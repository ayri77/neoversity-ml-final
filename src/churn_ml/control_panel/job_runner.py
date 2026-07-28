"""Durable terminal-record wrapper for UI-launched background jobs."""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


TERMINAL_FILE_NAME = "terminal.json"
TERMINAL_SCHEMA_VERSION = 1
TERMINAL_KEYS = frozenset(
    {
        "schema_version",
        "job_id",
        "terminal_status",
        "exit_code",
        "started_at_utc",
        "finished_at_utc",
    }
)


def utc_now_text() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_repository_import_path()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--terminal-file", required=True)
    parser.add_argument("target", nargs=argparse.REMAINDER)
    args = parser.parse_args(list(argv) if argv is not None else None)
    target = list(args.target)
    if target and target[0] == "--":
        target = target[1:]
    if not target or not all(isinstance(item, str) and item for item in target):
        return 2
    try:
        job_id = _canonical_job_id(args.job_id)
        terminal_path = _validated_terminal_path(Path(args.terminal_file), job_id)
    except ValueError:
        return 2
    started = utc_now_text()
    completed = subprocess.run(
        target,
        shell=False,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    exit_code = int(completed.returncode)
    try:
        write_terminal_record(
            terminal_path,
            job_id=job_id,
            exit_code=exit_code,
            started_at_utc=started,
            finished_at_utc=utc_now_text(),
        )
    except OSError:
        return exit_code if exit_code != 0 else 1
    return exit_code


def write_terminal_record(
    path: Path,
    *,
    job_id: str,
    exit_code: int,
    started_at_utc: str,
    finished_at_utc: str,
) -> None:
    if type(exit_code) is not int:
        raise ValueError("exit_code must be an integer.")
    payload = {
        "schema_version": TERMINAL_SCHEMA_VERSION,
        "job_id": job_id,
        "terminal_status": "succeeded" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
    }
    _atomic_write_json(path, payload)


def _ensure_repository_import_path() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    root_text = str(repository_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def _canonical_job_id(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except ValueError as error:
        raise ValueError("job_id must be a canonical UUID.") from error
    text = str(parsed)
    if text != value:
        raise ValueError("job_id must be a canonical UUID.")
    return text


def _validated_terminal_path(path: Path, job_id: str) -> Path:
    from src.churn_ml.control_panel.path_safety import (
        PathSafetyError,
        path_exists_nonfollowing,
        require_regular_file,
        require_safe_directory,
        require_safe_existing_ancestors,
    )

    if path.name != TERMINAL_FILE_NAME:
        raise ValueError("Terminal record file name is invalid.")
    absolute = path if path.is_absolute() else path.absolute()
    job_directory = absolute.parent
    if job_directory.name != job_id:
        raise ValueError("Terminal record is outside the job directory.")
    try:
        require_safe_directory(job_directory)
        require_safe_existing_ancestors(absolute)
        if path_exists_nonfollowing(absolute):
            require_regular_file(absolute, reject_hardlinks=True)
    except PathSafetyError as error:
        raise ValueError("Terminal record path is unsafe.") from error
    if absolute.resolve(strict=False).parent != job_directory.resolve(strict=True):
        raise ValueError("Terminal record path escaped the job directory.")
    return absolute


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    from src.churn_ml.control_panel.path_safety import (
        PathSafetyError,
        path_exists_nonfollowing,
        require_regular_file,
        require_safe_directory,
    )

    try:
        require_safe_directory(path.parent)
        if path_exists_nonfollowing(path):
            require_regular_file(path, reject_hardlinks=True)
    except PathSafetyError as error:
        raise OSError(f"Unsafe terminal path: {path}") from error
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.chmod(temporary, stat.S_IRUSR | stat.S_IWUSR)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
