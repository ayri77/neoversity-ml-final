"""Durable artifact helpers for standalone AutoGluon runs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    """Atomically write JSON and durably flush it to disk."""
    _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_yaml(path: Path, payload: Any) -> None:
    """Atomically write portable YAML and durably flush it to disk."""
    _atomic_write(path, yaml.safe_dump(payload, sort_keys=False))


def write_text(path: Path, text: str) -> None:
    """Atomically write UTF-8 text and durably flush it to disk."""
    _atomic_write(path, text)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    replaced = False
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        replaced = True
    finally:
        if not replaced:
            temporary_path.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_inventory(
    run_dir: Path, hash_limit_bytes: int = 10 * 1024 * 1024
) -> dict[str, Any]:
    """Create a deterministic inventory without following directory symlinks."""
    entries: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(run_dir).as_posix()
        if relative in {"artifact_inventory.json", "_SUCCESS", "_FAILED"}:
            continue
        if path.is_symlink():
            entries.append(
                {"path": relative, "type": "symlink", "target": os.readlink(path)}
            )
        elif path.is_dir():
            entries.append({"path": relative, "type": "directory"})
        elif path.is_file():
            size = path.stat().st_size
            entry: dict[str, Any] = {
                "path": relative,
                "type": "file",
                "size_bytes": size,
            }
            if size <= hash_limit_bytes:
                entry["sha256"] = sha256_file(path)
            entries.append(entry)
    return {
        "generated_at_utc": utc_now(),
        "terminal_markers_excluded": True,
        "entries": entries,
    }


def bounded_text_tail(path: Path, max_bytes: int = 16_384, max_lines: int = 80) -> str:
    """Return a bounded UTF-8-safe tail while keeping the full log authoritative."""
    if not path.is_file():
        return ""
    size = path.stat().st_size
    with path.open("rb") as stream:
        if size > max_bytes:
            stream.seek(-max_bytes, os.SEEK_END)
        data = stream.read()
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def discover_model_directories(predictor_dir: Path) -> list[str]:
    """List filesystem model directories without asserting they are loadable."""
    models_dir = predictor_dir / "models"
    if not models_dir.is_dir():
        return []
    return sorted(
        path.relative_to(predictor_dir).as_posix()
        for path in models_dir.rglob("*")
        if path.is_dir()
    )
