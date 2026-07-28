from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.churn_ml.control_panel.process import LocalProcessBackend, ProcessBackend


class JobError(RuntimeError):
    """Raised for invalid job operations or state transitions."""


JOB_STATES = {
    "created",
    "running",
    "succeeded",
    "failed",
    "stop_requested",
    "stopped",
    "orphaned",
}
TERMINAL_STATES = {"succeeded", "failed", "stopped", "orphaned"}
TRANSITIONS = {
    "created": {"running", "failed"},
    "running": {"succeeded", "failed", "stop_requested", "orphaned"},
    "stop_requested": {"stopped", "failed", "orphaned"},
    "succeeded": set(),
    "failed": set(),
    "stopped": set(),
    "orphaned": set(),
}


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
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


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    root: Path
    job: Mapping[str, Any]
    command: Mapping[str, Any]
    status: Mapping[str, Any]


class JobManager:
    def __init__(
        self,
        jobs_root: Path,
        *,
        working_directory: Path,
        backend: ProcessBackend | None = None,
    ) -> None:
        self.jobs_root = jobs_root
        self.working_directory = working_directory.resolve(strict=True)
        self.backend = backend or LocalProcessBackend()
        self.jobs_root.mkdir(parents=True, exist_ok=True)

    def start(
        self,
        *,
        argv: Sequence[str],
        redacted_argv: Sequence[str],
        command_id: str,
        action_id: str,
        references: Mapping[str, str | int],
    ) -> JobRecord:
        job_id = str(uuid.uuid4())
        root = self.jobs_root / job_id
        root.mkdir(parents=False, exist_ok=False)
        created_at = utc_now_text()
        job = {
            "schema_version": 1,
            "job_id": job_id,
            "created_at_utc": created_at,
            "command_id": command_id,
            "action_id": action_id,
            "references": dict(references),
            "authoritative_result": False,
        }
        command = {
            "schema_version": 1,
            "argv": list(redacted_argv),
            "working_directory": str(self.working_directory),
            "shell": False,
        }
        status: dict[str, Any] = {
            "schema_version": 1,
            "state": "created",
            "created_at_utc": created_at,
            "started_at_utc": None,
            "finished_at_utc": None,
            "updated_at_utc": created_at,
            "pid": None,
            "process_group": None,
            "exit_code": None,
            "elapsed_seconds": 0.0,
        }
        atomic_write_json(root / "job.json", job)
        atomic_write_json(root / "command.json", command)
        atomic_write_json(root / "status.json", status)
        (root / "stdout.log").touch(exist_ok=False)
        (root / "stderr.log").touch(exist_ok=False)
        try:
            with (
                (root / "stdout.log").open("ab", buffering=0) as stdout,
                (root / "stderr.log").open("ab", buffering=0) as stderr,
            ):
                spawned = self.backend.spawn(
                    argv,
                    cwd=self.working_directory,
                    stdout=stdout,
                    stderr=stderr,
                )
            status.update(
                {
                    "state": "running",
                    "started_at_utc": utc_now_text(),
                    "updated_at_utc": utc_now_text(),
                    "pid": spawned.pid,
                    "process_group": spawned.process_group,
                }
            )
            atomic_write_json(root / "status.json", status)
        except BaseException as error:
            status.update(
                {
                    "state": "failed",
                    "finished_at_utc": utc_now_text(),
                    "updated_at_utc": utc_now_text(),
                    "exit_code": None,
                    "launch_error": f"{type(error).__name__}: {error}",
                }
            )
            atomic_write_json(root / "status.json", status)
            raise
        return JobRecord(job_id, root, job, command, status)

    def list_jobs(self, *, refresh: bool = True) -> list[JobRecord]:
        records: list[JobRecord] = []
        for path in sorted(
            (item for item in self.jobs_root.iterdir() if item.is_dir()),
            key=lambda item: item.name,
            reverse=True,
        ):
            try:
                record = self.load(path.name)
                records.append(self.refresh(path.name) if refresh else record)
            except (JobError, OSError, json.JSONDecodeError):
                continue
        return sorted(
            records,
            key=lambda item: str(item.job.get("created_at_utc", "")),
            reverse=True,
        )

    def load(self, job_id: str) -> JobRecord:
        try:
            uuid.UUID(job_id)
        except ValueError as error:
            raise JobError("Invalid job id.") from error
        root = self.jobs_root / job_id
        if root.parent != self.jobs_root or not root.is_dir():
            raise JobError(f"Unknown job id: {job_id}.")
        job = _read_json(root / "job.json")
        command = _read_json(root / "command.json")
        status = _read_json(root / "status.json")
        if status.get("state") not in JOB_STATES:
            raise JobError(f"Job {job_id} has an invalid state.")
        return JobRecord(job_id, root, job, command, status)

    def refresh(self, job_id: str) -> JobRecord:
        record = self.load(job_id)
        status = dict(record.status)
        state = str(status["state"])
        if state in TERMINAL_STATES or state == "created":
            return record
        pid = status.get("pid")
        if not isinstance(pid, int):
            self._transition(record.root, status, "orphaned")
            return self.load(job_id)
        exit_code = self.backend.poll(pid)
        if exit_code is not None:
            target = (
                "stopped"
                if state == "stop_requested"
                else ("succeeded" if exit_code == 0 else "failed")
            )
            self._transition(record.root, status, target, exit_code=exit_code)
        elif not self.backend.pid_exists(pid):
            self._transition(record.root, status, "orphaned")
        else:
            status["elapsed_seconds"] = _elapsed(status.get("started_at_utc"))
            status["updated_at_utc"] = utc_now_text()
            atomic_write_json(record.root / "status.json", status)
        return self.load(job_id)

    def request_stop(self, job_id: str) -> JobRecord:
        record = self.refresh(job_id)
        status = dict(record.status)
        if status["state"] != "running":
            raise JobError("Only a running job can be stopped.")
        self._transition(record.root, status, "stop_requested")
        pid = status.get("pid")
        if not isinstance(pid, int):
            raise JobError("Running job has no PID.")
        self.backend.stop(pid, status.get("process_group"))
        refreshed = self.refresh(job_id)
        if refreshed.status["state"] == "stop_requested":
            return refreshed
        return refreshed

    def _transition(
        self,
        root: Path,
        status: dict[str, Any],
        target: str,
        *,
        exit_code: int | None = None,
    ) -> None:
        current = str(status["state"])
        if target not in TRANSITIONS[current]:
            raise JobError(f"Invalid job transition: {current} -> {target}.")
        now = utc_now_text()
        status["state"] = target
        status["updated_at_utc"] = now
        status["elapsed_seconds"] = _elapsed(status.get("started_at_utc"))
        if target in TERMINAL_STATES:
            status["finished_at_utc"] = now
            status["exit_code"] = exit_code
        atomic_write_json(root / "status.json", status)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise JobError(f"{path.name} must contain a JSON object.")
    return payload


def _elapsed(started_at: Any) -> float:
    if not isinstance(started_at, str):
        return 0.0
    parsed = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())
