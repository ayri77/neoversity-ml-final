from __future__ import annotations

import json
import os
import re
import stat
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.churn_ml.control_panel.job_runner import (
    TERMINAL_FILE_NAME,
    TERMINAL_KEYS,
    TERMINAL_SCHEMA_VERSION,
)
from src.churn_ml.control_panel.path_safety import (
    PathSafetyError,
    path_exists_nonfollowing,
    require_regular_file,
    require_safe_directory,
    require_safe_existing_ancestors,
)
from src.churn_ml.control_panel.process import (
    LocalProcessBackend,
    ProcessBackend,
    ProcessIdentity,
    build_child_environment,
)
from src.churn_ml.control_panel.schemas import CommandSpec


class JobError(RuntimeError):
    """Raised for invalid job operations, persistence, or state transitions."""


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
REQUIRED_JOB_FILES = frozenset(
    {"job.json", "command.json", "status.json", "stdout.log", "stderr.log"}
)
OPTIONAL_JOB_FILES = frozenset({TERMINAL_FILE_NAME, "mlflow_index.json"})
ALLOWED_JOB_FILES = REQUIRED_JOB_FILES | OPTIONAL_JOB_FILES
JOB_FILES = REQUIRED_JOB_FILES  # backward-compatible alias for required layout
JOB_RUNNER_PATH = Path(__file__).resolve().parent / "job_runner.py"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
DIAGNOSTIC_LIMIT = 512


def utc_now_text() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        require_safe_directory(path.parent)
        if path_exists_nonfollowing(path):
            require_regular_file(path, reject_hardlinks=True)
    except PathSafetyError as error:
        raise JobError(f"Unsafe job metadata path: {path.name}.") from error
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
        commands: Mapping[str, CommandSpec],
        backend: ProcessBackend | None = None,
    ) -> None:
        self.commands = commands
        try:
            self.working_directory = working_directory.resolve(strict=True)
            require_safe_directory(self.working_directory)
            require_safe_existing_ancestors(jobs_root)
            jobs_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.jobs_root = jobs_root.resolve(strict=True)
            require_safe_directory(self.jobs_root)
        except (OSError, PathSafetyError) as error:
            raise JobError("Job storage or working directory is unsafe.") from error
        self.backend = backend or LocalProcessBackend()

    def start(
        self,
        *,
        argv: Sequence[str],
        redacted_argv: Sequence[str],
        command_id: str,
        action_id: str,
        references: Mapping[str, str | int],
    ) -> JobRecord:
        command_spec = self._registered_action(command_id, action_id)
        if len(argv) != len(redacted_argv) or not all(
            isinstance(item, str) for item in (*argv, *redacted_argv)
        ):
            raise JobError("Execution and redacted argv must be aligned strings.")
        argv_secrets = tuple(
            raw
            for raw, redacted in zip(argv, redacted_argv, strict=True)
            if raw != redacted and raw
        )
        safe_references = _sanitize_references(references, argv_secrets)
        try:
            environment, environment_secrets = build_child_environment(
                command_spec.environment
            )
        except RuntimeError as error:
            raise JobError(str(error)) from error
        job_id = str(uuid.uuid4())
        root = self.jobs_root / job_id
        try:
            root.mkdir(parents=False, exist_ok=False, mode=0o700)
            require_safe_directory(root)
        except (OSError, PathSafetyError) as error:
            raise JobError("Could not allocate a safe job directory.") from error
        created = utc_now_text()
        job = {
            "schema_version": 2,
            "job_id": job_id,
            "created_at_utc": created,
            "command_id": command_id,
            "action_id": action_id,
            "references": safe_references,
            "authoritative_result": False,
        }
        command = {
            "schema_version": 2,
            "argv": list(redacted_argv),
            "working_directory": str(self.working_directory),
            "shell": False,
        }
        status: dict[str, Any] = {
            "schema_version": 2,
            "state": "created",
            "created_at_utc": created,
            "started_at_utc": None,
            "finished_at_utc": None,
            "updated_at_utc": created,
            "pid": None,
            "process_identity": None,
            "exit_code": None,
            "elapsed_seconds": 0.0,
            "diagnostic": None,
        }
        atomic_write_json(root / "job.json", job)
        atomic_write_json(root / "command.json", command)
        atomic_write_json(root / "status.json", status)
        _create_private_file(root / "stdout.log")
        _create_private_file(root / "stderr.log")
        try:
            wrapper_argv = _wrapper_argv(
                job_id=job_id,
                terminal_path=root / TERMINAL_FILE_NAME,
                target_argv=argv,
            )
            spawned = self.backend.spawn(
                wrapper_argv,
                cwd=self.working_directory,
                stdout_path=root / "stdout.log",
                stderr_path=root / "stderr.log",
                environment=environment,
                redactions=argv_secrets + environment_secrets,
            )
            started = utc_now_text()
            status.update(
                {
                    "state": "running",
                    "started_at_utc": started,
                    "updated_at_utc": started,
                    "pid": spawned.identity.pid,
                    "process_identity": _identity_payload(spawned.identity),
                }
            )
            atomic_write_json(root / "status.json", status)
        except BaseException as error:
            failed = utc_now_text()
            status.update(
                {
                    "state": "failed",
                    "finished_at_utc": failed,
                    "updated_at_utc": failed,
                    "diagnostic": "Process launch failed before identity capture.",
                }
            )
            atomic_write_json(root / "status.json", status)
            raise JobError("Process launch failed before identity capture.") from error
        return JobRecord(job_id, root, job, command, status)

    def list_jobs(self, *, refresh: bool = True) -> list[JobRecord]:
        records: list[JobRecord] = []
        try:
            require_safe_directory(self.jobs_root)
            entries = sorted(os.scandir(self.jobs_root), key=lambda item: item.name)
        except (OSError, PathSafetyError) as error:
            raise JobError("Job storage became unsafe.") from error
        for entry in reversed(entries):
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                record = self.load(entry.name)
                records.append(self.refresh(entry.name) if refresh else record)
            except (JobError, OSError, json.JSONDecodeError):
                continue
        return sorted(
            records, key=lambda item: str(item.job["created_at_utc"]), reverse=True
        )

    def load(self, job_id: str) -> JobRecord:
        try:
            parsed_id = uuid.UUID(job_id)
        except ValueError as error:
            raise JobError("Invalid job id.") from error
        if str(parsed_id) != job_id:
            raise JobError("Job id must use canonical UUID text.")
        root = self.jobs_root / job_id
        if root.parent != self.jobs_root:
            raise JobError("Invalid job path.")
        try:
            require_safe_directory(self.jobs_root)
            require_safe_directory(root)
            names = {
                entry.name
                for entry in os.scandir(root)
                if not entry.name.startswith(".")
            }
            if not REQUIRED_JOB_FILES.issubset(names) or not names.issubset(
                ALLOWED_JOB_FILES
            ):
                raise JobError("Job directory has missing or unknown files.")
            for name in ("stdout.log", "stderr.log"):
                require_regular_file(root / name, reject_hardlinks=True)
            if TERMINAL_FILE_NAME in names:
                require_regular_file(root / TERMINAL_FILE_NAME, reject_hardlinks=True)
            if "mlflow_index.json" in names:
                require_regular_file(root / "mlflow_index.json", reject_hardlinks=True)
        except (OSError, PathSafetyError) as error:
            raise JobError(f"Unsafe job directory: {job_id}.") from error
        job = _read_json(root / "job.json")
        command = _read_json(root / "command.json")
        status = _read_json(root / "status.json")
        _validate_job(job, job_id, self.commands)
        _validate_command(command, self.working_directory)
        _validate_status(status, created_at=str(job["created_at_utc"]))
        return JobRecord(job_id, root, job, command, status)

    def refresh(self, job_id: str) -> JobRecord:
        record = self.load(job_id)
        status = dict(record.status)
        state = str(status["state"])
        if state in TERMINAL_STATES or state == "created":
            return record
        terminal_exit = _trusted_terminal_exit_code(record.root, job_id)
        if terminal_exit is not None:
            self._finish_from_exit(record.root, status, state, terminal_exit)
            return self.load(job_id)
        identity = _identity_from_payload(status["process_identity"])
        exit_code = self.backend.poll(identity)
        if exit_code is not None:
            self._finish_from_exit(record.root, status, state, exit_code)
            return self.load(job_id)
        check = self.backend.verify(identity)
        if check.state == "matching":
            status["elapsed_seconds"] = _elapsed(status["started_at_utc"])
            status["updated_at_utc"] = utc_now_text()
            status["diagnostic"] = None
            atomic_write_json(record.root / "status.json", status)
        else:
            raced_terminal_exit = _trusted_terminal_exit_code(record.root, job_id)
            if raced_terminal_exit is not None:
                self._finish_from_exit(record.root, status, state, raced_terminal_exit)
            else:
                raced_exit_code = self.backend.poll(identity)
                if raced_exit_code is not None:
                    self._finish_from_exit(record.root, status, state, raced_exit_code)
                else:
                    self._transition(
                        record.root,
                        status,
                        "orphaned",
                        diagnostic=check.diagnostic
                        or "Process identity is unverifiable.",
                    )
        return self.load(job_id)

    def request_stop(self, job_id: str) -> JobRecord:
        record = self.refresh(job_id)
        status = dict(record.status)
        if status["state"] != "running":
            raise JobError("Only a verified running job can be stopped.")
        identity = _identity_from_payload(status["process_identity"])
        self._transition(record.root, status, "stop_requested")
        check = self.backend.stop(identity)
        if check.state in {"mismatched", "unverifiable"}:
            current = dict(self.load(job_id).status)
            self._transition(
                record.root,
                current,
                "orphaned",
                diagnostic=check.diagnostic or "Process identity changed before stop.",
            )
            return self.load(job_id)
        return self.refresh(job_id)

    def _registered_action(self, command_id: str, action_id: str) -> CommandSpec:
        command = self.commands.get(command_id)
        if command is None or action_id not in command.actions:
            raise JobError("Command/action is not present in the current registry.")
        return command

    def _finish_from_exit(
        self, root: Path, status: dict[str, Any], state: str, exit_code: int
    ) -> None:
        target = (
            "stopped"
            if state == "stop_requested"
            else ("succeeded" if exit_code == 0 else "failed")
        )
        self._transition(root, status, target, exit_code=exit_code)

    def _transition(
        self,
        root: Path,
        status: dict[str, Any],
        target: str,
        *,
        exit_code: int | None = None,
        diagnostic: str | None = None,
    ) -> None:
        current = str(status["state"])
        if target not in TRANSITIONS[current]:
            raise JobError(f"Invalid job transition: {current} -> {target}.")
        now = utc_now_text()
        status["state"] = target
        status["updated_at_utc"] = now
        status["elapsed_seconds"] = _elapsed(status["started_at_utc"])
        status["diagnostic"] = _bounded_diagnostic(diagnostic)
        if target in TERMINAL_STATES:
            status["finished_at_utc"] = now
            status["exit_code"] = exit_code
        atomic_write_json(root / "status.json", status)


def _wrapper_argv(
    *,
    job_id: str,
    terminal_path: Path,
    target_argv: Sequence[str],
) -> list[str]:
    if not JOB_RUNNER_PATH.is_file():
        raise JobError("Job runner wrapper is missing.")
    return [
        sys.executable,
        str(JOB_RUNNER_PATH),
        "--job-id",
        job_id,
        "--terminal-file",
        str(terminal_path),
        "--",
        *target_argv,
    ]


def _trusted_terminal_exit_code(root: Path, job_id: str) -> int | None:
    path = root / TERMINAL_FILE_NAME
    try:
        if not path_exists_nonfollowing(path):
            return None
        require_regular_file(path, reject_hardlinks=True)
        if path.resolve(strict=True).parent != root.resolve(strict=True):
            return None
        payload = _read_json(path)
    except (OSError, PathSafetyError, JobError, json.JSONDecodeError):
        return None
    try:
        return _validate_terminal_record(payload, job_id)
    except JobError:
        return None


def _validate_terminal_record(payload: Mapping[str, Any], job_id: str) -> int:
    _exact_keys(payload, set(TERMINAL_KEYS), TERMINAL_FILE_NAME)
    _exact_int(
        payload["schema_version"],
        f"{TERMINAL_FILE_NAME}.schema_version",
        expected=TERMINAL_SCHEMA_VERSION,
    )
    if payload["job_id"] != job_id:
        raise JobError("terminal.json job_id does not match its directory.")
    status = _exact_string(
        payload["terminal_status"], f"{TERMINAL_FILE_NAME}.terminal_status"
    )
    if status not in {"succeeded", "failed"}:
        raise JobError("terminal.json terminal_status is invalid.")
    exit_code = _exact_int(payload["exit_code"], f"{TERMINAL_FILE_NAME}.exit_code")
    if status == "succeeded" and exit_code != 0:
        raise JobError("terminal.json succeeded status requires exit_code 0.")
    if status == "failed" and exit_code == 0:
        raise JobError("terminal.json failed status requires a nonzero exit_code.")
    _canonical_timestamp(
        payload["started_at_utc"], f"{TERMINAL_FILE_NAME}.started_at_utc"
    )
    _canonical_timestamp(
        payload["finished_at_utc"], f"{TERMINAL_FILE_NAME}.finished_at_utc"
    )
    return exit_code


def _create_private_file(path: Path) -> None:
    descriptor = os.open(
        path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, stat.S_IRUSR | stat.S_IWUSR
    )
    os.close(descriptor)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        require_regular_file(path, reject_hardlinks=True)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except PathSafetyError as error:
        raise JobError(f"Unsafe metadata file: {path.name}.") from error
    if not isinstance(payload, dict):
        raise JobError(f"{path.name} must contain a JSON object.")
    return payload


def _validate_job(
    payload: Mapping[str, Any], job_id: str, commands: Mapping[str, CommandSpec]
) -> None:
    _exact_keys(
        payload,
        {
            "schema_version",
            "job_id",
            "created_at_utc",
            "command_id",
            "action_id",
            "references",
            "authoritative_result",
        },
        "job.json",
    )
    _exact_int(payload["schema_version"], "job.json.schema_version", expected=2)
    if payload["job_id"] != job_id:
        raise JobError("job.json job_id does not match its directory.")
    _canonical_timestamp(payload["created_at_utc"], "job.json.created_at_utc")
    command_id = _exact_string(payload["command_id"], "job.json.command_id")
    action_id = _exact_string(payload["action_id"], "job.json.action_id")
    command = commands.get(command_id)
    if command is None or action_id not in command.actions:
        raise JobError("Persisted command/action is absent from the current registry.")
    if payload["authoritative_result"] is not False:
        raise JobError("job.json authoritative_result must be false.")
    references = payload["references"]
    if not isinstance(references, dict) or not all(
        isinstance(key, str) and type(value) in {str, int}
        for key, value in references.items()
    ):
        raise JobError("job.json references have invalid primitive types.")


def _validate_command(payload: Mapping[str, Any], working_directory: Path) -> None:
    _exact_keys(
        payload,
        {"schema_version", "argv", "working_directory", "shell"},
        "command.json",
    )
    _exact_int(payload["schema_version"], "command.json.schema_version", expected=2)
    argv = payload["argv"]
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(item, str) for item in argv)
    ):
        raise JobError("command.json argv must be a non-empty string list.")
    if payload["working_directory"] != str(working_directory):
        raise JobError("command.json working_directory is not canonical.")
    if payload["shell"] is not False:
        raise JobError("command.json shell must be false.")


def _validate_status(payload: Mapping[str, Any], *, created_at: str) -> None:
    _exact_keys(
        payload,
        {
            "schema_version",
            "state",
            "created_at_utc",
            "started_at_utc",
            "finished_at_utc",
            "updated_at_utc",
            "pid",
            "process_identity",
            "exit_code",
            "elapsed_seconds",
            "diagnostic",
        },
        "status.json",
    )
    _exact_int(payload["schema_version"], "status.json.schema_version", expected=2)
    state = _exact_string(payload["state"], "status.json.state")
    if state not in JOB_STATES:
        raise JobError("status.json state is invalid.")
    if payload["created_at_utc"] != created_at:
        raise JobError("status.json created_at_utc does not match job.json.")
    for key in ("created_at_utc", "updated_at_utc"):
        _canonical_timestamp(payload[key], f"status.json.{key}")
    for key in ("started_at_utc", "finished_at_utc"):
        if payload[key] is not None:
            _canonical_timestamp(payload[key], f"status.json.{key}")
    elapsed = payload["elapsed_seconds"]
    if type(elapsed) not in {int, float} or elapsed < 0:
        raise JobError("status.json elapsed_seconds must be non-negative.")
    if payload["exit_code"] is not None and type(payload["exit_code"]) is not int:
        raise JobError("status.json exit_code must be an integer or null.")
    diagnostic = payload["diagnostic"]
    if diagnostic is not None and (
        not isinstance(diagnostic, str) or len(diagnostic) > DIAGNOSTIC_LIMIT
    ):
        raise JobError("status.json diagnostic is invalid.")
    if state == "created" or (
        state == "failed"
        and payload["started_at_utc"] is None
        and payload["pid"] is None
        and payload["process_identity"] is None
    ):
        if payload["pid"] is not None or payload["process_identity"] is not None:
            raise JobError("Unstarted job must not have process identity.")
    else:
        identity = _identity_from_payload(payload["process_identity"])
        if payload["pid"] != identity.pid:
            raise JobError("status.json PID does not match process identity.")


def _identity_payload(identity: ProcessIdentity) -> dict[str, Any]:
    return {
        "pid": identity.pid,
        "creation_time_utc": identity.creation_time_utc,
        "executable_path": identity.executable_path,
        "argv_sha256": identity.argv_sha256,
        "process_group": identity.process_group,
        "session_id": identity.session_id,
    }


def _identity_from_payload(payload: Any) -> ProcessIdentity:
    if not isinstance(payload, dict):
        raise JobError("status.json process_identity must be an object.")
    _exact_keys(
        payload,
        {
            "pid",
            "creation_time_utc",
            "executable_path",
            "argv_sha256",
            "process_group",
            "session_id",
        },
        "status.json.process_identity",
    )
    pid = _exact_int(payload["pid"], "process_identity.pid")
    if pid <= 0:
        raise JobError("process_identity.pid must be positive.")
    creation = _canonical_timestamp(
        payload["creation_time_utc"], "process_identity.creation_time_utc"
    )
    executable = _exact_string(
        payload["executable_path"], "process_identity.executable_path"
    )
    if not Path(executable).is_absolute():
        raise JobError("process_identity.executable_path must be absolute.")
    fingerprint = _exact_string(payload["argv_sha256"], "process_identity.argv_sha256")
    if SHA256.fullmatch(fingerprint) is None:
        raise JobError("process_identity.argv_sha256 is invalid.")
    return ProcessIdentity(
        pid=pid,
        creation_time_utc=creation,
        executable_path=executable,
        argv_sha256=fingerprint,
        process_group=_optional_int(
            payload["process_group"], "process_identity.process_group"
        ),
        session_id=_optional_int(payload["session_id"], "process_identity.session_id"),
    )


def _sanitize_references(
    references: Mapping[str, str | int], secrets: Sequence[str]
) -> dict[str, str | int]:
    result: dict[str, str | int] = {}
    for key, value in references.items():
        if not isinstance(key, str) or type(value) not in {str, int}:
            raise JobError(
                "References must contain only string keys and string/int values."
            )
        if isinstance(value, str):
            for secret in secrets:
                if secret and secret in value:
                    value = value.replace(secret, "<redacted>")
        result[key] = value
    return result


def _exact_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise JobError(
            f"{label} keys are invalid; missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}."
        )


def _exact_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise JobError(f"{label} must be a non-empty string.")
    return value


def _exact_int(value: Any, label: str, *, expected: int | None = None) -> int:
    if type(value) is not int or (expected is not None and value != expected):
        raise JobError(f"{label} must be the expected integer.")
    return value


def _optional_int(value: Any, label: str) -> int | None:
    return None if value is None else _exact_int(value, label)


def _canonical_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise JobError(f"{label} must be a canonical UTC timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise JobError(f"{label} must be a canonical UTC timestamp.") from error
    canonical = (
        parsed.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
    if value != canonical:
        raise JobError(f"{label} must be a canonical UTC timestamp.")
    return value


def _bounded_diagnostic(value: str | None) -> str | None:
    return (
        None
        if value is None
        else value.replace("\r", " ").replace("\n", " ")[:DIAGNOSTIC_LIMIT]
    )


def _elapsed(started_at: Any) -> float:
    if not isinstance(started_at, str):
        return 0.0
    parsed = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())
