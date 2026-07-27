"""Parent-process supervision for crash-resilient AutoGluon runs."""

from __future__ import annotations

import ctypes
import importlib.metadata
import json
import logging
import os
import platform
import re
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ContextManager, TextIO

from src.churn_ml.autogluon_artifacts import (
    bounded_text_tail,
    build_inventory,
    discover_model_directories,
    utc_now,
    write_json,
    write_yaml,
)
from src.churn_ml.autogluon_completion import (
    CompletionExpectations,
    CompletionValidation,
    load_and_validate_completion,
)
from src.churn_ml.autogluon_config import load_config
from src.churn_ml.autogluon_profiles import (
    SUPPORTED_AUTOGLUON_VERSION,
    get_profile,
    profile_sha256,
    profile_summary,
)


RUN_ID_MIN_LENGTH = 3
RUN_ID_MAX_LENGTH = 80
_RUN_ID_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$")


class RunAlreadyExistsError(FileExistsError):
    """Raised before allocation when an explicit run directory already exists."""


@dataclass(frozen=True)
class SupervisorResult:
    run_dir: Path
    succeeded: bool
    child_exit_code: int | None


WorkerCommandFactory = Callable[[Path, Path], list[str]]
LogWriterFactory = Callable[[Path], ContextManager[TextIO]]


def validate_run_id(run_id: str) -> str:
    """Validate a bounded filesystem-safe slug."""
    if not RUN_ID_MIN_LENGTH <= len(run_id) <= RUN_ID_MAX_LENGTH:
        raise ValueError(
            f"run ID length must be between {RUN_ID_MIN_LENGTH} and "
            f"{RUN_ID_MAX_LENGTH} characters"
        )
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run ID must be a lowercase safe slug containing only letters, digits, "
            "periods, underscores, or hyphens, with an alphanumeric first and last character"
        )
    return run_id


def generate_run_id(now: datetime | None = None) -> str:
    """Generate a UTC timestamped, collision-resistant safe run ID."""
    timestamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dt%H%M%Sz")
    return f"autogluon-{timestamp}-{uuid.uuid4().hex[:10]}"


def unsigned_windows_exit_code(exit_code: int | None) -> int | None:
    """Preserve the unsigned 32-bit representation of a Windows process result."""
    return exit_code & 0xFFFFFFFF if exit_code is not None else None


def _resolve_worker_pid(process: subprocess.Popen[Any]) -> int:
    """Resolve a uv/venv launcher's single Python descendant on Windows."""
    if os.name != "nt":
        return process.pid
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and process.poll() is None:
        descendants = _windows_descendant_process_ids(process.pid)
        if descendants:
            parents = {parent for _pid, parent in descendants}
            leaves = sorted(pid for pid, _parent in descendants if pid not in parents)
            if len(leaves) == 1:
                return leaves[0]
        time.sleep(0.01)
    return process.pid


def _windows_descendant_process_ids(root_pid: int) -> set[tuple[int, int]]:
    """Return descendant PID/parent-PID pairs from a Toolhelp process snapshot."""
    from ctypes import wintypes

    class ProcessEntry32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel32: Any = ctypes.WinDLL("kernel32", use_last_error=True)
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if snapshot == invalid_handle:
        return set()
    parent_by_pid: dict[int, int] = {}
    executable_by_pid: dict[int, str] = {}
    entry = ProcessEntry32W()
    entry.dwSize = ctypes.sizeof(entry)
    try:
        success = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while success:
            pid = int(entry.th32ProcessID)
            parent_by_pid[pid] = int(entry.th32ParentProcessID)
            executable_by_pid[pid] = str(entry.szExeFile).lower()
            success = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)

    descendants: set[int] = set()
    changed = True
    while changed:
        changed = False
        known_parents = descendants | {root_pid}
        for pid, parent_pid in parent_by_pid.items():
            if pid not in descendants and parent_pid in known_parents:
                descendants.add(pid)
                changed = True
    return {
        (pid, parent_by_pid[pid])
        for pid in descendants
        if executable_by_pid.get(pid, "").startswith("python")
    }


def train_supervised(
    config_path: Path,
    repository_root: Path,
    *,
    run_id: str | None = None,
    worker_command_factory: WorkerCommandFactory | None = None,
    tee_progress: bool = True,
    log_writer_factory: LogWriterFactory | None = None,
) -> SupervisorResult:
    """Validate, allocate, launch one worker, and write exactly one terminal marker."""
    config = load_config(config_path, repository_root, require_data_files=True)
    selected_run_id = validate_run_id(run_id) if run_id else generate_run_id()
    run_dir = config.paths.artifacts_root / selected_run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise RunAlreadyExistsError(
            f"run directory already exists: {run_dir}"
        ) from error

    logs_dir = run_dir / "logs"
    logs_dir.mkdir()
    supervisor_log = logs_dir / "supervisor.log"
    stdout_log = logs_dir / "worker.stdout.log"
    stderr_log = logs_dir / "worker.stderr.log"
    logger = _create_logger(supervisor_log, selected_run_id)
    started_at = utc_now()
    monotonic_start = time.monotonic()
    command_factory = worker_command_factory or _default_worker_command
    writer_factory = log_writer_factory or _open_log_writer
    process: subprocess.Popen[Any] | None = None
    process_exit_code: int | None = None
    last_stages: list[str] = []
    shared_lock = threading.Lock()
    pump_failures: list[dict[str, str]] = []

    profile = get_profile(config.profile_id)
    profile_identity = profile_summary(profile, config.seed, config.resources.num_gpus)
    profile_hash = profile_sha256(profile, config.seed, config.resources.num_gpus)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "run_id": selected_run_id,
        "config_identity_sha256": config.identity_sha256,
        "profile_id": config.profile_id,
        "profile_sha256": profile_hash,
        "profile": profile_identity,
        "dataset_version": config.dataset.version,
        "requested_seed": config.seed,
        "resources": {
            "time_limit_seconds": config.resources.time_limit_seconds,
            "num_cpus": config.resources.num_cpus,
            "gpu_budget": config.resources.num_gpus,
            "top_level_num_gpus_passed_to_fit": False,
            "fit_strategy": config.resources.fit_strategy,
            "fold_fitting_strategy": config.resources.fold_fitting_strategy,
        },
        "started_at_utc": started_at,
        "child_pid": None,
        "launched_process_pid": None,
        "local_operational_nonportable": {
            "repository_root": str(config.paths.repository_root),
            "run_directory": str(run_dir),
            "python_executable": sys.executable,
        },
    }
    environment = _environment_payload()
    status: dict[str, Any] = {
        "status": "running",
        "run_id": selected_run_id,
        "config_identity_sha256": config.identity_sha256,
        "profile_sha256": profile_hash,
        "requested_seed": config.seed,
        "started_at_utc": started_at,
        "child_pid": None,
        "launched_process_pid": None,
        "child_process_exit_code": None,
        "windows_exit_code_unsigned": None,
        "last_completed_observable_stage": "supervisor_initialized",
        "predictor_loading_attempted": False,
        "predictor_loading_succeeded": False,
    }
    write_yaml(run_dir / "resolved_config.yaml", config.portable)
    write_json(run_dir / "run_metadata.json", metadata)
    write_json(run_dir / "environment.json", environment)
    write_json(run_dir / "execution_status.json", status)
    logger.info("Allocated run %s", selected_run_id)
    logger.info("Profile: %s", config.profile_id)
    logger.info("Requested seed: %s", config.seed)

    supervisor_failure: str | None = None
    try:
        command = command_factory(run_dir, config.paths.repository_root)
        if not command or not all(type(part) is str for part in command):
            raise ValueError("worker command factory returned an invalid argument list")
        metadata["local_operational_nonportable"]["worker_command"] = command
        write_json(run_dir / "run_metadata.json", metadata)
        logger.info("Launching child worker")
        environment_vars = os.environ.copy()
        environment_vars["PYTHONUNBUFFERED"] = "1"
        with (
            writer_factory(stdout_log) as stdout_file,
            writer_factory(stderr_log) as stderr_file,
        ):
            process = subprocess.Popen(
                command,
                cwd=config.paths.repository_root,
                env=environment_vars,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                bufsize=0,
                shell=False,
            )
            worker_pid = _resolve_worker_pid(process)
            metadata["child_pid"] = worker_pid
            metadata["launched_process_pid"] = process.pid
            status["child_pid"] = worker_pid
            status["launched_process_pid"] = process.pid
            write_json(run_dir / "run_metadata.json", metadata)
            write_json(run_dir / "execution_status.json", status)
            logger.info(
                "Worker PID: %s (launched process PID: %s)", worker_pid, process.pid
            )

            assert process.stdout is not None
            assert process.stderr is not None
            threads = [
                threading.Thread(
                    target=_pump_stream,
                    args=(
                        "stdout",
                        process.stdout,
                        stdout_file,
                        last_stages,
                        tee_progress,
                        pump_failures,
                        shared_lock,
                    ),
                    daemon=False,
                ),
                threading.Thread(
                    target=_pump_stream,
                    args=(
                        "stderr",
                        process.stderr,
                        stderr_file,
                        last_stages,
                        False,
                        pump_failures,
                        shared_lock,
                    ),
                    daemon=False,
                ),
            ]
            for thread in threads:
                thread.start()
            process_exit_code = process.wait()
            for thread in threads:
                thread.join()
            if any(thread.is_alive() for thread in threads):
                _record_pump_failure(
                    pump_failures,
                    shared_lock,
                    "supervisor",
                    "join",
                    RuntimeError("log pump thread did not terminate"),
                )
        logger.info("Child exited with process code %s", process_exit_code)
    except Exception as error:
        supervisor_failure = f"supervisor_exception:{type(error).__name__}:{error}"
        logger.exception("Supervisor failed while running the worker")
        if process is not None and process.poll() is None:
            process.terminate()
            process_exit_code = process.wait()

    completion_validation = _validate_worker_completion(
        run_dir,
        metadata.get("child_pid") if type(metadata.get("child_pid")) is int else None,
        config.profile_id,
        profile_hash,
        profile_identity,
        config.dataset.version,
        config.seed,
    )
    expected_missing = _missing_success_artifacts(run_dir)
    failure_codes: list[str] = []
    if supervisor_failure:
        failure_codes.append(supervisor_failure)
    if process_exit_code != 0:
        failure_codes.append(f"worker_exit_code:{process_exit_code}")
    if pump_failures:
        failure_codes.append("log_pump_failure")
    if expected_missing:
        failure_codes.append("missing_expected_artifacts")
    failure_codes.extend(completion_validation.reason_codes)

    logger_close_failures = _close_logger(logger)
    pump_failures.extend(logger_close_failures)
    if logger_close_failures:
        failure_codes.append("supervisor_log_durability_failure")
    succeeded = not failure_codes
    ended_at = utc_now()
    duration = float(time.monotonic() - monotonic_start)
    last_stage = last_stages[-1] if last_stages else "worker_not_observed"
    unsigned_exit = unsigned_windows_exit_code(process_exit_code)
    failure_codes = list(dict.fromkeys(failure_codes))
    failure_reason = ";".join(failure_codes) if failure_codes else None

    status.update(
        {
            "status": "completed" if succeeded else "failed",
            "ended_at_utc": ended_at,
            "duration_seconds": duration,
            "child_process_exit_code": process_exit_code,
            "windows_exit_code_unsigned": unsigned_exit,
            "last_completed_observable_stage": last_stage,
            "failure_reason": failure_reason,
            "failure_codes": failure_codes,
            "missing_expected_artifacts": expected_missing,
            "worker_completion_valid": completion_validation.valid,
            "worker_completion_reason_codes": list(completion_validation.reason_codes),
            "worker_completion_details": list(completion_validation.details),
            "log_pump_failures": pump_failures,
            "predictor_loading_attempted": False,
            "predictor_loading_succeeded": False,
            "discovered_model_directories": discover_model_directories(
                run_dir / "predictor"
            ),
            "stderr_tail_bounded": bounded_text_tail(stderr_log),
            "full_logs_authoritative": True,
        }
    )
    metadata.update(
        {
            "ended_at_utc": ended_at,
            "duration_seconds": duration,
            "child_process_exit_code": process_exit_code,
            "windows_exit_code_unsigned": unsigned_exit,
            "worker_completion_valid": completion_validation.valid,
            "worker_completion_reason_codes": list(completion_validation.reason_codes),
            "log_pump_failures": pump_failures,
        }
    )
    write_json(run_dir / "run_metadata.json", metadata)
    write_json(run_dir / "execution_status.json", status)
    write_json(run_dir / "artifact_inventory.json", build_inventory(run_dir))

    marker = "_SUCCESS" if succeeded else "_FAILED"
    marker_payload = {
        "status": "completed" if succeeded else "failed",
        "run_id": selected_run_id,
        "ended_at_utc": ended_at,
        "child_process_exit_code": process_exit_code,
        "windows_exit_code_unsigned": unsigned_exit,
        "failure_reason": failure_reason,
        "failure_codes": failure_codes,
    }
    result = SupervisorResult(
        run_dir=run_dir,
        succeeded=succeeded,
        child_exit_code=process_exit_code,
    )
    write_json(run_dir / marker, marker_payload)
    return result


def _validate_worker_completion(
    run_dir: Path,
    child_pid: int | None,
    profile_id: str,
    profile_hash: str,
    profile_identity: dict[str, Any],
    dataset_version: str,
    requested_seed: int,
) -> CompletionValidation:
    if child_pid is None:
        return CompletionValidation(
            None,
            ("worker_process_not_started",),
            (),
        )
    profile_resolution = _read_json(run_dir / "profile_resolution.json")
    inspection_summary = _read_json(run_dir / "inspection" / "summary.json")
    prevalidation_reasons: list[str] = []
    prevalidation_details: list[str] = []
    resolved_families: tuple[str, ...] | None = None
    if profile_resolution is None:
        prevalidation_reasons.append("profile_resolution_invalid")
    else:
        expected_resolution = {
            "profile_id": profile_id,
            "profile_sha256": profile_hash,
            "profile_identity": profile_identity,
            "dataset_version": dataset_version,
            "requested_seed": requested_seed,
            "top_level_num_gpus_passed_to_fit": False,
        }
        for key, expected in expected_resolution.items():
            if profile_resolution.get(key) != expected:
                prevalidation_reasons.append(f"profile_resolution_{key}_mismatch")
        values = profile_resolution.get("resolved_families")
        if (
            type(values) is list
            and values
            and all(type(item) is str and item for item in values)
        ):
            resolved_families = tuple(values)
        else:
            prevalidation_reasons.append("profile_resolution_families_invalid")
        if type(profile_resolution.get("family_resources")) is not dict:
            prevalidation_reasons.append("profile_resolution_resources_invalid")
    if inspection_summary is None:
        prevalidation_reasons.append("inspection_summary_invalid")
    expectations = CompletionExpectations(
        worker_pid=child_pid,
        profile_id=profile_id,
        profile_sha256=profile_hash,
        dataset_version=dataset_version,
        requested_seed=requested_seed,
        resolved_families=resolved_families,
    )
    completion = load_and_validate_completion(
        run_dir / "worker_result.json",
        expectations,
        inspection_summary=inspection_summary,
    )
    return CompletionValidation(
        completion.payload,
        tuple(dict.fromkeys(prevalidation_reasons + list(completion.reason_codes))),
        tuple(prevalidation_details + list(completion.details)),
    )


def _pump_stream(
    stream_name: str,
    source: Any,
    destination: TextIO,
    last_stages: list[str],
    tee_progress: bool,
    failures: list[dict[str, str]],
    lock: threading.Lock,
) -> None:
    """Drain a child stream, preserving read/write/durability failures for the parent."""
    destination_failed = False
    try:
        iterator = iter(source)
    except Exception as error:
        _record_pump_failure(failures, lock, stream_name, "read", error)
        iterator = iter(())
    while True:
        try:
            line = next(iterator)
        except StopIteration:
            break
        except Exception as error:
            _record_pump_failure(failures, lock, stream_name, "read", error)
            break
        if isinstance(line, bytes):
            try:
                decoded_line = line.decode("utf-8", errors="strict")
            except UnicodeDecodeError as error:
                _record_pump_failure(failures, lock, stream_name, "encoding", error)
                decoded_line = line.decode("utf-8", errors="replace")
        elif type(line) is str:
            decoded_line = line
        else:
            unexpected_type_error = TypeError(
                f"stream yielded {type(line).__name__}, expected bytes or str"
            )
            _record_pump_failure(
                failures, lock, stream_name, "encoding", unexpected_type_error
            )
            decoded_line = str(line)
        if not destination_failed:
            try:
                destination.write(decoded_line)
                destination.flush()
            except Exception as error:
                destination_failed = True
                _record_pump_failure(failures, lock, stream_name, "write", error)
        if decoded_line.startswith("AUTOGLUON_STAGE:"):
            stage = decoded_line.partition(":")[2].strip()
            with lock:
                last_stages.append(stage)
            if tee_progress:
                try:
                    print(f"[worker] {stage}", flush=True)
                except Exception as error:
                    _record_pump_failure(failures, lock, stream_name, "tee", error)
    try:
        _durable_flush(destination)
    except Exception as error:
        _record_pump_failure(failures, lock, stream_name, "durable_flush", error)


def _durable_flush(stream: TextIO) -> None:
    stream.flush()
    os.fsync(stream.fileno())


def _record_pump_failure(
    failures: list[dict[str, str]],
    lock: threading.Lock,
    stream_name: str,
    operation: str,
    error: Exception,
) -> None:
    with lock:
        failures.append(
            {
                "stream": stream_name,
                "operation": operation,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )


def _default_worker_command(run_dir: Path, repository_root: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "src.churn_ml.autogluon_worker",
        "--run-dir",
        str(run_dir),
        "--repository-root",
        str(repository_root),
    ]


@contextmanager
def _open_log_writer(path: Path) -> Iterator[TextIO]:
    with path.open("w", encoding="utf-8", newline="\n", buffering=1) as stream:
        yield stream


def _missing_success_artifacts(run_dir: Path) -> list[str]:
    expected = (
        "worker_result.json",
        "dataset_manifest.json",
        "profile_resolution.json",
        "predictor/predictor.pkl",
        "predictor/learner.pkl",
        "predictor/version.txt",
        "inspection/leaderboard.csv",
        "inspection/summary.json",
        "logs/worker.stdout.log",
        "logs/worker.stderr.log",
    )
    return [relative for relative in expected if not (run_dir / relative).is_file()]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if type(value) is dict else None


def _environment_payload() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("autogluon.tabular", "pandas", "pyarrow", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "captured_at_utc": utc_now(),
        "portable": {
            "python_version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "packages": packages,
            "required_autogluon_version": SUPPORTED_AUTOGLUON_VERSION,
        },
        "local_operational_nonportable": {
            "python_executable": sys.executable,
        },
    }


def _create_logger(path: Path, run_id: str) -> logging.Logger:
    logger = logging.getLogger(f"autogluon_supervisor.{run_id}.{uuid.uuid4().hex}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(path, encoding="utf-8")
    formatter = logging.Formatter(
        fmt="%(asctime)sZ %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def _close_logger(logger: logging.Logger) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    lock = threading.Lock()
    for handler in tuple(logger.handlers):
        try:
            handler.flush()
            stream = getattr(handler, "stream", None)
            if stream is not None:
                os.fsync(stream.fileno())
        except Exception as error:
            _record_pump_failure(failures, lock, "supervisor", "durable_flush", error)
        finally:
            try:
                handler.close()
            except Exception as error:
                _record_pump_failure(failures, lock, "supervisor", "close", error)
            logger.removeHandler(handler)
    return failures
