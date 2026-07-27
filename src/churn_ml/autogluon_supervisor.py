"""Parent-process supervision for crash-resilient AutoGluon runs."""

from __future__ import annotations

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
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from src.churn_ml.autogluon_artifacts import (
    bounded_text_tail,
    build_inventory,
    discover_model_directories,
    utc_now,
    write_json,
    write_yaml,
)
from src.churn_ml.autogluon_config import load_config
from src.churn_ml.autogluon_profiles import (
    SUPPORTED_AUTOGLUON_VERSION,
    get_profile,
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


def train_supervised(
    config_path: Path,
    repository_root: Path,
    *,
    run_id: str | None = None,
    worker_command_factory: WorkerCommandFactory | None = None,
    tee_progress: bool = True,
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
    process: subprocess.Popen[str] | None = None
    process_exit_code: int | None = None
    last_stages: list[str] = []

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "run_id": selected_run_id,
        "config_identity_sha256": config.identity_sha256,
        "profile": profile_summary(get_profile(config.profile_id)),
        "dataset_version": config.dataset.version,
        "resources": {
            "time_limit_seconds": config.resources.time_limit_seconds,
            "num_cpus": config.resources.num_cpus,
            "num_gpus": config.resources.num_gpus,
            "fit_strategy": config.resources.fit_strategy,
            "fold_fitting_strategy": config.resources.fold_fitting_strategy,
        },
        "started_at_utc": started_at,
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
        "started_at_utc": started_at,
        "child_pid": None,
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

    failure_reason: str | None = None
    try:
        command = command_factory(run_dir, config.paths.repository_root)
        if not command or not all(isinstance(part, str) for part in command):
            raise ValueError("worker command factory returned an invalid argument list")
        metadata["local_operational_nonportable"]["worker_command"] = command
        write_json(run_dir / "run_metadata.json", metadata)
        logger.info("Launching child worker")
        environment_vars = os.environ.copy()
        environment_vars["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=config.paths.repository_root,
            env=environment_vars,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            shell=False,
        )
        metadata["child_pid"] = process.pid
        status["child_pid"] = process.pid
        write_json(run_dir / "run_metadata.json", metadata)
        write_json(run_dir / "execution_status.json", status)
        logger.info("Child PID: %s", process.pid)

        assert process.stdout is not None
        assert process.stderr is not None
        with (
            stdout_log.open(
                "w", encoding="utf-8", newline="\n", buffering=1
            ) as stdout_file,
            stderr_log.open(
                "w", encoding="utf-8", newline="\n", buffering=1
            ) as stderr_file,
        ):
            threads = [
                threading.Thread(
                    target=_pump_stream,
                    args=(process.stdout, stdout_file, last_stages, tee_progress),
                    daemon=True,
                ),
                threading.Thread(
                    target=_pump_stream,
                    args=(process.stderr, stderr_file, last_stages, False),
                    daemon=True,
                ),
            ]
            for thread in threads:
                thread.start()
            process_exit_code = process.wait()
            for thread in threads:
                thread.join()
        logger.info("Child exited with process code %s", process_exit_code)
    except Exception as error:
        failure_reason = f"supervisor_exception: {type(error).__name__}: {error}"
        logger.exception("Supervisor failed while running the worker")
        if process is not None and process.poll() is None:
            process.terminate()
            process_exit_code = process.wait()

    ended_at = utc_now()
    duration = time.monotonic() - monotonic_start
    last_stage = last_stages[-1] if last_stages else "worker_not_observed"
    worker_result = _read_json(run_dir / "worker_result.json")
    expected_missing = _missing_success_artifacts(run_dir)
    succeeded = (
        failure_reason is None
        and process_exit_code == 0
        and not expected_missing
        and worker_result is not None
        and worker_result.get("status") == "completed"
    )
    if not succeeded and failure_reason is None:
        if process_exit_code != 0:
            failure_reason = f"worker_exit_code:{process_exit_code}"
        elif expected_missing:
            failure_reason = f"missing_expected_artifacts:{','.join(expected_missing)}"
        else:
            failure_reason = "invalid_worker_result"

    status.update(
        {
            "status": "succeeded" if succeeded else "failed",
            "ended_at_utc": ended_at,
            "duration_seconds": duration,
            "child_process_exit_code": process_exit_code,
            "windows_exit_code_unsigned": (
                process_exit_code & 0xFFFFFFFF
                if process_exit_code is not None
                else None
            ),
            "last_completed_observable_stage": last_stage,
            "failure_reason": failure_reason,
            "missing_expected_artifacts": expected_missing,
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
            "windows_exit_code_unsigned": status["windows_exit_code_unsigned"],
        }
    )
    write_json(run_dir / "run_metadata.json", metadata)
    write_json(run_dir / "execution_status.json", status)
    logger.info("Final status prepared: %s", status["status"])
    _close_logger(logger)
    write_json(run_dir / "artifact_inventory.json", build_inventory(run_dir))

    marker = "_SUCCESS" if succeeded else "_FAILED"
    marker_payload = {
        "status": status["status"],
        "run_id": selected_run_id,
        "ended_at_utc": ended_at,
        "child_process_exit_code": process_exit_code,
        "windows_exit_code_unsigned": status["windows_exit_code_unsigned"],
        "failure_reason": failure_reason,
    }
    result = SupervisorResult(
        run_dir=run_dir,
        succeeded=succeeded,
        child_exit_code=process_exit_code,
    )
    write_json(run_dir / marker, marker_payload)
    return result


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


def _pump_stream(
    source: TextIO,
    destination: TextIO,
    last_stages: list[str],
    tee_progress: bool,
) -> None:
    for line in source:
        destination.write(line)
        destination.flush()
        if line.startswith("AUTOGLUON_STAGE:"):
            stage = line.partition(":")[2].strip()
            last_stages.append(stage)
            if tee_progress:
                print(f"[worker] {stage}", flush=True)


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
    )
    return [relative for relative in expected if not (run_dir / relative).is_file()]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


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


def _close_logger(logger: logging.Logger) -> None:
    for handler in tuple(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
