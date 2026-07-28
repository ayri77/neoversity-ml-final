from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from src.churn_ml.control_panel.jobs import JobError, JobManager
from src.churn_ml.control_panel.process import (
    ProcessCheck,
    ProcessIdentity,
    SpawnedProcess,
)
from src.churn_ml.control_panel.registry import load_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMAND_ID = "experiment_core_v2"
ACTION_ID = "validate"


class FakeBackend:
    def __init__(self, check: ProcessCheck | None = None) -> None:
        self.exit_code: int | None = None
        self.check = check or ProcessCheck("matching")
        self.argv: list[str] = []
        self.environment: dict[str, str] = {}
        self.stop_signal_calls = 0
        self.identity = ProcessIdentity(
            pid=41001,
            creation_time_utc="2026-01-01T00:00:00.000000Z",
            executable_path=str(Path(sys.executable).resolve()),
            argv_sha256="1" * 64,
            process_group=None,
            session_id=None,
        )

    def spawn(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        stdout_path: Path,
        stderr_path: Path,
        environment: Mapping[str, str],
        redactions: Sequence[str],
    ) -> SpawnedProcess:
        del cwd, stdout_path, stderr_path, redactions
        self.argv = list(argv)
        self.environment = dict(environment)
        return SpawnedProcess(self.identity)

    def poll(self, identity: ProcessIdentity) -> int | None:
        assert identity.pid == self.identity.pid
        return self.exit_code

    def verify(self, identity: ProcessIdentity) -> ProcessCheck:
        assert identity.pid == self.identity.pid
        return self.check

    def stop(self, identity: ProcessIdentity) -> ProcessCheck:
        assert identity.pid == self.identity.pid
        if self.check.state == "matching":
            self.stop_signal_calls += 1
            self.exit_code = -15
        return self.check


def _manager(tmp_path: Path, backend: FakeBackend | None = None) -> JobManager:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return JobManager(
        tmp_path / "jobs",
        working_directory=tmp_path,
        commands=load_registry(PROJECT_ROOT).commands,
        backend=backend,
    )


def _start(manager: JobManager, *, secret: str | None = None):
    argv = ["safe", secret or "arg"]
    redacted = ["safe", "<redacted>" if secret else "arg"]
    return manager.start(
        argv=argv,
        redacted_argv=redacted,
        command_id=COMMAND_ID,
        action_id=ACTION_ID,
        references={"config": secret or "config.yaml"},
    )


def test_atomic_successful_and_failed_state_transitions(tmp_path: Path) -> None:
    backend = FakeBackend()
    manager = _manager(tmp_path, backend)
    record = _start(manager, secret="canary-secret")
    assert record.status["state"] == "running"
    assert record.command["argv"] == ["safe", "<redacted>"]
    assert record.job["references"] == {"config": "<redacted>"}
    assert record.command["shell"] is False
    backend.exit_code = 0
    assert manager.refresh(record.job_id).status["state"] == "succeeded"

    backend = FakeBackend()
    manager = _manager(tmp_path / "second", backend)
    failed = _start(manager)
    backend.exit_code = 7
    refreshed = manager.refresh(failed.job_id)
    assert refreshed.status["state"] == "failed"
    assert refreshed.status["exit_code"] == 7


def test_matching_live_identity_refreshes_and_legitimate_stop_succeeds(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    manager = _manager(tmp_path, backend)
    record = _start(manager)
    assert manager.refresh(record.job_id).status["state"] == "running"
    stopped = manager.request_stop(record.job_id)
    assert stopped.status["state"] == "stopped"
    assert backend.stop_signal_calls == 1


@pytest.mark.parametrize(
    "check",
    [
        ProcessCheck("mismatched", "Process creation time does not match."),
        ProcessCheck("mismatched", "Process executable does not match."),
        ProcessCheck("unverifiable", "Process identity access was denied."),
    ],
)
def test_restart_mismatch_or_unverifiable_process_becomes_orphaned(
    tmp_path: Path, check: ProcessCheck
) -> None:
    creator = FakeBackend()
    record = _start(_manager(tmp_path, creator))
    restarted = _manager(tmp_path, FakeBackend(check))
    refreshed = restarted.refresh(record.job_id)
    assert refreshed.status["state"] == "orphaned"
    assert refreshed.status["diagnostic"] == check.diagnostic


def test_stop_never_signals_mismatched_reused_pid(tmp_path: Path) -> None:
    backend = FakeBackend()
    manager = _manager(tmp_path, backend)
    record = _start(manager)
    backend.check = ProcessCheck("mismatched", "Process creation time does not match.")
    with pytest.raises(JobError, match="verified running"):
        manager.request_stop(record.job_id)
    assert backend.stop_signal_calls == 0
    assert manager.load(record.job_id).status["state"] == "orphaned"


def test_already_exited_reconciles_safely_as_orphaned_after_restart(
    tmp_path: Path,
) -> None:
    record = _start(_manager(tmp_path, FakeBackend()))
    restarted = _manager(
        tmp_path, FakeBackend(ProcessCheck("exited", "Process has exited."))
    )
    assert restarted.refresh(record.job_id).status["state"] == "orphaned"


def test_strict_metadata_rejects_unknown_keys_and_registry_ids(tmp_path: Path) -> None:
    manager = _manager(tmp_path, FakeBackend())
    record = _start(manager)
    job_path = record.root / "job.json"
    payload = json.loads(job_path.read_text(encoding="utf-8"))
    payload["unknown"] = True
    job_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(JobError, match="keys are invalid"):
        manager.load(record.job_id)


@pytest.mark.skipif(os.name != "nt", reason="Windows junction semantics")
def test_uuid_named_windows_junction_is_never_loaded(tmp_path: Path) -> None:
    manager = _manager(tmp_path, FakeBackend())
    external = tmp_path / "external"
    external.mkdir()
    junction = manager.jobs_root / str(uuid.uuid4())
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(external)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip("Junction creation is unavailable")
    with pytest.raises(JobError, match="Unsafe job directory"):
        manager.load(junction.name)
    assert manager.list_jobs(refresh=False) == []


def _wait_terminal(manager: JobManager, job_id: str) -> str:
    deadline = time.monotonic() + 10
    record = manager.refresh(job_id)
    while (
        record.status["state"] in {"running", "stop_requested"}
        and time.monotonic() < deadline
    ):
        time.sleep(0.02)
        record = manager.refresh(job_id)
    return str(record.status["state"])


def test_harmless_real_success_failure_stop_and_secret_redaction(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    for expected, code in (("succeeded", 0), ("failed", 3)):
        record = manager.start(
            argv=[
                sys.executable,
                "-c",
                f"import time;time.sleep(.1);raise SystemExit({code})",
            ],
            redacted_argv=[sys.executable, "-c", "<harmless>"],
            command_id=COMMAND_ID,
            action_id=ACTION_ID,
            references={},
        )
        assert _wait_terminal(manager, record.job_id) == expected

    sleeper = manager.start(
        argv=[sys.executable, "-c", "import time;time.sleep(30)"],
        redacted_argv=[sys.executable, "-c", "<harmless sleep>"],
        command_id=COMMAND_ID,
        action_id=ACTION_ID,
        references={},
    )
    time.sleep(0.1)
    manager.request_stop(sleeper.job_id)
    assert _wait_terminal(manager, sleeper.job_id) == "stopped"

    secret = "CANARY_SECRET_8b3e5e"
    secret_job = manager.start(
        argv=[sys.executable, "-c", "import sys;print(sys.argv[1])", secret],
        redacted_argv=[sys.executable, "-c", "<harmless print>", "<redacted>"],
        command_id=COMMAND_ID,
        action_id=ACTION_ID,
        references={"value": secret},
    )
    assert _wait_terminal(manager, secret_job.job_id) == "succeeded"
    time.sleep(0.1)
    for path in secret_job.root.iterdir():
        assert secret not in path.read_text(encoding="utf-8", errors="replace")
