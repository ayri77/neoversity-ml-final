from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import IO, Sequence

from src.churn_ml.control_panel.jobs import JobManager
from src.churn_ml.control_panel.process import SpawnedProcess


class FakeBackend:
    def __init__(self) -> None:
        self.exit_code: int | None = None
        self.exists = True
        self.argv: list[str] = []

    def spawn(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        stdout: IO[bytes],
        stderr: IO[bytes],
    ) -> SpawnedProcess:
        del cwd, stdout, stderr
        self.argv = list(argv)
        return SpawnedProcess(pid=41001, process_group=41001)

    def poll(self, pid: int) -> int | None:
        assert pid == 41001
        return self.exit_code

    def pid_exists(self, pid: int) -> bool:
        assert pid == 41001
        return self.exists

    def stop(self, pid: int, process_group: int | None) -> None:
        assert (pid, process_group) == (41001, 41001)
        self.exit_code = -15


def _manager(tmp_path: Path, backend: FakeBackend | None = None) -> JobManager:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return JobManager(
        tmp_path / "jobs",
        working_directory=tmp_path,
        backend=backend,
    )


def test_atomic_successful_and_failed_state_transitions(tmp_path: Path) -> None:
    backend = FakeBackend()
    manager = _manager(tmp_path, backend)
    record = manager.start(
        argv=["safe", "arg"],
        redacted_argv=["safe", "<redacted>"],
        command_id="command",
        action_id="run",
        references={"config": "config.yaml"},
    )
    assert record.status["state"] == "running"
    assert record.command["argv"] == ["safe", "<redacted>"]
    assert record.command["shell"] is False
    assert all(
        (record.root / name).exists()
        for name in {
            "job.json",
            "command.json",
            "status.json",
            "stdout.log",
            "stderr.log",
        }
    )
    backend.exit_code = 0
    assert manager.refresh(record.job_id).status["state"] == "succeeded"

    backend = FakeBackend()
    manager = _manager(tmp_path / "second", backend)
    failed = manager.start(
        argv=["safe"],
        redacted_argv=["safe"],
        command_id="command",
        action_id="run",
        references={},
    )
    backend.exit_code = 7
    refreshed = manager.refresh(failed.job_id)
    assert refreshed.status["state"] == "failed"
    assert refreshed.status["exit_code"] == 7


def test_confirmed_stop_state(tmp_path: Path) -> None:
    backend = FakeBackend()
    manager = _manager(tmp_path, backend)
    record = manager.start(
        argv=["safe"],
        redacted_argv=["safe"],
        command_id="command",
        action_id="run",
        references={},
    )
    stopped = manager.request_stop(record.job_id)
    assert stopped.status["state"] == "stopped"
    assert stopped.status["exit_code"] == -15


def test_restart_refresh_marks_disappeared_process_orphaned(tmp_path: Path) -> None:
    backend = FakeBackend()
    manager = _manager(tmp_path, backend)
    record = manager.start(
        argv=["safe"],
        redacted_argv=["safe"],
        command_id="command",
        action_id="run",
        references={},
    )
    restarted_backend = FakeBackend()
    restarted_backend.exists = False
    restarted = _manager(tmp_path, restarted_backend)
    assert restarted.refresh(record.job_id).status["state"] == "orphaned"


def test_harmless_real_success_and_failure_processes(tmp_path: Path) -> None:
    manager = JobManager(tmp_path / "jobs", working_directory=tmp_path)
    for expected, code in (("succeeded", 0), ("failed", 3)):
        record = manager.start(
            argv=[sys.executable, "-c", f"raise SystemExit({code})"],
            redacted_argv=[sys.executable, "-c", "<dummy>"],
            command_id="dummy",
            action_id="run",
            references={},
        )
        deadline = time.monotonic() + 10
        refreshed = manager.refresh(record.job_id)
        while refreshed.status["state"] == "running" and time.monotonic() < deadline:
            time.sleep(0.02)
            refreshed = manager.refresh(record.job_id)
        assert refreshed.status["state"] == expected


def test_harmless_real_process_can_be_stopped(tmp_path: Path) -> None:
    manager = JobManager(tmp_path / "jobs", working_directory=tmp_path)
    record = manager.start(
        argv=[sys.executable, "-c", "import time; time.sleep(30)"],
        redacted_argv=[sys.executable, "-c", "<dummy sleep>"],
        command_id="dummy",
        action_id="run",
        references={},
    )
    time.sleep(0.1)
    refreshed = manager.request_stop(record.job_id)
    deadline = time.monotonic() + 10
    while refreshed.status["state"] == "stop_requested" and time.monotonic() < deadline:
        time.sleep(0.02)
        refreshed = manager.refresh(record.job_id)
    assert refreshed.status["state"] == "stopped"
