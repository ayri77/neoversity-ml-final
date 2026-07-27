from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TextIO

import pytest
import yaml

import src.churn_ml.autogluon_supervisor as supervisor_module
from src.churn_ml.autogluon_supervisor import (
    RunAlreadyExistsError,
    _durable_flush,
    _pump_stream,
    train_supervised,
    unsigned_windows_exit_code,
)
from tests.test_autogluon_config import valid_payload


FAKE_WORKER = Path(__file__).with_name("autogluon_fake_worker.py")


@pytest.fixture
def repository(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repository with spaces"
    data_dir = root / "data" / "processed" / "v3_targeted_missingness"
    data_dir.mkdir(parents=True)
    (data_dir / "X_train.parquet").write_bytes(b"features")
    (data_dir / "y_train.parquet").write_bytes(b"target")
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(valid_payload()), encoding="utf-8")
    return root, config_path


def fake_command(mode: str, exit_code: int = 23):
    def factory(run_dir: Path, _repository_root: Path) -> list[str]:
        return [
            sys.executable,
            "-u",
            str(FAKE_WORKER),
            "--run-dir",
            str(run_dir),
            "--mode",
            mode,
            "--exit-code",
            str(exit_code),
        ]

    return factory


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_supervisor_success_and_success_marker_written_last(
    repository: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config_path = repository
    created_threads: list[threading.Thread] = []

    class TrackingThread(threading.Thread):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            created_threads.append(self)

    monkeypatch.setattr(supervisor_module.threading, "Thread", TrackingThread)
    result = train_supervised(
        config_path,
        root,
        run_id="fake-success",
        worker_command_factory=fake_command("success"),
        tee_progress=False,
    )
    assert result.succeeded
    marker = result.run_dir / "_SUCCESS"
    assert marker.is_file()
    assert not (result.run_dir / "_FAILED").exists()
    assert len(created_threads) == 2
    assert all(
        not thread.daemon and not thread.is_alive() for thread in created_threads
    )
    status = load_json(result.run_dir / "execution_status.json")
    assert status["status"] == "completed"
    assert status["worker_completion_valid"] is True
    assert status["log_pump_failures"] == []
    inspection = load_json(result.run_dir / "inspection" / "summary.json")
    assert inspection["effective_seed_status"] == "verified"
    assert inspection["effective_seed"] == 42
    assert inspection["auxiliary_effective_seeds_observed"] == [0]
    inventory = load_json(result.run_dir / "artifact_inventory.json")
    inventory_paths = {entry["path"] for entry in inventory["entries"]}
    assert not {"_SUCCESS", "_FAILED", "artifact_inventory.json"} & inventory_paths
    marker_time = marker.stat().st_mtime_ns
    assert all(
        path.stat().st_mtime_ns <= marker_time
        for path in result.run_dir.rglob("*")
        if path.is_file() and path != marker
    )
    metadata = load_json(result.run_dir / "run_metadata.json")
    assert metadata["requested_seed"] == 42
    assert metadata["profile"]["seed"] == 42
    assert metadata["resources"]["top_level_num_gpus_passed_to_fit"] is False
    command = metadata["local_operational_nonportable"]["worker_command"]
    assert isinstance(command, list)
    assert "repository with spaces" in command[4]


def test_subprocess_invocation_uses_argument_list_and_shell_false(
    repository: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config_path = repository
    real_popen = subprocess.Popen
    captured: dict[str, Any] = {}

    def recording_popen(command: Any, *args: Any, **kwargs: Any) -> Any:
        captured["command"] = command
        captured["shell"] = kwargs.get("shell")
        return real_popen(command, *args, **kwargs)

    monkeypatch.setattr(supervisor_module.subprocess, "Popen", recording_popen)
    result = train_supervised(
        config_path,
        root,
        run_id="popen-contract",
        worker_command_factory=fake_command("success"),
        tee_progress=False,
    )
    assert result.succeeded
    assert type(captured["command"]) is list
    assert all(type(item) is str for item in captured["command"])
    assert captured["shell"] is False


def test_existing_run_directory_is_refused_without_overwrite(
    repository: tuple[Path, Path],
) -> None:
    root, config_path = repository
    existing = root / "artifacts" / "autogluon_runs" / "existing-run"
    existing.mkdir(parents=True)
    sentinel = existing / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(RunAlreadyExistsError):
        train_supervised(
            config_path,
            root,
            run_id="existing-run",
            worker_command_factory=fake_command("success"),
        )
    assert sentinel.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize(
    ("mode", "reason"),
    (
        ("missing-result", "worker_result_missing"),
        ("invalid-json", "worker_result_invalid_json"),
        ("minimal", "worker_result_missing_fields"),
        ("missing-field", "worker_result_missing_fields"),
        ("unknown-field", "worker_result_unknown_fields"),
        ("wrong-type", "worker_result_wrong_field_type"),
        ("wrong-pid", "worker_result_pid_mismatch"),
        ("wrong-profile", "worker_result_profile_id_mismatch"),
        ("wrong-profile-hash", "worker_result_profile_hash_mismatch"),
        ("wrong-dataset", "worker_result_dataset_version_mismatch"),
        ("wrong-path", "worker_result_predictor_path_mismatch"),
        ("inconsistent-time", "worker_result_timestamp_order_invalid"),
        ("mismatch-inspection", "worker_result_best_model_mismatch"),
    ),
)
def test_zero_exit_with_invalid_completion_produces_only_failed(
    repository: tuple[Path, Path],
    mode: str,
    reason: str,
) -> None:
    root, config_path = repository
    result = train_supervised(
        config_path,
        root,
        run_id=f"completion-{mode}",
        worker_command_factory=fake_command(mode),
        tee_progress=False,
    )
    assert result.child_exit_code == 0
    assert not result.succeeded
    assert (result.run_dir / "_FAILED").is_file()
    assert not (result.run_dir / "_SUCCESS").exists()
    status = load_json(result.run_dir / "execution_status.json")
    metadata = load_json(result.run_dir / "run_metadata.json")
    failed = load_json(result.run_dir / "_FAILED")
    inventory = load_json(result.run_dir / "artifact_inventory.json")
    assert status["status"] == "failed"
    assert status["worker_completion_valid"] is False
    assert reason in status["worker_completion_reason_codes"]
    assert reason in metadata["worker_completion_reason_codes"]
    assert reason in failed["failure_codes"]
    inventory_paths = {entry["path"] for entry in inventory["entries"]}
    assert {
        "run_metadata.json",
        "execution_status.json",
        "logs/worker.stdout.log",
        "logs/worker.stderr.log",
    }.issubset(inventory_paths)
    assert "_FAILED" not in inventory_paths


@pytest.mark.parametrize(
    ("mode", "exit_code"),
    (("python-failure", 1), ("native-like", 77)),
)
def test_supervisor_records_worker_failures(
    repository: tuple[Path, Path],
    mode: str,
    exit_code: int,
) -> None:
    root, config_path = repository
    result = train_supervised(
        config_path,
        root,
        run_id=f"failure-{mode}",
        worker_command_factory=fake_command(mode, exit_code),
        tee_progress=False,
    )
    assert not result.succeeded
    failed = load_json(result.run_dir / "_FAILED")
    status = load_json(result.run_dir / "execution_status.json")
    metadata = load_json(result.run_dir / "run_metadata.json")
    assert failed["child_process_exit_code"] == exit_code
    assert status["child_process_exit_code"] == exit_code
    assert status["windows_exit_code_unsigned"] == exit_code
    assert status["predictor_loading_attempted"] is False
    assert metadata["requested_seed"] == 42
    assert (result.run_dir / "logs" / "worker.stderr.log").is_file()


def test_signed_windows_crash_code_has_unsigned_persisted_representation() -> None:
    assert unsigned_windows_exit_code(-1073741819) == 3221225477


@pytest.mark.skipif(os.name != "nt", reason="native Windows ExitProcess contract")
def test_actual_native_windows_crash_persists_raw_and_unsigned_codes(
    repository: tuple[Path, Path],
) -> None:
    root, config_path = repository
    result = train_supervised(
        config_path,
        root,
        run_id="native-access-violation",
        worker_command_factory=fake_command("native-like", 0xC0000005),
        tee_progress=False,
    )
    status = load_json(result.run_dir / "execution_status.json")
    assert not result.succeeded
    assert status["child_process_exit_code"] in {-1073741819, 3221225477}
    assert status["windows_exit_code_unsigned"] == 3221225477


def test_partial_artifacts_are_preserved_and_reported(
    repository: tuple[Path, Path],
) -> None:
    root, config_path = repository
    result = train_supervised(
        config_path,
        root,
        run_id="partial-native",
        worker_command_factory=fake_command("partial", 91),
        tee_progress=False,
    )
    partial = result.run_dir / "predictor" / "models" / "CompletedFamily" / "model.pkl"
    assert partial.read_text(encoding="utf-8") == "partial model"
    status = load_json(result.run_dir / "execution_status.json")
    metadata = load_json(result.run_dir / "run_metadata.json")
    assert status["last_completed_observable_stage"] == "partial_model_written"
    assert "models/CompletedFamily" in status["discovered_model_directories"]
    assert "simulated native failure" in status["stderr_tail_bounded"]
    assert metadata["requested_seed"] == 42
    assert metadata["profile"]["seed"] == 42
    assert (result.run_dir / "_FAILED").is_file()


class FailingWriter(io.StringIO):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def write(self, value: str) -> int:
        raise OSError(self.message)


class FailingSource:
    def __iter__(self) -> FailingSource:
        return self

    def __next__(self) -> str:
        raise UnicodeError("decode failed")


@pytest.mark.parametrize("stream_name", ("stdout", "stderr"))
def test_pump_captures_writer_failure_and_continues_drain(
    monkeypatch: pytest.MonkeyPatch,
    stream_name: str,
) -> None:
    source = io.StringIO("one\nAUTOGLUON_STAGE:drained\n")
    destination = FailingWriter(f"{stream_name} write failed")
    failures: list[dict[str, str]] = []
    stages: list[str] = []
    monkeypatch.setattr(supervisor_module, "_durable_flush", lambda _stream: None)
    _pump_stream(
        stream_name,
        source,
        destination,
        stages,
        False,
        failures,
        threading.Lock(),
    )
    assert failures[0]["stream"] == stream_name
    assert failures[0]["operation"] == "write"
    assert stages == ["drained"]


def test_pump_thread_propagates_read_failure_to_shared_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failures: list[dict[str, str]] = []
    monkeypatch.setattr(supervisor_module, "_durable_flush", lambda _stream: None)
    thread = threading.Thread(
        target=_pump_stream,
        args=(
            "stdout",
            FailingSource(),
            io.StringIO(),
            [],
            False,
            failures,
            threading.Lock(),
        ),
        daemon=False,
    )
    thread.start()
    thread.join()
    assert not thread.is_alive()
    assert failures == [
        {
            "stream": "stdout",
            "operation": "read",
            "error_type": "UnicodeError",
            "error": "decode failed",
        }
    ]


def test_pump_records_encoding_failure_and_continues_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failures: list[dict[str, str]] = []
    stages: list[str] = []
    destination = io.StringIO()
    monkeypatch.setattr(supervisor_module, "_durable_flush", lambda _stream: None)
    _pump_stream(
        "stdout",
        iter([b"invalid:\xff\n", b"AUTOGLUON_STAGE:after_invalid_utf8\n"]),
        destination,
        stages,
        False,
        failures,
        threading.Lock(),
    )
    assert failures[0]["operation"] == "encoding"
    assert "invalid:" in destination.getvalue()
    assert stages == ["after_invalid_utf8"]


def test_durable_flush_calls_flush_and_fsync(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []

    class Stream:
        def flush(self) -> None:
            calls.append("flush")

        def fileno(self) -> int:
            calls.append("fileno")
            return 123

    monkeypatch.setattr(
        os, "fsync", lambda descriptor: calls.append(("fsync", descriptor))
    )
    _durable_flush(Stream())  # type: ignore[arg-type]
    assert calls == ["flush", "fileno", ("fsync", 123)]


def test_child_zero_exit_plus_log_failure_produces_failed(
    repository: tuple[Path, Path],
) -> None:
    root, config_path = repository

    @contextmanager
    def writer_factory(path: Path) -> Iterator[TextIO]:
        with path.open("w", encoding="utf-8") as stream:
            if path.name == "worker.stdout.log":
                yield FailingWriter("injected stdout failure")
            else:
                yield stream

    result = train_supervised(
        config_path,
        root,
        run_id="log-pump-failure",
        worker_command_factory=fake_command("success"),
        tee_progress=False,
        log_writer_factory=writer_factory,
    )
    status = load_json(result.run_dir / "execution_status.json")
    assert result.child_exit_code == 0
    assert not result.succeeded
    assert status["status"] == "failed"
    assert "log_pump_failure" in status["failure_codes"]
    assert any(
        failure["stream"] == "stdout" and failure["operation"] == "write"
        for failure in status["log_pump_failures"]
    )
    assert (result.run_dir / "_FAILED").is_file()
    assert not (result.run_dir / "_SUCCESS").exists()
