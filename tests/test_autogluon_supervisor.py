from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from src.churn_ml.autogluon_supervisor import (
    RunAlreadyExistsError,
    train_supervised,
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


def test_supervisor_success_and_success_marker_written_last(
    repository: tuple[Path, Path],
) -> None:
    root, config_path = repository
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
    status = json.loads((result.run_dir / "execution_status.json").read_text())
    assert status["status"] == "succeeded"
    inventory = json.loads((result.run_dir / "artifact_inventory.json").read_text())
    inventory_paths = {entry["path"] for entry in inventory["entries"]}
    assert "_SUCCESS" not in inventory_paths
    marker_time = marker.stat().st_mtime_ns
    assert all(
        path.stat().st_mtime_ns <= marker_time
        for path in result.run_dir.rglob("*")
        if path.is_file() and path != marker
    )
    metadata = json.loads((result.run_dir / "run_metadata.json").read_text())
    command = metadata["local_operational_nonportable"]["worker_command"]
    assert isinstance(command, list)
    assert "repository with spaces" in command[4]


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
    failed = json.loads((result.run_dir / "_FAILED").read_text())
    status = json.loads((result.run_dir / "execution_status.json").read_text())
    assert failed["child_process_exit_code"] == exit_code
    assert status["child_process_exit_code"] == exit_code
    assert status["windows_exit_code_unsigned"] == exit_code
    assert status["predictor_loading_attempted"] is False
    assert (result.run_dir / "logs" / "worker.stderr.log").is_file()


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
    status = json.loads((result.run_dir / "execution_status.json").read_text())
    assert status["last_completed_observable_stage"] == "partial_model_written"
    assert "models/CompletedFamily" in status["discovered_model_directories"]
    assert "simulated native failure" in status["stderr_tail_bounded"]
    assert (result.run_dir / "_FAILED").is_file()
