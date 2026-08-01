"""CLI-focused tests for AutoGluon prediction-candidate import."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.churn_ml.autogluon_config import load_config
from src.churn_ml.prediction_candidates import cli as candidate_cli
from src.churn_ml.prediction_candidates.contract_v1 import (
    CANDIDATE_ROOT_RELATIVE,
    SUCCESS_FILENAME,
)
from tests.dataset_registry_support import (
    create_synthetic_legacy_tree,
    register_package,
    snapshot_tree,
)
from tests.test_autogluon_config import valid_payload
from tests.test_prediction_candidate_autogluon_v1 import FakePredictor


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    processed = root / "data" / "processed"
    create_synthetic_legacy_tree(processed)
    register_package(
        processed,
        "v0_raw_minimal",
        parent_dataset_id=None,
        hypothesis="synthetic v0",
    )
    run_dir = root / "artifacts" / "autogluon_runs" / "cli-run"
    _write_minimal_run(root, run_dir)
    return root


def _write_minimal_run(root: Path, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictor").mkdir(exist_ok=True)
    payload = valid_payload()
    payload["dataset"]["version"] = "v0_raw_minimal"
    payload["dataset"]["directory"] = "data/processed/v0_raw_minimal"
    payload["dataset"]["label"] = "y"
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(payload), encoding="utf-8"
    )
    config = load_config(
        run_dir / "resolved_config.yaml", root, require_data_files=False
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "config_identity_sha256": config.identity_sha256,
                "dataset_version": "v0_raw_minimal",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "worker_result.json").write_text(
        json.dumps(
            {
                "dataset_version": "v0_raw_minimal",
                "best_model": "WeightedEnsemble_L2",
                "model_names": ["LightGBM_BAG_L1", "WeightedEnsemble_L2"],
                "autogluon_version": "1.5.0",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "inspection").mkdir(exist_ok=True)
    pd.DataFrame(
        [
            {
                "model": "WeightedEnsemble_L2",
                "score_val": 0.91,
                "eval_metric": "balanced_accuracy",
            }
        ]
    ).to_csv(run_dir / "inspection" / "leaderboard.csv", index=False)


def test_list_is_read_only(repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = repo / "artifacts" / "autogluon_runs" / "cli-run"
    before = snapshot_tree(repo)
    code = candidate_cli.main(
        ["list", "--run-dir", str(run_dir)],
        repository_root=repo,
    )
    after = snapshot_tree(repo)
    assert code == 0
    assert before == after
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["artifacts_written"] is False
    assert payload["best_model"] == "WeightedEnsemble_L2"


def test_validate_is_read_only(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = repo / "artifacts" / "autogluon_runs" / "cli-run"
    predictor = FakePredictor()

    def _loader(_path: str) -> FakePredictor:
        return predictor

    monkeypatch.setattr(
        candidate_cli,
        "load_predictor",
        lambda predictor_dir, loader=None: _loader(str(predictor_dir)),
    )
    before = snapshot_tree(repo)
    code = candidate_cli.main(
        ["validate", "--run-dir", str(run_dir), "--best"],
        repository_root=repo,
    )
    after = snapshot_tree(repo)
    assert code == 0
    assert before == after
    payload = json.loads(capsys.readouterr().out)
    assert payload["artifacts_written"] is False
    assert payload["results"][0]["ok"] is True


def test_import_writes_success_last(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = repo / "artifacts" / "autogluon_runs" / "cli-run"
    predictor = FakePredictor()
    monkeypatch.setattr(
        candidate_cli,
        "load_predictor",
        lambda predictor_dir, loader=None: predictor,
    )
    code = candidate_cli.main(
        ["import", "--run-dir", str(run_dir), "--best"],
        repository_root=repo,
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["artifacts_written"] is True
    candidate_id = payload["imported"][0]["candidate_id"]
    package_dir = repo / Path(*CANDIDATE_ROOT_RELATIVE.split("/")) / candidate_id
    success = package_dir / SUCCESS_FILENAME
    assert success.is_file()
    success_mtime = success.stat().st_mtime_ns
    for path in package_dir.iterdir():
        if path.name == SUCCESS_FILENAME:
            continue
        assert path.stat().st_mtime_ns <= success_mtime


def test_inspect_returns_source_and_identity_summary(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = repo / "artifacts" / "autogluon_runs" / "cli-run"
    predictor = FakePredictor()
    monkeypatch.setattr(
        candidate_cli,
        "load_predictor",
        lambda predictor_dir, loader=None: predictor,
    )
    assert (
        candidate_cli.main(
            ["import", "--run-dir", str(run_dir), "--best"],
            repository_root=repo,
        )
        == 0
    )
    imported = json.loads(capsys.readouterr().out)["imported"][0]
    code = candidate_cli.main(
        ["inspect", "--candidate-id", imported["candidate_id"]],
        repository_root=repo,
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["source"]["source_model_name"] == "WeightedEnsemble_L2"
    assert payload["identity"]["dataset_id"] == "v0_raw_minimal"
    assert "train_anchor_hash" in payload["identity"]
    assert "probability_range" in payload
