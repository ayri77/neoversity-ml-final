"""Focused tests for the AutoGluon prediction-candidate adapter."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from src.churn_ml.autogluon_config import load_config
from src.churn_ml.prediction_candidates.autogluon_v1 import (
    AutoGluonRunInventory,
    extract_model_predictions,
    import_autogluon_model_candidate,
    inventory_autogluon_run,
    resolve_positive_probability_column,
    resolve_selected_models,
    validate_autogluon_model_import,
)
from src.churn_ml.prediction_candidates.contract_v1 import (
    CandidateValidationError,
    PredictionCandidateError,
)
from tests.dataset_registry_support import (
    create_synthetic_legacy_tree,
    register_package,
    snapshot_tree,
)
from tests.test_autogluon_config import valid_payload


class FakePredictor:
    def __init__(
        self,
        *,
        class_labels: list[Any] | None = None,
        column_order: list[Any] | None = None,
        models: list[str] | None = None,
        best: str = "WeightedEnsemble_L2",
        support_oof: bool = True,
        oof_fail_models: set[str] | None = None,
    ) -> None:
        self.class_labels = class_labels or [0, 1]
        self.column_order = column_order or list(self.class_labels)
        self.positive_class = 1
        self.model_best = best
        self._models = models or ["LightGBM_BAG_L1", "WeightedEnsemble_L2"]
        self._support_oof = support_oof
        self._oof_fail_models = oof_fail_models or set()
        self.train_predict_called = False
        self.last_test_columns: list[str] | None = None

    def model_names(self) -> list[str]:
        return list(self._models)

    def predict_proba_oof(self, model: str | None = None, **_kwargs: Any) -> pd.DataFrame:
        if not self._support_oof:
            raise AttributeError("predict_proba_oof unavailable")
        name = model or self.model_best
        if name in self._oof_fail_models:
            raise ValueError(f"OOF unsupported for {name}")
        n = 6
        values = np.linspace(0.15, 0.85, n)
        if self.column_order[0] == 1 or self.column_order[0] == "1":
            return pd.DataFrame(
                {self.column_order[0]: values, self.column_order[1]: 1.0 - values},
                index=pd.RangeIndex(n),
            )
        return pd.DataFrame(
            {self.column_order[0]: 1.0 - values, self.column_order[1]: values},
            index=pd.RangeIndex(n),
        )

    def predict_proba(self, data: pd.DataFrame, model: str | None = None, **_kwargs: Any) -> pd.DataFrame:
        self.last_test_columns = list(data.columns)
        if "index" in data.columns or "id" in data.columns:
            raise AssertionError("submission ID entered model input")
        n = len(data)
        values = np.linspace(0.2, 0.7, n)
        if self.column_order[0] == 1 or self.column_order[0] == "1":
            return pd.DataFrame(
                {self.column_order[0]: values, self.column_order[1]: 1.0 - values},
                index=pd.RangeIndex(n),
            )
        return pd.DataFrame(
            {self.column_order[0]: 1.0 - values, self.column_order[1]: values},
            index=pd.RangeIndex(n),
        )

    def model_info(self, model: str) -> dict[str, Any]:
        return {"model_type": "BagModel", "name": model}


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
    return root


def _inventory(repo: Path) -> AutoGluonRunInventory:
    return AutoGluonRunInventory(
        run_dir=repo / "artifacts" / "autogluon_runs" / "fake-run",
        run_path="artifacts/autogluon_runs/fake-run",
        dataset_id="v0_raw_minimal",
        predictor_path="artifacts/autogluon_runs/fake-run/predictor",
        config_path="artifacts/autogluon_runs/fake-run/resolved_config.yaml",
        config_sha256="d" * 64,
        autogluon_version="1.5.0",
        best_model="WeightedEnsemble_L2",
        model_names=("LightGBM_BAG_L1", "WeightedEnsemble_L2"),
        leaderboard=(
            {
                "model": "WeightedEnsemble_L2",
                "score_val": 0.88,
                "eval_metric": "balanced_accuracy",
            },
            {
                "model": "LightGBM_BAG_L1",
                "score_val": 0.8,
                "eval_metric": "balanced_accuracy",
            },
        ),
        classification="complete",
        limitations=(),
    )


def test_explicit_positive_class_selected() -> None:
    frame = pd.DataFrame({0: [0.9, 0.1], 1: [0.1, 0.9]})
    series = resolve_positive_probability_column(
        frame, class_labels=[0, 1], positive_class_label=1
    )
    assert series.tolist() == [0.1, 0.9]


def test_reversed_class_column_order_handled() -> None:
    frame = pd.DataFrame({1: [0.2, 0.8], 0: [0.8, 0.2]})
    series = resolve_positive_probability_column(
        frame, class_labels=[1, 0], positive_class_label=1
    )
    assert series.tolist() == [0.2, 0.8]


def test_missing_positive_class_rejected() -> None:
    frame = pd.DataFrame({0: [0.9, 0.1], 2: [0.1, 0.9]})
    with pytest.raises(CandidateValidationError, match="absent"):
        resolve_positive_probability_column(
            frame, class_labels=[0, 2], positive_class_label=1
        )


def test_best_model_resolution(repo: Path) -> None:
    inventory = _inventory(repo)
    selected = resolve_selected_models(inventory, best=True, models=[])
    assert selected == ("WeightedEnsemble_L2",)


def test_explicit_model_resolution(repo: Path) -> None:
    inventory = _inventory(repo)
    selected = resolve_selected_models(
        inventory, best=False, models=["LightGBM_BAG_L1"]
    )
    assert selected == ("LightGBM_BAG_L1",)


def test_unknown_model_rejected(repo: Path) -> None:
    inventory = _inventory(repo)
    with pytest.raises(PredictionCandidateError, match="Unknown model"):
        resolve_selected_models(inventory, best=False, models=["MissingModel"])


def test_genuine_oof_method_required(repo: Path) -> None:
    inventory = _inventory(repo)

    class NoOofPredictor:
        class_labels = [0, 1]
        positive_class = 1
        model_best = "WeightedEnsemble_L2"

        def predict_proba(self, data: pd.DataFrame, model: str | None = None) -> pd.DataFrame:
            n = len(data)
            return pd.DataFrame({0: np.full(n, 0.4), 1: np.full(n, 0.6)})

    with pytest.raises(PredictionCandidateError, match="predict_proba_oof"):
        extract_model_predictions(
            inventory,
            "WeightedEnsemble_L2",
            repository_root=repo,
            predictor=NoOofPredictor(),
        )


def test_insample_fallback_prohibited(repo: Path) -> None:
    inventory = _inventory(repo)

    class InSampleOnly:
        class_labels = [0, 1]
        positive_class = 1
        model_best = "WeightedEnsemble_L2"

        def predict_proba(self, data: pd.DataFrame, model: str | None = None) -> pd.DataFrame:
            n = len(data)
            return pd.DataFrame({0: np.full(n, 0.4), 1: np.full(n, 0.6)})

    with pytest.raises(PredictionCandidateError, match="predict_proba_oof"):
        extract_model_predictions(
            inventory,
            "WeightedEnsemble_L2",
            repository_root=repo,
            predictor=InSampleOnly(),
        )


def test_test_features_preserve_dataset_package_order(repo: Path) -> None:
    inventory = _inventory(repo)
    predictor = FakePredictor()
    extracted = extract_model_predictions(
        inventory,
        "WeightedEnsemble_L2",
        repository_root=repo,
        predictor=predictor,
    )
    expected = list(
        pd.read_parquet(
            repo / "data" / "processed" / "v0_raw_minimal" / "X_test.parquet"
        ).columns
    )
    assert predictor.last_test_columns == expected
    assert list(extracted.test["row_position"]) == list(range(len(extracted.test)))


def test_submission_id_never_enters_model_input(repo: Path) -> None:
    inventory = _inventory(repo)
    predictor = FakePredictor()
    extract_model_predictions(
        inventory,
        "WeightedEnsemble_L2",
        repository_root=repo,
        predictor=predictor,
    )
    assert predictor.last_test_columns is not None
    assert "index" not in predictor.last_test_columns
    assert "id" not in predictor.last_test_columns


def test_individual_model_limitation_reported(repo: Path) -> None:
    inventory = _inventory(repo)
    predictor = FakePredictor(oof_fail_models={"LightGBM_BAG_L1"})
    with pytest.raises(PredictionCandidateError, match="Individual-model OOF|OOF export"):
        extract_model_predictions(
            inventory,
            "LightGBM_BAG_L1",
            repository_root=repo,
            predictor=predictor,
        )


def test_autogluon_import_is_lazy_and_does_not_break_main_env() -> None:
    module_name = "src.churn_ml.prediction_candidates.autogluon_v1"
    sys.modules.pop(module_name, None)
    # Ensure AutoGluon is not required merely to import the adapter module.
    before = {name for name in sys.modules if name.startswith("autogluon")}
    module = importlib.import_module(module_name)
    after = {name for name in sys.modules if name.startswith("autogluon")}
    assert after == before
    assert module.SOURCE_KIND == "autogluon_standalone_v1"
    # Reloading contract package must still work without AutoGluon.
    importlib.import_module("src.churn_ml.prediction_candidates.contract_v1")


def test_import_best_model_writes_package(repo: Path) -> None:
    run_dir = repo / "artifacts" / "autogluon_runs" / "fake-run"
    run_dir.mkdir(parents=True)
    (run_dir / "predictor").mkdir()
    inventory = _inventory(repo)
    inventory = AutoGluonRunInventory(**{**inventory.__dict__, "run_dir": run_dir})
    predictor = FakePredictor()
    before = snapshot_tree(run_dir)
    package = import_autogluon_model_candidate(
        inventory,
        "WeightedEnsemble_L2",
        repository_root=repo,
        predictor=predictor,
    )
    after = snapshot_tree(run_dir)
    assert before == after
    assert package.manifest["source_model_name"] == "WeightedEnsemble_L2"
    assert package.manifest["oof_protocol"] == "autogluon_bagged_predict_proba_oof"
    validated = validate_autogluon_model_import(
        inventory,
        "WeightedEnsemble_L2",
        repository_root=repo,
        predictor=predictor,
    )
    assert validated["artifacts_written"] is False
    assert validated["candidate_id"] == package.candidate_id


def test_inventory_reads_run_metadata(repo: Path) -> None:
    run_dir = repo / "artifacts" / "autogluon_runs" / "inv-run"
    run_dir.mkdir(parents=True)
    payload = valid_payload()
    payload["dataset"]["version"] = "v0_raw_minimal"
    payload["dataset"]["directory"] = "data/processed/v0_raw_minimal"
    payload["dataset"]["label"] = "y"
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(payload), encoding="utf-8"
    )
    config = load_config(
        run_dir / "resolved_config.yaml", repo, require_data_files=False
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
    (run_dir / "inspection").mkdir()
    pd.DataFrame(
        [
            {
                "model": "WeightedEnsemble_L2",
                "score_val": 0.9,
                "eval_metric": "balanced_accuracy",
            }
        ]
    ).to_csv(run_dir / "inspection" / "leaderboard.csv", index=False)

    inventory = inventory_autogluon_run(run_dir, repository_root=repo)
    assert inventory.best_model == "WeightedEnsemble_L2"
    assert inventory.dataset_id == "v0_raw_minimal"
    assert inventory.config_sha256 == config.identity_sha256
