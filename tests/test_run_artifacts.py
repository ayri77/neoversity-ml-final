from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.churn_ml.config import (
    ManualExperimentConfig,
    load_manual_experiment_config,
)
from src.churn_ml.manual_lightgbm import (
    CrossValidationResult,
    EvaluationResult,
    FeatureSchema,
    FoldResult,
    ParityReport,
)
from src.churn_ml.run_artifacts import (
    RunArtifactStore,
    RunFailureFinalizationError,
)
from src.churn_ml.target_encoding import AutoGluonBinaryOOFTargetEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs/experiments/manual_lightgbmprep_r31.yaml"
)


def _artifact_config(tmp_path: Path) -> ManualExperimentConfig:
    baseline = load_manual_experiment_config(
        CONFIG_PATH,
        project_root=PROJECT_ROOT,
    )
    payload = deepcopy(baseline.payload)
    payload["artifacts"]["root"] = str(tmp_path / "runs")
    return ManualExperimentConfig(
        payload=payload,
        source_path=baseline.source_path,
        project_root=baseline.project_root,
    )


def _schema() -> FeatureSchema:
    return FeatureSchema(
        source_feature_names=["num", "cat"],
        dropped_features=[],
        model_feature_names=["num", "cat"],
        categorical_features=["cat"],
        numerical_features=["num"],
        transformed_feature_names=["num", "cat__te"],
    )


def _assignments(train_index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_position": np.arange(len(train_index)),
            "row_index": train_index,
            "fold": np.repeat(np.arange(1, 9), 2),
        }
    )


def _fitted_encoder() -> AutoGluonBinaryOOFTargetEncoder:
    encoder = AutoGluonBinaryOOFTargetEncoder(n_splits=2)
    encoder.fit_transform(
        pd.DataFrame({"num": [0, 1, 2, 3], "cat": ["a", "b", "a", "b"]}),
        pd.Series([0, 1, 0, 1]),
    )
    return encoder


def _fold(fold: int) -> FoldResult:
    validation_positions = np.array([2 * (fold - 1), 2 * (fold - 1) + 1])
    train_positions = np.setdiff1d(np.arange(16), validation_positions)
    return FoldResult(
        fold=fold,
        train_positions=train_positions,
        validation_positions=validation_positions,
        validation_targets=np.array([0, 1]),
        validation_probabilities=np.array([0.1, 0.8]),
        test_probabilities=np.array([0.2, 0.7]),
        balanced_accuracy=1.0,
        duration_seconds=0.01,
        model={"fold": fold, "fitted": True},
        encoder=_fitted_encoder(),
    )


def _make_store(tmp_path: Path, run_id: str = "unit-test-run") -> RunArtifactStore:
    return RunArtifactStore(
        _artifact_config(tmp_path),
        train_index=pd.Index(np.arange(100, 116)),
        test_index=pd.Index([200, 201]),
        run_id=run_id,
    )


def _save_artifacts(
    store: RunArtifactStore,
    *,
    fold_count: int = 8,
    parity_passed: bool = True,
    include_parity: bool = True,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"run_id": store.run_id, "status": "running"}
    assignments = _assignments(pd.Index(store.train_index))
    store.save_initial_contract(metadata, _schema(), assignments)
    for fold in range(1, fold_count + 1):
        store.save_fold(_fold(fold))

    fold_metrics = pd.DataFrame(store.fold_records)
    oof_predictions = pd.DataFrame(
        {
            "row_position": np.arange(16),
            "row_index": store.train_index,
            "fold": assignments["fold"],
            "target": [0, 1] * 8,
            "probability": [0.1, 0.8] * 8,
            "prediction_0_117": [0, 1] * 8,
        }
    )
    test_predictions = pd.DataFrame(
        {
            "row_position": [0, 1],
            "row_index": store.test_index,
            "probability": [0.2, 0.7],
            "prediction_0_117": [1, 1],
        }
    )
    result = CrossValidationResult(
        fold_metrics=fold_metrics,
        fold_assignments=assignments,
        oof_predictions=oof_predictions,
        test_predictions=test_predictions,
        test_probabilities_by_fold=np.tile(np.array([[0.2], [0.7]]), (1, 8)),
        duration_seconds=0.08,
    )
    evaluation = EvaluationResult(
        metrics={"primary": {"balanced_accuracy": 1.0}},
        global_threshold_curve=pd.DataFrame(
            {"threshold": [0.117], "balanced_accuracy": [1.0]}
        ),
        legacy_cross_fitted_fold_metrics=pd.DataFrame(
            {"fold": [1], "balanced_accuracy": [1.0]}
        ),
        legacy_cross_fitted_predictions=pd.DataFrame(
            {"row_position": [0], "probability": [0.1]}
        ),
    )
    parity = (
        ParityReport(
            enabled=True,
            required=True,
            passed=parity_passed,
            primary_gates={"unit_gate": {"passed": parity_passed}},
            secondary_diagnostics={},
        )
        if include_parity
        else None
    )
    submission = pd.DataFrame({"index": [200, 201], "y": [1, 1]})
    store.save_final_outputs(result, evaluation, submission, parity)
    return metadata


def _assert_failed_without_success(store: RunArtifactStore) -> None:
    status = json.loads(
        (store.root / "execution_status.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "failed"
    assert (store.root / "_FAILED").exists()
    assert not (store.root / "_SUCCESS").exists()




def test_store_rejects_overwrite_for_existing_run(tmp_path: Path) -> None:
    store = _make_store(tmp_path)

    with pytest.raises(FileExistsError):
        _make_store(tmp_path)

    assert store.root.exists()
    assert not (store.root / "_SUCCESS").exists()


def test_initial_write_failure_is_failed_and_original_error_reaches_caller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_write_yaml = RunArtifactStore._write_yaml
    failed_once = False

    def fail_once(path: Path, payload: dict[str, Any]) -> None:
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise OSError("injected resolved-config write failure")
        original_write_yaml(path, payload)

    monkeypatch.setattr(
        RunArtifactStore,
        "_write_yaml",
        staticmethod(fail_once),
    )

    with pytest.raises(OSError, match="injected resolved-config write failure"):
        _make_store(tmp_path, run_id="initial-write-failure")

    root = tmp_path / "runs/initial-write-failure"
    assert root.exists()
    assert (root / "models").is_dir()
    assert (root / "_FAILED").exists()
    assert not (root / "_SUCCESS").exists()
    status = json.loads(
        (root / "execution_status.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "failed"
    assert status["failure"]["message"] == "injected resolved-config write failure"


def test_failure_finalization_surfaces_primary_and_secondary_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _make_store(tmp_path)
    primary = ValueError("primary failure")

    def fail_finalization(
        error: BaseException,
        metadata: dict[str, Any],
    ) -> None:
        raise OSError("secondary failure")

    monkeypatch.setattr(store, "fail", fail_finalization)

    with pytest.raises(RunFailureFinalizationError) as caught:
        store.fail_preserving(primary, {"run_id": store.run_id})

    assert caught.value.primary_error is primary
    assert isinstance(caught.value.finalization_error, OSError)
    assert "primary failure" in str(caught.value)
    assert "secondary failure" in str(caught.value)


def test_failed_fold_preserves_partial_model_and_encoder(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    metadata = {"run_id": store.run_id, "status": "running"}
    store.save_initial_contract(
        metadata,
        _schema(),
        _assignments(pd.Index(store.train_index)),
    )
    store.save_fold(_fold(1))

    store.fail(RuntimeError("intentional failure"), metadata)

    _assert_failed_without_success(store)
    assert (store.root / "models/fold_01/model.joblib").exists()
    assert (store.root / "models/fold_01/encoder.joblib").exists()


def test_completion_requires_all_eight_folds(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    metadata = _save_artifacts(store, fold_count=7)

    with pytest.raises(RuntimeError, match="expected 8 fold records") as caught:
        store.complete(metadata)
    store.fail(caught.value, metadata)

    _assert_failed_without_success(store)


@pytest.mark.parametrize("filename", ["model.joblib", "encoder.joblib"])
def test_completion_requires_fold_inference_state(
    tmp_path: Path,
    filename: str,
) -> None:
    store = _make_store(tmp_path)
    metadata = _save_artifacts(store)
    (store.root / "models/fold_04" / filename).unlink()

    with pytest.raises(RuntimeError, match=filename) as caught:
        store.complete(metadata)
    store.fail(caught.value, metadata)

    _assert_failed_without_success(store)


@pytest.mark.parametrize(
    "relative_path",
    [
        "oof_predictions.parquet",
        "metrics.json",
        "submission/manual_lightgbmprep_r31_threshold_0117.csv",
    ],
)
def test_completion_requires_final_artifacts(
    tmp_path: Path,
    relative_path: str,
) -> None:
    store = _make_store(tmp_path)
    metadata = _save_artifacts(store)
    (store.root / relative_path).unlink()

    with pytest.raises(RuntimeError, match="missing required final artifacts") as caught:
        store.complete(metadata)
    store.fail(caught.value, metadata)

    _assert_failed_without_success(store)


@pytest.mark.parametrize(
    ("include_parity", "parity_passed"),
    [(False, False), (True, False)],
)
def test_completion_requires_passing_required_parity(
    tmp_path: Path,
    include_parity: bool,
    parity_passed: bool,
) -> None:
    store = _make_store(tmp_path)
    metadata = _save_artifacts(
        store,
        include_parity=include_parity,
        parity_passed=parity_passed,
    )

    with pytest.raises(RuntimeError, match="parity") as caught:
        store.complete(metadata)
    store.fail(caught.value, metadata)

    _assert_failed_without_success(store)


def test_completion_rejects_existing_failed_marker(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    metadata = _save_artifacts(store)
    store.fail(RuntimeError("earlier failure"), metadata)

    with pytest.raises(RuntimeError, match="_FAILED"):
        store.complete(metadata)

    _assert_failed_without_success(store)


def test_success_inventory_matches_final_filesystem(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    metadata = _save_artifacts(store)

    store.complete(metadata)

    status = json.loads(
        (store.root / "execution_status.json").read_text(encoding="utf-8")
    )
    saved_metadata = json.loads(
        (store.root / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "completed"
    assert (store.root / "_SUCCESS").exists()
    assert not (store.root / "_FAILED").exists()
    assert saved_metadata["artifact_inventory"] == store.inventory()
    assert "_SUCCESS" in saved_metadata["artifact_inventory"]