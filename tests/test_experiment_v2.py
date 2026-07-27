from __future__ import annotations

import argparse
import shutil
import socket
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from src.churn_ml import research_v2_cli
from src.churn_ml.experiment_v2 import (
    ExperimentV2ContractError,
    candidate_adapter_registry,
    feature_pipeline_registry,
    get_candidate_adapter,
    get_feature_pipeline,
    validate_positive_class_probabilities,
)
from src.churn_ml.research_manual_lightgbm import fit_predict_manual_candidate
from src.churn_ml.research_protocol import (
    build_evaluation_assignments,
    select_balanced_accuracy_threshold,
)
from src.churn_ml.research_v2_artifact_validation import (
    validate_portable_payload_paths,
)
from src.churn_ml.research_v2_artifacts import ResearchV2ArtifactStore
from src.churn_ml.research_v2_config import (
    ResearchV2ConfigurationError,
    load_research_v2_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = (
    PROJECT_ROOT / "configs/research_v2/manual_lightgbm_te_v1_compat_smoke.yaml"
)
DEVELOPMENT_CONFIG = (
    PROJECT_ROOT / "configs/research_v2/manual_lightgbm_te_v1_compat_development.yaml"
)
TRAIN_PATH = PROJECT_ROOT / "data/processed/v3_targeted_missingness/X_train.parquet"


def test_registries_are_explicit_read_only_and_reject_unknown_ids() -> None:
    assert set(feature_pipeline_registry()) == {"manual_v3_pipeline_v1_compat"}
    assert set(candidate_adapter_registry()) == {
        "manual_lightgbm_te_v1_compat",
        "xgboost_numeric_v1",
        "catboost_numeric_v1",
    }
    with pytest.raises(TypeError):
        feature_pipeline_registry()["new"] = object()  # type: ignore[index,assignment]
    with pytest.raises(ExperimentV2ContractError, match="Unknown feature pipeline"):
        get_feature_pipeline("unknown")
    with pytest.raises(ExperimentV2ContractError, match="Unknown candidate adapter"):
        get_candidate_adapter("unknown")


@pytest.mark.parametrize(
    "values,rows",
    [
        ([0.1, np.nan], 2),
        ([0.1, np.inf], 2),
        ([-0.1, 0.2], 2),
        ([0.1, 1.1], 2),
        ([[0.1], [0.2]], 2),
        ([0.1], 2),
    ],
)
def test_probability_validation_rejects_invalid_outputs(
    values: Any,
    rows: int,
) -> None:
    with pytest.raises(ExperimentV2ContractError):
        validate_positive_class_probabilities(values, expected_rows=rows)


def test_probability_validation_accepts_finite_positive_class_vector() -> None:
    actual = validate_positive_class_probabilities([0.0, 0.5, 1.0], expected_rows=3)
    np.testing.assert_array_equal(actual, np.array([0.0, 0.5, 1.0]))


def test_smoke_and_development_plan_contracts_are_fixed() -> None:
    smoke = load_research_v2_config(SMOKE_CONFIG, project_root=PROJECT_ROOT)
    development = load_research_v2_config(
        DEVELOPMENT_CONFIG,
        project_root=PROJECT_ROOT,
    )
    assert smoke.plan_payload["outer_evaluation"] == {
        "splitter": "stratified_kfold",
        "n_splits": 3,
        "shuffle": True,
        "repeat_seeds": [0],
    }
    assert smoke.plan_payload["threshold_selection"]["n_splits"] == 2
    assert development.plan_payload["outer_evaluation"] == {
        "splitter": "stratified_kfold",
        "n_splits": 5,
        "shuffle": True,
        "repeat_seeds": [0, 17],
    }
    assert development.plan_payload["threshold_selection"]["n_splits"] == 3
    assert (
        smoke.plan_payload["threshold_policy"]
        == development.plan_payload["threshold_policy"]
    )


def test_assignments_are_deterministic_and_fold_complete() -> None:
    config = load_research_v2_config(SMOKE_CONFIG, project_root=PROJECT_ROOT)
    y = pd.Series(([0, 1] * 30), dtype="int64")
    first = build_evaluation_assignments(y, config.plan_payload)
    second = build_evaluation_assignments(y, config.plan_payload)
    pd.testing.assert_frame_equal(first.outer, second.outer)
    pd.testing.assert_frame_equal(
        first.threshold_selection,
        second.threshold_selection,
    )
    assert first.outer["row_position"].value_counts().eq(1).all()


def test_threshold_comparison_is_exact_greater_than_or_equal() -> None:
    policy = {
        "minimum": 0.1,
        "maximum": 0.9,
        "step": 0.1,
        "maximizer_absolute_tolerance": 1e-12,
        "constant_probability_fallback": 0.5,
    }
    result = select_balanced_accuracy_threshold(
        np.array([0, 1]),
        np.array([0.4, 0.5]),
        policy,
    )
    predictions = (np.array([0.4, 0.5]) >= result.threshold).astype("int8")
    assert predictions.tolist() == [0, 1]
    assert result.threshold == 0.5


def test_config_is_recursive_strict_and_rejects_traversal(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
    payload["candidate_adapter"]["contract"]["target_encoder"]["unknown"] = True
    strict_path = PROJECT_ROOT / "configs/research_v2/_test_invalid_strict.yaml"
    traversal_path = PROJECT_ROOT / "configs/research_v2/_test_invalid_path.yaml"
    try:
        strict_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        with pytest.raises(ResearchV2ConfigurationError, match="unknown"):
            load_research_v2_config(strict_path, project_root=PROJECT_ROOT)
        payload = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
        payload["artifacts"]["root"] = "../escape"
        traversal_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        with pytest.raises(
            ResearchV2ConfigurationError,
            match="repository-relative",
        ):
            load_research_v2_config(traversal_path, project_root=PROJECT_ROOT)
    finally:
        strict_path.unlink(missing_ok=True)
        traversal_path.unlink(missing_ok=True)


def test_network_guard_denies_and_restores_sockets() -> None:
    original = socket.socket
    with research_v2_cli.network_disabled():
        with pytest.raises(RuntimeError, match="prohibited"):
            socket.socket()
    assert socket.socket is original


@pytest.mark.skipif(not TRAIN_PATH.exists(), reason="Local v3 data unavailable.")
def test_preflight_identities_are_deterministic_portable_and_train_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    original_read_parquet = pd.read_parquet
    original_open = Path.open

    def read_parquet(path: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(str(path))
        return original_read_parquet(path, *args, **kwargs)

    def path_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        opened.append(str(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read_parquet)
    monkeypatch.setattr(Path, "open", path_open)
    first = research_v2_cli.preflight(SMOKE_CONFIG)
    second = research_v2_cli.preflight(SMOKE_CONFIG)
    assert first.hashes == second.hashes
    validate_portable_payload_paths(
        first.identities,
        label="preflight.identities",
        project_root=PROJECT_ROOT,
    )
    normalized = [name.replace("\\", "/").lower() for name in opened]
    forbidden = (
        "x_test.parquet",
        "sample_submission",
        "/submissions/",
        "artifacts/reference",
        "artifacts/autogluon",
        "kaggle",
    )
    assert not any(marker in name for name in normalized for marker in forbidden)


@pytest.mark.skipif(not TRAIN_PATH.exists(), reason="Local v3 data unavailable.")
def test_validate_only_allocates_no_run(tmp_path: Path) -> None:
    payload = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
    artifact_root = PROJECT_ROOT / "artifacts/research_v2_tests" / tmp_path.name
    artifact_relative = artifact_root.relative_to(PROJECT_ROOT).as_posix()
    payload["artifacts"]["root"] = artifact_relative
    config_path = PROJECT_ROOT / "configs/research_v2/_test_validate_only.yaml"
    try:
        config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        before = list(artifact_root.rglob("*")) if artifact_root.exists() else []
        code = research_v2_cli.execute(
            argparse.Namespace(
                config=config_path,
                validate_only=True,
                run_id=None,
            )
        )
        assert code == 0
        after = list(artifact_root.rglob("*")) if artifact_root.exists() else []
        assert after == before
    finally:
        config_path.unlink(missing_ok=True)
        shutil.rmtree(artifact_root, ignore_errors=True)


@pytest.mark.skipif(not TRAIN_PATH.exists(), reason="Local v3 data unavailable.")
def test_v2_pipeline_and_adapter_match_v1_on_identical_partition() -> None:
    prepared = research_v2_cli.preflight(SMOKE_CONFIG)
    training = np.arange(0, 600, dtype=np.int64)
    prediction = np.arange(600, 700, dtype=np.int64)
    X_train = prepared.data.X.iloc[training]
    y_train = prepared.data.y.iloc[training]
    X_prediction = prepared.data.X.iloc[prediction]
    legacy = fit_predict_manual_candidate(
        X_train,
        y_train,
        X_prediction,
        prepared.config.adapter_contract,
        model_training_positions=training,
        prediction_positions=prediction,
    )
    audit_calls: list[tuple[set[int], set[int]]] = []

    def audit(train: np.ndarray, predict: np.ndarray) -> None:
        train_set = set(train.tolist())
        predict_set = set(predict.tolist())
        assert not train_set & predict_set
        audit_calls.append((train_set, predict_set))

    v2 = prepared.adapter.fit_predict(
        X_train,
        y_train,
        X_prediction,
        prepared.config.adapter_contract,
        model_training_positions=training,
        prediction_positions=prediction,
        audit_callback=audit,
    )
    np.testing.assert_array_equal(v2, legacy)
    assert len(audit_calls) == 1
    assert (
        prepared.data.pipeline_output.schema.model_input_schema_sha256
        == "e19efe336426c09071db296920353a0b967284a1288a1d9ab54aaa83ca179616"
    )
    assert list(X_train.columns) == list(X_prediction.columns)


def test_failed_lifecycle_writes_failed_and_never_success(
    tmp_path: Path,
) -> None:
    config = load_research_v2_config(SMOKE_CONFIG, project_root=PROJECT_ROOT)
    artifact_root = PROJECT_ROOT / "artifacts/research_v2_tests" / tmp_path.name
    payload = dict(config.payload)
    payload["artifacts"] = {"root": artifact_root.relative_to(PROJECT_ROOT).as_posix()}
    test_config = type(config)(
        payload=payload,
        plan_payload=config.plan_payload,
        source_path=config.source_path,
        plan_path=config.plan_path,
        project_root=config.project_root,
    )
    try:
        store = ResearchV2ArtifactStore(
            test_config,
            plan_hash="a" * 64,
            candidate_hash="b" * 64,
            run_id="failed-unit",
        )
        store.fail(RuntimeError("intentional"), {"status": "running"})
        assert (store.root / "_FAILED").is_file()
        assert not (store.root / "_SUCCESS").exists()
    finally:
        shutil.rmtree(artifact_root, ignore_errors=True)
