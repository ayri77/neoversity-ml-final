from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import yaml

from src.churn_ml.deployment_v1 import execute_deployment
from src.churn_ml.deployment_v1_auth import (
    DeploymentAuthenticationError,
    authenticate_source,
    load_threshold_evidence,
    derive_approval_id,
    load_synthetic_fixture,
)
from src.churn_ml.deployment_v1_artifacts import (
    DeploymentArtifactError,
    DeploymentArtifactStore,
    build_inventory,
    build_manifest,
    build_submission,
    load_completed_deployment,
    validate_deployment_artifacts,
)
from src.churn_ml.deployment_v1_contracts import (
    CandidateApproval,
    DeploymentConfig,
    DeploymentContractError,
    ValidatedDeployment,
    load_candidate_approval,
    load_deployment_config,
    validate_deployment,
)
from src.churn_ml.deployment_v1_features import (
    DeploymentData,
    DeploymentDataError,
    _validate_sample_and_alignment,
    build_full_data_encoding,
)
from src.churn_ml.deployment_v1_paths import (
    DeploymentPathError,
    prewalk_regular_tree,
    validate_new_path,
    validate_path_chain,
    validate_regular_file,
)
from src.churn_ml.deployment_v1_models import (
    build_fixed_blend,
    classify_fixed,
    fit_component_bags,
)
from src.churn_ml.experiment_v2_adapter import EXPECTED_ADAPTER_CONTRACT
from src.churn_ml.research_data import canonical_sha256
from tests.deployment_v1_loader_support import build_production_loader_fixture


def test_production_loader_reconstructs_fully_authenticated_tiny_deployment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_production_loader_fixture(tmp_path.name, monkeypatch)
    try:
        assert fixture.loaded.root == fixture.output
        assert fixture.loaded.prediction_summary["row_count"] == 3
    finally:
        fixture.cleanup()


def test_config_is_exact_and_rejects_bool_numbers_and_optimization(
    tmp_path: Path,
) -> None:
    payload = _config_payload()
    path = _write_yaml(tmp_path / "config.yaml", payload)
    loaded = load_deployment_config(path, project_root=tmp_path)
    assert loaded.deployment_id == "synthetic-deployment"

    invalid_cases = []
    unknown = deepcopy(payload)
    unknown["unexpected"] = True
    invalid_cases.append(unknown)
    bool_seed = deepcopy(payload)
    bool_seed["components"][0]["bag_seeds"] = [True]
    invalid_cases.append(bool_seed)
    bad_weight = deepcopy(payload)
    bad_weight["components"][0]["component_weight"] = 0.9
    invalid_cases.append(bad_weight)
    bad_threshold = deepcopy(payload)
    bad_threshold["threshold"]["value"] = 1.1
    invalid_cases.append(bad_threshold)
    optimization = deepcopy(payload)
    optimization["components"][0]["fixed_parameters"] = {
        "optuna_distribution": "uniform"
    }
    invalid_cases.append(optimization)
    escaped = deepcopy(payload)
    escaped["test_data"]["path"] = "../competition.parquet"
    invalid_cases.append(escaped)
    duplicate = deepcopy(payload)
    duplicate["components"].append(deepcopy(duplicate["components"][0]))
    duplicate["components"][0]["component_weight"] = 0.5
    duplicate["components"][1]["component_weight"] = 0.5
    invalid_cases.append(duplicate)

    for index, invalid in enumerate(invalid_cases):
        invalid_path = _write_yaml(tmp_path / f"invalid-{index}.yaml", invalid)
        with pytest.raises(DeploymentContractError):
            load_deployment_config(invalid_path, project_root=tmp_path)


def test_approval_authenticates_completed_run_and_rejects_revocation(
    tmp_path: Path,
) -> None:
    run = _fake_run(tmp_path)
    approval_path, payload = _approval_file(tmp_path, run)

    def loader(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return run

    approval = load_candidate_approval(
        approval_path,
        project_root=tmp_path,
        research_loader=loader,
    )
    assert approval.adapter_id == "manual_lightgbm_te_v1_compat"
    assert approval.research_run is run

    revoked = deepcopy(payload)
    revoked["manual_approval"]["status"] = "revoked"
    revoked_path = _write_yaml(tmp_path / "approvals/revoked.yaml", revoked)
    with pytest.raises(DeploymentContractError, match="approved"):
        load_candidate_approval(
            revoked_path,
            project_root=tmp_path,
            research_loader=loader,
        )

    mismatch = deepcopy(payload)
    mismatch["research_run"]["manifest_sha256"] = "f" * 64
    mismatch_path = _write_yaml(tmp_path / "approvals/mismatch.yaml", mismatch)
    with pytest.raises(DeploymentContractError, match="manifest_sha256"):
        load_candidate_approval(
            mismatch_path,
            project_root=tmp_path,
            research_loader=loader,
        )


def test_validate_never_reads_competition_paths(tmp_path: Path) -> None:
    run = _fake_run(tmp_path)
    approval_path, approval_payload = _approval_file(tmp_path, run)
    payload = _config_payload()
    payload["components"][0]["approval_artifact_path"] = approval_path.relative_to(
        tmp_path
    ).as_posix()
    payload["components"][0]["fixed_parameters"] = deepcopy(
        cast(dict[str, Any], EXPECTED_ADAPTER_CONTRACT["lightgbm"])["parameters"]
    )
    payload["threshold"]["evidence"] = deepcopy(approval_payload["threshold_evidence"])
    config_path = _write_yaml(tmp_path / "deployment.yaml", payload)

    def loader(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return run

    validated = validate_deployment(
        config_path,
        project_root=tmp_path,
        research_loader=loader,
    )
    assert validated.config.deployment_id == "synthetic-deployment"
    assert not (tmp_path / "competition/never-read.parquet").exists()
    assert not (tmp_path / "competition/never-read.csv").exists()
    assert not (tmp_path / "artifacts/deployments").exists()


def test_full_data_encoding_is_oof_and_uses_full_mapping_for_test() -> None:
    train = pd.DataFrame(
        {
            "index": np.arange(12, dtype=np.int64),
            "number": [np.nan, 1.0, 2.0, 3.0] * 3,
            "category": ["a", "a", "b", None] * 3,
        }
    )
    target = pd.Series([0, 1] * 6, name="y", dtype="int8")
    test = pd.DataFrame(
        {
            "index": [100, 101, 102],
            "number": [np.nan, 4.0, 5.0],
            "category": ["a", "unknown", None],
        }
    )
    contract = {
        "implementation": "custom_autogluon_compatible_binary_oof_target_encoder",
        "inner_splits": 3,
        "shuffle": True,
        "random_state": 42,
        "alpha": 10.0,
        "prior": "unweighted_mean_of_category_level_target_means",
        "keep_original_categorical_features": False,
    }
    first = build_full_data_encoding(train, target, test, contract)
    second = build_full_data_encoding(train, target, test, contract)
    pd.testing.assert_frame_equal(first.train, second.train, check_exact=True)
    pd.testing.assert_frame_equal(first.test, second.test, check_exact=True)
    pd.testing.assert_frame_equal(
        first.assignments, second.assignments, check_exact=True
    )
    assert first.assignments["encoding_fold"].nunique() == 3
    assert np.isnan(first.train.loc[0, "number"])
    assert np.isnan(first.test.loc[0, "number"])
    assert np.isfinite(first.test["category__te"]).all()
    assert first.identity["test_fit_performed"] is False
    assert first.identity["imputation_performed"] is False
    assert first.identity["sha256"] == second.identity["sha256"]


def test_seed_bagging_uses_all_rows_and_exact_average(tmp_path: Path) -> None:
    run = _fake_run(tmp_path)
    approval_path, payload = _approval_file(tmp_path, run)
    approval = CandidateApproval(
        payload=payload,
        source_path=approval_path,
        source_sha256=hashlib.sha256(approval_path.read_bytes()).hexdigest(),
        research_run=run,
    )
    train = pd.DataFrame({"x": np.arange(10, dtype=float)})
    test = pd.DataFrame({"x": [10.0, 11.0, 12.0]})
    target = pd.Series([0, 1] * 5, dtype="int8")
    encoding = SimpleNamespace(train=train, test=test)
    fitted_rows: list[int] = []

    class FakeEstimator:
        def __init__(self, seed: int) -> None:
            self.seed = seed
            self.classes_ = np.array([1, 0])

        def fit(self, features: pd.DataFrame, labels: pd.Series) -> None:
            fitted_rows.append(len(features))
            assert len(features) == len(labels) == 10

        def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
            positive = np.full(len(features), 0.2 + self.seed / 100.0)
            return np.column_stack([positive, 1.0 - positive])

    def factory(adapter_id: str, parameters: dict[str, Any]) -> FakeEstimator:
        assert adapter_id == "manual_lightgbm_te_v1_compat"
        return FakeEstimator(int(parameters["random_state"]))

    component = {
        "component_id": "manual-component",
        "adapter_id": "manual_lightgbm_te_v1_compat",
        "fixed_parameters": deepcopy(
            cast(dict[str, Any], EXPECTED_ADAPTER_CONTRACT["lightgbm"])["parameters"]
        ),
        "bag_seeds": [1, 3],
        "component_weight": 1.0,
    }
    first = fit_component_bags(
        component,
        approval,
        encoding,  # type: ignore[arg-type]
        target,
        row_keys=(100, 101, 102),
        estimator_factory=factory,
    )
    second = fit_component_bags(
        component,
        approval,
        encoding,  # type: ignore[arg-type]
        target,
        row_keys=(100, 101, 102),
        estimator_factory=factory,
    )
    assert fitted_rows == [10, 10, 10, 10]
    np.testing.assert_allclose(
        first.probabilities, np.full(3, 0.22), rtol=0.0, atol=1e-15
    )
    np.testing.assert_array_equal(first.probabilities, second.probabilities)
    assert all(record["evaluation_set"] is False for record in first.bag_records)
    assert all(record["early_stopping"] is False for record in first.bag_records)
    assert all(record["model_persisted"] is False for record in first.bag_records)


def test_fixed_blend_threshold_and_submission_round_trip() -> None:
    row_keys = (9, 8)
    row_identity = canonical_sha256(list(row_keys))
    left = SimpleNamespace(
        component_id="left",
        probabilities=np.array([0.25, 0.5]),
        row_keys=row_keys,
        row_identity_sha256=row_identity,
    )
    right = SimpleNamespace(
        component_id="right",
        probabilities=np.array([0.75, 0.0]),
        row_keys=row_keys,
        row_identity_sha256=row_identity,
    )
    blend = build_fixed_blend(
        [left, right],  # type: ignore[list-item]
        {"left": 0.5, "right": 0.5},
    )
    np.testing.assert_array_equal(blend, np.array([0.5, 0.25]))
    labels = classify_fixed(blend, 0.5)
    np.testing.assert_array_equal(labels, np.array([1, 0], dtype=np.int8))
    sample = pd.DataFrame({"index": [9, 8], "y": [0, 0]})
    submission = build_submission(
        sample,
        labels,
        id_column="index",
        target_column="y",
    )
    assert submission.to_csv(index=False, lineterminator="\n") == "index,y\n9,1\n8,0\n"
    assert submission.columns.tolist() == ["index", "y"]


def test_synthetic_lifecycle_success_failure_and_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated, data, fixture = _execution_fixture(tmp_path)
    monkeypatch.setattr(
        "src.churn_ml.deployment_v1.load_deployment_data",
        lambda *args, **kwargs: data,
    )
    monkeypatch.setattr(
        "src.churn_ml.deployment_v1.source_provenance",
        lambda root: {"schema_version": 1, "root": str(root)},
    )
    monkeypatch.setattr(
        "src.churn_ml.deployment_v1_artifacts.source_provenance",
        lambda root: {"schema_version": 1, "root": str(root)},
    )

    class Estimator:
        classes_ = np.array([0, 1])

        def __init__(self, seed: int) -> None:
            self.seed = seed

        def fit(self, features: pd.DataFrame, labels: pd.Series) -> None:
            assert len(features) == len(labels)

        def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
            positive = np.linspace(0.3, 0.7, len(features)) + self.seed * 0.0
            return np.column_stack([1.0 - positive, positive])

    def factory(adapter_id: str, parameters: dict[str, Any]) -> Estimator:
        del adapter_id
        return Estimator(int(parameters["random_state"]))

    output = tmp_path / "dry-output"
    result = execute_deployment(
        validated,
        mode="dry-run",
        output_dir=output,
        fixture_dir=fixture,
        estimator_factory=factory,
    )
    assert result.root == output
    assert (output / "_SUCCESS").is_file()
    assert not (output / "_FAILED").exists()
    assert not list(output.rglob("*.joblib"))
    validate_deployment_artifacts(
        output,
        validated=validated,
        require_success=True,
        verify_manifest=True,
        data=data,
    )
    with pytest.raises(FileExistsError):
        execute_deployment(
            validated,
            mode="dry-run",
            output_dir=output,
            fixture_dir=fixture,
            estimator_factory=factory,
        )

    component_path = output / "component_probabilities.parquet"
    original_component = pd.read_parquet(component_path)
    tampered = original_component.copy()
    tampered.loc[0, "manual-component"] = 0.99
    tampered.to_parquet(component_path, index=False)
    _reauthenticate_completed_tree(output)
    with pytest.raises(DeploymentArtifactError):
        validate_deployment_artifacts(
            output,
            validated=validated,
            require_success=True,
            verify_manifest=True,
            data=data,
        )

    original_component.to_parquet(component_path, index=False)
    submission_path = output / "submission.csv"
    rows = submission_path.read_text(encoding="utf-8").splitlines()
    first_id = rows[1].split(",", maxsplit=1)[0]
    rows[1] = f"{first_id},0.5"
    submission_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    _reauthenticate_completed_tree(output)
    with pytest.raises(DeploymentArtifactError, match="canonical"):
        validate_deployment_artifacts(
            output,
            validated=validated,
            require_success=True,
            verify_manifest=True,
            data=data,
        )

    failed_root = tmp_path / "failed-output"
    store = DeploymentArtifactStore(failed_root)
    store.fail(RuntimeError("synthetic failure"))
    assert (failed_root / "_FAILED").is_file()
    assert not (failed_root / "_SUCCESS").exists()


@pytest.mark.parametrize(
    "corruption",
    [
        "duplicate_encoding_assignment",
        "missing_encoding_assignment",
        "forged_encoding_identity",
        "forged_fold_assignment",
        "forged_bag_seed",
        "forged_bag_probability_hash",
        "altered_component_average",
        "altered_blend_probability",
        "altered_threshold_summary",
        "nonbinary_submission",
        "wrong_train_schema",
        "wrong_feature_identity",
        "forged_runtime",
        "forged_environment",
    ],
)
def test_completed_semantics_reject_coherently_remanifested_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    validated, data, fixture = _execution_fixture(tmp_path)
    monkeypatch.setattr(
        "src.churn_ml.deployment_v1.load_deployment_data",
        lambda *args, **kwargs: data,
    )

    def provenance(root: Path) -> dict[str, Any]:
        return {"schema_version": 1, "root": str(root)}

    monkeypatch.setattr("src.churn_ml.deployment_v1.source_provenance", provenance)
    monkeypatch.setattr(
        "src.churn_ml.deployment_v1_artifacts.source_provenance", provenance
    )

    class Estimator:
        classes_ = np.array([0, 1])

        def fit(self, features: pd.DataFrame, labels: pd.Series) -> None:
            assert len(features) == len(labels)

        def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
            positive = np.linspace(0.3, 0.7, len(features))
            return np.column_stack([1.0 - positive, positive])

    output = tmp_path / "completed"
    execute_deployment(
        validated,
        mode="dry-run",
        output_dir=output,
        fixture_dir=fixture,
        estimator_factory=lambda *_: Estimator(),
    )

    if corruption in {
        "duplicate_encoding_assignment",
        "missing_encoding_assignment",
        "forged_fold_assignment",
    }:
        path = output / "encoding_assignments.parquet"
        frame = pd.read_parquet(path)
        if corruption == "duplicate_encoding_assignment":
            frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        elif corruption == "missing_encoding_assignment":
            frame = frame.iloc[1:].reset_index(drop=True)
        else:
            frame.loc[0, "encoding_fold"] = (int(frame.loc[0, "encoding_fold"]) % 3) + 1
        frame.to_parquet(path, index=False)
    elif corruption == "forged_encoding_identity":
        path = output / "encoding_identity.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["components"]["manual-component"]["test_fit_performed"] = True
        _write_json(path, payload)
    elif corruption in {"forged_bag_seed", "forged_bag_probability_hash"}:
        path = output / "bag_summary.csv"
        frame = pd.read_csv(path)
        if corruption == "forged_bag_seed":
            frame.loc[0, "bag_seed"] = 999
        else:
            frame.loc[0, "probability_sha256"] = "f" * 64
        frame.to_csv(path, index=False)
    elif corruption == "altered_component_average":
        path = output / "component_probabilities.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "manual-component"] = 0.99
        frame.to_parquet(path, index=False)
    elif corruption == "altered_blend_probability":
        path = output / "blend_probabilities.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "probability"] = 0.99
        frame.to_parquet(path, index=False)
    elif corruption == "altered_threshold_summary":
        path = output / "prediction_summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["threshold"] = 0.25
        _write_json(path, payload)
    elif corruption == "nonbinary_submission":
        path = output / "submission.csv"
        rows = path.read_text(encoding="utf-8").splitlines()
        identifier = rows[1].split(",", maxsplit=1)[0]
        rows[1] = f"{identifier},0.5"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    elif corruption == "wrong_train_schema":
        path = output / "train_schema.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["source_rows"] = 999
        _write_json(path, payload)
    elif corruption == "wrong_feature_identity":
        path = output / "feature_identity.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["pipeline_id"] = "forged"
        _write_json(path, payload)
    elif corruption == "forged_runtime":
        path = output / "runtime.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["network_access"] = True
        _write_json(path, payload)
    elif corruption == "forged_environment":
        path = output / "environment.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["python"] = "0.0.0"
        _write_json(path, payload)
    else:  # pragma: no cover - guards the explicit parameter allowlist.
        raise AssertionError(corruption)

    _reauthenticate_completed_tree(output)
    with pytest.raises(DeploymentArtifactError):
        validate_deployment_artifacts(
            output,
            validated=validated,
            require_success=True,
            verify_manifest=True,
            data=data,
        )


PRODUCTION_CORRUPTIONS = (
    "duplicate_encoding_assignment",
    "missing_encoding_assignment",
    "forged_encoding_identity",
    "forged_fold_assignment",
    "forged_bag_seed",
    "forged_bag_probability_hash",
    "altered_component_average",
    "altered_blend_probability",
    "altered_threshold_summary",
    "nonbinary_submission",
    "wrong_train_schema",
    "wrong_feature_identity",
    "forged_runtime",
    "forged_environment",
    "float_encoding_row_position",
    "bool_component_row_position",
    "string_bag_row_position",
    "float_blend_row_position",
    "float_bag_index",
    "bool_bag_index",
    "forged_threshold_label",
    "float_label",
    "string_label",
    "bool_label",
    "null_label",
    "arbitrary_duration",
    "malformed_start_timestamp",
    "malformed_completion_timestamp",
    "non_utc_timestamp",
    "completion_before_start",
    "malformed_success_timestamp",
    "success_completion_mismatch",
    "success_wrong_deployment_id",
    "success_wrong_manifest",
    "both_terminal_markers",
    "terminal_not_newest",
    "quoted_numeric_id",
    "crlf_submission",
    "bom_submission",
    "whitespace_submission",
    "trailing_blank_submission",
    "alternate_quoting_submission",
    "float_text_label",
    "reordered_submission_columns",
)


def test_production_loader_rejects_coherently_remanifested_corruption_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_production_loader_fixture(tmp_path.name, monkeypatch)
    try:
        for index, corruption in enumerate(PRODUCTION_CORRUPTIONS):
            root = fixture.root / f"corruption-{index:02d}"
            shutil.copytree(fixture.output, root)
            _mutate_completed_deployment(root, corruption)
            if corruption not in {
                "malformed_success_timestamp",
                "success_completion_mismatch",
                "success_wrong_deployment_id",
                "success_wrong_manifest",
                "both_terminal_markers",
                "terminal_not_newest",
            }:
                _reauthenticate_completed_tree(root)
            else:
                _reauthenticate_completed_tree(root)
                _mutate_terminal(root, corruption)
            with pytest.raises(DeploymentArtifactError):
                load_completed_deployment(root, project_root=fixture.root.parents[2])
    finally:
        fixture.cleanup()


def _mutate_completed_deployment(root: Path, corruption: str) -> None:
    if corruption in {
        "duplicate_encoding_assignment",
        "missing_encoding_assignment",
        "forged_fold_assignment",
        "float_encoding_row_position",
    }:
        path = root / "encoding_assignments.parquet"
        frame = pd.read_parquet(path)
        if corruption == "duplicate_encoding_assignment":
            frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        elif corruption == "missing_encoding_assignment":
            frame = frame.iloc[1:].reset_index(drop=True)
        elif corruption == "forged_fold_assignment":
            frame.loc[0, "encoding_fold"] = 99
        else:
            frame["row_position"] = frame["row_position"].astype("float64")
        frame.to_parquet(path, index=False)
    elif corruption == "forged_encoding_identity":
        path = root / "encoding_identity.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["components"]["manual-component"]["test_fit_performed"] = True
        _write_json(path, payload)
    elif corruption in {
        "forged_bag_seed",
        "forged_bag_probability_hash",
        "float_bag_index",
        "bool_bag_index",
    }:
        path = root / "bag_summary.csv"
        rows = path.read_text(encoding="utf-8").splitlines()
        header = rows[0].split(",")
        values = rows[1].split(",")
        if corruption == "forged_bag_seed":
            values[header.index("bag_seed")] = "999"
        elif corruption == "forged_bag_probability_hash":
            values[header.index("probability_sha256")] = "f" * 64
        elif corruption == "float_bag_index":
            values[header.index("bag_index")] = "1.0"
        else:
            values[header.index("bag_index")] = "True"
        path.write_text(
            ",".join(header) + "\n" + ",".join(values) + "\n", encoding="utf-8"
        )
    elif corruption in {"altered_component_average", "bool_component_row_position"}:
        path = root / "component_probabilities.parquet"
        frame = pd.read_parquet(path)
        if corruption == "altered_component_average":
            frame.loc[0, "manual-component"] = 0.99
        else:
            frame["row_position"] = frame["row_position"].astype("bool")
        frame.to_parquet(path, index=False)
    elif corruption == "string_bag_row_position":
        path = root / "bag_probabilities.parquet"
        frame = pd.read_parquet(path)
        frame["row_position"] = frame["row_position"].astype("string")
        frame.to_parquet(path, index=False)
    elif corruption in {
        "altered_blend_probability",
        "float_blend_row_position",
        "forged_threshold_label",
        "float_label",
        "string_label",
        "bool_label",
        "null_label",
    }:
        path = root / "blend_probabilities.parquet"
        frame = pd.read_parquet(path)
        if corruption == "altered_blend_probability":
            frame.loc[0, "probability"] = 0.99
        elif corruption == "float_blend_row_position":
            frame["row_position"] = frame["row_position"].astype("float64")
        elif corruption == "forged_threshold_label":
            frame.loc[0, "probability"] = 0.5
            frame.loc[0, "prediction"] = 0
        elif corruption == "float_label":
            frame["prediction"] = frame["prediction"].astype("float64")
        elif corruption == "string_label":
            frame["prediction"] = frame["prediction"].astype("string")
        elif corruption == "bool_label":
            frame["prediction"] = frame["prediction"].astype("bool")
        else:
            frame["prediction"] = frame["prediction"].astype("float64")
            frame.loc[0, "prediction"] = np.nan
        frame.to_parquet(path, index=False)
    elif corruption == "altered_threshold_summary":
        path = root / "prediction_summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["threshold"] = 0.25
        _write_json(path, payload)
    elif corruption in {
        "nonbinary_submission",
        "quoted_numeric_id",
        "crlf_submission",
        "bom_submission",
        "whitespace_submission",
        "trailing_blank_submission",
        "alternate_quoting_submission",
        "float_text_label",
        "reordered_submission_columns",
    }:
        path = root / "submission.csv"
        raw = path.read_bytes()
        if corruption in {"nonbinary_submission", "float_text_label"}:
            raw = raw.replace(b"100,0\n", b"100,0.0\n", 1)
        elif corruption == "quoted_numeric_id":
            raw = raw.replace(b"100,", b'"100",', 1)
        elif corruption == "crlf_submission":
            raw = raw.replace(b"\n", b"\r\n")
        elif corruption == "bom_submission":
            raw = b"\xef\xbb\xbf" + raw
        elif corruption == "whitespace_submission":
            raw = raw.replace(b"100,", b" 100,", 1)
        elif corruption == "trailing_blank_submission":
            raw += b"\n"
        elif corruption == "alternate_quoting_submission":
            raw = b'"index","y"\n"100","0"\n"101","1"\n"102","1"\n'
        else:
            raw = b"y,index\n0,100\n1,101\n1,102\n"
        path.write_bytes(raw)
    elif corruption == "wrong_train_schema":
        path = root / "train_schema.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["source_rows"] = 999
        _write_json(path, payload)
    elif corruption == "wrong_feature_identity":
        path = root / "feature_identity.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["pipeline_id"] = "forged"
        _write_json(path, payload)
    elif corruption in {
        "forged_runtime",
        "arbitrary_duration",
        "malformed_start_timestamp",
        "malformed_completion_timestamp",
        "non_utc_timestamp",
        "completion_before_start",
    }:
        path = root / "runtime.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if corruption == "forged_runtime":
            payload["network_access"] = True
        elif corruption == "arbitrary_duration":
            payload["duration_seconds"] += 1.0
        elif corruption == "malformed_start_timestamp":
            payload["started_at_utc"] = "not-a-timestamp"
        elif corruption == "malformed_completion_timestamp":
            payload["finished_at_utc"] = "2026-02-30T00:00:00.000000Z"
        elif corruption == "non_utc_timestamp":
            payload["started_at_utc"] = "2026-01-01T00:00:00.000000+01:00"
        else:
            payload["started_at_utc"] = "2026-01-01T00:00:01.000000Z"
            payload["finished_at_utc"] = "2026-01-01T00:00:00.000000Z"
            payload["duration_seconds"] = 0.0
        _write_json(path, payload)
    elif corruption == "forged_environment":
        path = root / "environment.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["python"] = "0.0.0"
        _write_json(path, payload)
    elif corruption not in {
        "malformed_success_timestamp",
        "success_completion_mismatch",
        "success_wrong_deployment_id",
        "success_wrong_manifest",
        "both_terminal_markers",
        "terminal_not_newest",
    }:
        raise AssertionError(corruption)


def _mutate_terminal(root: Path, corruption: str) -> None:
    success_path = root / "_SUCCESS"
    payload = json.loads(success_path.read_text(encoding="utf-8"))
    if corruption == "malformed_success_timestamp":
        payload["completed_at_utc"] = "not-a-timestamp"
    elif corruption == "success_completion_mismatch":
        payload["completed_at_utc"] = "2026-01-01T00:00:00.000000Z"
    elif corruption == "success_wrong_deployment_id":
        payload["deployment_id"] = "wrong-deployment"
    elif corruption == "success_wrong_manifest":
        payload["manifest_sha256"] = "f" * 64
    elif corruption == "both_terminal_markers":
        _write_json(
            root / "_FAILED",
            {
                "schema_version": 1,
                "status": "failed",
                "failed_at_utc": "2026-01-01T00:00:00.000000Z",
                "failure": {"type": "Synthetic", "message": "forged"},
            },
        )
        return
    elif corruption == "terminal_not_newest":
        success_mtime = success_path.stat().st_mtime_ns
        os.utime(
            root / "runtime.json",
            ns=(success_mtime + 1_000_000_000, success_mtime + 1_000_000_000),
        )
        return
    else:
        raise AssertionError(corruption)
    _write_json(success_path, payload)


def test_execution_reauthenticates_every_decision_and_fixture_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated, _, fixture_root = _execution_fixture(tmp_path)
    comparison_root = tmp_path / "comparison"
    comparison_root.mkdir()
    _write_json(comparison_root / "manifest.json", {"manifest_sha256": "a" * 64})
    _write_json(comparison_root / "decision.json", {"decision": "promising"})
    nested = comparison_root / "nested"
    nested.mkdir()
    _write_json(nested / "metric.json", {"value": 1.0})
    approval = replace(
        validated.approvals[0],
        paired_comparison_source=authenticate_source(
            comparison_root,
            project_root=tmp_path,
            source_type="paired_comparison",
            semantic_identity="a" * 64,
            expected_kind="directory",
        ),
    )
    validated = replace(validated, approvals=(approval,))
    threshold_source = approval.threshold_evidence_source
    assert threshold_source is not None
    targets = {
        "threshold_content": threshold_source,
        "approval": approval.approval_source,
        "research": approval.research_run_source,
        "comparison_file": approval.paired_comparison_source,
        "comparison_manifest": approval.paired_comparison_source,
        "comparison_nested": approval.paired_comparison_source,
        "fixture_manifest": fixture_root / "fixture_manifest.yaml",
        "fixture_data": fixture_root / "X_train.parquet",
        "fixture_test": fixture_root / "X_test.parquet",
        "fixture_sample": fixture_root / "sample_submission.csv",
    }
    paths = {
        "threshold_content": tmp_path / threshold_source.repository_relative_path,
        "approval": approval.source_path,
        "research": approval.research_run.root / "identities/candidate.json",
        "comparison_file": comparison_root / "decision.json",
        "comparison_manifest": comparison_root / "manifest.json",
        "comparison_nested": nested / "metric.json",
        "fixture_manifest": fixture_root / "fixture_manifest.yaml",
        "fixture_data": fixture_root / "X_train.parquet",
        "fixture_test": fixture_root / "X_test.parquet",
        "fixture_sample": fixture_root / "sample_submission.csv",
    }
    assert all(item is not None for item in targets.values())
    for index, (name, path) in enumerate(paths.items()):
        original = path.read_bytes()

        def mutate(
            checkpoint: str, *, source: Path = path, raw: bytes = original
        ) -> None:
            if checkpoint != "after_initial_authentication":
                return
            if name == "threshold_content":
                changed = raw.replace(b"threshold: 0.5", b"threshold: 0.6", 1)
                assert len(changed) == len(raw) and changed != raw
                source.write_bytes(changed)
            else:
                source.write_bytes(raw + b"\n")

        monkeypatch.setattr("src.churn_ml.deployment_v1._lifecycle_checkpoint", mutate)
        output = tmp_path / f"mutation-output-{index:02d}"
        try:
            with pytest.raises(
                DeploymentAuthenticationError, match="Authenticated source changed"
            ):
                execute_deployment(
                    validated,
                    mode="dry-run",
                    output_dir=output,
                    fixture_dir=fixture_root,
                    estimator_factory=lambda *_: (_ for _ in ()).throw(
                        AssertionError("training must not start after input mutation")
                    ),
                )
        finally:
            path.write_bytes(original)
        assert not (output / "_SUCCESS").exists(), name
        assert (output / "_FAILED").is_file(), name
        assert path.read_bytes() == original, name


def test_threshold_ancestor_replacement_is_rejected_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated, _, fixture_root = _execution_fixture(tmp_path)
    evidence = tmp_path / "evidence"
    original = tmp_path / "evidence-original"
    external = tmp_path / "evidence-external"
    external.mkdir()
    shutil.copy2(evidence / "threshold.yaml", external / "threshold.yaml")
    output = tmp_path / "junction-mutation-output"

    def mutate(checkpoint: str) -> None:
        if checkpoint != "after_initial_authentication":
            return
        evidence.rename(original)
        created = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(evidence), str(external)],
            check=False,
            capture_output=True,
        )
        if created.returncode != 0:
            original.rename(evidence)
            pytest.skip("Junction creation is unavailable on this Windows host.")

    monkeypatch.setattr("src.churn_ml.deployment_v1._lifecycle_checkpoint", mutate)
    try:
        with pytest.raises(
            DeploymentAuthenticationError, match="junctions|reparse|Links"
        ):
            execute_deployment(
                validated,
                mode="dry-run",
                output_dir=output,
                fixture_dir=fixture_root,
                estimator_factory=lambda *_: None,
            )
    finally:
        if evidence.exists() and getattr(evidence, "is_junction", lambda: False)():
            os.rmdir(evidence)
        if original.exists():
            original.rename(evidence)
    assert not (output / "_SUCCESS").exists()
    assert (output / "_FAILED").is_file()


def test_final_success_authentication_rejects_late_threshold_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated, data, fixture_root = _execution_fixture(tmp_path)
    monkeypatch.setattr(
        "src.churn_ml.deployment_v1.load_deployment_data", lambda *args, **kwargs: data
    )

    def provenance(root: Path) -> dict[str, Any]:
        return {"schema_version": 1, "root": str(root)}

    monkeypatch.setattr("src.churn_ml.deployment_v1.source_provenance", provenance)
    monkeypatch.setattr(
        "src.churn_ml.deployment_v1_artifacts.source_provenance", provenance
    )
    threshold_path = tmp_path / "evidence/threshold.yaml"
    original = threshold_path.read_bytes()

    class Estimator:
        classes_ = np.array([0, 1])

        def fit(self, features: pd.DataFrame, labels: pd.Series) -> None:
            assert len(features) == len(labels)

        def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
            positive = np.linspace(0.25, 0.75, len(features), dtype=np.float64)
            return np.column_stack([1.0 - positive, positive])

    def mutate(checkpoint: str) -> None:
        if checkpoint == "before_success_authentication":
            threshold_path.write_bytes(original + b"\n")

    monkeypatch.setattr("src.churn_ml.deployment_v1._lifecycle_checkpoint", mutate)
    output = tmp_path / "late-mutation-output"
    try:
        with pytest.raises(
            DeploymentAuthenticationError, match="Authenticated source changed"
        ):
            execute_deployment(
                validated,
                mode="dry-run",
                output_dir=output,
                fixture_dir=fixture_root,
                estimator_factory=lambda *_: Estimator(),
            )
    finally:
        threshold_path.write_bytes(original)
    assert not (output / "_SUCCESS").exists()
    assert (output / "_FAILED").is_file()


def test_ancestor_junction_is_rejected_across_all_deployment_path_surfaces(
    tmp_path: Path,
) -> None:
    external = tmp_path.parent / f"{tmp_path.name}-external"
    nested = external / "level-one/level-two"
    nested.mkdir(parents=True)
    for name in ("leaf.yaml", "threshold.yaml", "test.parquet", "sample.csv"):
        (nested / name).write_text("safe", encoding="utf-8")
    for name in ("fixture", "comparison", "deployment"):
        (nested / name).mkdir()
    alias = tmp_path / "alias"
    if os.name == "nt":
        created = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(alias), str(external)],
            check=False,
            capture_output=True,
        )
        if created.returncode != 0:
            shutil.rmtree(external, ignore_errors=True)
            pytest.skip("Junction creation is unavailable on this Windows host.")
    else:
        try:
            alias.symlink_to(external, target_is_directory=True)
        except OSError:
            shutil.rmtree(external, ignore_errors=True)
            pytest.skip("Directory-link creation is unavailable.")
    try:
        ordinary = tmp_path / "ordinary/leaf.txt"
        ordinary.parent.mkdir()
        ordinary.write_text("safe", encoding="utf-8")
        safe = validate_path_chain(
            containment_root=tmp_path,
            requested_path=ordinary,
            require_exists=True,
            expected_kind="file",
        )
        assert safe.canonical == ordinary.absolute()
        linked_leaf = alias / "level-one/level-two/leaf.yaml"
        calls = (
            lambda: validate_path_chain(
                containment_root=tmp_path,
                requested_path=linked_leaf,
                require_exists=True,
                expected_kind="file",
            ),
            lambda: prewalk_regular_tree(
                alias / "level-one/level-two/deployment", reject_hardlinks=True
            ),
            lambda: validate_new_path(
                alias / "level-one/level-two/new-output", containment_root=tmp_path
            ),
            lambda: validate_regular_file(
                alias / "level-one/level-two/test.parquet",
                containment_root=tmp_path,
                reject_hardlinks=True,
            ),
            lambda: validate_regular_file(
                alias / "level-one/level-two/sample.csv",
                containment_root=tmp_path,
                reject_hardlinks=True,
            ),
            lambda: authenticate_source(
                alias / "level-one/level-two/comparison",
                project_root=tmp_path,
                source_type="paired_comparison",
                semantic_identity="a" * 64,
                expected_kind="directory",
            ),
            lambda: load_synthetic_fixture(
                alias / "level-one/level-two/fixture",
                project_root=tmp_path,
                forbidden_hashes=set(),
            ),
            lambda: load_candidate_approval(
                linked_leaf,
                project_root=tmp_path,
                research_loader=cast(Any, lambda *args, **kwargs: None),
            ),
            lambda: load_threshold_evidence(
                {
                    **_threshold_reference(),
                    "path": "alias/level-one/level-two/threshold.yaml",
                },
                project_root=tmp_path,
                research_run=SimpleNamespace(),
            ),
            lambda: DeploymentArtifactStore(
                alias / "level-one/level-two/new-deployment"
            ),
            lambda: load_completed_deployment(
                alias / "level-one/level-two/deployment", project_root=tmp_path
            ),
        )
        for call in calls:
            with pytest.raises(
                (ValueError, RuntimeError), match="junctions|reparse|Links"
            ):
                call()
    finally:
        if alias.exists() and (
            alias.is_symlink() or getattr(alias, "is_junction", lambda: False)()
        ):
            os.rmdir(alias)
        shutil.rmtree(external, ignore_errors=True)


@pytest.mark.parametrize(
    ("test_ids", "sample_ids", "message"),
    [
        (["", "b", "c"], ["", "b", "c"], "empty/whitespace"),
        (["a", " ", "c"], ["a", " ", "c"], "empty/whitespace"),
        (["a", "b", "c"], ["", "b", "c"], "empty/whitespace"),
        (["", "", "c"], ["", "", "c"], "duplicates"),
        (["a", 2, "c"], ["a", 2, "c"], "homogeneous"),
    ],
)
def test_id_validation_rejects_empty_whitespace_duplicate_and_mixed_values(
    tmp_path: Path,
    test_ids: list[Any],
    sample_ids: list[Any],
    message: str,
) -> None:
    config = DeploymentConfig(_config_payload(), tmp_path / "config.yaml", tmp_path)
    test = pd.DataFrame({"index": pd.Series(test_ids, dtype="object")})
    sample = pd.DataFrame(
        {"index": pd.Series(sample_ids, dtype="object"), "y": [0, 0, 0]}
    )
    with pytest.raises(DeploymentDataError, match=message):
        _validate_sample_and_alignment(test, sample, config, False)


def test_id_validation_accepts_nonempty_strings_and_exact_integers(
    tmp_path: Path,
) -> None:
    config = DeploymentConfig(_config_payload(), tmp_path / "config.yaml", tmp_path)
    for values in (
        pd.Series(["a", "b", "c"], dtype="object"),
        pd.Series([1, 2, 3], dtype="int64"),
    ):
        test = pd.DataFrame({"index": values})
        sample = pd.DataFrame({"index": values.copy(), "y": [0, 0, 0]})
        keys, _ = _validate_sample_and_alignment(test, sample, config, False)
        assert keys == tuple(values.tolist())


def test_approval_id_excludes_only_documented_audit_fields(tmp_path: Path) -> None:
    run = _fake_run(tmp_path)
    _, payload = _approval_file(tmp_path, run)
    original = payload["approval_id"]
    audit = deepcopy(payload)
    audit["component_name"] = "Renamed display label"
    audit["manual_approval"]["approver"] = "another-auditor"
    audit["manual_approval"]["approved_at_utc"] = "2026-02-01T00:00:00+00:00"
    assert derive_approval_id(audit) == original
    immutable = deepcopy(payload)
    immutable["fixed_resolved_model_parameters"]["n_estimators"] += 1
    assert derive_approval_id(immutable) != original


def test_seed_order_is_canonical_and_blend_rejects_row_permutation(
    tmp_path: Path,
) -> None:
    payload = _config_payload()
    payload["components"][0]["bag_seeds"] = [9, 1, 5]
    loaded = load_deployment_config(
        _write_yaml(tmp_path / "seed-order.yaml", payload), project_root=tmp_path
    )
    assert loaded.payload["components"][0]["bag_seeds"] == [1, 5, 9]
    left = SimpleNamespace(
        component_id="left",
        probabilities=np.array([0.1, 0.9]),
        row_keys=(10, 11),
        row_identity_sha256=canonical_sha256([10, 11]),
    )
    permuted = SimpleNamespace(
        component_id="right",
        probabilities=np.array([0.9, 0.1]),
        row_keys=(11, 10),
        row_identity_sha256=canonical_sha256([11, 10]),
    )
    with pytest.raises(ValueError, match="row alignment"):
        build_fixed_blend(
            [left, permuted],  # type: ignore[list-item]
            {"left": 0.5, "right": 0.5},
        )


@pytest.mark.parametrize("which", ["test", "sample"])
@pytest.mark.parametrize("bad", ["duplicate", "null"])
def test_test_and_submission_ids_reject_duplicates_and_nulls_independently(
    tmp_path: Path,
    which: str,
    bad: str,
) -> None:
    config = DeploymentConfig(_config_payload(), tmp_path / "c.yaml", tmp_path)
    test = pd.DataFrame({"index": pd.Series([10, 11, 12], dtype="Int64")})
    sample = pd.DataFrame(
        {"index": pd.Series([10, 11, 12], dtype="Int64"), "y": [0, 0, 0]}
    )
    target = test if which == "test" else sample
    target.loc[1, "index"] = target.loc[0, "index"] if bad == "duplicate" else pd.NA
    with pytest.raises(DeploymentDataError, match="duplicates|null"):
        _validate_sample_and_alignment(test, sample, config, False)


def test_authenticated_fixture_rejects_extra_and_linked_files(tmp_path: Path) -> None:
    _, _, fixture = _execution_fixture(tmp_path)
    (fixture / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="extra"):
        load_synthetic_fixture(
            fixture, project_root=tmp_path, forbidden_hashes={"0" * 64}
        )
    (fixture / "extra.txt").unlink()
    link = fixture / "linked.csv"
    try:
        os.link(fixture / "sample_submission.csv", link)
    except OSError:
        pytest.skip("Hard-link creation is unavailable on this filesystem.")
    if os.stat(link, follow_symlinks=False).st_nlink <= 1:
        pytest.skip("Filesystem does not report hard-link counts.")
    with pytest.raises(ValueError, match="linked"):
        prewalk_regular_tree(fixture, reject_hardlinks=True)


def test_tree_prewalk_rejects_root_reparse_link(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "file.txt").write_text("safe", encoding="utf-8")
    link = tmp_path / "linked-root"
    if os.name == "nt":
        created = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(source)],
            check=False,
            capture_output=True,
        )
        if created.returncode != 0:
            pytest.skip("Junction creation is unavailable on this Windows host.")
    else:
        try:
            link.symlink_to(source, target_is_directory=True)
        except OSError:
            pytest.skip("Directory-link creation is unavailable.")
    with pytest.raises(DeploymentPathError, match="junctions|reparse|Links"):
        prewalk_regular_tree(link, reject_hardlinks=True)


def _reauthenticate_completed_tree(root: Path) -> None:
    inventory = build_inventory(root)
    _write_json(root / "artifact_inventory.json", inventory)
    manifest = build_manifest(root)
    _write_json(root / "manifest.json", manifest)
    runtime = json.loads((root / "runtime.json").read_text(encoding="utf-8"))
    resolved = yaml.safe_load(
        (root / "resolved_deployment_config.yaml").read_text(encoding="utf-8")
    )
    _write_json(
        root / "_SUCCESS",
        {
            "schema_version": 1,
            "deployment_id": resolved["deployment_id"],
            "status": "completed",
            "manifest_sha256": manifest["manifest_sha256"],
            "completed_at_utc": runtime["finished_at_utc"],
        },
    )


def _threshold_reference() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "path": "evidence/threshold.yaml",
        "size_bytes": 1,
        "sha256": "a" * 64,
        "source_type": "experiment_core_v2_manual_threshold_v1",
        "source_run_id": "run-one",
        "source_manifest_sha256": "5" * 64,
        "threshold": 0.5,
        "threshold_policy_id": "fixed_manual_threshold_v1",
        "plan_sha256": "1" * 64,
        "candidate_sha256": "4" * 64,
    }


def _write_threshold_evidence(tmp_path: Path, run: Any) -> dict[str, Any]:
    artifact = {
        "schema_version": 1,
        "source_type": "experiment_core_v2_manual_threshold_v1",
        "source_run_id": run.metadata["run_id"],
        "source_manifest_sha256": run.manifest["manifest_sha256"],
        "threshold": 0.5,
        "threshold_policy_id": "fixed_manual_threshold_v1",
        "plan_sha256": "1" * 64,
        "candidate_sha256": "4" * 64,
    }
    path = _write_yaml(tmp_path / "evidence/threshold.yaml", artifact)
    raw = path.read_bytes()
    return {
        **artifact,
        "path": path.relative_to(tmp_path).as_posix(),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _config_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "deployment_id": "synthetic-deployment",
        "dataset_version": "v3_targeted_missingness",
        "pipeline_id": "manual_v3_pipeline_v1_compat",
        "components": [
            {
                "component_id": "manual-component",
                "approval_artifact_path": "approvals/manual.yaml",
                "adapter_id": "manual_lightgbm_te_v1_compat",
                "fixed_parameters": {},
                "bag_seeds": [0],
                "component_weight": 1.0,
            }
        ],
        "blend": {
            "method": "fixed_weighted_mean",
            "weight_sum_tolerance": 1e-12,
        },
        "threshold": {
            "value": 0.5,
            "evidence": _threshold_reference(),
            "comparison": "greater_than_or_equal",
        },
        "bagging": {
            "method": "full_data_seed_bagging",
            "aggregation": "arithmetic_mean",
            "model_persistence": False,
        },
        "test_data": {
            "path": "competition/never-read.parquet",
            "sha256": "0" * 64,
            "expected_rows": 3,
            "ordered_schema_sha256": "0" * 64,
        },
        "sample_submission": {
            "path": "competition/never-read.csv",
            "sha256": "0" * 64,
            "expected_rows": 3,
            "id_column": "index",
            "target_column": "y",
        },
        "output": {
            "root": "artifacts/deployments",
            "submission_filename": "submission.csv",
        },
        "runtime": {"tracking_enabled": False, "network_enabled": False},
    }


def _fake_run(tmp_path: Path) -> Any:
    root = tmp_path / "artifacts/research_v2/run-one"
    identity_dir = root / "identities"
    identity_dir.mkdir(parents=True, exist_ok=True)
    hashes = {
        "plan": "1" * 64,
        "pipeline": "2" * 64,
        "adapter": "3" * 64,
        "candidate": "4" * 64,
    }
    _write_json(
        identity_dir / "evaluation_plan.json",
        {"sha256": hashes["plan"], "canonical": {}},
    )
    _write_json(
        identity_dir / "feature_pipeline.json",
        {"sha256": hashes["pipeline"], "canonical": {}},
    )
    _write_json(
        identity_dir / "candidate_adapter.json",
        {
            "sha256": hashes["adapter"],
            "canonical": {
                "runtime_dependencies": {
                    "lightgbm": importlib.metadata.version("lightgbm")
                }
            },
        },
    )
    _write_json(
        identity_dir / "candidate.json",
        {"sha256": hashes["candidate"], "canonical": {}},
    )
    fingerprints = {"dataset_version": "v3_targeted_missingness", "rows": 12}
    config = SimpleNamespace(
        plan_id="synthetic-plan",
        pipeline_id="manual_v3_pipeline_v1_compat",
        adapter_id="manual_lightgbm_te_v1_compat",
        dataset_version="v3_targeted_missingness",
        adapter_contract=deepcopy(EXPECTED_ADAPTER_CONTRACT),
        pipeline_contract={},
        plan_payload={"dataset": {"files": {}}},
        dataset_dir=tmp_path / "data/processed/v3_targeted_missingness",
        project_root=tmp_path,
    )
    return SimpleNamespace(
        root=root,
        config=config,
        metadata={"run_id": "run-one"},
        manifest={"manifest_sha256": "5" * 64},
        dataset_fingerprints=fingerprints,
        evaluation_plan_identity={"sha256": hashes["plan"]},
    )


def _approval_file(tmp_path: Path, run: Any) -> tuple[Path, dict[str, Any]]:
    payload = {
        "schema_version": 1,
        "approval_id": "0" * 64,
        "component_id": "manual-component",
        "component_name": "Synthetic manual component",
        "research_run": {
            "path": run.root.relative_to(tmp_path).as_posix(),
            "run_id": "run-one",
            "manifest_sha256": "5" * 64,
            "plan_id": "synthetic-plan",
            "plan_sha256": "1" * 64,
            "dataset_identity_sha256": canonical_sha256(run.dataset_fingerprints),
            "pipeline_id": "manual_v3_pipeline_v1_compat",
            "pipeline_sha256": "2" * 64,
            "adapter_id": "manual_lightgbm_te_v1_compat",
            "adapter_sha256": "3" * 64,
            "candidate_sha256": "4" * 64,
        },
        "fixed_resolved_model_parameters": deepcopy(
            cast(dict[str, Any], EXPECTED_ADAPTER_CONTRACT["lightgbm"])["parameters"]
        ),
        "threshold_evidence": _write_threshold_evidence(tmp_path, run),
        "paired_comparison": {
            "reference": None,
            "manifest_sha256": None,
            "exception": {
                "granted": True,
                "reason": "Synthetic test explicitly exercises the exception.",
            },
        },
        "manual_approval": {
            "status": "approved",
            "approver": "synthetic-test",
            "approved_at_utc": "2026-01-01T00:00:00+00:00",
        },
        "intended_deployment_role": "synthetic-test-only",
    }
    payload["approval_id"] = derive_approval_id(payload)
    path = _write_yaml(tmp_path / "approvals/manual.yaml", payload)
    return path, payload


def _execution_fixture(
    tmp_path: Path,
) -> tuple[ValidatedDeployment, DeploymentData, Path]:
    run = _fake_run(tmp_path)
    approval_path, approval_payload = _approval_file(tmp_path, run)
    threshold_evidence = load_threshold_evidence(
        approval_payload["threshold_evidence"],
        project_root=tmp_path,
        research_run=run,
    )
    approval = CandidateApproval(
        payload=approval_payload,
        source_path=approval_path,
        source_sha256=hashlib.sha256(approval_path.read_bytes()).hexdigest(),
        research_run=run,
        approval_source=authenticate_source(
            approval_path,
            project_root=tmp_path,
            source_type="candidate_approval",
            semantic_identity=approval_payload["approval_id"],
            expected_kind="file",
        ),
        research_run_source=authenticate_source(
            run.root,
            project_root=tmp_path,
            source_type="completed_research_run",
            semantic_identity=run.manifest["manifest_sha256"],
            expected_kind="directory",
        ),
        threshold_evidence_source=threshold_evidence.source,
    )
    payload = _config_payload()
    payload["components"][0]["fixed_parameters"] = deepcopy(
        cast(dict[str, Any], EXPECTED_ADAPTER_CONTRACT["lightgbm"])["parameters"]
    )
    payload["threshold"]["evidence"] = deepcopy(approval_payload["threshold_evidence"])
    config_path = _write_yaml(tmp_path / "deployment.yaml", payload)
    config = DeploymentConfig(payload, config_path, tmp_path)
    identity = {"schema_version": 1, "synthetic": True}
    validated = ValidatedDeployment(
        config=config,
        approvals=(approval,),
        identity=identity,
        identity_sha256=canonical_sha256(identity),
    )
    X_train = pd.DataFrame(
        {
            "index": np.arange(12, dtype=np.int64),
            "number": np.arange(12, dtype=float),
            "category": ["a", "b"] * 6,
        }
    )
    y_train = pd.Series([0, 1] * 6, name="y", dtype="int8")
    X_test = pd.DataFrame(
        {
            "index": [100, 101, 102],
            "number": [12.0, 13.0, np.nan],
            "category": ["a", "unknown", None],
        }
    )
    sample = pd.DataFrame({"index": [100, 101, 102], "y": [0, 0, 0]})
    dataset_identity = {
        "schema_version": 1,
        "dataset_version": "v3_targeted_missingness",
        "train_rows": 12,
        "test_rows": 3,
        "train_row_order_sha256": canonical_sha256(list(range(12))),
        "test_row_order_sha256": canonical_sha256(list(range(3))),
        "target_sha256": canonical_sha256(y_train.tolist()),
        "test_id_sha256": canonical_sha256([100, 101, 102]),
        "sample_id_sha256": canonical_sha256([100, 101, 102]),
    }
    data = DeploymentData(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        sample_submission=sample,
        source_train=X_train,
        source_test=X_test,
        train_schema={"synthetic": True},
        test_schema={"synthetic": True},
        dataset_identity=dataset_identity,
        test_row_keys=(100, 101, 102),
        test_row_identity_sha256=canonical_sha256([100, 101, 102]),
        fixture_identity={},
    )
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    X_train.to_parquet(fixture / "X_train.parquet", index=False)
    y_train.to_frame().to_parquet(fixture / "y_train.parquet", index=False)
    X_test.to_parquet(fixture / "X_test.parquet", index=False)
    sample.to_csv(fixture / "sample_submission.csv", index=False)
    files = {}
    for role, name in {
        "train": "X_train.parquet",
        "labels": "y_train.parquet",
        "test": "X_test.parquet",
        "sample_submission": "sample_submission.csv",
    }.items():
        raw = (fixture / name).read_bytes()
        files[role] = {
            "path": name,
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    manifest = {
        "schema_version": 1,
        "fixture_id": "0" * 64,
        "fixture_type": "synthetic_noncompetition",
        "files": files,
        "generation": {"generator_id": "pytest_fixture_v1", "seed": 42},
    }
    manifest["fixture_id"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "fixture_id"}
    )
    _write_yaml(fixture / "fixture_manifest.yaml", manifest)
    authenticated = load_synthetic_fixture(
        fixture, project_root=tmp_path, forbidden_hashes={"0" * 64}
    )
    data = DeploymentData(
        **{**data.__dict__, "fixture_identity": authenticated.identity}
    )
    return validated, data, fixture


def _write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
