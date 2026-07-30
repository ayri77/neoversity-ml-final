"""Phase 1 Registry-backed Experiment Core integration tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.churn_ml.dataset_registry import (
    discover_registered_datasets,
    resolve_dataset_package,
)
from src.churn_ml.dataset_registry.alignment import prove_alignment
from src.churn_ml.dataset_registry.constants import MANIFEST_FILENAME
from src.churn_ml.dataset_registry.errors import DatasetRegistryError
from src.churn_ml.dataset_registry.manifest import build_manifest, write_manifest
from src.churn_ml.dataset_registry.package import load_package_artifacts
from src.churn_ml.experiment_v2 import (
    REGISTERED_PREPARED_PASSTHROUGH_V1,
    get_feature_pipeline,
)
from src.churn_ml.experiment_v2_pipeline import (
    REGISTERED_PREPARED_PASSTHROUGH_CONTRACT,
    ExperimentV2PipelineContractError,
)
from src.churn_ml.experiment_v2_schema import ordered_feature_schema_sha256
from src.churn_ml.research_data import canonical_sha256
from src.churn_ml.research_v2_config import ResearchV2Config
from src.churn_ml.research_v2_data import ResearchV2DataError, load_research_v2_training_data
from src.churn_ml.run_artifacts import fingerprint_file
from src.churn_ml.target_encoding import AutoGluonBinaryOOFTargetEncoder
from tests.dataset_registry_support import write_package_files


def _frames_shared_target(
    *,
    extra_feature: str | None = None,
    string_pattern: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], name="y", dtype="int64")
    train = {
        "num_a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "num_b": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        "cat_obj": ["a", "a", "b", "b", "a", "b", "a", "b"],
    }
    test = {
        "num_a": [10.0, 11.0, 12.0],
        "num_b": [0.0, 1.0, 0.0],
        "cat_obj": ["a", "b", "a"],
    }
    if extra_feature is not None:
        train[extra_feature] = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
        test[extra_feature] = [8.5, 9.5, 10.5]
    if string_pattern:
        train["missingness_pattern_id"] = pd.Series(
            ["01", "01", "02", "02", "01", "02", "01", "02"],
            dtype="string",
        )
        test["missingness_pattern_id"] = pd.Series(["01", "02", "03"], dtype="string")
    X_train = pd.DataFrame(train)
    X_test = pd.DataFrame(test)
    return X_train, y, X_test


def _write_registered(
    root: Path,
    dataset_id: str,
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    parent_dataset_id: str | None,
    hypothesis: str,
    target_dependency: str = "none",
    summary_features: tuple[str, ...] = (),
    binary_indicator_features: tuple[str, ...] = (),
) -> Path:
    package_dir = write_package_files(
        root / dataset_id,
        dataset_id=dataset_id,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        metadata={"description": hypothesis},
    )
    artifacts = load_package_artifacts(package_dir, dataset_id=dataset_id)
    parent_artifacts = None
    if parent_dataset_id is not None:
        parent_artifacts = load_package_artifacts(
            root / parent_dataset_id,
            dataset_id=parent_dataset_id,
        )
        anchor = tuple(parent_artifacts.X_train.columns.tolist())
    else:
        anchor = tuple(artifacts.X_train.columns.tolist())
    alignment = prove_alignment(
        artifacts,
        anchor_feature_names=anchor,
        parent_artifacts=parent_artifacts,
    )
    manifest = build_manifest(
        artifacts,
        hypothesis=hypothesis,
        parent_dataset_id=parent_dataset_id,
        target_dependency=target_dependency,  # type: ignore[arg-type]
        transformations=[{"type": "synthetic", "dataset_id": dataset_id}],
        alignment=alignment,
        summary_features=summary_features,
        binary_indicator_features=binary_indicator_features,
    )
    write_manifest(package_dir / MANIFEST_FILENAME, manifest)
    return package_dir



def _plan_and_config_for_package(
    tmp_path: Path,
    *,
    dataset_id: str,
    package_dir: Path,
    project_root: Path,
    target_dependency_visible: str | None = None,
) -> ResearchV2Config:
    del target_dependency_visible
    X_train = pd.read_parquet(package_dir / "X_train.parquet")
    y = pd.read_parquet(package_dir / "y_train.parquet").iloc[:, 0]
    metadata_path = package_dir / "metadata.json"
    train_path = package_dir / "X_train.parquet"
    target_path = package_dir / "y_train.parquet"
    processed_rel = package_dir.parent.relative_to(project_root).as_posix()
    plan = {
        "plan": {"id": f"plan_{dataset_id}", "schema_version": 1},
        "dataset": {
            "version": dataset_id,
            "processed_dir": processed_rel,
            "expected_rows": len(X_train),
            "expected_source_features": X_train.shape[1],
            "ordered_source_schema_sha256": ordered_feature_schema_sha256(
                X_train.columns.tolist()
            ),
            "ordered_dtype_schema_sha256": canonical_sha256(
                [
                    {"name": name, "dtype": str(dtype)}
                    for name, dtype in X_train.dtypes.items()
                ]
            ),
            "files": {
                "train_features": {
                    "name": "X_train.parquet",
                    "sha256": fingerprint_file(train_path)["sha256"],
                },
                "target": {
                    "name": "y_train.parquet",
                    "sha256": fingerprint_file(target_path)["sha256"],
                },
                "metadata": {
                    "name": "metadata.json",
                    "sha256": fingerprint_file(metadata_path)["sha256"],
                },
            },
            "target": {
                "name": str(y.name),
                "dtype": str(y.dtype),
                "negative_label": 0,
                "positive_label": 1,
                "expected_negative_rows": int((y == 0).sum()),
                "expected_positive_rows": int((y == 1).sum()),
                "values_sha256": canonical_sha256(
                    {
                        "name": str(y.name),
                        "dtype": str(y.dtype),
                        "values": y.tolist(),
                    }
                ),
            },
        },
        "outer_evaluation": {
            "n_splits": 2,
            "shuffle": True,
            "repeat_seeds": [11],
            "stratify": True,
        },
        "threshold_selection": {
            "n_splits": 2,
            "shuffle": True,
            "random_state": 7,
            "stratify": True,
        },
        "metrics": {"primary": "balanced_accuracy"},
    }
    # Minimal plan shape accepted by ResearchV2Config requires full research plan.
    # Use load path by constructing config directly for unit tests.
    payload = {
        "schema_version": 2,
        "experiment": {"id": f"exp_{dataset_id}"},
        "dataset": {"version": dataset_id},
        "evaluation_plan_path": "unused.yaml",
        "feature_pipeline": {
            "id": REGISTERED_PREPARED_PASSTHROUGH_V1,
            "contract": dict(REGISTERED_PREPARED_PASSTHROUGH_CONTRACT),
        },
        "candidate_adapter": {
            "id": "xgboost_numeric_v1",
            "contract": {
                "implementation": "existing_v2_fold_local_oof_target_encoding",
                "inner_splits": 2,
                "shuffle": True,
                "random_state": 42,
                "alpha": 10.0,
                "prior": "unweighted_mean_of_category_level_target_means",
                "keep_original_categorical_features": False,
                "output": "numeric_matrix",
                "missing_value_policy": "native_nan",
                "model_family": "xgboost",
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "n_estimators": 10,
                "learning_rate": 0.1,
                "max_depth": 2,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_lambda": 1.0,
                "reg_alpha": 0.0,
                "min_child_weight": 1.0,
                "gamma": 0.0,
                "tree_method": "hist",
                "n_jobs": 1,
            },
        },
        "artifacts": {"root": "artifacts/tmp"},
        "persistence": {
            "resolved_config": True,
            "identities": True,
            "assignments": True,
            "predictions": True,
            "metrics": True,
            "threshold_curves": False,
            "models": False,
        },
        "tracking": {"enabled": False},
    }
    return ResearchV2Config(
        payload=payload,
        plan_payload=plan,
        source_path=tmp_path / "config.yaml",
        plan_path=tmp_path / "plan.yaml",
        project_root=project_root,
    )


def test_discover_registered_datasets_dynamic_and_sorted(tmp_path: Path) -> None:
    root = tmp_path / "processed"
    root.mkdir()
    X_a, y, X_test = _frames_shared_target()
    X_b, _, X_test_b = _frames_shared_target(extra_feature="extra_num")
    _write_registered(
        root,
        "ds_b",
        X_train=X_b,
        y_train=y,
        X_test=X_test_b,
        parent_dataset_id=None,
        hypothesis="second",
    )
    _write_registered(
        root,
        "ds_a",
        X_train=X_a,
        y_train=y,
        X_test=X_test,
        parent_dataset_id=None,
        hypothesis="first",
        target_dependency="exploratory",
    )
    # Unregistered / legacy noise
    write_package_files(
        root / "legacy_only",
        dataset_id="legacy_only",
        X_train=X_a,
        y_train=y,
        X_test=X_test,
    )
    (root / "legacy_only" / "manifest.json").write_text("{}", encoding="utf-8")
    (root / "empty_dir").mkdir()

    discovered = discover_registered_datasets(root)
    assert [item.dataset_id for item in discovered] == ["ds_a", "ds_b"]
    assert discovered[0].target_dependency == "exploratory"
    assert discovered[0].n_features == 3
    assert discovered[1].n_features == 4
    assert discovered[0].train_content_hash
    assert discovered[0].target_hash == discovered[1].target_hash
    assert discovered[0].train_row_identity_hash
    assert discovered[0].schema_hash != discovered[1].schema_hash


def test_discover_rejects_directory_manifest_id_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "processed"
    root.mkdir()
    X_train, y, X_test = _frames_shared_target()
    package = _write_registered(
        root,
        "ds_ok",
        X_train=X_train,
        y_train=y,
        X_test=X_test,
        parent_dataset_id=None,
        hypothesis="ok",
    )
    # Corrupt by renaming directory while leaving manifest dataset_id unchanged.
    mismatched = root / "ds_mismatch"
    package.rename(mismatched)
    with pytest.raises(DatasetRegistryError):
        discover_registered_datasets(root)


def test_registry_loader_rejects_legacy_manifest_json(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    processed = project_root / "data" / "processed"
    processed.mkdir(parents=True)
    X_train, y, X_test = _frames_shared_target()
    package_dir = write_package_files(
        processed / "legacy_ds",
        dataset_id="legacy_ds",
        X_train=X_train,
        y_train=y,
        X_test=X_test,
    )
    (package_dir / "manifest.json").write_text("{}", encoding="utf-8")
    config = _plan_and_config_for_package(
        tmp_path,
        dataset_id="legacy_ds",
        package_dir=package_dir,
        project_root=project_root,
    )
    # Force registry path via passthrough pipeline; package has no dataset_manifest.json.
    with pytest.raises(ResearchV2DataError, match="Registry-backed"):
        load_research_v2_training_data(config)


def test_registry_backed_loading_and_passthrough_schema(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    processed = project_root / "data" / "processed"
    processed.mkdir(parents=True)
    X_train, y, X_test = _frames_shared_target(string_pattern=True)
    package_dir = _write_registered(
        processed,
        "ds_string",
        X_train=X_train,
        y_train=y,
        X_test=X_test,
        parent_dataset_id=None,
        hypothesis="string categorical",
        target_dependency="none",
    )
    config = _plan_and_config_for_package(
        tmp_path,
        dataset_id="ds_string",
        package_dir=package_dir,
        project_root=project_root,
    )
    data = load_research_v2_training_data(config)
    assert list(data.X.columns) == list(X_train.columns)
    assert data.X.dtypes.astype(str).tolist() == X_train.dtypes.astype(str).tolist()
    assert data.dataset_provenance["dataset_id"] == "ds_string"
    assert data.dataset_provenance["target_dependency"] == "none"
    assert data.dataset_provenance["registry_schema_version"] == "dataset_package_v1"
    assert (
        data.fingerprints["row_position_identity"]["bound_train_row_identity_hash"]
        == data.dataset_provenance["train_row_identity_hash"]
    )
    schema = data.pipeline_output.schema
    assert schema.dropped_features == []
    assert "missingness_pattern_id" in schema.categorical_features
    assert "cat_obj" in schema.categorical_features
    assert schema.transformed_feature_names == schema.numerical_features + [
        f"{name}__te" for name in schema.categorical_features
    ]
    # X_test must not be exposed on the research training object.
    assert not hasattr(data, "X_test")


def test_passthrough_pipeline_rejects_unsupported_dtype() -> None:
    pipeline = get_feature_pipeline(REGISTERED_PREPARED_PASSTHROUGH_V1)
    frame = pd.DataFrame(
        {
            "num": [1.0, 2.0],
            "when": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        }
    )
    with pytest.raises(Exception, match="unsupported dtypes"):
        pipeline.transform(frame, REGISTERED_PREPARED_PASSTHROUGH_CONTRACT)


def test_passthrough_pipeline_rejects_target_encoding_name_collision() -> None:
    pipeline = get_feature_pipeline(REGISTERED_PREPARED_PASSTHROUGH_V1)
    frame = pd.DataFrame(
        {
            "foo": pd.Series(["a", "b"], dtype="string"),
            "foo__te": [0.1, 0.2],
        }
    )
    with pytest.raises(
        ExperimentV2PipelineContractError,
        match=(
            r"transformed feature names collide: "
            r"'foo__te' <- "
            r"passthrough numeric/bool source feature 'foo__te'; "
            r"target-encoded categorical source feature 'foo'\."
        ),
    ):
        pipeline.transform(frame, REGISTERED_PREPARED_PASSTHROUGH_CONTRACT)


def test_passthrough_pipeline_accepts_non_colliding_transformed_schema() -> None:
    pipeline = get_feature_pipeline(REGISTERED_PREPARED_PASSTHROUGH_V1)
    frame = pd.DataFrame(
        {
            "num_a": [1.0, 2.0],
            "flag_bool": [True, False],
            "pattern_id": pd.Series(["01", "02"], dtype="string"),
        }
    )
    output = pipeline.transform(frame, REGISTERED_PREPARED_PASSTHROUGH_CONTRACT)
    assert output.schema.transformed_feature_names == [
        "num_a",
        "flag_bool",
        "pattern_id__te",
    ]
    assert output.schema.categorical_features == ["pattern_id"]
    assert output.schema.numerical_features == ["num_a", "flag_bool"]
    assert output.schema.dropped_features == []


def test_fold_local_string_target_encoding_unseen_category() -> None:
    frame = pd.DataFrame(
        {
            "num": np.arange(8, dtype=float),
            "missingness_pattern_id": pd.Series(
                ["01", "01", "02", "02", "01", "02", "01", "02"],
                dtype="string",
            ),
        }
    )
    target = pd.Series([1, 1, 0, 0, 1, 0, 1, 0], name="y")
    encoder = AutoGluonBinaryOOFTargetEncoder(n_splits=2, alpha=10.0, random_state=42)
    encoded = encoder.fit_transform(frame, target)
    assert encoder.categorical_columns_ == ["missingness_pattern_id"]
    assert encoded.columns.tolist() == ["num", "missingness_pattern_id__te"]
    probe = pd.DataFrame(
        {
            "num": [0.0, 1.0],
            "missingness_pattern_id": pd.Series(["01", "99"], dtype="string"),
        }
    )
    transformed = encoder.transform(probe)
    known = float(transformed.loc[0, "missingness_pattern_id__te"])
    unseen = float(transformed.loc[1, "missingness_pattern_id__te"])
    assert np.isfinite(known)
    assert np.isfinite(unseen)
    # Unseen uses training-fold global fallback stored on the encoder.
    assert unseen == pytest.approx(encoder.encodings_["missingness_pattern_id"]["global_mean"])


def test_two_datasets_identical_rows_different_features(tmp_path: Path) -> None:
    root = tmp_path / "processed"
    root.mkdir()
    X_a, y, X_test = _frames_shared_target()
    X_b, _, X_test_b = _frames_shared_target(extra_feature="extra_num")
    _write_registered(
        root,
        "base_a",
        X_train=X_a,
        y_train=y,
        X_test=X_test,
        parent_dataset_id=None,
        hypothesis="a",
    )
    _write_registered(
        root,
        "child_b",
        X_train=X_b,
        y_train=y,
        X_test=X_test_b,
        parent_dataset_id="base_a",
        hypothesis="b",
        target_dependency="exploratory",
    )
    pkg_a = resolve_dataset_package(root, "base_a")
    pkg_b = resolve_dataset_package(root, "child_b")
    assert pkg_a.manifest.target.hash == pkg_b.manifest.target.hash
    assert pkg_a.manifest.n_features != pkg_b.manifest.n_features
    assert pkg_a.manifest.schema_hash != pkg_b.manifest.schema_hash
    assert pkg_b.manifest.target_dependency == "exploratory"
    assert not (root / "base_a" / "manifest.json").exists()
    assert (root / "base_a" / MANIFEST_FILENAME).is_file()


def test_oof_alignment_key_contract_columns() -> None:
    # Document the outer-validation alignment key required for later blending.
    required = {"repeat", "repeat_seed", "outer_fold", "row_position", "target", "probability"}
    from src.churn_ml.research_v2_artifact_validation import OUTER_PREDICTION_SCHEMA

    names = {name for name, _ in OUTER_PREDICTION_SCHEMA}
    assert required.issubset(names)
