from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from src.churn_ml.deployment_v1_auth import (
    AuthenticatedSource,
    load_synthetic_fixture,
    reauthenticate_source,
)
from src.churn_ml.deployment_v1_contracts import (
    ValidatedDeployment,
    load_deployment_config,
    validate_loaded_deployment,
)
from src.churn_ml.deployment_v1_features import (
    DeploymentData,
    build_full_data_encoding,
    load_deployment_data,
)
from src.churn_ml.deployment_v1_models import (
    bag_parameter_identity,
    classify_fixed,
    probability_sha256,
)
from src.churn_ml.deployment_v1_physical import (
    DeploymentPhysicalError,
    canonical_csv_bytes,
    exact_mapping,
    format_utc_timestamp,
    parse_utc_timestamp,
    read_exact_bag_summary,
    require_columns,
    require_dtype,
    require_exact_integer_values,
    require_float64_probabilities,
    require_row_id_dtype,
)
from src.churn_ml.deployment_v1_paths import (
    DeploymentPathError,
    prewalk_regular_tree,
    validate_new_path,
    validate_regular_file,
)
from src.churn_ml.research_data import canonical_sha256


DEPLOYMENT_SOURCE_PATHS = (
    "src/churn_ml/deployment_v1_auth.py",
    "src/churn_ml/deployment_v1_paths.py",
    "src/churn_ml/deployment_v1_physical.py",
    "src/churn_ml/deployment_v1_contracts.py",
    "src/churn_ml/deployment_v1_features.py",
    "src/churn_ml/deployment_v1_models.py",
    "src/churn_ml/deployment_v1_artifacts.py",
    "src/churn_ml/deployment_v1.py",
    "src/churn_ml/deployment_v1_cli.py",
    "scripts/run_deployment_v1.py",
)
ORDINARY_FILES = {
    "resolved_deployment_config.yaml",
    "deployment_identity.json",
    "input_authentication.json",
    "dataset_identity.json",
    "fixture_identity.json",
    "train_schema.json",
    "test_schema.json",
    "feature_identity.json",
    "encoding_identity.json",
    "encoding_assignments.parquet",
    "component_summary.json",
    "bag_summary.csv",
    "bag_probabilities.parquet",
    "component_probabilities.parquet",
    "blend_probabilities.parquet",
    "prediction_summary.json",
    "submission.csv",
    "runtime.json",
    "environment.json",
    "source_provenance.json",
}
ARTIFACT_DIRECTORIES = {"approval_snapshot"}


class DeploymentArtifactError(RuntimeError):
    """Raised when P4 immutable artifact semantics cannot be proven."""


@dataclass(frozen=True)
class CompletedDeployment:
    root: Path
    deployment_identity: dict[str, Any]
    component_summary: dict[str, Any]
    prediction_summary: dict[str, Any]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class FailedDeployment:
    root: Path
    failure: dict[str, Any]


class DeploymentArtifactStore:
    def __init__(self, root: Path) -> None:
        try:
            self.root = validate_new_path(root)
        except DeploymentPathError as error:
            raise DeploymentArtifactError(str(error)) from error
        self._terminal = False
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "approval_snapshot").mkdir()

    def write_json(self, relative: str, payload: Mapping[str, Any]) -> None:
        self._guard_write()
        _atomic_bytes(
            self.root / relative,
            (
                json.dumps(
                    payload,
                    indent=2,
                    ensure_ascii=False,
                    default=_json_default,
                )
                + "\n"
            ).encode("utf-8"),
        )

    def write_yaml(self, relative: str, payload: Mapping[str, Any]) -> None:
        self._guard_write()
        raw = yaml.safe_dump(
            dict(payload),
            sort_keys=False,
            allow_unicode=True,
        ).encode("utf-8")
        _atomic_bytes(self.root / relative, raw)

    def write_csv(self, relative: str, frame: pd.DataFrame) -> None:
        self._guard_write()
        _atomic_bytes(self.root / relative, canonical_csv_bytes(frame))

    def write_parquet(self, relative: str, frame: pd.DataFrame) -> None:
        self._guard_write()
        path = self.root / relative
        temporary = path.with_name(path.name + ".tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)

    def fail(self, error: BaseException) -> None:
        if self._terminal or (self.root / "_SUCCESS").exists():
            raise DeploymentArtifactError("A successful deployment cannot be failed.")
        failure = {
            "schema_version": 1,
            "status": "failed",
            "failed_at_utc": format_utc_timestamp(datetime.now(timezone.utc)),
            "failure": {"type": type(error).__name__, "message": str(error)},
        }
        try:
            self.write_json("_FAILED", failure)
        finally:
            self._terminal = True

    def complete(self, validated: ValidatedDeployment, *, data: DeploymentData) -> None:
        self._guard_write()
        validate_deployment_artifacts(
            self.root,
            validated=validated,
            require_success=False,
            verify_manifest=False,
            data=data,
        )
        inventory = build_inventory(self.root)
        self.write_json("artifact_inventory.json", inventory)
        manifest = build_manifest(self.root)
        self.write_json("manifest.json", manifest)
        validate_deployment_artifacts(
            self.root,
            validated=validated,
            require_success=False,
            verify_manifest=True,
            data=data,
        )
        runtime = _read_json(self.root / "runtime.json")
        self.write_json(
            "_SUCCESS",
            {
                "schema_version": 1,
                "deployment_id": validated.config.deployment_id,
                "status": "completed",
                "manifest_sha256": manifest["manifest_sha256"],
                "completed_at_utc": runtime["finished_at_utc"],
            },
        )
        self._terminal = True

    def _guard_write(self) -> None:
        if self._terminal or (self.root / "_SUCCESS").exists():
            raise DeploymentArtifactError("No writes are allowed after _SUCCESS.")


def build_submission(
    sample: pd.DataFrame,
    labels: np.ndarray,
    *,
    id_column: str,
    target_column: str,
) -> pd.DataFrame:
    values = np.asarray(labels)
    if values.ndim != 1 or len(values) != len(sample):
        raise DeploymentArtifactError("Prediction and sample row counts differ.")
    if values.dtype != np.dtype("int8") or not set(np.unique(values)).issubset({0, 1}):
        raise DeploymentArtifactError(
            "Submission labels must be exact nonnullable int8 values."
        )
    expected_columns = [id_column, target_column]
    if sample.columns.tolist() != expected_columns:
        raise DeploymentArtifactError("Sample submission schema differs.")
    result = sample[[id_column]].copy()
    result[target_column] = values.astype(np.int8)
    if result.columns.tolist() != expected_columns:
        raise DeploymentArtifactError("Submission contains an extra index column.")
    return result


def source_provenance(project_root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for relative in DEPLOYMENT_SOURCE_PATHS:
        try:
            path = validate_regular_file(
                project_root / relative,
                containment_root=project_root,
                reject_hardlinks=True,
            )
        except DeploymentPathError as error:
            raise DeploymentArtifactError(str(error)) from error
        raw = path.read_bytes()
        records.append(
            {"path": relative, "size_bytes": len(raw), "sha256": _sha256(raw)}
        )
    canonical = {
        "schema_version": 1,
        "hashing_method": "sha256_repository_relative_source_bytes",
        "files": records,
    }
    return {**canonical, "sha256": canonical_sha256(canonical)}


def environment_record() -> dict[str, Any]:
    distributions: dict[str, str | None] = {}
    for distribution in (
        "numpy",
        "pandas",
        "scikit-learn",
        "pyarrow",
        "lightgbm",
        "xgboost",
        "catboost",
    ):
        try:
            distributions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            distributions[distribution] = None
    return {
        "schema_version": 1,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "distributions": distributions,
    }


def build_inventory(root: Path) -> dict[str, Any]:
    excluded = {"artifact_inventory.json", "manifest.json", "_SUCCESS", "_FAILED"}
    files, directories = _tree_records(root, excluded=excluded)
    return {
        "schema_version": 1,
        "hashing_method": "sha256_raw_file_bytes_recursive_posix_paths",
        "directories": directories,
        "files": files,
    }


def build_manifest(root: Path) -> dict[str, Any]:
    excluded = {"manifest.json", "_SUCCESS", "_FAILED"}
    files, directories = _tree_records(root, excluded=excluded)
    canonical = {
        "schema_version": 1,
        "hashing_method": "sha256_raw_file_bytes_recursive_posix_paths",
        "directories": directories,
        "files": files,
    }
    return {**canonical, "manifest_sha256": canonical_sha256(canonical)}


def validate_deployment_artifacts(
    root: Path,
    *,
    validated: ValidatedDeployment,
    require_success: bool,
    verify_manifest: bool,
    data: DeploymentData,
) -> None:
    try:
        _validate_deployment_artifacts_impl(
            root,
            validated=validated,
            require_success=require_success,
            verify_manifest=verify_manifest,
            data=data,
        )
    except (DeploymentPhysicalError, KeyError, TypeError) as error:
        raise DeploymentArtifactError(str(error)) from error


def _validate_deployment_artifacts_impl(
    root: Path,
    *,
    validated: ValidatedDeployment,
    require_success: bool,
    verify_manifest: bool,
    data: DeploymentData,
) -> None:
    try:
        deployment_root = prewalk_regular_tree(root, reject_hardlinks=True).root
    except DeploymentPathError as error:
        raise DeploymentArtifactError(str(error)) from error
    if not deployment_root.is_dir():
        raise DeploymentArtifactError("Deployment directory is missing.")
    if (deployment_root / "_FAILED").exists():
        raise DeploymentArtifactError("Deployment contains _FAILED.")
    success = deployment_root / "_SUCCESS"
    if require_success and not success.is_file():
        raise DeploymentArtifactError("Deployment does not contain _SUCCESS.")
    if not require_success and success.exists():
        raise DeploymentArtifactError("_SUCCESS exists before validation.")
    files, directories = _tree_paths(deployment_root)
    expected = set(ORDINARY_FILES)
    expected.update(
        f"approval_snapshot/{item.component_id}.yaml" for item in validated.approvals
    )
    if verify_manifest:
        expected.update({"artifact_inventory.json", "manifest.json"})
    if require_success:
        expected.add("_SUCCESS")
    if files != expected or directories != ARTIFACT_DIRECTORIES:
        raise DeploymentArtifactError(
            "Deployment artifact inventory differs; "
            f"missing={sorted(expected - files)}, unexpected={sorted(files - expected)}, "
            f"directories={sorted(directories)}."
        )

    resolved = _read_yaml(deployment_root / "resolved_deployment_config.yaml")
    if resolved != validated.config.resolved_payload():
        raise DeploymentArtifactError("Resolved deployment config differs.")
    identity = _read_json(deployment_root / "deployment_identity.json")
    expected_identity = {
        "sha256": validated.identity_sha256,
        "canonical": validated.identity,
    }
    if identity != expected_identity:
        raise DeploymentArtifactError("Deployment identity differs.")
    for approval in validated.approvals:
        snapshot = _read_yaml(
            deployment_root / "approval_snapshot" / f"{approval.component_id}.yaml"
        )
        if snapshot != approval.payload:
            raise DeploymentArtifactError("Approval snapshot differs.")
        if _sha256(approval.source_path.read_bytes()) != approval.source_sha256:
            raise DeploymentArtifactError(
                "An approval artifact changed after deployment validation."
            )

    _validate_input_authentication(deployment_root, validated, data)

    persisted_source = _read_json(deployment_root / "source_provenance.json")
    if persisted_source != source_provenance(validated.config.project_root):
        raise DeploymentArtifactError("Deployment source provenance differs.")

    if _read_json(deployment_root / "dataset_identity.json") != data.dataset_identity:
        raise DeploymentArtifactError(
            "Dataset identity differs from authenticated input."
        )
    if _read_json(deployment_root / "fixture_identity.json") != data.fixture_identity:
        raise DeploymentArtifactError(
            "Fixture identity differs from authenticated input."
        )
    if _read_json(deployment_root / "train_schema.json") != data.train_schema:
        raise DeploymentArtifactError(
            "Training schema differs from authenticated input."
        )
    if _read_json(deployment_root / "test_schema.json") != data.test_schema:
        raise DeploymentArtifactError("Test schema differs from authenticated input.")
    pipeline_identity = validated.approvals[0].research_run.evaluation_plan_identity
    expected_feature_identity = {
        "schema_version": 1,
        "pipeline_id": validated.config.payload["pipeline_id"],
        "pipeline_sha256": validated.approvals[0].payload["research_run"][
            "pipeline_sha256"
        ],
        "transformed_columns": data.X_train.columns.tolist(),
        "transformed_columns_sha256": canonical_sha256(data.X_train.columns.tolist()),
        "evaluation_plan_sha256": pipeline_identity["sha256"],
    }
    if (
        _read_json(deployment_root / "feature_identity.json")
        != expected_feature_identity
    ):
        raise DeploymentArtifactError(
            "Feature identity differs from authoritative input."
        )

    stored_encoding = _read_json(deployment_root / "encoding_identity.json")
    if (
        set(stored_encoding) != {"schema_version", "components"}
        or stored_encoding["schema_version"] != 1
        or not isinstance(stored_encoding["components"], dict)
    ):
        raise DeploymentArtifactError("Encoding identity schema differs.")
    stored_assignments = pd.read_parquet(
        deployment_root / "encoding_assignments.parquet"
    )
    require_columns(
        stored_assignments,
        ["row_position", "encoding_fold"],
        "encoding_assignments",
    )
    require_dtype(stored_assignments, "row_position", "int64", "encoding_assignments")
    require_dtype(stored_assignments, "encoding_fold", "int64", "encoding_assignments")
    require_exact_integer_values(
        stored_assignments["row_position"],
        allowed=None,
        minimum=0,
        label="encoding_assignments.row_position",
    )
    require_exact_integer_values(
        stored_assignments["encoding_fold"],
        allowed=None,
        minimum=1,
        label="encoding_assignments.encoding_fold",
    )
    expected_assignments: pd.DataFrame | None = None
    expected_encoding_ids: dict[str, Any] = {}
    for approval in validated.approvals:
        contract = approval.research_run.config.adapter_contract
        encoder_contract = (
            contract["target_encoder"]
            if approval.adapter_id == "manual_lightgbm_te_v1_compat"
            else contract["numeric_features"]
        )
        encoding = build_full_data_encoding(
            data.X_train, data.y_train, data.X_test, encoder_contract
        )
        expected_encoding_ids[approval.component_id] = encoding.identity
        if expected_assignments is None:
            expected_assignments = encoding.assignments
        else:
            _assert_frame_exact(expected_assignments, encoding.assignments, "encoding")
    assert expected_assignments is not None
    if stored_encoding["components"] != expected_encoding_ids:
        raise DeploymentArtifactError("Encoding identity semantics differ.")
    _assert_frame_exact(
        stored_assignments, expected_assignments, "encoding assignments"
    )

    if _read_json(deployment_root / "environment.json") != environment_record():
        raise DeploymentArtifactError("Runtime environment identity differs.")
    runtime = _validate_runtime_artifact(
        deployment_root / "runtime.json",
        expected_mode="dry-run"
        if data.fixture_identity["mode"] == "synthetic"
        else "run",
        deployment_id=validated.config.deployment_id,
        component_count=len(validated.approvals),
        bag_count=sum(
            len(item["bag_seeds"]) for item in validated.config.payload["components"]
        ),
    )
    component_summary = _read_json(deployment_root / "component_summary.json")
    components = component_summary.get("components")
    if not isinstance(components, list) or len(components) != len(validated.approvals):
        raise DeploymentArtifactError("Component summary differs.")
    component_frame = pd.read_parquet(
        deployment_root / "component_probabilities.parquet"
    )
    config_components = validated.config.payload["components"]
    expected_component_columns = [
        str(item["component_id"]) for item in config_components
    ]
    require_columns(
        component_frame,
        ["row_position", "row_id", *expected_component_columns],
        "components",
    )
    require_dtype(component_frame, "row_position", "int64", "components")
    require_row_id_dtype(
        component_frame["row_id"], data.test_row_keys, "components.row_id"
    )
    if component_frame.columns[:2].tolist() != ["row_position", "row_id"]:
        raise DeploymentArtifactError("Component row identity is absent.")
    expected_positions = list(range(len(component_frame)))
    if (
        component_frame["row_position"].tolist() != expected_positions
        or tuple(component_frame["row_id"].tolist()) != data.test_row_keys
    ):
        raise DeploymentArtifactError("Component row identity/order differs.")
    if component_frame.columns.tolist() != [
        "row_position",
        "row_id",
        *expected_component_columns,
    ]:
        raise DeploymentArtifactError("Component probability columns differ.")
    if (
        set(component_summary) != {"schema_version", "components"}
        or component_summary["schema_version"] != 1
    ):
        raise DeploymentArtifactError("Component summary schema differs.")
    bag_frame = pd.read_parquet(deployment_root / "bag_probabilities.parquet")
    expected_bag_columns = ["row_position", "row_id"]
    for item in config_components:
        for seed in sorted(item["bag_seeds"]):
            expected_bag_columns.append(f"{item['component_id']}__seed_{int(seed)}")
    require_columns(bag_frame, expected_bag_columns, "bags")
    require_dtype(bag_frame, "row_position", "int64", "bags")
    require_row_id_dtype(bag_frame["row_id"], data.test_row_keys, "bags.row_id")
    if bag_frame.columns.tolist() != expected_bag_columns:
        raise DeploymentArtifactError("Per-bag probability schema differs.")
    if (
        bag_frame["row_position"].tolist() != expected_positions
        or tuple(bag_frame["row_id"].tolist()) != data.test_row_keys
    ):
        raise DeploymentArtifactError("Per-bag row identity/order differs.")
    bag_columns = [
        "component_id",
        "adapter_id",
        "bag_index",
        "bag_seed",
        "training_rows",
        "test_rows",
        "parameter_sha256",
        "probability_sha256",
        "probability_column",
        "row_identity_sha256",
        "probability_bytes",
        "duration_seconds",
        "early_stopping",
        "evaluation_set",
        "model_persisted",
    ]
    bag_summary = read_exact_bag_summary(
        deployment_root / "bag_summary.csv", bag_columns
    )
    if bag_summary.columns.tolist() != bag_columns:
        raise DeploymentArtifactError("Bag summary schema differs.")
    expected_bag_count = sum(len(item["bag_seeds"]) for item in config_components)
    if len(bag_summary) != expected_bag_count:
        raise DeploymentArtifactError("Bag summary count differs.")
    environments = environment_record()["distributions"]
    distribution_key = {
        "manual_lightgbm_te_v1_compat": "lightgbm",
        "xgboost_numeric_v1": "xgboost",
        "catboost_numeric_v1": "catboost",
    }
    expected_summaries: list[dict[str, Any]] = []
    record_index = 0
    for item, approval in zip(config_components, validated.approvals, strict=True):
        accumulator = np.zeros(len(data.test_row_keys), dtype=np.float64)
        seeds = sorted(int(seed) for seed in item["bag_seeds"])
        for bag_index, seed in enumerate(seeds, start=1):
            column = f"{item['component_id']}__seed_{seed}"
            probabilities = require_float64_probabilities(
                bag_frame[column], f"bags.{column}"
            )
            record = bag_summary.iloc[record_index].to_dict()
            expected_values = {
                "component_id": item["component_id"],
                "adapter_id": item["adapter_id"],
                "bag_index": bag_index,
                "bag_seed": seed,
                "training_rows": len(data.X_train),
                "test_rows": len(data.X_test),
                "parameter_sha256": canonical_sha256(
                    bag_parameter_identity(item, approval, seed)
                ),
                "probability_sha256": probability_sha256(probabilities),
                "probability_column": column,
                "row_identity_sha256": data.test_row_identity_sha256,
                "probability_bytes": len(probabilities) * 8,
                "early_stopping": False,
                "evaluation_set": False,
                "model_persisted": False,
            }
            for key, value in expected_values.items():
                if record[key] != value:
                    raise DeploymentArtifactError(f"Bag summary {key} differs.")
            duration = record["duration_seconds"]
            if (
                not isinstance(duration, (int, float))
                or not np.isfinite(duration)
                or duration < 0
            ):
                raise DeploymentArtifactError("Bag duration differs.")
            accumulator += probabilities
            record_index += 1
        average = np.asarray(accumulator / len(seeds), dtype=np.float64)
        component_values = require_float64_probabilities(
            component_frame[item["component_id"]],
            f"components.{item['component_id']}",
        )
        if not np.array_equal(component_values, average):
            raise DeploymentArtifactError("Component bag average differs.")
        expected_summaries.append(
            {
                "schema_version": 1,
                "component_id": item["component_id"],
                "adapter_id": item["adapter_id"],
                "bag_seeds": seeds,
                "bag_count": len(seeds),
                "training_rows_per_bag": len(data.X_train),
                "all_training_rows_used": True,
                "aggregation": "arithmetic_mean",
                "component_probability_sha256": probability_sha256(average),
                "runtime_library_version": environments[
                    distribution_key[item["adapter_id"]]
                ],
                "row_identity_sha256": data.test_row_identity_sha256,
                "model_persistence": False,
            }
        )
    if components != expected_summaries:
        raise DeploymentArtifactError("Component summary semantics differ.")

    summaries = {
        str(item["component_id"]): item
        for item in components
        if isinstance(item, dict) and "component_id" in item
    }
    for component_id in expected_component_columns:
        values = require_float64_probabilities(
            component_frame[component_id], f"components.{component_id}"
        )
        if summaries[component_id]["component_probability_sha256"] != (
            probability_sha256(values)
        ):
            raise DeploymentArtifactError("Component probability hash differs.")

    blend_frame = pd.read_parquet(deployment_root / "blend_probabilities.parquet")
    require_columns(
        blend_frame,
        ["row_position", "row_id", "probability", "prediction"],
        "blend",
    )
    require_dtype(blend_frame, "row_position", "int64", "blend")
    require_row_id_dtype(blend_frame["row_id"], data.test_row_keys, "blend.row_id")
    stored = require_float64_probabilities(
        blend_frame["probability"], "blend.probability"
    )
    require_dtype(blend_frame, "prediction", "int8", "blend")
    require_exact_integer_values(
        blend_frame["prediction"],
        allowed={0, 1},
        minimum=None,
        label="blend.prediction",
    )
    if blend_frame.columns.tolist() != [
        "row_position",
        "row_id",
        "probability",
        "prediction",
    ]:
        raise DeploymentArtifactError("Blend probability schema differs.")
    if (
        blend_frame["row_position"].tolist() != expected_positions
        or tuple(blend_frame["row_id"].tolist()) != data.test_row_keys
    ):
        raise DeploymentArtifactError("Blend row identity/order differs.")
    recomputed = np.zeros(len(component_frame), dtype=np.float64)
    for item in config_components:
        recomputed += float(item["component_weight"]) * component_frame[
            item["component_id"]
        ].to_numpy(dtype=np.float64)
    if not np.array_equal(stored, recomputed):
        raise DeploymentArtifactError("Fixed blend differs from exact arithmetic.")
    threshold = float(validated.config.payload["threshold"]["value"])
    predictions = classify_fixed(stored, threshold)
    if not np.array_equal(
        predictions,
        blend_frame["prediction"].to_numpy(copy=False),
    ):
        raise DeploymentArtifactError("Fixed >= threshold labels differ.")

    sample_config = validated.config.payload["sample_submission"]
    submission = _read_strict_submission(
        deployment_root / "submission.csv",
        id_column=str(sample_config["id_column"]),
        target_column=str(sample_config["target_column"]),
        expected_ids=data.test_row_keys,
        expected_labels=predictions,
    )
    labels = submission[sample_config["target_column"]].to_numpy(copy=False)
    if not np.array_equal(labels, predictions):
        raise DeploymentArtifactError("Submission labels differ from predictions.")
    summary = _read_json(deployment_root / "prediction_summary.json")
    expected_summary = {
        "schema_version": 1,
        "row_count": len(stored),
        "blend_probability_sha256": probability_sha256(stored),
        "threshold": threshold,
        "comparison": "greater_than_or_equal",
        "positive_count": int(predictions.sum()),
        "positive_rate": float(predictions.mean()),
        "submission_sha256": _sha256((deployment_root / "submission.csv").read_bytes()),
    }
    if summary != expected_summary:
        raise DeploymentArtifactError("Prediction summary differs.")

    if verify_manifest:
        inventory = _read_json(deployment_root / "artifact_inventory.json")
        if inventory != build_inventory(deployment_root):
            raise DeploymentArtifactError("Recursive artifact inventory differs.")
        manifest = _read_json(deployment_root / "manifest.json")
        if manifest != build_manifest(deployment_root):
            raise DeploymentArtifactError("Deployment manifest differs.")
        if require_success:
            success_payload = exact_mapping(
                _read_json(success),
                keys={
                    "schema_version",
                    "deployment_id",
                    "status",
                    "manifest_sha256",
                    "completed_at_utc",
                },
                types={
                    "schema_version": int,
                    "deployment_id": str,
                    "status": str,
                    "manifest_sha256": str,
                    "completed_at_utc": str,
                },
                label="_SUCCESS",
            )
            parse_utc_timestamp(success_payload["completed_at_utc"])
            if (
                success_payload["schema_version"] != 1
                or success_payload["deployment_id"] != validated.config.deployment_id
                or success_payload["status"] != "completed"
                or success_payload["manifest_sha256"] != manifest["manifest_sha256"]
                or success_payload["completed_at_utc"] != runtime["finished_at_utc"]
            ):
                raise DeploymentArtifactError("_SUCCESS authentication differs.")
            success_mtime = success.stat().st_mtime_ns
            terminal_tree = prewalk_regular_tree(deployment_root, reject_hardlinks=True)
            if any(
                (terminal_tree.root / relative).stat().st_mtime_ns > success_mtime
                for relative in terminal_tree.files
                if relative.as_posix() != "_SUCCESS"
            ):
                raise DeploymentArtifactError("_SUCCESS is not the newest artifact.")


def load_completed_deployment(
    root: Path,
    *,
    project_root: Path,
) -> CompletedDeployment:
    try:
        deployment_root = prewalk_regular_tree(root, reject_hardlinks=True).root
    except DeploymentPathError as error:
        raise DeploymentArtifactError(str(error)) from error
    resolved = _read_yaml(deployment_root / "resolved_deployment_config.yaml")
    temporary_config = deployment_root / "resolved_deployment_config.yaml"
    config = load_deployment_config(
        temporary_config,
        project_root=project_root,
        allow_external_source=True,
    )
    validated = validate_loaded_deployment(config)
    if resolved != validated.config.resolved_payload():
        raise DeploymentArtifactError("Persisted deployment config differs.")
    fixture_identity = _read_json(deployment_root / "fixture_identity.json")
    mode = fixture_identity.get("mode")
    fixture = None
    if mode == "synthetic":
        relative = fixture_identity.get("canonical", {}).get("repository_relative_path")
        if not isinstance(relative, str):
            raise DeploymentArtifactError("Synthetic fixture identity differs.")
        fixture = load_synthetic_fixture(
            project_root / relative,
            project_root=project_root,
            forbidden_hashes={
                str(validated.config.payload["test_data"]["sha256"]),
                str(validated.config.payload["sample_submission"]["sha256"]),
            },
        )
        if fixture.identity != fixture_identity:
            raise DeploymentArtifactError("Synthetic fixture authentication differs.")
    elif mode != "competition":
        raise DeploymentArtifactError("Fixture identity mode differs.")
    data = load_deployment_data(validated, fixture=fixture)
    validate_deployment_artifacts(
        deployment_root,
        validated=validated,
        require_success=True,
        verify_manifest=True,
        data=data,
    )
    return CompletedDeployment(
        root=deployment_root,
        deployment_identity=_read_json(deployment_root / "deployment_identity.json"),
        component_summary=_read_json(deployment_root / "component_summary.json"),
        prediction_summary=_read_json(deployment_root / "prediction_summary.json"),
        manifest=_read_json(deployment_root / "manifest.json"),
    )


def load_failed_deployment(root: Path) -> FailedDeployment:
    try:
        deployment_root = prewalk_regular_tree(root, reject_hardlinks=True).root
    except DeploymentPathError as error:
        raise DeploymentArtifactError(str(error)) from error
    if (deployment_root / "_SUCCESS").exists():
        raise DeploymentArtifactError("Successful deployment is not failed.")
    failure = exact_mapping(
        _read_json(deployment_root / "_FAILED"),
        keys={"schema_version", "status", "failed_at_utc", "failure"},
        types={
            "schema_version": int,
            "status": str,
            "failed_at_utc": str,
            "failure": dict,
        },
        label="_FAILED",
    )
    parse_utc_timestamp(failure["failed_at_utc"])
    if failure["schema_version"] != 1 or failure["status"] != "failed":
        raise DeploymentArtifactError("Failure marker schema differs.")
    detail = exact_mapping(
        failure["failure"],
        keys={"type", "message"},
        types={"type": str, "message": str},
        label="_FAILED.failure",
    )
    if not detail["type"] or not detail["message"]:
        raise DeploymentArtifactError("Failure marker detail differs.")
    return FailedDeployment(root=deployment_root, failure=failure)


def _tree_paths(root: Path) -> tuple[set[str], set[str]]:
    try:
        tree = prewalk_regular_tree(root, reject_hardlinks=True)
    except DeploymentPathError as error:
        raise DeploymentArtifactError(str(error)) from error
    return (
        {item.as_posix() for item in tree.files},
        {item.as_posix() for item in tree.directories},
    )


def _tree_records(
    root: Path,
    *,
    excluded: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    file_paths, directories = _tree_paths(root)
    records = []
    for relative in sorted(file_paths - excluded):
        raw = (root / Path(*relative.split("/"))).read_bytes()
        records.append(
            {"path": relative, "size_bytes": len(raw), "sha256": _sha256(raw)}
        )
    return records, sorted(directories)


def _validate_input_authentication(
    deployment_root: Path,
    validated: ValidatedDeployment,
    data: DeploymentData,
) -> None:
    payload = exact_mapping(
        _read_json(deployment_root / "input_authentication.json"),
        keys={"schema_version", "sources"},
        types={"schema_version": int, "sources": list},
        label="input_authentication",
    )
    if payload["schema_version"] != 1 or not payload["sources"]:
        raise DeploymentArtifactError("Input authentication schema differs.")
    fields = {
        "source_type",
        "repository_relative_path",
        "kind",
        "size_bytes",
        "sha256",
        "semantic_identity",
        "path_chain_sha256",
    }
    types = {
        "source_type": str,
        "repository_relative_path": str,
        "kind": str,
        "size_bytes": int,
        "sha256": str,
        "semantic_identity": str,
        "path_chain_sha256": str,
    }
    observed: list[AuthenticatedSource] = []
    for index, record in enumerate(payload["sources"]):
        exact = exact_mapping(
            record, keys=fields, types=types, label=f"input_authentication[{index}]"
        )
        source = AuthenticatedSource(**exact)
        reauthenticate_source(source, project_root=validated.config.project_root)
        observed.append(source)
    config_sources = [
        item for item in observed if item.source_type == "deployment_config"
    ]
    if (
        len(config_sources) != 1
        or config_sources[0].semantic_identity != validated.identity_sha256
    ):
        raise DeploymentArtifactError(
            "Deployment config source authentication differs."
        )
    config_source_path = validated.config.project_root / Path(
        *config_sources[0].repository_relative_path.split("/")
    )
    if _read_yaml(config_source_path) != validated.config.resolved_payload():
        raise DeploymentArtifactError("Authenticated deployment config differs.")
    expected: list[AuthenticatedSource] = []
    for approval in validated.approvals:
        closure = (
            approval.approval_source,
            approval.research_run_source,
            approval.threshold_evidence_source,
        )
        if any(item is None for item in closure):
            raise DeploymentArtifactError("Approval source closure is incomplete.")
        expected.extend(item for item in closure if item is not None)
        if approval.paired_comparison_source is not None:
            expected.append(approval.paired_comparison_source)
    nonconfig = [item for item in observed if item.source_type != "deployment_config"]
    if nonconfig[: len(expected)] != expected:
        raise DeploymentArtifactError("Approval evidence source snapshot differs.")
    mode = data.fixture_identity["mode"]
    if mode == "synthetic":
        fixtures = [
            item for item in nonconfig if item.source_type == "synthetic_fixture"
        ]
        if (
            len(fixtures) != 1
            or fixtures[0].semantic_identity != data.fixture_identity["sha256"]
            or len(nonconfig) != len(expected) + 1
        ):
            raise DeploymentArtifactError("Synthetic fixture source snapshot differs.")
    elif mode == "competition":
        for source_type, section in (
            ("competition_test", "test_data"),
            ("sample_submission", "sample_submission"),
        ):
            matches = [item for item in nonconfig if item.source_type == source_type]
            if (
                len(matches) != 1
                or matches[0].semantic_identity
                != validated.config.payload[section]["sha256"]
                or len(nonconfig) != len(expected) + 2
            ):
                raise DeploymentArtifactError("Competition source snapshot differs.")
    allowed_types = {
        "deployment_config",
        "candidate_approval",
        "completed_research_run",
        "threshold_evidence",
        "paired_comparison",
        "synthetic_fixture",
        "competition_test",
        "sample_submission",
    }
    if any(item.source_type not in allowed_types for item in observed):
        raise DeploymentArtifactError("Unexpected authenticated input source.")


def _assert_frame_exact(
    actual: pd.DataFrame, expected: pd.DataFrame, label: str
) -> None:
    try:
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    except AssertionError as error:
        raise DeploymentArtifactError(f"{label} differs.") from error


def _validate_runtime_artifact(
    path: Path,
    *,
    expected_mode: str,
    deployment_id: str | None = None,
    component_count: int | None = None,
    bag_count: int | None = None,
) -> dict[str, Any]:
    payload = exact_mapping(
        _read_json(path),
        keys={
            "schema_version",
            "deployment_id",
            "status",
            "mode",
            "started_at_utc",
            "finished_at_utc",
            "duration_seconds",
            "component_count",
            "bag_count",
            "model_persistence",
            "competition_test_access_authorized",
            "network_access",
            "tracking_enabled",
        },
        types={
            "schema_version": int,
            "deployment_id": str,
            "status": str,
            "mode": str,
            "started_at_utc": str,
            "finished_at_utc": str,
            "duration_seconds": float,
            "component_count": int,
            "bag_count": int,
            "model_persistence": bool,
            "competition_test_access_authorized": bool,
            "network_access": bool,
            "tracking_enabled": bool,
        },
        label="runtime",
    )
    if payload["schema_version"] != 1 or payload["status"] != "completed":
        raise DeploymentArtifactError("Runtime lifecycle state differs.")
    if payload["mode"] != expected_mode:
        raise DeploymentArtifactError("Runtime execution mode differs.")
    if deployment_id is not None and payload["deployment_id"] != deployment_id:
        raise DeploymentArtifactError("Runtime deployment identity differs.")
    if component_count is not None and payload["component_count"] != component_count:
        raise DeploymentArtifactError("Runtime component count differs.")
    if bag_count is not None and payload["bag_count"] != bag_count:
        raise DeploymentArtifactError("Runtime bag count differs.")
    if payload["component_count"] <= 0 or payload["bag_count"] <= 0:
        raise DeploymentArtifactError("Runtime component/bag counts differ.")
    expected_access = expected_mode == "run"
    if payload["competition_test_access_authorized"] is not expected_access:
        raise DeploymentArtifactError("Competition access flag differs.")
    if (
        payload["network_access"] is not False
        or payload["tracking_enabled"] is not False
        or payload["model_persistence"] is not False
    ):
        raise DeploymentArtifactError("Runtime isolation flags differ.")
    started = parse_utc_timestamp(payload["started_at_utc"])
    finished = parse_utc_timestamp(payload["finished_at_utc"])
    if finished < started:
        raise DeploymentArtifactError("Runtime timestamp order differs.")
    wall_duration = (finished - started).total_seconds()
    if (
        not np.isfinite(payload["duration_seconds"])
        or payload["duration_seconds"] < 0.0
        or payload["duration_seconds"] != wall_duration
    ):
        raise DeploymentArtifactError("Runtime duration differs from timestamps.")
    return payload


def _read_strict_submission(
    path: Path,
    *,
    id_column: str,
    target_column: str,
    expected_ids: tuple[Any, ...],
    expected_labels: np.ndarray,
) -> pd.DataFrame:
    labels = np.asarray(expected_labels)
    if labels.dtype != np.dtype("int8") or labels.ndim != 1:
        raise DeploymentArtifactError("Expected submission labels differ.")
    expected = pd.DataFrame(
        {
            id_column: list(expected_ids),
            target_column: labels,
        }
    )
    raw = path.read_bytes()
    canonical = canonical_csv_bytes(expected)
    if raw != canonical:
        raise DeploymentArtifactError("Submission raw bytes are not canonical.")
    return expected


def _validate_probabilities(values: np.ndarray) -> None:
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or ((values < 0.0) | (values > 1.0)).any()
    ):
        raise DeploymentArtifactError("Persisted probabilities are invalid.")


def _atomic_bytes(path: Path, raw: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DeploymentArtifactError(f"Cannot read JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise DeploymentArtifactError(f"JSON artifact is not an object: {path}")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise DeploymentArtifactError(f"Cannot read YAML artifact: {path}") from error
    if not isinstance(value, dict):
        raise DeploymentArtifactError(f"YAML artifact is not a mapping: {path}")
    return value


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}.")
