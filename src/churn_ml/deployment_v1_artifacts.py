from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from src.churn_ml.deployment_v1_contracts import (
    ValidatedDeployment,
    load_deployment_config,
    validate_loaded_deployment,
)
from src.churn_ml.deployment_v1_models import (
    classify_fixed,
    probability_sha256,
)
from src.churn_ml.research_data import canonical_sha256


DEPLOYMENT_SOURCE_PATHS = (
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
    "dataset_identity.json",
    "train_schema.json",
    "test_schema.json",
    "feature_identity.json",
    "encoding_identity.json",
    "encoding_assignments.parquet",
    "component_summary.json",
    "bag_summary.csv",
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
        self.root = root.resolve()
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
        _atomic_bytes(
            self.root / relative,
            frame.to_csv(index=False, lineterminator="\n").encode("utf-8"),
        )

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
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "failure": {"type": type(error).__name__, "message": str(error)},
        }
        try:
            self.write_json("_FAILED", failure)
        finally:
            self._terminal = True

    def complete(self, validated: ValidatedDeployment) -> None:
        self._guard_write()
        validate_deployment_artifacts(
            self.root,
            validated=validated,
            require_success=False,
            verify_manifest=False,
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
        )
        self.write_json(
            "_SUCCESS",
            {
                "schema_version": 1,
                "manifest_sha256": manifest["manifest_sha256"],
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
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
    if values.dtype == bool or not set(np.unique(values)).issubset({0, 1}):
        raise DeploymentArtifactError("Submission labels must be binary integers.")
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
        path = project_root / relative
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
) -> None:
    deployment_root = root.resolve()
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

    persisted_source = _read_json(deployment_root / "source_provenance.json")
    if persisted_source != source_provenance(validated.config.project_root):
        raise DeploymentArtifactError("Deployment source provenance differs.")

    component_summary = _read_json(deployment_root / "component_summary.json")
    components = component_summary.get("components")
    if not isinstance(components, list) or len(components) != len(validated.approvals):
        raise DeploymentArtifactError("Component summary differs.")
    component_frame = pd.read_parquet(
        deployment_root / "component_probabilities.parquet"
    )
    if component_frame.columns[0] != "row_position":
        raise DeploymentArtifactError("Component row identity is absent.")
    expected_positions = list(range(len(component_frame)))
    if component_frame["row_position"].tolist() != expected_positions:
        raise DeploymentArtifactError("Component row order differs.")
    config_components = validated.config.payload["components"]
    expected_component_columns = [
        str(item["component_id"]) for item in config_components
    ]
    if component_frame.columns.tolist() != [
        "row_position",
        *expected_component_columns,
    ]:
        raise DeploymentArtifactError("Component probability columns differ.")
    summaries = {
        str(item["component_id"]): item
        for item in components
        if isinstance(item, dict) and "component_id" in item
    }
    for component_id in expected_component_columns:
        values = component_frame[component_id].to_numpy(dtype=np.float64)
        _validate_probabilities(values)
        if summaries[component_id]["component_probability_sha256"] != (
            probability_sha256(values)
        ):
            raise DeploymentArtifactError("Component probability hash differs.")

    blend_frame = pd.read_parquet(deployment_root / "blend_probabilities.parquet")
    if blend_frame.columns.tolist() != [
        "row_position",
        "probability",
        "prediction",
    ]:
        raise DeploymentArtifactError("Blend probability schema differs.")
    if blend_frame["row_position"].tolist() != expected_positions:
        raise DeploymentArtifactError("Blend row order differs.")
    recomputed = np.zeros(len(component_frame), dtype=np.float64)
    for item in config_components:
        recomputed += float(item["component_weight"]) * component_frame[
            item["component_id"]
        ].to_numpy(dtype=np.float64)
    stored = blend_frame["probability"].to_numpy(dtype=np.float64)
    if not np.array_equal(stored, recomputed):
        raise DeploymentArtifactError("Fixed blend differs from exact arithmetic.")
    threshold = float(validated.config.payload["threshold"]["value"])
    predictions = classify_fixed(stored, threshold)
    if not np.array_equal(
        predictions,
        blend_frame["prediction"].to_numpy(dtype=np.int8),
    ):
        raise DeploymentArtifactError("Fixed >= threshold labels differ.")

    submission = pd.read_csv(deployment_root / "submission.csv")
    sample_config = validated.config.payload["sample_submission"]
    if submission.columns.tolist() != [
        sample_config["id_column"],
        sample_config["target_column"],
    ]:
        raise DeploymentArtifactError("Submission schema differs.")
    labels = submission[sample_config["target_column"]].to_numpy(dtype=np.int8)
    if not np.array_equal(labels, predictions):
        raise DeploymentArtifactError("Submission labels differ from predictions.")
    if set(np.unique(labels)).difference({0, 1}):
        raise DeploymentArtifactError("Submission labels are not binary.")
    dataset_identity = _read_json(deployment_root / "dataset_identity.json")
    if (
        canonical_sha256(submission[sample_config["id_column"]].tolist())
        != (dataset_identity["sample_id_sha256"])
    ):
        raise DeploymentArtifactError("Submission ID/order identity differs.")
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

    roundtrip = pd.read_csv(deployment_root / "submission.csv")
    try:
        pd.testing.assert_frame_equal(roundtrip, submission, check_exact=True)
    except AssertionError as error:
        raise DeploymentArtifactError("Submission CSV round-trip differs.") from error

    if verify_manifest:
        inventory = _read_json(deployment_root / "artifact_inventory.json")
        if inventory != build_inventory(deployment_root):
            raise DeploymentArtifactError("Recursive artifact inventory differs.")
        manifest = _read_json(deployment_root / "manifest.json")
        if manifest != build_manifest(deployment_root):
            raise DeploymentArtifactError("Deployment manifest differs.")
        if require_success:
            success_payload = _read_json(success)
            if (
                set(success_payload)
                != {"schema_version", "manifest_sha256", "completed_at_utc"}
                or success_payload["schema_version"] != 1
                or success_payload["manifest_sha256"] != manifest["manifest_sha256"]
            ):
                raise DeploymentArtifactError("_SUCCESS authentication differs.")
            success_mtime = success.stat().st_mtime_ns
            if any(
                path.stat().st_mtime_ns > success_mtime
                for path in deployment_root.rglob("*")
                if path.is_file() and path != success
            ):
                raise DeploymentArtifactError("_SUCCESS is not the newest artifact.")


def load_completed_deployment(
    root: Path,
    *,
    project_root: Path,
) -> CompletedDeployment:
    deployment_root = root.resolve()
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
    validate_deployment_artifacts(
        deployment_root,
        validated=validated,
        require_success=True,
        verify_manifest=True,
    )
    return CompletedDeployment(
        root=deployment_root,
        deployment_identity=_read_json(deployment_root / "deployment_identity.json"),
        component_summary=_read_json(deployment_root / "component_summary.json"),
        prediction_summary=_read_json(deployment_root / "prediction_summary.json"),
        manifest=_read_json(deployment_root / "manifest.json"),
    )


def load_failed_deployment(root: Path) -> FailedDeployment:
    deployment_root = root.resolve()
    if (deployment_root / "_SUCCESS").exists():
        raise DeploymentArtifactError("Successful deployment is not failed.")
    failure = _read_json(deployment_root / "_FAILED")
    if failure.get("status") != "failed":
        raise DeploymentArtifactError("Failure marker schema differs.")
    return FailedDeployment(root=deployment_root, failure=failure)


def _tree_paths(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    for directory, directory_names, file_names in os.walk(root):
        base = Path(directory)
        for name in directory_names:
            path = base / name
            if path.is_symlink():
                raise DeploymentArtifactError(
                    "Linked artifact directories are forbidden."
                )
            directories.add(path.relative_to(root).as_posix())
        for name in file_names:
            path = base / name
            if path.is_symlink():
                raise DeploymentArtifactError("Linked artifact files are forbidden.")
            files.add(path.relative_to(root).as_posix())
    return files, directories


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
