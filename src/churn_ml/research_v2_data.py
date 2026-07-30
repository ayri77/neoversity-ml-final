from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.churn_ml.dataset_registry.api import resolve_dataset_package
from src.churn_ml.dataset_registry.errors import DatasetRegistryError
from src.churn_ml.dataset_registry.experiment_discovery import build_dataset_provenance
from src.churn_ml.dataset_registry.hashing import file_content_sha256
from src.churn_ml.dataset_registry.package import has_manifest
from src.churn_ml.dataset_registry.schema import DatasetManifest
from src.churn_ml.experiment_v2 import (
    REGISTERED_PREPARED_PASSTHROUGH_V1,
    PipelineOutput,
    get_feature_pipeline,
)
from src.churn_ml.experiment_v2_schema import ordered_feature_schema_sha256
from src.churn_ml.research_data import canonical_sha256
from src.churn_ml.research_v2_config import ResearchV2Config
from src.churn_ml.run_artifacts import fingerprint_file


class ResearchV2DataError(ValueError):
    """Raised when train-only v2 data identity or schema validation fails."""


@dataclass(frozen=True)
class ResearchV2TrainingData:
    X: pd.DataFrame
    y: pd.Series
    pipeline_output: PipelineOutput
    metadata: dict[str, Any]
    fingerprints: dict[str, Any]
    dataset_provenance: dict[str, Any]


def load_research_v2_training_data(
    config: ResearchV2Config,
) -> ResearchV2TrainingData:
    project_root = config.project_root.resolve()
    processed_root = config.processed_data_root.resolve()
    dataset_dir = config.dataset_dir.resolve()
    if not _is_contained(processed_root, project_root):
        raise ResearchV2DataError(
            "Resolved processed-data root escapes the project root."
        )
    if not _is_contained(dataset_dir, processed_root) or not _is_contained(
        dataset_dir, project_root
    ):
        raise ResearchV2DataError(
            "Resolved dataset version directory escapes its permitted roots."
        )

    requires_registry = (
        config.pipeline_id == REGISTERED_PREPARED_PASSTHROUGH_V1
        or has_manifest(dataset_dir)
    )
    if requires_registry:
        return _load_registry_backed_training_data(
            config,
            project_root=project_root,
            processed_root=processed_root,
            dataset_dir=dataset_dir,
        )
    return _load_legacy_plan_backed_training_data(
        config,
        project_root=project_root,
        processed_root=processed_root,
        dataset_dir=dataset_dir,
    )


def _load_registry_backed_training_data(
    config: ResearchV2Config,
    *,
    project_root: Path,
    processed_root: Path,
    dataset_dir: Path,
) -> ResearchV2TrainingData:
    del project_root  # containment already verified by caller
    try:
        package = resolve_dataset_package(processed_root, config.dataset_version)
    except DatasetRegistryError as error:
        raise ResearchV2DataError(
            "Registry-backed dataset resolution failed; legacy fallback is refused. "
            f"{error}"
        ) from error
    if package.package_dir != dataset_dir:
        raise ResearchV2DataError(
            "Resolved Registry package directory does not match config.dataset_dir."
        )

    X_source = package.artifacts.X_train.copy()
    y = package.artifacts.y_train.copy()
    metadata_payload = package.artifacts.metadata
    nested_metadata = metadata_payload.get("metadata", {})
    if not isinstance(nested_metadata, dict):
        nested_metadata = {}
    metadata = dict(nested_metadata)

    _cross_check_plan_against_registry(config, package.manifest, X_source, y)
    _validate_source_common(X_source, y, metadata, config)

    provenance = build_dataset_provenance(package.manifest)
    file_fingerprints = {
        "train_features": fingerprint_file(package.artifacts.paths["X_train"]),
        "target": fingerprint_file(package.artifacts.paths["y_train"]),
        "metadata": fingerprint_file(package.artifacts.paths["metadata"]),
        "dataset_manifest": fingerprint_file(
            package.package_dir / "dataset_manifest.json"
        ),
    }
    for key, artifact_key in (
        ("train_features", "X_train"),
        ("target", "y_train"),
        ("metadata", "metadata"),
    ):
        expected = package.manifest.content_hashes[artifact_key]
        actual = file_fingerprints[key]["sha256"]
        if actual != expected:
            raise ResearchV2DataError(
                f"Registry content hash mismatch for {artifact_key}."
            )
        plan_files = config.plan_payload["dataset"].get("files", {})
        if key in plan_files:
            plan_hash = plan_files[key].get("sha256")
            if plan_hash is not None and plan_hash != expected:
                raise ResearchV2DataError(
                    f"Evaluation-plan file hash for {key} disagrees with "
                    "dataset_manifest.json content_hashes."
                )

    pipeline = get_feature_pipeline(config.pipeline_id)
    output = pipeline.transform(X_source, config.pipeline_contract)
    identity = _build_identity(
        config=config,
        X_source=X_source,
        y=y,
        file_fingerprints=file_fingerprints,
        provenance=provenance,
    )
    return ResearchV2TrainingData(
        X=output.features,
        y=y,
        pipeline_output=output,
        metadata=metadata,
        fingerprints=identity,
        dataset_provenance=provenance,
    )


def _load_legacy_plan_backed_training_data(
    config: ResearchV2Config,
    *,
    project_root: Path,
    processed_root: Path,
    dataset_dir: Path,
) -> ResearchV2TrainingData:
    dataset = config.plan_payload["dataset"]
    paths: dict[str, Path] = {}
    fingerprints: dict[str, dict[str, Any]] = {}
    for key, item in dataset["files"].items():
        path = (dataset_dir / str(item["name"])).resolve()
        if (
            path.parent != dataset_dir
            or not _is_contained(path, processed_root)
            or not _is_contained(path, project_root)
        ):
            raise ResearchV2DataError("Dataset file escapes its version directory.")
        actual = fingerprint_file(path)
        if actual["sha256"] != item["sha256"]:
            raise ResearchV2DataError(f"Dataset fingerprint mismatch for {key}.")
        paths[key] = path
        fingerprints[key] = actual
    X_source = pd.read_parquet(paths["train_features"])
    target_frame = pd.read_parquet(paths["target"])
    with paths["metadata"].open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if target_frame.shape[1] != 1:
        raise ResearchV2DataError("Target file must contain exactly one column.")
    y = target_frame.iloc[:, 0]
    _validate_source_legacy(X_source, y, metadata, config)
    pipeline = get_feature_pipeline(config.pipeline_id)
    output = pipeline.transform(X_source, config.pipeline_contract)
    provenance = {
        "dataset_id": config.dataset_version,
        "parent_dataset_id": None,
        "hypothesis": None,
        "n_features": int(X_source.shape[1]),
        "schema_hash": None,
        "train_content_hash": file_content_sha256(paths["train_features"]),
        "target_hash": canonical_sha256(
            {
                "name": str(y.name),
                "dtype": str(y.dtype),
                "values": y.tolist(),
            }
        ),
        "target_dependency": None,
        "train_row_identity_hash": None,
        "registry_schema_version": None,
        "source": "legacy_plan",
    }
    identity = _build_identity(
        config=config,
        X_source=X_source,
        y=y,
        file_fingerprints=fingerprints,
        provenance=provenance,
    )
    return ResearchV2TrainingData(
        X=output.features,
        y=y,
        pipeline_output=output,
        metadata=metadata if isinstance(metadata, dict) else {},
        fingerprints=identity,
        dataset_provenance=provenance,
    )


def _build_identity(
    *,
    config: ResearchV2Config,
    X_source: pd.DataFrame,
    y: pd.Series,
    file_fingerprints: dict[str, dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    row_positions = list(range(len(X_source)))
    row_position_identity: dict[str, Any] = {
        "kind": "contiguous_zero_based",
        "sha256": canonical_sha256(row_positions),
    }
    train_row_identity_hash = provenance.get("train_row_identity_hash")
    if train_row_identity_hash is not None:
        row_position_identity["bound_train_row_identity_hash"] = train_row_identity_hash
    return {
        "dataset_version": config.dataset_version,
        "files": file_fingerprints,
        "row_count": len(X_source),
        "row_position_identity": row_position_identity,
        "dataset_provenance": provenance,
        "source_schema": {
            "ordered_names": X_source.columns.tolist(),
            "ordered_names_sha256": ordered_feature_schema_sha256(
                X_source.columns.tolist()
            ),
            "ordered_dtypes": [
                {"name": name, "dtype": str(dtype)}
                for name, dtype in X_source.dtypes.items()
            ],
            "ordered_dtypes_sha256": canonical_sha256(
                [
                    {"name": name, "dtype": str(dtype)}
                    for name, dtype in X_source.dtypes.items()
                ]
            ),
        },
        "target": {
            "name": str(y.name),
            "dtype": str(y.dtype),
            "negative_count": int((y == 0).sum()),
            "positive_count": int((y == 1).sum()),
            "values_sha256": canonical_sha256(
                {
                    "name": str(y.name),
                    "dtype": str(y.dtype),
                    "values": y.tolist(),
                }
            ),
        },
    }


def _cross_check_plan_against_registry(
    config: ResearchV2Config,
    manifest: DatasetManifest,
    X_source: pd.DataFrame,
    y: pd.Series,
) -> None:
    dataset = config.plan_payload["dataset"]
    if int(dataset["expected_rows"]) != int(manifest.train_row_count):
        raise ResearchV2DataError(
            "Evaluation-plan expected_rows disagrees with Registry train_row_count."
        )
    if int(dataset["expected_source_features"]) != int(manifest.n_features):
        raise ResearchV2DataError(
            "Evaluation-plan expected_source_features disagrees with Registry "
            "n_features."
        )
    feature_names = [feature.name for feature in manifest.features]
    if feature_names != X_source.columns.tolist():
        raise ResearchV2DataError(
            "Registry manifest feature order disagrees with X_train columns."
        )
    if ordered_feature_schema_sha256(feature_names) != dataset[
        "ordered_source_schema_sha256"
    ]:
        raise ResearchV2DataError(
            "Evaluation-plan ordered source schema disagrees with Registry features."
        )
    dtype_hash = canonical_sha256(
        [{"name": name, "dtype": str(dtype)} for name, dtype in X_source.dtypes.items()]
    )
    if dtype_hash != dataset["ordered_dtype_schema_sha256"]:
        raise ResearchV2DataError(
            "Evaluation-plan ordered dtype schema disagrees with Registry package."
        )
    target = dataset["target"]
    target_hash = canonical_sha256(
        {"name": str(y.name), "dtype": str(y.dtype), "values": y.tolist()}
    )
    if target_hash != manifest.target.hash:
        raise ResearchV2DataError(
            "Loaded target hash disagrees with Registry target.hash."
        )
    if target_hash != target["values_sha256"]:
        raise ResearchV2DataError(
            "Evaluation-plan target fingerprint disagrees with Registry target.hash."
        )


def _validate_source_common(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: dict[str, Any],
    config: ResearchV2Config,
) -> None:
    del metadata
    dataset = config.plan_payload["dataset"]
    target = dataset["target"]
    rows = int(dataset["expected_rows"])
    if len(X) != rows or len(y) != rows or not X.index.equals(y.index):
        raise ResearchV2DataError("Training rows are not aligned with the plan.")
    if not X.index.equals(pd.RangeIndex(rows)):
        raise ResearchV2DataError("Row identity must be a zero-based RangeIndex.")
    if str(y.name) != target["name"] or str(y.dtype) != target["dtype"]:
        raise ResearchV2DataError("Target name or dtype differs from the plan.")
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ResearchV2DataError(
            "Target must contain only non-missing labels 0 and 1."
        )
    counts = y.value_counts().sort_index().to_dict()
    expected_counts = {
        int(target["negative_label"]): int(target["expected_negative_rows"]),
        int(target["positive_label"]): int(target["expected_positive_rows"]),
    }
    if counts != expected_counts:
        raise ResearchV2DataError("Target class counts differ from the plan.")


def _validate_source_legacy(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: dict[str, Any],
    config: ResearchV2Config,
) -> None:
    dataset = config.plan_payload["dataset"]
    _validate_source_common(X, y, metadata if isinstance(metadata, dict) else {}, config)
    if X.shape[1] != int(dataset["expected_source_features"]):
        raise ResearchV2DataError("Source feature count differs from the plan.")
    if (
        ordered_feature_schema_sha256(X.columns.tolist())
        != dataset["ordered_source_schema_sha256"]
    ):
        raise ResearchV2DataError("Ordered source schema differs from the plan.")
    dtype_hash = canonical_sha256(
        [{"name": name, "dtype": str(dtype)} for name, dtype in X.dtypes.items()]
    )
    if dtype_hash != dataset["ordered_dtype_schema_sha256"]:
        raise ResearchV2DataError("Ordered dtype schema differs from the plan.")
    target = dataset["target"]
    target_hash = canonical_sha256(
        {"name": str(y.name), "dtype": str(y.dtype), "values": y.tolist()}
    )
    if target_hash != target["values_sha256"]:
        raise ResearchV2DataError("Target fingerprint differs from the plan.")
    if not isinstance(metadata, dict) or metadata.get("version") != config.dataset_version:
        raise ResearchV2DataError("Dataset metadata version differs from the plan.")


def _is_contained(path: Path, root: Path) -> bool:
    return path == root or root in path.parents
