"""Validate registered dataset packages against on-disk artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.churn_ml.dataset_registry.alignment import (
    load_anchor_feature_names,
    prove_alignment,
)
from src.churn_ml.dataset_registry.constants import ROW_IDENTITY_ANCHOR_DATASET_ID
from src.churn_ml.dataset_registry.errors import (
    DatasetRegistryError,
    PackageStatus,
    UnverifiableAlignmentError,
)
from src.churn_ml.dataset_registry.hashing import (
    class_counts,
    file_content_sha256,
    schema_hash,
    target_hash,
)
from src.churn_ml.dataset_registry.manifest import read_manifest
from src.churn_ml.dataset_registry.package import (
    PackageArtifacts,
    has_manifest,
    has_package_files,
    load_package_artifacts,
    manifest_path,
    package_dir_for,
    validate_basic_shapes,
)
from src.churn_ml.dataset_registry.schema import DatasetManifest


@dataclass(frozen=True)
class ValidationResult:
    status: PackageStatus
    dataset_id: str
    package_dir: Path
    message: str
    manifest: DatasetManifest | None = None
    mismatches: tuple[str, ...] = ()
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "dataset_id": self.dataset_id,
            "package_dir": str(self.package_dir),
            "message": self.message,
            "mismatches": list(self.mismatches),
        }
        if self.manifest is not None:
            payload["manifest"] = self.manifest.to_dict()
        if self.details is not None:
            payload["details"] = self.details
        return payload


def validate_package_dir(package_dir: Path, *, root: Path | None = None) -> ValidationResult:
    package_dir = package_dir.resolve()
    dataset_id = package_dir.name
    if not package_dir.is_dir():
        return ValidationResult(
            status="invalid",
            dataset_id=dataset_id,
            package_dir=package_dir,
            message="Package directory does not exist.",
        )
    if not has_package_files(package_dir):
        if has_manifest(package_dir):
            return ValidationResult(
                status="invalid",
                dataset_id=dataset_id,
                package_dir=package_dir,
                message="Manifest exists but package parquet/metadata files are incomplete.",
            )
        return ValidationResult(
            status="unregistered",
            dataset_id=dataset_id,
            package_dir=package_dir,
            message="Directory is not a complete dataset package.",
        )
    if not has_manifest(package_dir):
        return ValidationResult(
            status="unregistered",
            dataset_id=dataset_id,
            package_dir=package_dir,
            message="Package files are present but dataset_manifest.json is missing.",
        )
    try:
        manifest = read_manifest(manifest_path(package_dir))
    except DatasetRegistryError as error:
        return ValidationResult(
            status="invalid",
            dataset_id=dataset_id,
            package_dir=package_dir,
            message=str(error),
        )
    if manifest.dataset_id != dataset_id:
        return ValidationResult(
            status="invalid",
            dataset_id=dataset_id,
            package_dir=package_dir,
            message=(
                f"Manifest dataset_id {manifest.dataset_id!r} does not match "
                f"directory name {dataset_id!r}."
            ),
            manifest=manifest,
            mismatches=("dataset_id",),
        )
    try:
        artifacts = load_package_artifacts(package_dir, dataset_id=dataset_id)
        validate_basic_shapes(artifacts)
    except DatasetRegistryError as error:
        return ValidationResult(
            status="invalid",
            dataset_id=dataset_id,
            package_dir=package_dir,
            message=str(error),
            manifest=manifest,
        )
    mismatches = _content_and_schema_mismatches(manifest, artifacts)
    registry_root = root.resolve() if root is not None else package_dir.parent
    try:
        parent_artifacts = _load_parent(manifest, registry_root)
        alignment = prove_alignment(
            artifacts,
            anchor_feature_names=_anchor_names(
                artifacts,
                parent_artifacts=parent_artifacts,
                root=registry_root,
            ),
            parent_artifacts=parent_artifacts,
            root=registry_root,
        )
    except UnverifiableAlignmentError as error:
        if mismatches:
            mismatches.append("row_identity.unverifiable")
            return ValidationResult(
                status="invalid",
                dataset_id=dataset_id,
                package_dir=package_dir,
                message=(
                    "Protected package properties no longer match the manifest; "
                    f"alignment also unverifiable: {error}"
                ),
                manifest=manifest,
                mismatches=tuple(mismatches),
            )
        return ValidationResult(
            status="unverifiable_alignment",
            dataset_id=dataset_id,
            package_dir=package_dir,
            message=str(error),
            manifest=manifest,
        )
    except DatasetRegistryError as error:
        return ValidationResult(
            status="invalid",
            dataset_id=dataset_id,
            package_dir=package_dir,
            message=str(error),
            manifest=manifest,
            mismatches=tuple(mismatches),
        )
    if alignment.train_hash != manifest.row_identity.train_hash:
        mismatches.append("row_identity.train_hash")
    if alignment.test_hash != manifest.row_identity.test_hash:
        mismatches.append("row_identity.test_hash")
    if alignment.train_anchor_hash != manifest.row_identity.train_anchor_hash:
        mismatches.append("row_identity.train_anchor_hash")
    if alignment.test_anchor_hash != manifest.row_identity.test_anchor_hash:
        mismatches.append("row_identity.test_anchor_hash")
    if manifest.row_identity.alignment_status != "proven":
        mismatches.append("row_identity.alignment_status")
    if mismatches:
        return ValidationResult(
            status="invalid",
            dataset_id=dataset_id,
            package_dir=package_dir,
            message="Protected package properties no longer match the manifest.",
            manifest=manifest,
            mismatches=tuple(mismatches),
        )
    return ValidationResult(
        status="valid",
        dataset_id=dataset_id,
        package_dir=package_dir,
        message="Package is valid and immutable properties match on-disk artifacts.",
        manifest=manifest,
        details={
            "parent_dataset_id": manifest.parent_dataset_id,
            "schema_hash": manifest.schema_hash,
            "target_dependency": manifest.target_dependency,
            "alignment_method": manifest.row_identity.alignment_method,
        },
    )


def _content_and_schema_mismatches(
    manifest: DatasetManifest,
    artifacts: PackageArtifacts,
) -> list[str]:
    mismatches: list[str] = []
    for key, path in artifacts.paths.items():
        if file_content_sha256(path) != manifest.content_hashes[key]:
            mismatches.append(f"content_hashes.{key}")
    actual_names = [str(name) for name in artifacts.X_train.columns.tolist()]
    actual_dtypes = [str(dtype) for dtype in artifacts.X_train.dtypes.tolist()]
    expected_names = [feature.name for feature in manifest.features]
    expected_dtypes = [feature.dtype for feature in manifest.features]
    if actual_names != expected_names:
        mismatches.append("feature_order")
    if actual_dtypes != expected_dtypes:
        mismatches.append("feature_dtypes")
    if len(artifacts.X_train) != manifest.train_row_count:
        mismatches.append("train_row_count")
    if len(artifacts.X_test) != manifest.test_row_count:
        mismatches.append("test_row_count")
    if len(actual_names) != manifest.n_features:
        mismatches.append("n_features")
    if schema_hash(manifest.features) != manifest.schema_hash:
        mismatches.append("schema_hash")
    if str(artifacts.y_train.name) != manifest.target.name:
        mismatches.append("target.name")
    if str(artifacts.y_train.dtype) != manifest.target.dtype:
        mismatches.append("target.dtype")
    if class_counts(artifacts.y_train) != manifest.target.class_counts:
        mismatches.append("target.class_counts")
    if target_hash(artifacts.y_train) != manifest.target.hash:
        mismatches.append("target.hash")
    return mismatches


def _load_parent(
    manifest: DatasetManifest,
    root: Path,
) -> PackageArtifacts | None:
    if manifest.parent_dataset_id is None:
        return None
    parent_dir = package_dir_for(root, manifest.parent_dataset_id)
    return load_package_artifacts(parent_dir, dataset_id=manifest.parent_dataset_id)


def _anchor_names(
    artifacts: PackageArtifacts,
    *,
    parent_artifacts: PackageArtifacts | None,
    root: Path,
) -> tuple[str, ...]:
    anchor_dir = package_dir_for(root, ROW_IDENTITY_ANCHOR_DATASET_ID)
    if anchor_dir.is_dir() and has_package_files(anchor_dir):
        return load_anchor_feature_names(root)
    if parent_artifacts is not None:
        return tuple(parent_artifacts.X_train.columns.tolist())
    return tuple(artifacts.X_train.columns.tolist())
