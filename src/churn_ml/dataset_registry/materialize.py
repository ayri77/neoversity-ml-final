"""Materialize Registry-owned dataset_manifest.json for native packages."""

from __future__ import annotations

from pathlib import Path

from src.churn_ml.dataset_registry.alignment import (
    load_anchor_feature_names,
    prove_alignment,
)
from src.churn_ml.dataset_registry.build_spec import DatasetBuildSpec
from src.churn_ml.dataset_registry.constants import (
    MANIFEST_FILENAME,
    ROW_IDENTITY_ANCHOR_DATASET_ID,
)
from src.churn_ml.dataset_registry.errors import (
    DatasetRegistryError,
    OverwriteRefusedError,
    UnverifiableAlignmentError,
)
from src.churn_ml.dataset_registry.hashing import file_content_sha256
from src.churn_ml.dataset_registry.manifest import build_manifest, write_manifest
from src.churn_ml.dataset_registry.package import (
    has_manifest,
    has_package_files,
    load_package_artifacts,
    manifest_path,
    package_dir_for,
    validate_basic_shapes,
)
from src.churn_ml.dataset_registry.roles import role_catalog_from_sets
from src.churn_ml.dataset_registry.schema import DatasetManifest
from src.churn_ml.dataset_registry.validation import validate_package_dir


def dataset_manifest_sha256(package_dir: Path) -> str:
    """Return the physical SHA-256 of dataset_manifest.json."""
    path = manifest_path(package_dir)
    if not path.is_file():
        raise DatasetRegistryError(
            f"dataset_manifest.json is missing under {package_dir}."
        )
    return file_content_sha256(path)


def refuse_registered_overwrite(package_dir: Path) -> None:
    if has_manifest(package_dir):
        raise OverwriteRefusedError(
            "Registered dataset packages are immutable. "
            f"Refusing to overwrite {manifest_path(package_dir)}. "
            "Use a new dataset_id instead."
        )


def materialize_dataset_manifest(
    package_dir: Path,
    *,
    root: Path,
    build_spec: DatasetBuildSpec,
) -> DatasetManifest:
    """
    Build, write, and strictly validate dataset_manifest.json.

    The four package artifacts must already exist under package_dir.
    Parent packages, when required, must already exist under root.
    """
    package_dir = package_dir.resolve()
    root = root.resolve()
    if build_spec.target_dependency == "fold_local":
        raise DatasetRegistryError(
            "target_dependency='fold_local' cannot be materialized as a prepared "
            "Parquet package; fold-local transforms belong inside a CV pipeline."
        )
    if build_spec.dataset_id != package_dir.name:
        raise DatasetRegistryError(
            f"build_spec.dataset_id {build_spec.dataset_id!r} does not match "
            f"package directory name {package_dir.name!r}."
        )
    if has_manifest(package_dir):
        raise OverwriteRefusedError(
            f"Refusing to overwrite existing manifest: {manifest_path(package_dir)}"
        )
    if not has_package_files(package_dir):
        raise DatasetRegistryError(
            f"Cannot materialize manifest; package files are incomplete under "
            f"{package_dir}."
        )

    artifacts = load_package_artifacts(
        package_dir,
        dataset_id=build_spec.dataset_id,
    )
    validate_basic_shapes(artifacts)

    parent_artifacts = None
    if build_spec.parent_dataset_id is not None:
        parent_dir = package_dir_for(root, build_spec.parent_dataset_id)
        if not has_package_files(parent_dir):
            raise DatasetRegistryError(
                f"Parent package {build_spec.parent_dataset_id!r} is missing or "
                "incomplete; child alignment cannot be proven."
            )
        parent_artifacts = load_package_artifacts(
            parent_dir,
            dataset_id=build_spec.parent_dataset_id,
        )
        validate_basic_shapes(parent_artifacts)

    anchor_names = _resolve_anchor_names(root, artifacts, parent_artifacts)
    try:
        alignment = prove_alignment(
            artifacts,
            anchor_feature_names=anchor_names,
            parent_artifacts=parent_artifacts,
            root=root,
        )
    except UnverifiableAlignmentError:
        raise

    role_catalog = role_catalog_from_sets(
        summary_features=build_spec.summary_features,
        binary_indicator_features=build_spec.binary_indicator_features,
        use_missing_suffix=build_spec.use_missing_suffix,
    )
    manifest = build_manifest(
        artifacts,
        hypothesis=build_spec.hypothesis,
        parent_dataset_id=build_spec.parent_dataset_id,
        target_dependency=build_spec.target_dependency,
        transformations=build_spec.transformations,
        alignment=alignment,
        role_catalog=role_catalog,
    )
    write_manifest(package_dir / MANIFEST_FILENAME, manifest)
    validation = validate_package_dir(package_dir, root=root)
    if validation.status != "valid":
        manifest_file = package_dir / MANIFEST_FILENAME
        if manifest_file.exists():
            manifest_file.unlink()
        raise DatasetRegistryError(
            f"Final package validation failed with status "
            f"{validation.status}: {validation.message}",
            status=validation.status,
        )
    return manifest


def _resolve_anchor_names(
    root: Path,
    artifacts,
    parent_artifacts,
) -> tuple[str, ...]:
    anchor_dir = package_dir_for(root, ROW_IDENTITY_ANCHOR_DATASET_ID)
    if anchor_dir.is_dir() and has_package_files(anchor_dir):
        return load_anchor_feature_names(root)
    if parent_artifacts is not None:
        return tuple(parent_artifacts.X_train.columns.tolist())
    return tuple(artifacts.X_train.columns.tolist())
