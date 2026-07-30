"""Public loader/resolver API for Dataset Package & Registry v1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.churn_ml.dataset_registry.discovery import ScanEntry, scan_registry
from src.churn_ml.dataset_registry.errors import DatasetRegistryError
from src.churn_ml.dataset_registry.manifest import read_manifest
from src.churn_ml.dataset_registry.package import (
    PackageArtifacts,
    load_package_artifacts,
    manifest_path,
    package_dir_for,
)
from src.churn_ml.dataset_registry.schema import DatasetManifest
from src.churn_ml.dataset_registry.validation import ValidationResult, validate_package_dir


@dataclass(frozen=True)
class DatasetPackage:
    """Resolved, validated dataset package suitable for later pipeline use."""

    root: Path
    package_dir: Path
    manifest: DatasetManifest
    artifacts: PackageArtifacts
    validation: ValidationResult

    @property
    def dataset_id(self) -> str:
        return self.manifest.dataset_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "package_dir": str(self.package_dir),
            "dataset_id": self.dataset_id,
            "manifest": self.manifest.to_dict(),
            "validation": self.validation.to_dict(),
        }


def load_dataset_manifest(package_dir: Path) -> DatasetManifest:
    return read_manifest(manifest_path(package_dir))


def validate_dataset_package(
    package_dir: Path,
    *,
    root: Path | None = None,
) -> ValidationResult:
    return validate_package_dir(package_dir, root=root)


def list_dataset_packages(root: Path) -> list[ScanEntry]:
    return scan_registry(root)


def resolve_dataset_package(root: Path, dataset_id: str) -> DatasetPackage:
    """
    Resolve and strictly validate a registered package under root.

    Raises DatasetRegistryError when the package is missing, unregistered,
    invalid, or has unverifiable alignment.
    """
    root = root.resolve()
    package_dir = package_dir_for(root, dataset_id)
    validation = validate_package_dir(package_dir, root=root)
    if validation.status != "valid" or validation.manifest is None:
        raise DatasetRegistryError(
            validation.message,
            status=validation.status,
        )
    artifacts = load_package_artifacts(package_dir, dataset_id=dataset_id)
    return DatasetPackage(
        root=root,
        package_dir=package_dir,
        manifest=validation.manifest,
        artifacts=artifacts,
        validation=validation,
    )
