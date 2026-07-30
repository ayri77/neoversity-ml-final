"""Public discovery of strictly registered prepared datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.churn_ml.dataset_registry.api import resolve_dataset_package
from src.churn_ml.dataset_registry.errors import DatasetRegistryError
from src.churn_ml.dataset_registry.package import has_manifest, has_package_files
from src.churn_ml.dataset_registry.schema import DatasetManifest


@dataclass(frozen=True)
class RegisteredDatasetSummary:
    """Authoritative summary of one strictly validated Registry package."""

    dataset_id: str
    parent_dataset_id: str | None
    n_features: int
    hypothesis: str
    target_dependency: str
    schema_hash: str
    train_content_hash: str
    target_hash: str
    train_row_identity_hash: str
    package_dir: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "parent_dataset_id": self.parent_dataset_id,
            "n_features": self.n_features,
            "hypothesis": self.hypothesis,
            "target_dependency": self.target_dependency,
            "schema_hash": self.schema_hash,
            "train_content_hash": self.train_content_hash,
            "target_hash": self.target_hash,
            "train_row_identity_hash": self.train_row_identity_hash,
            "package_dir": str(self.package_dir),
        }


def discover_registered_datasets(root: Path) -> list[RegisteredDatasetSummary]:
    """
    Discover immediately nested, strictly validated Registry packages.

    Recognition requires canonical dataset_manifest.json. Legacy manifest.json
    and incomplete four-file directories are never treated as registered. Results
    are sorted by dataset_id.
    """
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Processed-data root does not exist: {root}")

    summaries: list[RegisteredDatasetSummary] = []
    seen_ids: set[str] = set()
    for package_dir in sorted(
        path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
    ):
        if not has_package_files(package_dir) or not has_manifest(package_dir):
            continue
        package = resolve_dataset_package(root, package_dir.name)
        summary = summary_from_manifest(package.manifest, package_dir=package.package_dir)
        if summary.dataset_id != package_dir.name:
            raise DatasetRegistryError(
                "Directory name does not match dataset_manifest.dataset_id: "
                f"directory={package_dir.name!r}, "
                f"dataset_id={summary.dataset_id!r}."
            )
        if summary.dataset_id in seen_ids:
            raise DatasetRegistryError(
                f"Duplicate registered dataset_id discovered: {summary.dataset_id!r}."
            )
        seen_ids.add(summary.dataset_id)
        summaries.append(summary)
    return sorted(summaries, key=lambda item: item.dataset_id)


def summary_from_manifest(
    manifest: DatasetManifest,
    *,
    package_dir: Path,
) -> RegisteredDatasetSummary:
    return RegisteredDatasetSummary(
        dataset_id=manifest.dataset_id,
        parent_dataset_id=manifest.parent_dataset_id,
        n_features=manifest.n_features,
        hypothesis=manifest.hypothesis,
        target_dependency=manifest.target_dependency,
        schema_hash=manifest.schema_hash,
        train_content_hash=manifest.content_hashes["X_train"],
        target_hash=manifest.target.hash,
        train_row_identity_hash=manifest.row_identity.train_hash,
        package_dir=package_dir.resolve(),
    )


def build_dataset_provenance(manifest: DatasetManifest) -> dict[str, Any]:
    """Canonical Research v2 dataset provenance payload from a Registry manifest."""
    return {
        "dataset_id": manifest.dataset_id,
        "parent_dataset_id": manifest.parent_dataset_id,
        "hypothesis": manifest.hypothesis,
        "n_features": manifest.n_features,
        "schema_hash": manifest.schema_hash,
        "train_content_hash": manifest.content_hashes["X_train"],
        "target_hash": manifest.target.hash,
        "target_dependency": manifest.target_dependency,
        "train_row_identity_hash": manifest.row_identity.train_hash,
        "registry_schema_version": manifest.schema_version,
    }
