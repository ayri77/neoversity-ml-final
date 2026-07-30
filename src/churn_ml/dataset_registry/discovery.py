"""Discover dataset package directories under a registry root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.churn_ml.dataset_registry.constants import LEGACY_UNREGISTERED_IDS
from src.churn_ml.dataset_registry.errors import PackageStatus
from src.churn_ml.dataset_registry.package import has_manifest, has_package_files
from src.churn_ml.dataset_registry.validation import ValidationResult, validate_package_dir


@dataclass(frozen=True)
class ScanEntry:
    dataset_id: str
    package_dir: Path
    status: PackageStatus
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "package_dir": str(self.package_dir),
            "status": self.status,
            "message": self.message,
        }


def iter_candidate_dirs(root: Path) -> list[Path]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Registry root does not exist: {root}")
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def classify_directory(package_dir: Path, *, root: Path | None = None) -> ScanEntry:
    package_dir = package_dir.resolve()
    dataset_id = package_dir.name
    if dataset_id in LEGACY_UNREGISTERED_IDS:
        return ScanEntry(
            dataset_id=dataset_id,
            package_dir=package_dir,
            status="unregistered",
            message="Obsolete or unimplemented dataset name; intentionally unregistered.",
        )
    if not has_package_files(package_dir):
        return ScanEntry(
            dataset_id=dataset_id,
            package_dir=package_dir,
            status="unregistered",
            message="Directory is not a complete dataset package.",
        )
    if not has_manifest(package_dir):
        return ScanEntry(
            dataset_id=dataset_id,
            package_dir=package_dir,
            status="unregistered",
            message="Package files are present but dataset_manifest.json is missing.",
        )
    result = validate_package_dir(package_dir, root=root or package_dir.parent)
    return ScanEntry(
        dataset_id=result.dataset_id,
        package_dir=result.package_dir,
        status=result.status,
        message=result.message,
    )


def scan_registry(root: Path) -> list[ScanEntry]:
    return [classify_directory(path, root=root) for path in iter_candidate_dirs(root)]


def validation_from_scan(entry: ScanEntry, root: Path) -> ValidationResult:
    return validate_package_dir(entry.package_dir, root=root)
