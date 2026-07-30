"""Load and inspect on-disk dataset package artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.churn_ml.dataset_registry.constants import MANIFEST_FILENAME, PACKAGE_FILES
from src.churn_ml.dataset_registry.errors import DatasetRegistryError


@dataclass(frozen=True)
class PackageArtifacts:
    package_dir: Path
    dataset_id: str
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    metadata: dict[str, Any]
    paths: dict[str, Path]


def package_dir_for(root: Path, dataset_id: str) -> Path:
    return root / dataset_id


def manifest_path(package_dir: Path) -> Path:
    return package_dir / MANIFEST_FILENAME


def has_package_files(package_dir: Path) -> bool:
    return all((package_dir / name).is_file() for name in PACKAGE_FILES)


def has_manifest(package_dir: Path) -> bool:
    return manifest_path(package_dir).is_file()


def load_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise DatasetRegistryError("metadata.json must be a JSON object.")
    return payload


def load_package_artifacts(package_dir: Path, *, dataset_id: str | None = None) -> PackageArtifacts:
    """Load parquet/metadata for a package directory. Does not read the manifest."""
    package_dir = package_dir.resolve()
    if not package_dir.is_dir():
        raise DatasetRegistryError(f"Package directory does not exist: {package_dir}")
    missing = [name for name in PACKAGE_FILES if not (package_dir / name).is_file()]
    if missing:
        raise DatasetRegistryError(
            f"Package is incomplete. Missing files: {missing}."
        )
    resolved_id = dataset_id or package_dir.name
    paths = {
        "X_train": package_dir / "X_train.parquet",
        "y_train": package_dir / "y_train.parquet",
        "X_test": package_dir / "X_test.parquet",
        "metadata": package_dir / "metadata.json",
    }
    X_train = pd.read_parquet(paths["X_train"])
    y_frame = pd.read_parquet(paths["y_train"])
    X_test = pd.read_parquet(paths["X_test"])
    if y_frame.shape[1] != 1:
        raise DatasetRegistryError("y_train.parquet must contain exactly one column.")
    y_train = y_frame.iloc[:, 0]
    metadata = load_metadata(paths["metadata"])
    version = metadata.get("version")
    if version is not None and version != resolved_id:
        raise DatasetRegistryError(
            f"metadata version {version!r} does not match dataset_id {resolved_id!r}."
        )
    return PackageArtifacts(
        package_dir=package_dir,
        dataset_id=resolved_id,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        metadata=metadata,
        paths=paths,
    )


def validate_basic_shapes(artifacts: PackageArtifacts) -> None:
    X_train = artifacts.X_train
    y_train = artifacts.y_train
    X_test = artifacts.X_test
    if len(X_train) != len(y_train):
        raise DatasetRegistryError("X_train and y_train row counts differ.")
    if list(X_train.columns) != list(X_test.columns):
        raise DatasetRegistryError(
            "X_train and X_test must share the same ordered feature columns."
        )
    if len(X_train.columns) == 0:
        raise DatasetRegistryError("Feature matrix must contain at least one column.")
    if not X_train.index.equals(pd.RangeIndex(len(X_train))):
        raise DatasetRegistryError("X_train index must be a contiguous RangeIndex.")
    if not y_train.index.equals(pd.RangeIndex(len(y_train))):
        raise DatasetRegistryError("y_train index must be a contiguous RangeIndex.")
    if not X_test.index.equals(pd.RangeIndex(len(X_test))):
        raise DatasetRegistryError("X_test index must be a contiguous RangeIndex.")
    if not X_train.index.equals(y_train.index):
        raise DatasetRegistryError("X_train and y_train indexes are not aligned.")
