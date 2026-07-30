"""Synthetic dataset package helpers for Dataset Registry v1 tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.churn_ml.dataset_registry.alignment import prove_alignment
from src.churn_ml.dataset_registry.constants import MANIFEST_FILENAME, SCHEMA_VERSION
from src.churn_ml.dataset_registry.manifest import build_manifest, write_manifest
from src.churn_ml.dataset_registry.package import load_package_artifacts


V0_FEATURES = ("f1", "f2", "f3")
V1_ENGINEERED = ("missing_count_total", "missing_rate_total")


def write_package_files(
    package_dir: Path,
    *,
    dataset_id: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    package_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(package_dir / "X_train.parquet", index=False)
    y_train.to_frame(name=y_train.name or "y").to_parquet(
        package_dir / "y_train.parquet",
        index=False,
    )
    X_test.to_parquet(package_dir / "X_test.parquet", index=False)
    payload = {
        "version": dataset_id,
        "metadata": dict(metadata or {"description": "synthetic"}),
    }
    (package_dir / "metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return package_dir


def make_v0_frames(
    *,
    train_rows: int = 6,
    test_rows: int = 3,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rng = pd.RangeIndex(train_rows)
    X_train = pd.DataFrame(
        {
            "f1": [float(i + seed) for i in rng],
            "f2": [float((i + seed) % 3) for i in rng],
            "f3": [f"c{(i + seed) % 2}" for i in rng],
        }
    )
    y_train = pd.Series(
        [int(i % 2) for i in rng],
        name="y",
        dtype="int64",
    )
    test_index = pd.RangeIndex(test_rows)
    X_test = pd.DataFrame(
        {
            "f1": [float(100 + i + seed) for i in test_index],
            "f2": [float((i + seed) % 3) for i in test_index],
            "f3": [f"c{(i + 1 + seed) % 2}" for i in test_index],
        }
    )
    return X_train, y_train, X_test


def add_v1_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["missing_count_total"] = 0
    result["missing_rate_total"] = 0.0
    return result


def create_synthetic_legacy_tree(root: Path) -> Path:
    """Create v0 + v1 synthetic packages sharing the ordered v0 projection."""
    root.mkdir(parents=True, exist_ok=True)
    X_train, y_train, X_test = make_v0_frames()
    write_package_files(
        root / "v0_raw_minimal",
        dataset_id="v0_raw_minimal",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        metadata={"source": "raw", "description": "synthetic v0"},
    )
    write_package_files(
        root / "v1_missingness_summary",
        dataset_id="v1_missingness_summary",
        X_train=add_v1_features(X_train),
        y_train=y_train,
        X_test=add_v1_features(X_test),
        metadata={
            "source": "v0_raw_minimal",
            "base_version": "v0_raw_minimal",
            "added_features": list(V1_ENGINEERED),
        },
    )
    return root


def register_package(
    root: Path,
    dataset_id: str,
    *,
    parent_dataset_id: str | None,
    hypothesis: str,
    target_dependency: str = "none",
    transformations: Sequence[Mapping[str, Any]] | None = None,
    summary_features: Sequence[str] | None = None,
    binary_indicator_features: Sequence[str] | None = None,
    use_missing_suffix: bool = False,
) -> Path:
    package_dir = root / dataset_id
    artifacts = load_package_artifacts(package_dir, dataset_id=dataset_id)
    parent_artifacts = None
    if parent_dataset_id is not None:
        parent_artifacts = load_package_artifacts(
            root / parent_dataset_id,
            dataset_id=parent_dataset_id,
        )
    alignment = prove_alignment(
        artifacts,
        parent_artifacts=parent_artifacts,
        root=root,
    )
    manifest = build_manifest(
        artifacts,
        hypothesis=hypothesis,
        parent_dataset_id=parent_dataset_id,
        target_dependency=target_dependency,  # type: ignore[arg-type]
        transformations=list(transformations or [{"type": "synthetic"}]),
        alignment=alignment,
        summary_features=summary_features,
        binary_indicator_features=binary_indicator_features,
        use_missing_suffix=use_missing_suffix,
    )
    write_manifest(package_dir / MANIFEST_FILENAME, manifest)
    assert manifest.schema_version == SCHEMA_VERSION
    return package_dir


def snapshot_tree(root: Path) -> dict[str, tuple[int, bytes]]:
    """Capture relative path -> (mtime_ns, content) for write-detection tests."""
    snapshot: dict[str, tuple[int, bytes]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            snapshot[rel] = (path.stat().st_mtime_ns, path.read_bytes())
    return snapshot
