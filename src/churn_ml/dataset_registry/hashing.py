"""Deterministic hashing helpers for dataset packages."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.churn_ml.dataset_registry.schema import FeatureSpec
from src.churn_ml.research_data import canonical_sha256
from src.churn_ml.run_artifacts import fingerprint_file


def file_content_sha256(path: Path) -> str:
    return str(fingerprint_file(path)["sha256"])


def schema_hash(features: Sequence[FeatureSpec]) -> str:
    payload = [
        {"name": feature.name, "dtype": feature.dtype, "role": feature.role}
        for feature in features
    ]
    return canonical_sha256(payload)


def dataframe_content_sha256(frame: pd.DataFrame) -> str:
    """Hash ordered frame content without relying on the pandas index."""
    return hashlib.sha256(
        frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def series_content_sha256(series: pd.Series) -> str:
    return canonical_sha256(
        {
            "name": str(series.name),
            "dtype": str(series.dtype),
            "values": series.tolist(),
        }
    )


def target_hash(series: pd.Series) -> str:
    return series_content_sha256(series)


def class_counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False).to_dict()
    return {
        str(key): int(value)
        for key, value in sorted(counts.items(), key=_sort_key)
    }


def _sort_key(item: tuple[Any, Any]) -> tuple[str, str]:
    key = item[0]
    return (type(key).__name__, str(key))
