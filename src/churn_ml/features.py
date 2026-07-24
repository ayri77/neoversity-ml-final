from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd
from collections.abc import Sequence


@dataclass(frozen=True)
class PreparedDataset:
    version: str
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    metadata: dict[str, Any]


def add_missingness_summary_features(
    dataframe: pd.DataFrame,
    *,
    very_high_missing_features: Sequence[str] = (),
) -> pd.DataFrame:
    """Add row-level missing-value summary features."""
    result = dataframe.copy()

    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()

    categorical_columns = dataframe.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    very_high_missing_columns = [
        column for column in very_high_missing_features if column in dataframe.columns
    ]

    result["missing_count_total"] = dataframe.isna().sum(axis=1).astype("int16")
    result["missing_rate_total"] = dataframe.isna().mean(axis=1).astype("float32")

    result["missing_count_numeric"] = (
        dataframe[numeric_columns].isna().sum(axis=1).astype("int16")
    )
    result["missing_rate_numeric"] = (
        dataframe[numeric_columns].isna().mean(axis=1).astype("float32")
    )

    result["missing_count_categorical"] = (
        dataframe[categorical_columns].isna().sum(axis=1).astype("int16")
    )
    result["missing_rate_categorical"] = (
        dataframe[categorical_columns].isna().mean(axis=1).astype("float32")
    )

    if very_high_missing_columns:
        result["missing_count_very_high"] = (
            dataframe[very_high_missing_columns].isna().sum(axis=1).astype("int16")
        )
        result["missing_rate_very_high"] = (
            dataframe[very_high_missing_columns].isna().mean(axis=1).astype("float32")
        )
    else:
        result["missing_count_very_high"] = 0
        result["missing_rate_very_high"] = 0.0

    return result


def add_missingness_indicator_features(
    dataframe: pd.DataFrame,
    *,
    indicator_features: Sequence[str],
    very_high_missing_features: Sequence[str] = (),
) -> pd.DataFrame:
    """
    Add row-level missing counts and per-feature missing indicators.

    Indicator features must be selected using the training dataset.
    The same feature list is then applied to validation and test data.
    """
    result = dataframe.copy()

    missing_indicator_features = sorted(
        set(indicator_features) - set(dataframe.columns)
    )

    if missing_indicator_features:
        raise KeyError(
            f"Missing indicator source columns: {missing_indicator_features}"
        )

    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()

    categorical_columns = dataframe.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    very_high_missing_columns = [
        column for column in very_high_missing_features if column in dataframe.columns
    ]

    result["missing_count_total"] = dataframe.isna().sum(axis=1).astype("int16")

    result["missing_count_numeric"] = (
        dataframe[numeric_columns].isna().sum(axis=1).astype("int16")
    )

    result["missing_count_categorical"] = (
        dataframe[categorical_columns].isna().sum(axis=1).astype("int16")
    )

    result["missing_count_very_high"] = (
        dataframe[very_high_missing_columns].isna().sum(axis=1).astype("int16")
        if very_high_missing_columns
        else pd.Series(
            0,
            index=dataframe.index,
            dtype="int16",
        )
    )

    for column in indicator_features:
        indicator_column = f"{column}_is_missing"

        if indicator_column in result.columns:
            raise ValueError(f"Generated column already exists: {indicator_column}")

        result[indicator_column] = dataframe[column].isna().astype("int8")

    return result


def add_selected_missing_indicators(
    dataframe: pd.DataFrame,
    *,
    indicator_features: Sequence[str],
) -> pd.DataFrame:
    """Add binary missing indicators for selected source features."""
    result = dataframe.copy()

    missing_source_columns = [
        column for column in indicator_features if column not in dataframe.columns
    ]

    if missing_source_columns:
        raise KeyError(f"Missing indicator source columns: {missing_source_columns}")

    for column in indicator_features:
        indicator_column = f"{column}_is_missing"

        if indicator_column in result.columns:
            raise ValueError(f"Generated column already exists: {indicator_column}")

        result[indicator_column] = dataframe[column].isna().astype("int8")

    return result


def save_dataset(
    dataset: PreparedDataset,
    processed_dir: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Save a prepared dataset version to disk."""
    dataset_dir = processed_dir / dataset.version

    if dataset_dir.exists() and any(dataset_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Dataset version already exists: {dataset_dir}. "
            "Use overwrite=True to replace it."
        )

    if len(dataset.X_train) != len(dataset.y_train):
        raise ValueError("X_train and y_train must contain the same number of rows.")

    if list(dataset.X_train.columns) != list(dataset.X_test.columns):
        raise ValueError(
            "X_train and X_test must contain the same columns in the same order."
        )

    dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset.X_train.to_parquet(
        dataset_dir / "X_train.parquet",
        index=False,
    )
    dataset.y_train.to_frame(name=dataset.y_train.name or "target").to_parquet(
        dataset_dir / "y_train.parquet",
        index=False,
    )
    dataset.X_test.to_parquet(
        dataset_dir / "X_test.parquet",
        index=False,
    )

    metadata_payload = {
        "version": dataset.version,
        "metadata": dataset.metadata,
    }

    with (dataset_dir / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata_payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return dataset_dir


def load_dataset(
    version: str,
    processed_dir: Path,
) -> PreparedDataset:
    """Load a prepared dataset version from disk."""
    dataset_dir = processed_dir / version

    required_files = [
        dataset_dir / "X_train.parquet",
        dataset_dir / "y_train.parquet",
        dataset_dir / "X_test.parquet",
        dataset_dir / "metadata.json",
    ]

    missing_files = [path.name for path in required_files if not path.exists()]

    if missing_files:
        raise FileNotFoundError(
            f"Dataset version '{version}' is incomplete. Missing files: {missing_files}"
        )

    X_train = pd.read_parquet(dataset_dir / "X_train.parquet")
    y_frame = pd.read_parquet(dataset_dir / "y_train.parquet")
    X_test = pd.read_parquet(dataset_dir / "X_test.parquet")

    if y_frame.shape[1] != 1:
        raise ValueError("The stored target file must contain exactly one column.")

    with (dataset_dir / "metadata.json").open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata_payload = json.load(file)

    stored_version = metadata_payload.get("version")

    if stored_version != version:
        raise ValueError(
            f"Requested version '{version}', but metadata contains '{stored_version}'."
        )

    if len(X_train) != len(y_frame):
        raise ValueError("Loaded X_train and y_train contain different row counts.")

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Loaded X_train and X_test contain different columns.")

    return PreparedDataset(
        version=stored_version,
        X_train=X_train,
        y_train=y_frame.iloc[:, 0],
        X_test=X_test,
        metadata=metadata_payload.get("metadata", {}),
    )
