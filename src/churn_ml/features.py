from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Literal

import pandas as pd
import numpy as np

from src.churn_ml.dataset_registry.api import resolve_dataset_package
from src.churn_ml.dataset_registry.build_spec import (
    DatasetBuildSpec,
    native_build_spec,
    parse_build_spec,
)
from src.churn_ml.dataset_registry.errors import DatasetRegistryError
from src.churn_ml.dataset_registry.materialize import (
    dataset_manifest_sha256,
    materialize_dataset_manifest,
    refuse_registered_overwrite,
)
from src.churn_ml.dataset_registry.package import has_manifest, has_package_files
from src.churn_ml.dataset_registry.schema import DatasetManifest


TargetDependency = Literal[
    "none",
    "exploratory",
    "fold_local",
]


@dataclass(frozen=True)
class PreparedDataset:
    version: str
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    metadata: dict[str, Any]
    build_spec: DatasetBuildSpec | None = None
    dataset_manifest: DatasetManifest | None = None


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


def deduplicate_zero_mask_features(
    dataframe: pd.DataFrame,
    *,
    features: Sequence[str],
) -> tuple[list[str], list[list[str]]]:
    """
    Group features with identical zero and observation masks.

    The first feature in every group is retained as the
    representative. The operation does not use the target.
    """
    feature_list = list(features)

    if len(feature_list) != len(set(feature_list)):
        raise ValueError("features must not contain duplicate column names.")

    missing_features = [
        column for column in feature_list if column not in dataframe.columns
    ]

    if missing_features:
        raise KeyError(f"Missing zero-mask source columns: {missing_features}")

    non_numeric_features = [
        column
        for column in feature_list
        if not pd.api.types.is_numeric_dtype(dataframe[column])
    ]

    if non_numeric_features:
        raise TypeError(
            f"Zero-mask source columns must be numeric: {non_numeric_features}"
        )

    feature_frame = dataframe[feature_list]

    zero_masks = feature_frame.eq(0)
    observed_masks = feature_frame.notna()

    groups: list[list[str]] = []

    for column in feature_list:
        for group in groups:
            representative = group[0]

            same_zero_mask = zero_masks[column].equals(zero_masks[representative])

            same_observed_mask = observed_masks[column].equals(
                observed_masks[representative]
            )

            if same_zero_mask and same_observed_mask:
                group.append(column)
                break
        else:
            groups.append([column])

    representatives = [group[0] for group in groups]

    return representatives, groups


def select_supported_zero_features(
    dataframe: pd.DataFrame,
    *,
    candidate_features: Sequence[str] | None = None,
    min_present: int = 1_000,
    min_zero: int = 100,
    min_nonzero: int = 100,
) -> list[str]:
    """
    Select numerical features with sufficiently supported zero
    and non-zero values.

    Selection does not use the target. It must be performed on
    training data, and the resulting feature list must then be
    applied unchanged to validation and test data.
    """
    if min_present < 1:
        raise ValueError("min_present must be at least 1.")

    if min_zero < 1:
        raise ValueError("min_zero must be at least 1.")

    if min_nonzero < 1:
        raise ValueError("min_nonzero must be at least 1.")

    if candidate_features is None:
        features = dataframe.select_dtypes(include="number").columns.tolist()
    else:
        features = list(candidate_features)

    missing_features = [
        column for column in features if column not in dataframe.columns
    ]

    if missing_features:
        raise KeyError(f"Missing zero-feature candidate columns: {missing_features}")

    non_numeric_features = [
        column
        for column in features
        if not pd.api.types.is_numeric_dtype(dataframe[column])
    ]

    if non_numeric_features:
        raise TypeError(
            f"Zero-feature candidates must be numeric: {non_numeric_features}"
        )

    selected_features: list[str] = []

    for column in features:
        series = dataframe[column]

        present_mask = series.notna()
        zero_mask = series.eq(0)
        nonzero_mask = present_mask & ~zero_mask

        n_present = int(present_mask.sum())
        n_zero = int(zero_mask.sum())
        n_nonzero = int(nonzero_mask.sum())

        if n_present >= min_present and n_zero >= min_zero and n_nonzero >= min_nonzero:
            selected_features.append(column)

    return selected_features


def add_zero_value_summary_features(
    dataframe: pd.DataFrame,
    *,
    numeric_features: Sequence[str],
    supported_zero_features: Sequence[str],
) -> pd.DataFrame:
    """
    Add target-independent row-level zero-value summaries.

    The supported feature list must be selected from training data
    and then applied unchanged to validation and test data.
    """
    result = dataframe.copy()

    numeric_columns = list(numeric_features)
    supported_columns = list(supported_zero_features)

    if not numeric_columns:
        raise ValueError("numeric_features must contain at least one feature.")

    missing_numeric_columns = [
        column for column in numeric_columns if column not in dataframe.columns
    ]

    if missing_numeric_columns:
        raise KeyError(f"Missing numeric source columns: {missing_numeric_columns}")

    missing_supported_columns = [
        column for column in supported_columns if column not in dataframe.columns
    ]

    if missing_supported_columns:
        raise KeyError(
            f"Missing supported zero source columns: {missing_supported_columns}"
        )

    unsupported_columns = sorted(set(supported_columns) - set(numeric_columns))

    if unsupported_columns:
        raise ValueError(
            "supported_zero_features must be a subset of "
            f"numeric_features: {unsupported_columns}"
        )

    generated_columns = [
        "zero_count_numeric",
        "zero_rate_observed_numeric",
        "zero_count_supported_numeric",
        "zero_rate_observed_supported_numeric",
    ]

    existing_generated_columns = [
        column for column in generated_columns if column in result.columns
    ]

    if existing_generated_columns:
        raise ValueError(
            "Generated zero-summary columns already exist: "
            f"{existing_generated_columns}"
        )

    numeric_frame = dataframe[numeric_columns]

    zero_count_numeric = numeric_frame.eq(0).sum(axis=1).astype("int16")

    observed_count_numeric = numeric_frame.notna().sum(axis=1).astype("int16")

    result["zero_count_numeric"] = zero_count_numeric

    result["zero_rate_observed_numeric"] = (
        zero_count_numeric.div(
            observed_count_numeric.where(observed_count_numeric.ne(0))
        )
        .fillna(0.0)
        .astype("float32")
    )

    if supported_columns:
        supported_frame = dataframe[supported_columns]

        zero_count_supported = supported_frame.eq(0).sum(axis=1).astype("int16")

        observed_count_supported = supported_frame.notna().sum(axis=1).astype("int16")
    else:
        zero_count_supported = pd.Series(
            0,
            index=dataframe.index,
            dtype="int16",
        )

        observed_count_supported = pd.Series(
            0,
            index=dataframe.index,
            dtype="int16",
        )

    result["zero_count_supported_numeric"] = zero_count_supported

    result["zero_rate_observed_supported_numeric"] = (
        zero_count_supported.div(
            observed_count_supported.where(observed_count_supported.ne(0))
        )
        .fillna(0.0)
        .astype("float32")
    )

    return result


def _get_git_provenance(working_directory: Path) -> dict[str, Any]:
    """Record enough git state to distinguish a clean commit from a dirty tree."""

    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=working_directory,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        status = run("status", "--porcelain")
        return {
            "commit": run("rev-parse", "HEAD") or None,
            "dirty": bool(status),
            "status_porcelain": status.splitlines(),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {
            "commit": None,
            "dirty": None,
            "error": str(error),
        }


def _validate_dataset_structure(dataset: PreparedDataset) -> None:
    """Validate schema and row alignment before persistence."""
    if dataset.version.strip() == "":
        raise ValueError("dataset.version must be a non-empty string.")
    if len(dataset.X_train) != len(dataset.y_train):
        raise ValueError("X_train and y_train must contain the same number of rows.")
    if not dataset.X_train.index.equals(dataset.y_train.index):
        raise ValueError("X_train and y_train indices must be identical and ordered.")
    if not dataset.X_train.index.is_unique:
        raise ValueError("X_train index must be unique.")
    if not dataset.y_train.index.is_unique:
        raise ValueError("y_train index must be unique.")
    if not dataset.X_test.index.is_unique:
        raise ValueError("X_test index must be unique.")
    if list(dataset.X_train.columns) != list(dataset.X_test.columns):
        raise ValueError(
            "X_train and X_test must contain the same columns in the same order."
        )
    train_dtypes = dataset.X_train.dtypes.astype(str).tolist()
    test_dtypes = dataset.X_test.dtypes.astype(str).tolist()
    if train_dtypes != test_dtypes:
        raise ValueError(
            "X_train and X_test must contain identical dtypes in the same order."
        )


def _resolve_build_spec(dataset: PreparedDataset) -> DatasetBuildSpec:
    if dataset.build_spec is None:
        raise ValueError(
            "PreparedDataset.build_spec is required to save a registered package."
        )
    build_spec = parse_build_spec(dataset.build_spec)
    if build_spec.dataset_id != dataset.version:
        raise ValueError("build_spec.dataset_id must match dataset.version.")
    if build_spec.target_dependency == "fold_local":
        raise DatasetRegistryError(
            "target_dependency='fold_local' cannot be materialized as a prepared "
            "Parquet package; fold-local transforms belong inside a CV pipeline."
        )
    return build_spec


def save_dataset(
    dataset: PreparedDataset,
    processed_dir: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Validate and save a Registry-registered prepared dataset.

    Writes the four package artifacts and canonical dataset_manifest.json.
    Does not write or trust generator-owned manifest.json.
    """
    _validate_dataset_structure(dataset)
    build_spec = _resolve_build_spec(dataset)
    processed_dir = processed_dir.resolve()
    dataset_dir = processed_dir / dataset.version
    refuse_registered_overwrite(dataset_dir)

    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        if has_package_files(dataset_dir) and not has_manifest(dataset_dir):
            if not overwrite:
                raise FileExistsError(
                    f"Unregistered incomplete package already exists: {dataset_dir}. "
                    "Pass overwrite=True only for clearly unregistered development "
                    "directories."
                )
            shutil.rmtree(dataset_dir)
        elif not overwrite:
            raise FileExistsError(
                f"Dataset version already exists: {dataset_dir}. "
                "Registered packages are immutable; use a new dataset_id. "
                "overwrite=True is allowed only for unregistered/incomplete directories."
            )
        else:
            # Non-empty directory without a complete unregistered package.
            if has_manifest(dataset_dir):
                refuse_registered_overwrite(dataset_dir)
            shutil.rmtree(dataset_dir)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    try:
        train_path = dataset_dir / "X_train.parquet"
        target_path = dataset_dir / "y_train.parquet"
        test_path = dataset_dir / "X_test.parquet"
        metadata_path = dataset_dir / "metadata.json"

        target_name = dataset.y_train.name or "target"
        target_frame = dataset.y_train.to_frame(name=target_name)
        dataset.X_train.to_parquet(train_path, index=False)
        target_frame.to_parquet(target_path, index=False)
        dataset.X_test.to_parquet(test_path, index=False)

        project_root = processed_dir.parent.parent
        metadata_payload = {
            "version": dataset.version,
            "metadata": {
                **dataset.metadata,
                "created_from_git": _get_git_provenance(project_root),
            },
        }
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata_payload, file, indent=2, ensure_ascii=False)

        materialize_dataset_manifest(
            dataset_dir,
            root=processed_dir,
            build_spec=build_spec,
        )
    except BaseException:
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir, ignore_errors=True)
        raise

    return dataset_dir


def load_dataset(
    version: str,
    processed_dir: Path,
) -> PreparedDataset:
    """
    Load a prepared dataset version.

    When dataset_manifest.json exists, load through the strict Registry API.
    Legacy four-file directories without a Registry manifest remain explicitly
    unregistered compatibility loads and must not be used by future experiment
    integration.
    """
    processed_dir = processed_dir.resolve()
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

    if (dataset_dir / "dataset_manifest.json").is_file():
        package = resolve_dataset_package(processed_dir, version)
        nested_metadata = package.artifacts.metadata.get("metadata", {})
        if not isinstance(nested_metadata, dict):
            nested_metadata = {}
        return PreparedDataset(
            version=package.manifest.dataset_id,
            X_train=package.artifacts.X_train,
            y_train=package.artifacts.y_train,
            X_test=package.artifacts.X_test,
            metadata=nested_metadata,
            build_spec=None,
            dataset_manifest=package.manifest,
        )

    # Explicit legacy compatibility path (unregistered).
    if (dataset_dir / "manifest.json").is_file():
        raise DatasetRegistryError(
            "Generator-owned manifest.json is not the Registry contract. "
            "Re-save through save_dataset() to create dataset_manifest.json, "
            "or load a four-file legacy package that has no manifest.json."
        )

    X_train = pd.read_parquet(dataset_dir / "X_train.parquet")
    y_frame = pd.read_parquet(dataset_dir / "y_train.parquet")
    X_test = pd.read_parquet(dataset_dir / "X_test.parquet")
    if y_frame.shape[1] != 1:
        raise ValueError("The stored target file must contain exactly one column.")
    with (dataset_dir / "metadata.json").open("r", encoding="utf-8") as file:
        metadata_payload = json.load(file)
    stored_version = metadata_payload.get("version")
    if stored_version != version:
        raise ValueError(
            f"Requested version '{version}', but metadata contains '{stored_version}'."
        )
    dataset = PreparedDataset(
        version=stored_version,
        X_train=X_train,
        y_train=y_frame.iloc[:, 0],
        X_test=X_test,
        metadata=metadata_payload.get("metadata", {}),
        build_spec=None,
        dataset_manifest=None,
    )
    _validate_dataset_structure(dataset)
    return dataset


def deduplicate_missingness_mask_features(
    dataframe: pd.DataFrame,
    *,
    features: Sequence[str],
) -> tuple[list[str], list[list[str]]]:
    """
    Group features with identical missingness masks.

    The first feature in each group is retained as the
    representative. The operation does not use the target.
    """
    feature_list = list(features)

    if not feature_list:
        raise ValueError("features must contain at least one column.")

    if len(feature_list) != len(set(feature_list)):
        raise ValueError("features must not contain duplicate column names.")

    missing_features = [
        column for column in feature_list if column not in dataframe.columns
    ]

    if missing_features:
        raise KeyError(f"Missing missingness-mask source columns: {missing_features}")

    missing_masks = dataframe[feature_list].isna()

    groups: list[list[str]] = []

    for column in feature_list:
        for group in groups:
            representative = group[0]

            if missing_masks[column].equals(missing_masks[representative]):
                group.append(column)
                break
        else:
            groups.append([column])

    representatives = [group[0] for group in groups]

    return representatives, groups


def add_missingness_pattern_feature(
    dataframe: pd.DataFrame,
    *,
    source_features: Sequence[str],
    feature_name: str = "missingness_pattern_id",
) -> pd.DataFrame:
    """
    Add a deterministic categorical identifier for the exact
    row-level missingness pattern.

    The pattern is calculated from a fixed ordered source-feature
    list and does not use the target.
    """
    feature_list = list(source_features)

    if not feature_list:
        raise ValueError("source_features must contain at least one column.")

    if len(feature_list) != len(set(feature_list)):
        raise ValueError("source_features must not contain duplicate columns.")

    missing_features = [
        column for column in feature_list if column not in dataframe.columns
    ]

    if missing_features:
        raise KeyError(f"Missing pattern source columns: {missing_features}")

    if feature_name in dataframe.columns:
        raise ValueError(f"Feature already exists: {feature_name}")

    missing_matrix = dataframe[feature_list].isna().to_numpy(dtype="uint8")

    packed_patterns = np.packbits(
        missing_matrix,
        axis=1,
        bitorder="little",
    )

    pattern_values = [packed_row.tobytes().hex() for packed_row in packed_patterns]

    result = dataframe.copy()

    result[feature_name] = pd.Series(
        pattern_values,
        index=result.index,
        dtype="string",
    )

    return result


def add_selected_zero_indicators(
    dataframe: pd.DataFrame,
    *,
    indicator_features: Sequence[str],
) -> pd.DataFrame:
    """
    Add binary zero-value indicators for selected numeric features.

    Missing values are treated as not zero. The feature list must
    be selected using training data and then applied unchanged to
    validation and test data.
    """
    feature_list = list(indicator_features)

    if not feature_list:
        raise ValueError("indicator_features must contain at least one column.")

    if len(feature_list) != len(set(feature_list)):
        raise ValueError("indicator_features must not contain duplicate columns.")

    missing_source_columns = [
        column for column in feature_list if column not in dataframe.columns
    ]

    if missing_source_columns:
        raise KeyError(
            f"Missing zero-indicator source columns: {missing_source_columns}"
        )

    non_numeric_features = [
        column
        for column in feature_list
        if not pd.api.types.is_numeric_dtype(dataframe[column])
    ]

    if non_numeric_features:
        raise TypeError(
            f"Zero-indicator source columns must be numeric: {non_numeric_features}"
        )

    result = dataframe.copy()

    for column in feature_list:
        indicator_column = f"{column}_is_zero"

        if indicator_column in result.columns:
            raise ValueError(f"Generated column already exists: {indicator_column}")

        result[indicator_column] = dataframe[column].eq(0).astype("int8")

    return result


__all__ = [
    "DatasetBuildSpec",
    "PreparedDataset",
    "TargetDependency",
    "add_missingness_indicator_features",
    "add_missingness_pattern_feature",
    "add_missingness_summary_features",
    "add_selected_missing_indicators",
    "add_selected_zero_indicators",
    "add_zero_value_summary_features",
    "dataset_manifest_sha256",
    "deduplicate_missingness_mask_features",
    "deduplicate_zero_mask_features",
    "load_dataset",
    "native_build_spec",
    "save_dataset",
    "select_supported_zero_features",
]
