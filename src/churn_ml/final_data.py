from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.churn_ml.final_config import FinalConfig, contained_child
from src.churn_ml.manual_lightgbm import FeatureSchema, ordered_feature_schema_sha256
from src.churn_ml.research_protocol import canonical_sha256


class FinalDataError(ValueError):
    """Raised when final train, test, or sample inputs violate the contract."""


@dataclass(frozen=True)
class FinalData:
    X_train: pd.DataFrame
    y: pd.Series
    X_test: pd.DataFrame
    sample_submission: pd.DataFrame
    feature_schema: FeatureSchema
    fingerprints: dict[str, Any]
    submission_schema: dict[str, Any]


def load_final_data(config: FinalConfig) -> FinalData:
    inputs = config.payload["inputs"]
    dataset_dir = config.dataset_dir.resolve()
    paths: dict[str, Path] = {}
    for role in ("train_features", "target", "test_features", "metadata"):
        item = inputs[role]
        path = contained_child(dataset_dir, item["name"], f"inputs.{role}.name")
        if path.parent != dataset_dir:
            raise FinalDataError(
                f"{role} must be directly below the dataset directory."
            )
        paths[role] = path
    paths["sample_submission"] = config.sample_submission

    fingerprints: dict[str, Any] = {}
    for role, path in paths.items():
        expected = (
            inputs["sample_submission"]["sha256"]
            if role == "sample_submission"
            else inputs[role]["sha256"]
        )
        actual = _fingerprint(path, config.project_root)
        if actual["sha256"] != expected:
            raise FinalDataError(
                f"Input hash mismatch for {role}: "
                f"expected={expected}, actual={actual['sha256']}"
            )
        fingerprints[role] = actual

    X_source = pd.read_parquet(paths["train_features"])
    target_frame = pd.read_parquet(paths["target"])
    X_test_source = pd.read_parquet(paths["test_features"])
    sample = pd.read_csv(paths["sample_submission"])
    try:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise FinalDataError("Dataset metadata is not valid JSON.") from error

    if target_frame.shape[1] != 1:
        raise FinalDataError("Target file must contain exactly one column.")
    y = target_frame.iloc[:, 0]
    _validate_source_data(X_source, y, X_test_source, sample, metadata, config)
    X_train, X_test, schema = _prepare_candidate_features(
        X_source,
        X_test_source,
        config,
    )

    fingerprints["logical"] = {
        "dataset_version": inputs["dataset_version"],
        "train_rows": len(X_source),
        "test_rows": len(X_test_source),
        "train_row_positions_sha256": canonical_sha256(list(range(len(X_source)))),
        "test_row_positions_sha256": canonical_sha256(list(range(len(X_test_source)))),
        "source_schema_sha256": schema.source_schema_sha256,
        "source_dtype_schema_sha256": canonical_sha256(
            [
                {"name": name, "dtype": str(dtype)}
                for name, dtype in X_source.dtypes.items()
            ]
        ),
        "target_sha256": canonical_sha256(
            {
                "name": str(y.name),
                "dtype": str(y.dtype),
                "values": y.tolist(),
            }
        ),
        "sample_index_sha256": canonical_sha256(sample["index"].tolist()),
    }
    submission_schema = {
        "columns": sample.columns.tolist(),
        "target_column": "y",
        "row_count": len(sample),
        "pandas_index_kind": type(sample.index).__name__,
        "official_index_values_sha256": fingerprints["logical"]["sample_index_sha256"],
        "row_policy": "copy_sample_submission_and_replace_only_y",
    }
    return FinalData(
        X_train=X_train,
        y=y,
        X_test=X_test,
        sample_submission=sample,
        feature_schema=schema,
        fingerprints=fingerprints,
        submission_schema=submission_schema,
    )


def build_final_predictions(
    sample_submission: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(sample_submission):
        raise FinalDataError("Test probability shape differs from sample submission.")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise FinalDataError("Test probabilities must be finite and in [0, 1].")
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise FinalDataError("Operational threshold is invalid.")
    labels = (values >= threshold).astype(np.int8)
    if not np.isin(labels, [0, 1]).all():
        raise FinalDataError("Final predictions are not binary.")

    predictions = pd.DataFrame(
        {
            "row_position": np.arange(len(values), dtype=np.int64),
            "sample_index": sample_submission["index"].to_numpy(copy=True),
            "probability": values,
            "prediction": labels,
        }
    )
    submission = sample_submission.copy(deep=True)
    before = submission.drop(columns=["y"]).copy(deep=True)
    submission["y"] = labels
    if submission.columns.tolist() != ["index", "y"]:
        raise FinalDataError("Submission columns differ from ['index', 'y'].")
    if len(submission) != 2500:
        raise FinalDataError("Submission row count differs from 2500.")
    if not submission.index.equals(sample_submission.index):
        raise FinalDataError("Submission pandas index order changed.")
    if not submission.drop(columns=["y"]).equals(before):
        raise FinalDataError("A non-target submission column changed.")
    if submission["index"].duplicated().any():
        raise FinalDataError("Submission index values contain duplicates.")
    if not pd.api.types.is_integer_dtype(submission["y"]):
        raise FinalDataError("Submission labels must have integer dtype.")
    if set(submission["y"].unique().tolist()) - {0, 1}:
        raise FinalDataError("Submission labels must be binary integers.")
    return predictions, submission


def _validate_source_data(
    X_train: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    sample: pd.DataFrame,
    metadata: Any,
    config: FinalConfig,
) -> None:
    if len(X_train) != 10000 or len(y) != 10000 or len(X_test) != 2500:
        raise FinalDataError("Final train/test row counts differ from 10000/2500.")
    expected_train_index = pd.RangeIndex(10000)
    expected_test_index = pd.RangeIndex(2500)
    if (
        not X_train.index.equals(expected_train_index)
        or not y.index.equals(expected_train_index)
        or not X_test.index.equals(expected_test_index)
    ):
        raise FinalDataError("Train, target, and test require aligned RangeIndex.")
    if not X_train.index.is_unique or not X_test.index.is_unique:
        raise FinalDataError("Train/test row positions must be unique.")
    if X_train.columns.tolist() != X_test.columns.tolist():
        raise FinalDataError("Train/test ordered source columns differ.")
    train_dtypes = [str(dtype) for dtype in X_train.dtypes]
    test_dtypes = [str(dtype) for dtype in X_test.dtypes]
    if train_dtypes != test_dtypes:
        raise FinalDataError("Train/test ordered dtypes differ.")
    if X_train.shape[1] != 217:
        raise FinalDataError("Source feature count differs from 217.")
    research_plan = config.payload["promotion"]["research"]
    expected_source_hash = (
        "fa4cd8017a05520ca7f6c0fb81fb2f05b9cabe572e757eb43e424befebb9c62e"
    )
    if ordered_feature_schema_sha256(X_train.columns.tolist()) != expected_source_hash:
        raise FinalDataError(
            f"Frozen source feature schema differs for {research_plan['candidate_id']}."
        )
    if str(y.name) != "y" or str(y.dtype) != "int64":
        raise FinalDataError("Target name or dtype differs.")
    if y.isna().any() or set(y.unique().tolist()) != {0, 1}:
        raise FinalDataError("Target must contain finite binary labels.")
    if int((y == 0).sum()) != 8695 or int((y == 1).sum()) != 1305:
        raise FinalDataError("Target class counts differ.")
    if (
        not isinstance(metadata, dict)
        or metadata.get("version") != "v3_targeted_missingness"
    ):
        raise FinalDataError("Dataset metadata version differs.")
    expected_sample = config.payload["inputs"]["sample_submission"]
    if sample.columns.tolist() != expected_sample["columns"]:
        raise FinalDataError("Sample submission columns or order differ.")
    if len(sample) != expected_sample["expected_rows"]:
        raise FinalDataError("Sample submission row count differs.")
    if not sample.index.equals(expected_test_index):
        raise FinalDataError("Sample submission pandas row order differs.")
    if sample["index"].isna().any() or sample["index"].duplicated().any():
        raise FinalDataError("Sample submission index values are missing or duplicate.")


def _prepare_candidate_features(
    X_source: pd.DataFrame,
    X_test_source: pd.DataFrame,
    config: FinalConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, FeatureSchema]:
    drops = ["Var214", "Var220", "Var222", "Var218_is_missing"]
    missing = [name for name in drops if name not in X_source.columns]
    if missing:
        raise FinalDataError(f"Frozen dropped features are missing: {missing}")
    X_train = X_source.drop(columns=drops)
    X_test = X_test_source.drop(columns=drops)
    categorical = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    unsupported = [
        name
        for name in X_train.columns
        if name not in categorical and name not in numerical
    ]
    if unsupported:
        raise FinalDataError(f"Unsupported final feature dtypes: {unsupported}")
    transformed = numerical + [f"{name}__te" for name in categorical]
    schema = FeatureSchema(
        source_feature_names=X_source.columns.tolist(),
        dropped_features=drops,
        model_feature_names=X_train.columns.tolist(),
        categorical_features=categorical,
        numerical_features=numerical,
        transformed_feature_names=transformed,
    )
    expected = {
        "source": "fa4cd8017a05520ca7f6c0fb81fb2f05b9cabe572e757eb43e424befebb9c62e",
        "model": "e19efe336426c09071db296920353a0b967284a1288a1d9ab54aaa83ca179616",
        "transformed": "7c1fe2610edaa837baf4f041faf024361b896bb52e1fd81121c5ac6652180c41",
    }
    actual = {
        "source": schema.source_schema_sha256,
        "model": schema.model_input_schema_sha256,
        "transformed": schema.transformed_schema_sha256,
    }
    if actual != expected or len(X_train.columns) != 213 or len(categorical) != 31:
        raise FinalDataError(
            f"Final candidate feature contract differs: expected={expected}, actual={actual}"
        )
    if X_train.columns.tolist() != X_test.columns.tolist():
        raise FinalDataError("Final train/test model feature order differs.")
    return X_train, X_test, schema


def _fingerprint(path: Path, project_root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FinalDataError(f"Required input does not exist: {path}")
    raw = path.read_bytes()
    try:
        relative = path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise FinalDataError(f"Input is outside the repository: {path}") from error
    return {
        "path": relative,
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
