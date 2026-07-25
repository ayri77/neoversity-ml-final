from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any
from pathlib import Path

import pandas as pd

from src.churn_ml.manual_lightgbm import (
    FeatureSchema,
    ordered_feature_schema_sha256,
)
from src.churn_ml.research_config import ResearchConfig
from src.churn_ml.run_artifacts import fingerprint_file


class ResearchDataError(ValueError):
    """Raised when train-only data violates the frozen evaluation plan."""


@dataclass(frozen=True)
class ResearchTrainingData:
    X: pd.DataFrame
    y: pd.Series
    metadata: dict[str, Any]
    feature_schema: FeatureSchema
    fingerprints: dict[str, Any]


def load_research_training_data(config: ResearchConfig) -> ResearchTrainingData:
    """Load and validate only training features, target, and dataset metadata."""
    dataset = config.plan_payload["dataset"]
    files = dataset["files"]
    dataset_dir = config.dataset_dir.resolve()
    paths: dict[str, Path] = {}
    for key, value in files.items():
        path = (dataset_dir / str(value["name"])).resolve()
        if path.parent != dataset_dir:
            raise ResearchDataError(
                f"Dataset file must resolve directly beneath dataset directory: {key}"
            )
        paths[key] = path
    file_fingerprints: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        actual = fingerprint_file(path)
        expected = str(files[name]["sha256"])
        if actual["sha256"] != expected:
            raise ResearchDataError(
                f"Dataset file fingerprint mismatch for {name}: "
                f"expected={expected}, actual={actual['sha256']}"
            )
        file_fingerprints[name] = actual

    X = pd.read_parquet(paths["train_features"])
    target_frame = pd.read_parquet(paths["target"])
    with paths["metadata"].open("r", encoding="utf-8") as file:
        metadata_payload = json.load(file)
    if target_frame.shape[1] != 1:
        raise ResearchDataError("The target Parquet file must contain one column.")
    y = target_frame.iloc[:, 0]

    _validate_training_identity(X, y, metadata_payload, config)
    X_model, schema = _prepare_candidate_features(X, config)
    fingerprints = _build_fingerprints(
        X,
        y,
        file_fingerprints,
        config,
    )
    return ResearchTrainingData(
        X=X_model,
        y=y,
        metadata=metadata_payload,
        feature_schema=schema,
        fingerprints=fingerprints,
    )


def _validate_training_identity(
    X: pd.DataFrame,
    y: pd.Series,
    metadata_payload: dict[str, Any],
    config: ResearchConfig,
) -> None:
    dataset = config.plan_payload["dataset"]
    target = dataset["target"]
    expected_rows = int(dataset["expected_rows"])
    if len(X) != expected_rows or len(y) != expected_rows:
        raise ResearchDataError(
            f"Expected {expected_rows} aligned rows, got X={len(X)}, y={len(y)}."
        )
    if not X.index.equals(y.index):
        raise ResearchDataError("Training feature and target indices differ.")
    expected_index = pd.RangeIndex(expected_rows)
    if not X.index.equals(expected_index):
        raise ResearchDataError(
            "Research row identity requires the stored order-based RangeIndex."
        )
    if X.shape[1] != int(dataset["expected_source_features"]):
        raise ResearchDataError("Source feature count differs from the plan.")
    source_hash = ordered_feature_schema_sha256(X.columns.tolist())
    if source_hash != dataset["ordered_source_schema_sha256"]:
        raise ResearchDataError("Ordered source feature schema differs from the plan.")
    dtype_hash = canonical_sha256(
        [{"name": name, "dtype": str(dtype)} for name, dtype in X.dtypes.items()]
    )
    if dtype_hash != dataset["ordered_dtype_schema_sha256"]:
        raise ResearchDataError("Ordered dtype schema differs from the plan.")

    if str(y.name) != target["name"] or str(y.dtype) != target["dtype"]:
        raise ResearchDataError("Target name or dtype differs from the plan.")
    if y.isna().any():
        raise ResearchDataError("Target contains missing values.")
    class_counts = y.value_counts().sort_index().to_dict()
    expected_counts = {
        int(target["negative_label"]): int(target["expected_negative_rows"]),
        int(target["positive_label"]): int(target["expected_positive_rows"]),
    }
    if class_counts != expected_counts:
        raise ResearchDataError(
            f"Target class counts differ: expected={expected_counts}, actual={class_counts}"
        )
    target_hash = canonical_sha256(
        {
            "name": str(y.name),
            "dtype": str(y.dtype),
            "values": y.tolist(),
        }
    )
    if target_hash != target["values_sha256"]:
        raise ResearchDataError("Target value fingerprint differs from the plan.")
    if metadata_payload.get("version") != dataset["version"]:
        raise ResearchDataError("Dataset metadata version differs from the plan.")


def _prepare_candidate_features(
    X: pd.DataFrame,
    config: ResearchConfig,
) -> tuple[pd.DataFrame, FeatureSchema]:
    features = config.candidate_contract["features"]
    drops = list(features["drop"])
    missing_drops = [name for name in drops if name not in X.columns]
    if missing_drops:
        raise ResearchDataError(f"Candidate drop features are missing: {missing_drops}")
    X_model = X.drop(columns=drops)
    categorical = X_model.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical = X_model.select_dtypes(include=["number", "bool"]).columns.tolist()
    unsupported = [
        name
        for name in X_model.columns
        if name not in categorical and name not in numerical
    ]
    if unsupported:
        raise ResearchDataError(f"Unsupported candidate feature dtypes: {unsupported}")
    if categorical != list(features["categorical"]):
        raise ResearchDataError("Candidate categorical feature order differs.")
    transformed = numerical + [f"{name}__te" for name in categorical]
    schema = FeatureSchema(
        source_feature_names=X.columns.tolist(),
        dropped_features=drops,
        model_feature_names=X_model.columns.tolist(),
        categorical_features=categorical,
        numerical_features=numerical,
        transformed_feature_names=transformed,
    )
    expected_hashes = {
        "source": features["expected_source_schema_sha256"],
        "model input": features["expected_model_input_schema_sha256"],
        "transformed": features["expected_transformed_schema_sha256"],
    }
    actual_hashes = {
        "source": schema.source_schema_sha256,
        "model input": schema.model_input_schema_sha256,
        "transformed": schema.transformed_schema_sha256,
    }
    mismatches = [
        f"{name}: expected={expected_hashes[name]}, actual={actual}"
        for name, actual in actual_hashes.items()
        if actual != expected_hashes[name]
    ]
    if mismatches:
        raise ResearchDataError(
            "Candidate ordered feature schema mismatch:\n- " + "\n- ".join(mismatches)
        )
    return X_model, schema


def _build_fingerprints(
    X_source: pd.DataFrame,
    y: pd.Series,
    files: dict[str, dict[str, Any]],
    config: ResearchConfig,
) -> dict[str, Any]:
    row_positions = list(range(len(X_source)))
    return {
        "dataset_version": config.plan_payload["dataset"]["version"],
        "files": files,
        "row_count": len(X_source),
        "row_position_identity": {
            "kind": "contiguous_zero_based",
            "sha256": canonical_sha256(row_positions),
        },
        "source_schema": {
            "ordered_names": X_source.columns.tolist(),
            "ordered_names_sha256": ordered_feature_schema_sha256(
                X_source.columns.tolist()
            ),
            "ordered_dtypes": [
                {"name": name, "dtype": str(dtype)}
                for name, dtype in X_source.dtypes.items()
            ],
            "ordered_dtypes_sha256": canonical_sha256(
                [
                    {"name": name, "dtype": str(dtype)}
                    for name, dtype in X_source.dtypes.items()
                ]
            ),
        },
        "target": {
            "name": str(y.name),
            "dtype": str(y.dtype),
            "negative_count": int((y == 0).sum()),
            "positive_count": int((y == 1).sum()),
            "values_sha256": canonical_sha256(
                {
                    "name": str(y.name),
                    "dtype": str(y.dtype),
                    "values": y.tolist(),
                }
            ),
        },
    }


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
