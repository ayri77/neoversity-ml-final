from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.churn_ml.experiment_v2 import PipelineOutput, get_feature_pipeline
from src.churn_ml.experiment_v2_schema import ordered_feature_schema_sha256
from src.churn_ml.research_data import canonical_sha256
from src.churn_ml.research_v2_config import ResearchV2Config
from src.churn_ml.run_artifacts import fingerprint_file


class ResearchV2DataError(ValueError):
    """Raised when train-only v2 data identity or schema validation fails."""


@dataclass(frozen=True)
class ResearchV2TrainingData:
    X: pd.DataFrame
    y: pd.Series
    pipeline_output: PipelineOutput
    metadata: dict[str, Any]
    fingerprints: dict[str, Any]


def load_research_v2_training_data(
    config: ResearchV2Config,
) -> ResearchV2TrainingData:
    dataset = config.plan_payload["dataset"]
    project_root = config.project_root.resolve()
    processed_root = config.processed_data_root.resolve()
    dataset_dir = config.dataset_dir.resolve()
    if not _is_contained(processed_root, project_root):
        raise ResearchV2DataError(
            "Resolved processed-data root escapes the project root."
        )
    if not _is_contained(dataset_dir, processed_root) or not _is_contained(
        dataset_dir, project_root
    ):
        raise ResearchV2DataError(
            "Resolved dataset version directory escapes its permitted roots."
        )
    paths: dict[str, Path] = {}
    fingerprints: dict[str, dict[str, Any]] = {}
    for key, item in dataset["files"].items():
        path = (dataset_dir / str(item["name"])).resolve()
        if (
            path.parent != dataset_dir
            or not _is_contained(path, processed_root)
            or not _is_contained(path, project_root)
        ):
            raise ResearchV2DataError("Dataset file escapes its version directory.")
        actual = fingerprint_file(path)
        if actual["sha256"] != item["sha256"]:
            raise ResearchV2DataError(f"Dataset fingerprint mismatch for {key}.")
        paths[key] = path
        fingerprints[key] = actual
    X_source = pd.read_parquet(paths["train_features"])
    target_frame = pd.read_parquet(paths["target"])
    with paths["metadata"].open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if target_frame.shape[1] != 1:
        raise ResearchV2DataError("Target file must contain exactly one column.")
    y = target_frame.iloc[:, 0]
    _validate_source(X_source, y, metadata, config)
    pipeline = get_feature_pipeline(config.pipeline_id)
    output = pipeline.transform(X_source, config.pipeline_contract)
    row_positions = list(range(len(X_source)))
    identity = {
        "dataset_version": config.dataset_version,
        "files": fingerprints,
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
    return ResearchV2TrainingData(
        X=output.features,
        y=y,
        pipeline_output=output,
        metadata=metadata,
        fingerprints=identity,
    )


def _validate_source(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: dict[str, Any],
    config: ResearchV2Config,
) -> None:
    dataset = config.plan_payload["dataset"]
    target = dataset["target"]
    rows = int(dataset["expected_rows"])
    if len(X) != rows or len(y) != rows or not X.index.equals(y.index):
        raise ResearchV2DataError("Training rows are not aligned with the plan.")
    if not X.index.equals(pd.RangeIndex(rows)):
        raise ResearchV2DataError("Row identity must be a zero-based RangeIndex.")
    if X.shape[1] != int(dataset["expected_source_features"]):
        raise ResearchV2DataError("Source feature count differs from the plan.")
    if (
        ordered_feature_schema_sha256(X.columns.tolist())
        != dataset["ordered_source_schema_sha256"]
    ):
        raise ResearchV2DataError("Ordered source schema differs from the plan.")
    dtype_hash = canonical_sha256(
        [{"name": name, "dtype": str(dtype)} for name, dtype in X.dtypes.items()]
    )
    if dtype_hash != dataset["ordered_dtype_schema_sha256"]:
        raise ResearchV2DataError("Ordered dtype schema differs from the plan.")
    if str(y.name) != target["name"] or str(y.dtype) != target["dtype"]:
        raise ResearchV2DataError("Target name or dtype differs from the plan.")
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ResearchV2DataError(
            "Target must contain only non-missing labels 0 and 1."
        )
    counts = y.value_counts().sort_index().to_dict()
    expected_counts = {
        int(target["negative_label"]): int(target["expected_negative_rows"]),
        int(target["positive_label"]): int(target["expected_positive_rows"]),
    }
    if counts != expected_counts:
        raise ResearchV2DataError("Target class counts differ from the plan.")
    target_hash = canonical_sha256(
        {"name": str(y.name), "dtype": str(y.dtype), "values": y.tolist()}
    )
    if target_hash != target["values_sha256"]:
        raise ResearchV2DataError("Target fingerprint differs from the plan.")
    if metadata.get("version") != config.dataset_version:
        raise ResearchV2DataError("Dataset metadata version differs from the plan.")


def _is_contained(path: Path, root: Path) -> bool:
    return path == root or root in path.parents
