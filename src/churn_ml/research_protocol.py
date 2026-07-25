from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


class ResearchProtocolError(ValueError):
    """Raised when split, probability, threshold, or aggregation data is invalid."""


@dataclass(frozen=True)
class EvaluationAssignments:
    outer: pd.DataFrame
    threshold_selection: pd.DataFrame


@dataclass(frozen=True)
class ThresholdSelectionResult:
    threshold: float
    balanced_accuracy: float
    degenerate: bool
    status: str
    scores: pd.DataFrame


def build_evaluation_assignments(
    y: pd.Series,
    plan: dict[str, Any],
) -> EvaluationAssignments:
    """Build deterministic outer and threshold-selection assignments."""
    outer_config = plan["outer_evaluation"]
    threshold_config = plan["threshold_selection"]
    row_positions = np.arange(len(y), dtype=np.int64)
    outer_records: list[pd.DataFrame] = []
    threshold_records: list[pd.DataFrame] = []
    for repeat, repeat_seed in enumerate(
        outer_config["repeat_seeds"],
        start=1,
    ):
        splitter = StratifiedKFold(
            n_splits=int(outer_config["n_splits"]),
            shuffle=bool(outer_config["shuffle"]),
            random_state=int(repeat_seed),
        )
        for outer_fold, (outer_train_local, outer_validation_local) in enumerate(
            splitter.split(row_positions, y.to_numpy()),
            start=1,
        ):
            outer_validation = row_positions[outer_validation_local]
            outer_train = row_positions[outer_train_local]
            outer_records.append(
                pd.DataFrame(
                    {
                        "repeat": repeat,
                        "repeat_seed": int(repeat_seed),
                        "outer_fold": outer_fold,
                        "row_position": outer_validation,
                    }
                )
            )
            threshold_splitter = StratifiedKFold(
                n_splits=int(threshold_config["n_splits"]),
                shuffle=bool(threshold_config["shuffle"]),
                random_state=int(threshold_config["random_state"]),
            )
            outer_train_targets = y.iloc[outer_train].to_numpy()
            for threshold_fold, (_, threshold_validation_local) in enumerate(
                threshold_splitter.split(outer_train, outer_train_targets),
                start=1,
            ):
                threshold_records.append(
                    pd.DataFrame(
                        {
                            "repeat": repeat,
                            "repeat_seed": int(repeat_seed),
                            "outer_fold": outer_fold,
                            "threshold_selection_fold": threshold_fold,
                            "row_position": outer_train[threshold_validation_local],
                        }
                    )
                )
    assignments = EvaluationAssignments(
        outer=pd.concat(outer_records, ignore_index=True).astype("int64"),
        threshold_selection=pd.concat(
            threshold_records,
            ignore_index=True,
        ).astype("int64"),
    )
    validate_evaluation_assignments(assignments, len(y), plan)
    return assignments


def validate_evaluation_assignments(
    assignments: EvaluationAssignments,
    row_count: int,
    plan: dict[str, Any],
) -> None:
    outer = assignments.outer
    threshold = assignments.threshold_selection
    repeats = len(plan["outer_evaluation"]["repeat_seeds"])
    outer_folds = int(plan["outer_evaluation"]["n_splits"])
    threshold_folds = int(plan["threshold_selection"]["n_splits"])
    expected_rows = set(range(row_count))

    for repeat in range(1, repeats + 1):
        repeat_outer = outer.loc[outer["repeat"] == repeat]
        if len(repeat_outer) != row_count:
            raise ResearchProtocolError("Outer validation coverage is incomplete.")
        if set(repeat_outer["row_position"]) != expected_rows:
            raise ResearchProtocolError("Outer validation row identity is incomplete.")
        if repeat_outer["row_position"].duplicated().any():
            raise ResearchProtocolError(
                "A row appears more than once in one outer repeat."
            )
        if set(repeat_outer["outer_fold"]) != set(range(1, outer_folds + 1)):
            raise ResearchProtocolError("Outer fold identifiers are incomplete.")

    for repeat in range(1, repeats + 1):
        for outer_fold in range(1, outer_folds + 1):
            validation_rows = set(
                outer.loc[
                    (outer["repeat"] == repeat) & (outer["outer_fold"] == outer_fold),
                    "row_position",
                ]
            )
            outer_train_rows = expected_rows - validation_rows
            inner = threshold.loc[
                (threshold["repeat"] == repeat)
                & (threshold["outer_fold"] == outer_fold)
            ]
            if set(inner["row_position"]) != outer_train_rows:
                raise ResearchProtocolError(
                    "Threshold-selection rows do not equal outer-training rows."
                )
            if inner["row_position"].duplicated().any():
                raise ResearchProtocolError(
                    "Threshold-selection OOF coverage contains duplicates."
                )
            if validation_rows & set(inner["row_position"]):
                raise ResearchProtocolError(
                    "Outer-validation rows entered threshold selection."
                )
            if set(inner["threshold_selection_fold"]) != set(
                range(1, threshold_folds + 1)
            ):
                raise ResearchProtocolError(
                    "Threshold-selection fold identifiers are incomplete."
                )


def select_balanced_accuracy_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    policy: dict[str, Any],
) -> ThresholdSelectionResult:
    """Select the approved deterministic Balanced Accuracy threshold."""
    targets = np.asarray(y_true)
    probabilities_array = np.asarray(probabilities, dtype=float)
    _validate_threshold_inputs(targets, probabilities_array)
    thresholds = build_threshold_grid(policy)
    if np.unique(probabilities_array).size == 1:
        fallback = float(policy["constant_probability_fallback"])
        predictions = (probabilities_array >= fallback).astype("int8")
        score = float(balanced_accuracy_score(targets, predictions))
        scores = _threshold_scores(targets, probabilities_array, thresholds)
        return ThresholdSelectionResult(
            threshold=fallback,
            balanced_accuracy=score,
            degenerate=True,
            status="degenerate_constant_probabilities",
            scores=scores,
        )

    scores = _threshold_scores(targets, probabilities_array, thresholds)
    maximum = float(scores["balanced_accuracy"].max())
    tolerance = float(policy["maximizer_absolute_tolerance"])
    maximizers = scores.loc[
        (maximum - scores["balanced_accuracy"]) <= tolerance,
        "threshold",
    ].to_numpy()
    selected_position = (len(maximizers) - 1) // 2
    selected = float(maximizers[selected_position])
    return ThresholdSelectionResult(
        threshold=selected,
        balanced_accuracy=maximum,
        degenerate=False,
        status="selected",
        scores=scores,
    )


def build_threshold_grid(policy: dict[str, Any]) -> np.ndarray:
    scale = 1000
    minimum = float(policy["minimum"])
    maximum = float(policy["maximum"])
    step = float(policy["step"])
    scaled_values = [minimum * scale, maximum * scale, step * scale]
    rounded = [round(value) for value in scaled_values]
    if any(
        abs(value - integer) > 1e-9 for value, integer in zip(scaled_values, rounded)
    ):
        raise ResearchProtocolError(
            "Threshold bounds and step must align to integer thousandths."
        )
    minimum_tick, maximum_tick, step_tick = rounded
    if step_tick <= 0 or (maximum_tick - minimum_tick) % step_tick:
        raise ResearchProtocolError("Threshold grid is not exactly inclusive.")
    return (
        np.arange(
            minimum_tick,
            maximum_tick + step_tick,
            step_tick,
            dtype=np.int64,
        ).astype(float)
        / scale
    )


def calculate_outer_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    *,
    selected_threshold: float | None,
) -> dict[str, float | int | None]:
    targets = np.asarray(y_true)
    probabilities_array = np.asarray(probabilities, dtype=float)
    predictions_array = np.asarray(predictions)
    _validate_threshold_inputs(targets, probabilities_array)
    if predictions_array.shape != targets.shape:
        raise ResearchProtocolError("Predictions and targets must have equal shape.")
    if not set(np.unique(predictions_array)).issubset({0, 1}):
        raise ResearchProtocolError("Predictions must contain only labels 0 and 1.")
    tn, fp, fn, tp = confusion_matrix(
        targets,
        predictions_array,
        labels=[0, 1],
    ).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive_rate": float(predictions_array.mean()),
        "roc_auc": float(roc_auc_score(targets, probabilities_array)),
        "average_precision": float(
            average_precision_score(targets, probabilities_array)
        ),
        "brier_score": float(brier_score_loss(targets, probabilities_array)),
        "selected_threshold": selected_threshold,
    }


def build_repeat_metrics(outer_predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for repeat, frame in outer_predictions.groupby("repeat", sort=True):
        metrics = calculate_outer_metrics(
            frame["target"],
            frame["probability"].to_numpy(),
            frame["prediction"].to_numpy(),
            selected_threshold=None,
        )
        metrics.pop("selected_threshold")
        records.append(
            {
                "repeat": int(repeat),
                "validation_rows": len(frame),
                **metrics,
            }
        )
    return pd.DataFrame(records)


def aggregate_repeat_metrics(
    repeat_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> dict[str, Any]:
    metric_names = [
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "tn",
        "fp",
        "fn",
        "tp",
        "predicted_positive_rate",
        "roc_auc",
        "average_precision",
        "brier_score",
    ]
    aggregates: dict[str, Any] = {}
    for name in metric_names:
        values = repeat_metrics[name].to_numpy(dtype=float)
        aggregates[name] = {
            "mean": float(values.mean()),
            "sample_standard_deviation": (
                float(values.std(ddof=1)) if len(values) > 1 else None
            ),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return {
        "primary_metric": "balanced_accuracy",
        "aggregation_unit": "pooled_repeat_predictions",
        "repeat_count": len(repeat_metrics),
        "metrics": aggregates,
        "fold_descriptive_heterogeneity": {
            name: {
                "standard_deviation_sample": float(
                    fold_metrics[name].to_numpy(dtype=float).std(ddof=1)
                )
            }
            for name in (
                "balanced_accuracy",
                "sensitivity",
                "specificity",
                "roc_auc",
                "average_precision",
                "brier_score",
            )
        },
        "confidence_interval": None,
        "confidence_interval_reason": (
            "Outer folds and repeats overlap and are not independent observations."
        ),
    }


def summarize_thresholds(selected_thresholds: pd.DataFrame) -> dict[str, Any]:
    values = selected_thresholds["selected_threshold"].to_numpy(dtype=float)
    quartiles = np.quantile(values, [0.25, 0.5, 0.75])
    per_repeat: list[dict[str, Any]] = []
    for repeat, frame in selected_thresholds.groupby("repeat", sort=True):
        repeat_values = frame["selected_threshold"].to_numpy(dtype=float)
        per_repeat.append(
            {
                "repeat": int(repeat),
                "values": repeat_values.tolist(),
                "minimum": float(repeat_values.min()),
                "median": float(np.median(repeat_values)),
                "maximum": float(repeat_values.max()),
            }
        )
    return {
        "count": len(values),
        "minimum": float(values.min()),
        "first_quartile": float(quartiles[0]),
        "median": float(quartiles[1]),
        "third_quartile": float(quartiles[2]),
        "interquartile_range": float(quartiles[2] - quartiles[0]),
        "maximum": float(values.max()),
        "per_repeat": per_repeat,
    }


def build_evaluation_plan_identity(
    plan: dict[str, Any],
    dataset_fingerprints: dict[str, Any],
    assignments: EvaluationAssignments,
) -> tuple[dict[str, Any], str]:
    dataset_identity = deepcopy(dataset_fingerprints)
    dataset_identity = {
        name: dataset_identity[name]
        for name in (
            "dataset_version",
            "files",
            "row_count",
            "row_position_identity",
            "source_schema",
            "target",
        )
    }

    for file_fingerprint in dataset_identity.get("files", {}).values():
        if isinstance(file_fingerprint, dict):
            file_fingerprint.pop("path", None)
    canonical = {
        "schema_version": 1,
        "plan_id": plan["plan"]["id"],
        "dataset": dataset_identity,
        "outer_evaluation": plan["outer_evaluation"],
        "threshold_selection": plan["threshold_selection"],
        "threshold_policy": plan["threshold_policy"],
        "metrics": plan["metrics"],
        "aggregation": plan["aggregation"],
        "assignment_fingerprints": {
            "outer": dataframe_integer_sha256(assignments.outer),
            "threshold_selection": dataframe_integer_sha256(
                assignments.threshold_selection
            ),
        },
    }
    return canonical, canonical_sha256(canonical)


def build_candidate_contract_identity(
    candidate_contract: dict[str, Any],
    *,
    feature_schema: dict[str, Any],
    source_provenance: dict[str, Any],
    runtime_dependencies: dict[str, str],
) -> tuple[dict[str, Any], str]:
    canonical = {
        "schema_version": 2,
        "candidate_id": candidate_contract["candidate_id"],
        "implementation": candidate_contract["implementation"],
        "features": candidate_contract["features"],
        "resolved_feature_schema": feature_schema,
        "target_encoder": candidate_contract["target_encoder"],
        "lightgbm": candidate_contract["lightgbm"],
        "runtime_dependencies": dict(sorted(runtime_dependencies.items())),
        "source_provenance": source_provenance,
    }
    return canonical, canonical_sha256(canonical)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dataframe_integer_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            list(frame.columns),
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for name in frame.columns:
        values = frame[name].to_numpy(dtype="<i8", copy=True)
        digest.update(name.encode("utf-8"))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _validate_threshold_inputs(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    if targets.ndim != 1 or probabilities.ndim != 1:
        raise ResearchProtocolError(
            "Targets and probabilities must be one-dimensional."
        )
    if len(targets) != len(probabilities) or not len(targets):
        raise ResearchProtocolError(
            "Targets and probabilities must have equal non-zero length."
        )
    if pd.isna(targets).any():
        raise ResearchProtocolError("Targets contain missing values.")
    if set(np.unique(targets)) != {0, 1}:
        raise ResearchProtocolError(
            "Threshold-selection targets must contain both classes 0 and 1."
        )
    if not np.isfinite(probabilities).all():
        raise ResearchProtocolError("Probabilities contain non-finite values.")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ResearchProtocolError("Probabilities must lie within [0, 1].")


def _threshold_scores(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    scores = [
        balanced_accuracy_score(
            targets,
            (probabilities >= threshold).astype("int8"),
        )
        for threshold in thresholds
    ]
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "balanced_accuracy": np.asarray(scores, dtype=float),
        }
    )
