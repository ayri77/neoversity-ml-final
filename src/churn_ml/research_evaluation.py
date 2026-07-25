from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.churn_ml.research_manual_lightgbm import (
    FitAuditCallback,
    fit_predict_manual_candidate,
)
from src.churn_ml.research_protocol import (
    EvaluationAssignments,
    ThresholdSelectionResult,
    aggregate_repeat_metrics,
    build_repeat_metrics,
    calculate_outer_metrics,
    select_balanced_accuracy_threshold,
    summarize_thresholds,
)


FitPredictFunction = Callable[..., np.ndarray]
ThresholdSelector = Callable[
    [pd.Series | np.ndarray, np.ndarray, dict[str, Any]],
    ThresholdSelectionResult,
]
ThresholdAuditCallback = Callable[
    [int, int, np.ndarray, np.ndarray],
    None,
]


@dataclass(frozen=True)
class CompletedOuterFold:
    repeat: int
    outer_fold: int
    threshold_selection_oof: pd.DataFrame
    outer_validation: pd.DataFrame
    selected_threshold: dict[str, Any]
    threshold_curve: pd.DataFrame
    fold_metrics: dict[str, Any]


@dataclass(frozen=True)
class ResearchEvaluationResult:
    threshold_selection_oof: pd.DataFrame
    outer_validation: pd.DataFrame
    selected_thresholds: pd.DataFrame
    threshold_curves: pd.DataFrame
    outer_fold_metrics: pd.DataFrame
    repeat_metrics: pd.DataFrame
    aggregate_metrics: dict[str, Any]
    threshold_summary: dict[str, Any]
    duration_seconds: float


def run_research_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    assignments: EvaluationAssignments,
    plan: dict[str, Any],
    candidate_contract: dict[str, Any],
    *,
    fit_predict: FitPredictFunction = fit_predict_manual_candidate,
    threshold_selector: ThresholdSelector = select_balanced_accuracy_threshold,
    fit_audit_callback: FitAuditCallback | None = None,
    threshold_audit_callback: ThresholdAuditCallback | None = None,
    on_outer_fold_complete: Callable[[CompletedOuterFold], None] | None = None,
) -> ResearchEvaluationResult:
    """Run repeated outer evaluation with nested threshold selection."""
    all_positions = np.arange(len(X), dtype=np.int64)
    outer_frames: list[pd.DataFrame] = []
    threshold_frames: list[pd.DataFrame] = []
    selected_records: list[dict[str, Any]] = []
    curve_frames: list[pd.DataFrame] = []
    metric_records: list[dict[str, Any]] = []
    started = perf_counter()

    repeat_count = len(plan["outer_evaluation"]["repeat_seeds"])
    outer_fold_count = int(plan["outer_evaluation"]["n_splits"])
    threshold_fold_count = int(plan["threshold_selection"]["n_splits"])
    for repeat in range(1, repeat_count + 1):
        repeat_seed = int(plan["outer_evaluation"]["repeat_seeds"][repeat - 1])
        for outer_fold in range(1, outer_fold_count + 1):
            fold_started = perf_counter()
            outer_validation_positions = assignments.outer.loc[
                (assignments.outer["repeat"] == repeat)
                & (assignments.outer["outer_fold"] == outer_fold),
                "row_position",
            ].to_numpy(dtype=np.int64)
            outer_training_positions = np.setdiff1d(
                all_positions,
                outer_validation_positions,
                assume_unique=True,
            )
            if threshold_audit_callback is not None:
                threshold_audit_callback(
                    repeat,
                    outer_fold,
                    outer_training_positions.copy(),
                    outer_validation_positions.copy(),
                )

            fold_threshold_frames: list[pd.DataFrame] = []
            for threshold_fold in range(1, threshold_fold_count + 1):
                threshold_validation_positions = assignments.threshold_selection.loc[
                    (assignments.threshold_selection["repeat"] == repeat)
                    & (assignments.threshold_selection["outer_fold"] == outer_fold)
                    & (
                        assignments.threshold_selection["threshold_selection_fold"]
                        == threshold_fold
                    ),
                    "row_position",
                ].to_numpy(dtype=np.int64)
                model_training_positions = np.setdiff1d(
                    outer_training_positions,
                    threshold_validation_positions,
                    assume_unique=True,
                )
                probabilities = fit_predict(
                    X.iloc[model_training_positions],
                    y.iloc[model_training_positions],
                    X.iloc[threshold_validation_positions],
                    candidate_contract,
                    model_training_positions=model_training_positions,
                    prediction_positions=threshold_validation_positions,
                    audit_callback=fit_audit_callback,
                )
                fold_threshold_frames.append(
                    pd.DataFrame(
                        {
                            "repeat": repeat,
                            "repeat_seed": repeat_seed,
                            "outer_fold": outer_fold,
                            "threshold_selection_fold": threshold_fold,
                            "row_position": threshold_validation_positions,
                            "target": y.iloc[threshold_validation_positions].to_numpy(
                                dtype="int8"
                            ),
                            "probability": probabilities,
                        }
                    )
                )
            threshold_oof = pd.concat(
                fold_threshold_frames,
                ignore_index=True,
            ).sort_values("row_position", ignore_index=True)
            if len(threshold_oof) != len(outer_training_positions):
                raise RuntimeError("Threshold-selection OOF coverage is incomplete.")
            if threshold_oof["row_position"].duplicated().any():
                raise RuntimeError("Threshold-selection OOF coverage has duplicates.")
            selection = threshold_selector(
                threshold_oof["target"],
                threshold_oof["probability"].to_numpy(),
                plan["threshold_policy"],
            )
            threshold_oof["selected_threshold"] = selection.threshold
            threshold_oof["prediction"] = (
                threshold_oof["probability"] >= selection.threshold
            ).astype("int8")

            outer_probabilities = fit_predict(
                X.iloc[outer_training_positions],
                y.iloc[outer_training_positions],
                X.iloc[outer_validation_positions],
                candidate_contract,
                model_training_positions=outer_training_positions,
                prediction_positions=outer_validation_positions,
                audit_callback=fit_audit_callback,
            )
            outer_predictions = (outer_probabilities >= selection.threshold).astype(
                "int8"
            )
            outer_frame = pd.DataFrame(
                {
                    "repeat": repeat,
                    "repeat_seed": repeat_seed,
                    "outer_fold": outer_fold,
                    "row_position": outer_validation_positions,
                    "target": y.iloc[outer_validation_positions].to_numpy(dtype="int8"),
                    "probability": outer_probabilities,
                    "selected_threshold": selection.threshold,
                    "prediction": outer_predictions,
                }
            ).sort_values("row_position", ignore_index=True)
            metrics = calculate_outer_metrics(
                outer_frame["target"],
                outer_frame["probability"].to_numpy(),
                outer_frame["prediction"].to_numpy(),
                selected_threshold=selection.threshold,
            )
            fold_duration = perf_counter() - fold_started
            metric_record = {
                "repeat": repeat,
                "repeat_seed": repeat_seed,
                "outer_fold": outer_fold,
                "training_rows": len(outer_training_positions),
                "validation_rows": len(outer_validation_positions),
                "threshold_selection_balanced_accuracy": (selection.balanced_accuracy),
                "threshold_status": selection.status,
                "threshold_degenerate": selection.degenerate,
                "duration_seconds": fold_duration,
                **metrics,
            }
            selected_record = {
                "repeat": repeat,
                "repeat_seed": repeat_seed,
                "outer_fold": outer_fold,
                "selected_threshold": selection.threshold,
                "threshold_selection_balanced_accuracy": (selection.balanced_accuracy),
                "status": selection.status,
                "degenerate": selection.degenerate,
            }
            curve = selection.scores.copy()
            curve.insert(0, "outer_fold", outer_fold)
            curve.insert(0, "repeat", repeat)
            completed = CompletedOuterFold(
                repeat=repeat,
                outer_fold=outer_fold,
                threshold_selection_oof=threshold_oof,
                outer_validation=outer_frame,
                selected_threshold=selected_record,
                threshold_curve=curve,
                fold_metrics=metric_record,
            )
            if on_outer_fold_complete is not None:
                on_outer_fold_complete(completed)
            threshold_frames.append(threshold_oof)
            outer_frames.append(outer_frame)
            selected_records.append(selected_record)
            curve_frames.append(curve)
            metric_records.append(metric_record)

    threshold_predictions = pd.concat(threshold_frames, ignore_index=True)
    outer_predictions = pd.concat(outer_frames, ignore_index=True)
    selected_thresholds = pd.DataFrame(selected_records)
    fold_metrics = pd.DataFrame(metric_records)
    repeat_metrics = build_repeat_metrics(outer_predictions)
    aggregate_metrics = aggregate_repeat_metrics(repeat_metrics, fold_metrics)
    threshold_summary = summarize_thresholds(selected_thresholds)
    return ResearchEvaluationResult(
        threshold_selection_oof=threshold_predictions,
        outer_validation=outer_predictions,
        selected_thresholds=selected_thresholds,
        threshold_curves=pd.concat(curve_frames, ignore_index=True),
        outer_fold_metrics=fold_metrics,
        repeat_metrics=repeat_metrics,
        aggregate_metrics=aggregate_metrics,
        threshold_summary=threshold_summary,
        duration_seconds=perf_counter() - started,
    )
