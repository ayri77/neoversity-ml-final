from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)


@dataclass(frozen=True)
class PredictionResult:
    probabilities: np.ndarray
    predictions: np.ndarray
    threshold: float

    @classmethod
    def from_probabilities(
        cls,
        probabilities: np.ndarray,
        threshold: float,
    ) -> "PredictionResult":
        """Create binary predictions from positive-class probabilities."""
        probabilities_array = np.asarray(
            probabilities,
            dtype=float,
        )

        if probabilities_array.ndim != 1:
            raise ValueError("Probabilities must be a one-dimensional array.")

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")

        predictions = (probabilities_array >= threshold).astype(int)

        return cls(
            probabilities=probabilities_array,
            predictions=predictions,
            threshold=float(threshold),
        )


@dataclass(frozen=True)
class ThresholdOptimizationResult:
    threshold: float
    balanced_accuracy: float
    scores: pd.DataFrame


@dataclass(frozen=True)
class ThresholdPlateauResult:
    best_threshold: float
    best_balanced_accuracy: float
    lower_threshold: float
    upper_threshold: float
    midpoint_threshold: float
    width: float
    tolerance: float


@dataclass(frozen=True)
class CrossFittedThresholdResult:
    fold_metrics: pd.DataFrame
    oof_predictions: pd.DataFrame
    balanced_accuracy: float


def calculate_binary_metrics(
    y_true: pd.Series | np.ndarray,
    prediction_result: PredictionResult,
) -> dict[str, float]:
    """Calculate threshold-based and probability-based metrics."""
    y_true_array = np.asarray(y_true)

    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true_array,
                prediction_result.predictions,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true_array,
                prediction_result.probabilities,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_true_array,
                prediction_result.probabilities,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true_array,
                prediction_result.probabilities,
            )
        ),
        "decision_threshold": prediction_result.threshold,
    }


def optimize_balanced_accuracy_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
) -> ThresholdOptimizationResult:
    """Find the threshold maximizing balanced accuracy."""
    y_true_array = np.asarray(y_true)
    probabilities_array = np.asarray(
        probabilities,
        dtype=float,
    )

    if probabilities_array.ndim != 1:
        raise ValueError("Probabilities must be a one-dimensional array.")

    if len(y_true_array) != len(probabilities_array):
        raise ValueError("y_true and probabilities must have the same length.")

    if thresholds is None:
        thresholds = np.linspace(
            0.01,
            0.99,
            981,
        )

    threshold_values = np.asarray(
        thresholds,
        dtype=float,
    )

    scores = np.array(
        [
            balanced_accuracy_score(
                y_true_array,
                (probabilities_array >= threshold).astype(int),
            )
            for threshold in threshold_values
        ],
        dtype=float,
    )

    best_position = int(np.argmax(scores))

    scores_frame = pd.DataFrame(
        {
            "threshold": threshold_values,
            "balanced_accuracy": scores,
        }
    )

    return ThresholdOptimizationResult(
        threshold=float(threshold_values[best_position]),
        balanced_accuracy=float(scores[best_position]),
        scores=scores_frame,
    )


def optimize_threshold_by_folds(
    fold_metrics: pd.DataFrame,
    oof_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Optimize balanced-accuracy threshold separately for each fold."""
    required_fold_columns = {"fold"}
    required_oof_columns = {
        "fold",
        "target",
        "probability",
    }

    if not required_fold_columns.issubset(fold_metrics.columns):
        raise ValueError("fold_metrics must contain a 'fold' column.")

    if not required_oof_columns.issubset(oof_predictions.columns):
        raise ValueError(
            "oof_predictions must contain fold, target, and probability columns."
        )

    records: list[dict[str, float | int]] = []

    for fold_number in sorted(oof_predictions["fold"].unique()):
        fold_data = oof_predictions.loc[oof_predictions["fold"] == fold_number]

        optimization = optimize_balanced_accuracy_threshold(
            y_true=fold_data["target"],
            probabilities=fold_data["probability"],
        )

        records.append(
            {
                "fold": int(fold_number),
                "optimized_threshold": (optimization.threshold),
                "optimized_balanced_accuracy": (optimization.balanced_accuracy),
            }
        )

    return pd.DataFrame(records)


def evaluate_cross_fitted_thresholds(
    oof_predictions: pd.DataFrame,
    *,
    thresholds: np.ndarray | None = None,
) -> CrossFittedThresholdResult:
    """
    Evaluate threshold transfer using cross-fitting.

    For each holdout fold, the decision threshold is optimized on all
    remaining folds and then evaluated on the holdout fold.
    """
    required_columns = {
        "fold",
        "target",
        "probability",
    }

    missing_columns = required_columns.difference(oof_predictions.columns)

    if missing_columns:
        raise ValueError(
            f"oof_predictions is missing required columns: {sorted(missing_columns)}"
        )

    if oof_predictions[list(required_columns)].isna().any().any():
        raise ValueError("oof_predictions contains missing values in required columns.")

    fold_numbers = sorted(oof_predictions["fold"].unique())

    if len(fold_numbers) < 2:
        raise ValueError("At least two folds are required for cross-fitted evaluation.")

    fold_records: list[dict[str, float | int]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold_number in fold_numbers:
        holdout_mask = oof_predictions["fold"] == fold_number

        calibration_data = oof_predictions.loc[~holdout_mask]

        holdout_data = oof_predictions.loc[holdout_mask].copy()

        cross_fitted_optimization = optimize_balanced_accuracy_threshold(
            y_true=calibration_data["target"],
            probabilities=calibration_data["probability"],
            thresholds=thresholds,
        )

        holdout_prediction = PredictionResult.from_probabilities(
            probabilities=holdout_data["probability"].to_numpy(),
            threshold=cross_fitted_optimization.threshold,
        )

        cross_fitted_balanced_accuracy = float(
            balanced_accuracy_score(
                holdout_data["target"],
                holdout_prediction.predictions,
            )
        )

        fold_optimal = optimize_balanced_accuracy_threshold(
            y_true=holdout_data["target"],
            probabilities=holdout_data["probability"],
            thresholds=thresholds,
        )

        fold_records.append(
            {
                "fold": int(fold_number),
                "threshold_cross_fitted": (cross_fitted_optimization.threshold),
                "balanced_accuracy_cross_fitted": (cross_fitted_balanced_accuracy),
                "threshold_fold_optimal": (fold_optimal.threshold),
                "balanced_accuracy_fold_optimal": (fold_optimal.balanced_accuracy),
                "balanced_accuracy_regret": (
                    fold_optimal.balanced_accuracy - cross_fitted_balanced_accuracy
                ),
            }
        )

        holdout_data["threshold_cross_fitted"] = cross_fitted_optimization.threshold
        holdout_data["prediction_cross_fitted"] = holdout_prediction.predictions

        prediction_frames.append(holdout_data)

    cross_fitted_predictions = pd.concat(prediction_frames).sort_index()

    overall_balanced_accuracy = float(
        balanced_accuracy_score(
            cross_fitted_predictions["target"],
            cross_fitted_predictions["prediction_cross_fitted"],
        )
    )

    return CrossFittedThresholdResult(
        fold_metrics=pd.DataFrame(fold_records),
        oof_predictions=cross_fitted_predictions,
        balanced_accuracy=overall_balanced_accuracy,
    )


def summarize_threshold_plateau(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    tolerance: float = 0.001,
    thresholds: np.ndarray | None = None,
) -> ThresholdPlateauResult:
    """
    Summarize the threshold interval whose balanced accuracy remains
    within the specified tolerance of the optimum.
    """
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative.")

    optimization = optimize_balanced_accuracy_threshold(
        y_true=y_true,
        probabilities=probabilities,
        thresholds=thresholds,
    )

    minimum_accepted_score = optimization.balanced_accuracy - tolerance

    plateau_scores = optimization.scores.loc[
        optimization.scores["balanced_accuracy"] >= minimum_accepted_score
    ]

    lower_threshold = float(plateau_scores["threshold"].min())
    upper_threshold = float(plateau_scores["threshold"].max())

    return ThresholdPlateauResult(
        best_threshold=optimization.threshold,
        best_balanced_accuracy=optimization.balanced_accuracy,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        midpoint_threshold=(lower_threshold + upper_threshold) / 2,
        width=upper_threshold - lower_threshold,
        tolerance=float(tolerance),
    )
