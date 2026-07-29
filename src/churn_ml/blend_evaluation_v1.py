from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.churn_ml.paired_comparison import (
    COMPARISON_SCHEMA_VERSION,
    ComparisonCompatibilitySummary,
    CompletedResearchV2Run,
    InputRunReference,
    build_compatibility_summary,
    load_completed_research_v2_run,
)
from src.churn_ml.research_data import canonical_sha256
from src.churn_ml.research_protocol import (
    calculate_outer_metrics,
    select_balanced_accuracy_threshold,
)


BLEND_EVALUATION_SCHEMA_VERSION = 1
OUTER_KEYS = ["repeat", "repeat_seed", "outer_fold", "row_position"]
PRINCIPAL_METRICS = (
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "roc_auc",
    "average_precision",
    "brier_score",
)
HIGHER_IS_BETTER = {
    "balanced_accuracy": True,
    "sensitivity": True,
    "specificity": True,
    "roc_auc": True,
    "average_precision": True,
    "brier_score": False,
}
LIGHTGBM_ADAPTER = "manual_lightgbm_te_v1_compat"
XGBOOST_ADAPTER = "xgboost_numeric_v1"
DEFAULT_WEIGHT_TOLERANCE = 1.0e-12


class BlendEvaluationError(ValueError):
    """Raised when a leakage-safe blend evaluation contract cannot be satisfied."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "BLEND_EVALUATION_INVALID",
        field_path: str = "blend_evaluation",
    ) -> None:
        self.reason_code = reason_code
        self.field_path = field_path
        self.detail = message
        super().__init__(f"{reason_code} at {field_path}: {message}")


@dataclass(frozen=True)
class BlendWeightSelection:
    lightgbm_weight: float
    threshold: float
    training_balanced_accuracy: float
    threshold_status: str
    threshold_degenerate: bool
    training_rows: int
    evaluation_rows: int


@dataclass(frozen=True)
class BlendEvaluationResult:
    compatibility: ComparisonCompatibilitySummary
    lightgbm_reference: InputRunReference
    xgboost_reference: InputRunReference
    fold_selections: pd.DataFrame
    held_out_predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    repeat_metrics: pd.DataFrame
    summary: dict[str, Any]
    deployment_parameters: dict[str, Any]
    sensitivity_table: pd.DataFrame


def load_blend_component_run(
    run_dir: Any,
    *,
    project_root: Any,
    role: str,
    expected_adapter_id: str,
) -> CompletedResearchV2Run:
    run = load_completed_research_v2_run(
        run_dir,
        project_root=project_root,
        role=role,
    )
    if run.config.adapter_id != expected_adapter_id:
        raise BlendEvaluationError(
            f"Expected adapter_id {expected_adapter_id!r}, got {run.config.adapter_id!r}.",
            reason_code="ADAPTER_ID_MISMATCH",
            field_path=f"components.{role}.adapter_id",
        )
    _assert_outer_prediction_contract(run, role=role)
    return run


def validate_blend_inputs(
    lightgbm_run: CompletedResearchV2Run,
    xgboost_run: CompletedResearchV2Run,
) -> ComparisonCompatibilitySummary:
    if lightgbm_run.config.adapter_id != LIGHTGBM_ADAPTER:
        raise BlendEvaluationError(
            f"LightGBM component must use {LIGHTGBM_ADAPTER}.",
            reason_code="ADAPTER_ID_MISMATCH",
            field_path="components.lightgbm.adapter_id",
        )
    if xgboost_run.config.adapter_id != XGBOOST_ADAPTER:
        raise BlendEvaluationError(
            f"XGBoost component must use {XGBOOST_ADAPTER}.",
            reason_code="ADAPTER_ID_MISMATCH",
            field_path="components.xgboost.adapter_id",
        )
    compatibility = build_compatibility_summary(lightgbm_run, xgboost_run)
    if not compatibility.compatible:
        raise BlendEvaluationError(
            "Base artifacts are not pair-compatible for blending.",
            reason_code="COMPATIBILITY_FAILED",
            field_path="compatibility",
        )
    _assert_aligned_outer_coverage(lightgbm_run, xgboost_run)
    return compatibility


def build_weight_grid(
    *,
    minimum: float = 0.0,
    maximum: float = 1.0,
    step: float = 0.025,
) -> np.ndarray:
    for label, value in {
        "minimum": minimum,
        "maximum": maximum,
        "step": step,
    }.items():
        if type(value) is not float or not np.isfinite(value):
            raise BlendEvaluationError(
                f"Weight grid {label} must be a finite float.",
                field_path=f"blend.lightgbm_weight_grid.{label}",
            )
    if not 0.0 <= minimum <= maximum <= 1.0:
        raise BlendEvaluationError(
            "Weight grid bounds must satisfy 0 <= minimum <= maximum <= 1.",
            field_path="blend.lightgbm_weight_grid",
        )
    if step <= 0.0:
        raise BlendEvaluationError(
            "Weight grid step must be positive.",
            field_path="blend.lightgbm_weight_grid.step",
        )
    scale = 1000
    start = int(round(minimum * scale))
    stop = int(round(maximum * scale))
    step_i = int(round(step * scale))
    if step_i <= 0 or abs(step_i / scale - step) > 1.0e-12:
        raise BlendEvaluationError(
            "Weight grid step must be an exact thousandths multiple.",
            field_path="blend.lightgbm_weight_grid.step",
        )
    values = np.arange(start, stop + 1, step_i, dtype=np.int64) / float(scale)
    if values.size == 0:
        raise BlendEvaluationError(
            "Weight grid is empty.",
            field_path="blend.lightgbm_weight_grid",
        )
    if not np.isclose(values[0], minimum) or not np.isclose(values[-1], maximum):
        raise BlendEvaluationError(
            "Weight grid does not include both endpoints.",
            field_path="blend.lightgbm_weight_grid",
        )
    return np.asarray(values, dtype=np.float64)


def select_blend_weight_and_threshold(
    y_true: np.ndarray | pd.Series,
    lightgbm_probability: np.ndarray,
    xgboost_probability: np.ndarray,
    *,
    threshold_policy: Mapping[str, Any],
    weight_grid: Sequence[float] | np.ndarray,
    maximizer_absolute_tolerance: float = DEFAULT_WEIGHT_TOLERANCE,
    prefer_closest_to: float = 0.5,
) -> BlendWeightSelection:
    targets = np.asarray(y_true)
    p_lgbm = _probability_vector(lightgbm_probability, len(targets), "lightgbm")
    p_xgb = _probability_vector(xgboost_probability, len(targets), "xgboost")
    if targets.size < 2:
        raise BlendEvaluationError(
            "Blend training fold must contain at least two rows.",
            reason_code="INSUFFICIENT_BLEND_TRAINING_ROWS",
            field_path="cross_fitting.training_rows",
        )
    if (
        type(maximizer_absolute_tolerance) is not float
        or maximizer_absolute_tolerance < 0.0
    ):
        raise BlendEvaluationError(
            "Weight maximizer tolerance must be a non-negative float.",
            field_path="blend.weight_selection.maximizer_absolute_tolerance",
        )
    if type(prefer_closest_to) is not float or not 0.0 <= prefer_closest_to <= 1.0:
        raise BlendEvaluationError(
            "prefer_closest_to must be a float in [0, 1].",
            field_path="blend.weight_selection.prefer_closest_to",
        )

    candidates: list[tuple[float, float, Any]] = []
    for weight in weight_grid:
        w = float(weight)
        if not 0.0 <= w <= 1.0 or not np.isfinite(w):
            raise BlendEvaluationError(
                "Weight grid contains an invalid weight.",
                field_path="blend.lightgbm_weight_grid",
            )
        blended = w * p_lgbm + (1.0 - w) * p_xgb
        selection = select_balanced_accuracy_threshold(
            targets,
            blended,
            dict(threshold_policy),
        )
        candidates.append((float(selection.balanced_accuracy), w, selection))

    max_ba = max(item[0] for item in candidates)
    maximizers = [
        item
        for item in candidates
        if abs(item[0] - max_ba) <= maximizer_absolute_tolerance
    ]
    min_distance = min(abs(item[1] - prefer_closest_to) for item in maximizers)
    closest = [
        item
        for item in maximizers
        if abs(abs(item[1] - prefer_closest_to) - min_distance)
        <= maximizer_absolute_tolerance
    ]
    best_ba, best_weight, best_selection = min(closest, key=lambda item: item[1])
    return BlendWeightSelection(
        lightgbm_weight=float(best_weight),
        threshold=float(best_selection.threshold),
        training_balanced_accuracy=float(best_ba),
        threshold_status=str(best_selection.status),
        threshold_degenerate=bool(best_selection.degenerate),
        training_rows=int(targets.size),
        evaluation_rows=0,
    )


def run_cross_fit_blend_evaluation(
    lightgbm_run: CompletedResearchV2Run,
    xgboost_run: CompletedResearchV2Run,
    *,
    weight_grid: Sequence[float] | np.ndarray | None = None,
    maximizer_absolute_tolerance: float = DEFAULT_WEIGHT_TOLERANCE,
    prefer_closest_to: float = 0.5,
    sensitivity_weight_deltas: Sequence[float] = (0.05, 0.10),
    forbid_pooled_oof_fallback: bool = True,
) -> BlendEvaluationResult:
    if not forbid_pooled_oof_fallback:
        raise BlendEvaluationError(
            "Pooled OOF fallback is forbidden for competition-critical blends.",
            reason_code="OPTIMISTIC_FALLBACK_FORBIDDEN",
            field_path="blend.cross_fitting.forbid_pooled_oof_fallback",
        )
    compatibility = validate_blend_inputs(lightgbm_run, xgboost_run)
    grid = (
        np.asarray(weight_grid, dtype=np.float64)
        if weight_grid is not None
        else build_weight_grid()
    )
    lightgbm_outer, xgboost_outer = _aligned_outer_frames(lightgbm_run, xgboost_run)
    threshold_policy = lightgbm_run.config.plan_payload["threshold_policy"]

    fold_records: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for repeat in sorted(lightgbm_outer["repeat"].unique()):
        repeat_mask = lightgbm_outer["repeat"] == repeat
        repeat_folds = sorted(
            lightgbm_outer.loc[repeat_mask, "outer_fold"].unique().tolist()
        )
        if len(repeat_folds) < 2:
            raise BlendEvaluationError(
                "Second-level cross-fitting requires at least two outer folds per repeat.",
                reason_code="INSUFFICIENT_OUTER_FOLDS",
                field_path="cross_fitting.outer_folds",
            )
        repeat_seed = int(lightgbm_outer.loc[repeat_mask, "repeat_seed"].iloc[0])
        for outer_fold in repeat_folds:
            eval_mask = repeat_mask & (lightgbm_outer["outer_fold"] == outer_fold)
            train_mask = repeat_mask & (lightgbm_outer["outer_fold"] != outer_fold)
            if int(eval_mask.sum()) == 0 or int(train_mask.sum()) == 0:
                raise BlendEvaluationError(
                    "Held-out fold or blend-training fold is empty.",
                    reason_code="EMPTY_CROSS_FIT_SPLIT",
                    field_path="cross_fitting",
                )
            eval_positions = set(
                lightgbm_outer.loc[eval_mask, "row_position"].astype(int).tolist()
            )
            train_positions = set(
                lightgbm_outer.loc[train_mask, "row_position"].astype(int).tolist()
            )
            if eval_positions & train_positions:
                raise BlendEvaluationError(
                    "Evaluation fold leaked into blend-training rows.",
                    reason_code="EVALUATION_FOLD_LEAKAGE",
                    field_path="cross_fitting",
                )

            selection = select_blend_weight_and_threshold(
                lightgbm_outer.loc[train_mask, "target"],
                lightgbm_outer.loc[train_mask, "probability"].to_numpy(dtype=float),
                xgboost_outer.loc[train_mask, "probability"].to_numpy(dtype=float),
                threshold_policy=threshold_policy,
                weight_grid=grid,
                maximizer_absolute_tolerance=maximizer_absolute_tolerance,
                prefer_closest_to=prefer_closest_to,
            )
            eval_lgbm = lightgbm_outer.loc[eval_mask]
            eval_xgb = xgboost_outer.loc[eval_mask]
            weight = selection.lightgbm_weight
            blend_probability = weight * eval_lgbm["probability"].to_numpy(
                dtype=float
            ) + (1.0 - weight) * eval_xgb["probability"].to_numpy(dtype=float)
            prediction = (blend_probability >= selection.threshold).astype("int8")
            held_out = eval_lgbm[OUTER_KEYS + ["target"]].copy()
            held_out["lightgbm_probability"] = eval_lgbm["probability"].to_numpy(
                dtype=float
            )
            held_out["xgboost_probability"] = eval_xgb["probability"].to_numpy(
                dtype=float
            )
            held_out["probability"] = blend_probability
            held_out["selected_lightgbm_weight"] = weight
            held_out["selected_threshold"] = selection.threshold
            held_out["prediction"] = prediction
            prediction_frames.append(held_out)

            blend_metrics = _metrics_from_predictions(
                held_out["target"].to_numpy(),
                blend_probability,
                prediction,
            )
            lightgbm_metrics = _metrics_from_frame(eval_lgbm)
            xgboost_metrics = _metrics_from_frame(eval_xgb)
            record: dict[str, Any] = {
                "repeat": int(repeat),
                "repeat_seed": repeat_seed,
                "outer_fold": int(outer_fold),
                "training_rows": selection.training_rows,
                "evaluation_rows": int(len(held_out)),
                "selected_lightgbm_weight": weight,
                "selected_threshold": selection.threshold,
                "training_balanced_accuracy": selection.training_balanced_accuracy,
                "threshold_status": selection.threshold_status,
                "threshold_degenerate": selection.threshold_degenerate,
                "predicted_positive_rate": float(prediction.mean()),
            }
            for metric in PRINCIPAL_METRICS:
                record[f"blend_{metric}"] = blend_metrics[metric]
                record[f"lightgbm_{metric}"] = lightgbm_metrics[metric]
                record[f"xgboost_{metric}"] = xgboost_metrics[metric]
                record[f"blend_minus_lightgbm_{metric}"] = (
                    blend_metrics[metric] - lightgbm_metrics[metric]
                )
                record[f"blend_minus_xgboost_{metric}"] = (
                    blend_metrics[metric] - xgboost_metrics[metric]
                )
            fold_records.append(record)

    held_out_predictions = pd.concat(prediction_frames, ignore_index=True)
    _assert_complete_held_out_coverage(lightgbm_outer, held_out_predictions)
    fold_metrics = pd.DataFrame(fold_records).sort_values(
        ["repeat", "outer_fold"], kind="mergesort"
    )
    repeat_metrics = _build_repeat_metrics(
        held_out_predictions,
        lightgbm_outer,
        xgboost_outer,
        fold_metrics,
    )
    deployment_parameters = _deployment_parameters(fold_metrics)
    summary = _build_summary(
        fold_metrics=fold_metrics,
        repeat_metrics=repeat_metrics,
        deployment_parameters=deployment_parameters,
        weight_grid=grid,
        threshold_policy=threshold_policy,
    )
    sensitivity_table = _build_sensitivity_table(
        held_out_predictions,
        deployment_weight=float(deployment_parameters["deployment_lightgbm_weight"]),
        deployment_threshold=float(deployment_parameters["deployment_threshold"]),
        deltas=sensitivity_weight_deltas,
    )
    return BlendEvaluationResult(
        compatibility=compatibility,
        lightgbm_reference=lightgbm_run.reference("lightgbm"),
        xgboost_reference=xgboost_run.reference("xgboost"),
        fold_selections=fold_metrics[
            [
                "repeat",
                "repeat_seed",
                "outer_fold",
                "training_rows",
                "evaluation_rows",
                "selected_lightgbm_weight",
                "selected_threshold",
                "training_balanced_accuracy",
                "threshold_status",
                "threshold_degenerate",
            ]
        ].copy(),
        held_out_predictions=held_out_predictions,
        fold_metrics=fold_metrics,
        repeat_metrics=repeat_metrics,
        summary=summary,
        deployment_parameters=deployment_parameters,
        sensitivity_table=sensitivity_table,
    )


def blend_candidate_identity(
    lightgbm_candidate_sha256: str,
    xgboost_candidate_sha256: str,
    *,
    deployment_lightgbm_weight: float,
) -> str:
    return canonical_sha256(
        {
            "schema_version": BLEND_EVALUATION_SCHEMA_VERSION,
            "components": {
                "lightgbm_candidate_sha256": lightgbm_candidate_sha256,
                "xgboost_candidate_sha256": xgboost_candidate_sha256,
            },
            "deployment_lightgbm_weight": float(deployment_lightgbm_weight),
            "blend_method": "linear_probability_mean",
        }
    )


def apply_fixed_blend_probabilities(
    lightgbm_probability: np.ndarray,
    xgboost_probability: np.ndarray,
    *,
    lightgbm_weight: float,
) -> np.ndarray:
    if type(lightgbm_weight) is not float or not np.isfinite(lightgbm_weight):
        raise BlendEvaluationError("Blend weight must be a finite float.")
    if not 0.0 <= lightgbm_weight <= 1.0:
        raise BlendEvaluationError("Blend weight must be within [0, 1].")
    p_lgbm = _probability_vector(
        lightgbm_probability, len(lightgbm_probability), "lightgbm"
    )
    p_xgb = _probability_vector(
        xgboost_probability, len(xgboost_probability), "xgboost"
    )
    if p_lgbm.shape != p_xgb.shape:
        raise BlendEvaluationError("Component probability shapes differ.")
    return lightgbm_weight * p_lgbm + (1.0 - lightgbm_weight) * p_xgb


def clip_weight(weight: float) -> float:
    if type(weight) is not float or not np.isfinite(weight):
        raise BlendEvaluationError("Weight must be a finite float.")
    return float(min(1.0, max(0.0, weight)))


def _assert_outer_prediction_contract(
    run: CompletedResearchV2Run, *, role: str
) -> None:
    required = set(OUTER_KEYS + ["target", "probability"])
    columns = set(run.outer_predictions.columns)
    missing = sorted(required - columns)
    if missing:
        raise BlendEvaluationError(
            f"Outer predictions missing required columns: {missing}.",
            reason_code="OUTER_PREDICTION_CONTRACT_INCOMPLETE",
            field_path=f"components.{role}.outer_predictions",
        )
    frame = run.outer_predictions
    if frame.empty:
        raise BlendEvaluationError(
            "Outer predictions are empty.",
            reason_code="OUTER_PREDICTION_EMPTY",
            field_path=f"components.{role}.outer_predictions",
        )
    if frame[OUTER_KEYS].duplicated().any():
        raise BlendEvaluationError(
            "Outer predictions contain duplicate held-out keys.",
            reason_code="DUPLICATE_HELD_OUT_ROWS",
            field_path=f"components.{role}.outer_predictions",
        )
    if frame["target"].isna().any() or frame["probability"].isna().any():
        raise BlendEvaluationError(
            "Outer predictions contain missing targets or probabilities.",
            reason_code="MISSING_PROBABILITY_OR_TARGET",
            field_path=f"components.{role}.outer_predictions",
        )
    probabilities = frame["probability"].to_numpy(dtype=float)
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
    ):
        raise BlendEvaluationError(
            "Outer probabilities must be finite values in [0, 1].",
            reason_code="INVALID_PROBABILITY_SEMANTICS",
            field_path=f"components.{role}.probability",
        )


def _assert_aligned_outer_coverage(
    lightgbm_run: CompletedResearchV2Run,
    xgboost_run: CompletedResearchV2Run,
) -> None:
    left = lightgbm_run.outer_predictions.sort_values(OUTER_KEYS, kind="mergesort")
    right = xgboost_run.outer_predictions.sort_values(OUTER_KEYS, kind="mergesort")
    if (
        not left[OUTER_KEYS]
        .reset_index(drop=True)
        .equals(right[OUTER_KEYS].reset_index(drop=True))
    ):
        raise BlendEvaluationError(
            "LightGBM and XGBoost outer prediction keys are not aligned.",
            reason_code="OUTER_KEY_MISALIGNMENT",
            field_path="outer_predictions",
        )
    if not np.array_equal(
        left["target"].to_numpy(),
        right["target"].to_numpy(),
    ):
        raise BlendEvaluationError(
            "Aligned outer-validation targets differ.",
            reason_code="TARGET_MISMATCH",
            field_path="outer_predictions.target",
        )


def _aligned_outer_frames(
    lightgbm_run: CompletedResearchV2Run,
    xgboost_run: CompletedResearchV2Run,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = lightgbm_run.outer_predictions.sort_values(
        OUTER_KEYS, kind="mergesort"
    ).reset_index(drop=True)
    right = xgboost_run.outer_predictions.sort_values(
        OUTER_KEYS, kind="mergesort"
    ).reset_index(drop=True)
    return left, right


def _assert_complete_held_out_coverage(
    source_outer: pd.DataFrame,
    held_out: pd.DataFrame,
) -> None:
    expected = source_outer[OUTER_KEYS].sort_values(OUTER_KEYS, kind="mergesort")
    actual = held_out[OUTER_KEYS].sort_values(OUTER_KEYS, kind="mergesort")
    if not expected.reset_index(drop=True).equals(actual.reset_index(drop=True)):
        raise BlendEvaluationError(
            "Held-out blend predictions do not cover every outer validation row exactly once.",
            reason_code="HELD_OUT_COVERAGE_MISMATCH",
            field_path="held_out_predictions",
        )
    if held_out[OUTER_KEYS].duplicated().any():
        raise BlendEvaluationError(
            "Held-out blend predictions contain duplicates.",
            reason_code="DUPLICATE_HELD_OUT_ROWS",
            field_path="held_out_predictions",
        )


def _probability_vector(values: Any, expected_rows: int, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.shape[0] != expected_rows:
        raise BlendEvaluationError(
            f"{label} probability vector shape is invalid.",
            field_path=f"probabilities.{label}",
        )
    if not np.isfinite(array).all() or np.any(array < 0.0) or np.any(array > 1.0):
        raise BlendEvaluationError(
            f"{label} probabilities must be finite values in [0, 1].",
            field_path=f"probabilities.{label}",
        )
    return array


def _metrics_from_frame(frame: pd.DataFrame) -> dict[str, float]:
    return _metrics_from_predictions(
        frame["target"].to_numpy(),
        frame["probability"].to_numpy(dtype=float),
        frame["prediction"].to_numpy(dtype=np.int8)
        if "prediction" in frame.columns
        else (
            frame["probability"].to_numpy(dtype=float)
            >= frame["selected_threshold"].to_numpy(dtype=float)
        ).astype(np.int8),
    )


def _metrics_from_predictions(
    targets: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    *,
    selected_threshold: float | None = None,
) -> dict[str, float]:
    metrics = calculate_outer_metrics(
        targets,
        probabilities,
        predictions,
        selected_threshold=selected_threshold,
    )
    result: dict[str, float] = {}
    for name in PRINCIPAL_METRICS:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BlendEvaluationError(f"Metric {name} is not numeric.")
        result[name] = float(value)
    return result


def _build_repeat_metrics(
    held_out: pd.DataFrame,
    lightgbm_outer: pd.DataFrame,
    xgboost_outer: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for repeat in sorted(held_out["repeat"].unique()):
        blend_frame = held_out.loc[held_out["repeat"] == repeat]
        lightgbm_frame = lightgbm_outer.loc[lightgbm_outer["repeat"] == repeat]
        xgboost_frame = xgboost_outer.loc[xgboost_outer["repeat"] == repeat]
        blend_metrics = _metrics_from_predictions(
            blend_frame["target"].to_numpy(),
            blend_frame["probability"].to_numpy(dtype=float),
            blend_frame["prediction"].to_numpy(dtype=np.int8),
        )
        lightgbm_metrics = _metrics_from_frame(lightgbm_frame)
        xgboost_metrics = _metrics_from_frame(xgboost_frame)
        fold_subset = fold_metrics.loc[fold_metrics["repeat"] == repeat]
        record: dict[str, Any] = {
            "repeat": int(repeat),
            "validation_rows": int(len(blend_frame)),
            "median_selected_lightgbm_weight": float(
                np.median(fold_subset["selected_lightgbm_weight"].to_numpy(dtype=float))
            ),
            "median_selected_threshold": float(
                np.median(fold_subset["selected_threshold"].to_numpy(dtype=float))
            ),
            "predicted_positive_rate": float(blend_frame["prediction"].mean()),
        }
        for metric in PRINCIPAL_METRICS:
            record[f"blend_{metric}"] = blend_metrics[metric]
            record[f"lightgbm_{metric}"] = lightgbm_metrics[metric]
            record[f"xgboost_{metric}"] = xgboost_metrics[metric]
            record[f"blend_minus_lightgbm_{metric}"] = (
                blend_metrics[metric] - lightgbm_metrics[metric]
            )
            record[f"blend_minus_xgboost_{metric}"] = (
                blend_metrics[metric] - xgboost_metrics[metric]
            )
        records.append(record)
    return pd.DataFrame(records)


def _deployment_parameters(fold_metrics: pd.DataFrame) -> dict[str, Any]:
    weights = fold_metrics["selected_lightgbm_weight"].to_numpy(dtype=float)
    thresholds = fold_metrics["selected_threshold"].to_numpy(dtype=float)
    return {
        "schema_version": BLEND_EVALUATION_SCHEMA_VERSION,
        "deployment_lightgbm_weight": float(np.median(weights)),
        "deployment_threshold": float(np.median(thresholds)),
        "weight_aggregation": "median",
        "threshold_aggregation": "median",
        "weight_summary": _distribution_summary(weights),
        "threshold_summary": _distribution_summary(thresholds),
        "fold_count": int(len(fold_metrics)),
        "comparison": "greater_than_or_equal",
    }


def _build_summary(
    *,
    fold_metrics: pd.DataFrame,
    repeat_metrics: pd.DataFrame,
    deployment_parameters: dict[str, Any],
    weight_grid: np.ndarray,
    threshold_policy: Mapping[str, Any],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for metric in PRINCIPAL_METRICS:
        blend_values = repeat_metrics[f"blend_{metric}"].to_numpy(dtype=float)
        metrics[metric] = {
            "repeat_distribution": _distribution_summary(blend_values),
            "aggregate_mean": float(blend_values.mean()),
            "aggregate_median": float(np.median(blend_values)),
            "deltas_versus_lightgbm": _distribution_summary(
                repeat_metrics[f"blend_minus_lightgbm_{metric}"].to_numpy(dtype=float)
            ),
            "deltas_versus_xgboost": _distribution_summary(
                repeat_metrics[f"blend_minus_xgboost_{metric}"].to_numpy(dtype=float)
            ),
            "higher_is_better": HIGHER_IS_BETTER[metric],
            "confidence_intervals": None,
            "confidence_intervals_omitted_reason": (
                "Overlapping repeats are not independent; CIs are intentionally omitted."
            ),
        }
    positive_rates = repeat_metrics["predicted_positive_rate"].to_numpy(dtype=float)
    return {
        "schema_version": BLEND_EVALUATION_SCHEMA_VERSION,
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "protocol": "leave_one_outer_fold_out_within_repeat",
        "blend_formula": "p_blend = w * p_lightgbm + (1 - w) * p_xgboost",
        "weight_grid": [float(value) for value in weight_grid.tolist()],
        "threshold_policy_id": threshold_policy["id"],
        "tuning_excludes_evaluation_fold": True,
        "pooled_oof_fallback_used": False,
        "competition_test_used": False,
        "metrics": metrics,
        "predicted_positive_rate": _distribution_summary(positive_rates),
        "deployment_parameters": deployment_parameters,
        "weight_summary": deployment_parameters["weight_summary"],
        "threshold_summary": deployment_parameters["threshold_summary"],
        "fold_count": int(len(fold_metrics)),
        "repeat_count": int(len(repeat_metrics)),
    }


def _build_sensitivity_table(
    held_out: pd.DataFrame,
    *,
    deployment_weight: float,
    deployment_threshold: float,
    deltas: Sequence[float],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    offsets = [0.0]
    for delta in deltas:
        if type(delta) is not float or not np.isfinite(delta) or delta <= 0.0:
            raise BlendEvaluationError(
                "Sensitivity deltas must be positive finite floats.",
                field_path="deployment_parameters.sensitivity_weight_deltas",
            )
        offsets.extend([delta, -delta])
    for offset in offsets:
        weight = clip_weight(deployment_weight + float(offset))
        probability = apply_fixed_blend_probabilities(
            held_out["lightgbm_probability"].to_numpy(dtype=float),
            held_out["xgboost_probability"].to_numpy(dtype=float),
            lightgbm_weight=weight,
        )
        prediction = (probability >= deployment_threshold).astype(np.int8)
        metrics = _metrics_from_predictions(
            held_out["target"].to_numpy(),
            probability,
            prediction,
        )
        records.append(
            {
                "weight_offset": float(offset),
                "lightgbm_weight": weight,
                "threshold": float(deployment_threshold),
                "leaderboard_probe": abs(offset) > 0.0,
                "predicted_positive_rate": float(prediction.mean()),
                **{metric: metrics[metric] for metric in PRINCIPAL_METRICS},
            }
        )
    return pd.DataFrame(records)


def _distribution_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise BlendEvaluationError("Cannot summarize an empty value vector.")
    q25 = float(np.quantile(array, 0.25))
    q75 = float(np.quantile(array, 0.75))
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "range": float(array.max() - array.min()),
        "iqr": float(q75 - q25),
        "q25": q25,
        "q75": q75,
        "sample_standard_deviation": (
            float(array.std(ddof=1)) if array.size >= 2 else None
        ),
    }
