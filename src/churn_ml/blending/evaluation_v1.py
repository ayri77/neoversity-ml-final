"""Honest meta-cross-validation and descriptive metrics for blends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.churn_ml.blending.compatibility_v1 import CompatibleCandidateSet
from src.churn_ml.blending.optimization_v1 import (
    DEFAULT_THRESHOLD_POLICY,
    TIE_BREAK_RULE,
    WeightThresholdSelection,
    blend_probabilities,
    equal_weights,
    optimize_weights,
    parse_manual_weights,
    score_weights,
    select_threshold_for_probabilities,
    validate_weights,
)
from src.churn_ml.research_data import canonical_sha256
from src.churn_ml.research_protocol import calculate_outer_metrics


BLEND_SOURCE_KIND = "canonical_probability_blend_v1"
OOF_PROTOCOL = "meta_cv_mean_probability_across_repeats"
REPEAT_NORMALIZATION_POLICY = "mean_cross_fitted_probability_across_meta_repeats"
STRATEGIES = frozenset({"equal", "manual", "optimized"})


@dataclass(frozen=True)
class BlendSettings:
    strategy: str
    folds: int = 5
    repeats: int = 2
    seed: int = 42
    max_active_models: int | None = None
    manual_weights: tuple[str, ...] = ()
    threshold_policy: Mapping[str, Any] | None = None
    pairwise_grid_step: float = 0.1
    dirichlet_draws: int = 32

    def normalized(self) -> "BlendSettings":
        strategy = str(self.strategy)
        if strategy not in STRATEGIES:
            raise ValueError(f"Unsupported strategy {strategy!r}")
        if self.folds < 2:
            raise ValueError("folds must be >= 2")
        if self.repeats < 1:
            raise ValueError("repeats must be >= 1")
        return BlendSettings(
            strategy=strategy,
            folds=int(self.folds),
            repeats=int(self.repeats),
            seed=int(self.seed),
            max_active_models=(
                None
                if self.max_active_models is None
                else int(self.max_active_models)
            ),
            manual_weights=tuple(self.manual_weights),
            threshold_policy=dict(self.threshold_policy or DEFAULT_THRESHOLD_POLICY),
            pairwise_grid_step=float(self.pairwise_grid_step),
            dirichlet_draws=int(self.dirichlet_draws),
        )

    def identity_payload(
        self,
        *,
        parent_candidate_ids: Sequence[str],
        parent_manifest_hashes: Sequence[str],
    ) -> dict[str, Any]:
        settings = self.normalized()
        return {
            "contract_version": "prediction_candidate_v1",
            "blend_schema_version": "prediction_blend_v1",
            "parent_candidate_ids": list(parent_candidate_ids),
            "parent_manifest_hashes": list(parent_manifest_hashes),
            "strategy": settings.strategy,
            "folds": settings.folds,
            "repeats": settings.repeats,
            "seed": settings.seed,
            "max_active_models": settings.max_active_models,
            "manual_weights": list(settings.manual_weights),
            "threshold_policy": dict(settings.threshold_policy or {}),
            "pairwise_grid_step": settings.pairwise_grid_step,
            "dirichlet_draws": settings.dirichlet_draws,
            "oof_protocol": OOF_PROTOCOL,
            "repeat_normalization_policy": REPEAT_NORMALIZATION_POLICY,
            "tie_break_rule": TIE_BREAK_RULE,
        }


@dataclass(frozen=True)
class BlendEvaluationResult:
    settings: BlendSettings
    fixed_or_final_weights: np.ndarray
    final_threshold: float
    cross_fitted_oof: np.ndarray
    test_probabilities: np.ndarray
    assignments: pd.DataFrame
    assignment_hash: str
    fold_results: list[dict[str, Any]]
    fold_metrics: pd.DataFrame
    repeat_metrics: pd.DataFrame
    cross_fitted_metrics: dict[str, Any]
    full_oof_descriptive_metrics: dict[str, Any]
    candidate_descriptive_metrics: list[dict[str, Any]]
    search_budget: dict[str, Any]
    deployment: dict[str, Any]


def compute_candidate_descriptive_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    candidate_id: str,
) -> dict[str, Any]:
    probs = np.asarray(probabilities, dtype=np.float64)
    metrics_05 = calculate_outer_metrics(
        y_true,
        probs,
        (probs >= 0.5).astype(np.int8),
        selected_threshold=0.5,
    )
    optimal = select_threshold_for_probabilities(y_true, probs)
    preds_opt = (probs >= optimal.threshold).astype(np.int8)
    metrics_opt = calculate_outer_metrics(
        y_true, probs, preds_opt, selected_threshold=optimal.threshold
    )
    return {
        "candidate_id": candidate_id,
        "threshold_0_5": {
            "balanced_accuracy": metrics_05["balanced_accuracy"],
            "sensitivity": metrics_05["sensitivity"],
            "specificity": metrics_05["specificity"],
            "roc_auc": metrics_05["roc_auc"],
            "average_precision": metrics_05["average_precision"],
            "brier_score": metrics_05["brier_score"],
            "positive_prediction_rate": metrics_05["predicted_positive_rate"],
            "label": "fixed_threshold_0.5",
        },
        "descriptive_oof_optimal_threshold": float(optimal.threshold),
        "descriptive_oof_optimal_metrics": {
            "balanced_accuracy": metrics_opt["balanced_accuracy"],
            "sensitivity": metrics_opt["sensitivity"],
            "specificity": metrics_opt["specificity"],
            "roc_auc": metrics_opt["roc_auc"],
            "average_precision": metrics_opt["average_precision"],
            "brier_score": metrics_opt["brier_score"],
            "positive_prediction_rate": metrics_opt["predicted_positive_rate"],
            "label": "full_OOF_descriptive_optimal_threshold_not_unbiased",
        },
        "probability_mean": float(probs.mean()),
        "probability_std": float(probs.std(ddof=0)),
        "probability_min": float(probs.min()),
        "probability_max": float(probs.max()),
        "threshold_tie_break": (
            "middle maximizer on deterministic threshold grid "
            "(research_protocol.select_balanced_accuracy_threshold)"
        ),
    }


def build_meta_assignments(
    y_true: np.ndarray,
    *,
    folds: int,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    n = len(y_true)
    row_positions = np.arange(n, dtype=np.int64)
    for repeat in range(1, repeats + 1):
        repeat_seed = int(seed + repeat - 1)
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=repeat_seed,
        )
        for fold_index, (train_idx, val_idx) in enumerate(
            splitter.split(np.zeros(n), y_true), start=1
        ):
            for local_index, row in enumerate(train_idx):
                records.append(
                    {
                        "repeat": repeat,
                        "repeat_seed": repeat_seed,
                        "fold": fold_index,
                        "role": "meta_train",
                        "row_position": int(row_positions[row]),
                    }
                )
            for row in val_idx:
                records.append(
                    {
                        "repeat": repeat,
                        "repeat_seed": repeat_seed,
                        "fold": fold_index,
                        "role": "meta_validation",
                        "row_position": int(row_positions[row]),
                    }
                )
    frame = pd.DataFrame.from_records(records)
    frame = frame.sort_values(
        ["repeat", "fold", "role", "row_position"], kind="mergesort"
    ).reset_index(drop=True)
    return frame


def _resolve_strategy_weights(
    pool: CompatibleCandidateSet,
    settings: BlendSettings,
    *,
    y_train: np.ndarray,
    oof_train: np.ndarray,
) -> WeightThresholdSelection:
    settings = settings.normalized()
    policy = settings.threshold_policy
    if settings.strategy == "equal":
        weights = equal_weights(pool.n_candidates)
        return score_weights(y_train, oof_train, weights, policy=policy)
    if settings.strategy == "manual":
        weights = parse_manual_weights(
            settings.manual_weights, candidate_ids=pool.candidate_ids
        )
        return score_weights(y_train, oof_train, weights, policy=policy)
    return optimize_weights(
        y_train,
        oof_train,
        seed=settings.seed,
        max_active=settings.max_active_models,
        policy=policy,
        pairwise_grid_step=settings.pairwise_grid_step,
        dirichlet_draws=settings.dirichlet_draws,
    )


def run_blend_evaluation(
    pool: CompatibleCandidateSet,
    settings: BlendSettings,
) -> BlendEvaluationResult:
    """Run leakage-safe meta-CV blend evaluation and fit deployment parameters."""
    settings = settings.normalized()
    y = pool.target
    oof = pool.oof_matrix
    assignments = build_meta_assignments(
        y, folds=settings.folds, repeats=settings.repeats, seed=settings.seed
    )
    assignment_hash = canonical_sha256(
        assignments.astype({"row_position": "int64"}).to_dict(orient="list")
    )

    # Accumulate held-out probabilities per repeat.
    repeat_vectors = {
        repeat: np.full(len(y), np.nan, dtype=np.float64)
        for repeat in range(1, settings.repeats + 1)
    }
    fold_results: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    proposals_total = 0

    for (repeat, fold), group in assignments.groupby(["repeat", "fold"], sort=True):
        train_rows = group.loc[group["role"] == "meta_train", "row_position"].to_numpy(
            dtype=np.int64
        )
        val_rows = group.loc[
            group["role"] == "meta_validation", "row_position"
        ].to_numpy(dtype=np.int64)
        if len(train_rows) == 0 or len(val_rows) == 0:
            raise RuntimeError("Meta split produced an empty train or validation fold.")
        # Leakage guard: no overlap.
        if len(set(train_rows) & set(val_rows)) != 0:
            raise RuntimeError("Meta-train and meta-validation rows overlap.")

        selection = _resolve_strategy_weights(
            pool,
            settings,
            y_train=y[train_rows],
            oof_train=oof[train_rows],
        )
        proposals_total += selection.proposals_evaluated
        val_probs = blend_probabilities(oof[val_rows], selection.weights)
        # Threshold was selected on meta-train only; apply to validation.
        val_preds = (val_probs >= selection.threshold).astype(np.int8)
        metrics = calculate_outer_metrics(
            y[val_rows],
            val_probs,
            val_preds,
            selected_threshold=selection.threshold,
        )
        repeat_vectors[int(repeat)][val_rows] = val_probs
        fold_results.append(
            {
                "repeat": int(repeat),
                "fold": int(fold),
                "weights": selection.weights.tolist(),
                "threshold": selection.threshold,
                "train_rows": int(len(train_rows)),
                "validation_rows": int(len(val_rows)),
                "train_balanced_accuracy": selection.balanced_accuracy,
                "validation_metrics": metrics,
                "proposals_evaluated": selection.proposals_evaluated,
            }
        )
        fold_metric_rows.append(
            {
                "repeat": int(repeat),
                "fold": int(fold),
                **{
                    key: metrics[key]
                    for key in (
                        "balanced_accuracy",
                        "sensitivity",
                        "specificity",
                        "roc_auc",
                        "average_precision",
                        "brier_score",
                        "predicted_positive_rate",
                    )
                },
            }
        )

    # Every row must receive a held-out prediction in each repeat.
    for repeat, vector in repeat_vectors.items():
        if np.isnan(vector).any():
            missing = int(np.isnan(vector).sum())
            raise RuntimeError(
                f"Repeat {repeat} missing held-out predictions for {missing} rows."
            )

    stacked = np.column_stack([repeat_vectors[r] for r in sorted(repeat_vectors)])
    cross_fitted = stacked.mean(axis=1)
    # Normalize policy recorded: mean across meta repeats.

    final_selection = _resolve_strategy_weights(
        pool, settings, y_train=y, oof_train=oof
    )
    final_weights = validate_weights(
        final_selection.weights, n_candidates=pool.n_candidates
    )
    # Final deployment threshold fitted on all OOF with final weights.
    full_blended = blend_probabilities(oof, final_weights)
    final_threshold_result = select_threshold_for_probabilities(
        y, full_blended, policy=settings.threshold_policy
    )
    final_threshold = float(final_threshold_result.threshold)

    # Primary estimate uses cross-fitted probs. Threshold below is selected on
    # the aggregated cross-fitted vector for summary labeling and is recorded
    # separately from fold-specific thresholds and deployment threshold.
    cross_threshold = select_threshold_for_probabilities(
        y, cross_fitted, policy=settings.threshold_policy
    )
    cross_preds_honest = (cross_fitted >= cross_threshold.threshold).astype(np.int8)
    cross_metrics = calculate_outer_metrics(
        y,
        cross_fitted,
        cross_preds_honest,
        selected_threshold=cross_threshold.threshold,
    )
    cross_metrics_label = {
        **cross_metrics,
        "label": "cross_fitted_oof_with_threshold_selected_on_cross_fitted_vector",
        "note": (
            "Primary blend estimate uses held-out meta-CV probabilities. "
            "The threshold here is selected on the aggregated cross-fitted "
            "vector for summary labeling and is recorded separately from "
            "fold-specific thresholds."
        ),
    }

    full_oof_metrics = calculate_outer_metrics(
        y,
        full_blended,
        (full_blended >= final_threshold).astype(np.int8),
        selected_threshold=final_threshold,
    )
    full_oof_descriptive = {
        **full_oof_metrics,
        "label": "full_OOF_descriptive_with_deployment_weights_and_threshold",
        "note": (
            "Not unbiased evidence. Deployment weights/threshold were fitted on "
            "all canonical OOF rows."
        ),
    }

    candidate_metrics = [
        compute_candidate_descriptive_metrics(y, oof[:, index], candidate_id=candidate_id)
        for index, candidate_id in enumerate(pool.candidate_ids)
    ]

    repeat_metric_rows: list[dict[str, Any]] = []
    for repeat in sorted(repeat_vectors):
        probs = repeat_vectors[repeat]
        thr = select_threshold_for_probabilities(
            y, probs, policy=settings.threshold_policy
        )
        metrics = calculate_outer_metrics(
            y,
            probs,
            (probs >= thr.threshold).astype(np.int8),
            selected_threshold=thr.threshold,
        )
        repeat_metric_rows.append({"repeat": repeat, **metrics})

    test_probabilities = blend_probabilities(pool.test_matrix, final_weights)

    return BlendEvaluationResult(
        settings=settings,
        fixed_or_final_weights=final_weights,
        final_threshold=final_threshold,
        cross_fitted_oof=cross_fitted,
        test_probabilities=test_probabilities,
        assignments=assignments,
        assignment_hash=assignment_hash,
        fold_results=fold_results,
        fold_metrics=pd.DataFrame(fold_metric_rows),
        repeat_metrics=pd.DataFrame(repeat_metric_rows),
        cross_fitted_metrics=cross_metrics_label,
        full_oof_descriptive_metrics=full_oof_descriptive,
        candidate_descriptive_metrics=candidate_metrics,
        search_budget={
            "strategy": settings.strategy,
            "proposals_evaluated_total": proposals_total
            + final_selection.proposals_evaluated,
            "dirichlet_draws": settings.dirichlet_draws,
            "pairwise_grid_step": settings.pairwise_grid_step,
            "max_active_models": settings.max_active_models,
            "tie_break_rule": TIE_BREAK_RULE,
        },
        deployment={
            "weights": {
                candidate_id: float(weight)
                for candidate_id, weight in zip(
                    pool.candidate_ids, final_weights, strict=True
                )
            },
            "weight_vector": final_weights.tolist(),
            "threshold": final_threshold,
            "threshold_status": final_threshold_result.status,
            "label": "deployment_parameters_fitted_on_all_canonical_oof",
            "test_probability_rule": "sum_i weight_i * p_test_i",
        },
    )
