"""Descriptive diversity analysis for prediction-candidate pools."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.churn_ml.blending.compatibility_v1 import CompatibleCandidateSet
from src.churn_ml.blending.evaluation_v1 import compute_candidate_descriptive_metrics
from src.churn_ml.blending.optimization_v1 import (
    DEFAULT_THRESHOLD_POLICY,
    select_threshold_for_probabilities,
)
from src.churn_ml.research_protocol import calculate_outer_metrics


def analyze_diversity(pool: CompatibleCandidateSet) -> dict[str, Any]:
    """Compute pairwise diversity and descriptive candidate metrics."""
    ids = list(pool.candidate_ids)
    n = pool.n_candidates
    y = pool.target
    oof = pool.oof_matrix

    candidate_metrics = []
    thresholds: list[float] = []
    for index, candidate_id in enumerate(ids):
        metrics = compute_candidate_descriptive_metrics(
            y, oof[:, index], candidate_id=candidate_id
        )
        candidate_metrics.append(metrics)
        thresholds.append(float(metrics["descriptive_oof_optimal_threshold"]))

    pearson = np.corrcoef(oof, rowvar=False)
    spearman = np.ones((n, n), dtype=np.float64)
    mad = np.zeros((n, n), dtype=np.float64)
    rmse = np.zeros((n, n), dtype=np.float64)
    disagreement = np.zeros((n, n), dtype=np.float64)
    joint_error = np.zeros((n, n), dtype=np.float64)
    error_overlap = np.zeros((n, n), dtype=np.float64)
    pos_disagree = np.zeros((n, n), dtype=np.float64)
    neg_disagree = np.zeros((n, n), dtype=np.float64)

    predictions = [
        (oof[:, index] >= thresholds[index]).astype(np.int8) for index in range(n)
    ]
    errors = [predictions[index] != y for index in range(n)]

    pairwise_rows: list[dict[str, Any]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                spearman[i, j] = 1.0
                continue
            corr, _ = spearmanr(oof[:, i], oof[:, j])
            spearman[i, j] = float(corr)
            diff = oof[:, i] - oof[:, j]
            mad[i, j] = float(np.mean(np.abs(diff)))
            rmse[i, j] = float(np.sqrt(np.mean(diff**2)))
            disagreement[i, j] = float(np.mean(predictions[i] != predictions[j]))
            both_wrong = errors[i] & errors[j]
            either_wrong = errors[i] | errors[j]
            joint_error[i, j] = float(np.mean(both_wrong))
            error_overlap[i, j] = (
                float(both_wrong.sum() / either_wrong.sum())
                if either_wrong.any()
                else 0.0
            )
            positive_mask = y == 1
            negative_mask = y == 0
            pos_disagree[i, j] = (
                float(np.mean(predictions[i][positive_mask] != predictions[j][positive_mask]))
                if positive_mask.any()
                else 0.0
            )
            neg_disagree[i, j] = (
                float(np.mean(predictions[i][negative_mask] != predictions[j][negative_mask]))
                if negative_mask.any()
                else 0.0
            )

    for i in range(n):
        for j in range(i + 1, n):
            equal = np.full(n, 0.0)
            equal[i] = 0.5
            equal[j] = 0.5
            blended = oof @ equal
            threshold = select_threshold_for_probabilities(
                y, blended, policy=DEFAULT_THRESHOLD_POLICY
            )
            preds = (blended >= threshold.threshold).astype(np.int8)
            metrics = calculate_outer_metrics(
                y, blended, preds, selected_threshold=threshold.threshold
            )
            pairwise_rows.append(
                {
                    "candidate_a": ids[i],
                    "candidate_b": ids[j],
                    "pearson": float(pearson[i, j]),
                    "spearman": float(spearman[i, j]),
                    "mean_abs_diff": float(mad[i, j]),
                    "rmse": float(rmse[i, j]),
                    "prediction_disagreement_rate": float(disagreement[i, j]),
                    "joint_error_rate": float(joint_error[i, j]),
                    "error_overlap": float(error_overlap[i, j]),
                    "positive_class_disagreement": float(pos_disagree[i, j]),
                    "negative_class_disagreement": float(neg_disagree[i, j]),
                    "equal_weight_descriptive_balanced_accuracy": float(
                        metrics["balanced_accuracy"]
                    ),
                    "equal_weight_descriptive_threshold": float(threshold.threshold),
                    "threshold_basis": (
                        "each candidate descriptive OOF-optimal threshold; "
                        "equal-weight blend uses descriptive full-OOF threshold"
                    ),
                }
            )

    diversity_matrix = pd.DataFrame(pearson, index=ids, columns=ids)
    return {
        "candidate_metrics": candidate_metrics,
        "diversity_matrix": diversity_matrix,
        "pairwise_analysis": pd.DataFrame(pairwise_rows),
        "matrices": {
            "pearson": diversity_matrix,
            "spearman": pd.DataFrame(spearman, index=ids, columns=ids),
            "mean_abs_diff": pd.DataFrame(mad, index=ids, columns=ids),
            "rmse": pd.DataFrame(rmse, index=ids, columns=ids),
            "prediction_disagreement_rate": pd.DataFrame(
                disagreement, index=ids, columns=ids
            ),
            "joint_error_rate": pd.DataFrame(joint_error, index=ids, columns=ids),
            "error_overlap": pd.DataFrame(error_overlap, index=ids, columns=ids),
            "positive_class_disagreement": pd.DataFrame(
                pos_disagree, index=ids, columns=ids
            ),
            "negative_class_disagreement": pd.DataFrame(
                neg_disagree, index=ids, columns=ids
            ),
        },
        "notes": {
            "threshold_disagreement": (
                "Threshold-based disagreement uses each candidate's descriptive "
                "full-OOF optimal threshold and is not unbiased evidence."
            ),
            "no_automatic_discard": (
                "Candidates are never discarded automatically based on correlation."
            ),
        },
    }
