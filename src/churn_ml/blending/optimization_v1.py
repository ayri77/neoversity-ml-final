"""Deterministic weight and threshold optimization for probability blends."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np

from src.churn_ml.research_protocol import (
    ThresholdSelectionResult,
    calculate_outer_metrics,
    select_balanced_accuracy_threshold,
)


DEFAULT_THRESHOLD_POLICY: dict[str, Any] = {
    "minimum": 0.01,
    "maximum": 0.99,
    "step": 0.01,
    "maximizer_absolute_tolerance": 1.0e-12,
    "constant_probability_fallback": 0.5,
}

WEIGHT_TOLERANCE = 1.0e-12
ACTIVE_WEIGHT_EPSILON = 1.0e-8

# Tie-breaking for weight vectors (documented in artifacts):
# 1) higher Balanced Accuracy
# 2) higher min(sensitivity, specificity)
# 3) fewer active candidates
# 4) closer to equal weights
# 5) lexicographically smaller weight tuple
TIE_BREAK_RULE = (
    "higher_balanced_accuracy > higher_min_sens_spec > fewer_active > "
    "closer_to_equal > lexical_weights"
)


class BlendOptimizationError(ValueError):
    """Raised when blend weights or thresholds cannot be optimized."""

    def __init__(self, message: str, *, reason_code: str = "blend_optimization") -> None:
        self.reason_code = reason_code
        super().__init__(message)


@dataclass(frozen=True)
class WeightThresholdSelection:
    weights: np.ndarray
    threshold: float
    balanced_accuracy: float
    sensitivity: float
    specificity: float
    status: str
    proposals_evaluated: int


def blend_probabilities(probability_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights_array = validate_weights(weights, n_candidates=probability_matrix.shape[1])
    return probability_matrix @ weights_array


def validate_weights(
    weights: Sequence[float] | np.ndarray,
    *,
    n_candidates: int,
) -> np.ndarray:
    array = np.asarray(weights, dtype=np.float64)
    if array.shape != (n_candidates,):
        raise BlendOptimizationError(
            f"Expected {n_candidates} weights, got shape {array.shape}.",
            reason_code="weight_shape_invalid",
        )
    if not np.isfinite(array).all():
        raise BlendOptimizationError(
            "Weights must be finite.",
            reason_code="weight_nonfinite",
        )
    if (array < -WEIGHT_TOLERANCE).any():
        raise BlendOptimizationError(
            "Negative weights are rejected in v1.",
            reason_code="negative_weights",
        )
    array = np.clip(array, 0.0, None)
    total = float(array.sum())
    if abs(total - 1.0) > 1.0e-9:
        raise BlendOptimizationError(
            f"Weights must sum to 1 within tolerance; got {total}.",
            reason_code="weight_sum_invalid",
        )
    return array


def parse_manual_weights(
    weight_specs: Sequence[str],
    *,
    candidate_ids: Sequence[str],
) -> np.ndarray:
    if not weight_specs:
        raise BlendOptimizationError(
            "Manual strategy requires --weight entries.",
            reason_code="manual_weights_missing",
        )
    mapping: dict[str, float] = {}
    for spec in weight_specs:
        if "=" not in spec:
            raise BlendOptimizationError(
                f"Invalid weight spec {spec!r}; expected candidate_id=weight.",
                reason_code="weight_spec_invalid",
            )
        candidate_id, raw = spec.split("=", 1)
        candidate_id = candidate_id.strip()
        if candidate_id in mapping:
            raise BlendOptimizationError(
                f"Duplicate weight for {candidate_id}.",
                reason_code="duplicate_weight",
            )
        try:
            value = float(raw)
        except ValueError as error:
            raise BlendOptimizationError(
                f"Non-numeric weight for {candidate_id}: {raw!r}",
                reason_code="weight_spec_invalid",
            ) from error
        mapping[candidate_id] = value
    unknown = sorted(set(mapping) - set(candidate_ids))
    if unknown:
        raise BlendOptimizationError(
            f"Unknown weight candidate IDs: {unknown}",
            reason_code="unknown_weight_candidate",
        )
    missing = [item for item in candidate_ids if item not in mapping]
    if missing:
        raise BlendOptimizationError(
            f"Missing weights for candidates: {missing}",
            reason_code="missing_weight_candidate",
        )
    return validate_weights(
        [mapping[item] for item in candidate_ids],
        n_candidates=len(candidate_ids),
    )


def select_threshold_for_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    policy: Mapping[str, Any] | None = None,
) -> ThresholdSelectionResult:
    return select_balanced_accuracy_threshold(
        y_true,
        probabilities,
        dict(policy or DEFAULT_THRESHOLD_POLICY),
    )


def score_weights(
    y_true: np.ndarray,
    probability_matrix: np.ndarray,
    weights: np.ndarray,
    *,
    policy: Mapping[str, Any] | None = None,
) -> WeightThresholdSelection:
    blended = blend_probabilities(probability_matrix, weights)
    threshold = select_threshold_for_probabilities(y_true, blended, policy=policy)
    preds = (blended >= threshold.threshold).astype(np.int8)
    metrics = calculate_outer_metrics(
        y_true, blended, preds, selected_threshold=threshold.threshold
    )
    return WeightThresholdSelection(
        weights=validate_weights(weights, n_candidates=probability_matrix.shape[1]),
        threshold=float(threshold.threshold),
        balanced_accuracy=float(metrics["balanced_accuracy"]),
        sensitivity=float(metrics["sensitivity"]),
        specificity=float(metrics["specificity"]),
        status=str(threshold.status),
        proposals_evaluated=1,
    )


def _active_count(weights: np.ndarray) -> int:
    return int(np.sum(weights > ACTIVE_WEIGHT_EPSILON))


def _equal_distance(weights: np.ndarray) -> float:
    n = len(weights)
    equal = np.full(n, 1.0 / n, dtype=np.float64)
    return float(np.sum((weights - equal) ** 2))


def _selection_key(selection: WeightThresholdSelection) -> tuple[Any, ...]:
    return (
        -selection.balanced_accuracy,
        -min(selection.sensitivity, selection.specificity),
        _active_count(selection.weights),
        _equal_distance(selection.weights),
        tuple(np.round(selection.weights, 12).tolist()),
    )


def validate_max_active(max_active: int | None, *, n_candidates: int) -> int | None:
    if max_active is None:
        return None
    if max_active < 1:
        raise BlendOptimizationError(
            "max_active_models must be >= 1 when provided.",
            reason_code="max_active_invalid",
        )
    if max_active > n_candidates:
        raise BlendOptimizationError(
            f"max_active_models ({max_active}) exceeds selected candidate count "
            f"({n_candidates}).",
            reason_code="max_active_exceeds_candidates",
        )
    return int(max_active)


def softmax_from_latents(latents: np.ndarray) -> np.ndarray:
    """Numerically stable softmax mapping latents onto the probability simplex."""
    values = np.asarray(latents, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise BlendOptimizationError(
            "Latents must be a non-empty one-dimensional array.",
            reason_code="latent_shape_invalid",
        )
    if not np.isfinite(values).all():
        raise BlendOptimizationError(
            "Latents must be finite.",
            reason_code="latent_nonfinite",
        )
    shifted = values - float(np.max(values))
    exp = np.exp(shifted)
    total = float(exp.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise BlendOptimizationError(
            "Softmax normalization failed.",
            reason_code="softmax_failed",
        )
    return validate_weights(exp / total, n_candidates=len(values))


def enforce_max_active(weights: np.ndarray, max_active: int | None) -> np.ndarray:
    array = validate_weights(weights, n_candidates=len(weights))
    limit = validate_max_active(max_active, n_candidates=len(array))
    if limit is None:
        return array
    if _active_count(array) <= limit:
        return array
    # Prefer larger weight, break ties by canonical (lower) index order.
    order = sorted(
        range(len(array)),
        key=lambda index: (-float(array[index]), index),
    )
    keep = order[:limit]
    projected = np.zeros_like(array)
    projected[keep] = array[keep]
    total = float(projected.sum())
    if total <= 0:
        projected[keep[0]] = 1.0
    else:
        projected = projected / total
    return validate_weights(projected, n_candidates=len(array))


def generate_weight_proposals(
    n_candidates: int,
    *,
    seed: int,
    max_active: int | None,
    pairwise_grid_step: float = 0.1,
    dirichlet_draws: int = 32,
) -> list[np.ndarray]:
    """Deterministic bounded proposal set on the probability simplex."""
    if n_candidates < 1:
        raise BlendOptimizationError("n_candidates must be positive.")
    rng = np.random.default_rng(seed)
    proposals: list[np.ndarray] = []

    # Equal-weight seed.
    proposals.append(np.full(n_candidates, 1.0 / n_candidates, dtype=np.float64))
    # Vertices.
    for index in range(n_candidates):
        vertex = np.zeros(n_candidates, dtype=np.float64)
        vertex[index] = 1.0
        proposals.append(vertex)

    active_limit = (
        n_candidates
        if max_active is None
        else validate_max_active(max_active, n_candidates=n_candidates)
    )
    assert active_limit is not None or max_active is None
    active_limit = n_candidates if active_limit is None else active_limit
    # Pairwise grids on all pairs (or active subsets of size 2).
    if active_limit >= 2:
        for i, j in combinations(range(n_candidates), 2):
            steps = int(round(1.0 / pairwise_grid_step))
            for tick in range(steps + 1):
                w = tick * pairwise_grid_step
                weights = np.zeros(n_candidates, dtype=np.float64)
                weights[i] = w
                weights[j] = 1.0 - w
                proposals.append(weights)

    # Equal weights on subsets up to min(active_limit, 3) to bound combinatorial cost.
    subset_limit = min(active_limit, 3)
    for size in range(1, subset_limit + 1):
        for subset in combinations(range(n_candidates), size):
            weights = np.zeros(n_candidates, dtype=np.float64)
            weights[list(subset)] = 1.0 / size
            proposals.append(weights)

    # Seeded Dirichlet proposals, then optionally projected to max_active.
    alpha = np.ones(n_candidates, dtype=np.float64)
    for _ in range(dirichlet_draws):
        draw = rng.dirichlet(alpha)
        proposals.append(enforce_max_active(draw, max_active))

    # Deduplicate with stable rounding.
    unique: dict[tuple[float, ...], np.ndarray] = {}
    for proposal in proposals:
        projected = enforce_max_active(proposal, max_active)
        key = tuple(np.round(projected, 10).tolist())
        unique[key] = projected
    return list(unique.values())


def optimize_weights(
    y_true: np.ndarray,
    probability_matrix: np.ndarray,
    *,
    seed: int,
    max_active: int | None = None,
    policy: Mapping[str, Any] | None = None,
    pairwise_grid_step: float = 0.1,
    dirichlet_draws: int = 32,
) -> WeightThresholdSelection:
    """Search non-negative simplex weights maximizing Balanced Accuracy."""
    proposals = generate_weight_proposals(
        probability_matrix.shape[1],
        seed=seed,
        max_active=max_active,
        pairwise_grid_step=pairwise_grid_step,
        dirichlet_draws=dirichlet_draws,
    )
    best: WeightThresholdSelection | None = None
    for proposal in proposals:
        scored = score_weights(
            y_true, probability_matrix, proposal, policy=policy
        )
        if best is None or _selection_key(scored) < _selection_key(best):
            best = scored
    if best is None:
        raise BlendOptimizationError("Weight search produced no proposals.")

    # Coordinate refinement around the best point.
    refined = _coordinate_refine(
        y_true,
        probability_matrix,
        best.weights,
        seed=seed,
        max_active=max_active,
        policy=policy,
    )
    if _selection_key(refined) < _selection_key(best):
        best = refined
    return WeightThresholdSelection(
        weights=best.weights,
        threshold=best.threshold,
        balanced_accuracy=best.balanced_accuracy,
        sensitivity=best.sensitivity,
        specificity=best.specificity,
        status=best.status,
        proposals_evaluated=len(proposals) + refined.proposals_evaluated,
    )


def _coordinate_refine(
    y_true: np.ndarray,
    probability_matrix: np.ndarray,
    weights: np.ndarray,
    *,
    seed: int,
    max_active: int | None,
    policy: Mapping[str, Any] | None,
    steps: tuple[float, ...] = (0.05, 0.02),
    max_passes: int = 2,
) -> WeightThresholdSelection:
    current = score_weights(y_true, probability_matrix, weights, policy=policy)
    evaluated = current.proposals_evaluated
    n = len(weights)
    rng = np.random.default_rng(seed + 17)
    order = np.arange(n)
    rng.shuffle(order)
    for step in steps:
        for _pass in range(max_passes):
            improved = False
            for i in order:
                for j in order:
                    if i == j:
                        continue
                    trial = current.weights.copy()
                    delta = min(step, trial[i])
                    if delta <= 0:
                        continue
                    trial[i] -= delta
                    trial[j] += delta
                    trial = enforce_max_active(trial, max_active)
                    scored = score_weights(
                        y_true, probability_matrix, trial, policy=policy
                    )
                    evaluated += 1
                    if _selection_key(scored) < _selection_key(current):
                        current = scored
                        improved = True
            if not improved:
                break
    return WeightThresholdSelection(
        weights=current.weights,
        threshold=current.threshold,
        balanced_accuracy=current.balanced_accuracy,
        sensitivity=current.sensitivity,
        specificity=current.specificity,
        status=current.status,
        proposals_evaluated=evaluated,
    )


def equal_weights(n_candidates: int) -> np.ndarray:
    if n_candidates < 1:
        raise BlendOptimizationError("n_candidates must be positive.")
    return np.full(n_candidates, 1.0 / n_candidates, dtype=np.float64)
