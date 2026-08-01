"""Lazy Optuna backend for optimized probability blends.

Optuna is imported only when this optimizer runs. Equal/manual/native paths and
ordinary CLI commands must never import this module's Optuna dependency path
except through an explicit optimized+optuna request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from src.churn_ml.blending.optimization_v1 import (
    BlendOptimizationError,
    WeightThresholdSelection,
    enforce_max_active,
    score_weights,
    softmax_from_latents,
    validate_max_active,
    validate_weights,
)


LATENT_LOW = -6.0
LATENT_HIGH = 6.0
OPTIMIZER_BACKEND = "optuna"
SAMPLER_TYPE = "TPESampler"
PARAMETERIZATION = "bounded_latent_softmax_z_in_[-6,6]"


@dataclass(frozen=True)
class OptunaStudyRecord:
    study_role: str
    repeat: int | None
    fold: int | None
    resolved_seed: int
    optuna_version: str
    sampler_type: str
    sampler_settings: dict[str, Any]
    requested_trials: int
    completed_trials: int
    timeout_seconds: float | None
    best_trial_number: int
    best_objective: float
    best_latents: list[float]
    resolved_weights: list[float]
    resolved_threshold: float
    trial_state_counts: dict[str, int]
    trials: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    status: str = "completed"


StudyRunner = Callable[..., OptunaStudyRecord]


def lazy_import_optuna() -> Any:
    try:
        import optuna
    except ImportError as error:
        raise BlendOptimizationError(
            "Optuna is not available in this environment.",
            reason_code="optuna_unavailable",
        ) from error
    return optuna


def derive_study_seed(
    *,
    global_seed: int,
    optuna_seed: int,
    study_role: str,
    repeat: int | None,
    fold: int | None,
) -> int:
    """Deterministic 31-bit seed from blend/Optuna context."""
    role_code = {
        "meta_fold": 1,
        "final_deployment": 2,
    }.get(study_role)
    if role_code is None:
        raise BlendOptimizationError(
            f"Unknown study role {study_role!r}.",
            reason_code="study_role_invalid",
        )
    repeat_value = 0 if repeat is None else int(repeat)
    fold_value = 0 if fold is None else int(fold)
    mixed = (
        (int(global_seed) & 0xFFFFFFFF)
        ^ ((int(optuna_seed) & 0xFFFFFFFF) << 1)
        ^ (role_code << 20)
        ^ (repeat_value << 12)
        ^ (fold_value << 4)
    )
    return int(mixed % (2**31 - 1))


def optimize_weights_optuna(
    y_true: np.ndarray,
    probability_matrix: np.ndarray,
    *,
    global_seed: int,
    optuna_seed: int,
    n_trials: int,
    timeout_seconds: float | None,
    max_active: int | None,
    policy: Mapping[str, Any] | None,
    study_role: str,
    repeat: int | None,
    fold: int | None,
    study_runner: StudyRunner | None = None,
) -> tuple[WeightThresholdSelection, OptunaStudyRecord]:
    """Run one deterministic single-process Optuna study on meta-train only."""
    if n_trials < 1:
        raise BlendOptimizationError(
            "optuna_trials must be a positive integer.",
            reason_code="optuna_trials_invalid",
        )
    if timeout_seconds is not None and (
        not np.isfinite(timeout_seconds) or float(timeout_seconds) <= 0.0
    ):
        raise BlendOptimizationError(
            "optuna_timeout_seconds must be a positive finite number when set.",
            reason_code="optuna_timeout_invalid",
        )
    n_candidates = int(probability_matrix.shape[1])
    validate_max_active(max_active, n_candidates=n_candidates)
    resolved_seed = derive_study_seed(
        global_seed=global_seed,
        optuna_seed=optuna_seed,
        study_role=study_role,
        repeat=repeat,
        fold=fold,
    )
    runner = study_runner or _default_study_runner
    record = runner(
        y_true=y_true,
        probability_matrix=probability_matrix,
        resolved_seed=resolved_seed,
        n_trials=int(n_trials),
        timeout_seconds=None if timeout_seconds is None else float(timeout_seconds),
        max_active=max_active,
        policy=policy,
        study_role=study_role,
        repeat=repeat,
        fold=fold,
    )
    if record.status != "completed" or record.completed_trials < 1:
        raise BlendOptimizationError(
            f"Optuna study incomplete for role={study_role!r}: status={record.status}.",
            reason_code="optuna_study_incomplete",
        )
    weights = validate_weights(record.resolved_weights, n_candidates=n_candidates)
    scored = score_weights(y_true, probability_matrix, weights, policy=policy)
    selection = WeightThresholdSelection(
        weights=scored.weights,
        threshold=float(record.resolved_threshold),
        balanced_accuracy=float(record.best_objective),
        sensitivity=scored.sensitivity,
        specificity=scored.specificity,
        status=scored.status,
        proposals_evaluated=int(record.completed_trials),
    )
    return selection, record


def _default_study_runner(
    *,
    y_true: np.ndarray,
    probability_matrix: np.ndarray,
    resolved_seed: int,
    n_trials: int,
    timeout_seconds: float | None,
    max_active: int | None,
    policy: Mapping[str, Any] | None,
    study_role: str,
    repeat: int | None,
    fold: int | None,
) -> OptunaStudyRecord:
    optuna = lazy_import_optuna()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=int(resolved_seed))
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )
    n_candidates = int(probability_matrix.shape[1])
    trial_rows: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        latents = np.asarray(
            [
                trial.suggest_float(f"z_{index}", LATENT_LOW, LATENT_HIGH)
                for index in range(n_candidates)
            ],
            dtype=np.float64,
        )
        weights = enforce_max_active(softmax_from_latents(latents), max_active)
        # Objective uses only the arrays provided to this study (meta-train).
        scored = score_weights(y_true, probability_matrix, weights, policy=policy)
        trial.set_user_attr("weights", scored.weights.tolist())
        trial.set_user_attr("threshold", scored.threshold)
        trial.set_user_attr("latents", latents.tolist())
        return float(scored.balanced_accuracy)

    study.optimize(
        objective,
        n_trials=int(n_trials),
        timeout=timeout_seconds,
        n_jobs=1,
        show_progress_bar=False,
    )
    state_counts: dict[str, int] = {}
    for trial in study.trials:
        state_name = str(trial.state).split(".")[-1]
        state_counts[state_name] = state_counts.get(state_name, 0) + 1
        attrs = trial.user_attrs
        trial_rows.append(
            {
                "study_role": study_role,
                "repeat": repeat,
                "fold": fold,
                "trial_number": int(trial.number),
                "state": state_name,
                "objective": (
                    float(trial.value) if trial.value is not None else None
                ),
                "latents": attrs.get("latents"),
                "weights": attrs.get("weights"),
                "threshold": attrs.get("threshold"),
                "duration_seconds": (
                    float(trial.duration.total_seconds())
                    if trial.duration is not None
                    else None
                ),
            }
        )
    complete = [trial for trial in study.trials if trial.value is not None]
    if not complete or study.best_trial is None:
        return OptunaStudyRecord(
            study_role=study_role,
            repeat=repeat,
            fold=fold,
            resolved_seed=resolved_seed,
            optuna_version=str(optuna.__version__),
            sampler_type=SAMPLER_TYPE,
            sampler_settings={"seed": int(resolved_seed)},
            requested_trials=int(n_trials),
            completed_trials=0,
            timeout_seconds=timeout_seconds,
            best_trial_number=-1,
            best_objective=float("nan"),
            best_latents=[],
            resolved_weights=[],
            resolved_threshold=float("nan"),
            trial_state_counts=state_counts,
            trials=tuple(trial_rows),
            status="failed",
        )
    best = study.best_trial
    return OptunaStudyRecord(
        study_role=study_role,
        repeat=repeat,
        fold=fold,
        resolved_seed=resolved_seed,
        optuna_version=str(optuna.__version__),
        sampler_type=SAMPLER_TYPE,
        sampler_settings={"seed": int(resolved_seed)},
        requested_trials=int(n_trials),
        completed_trials=len(complete),
        timeout_seconds=timeout_seconds,
        best_trial_number=int(best.number),
        best_objective=float(best.value),
        best_latents=list(best.user_attrs.get("latents") or []),
        resolved_weights=list(best.user_attrs.get("weights") or []),
        resolved_threshold=float(best.user_attrs.get("threshold")),
        trial_state_counts=state_counts,
        trials=tuple(trial_rows),
        status="completed",
    )


def study_record_to_dict(record: OptunaStudyRecord) -> dict[str, Any]:
    return {
        "study_role": record.study_role,
        "repeat": record.repeat,
        "fold": record.fold,
        "resolved_seed": record.resolved_seed,
        "optuna_version": record.optuna_version,
        "sampler_type": record.sampler_type,
        "sampler_settings": dict(record.sampler_settings),
        "requested_trials": record.requested_trials,
        "completed_trials": record.completed_trials,
        "timeout_seconds": record.timeout_seconds,
        "best_trial_number": record.best_trial_number,
        "best_objective": record.best_objective,
        "best_latents": list(record.best_latents),
        "resolved_weights": list(record.resolved_weights),
        "resolved_threshold": record.resolved_threshold,
        "trial_state_counts": dict(record.trial_state_counts),
        "status": record.status,
        "parameterization": PARAMETERIZATION,
    }
