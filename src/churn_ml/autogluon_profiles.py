"""Versioned, non-user-extensible AutoGluon benchmark profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SUPPORTED_AUTOGLUON_VERSION = "1.5.0"


@dataclass(frozen=True)
class AutoGluonProfile:
    """A reviewed mapping to one AutoGluon 1.5.0 portfolio subset."""

    profile_id: str
    description: str
    portfolio: str
    included_model_types: tuple[str, ...] | None
    excluded_model_types: tuple[str, ...]
    force_num_gpus: int | None


_PROFILES = {
    "realtabpfn_only_v1": AutoGluonProfile(
        profile_id="realtabpfn_only_v1",
        description=(
            "All RealTabPFN-v2 configurations in the AutoGluon 1.5.0 "
            "zeroshot_2025_12_18_gpu portfolio."
        ),
        portfolio="zeroshot_2025_12_18_gpu",
        included_model_types=("REALTABPFN-V2",),
        excluded_model_types=(),
        force_num_gpus=1,
    ),
    "catboost_only_cpu_v1": AutoGluonProfile(
        profile_id="catboost_only_cpu_v1",
        description="The AutoGluon 1.5.0 CPU portfolio CatBoost family, forced to CPU.",
        portfolio="zeroshot_2025_12_18_cpu",
        included_model_types=("CAT",),
        excluded_model_types=(),
        force_num_gpus=0,
    ),
    "lightgbmprep_only_cpu_v1": AutoGluonProfile(
        profile_id="lightgbmprep_only_cpu_v1",
        description="The AutoGluon 1.5.0 CPU portfolio GBM_PREP family, forced to CPU.",
        portfolio="zeroshot_2025_12_18_cpu",
        included_model_types=("GBM_PREP",),
        excluded_model_types=(),
        force_num_gpus=0,
    ),
    "extreme_seqmem_v1": AutoGluonProfile(
        profile_id="extreme_seqmem_v1",
        description=(
            "Diagnostic extreme portfolio with sequential fitting; TABDPT, TABICL, "
            "and MITRA are excluded."
        ),
        portfolio="zeroshot_2025_12_18_gpu",
        included_model_types=None,
        excluded_model_types=("TABDPT", "TABICL", "MITRA"),
        force_num_gpus=None,
    ),
}


def get_profile(profile_id: str) -> AutoGluonProfile:
    """Return a registered profile or fail with the available IDs."""
    try:
        return _PROFILES[profile_id]
    except KeyError as error:
        available = ", ".join(sorted(_PROFILES))
        raise ValueError(
            f"Unknown AutoGluon profile {profile_id!r}. Available profiles: {available}"
        ) from error


def list_profiles() -> tuple[AutoGluonProfile, ...]:
    """Return profiles in stable ID order."""
    return tuple(_PROFILES[key] for key in sorted(_PROFILES))


def resolve_profile_hyperparameters(profile: AutoGluonProfile) -> dict[str, Any]:
    """Resolve and filter an AutoGluon portfolio inside the worker process only."""
    from autogluon.tabular.configs.hyperparameter_configs import (  # type: ignore[import-not-found]
        get_hyperparameter_config,
    )

    portfolio: dict[str, Any] = get_hyperparameter_config(profile.portfolio)
    if profile.included_model_types is not None:
        missing = sorted(set(profile.included_model_types).difference(portfolio))
        if missing:
            raise RuntimeError(
                "Profile model family is unavailable in AutoGluon "
                f"{SUPPORTED_AUTOGLUON_VERSION}: {', '.join(missing)}"
            )
        portfolio = {key: portfolio[key] for key in profile.included_model_types}
    else:
        portfolio = {
            key: value
            for key, value in portfolio.items()
            if key not in profile.excluded_model_types
        }

    if not portfolio:
        raise RuntimeError(
            f"Profile {profile.profile_id!r} resolved to no model families."
        )

    for configurations in portfolio.values():
        items = configurations if isinstance(configurations, list) else [configurations]
        for item in items:
            if not isinstance(item, dict):
                continue
            ensemble = item.setdefault("ag_args_ensemble", {})
            ensemble["fold_fitting_strategy"] = "sequential_local"
            if profile.force_num_gpus is not None:
                fit_args = item.setdefault("ag_args_fit", {})
                fit_args["num_gpus"] = profile.force_num_gpus
                if profile.force_num_gpus == 0:
                    item.pop("task_type", None)
                    if profile.profile_id == "catboost_only_cpu_v1":
                        item["task_type"] = "CPU"
    return portfolio


def profile_summary(profile: AutoGluonProfile) -> dict[str, Any]:
    """Return a portable profile identity payload."""
    return {
        "profile_id": profile.profile_id,
        "autogluon_version": SUPPORTED_AUTOGLUON_VERSION,
        "description": profile.description,
        "portfolio": profile.portfolio,
        "included_model_types": (
            list(profile.included_model_types)
            if profile.included_model_types is not None
            else None
        ),
        "excluded_model_types": list(profile.excluded_model_types),
        "force_num_gpus": profile.force_num_gpus,
    }
