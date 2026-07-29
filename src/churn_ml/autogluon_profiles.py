"""Versioned, non-user-extensible AutoGluon benchmark profiles."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any


SUPPORTED_AUTOGLUON_VERSION = "1.5.0"
CPU_ONLY_FAMILIES = frozenset({"GBM", "GBM_PREP", "CAT"})
GPU_SUPPORTED_FAMILIES = frozenset({"REALTABPFN-V2", "TABM"})


@dataclass(frozen=True)
class AutoGluonProfile:
    """A reviewed mapping to one AutoGluon 1.5.0 portfolio subset."""

    profile_id: str
    description: str
    portfolio: str
    included_model_types: tuple[str, ...] | None
    excluded_model_types: tuple[str, ...]
    minimum_gpu_budget: int


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
        minimum_gpu_budget=1,
    ),
    "tabm_only_gpu_v1": AutoGluonProfile(
        profile_id="tabm_only_gpu_v1",
        description=(
            "All TabM configurations in the AutoGluon 1.5.0 "
            "zeroshot_2025_12_18_gpu portfolio."
        ),
        portfolio="zeroshot_2025_12_18_gpu",
        included_model_types=("TABM",),
        excluded_model_types=(),
        minimum_gpu_budget=1,
    ),
    "catboost_only_cpu_v1": AutoGluonProfile(
        profile_id="catboost_only_cpu_v1",
        description="The AutoGluon 1.5.0 CPU portfolio CatBoost family, forced to CPU.",
        portfolio="zeroshot_2025_12_18_cpu",
        included_model_types=("CAT",),
        excluded_model_types=(),
        minimum_gpu_budget=0,
    ),
    "lightgbmprep_only_cpu_v1": AutoGluonProfile(
        profile_id="lightgbmprep_only_cpu_v1",
        description="The AutoGluon 1.5.0 CPU portfolio GBM_PREP family, forced to CPU.",
        portfolio="zeroshot_2025_12_18_cpu",
        included_model_types=("GBM_PREP",),
        excluded_model_types=(),
        minimum_gpu_budget=0,
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
        minimum_gpu_budget=1,
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


def profile_summary(
    profile: AutoGluonProfile,
    seed: int,
    gpu_budget: int,
) -> dict[str, Any]:
    """Return the exact portable profile identity, including seed and resources."""
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
        "seed": seed,
        "gpu_budget": gpu_budget,
        "minimum_gpu_budget": profile.minimum_gpu_budget,
        "resource_policy": {
            "cpu_only_families": sorted(CPU_ONLY_FAMILIES),
            "gpu_supported_families": sorted(GPU_SUPPORTED_FAMILIES),
            "mixed_profile_top_level_num_gpus": "omitted",
            "fold_fitting_strategy": "sequential_local",
        },
    }


def profile_sha256(profile: AutoGluonProfile, seed: int, gpu_budget: int) -> str:
    """Hash the portable profile identity without local paths."""
    encoded = json.dumps(
        profile_summary(profile, seed, gpu_budget),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_profile_hyperparameters(
    profile: AutoGluonProfile,
    *,
    seed: int,
    gpu_budget: int,
) -> dict[str, Any]:
    """Resolve a portfolio with explicit seed and per-family resources in the worker."""
    from autogluon.tabular.configs.hyperparameter_configs import (  # type: ignore[import-not-found]
        get_hyperparameter_config,
    )

    if gpu_budget < profile.minimum_gpu_budget:
        raise RuntimeError(
            f"Profile {profile.profile_id!r} requires GPU budget "
            f">={profile.minimum_gpu_budget}; found {gpu_budget}."
        )
    portfolio: dict[str, Any] = copy.deepcopy(
        get_hyperparameter_config(profile.portfolio)
    )
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

    unsupported = sorted(
        set(portfolio).difference(CPU_ONLY_FAMILIES | GPU_SUPPORTED_FAMILIES)
    )
    if unsupported:
        raise RuntimeError(
            "Profile contains families without an explicit resource policy: "
            + ", ".join(unsupported)
        )
    if not portfolio:
        raise RuntimeError(
            f"Profile {profile.profile_id!r} resolved to no model families."
        )

    for family, configurations in portfolio.items():
        items = configurations if isinstance(configurations, list) else [configurations]
        for item in items:
            if not isinstance(item, dict):
                raise RuntimeError(
                    f"Profile family {family!r} contains a non-mapping configuration."
                )
            ensemble = item.setdefault("ag_args_ensemble", {})
            ensemble["fold_fitting_strategy"] = "sequential_local"
            ensemble["model_random_seed"] = seed
            ensemble["vary_seed_across_folds"] = False
            fit_args = item.setdefault("ag_args_fit", {})
            fit_args["num_gpus"] = (
                1 if family in GPU_SUPPORTED_FAMILIES and gpu_budget >= 1 else 0
            )
            if family == "CAT":
                item["task_type"] = "CPU"
    return portfolio


def effective_family_resources(
    hyperparameters: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Summarize and verify effective per-family GPU and CatBoost settings."""
    result: dict[str, dict[str, Any]] = {}
    for family, configurations in hyperparameters.items():
        items = configurations if isinstance(configurations, list) else [configurations]
        gpu_values: list[int] = []
        task_types: list[str] = []
        for item in items:
            fit_args = item.get("ag_args_fit", {})
            gpu = fit_args.get("num_gpus")
            if type(gpu) is not int:
                raise RuntimeError(
                    f"Family {family!r} lacks an exact integer num_gpus."
                )
            gpu_values.append(gpu)
            if "task_type" in item:
                task_types.append(str(item["task_type"]))
        if family in CPU_ONLY_FAMILIES and any(value != 0 for value in gpu_values):
            raise RuntimeError(f"CPU family {family!r} resolved with GPU resources.")
        if family == "CAT" and (
            len(task_types) != len(items) or any(value != "CPU" for value in task_types)
        ):
            raise RuntimeError("CatBoost must resolve with task_type='CPU'.")
        result[family] = {
            "configuration_count": len(items),
            "num_gpus": sorted(set(gpu_values)),
            "task_types": sorted(set(task_types)),
        }
    return result
