"""Versioned, non-user-extensible AutoGluon benchmark profiles."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal


SUPPORTED_AUTOGLUON_VERSION = "1.5.0"
CPU_ONLY_FAMILIES = frozenset({"GBM", "GBM_PREP", "CAT"})
GPU_SUPPORTED_FAMILIES = frozenset({"REALTABPFN-V2", "TABM"})
COMPOSITE_PORTFOLIO_MARKER = "composite"


@dataclass(frozen=True)
class CompositeConfigSelection:
    """One statically selected configuration from a built-in AutoGluon portfolio."""

    portfolio: str
    family: str
    priority: int
    num_gpus: int
    name_suffix: str | None = None
    index: int | None = None

    def __post_init__(self) -> None:
        has_suffix = self.name_suffix is not None
        has_index = self.index is not None
        if has_suffix == has_index:
            raise ValueError(
                "CompositeConfigSelection requires exactly one of "
                "name_suffix or index."
            )
        if self.num_gpus < 0:
            raise ValueError("CompositeConfigSelection.num_gpus must not be negative.")

    @property
    def selector_kind(self) -> Literal["name_suffix", "index"]:
        return "name_suffix" if self.name_suffix is not None else "index"

    def selector_payload(self) -> dict[str, Any]:
        if self.name_suffix is not None:
            return {"kind": "name_suffix", "name_suffix": self.name_suffix}
        return {"kind": "index", "index": self.index}

    def selector_key(self) -> tuple[Any, ...]:
        if self.name_suffix is not None:
            return ("name_suffix", self.name_suffix)
        return ("index", self.index)


@dataclass(frozen=True)
class AutoGluonProfile:
    """A reviewed mapping to one AutoGluon 1.5.0 portfolio subset."""

    profile_id: str
    description: str
    portfolio: str
    included_model_types: tuple[str, ...] | None
    excluded_model_types: tuple[str, ...]
    minimum_gpu_budget: int
    composite_selections: tuple[CompositeConfigSelection, ...] | None = None


def _selection(
    *,
    portfolio: str,
    family: str,
    priority: int,
    num_gpus: int,
    name_suffix: str | None = None,
    index: int | None = None,
) -> CompositeConfigSelection:
    return CompositeConfigSelection(
        portfolio=portfolio,
        family=family,
        priority=priority,
        num_gpus=num_gpus,
        name_suffix=name_suffix,
        index=index,
    )


_FOCUSED_HYBRID_SELECTIONS = (
    _selection(
        portfolio="zeroshot_2025_12_18_gpu",
        family="REALTABPFN-V2",
        name_suffix="_r11",
        priority=100,
        num_gpus=1,
    ),
    _selection(
        portfolio="zeroshot_2025_12_18_gpu",
        family="GBM_PREP",
        name_suffix="_r31",
        priority=90,
        num_gpus=0,
    ),
    _selection(
        portfolio="zeroshot_2025_12_18_gpu",
        family="GBM_PREP",
        name_suffix="_r41",
        priority=89,
        num_gpus=0,
    ),
    _selection(
        portfolio="zeroshot_2025_12_18_gpu",
        family="GBM_PREP",
        name_suffix="_r13",
        priority=88,
        num_gpus=0,
    ),
    _selection(
        portfolio="zeroshot",
        family="XGB",
        index=0,
        priority=80,
        num_gpus=0,
    ),
    _selection(
        portfolio="zeroshot",
        family="XGB",
        name_suffix="_r33",
        priority=79,
        num_gpus=0,
    ),
    _selection(
        portfolio="zeroshot",
        family="XGB",
        name_suffix="_r89",
        priority=78,
        num_gpus=0,
    ),
    _selection(
        portfolio="zeroshot_2025_12_18_cpu",
        family="NN_TORCH",
        name_suffix="_r37",
        priority=70,
        num_gpus=1,
    ),
    _selection(
        portfolio="zeroshot_2025_12_18_cpu",
        family="NN_TORCH",
        name_suffix="_r31",
        priority=69,
        num_gpus=1,
    ),
    _selection(
        portfolio="zeroshot",
        family="XT",
        name_suffix="Gini",
        priority=60,
        num_gpus=0,
    ),
    _selection(
        portfolio="zeroshot",
        family="XT",
        name_suffix="_r42",
        priority=59,
        num_gpus=0,
    ),
)


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
    "focused_hybrid_v1": AutoGluonProfile(
        profile_id="focused_hybrid_v1",
        description=(
            "Focused post-screening composite of selected RealTabPFN-v2, LightGBMPrep, "
            "XGBoost, NeuralNetTorch, and ExtraTrees configurations from multiple "
            "AutoGluon 1.5.0 built-in portfolios. Not an automatic replacement for "
            "Research v2 evaluation."
        ),
        portfolio=COMPOSITE_PORTFOLIO_MARKER,
        included_model_types=(
            "REALTABPFN-V2",
            "GBM_PREP",
            "XGB",
            "NN_TORCH",
            "XT",
        ),
        excluded_model_types=(),
        minimum_gpu_budget=1,
        composite_selections=_FOCUSED_HYBRID_SELECTIONS,
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


def _composite_selections_summary(
    selections: tuple[CompositeConfigSelection, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "portfolio": selection.portfolio,
            "family": selection.family,
            "selector": selection.selector_payload(),
            "priority": selection.priority,
            "num_gpus": selection.num_gpus,
        }
        for selection in selections
    ]


def _resource_policy_summary(
    profile: AutoGluonProfile,
) -> dict[str, Any]:
    if profile.composite_selections is None:
        return {
            "cpu_only_families": sorted(CPU_ONLY_FAMILIES),
            "gpu_supported_families": sorted(GPU_SUPPORTED_FAMILIES),
            "mixed_profile_top_level_num_gpus": "omitted",
            "fold_fitting_strategy": "sequential_local",
        }

    cpu_only = sorted(
        {
            selection.family
            for selection in profile.composite_selections
            if selection.num_gpus == 0
        }
    )
    gpu_supported = sorted(
        {
            selection.family
            for selection in profile.composite_selections
            if selection.num_gpus > 0
        }
    )
    return {
        "cpu_only_families": cpu_only,
        "gpu_supported_families": gpu_supported,
        "mixed_profile_top_level_num_gpus": "omitted",
        "fold_fitting_strategy": "sequential_local",
    }


def profile_summary(
    profile: AutoGluonProfile,
    seed: int,
    gpu_budget: int,
) -> dict[str, Any]:
    """Return the exact portable profile identity, including seed and resources."""
    summary: dict[str, Any] = {
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
        "resource_policy": _resource_policy_summary(profile),
    }
    if profile.composite_selections is not None:
        summary["composite_selections"] = _composite_selections_summary(
            profile.composite_selections
        )
    return summary


def profile_sha256(profile: AutoGluonProfile, seed: int, gpu_budget: int) -> str:
    """Hash the portable profile identity without local paths."""
    encoded = json.dumps(
        profile_summary(profile, seed, gpu_budget),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _configuration_items(family: str, configurations: Any) -> list[Any]:
    if isinstance(configurations, list):
        return configurations
    return [configurations]


def _effective_name_suffix(configuration: dict[str, Any]) -> str:
    ag_args = configuration.get("ag_args")
    if not isinstance(ag_args, dict):
        return ""
    suffix = ag_args.get("name_suffix")
    if suffix is None:
        return ""
    return str(suffix)


def _select_configuration(
    *,
    portfolio_name: str,
    family: str,
    items: list[Any],
    selection: CompositeConfigSelection,
) -> dict[str, Any]:
    if selection.name_suffix is not None:
        matches: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                raise RuntimeError(
                    f"Profile family {family!r} in portfolio {portfolio_name!r} "
                    "contains a non-mapping configuration."
                )
            if _effective_name_suffix(item) == selection.name_suffix:
                matches.append(item)
        if not matches:
            raise RuntimeError(
                f"Configuration name_suffix {selection.name_suffix!r} for family "
                f"{family!r} is unavailable in portfolio {portfolio_name!r}."
            )
        if len(matches) > 1:
            raise RuntimeError(
                f"Configuration name_suffix {selection.name_suffix!r} for family "
                f"{family!r} is ambiguous in portfolio {portfolio_name!r}."
            )
        return copy.deepcopy(matches[0])

    assert selection.index is not None
    if selection.index < 0 or selection.index >= len(items):
        raise RuntimeError(
            f"Configuration index {selection.index} for family {family!r} is "
            f"unavailable in portfolio {portfolio_name!r} "
            f"(size={len(items)})."
        )
    item = items[selection.index]
    if not isinstance(item, dict):
        raise RuntimeError(
            f"Profile family {family!r} in portfolio {portfolio_name!r} "
            "contains a non-mapping configuration."
        )
    return copy.deepcopy(item)


def _apply_runtime_overrides(
    configuration: dict[str, Any],
    *,
    seed: int,
    priority: int,
    num_gpus: int,
) -> dict[str, Any]:
    ag_args = configuration.setdefault("ag_args", {})
    if not isinstance(ag_args, dict):
        raise RuntimeError("Configuration ag_args must be a mapping when present.")
    ag_args["priority"] = priority
    ensemble = configuration.setdefault("ag_args_ensemble", {})
    if not isinstance(ensemble, dict):
        raise RuntimeError(
            "Configuration ag_args_ensemble must be a mapping when present."
        )
    ensemble["fold_fitting_strategy"] = "sequential_local"
    ensemble["model_random_seed"] = seed
    ensemble["vary_seed_across_folds"] = False
    fit_args = configuration.setdefault("ag_args_fit", {})
    if not isinstance(fit_args, dict):
        raise RuntimeError("Configuration ag_args_fit must be a mapping when present.")
    fit_args["num_gpus"] = num_gpus
    return configuration


def _validate_composite_selection_table(
    selections: tuple[CompositeConfigSelection, ...],
) -> None:
    seen_selectors: set[tuple[str, str, tuple[Any, ...]]] = set()
    for selection in selections:
        selector_identity = (
            selection.portfolio,
            selection.family,
            selection.selector_key(),
        )
        if selector_identity in seen_selectors:
            raise RuntimeError(
                "Duplicate composite configuration selector "
                f"{selection.selector_payload()!r} for family "
                f"{selection.family!r} in portfolio {selection.portfolio!r}."
            )
        seen_selectors.add(selector_identity)


def _load_portfolio(portfolio_name: str, get_hyperparameter_config: Any) -> dict[str, Any]:
    try:
        portfolio = get_hyperparameter_config(portfolio_name)
    except Exception as error:
        raise RuntimeError(
            f"Source portfolio {portfolio_name!r} is unavailable in AutoGluon "
            f"{SUPPORTED_AUTOGLUON_VERSION}: {error}"
        ) from error
    if not isinstance(portfolio, dict):
        raise RuntimeError(
            f"Source portfolio {portfolio_name!r} did not resolve to a mapping."
        )
    return portfolio


def _resolve_composite_hyperparameters(
    profile: AutoGluonProfile,
    *,
    seed: int,
    gpu_budget: int,
    get_hyperparameter_config: Any,
) -> dict[str, Any]:
    assert profile.composite_selections is not None
    selections = profile.composite_selections
    if not selections:
        raise RuntimeError(
            f"Profile {profile.profile_id!r} resolved to no model families."
        )
    _validate_composite_selection_table(selections)

    portfolio_cache: dict[str, dict[str, Any]] = {}
    resolved: dict[str, list[dict[str, Any]]] = {}
    seen_model_identities: set[tuple[str, str]] = set()

    for selection in selections:
        if selection.num_gpus > 0 and gpu_budget < selection.num_gpus:
            raise RuntimeError(
                f"Profile {profile.profile_id!r} selection for family "
                f"{selection.family!r} requires num_gpus={selection.num_gpus}; "
                f"found GPU budget {gpu_budget}."
            )
        if selection.portfolio not in portfolio_cache:
            portfolio_cache[selection.portfolio] = copy.deepcopy(
                _load_portfolio(selection.portfolio, get_hyperparameter_config)
            )
        portfolio = portfolio_cache[selection.portfolio]
        if selection.family not in portfolio:
            raise RuntimeError(
                f"Profile model family {selection.family!r} is unavailable in "
                f"portfolio {selection.portfolio!r}."
            )
        items = _configuration_items(selection.family, portfolio[selection.family])
        configuration = _select_configuration(
            portfolio_name=selection.portfolio,
            family=selection.family,
            items=items,
            selection=selection,
        )
        model_identity = (selection.family, _effective_name_suffix(configuration))
        if model_identity in seen_model_identities:
            raise RuntimeError(
                "Duplicate effective model identity "
                f"{selection.family}{model_identity[1]!r} in profile "
                f"{profile.profile_id!r}."
            )
        seen_model_identities.add(model_identity)
        resolved.setdefault(selection.family, []).append(
            _apply_runtime_overrides(
                configuration,
                seed=seed,
                priority=selection.priority,
                num_gpus=selection.num_gpus,
            )
        )
    return resolved


def _resolve_legacy_hyperparameters(
    profile: AutoGluonProfile,
    *,
    seed: int,
    gpu_budget: int,
    get_hyperparameter_config: Any,
) -> dict[str, Any]:
    portfolio: dict[str, Any] = copy.deepcopy(
        _load_portfolio(profile.portfolio, get_hyperparameter_config)
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
        items = _configuration_items(family, configurations)
        normalized: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                raise RuntimeError(
                    f"Profile family {family!r} contains a non-mapping configuration."
                )
            configuration = copy.deepcopy(item)
            ensemble = configuration.setdefault("ag_args_ensemble", {})
            ensemble["fold_fitting_strategy"] = "sequential_local"
            ensemble["model_random_seed"] = seed
            ensemble["vary_seed_across_folds"] = False
            fit_args = configuration.setdefault("ag_args_fit", {})
            fit_args["num_gpus"] = (
                1 if family in GPU_SUPPORTED_FAMILIES and gpu_budget >= 1 else 0
            )
            if family == "CAT":
                configuration["task_type"] = "CPU"
            normalized.append(configuration)
        portfolio[family] = normalized
    return portfolio


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
    if profile.composite_selections is not None:
        return _resolve_composite_hyperparameters(
            profile,
            seed=seed,
            gpu_budget=gpu_budget,
            get_hyperparameter_config=get_hyperparameter_config,
        )
    return _resolve_legacy_hyperparameters(
        profile,
        seed=seed,
        gpu_budget=gpu_budget,
        get_hyperparameter_config=get_hyperparameter_config,
    )


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
