from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from typing import Any, Mapping

from src.churn_ml.optuna_search_config import SearchSpace


def suggest_parameters(trial: Any, search_space: SearchSpace) -> dict[str, Any]:
    """Resolve one strict adapter parameter mapping from an Optuna trial."""
    resolved: dict[str, Any] = {}
    for name, specification in search_space.parameters.items():
        distribution = specification["distribution"]
        if distribution == "int":
            resolved[name] = trial.suggest_int(
                name,
                int(specification["low"]),
                int(specification["high"]),
                step=int(specification["step"]),
                log=bool(specification["log"]),
            )
        elif distribution == "float":
            resolved[name] = trial.suggest_float(
                name,
                float(specification["low"]),
                float(specification["high"]),
                log=bool(specification["log"]),
            )
        else:
            zero_draw = trial.suggest_float(
                f"{name}__zero_draw",
                0.0,
                1.0,
            )
            if zero_draw < float(specification["zero_probability"]):
                resolved[name] = 0.0
            else:
                resolved[name] = trial.suggest_float(
                    name,
                    float(specification["low_positive"]),
                    float(specification["high"]),
                    log=True,
                )
    return resolved


def build_resolved_adapter_contract(
    base_contract: Mapping[str, Any],
    *,
    adapter_id: str,
    tuned_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    contract = deepcopy(dict(base_contract))
    section = "xgboost" if adapter_id == "xgboost_numeric_v1" else "catboost"
    parameters = contract[section]["parameters"]
    for name, value in tuned_parameters.items():
        if name not in parameters:
            raise ValueError(f"Unknown resolved adapter parameter: {name}.")
        parameters[name] = value
    return contract


def stateless_sample(
    *,
    seed: int,
    study_name: str,
    trial_number: int,
    parameter_name: str,
    distribution: Any,
) -> Any:
    """Sample deterministically from an Optuna distribution without mutable RNG."""
    import numpy as np
    from optuna.distributions import (
        CategoricalDistribution,
        FloatDistribution,
        IntDistribution,
    )

    material = (f"{seed}\0{study_name}\0{trial_number}\0{parameter_name}").encode(
        "utf-8"
    )
    derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    rng = np.random.default_rng(derived_seed)
    if isinstance(distribution, CategoricalDistribution):
        return distribution.choices[int(rng.integers(0, len(distribution.choices)))]
    if isinstance(distribution, IntDistribution):
        if distribution.log:
            low = math.log(distribution.low - 0.5)
            high = math.log(distribution.high + 0.5)
            sampled = int(round(math.exp(float(rng.uniform(low, high)))))
            sampled = min(max(sampled, distribution.low), distribution.high)
            return (
                distribution.low
                + ((sampled - distribution.low) // distribution.step)
                * distribution.step
            )
        count = (distribution.high - distribution.low) // distribution.step + 1
        return distribution.low + int(rng.integers(0, count)) * distribution.step
    if isinstance(distribution, FloatDistribution):
        unit = float(rng.random())
        if distribution.log:
            return math.exp(
                math.log(distribution.low)
                + unit * (math.log(distribution.high) - math.log(distribution.low))
            )
        if distribution.step is None:
            return distribution.low + unit * (distribution.high - distribution.low)
        count = round((distribution.high - distribution.low) / distribution.step) + 1
        return distribution.low + int(rng.integers(0, count)) * distribution.step
    raise TypeError(f"Unsupported Optuna distribution: {type(distribution).__name__}.")
