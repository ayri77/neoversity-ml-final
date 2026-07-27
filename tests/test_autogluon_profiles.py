from __future__ import annotations

import sys
from types import ModuleType

import pytest

from src.churn_ml.autogluon_profiles import (
    effective_family_resources,
    get_profile,
    profile_sha256,
    resolve_profile_hyperparameters,
)


def install_fake_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    autogluon = ModuleType("autogluon")
    tabular = ModuleType("autogluon.tabular")
    configs = ModuleType("autogluon.tabular.configs")
    module = ModuleType("autogluon.tabular.configs.hyperparameter_configs")

    def get_hyperparameter_config(_name: str) -> dict[str, object]:
        return {
            "GBM": [{}, {}],
            "GBM_PREP": [{}],
            "CAT": [{}],
            "REALTABPFN-V2": [{}, {}],
            "TABM": [{}],
            "TABDPT": [{}],
            "TABICL": [{}],
            "MITRA": [{}],
        }

    module.get_hyperparameter_config = get_hyperparameter_config  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "autogluon", autogluon)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", tabular)
    monkeypatch.setitem(sys.modules, "autogluon.tabular.configs", configs)
    monkeypatch.setitem(
        sys.modules,
        "autogluon.tabular.configs.hyperparameter_configs",
        module,
    )


def test_extreme_profile_resources_and_seed_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_registry(monkeypatch)
    resolved = resolve_profile_hyperparameters(
        get_profile("extreme_seqmem_v1"),
        seed=42,
        gpu_budget=1,
    )
    assert set(resolved) == {
        "GBM",
        "GBM_PREP",
        "CAT",
        "REALTABPFN-V2",
        "TABM",
    }
    resources = effective_family_resources(resolved)
    assert resources["GBM"]["num_gpus"] == [0]
    assert resources["GBM_PREP"]["num_gpus"] == [0]
    assert resources["CAT"]["num_gpus"] == [0]
    assert resources["CAT"]["task_types"] == ["CPU"]
    assert resources["REALTABPFN-V2"]["num_gpus"] == [1]
    assert resources["TABM"]["num_gpus"] == [1]
    for configurations in resolved.values():
        for configuration in configurations:
            assert configuration["ag_args_ensemble"]["model_random_seed"] == 42
            assert (
                configuration["ag_args_ensemble"]["fold_fitting_strategy"]
                == "sequential_local"
            )


def test_profile_hash_changes_with_seed() -> None:
    profile = get_profile("realtabpfn_only_v1")
    assert profile_sha256(profile, 42, 1) != profile_sha256(profile, 43, 1)


def test_gpu_profile_rejects_insufficient_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_registry(monkeypatch)
    with pytest.raises(RuntimeError, match="requires GPU budget"):
        resolve_profile_hyperparameters(
            get_profile("realtabpfn_only_v1"),
            seed=42,
            gpu_budget=0,
        )
