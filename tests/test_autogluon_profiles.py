from __future__ import annotations

import sys
from types import ModuleType

import pytest

from src.churn_ml.autogluon_profiles import (
    AutoGluonProfile,
    CompositeConfigSelection,
    effective_family_resources,
    get_profile,
    profile_sha256,
    profile_summary,
    resolve_profile_hyperparameters,
)

# Regression anchors for portable profile identity hashes before composite support.
_EXISTING_PROFILE_HASHES = {
    ("realtabpfn_only_v1", 42, 1): (
        "1b5c772a5d81b29845849228b073069e2af7c9b068387d3fc2e1dc0ea1623895"
    ),
    ("tabm_only_gpu_v1", 42, 1): (
        "d64195e60e653ec6fb8863a204bf345e7344d71452f283c701b9adfbe5ce2498"
    ),
    ("catboost_only_cpu_v1", 42, 0): (
        "bed2319d6a504dae597f1d6681351d8d949afe27084ed28770e0d2790caacd34"
    ),
    ("lightgbmprep_only_cpu_v1", 42, 0): (
        "9762c952f6a9350c14ab3ef36af4f3036b446f6e91a8c7ef1e6ec746c2217002"
    ),
    ("extreme_seqmem_v1", 42, 1): (
        "89d7897343044455a5351330918785e600ab3a8b92b28e06e3fa47c4d92ac1bf"
    ),
}


def install_fake_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    autogluon = ModuleType("autogluon")
    tabular = ModuleType("autogluon.tabular")
    configs = ModuleType("autogluon.tabular.configs")
    module = ModuleType("autogluon.tabular.configs.hyperparameter_configs")

    def get_hyperparameter_config(name: str) -> dict[str, object]:
        if name == "zeroshot_2025_12_18_gpu":
            return {
                "GBM": [{}, {}],
                "GBM_PREP": [
                    {
                        "learning_rate": 0.13,
                        "num_leaves": 13,
                        "ag_args": {"name_suffix": "_r13", "priority": -14},
                    },
                    {
                        "learning_rate": 0.41,
                        "num_leaves": 41,
                        "ag_args": {"name_suffix": "_r41", "priority": -16},
                    },
                    {
                        "learning_rate": 0.31,
                        "num_leaves": 31,
                        "ag_args": {"name_suffix": "_r31", "priority": -18},
                    },
                    {
                        "learning_rate": 0.21,
                        "ag_args": {"name_suffix": "_r21"},
                    },
                ],
                "CAT": [{}],
                "REALTABPFN-V2": [
                    {"softmax_temperature": 0.75, "ag_args": {"name_suffix": "_r13"}},
                    {
                        "softmax_temperature": 0.9,
                        "balance_probabilities": True,
                        "ag_args": {"name_suffix": "_r11", "priority": -6},
                    },
                    {"softmax_temperature": 0.8, "ag_args": {"name_suffix": "_c1"}},
                ],
                "TABM": [{}],
                "TABDPT": [{}],
                "TABICL": [{}],
                "MITRA": [{}],
            }
        if name == "zeroshot":
            return {
                "XGB": [
                    {},
                    {
                        "max_depth": 5,
                        "learning_rate": 0.1,
                        "ag_args": {"name_suffix": "_r33"},
                    },
                    {
                        "max_depth": 7,
                        "learning_rate": 0.05,
                        "ag_args": {"name_suffix": "_r89"},
                    },
                    {
                        "max_depth": 9,
                        "ag_args": {"name_suffix": "_r194"},
                    },
                ],
                "XT": [
                    {"criterion": "gini", "ag_args": {"name_suffix": "Gini"}},
                    {"criterion": "entropy", "ag_args": {"name_suffix": "Entr"}},
                    {
                        "max_features": 0.75,
                        "max_leaf_nodes": 18392,
                        "ag_args": {"name_suffix": "_r42", "priority": -9},
                    },
                ],
                "NN_TORCH": [{"ag_args": {"name_suffix": "_r79"}}],
                "GBM": [{}],
                "CAT": [{}],
            }
        if name == "zeroshot_2025_12_18_cpu":
            return {
                "GBM": [{}, {}],
                "GBM_PREP": [{}],
                "CAT": [{}],
                "NN_TORCH": [
                    {
                        "hidden_size": 109,
                        "num_layers": 3,
                        "ag_args": {"name_suffix": "_r37", "priority": -4},
                    },
                    {
                        "hidden_size": 81,
                        "num_layers": 4,
                        "ag_args": {"name_suffix": "_r31", "priority": -9},
                    },
                    {
                        "hidden_size": 50,
                        "ag_args": {"name_suffix": "_r193"},
                    },
                ],
                "FASTAI": [{}],
            }
        raise ValueError(f"Unknown fake portfolio {name!r}")

    module.get_hyperparameter_config = get_hyperparameter_config  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "autogluon", autogluon)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", tabular)
    monkeypatch.setitem(sys.modules, "autogluon.tabular.configs", configs)
    monkeypatch.setitem(
        sys.modules,
        "autogluon.tabular.configs.hyperparameter_configs",
        module,
    )


def _suffix(configuration: dict[str, object]) -> str | None:
    ag_args = configuration.get("ag_args")
    if not isinstance(ag_args, dict):
        return None
    suffix = ag_args.get("name_suffix")
    return None if suffix is None else str(suffix)


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


def test_existing_profile_hashes_remain_stable() -> None:
    for (profile_id, seed, gpu_budget), expected in _EXISTING_PROFILE_HASHES.items():
        assert (
            profile_sha256(get_profile(profile_id), seed, gpu_budget) == expected
        ), profile_id


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


def test_tabm_only_gpu_profile_resources_and_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_registry(monkeypatch)
    profile = get_profile("tabm_only_gpu_v1")
    assert profile.included_model_types == ("TABM",)
    assert profile.excluded_model_types == ()
    assert profile.minimum_gpu_budget == 1
    assert profile.portfolio == "zeroshot_2025_12_18_gpu"

    resolved = resolve_profile_hyperparameters(profile, seed=42, gpu_budget=1)
    assert set(resolved) == {"TABM"}
    resources = effective_family_resources(resolved)
    assert resources["TABM"]["num_gpus"] == [1]
    for configuration in resolved["TABM"]:
        assert configuration["ag_args_ensemble"]["model_random_seed"] == 42
        assert (
            configuration["ag_args_ensemble"]["fold_fitting_strategy"]
            == "sequential_local"
        )
        assert configuration["ag_args_ensemble"]["vary_seed_across_folds"] is False

    with pytest.raises(RuntimeError, match="requires GPU budget"):
        resolve_profile_hyperparameters(profile, seed=42, gpu_budget=0)


def test_focused_hybrid_resolves_exact_composite_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_registry(monkeypatch)
    profile = get_profile("focused_hybrid_v1")
    assert profile.minimum_gpu_budget == 1
    assert profile.portfolio == "composite"
    assert profile.composite_selections is not None
    assert len(profile.composite_selections) == 11

    resolved = resolve_profile_hyperparameters(profile, seed=42, gpu_budget=1)
    assert list(resolved) == [
        "REALTABPFN-V2",
        "GBM_PREP",
        "XGB",
        "NN_TORCH",
        "XT",
    ]
    assert {family: len(configs) for family, configs in resolved.items()} == {
        "REALTABPFN-V2": 1,
        "GBM_PREP": 3,
        "XGB": 3,
        "NN_TORCH": 2,
        "XT": 2,
    }
    assert [_suffix(item) for item in resolved["REALTABPFN-V2"]] == ["_r11"]
    assert [_suffix(item) for item in resolved["GBM_PREP"]] == ["_r31", "_r41", "_r13"]
    assert [_suffix(item) for item in resolved["XGB"]] == [None, "_r33", "_r89"]
    assert [_suffix(item) for item in resolved["NN_TORCH"]] == ["_r37", "_r31"]
    assert [_suffix(item) for item in resolved["XT"]] == ["Gini", "_r42"]

    # Predictive hyperparameters preserved from the selected source configs.
    assert resolved["REALTABPFN-V2"][0]["softmax_temperature"] == 0.9
    assert resolved["REALTABPFN-V2"][0]["balance_probabilities"] is True
    assert resolved["GBM_PREP"][0]["learning_rate"] == 0.31
    assert resolved["GBM_PREP"][1]["num_leaves"] == 41
    assert resolved["GBM_PREP"][2]["learning_rate"] == 0.13
    assert resolved["XGB"][0] == {
        "ag_args": {"priority": 80},
        "ag_args_ensemble": {
            "fold_fitting_strategy": "sequential_local",
            "model_random_seed": 42,
            "vary_seed_across_folds": False,
        },
        "ag_args_fit": {"num_gpus": 0},
    }
    assert resolved["XGB"][1]["max_depth"] == 5
    assert resolved["XGB"][2]["learning_rate"] == 0.05
    assert resolved["NN_TORCH"][0]["hidden_size"] == 109
    assert resolved["NN_TORCH"][1]["num_layers"] == 4
    assert resolved["XT"][0]["criterion"] == "gini"
    assert resolved["XT"][1]["max_leaf_nodes"] == 18392

    expected_priorities = {
        ("REALTABPFN-V2", "_r11"): 100,
        ("GBM_PREP", "_r31"): 90,
        ("GBM_PREP", "_r41"): 89,
        ("GBM_PREP", "_r13"): 88,
        ("XGB", None): 80,
        ("XGB", "_r33"): 79,
        ("XGB", "_r89"): 78,
        ("NN_TORCH", "_r37"): 70,
        ("NN_TORCH", "_r31"): 69,
        ("XT", "Gini"): 60,
        ("XT", "_r42"): 59,
    }
    for family, configurations in resolved.items():
        for configuration in configurations:
            key = (family, _suffix(configuration))
            assert configuration["ag_args"]["priority"] == expected_priorities[key]

    resources = effective_family_resources(resolved)
    assert resources["REALTABPFN-V2"]["num_gpus"] == [1]
    assert resources["NN_TORCH"]["num_gpus"] == [1]
    assert resources["GBM_PREP"]["num_gpus"] == [0]
    assert resources["XGB"]["num_gpus"] == [0]
    assert resources["XT"]["num_gpus"] == [0]

    for configurations in resolved.values():
        for configuration in configurations:
            ensemble = configuration["ag_args_ensemble"]
            assert ensemble["model_random_seed"] == 42
            assert ensemble["vary_seed_across_folds"] is False
            assert ensemble["fold_fitting_strategy"] == "sequential_local"

    with pytest.raises(RuntimeError, match="requires GPU budget"):
        resolve_profile_hyperparameters(profile, seed=42, gpu_budget=0)


def test_focused_hybrid_identity_includes_composite_provenance() -> None:
    summary = profile_summary(get_profile("focused_hybrid_v1"), 42, 1)
    assert "composite_selections" in summary
    assert len(summary["composite_selections"]) == 11
    first = summary["composite_selections"][0]
    assert first["portfolio"] == "zeroshot_2025_12_18_gpu"
    assert first["family"] == "REALTABPFN-V2"
    assert first["selector"] == {"kind": "name_suffix", "name_suffix": "_r11"}
    assert first["priority"] == 100
    assert first["num_gpus"] == 1
    xgb_default = summary["composite_selections"][4]
    assert xgb_default["selector"] == {"kind": "index", "index": 0}
    assert summary["resource_policy"]["cpu_only_families"] == [
        "GBM_PREP",
        "XGB",
        "XT",
    ]
    assert summary["resource_policy"]["gpu_supported_families"] == [
        "NN_TORCH",
        "REALTABPFN-V2",
    ]

    legacy = profile_summary(get_profile("realtabpfn_only_v1"), 42, 1)
    assert "composite_selections" not in legacy
    assert legacy["resource_policy"]["cpu_only_families"] == ["CAT", "GBM", "GBM_PREP"]
    assert legacy["resource_policy"]["gpu_supported_families"] == [
        "REALTABPFN-V2",
        "TABM",
    ]


def test_composite_resolution_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_registry(monkeypatch)

    def resolve(selections: tuple[CompositeConfigSelection, ...]) -> None:
        resolve_profile_hyperparameters(
            AutoGluonProfile(
                profile_id="composite_test",
                description="test",
                portfolio="composite",
                included_model_types=("XGB",),
                excluded_model_types=(),
                minimum_gpu_budget=0,
                composite_selections=selections,
            ),
            seed=42,
            gpu_budget=0,
        )

    with pytest.raises(RuntimeError, match="Source portfolio .* unavailable"):
        resolve(
            (
                CompositeConfigSelection(
                    portfolio="missing_portfolio",
                    family="XGB",
                    index=0,
                    priority=1,
                    num_gpus=0,
                ),
            )
        )

    with pytest.raises(RuntimeError, match="model family .* unavailable"):
        resolve(
            (
                CompositeConfigSelection(
                    portfolio="zeroshot",
                    family="MISSING_FAMILY",
                    name_suffix="_r33",
                    priority=1,
                    num_gpus=0,
                ),
            )
        )

    with pytest.raises(RuntimeError, match="name_suffix .* unavailable"):
        resolve(
            (
                CompositeConfigSelection(
                    portfolio="zeroshot",
                    family="XGB",
                    name_suffix="_missing",
                    priority=1,
                    num_gpus=0,
                ),
            )
        )

    with pytest.raises(RuntimeError, match="Configuration index .* unavailable"):
        resolve(
            (
                CompositeConfigSelection(
                    portfolio="zeroshot",
                    family="XGB",
                    index=99,
                    priority=1,
                    num_gpus=0,
                ),
            )
        )

    with pytest.raises(RuntimeError, match="Duplicate composite configuration selector"):
        resolve(
            (
                CompositeConfigSelection(
                    portfolio="zeroshot",
                    family="XGB",
                    name_suffix="_r33",
                    priority=2,
                    num_gpus=0,
                ),
                CompositeConfigSelection(
                    portfolio="zeroshot",
                    family="XGB",
                    name_suffix="_r33",
                    priority=1,
                    num_gpus=0,
                ),
            )
        )


def test_composite_rejects_duplicate_empty_model_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_registry(monkeypatch)
    # Build a temporary fake portfolio that exposes two empty-suffix XGB configs
    # under different selectors so model-identity collision is reachable.
    module = sys.modules["autogluon.tabular.configs.hyperparameter_configs"]
    original = module.get_hyperparameter_config

    def get_hyperparameter_config(name: str) -> dict[str, object]:
        if name == "zeroshot":
            return {
                "XGB": [
                    {},
                    {"ag_args": {}},
                ]
            }
        return original(name)

    module.get_hyperparameter_config = get_hyperparameter_config  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="Duplicate effective model identity"):
        resolve_profile_hyperparameters(
            AutoGluonProfile(
                profile_id="dup_identity",
                description="test",
                portfolio="composite",
                included_model_types=("XGB",),
                excluded_model_types=(),
                minimum_gpu_budget=0,
                composite_selections=(
                    CompositeConfigSelection(
                        portfolio="zeroshot",
                        family="XGB",
                        index=0,
                        priority=2,
                        num_gpus=0,
                    ),
                    CompositeConfigSelection(
                        portfolio="zeroshot",
                        family="XGB",
                        index=1,
                        priority=1,
                        num_gpus=0,
                    ),
                ),
            ),
            seed=42,
            gpu_budget=0,
        )


def test_composite_rejects_non_mapping_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_registry(monkeypatch)
    module = sys.modules["autogluon.tabular.configs.hyperparameter_configs"]
    original = module.get_hyperparameter_config

    def get_hyperparameter_config(name: str) -> dict[str, object]:
        if name == "zeroshot":
            return {"XGB": ["not-a-mapping"]}
        return original(name)

    module.get_hyperparameter_config = get_hyperparameter_config  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="non-mapping configuration"):
        resolve_profile_hyperparameters(
            AutoGluonProfile(
                profile_id="bad_config",
                description="test",
                portfolio="composite",
                included_model_types=("XGB",),
                excluded_model_types=(),
                minimum_gpu_budget=0,
                composite_selections=(
                    CompositeConfigSelection(
                        portfolio="zeroshot",
                        family="XGB",
                        index=0,
                        priority=1,
                        num_gpus=0,
                    ),
                ),
            ),
            seed=42,
            gpu_budget=0,
        )
