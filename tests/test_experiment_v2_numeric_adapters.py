from __future__ import annotations

import importlib.metadata
import inspect
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from src.churn_ml import experiment_v2_numeric_adapter, research_v2_cli
from src.churn_ml.experiment_v2 import (
    ExperimentV2AdapterDependencyError,
    candidate_adapter_registry,
    get_candidate_adapter,
)
from src.churn_ml.experiment_v2_adapter import ExperimentV2AdapterContractError
from src.churn_ml.experiment_v2_catboost_adapter import (
    EXPECTED_CATBOOST_NUMERIC_V1_CONTRACT,
    CatboostNumericV1Adapter,
    _default_estimator_factory as catboost_estimator_factory,
)
from src.churn_ml.experiment_v2_numeric_adapter import validate_numeric_matrix
from src.churn_ml.experiment_v2_xgboost_adapter import (
    EXPECTED_XGBOOST_NUMERIC_V1_CONTRACT,
    XgboostNumericV1Adapter,
    _default_estimator_factory as xgboost_estimator_factory,
)
from src.churn_ml.research_v2_artifact_validation import (
    validate_portable_payload_paths,
)
from src.churn_ml.research_v2_config import load_research_v2_config
from src.churn_ml.research_v2_identity import (
    ADAPTER_IMPLEMENTATION_SOURCES,
    CATBOOST_ADAPTER_IMPLEMENTATION_SOURCES,
    PIPELINE_IMPLEMENTATION_SOURCES,
    XGBOOST_ADAPTER_IMPLEMENTATION_SOURCES,
    build_component_identities,
)
from src.churn_ml.research_data import canonical_sha256
from src.churn_ml.research_v2_provenance import environment_identity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data/processed/v3_targeted_missingness/X_train.parquet"
MANUAL_CONFIG = (
    PROJECT_ROOT / "configs/research_v2/manual_lightgbm_te_v1_compat_smoke.yaml"
)
XGBOOST_SMOKE_CONFIG = (
    PROJECT_ROOT / "configs/research_v2/xgboost_numeric_v1_smoke.yaml"
)


def _contract(use_xgboost: bool) -> dict[str, Any]:
    source = (
        EXPECTED_XGBOOST_NUMERIC_V1_CONTRACT
        if use_xgboost
        else EXPECTED_CATBOOST_NUMERIC_V1_CONTRACT
    )
    return deepcopy(cast(dict[str, Any], source))


CONFIGS = (
    "xgboost_numeric_v1_smoke.yaml",
    "catboost_numeric_v1_smoke.yaml",
    "xgboost_numeric_v1_development.yaml",
    "catboost_numeric_v1_development.yaml",
)


class RecordingEstimator:
    def __init__(self, parameters: dict[str, Any]) -> None:
        self.parameters = parameters
        self.classes_ = np.array([1, 0], dtype=np.int8)
        self.fit_features: pd.DataFrame | None = None
        self.fit_labels: pd.Series | None = None

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> None:
        self.fit_features = features.copy()
        self.fit_labels = labels.copy()

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        numeric = features.iloc[:, 0].to_numpy(dtype=float)
        positive = 0.2 + np.nan_to_num(numeric, nan=0.0) / 100.0
        return np.column_stack([positive, 1.0 - positive])


def _data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    numeric = np.arange(20, dtype=float)
    numeric[[0, 7]] = np.nan
    train = pd.DataFrame(
        {
            "numeric": numeric,
            "all_nan": np.full(20, np.nan),
            "category": pd.Series(["a", "b"] * 10, dtype="object"),
        }
    )
    labels = pd.Series([0, 1] * 10, dtype="int64")
    prediction = pd.DataFrame(
        {
            "numeric": [5.0, np.nan, 8.0],
            "all_nan": np.full(3, np.nan),
            "category": pd.Series(["b", "a", "b"], dtype="object"),
        },
        index=[91, 17, 44],
    )
    return train, labels, prediction


@pytest.mark.parametrize(
    "adapter_id,model_key,expected_cpu",
    [
        (
            "xgboost_numeric_v1",
            "xgboost",
            {"device": "cpu", "tree_method": "hist", "booster": "gbtree"},
        ),
        (
            "catboost_numeric_v1",
            "catboost",
            {"task_type": "CPU", "allow_writing_files": False, "nan_mode": "Min"},
        ),
    ],
)
def test_adapters_follow_protocol_and_pass_only_training_data(
    adapter_id: str,
    model_key: str,
    expected_cpu: dict[str, Any],
) -> None:
    train, labels, prediction = _data()
    contract = _contract(model_key == "xgboost")
    created: list[RecordingEstimator] = []

    def factory(parameters: dict[str, Any]) -> RecordingEstimator:
        estimator = RecordingEstimator(parameters)
        created.append(estimator)
        return estimator

    adapter = (
        XgboostNumericV1Adapter(factory)
        if model_key == "xgboost"
        else CatboostNumericV1Adapter(factory)
    )
    assert set(inspect.signature(adapter.fit_predict).parameters) == {
        "train_features",
        "train_labels",
        "prediction_features",
        "contract",
        "model_training_positions",
        "prediction_positions",
        "audit_callback",
    }
    audit: list[tuple[np.ndarray, np.ndarray]] = []
    probabilities = adapter.fit_predict(
        train,
        labels,
        prediction,
        contract,
        model_training_positions=np.arange(20),
        prediction_positions=np.array([91, 17, 44]),
        audit_callback=lambda fit, predict: audit.append((fit, predict)),
    )
    np.testing.assert_allclose(probabilities, [0.25, 0.20, 0.28])
    assert probabilities.shape == (3,)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    assert not set(probabilities).issubset({0.0, 1.0})
    assert len(created) == 1
    assert created[0].fit_features is not None
    assert created[0].fit_features.select_dtypes(exclude="number").empty
    assert created[0].fit_features["numeric"].isna().sum() == 2
    assert created[0].fit_features["all_nan"].isna().all()
    if model_key == "xgboost":
        assert np.isnan(created[0].parameters["missing"])
        expected_parameters = deepcopy(contract[model_key]["parameters"])
        expected_parameters["missing"] = np.nan
        assert created[0].parameters.keys() == expected_parameters.keys()
        for key, value in created[0].parameters.items():
            if key != "missing":
                assert value == expected_parameters[key]
    else:
        assert created[0].parameters == contract[model_key]["parameters"]
    for key, expected in expected_cpu.items():
        assert created[0].parameters[key] == expected
    assert "early_stopping_rounds" not in created[0].parameters
    assert "eval_set" not in created[0].parameters
    assert len(audit) == 1
    np.testing.assert_array_equal(audit[0][1], [91, 17, 44])
    assert adapter.id == adapter_id


@pytest.mark.parametrize("adapter_id", ["xgboost_numeric_v1", "catboost_numeric_v1"])
@pytest.mark.parametrize(
    "labels",
    [
        pd.Series([0, 0] * 10, dtype="int64"),
        pd.Series([0, 2] * 10, dtype="int64"),
        pd.Series([False, True] * 10, dtype="bool"),
    ],
)
def test_adapters_reject_non_exact_binary_targets(
    adapter_id: str,
    labels: pd.Series,
) -> None:
    train, _, prediction = _data()
    adapter = get_candidate_adapter(adapter_id)
    contract = _contract(adapter_id.startswith("xgboost"))
    with pytest.raises(ExperimentV2AdapterContractError, match="binary labels"):
        adapter.fit_predict(
            train,
            labels,
            prediction,
            contract,
            model_training_positions=np.arange(20),
            prediction_positions=np.arange(3),
        )


@pytest.mark.parametrize("adapter_id", ["xgboost_numeric_v1", "catboost_numeric_v1"])
@pytest.mark.parametrize("bad_value", [np.inf, -np.inf])
def test_adapters_reject_infinite_numeric_features(
    adapter_id: str,
    bad_value: float,
) -> None:
    train, labels, prediction = _data()
    train.loc[1, "numeric"] = bad_value
    contract = _contract(adapter_id.startswith("xgboost"))
    with pytest.raises(ExperimentV2AdapterContractError, match="infinity_count"):
        get_candidate_adapter(adapter_id).fit_predict(
            train,
            labels,
            prediction,
            contract,
            model_training_positions=np.arange(20),
            prediction_positions=np.arange(3),
        )


def test_native_nan_diagnostics_are_exact_and_validation_does_not_mutate() -> None:
    features = pd.DataFrame(
        {"a": [1.0, np.nan, 3.0], "all_nan": [np.nan, np.nan, np.nan]},
        index=[10, 20, 30],
    )
    original = features.copy(deep=True)
    diagnostics = validate_numeric_matrix(features, "unit")
    pd.testing.assert_frame_equal(features, original)
    assert diagnostics.to_dict() == {
        "label": "unit",
        "row_count": 3,
        "column_count": 2,
        "total_missing_count": 4,
        "rows_with_missing_count": 3,
        "columns_with_missing_count": 2,
        "columns_with_missing": ("a", "all_nan"),
        "infinity_count": 0,
    }


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_numeric_validator_rejects_infinity_with_precise_path(value: float) -> None:
    features = pd.DataFrame({"a": [1.0, value]})
    with pytest.raises(
        ExperimentV2AdapterContractError,
        match=r"candidate_adapter\.numeric_matrix\.unit\.infinity_count must be 0",
    ):
        validate_numeric_matrix(features, "unit")


@pytest.mark.parametrize(
    "features",
    [
        pd.DataFrame({"a": ["1", "2"]}),
        pd.DataFrame({"a": [1.0, "2"]}, dtype="object"),
    ],
)
def test_numeric_validator_rejects_object_string_and_mixed_values(
    features: pd.DataFrame,
) -> None:
    with pytest.raises(
        ExperimentV2AdapterContractError,
        match=r"candidate_adapter\.numeric_matrix\.unit\.non_numeric_columns",
    ):
        validate_numeric_matrix(features, "unit")


@pytest.mark.parametrize("adapter_id", ["xgboost_numeric_v1", "catboost_numeric_v1"])
@pytest.mark.parametrize("bad_policy", ["impute", 1, True, None])
def test_missing_value_policy_is_exact_and_strict(
    adapter_id: str,
    bad_policy: Any,
) -> None:
    contract = _contract(adapter_id.startswith("xgboost"))
    contract["numeric_features"]["missing_value_policy"] = bad_policy
    with pytest.raises(
        ExperimentV2AdapterContractError,
        match=r"numeric_features\.missing_value_policy",
    ):
        get_candidate_adapter(adapter_id).validate_contract(contract)


@pytest.mark.parametrize(
    "adapter_id,model_key,parameter,bad_value",
    [
        ("xgboost_numeric_v1", "xgboost", "n_estimators", True),
        ("xgboost_numeric_v1", "xgboost", "n_estimators", "300"),
        ("xgboost_numeric_v1", "xgboost", "learning_rate", float("nan")),
        ("xgboost_numeric_v1", "xgboost", "learning_rate", float("inf")),
        ("xgboost_numeric_v1", "xgboost", "device", "cuda"),
        ("xgboost_numeric_v1", "xgboost", "tree_method", "gpu_hist"),
        ("catboost_numeric_v1", "catboost", "iterations", True),
        ("catboost_numeric_v1", "catboost", "iterations", "300"),
        ("catboost_numeric_v1", "catboost", "learning_rate", float("nan")),
        ("catboost_numeric_v1", "catboost", "learning_rate", float("inf")),
        ("catboost_numeric_v1", "catboost", "task_type", "GPU"),
        ("catboost_numeric_v1", "catboost", "allow_writing_files", True),
    ],
)
def test_parameter_contract_rejects_wrong_types_ranges_and_unsafe_modes(
    adapter_id: str,
    model_key: str,
    parameter: str,
    bad_value: Any,
) -> None:
    contract = _contract(model_key == "xgboost")
    contract[model_key]["parameters"][parameter] = bad_value
    with pytest.raises(ExperimentV2AdapterContractError, match=parameter):
        get_candidate_adapter(adapter_id).validate_contract(contract)


@pytest.mark.parametrize(
    "adapter_id,model_key",
    [("xgboost_numeric_v1", "xgboost"), ("catboost_numeric_v1", "catboost")],
)
def test_parameter_contract_allows_valid_override_and_rejects_unknown_key(
    adapter_id: str,
    model_key: str,
) -> None:
    contract = _contract(model_key == "xgboost")
    count_key = "n_estimators" if model_key == "xgboost" else "iterations"
    contract[model_key]["parameters"][count_key] = 7
    adapter = get_candidate_adapter(adapter_id)
    adapter.validate_contract(contract)
    first = adapter.identity_inputs(contract)
    second = adapter.identity_inputs(contract)
    assert first == second
    assert first["contract"][model_key]["parameters"][count_key] == 7
    assert first["missing_value_policy"] == "native_nan"
    assert first["contract"]["numeric_features"]["missing_value_policy"] == "native_nan"
    expected_native = (
        {"missing": "IEEE_NaN"} if model_key == "xgboost" else {"nan_mode": "Min"}
    )
    assert first["native_missing_configuration"] == expected_native
    changed_policy = deepcopy(first)
    changed_policy["missing_value_policy"] = "different_versioned_policy"
    changed_policy["contract"]["numeric_features"]["missing_value_policy"] = (
        "different_versioned_policy"
    )
    assert canonical_sha256(changed_policy) != canonical_sha256(first)
    json.dumps(first, allow_nan=False)
    validate_portable_payload_paths(
        first,
        label="adapter.identity",
        project_root=PROJECT_ROOT,
    )
    contract[model_key]["parameters"]["unknown"] = 1
    with pytest.raises(ExperimentV2AdapterContractError, match="unknown"):
        adapter.validate_contract(contract)


@pytest.mark.parametrize("bad_value", ["NaN", True, 1, None])
def test_xgboost_portable_missing_parameter_is_exact_and_strict(
    bad_value: Any,
) -> None:
    contract = _contract(True)
    contract["xgboost"]["parameters"]["missing"] = bad_value
    with pytest.raises(
        ExperimentV2AdapterContractError,
        match=r"xgboost\.parameters\.missing",
    ):
        get_candidate_adapter("xgboost_numeric_v1").validate_contract(contract)


def test_xgboost_portable_missing_parameter_is_required() -> None:
    contract = _contract(True)
    del contract["xgboost"]["parameters"]["missing"]
    with pytest.raises(
        ExperimentV2AdapterContractError,
        match=r"xgboost\.parameters keys differ; missing=\['missing'\]",
    ):
        get_candidate_adapter("xgboost_numeric_v1").validate_contract(contract)


def test_all_new_configs_and_legacy_config_validate_without_data_access() -> None:
    for name in CONFIGS:
        config = load_research_v2_config(
            PROJECT_ROOT / "configs/research_v2" / name,
            project_root=PROJECT_ROOT,
        )
        assert config.adapter_id in candidate_adapter_registry()
        assert config.adapter_contract["numeric_features"]["missing_value_policy"] == (
            "native_nan"
        )
        if config.adapter_id == "xgboost_numeric_v1":
            assert (
                config.adapter_contract["xgboost"]["parameters"]["missing"]
                == "IEEE_NaN"
            )
        json.dumps(config.resolved_payload(), allow_nan=False)
    legacy = load_research_v2_config(
        PROJECT_ROOT / "configs/research_v2/manual_lightgbm_te_v1_compat_smoke.yaml",
        project_root=PROJECT_ROOT,
    )
    assert legacy.adapter_id == "manual_lightgbm_te_v1_compat"


def test_missing_policy_changes_adapter_and_candidate_but_not_pipeline_identity() -> (
    None
):
    paths = set(PIPELINE_IMPLEMENTATION_SOURCES)
    paths.update(XGBOOST_ADAPTER_IMPLEMENTATION_SOURCES)
    sources = {path: f"{index:064x}" for index, path in enumerate(sorted(paths), 1)}
    adapter = get_candidate_adapter("xgboost_numeric_v1")
    adapter_inputs = adapter.identity_inputs(_contract(True))

    def build(inputs: dict[str, Any]) -> dict[str, str]:
        _, hashes = build_component_identities(
            pipeline_inputs={"id": "pipeline", "contract": {"version": 1}},
            adapter_inputs=inputs,
            resolved_feature_schema={"ordered": ["a"]},
            runtime_dependencies={"python": "3.12", "xgboost": "3.3.0"},
            dataset_version="unit",
            source_records=sources,
        )
        return hashes

    before = build(adapter_inputs)
    changed = deepcopy(adapter_inputs)
    changed["missing_value_policy"] = "future_versioned_policy"
    changed["contract"]["numeric_features"]["missing_value_policy"] = (
        "future_versioned_policy"
    )
    after = build(changed)
    assert after["feature_pipeline"] == before["feature_pipeline"]
    assert after["candidate_adapter"] != before["candidate_adapter"]
    assert after["candidate"] != before["candidate"]


def test_xgboost_missing_parameter_changes_only_xgboost_identities() -> None:
    paths = set(PIPELINE_IMPLEMENTATION_SOURCES)
    paths.update(ADAPTER_IMPLEMENTATION_SOURCES)
    paths.update(XGBOOST_ADAPTER_IMPLEMENTATION_SOURCES)
    paths.update(CATBOOST_ADAPTER_IMPLEMENTATION_SOURCES)
    sources = {path: f"{index:064x}" for index, path in enumerate(sorted(paths), 1)}

    def hashes(adapter_id: str, contract: dict[str, Any]) -> dict[str, str]:
        adapter = get_candidate_adapter(adapter_id)
        _, result = build_component_identities(
            pipeline_inputs={"id": "pipeline", "contract": {"version": 1}},
            adapter_inputs=adapter.identity_inputs(contract),
            resolved_feature_schema={"ordered": ["a"]},
            runtime_dependencies={"python": "3.12", adapter_id.split("_")[0]: "1.0"},
            dataset_version="unit",
            source_records=sources,
        )
        return result

    xgboost_contract = _contract(True)
    catboost_contract = _contract(False)
    xgboost_before = hashes("xgboost_numeric_v1", xgboost_contract)
    catboost_before = hashes("catboost_numeric_v1", catboost_contract)
    xgboost_inputs = get_candidate_adapter("xgboost_numeric_v1").identity_inputs(
        xgboost_contract
    )
    changed_inputs = deepcopy(xgboost_inputs)
    changed_inputs["contract"]["xgboost"]["parameters"]["missing"] = "future_marker"
    changed_inputs["native_missing_configuration"]["missing"] = "future_marker"
    _, changed_hashes = build_component_identities(
        pipeline_inputs={"id": "pipeline", "contract": {"version": 1}},
        adapter_inputs=changed_inputs,
        resolved_feature_schema={"ordered": ["a"]},
        runtime_dependencies={"python": "3.12", "xgboost": "1.0"},
        dataset_version="unit",
        source_records=sources,
    )
    assert changed_hashes["feature_pipeline"] == xgboost_before["feature_pipeline"]
    assert changed_hashes["candidate_adapter"] != xgboost_before["candidate_adapter"]
    assert changed_hashes["candidate"] != xgboost_before["candidate"]
    assert hashes("catboost_numeric_v1", catboost_contract) == catboost_before


def test_adapter_source_identity_mutations_are_model_scoped() -> None:
    paths = set(PIPELINE_IMPLEMENTATION_SOURCES)
    paths.update(ADAPTER_IMPLEMENTATION_SOURCES)
    paths.update(XGBOOST_ADAPTER_IMPLEMENTATION_SOURCES)
    paths.update(CATBOOST_ADAPTER_IMPLEMENTATION_SOURCES)
    sources = {path: f"{index:064x}" for index, path in enumerate(sorted(paths), 1)}

    def hashes(adapter_id: str, records: dict[str, str]) -> dict[str, str]:
        adapter = get_candidate_adapter(adapter_id)
        contract = deepcopy(
            EXPECTED_XGBOOST_NUMERIC_V1_CONTRACT
            if adapter_id.startswith("xgboost")
            else EXPECTED_CATBOOST_NUMERIC_V1_CONTRACT
        )
        _, result = build_component_identities(
            pipeline_inputs={"id": "pipeline", "contract": {"version": 1}},
            adapter_inputs=adapter.identity_inputs(contract),
            resolved_feature_schema={"ordered": ["a"]},
            runtime_dependencies={"python": "3.12", adapter_id.split("_")[0]: "1.0"},
            dataset_version="unit",
            source_records=records,
        )
        return result

    xgboost_before = hashes("xgboost_numeric_v1", sources)
    catboost_before = hashes("catboost_numeric_v1", sources)
    changed = dict(sources)
    changed["src/churn_ml/experiment_v2_xgboost_adapter.py"] = "f" * 64
    xgboost_after = hashes("xgboost_numeric_v1", changed)
    catboost_after = hashes("catboost_numeric_v1", changed)
    assert xgboost_after["candidate_adapter"] != xgboost_before["candidate_adapter"]
    assert xgboost_after["candidate"] != xgboost_before["candidate"]
    assert catboost_after == catboost_before

    changed = dict(sources)
    changed["src/churn_ml/experiment_v2_catboost_adapter.py"] = "e" * 64
    assert hashes("xgboost_numeric_v1", changed) == xgboost_before
    assert (
        hashes("catboost_numeric_v1", changed)["candidate_adapter"]
        != catboost_before["candidate_adapter"]
    )

    changed = dict(sources)
    changed["src/churn_ml/experiment_v2_numeric_adapter.py"] = "d" * 64
    assert (
        hashes("xgboost_numeric_v1", changed)["candidate_adapter"]
        != xgboost_before["candidate_adapter"]
    )
    assert (
        hashes("catboost_numeric_v1", changed)["candidate_adapter"]
        != catboost_before["candidate_adapter"]
    )


def test_runtime_provenance_is_selected_library_specific() -> None:
    legacy = environment_identity("manual_lightgbm_te_v1_compat")
    xgboost = environment_identity("xgboost_numeric_v1")
    catboost = environment_identity("catboost_numeric_v1")
    assert "xgboost" not in legacy and "catboost" not in legacy
    assert "xgboost" in xgboost and "catboost" not in xgboost
    assert "catboost" in catboost and "xgboost" not in catboost


@pytest.mark.parametrize(
    "adapter_id,package,factory",
    [
        ("xgboost_numeric_v1", "xgboost", xgboost_estimator_factory),
        ("catboost_numeric_v1", "catboost", catboost_estimator_factory),
    ],
)
def test_missing_top_level_adapter_import_has_precise_dependency_error(
    adapter_id: str,
    package: str,
    factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ModuleNotFoundError(f"No module named '{package}'", name=package)

    def missing_import(name: str) -> Any:
        assert name == package
        raise original

    monkeypatch.setattr(
        experiment_v2_numeric_adapter.importlib,
        "import_module",
        missing_import,
    )
    with pytest.raises(ExperimentV2AdapterDependencyError) as raised:
        factory({})
    expected_version = "3.3.0" if package == "xgboost" else "1.2.10"
    message = str(raised.value)
    assert adapter_id in message
    assert package in message
    assert expected_version in message
    assert "restore the locked project environment" in message
    assert raised.value.__cause__ is original


@pytest.mark.parametrize(
    "adapter_id,package",
    [
        ("xgboost_numeric_v1", "xgboost"),
        ("catboost_numeric_v1", "catboost"),
    ],
)
def test_missing_adapter_distribution_has_precise_dependency_error(
    adapter_id: str,
    package: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = importlib.metadata.PackageNotFoundError(package)

    def missing_version(name: str) -> str:
        assert name == package
        raise original

    monkeypatch.setattr(
        experiment_v2_numeric_adapter.importlib.metadata,
        "version",
        missing_version,
    )
    with pytest.raises(ExperimentV2AdapterDependencyError) as raised:
        experiment_v2_numeric_adapter.adapter_dependency_version(adapter_id)
    expected_version = "3.3.0" if package == "xgboost" else "1.2.10"
    message = str(raised.value)
    assert adapter_id in message
    assert package in message
    assert expected_version in message
    assert "restore the locked project environment" in message
    assert raised.value.__cause__ is original


@pytest.mark.parametrize(
    "factory,package",
    [
        (xgboost_estimator_factory, "xgboost"),
        (catboost_estimator_factory, "catboost"),
    ],
)
def test_nested_module_import_failure_is_preserved(
    factory: Any,
    package: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ModuleNotFoundError(
        "No module named 'package_internal_dependency'",
        name="package_internal_dependency",
    )

    def nested_failure(name: str) -> Any:
        assert name == package
        raise original

    monkeypatch.setattr(
        experiment_v2_numeric_adapter.importlib,
        "import_module",
        nested_failure,
    )
    with pytest.raises(ModuleNotFoundError) as raised:
        factory({})
    assert raised.value is original


def test_adapter_modules_do_not_eagerly_import_model_libraries() -> None:
    assert "xgboost" not in sys.modules
    assert "catboost" not in sys.modules


@pytest.mark.skipif(not TRAIN_PATH.exists(), reason="train-only dataset unavailable")
def test_preflight_uses_precise_adapter_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_version = importlib.metadata.version
    original = importlib.metadata.PackageNotFoundError("xgboost")

    def selected_missing_version(name: str) -> str:
        if name == "xgboost":
            raise original
        return original_version(name)

    monkeypatch.setattr(
        experiment_v2_numeric_adapter.importlib.metadata,
        "version",
        selected_missing_version,
    )
    with pytest.raises(ExperimentV2AdapterDependencyError) as raised:
        research_v2_cli.preflight(XGBOOST_SMOKE_CONFIG)
    assert "xgboost_numeric_v1" in str(raised.value)
    assert raised.value.__cause__ is original


@pytest.mark.skipif(not TRAIN_PATH.exists(), reason="train-only dataset unavailable")
def test_manual_lightgbm_identity_hashes_are_pinned() -> None:
    prepared = research_v2_cli.preflight(MANUAL_CONFIG)
    assert prepared.hashes["candidate_adapter"] == (
        "9b01221fa236dd4e589ea6156642e57e7eeb4a0d95cbf07e1d23280c2e26f3e8"
    ), "manual LightGBM adapter identity drift"
    assert prepared.hashes["candidate"] == (
        "5b3daa881c73f8901eed705a2bd68da81d79a8806570597d183d0b67100c286f"
    ), "manual LightGBM candidate identity drift"


@pytest.mark.parametrize("adapter_id", ["xgboost_numeric_v1", "catboost_numeric_v1"])
def test_tiny_real_cpu_fit_is_deterministic_and_writes_no_files(
    adapter_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("xgboost" if adapter_id.startswith("xgboost") else "catboost")
    train, labels, prediction = _data()
    contract = _contract(adapter_id.startswith("xgboost"))
    model_key = "xgboost" if adapter_id.startswith("xgboost") else "catboost"
    count_key = "n_estimators" if model_key == "xgboost" else "iterations"
    contract[model_key]["parameters"][count_key] = 5
    monkeypatch.chdir(tmp_path)
    adapter = get_candidate_adapter(adapter_id)

    def run() -> np.ndarray:
        return adapter.fit_predict(
            train,
            labels,
            prediction,
            contract,
            model_training_positions=np.arange(20),
            prediction_positions=np.array([91, 17, 44]),
        )

    first = run()
    second = run()
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()
    assert ((first >= 0.0) & (first <= 1.0)).all()
    assert list(tmp_path.iterdir()) == []
