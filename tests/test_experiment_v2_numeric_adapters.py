from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.churn_ml.experiment_v2 import (
    candidate_adapter_registry,
    get_candidate_adapter,
)
from src.churn_ml.experiment_v2_adapter import ExperimentV2AdapterContractError
from src.churn_ml.experiment_v2_catboost_adapter import (
    EXPECTED_CATBOOST_NUMERIC_V1_CONTRACT,
    CatboostNumericV1Adapter,
)
from src.churn_ml.experiment_v2_numeric_adapter import validate_numeric_matrix
from src.churn_ml.experiment_v2_xgboost_adapter import (
    EXPECTED_XGBOOST_NUMERIC_V1_CONTRACT,
    XgboostNumericV1Adapter,
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
    contract = deepcopy(
        EXPECTED_XGBOOST_NUMERIC_V1_CONTRACT
        if model_key == "xgboost"
        else EXPECTED_CATBOOST_NUMERIC_V1_CONTRACT
    )
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
        without_runtime_missing = dict(created[0].parameters)
        del without_runtime_missing["missing"]
        assert without_runtime_missing == contract[model_key]["parameters"]
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
    contract = deepcopy(
        EXPECTED_XGBOOST_NUMERIC_V1_CONTRACT
        if adapter_id.startswith("xgboost")
        else EXPECTED_CATBOOST_NUMERIC_V1_CONTRACT
    )
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
    contract = deepcopy(
        EXPECTED_XGBOOST_NUMERIC_V1_CONTRACT
        if adapter_id.startswith("xgboost")
        else EXPECTED_CATBOOST_NUMERIC_V1_CONTRACT
    )
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
    contract = deepcopy(
        EXPECTED_XGBOOST_NUMERIC_V1_CONTRACT
        if adapter_id.startswith("xgboost")
        else EXPECTED_CATBOOST_NUMERIC_V1_CONTRACT
    )
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
    contract = deepcopy(
        EXPECTED_XGBOOST_NUMERIC_V1_CONTRACT
        if model_key == "xgboost"
        else EXPECTED_CATBOOST_NUMERIC_V1_CONTRACT
    )
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
    contract = deepcopy(
        EXPECTED_XGBOOST_NUMERIC_V1_CONTRACT
        if model_key == "xgboost"
        else EXPECTED_CATBOOST_NUMERIC_V1_CONTRACT
    )
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
    adapter_inputs = adapter.identity_inputs(
        deepcopy(EXPECTED_XGBOOST_NUMERIC_V1_CONTRACT)
    )

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


@pytest.mark.parametrize("adapter_id", ["xgboost_numeric_v1", "catboost_numeric_v1"])
def test_tiny_real_cpu_fit_is_deterministic_and_writes_no_files(
    adapter_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("xgboost" if adapter_id.startswith("xgboost") else "catboost")
    train, labels, prediction = _data()
    contract = deepcopy(
        EXPECTED_XGBOOST_NUMERIC_V1_CONTRACT
        if adapter_id.startswith("xgboost")
        else EXPECTED_CATBOOST_NUMERIC_V1_CONTRACT
    )
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
