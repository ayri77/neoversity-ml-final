from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from src.churn_ml.deployment_v1_contracts import (
    CandidateApproval,
    DeploymentContractError,
    model_parameters,
)
from src.churn_ml.deployment_v1_features import EncodingResult
from src.churn_ml.experiment_v2_adapter import validate_positive_class_probabilities
from src.churn_ml.experiment_v2_numeric_adapter import (
    ADAPTER_DEPENDENCIES,
    positive_class_probabilities,
)
from src.churn_ml.research_data import canonical_sha256


EstimatorFactory = Callable[[str, dict[str, Any]], Any]


class DeploymentModelError(ValueError):
    """Raised when fixed full-data bagging or blending violates P4 semantics."""


@dataclass(frozen=True)
class ComponentPrediction:
    component_id: str
    adapter_id: str
    probabilities: np.ndarray
    bag_records: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


def fit_component_bags(
    component: Mapping[str, Any],
    approval: CandidateApproval,
    encoding: EncodingResult,
    y_train: pd.Series,
    *,
    estimator_factory: EstimatorFactory | None = None,
) -> ComponentPrediction:
    adapter_id = str(component["adapter_id"])
    contract = approval.research_run.config.adapter_contract
    base_parameters = model_parameters(adapter_id, contract)
    if base_parameters != dict(component["fixed_parameters"]):
        raise DeploymentModelError("Component parameters differ from approval.")
    _validate_runtime_version(approval)
    factory = estimator_factory or _default_estimator_factory
    bag_probabilities: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    seed_key = _seed_key(adapter_id)
    for bag_index, seed in enumerate(component["bag_seeds"], start=1):
        parameters = deepcopy(base_parameters)
        parameters[seed_key] = int(seed)
        _assert_seed_only_change(base_parameters, parameters, seed_key)
        native_parameters = _native_parameters(adapter_id, parameters)
        estimator = factory(adapter_id, native_parameters)
        started = perf_counter()
        estimator.fit(
            encoding.train,
            y_train.reset_index(drop=True),
        )
        values = positive_class_probabilities(estimator, encoding.test)
        values = validate_positive_class_probabilities(
            values,
            expected_rows=len(encoding.test),
        )
        probability_hash = probability_sha256(values)
        parameter_identity = {
            "schema_version": 1,
            "adapter_id": adapter_id,
            "bag_seed": int(seed),
            "seed_parameter": seed_key,
            "parameters": parameters,
        }
        records.append(
            {
                "component_id": str(component["component_id"]),
                "adapter_id": adapter_id,
                "bag_index": bag_index,
                "bag_seed": int(seed),
                "training_rows": len(encoding.train),
                "test_rows": len(encoding.test),
                "parameter_sha256": canonical_sha256(parameter_identity),
                "probability_sha256": probability_hash,
                "duration_seconds": perf_counter() - started,
                "early_stopping": False,
                "evaluation_set": False,
                "model_persisted": False,
            }
        )
        bag_probabilities.append(values)
    stacked = np.vstack(bag_probabilities)
    average = np.asarray(np.mean(stacked, axis=0, dtype=np.float64), dtype=np.float64)
    validate_positive_class_probabilities(average, expected_rows=len(encoding.test))
    summary = {
        "schema_version": 1,
        "component_id": str(component["component_id"]),
        "adapter_id": adapter_id,
        "bag_seeds": list(component["bag_seeds"]),
        "bag_count": len(bag_probabilities),
        "training_rows_per_bag": len(encoding.train),
        "all_training_rows_used": True,
        "aggregation": "arithmetic_mean",
        "component_probability_sha256": probability_sha256(average),
        "runtime_library_version": _runtime_library_version(adapter_id),
        "row_order_sha256": canonical_sha256(list(range(len(average)))),
        "model_persistence": False,
    }
    return ComponentPrediction(
        component_id=str(component["component_id"]),
        adapter_id=adapter_id,
        probabilities=average,
        bag_records=tuple(records),
        summary=summary,
    )


def build_fixed_blend(
    components: list[ComponentPrediction],
    weights: Mapping[str, float],
) -> np.ndarray:
    if not components:
        raise DeploymentModelError("At least one component is required.")
    expected_rows = len(components[0].probabilities)
    accumulator = np.zeros(expected_rows, dtype=np.float64)
    seen: set[str] = set()
    for component in components:
        if component.component_id in seen:
            raise DeploymentModelError("Duplicate blend component ID.")
        seen.add(component.component_id)
        if len(component.probabilities) != expected_rows:
            raise DeploymentModelError("Component probability row alignment differs.")
        values = validate_positive_class_probabilities(
            component.probabilities,
            expected_rows=expected_rows,
        )
        weight = weights.get(component.component_id)
        if type(weight) is not float or not np.isfinite(weight) or weight < 0.0:
            raise DeploymentModelError("Blend weight is invalid.")
        accumulator += weight * values
    if set(weights) != seen:
        raise DeploymentModelError("Blend weights and component IDs differ.")
    if not np.isclose(sum(weights.values()), 1.0, rtol=0.0, atol=1e-12):
        raise DeploymentModelError("Blend weights do not sum to 1.")
    return validate_positive_class_probabilities(
        accumulator,
        expected_rows=expected_rows,
    )


def classify_fixed(probabilities: Any, threshold: float) -> np.ndarray:
    values = validate_positive_class_probabilities(
        probabilities,
        expected_rows=len(probabilities),
    )
    if type(threshold) is not float or not np.isfinite(threshold):
        raise DeploymentModelError("Threshold must be a finite float.")
    if not 0.0 <= threshold <= 1.0:
        raise DeploymentModelError("Threshold must be within [0, 1].")
    return (values >= threshold).astype(np.int8)


def probability_sha256(values: Any) -> str:
    array = np.asarray(values, dtype="<f8")
    if array.ndim != 1:
        raise DeploymentModelError("Probability hash requires a vector.")
    digest = hashlib.sha256()
    digest.update(b"p4_probability_vector_float64_v1")
    digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _default_estimator_factory(adapter_id: str, parameters: dict[str, Any]) -> Any:
    module_name, class_name = {
        "manual_lightgbm_te_v1_compat": ("lightgbm", "LGBMClassifier"),
        "xgboost_numeric_v1": ("xgboost", "XGBClassifier"),
        "catboost_numeric_v1": ("catboost", "CatBoostClassifier"),
    }[adapter_id]
    estimator_class = getattr(importlib.import_module(module_name), class_name)
    return estimator_class(**parameters)


def _native_parameters(adapter_id: str, parameters: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(parameters)
    if adapter_id == "xgboost_numeric_v1":
        if result.pop("missing") != "IEEE_NaN":
            raise DeploymentModelError("XGBoost native missing contract differs.")
        result["missing"] = np.nan
    if adapter_id == "catboost_numeric_v1":
        if result.get("allow_writing_files") is not False:
            raise DeploymentModelError("CatBoost file writing must be disabled.")
    forbidden = {
        "early_stopping_rounds",
        "callbacks",
        "eval_set",
        "use_best_model",
    }
    if forbidden.intersection(result):
        raise DeploymentModelError("Deployment parameters contain fitting controls.")
    return result


def _seed_key(adapter_id: str) -> str:
    return {
        "manual_lightgbm_te_v1_compat": "random_state",
        "xgboost_numeric_v1": "random_state",
        "catboost_numeric_v1": "random_seed",
    }[adapter_id]


def _assert_seed_only_change(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    seed_key: str,
) -> None:
    if set(before) != set(after):
        raise DeploymentModelError("Bag parameters changed shape.")
    differing = {key for key in before if before[key] != after[key]}
    if not differing.issubset({seed_key}):
        raise DeploymentModelError("Only the approved model seed may vary by bag.")


def _runtime_library_version(adapter_id: str) -> str:
    distribution = {
        "manual_lightgbm_te_v1_compat": "lightgbm",
        "xgboost_numeric_v1": "xgboost",
        "catboost_numeric_v1": "catboost",
    }[adapter_id]
    return importlib.metadata.version(distribution)


def _validate_runtime_version(approval: CandidateApproval) -> None:
    adapter_id = approval.adapter_id
    current = _runtime_library_version(adapter_id)
    adapter_identity = (
        approval.research_run.root / "identities" / "candidate_adapter.json"
    )
    import json

    payload = json.loads(adapter_identity.read_text(encoding="utf-8"))
    dependencies = payload["canonical"]["runtime_dependencies"]
    dependency_key = {
        "manual_lightgbm_te_v1_compat": "lightgbm",
        "xgboost_numeric_v1": "xgboost",
        "catboost_numeric_v1": "catboost",
    }[adapter_id]
    approved_version = dependencies[dependency_key]
    if current != approved_version:
        raise DeploymentModelError(
            f"Runtime {dependency_key} version {current!r} differs from "
            f"approved {approved_version!r}."
        )
    if adapter_id in ADAPTER_DEPENDENCIES:
        _, expected = ADAPTER_DEPENDENCIES[adapter_id]
        if current != expected:
            raise DeploymentContractError(
                f"Adapter {adapter_id} requires locked version {expected!r}."
            )
