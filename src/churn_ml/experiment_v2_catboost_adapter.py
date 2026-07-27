from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from src.churn_ml.experiment_v2_adapter import (
    ExperimentV2AdapterContractError,
    validate_positive_class_probabilities,
)
from src.churn_ml.experiment_v2_contract import first_exact_difference
from src.churn_ml.experiment_v2_numeric_adapter import (
    EXPECTED_NUMERIC_FEATURE_CONTRACT,
    build_numeric_matrices,
    load_adapter_class,
    numeric_identity_inputs,
    positive_class_probabilities,
    validate_adapter_inputs,
)


CATBOOST_NUMERIC_V1 = "catboost_numeric_v1"
CATBOOST_NUMERIC_V1_DEFAULT_PARAMETERS = {
    "iterations": 300,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "bagging_temperature": 1.0,
    "border_count": 254,
    "random_seed": 42,
    "thread_count": 1,
    "bootstrap_type": "Bayesian",
    "grow_policy": "SymmetricTree",
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "nan_mode": "Min",
    "task_type": "CPU",
    "allow_writing_files": False,
    "verbose": False,
}
EXPECTED_CATBOOST_NUMERIC_V1_CONTRACT = {
    "numeric_features": EXPECTED_NUMERIC_FEATURE_CONTRACT,
    "catboost": {
        "estimator": "CatBoostClassifier",
        "parameters": CATBOOST_NUMERIC_V1_DEFAULT_PARAMETERS,
    },
}

EstimatorFactory = Callable[[dict[str, Any]], Any]


def _default_estimator_factory(parameters: dict[str, Any]) -> Any:
    estimator_class = load_adapter_class(CATBOOST_NUMERIC_V1, "CatBoostClassifier")
    return estimator_class(**parameters)


class CatboostNumericV1Adapter:
    id = CATBOOST_NUMERIC_V1

    def __init__(
        self,
        estimator_factory: EstimatorFactory = _default_estimator_factory,
    ) -> None:
        self._estimator_factory = estimator_factory

    def validate_contract(self, contract: Mapping[str, Any]) -> None:
        _exact_keys(
            contract,
            set(EXPECTED_CATBOOST_NUMERIC_V1_CONTRACT),
            "candidate_adapter.contract",
        )
        difference = first_exact_difference(
            contract["numeric_features"],
            EXPECTED_NUMERIC_FEATURE_CONTRACT,
            "candidate_adapter.contract.numeric_features",
        )
        if difference is not None:
            raise ExperimentV2AdapterContractError(difference)
        model = contract["catboost"]
        if not isinstance(model, Mapping):
            raise ExperimentV2AdapterContractError(
                "candidate_adapter.contract.catboost must be a mapping."
            )
        _exact_keys(
            model, {"estimator", "parameters"}, "candidate_adapter.contract.catboost"
        )
        if (
            type(model["estimator"]) is not str
            or model["estimator"] != "CatBoostClassifier"
        ):
            raise ExperimentV2AdapterContractError(
                "candidate_adapter.contract.catboost.estimator must be exactly "
                "'CatBoostClassifier'."
            )
        parameters = model["parameters"]
        if not isinstance(parameters, Mapping):
            raise ExperimentV2AdapterContractError(
                "candidate_adapter.contract.catboost.parameters must be a mapping."
            )
        _exact_keys(
            parameters,
            set(CATBOOST_NUMERIC_V1_DEFAULT_PARAMETERS),
            "candidate_adapter.contract.catboost.parameters",
        )
        _validate_parameters(parameters)

    def fit_predict(
        self,
        train_features: pd.DataFrame,
        train_labels: pd.Series,
        prediction_features: pd.DataFrame,
        contract: Mapping[str, Any],
        *,
        model_training_positions: np.ndarray,
        prediction_positions: np.ndarray,
        audit_callback: Any = None,
    ) -> np.ndarray:
        self.validate_contract(contract)
        validate_adapter_inputs(
            train_features,
            train_labels,
            prediction_features,
        )
        if audit_callback is not None:
            audit_callback(
                np.asarray(model_training_positions, dtype=np.int64),
                np.asarray(prediction_positions, dtype=np.int64),
            )
        encoded_train, encoded_prediction = build_numeric_matrices(
            train_features,
            train_labels,
            prediction_features,
            contract["numeric_features"],
        )
        parameters = deepcopy(dict(contract["catboost"]["parameters"]))
        estimator = self._estimator_factory(parameters)
        estimator.fit(encoded_train, train_labels.reset_index(drop=True))
        probabilities = positive_class_probabilities(estimator, encoded_prediction)
        return validate_positive_class_probabilities(
            probabilities,
            expected_rows=len(prediction_features),
        )

    def identity_inputs(self, contract: Mapping[str, Any]) -> dict[str, Any]:
        self.validate_contract(contract)
        return numeric_identity_inputs(
            adapter_id=self.id,
            contract=contract,
            estimator="catboost.CatBoostClassifier",
            native_missing_configuration={
                "nan_mode": str(contract["catboost"]["parameters"]["nan_mode"])
            },
        )


def _validate_parameters(parameters: Mapping[str, Any]) -> None:
    _integer_range(parameters, "iterations", 1, 100_000)
    _float_range(parameters, "learning_rate", 0.0, 1.0, lower_open=True)
    _integer_range(parameters, "depth", 1, 16)
    _float_range(parameters, "l2_leaf_reg", 0.0, None)
    _float_range(parameters, "random_strength", 0.0, None)
    _float_range(parameters, "bagging_temperature", 0.0, None)
    _integer_range(parameters, "border_count", 1, 65_535)
    _integer_range(parameters, "random_seed", 0, 2**31 - 1)
    _integer_range(parameters, "thread_count", 1, 256)
    for key, expected in {
        "bootstrap_type": "Bayesian",
        "grow_policy": "SymmetricTree",
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "nan_mode": "Min",
        "task_type": "CPU",
    }.items():
        if type(parameters[key]) is not str or parameters[key] != expected:
            raise ExperimentV2AdapterContractError(
                f"candidate_adapter.contract.catboost.parameters.{key} "
                f"must be exactly {expected!r}."
            )
    for key in ("allow_writing_files", "verbose"):
        if type(parameters[key]) is not bool or parameters[key] is not False:
            raise ExperimentV2AdapterContractError(
                f"candidate_adapter.contract.catboost.parameters.{key} "
                "must be exactly false."
            )


def _integer_range(
    parameters: Mapping[str, Any],
    key: str,
    minimum: int,
    maximum: int,
) -> None:
    value = parameters[key]
    if type(value) is not int or not minimum <= value <= maximum:
        raise ExperimentV2AdapterContractError(
            f"candidate_adapter.contract.catboost.parameters.{key} "
            f"must be an integer in [{minimum}, {maximum}]."
        )


def _float_range(
    parameters: Mapping[str, Any],
    key: str,
    minimum: float,
    maximum: float | None,
    *,
    lower_open: bool = False,
) -> None:
    value = parameters[key]
    valid = type(value) is float and math.isfinite(value)
    if valid:
        valid = value > minimum if lower_open else value >= minimum
        valid = valid and (maximum is None or value <= maximum)
    if not valid:
        interval = (
            f"({minimum}, {maximum}]" if lower_open else f"[{minimum}, {maximum}]"
        )
        raise ExperimentV2AdapterContractError(
            f"candidate_adapter.contract.catboost.parameters.{key} "
            f"must be a finite float in {interval}."
        )


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ExperimentV2AdapterContractError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}."
        )
