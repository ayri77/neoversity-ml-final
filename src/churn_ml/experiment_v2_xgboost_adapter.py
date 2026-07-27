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
    numeric_identity_inputs,
    positive_class_probabilities,
    validate_adapter_inputs,
)


XGBOOST_NUMERIC_V1 = "xgboost_numeric_v1"
XGBOOST_NUMERIC_V1_DEFAULT_PARAMETERS = {
    "objective": "binary:logistic",
    "booster": "gbtree",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 1.0,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "max_bin": 256,
    "tree_method": "hist",
    "device": "cpu",
    "random_state": 42,
    "n_jobs": 1,
    "eval_metric": "logloss",
    "verbosity": 0,
}
EXPECTED_XGBOOST_NUMERIC_V1_CONTRACT = {
    "numeric_features": EXPECTED_NUMERIC_FEATURE_CONTRACT,
    "xgboost": {
        "estimator": "XGBClassifier",
        "parameters": XGBOOST_NUMERIC_V1_DEFAULT_PARAMETERS,
    },
}

EstimatorFactory = Callable[[dict[str, Any]], Any]


def _default_estimator_factory(parameters: dict[str, Any]) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(**parameters)


class XgboostNumericV1Adapter:
    id = XGBOOST_NUMERIC_V1

    def __init__(
        self,
        estimator_factory: EstimatorFactory = _default_estimator_factory,
    ) -> None:
        self._estimator_factory = estimator_factory

    def validate_contract(self, contract: Mapping[str, Any]) -> None:
        _exact_keys(
            contract,
            set(EXPECTED_XGBOOST_NUMERIC_V1_CONTRACT),
            "candidate_adapter.contract",
        )
        difference = first_exact_difference(
            contract["numeric_features"],
            EXPECTED_NUMERIC_FEATURE_CONTRACT,
            "candidate_adapter.contract.numeric_features",
        )
        if difference is not None:
            raise ExperimentV2AdapterContractError(difference)
        model = contract["xgboost"]
        if not isinstance(model, Mapping):
            raise ExperimentV2AdapterContractError(
                "candidate_adapter.contract.xgboost must be a mapping."
            )
        _exact_keys(
            model, {"estimator", "parameters"}, "candidate_adapter.contract.xgboost"
        )
        if type(model["estimator"]) is not str or model["estimator"] != "XGBClassifier":
            raise ExperimentV2AdapterContractError(
                "candidate_adapter.contract.xgboost.estimator must be exactly "
                "'XGBClassifier'."
            )
        parameters = model["parameters"]
        if not isinstance(parameters, Mapping):
            raise ExperimentV2AdapterContractError(
                "candidate_adapter.contract.xgboost.parameters must be a mapping."
            )
        _exact_keys(
            parameters,
            set(XGBOOST_NUMERIC_V1_DEFAULT_PARAMETERS),
            "candidate_adapter.contract.xgboost.parameters",
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
        parameters = deepcopy(dict(contract["xgboost"]["parameters"]))
        parameters["missing"] = np.nan
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
            estimator="xgboost.XGBClassifier",
            native_missing_configuration={"missing": "IEEE_NaN"},
        )


def _validate_parameters(parameters: Mapping[str, Any]) -> None:
    _integer_range(parameters, "n_estimators", 1, 100_000)
    _float_range(parameters, "learning_rate", 0.0, 1.0, lower_open=True)
    _integer_range(parameters, "max_depth", 1, 64)
    _float_range(parameters, "min_child_weight", 0.0, None)
    _float_range(parameters, "subsample", 0.0, 1.0, lower_open=True)
    _float_range(parameters, "colsample_bytree", 0.0, 1.0, lower_open=True)
    _float_range(parameters, "gamma", 0.0, None)
    _float_range(parameters, "reg_alpha", 0.0, None)
    _float_range(parameters, "reg_lambda", 0.0, None)
    _integer_range(parameters, "max_bin", 2, 65_536)
    _integer_range(parameters, "random_state", 0, 2**31 - 1)
    _integer_range(parameters, "n_jobs", 1, 256)
    _integer_range(parameters, "verbosity", 0, 3)
    for key, expected in {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "tree_method": "hist",
        "device": "cpu",
        "eval_metric": "logloss",
    }.items():
        if type(parameters[key]) is not str or parameters[key] != expected:
            raise ExperimentV2AdapterContractError(
                f"candidate_adapter.contract.xgboost.parameters.{key} "
                f"must be exactly {expected!r}."
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
            f"candidate_adapter.contract.xgboost.parameters.{key} "
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
            f"candidate_adapter.contract.xgboost.parameters.{key} "
            f"must be a finite float in {interval}."
        )


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ExperimentV2AdapterContractError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}."
        )
