from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd

from src.churn_ml.config import EXPECTED_LIGHTGBM_PARAMETERS
from src.churn_ml.experiment_v2_contract import (
    ExperimentV2ContractError,
    first_exact_difference,
)
from src.churn_ml.research_manual_lightgbm import fit_predict_manual_candidate


MANUAL_LIGHTGBM_TE_V1_COMPAT = "manual_lightgbm_te_v1_compat"
EXPECTED_ADAPTER_CONTRACT = {
    "target_encoder": {
        "implementation": "custom_autogluon_compatible_binary_oof_target_encoder",
        "inner_splits": 5,
        "shuffle": True,
        "random_state": 42,
        "alpha": 10.0,
        "prior": "unweighted_mean_of_category_level_target_means",
        "keep_original_categorical_features": False,
    },
    "lightgbm": {
        "estimator": "LGBMClassifier",
        "parameters": EXPECTED_LIGHTGBM_PARAMETERS,
    },
}


class ExperimentV2AdapterContractError(ExperimentV2ContractError):
    """Raised when a v2 candidate adapter violates its frozen contract."""


class CandidateAdapter(Protocol):
    id: str

    def validate_contract(self, contract: Mapping[str, Any]) -> None: ...

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
    ) -> np.ndarray: ...

    def identity_inputs(self, contract: Mapping[str, Any]) -> dict[str, Any]: ...


class ManualLightgbmTeV1CompatAdapter:
    id = MANUAL_LIGHTGBM_TE_V1_COMPAT

    def validate_contract(self, contract: Mapping[str, Any]) -> None:
        difference = first_exact_difference(
            contract,
            EXPECTED_ADAPTER_CONTRACT,
            "candidate_adapter.contract",
        )
        if difference is not None:
            raise ExperimentV2AdapterContractError(difference)

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
        if len(train_features) != len(train_labels):
            raise ExperimentV2AdapterContractError(
                "Training features and labels are not aligned."
            )
        if list(train_features.columns) != list(prediction_features.columns):
            raise ExperimentV2AdapterContractError(
                "Training and prediction feature schemas differ."
            )
        probabilities = fit_predict_manual_candidate(
            train_features,
            train_labels,
            prediction_features,
            deepcopy(dict(contract)),
            model_training_positions=model_training_positions,
            prediction_positions=prediction_positions,
            audit_callback=audit_callback,
        )
        return validate_positive_class_probabilities(
            probabilities,
            expected_rows=len(prediction_features),
        )

    def identity_inputs(self, contract: Mapping[str, Any]) -> dict[str, Any]:
        self.validate_contract(contract)
        return {
            "id": self.id,
            "contract": deepcopy(dict(contract)),
            "probability_semantics": "binary_positive_class_label_1",
        }


def validate_positive_class_probabilities(
    values: Any,
    *,
    expected_rows: int,
) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 1 or len(probabilities) != expected_rows:
        raise ExperimentV2AdapterContractError(
            "Adapter must return one probability per prediction row."
        )
    if not np.isfinite(probabilities).all():
        raise ExperimentV2AdapterContractError("Adapter probabilities must be finite.")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ExperimentV2AdapterContractError(
            "Adapter probabilities must be within [0, 1]."
        )
    return probabilities


_ADAPTERS: Mapping[str, CandidateAdapter] = MappingProxyType(
    {MANUAL_LIGHTGBM_TE_V1_COMPAT: ManualLightgbmTeV1CompatAdapter()}
)


def candidate_adapter_registry() -> Mapping[str, CandidateAdapter]:
    return _ADAPTERS


def get_candidate_adapter(adapter_id: str) -> CandidateAdapter:
    try:
        return _ADAPTERS[adapter_id]
    except KeyError as error:
        raise ExperimentV2AdapterContractError(
            f"Unknown candidate adapter ID: {adapter_id}."
        ) from error
