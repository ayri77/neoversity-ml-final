from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from src.churn_ml.experiment_v2_adapter import (
    CandidateAdapter,
    ExperimentV2AdapterContractError,
    candidate_adapter_registry as legacy_candidate_adapter_registry,
)
from src.churn_ml.experiment_v2_catboost_adapter import CatboostNumericV1Adapter
from src.churn_ml.experiment_v2_xgboost_adapter import XgboostNumericV1Adapter


_ADAPTERS: Mapping[str, CandidateAdapter] = MappingProxyType(
    {
        **legacy_candidate_adapter_registry(),
        XgboostNumericV1Adapter.id: XgboostNumericV1Adapter(),
        CatboostNumericV1Adapter.id: CatboostNumericV1Adapter(),
    }
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
