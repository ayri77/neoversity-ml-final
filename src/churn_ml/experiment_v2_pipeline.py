from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol

import pandas as pd

from src.churn_ml.experiment_v2_contract import (
    ExperimentV2ContractError,
    first_exact_difference,
)
from src.churn_ml.experiment_v2_schema import FeatureSchema


MANUAL_V3_PIPELINE_V1_COMPAT = "manual_v3_pipeline_v1_compat"

EXPECTED_DROPS = ["Var214", "Var220", "Var222", "Var218_is_missing"]
EXPECTED_CATEGORICAL = [
    "Var192",
    "Var193",
    "Var194",
    "Var195",
    "Var196",
    "Var197",
    "Var198",
    "Var199",
    "Var200",
    "Var201",
    "Var202",
    "Var203",
    "Var204",
    "Var205",
    "Var206",
    "Var207",
    "Var208",
    "Var210",
    "Var211",
    "Var212",
    "Var216",
    "Var217",
    "Var218",
    "Var219",
    "Var221",
    "Var223",
    "Var225",
    "Var226",
    "Var227",
    "Var228",
    "Var229",
]
EXPECTED_PIPELINE_CONTRACT = {
    "source_feature_count": 217,
    "drop": EXPECTED_DROPS,
    "categorical": EXPECTED_CATEGORICAL,
    "expected_model_feature_count": 213,
    "expected_source_schema_sha256": (
        "fa4cd8017a05520ca7f6c0fb81fb2f05b9cabe572e757eb43e424befebb9c62e"
    ),
    "expected_model_input_schema_sha256": (
        "e19efe336426c09071db296920353a0b967284a1288a1d9ab54aaa83ca179616"
    ),
    "expected_transformed_schema_sha256": (
        "7c1fe2610edaa837baf4f041faf024361b896bb52e1fd81121c5ac6652180c41"
    ),
}


class ExperimentV2PipelineContractError(ExperimentV2ContractError):
    """Raised when a v2 feature pipeline violates its frozen contract."""


@dataclass(frozen=True)
class PipelineOutput:
    features: pd.DataFrame
    schema: FeatureSchema


class FeaturePipeline(Protocol):
    id: str

    def validate_contract(self, contract: Mapping[str, Any]) -> None: ...

    def transform(
        self,
        features: pd.DataFrame,
        contract: Mapping[str, Any],
    ) -> PipelineOutput: ...

    def identity_inputs(self, contract: Mapping[str, Any]) -> dict[str, Any]: ...


class ManualV3PipelineV1Compat:
    id = MANUAL_V3_PIPELINE_V1_COMPAT

    def validate_contract(self, contract: Mapping[str, Any]) -> None:
        difference = first_exact_difference(
            contract,
            EXPECTED_PIPELINE_CONTRACT,
            "feature_pipeline.contract",
        )
        if difference is not None:
            raise ExperimentV2PipelineContractError(difference)

    def transform(
        self,
        features: pd.DataFrame,
        contract: Mapping[str, Any],
    ) -> PipelineOutput:
        self.validate_contract(contract)
        if features.shape[1] != contract["source_feature_count"]:
            raise ExperimentV2PipelineContractError(
                "feature_pipeline.input.source_feature_count differs."
            )
        drops = list(contract["drop"])
        missing = [name for name in drops if name not in features.columns]
        if missing:
            raise ExperimentV2PipelineContractError(
                f"feature_pipeline.input.drop features are missing: {missing}."
            )
        model_features = features.drop(columns=drops)
        categorical = model_features.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        numerical = model_features.select_dtypes(
            include=["number", "bool"]
        ).columns.tolist()
        unsupported = [
            name
            for name in model_features.columns
            if name not in categorical and name not in numerical
        ]
        if unsupported:
            raise ExperimentV2PipelineContractError(
                f"feature_pipeline.input has unsupported dtypes: {unsupported}."
            )
        if categorical != contract["categorical"]:
            raise ExperimentV2PipelineContractError(
                "feature_pipeline.input categorical names or order differ."
            )
        transformed = numerical + [f"{name}__te" for name in categorical]
        schema = FeatureSchema(
            source_feature_names=features.columns.tolist(),
            dropped_features=drops,
            model_feature_names=model_features.columns.tolist(),
            categorical_features=categorical,
            numerical_features=numerical,
            transformed_feature_names=transformed,
        )
        actual = {
            "expected_source_schema_sha256": schema.source_schema_sha256,
            "expected_model_input_schema_sha256": schema.model_input_schema_sha256,
            "expected_transformed_schema_sha256": schema.transformed_schema_sha256,
        }
        mismatches = [
            f"{name}: expected={contract[name]}, actual={value}"
            for name, value in actual.items()
            if contract[name] != value
        ]
        if mismatches:
            raise ExperimentV2PipelineContractError(
                "Ordered pipeline schema mismatch:\n- " + "\n- ".join(mismatches)
            )
        return PipelineOutput(features=model_features, schema=schema)

    def identity_inputs(self, contract: Mapping[str, Any]) -> dict[str, Any]:
        self.validate_contract(contract)
        return {"id": self.id, "contract": deepcopy(dict(contract))}


_PIPELINES: Mapping[str, FeaturePipeline] = MappingProxyType(
    {MANUAL_V3_PIPELINE_V1_COMPAT: ManualV3PipelineV1Compat()}
)


def feature_pipeline_registry() -> Mapping[str, FeaturePipeline]:
    return _PIPELINES


def get_feature_pipeline(pipeline_id: str) -> FeaturePipeline:
    try:
        return _PIPELINES[pipeline_id]
    except KeyError as error:
        raise ExperimentV2PipelineContractError(
            f"Unknown feature pipeline ID: {pipeline_id}."
        ) from error
