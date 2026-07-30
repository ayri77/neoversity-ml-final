from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

import pandas as pd

from src.churn_ml.experiment_v2_contract import (
    ExperimentV2ContractError,
    first_exact_difference,
)
from src.churn_ml.experiment_v2_schema import FeatureSchema
from src.churn_ml.target_encoding import (
    categorical_feature_names,
    numerical_feature_names,
    unsupported_feature_names,
)


MANUAL_V3_PIPELINE_V1_COMPAT = "manual_v3_pipeline_v1_compat"
REGISTERED_PREPARED_PASSTHROUGH_V1 = "registered_prepared_passthrough_v1"

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

REGISTERED_PREPARED_PASSTHROUGH_CONTRACT = {
    "mode": "registry_prepared_passthrough_v1",
    "drop": [],
    "keep_all_features": True,
}


class ExperimentV2PipelineContractError(ExperimentV2ContractError):
    """Raised when a v2 feature pipeline violates its frozen contract."""


@dataclass(frozen=True)
class PipelineOutput:
    features: pd.DataFrame
    schema: FeatureSchema


def _transformed_feature_name_sources(
    *,
    numerical: Sequence[str],
    categorical: Sequence[str],
) -> dict[str, list[str]]:
    """Map each transformed name to the source features that produce it."""
    sources: dict[str, list[str]] = {}
    for name in numerical:
        sources.setdefault(name, []).append(
            f"passthrough numeric/bool source feature {name!r}"
        )
    for name in categorical:
        transformed = f"{name}__te"
        sources.setdefault(transformed, []).append(
            f"target-encoded categorical source feature {name!r}"
        )
    return sources


def _reject_transformed_feature_name_collisions(
    *,
    transformed: Sequence[str],
    numerical: Sequence[str],
    categorical: Sequence[str],
) -> None:
    """Fail closed when transformed feature names are not unique."""
    counts = Counter(transformed)
    colliding = sorted(name for name, count in counts.items() if count > 1)
    if not colliding:
        return
    sources = _transformed_feature_name_sources(
        numerical=numerical,
        categorical=categorical,
    )
    details = [
        f"{name!r} <- {'; '.join(sources.get(name, ['unknown source']))}"
        for name in colliding
    ]
    raise ExperimentV2PipelineContractError(
        "registered_prepared_passthrough_v1 transformed feature names collide: "
        + "; ".join(details)
        + "."
    )


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
        categorical = categorical_feature_names(model_features)
        numerical = numerical_feature_names(model_features)
        unsupported = unsupported_feature_names(model_features)
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


class RegisteredPreparedPassthroughV1:
    """
    Preserve a Registry-validated prepared dataset without feature mutation.

    Categorical detection includes object, category, and pandas string dtypes.
    Transformed names follow the numeric adapter's fold-local TE naming rule.
    """

    id = REGISTERED_PREPARED_PASSTHROUGH_V1

    def validate_contract(self, contract: Mapping[str, Any]) -> None:
        difference = first_exact_difference(
            contract,
            REGISTERED_PREPARED_PASSTHROUGH_CONTRACT,
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
        drops = list(contract["drop"])
        if drops:
            raise ExperimentV2PipelineContractError(
                "registered_prepared_passthrough_v1 must not drop features."
            )
        if not bool(contract["keep_all_features"]):
            raise ExperimentV2PipelineContractError(
                "registered_prepared_passthrough_v1 must keep all features."
            )
        model_features = features.copy()
        categorical = categorical_feature_names(model_features)
        numerical = numerical_feature_names(model_features)
        unsupported = unsupported_feature_names(model_features)
        if unsupported:
            raise ExperimentV2PipelineContractError(
                "registered_prepared_passthrough_v1 has unsupported dtypes: "
                f"{unsupported}."
            )
        if set(categorical) | set(numerical) != set(model_features.columns):
            raise ExperimentV2PipelineContractError(
                "registered_prepared_passthrough_v1 feature partition is incomplete."
            )
        transformed = numerical + [f"{name}__te" for name in categorical]
        _reject_transformed_feature_name_collisions(
            transformed=transformed,
            numerical=numerical,
            categorical=categorical,
        )
        schema = FeatureSchema(
            source_feature_names=features.columns.tolist(),
            dropped_features=[],
            model_feature_names=model_features.columns.tolist(),
            categorical_features=categorical,
            numerical_features=numerical,
            transformed_feature_names=transformed,
        )
        return PipelineOutput(features=model_features, schema=schema)

    def identity_inputs(self, contract: Mapping[str, Any]) -> dict[str, Any]:
        self.validate_contract(contract)
        return {"id": self.id, "contract": deepcopy(dict(contract))}


_PIPELINES: Mapping[str, FeaturePipeline] = MappingProxyType(
    {
        MANUAL_V3_PIPELINE_V1_COMPAT: ManualV3PipelineV1Compat(),
        REGISTERED_PREPARED_PASSTHROUGH_V1: RegisteredPreparedPassthroughV1(),
    }
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
