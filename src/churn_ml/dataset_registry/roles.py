"""Deterministic feature-role classification for Dataset Registry v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Sequence

from src.churn_ml.dataset_registry.constants import (
    V1_ENGINEERED_FEATURES,
    V2_SUMMARY_FEATURES,
    V3_ENGINEERED_FEATURES,
    V4_SUMMARY_FEATURES,
)
from src.churn_ml.dataset_registry.errors import DatasetRegistryError
from src.churn_ml.dataset_registry.schema import FeatureRole

# Precedence (highest first): summary > binary_indicator > categorical > numeric.
# Catalog/name membership wins over dtype inference. Classification never inspects
# y_train values, X_test values, correlations, or cardinality.
ROLE_PRECEDENCE: tuple[FeatureRole, ...] = (
    "summary",
    "binary_indicator",
    "categorical",
    "numeric",
)

BINARY_INDICATOR_SUFFIX = "_is_missing"

_CATEGORICAL_DTYPE_PREFIXES = ("object", "category", "string")
_NUMERIC_DTYPE_PREFIXES = (
    "int",
    "uint",
    "float",
    "bool",
    "boolean",
    "Float",
    "Int",
    "UInt",
)


@dataclass(frozen=True)
class RoleCatalog:
    """Declared name sets used before dtype-based role assignment."""

    summary_features: frozenset[str]
    binary_indicator_features: frozenset[str]
    use_missing_suffix: bool = False


def legacy_role_catalog(dataset_id: str) -> RoleCatalog:
    if dataset_id == "v0_raw_minimal":
        return RoleCatalog(
            summary_features=frozenset(),
            binary_indicator_features=frozenset(),
            use_missing_suffix=False,
        )
    if dataset_id == "v1_missingness_summary":
        return RoleCatalog(
            summary_features=frozenset(V1_ENGINEERED_FEATURES),
            binary_indicator_features=frozenset(),
            use_missing_suffix=False,
        )
    if dataset_id == "v2_missingness_indicators":
        return RoleCatalog(
            summary_features=frozenset(V2_SUMMARY_FEATURES),
            binary_indicator_features=frozenset(),
            use_missing_suffix=True,
        )
    if dataset_id == "v3_targeted_missingness":
        return RoleCatalog(
            summary_features=frozenset(V1_ENGINEERED_FEATURES),
            binary_indicator_features=frozenset(V3_ENGINEERED_FEATURES),
            use_missing_suffix=False,
        )
    if dataset_id == "v4_zero_value_summary":
        return RoleCatalog(
            summary_features=frozenset(V4_SUMMARY_FEATURES),
            binary_indicator_features=frozenset(),
            use_missing_suffix=False,
        )
    raise DatasetRegistryError(
        f"No legacy role catalog for dataset_id {dataset_id!r}."
    )


def classify_feature_role(
    name: str,
    dtype: str,
    *,
    summary_features: AbstractSet[str] = frozenset(),
    binary_indicator_features: AbstractSet[str] = frozenset(),
    use_missing_suffix: bool = False,
) -> FeatureRole:
    """
    Assign a feature role deterministically.

    Precedence:
    1. summary — name listed in declared summary features
    2. binary_indicator — name listed in declared indicators, or (when enabled)
       name ending with ``_is_missing``
    3. categorical — train column dtype is object/category/string
    4. numeric — remaining supported numeric/bool dtypes

    Raises DatasetRegistryError for unsupported dtypes after catalog checks.
    """
    if name in summary_features:
        return "summary"
    if name in binary_indicator_features:
        return "binary_indicator"
    if use_missing_suffix and name.endswith(BINARY_INDICATOR_SUFFIX):
        return "binary_indicator"
    if is_categorical_dtype_name(dtype):
        return "categorical"
    if is_numeric_or_bool_dtype_name(dtype):
        return "numeric"
    raise DatasetRegistryError(
        f"Unsupported feature dtype for role classification: "
        f"name={name!r}, dtype={dtype!r}."
    )


def classify_feature_role_with_catalog(
    name: str,
    dtype: str,
    catalog: RoleCatalog,
) -> FeatureRole:
    return classify_feature_role(
        name,
        dtype,
        summary_features=catalog.summary_features,
        binary_indicator_features=catalog.binary_indicator_features,
        use_missing_suffix=catalog.use_missing_suffix,
    )


def is_categorical_dtype_name(dtype: str) -> bool:
    text = str(dtype)
    return any(
        text == prefix or text.startswith(f"{prefix}[")
        for prefix in _CATEGORICAL_DTYPE_PREFIXES
    )


def is_numeric_or_bool_dtype_name(dtype: str) -> bool:
    text = str(dtype)
    if text in {"bool", "boolean"}:
        return True
    return any(text.startswith(prefix) for prefix in _NUMERIC_DTYPE_PREFIXES)


def role_catalog_from_sets(
    *,
    summary_features: Sequence[str] | None = None,
    binary_indicator_features: Sequence[str] | None = None,
    use_missing_suffix: bool = False,
) -> RoleCatalog:
    return RoleCatalog(
        summary_features=frozenset(summary_features or ()),
        binary_indicator_features=frozenset(binary_indicator_features or ()),
        use_missing_suffix=use_missing_suffix,
    )
