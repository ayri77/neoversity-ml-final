"""Constants and legacy catalog for Dataset Package & Registry v1."""

from __future__ import annotations

from typing import Any, Mapping

MANIFEST_FILENAME = "dataset_manifest.json"
SCHEMA_VERSION = "dataset_package_v1"

PACKAGE_FILES = (
    "X_train.parquet",
    "y_train.parquet",
    "X_test.parquet",
    "metadata.json",
)

TARGET_DEPENDENCY_VALUES = frozenset({"none", "exploratory", "fold_local"})
FEATURE_ROLE_VALUES = frozenset(
    {"numeric", "categorical", "binary_indicator", "summary"}
)

# Canonical implemented legacy datasets eligible for safe backfill.
LEGACY_CANONICAL_IDS = (
    "v0_raw_minimal",
    "v1_missingness_summary",
    "v2_missingness_indicators",
    "v3_targeted_missingness",
)

# Obsolete or unimplemented names that must remain unregistered.
LEGACY_UNREGISTERED_IDS = frozenset(
    {
        "v1_basic_clean",
        "v2_eda_features",
        "v3_native_categorical",
    }
)

# Additive engineered feature names known from baseline_manifest.yaml.
V1_ENGINEERED_FEATURES = (
    "missing_count_total",
    "missing_rate_total",
    "missing_count_numeric",
    "missing_rate_numeric",
    "missing_count_categorical",
    "missing_rate_categorical",
    "missing_count_very_high",
    "missing_rate_very_high",
)

V2_SUMMARY_FEATURES = (
    "missing_count_total",
    "missing_count_numeric",
    "missing_count_categorical",
    "missing_count_very_high",
)

V3_ENGINEERED_FEATURES = (
    "Var217_is_missing",
    "Var126_is_missing",
    "Var218_is_missing",
    "Var192_is_missing",
)

V4_SUMMARY_FEATURES = (
    "zero_count_numeric",
    "zero_rate_observed_numeric",
    "zero_count_supported_numeric",
    "zero_rate_observed_supported_numeric",
)

LEGACY_CATALOG: Mapping[str, Mapping[str, Any]] = {
    "v0_raw_minimal": {
        "parent_dataset_id": None,
        "hypothesis": (
            "Retain ordered raw competition features after dropping constant "
            "and all-missing columns."
        ),
        "target_dependency": "none",
        "transformations": [
            {
                "type": "drop_columns",
                "description": (
                    "Remove the union of constant and all-missing raw features."
                ),
            }
        ],
        "summary_features": (),
        "binary_indicator_features": (),
        "use_missing_suffix": False,
    },
    "v1_missingness_summary": {
        "parent_dataset_id": "v0_raw_minimal",
        "hypothesis": (
            "Additive row-level missingness summaries improve signal over raw "
            "minimal features."
        ),
        "target_dependency": "none",
        "transformations": [
            {
                "type": "additive_missingness_summary",
                "added_features": list(V1_ENGINEERED_FEATURES),
            }
        ],
        "summary_features": V1_ENGINEERED_FEATURES,
        "binary_indicator_features": (),
        "use_missing_suffix": False,
    },
    "v2_missingness_indicators": {
        "parent_dataset_id": "v0_raw_minimal",
        "hypothesis": (
            "Broad per-feature missing indicators help identify useful "
            "targeted missingness features."
        ),
        "target_dependency": "none",
        "transformations": [
            {
                "type": "additive_broad_missingness_indicators",
                "summary_features": list(V2_SUMMARY_FEATURES),
                "indicator_suffix": "_is_missing",
            }
        ],
        "summary_features": V2_SUMMARY_FEATURES,
        "binary_indicator_features": (),
        "use_missing_suffix": True,
    },
    "v3_targeted_missingness": {
        "parent_dataset_id": "v1_missingness_summary",
        "hypothesis": (
            "Four targeted missing indicators selected from a prior full-train "
            "v2 experiment improve over missingness summaries alone."
        ),
        "target_dependency": "exploratory",
        "transformations": [
            {
                "type": "additive_targeted_missingness_indicators",
                "indicator_source_features": [
                    "Var217",
                    "Var126",
                    "Var218",
                    "Var192",
                ],
                "added_features": list(V3_ENGINEERED_FEATURES),
                "selection_source_experiment": (
                    "xgboost_v2_missingness_indicators_cv5"
                ),
            }
        ],
        "summary_features": V1_ENGINEERED_FEATURES,
        "binary_indicator_features": V3_ENGINEERED_FEATURES,
        "use_missing_suffix": False,
    },
}

NATIVE_CATALOG: Mapping[str, Mapping[str, Any]] = {
    **LEGACY_CATALOG,
    "v4_zero_value_summary": {
        "parent_dataset_id": "v0_raw_minimal",
        "hypothesis": (
            "Target-independent row-level zero-value summaries add signal over "
            "raw minimal features."
        ),
        "target_dependency": "none",
        "transformations": [
            {
                "type": "additive_zero_value_summary",
                "added_features": list(V4_SUMMARY_FEATURES),
            }
        ],
        "summary_features": V4_SUMMARY_FEATURES,
        "binary_indicator_features": (),
        "use_missing_suffix": False,
    },
}

NATIVE_CANONICAL_IDS = (
    "v0_raw_minimal",
    "v1_missingness_summary",
    "v2_missingness_indicators",
    "v3_targeted_missingness",
    "v4_zero_value_summary",
)

# Assumption documented for legacy row-identity proof:
# All four implemented packages retain the ordered v0_raw_minimal feature
# projection with unchanged values and row order. Legacy metadata records
# parent/base_version/source but does not store an explicit row-identity key;
# therefore the v0 ordered column projection is the verifiable anchor.
ROW_IDENTITY_ANCHOR_DATASET_ID = "v0_raw_minimal"
ROW_IDENTITY_METHOD = "ordered_v0_raw_minimal_feature_projection"
