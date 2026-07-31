"""Constants for Dataset Campaign / Matrix Runner v1."""

from __future__ import annotations

CAMPAIGN_CONTRACT_VERSION = "dataset_campaign_v1"
MANIFEST_SCHEMA_VERSION = "dataset_campaign_manifest_v1"
STATUS_SCHEMA_VERSION = "dataset_campaign_status_v1"
SUMMARY_SCHEMA_VERSION = "dataset_campaign_summary_v1"

CAMPAIGN_SPEC_FILENAME = "campaign_spec.yaml"
MANIFEST_FILENAME = "campaign_manifest.json"
STATUS_FILENAME = "campaign_status.json"
SUMMARY_FILENAME = "campaign_summary.json"
PREPARED_SUBDIR = "prepared"
CELLS_SUBDIR = "cells"

DEFAULT_PROCESSED_ROOT = "data/processed"
DEFAULT_CAMPAIGN_ARTIFACTS_ROOT = "artifacts/dataset_campaigns"

CAMPAIGN_TYPES = frozenset({"smoke", "development", "confirmation", "exploratory"})
CLASSIFICATIONS = frozenset({"unbiased", "exploratory"})
EXECUTION_POLICIES = frozenset({"sequential"})
MODEL_FAMILIES = frozenset({"LightGBM", "XGBoost", "CatBoost"})

# Exact unbiased screening set for Stage D (order is canonical for expansion).
UNBIASED_SCREENING_DATASET_IDS: tuple[str, ...] = (
    "v0_raw_minimal",
    "v1_missingness_summary",
    "v2_missingness_indicators",
    "v4_zero_value_summary",
    "v5_joint_missingness_pattern",
    "v6_compact_missingness_indicators",
    "v7_compact_zero_indicators",
)

EXPLORATORY_DATASET_IDS: frozenset[str] = frozenset({"v3_targeted_missingness"})

DATASET_PRESETS: dict[str, tuple[str, ...]] = {
    "unbiased_screening_v1": UNBIASED_SCREENING_DATASET_IDS,
}

REQUIRED_OOF_COLUMNS: tuple[str, ...] = (
    "repeat",
    "repeat_seed",
    "outer_fold",
    "row_position",
    "target",
    "probability",
)

AGGREGATE_METRIC_FIELDS: tuple[str, ...] = (
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "roc_auc",
    "average_precision",
    "brier_score",
)

FAMILY_TO_ADAPTER_PREFIX: dict[str, tuple[str, ...]] = {
    "LightGBM": ("manual_lightgbm", "lightgbm"),
    "XGBoost": ("xgboost",),
    "CatBoost": ("catboost",),
}
