"""Deterministic Research Workspace comparability classification.

These labels are descriptive workspace badges. They do not claim official
same-dataset Paired Comparison v1 readiness and must not weaken that contract.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.churn_ml.control_panel.research_inventory import ResearchRunInventoryRow


COMPARABLE_DEVELOPMENT = "Comparable development"
EXPLORATORY_DATASET = "Exploratory dataset"
SMOKE = "Smoke"
DIFFERENT_PROTOCOL = "Different protocol"
DIFFERENT_MODEL_CONFIGURATION = "Different model configuration"
TUNED_OPTUNA = "Tuned / Optuna export"
LEGACY_DESCRIPTIVE = "Legacy / descriptive only"
INCOMPLETE_IDENTITY = "Incomplete identity"
INVALID = "Invalid"

COMPARABILITY_LABELS = (
    COMPARABLE_DEVELOPMENT,
    EXPLORATORY_DATASET,
    SMOKE,
    DIFFERENT_PROTOCOL,
    DIFFERENT_MODEL_CONFIGURATION,
    TUNED_OPTUNA,
    LEGACY_DESCRIPTIVE,
    INCOMPLETE_IDENTITY,
    INVALID,
)

DEFAULT_DEVELOPMENT_PROTOCOL = {
    "evaluation_mode": "Development",
    "repeat_seeds": (0, 17),
    "outer_fold_count": 5,
    "threshold_selection_protocol": (
        "grid_balanced_accuracy_v1|stratified_kfold|n_splits=3|random_state=314159"
    ),
    "feature_pipeline_id": "registered_prepared_passthrough_v1",
}

DEFAULT_FAMILY_ADAPTERS = {
    "LightGBM": "manual_lightgbm_te_v1_compat",
    "XGBoost": "xgboost_numeric_v1",
    "CatBoost": "catboost_numeric_v1",
}


@dataclass(frozen=True)
class ComparabilityAssessment:
    primary: str
    reasons: tuple[str, ...]

    def to_mapping(self) -> dict[str, object]:
        return {"primary": self.primary, "reasons": list(self.reasons)}


def classify_research_run(
    row: ResearchRunInventoryRow,
    *,
    reference_by_family: dict[str, ResearchRunInventoryRow] | None = None,
) -> ComparabilityAssessment:
    """Classify one inventory row with a primary label and explainable reasons."""
    reasons: list[str] = []

    if row.status == "invalid":
        reasons.append(row.diagnostic or "Run markers or identity are invalid.")
        return ComparabilityAssessment(INVALID, tuple(reasons))
    if row.status == "failed":
        reasons.append("Run status is failed.")
        return ComparabilityAssessment(INVALID, tuple(reasons))
    if row.status == "running":
        reasons.append("Run is still running or incomplete.")
        return ComparabilityAssessment(INVALID, tuple(reasons))

    if not row.identity_complete:
        missing = ", ".join(row.unavailable_fields) or "required identity fields"
        reasons.append(f"Incomplete identity: unavailable {missing}.")
        return ComparabilityAssessment(INCOMPLETE_IDENTITY, tuple(reasons))

    if row.evaluation_mode == "Smoke":
        reasons.append("Evaluation mode is Smoke.")
        return ComparabilityAssessment(SMOKE, tuple(reasons))

    if row.has_search_provenance:
        reasons.append("resolved_config.search_provenance is present (Optuna export).")
        return ComparabilityAssessment(TUNED_OPTUNA, tuple(reasons))

    if (
        row.target_dependency == "exploratory"
        or (row.dataset_id or "").startswith("v3_targeted_missingness")
    ):
        reasons.append(
            "Dataset is exploratory (v3_targeted_missingness / target_dependency)."
        )
        return ComparabilityAssessment(EXPLORATORY_DATASET, tuple(reasons))

    if row.feature_pipeline_id != DEFAULT_DEVELOPMENT_PROTOCOL["feature_pipeline_id"]:
        reasons.append(
            "Feature pipeline is not registered_prepared_passthrough_v1; "
            "treat as legacy/descriptive only."
        )
        return ComparabilityAssessment(LEGACY_DESCRIPTIVE, tuple(reasons))

    protocol_mismatch = _protocol_mismatch_reasons(row)
    if protocol_mismatch:
        reasons.extend(protocol_mismatch)
        return ComparabilityAssessment(DIFFERENT_PROTOCOL, tuple(reasons))

    expected_adapter = DEFAULT_FAMILY_ADAPTERS.get(row.model_family or "")
    if expected_adapter and row.adapter_id != expected_adapter:
        reasons.append(
            f"Adapter {row.adapter_id!r} differs from default family adapter "
            f"{expected_adapter!r}."
        )
        return ComparabilityAssessment(DIFFERENT_MODEL_CONFIGURATION, tuple(reasons))

    if reference_by_family and row.model_family in reference_by_family:
        reference = reference_by_family[row.model_family]
        config_reasons = controlled_comparison_mismatch_reasons(reference, row)
        config_only = [
            reason
            for reason in config_reasons
            if "model configuration" in reason.lower()
            or "adapter" in reason.lower()
            or "candidate" in reason.lower()
        ]
        if config_only and any(
            "configuration" in reason.lower() or "candidate" in reason.lower()
            for reason in config_reasons
        ):
            # Only elevate when the reference shares protocol/dataset contract
            # except configuration.
            non_config = [
                reason
                for reason in config_reasons
                if reason not in config_only
                and "feature schema" not in reason.lower()
                and "dataset_id" not in reason.lower()
                and "content hash" not in reason.lower()
                and "schema_hash" not in reason.lower()
            ]
            if not non_config and config_only:
                reasons.extend(config_only)
                return ComparabilityAssessment(
                    DIFFERENT_MODEL_CONFIGURATION, tuple(reasons)
                )

    reasons.append(
        "Development mode with complete identity, registered prepared pipeline, "
        "and matching default protocol/adapters for descriptive matrix use."
    )
    reasons.append(
        "Does not by itself claim official Paired Comparison v1 readiness."
    )
    return ComparabilityAssessment(COMPARABLE_DEVELOPMENT, tuple(reasons))


def controlled_comparison_compatible(
    left: ResearchRunInventoryRow,
    right: ResearchRunInventoryRow,
) -> bool:
    return not controlled_comparison_mismatch_reasons(left, right)


def controlled_comparison_mismatch_reasons(
    left: ResearchRunInventoryRow,
    right: ResearchRunInventoryRow,
) -> list[str]:
    """Reasons why two runs are not controlled-comparable within one model family.

    Different feature schemas are allowed for descriptive cross-dataset
    comparison. Official paired inference remains a separate Stage E contract.
    """
    reasons: list[str] = []
    if left.model_family != right.model_family:
        reasons.append("model family mismatch")
    if left.adapter_id != right.adapter_id:
        reasons.append("adapter/pipeline contract mismatch")
    if left.model_config_id != right.model_config_id:
        reasons.append("model configuration identity mismatch")
    if left.feature_pipeline_id != right.feature_pipeline_id:
        reasons.append("feature pipeline mismatch")
    if left.evaluation_mode != right.evaluation_mode:
        reasons.append("evaluation mode mismatch")
    if left.repeat_seeds != right.repeat_seeds:
        reasons.append("repeat seeds mismatch")
    if left.outer_fold_count != right.outer_fold_count:
        reasons.append("outer-fold count mismatch")
    if left.evaluation_plan_hash != right.evaluation_plan_hash and (
        left.dataset_id == right.dataset_id
    ):
        # Same dataset with different plan hashes is a protocol mismatch.
        reasons.append("evaluation-plan / outer-assignment protocol mismatch")
    if left.threshold_selection_protocol != right.threshold_selection_protocol:
        reasons.append("threshold-selection protocol mismatch")
    if left.target_hash != right.target_hash:
        reasons.append("target hash mismatch")
    if left.train_row_identity_hash != right.train_row_identity_hash:
        reasons.append("training row identity mismatch")
    for side, label in ((left, "left"), (right, "right")):
        if side.status != "completed":
            reasons.append(f"{label} run is not completed")
        if not side.identity_complete:
            reasons.append(f"{label} identity is incomplete")
    return reasons


def annotate_inventory_comparability(
    rows: list[ResearchRunInventoryRow],
) -> list[tuple[ResearchRunInventoryRow, ComparabilityAssessment]]:
    references = _reference_rows_by_family(rows)
    return [
        (row, classify_research_run(row, reference_by_family=references))
        for row in rows
    ]


def _reference_rows_by_family(
    rows: list[ResearchRunInventoryRow],
) -> dict[str, ResearchRunInventoryRow]:
    """Prefer v0 comparable-development rows as family configuration references."""
    selected: dict[str, ResearchRunInventoryRow] = {}
    for row in rows:
        if row.dataset_id != "v0_raw_minimal":
            continue
        if row.status != "completed" or not row.identity_complete:
            continue
        if row.evaluation_mode != "Development":
            continue
        if row.has_search_provenance:
            continue
        if row.feature_pipeline_id != DEFAULT_DEVELOPMENT_PROTOCOL["feature_pipeline_id"]:
            continue
        family = row.model_family
        if family is None:
            continue
        current = selected.get(family)
        if current is None or (row.created_at_utc or "") > (current.created_at_utc or ""):
            selected[family] = row
    return selected


def _protocol_mismatch_reasons(row: ResearchRunInventoryRow) -> list[str]:
    reasons: list[str] = []
    if row.evaluation_mode != DEFAULT_DEVELOPMENT_PROTOCOL["evaluation_mode"]:
        reasons.append(
            f"evaluation mode {row.evaluation_mode!r} is not Development."
        )
    if row.repeat_seeds != DEFAULT_DEVELOPMENT_PROTOCOL["repeat_seeds"]:
        reasons.append(
            f"repeat seeds {row.repeat_seeds!r} differ from default development "
            f"{DEFAULT_DEVELOPMENT_PROTOCOL['repeat_seeds']!r}."
        )
    if row.outer_fold_count != DEFAULT_DEVELOPMENT_PROTOCOL["outer_fold_count"]:
        reasons.append(
            f"outer-fold count {row.outer_fold_count!r} differs from default "
            f"{DEFAULT_DEVELOPMENT_PROTOCOL['outer_fold_count']!r}."
        )
    if (
        row.threshold_selection_protocol
        != DEFAULT_DEVELOPMENT_PROTOCOL["threshold_selection_protocol"]
    ):
        reasons.append(
            "threshold-selection protocol differs from default development protocol."
        )
    return reasons
