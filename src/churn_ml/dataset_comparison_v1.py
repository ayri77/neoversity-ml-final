from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from src.churn_ml.dataset_campaign.constants import EXPLORATORY_DATASET_IDS
from src.churn_ml.dataset_registry.constants import MANIFEST_FILENAME
from src.churn_ml.dataset_registry.errors import DatasetRegistryError
from src.churn_ml.dataset_registry.manifest import read_manifest
from src.churn_ml.dataset_registry.package import manifest_path, package_dir_for
from src.churn_ml.dataset_registry.schema import DatasetManifest
from src.churn_ml.paired_comparison import (
    COMPARISON_SCHEMA_VERSION,
    CompletedResearchV2Run,
    OUTER_KEYS,
    PRINCIPAL_METRICS,
    THRESHOLD_KEYS,
    build_aggregate_metrics,
    load_completed_research_v2_run,
)
from src.churn_ml.research_data import canonical_sha256
from src.churn_ml.research_protocol import (
    build_threshold_grid,
    calculate_outer_metrics,
    dataframe_integer_sha256,
)

COMPARISON_OPERATION = "dataset_comparison_v1"
PROJECT_CONTRACT = "experiment_core_v2_dataset_comparison_v1"
DEFAULT_OUTPUT_ROOT = Path("artifacts/research_v2_dataset_comparisons")
DEFAULT_TIE_EPSILON = 1.0e-12
FOLD_KEYS = ["repeat", "outer_fold"]

ParentChildRelation = Literal[
    "baseline_is_parent",
    "candidate_is_parent",
    "unrelated",
]

UNBIASED_PARENT_CHILD_PAIRS: tuple[tuple[str, str], ...] = (
    ("v0_raw_minimal", "v1_missingness_summary"),
    ("v0_raw_minimal", "v2_missingness_indicators"),
    ("v0_raw_minimal", "v4_zero_value_summary"),
    ("v1_missingness_summary", "v5_joint_missingness_pattern"),
    ("v1_missingness_summary", "v6_compact_missingness_indicators"),
    ("v4_zero_value_summary", "v7_compact_zero_indicators"),
)

EXPLORATORY_PARENT_CHILD_PAIRS: tuple[tuple[str, str], ...] = (
    ("v1_missingness_summary", "v3_targeted_missingness"),
)

DATASET_SPECIFIC_PIPELINE_KEYS = frozenset(
    {
        "source_feature_count",
        "expected_model_feature_count",
        "expected_source_schema_sha256",
        "expected_model_input_schema_sha256",
        "expected_transformed_schema_sha256",
        "drop",
        "categorical",
    }
)

REGISTERED_PREPARED_PASSTHROUGH_V1 = "registered_prepared_passthrough_v1"


class DatasetComparisonError(ValueError):
    """Raised when a dataset comparison contract cannot be satisfied."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "DATASET_COMPARISON_INVALID",
        field_path: str = "comparison",
    ) -> None:
        self.reason_code = reason_code
        self.field_path = field_path
        self.detail = message
        super().__init__(f"{reason_code} at {field_path}: {message}")


class DatasetCompatibilityError(DatasetComparisonError):
    """Raised when two valid runs are not dataset-comparison compatible."""

    def __init__(self, report: DatasetComparisonCompatibilitySummary) -> None:
        self.report = report
        details = "; ".join(
            f"{issue.reason_code} at {issue.field_path}" for issue in report.issues
        )
        super().__init__(f"Dataset comparison compatibility failed: {details}.")


@dataclass(frozen=True)
class CompatibilityIssue:
    reason_code: str
    field_path: str
    baseline_value: Any
    candidate_value: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason_code": self.reason_code,
            "field_path": self.field_path,
            "baseline_value": self.baseline_value,
            "candidate_value": self.candidate_value,
        }


@dataclass(frozen=True)
class NormalizedIdentities:
    baseline_independent_row_identity: dict[str, Any]
    candidate_independent_row_identity: dict[str, Any]
    baseline_target_identity: dict[str, Any]
    candidate_target_identity: dict[str, Any]
    evaluation_protocol: dict[str, Any]
    evaluation_protocol_sha256: str
    baseline_model_configuration: dict[str, Any]
    baseline_model_configuration_sha256: str
    candidate_model_configuration: dict[str, Any]
    candidate_model_configuration_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "baseline_independent_row_identity": self.baseline_independent_row_identity,
            "candidate_independent_row_identity": self.candidate_independent_row_identity,
            "baseline_target_identity": self.baseline_target_identity,
            "candidate_target_identity": self.candidate_target_identity,
            "evaluation_protocol": self.evaluation_protocol,
            "evaluation_protocol_sha256": self.evaluation_protocol_sha256,
            "baseline_model_configuration": self.baseline_model_configuration,
            "baseline_model_configuration_sha256": (
                self.baseline_model_configuration_sha256
            ),
            "candidate_model_configuration": self.candidate_model_configuration,
            "candidate_model_configuration_sha256": (
                self.candidate_model_configuration_sha256
            ),
        }


@dataclass(frozen=True)
class DatasetComparisonCompatibilitySummary:
    schema_version: int
    operation: str
    compatible: bool
    issues: tuple[CompatibilityIssue, ...]
    expected_differences: tuple[str, ...]
    parent_child_relation: ParentChildRelation
    exploratory: bool
    normalized_identities: NormalizedIdentities | None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "operation": self.operation,
            "compatible": self.compatible,
            "issues": [issue.to_dict() for issue in self.issues],
            "expected_differences": list(self.expected_differences),
            "parent_child_relation": self.parent_child_relation,
            "exploratory": self.exploratory,
        }
        if self.normalized_identities is not None:
            payload["normalized_identities"] = self.normalized_identities.to_dict()
        return payload


@dataclass(frozen=True)
class DatasetComparisonRunReference:
    schema_version: int
    role: str
    run_id: str
    dataset_id: str
    parent_dataset_id: str | None
    repository_relative_path: str
    artifact_manifest_sha256: str
    artifact_manifest_file_sha256: str
    evaluation_plan_sha256: str
    evaluation_protocol_sha256: str
    model_configuration_sha256: str
    feature_pipeline_id: str
    candidate_adapter_id: str
    model_family: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "parent_dataset_id": self.parent_dataset_id,
            "repository_relative_path": self.repository_relative_path,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "artifact_manifest_file_sha256": self.artifact_manifest_file_sha256,
            "evaluation_plan_sha256": self.evaluation_plan_sha256,
            "evaluation_protocol_sha256": self.evaluation_protocol_sha256,
            "model_configuration_sha256": self.model_configuration_sha256,
            "feature_pipeline_id": self.feature_pipeline_id,
            "candidate_adapter_id": self.candidate_adapter_id,
            "model_family": self.model_family,
        }


@dataclass(frozen=True)
class DatasetComparisonMetricResults:
    repeat_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    aggregate_summary: dict[str, Any]


@dataclass(frozen=True)
class DatasetComparisonResult:
    compatibility: DatasetComparisonCompatibilitySummary
    baseline_reference: DatasetComparisonRunReference
    candidate_reference: DatasetComparisonRunReference
    paired_metrics: DatasetComparisonMetricResults
    oof_diagnostics: dict[str, Any]
    fit_time_delta: dict[str, Any]


def default_dataset_comparison_id(
    model_family: str,
    baseline_dataset_id: str,
    candidate_dataset_id: str,
) -> str:
    model_slug = model_family.casefold().replace(" ", "_")
    return f"{model_slug}__{baseline_dataset_id}__vs__{candidate_dataset_id}"


def is_exploratory_dataset(dataset_id: str, target_dependency: str | None) -> bool:
    if dataset_id in EXPLORATORY_DATASET_IDS:
        return True
    return target_dependency == "exploratory"


def derive_model_family(adapter_id: str) -> str:
    normalized = adapter_id.casefold()
    if normalized.startswith("manual_lightgbm") or normalized.startswith("lightgbm"):
        return "lightgbm"
    if normalized.startswith("xgboost"):
        return "xgboost"
    if normalized.startswith("catboost"):
        return "catboost"
    return adapter_id


def resolve_dataset_package_manifest(
    dataset_id: str,
    *,
    project_root: Path,
) -> DatasetManifest:
    processed_root = project_root / "data" / "processed"
    package_dir = package_dir_for(processed_root, dataset_id)
    manifest_file = manifest_path(package_dir)
    if not package_dir.is_dir() or not manifest_file.is_file():
        raise DatasetComparisonError(
            f"Dataset package manifest unavailable for {dataset_id!r}.",
            reason_code="ROW_IDENTITY_UNAVAILABLE",
            field_path=f"data.processed.{dataset_id}.{MANIFEST_FILENAME}",
        )
    try:
        return read_manifest(manifest_file)
    except (OSError, DatasetRegistryError) as error:
        raise DatasetComparisonError(
            f"Cannot read dataset package manifest for {dataset_id!r}.",
            reason_code="ROW_IDENTITY_UNAVAILABLE",
            field_path=f"data.processed.{dataset_id}.{MANIFEST_FILENAME}",
        ) from error


def build_independent_row_identity(
    manifest: DatasetManifest,
    row_position_identity: Mapping[str, Any],
    train_row_count: int,
) -> dict[str, Any]:
    kind = row_position_identity.get("kind")
    sha256 = row_position_identity.get("sha256")
    if not isinstance(kind, str) or not isinstance(sha256, str):
        raise DatasetComparisonError(
            "row_position_identity must contain string kind and sha256.",
            reason_code="ROW_IDENTITY_UNAVAILABLE",
            field_path="dataset_fingerprints.row_position_identity",
        )
    return {
        "train_anchor_hash": manifest.row_identity.train_anchor_hash,
        "row_position_identity": {"kind": kind, "sha256": sha256},
        "train_row_count": train_row_count,
    }


def build_target_identity(
    run: CompletedResearchV2Run,
    manifest: DatasetManifest | None,
) -> dict[str, Any]:
    target_fp = run.dataset_fingerprints.get("target")
    if not isinstance(target_fp, Mapping):
        raise DatasetComparisonError(
            "dataset_fingerprints.target is malformed.",
            reason_code="TARGET_IDENTITY_MISMATCH",
            field_path="dataset_fingerprints.target",
        )
    plan_target = run.config.plan_payload.get("dataset", {}).get("target", {})
    positive_label = plan_target.get("positive_label")
    identity: dict[str, Any] = {
        "name": target_fp.get("name"),
        "dtype": target_fp.get("dtype"),
        "negative_count": target_fp.get("negative_count"),
        "positive_count": target_fp.get("positive_count"),
        "values_sha256": target_fp.get("values_sha256"),
        "positive_label": positive_label,
    }
    if manifest is not None:
        identity["manifest_target_hash"] = manifest.target.hash
    return identity


def build_evaluation_protocol_identity(
    plan_payload: Mapping[str, Any],
    outer_assignments: pd.DataFrame,
    threshold_assignments: pd.DataFrame,
    *,
    fold_product: Sequence[tuple[int, int]],
) -> tuple[dict[str, Any], str]:
    threshold_policy = dict(plan_payload["threshold_policy"])
    grid = build_threshold_grid(threshold_policy).tolist()
    canonical: dict[str, Any] = {
        "schema_version": 1,
        "outer_evaluation": deepcopy(dict(plan_payload["outer_evaluation"])),
        "threshold_selection": deepcopy(dict(plan_payload["threshold_selection"])),
        "threshold_policy": deepcopy(threshold_policy),
        "threshold_grid": grid,
        "metrics": deepcopy(dict(plan_payload["metrics"])),
        "aggregation": deepcopy(dict(plan_payload["aggregation"])),
        "positive_class_label": plan_payload.get("dataset", {})
        .get("target", {})
        .get("positive_label"),
        "label_convention": threshold_policy.get("comparison"),
        "probability_to_label_convention": "probability_greater_than_or_equal_threshold",
        "positive_class_probability_convention": "binary_positive_class_label_1",
        "assignment_fingerprints": {
            "outer": dataframe_integer_sha256(outer_assignments),
            "threshold_selection": dataframe_integer_sha256(threshold_assignments),
        },
        "fold_product": [list(item) for item in fold_product],
    }
    mode = plan_payload.get("mode")
    if mode is not None:
        canonical["mode"] = mode
    outer_context = plan_payload.get("outer_evaluation_context")
    if isinstance(outer_context, Mapping) and outer_context.get("mode") is not None:
        canonical["mode"] = outer_context["mode"]
    if "mode" not in canonical:
        plan_id = ""
        plan_section = plan_payload.get("plan")
        if isinstance(plan_section, Mapping):
            plan_id = str(plan_section.get("id") or "")
        lowered = plan_id.casefold()
        if "smoke" in lowered:
            canonical["mode"] = "smoke"
        elif "development" in lowered:
            canonical["mode"] = "development"
        elif "confirmation" in lowered:
            canonical["mode"] = "confirmation"
    digest = canonical_sha256(canonical)
    return canonical, digest


def build_model_configuration_identity(
    run: CompletedResearchV2Run,
) -> tuple[dict[str, Any], str]:
    pipeline = run.config.payload["feature_pipeline"]
    adapter = run.config.payload["candidate_adapter"]
    candidate_canonical = run.candidate_identity.get("canonical", {})
    pipeline_id = run.config.pipeline_id
    pipeline_contract = pipeline.get("contract", {})
    if pipeline_id == REGISTERED_PREPARED_PASSTHROUGH_V1:
        normalized_pipeline: dict[str, Any] = {"id": pipeline_id}
    else:
        contract = {
            key: deepcopy(value)
            for key, value in dict(pipeline_contract).items()
            if key not in DATASET_SPECIFIC_PIPELINE_KEYS
        }
        normalized_pipeline = {"id": pipeline_id, "contract": contract}
    canonical = {
        "schema_version": 1,
        "model_family": derive_model_family(run.config.adapter_id),
        "feature_pipeline": normalized_pipeline,
        "candidate_adapter": {
            "id": run.config.adapter_id,
            "contract": deepcopy(dict(adapter.get("contract", {}))),
        },
        "probability_semantics": candidate_canonical.get("probability_semantics"),
        "evaluation_boundary": candidate_canonical.get("evaluation_boundary"),
    }
    return canonical, canonical_sha256(canonical)


def resolve_parent_child_relation(
    baseline_dataset_id: str,
    candidate_dataset_id: str,
    baseline_manifest: DatasetManifest,
    candidate_manifest: DatasetManifest,
) -> ParentChildRelation:
    if candidate_manifest.parent_dataset_id == baseline_dataset_id:
        return "baseline_is_parent"
    if baseline_manifest.parent_dataset_id == candidate_dataset_id:
        return "candidate_is_parent"
    return "unrelated"


def build_expected_difference_fields() -> tuple[str, ...]:
    return (
        "dataset_id",
        "parent_dataset_id",
        "schema_hash",
        "feature_count",
        "feature_names",
        "dtypes",
        "train_content_hash",
        "test_content_hash",
        "experiment_id",
        "run_id",
        "timestamps",
        "evaluation_plan_id",
        "evaluation_plan_hash",
        "source_config_hash",
        "resolved_config_hash",
    )


def build_normalized_identities(
    baseline: CompletedResearchV2Run,
    candidate: CompletedResearchV2Run,
    *,
    project_root: Path,
) -> NormalizedIdentities:
    baseline_manifest = resolve_dataset_package_manifest(
        baseline.config.dataset_version,
        project_root=project_root,
    )
    candidate_manifest = resolve_dataset_package_manifest(
        candidate.config.dataset_version,
        project_root=project_root,
    )
    baseline_row_count = int(baseline.dataset_fingerprints.get("row_count", 0))
    candidate_row_count = int(candidate.dataset_fingerprints.get("row_count", 0))
    baseline_position = baseline.dataset_fingerprints.get("row_position_identity", {})
    candidate_position = candidate.dataset_fingerprints.get("row_position_identity", {})
    if not isinstance(baseline_position, Mapping) or not isinstance(
        candidate_position, Mapping
    ):
        raise DatasetComparisonError(
            "row_position_identity is malformed.",
            reason_code="ROW_IDENTITY_UNAVAILABLE",
            field_path="dataset_fingerprints.row_position_identity",
        )
    protocol, protocol_hash = build_evaluation_protocol_identity(
        baseline.config.plan_payload,
        baseline.outer_assignments,
        baseline.threshold_assignments,
        fold_product=_fold_product(baseline),
    )
    baseline_model, baseline_model_hash = build_model_configuration_identity(baseline)
    candidate_model, candidate_model_hash = build_model_configuration_identity(
        candidate
    )
    return NormalizedIdentities(
        baseline_independent_row_identity=build_independent_row_identity(
            baseline_manifest,
            baseline_position,
            baseline_row_count,
        ),
        candidate_independent_row_identity=build_independent_row_identity(
            candidate_manifest,
            candidate_position,
            candidate_row_count,
        ),
        baseline_target_identity=build_target_identity(baseline, baseline_manifest),
        candidate_target_identity=build_target_identity(candidate, candidate_manifest),
        evaluation_protocol=protocol,
        evaluation_protocol_sha256=protocol_hash,
        baseline_model_configuration=baseline_model,
        baseline_model_configuration_sha256=baseline_model_hash,
        candidate_model_configuration=candidate_model,
        candidate_model_configuration_sha256=candidate_model_hash,
    )


def build_compatibility_summary(
    baseline: CompletedResearchV2Run,
    candidate: CompletedResearchV2Run,
    *,
    project_root: Path,
) -> DatasetComparisonCompatibilitySummary:
    issues: list[CompatibilityIssue] = []

    def compare(
        baseline_value: Any,
        candidate_value: Any,
        reason_code: str,
        field_path: str,
    ) -> None:
        if not _exact_equal(baseline_value, candidate_value):
            issues.append(
                CompatibilityIssue(
                    reason_code=reason_code,
                    field_path=field_path,
                    baseline_value=_compact_value(baseline_value),
                    candidate_value=_compact_value(candidate_value),
                )
            )

    baseline_dataset_id = baseline.config.dataset_version
    candidate_dataset_id = candidate.config.dataset_version
    if baseline_dataset_id == candidate_dataset_id:
        issues.append(
            CompatibilityIssue(
                reason_code="SAME_DATASET_ID",
                field_path="resolved_config.dataset.version",
                baseline_value=baseline_dataset_id,
                candidate_value=candidate_dataset_id,
            )
        )

    baseline_manifest: DatasetManifest | None = None
    candidate_manifest: DatasetManifest | None = None
    parent_child_relation: ParentChildRelation = "unrelated"
    exploratory = False
    normalized: NormalizedIdentities | None = None

    try:
        baseline_manifest = resolve_dataset_package_manifest(
            baseline_dataset_id,
            project_root=project_root,
        )
        candidate_manifest = resolve_dataset_package_manifest(
            candidate_dataset_id,
            project_root=project_root,
        )
        parent_child_relation = resolve_parent_child_relation(
            baseline_dataset_id,
            candidate_dataset_id,
            baseline_manifest,
            candidate_manifest,
        )
        exploratory = is_exploratory_dataset(
            baseline_dataset_id,
            baseline_manifest.target_dependency,
        ) or is_exploratory_dataset(
            candidate_dataset_id,
            candidate_manifest.target_dependency,
        )
        normalized = build_normalized_identities(
            baseline,
            candidate,
            project_root=project_root,
        )
        compare(
            normalized.baseline_independent_row_identity,
            normalized.candidate_independent_row_identity,
            "INDEPENDENT_ROW_IDENTITY_MISMATCH",
            "independent_row_identity",
        )
        if normalized.baseline_independent_row_identity.get(
            "row_position_identity"
        ) != normalized.candidate_independent_row_identity.get("row_position_identity"):
            compare(
                normalized.baseline_independent_row_identity["row_position_identity"],
                normalized.candidate_independent_row_identity["row_position_identity"],
                "OBSERVATION_ORDER_MISMATCH",
                "independent_row_identity.row_position_identity",
            )
        compare(
            normalized.baseline_target_identity,
            normalized.candidate_target_identity,
            "TARGET_IDENTITY_MISMATCH",
            "target_identity",
        )
        baseline_protocol, baseline_protocol_hash = build_evaluation_protocol_identity(
            baseline.config.plan_payload,
            baseline.outer_assignments,
            baseline.threshold_assignments,
            fold_product=_fold_product(baseline),
        )
        candidate_protocol, candidate_protocol_hash = (
            build_evaluation_protocol_identity(
                candidate.config.plan_payload,
                candidate.outer_assignments,
                candidate.threshold_assignments,
                fold_product=_fold_product(candidate),
            )
        )
        compare(
            baseline_protocol_hash,
            candidate_protocol_hash,
            "EVALUATION_PROTOCOL_MISMATCH",
            "evaluation_protocol_sha256",
        )
        del baseline_protocol, candidate_protocol
        compare(
            normalized.baseline_model_configuration["model_family"],
            normalized.candidate_model_configuration["model_family"],
            "MODEL_FAMILY_MISMATCH",
            "model_configuration.model_family",
        )
        compare(
            normalized.baseline_model_configuration_sha256,
            normalized.candidate_model_configuration_sha256,
            "MODEL_CONFIGURATION_MISMATCH",
            "model_configuration_sha256",
        )
        baseline_plan_metrics = baseline.config.plan_payload.get("metrics")
        candidate_plan_metrics = candidate.config.plan_payload.get("metrics")
        if baseline_plan_metrics != candidate_plan_metrics:
            compare(
                baseline_plan_metrics,
                candidate_plan_metrics,
                "METRIC_CONTRACT_MISMATCH",
                "resolved_config.evaluation_plan.metrics",
            )
    except DatasetComparisonError as error:
        if error.reason_code == "ROW_IDENTITY_UNAVAILABLE":
            issues.append(
                CompatibilityIssue(
                    reason_code=error.reason_code,
                    field_path=error.field_path,
                    baseline_value=baseline_dataset_id,
                    candidate_value=candidate_dataset_id,
                )
            )
        else:
            raise

    _compare_frame(
        baseline.outer_assignments,
        candidate.outer_assignments,
        issues,
        reason_code="OUTER_ASSIGNMENTS_MISMATCH",
        field_path="splits.outer_assignments",
        sort_by=OUTER_KEYS,
    )
    _compare_frame(
        baseline.threshold_assignments,
        candidate.threshold_assignments,
        issues,
        reason_code="THRESHOLD_ASSIGNMENTS_MISMATCH",
        field_path="splits.threshold_selection_assignments",
        sort_by=THRESHOLD_KEYS,
    )
    issues.extend(
        _validate_oof_alignment_issues(baseline, candidate),
    )
    compare(
        _fold_product(baseline),
        _fold_product(candidate),
        "OOF_INCOMPLETE_FOLD_PRODUCT",
        "repeat_outer_fold_cartesian_product",
    )

    return DatasetComparisonCompatibilitySummary(
        schema_version=COMPARISON_SCHEMA_VERSION,
        operation=COMPARISON_OPERATION,
        compatible=not issues,
        issues=tuple(issues),
        expected_differences=build_expected_difference_fields(),
        parent_child_relation=parent_child_relation,
        exploratory=exploratory,
        normalized_identities=normalized,
    )


def build_dataset_comparison_metrics(
    baseline: CompletedResearchV2Run,
    candidate: CompletedResearchV2Run,
    *,
    project_root: Path | None = None,
    tie_epsilon: float = DEFAULT_TIE_EPSILON,
) -> DatasetComparisonMetricResults:
    epsilon = _validate_tie_epsilon(tie_epsilon)
    _require_compatible(
        baseline,
        candidate,
        project_root=project_root or baseline.config.project_root,
    )
    baseline_outer, candidate_outer = _aligned_frames(
        baseline.outer_predictions,
        candidate.outer_predictions,
        OUTER_KEYS,
        "outer prediction",
    )
    repeat_records: list[dict[str, Any]] = []
    for repeat in sorted(baseline_outer["repeat"].unique()):
        baseline_frame = baseline_outer.loc[baseline_outer["repeat"] == repeat]
        candidate_frame = candidate_outer.loc[candidate_outer["repeat"] == repeat]
        baseline_metrics = _metrics_from_predictions(baseline_frame)
        candidate_metrics = _metrics_from_predictions(candidate_frame)
        baseline_thresholds = baseline.selected_thresholds.loc[
            baseline.selected_thresholds["repeat"] == repeat,
            "selected_threshold",
        ].to_numpy(dtype=float)
        candidate_thresholds = candidate.selected_thresholds.loc[
            candidate.selected_thresholds["repeat"] == repeat,
            "selected_threshold",
        ].to_numpy(dtype=float)
        record: dict[str, Any] = {
            "repeat": int(repeat),
            "validation_rows": len(baseline_frame),
        }
        _add_metric_pair(record, baseline_metrics, candidate_metrics)
        record.update(
            {
                "baseline_median_threshold": float(np.median(baseline_thresholds)),
                "candidate_median_threshold": float(np.median(candidate_thresholds)),
                "threshold_delta": float(
                    np.median(candidate_thresholds) - np.median(baseline_thresholds)
                ),
            }
        )
        repeat_records.append(record)

    fold_records: list[dict[str, Any]] = []
    for repeat, outer_fold in _fold_product(baseline):
        baseline_frame = baseline_outer.loc[
            (baseline_outer["repeat"] == repeat)
            & (baseline_outer["outer_fold"] == outer_fold)
        ]
        candidate_frame = candidate_outer.loc[
            (candidate_outer["repeat"] == repeat)
            & (candidate_outer["outer_fold"] == outer_fold)
        ]
        baseline_metrics = _metrics_from_predictions(baseline_frame)
        candidate_metrics = _metrics_from_predictions(candidate_frame)
        baseline_threshold = float(baseline_frame["selected_threshold"].iloc[0])
        candidate_threshold = float(candidate_frame["selected_threshold"].iloc[0])
        record = {
            "repeat": repeat,
            "outer_fold": outer_fold,
            "validation_rows": len(baseline_frame),
        }
        _add_metric_pair(record, baseline_metrics, candidate_metrics)
        record.update(
            {
                "baseline_threshold": baseline_threshold,
                "candidate_threshold": candidate_threshold,
                "threshold_delta": candidate_threshold - baseline_threshold,
                "baseline_positive_prediction_rate": float(
                    baseline_frame["prediction"].mean()
                ),
                "candidate_positive_prediction_rate": float(
                    candidate_frame["prediction"].mean()
                ),
                "prediction_agreement_rate": float(
                    (
                        baseline_frame["prediction"].to_numpy()
                        == candidate_frame["prediction"].to_numpy()
                    ).mean()
                ),
            }
        )
        fold_records.append(record)

    repeat_metrics = pd.DataFrame(repeat_records)
    fold_metrics = pd.DataFrame(fold_records)
    summary = build_aggregate_metrics(
        repeat_metrics,
        fold_metrics,
        baseline.selected_thresholds,
        candidate.selected_thresholds,
        tie_epsilon=epsilon,
    )
    return DatasetComparisonMetricResults(repeat_metrics, fold_metrics, summary)


def build_oof_diagnostics(
    baseline: CompletedResearchV2Run,
    candidate: CompletedResearchV2Run,
    *,
    project_root: Path | None = None,
    probability_difference_quantiles: Sequence[float] | None = None,
) -> dict[str, Any]:
    _require_compatible(
        baseline,
        candidate,
        project_root=project_root or baseline.config.project_root,
    )
    baseline_frame, candidate_frame = _aligned_frames(
        baseline.outer_predictions,
        candidate.outer_predictions,
        OUTER_KEYS,
        "outer prediction",
    )
    baseline_probability = baseline_frame["probability"].to_numpy(dtype=float)
    candidate_probability = candidate_frame["probability"].to_numpy(dtype=float)
    baseline_prediction = baseline_frame["prediction"].to_numpy(dtype=np.int8)
    candidate_prediction = candidate_frame["prediction"].to_numpy(dtype=np.int8)
    target = baseline_frame["target"].to_numpy(dtype=np.int8)
    if not np.array_equal(target, candidate_frame["target"].to_numpy(dtype=np.int8)):
        raise DatasetComparisonError(
            "Aligned outer prediction targets differ.",
            reason_code="OOF_TARGET_DISAGREEMENT",
            field_path="predictions.outer_validation.target",
        )
    differences = np.abs(candidate_probability - baseline_probability)
    baseline_correct = baseline_prediction == target
    candidate_correct = candidate_prediction == target
    total = len(target)
    matrix = {
        "both_correct": _count_rate(baseline_correct & candidate_correct, total),
        "baseline_only_correct": _count_rate(
            baseline_correct & ~candidate_correct, total
        ),
        "candidate_only_correct": _count_rate(
            ~baseline_correct & candidate_correct, total
        ),
        "both_wrong": _count_rate(~baseline_correct & ~candidate_correct, total),
    }
    disagreement_by_class: dict[str, Any] = {}
    for true_class in (0, 1):
        mask = target == true_class
        class_count = int(mask.sum())
        disagreement_by_class[str(true_class)] = {
            "rows": class_count,
            "prediction_disagreement": _count_rate(
                (baseline_prediction != candidate_prediction) & mask,
                class_count,
            ),
            "candidate_positive_baseline_negative": _count_rate(
                (candidate_prediction == 1) & (baseline_prediction == 0) & mask,
                class_count,
            ),
            "baseline_positive_candidate_negative": _count_rate(
                (baseline_prediction == 1) & (candidate_prediction == 0) & mask,
                class_count,
            ),
        }
    quantiles = probability_difference_quantiles
    if quantiles is None:
        quantiles = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    normalized_quantiles = _validate_quantiles(quantiles)
    pearson_value, pearson_status = _correlation(
        baseline_probability, candidate_probability
    )
    baseline_ranks = pd.Series(baseline_probability).rank(method="average").to_numpy()
    candidate_ranks = pd.Series(candidate_probability).rank(method="average").to_numpy()
    spearman_value, spearman_status = _correlation(baseline_ranks, candidate_ranks)
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "operation": COMPARISON_OPERATION,
        "observation_unit": "outer_validation_row_per_repeat",
        "observation_count": total,
        "pearson_correlation": {
            "coefficient": pearson_value,
            "status": pearson_status,
        },
        "spearman_correlation": {
            "coefficient": spearman_value,
            "status": spearman_status,
        },
        "mean_absolute_probability_difference": float(differences.mean()),
        "maximum_absolute_probability_difference": float(differences.max()),
        "prediction_label_agreement_rate": float(
            (baseline_prediction == candidate_prediction).mean()
        ),
        "correctness_matrix": matrix,
        "candidate_positive_baseline_negative": _count_rate(
            (candidate_prediction == 1) & (baseline_prediction == 0), total
        ),
        "baseline_positive_candidate_negative": _count_rate(
            (baseline_prediction == 1) & (candidate_prediction == 0), total
        ),
        "disagreement_by_true_class": disagreement_by_class,
        "absolute_probability_difference_quantiles": {
            _quantile_label(quantile): float(np.quantile(differences, quantile))
            for quantile in normalized_quantiles
        },
    }


def build_fit_time_delta(
    baseline: CompletedResearchV2Run,
    candidate: CompletedResearchV2Run,
) -> dict[str, Any]:
    baseline_timing = _authoritative_fit_timing(baseline.metadata)
    candidate_timing = _authoritative_fit_timing(candidate.metadata)
    if baseline_timing is None or candidate_timing is None:
        return {
            "available": False,
            "reason": "authoritative_fit_timing_unavailable_on_one_or_both_runs",
        }
    return {
        "available": True,
        "baseline_seconds": baseline_timing,
        "candidate_seconds": candidate_timing,
        "candidate_minus_baseline_seconds": candidate_timing - baseline_timing,
    }


def build_comparison_result(
    baseline: CompletedResearchV2Run,
    candidate: CompletedResearchV2Run,
    *,
    project_root: Path,
    tie_epsilon: float = DEFAULT_TIE_EPSILON,
) -> DatasetComparisonResult:
    compatibility = build_compatibility_summary(
        baseline,
        candidate,
        project_root=project_root,
    )
    if not compatibility.compatible:
        raise DatasetCompatibilityError(compatibility)
    paired_metrics = build_dataset_comparison_metrics(
        baseline,
        candidate,
        project_root=project_root,
        tie_epsilon=tie_epsilon,
    )
    oof_diagnostics = build_oof_diagnostics(
        baseline,
        candidate,
        project_root=project_root,
    )
    return DatasetComparisonResult(
        compatibility=compatibility,
        baseline_reference=_run_reference(baseline, "baseline", compatibility),
        candidate_reference=_run_reference(candidate, "candidate", compatibility),
        paired_metrics=paired_metrics,
        oof_diagnostics=oof_diagnostics,
        fit_time_delta=build_fit_time_delta(baseline, candidate),
    )


def deterministic_comparison_identity(
    baseline_reference: DatasetComparisonRunReference,
    candidate_reference: DatasetComparisonRunReference,
) -> str:
    canonical = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "operation": COMPARISON_OPERATION,
        "baseline_manifest_sha256": baseline_reference.artifact_manifest_sha256,
        "candidate_manifest_sha256": candidate_reference.artifact_manifest_sha256,
        "baseline_dataset_id": baseline_reference.dataset_id,
        "candidate_dataset_id": candidate_reference.dataset_id,
        "evaluation_protocol_sha256": baseline_reference.evaluation_protocol_sha256,
        "baseline_model_configuration_sha256": (
            baseline_reference.model_configuration_sha256
        ),
        "candidate_model_configuration_sha256": (
            candidate_reference.model_configuration_sha256
        ),
    }
    return canonical_sha256(canonical)


def _run_reference(
    run: CompletedResearchV2Run,
    role: str,
    compatibility: DatasetComparisonCompatibilitySummary,
) -> DatasetComparisonRunReference:
    manifest_path_obj = run.root / "artifact_manifest.json"
    normalized = compatibility.normalized_identities
    if normalized is None:
        raise DatasetComparisonError(
            "Normalized identities are required for run references.",
            reason_code="DATASET_COMPARISON_INVALID",
            field_path="normalized_identities",
        )
    model_hash = (
        normalized.baseline_model_configuration_sha256
        if role == "baseline"
        else normalized.candidate_model_configuration_sha256
    )
    manifest = resolve_dataset_package_manifest(
        run.config.dataset_version,
        project_root=run.config.project_root,
    )
    return DatasetComparisonRunReference(
        schema_version=COMPARISON_SCHEMA_VERSION,
        role=role,
        run_id=str(run.metadata["run_id"]),
        dataset_id=run.config.dataset_version,
        parent_dataset_id=manifest.parent_dataset_id,
        repository_relative_path=run.root.relative_to(
            run.config.project_root
        ).as_posix(),
        artifact_manifest_sha256=str(run.manifest["manifest_sha256"]),
        artifact_manifest_file_sha256=hashlib.sha256(
            manifest_path_obj.read_bytes()
        ).hexdigest(),
        evaluation_plan_sha256=str(run.evaluation_plan_identity["sha256"]),
        evaluation_protocol_sha256=normalized.evaluation_protocol_sha256,
        model_configuration_sha256=model_hash,
        feature_pipeline_id=run.config.pipeline_id,
        candidate_adapter_id=run.config.adapter_id,
        model_family=derive_model_family(run.config.adapter_id),
    )


def _validate_oof_alignment_issues(
    baseline: CompletedResearchV2Run,
    candidate: CompletedResearchV2Run,
) -> list[CompatibilityIssue]:
    issues: list[CompatibilityIssue] = []
    for label, frame, keys in (
        ("outer", baseline.outer_predictions, OUTER_KEYS),
        ("threshold", baseline.threshold_predictions, THRESHOLD_KEYS),
    ):
        candidate_frame = (
            candidate.outer_predictions
            if label == "outer"
            else candidate.threshold_predictions
        )
        if frame.duplicated(keys).any():
            issues.append(
                CompatibilityIssue(
                    reason_code="OOF_DUPLICATE_KEYS",
                    field_path=f"predictions.{label}.baseline",
                    baseline_value={"rows": len(frame)},
                    candidate_value=None,
                )
            )
        if candidate_frame.duplicated(keys).any():
            issues.append(
                CompatibilityIssue(
                    reason_code="OOF_DUPLICATE_KEYS",
                    field_path=f"predictions.{label}.candidate",
                    baseline_value=None,
                    candidate_value={"rows": len(candidate_frame)},
                )
            )
        if not np.isfinite(frame["probability"].to_numpy(dtype=float)).all():
            issues.append(
                CompatibilityIssue(
                    reason_code="OOF_INVALID_PROBABILITIES",
                    field_path=f"predictions.{label}.baseline.probability",
                    baseline_value="non_finite",
                    candidate_value=None,
                )
            )
        if not np.isfinite(candidate_frame["probability"].to_numpy(dtype=float)).all():
            issues.append(
                CompatibilityIssue(
                    reason_code="OOF_INVALID_PROBABILITIES",
                    field_path=f"predictions.{label}.candidate.probability",
                    baseline_value=None,
                    candidate_value="non_finite",
                )
            )
    try:
        baseline_outer, candidate_outer = _aligned_frames(
            baseline.outer_predictions,
            candidate.outer_predictions,
            OUTER_KEYS,
            "outer prediction",
        )
    except DatasetComparisonError as error:
        code = error.reason_code
        if code == "OOF_DUPLICATE_KEYS":
            return issues
        issues.append(
            CompatibilityIssue(
                reason_code=(
                    "OOF_MISSING_KEYS"
                    if "coverage" in error.detail.lower()
                    else "OOF_ROW_COVERAGE_MISMATCH"
                ),
                field_path="predictions.outer_validation.keys",
                baseline_value={"rows": len(baseline.outer_predictions)},
                candidate_value={"rows": len(candidate.outer_predictions)},
            )
        )
        return issues
    if not np.array_equal(
        baseline_outer["target"].to_numpy(),
        candidate_outer["target"].to_numpy(),
    ):
        issues.append(
            CompatibilityIssue(
                reason_code="OOF_TARGET_DISAGREEMENT",
                field_path="predictions.outer_validation.target",
                baseline_value=_compact_value(baseline_outer["target"].tolist()[:8]),
                candidate_value=_compact_value(candidate_outer["target"].tolist()[:8]),
            )
        )
    return issues


def _require_compatible(
    baseline: CompletedResearchV2Run,
    candidate: CompletedResearchV2Run,
    *,
    project_root: Path | None = None,
) -> None:
    root = project_root or baseline.config.project_root
    report = build_compatibility_summary(
        baseline,
        candidate,
        project_root=root,
    )
    if not report.compatible:
        raise DatasetCompatibilityError(report)


def _authoritative_fit_timing(metadata: Mapping[str, Any]) -> float | None:
    timing = metadata.get("timing")
    if not isinstance(timing, Mapping):
        return None
    fit_seconds = timing.get("fit_seconds")
    if type(fit_seconds) is not float or not math.isfinite(fit_seconds):
        return None
    if timing.get("authoritative") is not True:
        return None
    return fit_seconds


def _add_metric_pair(
    record: dict[str, Any],
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
) -> None:
    for metric in PRINCIPAL_METRICS:
        baseline_value = float(baseline_metrics[metric])
        candidate_value = float(candidate_metrics[metric])
        record[f"baseline_{metric}"] = baseline_value
        record[f"candidate_{metric}"] = candidate_value
        record[f"{metric}_delta"] = candidate_value - baseline_value
        if metric == "brier_score":
            record["brier_score_improvement"] = baseline_value - candidate_value


def _metrics_from_predictions(frame: pd.DataFrame) -> dict[str, Any]:
    return calculate_outer_metrics(
        frame["target"],
        frame["probability"].to_numpy(dtype=float),
        frame["prediction"].to_numpy(dtype=np.int8),
        selected_threshold=float(frame["selected_threshold"].iloc[0]),
    )


def _aligned_frames(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    keys: list[str],
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if baseline.duplicated(keys).any() or candidate.duplicated(keys).any():
        raise DatasetComparisonError(
            f"{label} keys contain duplicates.",
            reason_code="OOF_DUPLICATE_KEYS",
            field_path=f"predictions.{label}.keys",
        )
    baseline_sorted = baseline.sort_values(keys, ignore_index=True)
    candidate_sorted = candidate.sort_values(keys, ignore_index=True)
    if not baseline_sorted[keys].equals(candidate_sorted[keys]):
        raise DatasetComparisonError(
            f"{label} key coverage differs.",
            reason_code="OOF_MISSING_KEYS",
            field_path=f"predictions.{label}.keys",
        )
    return baseline_sorted, candidate_sorted


def _fold_product(run: CompletedResearchV2Run) -> list[tuple[int, int]]:
    frame = run.selected_thresholds
    keys = list(frame[FOLD_KEYS].itertuples(index=False, name=None))
    if len(keys) != len(set(keys)):
        return [(-1, -1), *sorted(set(keys))]
    return sorted((int(repeat), int(fold)) for repeat, fold in keys)


def _compare_frame(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    issues: list[CompatibilityIssue],
    *,
    reason_code: str,
    field_path: str,
    sort_by: list[str],
) -> None:
    try:
        pd.testing.assert_frame_equal(
            baseline.sort_values(sort_by, ignore_index=True),
            candidate.sort_values(sort_by, ignore_index=True),
            check_dtype=True,
            check_exact=True,
        )
    except AssertionError:
        issues.append(
            CompatibilityIssue(
                reason_code=reason_code,
                field_path=field_path,
                baseline_value={
                    "rows": len(baseline),
                    "sha256": _frame_sha256(baseline, sort_by),
                },
                candidate_value={
                    "rows": len(candidate),
                    "sha256": _frame_sha256(candidate, sort_by),
                },
            )
        )


def _frame_sha256(frame: pd.DataFrame, sort_by: list[str]) -> str:
    normalized = frame.sort_values(sort_by, ignore_index=True)
    payload = {
        "columns": normalized.columns.tolist(),
        "dtypes": [str(dtype) for dtype in normalized.dtypes],
        "rows": normalized.astype(object)
        .where(pd.notna(normalized), None)
        .values.tolist(),
    }
    return canonical_sha256(payload)


def _count_rate(mask: np.ndarray, denominator: int) -> dict[str, Any]:
    count = int(np.asarray(mask, dtype=bool).sum())
    return {
        "count": count,
        "rate": None if denominator == 0 else float(count / denominator),
    }


def _correlation(left: np.ndarray, right: np.ndarray) -> tuple[float | None, str]:
    if len(left) < 2:
        return None, "undefined_fewer_than_two_observations"
    left_constant = bool(np.all(left == left[0]))
    right_constant = bool(np.all(right == right[0]))
    if left_constant or right_constant:
        if left_constant and right_constant and np.array_equal(left, right):
            return 1.0, "identical_constant_vectors"
        return None, "undefined_constant_vector"
    value = float(np.corrcoef(left, right)[0, 1])
    if not math.isfinite(value):
        return None, "undefined_non_finite_result"
    return value, "defined"


def _validate_quantiles(values: Sequence[float]) -> tuple[float, ...]:
    result: list[float] = []
    for value in values:
        if type(value) is not float or not math.isfinite(value):
            raise DatasetComparisonError("Probability quantiles must be exact floats.")
        if not 0.0 <= value <= 1.0:
            raise DatasetComparisonError("Probability quantiles must lie in [0, 1].")
        result.append(value)
    if result != sorted(set(result)):
        raise DatasetComparisonError(
            "Probability quantiles must be unique and increasing."
        )
    return tuple(result)


def _quantile_label(value: float) -> str:
    return format(value, ".12g")


def _validate_tie_epsilon(value: float) -> float:
    result = _strict_finite_float(value, "tie_epsilon")
    if result < 0.0:
        raise DatasetComparisonError("tie_epsilon must be non-negative.")
    return result


def _strict_finite_float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise DatasetComparisonError(f"{label} must be an exact finite float.")
    return value


def _exact_equal(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def _compact_value(value: Any) -> Any:
    encoded = json.dumps(value, sort_keys=True, default=str)
    if len(encoded) <= 500:
        return value
    return {"sha256": canonical_sha256(value), "serialized_length": len(encoded)}


__all__ = [
    "COMPARISON_OPERATION",
    "COMPARISON_SCHEMA_VERSION",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_TIE_EPSILON",
    "EXPLORATORY_PARENT_CHILD_PAIRS",
    "PROJECT_CONTRACT",
    "UNBIASED_PARENT_CHILD_PAIRS",
    "CompletedResearchV2Run",
    "DatasetComparisonError",
    "DatasetComparisonResult",
    "DatasetComparisonRunReference",
    "DatasetCompatibilityError",
    "build_comparison_result",
    "build_compatibility_summary",
    "build_evaluation_protocol_identity",
    "build_independent_row_identity",
    "build_model_configuration_identity",
    "build_normalized_identities",
    "build_oof_diagnostics",
    "default_dataset_comparison_id",
    "derive_model_family",
    "deterministic_comparison_identity",
    "is_exploratory_dataset",
    "load_completed_research_v2_run",
    "resolve_dataset_package_manifest",
]
