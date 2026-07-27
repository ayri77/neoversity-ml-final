from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from src.churn_ml.research_data import canonical_sha256
from src.churn_ml.research_protocol import (
    build_repeat_metrics,
    build_threshold_grid,
    calculate_outer_metrics,
    select_balanced_accuracy_threshold,
)
from src.churn_ml.research_v2_artifact_validation import validate_research_v2_run
from src.churn_ml.research_v2_config import ResearchV2Config


COMPARISON_SCHEMA_VERSION = 1
PRINCIPAL_METRICS = (
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "roc_auc",
    "average_precision",
    "brier_score",
)
HIGHER_IS_BETTER = {
    "balanced_accuracy": True,
    "sensitivity": True,
    "specificity": True,
    "roc_auc": True,
    "average_precision": True,
    "brier_score": False,
}
OUTER_KEYS = ["repeat", "repeat_seed", "outer_fold", "row_position"]
THRESHOLD_KEYS = [
    "repeat",
    "repeat_seed",
    "outer_fold",
    "threshold_selection_fold",
    "row_position",
]
FOLD_KEYS = ["repeat", "outer_fold"]


class PairedComparisonError(ValueError):
    """Raised when a paired comparison contract cannot be satisfied."""


class PairedCompatibilityError(PairedComparisonError):
    """Raised when two individually valid v2 runs are not pair-compatible."""

    def __init__(self, report: ComparisonCompatibilitySummary) -> None:
        self.report = report
        details = "; ".join(
            f"{issue.reason_code} at {issue.field_path}" for issue in report.issues
        )
        super().__init__(f"Comparison compatibility failed: {details}.")


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
class ComparisonCompatibilitySummary:
    schema_version: int
    compatible: bool
    issues: tuple[CompatibilityIssue, ...]
    compared_contracts: tuple[str, ...]
    intentionally_ignored_contracts: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "compatible": self.compatible,
            "issues": [issue.to_dict() for issue in self.issues],
            "compared_contracts": list(self.compared_contracts),
            "intentionally_ignored_contracts": list(
                self.intentionally_ignored_contracts
            ),
        }


@dataclass(frozen=True)
class ComparisonPolicy:
    schema_version: int
    policy_id: str
    tie_epsilon: float
    promising_min_mean_balanced_accuracy_delta: float
    promising_min_repeat_win_fraction: float
    threshold_stability_max_sample_standard_deviation: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy": {
                "id": self.policy_id,
                "tie_epsilon": self.tie_epsilon,
                "promising_min_mean_balanced_accuracy_delta": (
                    self.promising_min_mean_balanced_accuracy_delta
                ),
                "promising_min_repeat_win_fraction": (
                    self.promising_min_repeat_win_fraction
                ),
                "threshold_stability_max_sample_standard_deviation": (
                    self.threshold_stability_max_sample_standard_deviation
                ),
            },
        }


@dataclass(frozen=True)
class InputRunReference:
    schema_version: int
    role: str
    run_id: str
    repository_relative_path: str
    artifact_manifest_sha256: str
    artifact_manifest_file_sha256: str
    evaluation_plan_sha256: str
    dataset_version: str
    feature_pipeline_id: str
    candidate_adapter_id: str
    candidate_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "run_id": self.run_id,
            "repository_relative_path": self.repository_relative_path,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "artifact_manifest_file_sha256": self.artifact_manifest_file_sha256,
            "evaluation_plan_sha256": self.evaluation_plan_sha256,
            "dataset_version": self.dataset_version,
            "feature_pipeline_id": self.feature_pipeline_id,
            "candidate_adapter_id": self.candidate_adapter_id,
            "candidate_sha256": self.candidate_sha256,
        }


@dataclass(frozen=True)
class CompletedResearchV2Run:
    root: Path
    config: ResearchV2Config
    metadata: dict[str, Any]
    manifest: dict[str, Any]
    dataset_fingerprints: dict[str, Any]
    evaluation_plan_identity: dict[str, Any]
    candidate_identity: dict[str, Any]
    outer_assignments: pd.DataFrame
    threshold_assignments: pd.DataFrame
    outer_predictions: pd.DataFrame
    threshold_predictions: pd.DataFrame
    selected_thresholds: pd.DataFrame
    fold_metrics: pd.DataFrame
    repeat_metrics: pd.DataFrame

    def reference(self, role: str) -> InputRunReference:
        manifest_path = self.root / "artifact_manifest.json"
        return InputRunReference(
            schema_version=COMPARISON_SCHEMA_VERSION,
            role=role,
            run_id=str(self.metadata["run_id"]),
            repository_relative_path=self.root.relative_to(
                self.config.project_root
            ).as_posix(),
            artifact_manifest_sha256=str(self.manifest["manifest_sha256"]),
            artifact_manifest_file_sha256=hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            evaluation_plan_sha256=str(self.evaluation_plan_identity["sha256"]),
            dataset_version=self.config.dataset_version,
            feature_pipeline_id=self.config.pipeline_id,
            candidate_adapter_id=self.config.adapter_id,
            candidate_sha256=str(self.candidate_identity["sha256"]),
        )


@dataclass(frozen=True)
class PairedMetricResults:
    repeat_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    aggregate_summary: dict[str, Any]


@dataclass(frozen=True)
class BlendDiagnosticResults:
    repeat_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    summary: dict[str, Any]


@dataclass(frozen=True)
class PairedComparisonResult:
    compatibility: ComparisonCompatibilitySummary
    baseline_reference: InputRunReference
    candidate_reference: InputRunReference
    paired_metrics: PairedMetricResults
    prediction_comparison: dict[str, Any]
    blend: BlendDiagnosticResults
    decision_report: dict[str, Any]


def load_comparison_policy(path: Path) -> ComparisonPolicy:
    payload = _read_yaml(path)
    if set(payload) != {"schema_version", "policy"}:
        raise PairedComparisonError("Comparison policy root keys differ.")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise PairedComparisonError(
            "comparison policy schema_version must be integer 1."
        )
    policy = payload["policy"]
    if not isinstance(policy, Mapping):
        raise PairedComparisonError("comparison policy.policy must be a mapping.")
    expected = {
        "id",
        "tie_epsilon",
        "promising_min_mean_balanced_accuracy_delta",
        "promising_min_repeat_win_fraction",
        "threshold_stability_max_sample_standard_deviation",
    }
    if set(policy) != expected:
        raise PairedComparisonError("Comparison policy keys differ.")
    policy_id = policy["id"]
    if not isinstance(policy_id, str) or not policy_id:
        raise PairedComparisonError("comparison policy id must be a non-empty string.")
    numeric = {
        name: _strict_finite_float(policy[name], f"comparison policy {name}")
        for name in expected - {"id"}
    }
    if numeric["tie_epsilon"] < 0.0:
        raise PairedComparisonError(
            "comparison policy tie_epsilon must be non-negative."
        )
    if not 0.0 <= numeric["promising_min_repeat_win_fraction"] <= 1.0:
        raise PairedComparisonError(
            "comparison policy promising_min_repeat_win_fraction must be in [0, 1]."
        )
    if numeric["threshold_stability_max_sample_standard_deviation"] < 0.0:
        raise PairedComparisonError(
            "comparison policy threshold stability limit must be non-negative."
        )
    return ComparisonPolicy(
        schema_version=1,
        policy_id=policy_id,
        tie_epsilon=numeric["tie_epsilon"],
        promising_min_mean_balanced_accuracy_delta=numeric[
            "promising_min_mean_balanced_accuracy_delta"
        ],
        promising_min_repeat_win_fraction=numeric["promising_min_repeat_win_fraction"],
        threshold_stability_max_sample_standard_deviation=numeric[
            "threshold_stability_max_sample_standard_deviation"
        ],
    )


def load_completed_research_v2_run(
    run_dir: Path,
    *,
    project_root: Path,
) -> CompletedResearchV2Run:
    root = _validated_run_path(run_dir, project_root)
    resolved = _read_yaml(root / "resolved_config.yaml")
    plan = resolved.get("evaluation_plan")
    if not isinstance(plan, dict):
        raise PairedComparisonError(
            "Input resolved_config.evaluation_plan must be a mapping."
        )
    payload = dict(resolved)
    payload.pop("evaluation_plan")
    plan_relative = payload.get("evaluation_plan_path")
    if not isinstance(plan_relative, str):
        raise PairedComparisonError(
            "Input resolved_config.evaluation_plan_path must be a string."
        )
    config = ResearchV2Config(
        payload=payload,
        plan_payload=plan,
        source_path=project_root.resolve() / "resolved_input_run.yaml",
        plan_path=(project_root.resolve() / plan_relative).resolve(),
        project_root=project_root.resolve(),
    )
    metadata = _read_json(root / "run_metadata.json")
    hashes = metadata.get("hashes")
    if not isinstance(hashes, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in hashes.items()
    ):
        raise PairedComparisonError("Input run_metadata.hashes is malformed.")
    validate_research_v2_run(
        root,
        config,
        expected_hashes=dict(hashes),
        require_success=True,
        verify_manifest=True,
    )
    return CompletedResearchV2Run(
        root=root,
        config=config,
        metadata=metadata,
        manifest=_read_json(root / "artifact_manifest.json"),
        dataset_fingerprints=_read_json(root / "dataset_fingerprints.json"),
        evaluation_plan_identity=_read_json(
            root / "identities" / "evaluation_plan.json"
        ),
        candidate_identity=_read_json(root / "identities" / "candidate.json"),
        outer_assignments=pd.read_parquet(
            root / "splits" / "outer_assignments.parquet"
        ),
        threshold_assignments=pd.read_parquet(
            root / "splits" / "threshold_selection_assignments.parquet"
        ),
        outer_predictions=pd.read_parquet(
            root / "predictions" / "outer_validation.parquet"
        ),
        threshold_predictions=pd.read_parquet(
            root / "predictions" / "threshold_selection_oof.parquet"
        ),
        selected_thresholds=pd.read_csv(
            root / "thresholds" / "selected_thresholds.csv"
        ),
        fold_metrics=pd.read_csv(root / "metrics" / "outer_folds.csv"),
        repeat_metrics=pd.read_csv(root / "metrics" / "repeats.csv"),
    )


def build_compatibility_summary(
    baseline: CompletedResearchV2Run,
    candidate: CompletedResearchV2Run,
) -> ComparisonCompatibilitySummary:
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

    baseline_plan = baseline.config.plan_payload
    candidate_plan = candidate.config.plan_payload
    compare(
        baseline.config.payload.get("schema_version"),
        candidate.config.payload.get("schema_version"),
        "SCHEMA_VERSION_MISMATCH",
        "resolved_config.schema_version",
    )
    compare(
        baseline_plan.get("schema_version"),
        candidate_plan.get("schema_version"),
        "PROTOCOL_VERSION_MISMATCH",
        "resolved_config.evaluation_plan.schema_version",
    )
    compare(
        baseline.evaluation_plan_identity["sha256"],
        candidate.evaluation_plan_identity["sha256"],
        "EVALUATION_PLAN_IDENTITY_MISMATCH",
        "identities.evaluation_plan.sha256",
    )
    compare(
        baseline_plan.get("plan"),
        candidate_plan.get("plan"),
        "EVALUATION_PLAN_ID_MISMATCH",
        "resolved_config.evaluation_plan.plan",
    )
    compare(
        baseline.config.dataset_version,
        candidate.config.dataset_version,
        "DATASET_VERSION_MISMATCH",
        "resolved_config.dataset.version",
    )
    compare(
        baseline.dataset_fingerprints.get("files"),
        candidate.dataset_fingerprints.get("files"),
        "DATASET_CONTENT_MISMATCH",
        "dataset_fingerprints.files",
    )
    compare(
        baseline.dataset_fingerprints.get("row_count"),
        candidate.dataset_fingerprints.get("row_count"),
        "SAMPLE_COUNT_MISMATCH",
        "dataset_fingerprints.row_count",
    )
    compare(
        baseline.dataset_fingerprints.get("row_position_identity"),
        candidate.dataset_fingerprints.get("row_position_identity"),
        "SAMPLE_ORDER_IDENTITY_MISMATCH",
        "dataset_fingerprints.row_position_identity",
    )
    compare(
        baseline.dataset_fingerprints.get("target"),
        candidate.dataset_fingerprints.get("target"),
        "TARGET_IDENTITY_MISMATCH",
        "dataset_fingerprints.target",
    )
    compare(
        baseline_plan.get("metrics"),
        candidate_plan.get("metrics"),
        "METRIC_CONTRACT_MISMATCH",
        "resolved_config.evaluation_plan.metrics",
    )
    compare(
        baseline_plan.get("aggregation"),
        candidate_plan.get("aggregation"),
        "METRIC_AGGREGATION_MISMATCH",
        "resolved_config.evaluation_plan.aggregation",
    )
    compare(
        baseline_plan.get("threshold_policy"),
        candidate_plan.get("threshold_policy"),
        "THRESHOLD_POLICY_MISMATCH",
        "resolved_config.evaluation_plan.threshold_policy",
    )
    baseline_grid = build_threshold_grid(baseline_plan["threshold_policy"]).tolist()
    candidate_grid = build_threshold_grid(candidate_plan["threshold_policy"]).tolist()
    compare(
        baseline_grid,
        candidate_grid,
        "THRESHOLD_GRID_MISMATCH",
        "resolved_config.evaluation_plan.threshold_policy.grid",
    )
    compare(
        baseline_plan.get("outer_evaluation"),
        candidate_plan.get("outer_evaluation"),
        "OUTER_PROTOCOL_MISMATCH",
        "resolved_config.evaluation_plan.outer_evaluation",
    )
    compare(
        baseline_plan.get("threshold_selection"),
        candidate_plan.get("threshold_selection"),
        "THRESHOLD_SELECTION_PROTOCOL_MISMATCH",
        "resolved_config.evaluation_plan.threshold_selection",
    )
    compare(
        baseline_plan.get("dataset", {}).get("target", {}).get("positive_label"),
        candidate_plan.get("dataset", {}).get("target", {}).get("positive_label"),
        "POSITIVE_CLASS_SEMANTICS_MISMATCH",
        "resolved_config.evaluation_plan.dataset.target.positive_label",
    )
    compare(
        baseline.candidate_identity.get("canonical", {}).get("probability_semantics"),
        candidate.candidate_identity.get("canonical", {}).get("probability_semantics"),
        "PROBABILITY_SEMANTICS_MISMATCH",
        "identities.candidate.canonical.probability_semantics",
    )
    compare(
        baseline_plan.get("threshold_policy", {}).get("comparison"),
        candidate_plan.get("threshold_policy", {}).get("comparison"),
        "LABEL_CONVENTION_MISMATCH",
        "resolved_config.evaluation_plan.threshold_policy.comparison",
    )
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
    _compare_frame(
        baseline.outer_predictions[OUTER_KEYS + ["target"]],
        candidate.outer_predictions[OUTER_KEYS + ["target"]],
        issues,
        reason_code="OUTER_TARGET_ORDER_MISMATCH",
        field_path="predictions.outer_validation.keys_and_target",
        sort_by=OUTER_KEYS,
    )
    _compare_frame(
        baseline.threshold_predictions[THRESHOLD_KEYS + ["target"]],
        candidate.threshold_predictions[THRESHOLD_KEYS + ["target"]],
        issues,
        reason_code="THRESHOLD_TARGET_ORDER_MISMATCH",
        field_path="predictions.threshold_selection_oof.keys_and_target",
        sort_by=THRESHOLD_KEYS,
    )
    compare(
        _fold_product(baseline),
        _fold_product(candidate),
        "FOLD_PRODUCT_MISMATCH",
        "repeat_outer_fold_cartesian_product",
    )
    return ComparisonCompatibilitySummary(
        schema_version=COMPARISON_SCHEMA_VERSION,
        compatible=not issues,
        issues=tuple(issues),
        compared_contracts=(
            "schema_and_protocol_versions",
            "evaluation_plan_identity",
            "dataset_version_and_content",
            "outer_and_threshold_assignments",
            "target_vector_and_sample_order",
            "metric_and_aggregation_contract",
            "threshold_policy_and_grid",
            "positive_class_probability_and_label_semantics",
            "repeat_fold_cartesian_product",
        ),
        intentionally_ignored_contracts=(
            "feature_pipeline_identity",
            "candidate_adapter_identity",
            "complete_candidate_identity",
            "source_provenance",
            "run_id",
            "timestamps",
        ),
    )


def build_paired_metrics(
    baseline: CompletedResearchV2Run,
    candidate: CompletedResearchV2Run,
    *,
    tie_epsilon: float,
) -> PairedMetricResults:
    epsilon = _validate_tie_epsilon(tie_epsilon)
    _require_compatible(baseline, candidate)
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
    return PairedMetricResults(repeat_metrics, fold_metrics, summary)


def build_aggregate_metrics(
    repeat_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    baseline_thresholds: pd.DataFrame,
    candidate_thresholds: pd.DataFrame,
    *,
    tie_epsilon: float,
) -> dict[str, Any]:
    epsilon = _validate_tie_epsilon(tie_epsilon)
    metric_summaries: dict[str, Any] = {}
    for metric in PRINCIPAL_METRICS:
        repeat_delta = repeat_metrics[f"{metric}_delta"].to_numpy(dtype=float)
        fold_delta = fold_metrics[f"{metric}_delta"].to_numpy(dtype=float)
        metric_summaries[metric] = {
            "candidate_minus_baseline": {
                "mean": float(repeat_delta.mean()),
                "median": float(np.median(repeat_delta)),
                "sample_standard_deviation": _sample_sd(repeat_delta),
            },
            "repeat_wins_ties_losses": _wins_ties_losses(
                repeat_delta,
                epsilon,
                higher_is_better=HIGHER_IS_BETTER[metric],
            ),
            "fold_wins_ties_losses_descriptive_only": _wins_ties_losses(
                fold_delta,
                epsilon,
                higher_is_better=HIGHER_IS_BETTER[metric],
            ),
            "optimization_direction": (
                "higher_is_better" if HIGHER_IS_BETTER[metric] else "lower_is_better"
            ),
            "delta_interpretation": (
                "positive_favors_candidate"
                if HIGHER_IS_BETTER[metric]
                else "negative_favors_candidate"
            ),
        }
        if metric == "brier_score":
            metric_summaries[metric]["mean_improvement_baseline_minus_candidate"] = (
                float(-repeat_delta.mean())
            )
    baseline_values = baseline_thresholds["selected_threshold"].to_numpy(dtype=float)
    candidate_values = candidate_thresholds["selected_threshold"].to_numpy(dtype=float)
    paired_thresholds = _aligned_threshold_values(
        baseline_thresholds, candidate_thresholds
    )
    threshold_deltas = paired_thresholds[1] - paired_thresholds[0]
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "primary_unit": "repeat",
        "fold_results": "descriptive_only",
        "tie_epsilon": epsilon,
        "metrics": metric_summaries,
        "sensitivity_specificity_trade_off": _trade_off_summary(
            repeat_metrics, epsilon
        ),
        "thresholds": {
            "baseline": _distribution_summary(baseline_values),
            "candidate": _distribution_summary(candidate_values),
            "candidate_minus_baseline": {
                **_distribution_summary(threshold_deltas),
                "paired_count": len(threshold_deltas),
            },
        },
        "formal_inference": {
            "performed": False,
            "reason": "overlapping repeated cross-validation observations",
        },
    }


def build_prediction_comparison(
    baseline: CompletedResearchV2Run,
    candidate: CompletedResearchV2Run,
    *,
    probability_difference_quantiles: Sequence[float] | None = None,
) -> dict[str, Any]:
    _require_compatible(baseline, candidate)
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
        raise PairedComparisonError("Aligned outer prediction targets differ.")
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


def build_fixed_blend_diagnostic(
    baseline: CompletedResearchV2Run,
    candidate: CompletedResearchV2Run,
    *,
    tie_epsilon: float,
) -> BlendDiagnosticResults:
    epsilon = _validate_tie_epsilon(tie_epsilon)
    _require_compatible(baseline, candidate)
    baseline_threshold, candidate_threshold = _aligned_frames(
        baseline.threshold_predictions,
        candidate.threshold_predictions,
        THRESHOLD_KEYS,
        "threshold-selection prediction",
    )
    baseline_outer, candidate_outer = _aligned_frames(
        baseline.outer_predictions,
        candidate.outer_predictions,
        OUTER_KEYS,
        "outer prediction",
    )
    if not np.array_equal(
        baseline_threshold["target"].to_numpy(),
        candidate_threshold["target"].to_numpy(),
    ):
        raise PairedComparisonError("Aligned threshold-selection targets differ.")
    if not np.array_equal(
        baseline_outer["target"].to_numpy(),
        candidate_outer["target"].to_numpy(),
    ):
        raise PairedComparisonError("Aligned outer-validation targets differ.")

    blend_outer_frames: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []
    policy = baseline.config.plan_payload["threshold_policy"]
    for repeat, outer_fold in _fold_product(baseline):
        threshold_mask = (baseline_threshold["repeat"] == repeat) & (
            baseline_threshold["outer_fold"] == outer_fold
        )
        baseline_selection = baseline_threshold.loc[threshold_mask]
        candidate_selection = candidate_threshold.loc[threshold_mask]
        blend_selection_probability = 0.5 * baseline_selection["probability"].to_numpy(
            dtype=float
        ) + 0.5 * candidate_selection["probability"].to_numpy(dtype=float)
        selection = select_balanced_accuracy_threshold(
            baseline_selection["target"],
            blend_selection_probability,
            policy,
        )
        outer_mask = (baseline_outer["repeat"] == repeat) & (
            baseline_outer["outer_fold"] == outer_fold
        )
        baseline_fold = baseline_outer.loc[outer_mask]
        candidate_fold = candidate_outer.loc[outer_mask]
        blend_probability = 0.5 * baseline_fold["probability"].to_numpy(
            dtype=float
        ) + 0.5 * candidate_fold["probability"].to_numpy(dtype=float)
        blend_prediction = (blend_probability >= selection.threshold).astype("int8")
        blend_frame = baseline_fold[OUTER_KEYS + ["target"]].copy()
        blend_frame["probability"] = blend_probability
        blend_frame["selected_threshold"] = selection.threshold
        blend_frame["prediction"] = blend_prediction
        blend_outer_frames.append(blend_frame)
        baseline_metrics = _metrics_from_predictions(baseline_fold)
        candidate_metrics = _metrics_from_predictions(candidate_fold)
        blend_metrics = _metrics_from_predictions(blend_frame)
        record: dict[str, Any] = {
            "repeat": repeat,
            "outer_fold": outer_fold,
            "validation_rows": len(blend_frame),
            "blend_threshold": selection.threshold,
            "blend_threshold_selection_balanced_accuracy": (
                selection.balanced_accuracy
            ),
            "blend_threshold_status": selection.status,
            "blend_threshold_degenerate": selection.degenerate,
            "baseline_threshold": float(baseline_fold["selected_threshold"].iloc[0]),
            "candidate_threshold": float(candidate_fold["selected_threshold"].iloc[0]),
        }
        _add_blend_metric_triplet(
            record, baseline_metrics, candidate_metrics, blend_metrics
        )
        fold_records.append(record)
    blend_outer = pd.concat(blend_outer_frames, ignore_index=True)
    blend_repeat_base = build_repeat_metrics(blend_outer)
    repeat_records: list[dict[str, Any]] = []
    for repeat in sorted(blend_repeat_base["repeat"].unique()):
        blend_frame = blend_outer.loc[blend_outer["repeat"] == repeat]
        baseline_frame = baseline_outer.loc[baseline_outer["repeat"] == repeat]
        candidate_frame = candidate_outer.loc[candidate_outer["repeat"] == repeat]
        baseline_metrics = _metrics_from_predictions(baseline_frame)
        candidate_metrics = _metrics_from_predictions(candidate_frame)
        blend_metrics = _metrics_from_predictions(blend_frame)
        thresholds = [
            record["blend_threshold"]
            for record in fold_records
            if record["repeat"] == repeat
        ]
        record = {
            "repeat": int(repeat),
            "validation_rows": len(blend_frame),
            "blend_median_threshold": float(np.median(thresholds)),
        }
        _add_blend_metric_triplet(
            record, baseline_metrics, candidate_metrics, blend_metrics
        )
        repeat_records.append(record)
    repeat_metrics = pd.DataFrame(repeat_records)
    fold_metrics = pd.DataFrame(fold_records)
    summary: dict[str, Any] = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "diagnostic_only": True,
        "blend_weights": {"baseline": 0.5, "candidate": 0.5},
        "threshold_selection_source": (
            "aligned_threshold_selection_oof_probabilities_only"
        ),
        "outer_validation_target_used_for_threshold_selection": False,
        "weight_search_performed": False,
        "metrics": {},
    }
    for metric in PRINCIPAL_METRICS:
        versus: dict[str, Any] = {}
        absolute = {
            component: _distribution_summary(
                repeat_metrics[f"{component}_{metric}"].to_numpy(dtype=float)
            )
            for component in ("baseline", "candidate", "blend")
        }
        for component in ("baseline", "candidate"):
            values = repeat_metrics[f"blend_minus_{component}_{metric}"].to_numpy(
                dtype=float
            )
            versus[component] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "sample_standard_deviation": _sample_sd(values),
                "repeat_wins_ties_losses": _wins_ties_losses(
                    values,
                    epsilon,
                    higher_is_better=HIGHER_IS_BETTER[metric],
                ),
                "delta_interpretation": (
                    "positive_favors_blend"
                    if HIGHER_IS_BETTER[metric]
                    else "negative_favors_blend"
                ),
            }
        summary["metrics"][metric] = {
            "repeat_distributions": absolute,
            "blend_versus": versus,
        }
    summary["thresholds"] = _distribution_summary(
        fold_metrics["blend_threshold"].to_numpy(dtype=float)
    )
    return BlendDiagnosticResults(repeat_metrics, fold_metrics, summary)


def build_decision_status(
    paired_metrics: PairedMetricResults,
    blend: BlendDiagnosticResults,
    policy: ComparisonPolicy,
) -> dict[str, Any]:
    balanced = paired_metrics.aggregate_summary["metrics"]["balanced_accuracy"]
    mean_delta = float(balanced["candidate_minus_baseline"]["mean"])
    counts = balanced["repeat_wins_ties_losses"]
    repeat_count = int(counts["wins"] + counts["ties"] + counts["losses"])
    win_fraction = counts["wins"] / repeat_count
    if (
        mean_delta >= policy.promising_min_mean_balanced_accuracy_delta
        and win_fraction >= policy.promising_min_repeat_win_fraction
    ):
        status = "promising"
    elif mean_delta > policy.tie_epsilon or counts["wins"] > counts["losses"]:
        status = "mixed"
    else:
        status = "not_improved"
    candidate_threshold_sd = paired_metrics.aggregate_summary["thresholds"][
        "candidate"
    ]["sample_standard_deviation"]
    blend_balanced = blend.summary["metrics"]["balanced_accuracy"]["blend_versus"]
    blend_improved_over_both = all(
        blend_balanced[component]["mean"] > policy.tie_epsilon
        for component in ("baseline", "candidate")
    )
    repeat_delta = paired_metrics.repeat_metrics["balanced_accuracy_delta"].to_numpy(
        dtype=float
    )
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "policy": policy.to_dict(),
        "status": status,
        "status_kind": "deterministic_development_guidance",
        "candidate_mean_repeat_balanced_accuracy_delta": mean_delta,
        "repeat_wins_ties_losses": counts,
        "every_repeat_improved": bool(np.all(repeat_delta > policy.tie_epsilon)),
        "sensitivity_specificity_trade_off": paired_metrics.aggregate_summary[
            "sensitivity_specificity_trade_off"
        ],
        "threshold_stability": {
            "candidate_sample_standard_deviation": candidate_threshold_sd,
            "policy_maximum": (
                policy.threshold_stability_max_sample_standard_deviation
            ),
            "within_policy_limit": (
                candidate_threshold_sd is None
                or candidate_threshold_sd
                <= policy.threshold_stability_max_sample_standard_deviation
            ),
        },
        "fixed_blend_improved_over_both": blend_improved_over_both,
        "formal_inference_performed": False,
        "deployment_recommendation": None,
    }


def build_comparison_result(
    baseline: CompletedResearchV2Run,
    candidate: CompletedResearchV2Run,
    policy: ComparisonPolicy,
) -> PairedComparisonResult:
    compatibility = build_compatibility_summary(baseline, candidate)
    if not compatibility.compatible:
        raise PairedCompatibilityError(compatibility)
    paired_metrics = build_paired_metrics(
        baseline, candidate, tie_epsilon=policy.tie_epsilon
    )
    prediction = build_prediction_comparison(baseline, candidate)
    blend = build_fixed_blend_diagnostic(
        baseline, candidate, tie_epsilon=policy.tie_epsilon
    )
    decision = build_decision_status(paired_metrics, blend, policy)
    return PairedComparisonResult(
        compatibility=compatibility,
        baseline_reference=baseline.reference("baseline"),
        candidate_reference=candidate.reference("candidate"),
        paired_metrics=paired_metrics,
        prediction_comparison=prediction,
        blend=blend,
        decision_report=decision,
    )


def deterministic_comparison_identity(
    baseline_reference: InputRunReference,
    candidate_reference: InputRunReference,
    policy: ComparisonPolicy,
) -> str:
    canonical = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "baseline_manifest_sha256": baseline_reference.artifact_manifest_sha256,
        "candidate_manifest_sha256": candidate_reference.artifact_manifest_sha256,
        "comparison_policy": policy.to_dict(),
        "blend": {"baseline_weight": 0.5, "candidate_weight": 0.5},
    }
    return canonical_sha256(canonical)


def _require_compatible(
    baseline: CompletedResearchV2Run,
    candidate: CompletedResearchV2Run,
) -> None:
    report = build_compatibility_summary(baseline, candidate)
    if not report.compatible:
        raise PairedCompatibilityError(report)


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


def _add_blend_metric_triplet(
    record: dict[str, Any],
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    blend_metrics: Mapping[str, Any],
) -> None:
    for metric in PRINCIPAL_METRICS:
        baseline_value = float(baseline_metrics[metric])
        candidate_value = float(candidate_metrics[metric])
        blend_value = float(blend_metrics[metric])
        record[f"baseline_{metric}"] = baseline_value
        record[f"candidate_{metric}"] = candidate_value
        record[f"blend_{metric}"] = blend_value
        record[f"blend_minus_baseline_{metric}"] = blend_value - baseline_value
        record[f"blend_minus_candidate_{metric}"] = blend_value - candidate_value


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
        raise PairedComparisonError(f"{label} keys contain duplicates.")
    baseline_sorted = baseline.sort_values(keys, ignore_index=True)
    candidate_sorted = candidate.sort_values(keys, ignore_index=True)
    if not baseline_sorted[keys].equals(candidate_sorted[keys]):
        raise PairedComparisonError(f"{label} key coverage differs.")
    return baseline_sorted, candidate_sorted


def _aligned_threshold_values(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    baseline_sorted, candidate_sorted = _aligned_frames(
        baseline, candidate, FOLD_KEYS, "selected threshold"
    )
    return (
        baseline_sorted["selected_threshold"].to_numpy(dtype=float),
        candidate_sorted["selected_threshold"].to_numpy(dtype=float),
    )


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


def _wins_ties_losses(
    deltas: np.ndarray,
    epsilon: float,
    *,
    higher_is_better: bool,
) -> dict[str, int]:
    directional = deltas if higher_is_better else -deltas
    return {
        "wins": int((directional > epsilon).sum()),
        "ties": int((np.abs(directional) <= epsilon).sum()),
        "losses": int((directional < -epsilon).sum()),
    }


def _trade_off_summary(
    repeat_metrics: pd.DataFrame,
    epsilon: float,
) -> dict[str, Any]:
    sensitivity = repeat_metrics["sensitivity_delta"].to_numpy(dtype=float)
    specificity = repeat_metrics["specificity_delta"].to_numpy(dtype=float)
    return {
        "mean_sensitivity_delta": float(sensitivity.mean()),
        "mean_specificity_delta": float(specificity.mean()),
        "both_improved_repeats": int(
            ((sensitivity > epsilon) & (specificity > epsilon)).sum()
        ),
        "sensitivity_up_specificity_down_repeats": int(
            ((sensitivity > epsilon) & (specificity < -epsilon)).sum()
        ),
        "sensitivity_down_specificity_up_repeats": int(
            ((sensitivity < -epsilon) & (specificity > epsilon)).sum()
        ),
        "both_declined_repeats": int(
            ((sensitivity < -epsilon) & (specificity < -epsilon)).sum()
        ),
        "other_or_tied_repeats": int(
            (
                ~(
                    ((sensitivity > epsilon) & (specificity > epsilon))
                    | ((sensitivity > epsilon) & (specificity < -epsilon))
                    | ((sensitivity < -epsilon) & (specificity > epsilon))
                    | ((sensitivity < -epsilon) & (specificity < -epsilon))
                )
            ).sum()
        ),
    }


def _distribution_summary(values: np.ndarray) -> dict[str, Any]:
    if not len(values):
        raise PairedComparisonError("Cannot summarize an empty distribution.")
    return {
        "count": len(values),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "range": float(values.max() - values.min()),
        "median": float(np.median(values)),
        "sample_standard_deviation": _sample_sd(values),
    }


def _sample_sd(values: np.ndarray) -> float | None:
    return float(values.std(ddof=1)) if len(values) > 1 else None


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
            raise PairedComparisonError("Probability quantiles must be exact floats.")
        if not 0.0 <= value <= 1.0:
            raise PairedComparisonError("Probability quantiles must lie in [0, 1].")
        result.append(value)
    if result != sorted(set(result)):
        raise PairedComparisonError(
            "Probability quantiles must be unique and increasing."
        )
    return tuple(result)


def _quantile_label(value: float) -> str:
    return format(value, ".12g")


def _validate_tie_epsilon(value: float) -> float:
    result = _strict_finite_float(value, "tie_epsilon")
    if result < 0.0:
        raise PairedComparisonError("tie_epsilon must be non-negative.")
    return result


def _strict_finite_float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise PairedComparisonError(f"{label} must be an exact finite float.")
    return value


def _validated_run_path(path: Path, project_root: Path) -> Path:
    root = project_root.resolve()
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise PairedComparisonError(
            f"Input run directory does not exist: {path}."
        ) from error
    if not resolved.is_dir():
        raise PairedComparisonError(f"Input run path is not a directory: {path}.")
    if resolved == root or root not in resolved.parents:
        raise PairedComparisonError("Input run directory escapes the repository.")
    normalized = resolved.relative_to(root).as_posix().lower()
    forbidden = (
        "data/raw/",
        "data/test/",
        "submissions/",
        "kaggle",
        "autogluon",
        "sample_submission",
        "competition_test",
        "competition-test",
        "x_test",
        "final-submission",
        "final_submission",
        "final-model",
        "final_model",
        "model-artifact",
        "model_artifact",
        "models/",
        "/models/",
    )
    if any(marker in normalized for marker in forbidden):
        raise PairedComparisonError("Input path is in a forbidden asset namespace.")
    return resolved


def _exact_equal(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def _compact_value(value: Any) -> Any:
    encoded = json.dumps(value, sort_keys=True, default=str)
    if len(encoded) <= 500:
        return value
    return {"sha256": canonical_sha256(value), "serialized_length": len(encoded)}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PairedComparisonError(f"Cannot read JSON artifact: {path}.") from error
    if not isinstance(value, dict):
        raise PairedComparisonError(f"JSON artifact must be a mapping: {path}.")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise PairedComparisonError(f"Cannot read YAML artifact: {path}.") from error
    if not isinstance(value, dict):
        raise PairedComparisonError(f"YAML artifact must be a mapping: {path}.")
    return value
