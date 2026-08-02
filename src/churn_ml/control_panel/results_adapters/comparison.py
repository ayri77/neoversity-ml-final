"""Paired and dataset comparison → Comparison entity adapter."""

from __future__ import annotations

from pathlib import Path

from src.churn_ml.control_panel.artifacts import ArtifactRecord
from src.churn_ml.control_panel.results_adapters._common import (
    bool_field,
    load_json,
    mapping_get,
    na,
    nested_number,
    payload,
    schema_field,
    state_field,
    unrecorded,
)
from src.churn_ml.control_panel.results_entities import (
    ResultEntity,
    ResultEntityKind,
    ResultLink,
)
from src.churn_ml.control_panel.results_fields import (
    ResultField,
    available,
    missing_unexpectedly,
)


class ComparisonAdapter:
    entity_kind = ResultEntityKind.COMPARISON
    reader_ids = ("paired_comparison", "dataset_comparison_v1")
    supported_schema_versions = frozenset({1})

    def adapt(
        self,
        artifact: ArtifactRecord,
        *,
        repository_root: Path,
        archived: bool = False,
    ) -> ResultEntity:
        del repository_root
        comparison_id = Path(artifact.relative_path).name
        if artifact.reader_id == "dataset_comparison_v1":
            return _adapt_dataset_comparison(artifact, comparison_id, archived=archived)
        return _adapt_paired_comparison(artifact, comparison_id, archived=archived)


def _adapt_paired_comparison(
    artifact: ArtifactRecord, comparison_id: str, *, archived: bool
) -> ResultEntity:
    aggregate = payload(artifact, "aggregate_summary.json") or load_json(
        artifact.root, "aggregate_summary.json"
    )
    decision = payload(artifact, "decision_report.json") or load_json(
        artifact.root, "decision_report.json"
    )
    schema = schema_field(
        None if aggregate is None else aggregate.get("schema_version"),
        expected=frozenset({1}),
        source="aggregate_summary.json.schema_version",
    )
    delta = nested_number(
        aggregate if isinstance(aggregate, dict) else None,
        "metrics",
        "balanced_accuracy",
        "candidate_minus_baseline",
        "mean",
        source="aggregate_summary.json",
        missing=missing_unexpectedly("Mean BA delta missing."),
    )
    # Baseline/candidate absolute values are not always present; deltas are authoritative.
    fields: dict[str, ResultField] = {
        "comparison_type": available("paired_research_v1", source="adapter"),
        "comparison_id": available(comparison_id, source="path"),
        "baseline": unrecorded(
            "Paired comparison packages do not embed absolute baseline run labels in summary."
        ),
        "candidate": unrecorded(
            "Paired comparison packages do not embed absolute candidate run labels in summary."
        ),
        "baseline_dataset": unrecorded(
            "Baseline dataset is not recorded in paired comparison summaries."
        ),
        "candidate_dataset": unrecorded(
            "Candidate dataset is not recorded in paired comparison summaries."
        ),
        "primary_metric_name": available(
            "balanced_accuracy_delta", source="adapter"
        ),
        "baseline_value": unrecorded("Absolute baseline BA not recorded in summary."),
        "candidate_value": unrecorded("Absolute candidate BA not recorded in summary."),
        "delta": delta,
        "decision": mapping_get(
            decision,
            "status",
            source="decision_report.json",
            missing=missing_unexpectedly("Decision status missing."),
        ),
        "compatibility": available(
            True, source="paired_comparison_contract"
        ),
        "model_family": na(
            "Comparisons are not model packages; model family does not apply."
        ),
    }
    return ResultEntity(
        kind=ResultEntityKind.COMPARISON,
        entity_id=comparison_id,
        display_label=f"Paired · {comparison_id}",
        reader_id=artifact.reader_id,
        source_path=artifact.relative_path,
        schema_version=schema,
        state=state_field(artifact),
        created_at=unrecorded("Created timestamp is not recorded for paired comparisons."),
        dataset_id=unrecorded("Single dataset ID is not recorded on paired comparison packages."),
        exploratory=bool_field(
            decision,
            "exploratory",
            source="decision_report.json",
            missing=available(False, source="default_non_exploratory"),
        )
        if isinstance(decision, dict) and "exploratory" in decision
        else available(False, source="default_non_exploratory"),
        fields=fields,
        lineage=(),
        diagnostics=tuple(
            [artifact.diagnostic] if artifact.diagnostic else []
        ),
        archived=archived,
    )


def _adapt_dataset_comparison(
    artifact: ArtifactRecord, comparison_id: str, *, archived: bool
) -> ResultEntity:
    summary = payload(artifact, "summary.json") or load_json(artifact.root, "summary.json")
    compatibility = payload(artifact, "compatibility_report.json") or load_json(
        artifact.root, "compatibility_report.json"
    )
    baseline_ref = payload(artifact, "baseline_run_reference.json") or load_json(
        artifact.root, "baseline_run_reference.json"
    )
    candidate_ref = payload(artifact, "candidate_run_reference.json") or load_json(
        artifact.root, "candidate_run_reference.json"
    )
    schema = schema_field(
        None if summary is None else summary.get("schema_version"),
        expected=frozenset({1}),
        source="summary.json.schema_version",
    )
    aggregate = (
        summary.get("aggregate_summary") if isinstance(summary, dict) else None
    )
    if not isinstance(aggregate, dict):
        aggregate = None
    delta = nested_number(
        aggregate,
        "metrics",
        "balanced_accuracy",
        "candidate_minus_baseline",
        "mean",
        source="summary.json.aggregate_summary",
        missing=missing_unexpectedly("Mean BA delta missing."),
    )
    baseline_dataset = mapping_get(
        baseline_ref,
        "dataset_id",
        source="baseline_run_reference.json",
        missing=missing_unexpectedly("Baseline dataset missing."),
    )
    candidate_dataset = mapping_get(
        candidate_ref,
        "dataset_id",
        source="candidate_run_reference.json",
        missing=missing_unexpectedly("Candidate dataset missing."),
    )
    baseline_run = mapping_get(
        baseline_ref,
        "run_id",
        source="baseline_run_reference.json",
        missing=missing_unexpectedly("Baseline run ID missing."),
    )
    candidate_run = mapping_get(
        candidate_ref,
        "run_id",
        source="candidate_run_reference.json",
        missing=missing_unexpectedly("Candidate run ID missing."),
    )
    model_family = mapping_get(
        baseline_ref,
        "model_family",
        source="baseline_run_reference.json",
        missing=unrecorded("Model family not recorded on comparison references."),
    )
    dataset_pair = available(
        f"{baseline_dataset.value} → {candidate_dataset.value}"
        if baseline_dataset.is_available() and candidate_dataset.is_available()
        else comparison_id,
        source="dataset_pair",
    )
    fields: dict[str, ResultField] = {
        "comparison_type": available("dataset_comparison_v1", source="adapter"),
        "comparison_id": available(comparison_id, source="path"),
        "baseline": baseline_run,
        "candidate": candidate_run,
        "baseline_dataset": baseline_dataset,
        "candidate_dataset": candidate_dataset,
        "dataset_pair": dataset_pair,
        "primary_metric_name": available(
            "balanced_accuracy_delta", source="adapter"
        ),
        "baseline_value": unrecorded(
            "Absolute baseline BA is not embedded in dataset comparison summary."
        ),
        "candidate_value": unrecorded(
            "Absolute candidate BA is not embedded in dataset comparison summary."
        ),
        "delta": delta,
        "decision": unrecorded(
            "Dataset comparison v1 records wins/ties/losses rather than a paired decision status."
        ),
        "compatibility": bool_field(
            compatibility,
            "compatible",
            source="compatibility_report.json",
            missing=missing_unexpectedly("Compatibility flag missing."),
        ),
        "repeat_wins": nested_number(
            aggregate,
            "metrics",
            "balanced_accuracy",
            "repeat_wins_ties_losses",
            "wins",
            source="summary.json",
            missing=unrecorded("Repeat wins not recorded."),
        ),
        "model_family": model_family,
        # Explicitly prevent blend misclassification.
        "is_blend": available(False, source="adapter"),
    }
    lineage = []
    if baseline_ref and baseline_ref.get("repository_relative_path"):
        lineage.append(
            ResultLink(
                relation="baseline_run",
                target_kind=ResultEntityKind.MODEL_RUN,
                target_id=str(baseline_ref.get("run_id") or ""),
                target_path=str(baseline_ref["repository_relative_path"]),
                resolved=True,
            )
        )
    if candidate_ref and candidate_ref.get("repository_relative_path"):
        lineage.append(
            ResultLink(
                relation="candidate_run",
                target_kind=ResultEntityKind.MODEL_RUN,
                target_id=str(candidate_ref.get("run_id") or ""),
                target_path=str(candidate_ref["repository_relative_path"]),
                resolved=True,
            )
        )
    label = f"Dataset · {comparison_id}"
    if model_family.is_available():
        label = f"Dataset · {model_family.value} · {comparison_id}"
    return ResultEntity(
        kind=ResultEntityKind.COMPARISON,
        entity_id=comparison_id,
        display_label=label,
        reader_id=artifact.reader_id,
        source_path=artifact.relative_path,
        schema_version=schema,
        state=state_field(artifact),
        created_at=unrecorded(
            "Created timestamp is not recorded for dataset comparisons."
        ),
        dataset_id=dataset_pair,
        exploratory=bool_field(
            compatibility,
            "exploratory",
            source="compatibility_report.json",
            missing=available(False, source="default_non_exploratory"),
        ),
        fields=fields,
        lineage=tuple(lineage),
        diagnostics=tuple([artifact.diagnostic] if artifact.diagnostic else []),
        archived=archived,
    )
