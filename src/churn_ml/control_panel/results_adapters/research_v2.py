"""Research v2 → Model run adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.churn_ml.control_panel.artifacts import ArtifactRecord
from src.churn_ml.control_panel.dataset_identity import read_dataset_identity_safe
from src.churn_ml.control_panel.presentation import (
    normalize_mode,
    normalize_model_family,
    parse_run_id_timestamp,
)
from src.churn_ml.control_panel.results_adapters._common import (
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
)
from src.churn_ml.control_panel.results_fields import (
    ResultField,
    available,
    from_optional,
    missing_unexpectedly,
)


class ResearchV2Adapter:
    entity_kind = ResultEntityKind.MODEL_RUN
    reader_ids = ("research_v2",)
    supported_schema_versions = frozenset({2})

    def adapt(
        self,
        artifact: ArtifactRecord,
        *,
        repository_root: Path,
        archived: bool = False,
    ) -> ResultEntity:
        del repository_root
        meta = payload(artifact, "run_metadata.json") or load_json(
            artifact.root, "run_metadata.json"
        )
        metrics_root = payload(artifact, "metrics/aggregate.json") or load_json(
            artifact.root, "metrics/aggregate.json"
        )
        thresholds = payload(artifact, "thresholds/threshold_summary.json") or load_json(
            artifact.root, "thresholds/threshold_summary.json"
        )
        provenance = load_json(artifact.root, "dataset_provenance.json")
        identity = read_dataset_identity_safe(artifact.root)

        schema = schema_field(
            None if meta is None else meta.get("schema_version"),
            expected=self.supported_schema_versions,
            source="run_metadata.json.schema_version",
        )
        run_id = _text(meta, "run_id") or Path(artifact.relative_path).name
        created = _created(meta, run_id)
        dataset = _dataset(identity, provenance, meta)
        parent = _parent_dataset(identity, provenance, dataset)
        model_family = _model_family(meta, artifact.relative_path)
        mode = _mode(meta, artifact.relative_path)
        adapter_id = _text(meta, "candidate_adapter_id")
        pipeline = _text(meta, "feature_pipeline_id")
        diagnostics: list[str] = []
        if identity.diagnostic:
            diagnostics.append(identity.diagnostic)
        if artifact.diagnostic:
            diagnostics.append(artifact.diagnostic)

        metrics = (
            metrics_root.get("metrics") if isinstance(metrics_root, dict) else None
        )
        if not isinstance(metrics, dict):
            metrics = None
        ba = nested_number(
            metrics,
            "balanced_accuracy",
            "mean",
            source="metrics/aggregate.json",
            missing=missing_unexpectedly(
                "Balanced Accuracy mean is required for completed Research v2 runs."
            )
            if artifact.state == "completed"
            else unrecorded("Metrics unavailable for incomplete runs."),
        )
        ba_std = nested_number(
            metrics,
            "balanced_accuracy",
            "sample_standard_deviation",
            source="metrics/aggregate.json",
            missing=unrecorded("CV std was not recorded for this run."),
        )
        ba_min = nested_number(
            metrics,
            "balanced_accuracy",
            "minimum",
            source="metrics/aggregate.json",
            missing=unrecorded("CV minimum was not recorded for this run."),
        )
        ba_max = nested_number(
            metrics,
            "balanced_accuracy",
            "maximum",
            source="metrics/aggregate.json",
            missing=unrecorded("CV maximum was not recorded for this run."),
        )

        fields: dict[str, ResultField] = {
            "run_type": available("research_v2", source="adapter"),
            "framework": available("Research v2", source="adapter"),
            "run_id": available(run_id, source="run_metadata.json.run_id"),
            "model_family": model_family,
            "mode": mode,
            "configuration": from_optional(
                adapter_id,
                source="run_metadata.json.candidate_adapter_id",
                missing=missing_unexpectedly("Adapter ID missing."),
            ),
            "feature_pipeline": from_optional(
                pipeline,
                source="run_metadata.json.feature_pipeline_id",
                missing=unrecorded("Feature pipeline ID not recorded."),
            ),
            "feature_count": _feature_count(identity, provenance),
            "target_dependency": _target_dependency(identity, provenance),
            "parent_dataset_id": parent,
            "primary_metric_name": available(
                "balanced_accuracy", source="metrics/aggregate.json.primary_metric"
            )
            if metrics_root is not None
            else missing_unexpectedly("Primary metric unavailable."),
            "primary_metric_value": ba,
            "balanced_accuracy": ba,
            "sensitivity": nested_number(
                metrics, "sensitivity", "mean", source="metrics/aggregate.json"
            ),
            "specificity": nested_number(
                metrics, "specificity", "mean", source="metrics/aggregate.json"
            ),
            "roc_auc": nested_number(
                metrics, "roc_auc", "mean", source="metrics/aggregate.json"
            ),
            "average_precision": nested_number(
                metrics, "average_precision", "mean", source="metrics/aggregate.json"
            ),
            "brier_score": nested_number(
                metrics, "brier_score", "mean", source="metrics/aggregate.json"
            ),
            "threshold": mapping_get(
                thresholds,
                "median",
                source="thresholds/threshold_summary.json",
                missing=missing_unexpectedly("Threshold median missing.")
                if artifact.state == "completed"
                else unrecorded("Threshold summary unavailable."),
            ),
            "cv_mean": ba,
            "cv_std": ba_std,
            "cv_min": ba_min,
            "cv_max": ba_max,
            "fold_count": _fold_count(metrics_root, meta),
            "repeat_count": mapping_get(
                metrics_root if isinstance(metrics_root, dict) else None,
                "repeat_count",
                source="metrics/aggregate.json",
                missing=unrecorded("Repeat count not recorded."),
            ),
            "seed": unrecorded("Research v2 stores repeat seeds, not a single seed."),
            "duration_seconds": mapping_get(
                meta,
                "evaluation_duration_seconds",
                source="run_metadata.json",
                missing=unrecorded("Duration not recorded."),
            ),
            "best_model": na("Best-model leaderboard is AutoGluon-specific."),
            "profile": na("Profile is AutoGluon-specific."),
            "mlflow_status": unrecorded("Resolved by Results MLflow projection."),
        }

        label_parts = [
            str(dataset.value) if dataset.is_available() else None,
            str(model_family.value) if model_family.is_available() else None,
            str(mode.value) if mode.is_available() else None,
            run_id,
        ]
        display = " · ".join(part for part in label_parts if part)

        return ResultEntity(
            kind=self.entity_kind,
            entity_id=run_id,
            display_label=display,
            reader_id=artifact.reader_id,
            source_path=artifact.relative_path,
            schema_version=schema,
            state=state_field(artifact),
            created_at=created,
            dataset_id=dataset,
            exploratory=_exploratory(provenance, identity),
            fields=fields,
            lineage=(),
            diagnostics=tuple(diagnostics),
            archived=archived,
        )


def _text(payload: dict[str, Any] | None, key: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    return str(value) if isinstance(value, str) and value else None


def _created(meta: dict[str, Any] | None, run_id: str) -> ResultField:
    for key in ("started_at_utc", "finished_at_utc"):
        value = _text(meta, key)
        if value:
            return available(value, source=f"run_metadata.json.{key}")
    stamp = parse_run_id_timestamp(run_id)
    if stamp:
        return available(stamp, source="run_id_stamp")
    return unrecorded("Created timestamp not recorded for this run ID form.")


def _dataset(identity: Any, provenance: dict[str, Any] | None, meta: dict[str, Any] | None) -> ResultField:
    if identity.dataset_id:
        return available(identity.dataset_id, source="dataset_identity")
    if provenance and provenance.get("dataset_id"):
        return available(str(provenance["dataset_id"]), source="dataset_provenance.json")
    plan = _text(meta, "plan_id")
    if plan:
        return available(plan, source="run_metadata.json.plan_id")
    return missing_unexpectedly("Dataset identity is missing.")


def _parent_dataset(
    identity: Any, provenance: dict[str, Any] | None, dataset: ResultField
) -> ResultField:
    parent = identity.parent_dataset_id
    if parent is None and provenance is not None:
        parent = provenance.get("parent_dataset_id")
    if parent in {None, "", "None"}:
        if dataset.is_available() and str(dataset.value).startswith("v0"):
            return na("Root dataset has no parent.")
        if provenance is not None and "parent_dataset_id" in provenance:
            return na("Parent dataset is explicitly null.")
        return unrecorded("Parent dataset was not recorded.")
    return available(str(parent), source="dataset_provenance.json")


def _feature_count(identity: Any, provenance: dict[str, Any] | None) -> ResultField:
    if identity.n_features is not None:
        return available(identity.n_features, source="dataset_identity")
    if provenance and provenance.get("n_features") is not None:
        return available(provenance["n_features"], source="dataset_provenance.json")
    return unrecorded("Feature count was not recorded.")


def _target_dependency(identity: Any, provenance: dict[str, Any] | None) -> ResultField:
    value = identity.target_dependency
    if value is None and provenance is not None:
        value = provenance.get("target_dependency")
    if value is None:
        return unrecorded("Target dependency was not recorded.")
    return available(str(value), source="dataset_provenance.json")


def _model_family(meta: dict[str, Any] | None, relative_path: str) -> ResultField:
    adapter = _text(meta, "candidate_adapter_id")
    if adapter:
        return available(normalize_model_family(adapter), source="candidate_adapter_id")
    parts = Path(relative_path).parts
    if len(parts) >= 2 and "__" in parts[-2]:
        token = parts[-2].split("__", 1)[1]
        return available(normalize_model_family(token), source="path.adapter_token")
    return missing_unexpectedly("Model family cannot be determined.")


def _mode(meta: dict[str, Any] | None, relative_path: str) -> ResultField:
    experiment = _text(meta, "experiment_id") or ""
    for token in ("smoke", "development", "deployment"):
        if token in experiment.lower() or token in relative_path.lower():
            return available(normalize_mode(token), source="experiment_id_or_path")
    return unrecorded("Evaluation mode was not recorded.")


def _exploratory(provenance: dict[str, Any] | None, identity: Any) -> ResultField:
    del identity
    if provenance and isinstance(provenance.get("exploratory"), bool):
        return available(bool(provenance["exploratory"]), source="dataset_provenance.json")
    # Research packages encode exploratory via dataset contracts; default false when identity complete.
    return available(False, source="default_non_exploratory")


def _fold_count(
    metrics_root: dict[str, Any] | None, meta: dict[str, Any] | None
) -> ResultField:
    del meta
    if isinstance(metrics_root, dict):
        unit = metrics_root.get("aggregation_unit")
        if unit is not None:
            return unrecorded(
                "Fold count is derived from evaluation plan assignments, not aggregate.json."
            )
    return unrecorded("Fold count was not recorded in aggregate metrics.")
