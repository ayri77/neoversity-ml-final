"""Table projections and inventory helpers for schema-aware Results views."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from src.churn_ml.control_panel.artifacts import ArtifactRecord, discover_artifacts
from src.churn_ml.control_panel.results_adapters import (
    ENTITY_VIEW_READERS,
    adapt_artifact,
)
from src.churn_ml.control_panel.results_entities import ResultEntity, ResultEntityKind
from src.churn_ml.control_panel.results_fields import ResultField, render_field
from src.churn_ml.control_panel.results_mlflow import (
    MLflowEntityStatus,
    MLflowProjection,
    annotate_diagnostics_for_paths,
    resolve_model_run_mlflow_status,
)
from src.churn_ml.control_panel.schemas import ReaderSpec


def inventory_entities(
    *,
    repository_root: Path,
    readers: Mapping[str, ReaderSpec],
    reader_ids: Iterable[str],
    artifacts_by_reader: Mapping[str, list[ArtifactRecord]],
    archived_keys: frozenset[tuple[str, str]],
    show_archived: bool,
) -> list[ResultEntity]:
    """Adapt discovered artifacts for one Results entity view."""
    del readers
    entities: list[ResultEntity] = []
    for reader_id in reader_ids:
        for artifact in artifacts_by_reader.get(reader_id, []):
            archived = (reader_id, artifact.relative_path) in archived_keys
            if archived and not show_archived:
                continue
            entity = adapt_artifact(
                artifact,
                repository_root=repository_root,
                archived=archived,
            )
            if entity is not None:
                entities.append(entity)
    return entities


def discover_for_view(
    *,
    repository_root: Path,
    readers: Mapping[str, ReaderSpec],
    view_name: str,
) -> dict[str, list[ArtifactRecord]]:
    reader_ids = ENTITY_VIEW_READERS.get(view_name, ())
    out: dict[str, list[ArtifactRecord]] = {}
    for reader_id in reader_ids:
        reader = readers.get(reader_id)
        if reader is None:
            continue
        out[reader_id] = discover_artifacts(repository_root, reader)
    return out


def attach_mlflow_statuses(
    entities: list[ResultEntity],
    *,
    projection: MLflowProjection,
) -> list[tuple[ResultEntity, MLflowEntityStatus | None]]:
    rows: list[tuple[ResultEntity, MLflowEntityStatus | None]] = []
    for entity in entities:
        if entity.kind is not ResultEntityKind.MODEL_RUN:
            rows.append((entity, None))
            continue
        run_type = entity.fields.get("run_type")
        source_type = (
            str(run_type.value)
            if run_type is not None and run_type.is_available()
            else ""
        )
        status = resolve_model_run_mlflow_status(
            source_type=source_type,
            source_path=entity.source_path,
            projection=projection,
        )
        rows.append((entity, status))
    return rows


def model_runs_diagnostics(
    entities: list[ResultEntity],
    *,
    projection: MLflowProjection,
) -> Any:
    eligible = {}
    for entity in entities:
        run_type = entity.fields.get("run_type")
        if run_type is None or not run_type.is_available():
            continue
        eligible[entity.source_path] = str(run_type.value)
    return annotate_diagnostics_for_paths(projection, eligible_paths=eligible)


def archive_counts(
    *,
    total: int,
    shown: int,
    archived_hidden: int,
) -> str:
    return f"{shown} shown · {archived_hidden} archived hidden · {total} total"


def table_rows_for_view(
    view_name: str,
    entities: list[ResultEntity],
    *,
    mlflow_by_path: Mapping[str, MLflowEntityStatus] | None = None,
) -> list[dict[str, Any]]:
    if view_name == "Model runs":
        return [_model_run_row(entity, mlflow_by_path) for entity in entities]
    if view_name == "Comparisons":
        return [_comparison_row(entity) for entity in entities]
    if view_name == "Prediction candidates":
        return [_candidate_row(entity) for entity in entities]
    if view_name == "Blends":
        return [_blend_row(entity) for entity in entities]
    if view_name == "Submissions":
        return [_submission_row(entity) for entity in entities]
    return []


def detail_rows(entity: ResultEntity) -> list[dict[str, str]]:
    rows = [
        {"field": "entity_kind", "value": entity.kind.value, "status": "available"},
        {"field": "entity_id", "value": entity.entity_id, "status": "available"},
        {"field": "display_label", "value": entity.display_label, "status": "available"},
        {"field": "source_path", "value": entity.source_path, "status": "available"},
        _detail("schema_version", entity.schema_version),
        _detail("state", entity.state),
        _detail("created_at", entity.created_at),
        _detail("dataset_id", entity.dataset_id),
        _detail("exploratory", entity.exploratory),
    ]
    for key, field in sorted(entity.fields.items()):
        rows.append(_detail(key, field))
    return rows


def lineage_rows(entity: ResultEntity) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for link in entity.lineage:
        rows.append(
            {
                "relation": link.relation,
                "target_kind": link.target_kind.value if link.target_kind else "",
                "target_id": link.target_id or "",
                "target_path": link.target_path or "",
                "resolved": "true" if link.resolved else "false",
                "message": link.message or "",
            }
        )
    return rows


def _model_run_row(
    entity: ResultEntity,
    mlflow_by_path: Mapping[str, MLflowEntityStatus] | None,
) -> dict[str, Any]:
    mlflow = (mlflow_by_path or {}).get(entity.source_path)
    return {
        "Run": entity.display_label,
        "Type": _cell(entity.fields.get("run_type")),
        "Dataset": _cell(entity.dataset_id),
        "Model": _cell(entity.fields.get("model_family")),
        "Framework": _cell(entity.fields.get("framework")),
        "Mode / Profile": _mode_or_profile(entity),
        "Primary metric": _cell(entity.fields.get("primary_metric_name")),
        "Metric value": _cell(entity.fields.get("primary_metric_value")),
        "BA": _cell(entity.fields.get("balanced_accuracy")),
        "CV std": _cell(entity.fields.get("cv_std")),
        "State": _cell(entity.state),
        "Created": _cell(entity.created_at),
        "Exploratory": _cell(entity.exploratory),
        "MLflow": _mlflow_cell(mlflow),
        "_source_path": entity.source_path,
        "_entity_id": entity.entity_id,
        "_reader_id": entity.reader_id,
        "_created_sort": entity.created_at.value
        if entity.created_at.is_available()
        else "",
        "_archived": entity.archived,
    }


def _comparison_row(entity: ResultEntity) -> dict[str, Any]:
    dataset = entity.fields.get("dataset_pair") or entity.dataset_id
    return {
        "Comparison": entity.display_label,
        "Type": _cell(entity.fields.get("comparison_type")),
        "Baseline": _cell(entity.fields.get("baseline")),
        "Candidate": _cell(entity.fields.get("candidate")),
        "Dataset / Dataset pair": _cell(dataset),
        "Metric": _cell(entity.fields.get("primary_metric_name")),
        "Baseline value": _cell(entity.fields.get("baseline_value")),
        "Candidate value": _cell(entity.fields.get("candidate_value")),
        "Delta": _cell(entity.fields.get("delta")),
        "Decision": _cell(entity.fields.get("decision")),
        "Compatibility": _cell(entity.fields.get("compatibility")),
        "State": _cell(entity.state),
        "_source_path": entity.source_path,
        "_entity_id": entity.entity_id,
        "_reader_id": entity.reader_id,
    }


def _candidate_row(entity: ResultEntity) -> dict[str, Any]:
    return {
        "Candidate": entity.display_label,
        "Source": _cell(entity.fields.get("source_kind")),
        "Dataset": _cell(entity.dataset_id),
        "Metric": _cell(entity.fields.get("source_metric_name")),
        "Metric value": _cell(entity.fields.get("source_metric_value")),
        "OOF rows": _cell(entity.fields.get("oof_row_count")),
        "Test rows": _cell(entity.fields.get("test_row_count")),
        "Exploratory": _cell(entity.exploratory),
        "State": _cell(entity.state),
        "Created": _cell(entity.created_at),
        "_source_path": entity.source_path,
        "_entity_id": entity.entity_id,
        "_reader_id": entity.reader_id,
        "_created_sort": entity.created_at.value
        if entity.created_at.is_available()
        else "",
    }


def _blend_row(entity: ResultEntity) -> dict[str, Any]:
    weights = entity.fields.get("nonzero_weights")
    weight_text = "N/A"
    if weights is not None and weights.is_available():
        pairs = weights.value
        if isinstance(pairs, tuple):
            weight_text = ", ".join(f"{key}:{value:g}" for key, value in pairs)
        else:
            weight_text = render_field(weights)
    return {
        "Blend": entity.display_label,
        "Optimizer": _cell(entity.fields.get("optimizer")),
        "Parents": _cell(entity.fields.get("parent_count")),
        "Honest BA": _cell(entity.fields.get("honest_mean_ba")),
        "Std": _cell(entity.fields.get("honest_ba_std")),
        "Minimum": _cell(entity.fields.get("honest_ba_min")),
        "Threshold": _cell(entity.fields.get("threshold")),
        "Non-zero weights": weight_text,
        "Readiness": _cell(entity.fields.get("submission_readiness")),
        "Exploratory": _cell(entity.exploratory),
        "State": _cell(entity.state),
        "Created": _cell(entity.created_at),
        "_source_path": entity.source_path,
        "_entity_id": entity.entity_id,
        "_reader_id": entity.reader_id,
        "_created_sort": entity.created_at.value
        if entity.created_at.is_available()
        else "",
    }


def _submission_row(entity: ResultEntity) -> dict[str, Any]:
    return {
        "Submission": entity.display_label,
        "Candidate": _cell(entity.fields.get("candidate_id")),
        "Blend": _cell(entity.fields.get("blend_id")),
        "Rows": _cell(entity.fields.get("row_count")),
        "Threshold": _cell(entity.fields.get("threshold")),
        "Ready": _cell(entity.fields.get("readiness")),
        "Network access": _cell(entity.fields.get("network_access")),
        "Kaggle upload": _cell(entity.fields.get("kaggle_upload")),
        "Public score": _cell(entity.fields.get("kaggle_public_score")),
        "State": _cell(entity.state),
        "Created": _cell(entity.created_at),
        "_source_path": entity.source_path,
        "_entity_id": entity.entity_id,
        "_reader_id": entity.reader_id,
        "_created_sort": entity.created_at.value
        if entity.created_at.is_available()
        else "",
    }


def _mode_or_profile(entity: ResultEntity) -> str:
    mode = entity.fields.get("mode")
    profile = entity.fields.get("profile")
    if mode is not None and mode.is_available():
        return render_field(mode)
    if profile is not None and profile.is_available():
        return render_field(profile)
    if mode is not None:
        return render_field(mode)
    if profile is not None:
        return render_field(profile)
    return "N/A"


def _cell(field: ResultField | None) -> str:
    if field is None:
        return "N/A"
    return render_field(field)


def _mlflow_cell(status: MLflowEntityStatus | None) -> str:
    if status is None:
        return "N/A"
    return status.status.value


def _detail(name: str, field: ResultField) -> dict[str, str]:
    return {
        "field": name,
        "value": render_field(field),
        "status": field.status.value,
        "source": field.source or "",
        "message": field.message or "",
    }


def sort_entities_newest_first(entities: list[ResultEntity]) -> list[ResultEntity]:
    return sorted(
        entities,
        key=lambda item: (
            str(item.created_at.value) if item.created_at.is_available() else "",
            item.source_path,
        ),
        reverse=True,
    )
