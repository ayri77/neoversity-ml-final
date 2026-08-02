"""Streamlit rendering helpers for schema-aware Results entity views."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from src.churn_ml.control_panel.archive_registry import ArchiveRegistry
from src.churn_ml.control_panel.artifacts import ArtifactRecord
from src.churn_ml.control_panel.registry import ControlPanelRegistry
from src.churn_ml.control_panel.results_adapters import ENTITY_VIEW_READERS
from src.churn_ml.control_panel.results_entities import ResultEntity
from src.churn_ml.control_panel.results_mlflow import (
    MLflowEntityStatus,
    MLflowProjection,
)
from src.churn_ml.control_panel.results_tables import (
    archive_counts,
    attach_mlflow_statuses,
    detail_rows,
    inventory_entities,
    lineage_rows,
    model_runs_diagnostics,
    sort_entities_newest_first,
    table_rows_for_view,
)


DiscoverFn = Callable[[str], list[ArtifactRecord]]


def load_view_entities(
    *,
    repository_root: Path,
    loaded: ControlPanelRegistry,
    view_name: str,
    archive: ArchiveRegistry,
    show_archived: bool,
    discover: DiscoverFn,
) -> tuple[list[ResultEntity], int, int, int]:
    """Return entities plus archive counts for one lazy Results view."""
    reader_ids = ENTITY_VIEW_READERS.get(view_name, ())
    artifacts_by_reader: dict[str, list[ArtifactRecord]] = {}
    total = 0
    archived_keys = archive.archived_artifact_keys()
    archived_hidden = 0
    for reader_id in reader_ids:
        artifacts = discover(reader_id)
        artifacts_by_reader[reader_id] = artifacts
        total += len(artifacts)
        for artifact in artifacts:
            if (reader_id, artifact.relative_path) in archived_keys and not show_archived:
                archived_hidden += 1
    entities = inventory_entities(
        repository_root=repository_root,
        readers=loaded.readers,
        reader_ids=reader_ids,
        artifacts_by_reader=artifacts_by_reader,
        archived_keys=archived_keys,
        show_archived=show_archived,
    )
    entities = sort_entities_newest_first(entities)
    return entities, len(entities), archived_hidden, total


def render_archive_caption(shown: int, archived_hidden: int, total: int, st: Any) -> None:
    st.caption(archive_counts(total=total, shown=shown, archived_hidden=archived_hidden))


def render_entity_table(
    st: Any,
    view_name: str,
    entities: list[ResultEntity],
    *,
    mlflow_by_path: Mapping[str, MLflowEntityStatus] | None = None,
) -> list[dict[str, Any]]:
    rows = table_rows_for_view(view_name, entities, mlflow_by_path=mlflow_by_path)
    display = [
        {key: value for key, value in row.items() if not str(key).startswith("_")}
        for row in rows
    ]
    if not display:
        st.info(f"No {view_name.lower()} match the current filters.")
        return rows
    st.dataframe(display, width="stretch", hide_index=True)
    return rows


def render_entity_details(st: Any, entity: ResultEntity) -> None:
    st.subheader("Normalized entity")
    st.markdown(
        "  \n".join(
            [
                f"**Kind:** `{entity.kind.value}`",
                f"**ID:** `{entity.entity_id}`",
                f"**Label:** `{entity.display_label}`",
                f"**Reader:** `{entity.reader_id}`",
                f"**Path:** `{entity.source_path}`",
            ]
        )
    )
    st.dataframe(detail_rows(entity), width="stretch", hide_index=True)
    lineage = lineage_rows(entity)
    if lineage:
        st.subheader("Lineage")
        st.dataframe(lineage, width="stretch", hide_index=True)
    if entity.diagnostics:
        st.subheader("Diagnostics")
        for item in entity.diagnostics:
            st.warning(item)


def render_mlflow_diagnostics(st: Any, projection: MLflowProjection, entities: list[ResultEntity]) -> None:
    summary = model_runs_diagnostics(entities, projection=projection)
    st.subheader("MLflow index diagnostics")
    st.write(
        {
            "store_available": summary.store_available,
            "indexed_local_runs": summary.indexed_local_runs,
            "unindexed_eligible_runs": summary.unindexed_eligible_runs,
            "failed_index_records": summary.failed_index_records,
            "stale_failure_records": summary.stale_failure_records,
            "orphaned_mlflow_runs": summary.orphaned_mlflow_runs,
            "invalid_records": summary.invalid_records,
            "duplicate_source_identities": summary.duplicate_source_identities,
        }
    )
    if projection.orphans:
        with st.expander(f"Orphaned MLflow runs ({len(projection.orphans)})", expanded=False):
            st.dataframe(
                [
                    {
                        "run_uuid": item.run_uuid,
                        "source_path": item.source_relative_path,
                        "source_type": item.source_type,
                        "status": "orphaned_source",
                        "message": "Local source artifact not found",
                    }
                    for item in projection.orphans
                ],
                width="stretch",
                hide_index=True,
            )


def filter_model_runs(
    entities: list[ResultEntity],
    *,
    run_type: str,
    dataset: str,
    model_family: str,
    framework: str,
    state: str,
    exploratory: str,
    mlflow_status: str,
    mlflow_by_path: Mapping[str, MLflowEntityStatus],
) -> list[ResultEntity]:
    filtered: list[ResultEntity] = []
    for entity in entities:
        if run_type != "All":
            value = entity.fields.get("run_type")
            if value is None or not value.is_available() or str(value.value) != run_type:
                continue
        if dataset != "All" and (
            not entity.dataset_id.is_available()
            or str(entity.dataset_id.value) != dataset
        ):
            continue
        if model_family != "All":
            value = entity.fields.get("model_family")
            if (
                value is None
                or not value.is_available()
                or str(value.value) != model_family
            ):
                continue
        if framework != "All":
            value = entity.fields.get("framework")
            if (
                value is None
                or not value.is_available()
                or str(value.value) != framework
            ):
                continue
        if state != "All" and (
            not entity.state.is_available() or str(entity.state.value) != state
        ):
            continue
        if exploratory != "All":
            wanted = exploratory == "true"
            if (
                not entity.exploratory.is_available()
                or bool(entity.exploratory.value) != wanted
            ):
                continue
        if mlflow_status != "All":
            status = mlflow_by_path.get(entity.source_path)
            if status is None or status.status.value != mlflow_status:
                continue
        filtered.append(entity)
    return filtered


def mlflow_status_map(
    entities: list[ResultEntity],
    projection: MLflowProjection,
) -> dict[str, MLflowEntityStatus]:
    mapping: dict[str, MLflowEntityStatus] = {}
    for entity, status in attach_mlflow_statuses(entities, projection=projection):
        if status is not None:
            mapping[entity.source_path] = status
    return mapping


def entity_options(entities: list[ResultEntity]) -> list[tuple[str, str]]:
    return [(entity.source_path, entity.display_label) for entity in entities]


def select_entity(
    entities: list[ResultEntity], source_path: str | None
) -> ResultEntity | None:
    if not source_path:
        return None
    for entity in entities:
        if entity.source_path == source_path:
            return entity
    return None


def dataframe_preview(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)
