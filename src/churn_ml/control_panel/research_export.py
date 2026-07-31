"""CSV export helpers for the Research Workspace.

Exports are derived from the currently visible filtered inventory or matrix
selection and never mutate persisted Control Panel state.
"""

from __future__ import annotations

import csv
import io
from typing import Any, Iterable, Sequence

from src.churn_ml.control_panel.research_inventory import UNAVAILABLE
from src.churn_ml.control_panel.research_matrix import AnnotatedResearchRun


EXPORT_COLUMNS = (
    "relative_path",
    "run_id",
    "dataset_id",
    "parent_dataset_id",
    "model_family",
    "adapter_id",
    "model_config_id",
    "feature_pipeline_id",
    "evaluation_plan_id",
    "evaluation_plan_hash",
    "evaluation_mode",
    "comparability",
    "comparability_reasons",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "roc_auc",
    "average_precision",
    "brier_score",
    "threshold_median",
    "tags",
    "shortlisted",
    "note",
    "artifact_path",
    "status",
    "created_at_utc",
)


def export_annotated_runs_csv(runs: Sequence[AnnotatedResearchRun]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=EXPORT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for run in runs:
        writer.writerow(_row_dict(run))
    return buffer.getvalue()


def export_annotated_runs_records(
    runs: Sequence[AnnotatedResearchRun],
) -> list[dict[str, Any]]:
    return [_row_dict(run) for run in runs]


def _row_dict(run: AnnotatedResearchRun) -> dict[str, Any]:
    row = run.row
    return {
        "relative_path": row.relative_path,
        "run_id": _value(row.run_id),
        "dataset_id": _value(row.dataset_id),
        "parent_dataset_id": _value(row.parent_dataset_id),
        "model_family": _value(row.model_family),
        "adapter_id": _value(row.adapter_id),
        "model_config_id": _value(row.model_config_id),
        "feature_pipeline_id": _value(row.feature_pipeline_id),
        "evaluation_plan_id": _value(row.evaluation_plan_id),
        "evaluation_plan_hash": _value(row.evaluation_plan_hash),
        "evaluation_mode": _value(row.evaluation_mode),
        "comparability": run.comparability.primary,
        "comparability_reasons": " | ".join(run.comparability.reasons),
        "balanced_accuracy": _value(row.balanced_accuracy),
        "sensitivity": _value(row.sensitivity),
        "specificity": _value(row.specificity),
        "roc_auc": _value(row.roc_auc),
        "average_precision": _value(row.average_precision),
        "brier_score": _value(row.brier_score),
        "threshold_median": _value(row.threshold_median),
        "tags": ",".join(run.tags),
        "shortlisted": "true" if run.shortlisted else "false",
        "note": run.note,
        "artifact_path": row.relative_path,
        "status": row.status,
        "created_at_utc": _value(row.created_at_utc),
    }


def _value(value: Any) -> Any:
    if value is None:
        return UNAVAILABLE
    return value


def visible_runs_from_paths(
    runs: Iterable[AnnotatedResearchRun],
    relative_paths: set[str],
) -> list[AnnotatedResearchRun]:
    wanted = set(relative_paths)
    return [run for run in runs if run.row.relative_path in wanted]
