"""Research Workspace Results Matrix v1 helpers.

Rows are Dataset Packages, columns are model families, and cells contain an
explicitly selected Research v2 run. Duplicate cells never silently promote the
maximum Balanced Accuracy as canonical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from src.churn_ml.control_panel.research_comparability import (
    COMPARABLE_DEVELOPMENT,
    EXPLORATORY_DATASET,
    INCOMPLETE_IDENTITY,
    INVALID,
    LEGACY_DESCRIPTIVE,
    SMOKE,
    ComparabilityAssessment,
    annotate_inventory_comparability,
)
from src.churn_ml.control_panel.research_inventory import ResearchRunInventoryRow


DEFAULT_MODEL_ORDER = ("LightGBM", "XGBoost", "CatBoost")
DEFAULT_BASELINE_DATASET_ID = "v0_raw_minimal"
MATRIX_METRICS = (
    ("balanced_accuracy", "Balanced Accuracy"),
    ("sensitivity", "Sensitivity"),
    ("specificity", "Specificity"),
    ("roc_auc", "ROC AUC"),
    ("average_precision", "Average Precision"),
    ("brier_score", "Brier score"),
)

SELECTION_POLICY = (
    "Prefer Comparable development, then newest created_at_utc, "
    "then newest run_id, then lexicographic relative_path. "
    "Never select by maximum BA."
)


@dataclass(frozen=True)
class AnnotatedResearchRun:
    row: ResearchRunInventoryRow
    comparability: ComparabilityAssessment
    tags: tuple[str, ...] = ()
    note: str = ""
    shortlisted: bool = False
    archived: bool = False


@dataclass(frozen=True)
class MatrixFilters:
    development_only: bool = True
    include_exploratory: bool = False
    model_families: tuple[str, ...] = ()
    dataset_ids: tuple[str, ...] = ()
    adapter_or_config: str = ""
    statuses: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    shortlist_only: bool = False
    show_archived: bool = False


@dataclass(frozen=True)
class MatrixCell:
    dataset_id: str
    model_family: str
    selected: AnnotatedResearchRun | None
    candidates: tuple[AnnotatedResearchRun, ...]
    duplicate_count: int
    selection_policy: str
    delta_ba_vs_baseline: float | None = None

    @property
    def has_duplicates(self) -> bool:
        return self.duplicate_count > 1


@dataclass(frozen=True)
class ResearchMatrix:
    model_families: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    cells: dict[tuple[str, str], MatrixCell]
    baseline_dataset_id: str
    filtered_runs: tuple[AnnotatedResearchRun, ...]
    parent_by_dataset: dict[str, str | None] = field(default_factory=dict)


def annotate_runs(
    rows: Sequence[ResearchRunInventoryRow],
    *,
    annotations_by_path: Mapping[str, Mapping[str, Any]] | None = None,
    archived_paths: Iterable[str] | None = None,
) -> list[AnnotatedResearchRun]:
    archived = set(archived_paths or ())
    annotations = annotations_by_path or {}
    annotated: list[AnnotatedResearchRun] = []
    for row, assessment in annotate_inventory_comparability(list(rows)):
        payload = annotations.get(row.relative_path, {})
        tags = tuple(str(tag) for tag in payload.get("tags", []) or [])
        note = str(payload.get("note") or "")
        shortlisted = bool(payload.get("shortlisted", False))
        annotated.append(
            AnnotatedResearchRun(
                row=row,
                comparability=assessment,
                tags=tags,
                note=note,
                shortlisted=shortlisted,
                archived=row.relative_path in archived,
            )
        )
    return annotated


def filter_annotated_runs(
    runs: Sequence[AnnotatedResearchRun],
    filters: MatrixFilters,
) -> list[AnnotatedResearchRun]:
    comparable_keys = {
        (run.row.dataset_id, run.row.model_family)
        for run in runs
        if run.comparability.primary == COMPARABLE_DEVELOPMENT
        and run.row.dataset_id
        and run.row.model_family
    }
    selected: list[AnnotatedResearchRun] = []
    for run in runs:
        if not _passes_filters(run, filters, comparable_keys):
            continue
        selected.append(run)
    selected.sort(key=_run_sort_key)
    return selected


def discover_model_families(runs: Sequence[AnnotatedResearchRun]) -> tuple[str, ...]:
    present = {
        run.row.model_family
        for run in runs
        if run.row.model_family
    }
    ordered = [family for family in DEFAULT_MODEL_ORDER if family in present]
    extras = sorted(family for family in present if family not in DEFAULT_MODEL_ORDER)
    return tuple(ordered + extras)


def discover_dataset_ids(runs: Sequence[AnnotatedResearchRun]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                run.row.dataset_id
                for run in runs
                if run.row.dataset_id
            }
        )
    )


def select_cell_run(
    candidates: Sequence[AnnotatedResearchRun],
    *,
    explicit_relative_path: str | None = None,
) -> tuple[AnnotatedResearchRun | None, str]:
    """Deterministic cell selection with optional explicit override."""
    if not candidates:
        return None, SELECTION_POLICY
    if explicit_relative_path:
        for candidate in candidates:
            if candidate.row.relative_path == explicit_relative_path:
                return candidate, f"Explicit selection: {explicit_relative_path}"
    ordered = sorted(candidates, key=_cell_preference_key)
    return ordered[0], SELECTION_POLICY


def build_research_matrix(
    runs: Sequence[AnnotatedResearchRun],
    *,
    filters: MatrixFilters | None = None,
    baseline_dataset_id: str = DEFAULT_BASELINE_DATASET_ID,
    explicit_selections: Mapping[tuple[str, str], str] | None = None,
) -> ResearchMatrix:
    active_filters = filters or MatrixFilters()
    filtered = filter_annotated_runs(runs, active_filters)
    model_families = discover_model_families(filtered)
    if not model_families:
        model_families = discover_model_families(runs) or DEFAULT_MODEL_ORDER
    dataset_ids = discover_dataset_ids(filtered)
    parent_by_dataset = {
        run.row.dataset_id: run.row.parent_dataset_id
        for run in filtered
        if run.row.dataset_id
    }
    grouped: dict[tuple[str, str], list[AnnotatedResearchRun]] = {}
    for run in filtered:
        dataset_id = run.row.dataset_id
        model_family = run.row.model_family
        if not dataset_id or not model_family:
            continue
        grouped.setdefault((dataset_id, model_family), []).append(run)

    baseline_by_family: dict[str, AnnotatedResearchRun] = {}
    for model_family in model_families:
        candidates = grouped.get((baseline_dataset_id, model_family), [])
        selected, _ = select_cell_run(candidates)
        if selected is not None:
            baseline_by_family[model_family] = selected

    cells: dict[tuple[str, str], MatrixCell] = {}
    explicit = explicit_selections or {}
    for dataset_id in dataset_ids:
        for model_family in model_families:
            key = (dataset_id, model_family)
            candidates = tuple(sorted(grouped.get(key, []), key=_run_sort_key))
            selected, policy = select_cell_run(
                candidates,
                explicit_relative_path=explicit.get(key),
            )
            delta = None
            baseline = baseline_by_family.get(model_family)
            if (
                selected is not None
                and baseline is not None
                and selected.row.balanced_accuracy is not None
                and baseline.row.balanced_accuracy is not None
                and dataset_id != baseline_dataset_id
            ):
                delta = selected.row.balanced_accuracy - baseline.row.balanced_accuracy
            cells[key] = MatrixCell(
                dataset_id=dataset_id,
                model_family=model_family,
                selected=selected,
                candidates=candidates,
                duplicate_count=len(candidates),
                selection_policy=policy,
                delta_ba_vs_baseline=delta,
            )

    return ResearchMatrix(
        model_families=model_families,
        dataset_ids=dataset_ids,
        cells=cells,
        baseline_dataset_id=baseline_dataset_id,
        filtered_runs=tuple(filtered),
        parent_by_dataset=parent_by_dataset,
    )


def matrix_display_rows(
    matrix: ResearchMatrix,
    *,
    primary_metric: str = "balanced_accuracy",
) -> list[dict[str, Any]]:
    """Flatten the matrix into table-friendly rows for Streamlit/CSV."""
    rows: list[dict[str, Any]] = []
    for dataset_id in matrix.dataset_ids:
        row: dict[str, Any] = {
            "Dataset ID": dataset_id,
            "Parent Dataset ID": matrix.parent_by_dataset.get(dataset_id) or "—",
        }
        for model_family in matrix.model_families:
            cell = matrix.cells[(dataset_id, model_family)]
            selected = cell.selected
            if selected is None:
                row[f"{model_family} metric"] = None
                row[f"{model_family} BA"] = None
                row[f"{model_family} sensitivity"] = None
                row[f"{model_family} specificity"] = None
                row[f"{model_family} threshold median"] = None
                row[f"{model_family} badge"] = None
                row[f"{model_family} run"] = None
                row[f"{model_family} duplicates"] = 0
                row[f"{model_family} delta BA vs {matrix.baseline_dataset_id}"] = None
                continue
            metric_value = getattr(selected.row, primary_metric, None)
            row[f"{model_family} metric"] = metric_value
            row[f"{model_family} BA"] = selected.row.balanced_accuracy
            row[f"{model_family} sensitivity"] = selected.row.sensitivity
            row[f"{model_family} specificity"] = selected.row.specificity
            row[f"{model_family} threshold median"] = selected.row.threshold_median
            row[f"{model_family} badge"] = selected.comparability.primary
            row[f"{model_family} run"] = (
                selected.row.created_at_utc or selected.row.run_id
            )
            row[f"{model_family} duplicates"] = cell.duplicate_count
            row[f"{model_family} delta BA vs {matrix.baseline_dataset_id}"] = (
                cell.delta_ba_vs_baseline
            )
            row[f"{model_family} path"] = selected.row.relative_path
        rows.append(row)
    return rows


def descriptive_comparison_table(
    runs: Sequence[AnnotatedResearchRun],
    *,
    reference_relative_path: str | None = None,
) -> list[dict[str, Any]]:
    if not runs:
        return []
    reference = None
    if reference_relative_path:
        for run in runs:
            if run.row.relative_path == reference_relative_path:
                reference = run
                break
    if reference is None:
        reference = runs[0]
    table: list[dict[str, Any]] = []
    for run in runs:
        row = run.row
        delta_ba = None
        if (
            row.balanced_accuracy is not None
            and reference.row.balanced_accuracy is not None
        ):
            delta_ba = row.balanced_accuracy - reference.row.balanced_accuracy
        table.append(
            {
                "relative_path": row.relative_path,
                "dataset_id": row.dataset_id,
                "parent_dataset_id": row.parent_dataset_id,
                "model_family": row.model_family,
                "adapter_id": row.adapter_id,
                "model_config_id": row.model_config_id,
                "evaluation_plan_id": row.evaluation_plan_id,
                "evaluation_plan_hash": row.evaluation_plan_hash,
                "evaluation_mode": row.evaluation_mode,
                "comparability": run.comparability.primary,
                "balanced_accuracy": row.balanced_accuracy,
                "sensitivity": row.sensitivity,
                "specificity": row.specificity,
                "roc_auc": row.roc_auc,
                "average_precision": row.average_precision,
                "brier_score": row.brier_score,
                "threshold_median": row.threshold_median,
                "delta_BA_vs_reference": delta_ba,
                "is_reference": row.relative_path == reference.row.relative_path,
                "note": "Descriptive aggregate comparison only; not Stage E paired inference.",
            }
        )
    return table


def _passes_filters(
    run: AnnotatedResearchRun,
    filters: MatrixFilters,
    comparable_keys: set[tuple[str | None, str | None]],
) -> bool:
    row = run.row
    label = run.comparability.primary

    if not filters.show_archived and run.archived:
        return False
    if filters.shortlist_only and not run.shortlisted:
        return False
    if filters.statuses and row.status not in filters.statuses:
        return False
    if filters.model_families and (row.model_family or "") not in filters.model_families:
        return False
    if filters.dataset_ids and (row.dataset_id or "") not in filters.dataset_ids:
        return False
    if filters.tags and not set(run.tags).intersection(filters.tags):
        return False

    needle = filters.adapter_or_config.strip().lower()
    if needle:
        haystack = " ".join(
            filter(
                None,
                [
                    row.adapter_id,
                    row.model_config_id,
                    row.feature_pipeline_id,
                    row.evaluation_plan_id,
                ],
            )
        ).lower()
        if needle not in haystack:
            return False

    if label == INVALID or row.status in {"failed", "invalid", "running"}:
        return False
    if filters.development_only and label == INCOMPLETE_IDENTITY:
        return False
    if filters.development_only and (label == SMOKE or row.evaluation_mode == "Smoke"):
        return False
    if label == EXPLORATORY_DATASET and not filters.include_exploratory:
        return False
    if label == LEGACY_DESCRIPTIVE and (
        row.dataset_id,
        row.model_family,
    ) in comparable_keys:
        return False
    if filters.development_only and row.evaluation_mode not in {
        None,
        "Development",
    }:
        if label != EXPLORATORY_DATASET:
            return False
    return True


def _run_sort_key(run: AnnotatedResearchRun) -> tuple[Any, ...]:
    return (
        run.row.dataset_id or "",
        run.row.model_family or "",
        run.row.created_at_utc or "",
        run.row.relative_path,
    )


def _cell_preference_key(run: AnnotatedResearchRun) -> tuple[Any, ...]:
    comparable_rank = 0 if run.comparability.primary == COMPARABLE_DEVELOPMENT else 1
    created = run.row.created_at_utc or ""
    run_id = run.row.run_id or ""
    return (
        comparable_rank,
        # Ascending key: empty timestamps last, otherwise newest first.
        1 if not created else 0,
        tuple(-ord(ch) for ch in created),
        # When created_at ties (common in synthetic fixtures), prefer newer run_id.
        1 if not run_id else 0,
        tuple(-ord(ch) for ch in run_id),
        run.row.relative_path,
    )
