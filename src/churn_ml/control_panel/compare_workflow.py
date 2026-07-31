"""Compare workflow helpers for the Control Panel Run tab.

Supports three comparison scopes:
- same dataset / different candidates (official Paired Comparison v1)
- same model / different datasets (Dataset Comparison v1)
- descriptive comparison (Research Workspace only)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.churn_ml.control_panel.artifacts import sanitize_comparison_id
from src.churn_ml.control_panel.deployment_candidates import candidate_label
from src.churn_ml.control_panel.path_safety import (
    PathSafetyError,
    require_regular_file,
    require_safe_directory,
)
from src.churn_ml.control_panel.research_inventory import ResearchRunInventoryRow
from src.churn_ml.dataset_comparison_v1 import (
    DEFAULT_OUTPUT_ROOT as DATASET_COMPARISON_DEFAULT_OUTPUT_ROOT,
    DatasetComparisonCompatibilitySummary,
    build_compatibility_summary,
    default_dataset_comparison_id,
    derive_model_family,
    is_exploratory_dataset,
    load_completed_research_v2_run,
)

COMPARE_SCOPE_SAME_DATASET = "same_dataset"
COMPARE_SCOPE_DIFFERENT_DATASETS = "different_datasets"
COMPARE_SCOPE_DESCRIPTIVE = "descriptive"

COMPARE_SCOPE_LABELS: tuple[tuple[str, str], ...] = (
    (COMPARE_SCOPE_SAME_DATASET, "Same dataset / different candidates"),
    (COMPARE_SCOPE_DIFFERENT_DATASETS, "Same model / different datasets"),
    (COMPARE_SCOPE_DESCRIPTIVE, "Descriptive comparison"),
)

SAME_DATASET_OUTPUT_ROOT = "artifacts/research_v2_comparisons"
DATASET_COMPARISON_OUTPUT_ROOT = DATASET_COMPARISON_DEFAULT_OUTPUT_ROOT.as_posix()

_COMPARISON_ID_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_MAX_SIDECAR_BYTES = 4_000_000


@dataclass(frozen=True)
class DatasetComparisonReadiness:
    ready: bool
    compatible: bool
    summary: DatasetComparisonCompatibilitySummary | None
    diagnostic: str | None

    def to_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ready": self.ready,
            "compatible": self.compatible,
            "diagnostic": self.diagnostic,
        }
        if self.summary is not None:
            payload["parent_child_relation"] = self.summary.parent_child_relation
            payload["exploratory"] = self.summary.exploratory
            payload["expected_differences"] = list(self.summary.expected_differences)
            payload["issue_count"] = len(self.summary.issues)
        return payload


def comparison_scope_label(scope: str) -> str:
    for key, label in COMPARE_SCOPE_LABELS:
        if key == scope:
            return label
    return scope


def is_descriptive_comparison_scope(scope: str) -> bool:
    return scope == COMPARE_SCOPE_DESCRIPTIVE


def exploratory_comparison_warning(
    *,
    baseline_dataset_id: str | None,
    candidate_dataset_id: str | None,
    baseline_target_dependency: str | None,
    candidate_target_dependency: str | None,
    summary: DatasetComparisonCompatibilitySummary | None = None,
) -> str | None:
    exploratory = False
    if summary is not None:
        exploratory = summary.exploratory
    elif baseline_dataset_id is not None and candidate_dataset_id is not None:
        exploratory = is_exploratory_dataset(
            baseline_dataset_id, baseline_target_dependency
        ) or is_exploratory_dataset(
            candidate_dataset_id, candidate_target_dependency
        )
    if not exploratory:
        return None
    return (
        "This pair includes an exploratory Dataset Package (for example "
        "v3_targeted_missingness). Results are not suitable for unbiased "
        "ranking or deployment selection."
    )


def list_completed_development_runs(
    rows: list[ResearchRunInventoryRow],
    *,
    archived_paths: frozenset[str] = frozenset(),
) -> list[ResearchRunInventoryRow]:
    """Inventory rows eligible as Compare workflow baseline selectors."""
    eligible: list[ResearchRunInventoryRow] = []
    for row in rows:
        if row.status != "completed" or not row.identity_complete:
            continue
        if row.relative_path in archived_paths:
            continue
        if row.evaluation_mode not in (None, "Development"):
            continue
        if row.dataset_id is None or row.run_id is None:
            continue
        eligible.append(row)
    eligible.sort(
        key=lambda item: (
            item.dataset_id or "",
            item.model_family or "",
            item.created_at_utc or "",
            item.run_id or "",
        )
    )
    return eligible


def filter_same_dataset_candidates(
    baseline: ResearchRunInventoryRow,
    rows: list[ResearchRunInventoryRow],
) -> list[ResearchRunInventoryRow]:
    """Other completed Development runs on the same dataset as baseline."""
    candidates: list[ResearchRunInventoryRow] = []
    for row in rows:
        if row.relative_path == baseline.relative_path:
            continue
        if row.status != "completed" or not row.identity_complete:
            continue
        if row.evaluation_mode not in (None, "Development"):
            continue
        if row.dataset_id != baseline.dataset_id:
            continue
        candidates.append(row)
    candidates.sort(key=lambda item: candidate_label(item))
    return candidates


def filter_dataset_comparison_candidates(
    baseline: ResearchRunInventoryRow,
    rows: list[ResearchRunInventoryRow],
) -> list[ResearchRunInventoryRow]:
    """Inventory prefilter for cross-dataset comparison (deep validate on selection)."""
    candidates: list[ResearchRunInventoryRow] = []
    for row in rows:
        if row.relative_path == baseline.relative_path:
            continue
        if not _inventory_dataset_comparison_compatible(baseline, row):
            continue
        candidates.append(row)
    candidates.sort(key=lambda item: (item.dataset_id or "", candidate_label(item)))
    return candidates


def _inventory_dataset_comparison_compatible(
    baseline: ResearchRunInventoryRow,
    candidate: ResearchRunInventoryRow,
) -> bool:
    """Inventory prefilter for cross-dataset pairs.

    Full plan hashes, candidate-identity hashes, and feature-bound train-row
    hashes legitimately differ across Dataset Packages. Those are not used here.
    Official Dataset Comparison v1 readiness still runs on the selected pair.
    """
    if candidate.status != "completed" or not candidate.identity_complete:
        return False
    if candidate.evaluation_mode not in (None, "Development"):
        return False
    if baseline.dataset_id is None or candidate.dataset_id is None:
        return False
    if baseline.dataset_id == candidate.dataset_id:
        return False
    if baseline.model_family != candidate.model_family:
        return False
    if baseline.adapter_id != candidate.adapter_id:
        return False
    if baseline.feature_pipeline_id != candidate.feature_pipeline_id:
        return False
    if baseline.repeat_seeds != candidate.repeat_seeds:
        return False
    if baseline.outer_fold_count != candidate.outer_fold_count:
        return False
    if baseline.threshold_selection_protocol != candidate.threshold_selection_protocol:
        return False
    if baseline.target_hash is None or candidate.target_hash is None:
        return False
    if baseline.target_hash != candidate.target_hash:
        return False
    return True


def evaluate_dataset_comparison_readiness(
    *,
    baseline_run_dir: Path,
    candidate_run_dir: Path,
    repository_root: Path,
) -> DatasetComparisonReadiness:
    try:
        baseline = load_completed_research_v2_run(
            baseline_run_dir,
            project_root=repository_root,
            role="baseline_run",
        )
        candidate = load_completed_research_v2_run(
            candidate_run_dir,
            project_root=repository_root,
            role="candidate_run",
        )
        summary = build_compatibility_summary(
            baseline,
            candidate,
            project_root=repository_root,
        )
    except Exception as error:  # noqa: BLE001 - fail closed for UI gate
        return DatasetComparisonReadiness(
            ready=False,
            compatible=False,
            summary=None,
            diagnostic=str(error),
        )
    return DatasetComparisonReadiness(
        ready=summary.compatible,
        compatible=summary.compatible,
        summary=summary,
        diagnostic=None if summary.compatible else _compatibility_diagnostic(summary),
    )


def default_same_dataset_comparison_id(
    baseline_run_root: Path | str,
    candidate_run_root: Path | str,
    repo_root: Path,
) -> str:
    """Authoritative same-dataset comparison ID from run metadata."""
    baseline_dir = _resolve_run_dir(baseline_run_root, repo_root)
    candidate_dir = _resolve_run_dir(candidate_run_root, repo_root)
    dataset_id = _run_dataset_id(baseline_dir) or _run_dataset_id(candidate_dir) or "dataset"
    baseline_token = _run_model_identity_token(baseline_dir)
    candidate_token = _run_model_identity_token(candidate_dir)
    if candidate_token.endswith("_tuned") and not baseline_token.endswith("_tuned"):
        baseline_token = f"{_family_token(baseline_dir)}_base"
    elif baseline_token.endswith("_tuned") and not candidate_token.endswith("_tuned"):
        candidate_token = f"{_family_token(candidate_dir)}_base"
    raw = f"{dataset_id}__{baseline_token}__vs__{candidate_token}"
    return _sanitize_comparison_slug(raw)


def default_dataset_comparison_output_root() -> str:
    return DATASET_COMPARISON_OUTPUT_ROOT


def default_same_dataset_output_root() -> str:
    return SAME_DATASET_OUTPUT_ROOT


def default_dataset_comparison_id_for_runs(
    baseline: ResearchRunInventoryRow,
    candidate: ResearchRunInventoryRow,
) -> str:
    model_family = derive_model_family(
        baseline.adapter_id or baseline.model_family or "model"
    )
    return default_dataset_comparison_id(
        model_family,
        baseline.dataset_id or "baseline",
        candidate.dataset_id or "candidate",
    )


def dataset_comparison_artifact_label(
    artifact_summaries: Mapping[str, Any],
    *,
    comparison_name: str,
) -> str:
    model = artifact_summaries.get("Model family") or "Model"
    baseline = artifact_summaries.get("Baseline dataset") or "baseline"
    candidate = artifact_summaries.get("Candidate dataset") or "candidate"
    short_id = comparison_name
    if len(short_id) > 48:
        short_id = f"{short_id[:45]}..."
    return f"{model} · {baseline} → {candidate} · {short_id}"


def _compatibility_diagnostic(summary: DatasetComparisonCompatibilitySummary) -> str:
    if not summary.issues:
        return "Dataset comparison compatibility failed."
    return "; ".join(
        f"{issue.reason_code} at {issue.field_path}" for issue in summary.issues[:5]
    )


def _resolve_run_dir(path: Path | str, repo_root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def _run_dataset_id(run_dir: Path) -> str | None:
    for relative in ("dataset_provenance.json", "dataset_fingerprints.json"):
        payload = _read_json(run_dir / relative)
        if payload is None:
            continue
        for key in ("dataset_id", "dataset_version"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
    resolved = _read_yaml(run_dir / "resolved_config.yaml")
    if resolved is not None:
        dataset = resolved.get("dataset")
        if isinstance(dataset, dict):
            version = dataset.get("version")
            if isinstance(version, str) and version:
                return version
    return None


def _run_model_identity_token(run_dir: Path) -> str:
    metadata = _read_json(run_dir / "run_metadata.json") or {}
    resolved = _read_yaml(run_dir / "resolved_config.yaml") or {}
    adapter_id = metadata.get("candidate_adapter_id")
    if not isinstance(adapter_id, str) or not adapter_id:
        adapter_identity = _read_json(run_dir / "identities" / "candidate_adapter.json")
        if isinstance(adapter_identity, dict):
            canonical = adapter_identity.get("canonical")
            if isinstance(canonical, dict):
                adapter_id = canonical.get("id")
    adapter_id = str(adapter_id or "model")
    family = derive_model_family(adapter_id)
    if isinstance(resolved.get("search_provenance"), dict):
        return f"{family}_tuned"
    default_adapters = {
        "lightgbm": "manual_lightgbm_te_v1_compat",
        "xgboost": "xgboost_numeric_v1",
        "catboost": "catboost_numeric_v1",
    }
    if adapter_id == default_adapters.get(family):
        return family
    adapter_suffix = adapter_id
    for prefix in (f"manual_{family}_", f"{family}_", "manual_"):
        if adapter_suffix.lower().startswith(prefix):
            adapter_suffix = adapter_suffix[len(prefix) :]
            break
    adapter_suffix = re.sub(r"[^A-Za-z0-9]+", "_", adapter_suffix).strip("_")
    if adapter_suffix:
        return f"{family}_{adapter_suffix}"
    return family


def _family_token(run_dir: Path) -> str:
    metadata = _read_json(run_dir / "run_metadata.json") or {}
    adapter_id = str(metadata.get("candidate_adapter_id") or "model")
    return derive_model_family(adapter_id)


def _sanitize_comparison_slug(value: str) -> str:
    lowered = value.strip().lower()
    if "__vs__" in lowered:
        left, candidate_part = lowered.rsplit("__vs__", 1)
        dataset_part, baseline_part = left.rsplit("__", 1)
        text = (
            f"{_slug_token(dataset_part)}__{_slug_token(baseline_part)}"
            f"__vs__{_slug_token(candidate_part)}"
        )
    else:
        text = _slug_token(lowered)
    if not text:
        text = "compare"
    if text[0] in "-_":
        text = f"c{text}"
    text = text[:128]
    if _COMPARISON_ID_SAFE.fullmatch(text) is None:
        return sanitize_comparison_id(value)
    return text


def _slug_token(value: str) -> str:
    text = re.sub(r"[^a-z0-9_]+", "_", value.strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "item"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        require_safe_directory(path.parent)
        require_regular_file(path, reject_hardlinks=True)
    except (PathSafetyError, OSError):
        return None
    if path.stat().st_size > _MAX_SIDECAR_BYTES:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_yaml(path: Path) -> dict[str, Any] | None:
    try:
        require_safe_directory(path.parent)
        require_regular_file(path, reject_hardlinks=True)
    except (PathSafetyError, OSError):
        return None
    if path.stat().st_size > _MAX_SIDECAR_BYTES:
        return None
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return payload if isinstance(payload, dict) else None


__all__ = [
    "COMPARE_SCOPE_DESCRIPTIVE",
    "COMPARE_SCOPE_DIFFERENT_DATASETS",
    "COMPARE_SCOPE_LABELS",
    "COMPARE_SCOPE_SAME_DATASET",
    "DATASET_COMPARISON_OUTPUT_ROOT",
    "DatasetComparisonReadiness",
    "SAME_DATASET_OUTPUT_ROOT",
    "comparison_scope_label",
    "dataset_comparison_artifact_label",
    "default_dataset_comparison_id_for_runs",
    "default_dataset_comparison_output_root",
    "default_same_dataset_comparison_id",
    "default_same_dataset_output_root",
    "evaluate_dataset_comparison_readiness",
    "exploratory_comparison_warning",
    "filter_dataset_comparison_candidates",
    "filter_same_dataset_candidates",
    "is_descriptive_comparison_scope",
    "list_completed_development_runs",
]
