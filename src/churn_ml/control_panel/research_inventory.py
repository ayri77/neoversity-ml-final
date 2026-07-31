"""Normalized Research v2 inventory for the Control Panel Research Workspace.

Filesystem Research v2 artifacts remain authoritative. Missing identity fields
are recorded as unavailable rather than inferred. Invalid runs are isolated and
do not crash discovery.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

from src.churn_ml.control_panel.dataset_identity import (
    DatasetIdentityConflictError,
    DatasetIdentityMalformedError,
    DatasetIdentityUnsafeError,
    read_dataset_identity_safe,
)
from src.churn_ml.control_panel.path_safety import (
    PathSafetyError,
    path_exists_nonfollowing,
    require_regular_file,
    require_safe_directory,
)
from src.churn_ml.control_panel.presentation import (
    normalize_mode,
    normalize_model_family,
    parse_run_id_timestamp,
)


UNAVAILABLE = "unavailable"
RESEARCH_V2_ROOT = "artifacts/research_v2"
AUTHORITATIVE_OOF_RELATIVE = "predictions/outer_validation.parquet"
INVENTORY_CACHE_VERSION = "research_workspace_inventory_v1"
_MAX_JSON_BYTES = 2_000_000
_MAX_YAML_BYTES = 512 * 1024


@dataclass(frozen=True)
class ResearchRunInventoryRow:
    """Normalized Research v2 inventory row for Research Workspace views."""

    relative_path: str
    run_id: str | None
    created_at_utc: str | None
    status: str
    dataset_id: str | None
    parent_dataset_id: str | None
    target_dependency: str | None
    n_features: int | None
    model_family: str | None
    adapter_id: str | None
    model_config_id: str | None
    source_config_path: str | None
    source_config_hash: str | None
    resolved_config_hash: str | None
    evaluation_plan_id: str | None
    evaluation_plan_hash: str | None
    evaluation_mode: str | None
    repeat_seeds: tuple[int, ...] | None
    outer_fold_count: int | None
    threshold_selection_protocol: str | None
    feature_pipeline_id: str | None
    balanced_accuracy: float | None
    sensitivity: float | None
    specificity: float | None
    roc_auc: float | None
    average_precision: float | None
    brier_score: float | None
    threshold_median: float | None
    competition_assets_accessed: bool | None
    authoritative_oof_path: str | None
    target_hash: str | None
    train_row_identity_hash: str | None
    candidate_adapter_hash: str | None
    candidate_hash: str | None
    train_content_hash: str | None
    schema_hash: str | None
    has_search_provenance: bool
    identity_complete: bool
    diagnostic: str | None = None
    unavailable_fields: tuple[str, ...] = field(default_factory=tuple)

    def display(self, field_name: str) -> str:
        value = getattr(self, field_name, None)
        if value is None:
            return UNAVAILABLE
        if isinstance(value, tuple):
            return ",".join(str(item) for item in value)
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def to_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["repeat_seeds"] = (
            list(self.repeat_seeds) if self.repeat_seeds is not None else None
        )
        payload["unavailable_fields"] = list(self.unavailable_fields)
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ResearchRunInventoryRow:
        data = dict(payload)
        seeds = data.get("repeat_seeds")
        if isinstance(seeds, list):
            data["repeat_seeds"] = tuple(int(item) for item in seeds)
        unavailable = data.get("unavailable_fields")
        if isinstance(unavailable, list):
            data["unavailable_fields"] = tuple(str(item) for item in unavailable)
        return cls(**data)


def discover_research_v2_run_roots(repository_root: Path) -> list[Path]:
    """Discover Research v2 run directories under the configured artifact root."""
    root = repository_root.resolve()
    artifact_root = root / RESEARCH_V2_ROOT
    if not path_exists_nonfollowing(artifact_root):
        return []
    try:
        require_safe_directory(artifact_root)
    except PathSafetyError:
        return []
    runs: list[Path] = []
    for plan_dir in sorted(artifact_root.iterdir(), key=lambda path: path.name):
        if not _is_safe_dir(plan_dir):
            continue
        for pipeline_dir in sorted(plan_dir.iterdir(), key=lambda path: path.name):
            if not _is_safe_dir(pipeline_dir):
                continue
            for run_dir in sorted(pipeline_dir.iterdir(), key=lambda path: path.name):
                if not _is_safe_dir(run_dir):
                    continue
                runs.append(run_dir)
    return runs


def build_research_inventory(
    repository_root: Path,
) -> list[ResearchRunInventoryRow]:
    """Build a deterministic normalized inventory of discoverable Research v2 runs."""
    root = repository_root.resolve()
    rows: list[ResearchRunInventoryRow] = []
    for run_dir in discover_research_v2_run_roots(root):
        try:
            relative = _repository_relative(root, run_dir)
            rows.append(normalize_research_run(root, run_dir, relative_path=relative))
        except Exception as error:  # noqa: BLE001 - isolate corrupt runs
            try:
                relative = _repository_relative(root, run_dir)
            except Exception:  # noqa: BLE001
                relative = run_dir.as_posix()
            rows.append(_invalid_row(relative, f"Inventory isolation: {error}"))
    rows.sort(key=_inventory_sort_key)
    return rows


def normalize_research_run(
    repository_root: Path,
    run_dir: Path,
    *,
    relative_path: str | None = None,
) -> ResearchRunInventoryRow:
    """Normalize one Research v2 run directory into an inventory row."""
    root = repository_root.resolve()
    absolute = run_dir if run_dir.is_absolute() else root / run_dir
    try:
        require_safe_directory(absolute)
    except PathSafetyError as error:
        rel = relative_path or absolute.as_posix()
        return _invalid_row(rel, f"Unsafe run directory: {error}")

    relative = relative_path or _repository_relative(root, absolute)
    status = _status_from_markers(absolute)
    metadata = _read_json(absolute / "run_metadata.json")
    aggregate = _read_json(absolute / "metrics" / "aggregate.json")
    thresholds = _read_json(absolute / "thresholds" / "threshold_summary.json")
    plan_identity = _read_json(absolute / "identities" / "evaluation_plan.json")
    source_identity = _read_json(absolute / "identities" / "source.json")
    candidate_identity = _read_json(absolute / "identities" / "candidate.json")
    adapter_identity = _read_json(absolute / "identities" / "candidate_adapter.json")
    resolved = _read_yaml(absolute / "resolved_config.yaml")
    fingerprints = _read_json(absolute / "dataset_fingerprints.json")

    identity_view = read_dataset_identity_safe(absolute)
    identity_diagnostic = identity_view.diagnostic

    run_id = _as_str(_dig(metadata, "run_id")) or absolute.name
    created_at = _as_str(_dig(metadata, "started_at_utc")) or parse_run_id_timestamp(
        run_id
    )
    adapter_id = _as_str(_dig(metadata, "candidate_adapter_id")) or _as_str(
        _dig(adapter_identity, "canonical", "id")
    )
    model_family = normalize_model_family(adapter_id) if adapter_id else None
    feature_pipeline_id = _as_str(_dig(metadata, "feature_pipeline_id"))
    plan_id = _as_str(_dig(metadata, "plan_id")) or _as_str(
        _dig(plan_identity, "canonical", "plan_id")
    )
    plan_hash = _as_str(_dig(metadata, "hashes", "plan")) or _as_str(
        _dig(plan_identity, "sha256")
    )
    candidate_hash = _as_str(_dig(metadata, "hashes", "candidate")) or _as_str(
        _dig(candidate_identity, "sha256")
    )
    candidate_adapter_hash = _as_str(
        _dig(metadata, "hashes", "candidate_adapter")
    ) or _as_str(_dig(adapter_identity, "sha256"))
    model_config_id = candidate_hash or candidate_adapter_hash or adapter_id

    source_config_path, source_config_hash = _source_config_identity(source_identity)
    if source_config_path is None:
        source_config_path = _as_str(_dig(resolved, "evaluation_plan_path"))
    resolved_config_hash = _file_sha256_if_safe(absolute / "resolved_config.yaml")

    evaluation_mode = _infer_evaluation_mode(
        plan_id=plan_id,
        relative_path=relative,
        resolved=resolved,
    )
    repeat_seeds, outer_fold_count, threshold_protocol = _protocol_fields(
        plan_identity, resolved
    )

    metrics = _metric_means(aggregate)
    threshold_median = _as_float(_dig(thresholds, "median"))
    competition_assets = _as_bool(_dig(metadata, "competition_assets_accessed"))
    if competition_assets is None:
        competition_assets = _as_bool(
            _dig(metadata, "runtime", "competition_assets_accessed")
        )

    oof_relative = None
    oof_path = absolute / AUTHORITATIVE_OOF_RELATIVE
    if path_exists_nonfollowing(oof_path):
        oof_relative = f"{relative}/{AUTHORITATIVE_OOF_RELATIVE}"

    has_search_provenance = isinstance(_dig(resolved, "search_provenance"), dict)

    dataset_id = identity_view.dataset_id
    parent_dataset_id = identity_view.parent_dataset_id
    target_dependency = identity_view.target_dependency
    n_features = identity_view.n_features
    target_hash = identity_view.target_hash
    train_row_identity_hash = identity_view.train_row_identity_hash
    train_content_hash = identity_view.train_content_hash
    schema_hash = identity_view.schema_hash

    if dataset_id is None:
        dataset_id = _as_str(_dig(fingerprints, "dataset_version")) or _as_str(
            _dig(resolved, "dataset", "version")
        )

    metadata_status = _as_str(_dig(metadata, "status"))
    if status == "completed" and metadata_status and metadata_status != "completed":
        status = metadata_status
    if status == "invalid" and metadata_status:
        status = metadata_status

    diagnostic_parts: list[str] = []
    if identity_diagnostic:
        diagnostic_parts.append(identity_diagnostic)

    required_for_complete = {
        "dataset_id": dataset_id,
        "model_family": model_family,
        "adapter_id": adapter_id,
        "model_config_id": model_config_id,
        "evaluation_plan_id": plan_id,
        "evaluation_plan_hash": plan_hash,
        "evaluation_mode": evaluation_mode,
        "repeat_seeds": repeat_seeds,
        "outer_fold_count": outer_fold_count,
        "threshold_selection_protocol": threshold_protocol,
        "feature_pipeline_id": feature_pipeline_id,
        "target_hash": target_hash,
        "train_row_identity_hash": train_row_identity_hash,
    }
    unavailable_fields = tuple(
        name for name, value in required_for_complete.items() if value is None
    )
    identity_complete = not unavailable_fields and status == "completed"

    if status not in {"completed", "failed", "running"}:
        status = "invalid"
        diagnostic_parts.append("Unrecognized or missing run markers.")

    return ResearchRunInventoryRow(
        relative_path=relative,
        run_id=run_id,
        created_at_utc=created_at,
        status=status,
        dataset_id=dataset_id,
        parent_dataset_id=parent_dataset_id,
        target_dependency=target_dependency,
        n_features=n_features,
        model_family=model_family,
        adapter_id=adapter_id,
        model_config_id=model_config_id,
        source_config_path=source_config_path,
        source_config_hash=source_config_hash,
        resolved_config_hash=resolved_config_hash,
        evaluation_plan_id=plan_id,
        evaluation_plan_hash=plan_hash,
        evaluation_mode=evaluation_mode,
        repeat_seeds=repeat_seeds,
        outer_fold_count=outer_fold_count,
        threshold_selection_protocol=threshold_protocol,
        feature_pipeline_id=feature_pipeline_id,
        balanced_accuracy=metrics.get("balanced_accuracy"),
        sensitivity=metrics.get("sensitivity"),
        specificity=metrics.get("specificity"),
        roc_auc=metrics.get("roc_auc"),
        average_precision=metrics.get("average_precision"),
        brier_score=metrics.get("brier_score"),
        threshold_median=threshold_median,
        competition_assets_accessed=competition_assets,
        authoritative_oof_path=oof_relative,
        target_hash=target_hash,
        train_row_identity_hash=train_row_identity_hash,
        candidate_adapter_hash=candidate_adapter_hash,
        candidate_hash=candidate_hash,
        train_content_hash=train_content_hash,
        schema_hash=schema_hash,
        has_search_provenance=has_search_provenance,
        identity_complete=identity_complete,
        diagnostic="; ".join(diagnostic_parts) if diagnostic_parts else None,
        unavailable_fields=unavailable_fields,
    )


def inventory_rows_as_mappings(
    rows: list[ResearchRunInventoryRow],
) -> list[dict[str, Any]]:
    return [row.to_mapping() for row in rows]


def inventory_rows_from_mappings(
    payloads: list[Mapping[str, Any]],
) -> list[ResearchRunInventoryRow]:
    return [ResearchRunInventoryRow.from_mapping(payload) for payload in payloads]


def _invalid_row(relative_path: str, diagnostic: str) -> ResearchRunInventoryRow:
    return ResearchRunInventoryRow(
        relative_path=relative_path,
        run_id=None,
        created_at_utc=None,
        status="invalid",
        dataset_id=None,
        parent_dataset_id=None,
        target_dependency=None,
        n_features=None,
        model_family=None,
        adapter_id=None,
        model_config_id=None,
        source_config_path=None,
        source_config_hash=None,
        resolved_config_hash=None,
        evaluation_plan_id=None,
        evaluation_plan_hash=None,
        evaluation_mode=None,
        repeat_seeds=None,
        outer_fold_count=None,
        threshold_selection_protocol=None,
        feature_pipeline_id=None,
        balanced_accuracy=None,
        sensitivity=None,
        specificity=None,
        roc_auc=None,
        average_precision=None,
        brier_score=None,
        threshold_median=None,
        competition_assets_accessed=None,
        authoritative_oof_path=None,
        target_hash=None,
        train_row_identity_hash=None,
        candidate_adapter_hash=None,
        candidate_hash=None,
        train_content_hash=None,
        schema_hash=None,
        has_search_provenance=False,
        identity_complete=False,
        diagnostic=diagnostic,
        unavailable_fields=(
            "dataset_id",
            "model_family",
            "adapter_id",
            "model_config_id",
            "evaluation_plan_id",
            "evaluation_plan_hash",
            "evaluation_mode",
            "repeat_seeds",
            "outer_fold_count",
            "threshold_selection_protocol",
            "feature_pipeline_id",
            "target_hash",
            "train_row_identity_hash",
        ),
    )


def _inventory_sort_key(row: ResearchRunInventoryRow) -> tuple[Any, ...]:
    return (
        row.dataset_id or "",
        row.model_family or "",
        row.created_at_utc or "",
        row.relative_path,
    )


def _status_from_markers(run_dir: Path) -> str:
    success = path_exists_nonfollowing(run_dir / "_SUCCESS")
    failed = path_exists_nonfollowing(run_dir / "_FAILED")
    if success and not failed:
        return "completed"
    if failed and not success:
        return "failed"
    if success and failed:
        return "invalid"
    return "running"


def _infer_evaluation_mode(
    *,
    plan_id: str | None,
    relative_path: str,
    resolved: Mapping[str, Any] | None,
) -> str | None:
    tokens = " ".join(
        filter(
            None,
            [
                plan_id,
                relative_path,
                _as_str(_dig(resolved, "experiment", "id")),
            ],
        )
    ).lower()
    if "smoke" in tokens:
        return normalize_mode("smoke")
    if "development" in tokens:
        return normalize_mode("development")
    if "deployment" in tokens:
        return normalize_mode("deployment")
    return None


def _protocol_fields(
    plan_identity: Mapping[str, Any] | None,
    resolved: Mapping[str, Any] | None,
) -> tuple[tuple[int, ...] | None, int | None, str | None]:
    canonical = _dig(plan_identity, "canonical")
    outer = None
    threshold_selection = None
    threshold_policy = None
    if isinstance(canonical, dict):
        outer = canonical.get("outer_evaluation")
        threshold_selection = canonical.get("threshold_selection")
        threshold_policy = canonical.get("threshold_policy")
    if outer is None and isinstance(resolved, dict):
        plan = resolved.get("evaluation_plan")
        if isinstance(plan, dict):
            outer = plan.get("outer_evaluation")
            threshold_selection = plan.get("threshold_selection")
            threshold_policy = plan.get("threshold_policy")
    seeds = None
    fold_count = None
    if isinstance(outer, dict):
        raw_seeds = outer.get("repeat_seeds")
        if isinstance(raw_seeds, list) and all(isinstance(item, int) for item in raw_seeds):
            seeds = tuple(raw_seeds)
        fold_count = _as_int(outer.get("n_splits"))
    protocol = None
    policy_id = _as_str(_dig(threshold_policy, "id")) if isinstance(threshold_policy, dict) else None
    selection_bits: list[str] = []
    if isinstance(threshold_selection, dict):
        splitter = _as_str(threshold_selection.get("splitter"))
        n_splits = _as_int(threshold_selection.get("n_splits"))
        random_state = threshold_selection.get("random_state")
        if splitter:
            selection_bits.append(splitter)
        if n_splits is not None:
            selection_bits.append(f"n_splits={n_splits}")
        if isinstance(random_state, int):
            selection_bits.append(f"random_state={random_state}")
    if policy_id:
        selection_bits.insert(0, policy_id)
    if selection_bits:
        protocol = "|".join(selection_bits)
    return seeds, fold_count, protocol


def _metric_means(aggregate: Mapping[str, Any] | None) -> dict[str, float | None]:
    names = (
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "roc_auc",
        "average_precision",
        "brier_score",
    )
    result: dict[str, float | None] = {}
    for name in names:
        result[name] = _as_float(_dig(aggregate, "metrics", name, "mean"))
    return result


def _source_config_identity(
    source_identity: Mapping[str, Any] | None,
) -> tuple[str | None, str | None]:
    if not isinstance(source_identity, dict):
        return None, None
    source_hash = _as_str(source_identity.get("sha256"))
    canonical = source_identity.get("canonical")
    path = None
    if isinstance(canonical, dict):
        files = canonical.get("files")
        if isinstance(files, list) and files:
            first = files[0]
            if isinstance(first, dict):
                path = _as_str(first.get("path"))
                if source_hash is None:
                    source_hash = _as_str(first.get("sha256"))
    return path, source_hash


def _file_sha256_if_safe(path: Path) -> str | None:
    if not path_exists_nonfollowing(path):
        return None
    try:
        require_regular_file(path, reject_hardlinks=True)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, PathSafetyError):
        return None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path_exists_nonfollowing(path):
        return None
    try:
        require_regular_file(path, reject_hardlinks=True)
        if path.stat().st_size > _MAX_JSON_BYTES:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        PathSafetyError,
    ):
        return None
    return payload if isinstance(payload, dict) else None


def _read_yaml(path: Path) -> dict[str, Any] | None:
    if not path_exists_nonfollowing(path):
        return None
    try:
        require_regular_file(path, reject_hardlinks=True)
        if path.stat().st_size > _MAX_YAML_BYTES:
            return None
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError, PathSafetyError):
        return None
    return payload if isinstance(payload, dict) else None


def _repository_relative(repository_root: Path, path: Path) -> str:
    relative = path.resolve().relative_to(repository_root.resolve())
    return PurePosixPath(*relative.parts).as_posix()


def _is_safe_dir(path: Path) -> bool:
    try:
        require_safe_directory(path)
        return True
    except PathSafetyError:
        return False


def _dig(payload: Any, *keys: str) -> Any:
    current = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return str(value)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


# Re-export identity exceptions for callers that want to distinguish diagnostics.
__all__ = [
    "AUTHORITATIVE_OOF_RELATIVE",
    "INVENTORY_CACHE_VERSION",
    "RESEARCH_V2_ROOT",
    "UNAVAILABLE",
    "ResearchRunInventoryRow",
    "build_research_inventory",
    "discover_research_v2_run_roots",
    "inventory_rows_as_mappings",
    "inventory_rows_from_mappings",
    "normalize_research_run",
    "DatasetIdentityConflictError",
    "DatasetIdentityMalformedError",
    "DatasetIdentityUnsafeError",
]
