"""Read-only MLflow status projection for Results Model runs.

Never syncs, writes receipts, mutates job index records, or opens the store
for writes. Missing/unreadable stores must not break artifact browsing.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class MLflowResultStatus(str, Enum):
    INDEXED = "indexed"
    NOT_INDEXED = "not_indexed"
    INDEX_FAILED = "index_failed"
    INDEXED_WITH_STALE_FAILURE_RECORD = "indexed_with_stale_failure_record"
    ORPHANED_SOURCE = "orphaned_source"
    INVALID_INDEX_RECORD = "invalid_index_record"
    NOT_APPLICABLE = "not_applicable"
    STORE_UNAVAILABLE = "store_unavailable"


ELIGIBLE_SOURCE_TYPES = frozenset({"research_v2", "autogluon"})


@dataclass(frozen=True)
class MLflowRunEvidence:
    run_uuid: str
    experiment_id: str
    experiment_name: str
    source_type: str | None
    source_identity: str | None
    source_relative_path: str | None
    source_run_id: str | None
    source_key: str | None
    status: str | None
    lifecycle_stage: str | None


@dataclass(frozen=True)
class JobIndexEvidence:
    job_id: str
    source_type: str | None
    artifact_path: str | None
    status: str | None
    message: str | None
    attempted: bool | None


@dataclass(frozen=True)
class MLflowEntityStatus:
    status: MLflowResultStatus
    experiment_name: str | None = None
    mlflow_run_id: str | None = None
    source_relative_path: str | None = None
    source_identity: str | None = None
    message: str | None = None
    job_id: str | None = None


@dataclass(frozen=True)
class MLflowDiagnosticsSummary:
    indexed_local_runs: int
    unindexed_eligible_runs: int
    failed_index_records: int
    stale_failure_records: int
    orphaned_mlflow_runs: int
    invalid_records: int
    store_available: bool
    duplicate_source_identities: int


@dataclass(frozen=True)
class MLflowProjection:
    store_available: bool
    by_source_path: Mapping[str, MLflowRunEvidence]
    by_source_identity: Mapping[str, tuple[MLflowRunEvidence, ...]]
    job_indexes: tuple[JobIndexEvidence, ...]
    orphans: tuple[MLflowRunEvidence, ...]
    experiment_names: Mapping[str, str]
    diagnostics: MLflowDiagnosticsSummary


def mlflow_projection_fingerprint(repository_root: Path) -> str:
    """Cheap fingerprint for cache invalidation (no DB hashing)."""
    root = repository_root.resolve()
    entries: list[dict[str, Any]] = []
    for relative in (
        "configs/mlflow/local.yaml",
        "artifacts/mlflow/mlflow.db",
        "artifacts/mlflow/sync_receipts",
        "artifacts/ui_jobs",
    ):
        path = root / relative
        entries.append(_stat_entry(relative, path))
    return json.dumps(entries, sort_keys=True)


def build_mlflow_projection(repository_root: Path) -> MLflowProjection:
    """Load a read-only projection of local MLflow + job index evidence."""
    root = repository_root.resolve()
    db = root / "artifacts" / "mlflow" / "mlflow.db"
    experiment_names: dict[str, str] = {}
    runs: list[MLflowRunEvidence] = []
    store_available = False
    if db.is_file():
        try:
            experiment_names, runs = _read_sqlite_readonly(db)
            store_available = True
        except Exception:
            store_available = False
            experiment_names, runs = {}, []

    by_path: dict[str, MLflowRunEvidence] = {}
    by_identity: dict[str, list[MLflowRunEvidence]] = {}
    orphans: list[MLflowRunEvidence] = []
    for run in runs:
        if run.source_type not in ELIGIBLE_SOURCE_TYPES:
            continue
        if run.source_relative_path:
            normalized = run.source_relative_path.replace("\\", "/")
            by_path[normalized] = run
            local = root / normalized
            if not local.exists():
                orphans.append(run)
        if run.source_identity:
            by_identity.setdefault(run.source_identity, []).append(run)

    job_indexes = tuple(_load_job_indexes(root))
    duplicate_identities = sum(1 for items in by_identity.values() if len(items) > 1)

    # Diagnostics counts are filled by callers that know local eligible paths;
    # here we store store-level orphan/invalid/failure evidence.
    failed = sum(1 for item in job_indexes if item.status == "failed")
    invalid = sum(
        1
        for item in job_indexes
        if item.status is None and item.artifact_path is None and item.attempted is None
    )
    stale = 0
    for item in job_indexes:
        if item.status != "failed" or not item.artifact_path:
            continue
        path = item.artifact_path.replace("\\", "/")
        if path in by_path:
            stale += 1

    diagnostics = MLflowDiagnosticsSummary(
        indexed_local_runs=0,
        unindexed_eligible_runs=0,
        failed_index_records=failed,
        stale_failure_records=stale,
        orphaned_mlflow_runs=len(orphans),
        invalid_records=invalid,
        store_available=store_available,
        duplicate_source_identities=duplicate_identities,
    )
    return MLflowProjection(
        store_available=store_available,
        by_source_path=by_path,
        by_source_identity={key: tuple(value) for key, value in by_identity.items()},
        job_indexes=job_indexes,
        orphans=tuple(orphans),
        experiment_names=experiment_names,
        diagnostics=diagnostics,
    )


def resolve_model_run_mlflow_status(
    *,
    source_type: str,
    source_path: str,
    projection: MLflowProjection,
) -> MLflowEntityStatus:
    if source_type not in ELIGIBLE_SOURCE_TYPES:
        return MLflowEntityStatus(
            status=MLflowResultStatus.NOT_APPLICABLE,
            message="Entity type is not configured for MLflow indexing.",
            source_relative_path=source_path,
        )
    if not projection.store_available:
        return MLflowEntityStatus(
            status=MLflowResultStatus.STORE_UNAVAILABLE,
            message="MLflow SQLite store is missing or unreadable.",
            source_relative_path=source_path,
        )

    path = source_path.replace("\\", "/")
    run = projection.by_source_path.get(path)
    job = _job_for_path(projection.job_indexes, path)

    if run is not None:
        if job is not None and job.status == "failed":
            return MLflowEntityStatus(
                status=MLflowResultStatus.INDEXED_WITH_STALE_FAILURE_RECORD,
                experiment_name=run.experiment_name,
                mlflow_run_id=run.run_uuid,
                source_relative_path=path,
                source_identity=run.source_identity,
                message=job.message
                or "Job index record failed earlier, but an MLflow run exists.",
                job_id=job.job_id,
            )
        return MLflowEntityStatus(
            status=MLflowResultStatus.INDEXED,
            experiment_name=run.experiment_name,
            mlflow_run_id=run.run_uuid,
            source_relative_path=path,
            source_identity=run.source_identity,
            job_id=job.job_id if job else None,
        )

    if job is not None and job.status == "failed":
        return MLflowEntityStatus(
            status=MLflowResultStatus.INDEX_FAILED,
            source_relative_path=path,
            message=job.message or "MLflow indexing failed.",
            job_id=job.job_id,
        )
    if job is not None and job.status in {"succeeded", "idempotent", "unchanged"}:
        return MLflowEntityStatus(
            status=MLflowResultStatus.INVALID_INDEX_RECORD,
            source_relative_path=path,
            message="Job index claims success but no MLflow run was found.",
            job_id=job.job_id,
        )
    return MLflowEntityStatus(
        status=MLflowResultStatus.NOT_INDEXED,
        source_relative_path=path,
        message="Eligible local run has no MLflow index evidence.",
    )


def annotate_diagnostics_for_paths(
    projection: MLflowProjection,
    *,
    eligible_paths: Mapping[str, str],
) -> MLflowDiagnosticsSummary:
    """Recompute path-relative diagnostics using current Model-run inventory."""
    indexed = 0
    unindexed = 0
    for path, source_type in eligible_paths.items():
        status = resolve_model_run_mlflow_status(
            source_type=source_type,
            source_path=path,
            projection=projection,
        )
        if status.status in {
            MLflowResultStatus.INDEXED,
            MLflowResultStatus.INDEXED_WITH_STALE_FAILURE_RECORD,
        }:
            indexed += 1
        elif status.status in {
            MLflowResultStatus.NOT_INDEXED,
            MLflowResultStatus.INDEX_FAILED,
        }:
            unindexed += 1
    base = projection.diagnostics
    return MLflowDiagnosticsSummary(
        indexed_local_runs=indexed,
        unindexed_eligible_runs=unindexed,
        failed_index_records=base.failed_index_records,
        stale_failure_records=base.stale_failure_records,
        orphaned_mlflow_runs=base.orphaned_mlflow_runs,
        invalid_records=base.invalid_records,
        store_available=base.store_available,
        duplicate_source_identities=base.duplicate_source_identities,
    )


def _read_sqlite_readonly(
    db: Path,
) -> tuple[dict[str, str], list[MLflowRunEvidence]]:
    uri = f"file:{db.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        # Refuse writes even if a caller later misuses the connection pattern.
        conn.execute("PRAGMA query_only = ON")
        experiments = {
            str(row["experiment_id"]): str(row["name"])
            for row in conn.execute(
                "SELECT experiment_id, name FROM experiments"
            )
        }
        rows = conn.execute(
            """
            SELECT r.run_uuid, r.experiment_id, r.status, r.lifecycle_stage,
                   MAX(CASE WHEN t.key='mlflow_index.source_type' THEN t.value END)
                     AS source_type,
                   MAX(CASE WHEN t.key='mlflow_index.source_identity' THEN t.value END)
                     AS source_identity,
                   MAX(CASE WHEN t.key='mlflow_index.source_relative_path' THEN t.value END)
                     AS source_relative_path,
                   MAX(CASE WHEN t.key='mlflow_index.source_run_id' THEN t.value END)
                     AS source_run_id,
                   MAX(CASE WHEN t.key='mlflow_index.source_key' THEN t.value END)
                     AS source_key
            FROM runs r
            LEFT JOIN tags t ON t.run_uuid = r.run_uuid
            GROUP BY r.run_uuid
            """
        )
        evidence: list[MLflowRunEvidence] = []
        for row in rows:
            exp_id = str(row["experiment_id"])
            evidence.append(
                MLflowRunEvidence(
                    run_uuid=str(row["run_uuid"]),
                    experiment_id=exp_id,
                    experiment_name=experiments.get(exp_id, exp_id),
                    source_type=row["source_type"],
                    source_identity=row["source_identity"],
                    source_relative_path=row["source_relative_path"],
                    source_run_id=row["source_run_id"],
                    source_key=row["source_key"],
                    status=row["status"],
                    lifecycle_stage=row["lifecycle_stage"],
                )
            )
        return experiments, evidence
    finally:
        conn.close()


def _load_job_indexes(root: Path) -> list[JobIndexEvidence]:
    jobs_root = root / "artifacts" / "ui_jobs"
    if not jobs_root.is_dir():
        return []
    items: list[JobIndexEvidence] = []
    for job_dir in jobs_root.iterdir():
        if not job_dir.is_dir():
            continue
        path = job_dir / "mlflow_index.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            items.append(
                JobIndexEvidence(
                    job_id=job_dir.name,
                    source_type=None,
                    artifact_path=None,
                    status=None,
                    message="Invalid mlflow_index.json",
                    attempted=None,
                )
            )
            continue
        if not isinstance(payload, dict):
            items.append(
                JobIndexEvidence(
                    job_id=job_dir.name,
                    source_type=None,
                    artifact_path=None,
                    status=None,
                    message="mlflow_index.json is not an object",
                    attempted=None,
                )
            )
            continue
        artifact_path = payload.get("artifact_path")
        items.append(
            JobIndexEvidence(
                job_id=job_dir.name,
                source_type=(
                    str(payload["source_type"])
                    if payload.get("source_type") is not None
                    else None
                ),
                artifact_path=str(artifact_path) if artifact_path else None,
                status=str(payload["status"]) if payload.get("status") else None,
                message=str(payload["message"]) if payload.get("message") else None,
                attempted=bool(payload["attempted"])
                if isinstance(payload.get("attempted"), bool)
                else None,
            )
        )
    return items


def _job_for_path(
    jobs: tuple[JobIndexEvidence, ...], path: str
) -> JobIndexEvidence | None:
    matches = [
        item
        for item in jobs
        if item.artifact_path and item.artifact_path.replace("\\", "/") == path
    ]
    if not matches:
        return None
    # Prefer failed records for stale detection, else newest-ish by job_id.
    failed = [item for item in matches if item.status == "failed"]
    if failed:
        return failed[-1]
    return matches[-1]


def _stat_entry(relative: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": relative, "exists": False}
    try:
        stat = path.stat()
        entry: dict[str, Any] = {
            "path": relative,
            "exists": True,
            "size": int(getattr(stat, "st_size", 0) or 0),
            "mtime_ns": int(getattr(stat, "st_mtime_ns", 0) or 0),
        }
    except OSError:
        return {"path": relative, "exists": False}
    if not path.is_dir():
        return entry
    try:
        children = list(path.iterdir())
    except OSError:
        children = []
    child_names = sorted(child.name for child in children)
    entry["child_count"] = len(child_names)
    entry["child_names_sample"] = child_names[:50]
    if relative.endswith("ui_jobs"):
        index_mtime: list[int] = []
        for child in children:
            index = child / "mlflow_index.json"
            if index.is_file():
                try:
                    index_mtime.append(int(index.stat().st_mtime_ns))
                except OSError:
                    continue
        entry["mlflow_index_count"] = len(index_mtime)
        entry["mlflow_index_mtime_max"] = max(index_mtime) if index_mtime else 0
    if relative.endswith("sync_receipts"):
        try:
            receipt_count = sum(1 for _ in path.rglob("*.json"))
        except OSError:
            receipt_count = 0
        entry["receipt_count"] = receipt_count
    return entry
