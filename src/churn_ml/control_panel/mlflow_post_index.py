"""Automatic post-success MLflow indexing for control-panel training jobs.

Filesystem training status remains authoritative. Indexing failures are recorded
separately and never rewrite a successful job status. Repeat syncs rely on the
existing MLflow sync idempotency (unchanged receipts / no duplicate runs).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.churn_ml.mlflow_config import load_mlflow_config
from src.churn_ml.mlflow_sources import default_source_registry
from src.churn_ml.mlflow_sync import SyncSummary, sync_sources


MLFLOW_INDEX_FILE = "mlflow_index.json"
DEFAULT_MLFLOW_CONFIG = "configs/mlflow/local.yaml"
_RUN_DIRECTORY_RE = re.compile(
    r"(?im)^(?:Run directory|Failed run directory):\s*(.+?)\s*$"
)
_JSON_RUN_DIRECTORY_RE = re.compile(r'"run_directory"\s*:\s*"((?:\\.|[^"\\])*)"')


@dataclass(frozen=True)
class MLflowIndexResult:
    schema_version: int
    attempted: bool
    status: str
    source_type: str | None
    artifact_path: str | None
    message: str | None
    sync_summary: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def source_type_for_job(
    command_id: str, action_id: str, argv: Sequence[str] | None = None
) -> str | None:
    """Map a completed control-panel job to an MLflow source type."""
    joined = " ".join(argv or ()).replace("\\", "/").lower()
    if command_id == "experiment_core_v2" and action_id == "run":
        return "research_v2"
    if command_id in {"autogluon", "autogluon_train"} and action_id in {
        "train",
        "run",
    }:
        return "autogluon"
    if "run_autogluon.py" in joined and "train" in joined:
        return "autogluon"
    return None


def extract_run_directory(log_text: str, *, repository_root: Path) -> Path | None:
    """Parse the produced artifact path from job logs; fail closed when ambiguous."""
    candidates: list[Path] = []
    for match in _RUN_DIRECTORY_RE.finditer(log_text or ""):
        candidates.append(Path(match.group(1).strip().strip('"')))
    for match in _JSON_RUN_DIRECTORY_RE.finditer(log_text or ""):
        raw = bytes(match.group(1), "utf-8").decode("unicode_escape")
        candidates.append(Path(raw))
    if not candidates:
        return None
    resolved_root = repository_root.resolve()
    portable: list[Path] = []
    for candidate in candidates:
        path = candidate if candidate.is_absolute() else resolved_root / candidate
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved == resolved_root or resolved_root not in resolved.parents:
            continue
        portable.append(resolved)
    if not portable:
        return None
    # Prefer the last mention (completion line) when several appear.
    return portable[-1]


def load_index_record(job_root: Path) -> dict[str, Any] | None:
    path = job_root / MLFLOW_INDEX_FILE
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_index_record(job_root: Path, result: MLflowIndexResult) -> None:
    path = job_root / MLFLOW_INDEX_FILE
    path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def maybe_index_successful_job(
    *,
    job_root: Path,
    command_id: str,
    action_id: str,
    argv: Sequence[str] | None,
    job_status: str,
    stdout_text: str,
    repository_root: Path,
    mlflow_config_path: str = DEFAULT_MLFLOW_CONFIG,
    force: bool = False,
) -> MLflowIndexResult:
    """Index the exact completed artifact after a successful train/run job.

    Never mutates job status. Skips when already recorded unless ``force``.
    """
    if job_status != "succeeded":
        return MLflowIndexResult(
            schema_version=1,
            attempted=False,
            status="skipped_not_succeeded",
            source_type=None,
            artifact_path=None,
            message="Job is not succeeded; MLflow indexing was not attempted.",
        )

    source_type = source_type_for_job(command_id, action_id, argv)
    if source_type is None:
        return MLflowIndexResult(
            schema_version=1,
            attempted=False,
            status="skipped_unsupported",
            source_type=None,
            artifact_path=None,
            message="Command is not configured for automatic MLflow indexing.",
        )

    existing = load_index_record(job_root)
    if existing is not None and not force:
        status = str(existing.get("status") or "recorded")
        if status in {"succeeded", "unchanged", "idempotent"}:
            return MLflowIndexResult(
                schema_version=1,
                attempted=False,
                status="skipped_already_indexed",
                source_type=str(existing.get("source_type") or source_type),
                artifact_path=(
                    str(existing["artifact_path"])
                    if existing.get("artifact_path")
                    else None
                ),
                message="Previous MLflow index record is present; skipping.",
                sync_summary=existing.get("sync_summary")
                if isinstance(existing.get("sync_summary"), dict)
                else None,
            )

    run_dir = extract_run_directory(stdout_text, repository_root=repository_root)
    if run_dir is None:
        result = MLflowIndexResult(
            schema_version=1,
            attempted=True,
            status="failed",
            source_type=source_type,
            artifact_path=None,
            message="Could not determine the completed artifact path from job logs.",
        )
        write_index_record(job_root, result)
        return result

    try:
        relative = run_dir.relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        result = MLflowIndexResult(
            schema_version=1,
            attempted=True,
            status="failed",
            source_type=source_type,
            artifact_path=None,
            message="Completed artifact path escapes the repository root.",
        )
        write_index_record(job_root, result)
        return result

    try:
        summary = _sync_one_artifact(
            repository_root=repository_root,
            mlflow_config_path=mlflow_config_path,
            source_type=source_type,
            run_dir=run_dir,
        )
    except Exception as error:  # noqa: BLE001 - indexing must never break training status
        result = MLflowIndexResult(
            schema_version=1,
            attempted=True,
            status="failed",
            source_type=source_type,
            artifact_path=relative,
            message=str(error),
        )
        write_index_record(job_root, result)
        return result

    status = _status_from_summary(summary)
    result = MLflowIndexResult(
        schema_version=1,
        attempted=True,
        status=status,
        source_type=source_type,
        artifact_path=relative,
        message=None
        if status in {"succeeded", "unchanged", "idempotent"}
        else ("MLflow indexing reported failures; training status is unchanged."),
        sync_summary=summary.to_dict(),
    )
    write_index_record(job_root, result)
    return result


def _sync_one_artifact(
    *,
    repository_root: Path,
    mlflow_config_path: str,
    source_type: str,
    run_dir: Path,
) -> SyncSummary:
    config = load_mlflow_config(
        repository_root / mlflow_config_path,
        repository_root=repository_root,
    )
    return sync_sources(
        config,
        registry=default_source_registry(),
        source_types=(source_type,),
        run_dir=run_dir,
        dry_run=False,
        fail_fast=False,
    )


def _status_from_summary(summary: SyncSummary) -> str:
    if summary.has_failures:
        return "failed"
    counts = summary.counts
    created = int(counts.get("created", 0) or 0)
    resumed = int(counts.get("resumed", 0) or 0)
    unchanged = int(counts.get("unchanged", 0) or 0)
    if unchanged and not created and not resumed:
        return "idempotent"
    if created or resumed or unchanged:
        return "succeeded"
    return "succeeded"
