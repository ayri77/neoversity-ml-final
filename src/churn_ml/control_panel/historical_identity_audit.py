"""Read-only historical dataset-identity audit for Control Panel sources.

Never mutates jobs, artifacts, manifests, or MLflow state. Never invents a
canonical Registry ID such as ``v0_raw_minimal``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from src.churn_ml.control_panel.artifacts import discover_artifacts
from src.churn_ml.control_panel.dataset_identity import (
    DatasetIdentityConflictError,
    DatasetIdentityMalformedError,
    DatasetIdentityUnsafeError,
    read_dataset_identity,
)
from src.churn_ml.control_panel.jobs import JobManager
from src.churn_ml.control_panel.path_safety import PathSafetyError
from src.churn_ml.control_panel.registry import load_registry


CLASSIFICATIONS = (
    "persisted_and_complete",
    "persisted_but_partial",
    "resolved_through_legacy_fallback",
    "conflicting",
    "missing",
    "unsafe_unreadable",
)


@dataclass(frozen=True)
class IdentityAuditRecord:
    kind: str
    path: str
    classification: str
    recorded_dataset_id: str | None
    source: str | None
    config_fallback: str | None
    n_features: int | None
    schema_hash: str | None
    train_content_hash: str | None
    target_hash: str | None
    train_row_identity_hash: str | None
    canonical_registry_equivalent: str | None
    missing_evidence: tuple[str, ...]
    diagnostic: str | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["missing_evidence"] = list(self.missing_evidence)
        return payload


@dataclass(frozen=True)
class IdentityAuditReport:
    records: tuple[IdentityAuditRecord, ...]

    def counts(self) -> dict[str, int]:
        totals = {key: 0 for key in CLASSIFICATIONS}
        for record in self.records:
            totals[record.classification] = totals.get(record.classification, 0) + 1
        return totals

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": self.counts(),
            "records": [record.to_dict() for record in self.records],
            "proposed_legacy_to_canonical_mappings": [],
            "notes": [
                "No historical records were modified.",
                "Canonical Registry equivalence is never invented; "
                "blanket v0_raw_minimal backfill is prohibited.",
            ],
        }


def audit_historical_dataset_identity(
    repository_root: Path,
    *,
    include_jobs: bool = True,
    include_artifacts: bool = True,
) -> IdentityAuditReport:
    """Scan persisted UI jobs and configured artifact roots read-only."""
    root = Path(repository_root).resolve()
    loaded = load_registry(root)
    records: list[IdentityAuditRecord] = []

    if include_jobs:
        manager = JobManager(
            root / loaded.settings.jobs_root,
            working_directory=root / loaded.settings.working_directory,
            commands=loaded.commands,
        )
        for job_record in manager.list_jobs(refresh=False):
            records.append(_audit_job(job_record.job, repository_root=root))

    if include_artifacts:
        for reader_id, reader in loaded.readers.items():
            if reader_id not in {
                "research_v2",
                "paired_comparison",
                "research_v2_comparisons",
            } and "research_v2" not in reader_id:
                # Still scan every configured reader; identity is best-effort.
                pass
            for artifact in discover_artifacts(root, reader):
                records.append(
                    _audit_artifact(
                        kind=f"artifact:{reader_id}",
                        path=artifact.relative_path,
                        run_root=artifact.root,
                    )
                )

    return IdentityAuditReport(records=tuple(records))


def _audit_job(job: Mapping[str, Any], *, repository_root: Path) -> IdentityAuditRecord:
    job_id = str(job.get("job_id") or "unknown")
    references = job.get("references")
    if not isinstance(references, Mapping):
        references = {}
    dataset_id = references.get("dataset_id")
    config = references.get("config")
    config_fallback = str(config) if isinstance(config, str) and config else None
    missing: list[str] = []
    if dataset_id:
        classification = "persisted_and_complete"
        if not references.get("experiment_id") or not references.get("plan_id"):
            classification = "persisted_but_partial"
            missing.append("experiment_or_plan_reference")
        return IdentityAuditRecord(
            kind="ui_job",
            path=f"artifacts/ui_jobs/{job_id}/job.json",
            classification=classification,
            recorded_dataset_id=str(dataset_id),
            source="job.references.dataset_id",
            config_fallback=config_fallback,
            n_features=None,
            schema_hash=None,
            train_content_hash=None,
            target_hash=None,
            train_row_identity_hash=None,
            canonical_registry_equivalent=None,
            missing_evidence=tuple(missing),
            diagnostic=None,
        )
    if config_fallback:
        return IdentityAuditRecord(
            kind="ui_job",
            path=f"artifacts/ui_jobs/{job_id}/job.json",
            classification="resolved_through_legacy_fallback",
            recorded_dataset_id=None,
            source="job.references.config",
            config_fallback=config_fallback,
            n_features=None,
            schema_hash=None,
            train_content_hash=None,
            target_hash=None,
            train_row_identity_hash=None,
            canonical_registry_equivalent=None,
            missing_evidence=("dataset_id_reference",),
            diagnostic="Legacy job without persisted dataset_id.",
        )
    return IdentityAuditRecord(
        kind="ui_job",
        path=f"artifacts/ui_jobs/{job_id}/job.json",
        classification="missing",
        recorded_dataset_id=None,
        source=None,
        config_fallback=None,
        n_features=None,
        schema_hash=None,
        train_content_hash=None,
        target_hash=None,
        train_row_identity_hash=None,
        canonical_registry_equivalent=None,
        missing_evidence=("dataset_id_reference", "config_reference"),
        diagnostic="No dataset identity persisted on job.",
    )


def _audit_artifact(*, kind: str, path: str, run_root: Path) -> IdentityAuditRecord:
    try:
        identity = read_dataset_identity(run_root)
    except DatasetIdentityConflictError as error:
        return IdentityAuditRecord(
            kind=kind,
            path=path,
            classification="conflicting",
            recorded_dataset_id=None,
            source="conflict",
            config_fallback=None,
            n_features=None,
            schema_hash=None,
            train_content_hash=None,
            target_hash=None,
            train_row_identity_hash=None,
            canonical_registry_equivalent=None,
            missing_evidence=("consistent_run_local_identity",),
            diagnostic=str(error),
        )
    except (
        DatasetIdentityUnsafeError,
        DatasetIdentityMalformedError,
        PathSafetyError,
    ) as error:
        return IdentityAuditRecord(
            kind=kind,
            path=path,
            classification="unsafe_unreadable",
            recorded_dataset_id=None,
            source="unsafe",
            config_fallback=None,
            n_features=None,
            schema_hash=None,
            train_content_hash=None,
            target_hash=None,
            train_row_identity_hash=None,
            canonical_registry_equivalent=None,
            missing_evidence=("safe_metadata",),
            diagnostic=str(error),
        )

    missing = _missing_hash_evidence(identity)
    if identity.dataset_id is None:
        classification = "missing"
    elif identity.source == "dataset_provenance" and not missing:
        classification = "persisted_and_complete"
    elif identity.source == "dataset_provenance":
        classification = "persisted_but_partial"
    elif identity.source in {"resolved_config", "fingerprints"}:
        classification = "resolved_through_legacy_fallback"
    else:
        classification = "missing"

    return IdentityAuditRecord(
        kind=kind,
        path=path,
        classification=classification,
        recorded_dataset_id=identity.dataset_id,
        source=identity.source,
        config_fallback=None,
        n_features=identity.n_features,
        schema_hash=identity.schema_hash,
        train_content_hash=identity.train_content_hash,
        target_hash=identity.target_hash,
        train_row_identity_hash=identity.train_row_identity_hash,
        canonical_registry_equivalent=None,
        missing_evidence=tuple(missing),
        diagnostic=identity.diagnostic,
    )


def _missing_hash_evidence(identity: Any) -> list[str]:
    missing: list[str] = []
    for field in (
        "n_features",
        "schema_hash",
        "train_content_hash",
        "target_hash",
        "train_row_identity_hash",
    ):
        if getattr(identity, field, None) in (None, ""):
            missing.append(field)
    return missing


def render_audit_text(report: IdentityAuditReport) -> str:
    lines = ["Dataset identity audit (read-only)", ""]
    counts = report.counts()
    for key in CLASSIFICATIONS:
        lines.append(f"{key}: {counts.get(key, 0)}")
    lines.append("")
    lines.append("Proposed legacy->canonical mappings: none (equivalence unproven).")
    lines.append("No historical records were modified.")
    return "\n".join(lines) + "\n"


def write_audit_json(report: IdentityAuditReport, path: Path) -> None:
    path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
