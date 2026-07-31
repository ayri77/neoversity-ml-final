"""Official Paired Comparison readiness for the Control Panel.

The lightweight ``display_compatibility_summary`` remains descriptive only.
This module is the UI gate for Prepare Paired Comparison and always delegates
compatibility authority to ``paired_comparison.build_compatibility_summary``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OfficialPairedReadiness:
    """UI-facing official readiness result (fail-closed)."""

    ready: bool
    compatible: bool
    reason_codes: tuple[str, ...]
    field_paths: tuple[str, ...]
    diagnostic: str | None
    left_dataset_version: str | None
    right_dataset_version: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "compatible": self.compatible,
            "reason_codes": list(self.reason_codes),
            "field_paths": list(self.field_paths),
            "diagnostic": self.diagnostic,
            "left_dataset_version": self.left_dataset_version,
            "right_dataset_version": self.right_dataset_version,
        }


def evaluate_official_paired_readiness(
    *,
    left_root: Path,
    right_root: Path,
    repository_root: Path,
    left_state: str,
    right_state: str,
) -> OfficialPairedReadiness:
    """Return whether the official Paired Comparison action may be prepared.

    Requires both artifacts to be completed, loads them through the validated
    completed-run loader, and calls the authoritative compatibility contract.
    Any load, safety, or comparison failure fails closed.
    """
    if left_state != "completed" or right_state != "completed":
        return OfficialPairedReadiness(
            ready=False,
            compatible=False,
            reason_codes=("RUN_NOT_COMPLETED",),
            field_paths=("artifact.state",),
            diagnostic=(
                "Official Paired Comparison requires two completed Experiment "
                "Core runs."
            ),
            left_dataset_version=None,
            right_dataset_version=None,
        )

    # Lazy import keeps Control Panel package import free of Experiment Core /
    # paired-comparison validation modules until readiness is evaluated.
    from src.churn_ml.paired_comparison import (
        PairedComparisonError,
        build_compatibility_summary,
        load_completed_research_v2_run,
    )

    try:
        baseline = load_completed_research_v2_run(
            Path(left_root),
            project_root=Path(repository_root),
            role="baseline_run",
        )
        candidate = load_completed_research_v2_run(
            Path(right_root),
            project_root=Path(repository_root),
            role="candidate_run",
        )
        summary = build_compatibility_summary(baseline, candidate)
    except PairedComparisonError as error:
        return OfficialPairedReadiness(
            ready=False,
            compatible=False,
            reason_codes=(getattr(error, "reason_code", "PAIRED_COMPARISON_INVALID"),),
            field_paths=(getattr(error, "field_path", "paths"),),
            diagnostic=str(error),
            left_dataset_version=None,
            right_dataset_version=None,
        )
    except Exception as error:  # fail closed on unexpected loader failures
        return OfficialPairedReadiness(
            ready=False,
            compatible=False,
            reason_codes=("OFFICIAL_READINESS_FAILED",),
            field_paths=("paths",),
            diagnostic=f"Official readiness failed closed: {error}",
            left_dataset_version=None,
            right_dataset_version=None,
        )

    left_version = None
    right_version = None
    try:
        left_version = str(baseline.config.dataset_version)
        right_version = str(candidate.config.dataset_version)
    except Exception:
        left_version = None
        right_version = None

    reason_codes = tuple(issue.reason_code for issue in summary.issues)
    field_paths = tuple(issue.field_path for issue in summary.issues)
    diagnostic = None
    if not summary.compatible:
        parts = [
            f"{issue.reason_code} at {issue.field_path}" for issue in summary.issues[:8]
        ]
        diagnostic = "; ".join(parts) if parts else "Officially incompatible."
    return OfficialPairedReadiness(
        ready=bool(summary.compatible),
        compatible=bool(summary.compatible),
        reason_codes=reason_codes,
        field_paths=field_paths,
        diagnostic=diagnostic,
        left_dataset_version=left_version,
        right_dataset_version=right_version,
    )
