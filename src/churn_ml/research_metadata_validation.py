from __future__ import annotations

import math
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.churn_ml.research_config import ResearchConfig
from src.churn_ml.research_evaluation import ResearchEvaluationResult
from src.churn_ml.research_runtime_provenance import REQUIRED_RESEARCH_DISTRIBUTIONS


FINAL_METADATA_KEYS = {
    "experiment_id",
    "candidate_id",
    "plan_id",
    "plan_sha256",
    "candidate_sha256",
    "candidate_source_manifest_sha256",
    "run_implementation_sha256",
    "loaded_module_sha256",
    "status",
    "started_at_utc",
    "finished_at_utc",
    "duration_seconds",
    "evaluation_duration_seconds",
    "run_id",
    "config_source_path",
    "plan_source_path",
    "environment",
    "invocation",
    "git",
    "upstream_feature_selection_limitation",
    "competition_assets_accessed",
    "record_counts",
}
STATUS_KEYS = {
    "run_id",
    "status",
    "started_at_utc",
    "finished_at_utc",
    "completed_outer_folds",
    "expected_outer_folds",
    "failure",
}
INVOCATION_KEYS = {
    "python_executable",
    "python_version",
    "sys_argv",
    "entry_point",
    "current_working_directory",
    "process_id",
    "process_started_at_utc",
}
GIT_KEYS = {
    "branch",
    "commit",
    "dirty",
    "status_porcelain",
    "untracked_implementation_files",
}
RECORD_COUNT_KEYS = {
    "repeat_count",
    "outer_fold_count",
    "outer_assignment_count",
    "outer_prediction_count",
    "threshold_assignment_count",
    "threshold_prediction_count",
    "selected_threshold_count",
}


class FinalMetadataValidationError(RuntimeError):
    """Raised when terminal metadata cannot prove a successful run."""


def build_record_counts(
    config: ResearchConfig,
    result: ResearchEvaluationResult,
) -> dict[str, int]:
    repeats = len(config.plan_payload["outer_evaluation"]["repeat_seeds"])
    rows = int(config.plan_payload["dataset"]["expected_rows"])
    folds = int(config.plan_payload["outer_evaluation"]["n_splits"])
    return {
        "repeat_count": len(result.repeat_metrics),
        "outer_fold_count": len(result.outer_fold_metrics),
        "outer_assignment_count": repeats * rows,
        "outer_prediction_count": len(result.outer_validation),
        "threshold_assignment_count": repeats * rows * (folds - 1),
        "threshold_prediction_count": len(result.threshold_selection_oof),
        "selected_threshold_count": len(result.selected_thresholds),
    }


def validate_final_metadata_status(
    metadata: Mapping[str, Any],
    status: Mapping[str, Any],
    *,
    root: Path,
    config: ResearchConfig,
    result: ResearchEvaluationResult,
    run_id: str,
    started_at: datetime,
    finished_at: datetime,
    plan_hash: str,
    candidate_hash: str,
    candidate_source_manifest_hash: str,
    run_implementation_hash: str,
    loaded_module_hash: str,
    completed_outer_folds: int,
    expected_outer_folds: int,
) -> None:
    _exact_keys(metadata, FINAL_METADATA_KEYS, "final run metadata")
    _exact_keys(status, STATUS_KEYS, "final execution status")
    effective_plan = f"{config.plan_id}_{plan_hash[:12]}"
    expected_root = (
        config.artifact_root.resolve() / effective_plan / config.candidate_id / run_id
    ).resolve()
    if root.resolve() != expected_root or root.name != run_id:
        raise FinalMetadataValidationError("Run directory identity differs.")
    expected_values = {
        "experiment_id": config.experiment_id,
        "candidate_id": config.candidate_id,
        "plan_id": config.plan_id,
        "plan_sha256": plan_hash,
        "candidate_sha256": candidate_hash,
        "candidate_source_manifest_sha256": candidate_source_manifest_hash,
        "run_implementation_sha256": run_implementation_hash,
        "loaded_module_sha256": loaded_module_hash,
        "status": "completed",
        "run_id": run_id,
        "config_source_path": str(config.source_path),
        "plan_source_path": str(config.plan_path),
        "competition_assets_accessed": False,
    }
    for key, expected in expected_values.items():
        if metadata[key] != expected or type(metadata[key]) is not type(expected):
            raise FinalMetadataValidationError(f"Metadata {key} differs.")
    if not isinstance(metadata["upstream_feature_selection_limitation"], str):
        raise FinalMetadataValidationError("Limitation must be a string.")
    parsed_start = _parse_utc(metadata["started_at_utc"], "started_at_utc")
    parsed_finish = _parse_utc(metadata["finished_at_utc"], "finished_at_utc")
    if parsed_start != started_at or parsed_finish != finished_at:
        raise FinalMetadataValidationError("Lifecycle timestamps differ.")
    if parsed_finish < parsed_start:
        raise FinalMetadataValidationError("Run finished before it started.")
    duration = (parsed_finish - parsed_start).total_seconds()
    _finite_nonnegative(metadata["duration_seconds"], "duration_seconds")
    _finite_nonnegative(
        metadata["evaluation_duration_seconds"],
        "evaluation_duration_seconds",
    )
    if not math.isclose(
        float(metadata["duration_seconds"]), duration, rel_tol=0.0, abs_tol=1e-6
    ):
        raise FinalMetadataValidationError("Lifecycle duration differs.")
    evaluation_duration = float(metadata["evaluation_duration_seconds"])
    if (
        not math.isclose(
            evaluation_duration,
            float(result.duration_seconds),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or duration + 1e-6 < evaluation_duration
    ):
        raise FinalMetadataValidationError("Evaluation duration differs.")
    _exact_keys(metadata["record_counts"], RECORD_COUNT_KEYS, "record counts")
    if dict(metadata["record_counts"]) != build_record_counts(config, result):
        raise FinalMetadataValidationError("Record counts differ.")
    environment = metadata["environment"]
    _exact_keys(
        environment,
        {"python", *REQUIRED_RESEARCH_DISTRIBUTIONS},
        "environment",
    )
    if not all(
        isinstance(value, str) and value.strip() for value in environment.values()
    ):
        raise FinalMetadataValidationError("Environment versions must be strings.")
    invocation = metadata["invocation"]
    _exact_keys(invocation, INVOCATION_KEYS, "invocation")
    for key in (
        "python_executable",
        "python_version",
        "entry_point",
        "current_working_directory",
    ):
        if not isinstance(invocation[key], str) or not invocation[key]:
            raise FinalMetadataValidationError(f"Invocation {key} is invalid.")
    if (
        not isinstance(invocation["sys_argv"], list)
        or not invocation["sys_argv"]
        or not all(isinstance(item, str) for item in invocation["sys_argv"])
    ):
        raise FinalMetadataValidationError("Invocation sys_argv is invalid.")
    if (
        not isinstance(invocation["process_id"], int)
        or isinstance(invocation["process_id"], bool)
        or invocation["process_id"] <= 0
    ):
        raise FinalMetadataValidationError("Invocation process_id is invalid.")
    if (
        _parse_utc(
            invocation["process_started_at_utc"],
            "invocation.process_started_at_utc",
        )
        > parsed_start
    ):
        raise FinalMetadataValidationError("Process start follows run start.")
    git = metadata["git"]
    _exact_keys(git, GIT_KEYS, "git provenance")
    if not isinstance(git["branch"], str):
        raise FinalMetadataValidationError("Git branch is invalid.")
    if (
        not isinstance(git["commit"], str)
        or re.fullmatch(r"[0-9a-f]{40}", git["commit"]) is None
    ):
        raise FinalMetadataValidationError("Git commit is invalid.")
    if type(git["dirty"]) is not bool:
        raise FinalMetadataValidationError("Git dirty is invalid.")
    for key in ("status_porcelain", "untracked_implementation_files"):
        if not isinstance(git[key], list) or not all(
            isinstance(item, str) for item in git[key]
        ):
            raise FinalMetadataValidationError(f"Git {key} is invalid.")
    expected_status = {
        "run_id": run_id,
        "status": "completed",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "completed_outer_folds": completed_outer_folds,
        "expected_outer_folds": expected_outer_folds,
        "failure": None,
    }
    if dict(status) != expected_status or completed_outer_folds != expected_outer_folds:
        raise FinalMetadataValidationError("Final execution status differs.")
    _all_utc(metadata, "metadata")
    _all_utc(status, "status")


def _exact_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise FinalMetadataValidationError(
            f"{label} keys differ: expected={sorted(expected)}, actual={actual}"
        )


def _finite_nonnegative(value: Any, label: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise FinalMetadataValidationError(f"{label} is invalid.")


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not (
        value.endswith("+00:00") or value.endswith("Z")
    ):
        raise FinalMetadataValidationError(f"{label} lacks explicit UTC.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise FinalMetadataValidationError(f"{label} is invalid.") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise FinalMetadataValidationError(f"{label} is not UTC.")
    return parsed


def _all_utc(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            item_path = f"{path}.{key}"
            if key.endswith("_utc"):
                _parse_utc(item, item_path)
            else:
                _all_utc(item, item_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _all_utc(item, f"{path}[{index}]")
