"""Persistence helpers for Blend Campaign Runner v1 artifacts."""

from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from src.churn_ml.research_data import canonical_sha256


CAMPAIGN_ROOT_RELATIVE = "artifacts/blend_campaigns"
FROZEN_CONFIG = "frozen_config.yaml"
PLAN = "campaign_plan.json"
STATUS = "campaign_status.json"
RESULTS_CSV = "experiment_results.csv"
RESULTS_JSON = "experiment_results.json"
FAILURES = "failures.json"
RANKING = "ranking.csv"
SELECTED = "selected_for_materialization.json"
SUBMISSIONS = "generated_submissions.json"
ENVIRONMENT = "environment.json"
SUCCESS = "_SUCCESS"

RESULT_COLUMNS = (
    "experiment_id",
    "ordered_candidate_ids",
    "aliases",
    "candidate_manifest_hashes",
    "strategy",
    "optimizer",
    "folds",
    "repeats",
    "seed",
    "max_active_models",
    "threshold_policy",
    "expected_blend_id",
    "status",
    "runtime_seconds",
    "honest_mean_balanced_accuracy",
    "honest_balanced_accuracy_std",
    "min_repeat_balanced_accuracy",
    "max_repeat_balanced_accuracy",
    "pooled_sensitivity",
    "pooled_specificity",
    "descriptive_full_oof_balanced_accuracy",
    "final_weights",
    "final_threshold",
    "predicted_positive_count",
    "active_model_count",
    "exploratory",
    "failure_reason",
    "source_result_path",
)


class BlendCampaignArtifactError(RuntimeError):
    """Raised for campaign artifact conflicts or corruption."""


def campaign_dir(repository_root: Path, campaign_id: str) -> Path:
    return (repository_root / CAMPAIGN_ROOT_RELATIVE / campaign_id).resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        (json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode(),
    )


def atomic_write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(path, yaml.safe_dump(dict(payload), sort_keys=False).encode("utf-8"))


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_sha256(existing) != canonical_sha256(dict(payload)):
            raise BlendCampaignArtifactError(
                f"Refusing to overwrite immutable file: {path}"
            )
        return
    atomic_write_json(path, payload)


def write_immutable_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        existing = yaml.safe_load(path.read_text(encoding="utf-8"))
        if canonical_sha256(existing) != canonical_sha256(dict(payload)):
            raise BlendCampaignArtifactError(
                f"Refusing to overwrite immutable file: {path}"
            )
        return
    atomic_write_yaml(path, payload)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BlendCampaignArtifactError(f"Required campaign artifact missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BlendCampaignArtifactError(
            f"Invalid JSON artifact {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise BlendCampaignArtifactError(f"Campaign artifact must be a mapping: {path}")
    return payload


def write_result_views(directory: Path, results: Sequence[Mapping[str, Any]]) -> None:
    ordered = [dict(row) for row in results]
    atomic_write_json(
        directory / RESULTS_JSON, {"schema_version": 1, "results": ordered}
    )
    frame = pd.DataFrame(ordered, columns=RESULT_COLUMNS)
    for column in (
        "ordered_candidate_ids",
        "aliases",
        "candidate_manifest_hashes",
        "threshold_policy",
        "final_weights",
    ):
        if column in frame:
            frame[column] = frame[column].map(
                lambda value: (
                    json.dumps(value, sort_keys=True) if value is not None else ""
                )
            )
    _atomic_write(
        directory / RESULTS_CSV, frame.to_csv(index=False, lineterminator="\n").encode()
    )


def write_ranking(directory: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    frame = pd.DataFrame([dict(row) for row in rows])
    _atomic_write(
        directory / RANKING, frame.to_csv(index=False, lineterminator="\n").encode()
    )


def environment_payload() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("numpy", "pandas", "scikit-learn", "pyyaml", "optuna"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    return {
        "schema_version": 1,
        "python_version": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": packages,
        "network_access": False,
        "kaggle_upload": False,
    }


def write_success_last(directory: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(directory / SUCCESS, payload)


def remove_success_marker(directory: Path) -> None:
    (directory / SUCCESS).unlink(missing_ok=True)


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    replaced = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        replaced = True
    finally:
        if not replaced:
            temporary.unlink(missing_ok=True)


__all__ = [
    "BlendCampaignArtifactError",
    "ENVIRONMENT",
    "FAILURES",
    "FROZEN_CONFIG",
    "PLAN",
    "RESULTS_CSV",
    "RESULTS_JSON",
    "SELECTED",
    "STATUS",
    "SUBMISSIONS",
    "SUCCESS",
    "atomic_write_json",
    "campaign_dir",
    "environment_payload",
    "load_json",
    "remove_success_marker",
    "utc_now",
    "write_immutable_json",
    "write_immutable_yaml",
    "write_ranking",
    "write_result_views",
    "write_success_last",
]
