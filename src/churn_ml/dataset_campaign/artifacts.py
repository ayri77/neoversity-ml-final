"""Campaign artifact persistence: frozen manifest and mutable status."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.churn_ml.control_panel.path_safety import PathSafetyError, require_safe_directory
from src.churn_ml.dataset_campaign.constants import (
    AGGREGATE_METRIC_FIELDS,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    STATUS_FILENAME,
    STATUS_SCHEMA_VERSION,
    SUMMARY_FILENAME,
    SUMMARY_SCHEMA_VERSION,
)
from src.churn_ml.dataset_campaign.errors import CampaignManifestError
from src.churn_ml.research_data import canonical_sha256


def campaign_dir_for(
    *,
    project_root: Path,
    artifacts_root: str,
    campaign_id: str,
) -> Path:
    return (project_root / Path(*Path(artifacts_root.replace("\\", "/")).parts) / campaign_id).resolve()


def write_frozen_manifest(campaign_dir: Path, manifest: Mapping[str, Any]) -> Path:
    """Persist the immutable campaign manifest without allocating experiment runs."""
    campaign_dir.mkdir(parents=True, exist_ok=True)
    path = campaign_dir / MANIFEST_FILENAME
    payload = dict(manifest)
    if "manifest_hash" not in payload:
        raise CampaignManifestError("Frozen manifest is missing manifest_hash.")
    body = {key: value for key, value in payload.items() if key != "manifest_hash"}
    recomputed = canonical_sha256(body)
    if recomputed != payload["manifest_hash"]:
        raise CampaignManifestError(
            "Frozen manifest_hash does not match canonical body hash."
        )
    if path.exists():
        existing = load_frozen_manifest(campaign_dir)
        if existing.get("manifest_hash") != payload["manifest_hash"]:
            raise CampaignManifestError(
                "Refusing to overwrite campaign_manifest.json with a different "
                f"manifest_hash (existing={existing.get('manifest_hash')}, "
                f"new={payload['manifest_hash']})."
            )
        return path
    _write_json(path, payload)
    return path


def load_frozen_manifest(campaign_dir: Path) -> dict[str, Any]:
    path = campaign_dir / MANIFEST_FILENAME
    if not path.is_file():
        raise CampaignManifestError(
            f"Frozen campaign manifest not found: {path}."
        )
    payload = _read_json(path)
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise CampaignManifestError(
            f"Unsupported campaign manifest schema_version: "
            f"{payload.get('schema_version')!r}."
        )
    body = {key: value for key, value in payload.items() if key != "manifest_hash"}
    recomputed = canonical_sha256(body)
    if recomputed != payload.get("manifest_hash"):
        raise CampaignManifestError(
            "Campaign manifest_hash drift detected; frozen identity is corrupt."
        )
    return payload


def initial_status(manifest: Mapping[str, Any]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for cell in manifest["cells"]:
        cells[cell["cell_id"]] = {
            "cell_id": cell["cell_id"],
            "execution_order": cell["execution_order"],
            "dataset_id": cell["dataset_id"],
            "model_family": cell["model_family"],
            "state": "pending",
            "attempts": [],
        }
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "campaign_id": manifest["campaign_id"],
        "manifest_hash": manifest["manifest_hash"],
        "state": "validated",
        "updated_at_utc": _utc_now(),
        "cells": cells,
    }


def write_status(campaign_dir: Path, status: Mapping[str, Any]) -> Path:
    path = campaign_dir / STATUS_FILENAME
    campaign_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(status)
    payload["updated_at_utc"] = _utc_now()
    _write_json(path, payload)
    return path


def load_status(campaign_dir: Path) -> dict[str, Any]:
    path = campaign_dir / STATUS_FILENAME
    if not path.is_file():
        raise CampaignManifestError(f"Campaign status not found: {path}.")
    payload = _read_json(path)
    if payload.get("schema_version") != STATUS_SCHEMA_VERSION:
        raise CampaignManifestError(
            f"Unsupported campaign status schema_version: "
            f"{payload.get('schema_version')!r}."
        )
    return payload


def assert_status_matches_manifest(
    status: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    if status.get("manifest_hash") != manifest.get("manifest_hash"):
        raise CampaignManifestError(
            "Campaign status manifest_hash does not match frozen manifest; "
            "resume rejected due to identity drift."
        )
    if status.get("campaign_id") != manifest.get("campaign_id"):
        raise CampaignManifestError(
            "Campaign status campaign_id does not match frozen manifest."
        )


def build_summary(
    manifest: Mapping[str, Any],
    status: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for cell in sorted(manifest["cells"], key=lambda item: item["execution_order"]):
        cell_status = status["cells"][cell["cell_id"]]
        latest = _latest_attempt(cell_status)
        metrics = (latest or {}).get("metrics") or {}
        threshold = (latest or {}).get("threshold_summary") or {}
        row = {
            "cell_id": cell["cell_id"],
            "execution_order": cell["execution_order"],
            "dataset_id": cell["dataset_id"],
            "parent_dataset_id": cell.get("parent_dataset_id"),
            "target_dependency": cell.get("target_dependency"),
            "model_family": cell["model_family"],
            "adapter_id": cell["adapter_id"],
            "protocol_sha256": cell["protocol_sha256"],
            "state": cell_status.get("state"),
            "run_directory": (latest or {}).get("run_directory"),
            "attempt_count": len(cell_status.get("attempts") or []),
            "balanced_accuracy": _metric_mean(metrics, "balanced_accuracy"),
            "sensitivity": _metric_mean(metrics, "sensitivity"),
            "specificity": _metric_mean(metrics, "specificity"),
            "roc_auc": _metric_mean(metrics, "roc_auc"),
            "average_precision": _metric_mean(metrics, "average_precision"),
            "brier_score": _metric_mean(metrics, "brier_score"),
            "threshold_summary": threshold,
            "package_hashes": {
                "schema_hash": cell["package"]["schema_hash"],
                "train_content_hash": cell["package"]["train_content_hash"],
                "target_hash": cell["package"]["target_hash"],
                "train_row_identity_hash": cell["package"]["train_row_identity_hash"],
                "dataset_manifest_sha256": cell["package"]["dataset_manifest_sha256"],
            },
            "oof_relative": (latest or {}).get("oof_relative"),
            "error": (latest or {}).get("error"),
            "mlflow_index_status": (latest or {}).get("mlflow_index_status"),
        }
        rows.append(row)
    counts = {
        "pending": 0,
        "running": 0,
        "succeeded": 0,
        "failed": 0,
        "skipped": 0,
    }
    for cell_status in status["cells"].values():
        state = str(cell_status.get("state"))
        if state in counts:
            counts[state] += 1
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "campaign_id": manifest["campaign_id"],
        "manifest_hash": manifest["manifest_hash"],
        "campaign_state": status.get("state"),
        "classification": manifest.get("classification"),
        "campaign_type": manifest.get("campaign_type"),
        "cell_count": len(rows),
        "counts": counts,
        "rows": rows,
        "note": (
            "Descriptive campaign aggregation only. Official cross-dataset "
            "paired inference and winner selection belong to Stage E."
        ),
    }


def write_summary(campaign_dir: Path, summary: Mapping[str, Any]) -> Path:
    path = campaign_dir / SUMMARY_FILENAME
    _write_json(path, dict(summary))
    return path


def require_campaign_dir(path: Path) -> Path:
    try:
        return require_safe_directory(path.resolve())
    except PathSafetyError as error:
        raise CampaignManifestError(str(error)) from error


def _latest_attempt(cell_status: Mapping[str, Any]) -> dict[str, Any] | None:
    attempts = cell_status.get("attempts") or []
    if not attempts:
        return None
    return dict(attempts[-1])


def _metric_mean(metrics: Mapping[str, Any], name: str) -> float | None:
    if name not in AGGREGATE_METRIC_FIELDS:
        return None
    block = metrics.get(name)
    if isinstance(block, Mapping) and "mean" in block:
        return float(block["mean"])
    if isinstance(block, (int, float)):
        return float(block)
    return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(deepcopy(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignManifestError(
            f"Could not read campaign JSON artifact {path.name}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise CampaignManifestError(f"Campaign artifact must be a mapping: {path}.")
    return payload
