"""Validate-only and sequential campaign execution."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.churn_ml.dataset_campaign.artifacts import (
    assert_status_matches_manifest,
    build_summary,
    campaign_dir_for,
    initial_status,
    load_frozen_manifest,
    load_status,
    require_campaign_dir,
    write_frozen_manifest,
    write_status,
    write_summary,
)
from src.churn_ml.dataset_campaign.errors import (
    CampaignExecutionError,
    CampaignManifestError,
    CampaignValidationError,
)
from src.churn_ml.dataset_campaign.resolve import ResolvedCampaign, resolve_campaign
from src.churn_ml.dataset_campaign.runner import (
    CellRunner,
    CellRunResult,
    ResearchV2CellRunner,
    maybe_index_run_directory,
)
from src.churn_ml.dataset_campaign.schema import CampaignSpec


def validate_only(
    spec: CampaignSpec,
    *,
    project_root: Path,
    freeze: bool = False,
    runner: CellRunner | None = None,
    validate_prepared: bool = True,
) -> ResolvedCampaign:
    """Resolve and validate the full matrix without Experiment Core run allocation.

    When ``freeze`` is True, persists the immutable campaign manifest and an
    initial mutable status file under the campaign artifact directory. Prepared
    config/plan pairs may be written under that campaign directory; Research v2
    run directories and job records are never created.
    """
    root = project_root.resolve()
    campaign_dir = campaign_dir_for(
        project_root=root,
        artifacts_root=spec.artifacts_root,
        campaign_id=spec.campaign_id,
    )
    resolved = resolve_campaign(
        spec,
        project_root=root,
        campaign_dir=campaign_dir,
        materialize=True,
        validate_prepared=validate_prepared,
    )
    cell_runner = runner or ResearchV2CellRunner()
    failures: list[str] = []
    for cell in resolved.manifest["cells"]:
        cell_id = cell["cell_id"]
        config_path = resolved.prepared_configs.get(cell_id)
        if config_path is None:
            config_rel = cell.get("prepared_config_relative")
            config_path = Path(config_rel) if config_rel else None
        if config_path is None:
            failures.append(f"Cell {cell_id} missing prepared config.")
            continue
        try:
            cell_runner.validate(config_path, project_root=root)
        except Exception as error:  # noqa: BLE001
            failures.append(
                f"Cell {cell['dataset_id']} × {cell['model_family']}: {error}"
            )
    if failures:
        raise CampaignValidationError(
            "Validate-only failed with "
            f"{len(failures)} error(s):\n- " + "\n- ".join(failures),
            failures=failures,
        )

    if freeze:
        write_frozen_manifest(campaign_dir, resolved.manifest)
        if not (campaign_dir / "campaign_status.json").exists():
            write_status(campaign_dir, initial_status(resolved.manifest))
        write_summary(
            campaign_dir,
            build_summary(resolved.manifest, load_status(campaign_dir)),
        )
    return resolved


def run_campaign(
    *,
    project_root: Path,
    spec: CampaignSpec | None = None,
    campaign_dir: Path | None = None,
    runner: CellRunner | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute or resume a campaign sequentially from a frozen manifest."""
    root = project_root.resolve()
    cell_runner = runner or ResearchV2CellRunner()

    if campaign_dir is not None:
        directory = require_campaign_dir(campaign_dir)
        manifest = load_frozen_manifest(directory)
        if resume:
            status = load_status(directory)
            assert_status_matches_manifest(status, manifest)
        else:
            if (directory / "campaign_status.json").exists():
                status = load_status(directory)
                assert_status_matches_manifest(status, manifest)
            else:
                status = initial_status(manifest)
                write_status(directory, status)
    else:
        if spec is None:
            raise CampaignExecutionError(
                "run_campaign requires either campaign_dir or spec."
            )
        directory = campaign_dir_for(
            project_root=root,
            artifacts_root=spec.artifacts_root,
            campaign_id=spec.campaign_id,
        )
        if (directory / "campaign_manifest.json").exists():
            manifest = load_frozen_manifest(directory)
            # Reject drift against the provided spec identity.
            resolved = resolve_campaign(
                spec,
                project_root=root,
                campaign_dir=directory,
                materialize=True,
                validate_prepared=True,
            )
            if resolved.manifest_hash != manifest["manifest_hash"]:
                raise CampaignManifestError(
                    "Provided campaign specification drifted from the frozen "
                    f"manifest (frozen={manifest['manifest_hash']}, "
                    f"resolved={resolved.manifest_hash})."
                )
        else:
            resolved = resolve_campaign(
                spec,
                project_root=root,
                campaign_dir=directory,
                materialize=True,
                validate_prepared=True,
            )
            write_frozen_manifest(directory, resolved.manifest)
            manifest = resolved.manifest
        if (directory / "campaign_status.json").exists():
            status = load_status(directory)
            assert_status_matches_manifest(status, manifest)
        else:
            status = initial_status(manifest)
            write_status(directory, status)

    status["state"] = "running"
    write_status(directory, status)

    cells = sorted(manifest["cells"], key=lambda item: int(item["execution_order"]))
    for cell in cells:
        cell_id = cell["cell_id"]
        cell_status = status["cells"][cell_id]
        if cell_status.get("state") == "succeeded":
            # Skip only cells already completed under the exact same identity.
            cell_status["state"] = "skipped"
            # Keep succeeded semantics for aggregation: restore succeeded marker.
            cell_status["state"] = "succeeded"
            continue

        config_rel = cell.get("prepared_config_relative")
        if not config_rel:
            raise CampaignExecutionError(
                f"Frozen cell {cell_id} is missing prepared_config_relative."
            )

        attempt_id = _next_attempt_id(cell_status)
        attempt: dict[str, Any] = {
            "attempt_id": attempt_id,
            "started_at_utc": _utc_now(),
            "finished_at_utc": None,
            "run_directory": None,
            "metrics": None,
            "threshold_summary": None,
            "oof_relative": None,
            "error": None,
            "mlflow_index_status": None,
        }
        cell_status["state"] = "running"
        cell_status["attempts"] = list(cell_status.get("attempts") or []) + [attempt]
        write_status(directory, status)

        result = cell_runner.run(
            Path(config_rel),
            project_root=root,
            run_id=f"{cell_id}_{attempt_id}",
        )
        _apply_attempt_result(
            attempt,
            result,
            project_root=root,
            index_mlflow=bool(manifest.get("index_mlflow")),
            mlflow_config_path=str(
                manifest.get("mlflow_config_path") or "configs/mlflow/local.yaml"
            ),
        )
        cell_status["state"] = "succeeded" if result.success else "failed"
        cell_status["attempts"][-1] = attempt
        write_status(directory, status)

    status["state"] = _aggregate_campaign_state(status)
    write_status(directory, status)
    summary = build_summary(manifest, status)
    write_summary(directory, summary)
    return {
        "campaign_dir": str(directory),
        "manifest_hash": manifest["manifest_hash"],
        "status": deepcopy(status),
        "summary": summary,
    }


def inspect_campaign(campaign_dir: Path) -> dict[str, Any]:
    directory = require_campaign_dir(campaign_dir)
    manifest = load_frozen_manifest(directory)
    status_path = directory / "campaign_status.json"
    status = load_status(directory) if status_path.is_file() else None
    summary = (
        build_summary(manifest, status)
        if status is not None
        else {
            "schema_version": "dataset_campaign_summary_v1",
            "campaign_id": manifest["campaign_id"],
            "manifest_hash": manifest["manifest_hash"],
            "campaign_state": "frozen",
            "cell_count": manifest.get("cell_count"),
            "rows": [],
            "note": "Frozen manifest only; no mutable status yet.",
        }
    )
    return {
        "campaign_dir": str(directory),
        "manifest": manifest,
        "status": status,
        "summary": summary,
    }


def _apply_attempt_result(
    attempt: dict[str, Any],
    result: CellRunResult,
    *,
    project_root: Path,
    index_mlflow: bool,
    mlflow_config_path: str,
) -> None:
    attempt["finished_at_utc"] = _utc_now()
    attempt["run_directory"] = result.run_directory
    attempt["metrics"] = result.metrics
    attempt["threshold_summary"] = result.threshold_summary
    attempt["oof_relative"] = result.oof_relative
    attempt["error"] = result.error
    if result.success and index_mlflow and result.run_directory:
        run_dir = Path(result.run_directory)
        if not run_dir.is_absolute():
            run_dir = project_root / run_dir
        try:
            attempt["mlflow_index_status"] = maybe_index_run_directory(
                run_directory=run_dir,
                project_root=project_root,
                mlflow_config_path=mlflow_config_path,
            )
        except Exception as error:  # noqa: BLE001
            attempt["mlflow_index_status"] = (
                f"failed:{type(error).__name__}:{error}"
            )
        # Training success is preserved regardless of indexing outcome.
    else:
        attempt["mlflow_index_status"] = result.mlflow_index_status


def _next_attempt_id(cell_status: dict[str, Any]) -> str:
    prior = len(cell_status.get("attempts") or [])
    return f"attempt_{prior + 1:02d}"


def _aggregate_campaign_state(status: dict[str, Any]) -> str:
    states = [str(cell.get("state")) for cell in status["cells"].values()]
    if states and all(state == "succeeded" for state in states):
        return "completed"
    if any(state == "failed" for state in states) and any(
        state == "succeeded" for state in states
    ):
        return "partial"
    if any(state == "failed" for state in states):
        return "failed"
    if any(state == "pending" for state in states):
        return "partial"
    return "completed"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
