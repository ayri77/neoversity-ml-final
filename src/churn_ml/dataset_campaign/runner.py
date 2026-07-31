"""Cell execution adapters for Dataset Campaign Runner v1.

The campaign orchestrates the existing Research v2 single-run path. It does not
duplicate training, thresholding, metrics, or artifact creation.
"""

from __future__ import annotations

import argparse
import io
import json
import re
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.churn_ml.dataset_campaign.errors import CampaignExecutionError
from src.churn_ml.dataset_campaign.oof import (
    resolve_oof_path,
    validate_outer_validation_oof,
)


_RUN_DIRECTORY_RE = re.compile(
    r"(?im)^(?:Run directory|Failed run directory):\s*(.+?)\s*$"
)


@dataclass(frozen=True)
class CellRunResult:
    success: bool
    run_directory: str | None
    metrics: dict[str, Any] | None
    threshold_summary: dict[str, Any] | None
    oof_relative: str | None
    error: str | None
    mlflow_index_status: str | None = None


class CellRunner(Protocol):
    def validate(self, config_path: Path, *, project_root: Path) -> None:
        """Validate one prepared Research v2 config without allocating a run dir."""

    def run(
        self,
        config_path: Path,
        *,
        project_root: Path,
        run_id: str | None = None,
    ) -> CellRunResult:
        """Execute one Research v2 cell."""


class ResearchV2CellRunner:
    """Default runner that delegates to Research v2 preflight/execute."""

    def validate(self, config_path: Path, *, project_root: Path) -> None:
        from src.churn_ml.research_v2_config import (
            ResearchV2ConfigurationError,
            load_research_v2_config,
        )

        abs_config = _absolute_config(config_path, project_root)
        try:
            loaded = load_research_v2_config(abs_config, project_root=project_root)
        except ResearchV2ConfigurationError as error:
            raise CampaignExecutionError(str(error)) from error
        from src.churn_ml import research_v2_cli as cli_module

        if cli_module.PROJECT_ROOT.resolve() != project_root.resolve():
            # Synthetic test repositories cannot use module-global PROJECT_ROOT.
            return
        try:
            cli_module.preflight(abs_config)
        except Exception as error:  # noqa: BLE001
            raise CampaignExecutionError(str(error)) from error
        del loaded

    def run(
        self,
        config_path: Path,
        *,
        project_root: Path,
        run_id: str | None = None,
    ) -> CellRunResult:
        from src.churn_ml.research_v2_cli import execute

        abs_config = _absolute_config(config_path, project_root)
        args = argparse.Namespace(
            config=abs_config,
            validate_only=False,
            run_id=run_id,
        )
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer), redirect_stderr(buffer):
                code = int(execute(args))
        except Exception as error:  # noqa: BLE001
            return CellRunResult(
                success=False,
                run_directory=_parse_run_directory(buffer.getvalue(), project_root),
                metrics=None,
                threshold_summary=None,
                oof_relative=None,
                error=f"{type(error).__name__}: {error}",
            )
        text = buffer.getvalue()
        run_directory_text = _parse_run_directory(text, project_root)
        run_directory = (
            Path(run_directory_text) if run_directory_text is not None else None
        )
        if run_directory is not None and not run_directory.is_absolute():
            run_directory = (project_root / run_directory).resolve()
        if code != 0:
            return CellRunResult(
                success=False,
                run_directory=_portable(run_directory, project_root)
                if run_directory
                else None,
                metrics=None,
                threshold_summary=None,
                oof_relative=None,
                error=f"Research v2 execute returned exit code {code}.",
            )
        if run_directory is None:
            return CellRunResult(
                success=False,
                run_directory=None,
                metrics=None,
                threshold_summary=None,
                oof_relative=None,
                error="Research v2 reported success but run directory was not found.",
            )
        return collect_run_result(run_directory, project_root=project_root)


def collect_run_result(run_directory: Path, *, project_root: Path) -> CellRunResult:
    success_marker = run_directory / "_SUCCESS"
    if not success_marker.is_file():
        error = "Run directory exists without _SUCCESS."
        error_path = run_directory / "execution_status.json"
        if error_path.is_file():
            try:
                payload = json.loads(error_path.read_text(encoding="utf-8"))
                error = str(payload.get("error") or payload.get("status") or error)
            except (OSError, json.JSONDecodeError):
                pass
        return CellRunResult(
            success=False,
            run_directory=_portable(run_directory, project_root),
            metrics=None,
            threshold_summary=None,
            oof_relative=None,
            error=error,
        )

    metrics_path = run_directory / "metrics" / "aggregate.json"
    threshold_path = run_directory / "thresholds" / "threshold_summary.json"
    try:
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        threshold_payload = json.loads(threshold_path.read_text(encoding="utf-8"))
        metrics = dict(metrics_payload.get("metrics") or metrics_payload)
        oof_path = resolve_oof_path(run_directory)
        validate_outer_validation_oof(oof_path)
        oof_relative = _portable(oof_path, project_root)
    except Exception as error:  # noqa: BLE001
        return CellRunResult(
            success=False,
            run_directory=_portable(run_directory, project_root),
            metrics=None,
            threshold_summary=None,
            oof_relative=None,
            error=f"Post-run artifact collection failed: {error}",
        )
    return CellRunResult(
        success=True,
        run_directory=_portable(run_directory, project_root),
        metrics=metrics,
        threshold_summary=threshold_payload,
        oof_relative=oof_relative,
        error=None,
    )


def maybe_index_run_directory(
    *,
    run_directory: Path,
    project_root: Path,
    mlflow_config_path: str,
) -> str:
    """Best-effort MLflow index. Failures never rewrite training success."""
    try:
        from src.churn_ml.mlflow_config import load_mlflow_config
        from src.churn_ml.mlflow_sources import default_source_registry
        from src.churn_ml.mlflow_sync import sync_sources

        config = load_mlflow_config(
            project_root / mlflow_config_path,
            repository_root=project_root,
        )
        summary = sync_sources(
            config,
            registry=default_source_registry(),
            source_types=("research_v2",),
            run_dir=run_directory.resolve(),
            dry_run=False,
            fail_fast=False,
        )
        if summary.has_failures:
            return "failed"
        return "succeeded"
    except Exception as error:  # noqa: BLE001
        return f"failed:{type(error).__name__}:{error}"


def _absolute_config(config_path: Path, project_root: Path) -> Path:
    if config_path.is_absolute():
        return config_path.resolve()
    return (project_root / config_path).resolve()


def _parse_run_directory(text: str, project_root: Path) -> str | None:
    matches = list(_RUN_DIRECTORY_RE.finditer(text or ""))
    if not matches:
        return None
    raw = matches[-1].group(1).strip().strip('"')
    path = Path(raw)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _portable(path: Path | None, project_root: Path) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)
