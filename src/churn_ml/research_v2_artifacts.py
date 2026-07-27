from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import yaml

from src.churn_ml.research_evaluation import (
    CompletedOuterFold,
    ResearchEvaluationResult,
)
from src.churn_ml.research_protocol import EvaluationAssignments
from src.churn_ml.research_v2_artifact_validation import (
    build_artifact_manifest,
    validate_artifact_manifest,
    validate_research_v2_run,
)
from src.churn_ml.research_v2_config import ResearchV2Config


_SAFE_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


class ResearchV2ArtifactError(RuntimeError):
    """Raised when a v2 run cannot satisfy its artifact lifecycle."""


class ResearchV2ArtifactStore:
    def __init__(
        self,
        config: ResearchV2Config,
        *,
        plan_hash: str,
        candidate_hash: str,
        expected_hashes: Mapping[str, str] | None = None,
        run_id: str | None,
    ) -> None:
        self.config = config
        self.plan_hash = plan_hash
        self.candidate_hash = candidate_hash
        self.expected_hashes = dict(expected_hashes or {})
        self.started_at = datetime.now(timezone.utc)
        self.expected_folds = len(
            config.plan_payload["outer_evaluation"]["repeat_seeds"]
        ) * int(config.plan_payload["outer_evaluation"]["n_splits"])
        self.completed_folds: set[tuple[int, int]] = set()
        self.run_id = run_id or (
            self.started_at.strftime("%Y%m%dT%H%M%S%fZ") + f"_{candidate_hash[:8]}"
        )
        values = (
            ("experiment_id", config.experiment_id),
            ("plan_id", config.plan_id),
            ("pipeline_id", config.pipeline_id),
            ("adapter_id", config.adapter_id),
            ("run_id", self.run_id),
        )
        for label, value in values:
            if _SAFE_SLUG.fullmatch(value) is None:
                raise ResearchV2ArtifactError(f"{label} is not a safe slug.")
        root = config.artifact_root.resolve()
        self.root = (
            root
            / f"{config.plan_id}_{plan_hash[:12]}"
            / f"{config.pipeline_id}__{config.adapter_id}"
            / self.run_id
        ).resolve()
        if root not in self.root.parents:
            raise ResearchV2ArtifactError("Run directory escapes artifact root.")
        self.root.mkdir(parents=True, exist_ok=False)
        try:
            for directory in (
                "identities",
                "splits",
                "predictions",
                "thresholds",
                "metrics",
                "fold_progress",
            ):
                (self.root / directory).mkdir()
            self.write_yaml("resolved_config.yaml", config.resolved_payload())
            self.write_json("execution_status.json", self._status("running"))
        except BaseException as error:
            self.fail(error, {})
            raise

    def save_initial(
        self,
        *,
        metadata: dict[str, Any],
        identities: dict[str, dict[str, Any]],
        fingerprints: dict[str, Any],
        feature_schema: dict[str, Any],
        assignments: EvaluationAssignments,
    ) -> None:
        self.write_json("run_metadata.json", metadata)
        for name, payload in identities.items():
            self.write_json(f"identities/{name}.json", payload)
        self.write_json("dataset_fingerprints.json", fingerprints)
        self.write_json("feature_schema.json", feature_schema)
        assignments.outer.to_parquet(
            self.root / "splits/outer_assignments.parquet", index=False
        )
        assignments.threshold_selection.to_parquet(
            self.root / "splits/threshold_selection_assignments.parquet",
            index=False,
        )

    def save_fold(self, fold: CompletedOuterFold) -> None:
        key = (fold.repeat, fold.outer_fold)
        if key in self.completed_folds:
            raise ResearchV2ArtifactError(f"Fold already persisted: {key}.")
        directory = (
            self.root
            / "fold_progress"
            / f"repeat_{fold.repeat:02d}_outer_fold_{fold.outer_fold:02d}"
        )
        directory.mkdir()
        fold.threshold_selection_oof.to_parquet(
            directory / "threshold_selection_oof.parquet", index=False
        )
        fold.outer_validation.to_parquet(
            directory / "outer_validation.parquet", index=False
        )
        _write_json(directory / "selected_threshold.json", fold.selected_threshold)
        _write_json(directory / "fold_metrics.json", fold.fold_metrics)
        if self.config.payload["persistence"]["threshold_curves"]:
            fold.threshold_curve.to_parquet(
                directory / "threshold_curve.parquet", index=False
            )
        self.completed_folds.add(key)
        self.write_json("execution_status.json", self._status("running"))

    def save_result(self, result: ResearchEvaluationResult) -> None:
        result.threshold_selection_oof.to_parquet(
            self.root / "predictions/threshold_selection_oof.parquet", index=False
        )
        result.outer_validation.to_parquet(
            self.root / "predictions/outer_validation.parquet", index=False
        )
        result.selected_thresholds.to_csv(
            self.root / "thresholds/selected_thresholds.csv", index=False
        )
        self.write_json("thresholds/threshold_summary.json", result.threshold_summary)
        if self.config.payload["persistence"]["threshold_curves"]:
            result.threshold_curves.to_parquet(
                self.root / "thresholds/threshold_curves.parquet", index=False
            )
        result.outer_fold_metrics.to_csv(
            self.root / "metrics/outer_folds.csv", index=False
        )
        result.repeat_metrics.to_csv(self.root / "metrics/repeats.csv", index=False)
        self.write_json("metrics/aggregate.json", result.aggregate_metrics)

    def complete(
        self,
        metadata: dict[str, Any],
        result: ResearchEvaluationResult,
        pre_success_output: Callable[[], None] | None = None,
    ) -> None:
        if len(self.completed_folds) != self.expected_folds:
            raise ResearchV2ArtifactError("Not all outer folds were persisted.")
        if (self.root / "_FAILED").exists() or (self.root / "_SUCCESS").exists():
            raise ResearchV2ArtifactError("Terminal marker already exists.")
        if not self.expected_hashes:
            raise ResearchV2ArtifactError(
                "Complete identity hashes are required for terminal success."
            )
        finished = datetime.now(timezone.utc)
        completed_metadata = {
            **metadata,
            "status": "completed",
            "finished_at_utc": finished.isoformat(),
            "evaluation_duration_seconds": result.duration_seconds,
        }
        self.write_json("run_metadata.json", completed_metadata)
        self.write_json(
            "execution_status.json",
            self._status("completed", finished_at=finished),
        )
        validate_research_v2_run(
            self.root,
            self.config,
            expected_hashes=self.expected_hashes,
            require_success=False,
            verify_manifest=False,
        )
        manifest = build_artifact_manifest(self.root)
        self.write_json("artifact_manifest.json", manifest)
        validate_artifact_manifest(self.root, manifest)
        if pre_success_output is not None:
            self._run_pre_success_output(pre_success_output)
        self.write_json(
            "_SUCCESS",
            {
                "schema_version": 2,
                "artifact_manifest_sha256": manifest["manifest_sha256"],
            },
        )

    def _run_pre_success_output(self, output: Callable[[], None]) -> None:
        output()

    def fail(self, error: BaseException, metadata: dict[str, Any]) -> None:
        if (self.root / "_SUCCESS").exists():
            raise ResearchV2ArtifactError("Successful run cannot be failed.")
        finished = datetime.now(timezone.utc)
        failure = {"type": type(error).__name__, "message": str(error)}
        for action in (
            lambda: self.write_json(
                "run_metadata.json",
                {
                    **metadata,
                    "status": "failed",
                    "finished_at_utc": finished.isoformat(),
                    "failure": failure,
                },
            ),
            lambda: self.write_json(
                "execution_status.json",
                self._status("failed", finished_at=finished, failure=failure),
            ),
            lambda: (self.root / "_FAILED").touch(exist_ok=True),
        ):
            try:
                action()
            except BaseException:
                pass

    def write_json(self, relative: str, payload: dict[str, Any]) -> None:
        _write_json(self.root / relative, payload)

    def write_yaml(self, relative: str, payload: dict[str, Any]) -> None:
        path = self.root / relative
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=True)
        temporary.replace(path)

    def _status(
        self,
        status: str,
        *,
        finished_at: datetime | None = None,
        failure: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "run_id": self.run_id,
            "status": status,
            "started_at_utc": self.started_at.isoformat(),
            "finished_at_utc": (
                None if finished_at is None else finished_at.isoformat()
            ),
            "completed_outer_folds": len(self.completed_folds),
            "expected_outer_folds": self.expected_folds,
            "failure": failure,
        }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, default=_json_default)
        file.write("\n")
    temporary.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}.")
