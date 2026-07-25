from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from src.churn_ml.research_artifact_validation import (
    build_artifact_manifest,
    validate_artifact_manifest,
    validate_persisted_run,
)

from src.churn_ml.research_config import ResearchConfig
from src.churn_ml.research_metadata_validation import (
    build_record_counts,
    validate_final_metadata_status,
)
from src.churn_ml.research_evaluation import (
    CompletedOuterFold,
    ResearchEvaluationResult,
)
from src.churn_ml.research_protocol import EvaluationAssignments


class ResearchArtifactError(RuntimeError):
    """Raised when the research artifact contract cannot be completed."""


class ResearchArtifactStore:
    """Own the isolated, non-overwriting lifecycle of one research run."""

    def __init__(
        self,
        config: ResearchConfig,
        *,
        plan_hash: str,
        candidate_hash: str,
        candidate_source_manifest_hash: str,
        run_id: str | None = None,
        run_implementation_hash: str,
        loaded_module_hash: str,
    ) -> None:
        self.config = config
        self.plan_hash = plan_hash
        self.candidate_hash = candidate_hash
        self.candidate_source_manifest_hash = candidate_source_manifest_hash
        self.run_implementation_hash = run_implementation_hash
        self.loaded_module_hash = loaded_module_hash
        self.started_at = datetime.now(timezone.utc)
        self.expected_folds = len(
            config.plan_payload["outer_evaluation"]["repeat_seeds"]
        ) * int(config.plan_payload["outer_evaluation"]["n_splits"])
        self.completed_folds: set[tuple[int, int]] = set()
        self.run_id = run_id or (
            self.started_at.strftime("%Y%m%dT%H%M%S%fZ") + f"_{candidate_hash[:8]}"
        )
        for label, component in (
            ("experiment_id", config.experiment_id),
            ("plan_id", config.plan_id),
            ("candidate_id", config.candidate_id),
            ("run_id", self.run_id),
        ):
            if not component or any(
                character
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                for character in component
            ):
                raise ValueError(f"{label} contains unsupported characters.")
        effective_plan = f"{config.plan_id}_{plan_hash[:12]}"
        artifact_root = config.artifact_root.resolve()
        self.root = (
            artifact_root / effective_plan / config.candidate_id / self.run_id
        ).resolve()
        if artifact_root not in self.root.parents:
            raise ValueError("Resolved run directory escapes the artifact root.")
        self.root.mkdir(parents=True, exist_ok=False)
        try:
            for directory in (
                "splits",
                "predictions",
                "thresholds",
                "metrics",
                "fold_progress",
            ):
                (self.root / directory).mkdir()
            self._write_yaml(
                self.root / "resolved_config.yaml",
                config.resolved_payload(),
            )
            self._write_status("running")
        except BaseException as error:
            self.fail(error, {"status": "initialization_failed"})
            raise

    def save_initial_contract(
        self,
        *,
        metadata: dict[str, Any],
        plan_identity: dict[str, Any],
        candidate_identity: dict[str, Any],
        run_implementation_identity: dict[str, Any],
        loaded_module_identity: dict[str, Any],
        fingerprints: dict[str, Any],
        feature_schema: dict[str, Any],
        assignments: EvaluationAssignments,
    ) -> None:
        self._write_json(self.root / "run_metadata.json", metadata)
        self._write_json(
            self.root / "evaluation_plan.json",
            {"sha256": self.plan_hash, "canonical": plan_identity},
        )
        self._write_json(
            self.root / "candidate_contract.json",
            {"sha256": self.candidate_hash, "canonical": candidate_identity},
        )
        self._write_json(
            self.root / "run_implementation.json",
            {
                "sha256": self.run_implementation_hash,
                "canonical": run_implementation_identity,
            },
        )
        self._write_json(
            self.root / "loaded_modules.json",
            {"sha256": self.loaded_module_hash, **loaded_module_identity},
        )
        self._write_json(
            self.root / "dataset_fingerprints.json",
            fingerprints,
        )
        self._write_json(self.root / "feature_schema.json", feature_schema)
        assignments.outer.to_parquet(
            self.root / "splits" / "outer_assignments.parquet",
            index=False,
        )
        assignments.threshold_selection.to_parquet(
            self.root / "splits" / "threshold_selection_assignments.parquet",
            index=False,
        )

    def save_completed_fold(self, fold: CompletedOuterFold) -> None:
        key = (fold.repeat, fold.outer_fold)
        if key in self.completed_folds:
            raise ResearchArtifactError(f"Outer fold was already saved: {key}")
        directory = (
            self.root
            / "fold_progress"
            / f"repeat_{fold.repeat:02d}_outer_fold_{fold.outer_fold:02d}"
        )
        directory.mkdir(exist_ok=False)
        fold.threshold_selection_oof.to_parquet(
            directory / "threshold_selection_oof.parquet",
            index=False,
        )
        fold.outer_validation.to_parquet(
            directory / "outer_validation.parquet",
            index=False,
        )
        self._write_json(directory / "selected_threshold.json", fold.selected_threshold)
        self._write_json(directory / "fold_metrics.json", fold.fold_metrics)
        if self.config.payload["artifacts"]["save_threshold_curves"]:
            fold.threshold_curve.to_parquet(
                directory / "threshold_curve.parquet",
                index=False,
            )
        self.completed_folds.add(key)
        self._write_status("running")

    def save_final_outputs(self, result: ResearchEvaluationResult) -> None:
        result.threshold_selection_oof.to_parquet(
            self.root / "predictions" / "threshold_selection_oof.parquet",
            index=False,
        )
        result.outer_validation.to_parquet(
            self.root / "predictions" / "outer_validation.parquet",
            index=False,
        )
        result.selected_thresholds.to_csv(
            self.root / "thresholds" / "selected_thresholds.csv",
            index=False,
        )
        if self.config.payload["artifacts"]["save_threshold_curves"]:
            result.threshold_curves.to_parquet(
                self.root / "thresholds" / "threshold_curves.parquet",
                index=False,
            )
        self._write_json(
            self.root / "thresholds" / "threshold_summary.json",
            result.threshold_summary,
        )
        result.outer_fold_metrics.to_csv(
            self.root / "metrics" / "outer_folds.csv",
            index=False,
        )
        result.repeat_metrics.to_csv(
            self.root / "metrics" / "repeats.csv",
            index=False,
        )
        self._write_json(
            self.root / "metrics" / "aggregate.json",
            result.aggregate_metrics,
        )

    def complete(
        self,
        metadata: dict[str, Any],
        result: ResearchEvaluationResult,
    ) -> None:
        self._validate_completion(result)
        validate_persisted_run(
            self.root,
            self.config,
            expected_plan_hash=self.plan_hash,
            expected_candidate_hash=self.candidate_hash,
            expected_candidate_source_manifest_hash=(
                self.candidate_source_manifest_hash
            ),
            expected_run_implementation_hash=self.run_implementation_hash,
            expected_loaded_module_hash=self.loaded_module_hash,
        )
        finished = datetime.now(timezone.utc)
        final_metadata = {
            **metadata,
            "status": "completed",
            "finished_at_utc": finished.isoformat(),
            "duration_seconds": (finished - self.started_at).total_seconds(),
            "evaluation_duration_seconds": result.duration_seconds,
            "record_counts": build_record_counts(self.config, result),
        }
        final_status = self._status_payload("completed", finished_at=finished)

        def validate_terminal_payloads(
            metadata_payload: dict[str, Any],
            status_payload: dict[str, Any],
        ) -> None:
            validate_final_metadata_status(
                metadata_payload,
                status_payload,
                root=self.root,
                config=self.config,
                result=result,
                run_id=self.run_id,
                started_at=self.started_at,
                finished_at=finished,
                plan_hash=self.plan_hash,
                candidate_hash=self.candidate_hash,
                candidate_source_manifest_hash=(self.candidate_source_manifest_hash),
                run_implementation_hash=self.run_implementation_hash,
                loaded_module_hash=self.loaded_module_hash,
                completed_outer_folds=len(self.completed_folds),
                expected_outer_folds=self.expected_folds,
            )

        validate_terminal_payloads(final_metadata, final_status)
        self._write_json(self.root / "run_metadata.json", final_metadata)
        self._write_json(self.root / "execution_status.json", final_status)
        persisted_metadata = self._read_json(self.root / "run_metadata.json")
        persisted_status = self._read_json(self.root / "execution_status.json")
        validate_terminal_payloads(persisted_metadata, persisted_status)
        manifest_canonical, manifest_hash = build_artifact_manifest(self.root)
        manifest_payload = {
            **manifest_canonical,
            "manifest_sha256": manifest_hash,
        }
        self._write_json(self.root / "artifact_manifest.json", manifest_payload)
        persisted_manifest = self._read_json(self.root / "artifact_manifest.json")
        validate_artifact_manifest(self.root, persisted_manifest)
        self._write_json(
            self.root / "_SUCCESS",
            {
                "schema_version": 1,
                "artifact_manifest_sha256": manifest_hash,
            },
        )

    def fail(self, error: BaseException, metadata: dict[str, Any]) -> None:
        if (self.root / "_SUCCESS").exists():
            raise ResearchArtifactError("A successful run cannot be failed.")
        finished = datetime.now(timezone.utc)
        reason = {"type": type(error).__name__, "message": str(error)}
        failed_metadata = {
            **metadata,
            "status": "failed",
            "finished_at_utc": finished.isoformat(),
            "failure": reason,
        }
        errors: list[BaseException] = []
        for writer in (
            lambda: self._write_json(
                self.root / "run_metadata.json",
                failed_metadata,
            ),
            lambda: self._write_status(
                "failed",
                finished_at=finished,
                failure=reason,
            ),
            lambda: (self.root / "_FAILED").touch(exist_ok=True),
        ):
            try:
                writer()
            except BaseException as persistence_error:
                errors.append(persistence_error)
        if errors:
            details = "; ".join(str(item) for item in errors)
            raise ResearchArtifactError(
                f"Failed to preserve research failure: {details}"
            ) from error

    def inventory(self) -> list[str]:
        return sorted(
            str(path.relative_to(self.root)).replace("\\", "/")
            for path in self.root.rglob("*")
            if path.is_file()
        )

    def _validate_completion(self, result: ResearchEvaluationResult) -> None:
        plan = self.config.plan_payload
        repeat_count = len(plan["outer_evaluation"]["repeat_seeds"])
        row_count = int(plan["dataset"]["expected_rows"])
        outer_folds = int(plan["outer_evaluation"]["n_splits"])
        expected_threshold_rows = repeat_count * row_count * (outer_folds - 1)
        errors: list[str] = []
        if len(self.completed_folds) != self.expected_folds:
            errors.append(
                f"expected {self.expected_folds} completed folds, "
                f"got {len(self.completed_folds)}"
            )
        expected_lengths = {
            "outer validation": (
                len(result.outer_validation),
                repeat_count * row_count,
            ),
            "threshold-selection OOF": (
                len(result.threshold_selection_oof),
                expected_threshold_rows,
            ),
            "outer fold metrics": (
                len(result.outer_fold_metrics),
                self.expected_folds,
            ),
            "selected thresholds": (
                len(result.selected_thresholds),
                self.expected_folds,
            ),
            "repeat metrics": (len(result.repeat_metrics), repeat_count),
        }
        for label, (actual, expected) in expected_lengths.items():
            if actual != expected:
                errors.append(f"{label}: expected {expected}, got {actual}")
        expected_positions = set(range(row_count))
        for repeat in range(1, repeat_count + 1):
            repeat_outer = result.outer_validation.loc[
                result.outer_validation["repeat"] == repeat
            ]
            if (
                len(repeat_outer) != row_count
                or repeat_outer["row_position"].duplicated().any()
                or set(repeat_outer["row_position"]) != expected_positions
            ):
                errors.append(
                    f"outer prediction coverage is invalid for repeat {repeat}"
                )
            for outer_fold in range(1, outer_folds + 1):
                validation_positions = set(
                    repeat_outer.loc[
                        repeat_outer["outer_fold"] == outer_fold,
                        "row_position",
                    ]
                )
                threshold_oof = result.threshold_selection_oof.loc[
                    (result.threshold_selection_oof["repeat"] == repeat)
                    & (result.threshold_selection_oof["outer_fold"] == outer_fold)
                ]
                if (
                    threshold_oof["row_position"].duplicated().any()
                    or set(threshold_oof["row_position"])
                    != expected_positions - validation_positions
                    or validation_positions & set(threshold_oof["row_position"])
                ):
                    errors.append(
                        "threshold-selection OOF coverage is invalid for "
                        f"repeat {repeat}, outer fold {outer_fold}"
                    )
        for label, frame in (
            ("outer fold metrics", result.outer_fold_metrics),
            ("selected thresholds", result.selected_thresholds),
        ):
            if frame.duplicated(["repeat", "outer_fold"]).any():
                errors.append(f"{label} contains duplicate fold records")
        required = [
            "resolved_config.yaml",
            "run_metadata.json",
            "execution_status.json",
            "evaluation_plan.json",
            "candidate_contract.json",
            "run_implementation.json",
            "loaded_modules.json",
            "dataset_fingerprints.json",
            "feature_schema.json",
            "splits/outer_assignments.parquet",
            "splits/threshold_selection_assignments.parquet",
            "predictions/threshold_selection_oof.parquet",
            "predictions/outer_validation.parquet",
            "thresholds/selected_thresholds.csv",
            "thresholds/threshold_summary.json",
            "metrics/outer_folds.csv",
            "metrics/repeats.csv",
            "metrics/aggregate.json",
        ]
        if self.config.payload["artifacts"]["save_threshold_curves"]:
            required.append("thresholds/threshold_curves.parquet")
        missing = [name for name in required if not (self.root / name).is_file()]
        if missing:
            errors.append(f"missing required artifacts: {missing}")
        if (self.root / "_FAILED").exists():
            errors.append("_FAILED already exists")
        if (self.root / "_SUCCESS").exists():
            errors.append("_SUCCESS already exists")
        if errors:
            raise ResearchArtifactError(
                "Research completion checks failed:\n- " + "\n- ".join(errors)
            )

    def _status_payload(
        self,
        status: str,
        *,
        finished_at: datetime | None = None,
        failure: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return {
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

    def _write_status(
        self,
        status: str,
        *,
        finished_at: datetime | None = None,
        failure: dict[str, str] | None = None,
    ) -> None:
        self._write_json(
            self.root / "execution_status.json",
            self._status_payload(
                status,
                finished_at=finished_at,
                failure=failure,
            ),
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ResearchArtifactError(f"JSON artifact is not an object: {path}")
        return payload

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(
                payload,
                file,
                indent=2,
                ensure_ascii=False,
                default=_json_default,
            )
            file.write("\n")
        temporary.replace(path)

    @staticmethod
    def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=True)
        temporary.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}.")
