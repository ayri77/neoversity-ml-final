from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from src.churn_ml.final_artifact_validation import (
    finalize_and_validate_manifest,
    validate_final_manifest,
    validate_final_metadata_status,
    validate_persisted_final_run,
)
from src.churn_ml.final_config import FinalConfig, SAFE_SLUG
from src.churn_ml.final_data import FinalData
from src.churn_ml.final_manual_lightgbm import FinalFitResult
from src.churn_ml.final_promotion import OperationalThreshold, PromotionEvidence


class FinalArtifactError(RuntimeError):
    """Raised when final artifact lifecycle cannot be completed safely."""


class FinalArtifactStore:
    """Own one non-overwriting final-model and submission run."""

    def __init__(
        self,
        config: FinalConfig,
        *,
        promotion_hash: str,
        final_model_hash: str,
        run_id: str | None,
    ) -> None:
        self.config = config
        self.promotion_hash = promotion_hash
        self.final_model_hash = final_model_hash
        self.started_at = datetime.now(timezone.utc)
        self.run_id = run_id or (
            self.started_at.strftime("%Y%m%dT%H%M%S%fZ") + f"_{final_model_hash[:8]}"
        )
        if SAFE_SLUG.fullmatch(self.run_id) is None:
            raise FinalArtifactError("run_id is not a safe slug.")
        root = config.artifact_root.resolve()
        parent = (root / f"{config.promotion_id}_{promotion_hash[:12]}").resolve()
        self.root = (parent / self.run_id).resolve()
        if root not in self.root.parents:
            raise FinalArtifactError("Final run directory escapes artifact root.")
        self.root.mkdir(parents=True, exist_ok=False)
        self.phase = "allocated"
        try:
            for name in (
                "promotion",
                "identities",
                "provenance",
                "fingerprints",
                "schemas",
                "threshold",
                "model",
                "predictions",
                "diagnostics",
                "submission",
            ):
                (self.root / name).mkdir()
            self._write_yaml(
                self.root / "resolved_config.yaml",
                config.resolved_payload(),
            )
            self._write_initial_status()
        except BaseException as error:
            self.fail(error)
            raise

    def save_initial_contract(
        self,
        *,
        promotion: PromotionEvidence,
        threshold: OperationalThreshold,
        data: FinalData,
        promotion_identity: dict[str, Any],
        final_model_identity: dict[str, Any],
        source_manifest: dict[str, Any],
        source_hash: str,
        loaded_modules: dict[str, Any],
        loaded_hash: str,
        environment: dict[str, str],
        invocation: dict[str, Any],
        git: dict[str, Any],
    ) -> None:
        self.phase = "initial_contract"
        approved = self.config.payload["promotion"]
        promotion_record = {
            "schema_version": 1,
            "promotion_id": approved["id"],
            "candidate_id": approved["research"]["candidate_id"],
            "evaluation_plan_id": approved["research"]["plan_id"],
            "evaluation_plan_sha256": approved["research"]["plan_sha256"],
            "candidate_contract_sha256": approved["research"]["candidate_sha256"],
            "source_research_run_path": approved["research"]["run_path"],
            "source_research_run_id": approved["research"]["run_id"],
            "research_artifact_manifest_sha256": promotion.research_manifest_hash,
            "research_metric": approved["research"]["approved_metrics"],
            "approval": approved["approval"],
            "research_metrics_policy": (
                "Research metrics are persisted unchanged and will not be "
                "recomputed or replaced by final-training diagnostics."
            ),
            "operational_threshold": threshold.selection_record,
            "promotion_sha256": self.promotion_hash,
        }
        research_evidence = {
            "schema_version": 1,
            "verification": "passed",
            "metadata": promotion.research_metadata,
            "execution_status": promotion.research_status,
            "research_metrics": promotion.research_metrics,
            "artifact_manifest_sha256": promotion.research_manifest_hash,
        }
        self._write_json(self.root / "promotion" / "record.json", promotion_record)
        self._write_json(
            self.root / "promotion" / "research_evidence.json",
            research_evidence,
        )
        self._write_json(
            self.root / "identities" / "evaluation_plan.json",
            promotion.plan_identity,
        )
        self._write_json(
            self.root / "identities" / "candidate_contract.json",
            promotion.candidate_identity,
        )
        self._write_json(
            self.root / "identities" / "promotion.json",
            {"sha256": self.promotion_hash, "canonical": promotion_identity},
        )
        self._write_json(
            self.root / "identities" / "final_model.json",
            {"sha256": self.final_model_hash, "canonical": final_model_identity},
        )
        self._write_json(
            self.root / "provenance" / "source_manifest.json",
            {"sha256": source_hash, "canonical": source_manifest},
        )
        self._write_json(
            self.root / "provenance" / "loaded_modules.json",
            {"sha256": loaded_hash, **loaded_modules},
        )
        self._write_json(self.root / "provenance" / "environment.json", environment)
        self._write_json(self.root / "provenance" / "invocation.json", invocation)
        self._write_json(self.root / "provenance" / "git.json", git)
        self._write_json(self.root / "fingerprints" / "data.json", data.fingerprints)
        self._write_json(
            self.root / "schemas" / "feature_schema.json",
            data.feature_schema.to_dict(),
        )
        self._write_json(
            self.root / "schemas" / "submission_schema.json",
            data.submission_schema,
        )
        self._write_parquet(
            self.root / "threshold" / "averaged_oos_predictions.parquet",
            threshold.averaged_oos,
        )
        self._write_parquet(
            self.root / "threshold" / "threshold_curve.parquet",
            threshold.threshold_curve,
        )
        self._write_json(
            self.root / "threshold" / "selection.json",
            threshold.selection_record,
        )
        self._write_initial_status()

    def save_fit_outputs(
        self,
        fit: FinalFitResult,
        predictions: pd.DataFrame,
        submission: pd.DataFrame,
        diagnostics: dict[str, Any],
    ) -> None:
        self.phase = "fit_persistence"
        self._write_joblib(
            self.root / "model" / "target_encoder.joblib",
            fit.encoder,
        )
        self._write_joblib(
            self.root / "model" / "lightgbm_classifier.joblib",
            fit.model,
        )
        self._write_parquet(
            self.root / "predictions" / "test_predictions.parquet",
            predictions,
        )
        self._write_csv(
            self.root
            / "submission"
            / self.config.payload["artifacts"]["submission_filename"],
            submission,
        )
        self._write_json(
            self.root / "diagnostics" / "operational.json",
            diagnostics,
        )
        self._write_initial_status()

    def save_inference_verification(self, record: dict[str, Any]) -> None:
        self.phase = "reload_verified"
        self._write_json(self.root / "inference_verification.json", record)
        self._write_initial_status()

    def complete(
        self,
        *,
        data: FinalData,
        fit: FinalFitResult,
        threshold: OperationalThreshold,
        predictions: pd.DataFrame,
        submission: pd.DataFrame,
    ) -> None:
        self.phase = "semantic_validation"
        validate_persisted_final_run(
            self.root,
            self.config,
            data,
            fit,
            threshold,
            predictions,
            submission,
            promotion_hash=self.promotion_hash,
            final_model_hash=self.final_model_hash,
        )
        finished = datetime.now(timezone.utc)
        metadata = {
            "schema_version": 1,
            "run_id": self.run_id,
            "status": "completed",
            "started_at_utc": self.started_at.isoformat(),
            "finished_at_utc": finished.isoformat(),
            "duration_seconds": (finished - self.started_at).total_seconds(),
            "fit_duration_seconds": fit.duration_seconds,
            "promotion_sha256": self.promotion_hash,
            "final_model_sha256": self.final_model_hash,
            "model_count": 1,
            "encoder_count": 1,
            "probability_count": len(predictions),
            "submission_row_count": len(submission),
            "network_calls": False,
            "failure": None,
        }
        status = {
            "schema_version": 1,
            "run_id": self.run_id,
            "status": "completed",
            "started_at_utc": self.started_at.isoformat(),
            "finished_at_utc": finished.isoformat(),
            "phase": "completed",
            "failure": None,
        }
        validate_final_metadata_status(
            metadata,
            status,
            run_id=self.run_id,
            started_at=self.started_at,
            finished_at=finished,
            promotion_hash=self.promotion_hash,
            final_model_hash=self.final_model_hash,
            fit_duration_seconds=fit.duration_seconds,
        )
        self._write_json(self.root / "run_metadata.json", metadata)
        self._write_json(self.root / "execution_status.json", status)
        validate_final_metadata_status(
            self._read_json(self.root / "run_metadata.json"),
            self._read_json(self.root / "execution_status.json"),
            run_id=self.run_id,
            started_at=self.started_at,
            finished_at=finished,
            promotion_hash=self.promotion_hash,
            final_model_hash=self.final_model_hash,
            fit_duration_seconds=fit.duration_seconds,
        )
        manifest, digest = finalize_and_validate_manifest(self.root)
        self._write_json(self.root / "artifact_manifest.json", manifest)
        validate_final_manifest(
            self.root,
            self._read_json(self.root / "artifact_manifest.json"),
        )
        self.phase = "completed"
        self._write_json(
            self.root / "_SUCCESS",
            {
                "schema_version": 1,
                "artifact_manifest_sha256": digest,
            },
        )

    def fail(self, error: BaseException) -> None:
        if (self.root / "_SUCCESS").exists():
            raise FinalArtifactError("A successful final run cannot be failed.")
        finished = datetime.now(timezone.utc)
        failure = {
            "type": type(error).__name__,
            "message": str(error),
            "phase": self.phase,
            "finished_at_utc": finished.isoformat(),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        }
        failures: list[BaseException] = []
        payloads = (
            (
                self.root / "failure.json",
                failure,
            ),
            (
                self.root / "run_metadata.json",
                {
                    "schema_version": 1,
                    "run_id": self.run_id,
                    "status": "failed",
                    "started_at_utc": self.started_at.isoformat(),
                    "finished_at_utc": finished.isoformat(),
                    "phase": self.phase,
                    "failure": {
                        "type": failure["type"],
                        "message": failure["message"],
                    },
                },
            ),
            (
                self.root / "execution_status.json",
                {
                    "schema_version": 1,
                    "run_id": self.run_id,
                    "status": "failed",
                    "started_at_utc": self.started_at.isoformat(),
                    "finished_at_utc": finished.isoformat(),
                    "phase": self.phase,
                    "failure": {
                        "type": failure["type"],
                        "message": failure["message"],
                    },
                },
            ),
        )
        for path, payload in payloads:
            try:
                self._write_json(path, payload)
            except BaseException as persistence_error:
                failures.append(persistence_error)
        try:
            self._write_json(
                self.root / "_FAILED",
                {
                    "schema_version": 1,
                    "failure_type": failure["type"],
                    "phase": self.phase,
                },
            )
        except BaseException as persistence_error:
            failures.append(persistence_error)
        if failures:
            raise FinalArtifactError(
                "Failed to preserve final-run failure: "
                + "; ".join(str(item) for item in failures)
            ) from error

    def _write_initial_status(self) -> None:
        self._write_json(
            self.root / "run_metadata.json",
            {
                "schema_version": 1,
                "run_id": self.run_id,
                "status": "running",
                "started_at_utc": self.started_at.isoformat(),
                "phase": self.phase,
            },
        )
        self._write_json(
            self.root / "execution_status.json",
            {
                "schema_version": 1,
                "run_id": self.run_id,
                "status": "running",
                "started_at_utc": self.started_at.isoformat(),
                "phase": self.phase,
                "failure": None,
            },
        )

    @staticmethod
    def _temporary(path: Path) -> Path:
        return path.with_name(f".{path.name}.tmp")

    def _write_json(self, path: Path, payload: Any) -> None:
        temporary = self._temporary(path)
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _write_yaml(self, path: Path, payload: Any) -> None:
        temporary = self._temporary(path)
        temporary.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _write_parquet(self, path: Path, frame: pd.DataFrame) -> None:
        temporary = self._temporary(path)
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)

    def _write_csv(self, path: Path, frame: pd.DataFrame) -> None:
        temporary = self._temporary(path)
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)

    def _write_joblib(self, path: Path, value: Any) -> None:
        temporary = self._temporary(path)
        joblib.dump(value, temporary)
        os.replace(temporary, path)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise FinalArtifactError(f"JSON artifact is not an object: {path}")
        return payload


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")
