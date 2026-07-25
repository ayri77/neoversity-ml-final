from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from src.churn_ml.final_config import FinalConfig
from src.churn_ml.final_data import FinalData
from src.churn_ml.final_manual_lightgbm import FinalFitResult
from src.churn_ml.final_promotion import OperationalThreshold
from src.churn_ml.research_artifact_validation import (
    build_artifact_manifest,
    validate_artifact_manifest,
)


class FinalArtifactValidationError(RuntimeError):
    """Raised when persisted final artifacts cannot prove the final run."""


REQUIRED_PATHS = {
    "resolved_config.yaml",
    "promotion/record.json",
    "promotion/research_evidence.json",
    "identities/evaluation_plan.json",
    "identities/candidate_contract.json",
    "identities/promotion.json",
    "identities/final_model.json",
    "provenance/source_manifest.json",
    "provenance/loaded_modules.json",
    "provenance/environment.json",
    "provenance/invocation.json",
    "provenance/git.json",
    "fingerprints/data.json",
    "schemas/feature_schema.json",
    "schemas/submission_schema.json",
    "threshold/averaged_oos_predictions.parquet",
    "threshold/threshold_curve.parquet",
    "threshold/selection.json",
    "model/target_encoder.joblib",
    "model/lightgbm_classifier.joblib",
    "predictions/test_predictions.parquet",
    "diagnostics/operational.json",
    "inference_verification.json",
    "run_metadata.json",
    "execution_status.json",
}


def verify_reloaded_inference(
    root: Path,
    data: FinalData,
    fit: FinalFitResult,
    threshold: OperationalThreshold,
    predictions: pd.DataFrame,
    submission: pd.DataFrame,
) -> dict[str, Any]:
    encoder = joblib.load(root / "model" / "target_encoder.joblib")
    model = joblib.load(root / "model" / "lightgbm_classifier.joblib")
    reencoded = encoder.transform(data.X_test.reset_index(drop=True))
    try:
        pd.testing.assert_frame_equal(
            reencoded,
            fit.encoded_test,
            check_exact=True,
            check_dtype=True,
            check_column_type=True,
        )
    except AssertionError as error:
        raise FinalArtifactValidationError(
            f"Reloaded encoded test differs exactly: {error}"
        ) from error
    matrix = np.asarray(model.predict_proba(reencoded), dtype=np.float64)
    if matrix.shape != (2500, 2):
        raise FinalArtifactValidationError("Reloaded model probability shape differs.")
    probabilities = matrix[:, 1]
    if not np.array_equal(probabilities, fit.test_probabilities):
        raise FinalArtifactValidationError(
            "Reloaded probabilities are not exactly equal."
        )
    labels = (probabilities >= threshold.selected_threshold).astype(np.int8)
    if not np.array_equal(labels, predictions["prediction"].to_numpy(dtype=np.int8)):
        raise FinalArtifactValidationError("Reloaded labels are not exactly equal.")
    rebuilt = data.sample_submission.copy(deep=True)
    rebuilt["y"] = labels
    try:
        pd.testing.assert_frame_equal(
            rebuilt,
            submission,
            check_exact=True,
            check_dtype=True,
        )
    except AssertionError as error:
        raise FinalArtifactValidationError(
            f"Reloaded submission is not exactly equal: {error}"
        ) from error
    return {
        "schema_version": 1,
        "encoder_reload": "exact",
        "model_reload": "exact",
        "encoded_test_equal": True,
        "probabilities_equal": True,
        "labels_equal": True,
        "submission_equal": True,
        "probability_count": len(probabilities),
        "comparison_tolerance": None,
    }


def validate_persisted_final_run(
    root: Path,
    config: FinalConfig,
    data: FinalData,
    fit: FinalFitResult,
    threshold: OperationalThreshold,
    predictions: pd.DataFrame,
    submission: pd.DataFrame,
    *,
    promotion_hash: str,
    final_model_hash: str,
) -> None:
    missing = sorted(path for path in REQUIRED_PATHS if not (root / path).is_file())
    submission_relative = (
        "submission/" + config.payload["artifacts"]["submission_filename"]
    )
    if not (root / submission_relative).is_file():
        missing.append(submission_relative)
    if missing:
        raise FinalArtifactValidationError(f"Missing final artifacts: {missing}")
    if (root / "_SUCCESS").exists() or (root / "_FAILED").exists():
        raise FinalArtifactValidationError(
            "Terminal marker exists before completion validation."
        )

    saved_predictions = pd.read_parquet(
        root / "predictions" / "test_predictions.parquet"
    )
    try:
        pd.testing.assert_frame_equal(
            saved_predictions,
            predictions,
            check_exact=True,
            check_dtype=True,
        )
    except AssertionError as error:
        raise FinalArtifactValidationError(
            f"Persisted test predictions differ: {error}"
        ) from error
    values = saved_predictions["probability"].to_numpy(dtype=np.float64)
    if (
        len(values) != 2500
        or not np.isfinite(values).all()
        or np.any((values < 0.0) | (values > 1.0))
    ):
        raise FinalArtifactValidationError("Persisted probabilities are invalid.")
    expected_labels = (values >= threshold.selected_threshold).astype(np.int8)
    if not np.array_equal(
        expected_labels,
        saved_predictions["prediction"].to_numpy(dtype=np.int8),
    ):
        raise FinalArtifactValidationError("Persisted labels violate >= semantics.")

    submission_path = root / submission_relative
    saved_submission = pd.read_csv(submission_path)
    if (
        saved_submission.columns.tolist() != ["index", "y"]
        or len(saved_submission) != 2500
    ):
        raise FinalArtifactValidationError("Persisted submission schema differs.")
    if not np.array_equal(
        saved_submission["index"].to_numpy(),
        data.sample_submission["index"].to_numpy(),
    ):
        raise FinalArtifactValidationError("Persisted submission order differs.")
    if not np.array_equal(
        saved_submission["y"].to_numpy(dtype=np.int8),
        submission["y"].to_numpy(dtype=np.int8),
    ):
        raise FinalArtifactValidationError("Persisted submission labels differ.")
    if set(saved_submission["y"].unique().tolist()) - {0, 1}:
        raise FinalArtifactValidationError(
            "Persisted submission labels are not binary."
        )

    averaged = pd.read_parquet(root / "threshold" / "averaged_oos_predictions.parquet")
    curve = pd.read_parquet(root / "threshold" / "threshold_curve.parquet")
    try:
        pd.testing.assert_frame_equal(
            averaged,
            threshold.averaged_oos,
            check_exact=True,
            check_dtype=True,
        )
        pd.testing.assert_frame_equal(
            curve,
            threshold.threshold_curve,
            check_exact=True,
            check_dtype=True,
        )
    except AssertionError as error:
        raise FinalArtifactValidationError(
            f"Persisted threshold artifact differs: {error}"
        ) from error
    if _read_json(root / "threshold" / "selection.json") != threshold.selection_record:
        raise FinalArtifactValidationError("Persisted threshold selection differs.")
    verification = _read_json(root / "inference_verification.json")
    if (
        any(
            verification.get(name) is not True
            for name in (
                "encoded_test_equal",
                "probabilities_equal",
                "labels_equal",
                "submission_equal",
            )
        )
        or verification.get("comparison_tolerance") is not None
    ):
        raise FinalArtifactValidationError("Inference parity record is not exact.")

    promotion = _read_json(root / "identities" / "promotion.json")
    final_model = _read_json(root / "identities" / "final_model.json")
    if (
        promotion.get("sha256") != promotion_hash
        or final_model.get("sha256") != final_model_hash
    ):
        raise FinalArtifactValidationError("Persisted final identity hash differs.")
    if _read_json(root / "fingerprints" / "data.json") != data.fingerprints:
        raise FinalArtifactValidationError("Persisted data fingerprints differ.")
    if (
        _read_json(root / "schemas" / "feature_schema.json")
        != data.feature_schema.to_dict()
    ):
        raise FinalArtifactValidationError("Persisted feature schema differs.")


def validate_final_metadata_status(
    metadata: Mapping[str, Any],
    status: Mapping[str, Any],
    *,
    run_id: str,
    started_at: datetime,
    finished_at: datetime,
    promotion_hash: str,
    final_model_hash: str,
    fit_duration_seconds: float,
) -> None:
    metadata_keys = {
        "schema_version",
        "run_id",
        "status",
        "started_at_utc",
        "finished_at_utc",
        "duration_seconds",
        "fit_duration_seconds",
        "promotion_sha256",
        "final_model_sha256",
        "model_count",
        "encoder_count",
        "probability_count",
        "submission_row_count",
        "network_calls",
        "failure",
    }
    status_keys = {
        "schema_version",
        "run_id",
        "status",
        "started_at_utc",
        "finished_at_utc",
        "phase",
        "failure",
    }
    if set(metadata) != metadata_keys or set(status) != status_keys:
        raise FinalArtifactValidationError("Final metadata/status keys differ.")
    expected_metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "completed",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "fit_duration_seconds": fit_duration_seconds,
        "promotion_sha256": promotion_hash,
        "final_model_sha256": final_model_hash,
        "model_count": 1,
        "encoder_count": 1,
        "probability_count": 2500,
        "submission_row_count": 2500,
        "network_calls": False,
        "failure": None,
    }
    expected_status = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "completed",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "phase": "completed",
        "failure": None,
    }
    if dict(metadata) != expected_metadata or dict(status) != expected_status:
        raise FinalArtifactValidationError("Final metadata/status values differ.")
    for value in (
        metadata["started_at_utc"],
        metadata["finished_at_utc"],
        status["started_at_utc"],
        status["finished_at_utc"],
    ):
        _parse_utc(value)
    for name in ("duration_seconds", "fit_duration_seconds"):
        value = metadata[name]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise FinalArtifactValidationError(f"Final metadata {name} is invalid.")


def finalize_and_validate_manifest(root: Path) -> tuple[dict[str, Any], str]:
    canonical, digest = build_artifact_manifest(root)
    payload = {**canonical, "manifest_sha256": digest}
    return payload, digest


def validate_final_manifest(root: Path, payload: dict[str, Any]) -> None:
    validate_artifact_manifest(root, payload)


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str):
        raise FinalArtifactValidationError("UTC timestamp is missing.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise FinalArtifactValidationError("UTC timestamp is invalid.") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise FinalArtifactValidationError("Timestamp is not timezone-aware UTC.")
    return parsed


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FinalArtifactValidationError(
            f"Cannot read JSON artifact: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise FinalArtifactValidationError(f"JSON artifact is not an object: {path}")
    return payload
