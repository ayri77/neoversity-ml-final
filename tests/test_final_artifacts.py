from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from src.churn_ml.final_artifact_validation import (
    FinalArtifactValidationError,
    finalize_and_validate_manifest,
    validate_final_manifest,
    validate_final_metadata_status,
)
from src.churn_ml.final_artifacts import FinalArtifactStore
from src.churn_ml.final_config import FinalConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/final/manual_lightgbmprep_r31_submission_v1.yaml"


def _config(tmp_path: Path) -> FinalConfig:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload = deepcopy(payload)
    payload["artifacts"]["root"] = "runs"
    return FinalConfig(
        payload=payload,
        source_path=tmp_path / "final.yaml",
        project_root=tmp_path,
    )


def test_store_rejects_overwrite_and_preserves_failure(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = FinalArtifactStore(
        config,
        promotion_hash="a" * 64,
        final_model_hash="b" * 64,
        run_id="unit-run",
    )
    store.fail(RuntimeError("exact failure"))

    assert (store.root / "_FAILED").is_file()
    assert not (store.root / "_SUCCESS").exists()
    assert "exact failure" in (store.root / "failure.json").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        FinalArtifactStore(
            config,
            promotion_hash="a" * 64,
            final_model_hash="b" * 64,
            run_id="unit-run",
        )


def test_artifact_manifest_detects_corruption(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    artifact = root / "ordinary.json"
    artifact.write_text('{"value": 1}\n', encoding="utf-8")
    payload, _ = finalize_and_validate_manifest(root)
    manifest = root / "artifact_manifest.json"
    manifest.write_text(__import__("json").dumps(payload), encoding="utf-8")

    validate_final_manifest(root, payload)
    artifact.write_text('{"value": 2}\n', encoding="utf-8")

    with pytest.raises(Exception, match="manifest"):
        validate_final_manifest(root, payload)


def test_final_metadata_status_requires_exact_completed_utc_contract() -> None:
    started = datetime(2026, 7, 25, 16, 0, tzinfo=timezone.utc)
    finished = started + timedelta(seconds=2)
    metadata = {
        "schema_version": 1,
        "run_id": "unit",
        "status": "completed",
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "duration_seconds": 2.0,
        "fit_duration_seconds": 1.0,
        "promotion_sha256": "a" * 64,
        "final_model_sha256": "b" * 64,
        "model_count": 1,
        "encoder_count": 1,
        "probability_count": 2500,
        "submission_row_count": 2500,
        "network_calls": False,
        "failure": None,
    }
    status = {
        "schema_version": 1,
        "run_id": "unit",
        "status": "completed",
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "phase": "completed",
        "failure": None,
    }

    validate_final_metadata_status(
        metadata,
        status,
        run_id="unit",
        started_at=started,
        finished_at=finished,
        promotion_hash="a" * 64,
        final_model_hash="b" * 64,
        fit_duration_seconds=1.0,
    )
    metadata["started_at_utc"] = "2026-07-25T16:00:00"
    with pytest.raises(FinalArtifactValidationError):
        validate_final_metadata_status(
            metadata,
            status,
            run_id="unit",
            started_at=started,
            finished_at=finished,
            promotion_hash="a" * 64,
            final_model_hash="b" * 64,
            fit_duration_seconds=1.0,
        )
