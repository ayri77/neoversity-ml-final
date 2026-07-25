from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.churn_ml.research_artifact_validation import validate_artifact_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs/final/manual_lightgbmprep_r31_submission_v1.yaml"
REQUIRED = [
    PROJECT_ROOT / "data/processed/v3_targeted_missingness/X_train.parquet",
    PROJECT_ROOT / "data/processed/v3_targeted_missingness/y_train.parquet",
    PROJECT_ROOT / "data/processed/v3_targeted_missingness/X_test.parquet",
    PROJECT_ROOT / "data/raw/final_proj_sample_submission.csv",
    PROJECT_ROOT
    / "artifacts/research_evaluations"
    / "telecom_v3_nested_rs5x5_v1_09b3e1632848"
    / "manual_lightgbmprep_r31_v3"
    / "final-provenance-20260725"
    / "_SUCCESS",
]
RUN_ID = "pytest-final-integration-20260725"

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_FINAL_SUBMISSION_INTEGRATION") != "1"
    or not all(path.exists() for path in REQUIRED),
    reason="Set RUN_FINAL_SUBMISSION_INTEGRATION=1 with approved local assets.",
)


def test_real_local_final_submission_flow(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["artifacts"]["root"] = "artifacts/final_integration_tests"
    config_path = tmp_path / "final-integration.yaml"
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    process = subprocess.run(
        [
            sys.executable,
            "scripts/run_final_submission.py",
            "--config",
            str(config_path),
            "--run-id",
            RUN_ID,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert process.returncode == 0, process.stdout + process.stderr
    successes = list(
        (PROJECT_ROOT / "artifacts/final_integration_tests").rglob(f"{RUN_ID}/_SUCCESS")
    )
    assert len(successes) == 1
    root = successes[0].parent
    assert not (root / "_FAILED").exists()
    selection = json.loads(
        (root / "threshold/selection.json").read_text(encoding="utf-8")
    )
    assert selection["selected_threshold"] == 0.107
    assert selection["diagnostic_balanced_accuracy"] == pytest.approx(
        0.9079100376972717,
        abs=1e-15,
    )
    verification = json.loads(
        (root / "inference_verification.json").read_text(encoding="utf-8")
    )
    assert verification["probabilities_equal"] is True
    predictions = pd.read_parquet(root / "predictions/test_predictions.parquet")
    submission = pd.read_csv(
        root / "submission/manual_lightgbmprep_r31_submission_v1.csv"
    )
    assert len(predictions) == len(submission) == 2500
    assert predictions["probability"].between(0.0, 1.0).all()
    assert submission.columns.tolist() == ["index", "y"]
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    validate_artifact_manifest(root, manifest)
