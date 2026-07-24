from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs/experiments/manual_lightgbmprep_r31.yaml"
)
LOCAL_ASSETS = [
    PROJECT_ROOT
    / "data/processed/v3_targeted_missingness/X_train.parquet",
    PROJECT_ROOT
    / "data/processed/v3_targeted_missingness/y_train.parquet",
    PROJECT_ROOT
    / "data/processed/v3_targeted_missingness/X_test.parquet",
    PROJECT_ROOT / "data/raw/final_proj_sample_submission.csv",
    PROJECT_ROOT
    / "submissions/manual_lightgbmprep_r31_threshold_0117.csv",
]

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MANUAL_LIGHTGBM_PARITY") != "1"
    or not all(path.exists() for path in LOCAL_ASSETS),
    reason=(
        "Set RUN_MANUAL_LIGHTGBM_PARITY=1 and provide ignored local assets "
        "to run the end-to-end parity test."
    ),
)


def test_end_to_end_manual_lightgbm_parity(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["artifacts"]["root"] = str(tmp_path / "runs")
    config_path = tmp_path / "parity.yaml"
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_experiment.py",
            "--config",
            str(config_path),
            "--run-id",
            "pytest-parity",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    run_dir = tmp_path / "runs/pytest-parity"
    report = json.loads(
        (run_dir / "parity_report.json").read_text(encoding="utf-8")
    )
    assert report["passed"] is True
    assert all(
        gate["passed"] for gate in report["primary_gates"].values()
    )
    assert (run_dir / "_SUCCESS").exists()
