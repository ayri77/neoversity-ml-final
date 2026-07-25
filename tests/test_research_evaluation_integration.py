from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_CONFIG = PROJECT_ROOT / "configs/research/manual_lightgbmprep_r31_nested_rs5x5.yaml"
PLAN_CONFIG = PROJECT_ROOT / "configs/research/plans/telecom_v3_nested_rs5x5_v1.yaml"
LOCAL_TRAINING = [
    PROJECT_ROOT / "data/processed/v3_targeted_missingness/X_train.parquet",
    PROJECT_ROOT / "data/processed/v3_targeted_missingness/y_train.parquet",
    PROJECT_ROOT / "data/processed/v3_targeted_missingness/metadata.json",
]

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_RESEARCH_EVALUATION_INTEGRATION") != "1"
    or not all(path.exists() for path in LOCAL_TRAINING),
    reason=(
        "Set RUN_RESEARCH_EVALUATION_INTEGRATION=1 and provide the local "
        "v3 training dataset."
    ),
)


def test_reduced_local_research_evaluation(tmp_path: Path) -> None:
    plan = yaml.safe_load(PLAN_CONFIG.read_text(encoding="utf-8"))
    plan["plan"]["id"] = "pytest_reduced_nested"
    plan["outer_evaluation"]["n_splits"] = 2
    plan["outer_evaluation"]["repeat_seeds"] = [0]
    plan["threshold_selection"]["n_splits"] = 2
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        yaml.safe_dump(plan, sort_keys=False),
        encoding="utf-8",
    )
    run = yaml.safe_load(RUN_CONFIG.read_text(encoding="utf-8"))
    run["experiment"]["id"] = "pytest_reduced_nested"
    run["evaluation_plan_path"] = str(plan_path)
    run["artifacts"]["root"] = str(tmp_path / "runs")
    run_path = tmp_path / "run.yaml"
    run_path.write_text(
        yaml.safe_dump(run, sort_keys=False),
        encoding="utf-8",
    )

    process = subprocess.run(
        [
            sys.executable,
            "scripts/run_research_evaluation.py",
            "--config",
            str(run_path),
            "--run-id",
            "pytest-reduced",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert process.returncode == 0, process.stdout + process.stderr
    successes = list((tmp_path / "runs").rglob("_SUCCESS"))
    assert len(successes) == 1
    run_dir = successes[0].parent
    assert not (run_dir / "_FAILED").exists()
    outer = __import__("pandas").read_parquet(
        run_dir / "predictions/outer_validation.parquet"
    )
    threshold = __import__("pandas").read_parquet(
        run_dir / "predictions/threshold_selection_oof.parquet"
    )
    assert len(outer) == 10000
    assert len(threshold) == 10000
    assert len(__import__("pandas").read_csv(run_dir / "metrics/outer_folds.csv")) == 2
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["competition_assets_accessed"] is False
