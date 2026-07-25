from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from src.churn_ml.research_config import (
    ResearchConfigurationError,
    load_research_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_CONFIG = PROJECT_ROOT / "configs/research/manual_lightgbmprep_r31_nested_rs5x5.yaml"
PLAN_CONFIG = PROJECT_ROOT / "configs/research/plans/telecom_v3_nested_rs5x5_v1.yaml"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("plan.threshold_policy.minimum", float("nan")),
        ("plan.threshold_policy.maximum", float("inf")),
        ("plan.threshold_policy.step", float("-inf")),
        ("plan.threshold_policy.maximizer_absolute_tolerance", float("nan")),
        ("plan.threshold_policy.constant_probability_fallback", float("inf")),
        ("plan.metrics.secondary", ["roc_auc", "unknown_metric"]),
        ("plan.metrics.diagnostic", ["tn", "tn"]),
        ("plan.metrics.secondary", ["roc_auc", "balanced_accuracy"]),
        ("plan.outer_evaluation.n_splits", True),
        ("plan.outer_evaluation.n_splits", 101),
        ("plan.outer_evaluation.repeat_seeds", [-1]),
        ("plan.threshold_selection.random_state", 2**32),
        ("run.experiment.id", "../escape"),
        ("run.candidate.id", "candidate/name"),
        ("plan.plan.id", "."),
        ("plan.dataset.version", "unsafe version"),
        ("plan.dataset.files.train_features.name", ""),
        ("plan.dataset.files.train_features.name", " "),
        ("plan.dataset.files.train_features.name", "."),
        ("plan.dataset.files.train_features.name", ".."),
        ("plan.dataset.files.train_features.name", "../X_train.parquet"),
        ("plan.dataset.files.train_features.name", "nested/X_train.parquet"),
        ("plan.dataset.files.train_features.name", r"nested\X_train.parquet"),
        ("plan.dataset.files.train_features.name", "/tmp/X_train.parquet"),
        ("plan.dataset.files.train_features.name", r"C:\X_train.parquet"),
        ("plan.dataset.files.train_features.name", " X_train.parquet"),
    ],
)
def test_semantic_config_mutations_fail_preflight(
    tmp_path: Path,
    path: str,
    value: Any,
) -> None:
    run_payload = yaml.safe_load(RUN_CONFIG.read_text(encoding="utf-8"))
    plan_payload = yaml.safe_load(PLAN_CONFIG.read_text(encoding="utf-8"))
    root_name, *keys = path.split(".")
    payload = run_payload if root_name == "run" else plan_payload
    target = payload
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value

    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        yaml.safe_dump(plan_payload, sort_keys=False),
        encoding="utf-8",
    )
    run_payload["evaluation_plan_path"] = str(plan_path)
    run_path = tmp_path / "run.yaml"
    run_path.write_text(
        yaml.safe_dump(run_payload, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ResearchConfigurationError):
        load_research_config(run_path, project_root=PROJECT_ROOT)
