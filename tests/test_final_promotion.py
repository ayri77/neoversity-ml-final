from __future__ import annotations

import shutil
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.churn_ml.final_config import load_final_config
from src.churn_ml.final_promotion import (
    PromotionVerificationError,
    derive_operational_threshold,
    verify_approved_research_run,
)
from src.churn_ml.research_protocol import select_balanced_accuracy_threshold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/final/manual_lightgbmprep_r31_submission_v1.yaml"
SOURCE_RUN = (
    PROJECT_ROOT
    / "artifacts/research_evaluations"
    / "telecom_v3_nested_rs5x5_v1_09b3e1632848"
    / "manual_lightgbmprep_r31_v3"
    / "final-provenance-20260725"
)

pytestmark = pytest.mark.skipif(
    not SOURCE_RUN.is_dir(),
    reason="Approved local research artifacts are unavailable.",
)


class _ConfigView:
    def __init__(self, baseline, run_root: Path):
        self.payload = deepcopy(baseline.payload)
        self.project_root = baseline.project_root
        self.research_config = baseline.research_config
        self.research_run = run_root


def test_valid_promotion_and_operational_threshold_are_exact() -> None:
    config = load_final_config(CONFIG_PATH, project_root=PROJECT_ROOT)

    promotion = verify_approved_research_run(config)
    threshold = derive_operational_threshold(config, promotion)

    assert promotion.research_status["completed_outer_folds"] == 25
    assert threshold.selected_threshold == 0.107
    assert threshold.diagnostic_balanced_accuracy == pytest.approx(
        0.9079100376972717,
        abs=1e-15,
    )
    assert len(threshold.averaged_oos) == 10000
    assert (threshold.averaged_oos["repeat_count"] == 5).all()
    assert (
        threshold.selection_record["diagnostic_label"]
        == "operational_threshold_selection_diagnostic"
    )
    assert "unbiased_estimate" in threshold.selection_record["prohibited_claims"]


def test_failed_incomplete_hash_mismatch_and_tampering_are_rejected(
    tmp_path: Path,
) -> None:
    baseline = load_final_config(CONFIG_PATH, project_root=PROJECT_ROOT)
    copied = tmp_path / "final-provenance-20260725"
    shutil.copytree(SOURCE_RUN, copied)
    view = _ConfigView(baseline, copied)

    assert verify_approved_research_run(view).research_manifest_hash  # type: ignore[arg-type]

    failed = copied / "_FAILED"
    failed.write_text("failed", encoding="utf-8")
    with pytest.raises(PromotionVerificationError, match="_FAILED"):
        verify_approved_research_run(view)  # type: ignore[arg-type]
    failed.unlink()

    success = copied / "_SUCCESS"
    success_bytes = success.read_bytes()
    success.unlink()
    with pytest.raises(PromotionVerificationError, match="_SUCCESS"):
        verify_approved_research_run(view)  # type: ignore[arg-type]
    success.write_bytes(success_bytes)

    mismatch = _ConfigView(baseline, copied)
    mismatch.payload["promotion"]["research"]["plan_sha256"] = "0" * 64
    with pytest.raises(PromotionVerificationError, match="metadata plan_sha256"):
        verify_approved_research_run(mismatch)  # type: ignore[arg-type]

    aggregate = copied / "metrics" / "aggregate.json"
    aggregate.write_bytes(aggregate.read_bytes() + b" ")
    with pytest.raises(Exception, match="manifest"):
        verify_approved_research_run(view)  # type: ignore[arg-type]


def test_threshold_equality_and_constant_fallback_are_exact() -> None:
    policy = {
        "id": "grid_balanced_accuracy_v1",
        "metric": "balanced_accuracy",
        "minimum": 0.01,
        "maximum": 0.99,
        "step": 0.001,
        "comparison": "greater_than_or_equal",
        "maximizer_absolute_tolerance": 1e-12,
        "tie_break": "median_maximizer_lower_on_even",
        "constant_probability_fallback": 0.5,
    }
    result = select_balanced_accuracy_threshold(
        np.array([0, 1]),
        np.array([0.5, 0.5]),
        policy,
    )

    assert result.threshold == 0.5
    assert result.degenerate is True
    np.testing.assert_array_equal(
        (np.array([np.nextafter(0.5, -np.inf), 0.5]) >= 0.5).astype("int8"),
        [0, 1],
    )


def test_one_class_threshold_input_is_rejected_by_promotion_contract(
    tmp_path: Path,
) -> None:
    baseline = load_final_config(CONFIG_PATH, project_root=PROJECT_ROOT)
    source_dir = tmp_path / "predictions"
    source_dir.mkdir()
    frame = pd.DataFrame(
        {
            "repeat": np.tile(np.arange(1, 6), 2),
            "row_position": np.repeat(np.arange(2), 5),
            "target": np.zeros(10, dtype=np.int8),
            "probability": np.linspace(0.1, 0.9, 10),
        }
    )
    frame.to_parquet(source_dir / "outer_validation.parquet", index=False)
    promotion = SimpleNamespace(
        run_root=tmp_path,
        research_config=SimpleNamespace(plan_payload={"dataset": {"expected_rows": 2}}),
    )

    with pytest.raises(PromotionVerificationError, match="both classes"):
        derive_operational_threshold(baseline, promotion)  # type: ignore[arg-type]
