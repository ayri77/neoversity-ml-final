from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import pytest

from src.churn_ml.research_artifacts import (
    ResearchArtifactError,
    ResearchArtifactStore,
)
from src.churn_ml.research_config import ResearchConfig
from src.churn_ml.research_evaluation import (
    CompletedOuterFold,
    ResearchEvaluationResult,
)
from tests.research_test_support import build_persisted_test_run
from src.churn_ml.research_protocol import EvaluationAssignments


def _config(tmp_path: Path) -> ResearchConfig:
    plan = {
        "plan": {"id": "unit-plan"},
        "dataset": {"expected_rows": 4},
        "outer_evaluation": {"n_splits": 2, "repeat_seeds": [0]},
    }
    payload = {
        "schema_version": 1,
        "experiment": {"id": "unit"},
        "candidate": {
            "id": "candidate",
            "implementation": "manual_lightgbm_r31",
            "historical_config_path": "historical.yaml",
            "manifest_path": "manifest.yaml",
            "manifest_key": "manual_lightgbm_contract",
        },
        "evaluation_plan_path": "plan.yaml",
        "artifacts": {
            "root": str(tmp_path / "runs"),
            "save_threshold_curves": True,
            "save_models": False,
        },
    }
    return ResearchConfig(
        payload=payload,
        plan_payload=plan,
        candidate_contract={},
        source_path=tmp_path / "run.yaml",
        plan_path=tmp_path / "plan.yaml",
        project_root=tmp_path,
    )


def _assignments() -> EvaluationAssignments:
    return EvaluationAssignments(
        outer=pd.DataFrame(
            {
                "repeat": [1] * 4,
                "repeat_seed": [0] * 4,
                "outer_fold": [1, 1, 2, 2],
                "row_position": [0, 1, 2, 3],
            }
        ),
        threshold_selection=pd.DataFrame(
            {
                "repeat": [1] * 4,
                "repeat_seed": [0] * 4,
                "outer_fold": [1, 1, 2, 2],
                "threshold_selection_fold": [1, 2, 1, 2],
                "row_position": [2, 3, 0, 1],
            }
        ),
    )


def _fold(outer_fold: int) -> CompletedOuterFold:
    outer_positions = [0, 1] if outer_fold == 1 else [2, 3]
    threshold_positions = [2, 3] if outer_fold == 1 else [0, 1]
    threshold = pd.DataFrame(
        {
            "repeat": [1, 1],
            "outer_fold": [outer_fold, outer_fold],
            "row_position": threshold_positions,
            "target": [0, 1],
            "probability": [0.1, 0.9],
        }
    )
    outer = pd.DataFrame(
        {
            "repeat": [1, 1],
            "outer_fold": [outer_fold, outer_fold],
            "row_position": outer_positions,
            "target": [0, 1],
            "probability": [0.1, 0.9],
            "repeat_seed": [0, 0],
            "selected_threshold": [0.5, 0.5],
            "prediction": [0, 1],
        }
    )
    return CompletedOuterFold(
        repeat=1,
        outer_fold=outer_fold,
        threshold_selection_oof=threshold,
        outer_validation=outer,
        selected_threshold={
            "repeat": 1,
            "outer_fold": outer_fold,
            "selected_threshold": 0.5,
        },
        threshold_curve=pd.DataFrame({"threshold": [0.5], "balanced_accuracy": [1.0]}),
        fold_metrics={
            "repeat": 1,
            "outer_fold": outer_fold,
            "balanced_accuracy": 1.0,
        },
    )


def _result() -> ResearchEvaluationResult:
    folds = [_fold(1), _fold(2)]
    outer = pd.concat([fold.outer_validation for fold in folds], ignore_index=True)
    threshold = pd.concat(
        [fold.threshold_selection_oof for fold in folds],
        ignore_index=True,
    )
    return ResearchEvaluationResult(
        threshold_selection_oof=threshold,
        outer_validation=outer,
        selected_thresholds=pd.DataFrame([fold.selected_threshold for fold in folds]),
        threshold_curves=pd.concat(
            [fold.threshold_curve for fold in folds],
            ignore_index=True,
        ),
        outer_fold_metrics=pd.DataFrame([fold.fold_metrics for fold in folds]),
        repeat_metrics=pd.DataFrame([{"repeat": 1, "balanced_accuracy": 1.0}]),
        aggregate_metrics={"primary": 1.0},
        threshold_summary={"median": 0.5},
        duration_seconds=0.1,
    )


def _store(tmp_path: Path, run_id: str = "unit") -> ResearchArtifactStore:
    return ResearchArtifactStore(
        _config(tmp_path),
        plan_hash="a" * 64,
        candidate_hash="b" * 64,
        candidate_source_manifest_hash="d" * 64,
        run_implementation_hash="c" * 64,
        loaded_module_hash="e" * 64,
        run_id=run_id,
    )


def test_store_rejects_overwrite_and_preserves_failure(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(FileExistsError):
        _store(tmp_path)

    store.fail(RuntimeError("intentional"), {"run_id": store.run_id})

    assert (store.root / "_FAILED").exists()
    assert not (store.root / "_SUCCESS").exists()
    assert (store.root / "execution_status.json").exists()


def test_completion_requires_every_outer_fold(tmp_path: Path) -> None:
    store = _store(tmp_path, "incomplete")
    store.save_completed_fold(_fold(1))
    result = _result()
    store.save_final_outputs(result)

    with pytest.raises(ResearchArtifactError, match="expected 2 completed folds"):
        store.complete({"run_id": store.run_id}, result)


def test_successful_store_writes_terminal_success_only(tmp_path: Path) -> None:
    store, result, metadata = build_persisted_test_run(tmp_path)
    store.complete(metadata, result)
    success = json.loads((store.root / "_SUCCESS").read_text(encoding="utf-8"))
    manifest = json.loads(
        (store.root / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert success["artifact_manifest_sha256"] == manifest["manifest_sha256"]

    assert (store.root / "_SUCCESS").exists()
    assert not (store.root / "_FAILED").exists()
    assert (store.root / "metrics/aggregate.json").exists()
