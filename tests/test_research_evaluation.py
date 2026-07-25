from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.churn_ml.research_evaluation import run_research_evaluation
from src.churn_ml.research_protocol import build_evaluation_assignments


def _plan() -> dict:
    return {
        "outer_evaluation": {
            "n_splits": 2,
            "shuffle": True,
            "repeat_seeds": [0, 17],
        },
        "threshold_selection": {
            "n_splits": 2,
            "shuffle": True,
            "random_state": 314159,
        },
        "threshold_policy": {
            "minimum": 0.01,
            "maximum": 0.99,
            "step": 0.001,
            "maximizer_absolute_tolerance": 1e-12,
            "constant_probability_fallback": 0.5,
        },
    }


def test_nested_evaluation_keeps_prediction_rows_out_of_model_fitting() -> None:
    y = pd.Series(([0, 1] * 20), name="y")
    X = pd.DataFrame({"row": np.arange(len(y), dtype=float)})
    plan = _plan()
    assignments = build_evaluation_assignments(y, plan)
    fit_calls: list[tuple[set[int], set[int]]] = []
    threshold_calls: list[tuple[set[int], set[int]]] = []

    def fake_fit_predict(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_prediction: pd.DataFrame,
        candidate_contract: dict,
        *,
        model_training_positions: np.ndarray,
        prediction_positions: np.ndarray,
        audit_callback: Callable | None,
    ) -> np.ndarray:
        if audit_callback is not None:
            audit_callback(model_training_positions, prediction_positions)
        return np.where(
            X_prediction["row"].to_numpy(dtype=int) % 2 == 1,
            0.8,
            0.2,
        )

    def fit_audit(training: np.ndarray, prediction: np.ndarray) -> None:
        fit_calls.append((set(training), set(prediction)))
        assert not set(training) & set(prediction)

    def threshold_audit(
        repeat: int,
        outer_fold: int,
        threshold_rows: np.ndarray,
        outer_validation_rows: np.ndarray,
    ) -> None:
        threshold_calls.append((set(threshold_rows), set(outer_validation_rows)))
        assert not set(threshold_rows) & set(outer_validation_rows)

    result = run_research_evaluation(
        X,
        y,
        assignments,
        plan,
        {"unused": True},
        fit_predict=fake_fit_predict,
        fit_audit_callback=fit_audit,
        threshold_audit_callback=threshold_audit,
    )

    assert len(fit_calls) == 12
    assert len(threshold_calls) == 4
    assert len(result.outer_validation) == 80
    assert len(result.threshold_selection_oof) == 80
    for repeat in (1, 2):
        repeat_frame = result.outer_validation.loc[
            result.outer_validation["repeat"] == repeat
        ]
        assert repeat_frame["row_position"].value_counts().eq(1).all()
    assert (result.outer_fold_metrics["balanced_accuracy"] == 1.0).all()
    assert len(result.repeat_metrics) == 2
