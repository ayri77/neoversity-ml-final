from __future__ import annotations

import pandas as pd

from src.churn_ml.research_protocol import aggregate_repeat_metrics


def test_single_repeat_reports_undefined_sample_standard_deviation_as_null() -> None:
    metrics = pd.DataFrame(
        {
            "balanced_accuracy": [0.9],
            "sensitivity": [0.9],
            "specificity": [0.9],
            "tn": [90],
            "fp": [10],
            "fn": [10],
            "tp": [90],
            "predicted_positive_rate": [0.5],
            "roc_auc": [0.95],
            "average_precision": [0.85],
            "brier_score": [0.1],
        }
    )
    fold_metrics = pd.concat([metrics, metrics], ignore_index=True)

    aggregate = aggregate_repeat_metrics(metrics, fold_metrics)

    assert (
        aggregate["metrics"]["balanced_accuracy"]["sample_standard_deviation"] is None
    )
