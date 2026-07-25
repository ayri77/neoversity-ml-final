from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.churn_ml.research_manual_lightgbm import fit_predict_manual_candidate


def test_real_candidate_wrapper_exposes_only_active_training_targets() -> None:
    records: dict[str, Any] = {}
    training_positions = np.array([2, 5, 8, 11], dtype=np.int64)
    prediction_positions = np.array([1, 4, 7], dtype=np.int64)
    X_train = pd.DataFrame(
        {
            "row_id": training_positions,
            "category": ["a", "a", "b", "b"],
        },
        index=training_positions,
    )
    y_train = pd.Series([0, 1, 0, 1], index=training_positions, name="y")
    X_prediction = pd.DataFrame(
        {
            "row_id": prediction_positions,
            "category": ["a", "c", "b"],
        },
        index=prediction_positions,
    )

    class RecordingEncoder:
        def fit_transform(
            self,
            features: pd.DataFrame,
            targets: pd.Series,
        ) -> pd.DataFrame:
            records["encoder_fit_rows"] = features["row_id"].tolist()
            records["encoder_fit_targets"] = targets.tolist()
            records["encoder_mapping_categories"] = sorted(
                features["category"].unique().tolist()
            )
            return pd.DataFrame(
                {
                    "row_id": features["row_id"].to_numpy(dtype=float),
                    "category__te": targets.to_numpy(dtype=float),
                }
            )

        def transform(self, features: pd.DataFrame) -> pd.DataFrame:
            records["encoder_transform_rows"] = features["row_id"].tolist()
            return pd.DataFrame(
                {
                    "row_id": features["row_id"].to_numpy(dtype=float),
                    "category__te": np.full(len(features), 0.5),
                }
            )

    class RecordingModel:
        def fit(self, features: pd.DataFrame, targets: pd.Series) -> None:
            records["model_fit_rows"] = features["row_id"].astype(int).tolist()
            records["model_fit_targets"] = targets.tolist()

        def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
            records["model_predict_rows"] = features["row_id"].astype(int).tolist()
            positive = np.linspace(0.2, 0.8, len(features))
            return np.column_stack([1.0 - positive, positive])

    def encoder_factory(**parameters: Any) -> RecordingEncoder:
        records["encoder_parameters"] = parameters
        return RecordingEncoder()

    def model_factory(parameters: dict[str, Any]) -> RecordingModel:
        records["model_parameters"] = parameters
        return RecordingModel()

    contract = {
        "target_encoder": {
            "inner_splits": 2,
            "alpha": 10.0,
            "random_state": 42,
        },
        "lightgbm": {"parameters": {"n_estimators": 1}},
    }

    probabilities = fit_predict_manual_candidate(
        X_train,
        y_train,
        X_prediction,
        contract,
        model_training_positions=training_positions,
        prediction_positions=prediction_positions,
        audit_callback=lambda training, prediction: records.update(
            {
                "audit_training": training.tolist(),
                "audit_prediction": prediction.tolist(),
            }
        ),
        encoder_factory=encoder_factory,
        model_factory=model_factory,
    )

    assert records["audit_training"] == training_positions.tolist()
    assert records["audit_prediction"] == prediction_positions.tolist()
    assert records["encoder_fit_rows"] == training_positions.tolist()
    assert records["encoder_fit_targets"] == y_train.tolist()
    assert records["encoder_mapping_categories"] == ["a", "b"]
    assert records["encoder_transform_rows"] == prediction_positions.tolist()
    assert records["model_fit_rows"] == training_positions.tolist()
    assert records["model_fit_targets"] == y_train.tolist()
    assert records["model_predict_rows"] == prediction_positions.tolist()
    assert probabilities.tolist() == [0.2, 0.5, 0.8]
