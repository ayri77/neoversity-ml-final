from __future__ import annotations

from collections.abc import Callable
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.churn_ml.target_encoding import AutoGluonBinaryOOFTargetEncoder


EncoderFactory = Callable[..., Any]
ModelFactory = Callable[[dict[str, Any]], Any]


def _default_encoder_factory(**parameters: Any) -> Any:
    return AutoGluonBinaryOOFTargetEncoder(**parameters)


def _default_model_factory(parameters: dict[str, Any]) -> Any:
    return lgb.LGBMClassifier(**parameters)


FitAuditCallback = Callable[[np.ndarray, np.ndarray], None]


def fit_predict_manual_candidate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_prediction: pd.DataFrame,
    candidate_contract: dict[str, Any],
    *,
    model_training_positions: np.ndarray,
    prediction_positions: np.ndarray,
    audit_callback: FitAuditCallback | None = None,
    encoder_factory: EncoderFactory = _default_encoder_factory,
    model_factory: ModelFactory = _default_model_factory,
) -> np.ndarray:
    """Fit one manual candidate partition and predict its untouched rows."""
    if audit_callback is not None:
        audit_callback(
            np.asarray(model_training_positions, dtype=np.int64),
            np.asarray(prediction_positions, dtype=np.int64),
        )
    encoder_config = candidate_contract["target_encoder"]
    encoder = encoder_factory(
        n_splits=int(encoder_config["inner_splits"]),
        alpha=float(encoder_config["alpha"]),
        random_state=int(encoder_config["random_state"]),
    )
    encoded_train = encoder.fit_transform(
        X_train.reset_index(drop=True),
        y_train.reset_index(drop=True),
    )
    encoded_prediction = encoder.transform(X_prediction.reset_index(drop=True))
    model = model_factory(dict(candidate_contract["lightgbm"]["parameters"]))
    model.fit(encoded_train, y_train.reset_index(drop=True))
    probability_matrix = np.asarray(model.predict_proba(encoded_prediction))
    if probability_matrix.ndim != 2 or probability_matrix.shape[1] != 2:
        raise RuntimeError("Manual candidate did not return binary probabilities.")
    return probability_matrix[:, 1].astype(float, copy=False)
