from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.churn_ml.final_data import FinalData, FinalDataError
from src.churn_ml.target_encoding import AutoGluonBinaryOOFTargetEncoder


@dataclass(frozen=True)
class FinalFitResult:
    encoder: AutoGluonBinaryOOFTargetEncoder
    model: Any
    encoded_train: pd.DataFrame
    encoded_test: pd.DataFrame
    test_probabilities: np.ndarray
    duration_seconds: float


def fit_final_manual_lightgbm(
    data: FinalData,
    candidate_contract: dict[str, Any],
) -> FinalFitResult:
    """Fit exactly one full-data model using leakage-safe OOF train encodings."""
    encoder_contract = candidate_contract["target_encoder"]
    expected_encoder = {
        "implementation": "custom_autogluon_compatible_binary_oof_target_encoder",
        "inner_splits": 5,
        "shuffle": True,
        "random_state": 42,
        "alpha": 10.0,
        "prior": "unweighted_mean_of_category_level_target_means",
        "keep_original_categorical_features": False,
    }
    if encoder_contract != expected_encoder:
        raise FinalDataError("Final encoder contract differs from approved candidate.")

    parameters = dict(candidate_contract["lightgbm"]["parameters"])
    if parameters.get("n_estimators") != 376:
        raise FinalDataError("Final LightGBM model must use exactly 376 rounds.")
    forbidden = {"early_stopping_rounds", "callbacks"}
    if forbidden.intersection(parameters):
        raise FinalDataError("Final model parameters contain tuning callbacks.")

    start = perf_counter()
    encoder = AutoGluonBinaryOOFTargetEncoder(
        n_splits=5,
        alpha=10.0,
        random_state=42,
    )
    encoded_train = encoder.fit_transform(
        data.X_train.reset_index(drop=True),
        data.y.reset_index(drop=True),
    )
    encoded_test = encoder.transform(data.X_test.reset_index(drop=True))
    expected_columns = data.feature_schema.transformed_feature_names
    if (
        encoded_train.columns.tolist() != expected_columns
        or encoded_test.columns.tolist() != expected_columns
    ):
        raise FinalDataError("Encoded feature order differs from frozen schema.")
    if encoded_train.shape != (10000, 213) or encoded_test.shape != (2500, 213):
        raise FinalDataError("Encoded final-data shapes differ from contract.")

    model = lgb.LGBMClassifier(**parameters)
    model.fit(encoded_train, data.y.reset_index(drop=True))
    matrix = np.asarray(model.predict_proba(encoded_test), dtype=np.float64)
    if matrix.shape != (2500, 2):
        raise FinalDataError("Final model did not return 2500 binary probabilities.")
    probabilities = matrix[:, 1]
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise FinalDataError("Final test probabilities are invalid.")
    return FinalFitResult(
        encoder=encoder,
        model=model,
        encoded_train=encoded_train,
        encoded_test=encoded_test,
        test_probabilities=probabilities,
        duration_seconds=perf_counter() - start,
    )
