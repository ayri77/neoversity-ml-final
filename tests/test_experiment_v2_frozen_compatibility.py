from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.churn_ml.config import EXPECTED_LIGHTGBM_PARAMETERS
from src.churn_ml.experiment_v2 import get_candidate_adapter
from src.churn_ml.experiment_v2_schema import ordered_feature_schema_sha256
from src.churn_ml.research_data import canonical_sha256
from src.churn_ml.target_encoding import AutoGluonBinaryOOFTargetEncoder


FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "experiment_v2_compatibility_v1.json"
)


def test_production_v2_adapter_matches_independent_frozen_train_only_fixture() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    columns = fixture["source_columns"]
    X_train = pd.DataFrame(fixture["training_data"], columns=columns)
    y_train = pd.Series(fixture["training_labels"], name="y", dtype="int64")
    X_prediction = pd.DataFrame(fixture["prediction_data"], columns=columns)
    contract = fixture["adapter_contract"]

    assert contract["lightgbm"]["parameters"] == EXPECTED_LIGHTGBM_PARAMETERS
    encoder_contract = contract["target_encoder"]
    encoder = AutoGluonBinaryOOFTargetEncoder(
        n_splits=encoder_contract["inner_splits"],
        alpha=encoder_contract["alpha"],
        random_state=encoder_contract["random_state"],
    )
    encoded = encoder.fit_transform(X_train, y_train)
    encoded_payload = {
        "columns": encoded.columns.tolist(),
        "values": encoded.to_numpy().tolist(),
    }
    assert encoded.columns.tolist() == fixture["expected_transformed_columns"]
    assert (
        ordered_feature_schema_sha256(encoded.columns.tolist())
        == fixture["expected_transformed_schema_sha256"]
    )
    assert (
        canonical_sha256(encoded_payload) == fixture["expected_encoded_values_sha256"]
    )

    adapter = get_candidate_adapter("manual_lightgbm_te_v1_compat")
    probabilities = adapter.fit_predict(
        X_train,
        y_train,
        X_prediction,
        contract,
        model_training_positions=np.arange(len(X_train), dtype=np.int64),
        prediction_positions=np.arange(
            len(X_train),
            len(X_train) + len(X_prediction),
            dtype=np.int64,
        ),
    )
    np.testing.assert_allclose(
        probabilities,
        fixture["expected_probabilities"],
        rtol=0.0,
        atol=1e-15,
    )
    assert (
        canonical_sha256(probabilities.tolist())
        == fixture["expected_probabilities_sha256"]
    )
