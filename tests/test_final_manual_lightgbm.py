from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.churn_ml.final_artifact_validation import (
    FinalArtifactValidationError,
    verify_reloaded_inference,
)
from src.churn_ml.final_data import FinalData, build_final_predictions
from src.churn_ml.final_manual_lightgbm import FinalFitResult
from src.churn_ml.manual_lightgbm import FeatureSchema
from src.churn_ml.final_promotion import OperationalThreshold


class IdentityEncoder:
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.copy()


class FirstColumnProbabilityModel:
    def __init__(self, delta: float = 0.0) -> None:
        self.delta = delta

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        positive = frame.iloc[:, 0].to_numpy(dtype=np.float64) + self.delta
        return np.column_stack([1.0 - positive, positive])


def _objects() -> tuple[
    FinalData, FinalFitResult, OperationalThreshold, pd.DataFrame, pd.DataFrame
]:
    probabilities = np.linspace(0.0, 1.0, 2500, dtype=np.float64)
    test = pd.DataFrame({"probability_feature": probabilities})
    sample = pd.DataFrame(
        {
            "index": np.arange(2500, dtype=np.int64),
            "y": np.zeros(2500, dtype=np.int64),
        }
    )
    schema = FeatureSchema(
        source_feature_names=["probability_feature"],
        dropped_features=[],
        model_feature_names=["probability_feature"],
        categorical_features=[],
        numerical_features=["probability_feature"],
        transformed_feature_names=["probability_feature"],
    )
    data = FinalData(
        X_train=pd.DataFrame(),
        y=pd.Series(dtype="int64"),
        X_test=test,
        sample_submission=sample,
        feature_schema=schema,
        fingerprints={},
        submission_schema={},
    )
    fit = FinalFitResult(
        encoder=IdentityEncoder(),  # type: ignore[arg-type]
        model=FirstColumnProbabilityModel(),
        encoded_train=pd.DataFrame(),
        encoded_test=test,
        test_probabilities=probabilities,
        duration_seconds=0.0,
    )
    threshold = OperationalThreshold(
        averaged_oos=pd.DataFrame(),
        threshold_curve=pd.DataFrame(),
        selected_threshold=0.107,
        diagnostic_balanced_accuracy=0.9,
        averaged_oos_sha256="0" * 64,
        selection_record={},
    )
    predictions, submission = build_final_predictions(sample, probabilities, 0.107)
    return data, fit, threshold, predictions, submission


def test_saved_encoder_and_model_reload_with_exact_inference(tmp_path: Path) -> None:
    data, fit, threshold, predictions, submission = _objects()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    joblib.dump(fit.encoder, model_dir / "target_encoder.joblib")
    joblib.dump(fit.model, model_dir / "lightgbm_classifier.joblib")

    record = verify_reloaded_inference(
        tmp_path,
        data,
        fit,
        threshold,
        predictions,
        submission,
    )

    assert record["encoded_test_equal"] is True
    assert record["probabilities_equal"] is True
    assert record["comparison_tolerance"] is None


def test_reload_rejects_any_probability_difference(tmp_path: Path) -> None:
    data, fit, threshold, predictions, submission = _objects()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    joblib.dump(fit.encoder, model_dir / "target_encoder.joblib")
    joblib.dump(
        FirstColumnProbabilityModel(delta=np.finfo(np.float64).eps),
        model_dir / "lightgbm_classifier.joblib",
    )

    with pytest.raises(FinalArtifactValidationError, match="exactly equal"):
        verify_reloaded_inference(
            tmp_path,
            data,
            fit,
            threshold,
            predictions,
            submission,
        )
