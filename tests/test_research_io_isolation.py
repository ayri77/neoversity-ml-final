from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.churn_ml.research_config import load_research_config
from src.churn_ml.research_data import load_research_training_data
from src.churn_ml.research_manual_lightgbm import fit_predict_manual_candidate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs/research/manual_lightgbmprep_r31_nested_rs5x5.yaml"
)
TRAIN_PATH = PROJECT_ROOT / "data/processed/v3_targeted_missingness/X_train.parquet"

pytestmark = pytest.mark.skipif(
    not TRAIN_PATH.exists(),
    reason="The ignored local v3 training dataset is unavailable.",
)


def test_research_execution_reads_no_competition_assets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    original_read_parquet = pd.read_parquet
    original_path_open = Path.open

    def recording_read_parquet(path: object, *args: object, **kwargs: object):
        opened.append(str(path))
        return original_read_parquet(path, *args, **kwargs)

    def recording_path_open(
        path: Path,
        *args: object,
        **kwargs: object,
    ):
        opened.append(str(path))
        return original_path_open(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", recording_read_parquet)
    monkeypatch.setattr(Path, "open", recording_path_open)
    config = load_research_config(CONFIG_PATH, project_root=PROJECT_ROOT)
    data = load_research_training_data(config)
    training_positions = np.arange(40, dtype=np.int64)
    prediction_positions = np.arange(40, 47, dtype=np.int64)

    class FastEncoder:
        def fit_transform(
            self,
            features: pd.DataFrame,
            targets: pd.Series,
        ) -> pd.DataFrame:
            del targets
            return pd.DataFrame({"row_marker": np.arange(len(features), dtype=float)})

        def transform(self, features: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"row_marker": np.arange(len(features), dtype=float)})

    class FastModel:
        def fit(self, features: pd.DataFrame, targets: pd.Series) -> None:
            del features, targets

        def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
            positive = np.linspace(0.2, 0.8, len(features))
            return np.column_stack([1.0 - positive, positive])

    probabilities = fit_predict_manual_candidate(
        data.X.iloc[training_positions],
        data.y.iloc[training_positions],
        data.X.iloc[prediction_positions],
        config.candidate_contract,
        model_training_positions=training_positions,
        prediction_positions=prediction_positions,
        encoder_factory=lambda **parameters: FastEncoder(),
        model_factory=lambda parameters: FastModel(),
    )

    assert len(probabilities) == len(prediction_positions)
    normalized = [value.replace("\\", "/").lower() for value in opened]
    forbidden = (
        "x_test.parquet",
        "sample_submission",
        "/submissions/",
        "artifacts/reference",
        "artifacts/autogluon",
    )
    assert not any(marker in value for value in normalized for marker in forbidden)
