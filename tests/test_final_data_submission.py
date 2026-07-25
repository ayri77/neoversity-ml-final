from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.churn_ml.final_data import FinalDataError, build_final_predictions


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "index": np.arange(2500, dtype=np.int64),
            "y": np.zeros(2500, dtype=np.int64),
        }
    )


def test_submission_preserves_order_binary_dtype_and_exact_greater_equal() -> None:
    sample = _sample()
    probabilities = np.zeros(2500, dtype=np.float64)
    probabilities[:3] = [
        np.nextafter(0.107, -np.inf),
        0.107,
        np.nextafter(0.107, np.inf),
    ]

    predictions, submission = build_final_predictions(
        sample,
        probabilities,
        0.107,
    )

    assert predictions["prediction"].iloc[:3].tolist() == [0, 1, 1]
    assert submission.columns.tolist() == ["index", "y"]
    assert submission["index"].tolist() == sample["index"].tolist()
    assert submission["y"].iloc[:3].tolist() == [0, 1, 1]
    assert pd.api.types.is_integer_dtype(submission["y"])
    assert sample["y"].sum() == 0


@pytest.mark.parametrize(
    "probabilities",
    [
        np.zeros(2499),
        np.concatenate([np.zeros(2499), [np.nan]]),
        np.concatenate([np.zeros(2499), [np.inf]]),
        np.concatenate([np.zeros(2499), [-0.01]]),
        np.concatenate([np.zeros(2499), [1.01]]),
    ],
)
def test_submission_rejects_missing_or_invalid_probabilities(
    probabilities: np.ndarray,
) -> None:
    with pytest.raises(FinalDataError, match="probabilit"):
        build_final_predictions(_sample(), probabilities, 0.107)


def test_submission_rejects_duplicate_official_rows() -> None:
    sample = _sample()
    sample.loc[2499, "index"] = sample.loc[2498, "index"]

    with pytest.raises(FinalDataError, match="duplicates"):
        build_final_predictions(sample, np.zeros(2500), 0.107)


def test_submission_rejects_changed_sample_column_order() -> None:
    sample = _sample()[["y", "index"]]

    with pytest.raises(FinalDataError, match="columns"):
        build_final_predictions(sample, np.zeros(2500), 0.107)
