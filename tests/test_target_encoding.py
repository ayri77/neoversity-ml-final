from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.churn_ml.target_encoding import AutoGluonBinaryOOFTargetEncoder


def _training_data() -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.DataFrame(
        {
            "num_a": np.arange(8, dtype=float),
            "cat_a": ["A", "A", "A", "A", "B", "B", "C", "C"],
            "num_b": np.arange(10, 18, dtype=float),
            "cat_b": ["x", "y", "x", "y", "x", "y", "x", "y"],
        }
    )
    target = pd.Series([1, 1, 1, 1, 0, 0, 0, 0], name="y")
    return frame, target


def test_encoder_preserves_notebook_column_order_and_replacement() -> None:
    frame, target = _training_data()
    encoder = AutoGluonBinaryOOFTargetEncoder(
        n_splits=2,
        alpha=10.0,
        random_state=42,
    )

    encoded = encoder.fit_transform(frame, target)

    assert encoded.columns.tolist() == [
        "num_a",
        "num_b",
        "cat_a__te",
        "cat_b__te",
    ]
    np.testing.assert_array_equal(encoded["num_a"], frame["num_a"])
    np.testing.assert_array_equal(encoded["num_b"], frame["num_b"])
    assert encoded[["cat_a__te", "cat_b__te"]].notna().all().all()


def test_training_encoding_uses_exact_inner_oof_values() -> None:
    frame, target = _training_data()
    encoder = AutoGluonBinaryOOFTargetEncoder(
        n_splits=2,
        alpha=10.0,
        random_state=42,
    )

    encoded = encoder.fit_transform(frame[["cat_a"]], target)

    # Seed 42 yields inner training positions [1, 3, 5, 7] and
    # [0, 2, 4, 6]. Each has A: (count=2, mean=1), B/C:
    # (count=1, mean=0), and the unweighted category prior is 1/3.
    expected_oof = np.array(
        [4 / 9, 4 / 9, 4 / 9, 4 / 9, 10 / 33, 10 / 33, 10 / 33, 10 / 33]
    )
    # A full-target mapping would instead produce 11/21 for A and
    # 5/18 for B/C, so this assertion detects leakage through a full fit.
    leaked_full_mapping = np.array(
        [11 / 21, 11 / 21, 11 / 21, 11 / 21, 5 / 18, 5 / 18, 5 / 18, 5 / 18]
    )
    np.testing.assert_allclose(encoded["cat_a__te"], expected_oof)
    assert not np.allclose(encoded["cat_a__te"], leaked_full_mapping)


def test_unweighted_category_prior_handles_missing_and_unseen() -> None:
    frame, target = _training_data()
    encoder = AutoGluonBinaryOOFTargetEncoder(
        n_splits=2,
        alpha=10.0,
        random_state=42,
    )
    encoder.fit_transform(frame, target)
    probe = pd.DataFrame(
        {
            "num_a": [100.0, 101.0],
            "cat_a": ["unseen", None],
            "num_b": [200.0, 201.0],
            "cat_b": ["x", None],
        }
    )

    transformed = encoder.transform(probe)

    expected_prior = np.mean([1.0, 0.0, 0.0])
    assert encoder.encodings_["cat_a"]["global_mean"] == expected_prior
    np.testing.assert_allclose(
        transformed["cat_a__te"],
        [expected_prior, expected_prior],
    )
    assert transformed.loc[1, "cat_b__te"] == encoder.encodings_["cat_b"][
        "global_mean"
    ]


def test_encoder_state_round_trips_with_joblib(tmp_path: Path) -> None:
    frame, target = _training_data()
    encoder = AutoGluonBinaryOOFTargetEncoder(n_splits=2)
    encoder.fit_transform(frame, target)
    path = tmp_path / "encoder.joblib"
    joblib.dump(encoder, path)

    loaded = joblib.load(path)

    pd.testing.assert_frame_equal(loaded.transform(frame), encoder.transform(frame))
