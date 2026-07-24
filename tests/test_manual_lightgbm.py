from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.churn_ml.config import (
    ManualExperimentConfig,
    load_manual_experiment_config,
)
from src.churn_ml.metrics import PredictionResult
from src.churn_ml.manual_lightgbm import (
    FeatureContractError,
    build_submission,
    notebook_threshold_grid,
    ordered_feature_schema_sha256,
    prepare_model_features,
    run_manual_lightgbm_cross_validation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs/experiments/manual_lightgbmprep_r31.yaml"
)


def _small_config(tmp_path: Path) -> ManualExperimentConfig:
    baseline = load_manual_experiment_config(
        CONFIG_PATH,
        project_root=PROJECT_ROOT,
    )
    payload = deepcopy(baseline.payload)
    payload["dataset"].update(
        {
            "expected_train_rows": 20,
            "expected_test_rows": 4,
            "expected_source_feature_count": 4,
        }
    )
    payload["features"].update(
        {
            "drop": ["drop_me"],
            "expected_model_feature_count": 3,
            "expected_categorical_count": 1,
            "categorical": ["cat"],
        }
    )
    source_names = ["num_first", "cat", "num_second", "drop_me"]
    model_names = ["num_first", "cat", "num_second"]
    transformed_names = ["num_first", "num_second", "cat__te"]
    payload["features"].update(
        {
            "expected_source_schema_sha256": ordered_feature_schema_sha256(
                source_names
            ),
            "expected_model_input_schema_sha256": ordered_feature_schema_sha256(
                model_names
            ),
            "expected_transformed_schema_sha256": ordered_feature_schema_sha256(
                transformed_names
            ),
        }
    )
    payload["outer_cv"]["n_splits"] = 2
    payload["target_encoder"]["inner_splits"] = 2
    payload["lightgbm"]["parameters"]["n_estimators"] = 5
    payload["artifacts"]["root"] = str(tmp_path / "artifacts")
    payload["parity"].update(
        {
            "enabled": False,
            "required": False,
            "reference_submission_path": None,
        }
    )
    return ManualExperimentConfig(
        payload=payload,
        source_path=baseline.source_path,
        project_root=baseline.project_root,
    )


def _small_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "num_first": np.arange(20, dtype=float),
            "cat": ["a", "b", "c", "a", "b"] * 4,
            "num_second": np.arange(100, 120, dtype=float),
            "drop_me": np.zeros(20),
        },
        index=pd.Index(np.arange(100, 120), name="source_index"),
    )
    target = pd.Series(
        [0, 1] * 10,
        index=train.index,
        name="y",
    )
    test = pd.DataFrame(
        {
            "num_first": [21.0, 22.0, 23.0, 24.0],
            "cat": ["a", "b", "new", None],
            "num_second": [121.0, 122.0, 123.0, 124.0],
            "drop_me": np.zeros(4),
        },
        index=pd.Index([501, 502, 503, 504], name="source_index"),
    )
    return train, target, test


def test_feature_contract_freezes_notebook_transformed_order(
    tmp_path: Path,
) -> None:
    train, target, test = _small_data()
    config = _small_config(tmp_path)

    X_train, X_test, schema = prepare_model_features(
        train,
        target,
        test,
        config,
    )

    assert X_train.columns.tolist() == ["num_first", "cat", "num_second"]
    assert X_test.columns.tolist() == X_train.columns.tolist()
    assert schema.transformed_feature_names == [
        "num_first",
        "num_second",
        "cat__te",
    ]
    assert schema.source_schema_sha256 == config.payload["features"][
        "expected_source_schema_sha256"
    ]
    assert schema.model_input_schema_sha256 == config.payload["features"][
        "expected_model_input_schema_sha256"
    ]
    assert schema.transformed_schema_sha256 == config.payload["features"][
        "expected_transformed_schema_sha256"
    ]


def test_outer_cv_restores_positions_and_averages_test_probabilities(
    tmp_path: Path,
) -> None:
    train, target, test = _small_data()
    config = _small_config(tmp_path)
    X_train, X_test, _ = prepare_model_features(train, target, test, config)

    validation_positions: list[np.ndarray] = []
    result = run_manual_lightgbm_cross_validation(
        X_train,
        target,
        X_test,
        config,
        on_fold_complete=lambda fold: validation_positions.append(
            fold.validation_positions.copy()
        ),
    )

    combined_positions = np.concatenate(validation_positions)
    assert len(combined_positions) == len(np.unique(combined_positions)) == 20
    np.testing.assert_array_equal(np.sort(combined_positions), np.arange(20))
    assert result.fold_assignments["row_position"].is_unique
    assert (result.fold_assignments["fold"] > 0).all()
    assert result.fold_assignments["fold"].value_counts().sum() == 20
    assert result.oof_predictions["row_position"].tolist() == list(range(20))
    assert result.oof_predictions["row_index"].tolist() == list(range(100, 120))
    assert sorted(result.fold_assignments["fold"].unique().tolist()) == [1, 2]
    np.testing.assert_allclose(
        result.test_predictions["probability"],
        result.test_probabilities_by_fold.mean(axis=1),
    )
    assert result.test_predictions["row_index"].tolist() == [501, 502, 503, 504]


def test_notebook_threshold_grid_is_inclusive(tmp_path: Path) -> None:
    grid = notebook_threshold_grid(_small_config(tmp_path))

    assert len(grid) == 281
    assert grid[0] == 0.02
    assert np.isclose(grid[-1], 0.30)
    assert np.any(np.isclose(grid, 0.096))


def test_submission_preserves_sample_rows_and_order() -> None:
    sample = pd.DataFrame({"index": [9, 3, 7], "y": [0, 0, 0]})
    predictions = pd.DataFrame({"prediction_0_117": [1, 0, 1]})

    submission = build_submission(sample, predictions)

    assert submission["index"].tolist() == [9, 3, 7]
    assert submission["y"].tolist() == [1, 0, 1]
    assert sample["y"].tolist() == [0, 0, 0]


def test_replaced_numeric_feature_name_fails_frozen_schema(
    tmp_path: Path,
) -> None:
    train, target, test = _small_data()
    train = train.rename(columns={"num_first": "num_replaced"})
    test = test.rename(columns={"num_first": "num_replaced"})

    with pytest.raises(FeatureContractError, match="schema hash mismatch"):
        prepare_model_features(train, target, test, _small_config(tmp_path))


def test_count_preserving_feature_swap_fails_frozen_order(
    tmp_path: Path,
) -> None:
    train, target, test = _small_data()
    order = ["num_second", "cat", "num_first", "drop_me"]

    with pytest.raises(FeatureContractError, match="schema hash mismatch"):
        prepare_model_features(
            train[order],
            target,
            test[order],
            _small_config(tmp_path),
        )


def test_transformed_schema_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    train, target, test = _small_data()
    config = _small_config(tmp_path)
    config.payload["features"]["expected_transformed_schema_sha256"] = "0" * 64

    with pytest.raises(FeatureContractError, match="transformed"):
        prepare_model_features(train, target, test, config)


def test_threshold_equality_is_inclusive_for_positive_class() -> None:
    threshold = 0.117
    probabilities = np.array(
        [
            np.nextafter(threshold, -np.inf),
            threshold,
            np.nextafter(threshold, np.inf),
        ]
    )

    result = PredictionResult.from_probabilities(probabilities, threshold)

    np.testing.assert_array_equal(result.predictions, [0, 1, 1])
