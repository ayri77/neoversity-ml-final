from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from src.churn_ml.features import (
    PreparedDataset,
    add_zero_value_summary_features,
    load_dataset,
    save_dataset,
    select_supported_zero_features,
    deduplicate_zero_mask_features,
    add_missingness_pattern_feature,
    deduplicate_missingness_mask_features,
    add_selected_zero_indicators,
)


def test_select_supported_zero_features_without_target() -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 0.0, None],
            "feature_b": [0.0, 2.0, 3.0, 4.0],
            "category": ["a", "b", "c", "d"],
        }
    )

    selected = select_supported_zero_features(
        dataframe,
        candidate_features=["feature_a", "feature_b"],
        min_present=3,
        min_zero=2,
        min_nonzero=1,
    )

    assert selected == ["feature_a"]


def test_add_zero_value_summary_features() -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 0.0, None],
            "feature_b": [0.0, 2.0, 3.0, 4.0],
            "category": ["a", "b", "c", "d"],
        }
    )

    transformed = add_zero_value_summary_features(
        dataframe,
        numeric_features=["feature_a", "feature_b"],
        supported_zero_features=["feature_a"],
    )

    assert_series_equal(
        transformed["zero_count_numeric"],
        pd.Series([2, 0, 1, 0], dtype="int16"),
        check_names=False,
    )

    assert_series_equal(
        transformed["zero_rate_observed_numeric"],
        pd.Series(
            [1.0, 0.0, 0.5, 0.0],
            dtype="float32",
        ),
        check_names=False,
    )

    assert_series_equal(
        transformed["zero_count_supported_numeric"],
        pd.Series([1, 0, 1, 0], dtype="int16"),
        check_names=False,
    )

    assert_series_equal(
        transformed["zero_rate_observed_supported_numeric"],
        pd.Series(
            [1.0, 0.0, 1.0, 0.0],
            dtype="float32",
        ),
        check_names=False,
    )

    assert list(dataframe.columns) == [
        "feature_a",
        "feature_b",
        "category",
    ]


def test_dataset_manifest_round_trip(
    tmp_path: Path,
) -> None:
    X_train = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 2.0],
            "category": ["a", "b", "c"],
        }
    )

    y_train = pd.Series(
        [0, 1, 0],
        name="y",
    )

    X_test = pd.DataFrame(
        {
            "feature_a": [3.0, 4.0],
            "category": ["d", "e"],
        }
    )

    dataset = PreparedDataset(
        version="test_dataset",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        metadata={
            "description": "Test dataset.",
        },
        manifest={
            "dataset_id": "test_dataset",
            "parent_dataset_id": None,
            "hypothesis": "Manifest round-trip test.",
            "transformations": [
                "No feature transformations.",
            ],
            "target_dependency": "none",
        },
    )

    processed_dir = tmp_path / "data" / "processed"

    dataset_path = save_dataset(
        dataset,
        processed_dir,
    )

    assert (dataset_path / "X_train.parquet").exists()

    assert (dataset_path / "y_train.parquet").exists()

    assert (dataset_path / "X_test.parquet").exists()

    assert (dataset_path / "metadata.json").exists()

    manifest_path = dataset_path / "manifest.json"

    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["dataset_id"] == "test_dataset"
    assert manifest["parent_dataset_id"] is None
    assert manifest["target_dependency"] == "none"
    assert manifest["row_alignment_check"] == "passed"
    assert manifest["n_train_rows"] == 3
    assert manifest["n_test_rows"] == 2
    assert manifest["n_features"] == 2
    assert manifest["schema_hash"]
    assert manifest["train_content_hash"]
    assert manifest["test_content_hash"]
    assert manifest["target_hash"]

    loaded = load_dataset(
        "test_dataset",
        processed_dir,
    )

    assert_frame_equal(
        loaded.X_train,
        X_train,
    )

    assert_series_equal(
        loaded.y_train,
        y_train,
    )

    assert_frame_equal(
        loaded.X_test,
        X_test,
    )

    assert loaded.manifest["dataset_id"] == "test_dataset"


def test_deduplicate_zero_mask_features() -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 0.0, None],
            "feature_b": [0.0, 2.0, 0.0, None],
            "feature_c": [1.0, 0.0, 2.0, None],
            # Same zero mask as feature_a, but different
            # observation mask.
            "feature_d": [0.0, 3.0, 0.0, 4.0],
        }
    )

    representatives, groups = deduplicate_zero_mask_features(
        dataframe,
        features=[
            "feature_a",
            "feature_b",
            "feature_c",
            "feature_d",
        ],
    )

    assert representatives == [
        "feature_a",
        "feature_c",
        "feature_d",
    ]

    assert groups == [
        ["feature_a", "feature_b"],
        ["feature_c"],
        ["feature_d"],
    ]


def test_deduplicate_missingness_mask_features() -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [1.0, None, 3.0, None],
            "feature_b": ["a", None, "b", None],
            "feature_c": [None, 2.0, 3.0, None],
            "feature_d": [1.0, 2.0, 3.0, 4.0],
        }
    )

    representatives, groups = deduplicate_missingness_mask_features(
        dataframe,
        features=[
            "feature_a",
            "feature_b",
            "feature_c",
            "feature_d",
        ],
    )

    assert representatives == [
        "feature_a",
        "feature_c",
        "feature_d",
    ]

    assert groups == [
        ["feature_a", "feature_b"],
        ["feature_c"],
        ["feature_d"],
    ]


def test_add_missingness_pattern_feature() -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [1.0, None, None],
            "feature_b": [None, None, 2.0],
            "feature_c": [10, 20, 30],
        }
    )

    result = add_missingness_pattern_feature(
        dataframe,
        source_features=[
            "feature_a",
            "feature_b",
        ],
    )

    assert list(result["missingness_pattern_id"]) == [
        "02",
        "03",
        "01",
    ]

    assert str(result["missingness_pattern_id"].dtype) == "string"

    pd.testing.assert_frame_equal(
        result[dataframe.columns],
        dataframe,
    )


def test_add_selected_zero_indicators() -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, None, -1.0],
            "feature_b": [2, 0, 3, 0],
            "other_feature": ["a", "b", "c", "d"],
        }
    )

    result = add_selected_zero_indicators(
        dataframe,
        indicator_features=[
            "feature_a",
            "feature_b",
        ],
    )

    assert result["feature_a_is_zero"].tolist() == [
        1,
        0,
        0,
        0,
    ]

    assert result["feature_b_is_zero"].tolist() == [
        0,
        1,
        0,
        1,
    ]

    assert str(result["feature_a_is_zero"].dtype) == "int8"
    assert str(result["feature_b_is_zero"].dtype) == "int8"

    pd.testing.assert_frame_equal(
        result[dataframe.columns],
        dataframe,
    )
