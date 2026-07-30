"""Tests for prepared-dataset feature helpers and Registry-converged persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from src.churn_ml.dataset_registry.build_spec import DatasetBuildSpec
from src.churn_ml.dataset_registry.constants import (
    V1_ENGINEERED_FEATURES,
    V3_ENGINEERED_FEATURES,
    V4_SUMMARY_FEATURES,
)
from src.churn_ml.dataset_registry.errors import (
    DatasetRegistryError,
    OverwriteRefusedError,
)
from src.churn_ml.dataset_registry.materialize import dataset_manifest_sha256
from src.churn_ml.features import (
    PreparedDataset,
    add_missingness_pattern_feature,
    add_missingness_summary_features,
    add_selected_missing_indicators,
    add_selected_zero_indicators,
    add_zero_value_summary_features,
    deduplicate_missingness_mask_features,
    deduplicate_zero_mask_features,
    load_dataset,
    native_build_spec,
    save_dataset,
    select_supported_zero_features,
)


def _frames() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X_train = pd.DataFrame(
        {
            "f1": [0.0, 1.0, 2.0, 0.0],
            "f2": [1.0, 0.0, 1.0, 2.0],
            "category": ["a", "b", "c", "a"],
        }
    )
    y_train = pd.Series([0, 1, 0, 1], name="y", dtype="int64")
    X_test = pd.DataFrame(
        {
            "f1": [3.0, 0.0],
            "f2": [0.0, 4.0],
            "category": ["d", "e"],
        }
    )
    return X_train, y_train, X_test


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
    assert list(dataframe.columns) == ["feature_a", "feature_b", "category"]


def test_deduplicate_zero_mask_features() -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 0.0, None],
            "feature_b": [0.0, 2.0, 0.0, None],
            "feature_c": [1.0, 0.0, 2.0, None],
            "feature_d": [0.0, 3.0, 0.0, 4.0],
        }
    )
    representatives, groups = deduplicate_zero_mask_features(
        dataframe,
        features=["feature_a", "feature_b", "feature_c", "feature_d"],
    )
    assert representatives == ["feature_a", "feature_c", "feature_d"]
    assert groups == [["feature_a", "feature_b"], ["feature_c"], ["feature_d"]]


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


def test_native_save_load_round_trip_registry_valid(tmp_path: Path) -> None:
    X_train, y_train, X_test = _frames()
    processed = tmp_path / "processed"
    dataset = PreparedDataset(
        version="v0_raw_minimal",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        metadata={"description": "synthetic v0"},
        build_spec=native_build_spec("v0_raw_minimal"),
    )
    path = save_dataset(dataset, processed)
    assert (path / "dataset_manifest.json").is_file()
    assert not (path / "manifest.json").exists()
    loaded = load_dataset("v0_raw_minimal", processed)
    assert loaded.dataset_manifest is not None
    assert loaded.dataset_manifest.dataset_id == "v0_raw_minimal"
    assert loaded.dataset_manifest.parent_dataset_id is None
    assert loaded.dataset_manifest.row_identity.alignment_status == "proven"
    assert_frame_equal(loaded.X_train, X_train)
    assert_series_equal(loaded.y_train, y_train)
    assert "created_from_git" in loaded.metadata


def test_calculated_fields_cannot_be_spoofed() -> None:
    from src.churn_ml.dataset_registry.build_spec import parse_build_spec

    with pytest.raises(DatasetRegistryError, match="calculated fields"):
        parse_build_spec(
            {
                "dataset_id": "v0_raw_minimal",
                "parent_dataset_id": None,
                "hypothesis": "x",
                "transformations": [{"type": "drop_columns"}],
                "target_dependency": "none",
                "schema_hash": "0" * 64,
                "row_alignment_check": "passed",
            }
        )


def test_parent_child_alignment_and_roles(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    X_train, y_train, X_test = _frames()
    save_dataset(
        PreparedDataset(
            version="v0_raw_minimal",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            metadata={"source": "raw"},
            build_spec=native_build_spec("v0_raw_minimal"),
        ),
        processed,
    )
    v1_train = add_missingness_summary_features(X_train)
    v1_test = add_missingness_summary_features(X_test)
    path = save_dataset(
        PreparedDataset(
            version="v1_missingness_summary",
            X_train=v1_train,
            y_train=y_train,
            X_test=v1_test,
            metadata={"source": "v0_raw_minimal"},
            build_spec=native_build_spec("v1_missingness_summary"),
        ),
        processed,
    )
    loaded = load_dataset("v1_missingness_summary", processed)
    assert loaded.dataset_manifest is not None
    roles = {f.name: f.role for f in loaded.dataset_manifest.features}
    for name in V1_ENGINEERED_FEATURES:
        assert roles[name] == "summary"
    assert roles["category"] == "categorical"
    assert roles["f1"] == "numeric"
    assert loaded.dataset_manifest.target_dependency == "none"
    assert (path / "dataset_manifest.json").is_file()


def test_v3_exploratory_and_v4_target_independent_roles(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    X_train, y_train, X_test = _frames()
    # Expand base columns to include v3 indicator sources.
    for frame in (X_train, X_test):
        frame["Var217"] = frame["f1"]
        frame["Var126"] = frame["f2"]
        frame["Var218"] = frame["f1"]
        frame["Var192"] = frame["f2"]
    save_dataset(
        PreparedDataset(
            version="v0_raw_minimal",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            metadata={"source": "raw"},
            build_spec=native_build_spec("v0_raw_minimal"),
        ),
        processed,
    )
    v1_train = add_missingness_summary_features(X_train)
    v1_test = add_missingness_summary_features(X_test)
    save_dataset(
        PreparedDataset(
            version="v1_missingness_summary",
            X_train=v1_train,
            y_train=y_train,
            X_test=v1_test,
            metadata={"source": "v0_raw_minimal"},
            build_spec=native_build_spec("v1_missingness_summary"),
        ),
        processed,
    )
    v3_train = add_selected_missing_indicators(
        v1_train,
        indicator_features=["Var217", "Var126", "Var218", "Var192"],
    )
    v3_test = add_selected_missing_indicators(
        v1_test,
        indicator_features=["Var217", "Var126", "Var218", "Var192"],
    )
    save_dataset(
        PreparedDataset(
            version="v3_targeted_missingness",
            X_train=v3_train,
            y_train=y_train,
            X_test=v3_test,
            metadata={"source": "v1_missingness_summary"},
            build_spec=native_build_spec("v3_targeted_missingness"),
        ),
        processed,
    )
    v3 = load_dataset("v3_targeted_missingness", processed)
    assert v3.dataset_manifest is not None
    assert v3.dataset_manifest.target_dependency == "exploratory"
    roles = {f.name: f.role for f in v3.dataset_manifest.features}
    for name in V3_ENGINEERED_FEATURES:
        assert roles[name] == "binary_indicator"
    for name in V1_ENGINEERED_FEATURES:
        assert roles[name] == "summary"

    v4_train = add_zero_value_summary_features(
        X_train,
        numeric_features=["f1", "f2", "Var217", "Var126", "Var218", "Var192"],
        supported_zero_features=["f1"],
    )
    v4_test = add_zero_value_summary_features(
        X_test,
        numeric_features=["f1", "f2", "Var217", "Var126", "Var218", "Var192"],
        supported_zero_features=["f1"],
    )
    save_dataset(
        PreparedDataset(
            version="v4_zero_value_summary",
            X_train=v4_train,
            y_train=y_train,
            X_test=v4_test,
            metadata={"source": "v0_raw_minimal"},
            build_spec=native_build_spec("v4_zero_value_summary"),
        ),
        processed,
    )
    v4 = load_dataset("v4_zero_value_summary", processed)
    assert v4.dataset_manifest is not None
    assert v4.dataset_manifest.target_dependency == "none"
    roles_v4 = {f.name: f.role for f in v4.dataset_manifest.features}
    for name in V4_SUMMARY_FEATURES:
        assert roles_v4[name] == "summary"


def test_fold_local_materialization_rejected(tmp_path: Path) -> None:
    X_train, y_train, X_test = _frames()
    build_spec = DatasetBuildSpec(
        dataset_id="fold_local_demo",
        parent_dataset_id=None,
        hypothesis="should fail",
        transformations=[{"type": "fold_local_te"}],
        target_dependency="fold_local",
    )
    with pytest.raises(DatasetRegistryError, match="fold_local"):
        save_dataset(
            PreparedDataset(
                version="fold_local_demo",
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                metadata={},
                build_spec=build_spec,
            ),
            tmp_path / "processed",
        )
    assert not (tmp_path / "processed" / "fold_local_demo").exists()


def test_train_target_index_mismatch_rejected(tmp_path: Path) -> None:
    X_train, y_train, X_test = _frames()
    y_bad = y_train.copy()
    y_bad.index = pd.RangeIndex(1, len(y_bad) + 1)
    with pytest.raises(ValueError, match="indices must be identical"):
        save_dataset(
            PreparedDataset(
                version="v0_raw_minimal",
                X_train=X_train,
                y_train=y_bad,
                X_test=X_test,
                metadata={},
                build_spec=native_build_spec("v0_raw_minimal"),
            ),
            tmp_path / "processed",
        )


def test_train_test_dtype_or_order_mismatch_rejected(tmp_path: Path) -> None:
    X_train, y_train, X_test = _frames()
    X_test_bad = X_test[["category", "f1", "f2"]]
    with pytest.raises(ValueError, match="same columns"):
        save_dataset(
            PreparedDataset(
                version="v0_raw_minimal",
                X_train=X_train,
                y_train=y_train,
                X_test=X_test_bad,
                metadata={},
                build_spec=native_build_spec("v0_raw_minimal"),
            ),
            tmp_path / "processed",
        )
    X_test_dtype = X_test.copy()
    X_test_dtype["f1"] = X_test_dtype["f1"].astype("float32")
    with pytest.raises(ValueError, match="identical dtypes"):
        save_dataset(
            PreparedDataset(
                version="v0_raw_minimal",
                X_train=X_train,
                y_train=y_train,
                X_test=X_test_dtype,
                metadata={},
                build_spec=native_build_spec("v0_raw_minimal"),
            ),
            tmp_path / "processed",
        )


def test_registered_overwrite_refused(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    X_train, y_train, X_test = _frames()
    dataset = PreparedDataset(
        version="v0_raw_minimal",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        metadata={},
        build_spec=native_build_spec("v0_raw_minimal"),
    )
    save_dataset(dataset, processed)
    with pytest.raises(OverwriteRefusedError):
        save_dataset(dataset, processed, overwrite=True)


def test_tampering_invalidates_loading(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    X_train, y_train, X_test = _frames()
    save_dataset(
        PreparedDataset(
            version="v0_raw_minimal",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            metadata={},
            build_spec=native_build_spec("v0_raw_minimal"),
        ),
        processed,
    )
    package = processed / "v0_raw_minimal"
    frame = pd.read_parquet(package / "X_train.parquet")
    frame.iloc[0, 0] = 99.0
    frame.to_parquet(package / "X_train.parquet", index=False)
    with pytest.raises(DatasetRegistryError):
        load_dataset("v0_raw_minimal", processed)


def test_legacy_compatibility_explicit_and_unregistered(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    package = processed / "legacy_only"
    package.mkdir(parents=True)
    X_train, y_train, X_test = _frames()
    X_train.to_parquet(package / "X_train.parquet", index=False)
    y_train.to_frame(name="y").to_parquet(package / "y_train.parquet", index=False)
    X_test.to_parquet(package / "X_test.parquet", index=False)
    (package / "metadata.json").write_text(
        json.dumps({"version": "legacy_only", "metadata": {"legacy": True}}),
        encoding="utf-8",
    )
    loaded = load_dataset("legacy_only", processed)
    assert loaded.dataset_manifest is None
    (package / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DatasetRegistryError, match="manifest.json is not the Registry"):
        load_dataset("legacy_only", processed)


def test_manifest_sha256_deterministic(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    X_train, y_train, X_test = _frames()
    save_dataset(
        PreparedDataset(
            version="v0_raw_minimal",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            metadata={},
            build_spec=native_build_spec("v0_raw_minimal"),
        ),
        processed,
    )
    package = processed / "v0_raw_minimal"
    first = dataset_manifest_sha256(package)
    second = dataset_manifest_sha256(package)
    assert first == second
    assert len(first) == 64
