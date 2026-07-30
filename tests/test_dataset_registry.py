"""Tests for Dataset Package & Registry v1 using temporary synthetic packages."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.churn_ml.dataset_registry.api import (
    list_dataset_packages,
    resolve_dataset_package,
    validate_dataset_package,
)
from src.churn_ml.dataset_registry.backfill import backfill_legacy_dataset, backfill_legacy_registry
from src.churn_ml.dataset_registry.constants import MANIFEST_FILENAME
from src.churn_ml.dataset_registry.errors import (
    EXIT_INVALID,
    EXIT_REFUSED_OVERWRITE,
    EXIT_SUCCESS,
    EXIT_UNREGISTERED,
    EXIT_UNVERIFIABLE,
    DatasetRegistryError,
)
from src.churn_ml.dataset_registry.manifest import read_manifest, write_manifest
from src.churn_ml.dataset_registry.roles import classify_feature_role
from src.churn_ml.dataset_registry.schema import parse_manifest
from src.churn_ml.dataset_registry_cli import main as registry_main
from tests.dataset_registry_support import (
    V1_ENGINEERED,
    add_v1_features,
    create_synthetic_legacy_tree,
    make_v0_frames,
    register_package,
    snapshot_tree,
    write_package_files,
)


def _registered_tree(tmp_path: Path) -> Path:
    root = create_synthetic_legacy_tree(tmp_path / "processed")
    register_package(
        root,
        "v0_raw_minimal",
        parent_dataset_id=None,
        hypothesis="synthetic v0",
    )
    register_package(
        root,
        "v1_missingness_summary",
        parent_dataset_id="v0_raw_minimal",
        hypothesis="synthetic v1",
        summary_features=V1_ENGINEERED,
    )
    return root


def test_deterministic_manifest_generation(tmp_path: Path) -> None:
    root = create_synthetic_legacy_tree(tmp_path / "processed")
    first = backfill_legacy_dataset(root, "v0_raw_minimal", write=True)
    second_dir = tmp_path / "processed_copy"
    create_synthetic_legacy_tree(second_dir)
    second = backfill_legacy_dataset(second_dir, "v0_raw_minimal", write=True)
    assert first.manifest is not None
    assert second.manifest is not None
    assert first.manifest.to_dict() == second.manifest.to_dict()
    text_a = (root / "v0_raw_minimal" / MANIFEST_FILENAME).read_text(encoding="utf-8")
    text_b = (second_dir / "v0_raw_minimal" / MANIFEST_FILENAME).read_text(
        encoding="utf-8"
    )
    assert text_a == text_b


def test_valid_parent_child_lineage(tmp_path: Path) -> None:
    root = _registered_tree(tmp_path)
    child = resolve_dataset_package(root, "v1_missingness_summary")
    assert child.manifest.parent_dataset_id == "v0_raw_minimal"
    assert child.validation.status == "valid"
    assert child.manifest.row_identity.alignment_status == "proven"


def test_unknown_manifest_keys_fail(tmp_path: Path) -> None:
    root = _registered_tree(tmp_path)
    path = root / "v0_raw_minimal" / MANIFEST_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["extra_field"] = "nope"
    with pytest.raises(DatasetRegistryError, match="unknown keys"):
        parse_manifest(payload)


def test_missing_manifest_keys_fail(tmp_path: Path) -> None:
    root = _registered_tree(tmp_path)
    path = root / "v0_raw_minimal" / MANIFEST_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["schema_hash"]
    with pytest.raises(DatasetRegistryError, match="missing keys"):
        parse_manifest(payload)


def test_changed_parquet_content_invalidates(tmp_path: Path) -> None:
    root = _registered_tree(tmp_path)
    package = root / "v0_raw_minimal"
    frame = pd.read_parquet(package / "X_train.parquet")
    frame.iloc[0, 0] = 999.0
    frame.to_parquet(package / "X_train.parquet", index=False)
    result = validate_dataset_package(package, root=root)
    assert result.status == "invalid"
    assert "content_hashes.X_train" in result.mismatches


def test_changed_target_invalidates(tmp_path: Path) -> None:
    root = _registered_tree(tmp_path)
    package = root / "v0_raw_minimal"
    y = pd.read_parquet(package / "y_train.parquet")
    y.iloc[0, 0] = 1 - int(y.iloc[0, 0])
    y.to_parquet(package / "y_train.parquet", index=False)
    result = validate_dataset_package(package, root=root)
    assert result.status == "invalid"
    assert "target.hash" in result.mismatches


def test_feature_column_reordering_invalidates(tmp_path: Path) -> None:
    root = _registered_tree(tmp_path)
    package = root / "v0_raw_minimal"
    for name in ("X_train.parquet", "X_test.parquet"):
        frame = pd.read_parquet(package / name)
        frame = frame[list(reversed(frame.columns.tolist()))]
        frame.to_parquet(package / name, index=False)
    result = validate_dataset_package(package, root=root)
    assert result.status == "invalid"
    assert "feature_order" in result.mismatches


def test_dtype_change_invalidates(tmp_path: Path) -> None:
    root = _registered_tree(tmp_path)
    package = root / "v0_raw_minimal"
    for name in ("X_train.parquet", "X_test.parquet"):
        frame = pd.read_parquet(package / name)
        frame["f1"] = frame["f1"].astype("float32")
        frame.to_parquet(package / name, index=False)
    result = validate_dataset_package(package, root=root)
    assert result.status == "invalid"
    assert "feature_dtypes" in result.mismatches


def test_train_row_reordering_invalidates(tmp_path: Path) -> None:
    root = _registered_tree(tmp_path)
    package = root / "v0_raw_minimal"
    X = pd.read_parquet(package / "X_train.parquet")
    y = pd.read_parquet(package / "y_train.parquet")
    order = list(reversed(range(len(X))))
    X.iloc[order].reset_index(drop=True).to_parquet(
        package / "X_train.parquet", index=False
    )
    y.iloc[order].reset_index(drop=True).to_parquet(
        package / "y_train.parquet", index=False
    )
    result = validate_dataset_package(package, root=root)
    assert result.status == "invalid"
    assert any(item.startswith("row_identity") for item in result.mismatches)


def test_test_row_reordering_invalidates(tmp_path: Path) -> None:
    root = _registered_tree(tmp_path)
    package = root / "v0_raw_minimal"
    X = pd.read_parquet(package / "X_test.parquet")
    order = list(reversed(range(len(X))))
    X.iloc[order].reset_index(drop=True).to_parquet(
        package / "X_test.parquet", index=False
    )
    result = validate_dataset_package(package, root=root)
    assert result.status == "invalid"
    assert any(item.startswith("row_identity") for item in result.mismatches)


def test_row_count_mismatch_invalidates(tmp_path: Path) -> None:
    root = _registered_tree(tmp_path)
    package = root / "v0_raw_minimal"
    X = pd.read_parquet(package / "X_train.parquet")
    y = pd.read_parquet(package / "y_train.parquet")
    X = pd.concat([X, X.iloc[[0]]], ignore_index=True)
    y = pd.concat([y, y.iloc[[0]]], ignore_index=True)
    X.to_parquet(package / "X_train.parquet", index=False)
    y.to_parquet(package / "y_train.parquet", index=False)
    result = validate_dataset_package(package, root=root)
    assert result.status == "invalid"
    assert "train_row_count" in result.mismatches


def test_schema_hash_mismatch_invalidates(tmp_path: Path) -> None:
    root = _registered_tree(tmp_path)
    path = root / "v0_raw_minimal" / MANIFEST_FILENAME
    manifest = read_manifest(path)
    payload = manifest.to_dict()
    payload["schema_hash"] = "0" * 64
    write_manifest(path, parse_manifest(payload))
    result = validate_dataset_package(root / "v0_raw_minimal", root=root)
    assert result.status == "invalid"
    assert "schema_hash" in result.mismatches


def test_inability_to_prove_row_alignment(tmp_path: Path) -> None:
    root = create_synthetic_legacy_tree(tmp_path / "processed")
    # Break the child so the ordered v0 projection no longer matches the parent.
    child = root / "v1_missingness_summary"
    X_train = pd.read_parquet(child / "X_train.parquet")
    X_train["f1"] = X_train["f1"] + 1.0
    X_train.to_parquet(child / "X_train.parquet", index=False)
    result = backfill_legacy_dataset(root, "v1_missingness_summary", write=False)
    assert result.status == "unverifiable_alignment"


def test_refusal_to_overwrite_existing_manifest(tmp_path: Path) -> None:
    root = create_synthetic_legacy_tree(tmp_path / "processed")
    first = backfill_legacy_dataset(root, "v0_raw_minimal", write=True)
    assert first.wrote is True
    results = backfill_legacy_registry(root, dataset_ids=["v0_raw_minimal"], write=True)
    assert results[0].status == "refused_overwrite"
    code = registry_main(
        ["backfill", "--root", str(root), "--dataset-id", "v0_raw_minimal", "--write"]
    )
    assert code == EXIT_REFUSED_OVERWRITE


def test_discovery_registered_unregistered_invalid(tmp_path: Path) -> None:
    root = _registered_tree(tmp_path)
    X_train, y_train, X_test = make_v0_frames(seed=7)
    write_package_files(
        root / "orphan_unregistered",
        dataset_id="orphan_unregistered",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
    )
    write_package_files(
        root / "v1_basic_clean",
        dataset_id="v1_basic_clean",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
    )
    broken = root / "v0_raw_minimal"
    payload = json.loads((broken / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    payload["schema_hash"] = "1" * 64
    (broken / MANIFEST_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    entries = {entry.dataset_id: entry for entry in list_dataset_packages(root)}
    assert entries["v1_missingness_summary"].status == "valid"
    assert entries["orphan_unregistered"].status == "unregistered"
    assert entries["v1_basic_clean"].status == "unregistered"
    assert entries["v0_raw_minimal"].status == "invalid"


def test_scan_validate_inspect_dry_run_backfill_do_not_write(tmp_path: Path) -> None:
    root = create_synthetic_legacy_tree(tmp_path / "processed")
    before = snapshot_tree(root)
    assert registry_main(["scan", "--root", str(root)]) in {
        EXIT_SUCCESS,
        EXIT_UNREGISTERED,
    }
    assert before == snapshot_tree(root)
    assert registry_main(
        ["validate", "--root", str(root), "--dataset-id", "v0_raw_minimal"]
    ) == EXIT_UNREGISTERED
    assert before == snapshot_tree(root)
    assert registry_main(
        ["inspect", "--root", str(root), "--dataset-id", "v0_raw_minimal"]
    ) == EXIT_UNREGISTERED
    assert before == snapshot_tree(root)
    code = registry_main(
        ["backfill", "--root", str(root), "--dataset-id", "v0_raw_minimal"]
    )
    assert code == EXIT_SUCCESS
    assert before == snapshot_tree(root)
    assert not (root / "v0_raw_minimal" / MANIFEST_FILENAME).exists()


def test_backfill_write_and_validate_roundtrip(tmp_path: Path) -> None:
    root = create_synthetic_legacy_tree(tmp_path / "processed")
    results = backfill_legacy_registry(
        root,
        dataset_ids=["v0_raw_minimal", "v1_missingness_summary"],
        write=True,
    )
    assert all(result.status == "valid" and result.wrote for result in results)
    child = results[1].manifest
    assert child is not None
    assert child.parent_dataset_id == "v0_raw_minimal"
    roles = {feature.name: feature.role for feature in child.features}
    assert roles["f1"] == "numeric"
    assert roles["f3"] == "categorical"
    assert roles["missing_count_total"] == "summary"
    assert roles["missing_rate_total"] == "summary"
    assert (
        resolve_dataset_package(root, "v1_missingness_summary").validation.status
        == "valid"
    )


def test_strict_acceptance_of_all_four_roles() -> None:
    payload = {
        "schema_version": "dataset_package_v1",
        "dataset_id": "synthetic_roles",
        "parent_dataset_id": None,
        "hypothesis": "role acceptance",
        "files": {
            "X_train": "X_train.parquet",
            "y_train": "y_train.parquet",
            "X_test": "X_test.parquet",
            "metadata": "metadata.json",
        },
        "train_row_count": 1,
        "test_row_count": 1,
        "n_features": 4,
        "features": [
            {"name": "a", "dtype": "float64", "role": "numeric"},
            {"name": "b", "dtype": "object", "role": "categorical"},
            {"name": "c_is_missing", "dtype": "int8", "role": "binary_indicator"},
            {"name": "missing_count_total", "dtype": "int16", "role": "summary"},
        ],
        "transformations": [{"type": "synthetic"}],
        "target_dependency": "none",
        "schema_hash": "0" * 64,
        "content_hashes": {
            "X_train": "1" * 64,
            "y_train": "2" * 64,
            "X_test": "3" * 64,
            "metadata": "4" * 64,
        },
        "target": {
            "name": "y",
            "dtype": "int64",
            "class_counts": {"0": 1},
            "hash": "5" * 64,
        },
        "row_identity": {
            "train_hash": "6" * 64,
            "test_hash": "7" * 64,
            "alignment_status": "proven",
            "alignment_method": "ordered_v0_raw_minimal_feature_projection",
            "train_anchor_hash": "8" * 64,
            "test_anchor_hash": "9" * 64,
        },
    }
    manifest = parse_manifest(payload)
    assert [feature.role for feature in manifest.features] == [
        "numeric",
        "categorical",
        "binary_indicator",
        "summary",
    ]


@pytest.mark.parametrize("role", ["feature", "engineered", "unknown", ""])
def test_rejection_of_old_and_unknown_roles(role: str) -> None:
    payload = {
        "schema_version": "dataset_package_v1",
        "dataset_id": "synthetic_roles",
        "parent_dataset_id": None,
        "hypothesis": "role rejection",
        "files": {
            "X_train": "X_train.parquet",
            "y_train": "y_train.parquet",
            "X_test": "X_test.parquet",
            "metadata": "metadata.json",
        },
        "train_row_count": 1,
        "test_row_count": 1,
        "n_features": 1,
        "features": [{"name": "a", "dtype": "float64", "role": role}],
        "transformations": [{"type": "synthetic"}],
        "target_dependency": "none",
        "schema_hash": "0" * 64,
        "content_hashes": {
            "X_train": "1" * 64,
            "y_train": "2" * 64,
            "X_test": "3" * 64,
            "metadata": "4" * 64,
        },
        "target": {
            "name": "y",
            "dtype": "int64",
            "class_counts": {"0": 1},
            "hash": "5" * 64,
        },
        "row_identity": {
            "train_hash": "6" * 64,
            "test_hash": "7" * 64,
            "alignment_status": "proven",
            "alignment_method": "ordered_v0_raw_minimal_feature_projection",
            "train_anchor_hash": "8" * 64,
            "test_anchor_hash": "9" * 64,
        },
    }
    with pytest.raises(DatasetRegistryError):
        parse_manifest(payload)


def test_deterministic_role_classification_precedence() -> None:
    # Declared summary wins even when dtype is numeric.
    assert (
        classify_feature_role(
            "missing_count_total",
            "int16",
            summary_features={"missing_count_total"},
        )
        == "summary"
    )
    # Declared indicator wins over numeric dtype.
    assert (
        classify_feature_role(
            "Var217_is_missing",
            "int8",
            binary_indicator_features={"Var217_is_missing"},
        )
        == "binary_indicator"
    )
    # Suffix rule applies only when enabled.
    assert (
        classify_feature_role("f1_is_missing", "int8", use_missing_suffix=True)
        == "binary_indicator"
    )
    assert classify_feature_role("f1_is_missing", "int8") == "numeric"
    assert classify_feature_role("city", "object") == "categorical"
    assert classify_feature_role("amount", "float64") == "numeric"
    with pytest.raises(DatasetRegistryError, match="Unsupported feature dtype"):
        classify_feature_role("weird", "complex128")


def test_legacy_backfill_classifies_categorical_summary_and_indicators(
    tmp_path: Path,
) -> None:
    root = tmp_path / "processed"
    root.mkdir()
    X_train, y_train, X_test = make_v0_frames()
    write_package_files(
        root / "v0_raw_minimal",
        dataset_id="v0_raw_minimal",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        metadata={"source": "raw"},
    )
    v1_train = add_v1_features(X_train)
    v1_test = add_v1_features(X_test)
    write_package_files(
        root / "v1_missingness_summary",
        dataset_id="v1_missingness_summary",
        X_train=v1_train,
        y_train=y_train,
        X_test=v1_test,
        metadata={"base_version": "v0_raw_minimal", "source": "v0_raw_minimal"},
    )
    v2_train = v1_train.copy()
    v2_test = v1_test.copy()
    v2_train["f1_is_missing"] = 0
    v2_test["f1_is_missing"] = 0
    # Keep only v2 summary subset plus indicator for this synthetic package.
    for frame in (v2_train, v2_test):
        drop_cols = [
            name
            for name in ("missing_rate_total",)
            if name in frame.columns
        ]
        frame.drop(columns=drop_cols, inplace=True)
    write_package_files(
        root / "v2_missingness_indicators",
        dataset_id="v2_missingness_indicators",
        X_train=v2_train,
        y_train=y_train,
        X_test=v2_test,
        metadata={"base_version": "v0_raw_minimal", "source": "v0_raw_minimal"},
    )
    v3_train = add_v1_features(X_train)
    v3_test = add_v1_features(X_test)
    for frame in (v3_train, v3_test):
        frame["Var217_is_missing"] = 0
        frame["Var126_is_missing"] = 0
        frame["Var218_is_missing"] = 0
        frame["Var192_is_missing"] = 0
    write_package_files(
        root / "v3_targeted_missingness",
        dataset_id="v3_targeted_missingness",
        X_train=v3_train,
        y_train=y_train,
        X_test=v3_test,
        metadata={"base_version": "v1_missingness_summary", "source": "v1_missingness_summary"},
    )

    results = backfill_legacy_registry(
        root,
        dataset_ids=[
            "v0_raw_minimal",
            "v1_missingness_summary",
            "v2_missingness_indicators",
            "v3_targeted_missingness",
        ],
        write=False,
    )
    assert all(result.status == "valid" for result in results)
    by_id = {result.dataset_id: result.manifest for result in results}
    assert by_id["v0_raw_minimal"] is not None
    v0_roles = {f.name: f.role for f in by_id["v0_raw_minimal"].features}
    assert v0_roles == {"f1": "numeric", "f2": "numeric", "f3": "categorical"}
    assert [f.name for f in by_id["v0_raw_minimal"].features] == ["f1", "f2", "f3"]

    assert by_id["v1_missingness_summary"] is not None
    v1_roles = {f.name: f.role for f in by_id["v1_missingness_summary"].features}
    assert v1_roles["missing_count_total"] == "summary"
    assert v1_roles["f3"] == "categorical"

    assert by_id["v2_missingness_indicators"] is not None
    v2_roles = {f.name: f.role for f in by_id["v2_missingness_indicators"].features}
    assert v2_roles["missing_count_total"] == "summary"
    assert v2_roles["f1_is_missing"] == "binary_indicator"

    assert by_id["v3_targeted_missingness"] is not None
    v3_roles = {f.name: f.role for f in by_id["v3_targeted_missingness"].features}
    assert v3_roles["Var217_is_missing"] == "binary_indicator"
    assert v3_roles["missing_rate_total"] == "summary"

    # Determinism across repeated dry-runs.
    again = backfill_legacy_dataset(root, "v3_targeted_missingness", write=False)
    assert again.manifest is not None
    assert again.manifest.to_dict() == by_id["v3_targeted_missingness"].to_dict()


def test_role_tampering_invalidates_schema_hash(tmp_path: Path) -> None:
    root = _registered_tree(tmp_path)
    path = root / "v0_raw_minimal" / MANIFEST_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Tamper role without updating schema_hash.
    for feature in payload["features"]:
        if feature["name"] == "f3":
            feature["role"] = "numeric"
            break
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = validate_dataset_package(root / "v0_raw_minimal", root=root)
    assert result.status == "invalid"
    assert "schema_hash" in result.mismatches


def test_ordered_feature_schema_preserved_in_backfill(tmp_path: Path) -> None:
    root = create_synthetic_legacy_tree(tmp_path / "processed")
    result = backfill_legacy_dataset(root, "v1_missingness_summary", write=False)
    assert result.manifest is not None
    names = [feature.name for feature in result.manifest.features]
    assert names == list(pd.read_parquet(root / "v1_missingness_summary" / "X_train.parquet").columns)


def test_v3_target_dependency_exploratory_in_catalog() -> None:
    from src.churn_ml.dataset_registry.constants import LEGACY_CATALOG

    assert LEGACY_CATALOG["v3_targeted_missingness"]["target_dependency"] == (
        "exploratory"
    )


def test_cli_exit_codes_for_invalid_package(tmp_path: Path) -> None:
    root = _registered_tree(tmp_path)
    package = root / "v0_raw_minimal"
    frame = pd.read_parquet(package / "X_train.parquet")
    frame.iloc[0, 0] = -123.0
    frame.to_parquet(package / "X_train.parquet", index=False)
    code = registry_main(
        ["validate", "--root", str(root), "--dataset-id", "v0_raw_minimal"]
    )
    assert code == EXIT_INVALID


def test_cli_exit_code_unverifiable(tmp_path: Path) -> None:
    root = create_synthetic_legacy_tree(tmp_path / "processed")
    child = root / "v1_missingness_summary"
    X_train = pd.read_parquet(child / "X_train.parquet")
    X_train = X_train.drop(columns=["f1"])
    X_test = pd.read_parquet(child / "X_test.parquet").drop(columns=["f1"])
    X_train.to_parquet(child / "X_train.parquet", index=False)
    X_test.to_parquet(child / "X_test.parquet", index=False)
    code = registry_main(
        [
            "backfill",
            "--root",
            str(root),
            "--dataset-id",
            "v1_missingness_summary",
        ]
    )
    assert code == EXIT_UNVERIFIABLE
