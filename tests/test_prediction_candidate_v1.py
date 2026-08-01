"""Focused tests for prediction_candidate_v1 contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.churn_ml.prediction_candidates.contract_v1 import (
    CANDIDATE_ROOT_RELATIVE,
    MANIFEST_FILENAME,
    OOF_COLUMNS,
    OOF_FILENAME,
    SCHEMA_VERSION,
    SUCCESS_FILENAME,
    TEST_COLUMNS,
    TEST_FILENAME,
    UNAVAILABLE,
    CandidateConflictError,
    CandidateValidationError,
    PredictionCandidateError,
    build_candidate_id,
    create_candidate_package,
    discover_candidate_packages,
    load_aligned_oof_probabilities,
    load_aligned_test_probabilities,
    load_candidate_manifest,
    load_candidate_package,
    resolve_under_repository,
    validate_candidate_package,
    validate_probability_series,
)
from tests.dataset_registry_support import (
    create_synthetic_legacy_tree,
    register_package,
    snapshot_tree,
)


IDENTITY_BASE = {
    "schema_version": SCHEMA_VERSION,
    "source_kind": "unit_test_source_v1",
    "dataset_id": "v0_raw_minimal",
    "source_run_path": "artifacts/fake_runs/run-1",
    "source_config_sha256": "a" * 64,
    "source_model_name": "ModelA",
    "oof_protocol": "unit_test_oof_protocol",
    "positive_class_label": 1,
    "probability_semantics": "P(y=1)",
}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    processed = root / "data" / "processed"
    create_synthetic_legacy_tree(processed)
    register_package(
        processed,
        "v0_raw_minimal",
        parent_dataset_id=None,
        hypothesis="synthetic v0",
    )
    register_package(
        processed,
        "v1_missingness_summary",
        parent_dataset_id="v0_raw_minimal",
        hypothesis="synthetic v1",
        summary_features=("missing_count_total", "missing_rate_total"),
    )
    (root / "artifacts" / "fake_runs" / "run-1").mkdir(parents=True)
    return root


def _frames(repo: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = pd.read_parquet(
        repo / "data" / "processed" / "v0_raw_minimal" / "y_train.parquet"
    ).iloc[:, 0]
    n_train = len(y)
    n_test = len(
        pd.read_parquet(
            repo / "data" / "processed" / "v0_raw_minimal" / "X_test.parquet"
        )
    )
    rng = np.random.default_rng(0)
    oof = pd.DataFrame(
        {
            "row_position": np.arange(n_train, dtype=np.int64),
            "target": y.astype("int64").to_numpy(),
            "probability_positive": rng.uniform(0.05, 0.95, size=n_train),
        }
    )
    test = pd.DataFrame(
        {
            "row_position": np.arange(n_test, dtype=np.int64),
            "probability_positive": rng.uniform(0.05, 0.95, size=n_test),
        }
    )
    return oof, test


def _manifest_fields() -> dict[str, object]:
    return {
        "dataset_id": "v0_raw_minimal",
        "exploratory": False,
        "source_run_path": "artifacts/fake_runs/run-1",
        "source_config_path": "artifacts/fake_runs/run-1/resolved_config.yaml",
        "source_config_sha256": "a" * 64,
        "source_model_name": "ModelA",
        "source_model_type": "UnitTestModel",
        "source_predictor_path": "artifacts/fake_runs/run-1/predictor",
        "source_autogluon_version": UNAVAILABLE,
        "source_metric_name": "balanced_accuracy",
        "source_metric_value": 0.75,
        "source_leaderboard_metadata": {"model": "ModelA", "score_val": 0.75},
        "provenance": {"adapter": "unit_test"},
    }


def _create(repo: Path, **overrides: object):
    oof, test = _frames(repo)
    identity = dict(IDENTITY_BASE)
    fields = _manifest_fields()
    source_metadata = {"schema_version": 1, "note": "unit"}
    if "oof" in overrides:
        oof = overrides.pop("oof")  # type: ignore[assignment]
    if "test" in overrides:
        test = overrides.pop("test")  # type: ignore[assignment]
    if "identity" in overrides:
        identity = overrides.pop("identity")  # type: ignore[assignment]
    if "manifest_fields" in overrides:
        fields = overrides.pop("manifest_fields")  # type: ignore[assignment]
    if "source_metadata" in overrides:
        source_metadata = overrides.pop("source_metadata")  # type: ignore[assignment]
    assert not overrides
    return create_candidate_package(
        repository_root=repo,
        identity=identity,
        oof=oof,
        test=test,
        manifest_fields=fields,
        source_metadata=source_metadata,
    )


def test_valid_candidate_package(repo: Path) -> None:
    package = _create(repo)
    result = validate_candidate_package(package, repository_root=repo)
    assert result["ok"] is True
    assert package.candidate_id.startswith("pc1_")
    assert (package.package_dir / SUCCESS_FILENAME).is_file()
    assert list(package.oof.columns) == list(OOF_COLUMNS)
    assert list(package.test.columns) == list(TEST_COLUMNS)
    oof = load_aligned_oof_probabilities(package)
    test = load_aligned_test_probabilities(package)
    assert len(oof) == package.manifest["train_row_count"]
    assert len(test) == package.manifest["test_row_count"]


def test_exact_manifest_schema(repo: Path) -> None:
    package = _create(repo)
    manifest = load_candidate_manifest(package.package_dir / MANIFEST_FILENAME)
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["candidate_type"] == "prediction_candidate"
    assert manifest["oof_prediction_reference"]["columns"] == list(OOF_COLUMNS)
    assert manifest["test_prediction_reference"]["columns"] == list(TEST_COLUMNS)
    assert "\\" not in manifest["oof_prediction_reference"]["path"]
    assert not Path(manifest["oof_prediction_reference"]["path"]).is_absolute()


def test_deterministic_candidate_id() -> None:
    first = build_candidate_id(IDENTITY_BASE)
    second = build_candidate_id(IDENTITY_BASE)
    assert first == second
    altered = dict(IDENTITY_BASE)
    altered["source_model_name"] = "ModelB"
    assert build_candidate_id(altered) != first


def test_duplicate_train_rows_rejected(repo: Path) -> None:
    oof, test = _frames(repo)
    oof.loc[1, "row_position"] = 0
    with pytest.raises(CandidateValidationError, match="duplicate"):
        create_candidate_package(
            repository_root=repo,
            identity=IDENTITY_BASE,
            oof=oof,
            test=test,
            manifest_fields=_manifest_fields(),
            source_metadata={},
        )


def test_missing_train_rows_rejected(repo: Path) -> None:
    oof, test = _frames(repo)
    oof = oof.iloc[1:].reset_index(drop=True)
    with pytest.raises(CandidateValidationError, match="row count|missing"):
        create_candidate_package(
            repository_root=repo,
            identity=IDENTITY_BASE,
            oof=oof,
            test=test,
            manifest_fields=_manifest_fields(),
            source_metadata={},
        )


def test_target_mismatch_rejected(repo: Path) -> None:
    oof, test = _frames(repo)
    oof.loc[0, "target"] = 1 - int(oof.loc[0, "target"])
    with pytest.raises(CandidateValidationError, match="target"):
        create_candidate_package(
            repository_root=repo,
            identity=IDENTITY_BASE,
            oof=oof,
            test=test,
            manifest_fields=_manifest_fields(),
            source_metadata={},
        )


def test_train_anchor_mismatch_rejected(repo: Path) -> None:
    package = _create(repo)
    package.manifest["train_anchor_hash"] = "b" * 64
    with pytest.raises(CandidateValidationError, match="train_anchor"):
        validate_candidate_package(package, repository_root=repo)


def test_test_anchor_mismatch_rejected(repo: Path) -> None:
    package = _create(repo)
    package.manifest["test_anchor_hash"] = "c" * 64
    with pytest.raises(CandidateValidationError, match="test_anchor"):
        validate_candidate_package(package, repository_root=repo)


def test_nonfinite_probability_rejected() -> None:
    with pytest.raises(CandidateValidationError, match="non-finite"):
        validate_probability_series(
            pd.Series([0.1, np.nan, 0.2]),
            field_name="probability_positive",
        )


def test_out_of_range_probability_rejected() -> None:
    with pytest.raises(CandidateValidationError, match="outside"):
        validate_probability_series(
            pd.Series([0.1, 1.2, 0.2]),
            field_name="probability_positive",
        )


def test_label_values_cannot_masquerade_as_probabilities() -> None:
    with pytest.raises(CandidateValidationError, match="masquerade|floating-point"):
        validate_probability_series(
            pd.Series([0, 1, 0, 1], dtype="int64"),
            field_name="probability_positive",
        )
    with pytest.raises(CandidateValidationError, match="masquerade"):
        validate_probability_series(
            pd.Series([0.0, 1.0, 0.0, 1.0], dtype="float64"),
            field_name="probability_positive",
        )


def test_atomic_write_success_last(repo: Path) -> None:
    package = _create(repo)
    success = package.package_dir / SUCCESS_FILENAME
    others = [
        package.package_dir / MANIFEST_FILENAME,
        package.package_dir / OOF_FILENAME,
        package.package_dir / TEST_FILENAME,
    ]
    success_mtime = success.stat().st_mtime_ns
    assert all(path.stat().st_mtime_ns <= success_mtime for path in others)


def test_identical_import_idempotent(repo: Path) -> None:
    first = _create(repo)
    second = _create(repo)
    assert first.candidate_id == second.candidate_id
    assert first.package_dir == second.package_dir
    assert file_bytes(first.package_dir / OOF_FILENAME) == file_bytes(
        second.package_dir / OOF_FILENAME
    )


def test_conflicting_import_blocked(repo: Path) -> None:
    _create(repo)
    oof, test = _frames(repo)
    oof["probability_positive"] = np.clip(
        oof["probability_positive"] + 0.01, 0.01, 0.99
    )
    with pytest.raises(CandidateConflictError, match="different content"):
        create_candidate_package(
            repository_root=repo,
            identity=IDENTITY_BASE,
            oof=oof,
            test=test,
            manifest_fields=_manifest_fields(),
            source_metadata={"schema_version": 1, "note": "unit"},
        )


def test_path_traversal_rejected(repo: Path) -> None:
    with pytest.raises(PredictionCandidateError, match="traversal"):
        resolve_under_repository("../secrets.txt", repo)
    with pytest.raises(PredictionCandidateError, match="traversal|Absolute"):
        resolve_under_repository("/etc/passwd", repo)


def test_source_artifacts_remain_unchanged(repo: Path) -> None:
    source = repo / "artifacts" / "fake_runs" / "run-1"
    marker = source / "source.bin"
    marker.write_bytes(b"do-not-touch")
    before = snapshot_tree(source)
    _create(repo)
    after = snapshot_tree(source)
    assert before == after


def test_discover_and_reload(repo: Path) -> None:
    package = _create(repo)
    discovered = discover_candidate_packages(repo)
    assert len(discovered) == 1
    assert discovered[0]["candidate_id"] == package.candidate_id
    loaded = load_candidate_package(package.package_dir, repository_root=repo)
    assert loaded.candidate_id == package.candidate_id
    assert CANDIDATE_ROOT_RELATIVE in discovered[0]["package_path"]


def file_bytes(path: Path) -> bytes:
    return path.read_bytes()
