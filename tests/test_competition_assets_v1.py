"""Focused tests for authenticated competition submission assets.

Synthetic fixtures only. No Streamlit session, no Optuna, no Kaggle upload.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.churn_ml.competition_assets_v1 import (
    DEFAULT_ID_COLUMN,
    DEFAULT_TARGET_COLUMN,
    ROW_IDENTITY_ARTIFACT_PATH,
    SAMPLE_SUBMISSION_PATH,
    ensure_competition_row_identity,
    require_resolved_competition_draft,
    resolve_competition_assets,
    submission_ids_from_identity,
    write_competition_asset_registry,
)
from src.churn_ml.control_panel.deployment_candidates import (
    UNRESOLVED_SAMPLE_SUBMISSION_PATH,
)
from src.churn_ml.control_panel.deployment_draft_builder import (
    prepare_deployment_draft,
)
from src.churn_ml.control_panel.launch import LaunchAuthorizationError, authorize_launch
from src.churn_ml.control_panel.registry import load_registry
from src.churn_ml.deployment_v1_contracts import (
    CONFIG_KEYS,
    CONFIG_OPTIONAL_KEYS,
    DeploymentConfig,
    load_deployment_config,
)
from src.churn_ml.deployment_v1_features import (
    DeploymentDataError,
    _validate_sample_and_alignment,
)
from src.churn_ml.research_data import canonical_sha256
from tests.test_control_panel_deployment_workflow import (
    APPROVED_AT,
    APPROVER,
    DATASET_ID,
    _make_dataset_package,
    _make_run,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
N_ROWS = 8
TEST_ANCHOR = "a1" * 32


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _seed_competition_files(root: Path, *, n_rows: int = N_ROWS) -> None:
    sample = pd.DataFrame(
        {DEFAULT_ID_COLUMN: list(range(n_rows)), DEFAULT_TARGET_COLUMN: [0] * n_rows}
    )
    sample_path = root / Path(*Path(SAMPLE_SUBMISSION_PATH).parts)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(sample_path, index=False, lineterminator="\n")

    test = pd.DataFrame(
        {
            "feature_one": [float(i) for i in range(n_rows)],
            "feature_two": [float(i * 2) for i in range(n_rows)],
        }
    )
    test_path = root / "data" / "raw" / "final_proj_test.csv"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test.to_csv(test_path, index=False, lineterminator="\n")


def _seed_package_with_alignment(root: Path, *, n_rows: int = N_ROWS) -> None:
    _write_json(
        root / "data" / "processed" / DATASET_ID / "dataset_manifest.json",
        {
            "schema_version": "dataset_package_v1",
            "dataset_id": DATASET_ID,
            "target_dependency": "none",
            "schema_hash": "5c" * 32,
            "train_row_count": n_rows * 2,
            "test_row_count": n_rows,
            "files": {"X_test": "X_test.parquet"},
            "content_hashes": {"X_test": TEST_ANCHOR},
            "features": [
                {"name": "feature_one", "dtype": "float64"},
                {"name": "feature_two", "dtype": "float64"},
            ],
            "row_identity": {
                "alignment_status": "proven",
                "test_anchor_hash": TEST_ANCHOR,
                "test_row_count": n_rows,
            },
        },
    )


def test_resolve_registers_sample_and_builds_row_identity(tmp_path: Path) -> None:
    _seed_competition_files(tmp_path)
    _seed_package_with_alignment(tmp_path)
    resolution = ensure_competition_row_identity(tmp_path, dataset_version=DATASET_ID)
    assert resolution.ready is True
    assert resolution.sample_submission_path == SAMPLE_SUBMISSION_PATH
    assert resolution.id_column == DEFAULT_ID_COLUMN
    assert resolution.target_column == DEFAULT_TARGET_COLUMN
    assert resolution.sample_expected_rows == N_ROWS
    assert resolution.ordered_id_sha256 == canonical_sha256(list(range(N_ROWS)))
    assert (tmp_path / ROW_IDENTITY_ARTIFACT_PATH).is_file()
    registry_path = write_competition_asset_registry(tmp_path, resolution)
    assert (tmp_path / registry_path).is_file()
    payload = yaml.safe_load((tmp_path / registry_path).read_text(encoding="utf-8"))
    assert payload["sample_submission"]["columns"] == [
        DEFAULT_ID_COLUMN,
        DEFAULT_TARGET_COLUMN,
    ]
    assert payload["sample_submission"]["ordered_id_sha256"] == (
        resolution.ordered_id_sha256
    )


def test_feature_matrix_and_submission_ids_stay_separate(tmp_path: Path) -> None:
    _seed_competition_files(tmp_path)
    _seed_package_with_alignment(tmp_path)
    resolution = ensure_competition_row_identity(tmp_path, dataset_version=DATASET_ID)
    features = pd.DataFrame(
        {
            "feature_one": list(range(N_ROWS)),
            "feature_two": list(range(N_ROWS)),
        }
    )
    sample = pd.read_csv(tmp_path / SAMPLE_SUBMISSION_PATH)
    config = DeploymentConfig(
        payload={
            "sample_submission": {
                "id_column": DEFAULT_ID_COLUMN,
                "target_column": DEFAULT_TARGET_COLUMN,
                "expected_rows": N_ROWS,
            },
            "test_data": {},
            "submission_row_identity": {
                "path": resolution.row_identity_path,
                "sha256": resolution.row_identity_sha256,
                "expected_rows": N_ROWS,
                "id_column": DEFAULT_ID_COLUMN,
                "id_dtype": "int64",
                "id_semantics": "zero_based_row_position",
                "ordered_id_sha256": resolution.ordered_id_sha256,
                "row_position_identity_sha256": resolution.ordered_id_sha256,
                "test_anchor_hash": TEST_ANCHOR,
            },
        },
        source_path=tmp_path / "unused.yaml",
        project_root=tmp_path,
    )
    keys, identity = _validate_sample_and_alignment(
        features, sample, config, enforce_configured_rows=True
    )
    assert keys == tuple(range(N_ROWS))
    assert identity == resolution.ordered_id_sha256
    assert DEFAULT_ID_COLUMN not in features.columns

    poisoned = features.copy()
    poisoned[DEFAULT_ID_COLUMN] = list(range(N_ROWS))
    with pytest.raises(DeploymentDataError, match="must not appear"):
        _validate_sample_and_alignment(
            poisoned, sample, config, enforce_configured_rows=True
        )


def test_row_count_only_alignment_is_rejected(tmp_path: Path) -> None:
    _seed_competition_files(tmp_path)
    _seed_package_with_alignment(tmp_path)
    resolution = ensure_competition_row_identity(tmp_path, dataset_version=DATASET_ID)
    features = pd.DataFrame({"feature_one": list(range(N_ROWS))})
    sample = pd.read_csv(tmp_path / SAMPLE_SUBMISSION_PATH)
    wrong_hash = "0" * 64
    config = DeploymentConfig(
        payload={
            "sample_submission": {
                "id_column": DEFAULT_ID_COLUMN,
                "target_column": DEFAULT_TARGET_COLUMN,
                "expected_rows": N_ROWS,
            },
            "test_data": {},
            "submission_row_identity": {
                "path": resolution.row_identity_path,
                "sha256": resolution.row_identity_sha256,
                "expected_rows": N_ROWS,
                "id_column": DEFAULT_ID_COLUMN,
                "id_dtype": "int64",
                "id_semantics": "zero_based_row_position",
                "ordered_id_sha256": wrong_hash,
                "row_position_identity_sha256": wrong_hash,
                "test_anchor_hash": TEST_ANCHOR,
            },
        },
        source_path=tmp_path / "unused.yaml",
        project_root=tmp_path,
    )
    with pytest.raises(DeploymentDataError, match="Ordered submission ID hash"):
        _validate_sample_and_alignment(
            features, sample, config, enforce_configured_rows=True
        )


def test_id_order_mismatch_blocks(tmp_path: Path) -> None:
    _seed_competition_files(tmp_path)
    _seed_package_with_alignment(tmp_path)
    resolution = ensure_competition_row_identity(tmp_path, dataset_version=DATASET_ID)
    features = pd.DataFrame({"feature_one": list(range(N_ROWS))})
    sample = pd.read_csv(tmp_path / SAMPLE_SUBMISSION_PATH)
    sample[DEFAULT_ID_COLUMN] = list(reversed(range(N_ROWS)))
    config = DeploymentConfig(
        payload={
            "sample_submission": {
                "id_column": DEFAULT_ID_COLUMN,
                "target_column": DEFAULT_TARGET_COLUMN,
                "expected_rows": N_ROWS,
            },
            "test_data": {},
            "submission_row_identity": {
                "path": resolution.row_identity_path,
                "sha256": resolution.row_identity_sha256,
                "expected_rows": N_ROWS,
                "id_column": DEFAULT_ID_COLUMN,
                "id_dtype": "int64",
                "id_semantics": "zero_based_row_position",
                "ordered_id_sha256": resolution.ordered_id_sha256,
                "row_position_identity_sha256": resolution.ordered_id_sha256,
                "test_anchor_hash": TEST_ANCHOR,
            },
        },
        source_path=tmp_path / "unused.yaml",
        project_root=tmp_path,
    )
    with pytest.raises(DeploymentDataError, match="ordered IDs disagree|ID order"):
        _validate_sample_and_alignment(
            features, sample, config, enforce_configured_rows=True
        )


def test_resolved_and_unresolved_drafts_coexist(tmp_path: Path) -> None:
    _make_dataset_package(tmp_path)
    run = _make_run(tmp_path)
    unresolved = prepare_deployment_draft(
        tmp_path, run, approver=APPROVER, approved_at_utc=APPROVED_AT
    )
    assert (
        unresolved.payload.deployment_config["sample_submission"]["path"]
        == UNRESOLVED_SAMPLE_SUBMISSION_PATH
    )
    assert "submission_row_identity" not in unresolved.payload.deployment_config
    assert unresolved.payload.provenance["safety"][
        "sample_submission_identity_resolved"
    ] is False

    _seed_competition_files(tmp_path, n_rows=1000)
    # Align package expected rows with the synthetic run package (1000).
    _write_json(
        tmp_path / "data" / "processed" / DATASET_ID / "dataset_manifest.json",
        {
            "schema_version": "dataset_package_v1",
            "dataset_id": DATASET_ID,
            "target_dependency": "none",
            "schema_hash": "5c" * 32,
            "train_row_count": 4000,
            "test_row_count": 1000,
            "files": {"X_test": "X_test.parquet"},
            "content_hashes": {"X_test": TEST_ANCHOR},
            "features": [
                {"name": "feature_one", "dtype": "float64"},
                {"name": "feature_two", "dtype": "float64"},
            ],
            "row_identity": {
                "alignment_status": "proven",
                "test_anchor_hash": TEST_ANCHOR,
                "test_row_count": 1000,
            },
        },
    )
    resolved = prepare_deployment_draft(
        tmp_path, run, approver=APPROVER, approved_at_utc=APPROVED_AT
    )
    assert resolved.candidate_id != unresolved.candidate_id
    assert resolved.candidate_id.startswith(unresolved.candidate_id + "-r")
    assert (tmp_path / unresolved.deployment_config_path).is_file()
    assert (tmp_path / resolved.deployment_config_path).is_file()
    assert "submission_row_identity" in resolved.payload.deployment_config
    assert set(resolved.payload.deployment_config) == CONFIG_KEYS | CONFIG_OPTIONAL_KEYS
    assert resolved.payload.provenance["safety"][
        "sample_submission_identity_resolved"
    ] is True

    loaded = load_deployment_config(
        Path(resolved.deployment_config_path), project_root=tmp_path
    )
    assert loaded.payload["deployment_id"] == resolved.candidate_id
    assert "submission_row_identity" in loaded.payload

    again = prepare_deployment_draft(
        tmp_path, run, approver=APPROVER, approved_at_utc=APPROVED_AT
    )
    assert again.reused is True
    assert again.candidate_id == resolved.candidate_id


def test_existing_run_resolves_assets_without_mutation(tmp_path: Path) -> None:
    _seed_competition_files(tmp_path, n_rows=1000)
    _write_json(
        tmp_path / "data" / "processed" / DATASET_ID / "dataset_manifest.json",
        {
            "schema_version": "dataset_package_v1",
            "dataset_id": DATASET_ID,
            "target_dependency": "none",
            "schema_hash": "5c" * 32,
            "train_row_count": 4000,
            "test_row_count": 1000,
            "files": {"X_test": "X_test.parquet"},
            "content_hashes": {"X_test": TEST_ANCHOR},
            "features": [
                {"name": "feature_one", "dtype": "float64"},
                {"name": "feature_two", "dtype": "float64"},
            ],
            "row_identity": {
                "alignment_status": "proven",
                "test_anchor_hash": TEST_ANCHOR,
                "test_row_count": 1000,
            },
        },
    )
    run = _make_run(tmp_path)
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in (tmp_path / run).rglob("*")
        if path.is_file()
    }
    ensure_competition_row_identity(tmp_path, dataset_version=DATASET_ID)
    prepare_deployment_draft(
        tmp_path, run, approver=APPROVER, approved_at_utc=APPROVED_AT
    )
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in (tmp_path / run).rglob("*")
        if path.is_file()
    }
    assert before == after


def test_real_action_blocked_without_resolved_identity(tmp_path: Path) -> None:
    _make_dataset_package(tmp_path)
    run = _make_run(tmp_path)
    draft = prepare_deployment_draft(
        tmp_path, run, approver=APPROVER, approved_at_utc=APPROVED_AT
    )
    with pytest.raises(ValueError, match="resolved sample-submission|row_identity"):
        require_resolved_competition_draft(tmp_path, draft.deployment_config_path)

    registry = load_registry(REPOSITORY_ROOT)
    with pytest.raises(LaunchAuthorizationError, match="resolved|identity|ready"):
        authorize_launch(
            registry.commands,
            command_id="final_deployment_v1",
            action_id="run",
            values={"config": draft.deployment_config_path},
            repository_root=tmp_path,
            confirmed=True,
            high_risk_acknowledged=True,
            rendered=None,
            consumed_nonces=set(),
        )


def test_real_action_requires_resolved_draft_gate(tmp_path: Path) -> None:
    _seed_competition_files(tmp_path, n_rows=1000)
    _write_json(
        tmp_path / "data" / "processed" / DATASET_ID / "dataset_manifest.json",
        {
            "schema_version": "dataset_package_v1",
            "dataset_id": DATASET_ID,
            "target_dependency": "none",
            "schema_hash": "5c" * 32,
            "train_row_count": 4000,
            "test_row_count": 1000,
            "files": {"X_test": "X_test.parquet"},
            "content_hashes": {"X_test": TEST_ANCHOR},
            "features": [
                {"name": "feature_one", "dtype": "float64"},
                {"name": "feature_two", "dtype": "float64"},
            ],
            "row_identity": {
                "alignment_status": "proven",
                "test_anchor_hash": TEST_ANCHOR,
                "test_row_count": 1000,
            },
        },
    )
    run = _make_run(tmp_path)
    draft = prepare_deployment_draft(
        tmp_path, run, approver=APPROVER, approved_at_utc=APPROVED_AT
    )
    require_resolved_competition_draft(tmp_path, draft.deployment_config_path)
    ids = submission_ids_from_identity(
        json.loads((tmp_path / ROW_IDENTITY_ARTIFACT_PATH).read_text(encoding="utf-8"))
    )
    assert ids == list(range(1000))


def test_no_network_upload_in_runtime_contract(tmp_path: Path) -> None:
    _seed_competition_files(tmp_path, n_rows=1000)
    _write_json(
        tmp_path / "data" / "processed" / DATASET_ID / "dataset_manifest.json",
        {
            "schema_version": "dataset_package_v1",
            "dataset_id": DATASET_ID,
            "target_dependency": "none",
            "schema_hash": "5c" * 32,
            "train_row_count": 4000,
            "test_row_count": 1000,
            "files": {"X_test": "X_test.parquet"},
            "content_hashes": {"X_test": TEST_ANCHOR},
            "features": [
                {"name": "feature_one", "dtype": "float64"},
                {"name": "feature_two", "dtype": "float64"},
            ],
            "row_identity": {
                "alignment_status": "proven",
                "test_anchor_hash": TEST_ANCHOR,
                "test_row_count": 1000,
            },
        },
    )
    run = _make_run(tmp_path)
    draft = prepare_deployment_draft(
        tmp_path, run, approver=APPROVER, approved_at_utc=APPROVED_AT
    )
    assert draft.payload.deployment_config["runtime"] == {
        "tracking_enabled": False,
        "network_enabled": False,
    }
    assert draft.payload.deployment_config["output"]["root"] == "artifacts/deployments"


def test_missing_assets_keep_readiness_blocked(tmp_path: Path) -> None:
    resolution = resolve_competition_assets(tmp_path, dataset_version=DATASET_ID)
    assert resolution.ready is False
    assert resolution.summary()["Competition submission readiness"] == "blocked"
    assert any("Sample submission is missing" in reason for reason in resolution.blocking_reasons)
