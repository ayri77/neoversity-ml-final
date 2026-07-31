"""Unit tests for run-local dataset identity presentation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.churn_ml.control_panel.dataset_identity import (
    DatasetIdentityConflictError,
    enrich_experiment_core_references,
    extract_experiment_core_launch_identity,
    read_dataset_identity,
    read_dataset_identity_safe,
)
from src.churn_ml.control_panel.presentation import job_primary_label


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_run(
    root: Path,
    *,
    dataset_id: str = "v7_compact_zero_indicators",
    parent: str | None = "v4_zero_value_summary",
    target_dependency: str = "none",
    n_features: int = 234,
    include_provenance: bool = True,
    resolved_version: str | None = None,
    fingerprint_version: str | None = None,
    metadata_dataset_id: str | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    provenance = {
        "dataset_id": dataset_id,
        "parent_dataset_id": parent,
        "hypothesis": "test hypothesis",
        "n_features": n_features,
        "schema_hash": "a" * 64,
        "train_content_hash": "b" * 64,
        "target_hash": "c" * 64,
        "target_dependency": target_dependency,
        "train_row_identity_hash": "d" * 64,
        "registry_schema_version": "dataset_package_v1",
    }
    if include_provenance:
        (root / "dataset_provenance.json").write_text(
            json.dumps(provenance), encoding="utf-8"
        )
    metadata = {
        "experiment_id": "exp",
        "plan_id": "plan",
        "candidate_adapter_id": "manual_lightgbm_te_v1_compat",
        "dataset_provenance": {
            "dataset_id": metadata_dataset_id or dataset_id,
            "parent_dataset_id": parent,
            "n_features": n_features,
            "target_dependency": target_dependency,
            "schema_hash": "a" * 64,
            "train_content_hash": "b" * 64,
            "target_hash": "c" * 64,
            "train_row_identity_hash": "d" * 64,
            "registry_schema_version": "dataset_package_v1",
        },
    }
    (root / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "dataset": {"version": resolved_version or dataset_id},
                "experiment": {"id": "exp"},
            }
        ),
        encoding="utf-8",
    )
    (root / "dataset_fingerprints.json").write_text(
        json.dumps(
            {
                "dataset_version": fingerprint_version or dataset_id,
                "files": {"train_features": {"sha256": "e" * 64}},
                "row_position_identity": {"sha256": "f" * 64},
            }
        ),
        encoding="utf-8",
    )
    return root


def test_read_dataset_identity_prefers_run_local_provenance(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "run")
    identity = read_dataset_identity(run)
    assert identity.dataset_id == "v7_compact_zero_indicators"
    assert identity.parent_dataset_id == "v4_zero_value_summary"
    assert identity.n_features == 234
    assert identity.source == "dataset_provenance"


def test_read_dataset_identity_conflict_is_visible(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path / "run",
        dataset_id="v7_compact_zero_indicators",
        resolved_version="v0_raw_minimal",
    )
    with pytest.raises(DatasetIdentityConflictError, match="Conflicting"):
        read_dataset_identity(run)
    safe = read_dataset_identity_safe(run)
    assert safe.dataset_id is None
    assert safe.diagnostic is not None
    assert "Conflicting" in safe.diagnostic


def test_legacy_fallback_without_provenance(tmp_path: Path) -> None:
    run = tmp_path / "legacy"
    run.mkdir()
    (run / "resolved_config.yaml").write_text(
        "dataset:\n  version: v3_targeted_missingness\n",
        encoding="utf-8",
    )
    (run / "dataset_fingerprints.json").write_text(
        json.dumps({"dataset_version": "v3_targeted_missingness"}),
        encoding="utf-8",
    )
    identity = read_dataset_identity(run)
    assert identity.dataset_id == "v3_targeted_missingness"
    assert identity.source in {"resolved_config", "fingerprints"}
    assert identity.parent_dataset_id is None


def test_enrich_experiment_core_references_from_canonical_config() -> None:
    references = enrich_experiment_core_references(
        {"config": "configs/research_v2/manual_lightgbm_te_v1_compat_smoke.yaml"},
        repository_root=PROJECT_ROOT,
        command_id="experiment_core_v2",
    )
    assert references["dataset_id"] == "v3_targeted_missingness"
    assert references["experiment_id"] == "manual_lightgbm_te_v1_compat_smoke"
    assert "plan_id" in references
    assert references["model_family"] == "LightGBM"
    assert references["mode"] == "Smoke"
    # Unrelated commands unchanged.
    other = enrich_experiment_core_references(
        {"config": "configs/mlflow/local.yaml"},
        repository_root=PROJECT_ROOT,
        command_id="mlflow_local_index",
    )
    assert other == {"config": "configs/mlflow/local.yaml"}


def test_job_labels_include_dataset_and_remain_distinct() -> None:
    left = {
        "job_id": "11111111-1111-1111-1111-111111111111",
        "command_id": "experiment_core_v2",
        "action_id": "run",
        "created_at_utc": "2026-07-31T20:15:00+00:00",
        "references": {
            "config": "configs/research_v2/manual_lightgbm_te_v1_compat_smoke.yaml",
            "dataset_id": "v7_compact_zero_indicators",
            "model_family": "LightGBM",
            "mode": "Smoke",
        },
    }
    right = {
        **left,
        "job_id": "22222222-2222-2222-2222-222222222222",
        "references": {
            **left["references"],
            "dataset_id": "v0_raw_minimal",
        },
    }
    left_label = job_primary_label(left)
    right_label = job_primary_label(right)
    assert "v7_compact_zero_indicators" in left_label
    assert "v0_raw_minimal" in right_label
    assert left_label != right_label
    assert "LightGBM" in left_label
    assert "Smoke" in left_label


def test_old_jobs_without_dataset_reference_still_label() -> None:
    job = {
        "job_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "command_id": "experiment_core_v2",
        "action_id": "validate",
        "created_at_utc": "2026-07-31T20:15:00+00:00",
        "references": {
            "config": "configs/research_v2/manual_lightgbm_te_v1_compat_smoke.yaml",
        },
    }
    label = job_primary_label(job)
    assert "LightGBM" in label or "Experiment Core" in label or "Validate" in label


def test_extract_launch_identity_reads_dataset_not_filename(tmp_path: Path) -> None:
    config = tmp_path / "odd_name.yaml"
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "plan": {"id": "custom_plan"},
                "dataset": {"version": "v7_compact_zero_indicators"},
                "outer_evaluation": {
                    "splitter": "stratified_kfold",
                    "n_splits": 3,
                    "shuffle": True,
                    "repeat_seeds": [0],
                },
                "threshold_selection": {
                    "splitter": "stratified_kfold",
                    "n_splits": 2,
                    "shuffle": True,
                    "random_state": 1,
                },
                "threshold_policy": {"id": "grid"},
                "metrics": {"primary": "balanced_accuracy"},
                "aggregation": {"repeat_method": "pooled_predictions"},
            }
        ),
        encoding="utf-8",
    )
    # Minimal incomplete config is enough for identity extraction helpers.
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "experiment": {"id": "custom_exp_smoke"},
                "dataset": {"version": "v7_compact_zero_indicators"},
                "evaluation_plan_path": plan.relative_to(tmp_path).as_posix(),
                "candidate_adapter": {"id": "xgboost_numeric_v1", "contract": {}},
                "feature_pipeline": {"id": "registered_prepared_passthrough_v1"},
                "artifacts": {"root": "artifacts/research_v2"},
                "persistence": {"resolved_config": True},
                "tracking": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    identity = extract_experiment_core_launch_identity(
        tmp_path, config_relative=config.name
    )
    assert identity is not None
    assert identity.dataset_id == "v7_compact_zero_indicators"
    assert identity.experiment_id == "custom_exp_smoke"
    assert identity.model_family == "XGBoost"
