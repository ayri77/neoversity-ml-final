"""MLflow dataset provenance mapping and allowlist coverage."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from mlflow.tracking import MlflowClient

from src.churn_ml.mlflow_mapping import (
    build_research_mapping,
    deterministic_mlflow_run_name,
)
from src.churn_ml.mlflow_sources import _RESEARCH_ARTIFACT_ALLOWLIST
from src.churn_ml.mlflow_sync import sync_sources
from tests.test_mlflow_sync import make_record, setup_sync


def _provenance() -> dict[str, object]:
    return {
        "dataset_id": "v7_compact_zero_indicators",
        "parent_dataset_id": "v4_zero_value_summary",
        "n_features": 234,
        "target_dependency": "none",
        "schema_hash": "a" * 64,
        "train_content_hash": "b" * 64,
        "target_hash": "c" * 64,
        "train_row_identity_hash": "d" * 64,
        "registry_schema_version": "dataset_package_v1",
    }


def test_research_allowlist_includes_dataset_provenance() -> None:
    assert "dataset_provenance.json" in _RESEARCH_ARTIFACT_ALLOWLIST


def test_deterministic_run_name_includes_dataset() -> None:
    assert (
        deterministic_mlflow_run_name(
            source_type="research_v2",
            source_run_id="run-1",
            adapter_id="adapter",
            dataset_version="v7_compact_zero_indicators",
        )
        == "v7_compact_zero_indicators__adapter__run-1"
    )
    assert (
        deterministic_mlflow_run_name(
            source_type="autogluon",
            source_run_id="ag-1",
            adapter_id="ignored",
            dataset_version="v3",
        )
        == "ag-1"
    )


def test_completed_mapping_adds_provenance_params_and_tags(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "dataset_provenance.json").write_text(
        json.dumps(_provenance()), encoding="utf-8"
    )
    resolved = {
        "dataset": {"version": "v7_compact_zero_indicators"},
        "evaluation_plan": {
            "schema_version": 1,
            "dataset": {"version": "v7_compact_zero_indicators"},
            "outer_evaluation": {"repeat_seeds": [0], "n_splits": 3},
            "threshold_policy": {"id": "grid_v1"},
        },
    }
    metadata = {
        "schema_version": 2,
        "plan_id": "plan",
        "feature_pipeline_id": "pipeline",
        "candidate_adapter_id": "adapter",
        "hashes": {
            "plan": "1" * 64,
            "feature_pipeline": "2" * 64,
            "candidate_adapter": "3" * 64,
            "candidate": "4" * 64,
            "source": "5" * 64,
            "loaded_modules": "6" * 64,
        },
        "dataset_provenance": _provenance(),
    }
    completed = build_research_mapping(
        run_dir=run_dir,
        source_relative_path="plan/component/run-1",
        metadata=metadata,
        status={"failure": None},
        resolved_config=resolved,
        terminal_status="completed",
        source_identity="a" * 64,
        aggregate={
            "metrics": {
                "balanced_accuracy": {"mean": 0.9, "sample_standard_deviation": 0.01}
            }
        },
        threshold_summary={"minimum": 0.1, "median": 0.2, "maximum": 0.4},
        threshold_standard_deviation=0.03,
    )
    assert completed.params["dataset_version"] == "v7_compact_zero_indicators"
    assert completed.params["dataset_parent_id"] == "v4_zero_value_summary"
    assert completed.params["dataset_feature_count"] == 234
    assert completed.params["dataset_target_dependency"] == "none"
    assert completed.params["dataset_schema_sha256"] == "a" * 64
    assert completed.params["dataset_train_content_sha256"] == "b" * 64
    assert completed.params["dataset_target_sha256"] == "c" * 64
    assert completed.params["dataset_train_row_identity_sha256"] == "d" * 64
    assert completed.params["dataset_registry_schema_version"] == "dataset_package_v1"
    assert completed.tags["dataset.id"] == "v7_compact_zero_indicators"
    assert completed.tags["dataset.target_dependency"] == "none"
    assert completed.run_name == "v7_compact_zero_indicators__adapter__run-1"


def test_contradictory_failed_run_provenance_is_ignored(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-2"
    run_dir.mkdir()
    (run_dir / "dataset_provenance.json").write_text(
        json.dumps({**_provenance(), "dataset_id": "forged"}), encoding="utf-8"
    )
    resolved = {
        "dataset": {"version": "v7_compact_zero_indicators"},
        "evaluation_plan": {
            "schema_version": 1,
            "dataset": {"version": "v7_compact_zero_indicators"},
            "outer_evaluation": {"repeat_seeds": [0], "n_splits": 3},
            "threshold_policy": {"id": "grid_v1"},
        },
    }
    metadata = {
        "schema_version": 2,
        "plan_id": "plan",
        "feature_pipeline_id": "pipeline",
        "candidate_adapter_id": "adapter",
        "hashes": {
            "plan": "1" * 64,
            "feature_pipeline": "2" * 64,
            "candidate_adapter": "3" * 64,
            "candidate": "4" * 64,
            "source": "5" * 64,
            "loaded_modules": "6" * 64,
        },
        "dataset_provenance": _provenance(),
    }
    failed = build_research_mapping(
        run_dir=run_dir,
        source_relative_path="plan/component/run-2",
        metadata=metadata,
        status={"failure": {"type": "RuntimeError", "message": "boom"}},
        resolved_config=resolved,
        terminal_status="failed",
        source_identity="b" * 64,
    )
    assert "dataset_parent_id" not in failed.params
    assert "dataset.id" not in failed.tags


def test_contradictory_completed_provenance_is_omitted(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-3"
    run_dir.mkdir()
    (run_dir / "dataset_provenance.json").write_text(
        json.dumps({**_provenance(), "dataset_id": "other"}), encoding="utf-8"
    )
    resolved = {
        "dataset": {"version": "v7_compact_zero_indicators"},
        "evaluation_plan": {
            "schema_version": 1,
            "dataset": {"version": "v7_compact_zero_indicators"},
            "outer_evaluation": {"repeat_seeds": [0], "n_splits": 3},
            "threshold_policy": {"id": "grid_v1"},
        },
    }
    metadata = {
        "schema_version": 2,
        "plan_id": "plan",
        "feature_pipeline_id": "pipeline",
        "candidate_adapter_id": "adapter",
        "hashes": {
            "plan": "1" * 64,
            "feature_pipeline": "2" * 64,
            "candidate_adapter": "3" * 64,
            "candidate": "4" * 64,
            "source": "5" * 64,
            "loaded_modules": "6" * 64,
        },
        "dataset_provenance": _provenance(),
    }
    completed = build_research_mapping(
        run_dir=run_dir,
        source_relative_path="plan/component/run-3",
        metadata=metadata,
        status={"failure": None},
        resolved_config=resolved,
        terminal_status="completed",
        source_identity="c" * 64,
        aggregate={
            "metrics": {
                "balanced_accuracy": {"mean": 0.9, "sample_standard_deviation": 0.01}
            }
        },
        threshold_summary={"minimum": 0.1, "median": 0.2, "maximum": 0.4},
        threshold_standard_deviation=0.03,
    )
    assert completed.params["dataset_version"] == "v7_compact_zero_indicators"
    assert "dataset_parent_id" not in completed.params
    assert "dataset.id" not in completed.tags


def test_sync_renames_with_dataset_in_place(tmp_path: Path) -> None:
    source_root = tmp_path / "artifacts" / "research_v2"
    record = make_record(source_root, "run-1")
    record = replace(
        record,
        params={
            **record.params,
            "adapter_id": "xgboost_numeric_v1",
            "dataset_version": "v7_compact_zero_indicators",
        },
    )
    config, registry, _ = setup_sync(tmp_path, {"run-1": record})
    created = sync_sources(config, registry=registry, source_types=("research_v2",))
    assert created.counts["created"] == 1
    client = MlflowClient(tracking_uri=config.paths.tracking_uri)
    run_id = created.items[0].mlflow_run_id
    assert run_id is not None
    run = client.get_run(run_id)
    expected = "v7_compact_zero_indicators__xgboost_numeric_v1__run-1"
    assert run.info.run_name == expected
    source_key = run.data.tags["mlflow_index.source_key"]
    client.set_tag(run_id, "mlflow.runName", "whimsical-sponge-42")
    renamed = sync_sources(config, registry=registry, source_types=("research_v2",))
    assert renamed.counts["resumed"] == 1
    assert renamed.items[0].mlflow_run_id == run_id
    run = client.get_run(run_id)
    assert run.info.run_name == expected
    assert run.data.tags["mlflow_index.source_key"] == source_key
