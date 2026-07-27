from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.churn_ml.mlflow_config import load_mlflow_config
from src.churn_ml.mlflow_mapping import (
    build_autogluon_mapping,
    build_research_mapping,
)
from src.churn_ml.mlflow_sources import (
    AutoGluonSourceAdapter,
    ResearchV2SourceAdapter,
    SourceAdapterRegistry,
    SourceValidationError,
    default_source_registry,
)
from tests.test_mlflow_config import valid_payload, write_config


def test_source_adapter_registry_is_typed_and_extensible() -> None:
    registry = default_source_registry()
    assert registry.source_types() == ("autogluon", "research_v2")
    assert registry.get("research_v2").source_type == "research_v2"
    with pytest.raises(ValueError, match="already registered"):
        registry.register(ResearchV2SourceAdapter())
    with pytest.raises(ValueError, match="Unknown source type"):
        registry.get("future_comparison")
    assert isinstance(registry, SourceAdapterRegistry)


def test_completed_and_failed_research_mapping() -> None:
    resolved = {
        "dataset": {"version": "v3"},
        "evaluation_plan": {
            "schema_version": 1,
            "dataset": {"version": "v3"},
            "outer_evaluation": {"repeat_seeds": [0, 1], "n_splits": 5},
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
        "evaluation_duration_seconds": 12.5,
    }
    completed = build_research_mapping(
        run_dir=Path("run-1"),
        source_relative_path="plan/component/run-1",
        metadata=metadata,
        status={"failure": None},
        resolved_config=resolved,
        terminal_status="completed",
        source_identity="a" * 64,
        aggregate={
            "metrics": {
                "balanced_accuracy": {
                    "mean": 0.9,
                    "sample_standard_deviation": 0.01,
                },
                "brier_score": {
                    "mean": 0.1,
                    "sample_standard_deviation": 0.02,
                },
            }
        },
        threshold_summary={"minimum": 0.1, "median": 0.2, "maximum": 0.4},
        threshold_standard_deviation=0.03,
    )
    assert completed.mlflow_status == "FINISHED"
    assert completed.params["candidate_sha256"] == "4" * 64
    assert completed.params["repeat_count"] == 2
    assert completed.metrics["balanced_accuracy"] == 0.9
    assert completed.metrics["threshold_range"] == pytest.approx(0.3)
    assert completed.tags["metric_direction.brier_score"] == "lower_is_better"

    failed = build_research_mapping(
        run_dir=Path("run-2"),
        source_relative_path="plan/component/run-2",
        metadata=metadata,
        status={"failure": {"type": "RuntimeError", "message": "boom"}},
        resolved_config=resolved,
        terminal_status="failed",
        source_identity="b" * 64,
    )
    assert failed.mlflow_status == "FAILED"
    assert failed.metrics == {}
    assert failed.tags["failure_type"] == "RuntimeError"


def test_completed_and_failed_autogluon_mapping() -> None:
    common = {
        "run_dir": Path("ag-run"),
        "source_relative_path": "ag-run",
        "metadata": {
            "schema_version": 1,
            "profile_id": "profile",
            "profile_sha256": "1" * 64,
            "config_identity_sha256": "2" * 64,
            "dataset_version": "v3",
            "requested_seed": 42,
            "profile": {
                "autogluon_version": "1.5.0",
                "included_model_types": ["GBM_PREP"],
                "excluded_model_types": [],
            },
            "resources": {"num_cpus": 4, "gpu_budget": 0},
        },
        "resolved_config": {},
        "profile_resolution": {"resolved_families": ["GBM_PREP"]},
        "inspection_summary": {"effective_seed_status": "verified"},
    }
    completed = build_autogluon_mapping(
        **common,
        status={"duration_seconds": 3.0},
        worker_result={
            "model_names": ["A", "WeightedEnsemble_L2"],
            "best_model": "WeightedEnsemble_L2",
            "decision_threshold": 0.117,
            "autogluon_version": "1.5.0",
        },
        terminal_status="completed",
        source_identity="c" * 64,
        predictor_classification="complete",
    )
    assert completed.mlflow_status == "FINISHED"
    assert completed.params["model_count"] == 2
    assert completed.params["decision_threshold"] == 0.117
    assert completed.tags["predictor_loading_attempted"] == "false"

    failed = build_autogluon_mapping(
        **common,
        status={
            "duration_seconds": 4.0,
            "child_process_exit_code": 23,
            "failure_codes": ["worker_exit_code:23"],
            "last_completed_observable_stage": "fit_started",
        },
        worker_result=None,
        terminal_status="failed",
        source_identity="d" * 64,
        predictor_classification="failed_terminal_not_loaded",
    )
    assert failed.mlflow_status == "FAILED"
    assert failed.params["child_process_exit_code"] == 23
    assert failed.tags["failure_codes"] == "worker_exit_code:23"


def test_failed_research_artifact_policy_and_no_source_mutation(
    tmp_path: Path,
) -> None:
    payload = valid_payload()
    payload["sync"]["max_artifact_size_bytes"] = 450
    config = load_mlflow_config(
        write_config(tmp_path, payload), repository_root=tmp_path
    )
    run = config.paths.research_v2_root / "plan" / "candidate" / "failed-run"
    (run / "predictions").mkdir(parents=True)
    _write_json(
        run / "execution_status.json",
        {
            "schema_version": 2,
            "run_id": "failed-run",
            "status": "failed",
            "started_at_utc": "2026-01-01T00:00:00+00:00",
            "finished_at_utc": "2026-01-01T00:00:01+00:00",
            "completed_outer_folds": 0,
            "expected_outer_folds": 3,
            "failure": {"type": "RuntimeError", "message": "failed"},
        },
    )
    _write_json(
        run / "run_metadata.json",
        {
            "schema_version": 2,
            "run_id": "failed-run",
            "status": "failed",
            "hashes": {"candidate": "1" * 64},
        },
    )
    (run / "resolved_config.yaml").write_text(
        "padding: " + ("x" * 500), encoding="utf-8"
    )
    (run / "_FAILED").touch()
    (run / "predictions" / "outer_validation.parquet").write_bytes(b"prohibited")
    before = _tree_snapshot(run)

    record = ResearchV2SourceAdapter().prepare(run, config)

    assert record.terminal_status == "failed"
    assert "execution_status.json" in record.artifact_relative_paths
    assert "resolved_config.yaml" not in record.artifact_relative_paths
    assert all("predictions" not in item for item in record.artifact_relative_paths)
    assert _tree_snapshot(run) == before


def test_corrupt_completed_research_is_rejected_by_production_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_mlflow_config(
        write_config(tmp_path, valid_payload()), repository_root=tmp_path
    )
    run = config.paths.research_v2_root / "plan" / "candidate" / "bad-run"
    run.mkdir(parents=True)
    resolved = _minimal_resolved_research()
    (run / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    _write_json(
        run / "execution_status.json",
        {"status": "completed", "run_id": "bad-run"},
    )
    _write_json(
        run / "run_metadata.json",
        {
            "run_id": "bad-run",
            "status": "completed",
            "hashes": {"candidate": "1" * 64},
        },
    )
    (run / "_SUCCESS").touch()

    called = False

    def reject(*args: Any, **kwargs: Any) -> None:
        nonlocal called
        called = True
        raise RuntimeError("manifest corrupt")

    monkeypatch.setattr("src.churn_ml.mlflow_sources.validate_research_v2_run", reject)
    with pytest.raises(SourceValidationError, match="semantic validation"):
        ResearchV2SourceAdapter().prepare(run, config)
    assert called is True


def test_failed_autogluon_never_loads_predictor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_mlflow_config(
        write_config(tmp_path, valid_payload()), repository_root=tmp_path
    )
    run = config.paths.autogluon_root / "failed-ag-run"
    run.mkdir(parents=True)
    profile = {}
    profile_hash = hashlib.sha256(
        json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    metadata = {
        "schema_version": 1,
        "run_id": run.name,
        "config_identity_sha256": "1" * 64,
        "profile_id": "profile",
        "profile_sha256": profile_hash,
        "profile": profile,
        "dataset_version": "v3",
        "requested_seed": 42,
        "child_pid": 123,
    }
    status = {
        "status": "failed",
        "run_id": run.name,
        "config_identity_sha256": "1" * 64,
        "profile_sha256": profile_hash,
        "requested_seed": 42,
        "started_at_utc": "2026-01-01T00:00:00Z",
        "ended_at_utc": "2026-01-01T00:00:01Z",
        "duration_seconds": 1.0,
        "child_pid": 123,
        "child_process_exit_code": 23,
        "last_completed_observable_stage": "fit_started",
        "failure_codes": ["worker_exit_code:23"],
        "failure_reason": "worker_exit_code:23",
        "predictor_loading_attempted": False,
        "predictor_loading_succeeded": False,
    }
    _write_json(run / "run_metadata.json", metadata)
    _write_json(run / "execution_status.json", status)
    _write_json(
        run / "_FAILED",
        {
            "status": "failed",
            "run_id": run.name,
            "failure_codes": status["failure_codes"],
        },
    )

    def forbidden_loader(_path: Path) -> Any:
        raise AssertionError("predictor loader must not be called")

    monkeypatch.setattr(
        "src.churn_ml.autogluon_inspection._load_predictor", forbidden_loader
    )
    record = AutoGluonSourceAdapter().prepare(run, config)
    assert record.terminal_status == "failed"
    assert record.tags["predictor_loading_attempted"] == "false"


def _minimal_resolved_research() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "experiment": {"id": "experiment"},
        "dataset": {"version": "v3"},
        "evaluation_plan_path": "configs/plan.yaml",
        "feature_pipeline": {"id": "pipeline", "contract": {}},
        "candidate_adapter": {"id": "adapter", "contract": {}},
        "artifacts": {"root": "artifacts/research_v2"},
        "persistence": {
            "resolved_config": True,
            "identities": True,
            "assignments": True,
            "predictions": True,
            "metrics": True,
            "threshold_curves": False,
            "models": False,
        },
        "tracking": {"enabled": False},
        "evaluation_plan": {"schema_version": 1},
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
