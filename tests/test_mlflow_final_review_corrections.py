from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.churn_ml.mlflow_config import MLflowIndexConfig, load_mlflow_config
from src.churn_ml.mlflow_mapping import canonical_sha256
from src.churn_ml.mlflow_sources import (
    AutoGluonSourceAdapter,
    ResearchV2SourceAdapter,
    SourceValidationError,
    default_source_registry,
)
from src.churn_ml.mlflow_sync import sync_sources
from tests.test_mlflow_config import valid_payload
from tests.test_mlflow_mapping_and_sources import _write_exact_failed_autogluon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STARTED = "2026-01-01T00:00:00+00:00"
FINISHED = "2026-01-01T00:00:01+00:00"
FAILURE = {"type": "RuntimeError", "message": "early failure"}


def _config(
    root: Path,
    *,
    name: str = "local",
    research_root: str = "artifacts/research_v2",
    autogluon_root: str = "artifacts/autogluon_runs",
) -> MLflowIndexConfig:
    payload = valid_payload()
    payload["sources"]["research_v2_root"] = research_root
    payload["sources"]["autogluon_root"] = autogluon_root
    payload["sync"]["max_artifact_size_bytes"] = 1_000_000
    path = root / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return load_mlflow_config(path, repository_root=root)


def _resolved_config() -> dict[str, Any]:
    path = PROJECT_ROOT / "configs/research_v2/manual_lightgbm_te_v1_compat_smoke.yaml"
    resolved = yaml.safe_load(path.read_text(encoding="utf-8"))
    plan_path = PROJECT_ROOT / resolved["evaluation_plan_path"]
    resolved["evaluation_plan"] = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    return resolved


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_early_failure(
    run: Path,
    *,
    resolved_config: dict[str, Any] | None = None,
) -> None:
    run.mkdir(parents=True, exist_ok=True)
    _write_json(
        run / "execution_status.json",
        {
            "schema_version": 2,
            "run_id": run.name,
            "status": "failed",
            "started_at_utc": STARTED,
            "finished_at_utc": FINISHED,
            "completed_outer_folds": 0,
            "expected_outer_folds": 3,
            "failure": FAILURE,
        },
    )
    _write_json(
        run / "run_metadata.json",
        {
            "status": "failed",
            "finished_at_utc": FINISHED,
            "failure": FAILURE,
        },
    )
    if resolved_config is not None:
        artifacts_root = next(
            parent for parent in run.parents if parent.name == "artifacts"
        )
        repository_root = artifacts_root.parent
        plan_path = repository_root / resolved_config["evaluation_plan_path"]
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        if not plan_path.exists():
            plan_path.write_text(
                yaml.safe_dump(
                    resolved_config["evaluation_plan"],
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
        (run / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8"
        )
    (run / "_FAILED").touch()


def _write_identity(run: Path, name: str, canonical: dict[str, Any]) -> None:
    _write_json(
        run / "identities" / f"{name}.json",
        {"sha256": canonical_sha256(canonical), "canonical": canonical},
    )


def _candidate_identity(
    resolved: dict[str, Any],
    *,
    dataset_version: str | None = None,
    pipeline_id: str | None = None,
    adapter_id: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "dataset_version": dataset_version or resolved["dataset"]["version"],
        "feature_pipeline": {
            "id": pipeline_id or resolved["feature_pipeline"]["id"],
            "sha256": "1" * 64,
        },
        "candidate_adapter": {
            "id": adapter_id or resolved["candidate_adapter"]["id"],
            "sha256": "2" * 64,
        },
        "probability_semantics": "binary_positive_class_label_1",
        "evaluation_boundary": "adapter_receives_fold_local_data_only",
    }


def _plan_identity(
    resolved: dict[str, Any], *, plan_id: str | None = None
) -> dict[str, Any]:
    plan = resolved["evaluation_plan"]
    return {
        "schema_version": 1,
        "plan_id": plan_id or plan["plan"]["id"],
        "dataset": {"dataset_version": resolved["dataset"]["version"]},
        "outer_evaluation": deepcopy(plan["outer_evaluation"]),
        "threshold_selection": deepcopy(plan["threshold_selection"]),
        "threshold_policy": deepcopy(plan["threshold_policy"]),
        "metrics": deepcopy(plan["metrics"]),
        "aggregation": deepcopy(plan["aggregation"]),
        "assignment_fingerprints": {
            "outer": "3" * 64,
            "threshold_selection": "4" * 64,
        },
    }


def _valid_context() -> dict[str, dict[str, Any]]:
    return {
        "environment": {
            "python": "3.12",
            "numpy": "2",
            "pandas": "2",
            "scikit_learn": "1",
            "lightgbm": "4",
            "pyarrow": "18",
        },
        "runtime": {
            "process_started_at_utc": "2025-12-31T23:59:58+00:00",
            "preflight_at_utc": "2025-12-31T23:59:59+00:00",
            "entry_point": "scripts/run_research_v2.py",
            "fresh_process_required": True,
            "network_access": "disabled_during_execution",
            "competition_assets_accessed": False,
        },
        "git": {
            "branch": "feature",
            "commit": "a" * 40,
            "dirty": False,
            "status_porcelain": [],
        },
    }


def test_valid_early_failure_without_optional_config_is_accepted(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    run = config.paths.research_v2_root / "plan" / "candidate" / "run-1"
    _write_early_failure(run)

    record = ResearchV2SourceAdapter().prepare(run, config)

    assert record.terminal_status == "failed"
    assert record.params["dataset_version"] == "unknown"
    assert "resolved_config.yaml" not in record.artifact_relative_paths
    assert record.source_relative_path == "artifacts/research_v2/plan/candidate/run-1"


def test_valid_early_failure_with_strict_config_is_mapped_and_copyable(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    resolved = _resolved_config()
    run = config.paths.research_v2_root / "plan" / "candidate" / "run-1"
    _write_early_failure(run, resolved_config=resolved)

    record = ResearchV2SourceAdapter().prepare(run, config)

    assert record.params["dataset_version"] == resolved["dataset"]["version"]
    assert record.params["plan_schema_version"] == 1
    assert record.artifact_relative_paths.count("resolved_config.yaml") == 1


@pytest.mark.parametrize(
    "corruption",
    ["unknown_key", "wrong_type", "dataset", "pipeline", "adapter"],
)
def test_early_failure_rejects_strict_config_corruption(
    tmp_path: Path,
    corruption: str,
) -> None:
    config = _config(tmp_path)
    resolved = _resolved_config()
    if corruption == "unknown_key":
        resolved["unknown"] = True
    elif corruption == "wrong_type":
        resolved["schema_version"] = True
    elif corruption == "dataset":
        resolved["dataset"]["version"] = "forged_dataset"
    elif corruption == "pipeline":
        resolved["feature_pipeline"]["id"] = "forged_pipeline"
    else:
        resolved["candidate_adapter"]["id"] = "forged_adapter"
    run = config.paths.research_v2_root / "plan" / "candidate" / "run-1"
    _write_early_failure(run, resolved_config=resolved)

    with pytest.raises(SourceValidationError):
        ResearchV2SourceAdapter().prepare(run, config)


def test_early_failure_rejects_malformed_config(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run = config.paths.research_v2_root / "plan" / "candidate" / "run-1"
    _write_early_failure(run)
    (run / "resolved_config.yaml").write_text("[malformed", encoding="utf-8")

    with pytest.raises(SourceValidationError):
        ResearchV2SourceAdapter().prepare(run, config)


@pytest.mark.parametrize(
    ("identity_name", "canonical_factory"),
    [
        (
            "evaluation_plan",
            lambda resolved: _plan_identity(resolved, plan_id="forged_plan"),
        ),
        (
            "candidate",
            lambda resolved: _candidate_identity(
                resolved, pipeline_id="forged_pipeline"
            ),
        ),
        (
            "candidate",
            lambda resolved: _candidate_identity(resolved, adapter_id="forged_adapter"),
        ),
        (
            "candidate",
            lambda resolved: _candidate_identity(
                resolved, dataset_version="forged_dataset"
            ),
        ),
    ],
)
def test_early_failure_rejects_forged_optional_identities(
    tmp_path: Path,
    identity_name: str,
    canonical_factory: Any,
) -> None:
    config = _config(tmp_path)
    resolved = _resolved_config()
    run = config.paths.research_v2_root / "plan" / "candidate" / "run-1"
    _write_early_failure(run, resolved_config=resolved)
    _write_identity(run, identity_name, canonical_factory(resolved))

    with pytest.raises(SourceValidationError):
        ResearchV2SourceAdapter().prepare(run, config)


def test_valid_optional_identity_is_copied_only_after_validation(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    resolved = _resolved_config()
    run = config.paths.research_v2_root / "plan" / "candidate" / "run-1"
    _write_early_failure(run, resolved_config=resolved)
    _write_identity(run, "candidate", _candidate_identity(resolved))

    record = ResearchV2SourceAdapter().prepare(run, config)

    assert "identities/candidate.json" in record.artifact_relative_paths
    assert record.params["candidate_sha256"] not in {"1" * 64, "2" * 64}


def test_early_failure_rejects_contradictory_run_id(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run = config.paths.research_v2_root / "plan" / "candidate" / "run-1"
    _write_early_failure(run)
    path = run / "execution_status.json"
    status = json.loads(path.read_text(encoding="utf-8"))
    status["run_id"] = "other-run"
    _write_json(path, status)

    with pytest.raises(SourceValidationError):
        ResearchV2SourceAdapter().prepare(run, config)


def test_early_failure_rejects_malformed_optional_identity(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run = config.paths.research_v2_root / "plan" / "candidate" / "run-1"
    _write_early_failure(run, resolved_config=_resolved_config())
    path = run / "identities" / "candidate.json"
    path.parent.mkdir()
    path.write_text("{malformed", encoding="utf-8")

    with pytest.raises(SourceValidationError):
        ResearchV2SourceAdapter().prepare(run, config)


@pytest.mark.parametrize("name", ["environment", "runtime", "git"])
def test_early_failure_rejects_malformed_optional_context(
    tmp_path: Path,
    name: str,
) -> None:
    config = _config(tmp_path)
    run = config.paths.research_v2_root / "plan" / "candidate" / "run-1"
    _write_early_failure(run)
    _write_json(run / f"{name}.json", {"unknown": True})

    with pytest.raises(SourceValidationError):
        ResearchV2SourceAdapter().prepare(run, config)


def test_early_failure_rejects_context_inconsistent_with_status(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run = config.paths.research_v2_root / "plan" / "candidate" / "run-1"
    _write_early_failure(run)
    runtime = _valid_context()["runtime"]
    runtime["preflight_at_utc"] = "2026-01-01T00:00:00.500000+00:00"
    _write_json(run / "runtime.json", runtime)

    with pytest.raises(SourceValidationError):
        ResearchV2SourceAdapter().prepare(run, config)


def test_valid_optional_context_is_copied_after_exact_validation(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    run = config.paths.research_v2_root / "plan" / "candidate" / "run-1"
    _write_early_failure(run)
    for name, payload in _valid_context().items():
        _write_json(run / f"{name}.json", payload)

    record = ResearchV2SourceAdapter().prepare(run, config)

    assert {"environment.json", "runtime.json", "git.json"}.issubset(
        record.artifact_relative_paths
    )


def test_malformed_optional_artifact_stops_before_copy_mapping_and_mlflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    run = config.paths.research_v2_root / "plan" / "candidate" / "run-1"
    _write_early_failure(run)
    _write_json(run / "environment.json", {"unknown": True})
    called = {"copy": False, "mapping": False}

    def copied(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        called["copy"] = True
        return ()

    def mapped(*args: Any, **kwargs: Any) -> Any:
        called["mapping"] = True
        raise AssertionError("mapping must not run")

    monkeypatch.setattr("src.churn_ml.mlflow_sources.select_indexed_artifacts", copied)
    monkeypatch.setattr("src.churn_ml.mlflow_sources.build_research_mapping", mapped)

    summary = sync_sources(
        config,
        registry=default_source_registry(),
        source_types=("research_v2",),
    )

    assert summary.items[0].outcome == "rejected"
    assert called == {"copy": False, "mapping": False}
    assert config.paths.backend_store.exists() is False
    assert config.paths.artifact_root.exists() is False


def test_repository_relative_key_distinguishes_plans_and_repeats(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    first_run = config.paths.research_v2_root / "plan-a" / "candidate" / "run-1"
    second_run = config.paths.research_v2_root / "plan-b" / "candidate" / "run-1"
    _write_early_failure(first_run)
    _write_early_failure(second_run)
    adapter = ResearchV2SourceAdapter()

    first = adapter.prepare(first_run, config)
    repeated = adapter.prepare(first_run, config)
    second = adapter.prepare(second_run, config)

    assert first.source_key == repeated.source_key
    assert first.source_key != second.source_key
    assert first.source_key_payload["source_relative_path"].startswith("artifacts/")
    assert str(tmp_path.resolve()) not in json.dumps(first.source_key_payload)


def test_same_internal_path_under_two_configured_roots_is_distinct_in_sqlite(
    tmp_path: Path,
) -> None:
    first_config = _config(
        tmp_path,
        name="root-a",
        research_root="artifacts/research_v2_a",
    )
    second_config = _config(
        tmp_path,
        name="root-b",
        research_root="artifacts/research_v2_b",
    )
    internal = Path("plan") / "candidate" / "run-1"
    _write_early_failure(first_config.paths.research_v2_root / internal)
    _write_early_failure(second_config.paths.research_v2_root / internal)

    first = sync_sources(
        first_config,
        registry=default_source_registry(),
        source_types=("research_v2",),
    )
    second = sync_sources(
        second_config,
        registry=default_source_registry(),
        source_types=("research_v2",),
    )
    unchanged = sync_sources(
        first_config,
        registry=default_source_registry(),
        source_types=("research_v2",),
    )

    assert first.items[0].outcome == "created"
    assert second.items[0].outcome == "created"
    assert first.items[0].mlflow_run_id != second.items[0].mlflow_run_id
    assert unchanged.items[0].outcome == "unchanged"


def test_mutation_at_same_full_repository_path_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run = config.paths.research_v2_root / "plan" / "candidate" / "run-1"
    _write_early_failure(run)
    first = sync_sources(
        config,
        registry=default_source_registry(),
        source_types=("research_v2",),
    )
    status_path = run / "execution_status.json"
    metadata_path = run / "run_metadata.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    status["failure"]["message"] = "mutated failure"
    metadata["failure"]["message"] = "mutated failure"
    _write_json(status_path, status)
    _write_json(metadata_path, metadata)

    mutated = sync_sources(
        config,
        registry=default_source_registry(),
        source_types=("research_v2",),
    )

    assert first.items[0].outcome == "created"
    assert mutated.items[0].outcome == "rejected"
    assert mutated.items[0].reason_code == "source_mutation_detected"


def test_research_key_normalizes_backslashes_and_rejects_absolute_paths(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    run = config.paths.research_v2_root / "plan" / "candidate" / "run-1"
    _write_early_failure(run)
    record = ResearchV2SourceAdapter().prepare(run, config)

    normalized = replace(
        record,
        source_relative_path=record.source_relative_path.replace("/", "\\"),
    )

    assert normalized.source_key == record.source_key
    assert "\\" not in normalized.source_key_payload["source_relative_path"]
    with pytest.raises(ValueError):
        replace(record, source_relative_path=str(run.resolve()))


def test_repository_escape_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    outside = tmp_path / "outside" / "run-1"
    _write_early_failure(outside)

    with pytest.raises(SourceValidationError, match="inside"):
        ResearchV2SourceAdapter().prepare(outside, config)


def test_autogluon_key_uses_complete_repository_relative_source_root(
    tmp_path: Path,
) -> None:
    first_config = _config(
        tmp_path,
        name="auto-a",
        autogluon_root="artifacts/autogluon_runs_a",
    )
    second_config = _config(
        tmp_path,
        name="auto-b",
        autogluon_root="artifacts/autogluon_runs_b",
    )
    first_run = first_config.paths.autogluon_root / "run-1"
    second_run = second_config.paths.autogluon_root / "run-1"
    _write_exact_failed_autogluon(first_run)
    _write_exact_failed_autogluon(second_run)
    adapter = AutoGluonSourceAdapter()

    first = adapter.prepare(first_run, first_config)
    second = adapter.prepare(second_run, second_config)

    assert first.source_relative_path == "artifacts/autogluon_runs_a/run-1"
    assert second.source_relative_path == "artifacts/autogluon_runs_b/run-1"
    assert first.source_key != second.source_key
