from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from src.churn_ml.mlflow_config import MLflowIndexConfig, load_mlflow_config
from src.churn_ml.mlflow_mapping import IndexedRun
from src.churn_ml.mlflow_sources import (
    SourceAdapterRegistry,
    SourceValidationError,
)
from src.churn_ml.mlflow_sync import sync_sources
from tests.test_mlflow_config import valid_payload, write_config


class StaticAdapter:
    source_type = "research_v2"

    def __init__(self, root: Path, records: dict[str, IndexedRun | Exception]) -> None:
        self.root = root
        self.records = records

    def source_root(self, config: MLflowIndexConfig) -> Path:
        del config
        return self.root

    def experiment_name(self, config: MLflowIndexConfig) -> str:
        return config.experiments.research_v2

    def discover(
        self,
        config: MLflowIndexConfig,
        *,
        run_dir: Path | None = None,
    ) -> tuple[Path, ...]:
        del config
        return (
            (run_dir,)
            if run_dir is not None
            else tuple(self.root / name for name in self.records)
        )

    def prepare(
        self,
        run_dir: Path,
        config: MLflowIndexConfig,
    ) -> IndexedRun:
        del config
        result = self.records[run_dir.name]
        if isinstance(result, Exception):
            raise result
        return result


def make_record(
    root: Path,
    run_id: str,
    *,
    identity: str = "a" * 64,
    status: str = "completed",
) -> IndexedRun:
    run = root / run_id
    run.mkdir(parents=True, exist_ok=True)
    return IndexedRun(
        source_type="research_v2",
        source_run_id=run_id,
        source_relative_path=run_id,
        source_identity=identity,
        terminal_status="completed" if status == "completed" else "failed",
        mlflow_status="FINISHED" if status == "completed" else "FAILED",
        params={"candidate_sha256": "1" * 64, "repeat_count": 2},
        tags={"terminal_status": status},
        metrics={"balanced_accuracy": 0.9},
        artifacts=(),
        local_source_path=run,
    )


def setup_sync(
    tmp_path: Path,
    records: dict[str, IndexedRun | Exception],
) -> tuple[MLflowIndexConfig, SourceAdapterRegistry, StaticAdapter]:
    config = load_mlflow_config(
        write_config(tmp_path, valid_payload()), repository_root=tmp_path
    )
    root = config.paths.research_v2_root
    root.mkdir(parents=True, exist_ok=True)
    adapter = StaticAdapter(root, records)
    registry = SourceAdapterRegistry()
    registry.register(adapter)
    return config, registry, adapter


def test_dry_run_allocates_nothing_and_summary_is_json(tmp_path: Path) -> None:
    source_root = tmp_path / "artifacts" / "research_v2"
    record = make_record(source_root, "run-1")
    config, registry, _ = setup_sync(tmp_path, {"run-1": record})
    summary = sync_sources(
        config,
        registry=registry,
        source_types=("research_v2",),
        dry_run=True,
    )
    assert summary.counts["validated"] == 1
    assert config.paths.backend_store.exists() is False
    assert config.paths.artifact_root.exists() is False
    assert config.paths.receipts_root.exists() is False
    assert json.loads(json.dumps(summary.to_dict()))["dry_run"] is True


def test_local_sqlite_first_sync_second_sync_and_source_mutation(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "artifacts" / "research_v2"
    record = make_record(source_root, "run-1")
    config, registry, adapter = setup_sync(tmp_path, {"run-1": record})

    first = sync_sources(
        config,
        registry=registry,
        source_types=("research_v2",),
    )
    second = sync_sources(
        config,
        registry=registry,
        source_types=("research_v2",),
    )
    assert first.counts["created"] == 1
    assert second.counts["unchanged"] == 1
    assert first.items[0].mlflow_run_id == second.items[0].mlflow_run_id
    assert config.paths.backend_store.is_file()
    assert first.receipt_relative_path is not None

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=config.paths.tracking_uri)
    experiment = client.get_experiment_by_name(config.experiments.research_v2)
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.status == "FINISHED"

    adapter.records["run-1"] = replace(record, source_identity="b" * 64)
    mutated = sync_sources(
        config,
        registry=registry,
        source_types=("research_v2",),
    )
    assert mutated.counts["rejected"] == 1
    assert mutated.items[0].reason_code == "source_mutation_detected"
    assert len(client.search_runs([experiment.experiment_id])) == 1


def test_failed_run_resync_and_immutable_parameter_handling(tmp_path: Path) -> None:
    source_root = tmp_path / "artifacts" / "research_v2"
    record = make_record(source_root, "failed-run", status="failed")
    config, registry, adapter = setup_sync(tmp_path, {"failed-run": record})
    first = sync_sources(config, registry=registry, source_types=("research_v2",))
    second = sync_sources(config, registry=registry, source_types=("research_v2",))
    assert first.counts["created"] == 1
    assert second.counts["unchanged"] == 1

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=config.paths.tracking_uri)
    run = client.get_run(first.items[0].mlflow_run_id)
    assert run.info.status == "FAILED"

    adapter.records["failed-run"] = replace(
        record,
        params={**record.params, "repeat_count": 3},
    )
    conflict = sync_sources(config, registry=registry, source_types=("research_v2",))
    assert conflict.items[0].reason_code == "immutable_parameter_conflict"


def test_batch_continues_after_invalid_source_and_fail_fast(tmp_path: Path) -> None:
    source_root = tmp_path / "artifacts" / "research_v2"
    good = make_record(source_root, "good")
    bad_dir = source_root / "bad"
    bad_dir.mkdir(parents=True)
    records: dict[str, IndexedRun | Exception] = {
        "bad": SourceValidationError("corrupt", "bad source"),
        "good": good,
    }
    config, registry, _ = setup_sync(tmp_path, records)
    continued = sync_sources(
        config,
        registry=registry,
        source_types=("research_v2",),
        dry_run=True,
    )
    assert continued.counts["rejected"] == 1
    assert continued.counts["validated"] == 1

    stopped = sync_sources(
        config,
        registry=registry,
        source_types=("research_v2",),
        dry_run=True,
        fail_fast=True,
    )
    assert len(stopped.items) == 1
    assert stopped.items[0].outcome == "rejected"


def test_importing_mlflow_index_modules_is_lazy() -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import src.churn_ml.mlflow_config; "
                "import src.churn_ml.mlflow_sources; "
                "import src.churn_ml.mlflow_sync; "
                "assert 'mlflow' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr


def test_sync_sets_deterministic_run_name_and_renames_in_place(tmp_path: Path) -> None:
    source_root = tmp_path / "artifacts" / "research_v2"
    record = make_record(source_root, "run-1")
    record = replace(
        record,
        params={**record.params, "adapter_id": "xgboost_numeric_v1"},
    )
    config, registry, adapter = setup_sync(tmp_path, {"run-1": record})

    created = sync_sources(config, registry=registry, source_types=("research_v2",))
    assert created.counts["created"] == 1

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=config.paths.tracking_uri)
    experiment = client.get_experiment_by_name(config.experiments.research_v2)
    assert experiment is not None
    run_id = created.items[0].mlflow_run_id
    assert run_id is not None
    run = client.get_run(run_id)
    assert run.info.run_name == "xgboost_numeric_v1__run-1"
    assert run.data.tags["mlflow.runName"] == "xgboost_numeric_v1__run-1"
    source_key = run.data.tags["mlflow_index.source_key"]

    client.set_tag(run_id, "mlflow.runName", "whimsical-sponge-42")
    renamed = sync_sources(config, registry=registry, source_types=("research_v2",))
    assert renamed.counts["resumed"] == 1
    assert renamed.items[0].mlflow_run_id == run_id
    run = client.get_run(run_id)
    assert run.info.run_name == "xgboost_numeric_v1__run-1"
    assert run.data.tags["mlflow_index.source_key"] == source_key
    assert len(client.search_runs([experiment.experiment_id])) == 1

    unchanged = sync_sources(config, registry=registry, source_types=("research_v2",))
    assert unchanged.counts["unchanged"] == 1
    assert unchanged.items[0].mlflow_run_id == run_id


def test_sync_backfills_autogluon_quality_metrics_in_place(tmp_path: Path) -> None:
    from src.churn_ml.mlflow_mapping import build_autogluon_mapping

    config = load_mlflow_config(
        write_config(tmp_path, valid_payload()), repository_root=tmp_path
    )
    run_dir = config.paths.autogluon_root / "ag-v3-catboost"
    inspection = run_dir / "inspection"
    inspection.mkdir(parents=True)
    (inspection / "leaderboard.csv").write_text(
        "model,score_val,eval_metric,pred_time_val,fit_time\n"
        "WeightedEnsemble_L2,0.81,balanced_accuracy,0.2,10.5\n",
        encoding="utf-8",
    )
    common = {
        "run_dir": run_dir,
        "source_relative_path": "artifacts/autogluon_runs/ag-v3-catboost",
        "metadata": {
            "schema_version": 1,
            "profile_id": "profile",
            "profile_sha256": "1" * 64,
            "config_identity_sha256": "2" * 64,
            "dataset_version": "v3",
            "requested_seed": 42,
            "profile": {"autogluon_version": "1.5.0"},
            "resources": {},
        },
        "resolved_config": {},
        "profile_resolution": {},
        "worker_result": {
            "model_names": ["WeightedEnsemble_L2"],
            "best_model": "WeightedEnsemble_L2",
            "decision_threshold": 0.2,
            "autogluon_version": "1.5.0",
        },
        "terminal_status": "completed",
        "source_identity": "c" * 64,
        "predictor_classification": "complete",
        "status": {"duration_seconds": 9.0},
    }
    duration_only = build_autogluon_mapping(
        **common,
        inspection_summary={"effective_seed_status": "verified"},
    )
    # Simulate a previously indexed duration-only payload.
    duration_only = replace(
        duration_only,
        metrics={"duration_seconds": 9.0},
        tags={
            key: value
            for key, value in duration_only.tags.items()
            if key not in {"best_model", "metric_direction", "effective_seed_status"}
        },
        params={
            key: value
            for key, value in duration_only.params.items()
            if key != "eval_metric"
        },
    )
    enriched = build_autogluon_mapping(
        **common,
        inspection_summary={
            "effective_seed_status": "verified",
            "best_model": "WeightedEnsemble_L2",
            "leaderboard": [
                {
                    "model": "WeightedEnsemble_L2",
                    "score_val": 0.81,
                    "eval_metric": "balanced_accuracy",
                    "fit_time": 10.5,
                    "pred_time_val": 0.2,
                }
            ],
        },
    )
    assert "score_val" in enriched.metrics

    class AutoAdapter:
        source_type = "autogluon"

        def __init__(self) -> None:
            self.record = duration_only

        def source_root(self, config: MLflowIndexConfig) -> Path:
            return config.paths.autogluon_root

        def experiment_name(self, config: MLflowIndexConfig) -> str:
            return config.experiments.autogluon

        def discover(
            self,
            config: MLflowIndexConfig,
            *,
            run_dir: Path | None = None,
        ) -> tuple[Path, ...]:
            del config, run_dir
            return (self.record.local_source_path,)

        def prepare(
            self,
            run_dir: Path,
            config: MLflowIndexConfig,
        ) -> IndexedRun:
            del run_dir, config
            return self.record

    adapter = AutoAdapter()
    registry = SourceAdapterRegistry()
    registry.register(adapter)

    first = sync_sources(config, registry=registry, source_types=("autogluon",))
    assert first.counts["created"] == 1
    run_id = first.items[0].mlflow_run_id
    assert run_id is not None

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=config.paths.tracking_uri)
    experiment = client.get_experiment_by_name(config.experiments.autogluon)
    assert experiment is not None
    before = client.get_run(run_id)
    assert before.data.metrics == {"duration_seconds": 9.0}
    assert before.info.run_name == "ag-v3-catboost"
    source_key = before.data.tags["mlflow_index.source_key"]

    adapter.record = enriched
    backfilled = sync_sources(config, registry=registry, source_types=("autogluon",))
    assert backfilled.counts["resumed"] == 1
    assert backfilled.items[0].mlflow_run_id == run_id
    after = client.get_run(run_id)
    assert after.data.tags["mlflow_index.source_key"] == source_key
    assert after.data.metrics["duration_seconds"] == 9.0
    assert after.data.metrics["score_val"] == pytest.approx(0.81)
    assert after.data.metrics["best_model_fit_time_seconds"] == pytest.approx(10.5)
    assert after.data.metrics["best_model_pred_time_val_seconds"] == pytest.approx(0.2)
    assert after.data.params["eval_metric"] == "balanced_accuracy"
    assert after.data.tags["best_model"] == "WeightedEnsemble_L2"
    assert after.data.tags["metric_direction"] == "higher_is_better"
    assert len(client.search_runs([experiment.experiment_id])) == 1

    second = sync_sources(config, registry=registry, source_types=("autogluon",))
    assert second.counts["unchanged"] == 1
    assert second.items[0].mlflow_run_id == run_id
    final = client.get_run(run_id)
    assert set(final.data.metrics) == set(after.data.metrics)
    assert len(client.search_runs([experiment.experiment_id])) == 1
