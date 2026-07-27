from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from src.churn_ml.autogluon_artifacts import build_inventory
from src.churn_ml.mlflow_artifacts import (
    ArtifactContentError,
    IndexedArtifact,
    revalidate_indexed_artifact,
    select_indexed_artifacts,
)
from src.churn_ml.mlflow_config import (
    MLflowConfigError,
    MLflowIndexConfig,
    load_mlflow_config,
)
from src.churn_ml.mlflow_mapping import (
    IndexedRun,
    build_autogluon_mapping,
    canonical_source_relative_path,
)
from src.churn_ml.mlflow_sources import (
    AutoGluonSourceAdapter,
    ResearchV2SourceAdapter,
    SourceAdapterRegistry,
    SourceValidationError,
)
from src.churn_ml.mlflow_sync import SyncItem, _sync_one, _write_receipt, sync_sources
from tests.test_mlflow_config import valid_payload, write_config
from tests.test_mlflow_mapping_and_sources import _write_exact_failed_autogluon

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class OneRecordAdapter:
    root: Path
    record: IndexedRun
    source_type: str = "research_v2"

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
        return (run_dir or self.record.local_source_path,)

    def prepare(self, run_dir: Path, config: MLflowIndexConfig) -> IndexedRun:
        del run_dir, config
        return self.record


class RecordingFailOnceClient:
    def __init__(
        self,
        client: Any,
        failure: Callable[[str, tuple[Any, ...], dict[str, Any]], bool] | None,
    ) -> None:
        self.client = client
        self.failure = failure
        self.failed = False
        self.mutations: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Any:
        original = getattr(self.client, name)
        if name not in {
            "create_run",
            "log_param",
            "log_metric",
            "set_tag",
            "delete_tag",
            "log_artifact",
            "set_terminated",
        }:
            return original

        def call(*args: Any, **kwargs: Any) -> Any:
            self.mutations.append((name, args, kwargs))
            if (
                not self.failed
                and self.failure is not None
                and self.failure(name, args, kwargs)
            ):
                self.failed = True
                raise RuntimeError(f"injected failure at {name}")
            return original(*args, **kwargs)

        return call


def _record(root: Path, *, status: str = "completed") -> IndexedRun:
    run = root / "group" / "run-1"
    run.mkdir(parents=True, exist_ok=True)
    (run / "metadata.json").write_text('{"valid": true}\n', encoding="utf-8")
    artifacts = select_indexed_artifacts(
        run,
        ("metadata.json",),
        enabled=True,
        size_limit=1024,
    )
    return IndexedRun(
        source_type="research_v2",
        source_run_id="run-1",
        source_relative_path="group/run-1",
        source_identity="a" * 64,
        terminal_status="completed" if status == "completed" else "failed",
        mlflow_status="FINISHED" if status == "completed" else "FAILED",
        params={"candidate_sha256": "1" * 64, "repeat_count": 2},
        tags={"terminal_status": status},
        metrics={"balanced_accuracy": 0.9},
        artifacts=artifacts,
        local_source_path=run,
    )


def _client_and_experiment(tmp_path: Path) -> tuple[Any, str, MLflowIndexConfig]:
    config = load_mlflow_config(
        write_config(tmp_path, valid_payload()), repository_root=tmp_path
    )
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=config.paths.tracking_uri)
    experiment_id = str(
        client.create_experiment(
            config.experiments.research_v2,
            artifact_location=(config.paths.artifact_root / "research_v2").as_uri(),
        )
    )
    return client, experiment_id, config


def _stage_failure(
    stage: str,
) -> Callable[[str, tuple[Any, ...], dict[str, Any]], bool]:
    def predicate(name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
        if stage == "begin_marker":
            return name == "set_tag" and args[1:] == (
                "mlflow_index.sync_complete",
                "false",
            )
        if stage == "record_tag":
            return name == "set_tag" and args[1:2] == ("terminal_status",)
        if stage == "completion_marker":
            return name == "set_tag" and args[1:] == (
                "mlflow_index.sync_complete",
                "true",
            )
        return name == stage

    return predicate


@pytest.mark.parametrize(
    "stage",
    (
        "begin_marker",
        "log_param",
        "log_metric",
        "record_tag",
        "log_artifact",
        "set_terminated",
        "completion_marker",
    ),
)
def test_two_phase_sync_recovers_same_run_after_every_write_stage(
    tmp_path: Path,
    stage: str,
) -> None:
    client, experiment_id, config = _client_and_experiment(tmp_path)
    record = _record(config.paths.research_v2_root)
    failing = RecordingFailOnceClient(client, _stage_failure(stage))

    interrupted = _sync_one(failing, experiment_id, record)
    assert interrupted.outcome == "recoverable"
    assert interrupted.mlflow_run_id is not None
    partial = client.get_run(interrupted.mlflow_run_id)
    assert partial.data.tags.get("mlflow_index.sync_complete") != "true"

    retry = RecordingFailOnceClient(client, None)
    resumed = _sync_one(retry, experiment_id, record)
    assert resumed.outcome == "resumed"
    assert resumed.mlflow_run_id == interrupted.mlflow_run_id
    final = client.get_run(resumed.mlflow_run_id)
    assert final.info.status == "FINISHED"
    assert final.data.tags["mlflow_index.sync_complete"] == "true"
    last_mutation = retry.mutations[-1]
    assert last_mutation[0] == "set_tag"
    assert last_mutation[1][1:] == ("mlflow_index.sync_complete", "true")

    unchanged = _sync_one(client, experiment_id, record)
    assert unchanged.outcome == "unchanged"
    assert len(client.search_runs([experiment_id])) == 1


def test_poisoned_completion_marker_is_repaired_not_declared_unchanged(
    tmp_path: Path,
) -> None:
    client, experiment_id, config = _client_and_experiment(tmp_path)
    record = _record(config.paths.research_v2_root)
    first = _sync_one(client, experiment_id, record)
    assert first.outcome == "created"
    assert first.mlflow_run_id is not None
    client.set_terminated(first.mlflow_run_id, status="FAILED")

    repaired = _sync_one(client, experiment_id, record)
    assert repaired.outcome == "resumed"
    assert repaired.mlflow_run_id == first.mlflow_run_id
    run = client.get_run(first.mlflow_run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags["mlflow_index.sync_complete"] == "true"


def test_artifact_toctou_change_is_recoverable_and_not_committed(
    tmp_path: Path,
) -> None:
    client, experiment_id, config = _client_and_experiment(tmp_path)
    record = _record(config.paths.research_v2_root)
    (record.local_source_path / "metadata.json").write_text(
        '{"valid": false}\n', encoding="utf-8"
    )

    item = _sync_one(client, experiment_id, record)
    assert item.outcome == "recoverable"
    assert item.mlflow_run_id is not None
    run = client.get_run(item.mlflow_run_id)
    assert run.data.tags.get("mlflow_index.sync_complete") != "true"
    assert run.info.status == "RUNNING"


@pytest.mark.parametrize(
    ("filename", "body"),
    (
        ("bad.json", b'{"x": NaN}'),
        ("bad.yaml", b"items: [unterminated"),
        ("bad.csv", b"a,b\n1\n"),
        ("nul.json", b'{"x": "a\x00b"}'),
        ("nul.csv", b"a,b\n1,2\x00\n"),
        ("binary.csv", b"a,b\n\xff,2\n"),
        ("wrong-type.yaml", b"- one\n- two\n"),
        ("binary.json", b"\xff\xfe"),
    ),
)
def test_artifact_selection_rejects_invalid_human_readable_content(
    tmp_path: Path,
    filename: str,
    body: bytes,
) -> None:
    (tmp_path / filename).write_bytes(body)
    with pytest.raises(ArtifactContentError):
        select_indexed_artifacts(
            tmp_path,
            (filename,),
            enabled=True,
            size_limit=1024,
        )


def test_artifact_revalidation_rejects_selected_byte_change(tmp_path: Path) -> None:
    source = tmp_path / "metadata.json"
    source.write_text('{"x": 1}\n', encoding="utf-8")
    selected = select_indexed_artifacts(
        tmp_path,
        ("metadata.json",),
        enabled=True,
        size_limit=1024,
    )[0]
    source.write_text('{"x": 2}\n', encoding="utf-8")
    with pytest.raises(ArtifactContentError, match="changed"):
        revalidate_indexed_artifact(tmp_path, selected)


def test_scoped_source_key_uses_canonical_relative_path(tmp_path: Path) -> None:
    base = _record(tmp_path)
    same_spelling = IndexedRun(
        **{
            **base.__dict__,
            "source_relative_path": "group\\run-1",
        }
    )
    different_scope = IndexedRun(
        **{
            **base.__dict__,
            "source_relative_path": "other/run-1",
        }
    )
    assert base.source_key == same_spelling.source_key
    assert base.source_key != different_scope.source_key
    assert canonical_source_relative_path("group\\run-1") == "group/run-1"
    with pytest.raises(ValueError):
        canonical_source_relative_path("../run-1")


def test_deterministic_receipt_has_stable_path_and_bytes(tmp_path: Path) -> None:
    client, experiment_id, config = _client_and_experiment(tmp_path)
    record = _record(config.paths.research_v2_root)
    item = _sync_one(client, experiment_id, record)
    assert item.outcome == "created"
    first = _write_receipt(config, item, record)
    first_bytes = (config.paths.repository_root / first).read_bytes()

    unchanged = _sync_one(client, experiment_id, record)
    second = _write_receipt(config, unchanged, record)
    assert second == first
    assert (config.paths.repository_root / second).read_bytes() == first_bytes
    payload = json.loads(first_bytes)
    assert "created_at_utc" not in payload
    assert "tracking_backend_nonportable" not in payload
    changed_identity = replace(record, source_identity="b" * 64)
    changed_path = _write_receipt(config, item, changed_identity)
    assert changed_path != first


def test_sync_sources_reports_one_stable_receipt_per_committed_record(
    tmp_path: Path,
) -> None:
    config = load_mlflow_config(
        write_config(tmp_path, valid_payload()), repository_root=tmp_path
    )
    record = _record(config.paths.research_v2_root)
    registry = SourceAdapterRegistry()
    registry.register(OneRecordAdapter(config.paths.research_v2_root, record))
    first = sync_sources(config, registry=registry, source_types=("research_v2",))
    second = sync_sources(config, registry=registry, source_types=("research_v2",))
    assert first.receipt_relative_paths == second.receipt_relative_paths
    assert len(first.receipt_relative_paths) == 1


@pytest.mark.parametrize(
    "artifact_root",
    (
        "artifacts/mlflow/mlartifacts/mlflow.db/files",
        "artifacts/mlflow/sync_receipts",
        "artifacts/mlflow/sync_receipts/nested",
    ),
)
def test_storage_locations_reject_symmetric_overlap(
    tmp_path: Path,
    artifact_root: str,
) -> None:
    payload = valid_payload()
    if artifact_root.startswith("artifacts/mlflow/mlartifacts"):
        payload["tracking"]["backend_store_uri"] = (
            "sqlite:///artifacts/mlflow/mlartifacts/mlflow.db"
        )
    payload["tracking"]["artifact_root"] = artifact_root
    with pytest.raises(MLflowConfigError, match="non-overlapping"):
        load_mlflow_config(write_config(tmp_path, payload), repository_root=tmp_path)


def test_artifact_root_containing_database_is_rejected(tmp_path: Path) -> None:
    payload = valid_payload()
    payload["tracking"]["backend_store_uri"] = (
        "sqlite:///artifacts/mlflow/mlartifacts/mlflow.db"
    )
    payload["tracking"]["artifact_root"] = "artifacts/mlflow/mlartifacts"
    with pytest.raises(MLflowConfigError, match="non-overlapping"):
        load_mlflow_config(write_config(tmp_path, payload), repository_root=tmp_path)


def test_failed_autogluon_omits_stale_completion_fields_and_preserves_zero() -> None:
    record = build_autogluon_mapping(
        run_dir=Path("run"),
        source_relative_path="scope/run",
        metadata={
            "schema_version": 1,
            "profile_id": "profile",
            "profile_sha256": "1" * 64,
            "config_identity_sha256": "2" * 64,
            "dataset_version": "v3",
            "requested_seed": 42,
            "profile": {"autogluon_version": "1.5.0"},
            "resources": {},
            "duration_seconds": 8.0,
        },
        status={
            "duration_seconds": 0.0,
            "failure_codes": ["worker_exit_code:1"],
        },
        resolved_config={},
        profile_resolution={},
        worker_result={
            "model_names": ["stale"],
            "best_model": "stale",
            "decision_threshold": 0.5,
            "autogluon_version": "stale",
        },
        inspection_summary={},
        terminal_status="failed",
        source_identity="a" * 64,
        predictor_classification="failed_terminal_not_loaded",
    )
    assert record.metrics["duration_seconds"] == 0.0
    assert "model_count" not in record.params
    assert "best_model" not in record.params
    assert "decision_threshold" not in record.params
    assert record.params["autogluon_version"] == "1.5.0"


def test_receipt_rejects_unverified_item(tmp_path: Path) -> None:
    config = load_mlflow_config(
        write_config(tmp_path, valid_payload()), repository_root=tmp_path
    )
    record = _record(config.paths.research_v2_root)
    item = SyncItem(
        source_type=record.source_type,
        source_relative_path=record.source_relative_path,
        source_run_id=record.source_run_id,
        outcome="recoverable",
        mlflow_run_id="partial",
    )
    with pytest.raises(ValueError):
        _write_receipt(config, item, record)
    assert config.paths.receipts_root.exists() is False


@pytest.mark.parametrize(
    "corruption",
    (
        "status_schema",
        "exit_type",
        "unsigned_exit",
        "marker_exit",
        "inventory_schema",
        "inventory_hash",
        "inventory_size",
        "run_identity",
        "profile_identity",
        "config_identity",
        "timestamp",
        "negative_duration",
        "nan_duration",
        "empty_failure_codes",
        "completion_fields",
        "pid_fields",
        "missing_failed_marker",
        "stale_success_marker",
    ),
)
def test_exact_failed_autogluon_validator_rejects_corrupt_lifecycle(
    tmp_path: Path,
    corruption: str,
) -> None:
    config = load_mlflow_config(
        write_config(tmp_path, valid_payload()), repository_root=tmp_path
    )
    run = config.paths.autogluon_root / "failed-run"
    _write_exact_failed_autogluon(run)
    status_path = run / "execution_status.json"
    metadata_path = run / "run_metadata.json"
    marker_path = run / "_FAILED"
    inventory_path = run / "artifact_inventory.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    write_status = write_metadata = write_marker = write_inventory = False

    if corruption == "status_schema":
        status["unknown"] = True
        write_status = True
    elif corruption == "exit_type":
        status["windows_exit_code_unsigned"] = "23"
        write_status = True
    elif corruption == "unsigned_exit":
        status["windows_exit_code_unsigned"] = 24
        write_status = True
    elif corruption == "marker_exit":
        marker["child_process_exit_code"] = 99
        write_marker = True
    elif corruption == "inventory_schema":
        inventory["unknown"] = True
        write_inventory = True
    elif corruption in {"inventory_hash", "inventory_size"}:
        file_entry = next(
            entry for entry in inventory["entries"] if entry["type"] == "file"
        )
        if corruption == "inventory_hash":
            file_entry["sha256"] = "0" * 64
        else:
            file_entry["size_bytes"] += 1
        write_inventory = True
    elif corruption == "run_identity":
        metadata["run_id"] = "other-run"
        write_metadata = True
    elif corruption == "profile_identity":
        status["profile_sha256"] = "0" * 64
        write_status = True
    elif corruption == "config_identity":
        status["config_identity_sha256"] = "0" * 64
        write_status = True
    elif corruption == "timestamp":
        status["ended_at_utc"] = "not-a-timestamp"
        write_status = True
    elif corruption == "negative_duration":
        status["duration_seconds"] = -1.0
        write_status = True
    elif corruption == "nan_duration":
        status_path.write_text(
            json.dumps(status).replace(
                '"duration_seconds": 1.0', '"duration_seconds": NaN'
            ),
            encoding="utf-8",
        )
    elif corruption == "empty_failure_codes":
        status["failure_codes"] = []
        status["failure_reason"] = ""
        write_status = True
    elif corruption == "completion_fields":
        status["worker_completion_valid"] = True
        write_status = True
    elif corruption == "pid_fields":
        metadata["launched_process_pid"] = 456
        write_metadata = True
    elif corruption == "missing_failed_marker":
        marker_path.unlink()
    elif corruption == "stale_success_marker":
        (run / "_SUCCESS").write_text("{}\n", encoding="utf-8")
    if write_status:
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    if write_metadata:
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    if write_marker:
        marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    if write_inventory:
        inventory_path.write_text(
            json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
        )

    with pytest.raises(SourceValidationError):
        AutoGluonSourceAdapter().prepare(run, config)


def test_failed_autogluon_validation_does_not_infer_corruption_from_reason_text(
    tmp_path: Path,
) -> None:
    config = load_mlflow_config(
        write_config(tmp_path, valid_payload()), repository_root=tmp_path
    )
    run = config.paths.autogluon_root / "failed-run"
    _write_exact_failed_autogluon(run)
    status = json.loads((run / "execution_status.json").read_text(encoding="utf-8"))
    marker = json.loads((run / "_FAILED").read_text(encoding="utf-8"))
    codes = [*status["failure_codes"], "native_mismatch_diagnostic"]
    status["failure_codes"] = codes
    status["failure_reason"] = ";".join(codes)
    marker["failure_codes"] = codes
    marker["failure_reason"] = ";".join(codes)
    (run / "execution_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    (run / "artifact_inventory.json").write_text(
        json.dumps(build_inventory(run), indent=2) + "\n", encoding="utf-8"
    )
    (run / "_FAILED").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")

    record = AutoGluonSourceAdapter().prepare(run, config)
    assert record.terminal_status == "failed"
    assert record.tags["failure_codes"].endswith("native_mismatch_diagnostic")


def test_failed_terminalization_retry_uses_same_run(tmp_path: Path) -> None:
    client, experiment_id, config = _client_and_experiment(tmp_path)
    record = _record(config.paths.research_v2_root, status="failed")
    failing = RecordingFailOnceClient(client, _stage_failure("set_terminated"))
    interrupted = _sync_one(failing, experiment_id, record)
    assert interrupted.outcome == "recoverable"
    assert interrupted.mlflow_run_id is not None
    assert client.get_run(interrupted.mlflow_run_id).info.status == "RUNNING"

    resumed = _sync_one(client, experiment_id, record)
    assert resumed.outcome == "resumed"
    assert resumed.mlflow_run_id == interrupted.mlflow_run_id
    assert client.get_run(resumed.mlflow_run_id).info.status == "FAILED"
    assert len(client.search_runs([experiment_id])) == 1


def test_two_scoped_paths_with_same_leaf_create_distinct_runs(tmp_path: Path) -> None:
    client, experiment_id, config = _client_and_experiment(tmp_path)
    first = _record(config.paths.research_v2_root)
    second_dir = config.paths.research_v2_root / "other" / "run-1"
    second_dir.mkdir(parents=True)
    second = IndexedRun(
        **{
            **first.__dict__,
            "source_relative_path": "other/run-1",
            "local_source_path": second_dir,
            "artifacts": (),
        }
    )
    first_without_artifact = IndexedRun(**{**first.__dict__, "artifacts": ()})
    one = _sync_one(client, experiment_id, first_without_artifact)
    two = _sync_one(client, experiment_id, second)
    assert one.outcome == "created"
    assert two.outcome == "created"
    assert one.mlflow_run_id != two.mlflow_run_id
    assert len(client.search_runs([experiment_id])) == 2


def test_source_key_path_case_is_preserved(tmp_path: Path) -> None:
    record = _record(tmp_path)
    changed_case = IndexedRun(
        **{**record.__dict__, "source_relative_path": "Group/run-1"}
    )
    assert record.source_key != changed_case.source_key


def test_valid_human_readable_artifact_formats(tmp_path: Path) -> None:
    bodies = {
        "metadata.json": '{"ok": true}\n',
        "config.yaml": "value: 1\n",
        "table.csv": "a,b\n1,2\n",
        "_FAILED": "",
        "_SUCCESS": '{"schema_version": 1}\n',
    }
    for name, body in bodies.items():
        (tmp_path / name).write_text(body, encoding="utf-8")
    selected = select_indexed_artifacts(
        tmp_path,
        tuple(bodies),
        enabled=True,
        size_limit=1024,
    )
    assert {item.relative_path for item in selected} == set(bodies)


def test_artifact_revalidation_rejects_growth_above_cap(tmp_path: Path) -> None:
    source = tmp_path / "metadata.json"
    source.write_text('{"x": 1}\n', encoding="utf-8")
    selected = select_indexed_artifacts(
        tmp_path,
        ("metadata.json",),
        enabled=True,
        size_limit=20,
    )[0]
    source.write_text('{"value": "this is now too large"}\n', encoding="utf-8")
    with pytest.raises(ArtifactContentError, match="size cap"):
        revalidate_indexed_artifact(tmp_path, selected)


def test_artifact_revalidation_rejects_symlink_replacement(tmp_path: Path) -> None:
    source = tmp_path / "metadata.json"
    target = tmp_path / "target.json"
    source.write_text('{"x": 1}\n', encoding="utf-8")
    target.write_text('{"x": 1}\n', encoding="utf-8")
    selected = select_indexed_artifacts(
        tmp_path,
        ("metadata.json",),
        enabled=True,
        size_limit=1024,
    )[0]
    source.unlink()
    try:
        source.symlink_to(target)
    except OSError as error:
        pytest.skip(f"Symlink creation is unavailable: {error}")
    with pytest.raises(ArtifactContentError, match="symlink"):
        revalidate_indexed_artifact(tmp_path, selected)


def _write_early_failed_autogluon(run: Path) -> None:
    _write_exact_failed_autogluon(run)
    status_path = run / "execution_status.json"
    metadata_path = run / "run_metadata.json"
    marker_path = run / "_FAILED"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    codes = [
        "missing_expected_artifacts",
        "worker_process_not_started",
    ]
    for payload in (status, metadata, marker):
        payload["child_process_exit_code"] = None
        payload["windows_exit_code_unsigned"] = None
    status["child_pid"] = None
    status["launched_process_pid"] = None
    metadata["child_pid"] = None
    metadata["launched_process_pid"] = None
    metadata["local_operational_nonportable"].pop("worker_command")
    status["last_completed_observable_stage"] = "worker_not_observed"
    status["failure_codes"] = codes
    status["failure_reason"] = ";".join(codes)
    status["worker_completion_reason_codes"] = ["worker_process_not_started"]
    status["worker_completion_details"] = []
    metadata["worker_completion_reason_codes"] = ["worker_process_not_started"]
    marker["failure_codes"] = codes
    marker["failure_reason"] = ";".join(codes)
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (run / "artifact_inventory.json").write_text(
        json.dumps(build_inventory(run), indent=2) + "\n", encoding="utf-8"
    )
    marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")


def test_valid_early_autogluon_failure_before_predictor_creation(
    tmp_path: Path,
) -> None:
    config = load_mlflow_config(
        write_config(tmp_path, valid_payload()), repository_root=tmp_path
    )
    run = config.paths.autogluon_root / "early-failure"
    _write_early_failed_autogluon(run)
    record = AutoGluonSourceAdapter().prepare(run, config)
    assert record.terminal_status == "failed"
    assert record.params.get("child_process_exit_code") is None


def _write_failed_research(run: Path) -> None:
    run.mkdir(parents=True)
    failure = {"type": "RuntimeError", "message": "native failure"}
    status = {
        "schema_version": 2,
        "run_id": run.name,
        "status": "failed",
        "started_at_utc": "2026-01-01T00:00:00+00:00",
        "finished_at_utc": "2026-01-01T00:00:01+00:00",
        "completed_outer_folds": 0,
        "expected_outer_folds": 3,
        "failure": failure,
    }
    metadata = {
        "status": "failed",
        "finished_at_utc": status["finished_at_utc"],
        "failure": failure,
    }
    (run / "execution_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    (run / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (run / "_FAILED").write_bytes(b"")


@pytest.mark.parametrize(
    "corruption",
    (
        "timestamp",
        "ordering",
        "fold_bounds",
        "failure_payload",
        "metadata_disagreement",
        "stale_success",
        "stale_manifest",
        "forbidden_model",
    ),
)
def test_exact_failed_research_validator_rejects_corruption(
    tmp_path: Path,
    corruption: str,
) -> None:
    config = load_mlflow_config(
        write_config(tmp_path, valid_payload()), repository_root=tmp_path
    )
    run = config.paths.research_v2_root / "plan" / "candidate" / "failed-run"
    _write_failed_research(run)
    status_path = run / "execution_status.json"
    metadata_path = run / "run_metadata.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if corruption == "timestamp":
        status["started_at_utc"] = "not-a-timestamp"
    elif corruption == "ordering":
        status["finished_at_utc"] = "2025-01-01T00:00:00+00:00"
    elif corruption == "fold_bounds":
        status["completed_outer_folds"] = 4
    elif corruption == "failure_payload":
        status["failure"]["message"] = ""
    elif corruption == "metadata_disagreement":
        metadata["failure"]["message"] = "different"
    elif corruption == "stale_success":
        (run / "_SUCCESS").write_text("{}\n", encoding="utf-8")
    elif corruption == "stale_manifest":
        (run / "artifact_manifest.json").write_text("{}\n", encoding="utf-8")
    elif corruption == "forbidden_model":
        (run / "models").mkdir()
        (run / "models" / "model.pkl").write_bytes(b"model")
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(SourceValidationError):
        ResearchV2SourceAdapter().prepare(run, config)


def _write_full_failed_research(run: Path) -> None:
    run.mkdir(parents=True)
    failure = {"type": "RuntimeError", "message": "native failure"}
    started = "2026-01-01T00:00:00+00:00"
    finished = "2026-01-01T00:00:01+00:00"
    hashes = {
        "plan": "1" * 64,
        "feature_pipeline": "2" * 64,
        "candidate_adapter": "3" * 64,
        "candidate": "4" * 64,
        "source": "5" * 64,
        "loaded_modules": "6" * 64,
    }
    config_path = (
        PROJECT_ROOT / "configs/research_v2/manual_lightgbm_te_v1_compat_smoke.yaml"
    )
    resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    plan_path = PROJECT_ROOT / resolved["evaluation_plan_path"]
    resolved["evaluation_plan"] = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    metadata = {
        "schema_version": 2,
        "experiment_id": resolved["experiment"]["id"],
        "plan_id": resolved["evaluation_plan"]["plan"]["id"],
        "feature_pipeline_id": resolved["feature_pipeline"]["id"],
        "candidate_adapter_id": resolved["candidate_adapter"]["id"],
        "hashes": hashes,
        "status": "failed",
        "started_at_utc": started,
        "environment": {
            "python": "3.12",
            "numpy": "2",
            "pandas": "2",
            "scikit_learn": "1",
            "lightgbm": "4",
            "pyarrow": "18",
        },
        "runtime": {
            "process_started_at_utc": started,
            "preflight_at_utc": started,
            "entry_point": "scripts/run.py",
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
        "competition_assets_accessed": False,
        "tracking_enabled": False,
        "run_id": run.name,
        "finished_at_utc": finished,
        "failure": failure,
    }
    status = {
        "schema_version": 2,
        "run_id": run.name,
        "status": "failed",
        "started_at_utc": started,
        "finished_at_utc": finished,
        "completed_outer_folds": 0,
        "expected_outer_folds": 3,
        "failure": failure,
    }
    (run / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    (run / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (run / "execution_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    (run / "_FAILED").write_bytes(b"")


def test_valid_full_native_failed_research_metadata(tmp_path: Path) -> None:
    config = load_mlflow_config(
        write_config(tmp_path, valid_payload()), repository_root=tmp_path
    )
    run = config.paths.research_v2_root / "plan" / "candidate" / "full-failure"
    _write_full_failed_research(run)
    record = ResearchV2SourceAdapter().prepare(run, config)
    assert record.terminal_status == "failed"
    assert record.params["candidate_sha256"] == "4" * 64


def test_full_failed_research_rejects_corrupt_runtime_context(tmp_path: Path) -> None:
    config = load_mlflow_config(
        write_config(tmp_path, valid_payload()), repository_root=tmp_path
    )
    run = config.paths.research_v2_root / "plan" / "candidate" / "full-failure"
    _write_full_failed_research(run)
    path = run / "run_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["runtime"]["competition_assets_accessed"] = True
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(SourceValidationError):
        ResearchV2SourceAdapter().prepare(run, config)


def test_indexed_artifact_record_is_immutable() -> None:
    artifact = IndexedArtifact(
        relative_path="metadata.json",
        content_type="json",
        size_bytes=2,
        sha256="a" * 64,
        max_size_bytes=10,
    )
    with pytest.raises(AttributeError):
        artifact.size_bytes = 3  # type: ignore[misc]
