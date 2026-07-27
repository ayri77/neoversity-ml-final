"""Idempotent synchronization of validated filesystem artifacts into MLflow."""

from __future__ import annotations

import importlib
import json
import math
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from src.churn_ml.mlflow_config import MLflowIndexConfig
from src.churn_ml.mlflow_mapping import IndexedRun
from src.churn_ml.mlflow_sources import (
    SourceAdapter,
    SourceAdapterRegistry,
    SourceError,
    SourceSkip,
)


SYNC_SCHEMA_VERSION = 1
Outcome = Literal[
    "created",
    "unchanged",
    "resumed",
    "validated",
    "skipped",
    "rejected",
    "error",
]


@dataclass(frozen=True)
class SyncItem:
    source_type: str
    source_relative_path: str
    source_run_id: str | None
    outcome: Outcome
    reason_code: str | None = None
    message: str | None = None
    mlflow_run_id: str | None = None


@dataclass(frozen=True)
class SyncSummary:
    schema_version: int
    dry_run: bool
    fail_fast: bool
    source_types: tuple[str, ...]
    counts: dict[str, int]
    items: tuple[SyncItem, ...]
    receipt_relative_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dry_run": self.dry_run,
            "fail_fast": self.fail_fast,
            "source_types": list(self.source_types),
            "counts": dict(self.counts),
            "items": [asdict(item) for item in self.items],
            "receipt_relative_path": self.receipt_relative_path,
        }

    @property
    def has_failures(self) -> bool:
        return bool(self.counts.get("rejected", 0) or self.counts.get("error", 0))


def sync_sources(
    config: MLflowIndexConfig,
    *,
    registry: SourceAdapterRegistry,
    source_types: tuple[str, ...],
    run_dir: Path | None = None,
    dry_run: bool = False,
    fail_fast: bool = False,
) -> SyncSummary:
    """Validate and optionally index independent source runs."""
    items: list[SyncItem] = []
    client: Any | None = None
    experiment_ids: dict[str, str] = {}
    stop = False
    for source_type in source_types:
        if stop:
            break
        adapter = registry.get(source_type)
        try:
            directories = adapter.discover(config, run_dir=run_dir)
        except SourceError as error:
            items.append(
                SyncItem(
                    source_type=source_type,
                    source_relative_path=_portable_input_path(run_dir),
                    source_run_id=run_dir.name if run_dir is not None else None,
                    outcome="rejected",
                    reason_code=error.code,
                    message=str(error),
                )
            )
            if fail_fast:
                break
            continue
        for directory in directories:
            relative = _relative_or_name(directory, adapter.source_root(config))
            try:
                record = adapter.prepare(directory, config)
            except SourceSkip as error:
                items.append(
                    SyncItem(
                        source_type=source_type,
                        source_relative_path=relative,
                        source_run_id=directory.name,
                        outcome="skipped",
                        reason_code=error.code,
                        message=str(error),
                    )
                )
                continue
            except SourceError as error:
                items.append(
                    SyncItem(
                        source_type=source_type,
                        source_relative_path=relative,
                        source_run_id=directory.name,
                        outcome="rejected",
                        reason_code=error.code,
                        message=str(error),
                    )
                )
                if fail_fast:
                    stop = True
                    break
                continue
            except Exception as error:
                items.append(
                    SyncItem(
                        source_type=source_type,
                        source_relative_path=relative,
                        source_run_id=directory.name,
                        outcome="error",
                        reason_code="unexpected_source_error",
                        message=f"{type(error).__name__}: {error}",
                    )
                )
                if fail_fast:
                    stop = True
                    break
                continue

            if dry_run:
                items.append(
                    SyncItem(
                        source_type=source_type,
                        source_relative_path=record.source_relative_path,
                        source_run_id=record.source_run_id,
                        outcome="validated",
                        message="Source is valid and no MLflow storage was allocated.",
                    )
                )
                continue
            try:
                if client is None:
                    client = _new_mlflow_client(config.paths.tracking_uri)
                experiment_id = experiment_ids.get(source_type)
                if experiment_id is None:
                    experiment_id = _get_or_create_experiment(
                        client,
                        adapter,
                        config,
                    )
                    experiment_ids[source_type] = experiment_id
                item = _sync_one(client, experiment_id, record)
                items.append(item)
            except Exception as error:
                items.append(
                    SyncItem(
                        source_type=source_type,
                        source_relative_path=record.source_relative_path,
                        source_run_id=record.source_run_id,
                        outcome="error",
                        reason_code="mlflow_sync_error",
                        message=f"{type(error).__name__}: {error}",
                    )
                )
                if fail_fast:
                    stop = True
                    break

    counts = _count_outcomes(items)
    provisional = SyncSummary(
        schema_version=SYNC_SCHEMA_VERSION,
        dry_run=dry_run,
        fail_fast=fail_fast,
        source_types=source_types,
        counts=counts,
        items=tuple(items),
        receipt_relative_path=None,
    )
    if dry_run:
        return provisional
    receipt = _write_receipt(config, provisional)
    return SyncSummary(
        schema_version=provisional.schema_version,
        dry_run=False,
        fail_fast=fail_fast,
        source_types=source_types,
        counts=counts,
        items=tuple(items),
        receipt_relative_path=receipt,
    )


def _new_mlflow_client(tracking_uri: str) -> Any:
    """Import MLflow only for an actual synchronization operation."""
    tracking = importlib.import_module("mlflow.tracking")
    return tracking.MlflowClient(tracking_uri=tracking_uri)


def _get_or_create_experiment(
    client: Any,
    adapter: SourceAdapter,
    config: MLflowIndexConfig,
) -> str:
    name = adapter.experiment_name(config)
    existing = client.get_experiment_by_name(name)
    if existing is not None:
        return str(existing.experiment_id)
    artifact_location = (config.paths.artifact_root / str(adapter.source_type)).as_uri()
    return str(client.create_experiment(name, artifact_location=artifact_location))


def _sync_one(client: Any, experiment_id: str, record: IndexedRun) -> SyncItem:
    immutable_tags = {
        "mlflow_index.sync_schema_version": str(SYNC_SCHEMA_VERSION),
        "mlflow_index.source_key": record.source_key,
        "mlflow_index.source_type": record.source_type,
        "mlflow_index.source_run_id": record.source_run_id,
        "mlflow_index.source_identity": record.source_identity,
        "mlflow_index.source_relative_path": record.source_relative_path,
    }
    matches = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=(f"tags.mlflow_index.source_key = '{record.source_key}'"),
        max_results=100,
    )
    if len(matches) > 1:
        return _rejected(
            record, "duplicate_index_entries", "Multiple MLflow runs exist"
        )
    expected_params = {
        key: _parameter_value(value) for key, value in sorted(record.params.items())
    }
    expected_params["sync_schema_version"] = str(SYNC_SCHEMA_VERSION)
    created = False
    resumed = False
    if matches:
        run = matches[0]
        run_id = str(run.info.run_id)
        existing_identity = run.data.tags.get("mlflow_index.source_identity")
        if existing_identity != record.source_identity:
            return _rejected(
                record,
                "source_mutation_detected",
                "The same source run ID now has a different immutable source identity",
                run_id,
            )
        tag_conflicts = {
            key: (run.data.tags.get(key), expected)
            for key, expected in immutable_tags.items()
            if run.data.tags.get(key) not in {None, expected}
        }
        if tag_conflicts:
            return _rejected(
                record,
                "immutable_tag_conflict",
                f"Immutable MLflow tags differ: {tag_conflicts}",
                run_id,
            )
        parameter_conflicts = {
            key: (run.data.params[key], expected)
            for key, expected in expected_params.items()
            if key in run.data.params and run.data.params[key] != expected
        }
        if parameter_conflicts:
            return _rejected(
                record,
                "immutable_parameter_conflict",
                f"Immutable MLflow params differ: {parameter_conflicts}",
                run_id,
            )
        if run.data.tags.get("mlflow_index.sync_complete") == "true":
            return SyncItem(
                source_type=record.source_type,
                source_relative_path=record.source_relative_path,
                source_run_id=record.source_run_id,
                outcome="unchanged",
                mlflow_run_id=run_id,
            )
        resumed = True
    else:
        initial_tags = {
            **immutable_tags,
            **record.tags,
            "mlflow_index.sync_complete": "false",
            "mlflow_index.local_source_path_nonportable": str(record.local_source_path),
        }
        run = client.create_run(experiment_id, tags=initial_tags)
        run_id = str(run.info.run_id)
        created = True

    try:
        current = client.get_run(run_id)
        for key, parameter_value in expected_params.items():
            if key not in current.data.params:
                client.log_param(run_id, key, parameter_value)
        for key, metric_value in sorted(record.metrics.items()):
            current_value = current.data.metrics.get(key)
            if current_value is None or not math.isclose(
                float(current_value), metric_value, rel_tol=0.0, abs_tol=0.0
            ):
                client.log_metric(run_id, key, metric_value)
        for key, value in {**record.tags, **immutable_tags}.items():
            client.set_tag(run_id, key, value)
        for relative in record.artifact_relative_paths:
            source = record.local_source_path / Path(*relative.split("/"))
            parent = Path(relative).parent.as_posix()
            artifact_path = (
                "source_metadata" if parent == "." else f"source_metadata/{parent}"
            )
            client.log_artifact(run_id, str(source), artifact_path=artifact_path)
        client.set_tag(run_id, "mlflow_index.sync_complete", "true")
        client.set_terminated(run_id, status=record.mlflow_status)
    except Exception as error:
        try:
            client.set_tag(
                run_id,
                "mlflow_index.sync_error",
                f"{type(error).__name__}: {error}"[:5000],
            )
            client.set_terminated(run_id, status="FAILED")
        except Exception:
            pass
        raise
    return SyncItem(
        source_type=record.source_type,
        source_relative_path=record.source_relative_path,
        source_run_id=record.source_run_id,
        outcome="created" if created else "resumed" if resumed else "unchanged",
        mlflow_run_id=run_id,
    )


def _rejected(
    record: IndexedRun,
    code: str,
    message: str,
    run_id: str | None = None,
) -> SyncItem:
    return SyncItem(
        source_type=record.source_type,
        source_relative_path=record.source_relative_path,
        source_run_id=record.source_run_id,
        outcome="rejected",
        reason_code=code,
        message=message,
        mlflow_run_id=run_id,
    )


def _parameter_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _count_outcomes(items: list[SyncItem]) -> dict[str, int]:
    names = (
        "created",
        "unchanged",
        "resumed",
        "validated",
        "skipped",
        "rejected",
        "error",
    )
    return {name: sum(item.outcome == name for item in items) for name in names}


def _write_receipt(config: MLflowIndexConfig, summary: SyncSummary) -> str:
    config.paths.receipts_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = f"sync-{timestamp}-{uuid.uuid4().hex[:12]}.json"
    path = config.paths.receipts_root / name
    payload = {
        **summary.to_dict(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tracking_backend_nonportable": str(config.paths.backend_store),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path.relative_to(config.paths.repository_root).as_posix()


def _relative_or_name(path: Path, root: Path) -> str:
    try:
        return (
            path.resolve(strict=False)
            .relative_to(root.resolve(strict=False))
            .as_posix()
        )
    except ValueError:
        return path.name


def _portable_input_path(path: Path | None) -> str:
    return path.name if path is not None else "."
