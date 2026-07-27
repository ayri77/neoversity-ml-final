"""Idempotent two-phase synchronization into the optional local MLflow index."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from src.churn_ml.mlflow_artifacts import revalidate_indexed_artifact
from src.churn_ml.mlflow_config import MLflowIndexConfig
from src.churn_ml.mlflow_mapping import IndexedRun
from src.churn_ml.mlflow_sources import (
    SourceAdapter,
    SourceAdapterRegistry,
    SourceError,
    SourceSkip,
)


SYNC_SCHEMA_VERSION = 2
Outcome = Literal[
    "created",
    "unchanged",
    "resumed",
    "validated",
    "skipped",
    "rejected",
    "recoverable",
    "error",
]
_SUCCESS_OUTCOMES = {"created", "unchanged", "resumed"}


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
    receipt_relative_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dry_run": self.dry_run,
            "fail_fast": self.fail_fast,
            "source_types": list(self.source_types),
            "counts": dict(self.counts),
            "items": [asdict(item) for item in self.items],
            "receipt_relative_path": self.receipt_relative_path,
            "receipt_relative_paths": list(self.receipt_relative_paths),
        }

    @property
    def has_failures(self) -> bool:
        return any(
            self.counts.get(name, 0) for name in ("rejected", "recoverable", "error")
        )


def sync_sources(
    config: MLflowIndexConfig,
    *,
    registry: SourceAdapterRegistry,
    source_types: tuple[str, ...],
    run_dir: Path | None = None,
    dry_run: bool = False,
    fail_fast: bool = False,
) -> SyncSummary:
    """Validate sources and optionally commit their searchable MLflow records."""
    items: list[SyncItem] = []
    receipts: list[str] = []
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
                    experiment_id = _get_or_create_experiment(client, adapter, config)
                    experiment_ids[source_type] = experiment_id
                item = _sync_one(client, experiment_id, record)
                if item.outcome in _SUCCESS_OUTCOMES:
                    receipts.append(_write_receipt(config, item, record))
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
            if fail_fast and items[-1].outcome in {
                "rejected",
                "recoverable",
                "error",
            }:
                stop = True
                break

    counts = _count_outcomes(items)
    receipt_paths = tuple(sorted(set(receipts)))
    return SyncSummary(
        schema_version=SYNC_SCHEMA_VERSION,
        dry_run=dry_run,
        fail_fast=fail_fast,
        source_types=source_types,
        counts=counts,
        items=tuple(items),
        receipt_relative_path=receipt_paths[0] if receipt_paths else None,
        receipt_relative_paths=receipt_paths,
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
    """Commit one record with a recoverable completion marker written last."""
    immutable_tags = {
        "mlflow_index.sync_schema_version": str(SYNC_SCHEMA_VERSION),
        "mlflow_index.source_key": record.source_key,
        "mlflow_index.source_type": record.source_type,
        "mlflow_index.source_run_id": record.source_run_id,
        "mlflow_index.source_identity": record.source_identity,
        "mlflow_index.source_relative_path": record.source_relative_path,
    }
    expected_tags = {**record.tags, **immutable_tags}
    expected_params = {
        key: _parameter_value(value) for key, value in sorted(record.params.items())
    }
    expected_params["sync_schema_version"] = str(SYNC_SCHEMA_VERSION)
    matches = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow_index.source_key = '{record.source_key}'",
        max_results=100,
    )
    if len(matches) > 1:
        return _rejected(
            record, "duplicate_index_entries", "Multiple MLflow runs exist"
        )

    created = False
    if matches:
        run_id = str(matches[0].info.run_id)
        current = client.get_run(run_id)
        conflict = _immutable_conflict(
            current,
            record,
            immutable_tags,
            expected_params,
        )
        if conflict is not None:
            return conflict
        extras = _unrepairable_extras(client, current, run_id, record, expected_tags)
        if extras is not None:
            return _rejected(record, extras[0], extras[1], run_id)
        if current.data.tags.get("mlflow_index.sync_complete") == "true":
            problems = _committed_problems(
                client,
                run_id,
                record,
                expected_params,
                expected_tags,
            )
            if not problems:
                return _item(record, "unchanged", run_id)
    else:
        initial_tags = {
            **expected_tags,
            "mlflow_index.sync_complete": "false",
            "mlflow_index.local_source_path_nonportable": str(record.local_source_path),
        }
        run_id = str(client.create_run(experiment_id, tags=initial_tags).info.run_id)
        created = True

    try:
        client.set_tag(run_id, "mlflow_index.sync_complete", "false")
        current = client.get_run(run_id)
        for key, parameter_value in expected_params.items():
            if key not in current.data.params:
                client.log_param(run_id, key, parameter_value)
        for key, metric_value in sorted(record.metrics.items()):
            current_value = client.get_run(run_id).data.metrics.get(key)
            if current_value is None or not math.isclose(
                float(current_value), metric_value, rel_tol=0.0, abs_tol=0.0
            ):
                client.log_metric(run_id, key, metric_value)
        for key, tag_value in sorted(expected_tags.items()):
            client.set_tag(run_id, key, tag_value)
        if "mlflow_index.sync_error" in client.get_run(run_id).data.tags:
            client.delete_tag(run_id, "mlflow_index.sync_error")
        for artifact in record.artifacts:
            source = revalidate_indexed_artifact(record.local_source_path, artifact)
            parent = Path(artifact.relative_path).parent.as_posix()
            artifact_path = (
                "source_metadata" if parent == "." else f"source_metadata/{parent}"
            )
            client.log_artifact(run_id, str(source), artifact_path=artifact_path)
        precommit = _payload_problems(
            client,
            run_id,
            record,
            expected_params,
            expected_tags,
            expected_sync_complete="false",
        )
        if precommit:
            raise RuntimeError("precommit verification failed: " + "; ".join(precommit))
        client.set_terminated(run_id, status=record.mlflow_status)
        if client.get_run(run_id).info.status != record.mlflow_status:
            raise RuntimeError("terminal status verification failed")
        client.set_tag(run_id, "mlflow_index.sync_complete", "true")
        final = _committed_problems(
            client,
            run_id,
            record,
            expected_params,
            expected_tags,
        )
        if final:
            raise RuntimeError(
                "final read-only verification failed: " + "; ".join(final)
            )
    except Exception as error:
        try:
            marker = client.get_run(run_id).data.tags.get("mlflow_index.sync_complete")
            if marker != "true":
                client.set_tag(
                    run_id,
                    "mlflow_index.sync_error",
                    f"{type(error).__name__}: {error}"[:5000],
                )
        except Exception:
            pass
        return SyncItem(
            source_type=record.source_type,
            source_relative_path=record.source_relative_path,
            source_run_id=record.source_run_id,
            outcome="recoverable",
            reason_code="partial_sync_recoverable",
            message=f"{type(error).__name__}: {error}",
            mlflow_run_id=run_id,
        )
    return _item(record, "created" if created else "resumed", run_id)


def _immutable_conflict(
    current: Any,
    record: IndexedRun,
    immutable_tags: dict[str, str],
    expected_params: dict[str, str],
) -> SyncItem | None:
    run_id = str(current.info.run_id)
    identity = current.data.tags.get("mlflow_index.source_identity")
    if identity not in {None, record.source_identity}:
        return _rejected(
            record,
            "source_mutation_detected",
            "The same scoped source now has a different immutable identity",
            run_id,
        )
    tag_conflicts = {
        key: (current.data.tags.get(key), expected)
        for key, expected in immutable_tags.items()
        if current.data.tags.get(key) not in {None, expected}
    }
    if tag_conflicts:
        return _rejected(
            record,
            "immutable_tag_conflict",
            f"Immutable MLflow tags differ: {tag_conflicts}",
            run_id,
        )
    parameter_conflicts = {
        key: (current.data.params[key], expected)
        for key, expected in expected_params.items()
        if key in current.data.params and current.data.params[key] != expected
    }
    if parameter_conflicts:
        return _rejected(
            record,
            "immutable_parameter_conflict",
            f"Immutable MLflow params differ: {parameter_conflicts}",
            run_id,
        )
    return None


def _unrepairable_extras(
    client: Any,
    current: Any,
    run_id: str,
    record: IndexedRun,
    expected_tags: dict[str, str],
) -> tuple[str, str] | None:
    expected_params = set(record.params) | {"sync_schema_version"}
    extra_params = sorted(set(current.data.params) - expected_params)
    if extra_params:
        return "unexpected_parameters", f"Unexpected immutable params: {extra_params}"
    extra_metrics = sorted(set(current.data.metrics) - set(record.metrics))
    if extra_metrics:
        return (
            "unexpected_metrics",
            f"Unexpected metrics cannot be removed: {extra_metrics}",
        )
    allowed_tags = set(expected_tags) | {
        "mlflow_index.sync_complete",
        "mlflow_index.sync_error",
        "mlflow_index.local_source_path_nonportable",
    }
    extra_tags = sorted(
        key
        for key in current.data.tags
        if key not in allowed_tags and not key.startswith("mlflow.")
    )
    if extra_tags:
        return "unexpected_tags", f"Unexpected tags cannot be removed: {extra_tags}"
    actual_artifacts = _artifact_inventory(client, run_id)
    expected_artifacts = {
        f"source_metadata/{item.relative_path}": (item.size_bytes, item.sha256)
        for item in record.artifacts
    }
    extras = sorted(set(actual_artifacts) - set(expected_artifacts))
    if extras:
        return (
            "unexpected_artifacts",
            f"Unexpected artifacts cannot be removed: {extras}",
        )
    return None


def _committed_problems(
    client: Any,
    run_id: str,
    record: IndexedRun,
    expected_params: dict[str, str],
    expected_tags: dict[str, str],
) -> list[str]:
    problems = _payload_problems(
        client,
        run_id,
        record,
        expected_params,
        expected_tags,
        expected_sync_complete="true",
    )
    run = client.get_run(run_id)
    if run.info.status != record.mlflow_status:
        problems.append(
            f"status={run.info.status!r}, expected={record.mlflow_status!r}"
        )
    return problems


def _payload_problems(
    client: Any,
    run_id: str,
    record: IndexedRun,
    expected_params: dict[str, str],
    expected_tags: dict[str, str],
    *,
    expected_sync_complete: str,
) -> list[str]:
    run = client.get_run(run_id)
    problems: list[str] = []
    if dict(run.data.params) != expected_params:
        problems.append("params differ")
    for key, tag_value in expected_tags.items():
        if run.data.tags.get(key) != tag_value:
            problems.append(f"tag {key} differs")
    if run.data.tags.get("mlflow_index.sync_complete") != expected_sync_complete:
        problems.append("sync completion marker differs")
    if set(run.data.metrics) != set(record.metrics):
        problems.append("metric keys differ")
    else:
        for key, metric_value in record.metrics.items():
            actual = run.data.metrics.get(key)
            if actual is None or not math.isclose(
                float(actual), metric_value, rel_tol=0.0, abs_tol=0.0
            ):
                problems.append(f"metric {key} differs")
    actual_artifacts = _artifact_inventory(client, run_id)
    expected_artifacts = {
        f"source_metadata/{item.relative_path}": (item.size_bytes, item.sha256)
        for item in record.artifacts
    }
    if actual_artifacts != expected_artifacts:
        problems.append("artifact inventory differs")
    return problems


def _artifact_inventory(client: Any, run_id: str) -> dict[str, tuple[int, str]]:
    files: dict[str, tuple[int, str]] = {}

    def walk(path: str) -> None:
        for item in client.list_artifacts(run_id, path):
            item_path = str(item.path).replace("\\", "/")
            if item.is_dir:
                walk(item_path)
            else:
                size = item.file_size
                if type(size) is not int or size < 0:
                    raise RuntimeError(f"MLflow artifact size is invalid: {item_path}")
                downloaded = Path(client.download_artifacts(run_id, item_path))
                if not downloaded.is_file() or downloaded.stat().st_size != size:
                    raise RuntimeError(
                        f"MLflow artifact download differs from inventory: {item_path}"
                    )
                files[item_path] = (size, _sha256_file(downloaded))

    walk("")
    return files


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _item(record: IndexedRun, outcome: Outcome, run_id: str) -> SyncItem:
    return SyncItem(
        source_type=record.source_type,
        source_relative_path=record.source_relative_path,
        source_run_id=record.source_run_id,
        outcome=outcome,
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
        "recoverable",
        "error",
    )
    return {name: sum(item.outcome == name for item in items) for name in names}


def _write_receipt(
    config: MLflowIndexConfig,
    item: SyncItem,
    record: IndexedRun,
) -> str:
    if item.mlflow_run_id is None or item.outcome not in _SUCCESS_OUTCOMES:
        raise ValueError("Receipt requires a successfully verified MLflow record")
    payload = {
        "schema_version": SYNC_SCHEMA_VERSION,
        "source_type": record.source_type,
        "source_key": record.source_key,
        "source_run_id": record.source_run_id,
        "source_relative_path": record.source_relative_path,
        "source_identity": record.source_identity,
        "mlflow_run_id": item.mlflow_run_id,
        "final_status": record.mlflow_status,
    }
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    directory = config.paths.receipts_root / record.source_type
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.json"
    if path.exists():
        existing = path.read_bytes()
        if existing != encoded:
            raise RuntimeError(
                "Deterministic receipt path already contains different content"
            )
    else:
        temporary = directory / f".{digest}.tmp"
        temporary.write_bytes(encoded)
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
