"""Pure mappings from validated filesystem metadata to MLflow index records."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Mapping


SourceType = str
TerminalStatus = Literal["completed", "failed"]
MLflowTerminalStatus = Literal["FINISHED", "FAILED"]


@dataclass(frozen=True)
class IndexedRun:
    """A validated, portable source record ready for optional MLflow indexing."""

    source_type: SourceType
    source_run_id: str
    source_relative_path: str
    source_identity: str
    terminal_status: TerminalStatus
    mlflow_status: MLflowTerminalStatus
    params: dict[str, Any]
    tags: dict[str, str]
    metrics: dict[str, float]
    artifact_relative_paths: tuple[str, ...]
    local_source_path: Path

    @property
    def source_key(self) -> str:
        return canonical_sha256(
            {
                "source_type": self.source_type,
                "source_run_id": self.source_run_id,
            }
        )


def canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible portable content deterministically."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_research_mapping(
    *,
    run_dir: Path,
    source_relative_path: str,
    metadata: Mapping[str, Any],
    status: Mapping[str, Any],
    resolved_config: Mapping[str, Any] | None,
    terminal_status: TerminalStatus,
    source_identity: str,
    aggregate: Mapping[str, Any] | None = None,
    threshold_summary: Mapping[str, Any] | None = None,
    threshold_standard_deviation: float | None = None,
    artifact_relative_paths: tuple[str, ...] = (),
) -> IndexedRun:
    """Map structurally/semantically validated research-v2 metadata."""
    hashes = _mapping(metadata.get("hashes"))
    plan = _mapping(_mapping(resolved_config).get("evaluation_plan"))
    outer = _mapping(plan.get("outer_evaluation"))
    threshold_policy = _mapping(plan.get("threshold_policy"))
    dataset = _mapping(resolved_config).get("dataset")
    dataset_version = (
        _mapping(dataset).get("version") or metadata.get("dataset_version") or "unknown"
    )
    params: dict[str, Any] = {
        "source_run_id": run_dir.name,
        "research_schema_version": metadata.get("schema_version", 2),
        "plan_id": metadata.get("plan_id"),
        "plan_sha256": hashes.get("plan"),
        "dataset_version": dataset_version,
        "dataset_identity_sha256": _optional_hash(
            _mapping(plan.get("dataset")) or None
        ),
        "pipeline_id": metadata.get("feature_pipeline_id"),
        "pipeline_sha256": hashes.get("feature_pipeline"),
        "adapter_id": metadata.get("candidate_adapter_id"),
        "adapter_sha256": hashes.get("candidate_adapter"),
        "candidate_sha256": hashes.get("candidate"),
        "source_provenance_sha256": hashes.get("source"),
        "loaded_modules_sha256": hashes.get("loaded_modules"),
        "plan_schema_version": plan.get("schema_version"),
        "repeat_count": len(outer.get("repeat_seeds", []))
        if type(outer.get("repeat_seeds")) is list
        else None,
        "fold_count_per_repeat": outer.get("n_splits"),
        "threshold_policy": threshold_policy.get("id"),
        "source_identity": source_identity,
    }
    metrics: dict[str, float] = {}
    if terminal_status == "completed":
        aggregate_metrics = _mapping(_mapping(aggregate).get("metrics"))
        for name in (
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "roc_auc",
            "average_precision",
            "brier_score",
        ):
            summary = _mapping(aggregate_metrics.get(name))
            _metric(metrics, name, summary.get("mean"))
            _metric(metrics, f"repeat_{name}_mean", summary.get("mean"))
            _metric(
                metrics,
                f"repeat_{name}_std",
                summary.get("sample_standard_deviation"),
            )
        thresholds = _mapping(threshold_summary)
        _metric(metrics, "threshold_median", thresholds.get("median"))
        _metric(metrics, "threshold_minimum", thresholds.get("minimum"))
        _metric(metrics, "threshold_maximum", thresholds.get("maximum"))
        minimum = _finite_float(thresholds.get("minimum"))
        maximum = _finite_float(thresholds.get("maximum"))
        if minimum is not None and maximum is not None:
            metrics["threshold_range"] = maximum - minimum
        _metric(metrics, "threshold_std", threshold_standard_deviation)
        _metric(
            metrics,
            "duration_seconds",
            metadata.get("evaluation_duration_seconds"),
        )
    else:
        _metric(metrics, "duration_seconds", _terminal_duration(status))
    tags = {
        "source_type": "research_v2",
        "source_relative_path": source_relative_path,
        "terminal_status": terminal_status,
        "metric_direction.balanced_accuracy": "higher_is_better",
        "metric_direction.sensitivity": "higher_is_better",
        "metric_direction.specificity": "higher_is_better",
        "metric_direction.roc_auc": "higher_is_better",
        "metric_direction.average_precision": "higher_is_better",
        "metric_direction.brier_score": "lower_is_better",
        "filesystem_authority": "authoritative",
        "mlflow_role": "searchable_metadata_index",
    }
    failure = _mapping(status.get("failure"))
    if terminal_status == "failed" and failure:
        tags["failure_type"] = str(failure.get("type", "unknown"))
        tags["failure_message"] = str(failure.get("message", ""))[:5000]
    return IndexedRun(
        source_type="research_v2",
        source_run_id=run_dir.name,
        source_relative_path=source_relative_path,
        source_identity=source_identity,
        terminal_status=terminal_status,
        mlflow_status="FINISHED" if terminal_status == "completed" else "FAILED",
        params=_without_none(params),
        tags=tags,
        metrics=metrics,
        artifact_relative_paths=artifact_relative_paths,
        local_source_path=run_dir,
    )


def build_autogluon_mapping(
    *,
    run_dir: Path,
    source_relative_path: str,
    metadata: Mapping[str, Any],
    status: Mapping[str, Any],
    resolved_config: Mapping[str, Any] | None,
    profile_resolution: Mapping[str, Any] | None,
    worker_result: Mapping[str, Any] | None,
    inspection_summary: Mapping[str, Any] | None,
    terminal_status: TerminalStatus,
    source_identity: str,
    predictor_classification: str,
    artifact_relative_paths: tuple[str, ...] = (),
) -> IndexedRun:
    """Map validated standalone-runner metadata without loading a predictor."""
    profile = _mapping(metadata.get("profile"))
    resources = _mapping(metadata.get("resources"))
    resolution = _mapping(profile_resolution)
    completion = _mapping(worker_result)
    inspection = _mapping(inspection_summary)
    config = _mapping(resolved_config)
    configured_resources = _mapping(config.get("resources"))
    included = profile.get("included_model_types")
    excluded = profile.get("excluded_model_types")
    models = completion.get("model_names")
    model_count = len(models) if type(models) is list else None
    params: dict[str, Any] = {
        "source_run_id": run_dir.name,
        "runner_schema_version": metadata.get("schema_version", 1),
        "config_identity_sha256": metadata.get("config_identity_sha256"),
        "profile_id": metadata.get("profile_id"),
        "profile_sha256": metadata.get("profile_sha256"),
        "dataset_version": metadata.get("dataset_version"),
        "requested_seed": metadata.get("requested_seed"),
        "autogluon_version": completion.get("autogluon_version")
        or profile.get("autogluon_version"),
        "included_families": included,
        "excluded_families": excluded,
        "resolved_families": resolution.get("resolved_families"),
        "time_limit_seconds": resources.get("time_limit_seconds")
        or configured_resources.get("time_limit_seconds"),
        "num_cpus": resources.get("num_cpus") or configured_resources.get("num_cpus"),
        "gpu_budget": resources.get("gpu_budget")
        if "gpu_budget" in resources
        else configured_resources.get("num_gpus"),
        "best_model": completion.get("best_model")
        if terminal_status == "completed"
        else None,
        "decision_threshold": completion.get("decision_threshold")
        if terminal_status == "completed"
        else None,
        "model_count": model_count,
        "child_process_exit_code": status.get("child_process_exit_code"),
        "windows_exit_code_unsigned": status.get("windows_exit_code_unsigned"),
        "failure_codes": status.get("failure_codes"),
        "last_completed_observable_stage": status.get(
            "last_completed_observable_stage"
        ),
        "source_identity": source_identity,
    }
    metrics: dict[str, float] = {}
    _metric(
        metrics,
        "duration_seconds",
        status.get("duration_seconds") or metadata.get("duration_seconds"),
    )
    tags = {
        "source_type": "autogluon",
        "source_relative_path": source_relative_path,
        "terminal_status": terminal_status,
        "predictor_classification": predictor_classification,
        "predictor_loading_attempted": "false",
        "filesystem_authority": "authoritative",
        "mlflow_role": "searchable_metadata_index",
    }
    if terminal_status == "failed":
        codes = status.get("failure_codes")
        if type(codes) is list:
            tags["failure_codes"] = ",".join(str(item) for item in codes)[:5000]
        reason = status.get("failure_reason")
        if type(reason) is str:
            tags["failure_reason"] = reason[:5000]
    if terminal_status == "completed":
        tags["effective_seed_status"] = str(
            inspection.get("effective_seed_status", "unknown")
        )
    return IndexedRun(
        source_type="autogluon",
        source_run_id=run_dir.name,
        source_relative_path=source_relative_path,
        source_identity=source_identity,
        terminal_status=terminal_status,
        mlflow_status="FINISHED" if terminal_status == "completed" else "FAILED",
        params=_without_none(params),
        tags=tags,
        metrics=metrics,
        artifact_relative_paths=artifact_relative_paths,
        local_source_path=run_dir,
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_hash(value: Any) -> str | None:
    return canonical_sha256(value) if value is not None else None


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _terminal_duration(status: Mapping[str, Any]) -> float | None:
    explicit = _finite_float(status.get("duration_seconds"))
    if explicit is not None:
        return explicit
    started = status.get("started_at_utc")
    finished = status.get("finished_at_utc") or status.get("ended_at_utc")
    if type(started) is not str or type(finished) is not str:
        return None
    try:
        duration = (
            datetime.fromisoformat(finished.replace("Z", "+00:00"))
            - datetime.fromisoformat(started.replace("Z", "+00:00"))
        ).total_seconds()
    except ValueError:
        return None
    return duration if duration >= 0 and math.isfinite(duration) else None


def _metric(target: dict[str, float], name: str, value: Any) -> None:
    result = _finite_float(value)
    if result is not None:
        target[name] = result


def _without_none(values: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}
