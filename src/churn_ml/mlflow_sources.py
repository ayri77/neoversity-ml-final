"""Validated filesystem source adapters for the optional MLflow index."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import pandas as pd
import yaml

from src.churn_ml.autogluon_inspection import inspect_run
from src.churn_ml.mlflow_config import MLflowIndexConfig
from src.churn_ml.mlflow_mapping import (
    IndexedRun,
    SourceType,
    build_autogluon_mapping,
    build_research_mapping,
    canonical_sha256,
)
from src.churn_ml.research_v2_artifact_validation import (
    validate_research_v2_run,
)
from src.churn_ml.research_v2_config import ResearchV2Config


class SourceError(RuntimeError):
    """Base error for one independent source run."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class SourceSkip(SourceError):
    """A nonterminal source that is intentionally not indexed."""


class SourceValidationError(SourceError):
    """A terminal source that cannot be trusted or mapped."""


class SourceAdapter(Protocol):
    """Extension point for filesystem-backed index sources."""

    source_type: SourceType

    def source_root(self, config: MLflowIndexConfig) -> Path: ...

    def experiment_name(self, config: MLflowIndexConfig) -> str: ...

    def discover(
        self,
        config: MLflowIndexConfig,
        *,
        run_dir: Path | None = None,
    ) -> tuple[Path, ...]: ...

    def prepare(
        self,
        run_dir: Path,
        config: MLflowIndexConfig,
    ) -> IndexedRun: ...


@dataclass
class SourceAdapterRegistry:
    """Typed source registry with room for future comparison adapters."""

    _adapters: dict[str, SourceAdapter]

    def __init__(self) -> None:
        self._adapters = {}

    def register(self, adapter: SourceAdapter) -> None:
        if adapter.source_type in self._adapters:
            raise ValueError(
                f"Source adapter already registered: {adapter.source_type}"
            )
        self._adapters[adapter.source_type] = adapter

    def get(self, source_type: str) -> SourceAdapter:
        try:
            return self._adapters[source_type]
        except KeyError as error:
            available = ", ".join(sorted(self._adapters))
            raise ValueError(
                f"Unknown source type {source_type!r}; available: {available}"
            ) from error

    def source_types(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))


_RESEARCH_ARTIFACT_ALLOWLIST = (
    "_FAILED",
    "_SUCCESS",
    "artifact_manifest.json",
    "execution_status.json",
    "metrics/aggregate.json",
    "resolved_config.yaml",
    "run_metadata.json",
    "thresholds/threshold_summary.json",
)
_AUTOGLUON_ARTIFACT_ALLOWLIST = (
    "_FAILED",
    "_SUCCESS",
    "artifact_inventory.json",
    "execution_status.json",
    "inspection/leaderboard.csv",
    "inspection/summary.json",
    "profile_resolution.json",
    "resolved_config.yaml",
    "run_metadata.json",
    "worker_result.json",
)


class ResearchV2SourceAdapter:
    source_type: SourceType = "research_v2"

    def source_root(self, config: MLflowIndexConfig) -> Path:
        return config.paths.research_v2_root

    def experiment_name(self, config: MLflowIndexConfig) -> str:
        return config.experiments.research_v2

    def discover(
        self,
        config: MLflowIndexConfig,
        *,
        run_dir: Path | None = None,
    ) -> tuple[Path, ...]:
        return _discover_run_directories(self.source_root(config), run_dir)

    def prepare(
        self,
        run_dir: Path,
        config: MLflowIndexConfig,
    ) -> IndexedRun:
        source_root = self.source_root(config)
        resolved_run = _validated_source_directory(run_dir, source_root)
        relative = resolved_run.relative_to(source_root).as_posix()
        success = (resolved_run / "_SUCCESS").is_file()
        failed = (resolved_run / "_FAILED").is_file()
        if success and failed:
            raise SourceValidationError(
                "terminal_markers_conflict",
                f"Research run has both terminal markers: {relative}",
            )
        if not success and not failed:
            raise SourceSkip(
                "terminal_marker_missing",
                f"Research run is incomplete or running: {relative}",
            )

        status = _read_json_mapping(resolved_run / "execution_status.json")
        metadata = _read_json_mapping(resolved_run / "run_metadata.json")
        resolved_config = _read_optional_yaml_mapping(
            resolved_run / "resolved_config.yaml"
        )
        artifacts = _select_small_artifacts(
            resolved_run,
            _RESEARCH_ARTIFACT_ALLOWLIST,
            enabled=config.sync.log_small_artifacts,
            size_limit=config.sync.max_artifact_size_bytes,
        )
        if failed:
            _validate_failed_research_status(resolved_run, status, metadata)
            identity = canonical_sha256(
                {
                    "status": status,
                    "metadata_hashes": metadata.get("hashes"),
                    "resolved_config_sha256": (
                        canonical_sha256(resolved_config)
                        if resolved_config is not None
                        else None
                    ),
                }
            )
            return build_research_mapping(
                run_dir=resolved_run,
                source_relative_path=relative,
                metadata=metadata,
                status=status,
                resolved_config=resolved_config,
                terminal_status="failed",
                source_identity=identity,
                artifact_relative_paths=artifacts,
            )

        if resolved_config is None:
            raise SourceValidationError(
                "resolved_config_missing",
                f"Completed research run lacks resolved_config.yaml: {relative}",
            )
        research_config = _persisted_research_config(
            resolved_run,
            resolved_config,
            config.paths.repository_root,
        )
        hashes = _require_string_mapping(metadata.get("hashes"), "run_metadata.hashes")
        try:
            validate_research_v2_run(
                resolved_run,
                research_config,
                expected_hashes=hashes,
                require_success=True,
                verify_manifest=True,
            )
        except Exception as error:
            raise SourceValidationError(
                "semantic_validation_failed",
                f"Research run failed production semantic validation: {relative}: "
                f"{type(error).__name__}: {error}",
            ) from error
        manifest = _read_json_mapping(resolved_run / "artifact_manifest.json")
        manifest_hash = manifest.get("manifest_sha256")
        if not _is_sha256(manifest_hash):
            raise SourceValidationError(
                "manifest_identity_invalid",
                f"Research manifest identity is invalid: {relative}",
            )
        aggregate = _read_json_mapping(resolved_run / "metrics" / "aggregate.json")
        threshold_summary = _read_json_mapping(
            resolved_run / "thresholds" / "threshold_summary.json"
        )
        threshold_frame = pd.read_csv(
            resolved_run / "thresholds" / "selected_thresholds.csv"
        )
        threshold_std = float(
            threshold_frame["selected_threshold"].to_numpy(dtype=float).std(ddof=0)
        )
        return build_research_mapping(
            run_dir=resolved_run,
            source_relative_path=relative,
            metadata=metadata,
            status=status,
            resolved_config=resolved_config,
            terminal_status="completed",
            source_identity=cast(str, manifest_hash),
            aggregate=aggregate,
            threshold_summary=threshold_summary,
            threshold_standard_deviation=threshold_std,
            artifact_relative_paths=artifacts,
        )


class AutoGluonSourceAdapter:
    source_type: SourceType = "autogluon"

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
        return _discover_run_directories(self.source_root(config), run_dir)

    def prepare(
        self,
        run_dir: Path,
        config: MLflowIndexConfig,
    ) -> IndexedRun:
        source_root = self.source_root(config)
        resolved_run = _validated_source_directory(run_dir, source_root)
        relative = resolved_run.relative_to(source_root).as_posix()
        success = (resolved_run / "_SUCCESS").is_file()
        failed = (resolved_run / "_FAILED").is_file()
        if success and failed:
            raise SourceValidationError(
                "terminal_markers_conflict",
                f"AutoGluon run has both terminal markers: {relative}",
            )
        if not success and not failed:
            raise SourceSkip(
                "terminal_marker_missing",
                f"AutoGluon run is incomplete or running: {relative}",
            )

        report = inspect_run(resolved_run, attempt_load=False)
        if report.get("predictor_loading_attempted") is not False:
            raise SourceValidationError(
                "predictor_loading_attempted",
                f"AutoGluon inspection attempted predictor loading: {relative}",
            )
        metadata = _read_json_mapping(resolved_run / "run_metadata.json")
        status = _read_json_mapping(resolved_run / "execution_status.json")
        resolved_config = _read_optional_yaml_mapping(
            resolved_run / "resolved_config.yaml"
        )
        profile_resolution = _read_optional_json_mapping(
            resolved_run / "profile_resolution.json"
        )
        worker_result = _read_optional_json_mapping(resolved_run / "worker_result.json")
        inspection_summary = _read_optional_json_mapping(
            resolved_run / "inspection" / "summary.json"
        )
        artifacts = _select_small_artifacts(
            resolved_run,
            _AUTOGLUON_ARTIFACT_ALLOWLIST,
            enabled=config.sync.log_small_artifacts,
            size_limit=config.sync.max_artifact_size_bytes,
        )
        if success:
            if report.get("classification") != "complete":
                raise SourceValidationError(
                    "completed_run_corrupt",
                    f"AutoGluon completed run is not valid: {relative}: "
                    f"{report.get('reason_codes')}",
                )
            if worker_result is None:
                raise SourceValidationError(
                    "worker_result_missing",
                    f"AutoGluon completed run lacks worker_result.json: {relative}",
                )
            inventory = _read_json_mapping(resolved_run / "artifact_inventory.json")
            identity = canonical_sha256(
                {
                    "config_identity_sha256": metadata.get("config_identity_sha256"),
                    "profile_sha256": metadata.get("profile_sha256"),
                    "artifact_inventory": inventory,
                }
            )
            return build_autogluon_mapping(
                run_dir=resolved_run,
                source_relative_path=relative,
                metadata=metadata,
                status=status,
                resolved_config=resolved_config,
                profile_resolution=profile_resolution,
                worker_result=worker_result,
                inspection_summary=inspection_summary,
                terminal_status="completed",
                source_identity=identity,
                predictor_classification="complete",
                artifact_relative_paths=artifacts,
            )

        _validate_failed_autogluon_status(resolved_run, status, metadata)
        corrupt_reasons = [
            str(reason)
            for reason in report.get("reason_codes", [])
            if _failed_inspection_reason_is_corrupt(str(reason))
        ]
        if corrupt_reasons:
            raise SourceValidationError(
                "failed_run_corrupt",
                f"AutoGluon failed run metadata is corrupt: {relative}: "
                f"{corrupt_reasons}",
            )
        marker = _read_json_mapping(resolved_run / "_FAILED")
        identity = canonical_sha256(
            {
                "marker": marker,
                "status": status,
                "config_identity_sha256": metadata.get("config_identity_sha256"),
                "profile_sha256": metadata.get("profile_sha256"),
            }
        )
        return build_autogluon_mapping(
            run_dir=resolved_run,
            source_relative_path=relative,
            metadata=metadata,
            status=status,
            resolved_config=resolved_config,
            profile_resolution=profile_resolution,
            worker_result=worker_result,
            inspection_summary=inspection_summary,
            terminal_status="failed",
            source_identity=identity,
            predictor_classification="failed_terminal_not_loaded",
            artifact_relative_paths=artifacts,
        )


def default_source_registry() -> SourceAdapterRegistry:
    registry = SourceAdapterRegistry()
    registry.register(ResearchV2SourceAdapter())
    registry.register(AutoGluonSourceAdapter())
    return registry


def _discover_run_directories(
    source_root: Path,
    run_dir: Path | None,
) -> tuple[Path, ...]:
    if run_dir is not None:
        if run_dir.is_symlink():
            raise SourceValidationError(
                "source_symlink_rejected",
                f"Source run directory must not be a symlink: {run_dir}",
            )
        return (_validated_source_directory(run_dir, source_root),)
    if not source_root.exists():
        return ()
    if not source_root.is_dir():
        raise SourceValidationError(
            "source_root_not_directory",
            f"Configured source root is not a directory: {source_root}",
        )
    candidates: set[Path] = set()
    for name in ("execution_status.json", "_SUCCESS", "_FAILED"):
        for artifact in source_root.rglob(name):
            if artifact.is_file():
                candidates.add(artifact.parent.resolve(strict=True))
    return tuple(sorted(candidates, key=lambda item: item.as_posix()))


def _validated_source_directory(run_dir: Path, source_root: Path) -> Path:
    try:
        root = source_root.resolve(strict=False)
        resolved = run_dir.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise SourceValidationError(
            "source_path_escape",
            f"Source run must be a directory inside {source_root}: {run_dir}",
        ) from error
    if not resolved.is_dir():
        raise SourceValidationError(
            "source_not_directory", f"Source run is not a directory: {run_dir}"
        )
    return resolved


def _persisted_research_config(
    run_dir: Path,
    resolved: Mapping[str, Any],
    repository_root: Path,
) -> ResearchV2Config:
    expected = {
        "schema_version",
        "experiment",
        "dataset",
        "evaluation_plan_path",
        "feature_pipeline",
        "candidate_adapter",
        "artifacts",
        "persistence",
        "tracking",
        "evaluation_plan",
    }
    if set(resolved) != expected:
        raise SourceValidationError(
            "resolved_config_schema_invalid",
            "Persisted research resolved_config.yaml has missing or extra keys",
        )
    if (
        type(resolved.get("schema_version")) is not int
        or resolved["schema_version"] != 2
    ):
        raise SourceValidationError(
            "resolved_config_schema_version_invalid",
            "Persisted research schema_version must be exact integer 2",
        )
    plan = resolved.get("evaluation_plan")
    if type(plan) is not dict:
        raise SourceValidationError(
            "evaluation_plan_invalid",
            "Persisted research evaluation_plan must be a mapping",
        )
    payload = {
        key: value for key, value in resolved.items() if key != "evaluation_plan"
    }
    return ResearchV2Config(
        payload=json.loads(json.dumps(payload, allow_nan=False)),
        plan_payload=json.loads(json.dumps(plan, allow_nan=False)),
        source_path=run_dir / "resolved_config.yaml",
        plan_path=repository_root / str(payload["evaluation_plan_path"]),
        project_root=repository_root,
    )


def _validate_failed_research_status(
    run_dir: Path,
    status: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    expected = {
        "schema_version",
        "run_id",
        "status",
        "started_at_utc",
        "finished_at_utc",
        "completed_outer_folds",
        "expected_outer_folds",
        "failure",
    }
    if set(status) != expected:
        raise SourceValidationError(
            "failed_status_schema_invalid",
            "Failed research execution_status.json has missing or extra keys",
        )
    if (
        type(status["schema_version"]) is not int
        or status["schema_version"] != 2
        or status["status"] != "failed"
        or type(status["run_id"]) is not str
        or status["run_id"] != run_dir.name
        or type(status["started_at_utc"]) is not str
        or type(status["finished_at_utc"]) is not str
        or type(status["completed_outer_folds"]) is not int
        or type(status["expected_outer_folds"]) is not int
    ):
        raise SourceValidationError(
            "failed_status_value_invalid",
            "Failed research execution status contains invalid exact types or values",
        )
    failure = status["failure"]
    if (
        type(failure) is not dict
        or set(failure) != {"type", "message"}
        or type(failure["type"]) is not str
        or type(failure["message"]) is not str
    ):
        raise SourceValidationError(
            "failed_status_failure_invalid",
            "Failed research failure payload is invalid",
        )
    if metadata.get("status") != "failed" or metadata.get("run_id") != run_dir.name:
        raise SourceValidationError(
            "failed_metadata_mismatch",
            "Failed research metadata disagrees with the terminal run",
        )


def _validate_failed_autogluon_status(
    run_dir: Path,
    status: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    required_types: dict[str, type | tuple[type, ...]] = {
        "status": str,
        "run_id": str,
        "config_identity_sha256": str,
        "profile_sha256": str,
        "requested_seed": int,
        "started_at_utc": str,
        "ended_at_utc": str,
        "duration_seconds": (int, float),
        "child_process_exit_code": (int, type(None)),
        "last_completed_observable_stage": str,
        "failure_codes": list,
        "predictor_loading_attempted": bool,
        "predictor_loading_succeeded": bool,
    }
    for key, accepted in required_types.items():
        value = status.get(key)
        types = accepted if isinstance(accepted, tuple) else (accepted,)
        if not any(type(value) is item for item in types):
            raise SourceValidationError(
                "failed_status_schema_invalid",
                f"Failed AutoGluon status field has invalid type: {key}",
            )
    if (
        status["status"] != "failed"
        or status["run_id"] != run_dir.name
        or metadata.get("run_id") != run_dir.name
        or metadata.get("config_identity_sha256")
        != status.get("config_identity_sha256")
        or metadata.get("profile_sha256") != status.get("profile_sha256")
        or metadata.get("requested_seed") != status.get("requested_seed")
        or status["predictor_loading_attempted"] is not False
        or status["predictor_loading_succeeded"] is not False
        or not math.isfinite(float(status["duration_seconds"]))
        or float(status["duration_seconds"]) < 0
        or any(type(item) is not str for item in status["failure_codes"])
    ):
        raise SourceValidationError(
            "failed_status_value_invalid",
            "Failed AutoGluon status and metadata are inconsistent",
        )
    marker = _read_json_mapping(run_dir / "_FAILED")
    if (
        marker.get("status") != "failed"
        or marker.get("run_id") != run_dir.name
        or marker.get("failure_codes") != status.get("failure_codes")
    ):
        raise SourceValidationError(
            "failed_terminal_marker_invalid",
            "Failed AutoGluon marker disagrees with execution status",
        )


def _failed_inspection_reason_is_corrupt(reason: str) -> bool:
    return (
        "mismatch" in reason
        or "conflict" in reason
        or "malformed" in reason
        or reason.endswith("invalid_json")
    )


def _select_small_artifacts(
    run_dir: Path,
    allowlist: Iterable[str],
    *,
    enabled: bool,
    size_limit: int,
) -> tuple[str, ...]:
    if not enabled:
        return ()
    selected: list[str] = []
    resolved_run = run_dir.resolve(strict=True)
    for relative in allowlist:
        candidate = run_dir / Path(*relative.split("/"))
        if not candidate.exists():
            continue
        if candidate.is_symlink() or not candidate.is_file():
            raise SourceValidationError(
                "artifact_symlink_or_type_rejected",
                f"Allowlisted metadata artifact is not a regular file: {relative}",
            )
        try:
            candidate.resolve(strict=True).relative_to(resolved_run)
        except ValueError as error:
            raise SourceValidationError(
                "artifact_path_escape",
                f"Allowlisted metadata artifact escapes source run: {relative}",
            ) from error
        if candidate.stat().st_size <= size_limit:
            selected.append(relative)
    return tuple(selected)


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: _raise_nonfinite(path, token),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SourceValidationError(
            "metadata_json_invalid", f"Cannot read JSON metadata {path}: {error}"
        ) from error
    if type(value) is not dict:
        raise SourceValidationError(
            "metadata_schema_invalid", f"JSON metadata must be a mapping: {path}"
        )
    return value


def _read_optional_json_mapping(path: Path) -> dict[str, Any] | None:
    return _read_json_mapping(path) if path.is_file() else None


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise SourceValidationError(
            "metadata_yaml_invalid", f"Cannot read YAML metadata {path}: {error}"
        ) from error
    if type(value) is not dict:
        raise SourceValidationError(
            "metadata_schema_invalid", f"YAML metadata must be a mapping: {path}"
        )
    return value


def _read_optional_yaml_mapping(path: Path) -> dict[str, Any] | None:
    return _read_yaml_mapping(path) if path.is_file() else None


def _require_string_mapping(value: Any, label: str) -> dict[str, str]:
    if (
        type(value) is not dict
        or not value
        or any(
            type(key) is not str or type(item) is not str for key, item in value.items()
        )
    ):
        raise SourceValidationError(
            "identity_mapping_invalid", f"{label} must be a non-empty string mapping"
        )
    return dict(value)


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _raise_nonfinite(path: Path, token: str) -> Any:
    raise SourceValidationError(
        "metadata_nonfinite", f"Non-finite JSON value in {path}: {token}"
    )
