"""Validated filesystem source adapters for the optional MLflow index."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import pandas as pd
import yaml

from src.churn_ml.autogluon_inspection import inspect_run
from src.churn_ml.mlflow_artifacts import (
    ArtifactContentError,
    select_indexed_artifacts,
)
from src.churn_ml.mlflow_config import MLflowIndexConfig
from src.churn_ml.mlflow_lifecycle import (
    FailedLifecycleError,
    validate_failed_autogluon_run,
    validate_failed_research_run,
)
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
from src.churn_ml.research_v2_config import (
    ResearchV2Config,
    ResearchV2ConfigurationError,
    _validate_search_provenance,
)


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
_RESEARCH_FAILED_ARTIFACT_ALLOWLIST = (
    "_FAILED",
    "execution_status.json",
    "run_metadata.json",
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
_AUTOGLUON_FAILED_ARTIFACT_ALLOWLIST = (
    "_FAILED",
    "artifact_inventory.json",
    "execution_status.json",
    "profile_resolution.json",
    "resolved_config.yaml",
    "run_metadata.json",
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
        relative = _repository_relative_run_path(
            resolved_run, config.paths.repository_root
        )
        success = (resolved_run / "_SUCCESS").is_file()
        failed = (resolved_run / "_FAILED").is_file()
        if success and failed:
            raise SourceValidationError(
                "terminal_markers_conflict",
                f"Research run has both terminal markers: {relative}",
            )
        if not success and not failed:
            state = _status_without_marker(resolved_run)
            if state in {"completed", "failed"}:
                raise SourceValidationError(
                    "terminal_marker_missing",
                    f"Terminal research status lacks its marker: {relative}",
                )
            raise SourceSkip(
                "terminal_marker_missing",
                f"Research run is incomplete or running: {relative}",
            )

        status = _read_json_mapping(resolved_run / "execution_status.json")
        metadata = _read_json_mapping(resolved_run / "run_metadata.json")
        resolved_config = _read_optional_yaml_mapping(
            resolved_run / "resolved_config.yaml"
        )
        if failed:
            try:
                validated = validate_failed_research_run(
                    resolved_run,
                    status=status,
                    metadata=metadata,
                    resolved_config=resolved_config,
                    repository_root=config.paths.repository_root,
                )
                artifacts = select_indexed_artifacts(
                    resolved_run,
                    (
                        *_RESEARCH_FAILED_ARTIFACT_ALLOWLIST,
                        *validated.artifact_relative_paths,
                    ),
                    enabled=config.sync.log_small_artifacts,
                    size_limit=config.sync.max_artifact_size_bytes,
                )
            except (FailedLifecycleError, ArtifactContentError) as error:
                raise SourceValidationError(
                    "failed_research_lifecycle_invalid",
                    f"Failed research lifecycle is invalid: {relative}: {error}",
                ) from error
            trusted_metadata = dict(metadata)
            if validated.identity_hashes:
                trusted_metadata["hashes"] = dict(validated.identity_hashes)
            source_authentication_sha256 = validated.context_hashes.get(
                "source_authentication"
            )
            if source_authentication_sha256 is not None:
                trusted_metadata["source_authentication_sha256"] = (
                    source_authentication_sha256
                )
            if validated.resolved_config is not None:
                plan = cast(
                    Mapping[str, Any],
                    validated.resolved_config["evaluation_plan"],
                )
                trusted_metadata.update(
                    {
                        "plan_id": cast(Mapping[str, Any], plan["plan"])["id"],
                        "feature_pipeline_id": cast(
                            Mapping[str, Any],
                            validated.resolved_config["feature_pipeline"],
                        )["id"],
                        "candidate_adapter_id": cast(
                            Mapping[str, Any],
                            validated.resolved_config["candidate_adapter"],
                        )["id"],
                    }
                )
            identity = canonical_sha256(
                {
                    "status": status,
                    "metadata_hashes": metadata.get("hashes"),
                    "validated_identity_hashes": validated.identity_hashes,
                    "validated_context_hashes": validated.context_hashes,
                    "resolved_config_sha256": (
                        canonical_sha256(validated.resolved_config)
                        if validated.resolved_config is not None
                        else None
                    ),
                }
            )
            return build_research_mapping(
                run_dir=resolved_run,
                source_relative_path=relative,
                metadata=trusted_metadata,
                status=status,
                resolved_config=validated.resolved_config,
                terminal_status="failed",
                source_identity=identity,
                artifacts=artifacts,
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
        try:
            artifacts = select_indexed_artifacts(
                resolved_run,
                _RESEARCH_ARTIFACT_ALLOWLIST,
                enabled=config.sync.log_small_artifacts,
                size_limit=config.sync.max_artifact_size_bytes,
            )
        except ArtifactContentError as error:
            raise SourceValidationError(
                "artifact_content_invalid",
                f"Research metadata artifact is invalid: {relative}: {error}",
            ) from error
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
            artifacts=artifacts,
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
        relative = _repository_relative_run_path(
            resolved_run, config.paths.repository_root
        )
        success = (resolved_run / "_SUCCESS").is_file()
        failed = (resolved_run / "_FAILED").is_file()
        if success and failed:
            raise SourceValidationError(
                "terminal_markers_conflict",
                f"AutoGluon run has both terminal markers: {relative}",
            )
        if not success and not failed:
            state = _status_without_marker(resolved_run)
            if state in {"completed", "failed"}:
                raise SourceValidationError(
                    "terminal_marker_missing",
                    f"Terminal AutoGluon status lacks its marker: {relative}",
                )
            raise SourceSkip(
                "terminal_marker_missing",
                f"AutoGluon run is incomplete or running: {relative}",
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
        if success:
            report = inspect_run(resolved_run, attempt_load=False)
            if report.get("predictor_loading_attempted") is not False:
                raise SourceValidationError(
                    "predictor_loading_attempted",
                    f"AutoGluon inspection attempted predictor loading: {relative}",
                )
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
            try:
                artifacts = select_indexed_artifacts(
                    resolved_run,
                    _AUTOGLUON_ARTIFACT_ALLOWLIST,
                    enabled=config.sync.log_small_artifacts,
                    size_limit=config.sync.max_artifact_size_bytes,
                )
            except ArtifactContentError as error:
                raise SourceValidationError(
                    "artifact_content_invalid",
                    f"AutoGluon metadata artifact is invalid: {relative}: {error}",
                ) from error
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
                artifacts=artifacts,
            )

        marker = _read_json_mapping(resolved_run / "_FAILED")
        inventory = _read_json_mapping(resolved_run / "artifact_inventory.json")
        try:
            validate_failed_autogluon_run(
                resolved_run,
                status=status,
                metadata=metadata,
                marker=marker,
                resolved_config=resolved_config,
                profile_resolution=profile_resolution,
                inventory=inventory,
            )
            artifacts = select_indexed_artifacts(
                resolved_run,
                _AUTOGLUON_FAILED_ARTIFACT_ALLOWLIST,
                enabled=config.sync.log_small_artifacts,
                size_limit=config.sync.max_artifact_size_bytes,
            )
        except (FailedLifecycleError, ArtifactContentError) as error:
            raise SourceValidationError(
                "failed_autogluon_lifecycle_invalid",
                f"Failed AutoGluon lifecycle is invalid: {relative}: {error}",
            ) from error
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
            artifacts=artifacts,
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


def _repository_relative_run_path(run_dir: Path, repository_root: Path) -> str:
    try:
        root = repository_root.resolve(strict=True)
        relative = run_dir.resolve(strict=True).relative_to(root).as_posix()
    except (OSError, ValueError) as error:
        raise SourceValidationError(
            "source_path_escape",
            f"Source run must be inside repository root {repository_root}: {run_dir}",
        ) from error
    return relative


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
    actual = set(resolved)
    if actual not in (expected, expected | {"search_provenance"}):
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
    if "search_provenance" in resolved:
        try:
            _validate_search_provenance(resolved["search_provenance"])
        except ResearchV2ConfigurationError as error:
            raise SourceValidationError(
                "search_provenance_invalid",
                f"Persisted research search_provenance is invalid: {error}",
            ) from error
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


def _status_without_marker(run_dir: Path) -> Any:
    path = run_dir / "execution_status.json"
    if not path.is_file():
        return None
    return _read_json_mapping(path).get("status")


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
