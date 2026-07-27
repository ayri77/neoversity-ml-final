"""Exact failed-run lifecycle validators owned by the MLflow index layer."""

from __future__ import annotations

import hashlib
import json
import math
import stat
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from src.churn_ml.autogluon_artifacts import (
    build_inventory,
    discover_model_directories,
)
from src.churn_ml.autogluon_config import config_identity_sha256
from src.churn_ml.autogluon_profiles import (
    get_profile,
    profile_sha256,
    profile_summary,
)
from src.churn_ml.autogluon_supervisor import unsigned_windows_exit_code
from src.churn_ml.experiment_v2 import get_candidate_adapter, get_feature_pipeline
from src.churn_ml.experiment_v2_contract import first_exact_difference
from src.churn_ml.mlflow_mapping import canonical_sha256
from src.churn_ml.research_config import (
    _validate_plan_invariants,
    _validate_plan_structure,
)
from src.churn_ml.research_v2_artifact_validation import (
    validate_portable_payload_paths,
)
from src.churn_ml.research_v2_config import (
    ROOT_KEYS,
    SAFE_SLUG,
    SECTION_KEYS,
    _load_mapping as load_research_v2_plan_mapping,
    _repo_path as resolve_research_v2_repository_path,
    load_research_v2_config,
)
from src.churn_ml.research_v2_identity import (
    ADAPTER_IMPLEMENTATION_SOURCES,
    PIPELINE_IMPLEMENTATION_SOURCES,
)
from src.churn_ml.research_v2_provenance import (
    SOURCE_PATHS,
    file_identity as build_repository_file_identity,
)


class FailedLifecycleError(RuntimeError):
    """Raised when a failed source does not prove an exact trusted lifecycle."""


@dataclass(frozen=True)
class ValidatedFailedResearchArtifacts:
    """Optional artifacts trusted after exact failed-run validation."""

    resolved_config: dict[str, Any] | None
    identity_hashes: dict[str, str]
    context_hashes: dict[str, str]
    artifact_relative_paths: tuple[str, ...]


_RESEARCH_STATUS_KEYS = {
    "schema_version",
    "run_id",
    "status",
    "started_at_utc",
    "finished_at_utc",
    "completed_outer_folds",
    "expected_outer_folds",
    "failure",
}
_RESEARCH_FULL_METADATA_KEYS = {
    "schema_version",
    "experiment_id",
    "plan_id",
    "feature_pipeline_id",
    "candidate_adapter_id",
    "hashes",
    "status",
    "started_at_utc",
    "environment",
    "runtime",
    "git",
    "competition_assets_accessed",
    "tracking_enabled",
    "run_id",
    "finished_at_utc",
    "failure",
}
_RESEARCH_EARLY_METADATA_KEYS = {"status", "finished_at_utc", "failure"}
_RESEARCH_IDENTITY_NAMES = {
    "evaluation_plan": "plan",
    "feature_pipeline": "feature_pipeline",
    "candidate_adapter": "candidate_adapter",
    "candidate": "candidate",
    "source": "source",
    "loaded_modules": "loaded_modules",
}
_RESEARCH_HASH_KEYS = set(_RESEARCH_IDENTITY_NAMES.values())
_AUTO_STATUS_KEYS = {
    "status",
    "run_id",
    "config_identity_sha256",
    "profile_sha256",
    "requested_seed",
    "started_at_utc",
    "child_pid",
    "launched_process_pid",
    "child_process_exit_code",
    "windows_exit_code_unsigned",
    "last_completed_observable_stage",
    "predictor_loading_attempted",
    "predictor_loading_succeeded",
    "ended_at_utc",
    "duration_seconds",
    "failure_reason",
    "failure_codes",
    "missing_expected_artifacts",
    "worker_completion_valid",
    "worker_completion_reason_codes",
    "worker_completion_details",
    "log_pump_failures",
    "discovered_model_directories",
    "stderr_tail_bounded",
    "full_logs_authoritative",
}
_AUTO_METADATA_KEYS = {
    "schema_version",
    "run_id",
    "config_identity_sha256",
    "profile_id",
    "profile_sha256",
    "profile",
    "dataset_version",
    "requested_seed",
    "resources",
    "started_at_utc",
    "child_pid",
    "launched_process_pid",
    "local_operational_nonportable",
    "ended_at_utc",
    "duration_seconds",
    "child_process_exit_code",
    "windows_exit_code_unsigned",
    "worker_completion_valid",
    "worker_completion_reason_codes",
    "log_pump_failures",
}
_AUTO_MARKER_KEYS = {
    "status",
    "run_id",
    "ended_at_utc",
    "child_process_exit_code",
    "windows_exit_code_unsigned",
    "failure_reason",
    "failure_codes",
}
_AUTO_RESOURCE_KEYS = {
    "time_limit_seconds",
    "num_cpus",
    "gpu_budget",
    "top_level_num_gpus_passed_to_fit",
    "fit_strategy",
    "fold_fitting_strategy",
}
_PUMP_FAILURE_KEYS = {"stream", "operation", "error_type", "error"}
_AUTO_CONFIG_KEYS = {
    "schema_version",
    "profile_id",
    "seed",
    "dataset",
    "predictor",
    "resources",
    "fit",
    "artifacts",
}
_AUTO_DATASET_KEYS = {
    "version",
    "directory",
    "train_features_file",
    "train_target_file",
    "label",
}
_AUTO_PREDICTOR_KEYS = {
    "problem_type",
    "eval_metric",
    "positive_class",
    "verbosity",
}
_AUTO_CONFIG_RESOURCE_KEYS = {
    "time_limit_seconds",
    "num_cpus",
    "num_gpus",
    "fit_strategy",
    "fold_fitting_strategy",
}
_AUTO_FIT_KEYS = {"presets", "calibrate_decision_threshold"}
_AUTO_EXPECTED_SUCCESS_ARTIFACTS = (
    "worker_result.json",
    "dataset_manifest.json",
    "profile_resolution.json",
    "predictor/predictor.pkl",
    "predictor/learner.pkl",
    "predictor/version.txt",
    "inspection/leaderboard.csv",
    "inspection/summary.json",
    "logs/worker.stdout.log",
    "logs/worker.stderr.log",
)
_AUTO_PROFILE_RESOLUTION_KEYS = {
    "schema_version",
    "created_at_utc",
    "profile_id",
    "profile_sha256",
    "profile_identity",
    "dataset_version",
    "requested_seed",
    "autogluon_version",
    "portfolio",
    "resolved_families",
    "family_resources",
    "resolved_hyperparameters",
    "fit_strategy",
    "fold_fitting_strategy",
    "top_level_num_gpus_passed_to_fit",
}


def validate_failed_research_run(
    run_dir: Path,
    *,
    status: Mapping[str, Any],
    metadata: Mapping[str, Any],
    resolved_config: Mapping[str, Any] | None,
    repository_root: Path,
) -> ValidatedFailedResearchArtifacts:
    """Validate a failed v2 lifecycle and return only trusted optional artifacts."""
    _validate_terminal_files(run_dir, failed_marker_must_be_json=False)
    _validate_tree_safety(run_dir, research=True)
    _exact_keys(status, _RESEARCH_STATUS_KEYS, "research execution status")
    if (
        _exact_int(status["schema_version"]) != 2
        or _nonempty_string(status["run_id"], "research status run_id") != run_dir.name
        or status["status"] != "failed"
    ):
        raise FailedLifecycleError("Research status identity or terminal state differs")
    started = _utc(status["started_at_utc"], "research status started_at_utc")
    finished = _utc(status["finished_at_utc"], "research status finished_at_utc")
    if finished < started:
        raise FailedLifecycleError("Research failure finished before it started")
    completed = _exact_int(
        status["completed_outer_folds"], "research completed_outer_folds"
    )
    expected = _exact_int(
        status["expected_outer_folds"], "research expected_outer_folds"
    )
    if expected <= 0 or completed < 0 or completed > expected:
        raise FailedLifecycleError("Research fold counters are out of bounds")
    failure = _failure_mapping(status["failure"], "research status failure")

    metadata_keys = set(metadata)
    early = metadata_keys == _RESEARCH_EARLY_METADATA_KEYS
    full = metadata_keys == _RESEARCH_FULL_METADATA_KEYS
    if not early and not full:
        raise FailedLifecycleError(
            "Research failed metadata is neither the exact full nor early schema"
        )
    if metadata.get("status") != "failed":
        raise FailedLifecycleError("Research metadata status is not failed")
    if (
        _utc(metadata["finished_at_utc"], "research metadata finished_at_utc")
        != finished
        or _failure_mapping(metadata["failure"], "research metadata failure") != failure
    ):
        raise FailedLifecycleError("Research metadata disagrees with terminal status")
    hashes: dict[str, str] | None = None
    if early:
        if completed != 0:
            raise FailedLifecycleError("Research early failure claims fold progress")
    else:
        if (
            _exact_int(metadata["schema_version"]) != 2
            or _nonempty_string(metadata["run_id"], "research metadata run_id")
            != run_dir.name
            or _utc(metadata["started_at_utc"], "research metadata started_at_utc")
            != started
            or metadata["competition_assets_accessed"] is not False
            or metadata["tracking_enabled"] is not False
        ):
            raise FailedLifecycleError("Research full metadata identity differs")
        hashes = _sha_mapping(metadata["hashes"], "research metadata hashes")
        if set(hashes) != _RESEARCH_HASH_KEYS:
            raise FailedLifecycleError("Research identity hash keys differ")
        _validate_research_metadata_context(metadata)

    if full and resolved_config is None:
        raise FailedLifecycleError("Research full failure lacks resolved config")
    validated_config = (
        _validate_research_resolved_config(resolved_config, repository_root)
        if resolved_config is not None
        else None
    )
    if full:
        assert validated_config is not None
        plan = _mapping(
            validated_config.get("evaluation_plan"), "research evaluation plan"
        )
        outer = _mapping(plan.get("outer_evaluation"), "outer evaluation")
        configured_expected = len(_list(outer.get("repeat_seeds"))) * _exact_int(
            outer.get("n_splits"), "outer n_splits"
        )
        expected_values = {
            "experiment_id": _mapping(
                validated_config.get("experiment"), "research experiment"
            ).get("id"),
            "plan_id": _mapping(plan.get("plan"), "research plan").get("id"),
            "feature_pipeline_id": _mapping(
                validated_config.get("feature_pipeline"), "research feature pipeline"
            ).get("id"),
            "candidate_adapter_id": _mapping(
                validated_config.get("candidate_adapter"), "research candidate adapter"
            ).get("id"),
        }
        if any(metadata.get(key) != value for key, value in expected_values.items()):
            raise FailedLifecycleError(
                "Research plan/pipeline/adapter metadata disagrees with config"
            )
        if configured_expected != expected:
            raise FailedLifecycleError(
                "Research expected fold count disagrees with plan"
            )

    identity_hashes, identity_paths = _validate_research_identity_artifacts(
        run_dir,
        repository_root=repository_root,
        resolved_config=validated_config,
        metadata_hashes=hashes,
        require_all=full and completed > 0,
    )
    context_hashes, context_paths = _validate_optional_research_context_artifacts(
        run_dir,
        metadata=metadata,
        status_started=started,
        status_finished=finished,
    )
    if early:
        _reject_unsupported_early_research_artifacts(run_dir)
    artifact_paths = [
        *(["resolved_config.yaml"] if validated_config is not None else []),
        *identity_paths,
        *context_paths,
    ]
    return ValidatedFailedResearchArtifacts(
        resolved_config=validated_config,
        identity_hashes=identity_hashes,
        context_hashes=context_hashes,
        artifact_relative_paths=tuple(sorted(artifact_paths)),
    )


def _validate_research_resolved_config(
    config: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    """Apply the production v2 loader contract to a persisted resolved payload."""
    _exact_keys(config, ROOT_KEYS | {"evaluation_plan"}, "research resolved config")
    payload = {key: value for key, value in config.items() if key != "evaluation_plan"}
    for name, keys in SECTION_KEYS.items():
        section = _mapping(payload.get(name), f"research config {name}")
        _exact_keys(section, keys, f"research config {name}")
    if _exact_int(payload["schema_version"], "research config schema") != 2:
        raise FailedLifecycleError("Research resolved config schema differs")
    for value, label in (
        (_mapping(payload["experiment"], "experiment")["id"], "experiment.id"),
        (_mapping(payload["dataset"], "dataset")["version"], "dataset.version"),
        (
            _mapping(payload["feature_pipeline"], "feature pipeline")["id"],
            "feature_pipeline.id",
        ),
        (
            _mapping(payload["candidate_adapter"], "candidate adapter")["id"],
            "candidate_adapter.id",
        ),
    ):
        if SAFE_SLUG.fullmatch(_nonempty_string(value, label)) is None:
            raise FailedLifecycleError(f"Research config {label} is not a safe slug")
    root = repository_root.resolve()
    try:
        production_plan_path = resolve_research_v2_repository_path(
            _nonempty_string(
                payload["evaluation_plan_path"],
                "research evaluation_plan_path",
            ),
            root,
            "evaluation_plan_path",
        )
    except (TypeError, ValueError) as error:
        raise FailedLifecycleError(
            f"Research evaluation plan path differs: {error}"
        ) from error
    plan_path = _repository_path(
        payload["evaluation_plan_path"], root, "research evaluation_plan_path"
    )
    if plan_path != production_plan_path:
        raise FailedLifecycleError("Research evaluation plan path resolution differs")
    _validate_repository_regular_file(
        plan_path,
        repository_root=root,
        label="research evaluation plan",
    )
    _repository_path(
        _mapping(payload["artifacts"], "research artifacts")["root"],
        root,
        "research artifacts.root",
    )
    plan = _mapping(config["evaluation_plan"], "research evaluation plan")
    try:
        repository_plan = load_research_v2_plan_mapping(plan_path, "evaluation plan")
        _validate_plan_structure(repository_plan)
        _validate_plan_invariants(repository_plan)
        _validate_plan_structure(dict(plan))
        _validate_plan_invariants(dict(plan))
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise FailedLifecycleError(
            f"Research evaluation plan contract differs: {error}"
        ) from error
    difference = first_exact_difference(
        plan,
        repository_plan,
        "resolved_config.evaluation_plan",
    )
    if difference is not None:
        raise FailedLifecycleError(
            f"Persisted evaluation plan differs from repository plan: {difference}"
        )
    dataset = _mapping(payload["dataset"], "research dataset")
    plan_dataset = _mapping(plan["dataset"], "research plan dataset")
    if plan_dataset["version"] != dataset["version"]:
        raise FailedLifecycleError("Research plan/config dataset version differs")
    _repository_path(
        plan_dataset["processed_dir"], root, "research dataset.processed_dir"
    )
    files = _mapping(plan_dataset["files"], "research dataset files")
    for label, item in files.items():
        record = _mapping(item, f"research dataset file {label}")
        name = _nonempty_string(record.get("name"), f"dataset.files.{label}.name")
        if (
            Path(name).name != name
            or PurePosixPath(name).name != name
            or PureWindowsPath(name).name != name
        ):
            raise FailedLifecycleError(
                f"Research dataset.files.{label}.name is not a basename"
            )
    pipeline = _mapping(payload["feature_pipeline"], "research feature pipeline")
    adapter = _mapping(payload["candidate_adapter"], "research candidate adapter")
    try:
        get_feature_pipeline(str(pipeline["id"])).validate_contract(
            _mapping(pipeline["contract"], "research feature pipeline contract")
        )
        get_candidate_adapter(str(adapter["id"])).validate_contract(
            _mapping(adapter["contract"], "research candidate adapter contract")
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FailedLifecycleError(
            f"Research pipeline/adapter contract differs: {error}"
        ) from error
    persistence = _mapping(payload["persistence"], "research persistence")
    required_true = {
        "resolved_config",
        "identities",
        "assignments",
        "predictions",
        "metrics",
    }
    for key, value in persistence.items():
        if type(value) is not bool:
            raise FailedLifecycleError(f"Research persistence.{key} must be boolean")
        if key in required_true and value is not True:
            raise FailedLifecycleError(f"Research persistence.{key} must be true")
    if persistence["models"] is not False:
        raise FailedLifecycleError("Research model persistence must be disabled")
    if _mapping(payload["tracking"], "research tracking")["enabled"] is not False:
        raise FailedLifecycleError("Research tracking must remain disabled")
    try:
        validate_portable_payload_paths(
            config,
            label="resolved_config",
            project_root=root,
        )
    except Exception as error:
        raise FailedLifecycleError(
            f"Research resolved config contains a nonportable path: {error}"
        ) from error
    return deepcopy(dict(config))


def _validate_research_identity_artifacts(
    run_dir: Path,
    *,
    repository_root: Path,
    resolved_config: Mapping[str, Any] | None,
    metadata_hashes: Mapping[str, str] | None,
    require_all: bool,
) -> tuple[dict[str, str], list[str]]:
    identities = run_dir / "identities"
    if require_all and not identities.is_dir():
        raise FailedLifecycleError("Research progressed failure lacks identities")
    if identities.exists() and not identities.is_dir():
        raise FailedLifecycleError("Research identities path is not a directory")
    allowed = {f"{name}.json" for name in _RESEARCH_IDENTITY_NAMES}
    if identities.is_dir():
        unexpected = [
            path.relative_to(identities).as_posix()
            for path in identities.rglob("*")
            if path.is_file() and path.relative_to(identities).as_posix() not in allowed
        ]
        if unexpected:
            raise FailedLifecycleError(
                f"Research identities contain unknown artifacts: {sorted(unexpected)}"
            )
    hashes: dict[str, str] = {}
    paths: list[str] = []
    canonicals: dict[str, Mapping[str, Any]] = {}
    for filename, hash_key in _RESEARCH_IDENTITY_NAMES.items():
        path = identities / f"{filename}.json"
        if require_all and not path.is_file():
            raise FailedLifecycleError(
                f"Research identity artifact is missing: {filename}"
            )
        if not path.exists():
            continue
        if not path.is_file():
            raise FailedLifecycleError(
                f"Research identity artifact is not a file: {filename}"
            )
        payload = _strict_json(path, f"research identity {filename}")
        _exact_keys(payload, {"sha256", "canonical"}, f"identity {filename}")
        digest = _sha(payload["sha256"], f"research identity {filename} hash")
        canonical = _mapping(
            payload["canonical"], f"research identity {filename} canonical"
        )
        if canonical_sha256(canonical) != digest:
            raise FailedLifecycleError(f"Research identity hash differs: {filename}")
        try:
            validate_portable_payload_paths(
                canonical,
                label=f"identities.{filename}.canonical",
                project_root=repository_root,
            )
        except Exception as error:
            raise FailedLifecycleError(
                f"Research identity path differs: {filename}: {error}"
            ) from error
        if metadata_hashes is not None and digest != metadata_hashes[hash_key]:
            raise FailedLifecycleError(
                f"Research identity metadata hash differs: {filename}"
            )
        _validate_research_identity_semantics(
            filename,
            canonical,
            repository_root=repository_root,
            resolved_config=resolved_config,
        )
        hashes[hash_key] = digest
        canonicals[filename] = canonical
        paths.append(f"identities/{filename}.json")
    candidate = canonicals.get("candidate")
    if candidate is not None:
        for component, hash_key in (
            ("feature_pipeline", "feature_pipeline"),
            ("candidate_adapter", "candidate_adapter"),
        ):
            component_identity = _mapping(candidate[component], component)
            if hash_key in hashes and component_identity["sha256"] != hashes[hash_key]:
                raise FailedLifecycleError(
                    f"Research candidate/{component} identity differs"
                )
    return hashes, paths


def _validate_research_identity_semantics(
    name: str,
    canonical: Mapping[str, Any],
    *,
    repository_root: Path,
    resolved_config: Mapping[str, Any] | None,
) -> None:
    config = resolved_config or {}
    plan = _mapping(config.get("evaluation_plan"), "research evaluation plan")
    pipeline = _mapping(config.get("feature_pipeline"), "research feature pipeline")
    adapter = _mapping(config.get("candidate_adapter"), "research candidate adapter")
    dataset = _mapping(config.get("dataset"), "research dataset")
    if name == "evaluation_plan":
        _exact_keys(
            canonical,
            {
                "schema_version",
                "plan_id",
                "dataset",
                "outer_evaluation",
                "threshold_selection",
                "threshold_policy",
                "metrics",
                "aggregation",
                "assignment_fingerprints",
            },
            "evaluation plan identity",
        )
        if _exact_int(canonical["schema_version"]) != 1:
            raise FailedLifecycleError("Evaluation plan identity schema differs")
        if config and (
            canonical["plan_id"] != _mapping(plan.get("plan"), "research plan")["id"]
            or _mapping(canonical["dataset"], "plan identity dataset")[
                "dataset_version"
            ]
            != dataset["version"]
            or canonical["outer_evaluation"] != plan["outer_evaluation"]
            or canonical["threshold_selection"] != plan["threshold_selection"]
            or canonical["threshold_policy"] != plan["threshold_policy"]
            or canonical["metrics"] != plan["metrics"]
            or canonical["aggregation"] != plan["aggregation"]
        ):
            raise FailedLifecycleError("Evaluation plan identity disagrees with config")
    elif name in {"feature_pipeline", "candidate_adapter"}:
        expected = (
            {
                "schema_version",
                "id",
                "contract",
                "resolved_feature_schema",
                "implementation_sources",
            }
            if name == "feature_pipeline"
            else {
                "schema_version",
                "id",
                "contract",
                "probability_semantics",
                "runtime_dependencies",
                "implementation_sources",
            }
        )
        _exact_keys(canonical, expected, f"{name} identity")
        if _exact_int(canonical["schema_version"]) != 2:
            raise FailedLifecycleError(f"Research {name} identity schema differs")
        configured = pipeline if name == "feature_pipeline" else adapter
        if config and (
            canonical["id"] != configured["id"]
            or canonical["contract"] != configured["contract"]
        ):
            raise FailedLifecycleError(
                f"Research {name} identity disagrees with config"
            )
        _validate_source_file_manifest(
            _mapping(canonical["implementation_sources"], "implementation sources"),
            repository_root,
            f"{name} implementation sources",
            expected_paths=(
                PIPELINE_IMPLEMENTATION_SOURCES
                if name == "feature_pipeline"
                else ADAPTER_IMPLEMENTATION_SOURCES
            ),
        )
    elif name == "candidate":
        _exact_keys(
            canonical,
            {
                "schema_version",
                "dataset_version",
                "feature_pipeline",
                "candidate_adapter",
                "probability_semantics",
                "evaluation_boundary",
            },
            "candidate identity",
        )
        if _exact_int(canonical["schema_version"]) != 2:
            raise FailedLifecycleError("Research candidate identity schema differs")
        candidate_pipeline = _mapping(
            canonical["feature_pipeline"], "candidate pipeline"
        )
        candidate_adapter = _mapping(
            canonical["candidate_adapter"], "candidate adapter"
        )
        _exact_keys(candidate_pipeline, {"id", "sha256"}, "candidate pipeline")
        _exact_keys(candidate_adapter, {"id", "sha256"}, "candidate adapter")
        _sha(candidate_pipeline["sha256"], "candidate pipeline hash")
        _sha(candidate_adapter["sha256"], "candidate adapter hash")
        if config and (
            canonical["dataset_version"] != dataset["version"]
            or candidate_pipeline["id"] != pipeline["id"]
            or candidate_adapter["id"] != adapter["id"]
        ):
            raise FailedLifecycleError(
                "Research candidate identity disagrees with config"
            )
    elif name == "source":
        _validate_source_file_manifest(
            canonical,
            repository_root,
            "source identity",
            expected_paths=None,
            resolved_config=resolved_config,
        )
    elif name == "loaded_modules":
        _exact_keys(
            canonical,
            {"schema_version", "fresh_process_required", "modules"},
            "loaded modules identity",
        )
        if (
            _exact_int(canonical["schema_version"]) != 2
            or canonical["fresh_process_required"] is not True
            or type(canonical["modules"]) is not list
        ):
            raise FailedLifecycleError("Loaded modules identity contract differs")
        for item in canonical["modules"]:
            record = _mapping(item, "loaded module record")
            _exact_keys(
                record,
                {"module", "path", "loaded_source_sha256"},
                "loaded module record",
            )
            _nonempty_string(record["module"], "loaded module name")
            _validate_repository_file_hash(
                repository_root,
                record["path"],
                record["loaded_source_sha256"],
                "loaded module",
            )


def _validate_source_file_manifest(
    canonical: Mapping[str, Any],
    repository_root: Path,
    label: str,
    *,
    expected_paths: tuple[str, ...] | None,
    resolved_config: Mapping[str, Any] | None = None,
) -> None:
    _exact_keys(canonical, {"schema_version", "hashing_method", "files"}, label)
    source_manifest = label == "source identity"
    expected_schema = 2 if source_manifest else 1
    expected_method = (
        "sha256_file_bytes_sorted_repository_relative_paths"
        if source_manifest
        else "sha256_file_bytes_repository_relative_paths"
    )
    if _exact_int(canonical["schema_version"]) != expected_schema:
        raise FailedLifecycleError(f"{label} schema differs")
    if type(canonical["hashing_method"]) is not str:
        raise FailedLifecycleError(f"{label} hashing method type differs")
    if canonical["hashing_method"] != expected_method:
        raise FailedLifecycleError(f"{label} hashing method differs")
    files = canonical["files"]
    if type(files) is not list or not files:
        raise FailedLifecycleError(f"{label} files must be a nonempty list")
    seen: set[str] = set()
    ordered_paths: list[str] = []
    for item in files:
        record = _mapping(item, f"{label} file")
        _exact_keys(record, {"path", "sha256"}, f"{label} file")
        path = _nonempty_string(record["path"], f"{label} file path")
        if path in seen:
            raise FailedLifecycleError(f"{label} contains duplicate paths")
        seen.add(path)
        ordered_paths.append(path)
        _validate_repository_file_hash(
            repository_root,
            path,
            record["sha256"],
            label,
        )
    if ordered_paths != sorted(ordered_paths):
        raise FailedLifecycleError(f"{label} file order differs")
    if source_manifest:
        try:
            authenticated, _ = build_repository_file_identity(
                repository_root,
                ordered_paths,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise FailedLifecycleError(
                f"Research source identity byte authentication failed: {error}"
            ) from error
        difference = first_exact_difference(
            canonical,
            authenticated,
            "identities.source.canonical",
        )
        if difference is not None:
            raise FailedLifecycleError(
                f"Research source identity differs from repository bytes: {difference}"
            )
    if expected_paths is not None:
        if seen != set(expected_paths):
            raise FailedLifecycleError(
                f"{label} file set differs; "
                f"missing={sorted(set(expected_paths) - seen)}, "
                f"unexpected={sorted(seen - set(expected_paths))}"
            )
        return
    if resolved_config is None:
        raise FailedLifecycleError(
            "Research source identity cannot be authenticated without resolved config"
        )
    plan_path = _nonempty_string(
        resolved_config.get("evaluation_plan_path"),
        "research source identity evaluation plan path",
    )
    required = {*SOURCE_PATHS, plan_path}
    config_paths = seen - required
    if not required.issubset(seen) or len(config_paths) != 1:
        raise FailedLifecycleError(
            "Research source identity file set differs from the production contract"
        )
    config_path = next(iter(config_paths))
    config_parts = PurePosixPath(config_path).parts
    if (
        len(config_parts) < 3
        or config_parts[:2] != ("configs", "research_v2")
        or PurePosixPath(config_path).suffix not in {".yaml", ".yml"}
    ):
        raise FailedLifecycleError(
            "Research source identity config path differs from the production contract"
        )
    try:
        loaded = load_research_v2_config(
            _repository_path(
                config_path,
                repository_root.resolve(),
                "research source config path",
            ),
            project_root=repository_root,
        )
    except (OSError, TypeError, ValueError) as error:
        raise FailedLifecycleError(
            f"Research source config authentication failed: {error}"
        ) from error
    difference = first_exact_difference(
        resolved_config,
        loaded.resolved_payload(),
        "resolved_config",
    )
    if difference is not None:
        raise FailedLifecycleError(
            f"Research source config differs from repository config: {difference}"
        )


def _validate_repository_file_hash(
    repository_root: Path,
    value: Any,
    expected_sha256: Any,
    label: str,
) -> None:
    path = _repository_path(value, repository_root.resolve(), f"{label} path")
    expected = _sha(expected_sha256, f"{label} sha256")
    _validate_repository_regular_file(
        path,
        repository_root=repository_root,
        label=label,
    )
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise FailedLifecycleError(f"Cannot read {label} repository file") from error
    if actual != expected:
        raise FailedLifecycleError(f"{label} repository byte hash differs")


def _validate_repository_regular_file(
    path: Path,
    *,
    repository_root: Path,
    label: str,
) -> None:
    root = repository_root.resolve()
    try:
        path.resolve(strict=True).relative_to(root)
        info = path.stat(follow_symlinks=False)
    except (OSError, ValueError) as error:
        raise FailedLifecycleError(
            f"{label} repository file is missing or escapes the repository"
        ) from error
    if path.is_symlink() or _is_reparse_point(path) or not stat.S_ISREG(info.st_mode):
        raise FailedLifecycleError(
            f"{label} repository path is not a regular non-reparse file"
        )


def _validate_optional_research_context_artifacts(
    run_dir: Path,
    *,
    metadata: Mapping[str, Any],
    status_started: datetime,
    status_finished: datetime,
) -> tuple[dict[str, str], list[str]]:
    hashes: dict[str, str] = {}
    paths: list[str] = []
    for name in ("environment", "runtime", "git"):
        path = run_dir / f"{name}.json"
        if not path.exists():
            continue
        if not path.is_file():
            raise FailedLifecycleError(f"Research optional {name} is not a file")
        payload = _strict_json(path, f"research optional {name}")
        if name == "environment":
            _validate_research_environment(payload)
        elif name == "runtime":
            _validate_research_runtime(
                payload,
                status_started=status_started,
                status_finished=status_finished,
            )
        else:
            _validate_research_git(payload)
        if name in metadata and payload != metadata[name]:
            raise FailedLifecycleError(
                f"Research optional {name} disagrees with run metadata"
            )
        hashes[name] = canonical_sha256(payload)
        paths.append(f"{name}.json")
    return hashes, paths


def _reject_unsupported_early_research_artifacts(run_dir: Path) -> None:
    allowed_files = {
        "_FAILED",
        "execution_status.json",
        "run_metadata.json",
        "resolved_config.yaml",
        "environment.json",
        "runtime.json",
        "git.json",
        *(f"identities/{name}.json" for name in _RESEARCH_IDENTITY_NAMES),
    }
    actual_files = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    unexpected = sorted(actual_files - allowed_files)
    if unexpected:
        raise FailedLifecycleError(
            f"Research early failure has unsupported authoritative artifacts: {unexpected}"
        )


def _repository_path(value: Any, root: Path, label: str) -> Path:
    text = _nonempty_string(value, label)
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or ".." in posix.parts
        or ".." in windows.parts
        or "\\" in text
        or posix.as_posix() != text
        or (posix.parts and ":" in posix.parts[0])
    ):
        raise FailedLifecycleError(
            f"{label} must be a canonical repository-relative POSIX path "
            "without traversal"
        )
    lexical = root
    for part in posix.parts:
        lexical = lexical / part
        if lexical.exists() and (lexical.is_symlink() or _is_reparse_point(lexical)):
            raise FailedLifecycleError(f"{label} crosses a repository reparse path")
    resolved = lexical.resolve()
    if resolved != root and root not in resolved.parents:
        raise FailedLifecycleError(f"{label} escapes the repository")
    return resolved


def _validate_research_metadata_context(metadata: Mapping[str, Any]) -> None:
    _validate_research_environment(
        _mapping(metadata["environment"], "research environment")
    )
    _validate_research_runtime(_mapping(metadata["runtime"], "research runtime"))
    _validate_research_git(_mapping(metadata["git"], "research git state"))


def _validate_research_environment(environment: Mapping[str, Any]) -> None:
    environment_keys = {
        "python",
        "numpy",
        "pandas",
        "scikit_learn",
        "lightgbm",
        "pyarrow",
    }
    _exact_keys(environment, environment_keys, "research environment")
    for key in environment_keys:
        _nonempty_string(environment[key], f"research environment {key}")


def _validate_research_runtime(
    runtime: Mapping[str, Any],
    *,
    status_started: datetime | None = None,
    status_finished: datetime | None = None,
) -> None:
    runtime_keys = {
        "process_started_at_utc",
        "preflight_at_utc",
        "entry_point",
        "fresh_process_required",
        "network_access",
        "competition_assets_accessed",
    }
    _exact_keys(runtime, runtime_keys, "research runtime")
    process_started = _utc(
        runtime["process_started_at_utc"], "research process started"
    )
    preflight = _utc(runtime["preflight_at_utc"], "research preflight")
    if preflight < process_started:
        raise FailedLifecycleError("Research preflight predates process start")
    if status_started is not None and preflight > status_started:
        raise FailedLifecycleError("Research preflight follows run start")
    if status_finished is not None and process_started > status_finished:
        raise FailedLifecycleError("Research process starts after run finish")
    _nonempty_string(runtime["entry_point"], "research runtime entry point")
    if (
        runtime["fresh_process_required"] is not True
        or runtime["network_access"] != "disabled_during_execution"
        or runtime["competition_assets_accessed"] is not False
    ):
        raise FailedLifecycleError("Research runtime safety claims differ")


def _validate_research_git(git: Mapping[str, Any]) -> None:
    success_keys = {"branch", "commit", "dirty", "status_porcelain"}
    failure_keys = {"branch", "commit", "dirty", "error"}
    if set(git) == success_keys:
        if type(git["branch"]) is not str:
            raise FailedLifecycleError("Research git branch type differs")
        _git_object_id(git["commit"], "research git commit")
        if type(git["dirty"]) is not bool:
            raise FailedLifecycleError("Research git dirty type differs")
        _string_list(git["status_porcelain"], "research git status")
    elif set(git) == failure_keys:
        if any(git[key] is not None for key in ("branch", "commit", "dirty")):
            raise FailedLifecycleError("Research unavailable git fields differ")
        _nonempty_string(git["error"], "research git error")
    else:
        raise FailedLifecycleError("Research git state schema differs")


def validate_failed_autogluon_run(
    run_dir: Path,
    *,
    status: Mapping[str, Any],
    metadata: Mapping[str, Any],
    marker: Mapping[str, Any],
    resolved_config: Mapping[str, Any] | None,
    profile_resolution: Mapping[str, Any] | None,
    inventory: Mapping[str, Any],
) -> None:
    """Validate the exact standalone-runner failed lifecycle and inventory."""
    _validate_terminal_files(run_dir, failed_marker_must_be_json=True)
    _validate_tree_safety(run_dir, research=False)
    _exact_keys(status, _AUTO_STATUS_KEYS, "AutoGluon execution status")
    _exact_keys(metadata, _AUTO_METADATA_KEYS, "AutoGluon run metadata")
    _exact_keys(marker, _AUTO_MARKER_KEYS, "AutoGluon failed marker")
    if status["status"] != "failed" or marker["status"] != "failed":
        raise FailedLifecycleError("AutoGluon terminal state is not exactly failed")
    run_id = _nonempty_string(status["run_id"], "AutoGluon status run_id")
    if (
        run_id != run_dir.name
        or metadata["run_id"] != run_id
        or marker["run_id"] != run_id
    ):
        raise FailedLifecycleError("AutoGluon run IDs disagree")
    if _exact_int(metadata["schema_version"]) != 1:
        raise FailedLifecycleError("AutoGluon metadata schema version differs")
    config_hash = _sha(status["config_identity_sha256"], "AutoGluon config hash")
    profile_hash = _sha(status["profile_sha256"], "AutoGluon profile hash")
    seed = _exact_int(status["requested_seed"], "AutoGluon requested seed")
    if seed < 0:
        raise FailedLifecycleError("AutoGluon requested seed is negative")
    for value, expected, label in (
        (metadata["config_identity_sha256"], config_hash, "metadata config hash"),
        (metadata["profile_sha256"], profile_hash, "metadata profile hash"),
        (metadata["requested_seed"], seed, "metadata requested seed"),
    ):
        if value != expected or type(value) is not type(expected):
            raise FailedLifecycleError(f"AutoGluon {label} differs")
    profile_id = _nonempty_string(metadata["profile_id"], "AutoGluon profile ID")
    dataset_version = _nonempty_string(
        metadata["dataset_version"], "AutoGluon dataset version"
    )
    profile = _mapping(metadata["profile"], "AutoGluon profile")
    resources = _mapping(metadata["resources"], "AutoGluon resources")
    _exact_keys(resources, _AUTO_RESOURCE_KEYS, "AutoGluon resources")
    _validate_auto_resources(resources)
    try:
        configured_profile = get_profile(profile_id)
    except ValueError as error:
        raise FailedLifecycleError("AutoGluon profile ID is unknown") from error
    expected_profile = profile_summary(
        configured_profile,
        seed,
        _exact_int(resources["gpu_budget"], "AutoGluon GPU budget"),
    )
    if (
        profile != expected_profile
        or profile_sha256(
            configured_profile,
            seed,
            _exact_int(resources["gpu_budget"], "AutoGluon GPU budget"),
        )
        != profile_hash
    ):
        raise FailedLifecycleError("AutoGluon profile identity/hash differs")
    if resolved_config is None:
        raise FailedLifecycleError("AutoGluon failure lacks resolved config")
    _validate_auto_config(
        resolved_config,
        config_hash=config_hash,
        profile_id=profile_id,
        seed=seed,
        dataset_version=dataset_version,
        resources=resources,
    )
    _validate_local_operational(metadata["local_operational_nonportable"])
    _validate_auto_process_fields(status, metadata, marker)
    started = _utc(status["started_at_utc"], "AutoGluon started_at_utc")
    ended = _utc(status["ended_at_utc"], "AutoGluon ended_at_utc")
    if ended < started:
        raise FailedLifecycleError("AutoGluon failure ended before it started")
    if (
        _utc(metadata["started_at_utc"], "AutoGluon metadata started_at_utc") != started
        or _utc(metadata["ended_at_utc"], "AutoGluon metadata ended_at_utc") != ended
        or _utc(marker["ended_at_utc"], "AutoGluon marker ended_at_utc") != ended
    ):
        raise FailedLifecycleError("AutoGluon timestamps disagree")
    duration = _finite_nonnegative(
        status["duration_seconds"], "AutoGluon status duration"
    )
    if (
        _finite_nonnegative(metadata["duration_seconds"], "AutoGluon metadata duration")
        != duration
    ):
        raise FailedLifecycleError("AutoGluon duration disagrees")
    codes = _string_list(
        status["failure_codes"], "AutoGluon failure codes", nonempty=True
    )
    if len(codes) != len(set(codes)) or status["failure_reason"] != ";".join(codes):
        raise FailedLifecycleError("AutoGluon failure code/reason semantics differ")
    if (
        marker["failure_codes"] != codes
        or marker["failure_reason"] != status["failure_reason"]
    ):
        raise FailedLifecycleError("AutoGluon marker failure semantics differ")
    _nonempty_string(
        status["last_completed_observable_stage"], "AutoGluon last observable stage"
    )
    if (
        status["predictor_loading_attempted"] is not False
        or status["predictor_loading_succeeded"] is not False
        or status["full_logs_authoritative"] is not True
    ):
        raise FailedLifecycleError(
            "AutoGluon failed status claims invalid loading/log state"
        )
    missing = _string_list(
        status["missing_expected_artifacts"],
        "AutoGluon missing expected artifacts",
    )
    expected_missing = [
        relative
        for relative in _AUTO_EXPECTED_SUCCESS_ARTIFACTS
        if not (run_dir / Path(*relative.split("/"))).is_file()
    ]
    if missing != expected_missing:
        raise FailedLifecycleError("AutoGluon missing-artifact list differs")
    if (bool(missing)) != ("missing_expected_artifacts" in codes):
        raise FailedLifecycleError("AutoGluon missing-artifact failure code differs")
    raw_exit = status["child_process_exit_code"]
    if raw_exit not in {None, 0} and f"worker_exit_code:{raw_exit}" not in codes:
        raise FailedLifecycleError("AutoGluon worker exit failure code is missing")
    _validate_completion_fields(status, metadata)
    for reason in status["worker_completion_reason_codes"]:
        if reason not in codes:
            raise FailedLifecycleError(
                "AutoGluon completion reason is absent from failure codes"
            )
    _validate_pump_failures(status["log_pump_failures"], "status log pump failures")
    _validate_pump_failures(metadata["log_pump_failures"], "metadata log pump failures")
    if status["log_pump_failures"] != metadata["log_pump_failures"]:
        raise FailedLifecycleError("AutoGluon log-pump failures disagree")
    directories = _string_list(
        status["discovered_model_directories"],
        "AutoGluon discovered model directories",
    )
    if directories != discover_model_directories(run_dir / "predictor"):
        raise FailedLifecycleError("AutoGluon discovered model directories differ")
    if type(status["stderr_tail_bounded"]) is not str:
        raise FailedLifecycleError("AutoGluon bounded stderr tail type differs")
    if profile_resolution is not None:
        _exact_keys(
            profile_resolution,
            _AUTO_PROFILE_RESOLUTION_KEYS,
            "AutoGluon profile resolution",
        )
        expected_resolution = {
            "schema_version": 1,
            "profile_id": profile_id,
            "profile_sha256": profile_hash,
            "profile_identity": profile,
            "dataset_version": dataset_version,
            "requested_seed": seed,
            "portfolio": profile["portfolio"],
            "fit_strategy": resources["fit_strategy"],
            "fold_fitting_strategy": resources["fold_fitting_strategy"],
            "top_level_num_gpus_passed_to_fit": False,
        }
        if any(
            profile_resolution.get(key) != value
            for key, value in expected_resolution.items()
        ):
            raise FailedLifecycleError("AutoGluon profile resolution disagrees")
        _utc(
            profile_resolution["created_at_utc"],
            "AutoGluon profile resolution created_at_utc",
        )
        _nonempty_string(
            profile_resolution["autogluon_version"],
            "AutoGluon profile resolution version",
        )
        _string_list(
            profile_resolution["resolved_families"],
            "AutoGluon resolved families",
            nonempty=True,
        )
        _mapping(
            profile_resolution["family_resources"],
            "AutoGluon family resources",
        )
        _mapping(
            profile_resolution["resolved_hyperparameters"],
            "AutoGluon resolved hyperparameters",
        )
    _validate_auto_inventory(run_dir, inventory)


def _validate_auto_process_fields(
    status: Mapping[str, Any],
    metadata: Mapping[str, Any],
    marker: Mapping[str, Any],
) -> None:
    child = _optional_positive_int(status["child_pid"], "AutoGluon child PID")
    launched = _optional_positive_int(
        status["launched_process_pid"], "AutoGluon launched PID"
    )
    if (child is None) != (launched is None):
        raise FailedLifecycleError("AutoGluon process PID fields are inconsistent")
    if metadata["child_pid"] != child or metadata["launched_process_pid"] != launched:
        raise FailedLifecycleError("AutoGluon metadata PID fields disagree")
    raw = _optional_int(
        status["child_process_exit_code"], "AutoGluon child process exit code"
    )
    unsigned = _optional_int(
        status["windows_exit_code_unsigned"], "AutoGluon unsigned exit code"
    )
    if unsigned != unsigned_windows_exit_code(raw):
        raise FailedLifecycleError("AutoGluon signed/unsigned exit codes disagree")
    for payload, label in ((metadata, "metadata"), (marker, "marker")):
        if (
            payload["child_process_exit_code"] != raw
            or payload["windows_exit_code_unsigned"] != unsigned
        ):
            raise FailedLifecycleError(f"AutoGluon {label} exit codes disagree")


def _validate_completion_fields(
    status: Mapping[str, Any], metadata: Mapping[str, Any]
) -> None:
    valid = status["worker_completion_valid"]
    if type(valid) is not bool or metadata["worker_completion_valid"] is not valid:
        raise FailedLifecycleError("AutoGluon worker completion validity differs")
    reasons = _string_list(
        status["worker_completion_reason_codes"],
        "AutoGluon completion reason codes",
    )
    details = _string_list(
        status["worker_completion_details"], "AutoGluon completion details"
    )
    if metadata["worker_completion_reason_codes"] != reasons:
        raise FailedLifecycleError("AutoGluon completion reason codes disagree")
    if valid and reasons:
        raise FailedLifecycleError("Valid AutoGluon completion has reason codes")
    if not valid and not reasons:
        raise FailedLifecycleError("Invalid AutoGluon completion lacks reason codes")
    del details


def _validate_auto_inventory(run_dir: Path, inventory: Mapping[str, Any]) -> None:
    _exact_keys(
        inventory,
        {"generated_at_utc", "terminal_markers_excluded", "entries"},
        "AutoGluon artifact inventory",
    )
    _utc(inventory["generated_at_utc"], "AutoGluon inventory generated_at_utc")
    if inventory["terminal_markers_excluded"] is not True:
        raise FailedLifecycleError("AutoGluon inventory includes terminal markers")
    entries = inventory["entries"]
    if type(entries) is not list:
        raise FailedLifecycleError("AutoGluon inventory entries must be a list")
    paths: set[str] = set()
    for entry in entries:
        item = _mapping(entry, "AutoGluon inventory entry")
        path = _portable_relative(item.get("path"), "AutoGluon inventory entry path")
        if path in paths:
            raise FailedLifecycleError("AutoGluon inventory contains duplicate paths")
        paths.add(path)
        kind = item.get("type")
        if kind == "directory":
            _exact_keys(item, {"path", "type"}, "inventory directory entry")
        elif kind == "file":
            expected_keys = {"path", "type", "size_bytes"}
            size = _exact_int(item.get("size_bytes"), "inventory file size")
            if size < 0:
                raise FailedLifecycleError("AutoGluon inventory file size is negative")
            if size <= 10 * 1024 * 1024:
                expected_keys.add("sha256")
                _sha(item.get("sha256"), "inventory file hash")
            _exact_keys(item, expected_keys, "inventory file entry")
        else:
            raise FailedLifecycleError(
                "AutoGluon inventory contains a symlink or unknown entry type"
            )
    actual = build_inventory(run_dir)
    if entries != actual["entries"]:
        raise FailedLifecycleError("AutoGluon inventory hash/size/content differs")


def _validate_auto_config(
    config: Mapping[str, Any],
    *,
    config_hash: str,
    profile_id: str,
    seed: int,
    dataset_version: str,
    resources: Mapping[str, Any],
) -> None:
    _exact_keys(config, _AUTO_CONFIG_KEYS, "AutoGluon resolved config")
    if (
        _exact_int(config["schema_version"], "AutoGluon config schema") != 1
        or config_identity_sha256(dict(config)) != config_hash
        or config["profile_id"] != profile_id
        or config["seed"] != seed
    ):
        raise FailedLifecycleError("AutoGluon resolved config identity differs")
    dataset = _mapping(config["dataset"], "AutoGluon dataset config")
    _exact_keys(dataset, _AUTO_DATASET_KEYS, "AutoGluon dataset config")
    for key in _AUTO_DATASET_KEYS:
        _nonempty_string(dataset[key], f"AutoGluon dataset config {key}")
    if dataset["version"] != dataset_version:
        raise FailedLifecycleError("AutoGluon dataset version differs")
    predictor = _mapping(config["predictor"], "AutoGluon predictor config")
    _exact_keys(predictor, _AUTO_PREDICTOR_KEYS, "AutoGluon predictor config")
    if (
        predictor["problem_type"] != "binary"
        or predictor["eval_metric"] != "balanced_accuracy"
        or _exact_int(predictor["positive_class"], "AutoGluon positive class") != 1
        or _exact_int(predictor["verbosity"], "AutoGluon verbosity") < 0
    ):
        raise FailedLifecycleError("AutoGluon predictor config differs")
    configured_resources = _mapping(
        config["resources"], "AutoGluon configured resources"
    )
    _exact_keys(
        configured_resources,
        _AUTO_CONFIG_RESOURCE_KEYS,
        "AutoGluon configured resources",
    )
    expected_resources = {
        "time_limit_seconds": configured_resources["time_limit_seconds"],
        "num_cpus": configured_resources["num_cpus"],
        "gpu_budget": configured_resources["num_gpus"],
        "top_level_num_gpus_passed_to_fit": False,
        "fit_strategy": configured_resources["fit_strategy"],
        "fold_fitting_strategy": configured_resources["fold_fitting_strategy"],
    }
    if dict(resources) != expected_resources:
        raise FailedLifecycleError("AutoGluon resource/config metadata differs")
    fit = _mapping(config["fit"], "AutoGluon fit config")
    _exact_keys(fit, _AUTO_FIT_KEYS, "AutoGluon fit config")
    _nonempty_string(fit["presets"], "AutoGluon fit presets")
    if type(fit["calibrate_decision_threshold"]) is not bool:
        raise FailedLifecycleError("AutoGluon calibration flag type differs")
    artifacts = _mapping(config["artifacts"], "AutoGluon artifact config")
    _exact_keys(artifacts, {"root"}, "AutoGluon artifact config")
    _nonempty_string(artifacts["root"], "AutoGluon artifact root")


def _validate_local_operational(value: Any) -> None:
    local = _mapping(value, "AutoGluon local operational metadata")
    required = {"repository_root", "run_directory", "python_executable"}
    keys = frozenset(local)
    if keys not in {
        frozenset(required),
        frozenset(required | {"worker_command"}),
    }:
        raise FailedLifecycleError("AutoGluon local operational keys differ")
    for key in required:
        _nonempty_string(local[key], f"AutoGluon local operational {key}")
    if "worker_command" in local:
        _string_list(
            local["worker_command"],
            "AutoGluon worker command",
            nonempty=True,
        )


def _validate_auto_resources(resources: Mapping[str, Any]) -> None:
    if (
        _exact_int(resources["time_limit_seconds"]) <= 0
        or (type(resources["num_cpus"]) is not int and resources["num_cpus"] != "auto")
        or _exact_int(resources["gpu_budget"]) < 0
        or resources["top_level_num_gpus_passed_to_fit"] is not False
        or resources["fit_strategy"] != "sequential"
        or resources["fold_fitting_strategy"] != "sequential_local"
    ):
        raise FailedLifecycleError("AutoGluon resource metadata is invalid")


def _validate_terminal_files(
    run_dir: Path, *, failed_marker_must_be_json: bool
) -> None:
    if not (run_dir / "_FAILED").is_file():
        raise FailedLifecycleError("Failed source lacks _FAILED")
    if (run_dir / "_SUCCESS").exists():
        raise FailedLifecycleError("Failed source contains stale _SUCCESS")
    if failed_marker_must_be_json and (run_dir / "_FAILED").stat().st_size == 0:
        raise FailedLifecycleError("Failed terminal marker is empty")
    if (run_dir / "artifact_manifest.json").exists():
        raise FailedLifecycleError("Failed source contains a stale success manifest")


def _validate_tree_safety(run_dir: Path, *, research: bool) -> None:
    root = run_dir.resolve(strict=True)
    for path in run_dir.rglob("*"):
        relative = path.relative_to(run_dir).as_posix()
        if path.is_symlink() or _is_reparse_point(path):
            raise FailedLifecycleError(
                f"Failed source contains a reparse path: {relative}"
            )
        try:
            path.resolve(strict=True).relative_to(root)
        except (OSError, ValueError) as error:
            raise FailedLifecycleError(
                f"Failed source path escapes the run: {relative}"
            ) from error
        if research and _forbidden_research_path(relative):
            raise FailedLifecycleError(
                f"Failed research source contains a forbidden path: {relative}"
            )


def _is_reparse_point(path: Path) -> bool:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError:
        return False
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    junction = getattr(path, "is_junction", None)
    return bool(getattr(info, "st_file_attributes", 0) & flag) or bool(
        callable(junction) and junction()
    )


def _forbidden_research_path(relative: str) -> bool:
    normalized = relative.lower().replace("\\", "/")
    searchable = f"/{normalized}"
    return any(
        marker in searchable
        for marker in (
            "competition",
            "sample_submission",
            "x_test",
            "kaggle",
            "autogluon",
            "submission",
            "/models/",
            ".joblib",
            ".pkl",
        )
    )


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: _reject_constant(token),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise FailedLifecycleError(f"{label} is invalid JSON") from error
    return dict(_mapping(value, label))


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise FailedLifecycleError(
            f"{label} keys differ; missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise FailedLifecycleError(f"{label} must be an exact mapping")
    return value


def _list(value: Any) -> list[Any]:
    if type(value) is not list:
        raise FailedLifecycleError("Expected an exact list")
    return value


def _failure_mapping(value: Any, label: str) -> dict[str, str]:
    mapping = _mapping(value, label)
    _exact_keys(mapping, {"type", "message"}, label)
    result = {
        "type": _nonempty_string(mapping["type"], f"{label}.type"),
        "message": _nonempty_string(mapping["message"], f"{label}.message"),
    }
    return result


def _sha_mapping(value: Any, label: str) -> dict[str, str]:
    mapping = _mapping(value, label)
    return {str(key): _sha(item, f"{label}.{key}") for key, item in mapping.items()}


def _git_object_id(value: Any, label: str) -> str:
    text = _nonempty_string(value, label)
    if len(text) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise FailedLifecycleError(f"{label} must be a lowercase Git object ID")
    return text


def _sha(value: Any, label: str) -> str:
    text = _nonempty_string(value, label)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise FailedLifecycleError(f"{label} must be a lowercase SHA-256")
    return text


def _exact_int(value: Any, label: str = "value") -> int:
    if type(value) is not int:
        raise FailedLifecycleError(f"{label} must be an exact integer")
    return value


def _optional_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    return _exact_int(value, label)


def _optional_positive_int(value: Any, label: str) -> int | None:
    result = _optional_int(value, label)
    if result is not None and result <= 0:
        raise FailedLifecycleError(f"{label} must be positive")
    return result


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FailedLifecycleError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise FailedLifecycleError(f"{label} must be finite and nonnegative")
    return result


def _nonempty_string(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise FailedLifecycleError(f"{label} must be a nonempty string")
    return value


def _string_list(value: Any, label: str, *, nonempty: bool = False) -> list[str]:
    if type(value) is not list or any(
        type(item) is not str or not item for item in value
    ):
        raise FailedLifecycleError(f"{label} must be a list of nonempty strings")
    if nonempty and not value:
        raise FailedLifecycleError(f"{label} must not be empty")
    return list(value)


def _validate_pump_failures(value: Any, label: str) -> None:
    if type(value) is not list:
        raise FailedLifecycleError(f"{label} must be a list")
    for item in value:
        mapping = _mapping(item, label)
        _exact_keys(mapping, _PUMP_FAILURE_KEYS, label)
        for key in _PUMP_FAILURE_KEYS:
            _nonempty_string(mapping[key], f"{label}.{key}")


def _utc(value: Any, label: str) -> datetime:
    text = _nonempty_string(value, label)
    if not (text.endswith("Z") or text.endswith("+00:00")):
        raise FailedLifecycleError(f"{label} must use UTC")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise FailedLifecycleError(f"{label} is not a valid timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise FailedLifecycleError(f"{label} must use UTC")
    return parsed


def _portable_relative(value: Any, label: str) -> str:
    text = _nonempty_string(value, label).replace("\\", "/")
    posix = PurePosixPath(text)
    windows = PureWindowsPath(str(value))
    if (
        text.startswith("/")
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or ".." in posix.parts
        or "." in posix.parts
        or posix.as_posix() != text
    ):
        raise FailedLifecycleError(f"{label} must be a canonical relative path")
    return text


def _reject_constant(token: str) -> Any:
    raise ValueError(f"Non-finite JSON constant: {token}")
