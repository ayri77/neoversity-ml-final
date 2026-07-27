"""Exact failed-run lifecycle validators owned by the MLflow index layer."""

from __future__ import annotations

import json
import math
import stat
from collections.abc import Mapping
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
from src.churn_ml.mlflow_mapping import canonical_sha256


class FailedLifecycleError(RuntimeError):
    """Raised when a failed source does not prove an exact trusted lifecycle."""


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
_RESEARCH_RESOLVED_CONFIG_KEYS = {
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
_RESEARCH_PERSISTENCE_KEYS = {
    "resolved_config",
    "identities",
    "assignments",
    "predictions",
    "metrics",
    "threshold_curves",
    "models",
}

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
) -> None:
    """Validate the exact failed Experiment Core v2 lifecycle without promotion."""
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
    if early:
        if completed != 0:
            raise FailedLifecycleError("Research early failure claims fold progress")
        return

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
    if resolved_config is None:
        raise FailedLifecycleError("Research full failure lacks resolved config")
    _validate_research_resolved_config(resolved_config)
    plan = _mapping(resolved_config.get("evaluation_plan"), "evaluation plan")
    configured_expected = len(
        _list(
            _mapping(plan.get("outer_evaluation"), "outer evaluation").get(
                "repeat_seeds"
            )
        )
    ) * _exact_int(
        _mapping(plan.get("outer_evaluation"), "outer evaluation").get("n_splits"),
        "outer n_splits",
    )
    expected_values = {
        "experiment_id": _mapping(
            resolved_config.get("experiment"), "research experiment"
        ).get("id"),
        "plan_id": _mapping(plan.get("plan"), "research plan identity").get("id"),
        "feature_pipeline_id": _mapping(
            resolved_config.get("feature_pipeline"), "research feature pipeline"
        ).get("id"),
        "candidate_adapter_id": _mapping(
            resolved_config.get("candidate_adapter"), "research candidate adapter"
        ).get("id"),
    }
    if any(metadata.get(key) != value for key, value in expected_values.items()):
        raise FailedLifecycleError(
            "Research plan/dataset/pipeline/adapter metadata disagrees with config"
        )
    if configured_expected != expected:
        raise FailedLifecycleError("Research expected fold count disagrees with plan")
    identities = run_dir / "identities"
    if completed > 0 and not identities.is_dir():
        raise FailedLifecycleError("Research progressed failure lacks identities")
    for filename, hash_key in _RESEARCH_IDENTITY_NAMES.items():
        path = identities / f"{filename}.json"
        if completed > 0 and not path.is_file():
            raise FailedLifecycleError(
                f"Research identity artifact is missing: {filename}"
            )
        if path.exists():
            if not path.is_file():
                raise FailedLifecycleError(
                    f"Research identity artifact is not a file: {filename}"
                )
            payload = _strict_json(path, f"research identity {filename}")
            _exact_keys(payload, {"sha256", "canonical"}, f"identity {filename}")
            if (
                payload["sha256"] != hashes[hash_key]
                or canonical_sha256(payload["canonical"]) != hashes[hash_key]
            ):
                raise FailedLifecycleError(
                    f"Research identity hash differs: {filename}"
                )


def _validate_research_resolved_config(config: Mapping[str, Any]) -> None:
    _exact_keys(config, _RESEARCH_RESOLVED_CONFIG_KEYS, "research resolved config")
    if _exact_int(config["schema_version"], "research config schema") != 2:
        raise FailedLifecycleError("Research resolved config schema differs")
    sections = {
        "experiment": {"id"},
        "dataset": {"version"},
        "feature_pipeline": {"id", "contract"},
        "candidate_adapter": {"id", "contract"},
        "artifacts": {"root"},
        "persistence": _RESEARCH_PERSISTENCE_KEYS,
        "tracking": {"enabled"},
    }
    for name, keys in sections.items():
        section = _mapping(config[name], f"research config {name}")
        _exact_keys(section, keys, f"research config {name}")
    for section_name, key in (
        ("experiment", "id"),
        ("dataset", "version"),
        ("feature_pipeline", "id"),
        ("candidate_adapter", "id"),
        ("artifacts", "root"),
    ):
        _nonempty_string(
            _mapping(config[section_name], section_name)[key],
            f"research config {section_name}.{key}",
        )
    _nonempty_string(config["evaluation_plan_path"], "research evaluation plan path")
    _mapping(
        _mapping(config["feature_pipeline"], "feature pipeline")["contract"],
        "research feature pipeline contract",
    )
    _mapping(
        _mapping(config["candidate_adapter"], "candidate adapter")["contract"],
        "research candidate adapter contract",
    )
    persistence = _mapping(config["persistence"], "research persistence")
    if any(type(persistence[key]) is not bool for key in _RESEARCH_PERSISTENCE_KEYS):
        raise FailedLifecycleError("Research persistence flag types differ")
    if _mapping(config["tracking"], "research tracking")["enabled"] is not False:
        raise FailedLifecycleError("Research tracking must remain disabled")
    plan = _mapping(config["evaluation_plan"], "research evaluation plan")
    plan_identity = _mapping(plan.get("plan"), "research plan identity")
    _nonempty_string(plan_identity.get("id"), "research plan ID")
    plan_dataset = _mapping(plan.get("dataset"), "research plan dataset")
    if plan_dataset.get("version") != _mapping(config["dataset"], "dataset")["version"]:
        raise FailedLifecycleError("Research plan/config dataset version differs")


def _validate_research_metadata_context(metadata: Mapping[str, Any]) -> None:
    environment = _mapping(metadata["environment"], "research environment")
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
    runtime = _mapping(metadata["runtime"], "research runtime")
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
    _nonempty_string(runtime["entry_point"], "research runtime entry point")
    if (
        runtime["fresh_process_required"] is not True
        or runtime["network_access"] != "disabled_during_execution"
        or runtime["competition_assets_accessed"] is not False
    ):
        raise FailedLifecycleError("Research runtime safety claims differ")
    git = _mapping(metadata["git"], "research git state")
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
