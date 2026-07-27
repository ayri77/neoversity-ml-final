"""Read-only inspection and worker-side inspection export for AutoGluon runs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
from typing import Any

import yaml

from src.churn_ml.autogluon_artifacts import (
    build_inventory,
    discover_model_directories,
    write_json,
)
from src.churn_ml.autogluon_completion import (
    CompletionExpectations,
    completion_inspection_mismatches,
    load_and_validate_completion,
)
from src.churn_ml.autogluon_config import config_identity_sha256


_REQUIRED_PREDICTOR_FILES = ("predictor.pkl", "learner.pkl", "version.txt")
_REQUIRED_SUCCESS_FILES = (
    "resolved_config.yaml",
    "run_metadata.json",
    "environment.json",
    "execution_status.json",
    "dataset_manifest.json",
    "profile_resolution.json",
    "worker_result.json",
    "artifact_inventory.json",
    "logs/supervisor.log",
    "logs/worker.stdout.log",
    "logs/worker.stderr.log",
    "inspection/leaderboard.csv",
    "inspection/summary.json",
    "predictor/predictor.pkl",
    "predictor/learner.pkl",
    "predictor/version.txt",
)


def inspect_run(
    run_dir: Path,
    *,
    attempt_load: bool | None = None,
    predictor_loader: Any | None = None,
) -> dict[str, Any]:
    """Classify all terminal invariants before any optional predictor load."""
    run_dir = run_dir.resolve(strict=True)
    report, completion = _classify_filesystem(run_dir)
    should_load = (
        report["classification"] == "complete" if attempt_load is None else attempt_load
    )
    report["predictor_loading_attempted"] = should_load
    report["predictor_loading_succeeded"] = False
    report["load_error"] = None
    if not should_load:
        return report

    predictor_dir = run_dir / "predictor"
    try:
        loader = predictor_loader or _load_predictor
        predictor = loader(predictor_dir)
        requested_seed = completion.get("requested_seed") if completion else None
        loaded_report = predictor_report(predictor, requested_seed=requested_seed)
        report["predictor"] = loaded_report
        report["predictor_loading_succeeded"] = True
        if completion is not None:
            mismatches = completion_inspection_mismatches(completion, loaded_report)
            if mismatches:
                report["reason_codes"] = sorted(
                    set(report["reason_codes"]) | set(mismatches)
                )
                report["classification"] = "corrupt"
    except Exception as error:  # inspection must report corrupt/partial predictors
        report["load_error"] = f"{type(error).__name__}: {error}"
        if report["classification"] == "complete":
            report["classification"] = "corrupt"
            report["reason_codes"] = sorted(
                set(report["reason_codes"]) | {"predictor_load_failed"}
            )
    return report


def export_worker_inspection(
    run_dir: Path,
    predictor: Any,
    *,
    requested_seed: int,
) -> dict[str, Any]:
    """Export sanitized inspection artifacts after fit, before worker completion."""
    inspection_dir = run_dir / "inspection"
    inspection_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = predictor.leaderboard(silent=True)
    leaderboard.to_csv(inspection_dir / "leaderboard.csv", index=False)
    summary = predictor_report(predictor, requested_seed=requested_seed)
    write_json(inspection_dir / "summary.json", summary)
    return summary


def predictor_report(
    predictor: Any,
    *,
    requested_seed: int | None = None,
) -> dict[str, Any]:
    """Gather portable metadata through public APIs and isolate local paths."""
    models = list(predictor.model_names())
    best_model = getattr(predictor, "model_best", None)
    leaderboard_error: str | None = None
    try:
        leaderboard = predictor.leaderboard(silent=True).to_dict(orient="records")
    except Exception as error:
        leaderboard = []
        leaderboard_error = f"{type(error).__name__}: {error}"
    info_error: str | None = None
    try:
        raw_info = predictor.info()
    except Exception as error:
        raw_info = {}
        info_error = f"{type(error).__name__}: {error}"
    portable_info, nonportable = _sanitize_paths(raw_info)
    weights = _find_ensemble_weights(portable_info, best_model)
    effective_seeds = sorted(set(_find_effective_seeds(portable_info)))
    effective_seed = effective_seeds[0] if len(effective_seeds) == 1 else None
    if requested_seed is None or not effective_seeds:
        seed_status = "unavailable"
    elif effective_seeds == [requested_seed]:
        seed_status = "matched"
    else:
        seed_status = "mismatch"
    threshold = getattr(predictor, "decision_threshold", None)
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
    ):
        raise ValueError("Predictor decision threshold is not a finite number.")
    result: dict[str, Any] = {
        "decision_threshold": float(threshold),
        "models": models,
        "best_model": best_model,
        "leaderboard": _json_safe(leaderboard),
        "leaderboard_error": leaderboard_error,
        "ensemble_weights": weights,
        "ensemble_weights_source": (
            "best-effort traversal of public TabularPredictor.info()"
            if weights
            else None
        ),
        "predictor_info": _json_safe(portable_info),
        "predictor_info_error": info_error,
        "requested_seed": requested_seed,
        "effective_seed": effective_seed,
        "effective_seeds_observed": effective_seeds,
        "effective_seed_status": seed_status,
    }
    if nonportable:
        result["local_operational_nonportable"] = nonportable
    return result


def _classify_filesystem(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    reasons: list[str] = []
    success = (run_dir / "_SUCCESS").is_file()
    failed = (run_dir / "_FAILED").is_file()
    if success and failed:
        reasons.append("terminal_markers_conflict")
    elif not success and not failed:
        reasons.append("terminal_marker_missing")
    elif failed:
        reasons.append("terminal_failed")

    metadata, metadata_error = _read_json(run_dir / "run_metadata.json")
    status, status_error = _read_json(run_dir / "execution_status.json")
    profile_resolution, profile_error = _read_json(run_dir / "profile_resolution.json")
    inspection_summary, inspection_error = _read_json(
        run_dir / "inspection" / "summary.json"
    )
    inventory, inventory_error = _read_json(run_dir / "artifact_inventory.json")
    for error in (
        metadata_error,
        status_error,
        profile_error,
        inspection_error,
        inventory_error,
    ):
        if error:
            reasons.append(error)

    missing_files = [
        relative
        for relative in _REQUIRED_SUCCESS_FILES
        if not (run_dir / relative).is_file()
    ]
    if success and missing_files:
        reasons.append("success_required_artifacts_missing")
    predictor_complete = all(
        (run_dir / "predictor" / name).is_file() for name in _REQUIRED_PREDICTOR_FILES
    )
    if success and not predictor_complete:
        reasons.append("success_predictor_structure_incomplete")

    completion: dict[str, Any] | None = None
    expectations = _completion_expectations(metadata, profile_resolution, reasons)
    if expectations is not None:
        validation = load_and_validate_completion(
            run_dir / "worker_result.json",
            expectations,
            inspection_summary=inspection_summary,
        )
        reasons.extend(validation.reason_codes)
        completion = validation.payload
    elif success:
        reasons.append("completion_expectations_unavailable")

    _validate_execution_status(status, metadata, completion, success, reasons)
    _validate_config_hash(run_dir, metadata, reasons)
    _validate_metadata_profile_hash(metadata, reasons)
    _validate_profile_resolution(profile_resolution, metadata, reasons)
    _validate_inventory(run_dir, inventory, reasons)
    _validate_terminal_marker(run_dir, success, failed, metadata, reasons)

    unique_reasons = sorted(set(reasons))
    if success and not failed and not unique_reasons:
        classification = "complete"
    elif success or (success and failed):
        classification = "corrupt"
    elif any(
        reason.endswith(("invalid_json", "malformed"))
        or "conflict" in reason
        or "mismatch" in reason
        for reason in unique_reasons
    ):
        classification = "corrupt"
    else:
        classification = "incomplete"
    report: dict[str, Any] = {
        "classification": classification,
        "reason_codes": unique_reasons,
        "success_marker": success,
        "failed_marker": failed,
        "execution_status": status,
        "worker_completion_valid": completion is not None
        and not any(reason.startswith("worker_result_") for reason in unique_reasons),
        "predictor_structure_complete": predictor_complete,
        "missing_required_artifacts": missing_files,
        "discovered_model_directories": discover_model_directories(
            run_dir / "predictor"
        ),
        "local_operational_nonportable": {"run_directory": str(run_dir)},
    }
    return report, completion


def _completion_expectations(
    metadata: dict[str, Any] | None,
    profile_resolution: dict[str, Any] | None,
    reasons: list[str],
) -> CompletionExpectations | None:
    if metadata is None or profile_resolution is None:
        return None
    profile = metadata.get("profile")
    required = {
        "child_pid": metadata.get("child_pid"),
        "profile_id": metadata.get("profile_id"),
        "profile_sha256": metadata.get("profile_sha256"),
        "dataset_version": metadata.get("dataset_version"),
        "requested_seed": metadata.get("requested_seed"),
        "resolved_families": profile_resolution.get("resolved_families"),
    }
    if type(profile) is not dict:
        reasons.append("run_metadata_profile_missing")
    if (
        type(required["child_pid"]) is not int
        or type(required["profile_id"]) is not str
        or type(required["profile_sha256"]) is not str
        or type(required["dataset_version"]) is not str
        or type(required["requested_seed"]) is not int
        or type(required["resolved_families"]) is not list
        or any(type(item) is not str for item in required["resolved_families"])
    ):
        reasons.append("completion_expectations_invalid")
        return None
    return CompletionExpectations(
        worker_pid=required["child_pid"],
        profile_id=required["profile_id"],
        profile_sha256=required["profile_sha256"],
        dataset_version=required["dataset_version"],
        requested_seed=required["requested_seed"],
        resolved_families=tuple(required["resolved_families"]),
    )


def _validate_execution_status(
    status: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
    completion: dict[str, Any] | None,
    success: bool,
    reasons: list[str],
) -> None:
    if status is None:
        return
    expected = "completed" if success else "failed"
    if status.get("status") != expected:
        reasons.append("execution_status_terminal_mismatch")
    if metadata is not None:
        for key in ("run_id", "child_pid", "profile_sha256", "config_identity_sha256"):
            if status.get(key) != metadata.get(key):
                reasons.append(f"execution_status_{key}_mismatch")
    if completion is not None and status.get("child_pid") != completion.get(
        "worker_pid"
    ):
        reasons.append("execution_status_worker_pid_mismatch")


def _validate_config_hash(
    run_dir: Path,
    metadata: dict[str, Any] | None,
    reasons: list[str],
) -> None:
    if metadata is None:
        return
    try:
        payload = yaml.safe_load(
            (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError):
        reasons.append("resolved_config_unreadable")
        return
    if type(payload) is not dict:
        reasons.append("resolved_config_not_mapping")
        return
    if config_identity_sha256(payload) != metadata.get("config_identity_sha256"):
        reasons.append("resolved_config_hash_mismatch")


def _validate_metadata_profile_hash(
    metadata: dict[str, Any] | None,
    reasons: list[str],
) -> None:
    if metadata is None:
        return
    profile = metadata.get("profile")
    if type(profile) is not dict:
        reasons.append("run_metadata_profile_missing")
        return
    profile_hash = hashlib.sha256(
        json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if profile_hash != metadata.get("profile_sha256"):
        reasons.append("run_metadata_profile_hash_mismatch")


def _validate_profile_resolution(
    profile_resolution: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
    reasons: list[str],
) -> None:
    if profile_resolution is None or metadata is None:
        return
    comparisons = {
        "profile_id": metadata.get("profile_id"),
        "profile_sha256": metadata.get("profile_sha256"),
        "profile_identity": metadata.get("profile"),
        "dataset_version": metadata.get("dataset_version"),
        "requested_seed": metadata.get("requested_seed"),
        "top_level_num_gpus_passed_to_fit": False,
    }
    for key, expected in comparisons.items():
        if profile_resolution.get(key) != expected:
            reasons.append(f"profile_resolution_{key}_mismatch")


def _validate_inventory(
    run_dir: Path,
    inventory: dict[str, Any] | None,
    reasons: list[str],
) -> None:
    if inventory is None:
        return
    if set(inventory) != {"generated_at_utc", "terminal_markers_excluded", "entries"}:
        reasons.append("artifact_inventory_schema_invalid")
        return
    if (
        inventory.get("terminal_markers_excluded") is not True
        or type(inventory.get("entries")) is not list
    ):
        reasons.append("artifact_inventory_schema_invalid")
        return
    try:
        actual_entries = build_inventory(run_dir)["entries"]
    except OSError:
        reasons.append("artifact_inventory_recompute_failed")
        return
    if inventory["entries"] != actual_entries:
        reasons.append("artifact_inventory_mismatch")


def _validate_terminal_marker(
    run_dir: Path,
    success: bool,
    failed: bool,
    metadata: dict[str, Any] | None,
    reasons: list[str],
) -> None:
    marker_name = (
        "_SUCCESS"
        if success and not failed
        else "_FAILED"
        if failed and not success
        else None
    )
    if marker_name is None:
        return
    marker, error = _read_json(run_dir / marker_name)
    if error or marker is None:
        reasons.append("terminal_marker_malformed")
        return
    expected_status = "completed" if marker_name == "_SUCCESS" else "failed"
    if marker.get("status") != expected_status:
        reasons.append("terminal_marker_status_mismatch")
    if metadata is not None and marker.get("run_id") != metadata.get("run_id"):
        reasons.append("terminal_marker_run_id_mismatch")


def _sanitize_paths(value: Any) -> tuple[Any, dict[str, Any]]:
    nonportable: dict[str, Any] = {}

    def visit(item: Any, dotted: str) -> Any:
        if isinstance(item, dict):
            result: dict[str, Any] = {}
            for key, child in item.items():
                key_text = str(key)
                child_path = f"{dotted}.{key_text}" if dotted else key_text
                if "path" in key_text.lower() or _is_absolute_string(child):
                    nonportable[child_path] = _json_safe(child)
                else:
                    result[key_text] = visit(child, child_path)
            return result
        if isinstance(item, (list, tuple, set)):
            return [
                visit(child, f"{dotted}[{index}]") for index, child in enumerate(item)
            ]
        if _is_absolute_string(item):
            nonportable[dotted] = item
            return "<local-operational-path-removed>"
        return _json_safe(item)

    return visit(value, "predictor_info"), nonportable


def _is_absolute_string(value: Any) -> bool:
    if type(value) is not str or not value:
        return False
    windows = PureWindowsPath(value)
    return (
        Path(value).is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or value.startswith(("/", "\\"))
    )


def _find_effective_seeds(value: Any) -> list[int]:
    seeds: list[int] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "model_random_seed" and type(child) is int:
                seeds.append(child)
            seeds.extend(_find_effective_seeds(child))
    elif isinstance(value, list):
        for child in value:
            seeds.extend(_find_effective_seeds(child))
    return seeds


def _load_predictor(predictor_dir: Path) -> Any:
    from autogluon.tabular import TabularPredictor  # type: ignore[import-not-found]

    return TabularPredictor.load(str(predictor_dir))


def _find_ensemble_weights(
    info: Any, best_model: str | None
) -> dict[str, float] | None:
    if not isinstance(info, dict):
        return None
    model_info = info.get("model_info")
    root: Any = model_info.get(best_model, {}) if isinstance(model_info, dict) else info
    stack = [root]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"model_weights", "weights"} and isinstance(child, dict):
                    numeric = {
                        str(name): float(weight)
                        for name, weight in child.items()
                        if type(weight) in {int, float}
                    }
                    if numeric:
                        return numeric
                stack.append(child)
        elif isinstance(value, list):
            stack.extend(value)
    return None


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, f"{path.name}_missing"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, f"{path.name}_invalid_json"
    if type(value) is not dict:
        return None, f"{path.name}_not_object"
    return value, None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(child) for child in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (ValueError, TypeError):
            pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)
