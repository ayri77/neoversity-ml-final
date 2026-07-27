from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import yaml

from src.churn_ml.paired_comparison import (
    COMPARISON_SCHEMA_VERSION,
    ComparisonPolicy,
    PairedComparisonResult,
    build_comparison_result,
    deterministic_comparison_identity,
    load_completed_research_v2_run,
)
from src.churn_ml.research_data import canonical_sha256


SAFE_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
COMPARISON_SOURCE_PATHS = (
    "scripts/compare_research_v2.py",
    "src/churn_ml/paired_comparison.py",
    "src/churn_ml/paired_comparison_artifacts.py",
    "src/churn_ml/paired_comparison_cli.py",
)
PRIMARY_ARTIFACTS = {
    "resolved_comparison_config.yaml",
    "compatibility_report.json",
    "baseline_run_reference.json",
    "candidate_run_reference.json",
    "repeat_metrics.csv",
    "fold_metrics.csv",
    "aggregate_summary.json",
    "prediction_comparison.json",
    "blend_repeat_metrics.csv",
    "blend_fold_metrics.csv",
    "blend_summary.json",
    "decision_report.json",
}


class PairedComparisonArtifactError(RuntimeError):
    """Raised when comparison artifacts cannot satisfy their lifecycle."""


def comparison_source_provenance(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    records: list[dict[str, Any]] = []
    for relative in COMPARISON_SOURCE_PATHS:
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise PairedComparisonArtifactError(
                f"Comparison source is missing or outside the repository: {relative}."
            )
        raw = path.read_bytes()
        records.append(
            {
                "path": relative,
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    canonical = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "hashing_method": "sha256_file_bytes_repository_relative_paths",
        "files": records,
    }
    return {**canonical, "sha256": canonical_sha256(canonical)}


def create_comparison_artifacts(
    *,
    result: PairedComparisonResult,
    policy: ComparisonPolicy,
    project_root: Path,
    output_root: Path,
    comparison_id: str | None,
    before_success: Callable[[], None] | None = None,
) -> Path:
    root = project_root.resolve()
    output = _repository_path(output_root, root, "output root")
    identity = deterministic_comparison_identity(
        result.baseline_reference,
        result.candidate_reference,
        policy,
    )
    resolved_id = comparison_id or f"paired_{identity[:20]}"
    if SAFE_SLUG.fullmatch(resolved_id) is None:
        raise PairedComparisonArtifactError(
            "comparison_id must use the existing safe-slug format and 128-char limit."
        )
    comparison_root = (output / resolved_id).resolve()
    if comparison_root.parent != output:
        raise PairedComparisonArtifactError("Comparison directory escapes output root.")
    if comparison_root.exists():
        raise PairedComparisonArtifactError(
            f"Comparison directory already exists: {comparison_root}."
        )
    comparison_root.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    try:
        provenance = comparison_source_provenance(root)
        resolved = {
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "comparison_id": resolved_id,
            "comparison_identity_sha256": identity,
            "created_at_utc": started.isoformat(),
            "project_contract": "experiment_core_v2_paired_comparison",
            "baseline_run": result.baseline_reference.to_dict(),
            "candidate_run": result.candidate_reference.to_dict(),
            "comparison_policy": policy.to_dict(),
            "source_provenance": provenance,
            "prediction_alignment_keys": {
                "outer_validation": [
                    "repeat",
                    "repeat_seed",
                    "outer_fold",
                    "row_position",
                ],
                "threshold_selection": [
                    "repeat",
                    "repeat_seed",
                    "outer_fold",
                    "threshold_selection_fold",
                    "row_position",
                ],
            },
            "fixed_blend": {
                "baseline_weight": 0.5,
                "candidate_weight": 0.5,
                "weight_search": False,
                "threshold_source": "threshold_selection_oof",
            },
            "tracking_enabled": False,
            "competition_assets_accessed": False,
        }
        _atomic_yaml(comparison_root / "resolved_comparison_config.yaml", resolved)
        _atomic_json(
            comparison_root / "compatibility_report.json",
            result.compatibility.to_dict(),
        )
        _atomic_json(
            comparison_root / "baseline_run_reference.json",
            result.baseline_reference.to_dict(),
        )
        _atomic_json(
            comparison_root / "candidate_run_reference.json",
            result.candidate_reference.to_dict(),
        )
        _atomic_csv(
            comparison_root / "repeat_metrics.csv",
            result.paired_metrics.repeat_metrics,
        )
        _atomic_csv(
            comparison_root / "fold_metrics.csv",
            result.paired_metrics.fold_metrics,
        )
        _atomic_json(
            comparison_root / "aggregate_summary.json",
            result.paired_metrics.aggregate_summary,
        )
        _atomic_json(
            comparison_root / "prediction_comparison.json",
            result.prediction_comparison,
        )
        _atomic_csv(
            comparison_root / "blend_repeat_metrics.csv",
            result.blend.repeat_metrics,
        )
        _atomic_csv(
            comparison_root / "blend_fold_metrics.csv",
            result.blend.fold_metrics,
        )
        _atomic_json(comparison_root / "blend_summary.json", result.blend.summary)
        _atomic_json(comparison_root / "decision_report.json", result.decision_report)
        inventory = build_comparison_inventory(comparison_root)
        _atomic_json(comparison_root / "artifact_inventory.json", inventory)
        manifest = build_comparison_manifest(comparison_root)
        _atomic_json(comparison_root / "manifest.json", manifest)
        validate_comparison_artifacts(
            comparison_root,
            project_root=root,
            require_success=False,
            verify_manifest=True,
        )
        if before_success is not None:
            before_success()
        _atomic_json(
            comparison_root / "_SUCCESS",
            {
                "schema_version": COMPARISON_SCHEMA_VERSION,
                "manifest_sha256": manifest["manifest_sha256"],
            },
        )
        return comparison_root
    except BaseException as error:
        if not (comparison_root / "_SUCCESS").exists():
            _best_effort_failure_marker(comparison_root, error)
        raise


def validate_comparison_artifacts(
    comparison_root: Path,
    *,
    project_root: Path,
    require_success: bool,
    verify_manifest: bool,
) -> None:
    root = comparison_root.resolve()
    if not root.is_dir():
        raise PairedComparisonArtifactError("Comparison directory is missing.")
    success = root / "_SUCCESS"
    failed = root / "_FAILED"
    if failed.exists():
        raise PairedComparisonArtifactError("Comparison contains _FAILED.")
    if require_success and not success.is_file():
        raise PairedComparisonArtifactError("Comparison does not contain _SUCCESS.")
    if not require_success and success.exists():
        raise PairedComparisonArtifactError(
            "_SUCCESS exists before comparison semantic validation."
        )
    expected = PRIMARY_ARTIFACTS | {"artifact_inventory.json", "manifest.json"}
    if require_success:
        expected.add("_SUCCESS")
    actual = {path.name for path in root.iterdir() if path.is_file()}
    if actual != expected:
        raise PairedComparisonArtifactError(
            "Comparison artifact inventory differs; "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}."
        )
    resolved = _read_yaml(root / "resolved_comparison_config.yaml")
    _validate_resolved_config(resolved, root, project_root)
    policy = _policy_from_resolved(resolved["comparison_policy"])
    baseline_reference = _read_json(root / "baseline_run_reference.json")
    candidate_reference = _read_json(root / "candidate_run_reference.json")
    if baseline_reference != resolved["baseline_run"]:
        raise PairedComparisonArtifactError("Baseline reference differs.")
    if candidate_reference != resolved["candidate_run"]:
        raise PairedComparisonArtifactError("Candidate reference differs.")
    baseline = load_completed_research_v2_run(
        project_root / baseline_reference["repository_relative_path"],
        project_root=project_root,
    )
    candidate = load_completed_research_v2_run(
        project_root / candidate_reference["repository_relative_path"],
        project_root=project_root,
    )
    recomputed = build_comparison_result(baseline, candidate, policy)
    _assert_json_equal(
        _read_json(root / "compatibility_report.json"),
        recomputed.compatibility.to_dict(),
        "compatibility report",
    )
    _assert_frame_equal(
        pd.read_csv(root / "repeat_metrics.csv"),
        recomputed.paired_metrics.repeat_metrics,
        ["repeat"],
        "repeat metrics",
    )
    _assert_frame_equal(
        pd.read_csv(root / "fold_metrics.csv"),
        recomputed.paired_metrics.fold_metrics,
        ["repeat", "outer_fold"],
        "fold metrics",
    )
    _assert_json_equal(
        _read_json(root / "aggregate_summary.json"),
        recomputed.paired_metrics.aggregate_summary,
        "aggregate summary",
    )
    _assert_json_equal(
        _read_json(root / "prediction_comparison.json"),
        recomputed.prediction_comparison,
        "prediction comparison",
    )
    _assert_frame_equal(
        pd.read_csv(root / "blend_repeat_metrics.csv"),
        recomputed.blend.repeat_metrics,
        ["repeat"],
        "blend repeat metrics",
    )
    _assert_frame_equal(
        pd.read_csv(root / "blend_fold_metrics.csv"),
        recomputed.blend.fold_metrics,
        ["repeat", "outer_fold"],
        "blend fold metrics",
    )
    _assert_json_equal(
        _read_json(root / "blend_summary.json"),
        recomputed.blend.summary,
        "blend summary",
    )
    _assert_json_equal(
        _read_json(root / "decision_report.json"),
        recomputed.decision_report,
        "decision report",
    )
    _assert_json_equal(
        _read_json(root / "artifact_inventory.json"),
        build_comparison_inventory(root),
        "artifact inventory",
    )
    if verify_manifest:
        manifest = _read_json(root / "manifest.json")
        _assert_json_equal(
            manifest,
            build_comparison_manifest(root),
            "comparison manifest",
        )
        if require_success:
            success_payload = _read_json(success)
            expected_success = {
                "schema_version": COMPARISON_SCHEMA_VERSION,
                "manifest_sha256": manifest["manifest_sha256"],
            }
            _assert_json_equal(success_payload, expected_success, "_SUCCESS")


def build_comparison_inventory(root: Path) -> dict[str, Any]:
    files = _file_records(
        root,
        excluded={"artifact_inventory.json", "manifest.json", "_SUCCESS", "_FAILED"},
    )
    canonical = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "scope": "comparison_primary_artifacts",
        "files": files,
    }
    return {**canonical, "inventory_sha256": canonical_sha256(canonical)}


def build_comparison_manifest(root: Path) -> dict[str, Any]:
    files = _file_records(root, excluded={"manifest.json", "_SUCCESS", "_FAILED"})
    canonical = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "hashing_method": "sha256_raw_artifact_bytes",
        "files": files,
    }
    return {**canonical, "manifest_sha256": canonical_sha256(canonical)}


def _file_records(root: Path, *, excluded: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name in excluded:
            continue
        raw = path.read_bytes()
        records.append(
            {
                "path": path.name,
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return records


def _validate_resolved_config(
    payload: Mapping[str, Any],
    comparison_root: Path,
    project_root: Path,
) -> None:
    expected = {
        "schema_version",
        "comparison_id",
        "comparison_identity_sha256",
        "created_at_utc",
        "project_contract",
        "baseline_run",
        "candidate_run",
        "comparison_policy",
        "source_provenance",
        "prediction_alignment_keys",
        "fixed_blend",
        "tracking_enabled",
        "competition_assets_accessed",
    }
    if set(payload) != expected:
        raise PairedComparisonArtifactError("Resolved comparison config keys differ.")
    if payload["schema_version"] != COMPARISON_SCHEMA_VERSION:
        raise PairedComparisonArtifactError("Comparison schema version differs.")
    if payload["comparison_id"] != comparison_root.name:
        raise PairedComparisonArtifactError("Comparison ID differs from directory.")
    if (
        not isinstance(payload["comparison_id"], str)
        or SAFE_SLUG.fullmatch(payload["comparison_id"]) is None
    ):
        raise PairedComparisonArtifactError("Comparison ID is not a safe slug.")
    if payload["project_contract"] != "experiment_core_v2_paired_comparison":
        raise PairedComparisonArtifactError("Comparison project contract differs.")
    expected_alignment = {
        "outer_validation": [
            "repeat",
            "repeat_seed",
            "outer_fold",
            "row_position",
        ],
        "threshold_selection": [
            "repeat",
            "repeat_seed",
            "outer_fold",
            "threshold_selection_fold",
            "row_position",
        ],
    }
    _assert_json_equal(
        payload["prediction_alignment_keys"],
        expected_alignment,
        "prediction alignment keys",
    )
    expected_blend = {
        "baseline_weight": 0.5,
        "candidate_weight": 0.5,
        "weight_search": False,
        "threshold_source": "threshold_selection_oof",
    }
    _assert_json_equal(payload["fixed_blend"], expected_blend, "fixed blend")
    created = payload["created_at_utc"]
    if not isinstance(created, str) or not created.endswith("+00:00"):
        raise PairedComparisonArtifactError("Comparison timestamp is not UTC.")
    if payload["tracking_enabled"] is not False:
        raise PairedComparisonArtifactError("Tracking must remain disabled.")
    if payload["competition_assets_accessed"] is not False:
        raise PairedComparisonArtifactError("Competition assets must not be accessed.")
    provenance = payload["source_provenance"]
    _assert_json_equal(
        provenance,
        comparison_source_provenance(project_root),
        "source provenance",
    )
    baseline_reference = payload["baseline_run"]
    candidate_reference = payload["candidate_run"]
    if not isinstance(baseline_reference, Mapping) or not isinstance(
        candidate_reference, Mapping
    ):
        raise PairedComparisonArtifactError("Run references must be mappings.")
    policy = _policy_from_resolved(payload["comparison_policy"])
    baseline = load_completed_research_v2_run(
        project_root / str(baseline_reference["repository_relative_path"]),
        project_root=project_root,
    )
    candidate = load_completed_research_v2_run(
        project_root / str(candidate_reference["repository_relative_path"]),
        project_root=project_root,
    )
    _assert_json_equal(
        dict(baseline_reference),
        baseline.reference("baseline").to_dict(),
        "resolved baseline reference",
    )
    _assert_json_equal(
        dict(candidate_reference),
        candidate.reference("candidate").to_dict(),
        "resolved candidate reference",
    )
    expected_identity = deterministic_comparison_identity(
        baseline.reference("baseline"),
        candidate.reference("candidate"),
        policy,
    )
    if payload["comparison_identity_sha256"] != expected_identity:
        raise PairedComparisonArtifactError("Comparison identity differs.")


def _policy_from_resolved(payload: Any) -> ComparisonPolicy:
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "policy",
    }:
        raise PairedComparisonArtifactError("Resolved comparison policy is malformed.")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise PairedComparisonArtifactError("Resolved policy schema version differs.")
    values = payload["policy"]
    if not isinstance(values, Mapping):
        raise PairedComparisonArtifactError("Resolved policy values are malformed.")
    expected_keys = {
        "id",
        "tie_epsilon",
        "promising_min_mean_balanced_accuracy_delta",
        "promising_min_repeat_win_fraction",
        "threshold_stability_max_sample_standard_deviation",
    }
    if set(values) != expected_keys:
        raise PairedComparisonArtifactError("Resolved policy keys differ.")
    if not isinstance(values["id"], str) or not values["id"]:
        raise PairedComparisonArtifactError("Resolved policy ID is invalid.")
    float_keys = (
        "tie_epsilon",
        "promising_min_mean_balanced_accuracy_delta",
        "promising_min_repeat_win_fraction",
        "threshold_stability_max_sample_standard_deviation",
    )
    for key in float_keys:
        value = values.get(key)
        if type(value) is not float or not math.isfinite(value):
            raise PairedComparisonArtifactError(
                f"Resolved comparison policy {key} must be an exact finite float."
            )
    if values["tie_epsilon"] < 0.0:
        raise PairedComparisonArtifactError("Resolved tie epsilon is negative.")
    if not 0.0 <= values["promising_min_repeat_win_fraction"] <= 1.0:
        raise PairedComparisonArtifactError(
            "Resolved repeat win fraction is outside [0, 1]."
        )
    if values["threshold_stability_max_sample_standard_deviation"] < 0.0:
        raise PairedComparisonArtifactError(
            "Resolved threshold stability limit is negative."
        )
    return ComparisonPolicy(
        schema_version=payload["schema_version"],
        policy_id=values["id"],
        tie_epsilon=values["tie_epsilon"],
        promising_min_mean_balanced_accuracy_delta=values[
            "promising_min_mean_balanced_accuracy_delta"
        ],
        promising_min_repeat_win_fraction=values["promising_min_repeat_win_fraction"],
        threshold_stability_max_sample_standard_deviation=values[
            "threshold_stability_max_sample_standard_deviation"
        ],
    )


def _repository_path(path: Path, project_root: Path, label: str) -> Path:
    if not path.is_absolute():
        path = project_root / path
    resolved = path.resolve()
    if resolved == project_root or project_root not in resolved.parents:
        raise PairedComparisonArtifactError(f"{label} must be inside the repository.")
    normalized = resolved.relative_to(project_root).as_posix().lower()
    forbidden = (
        "data/raw/",
        "data/test/",
        "submissions/",
        "kaggle",
        "autogluon",
        "sample_submission",
        "competition_test",
        "competition-test",
        "x_test",
        "final-submission",
        "final_submission",
        "final-model",
        "final_model",
        "model-artifact",
        "model_artifact",
        "models/",
        "/models/",
    )
    if any(marker in normalized for marker in forbidden):
        raise PairedComparisonArtifactError(f"{label} is a forbidden asset path.")
    return resolved


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, default=_json_default)
        file.write("\n")
    temporary.replace(path)


def _atomic_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(dict(payload), file, sort_keys=False, allow_unicode=True)
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _best_effort_failure_marker(root: Path, error: BaseException) -> None:
    try:
        _atomic_json(
            root / "_FAILED",
            {
                "schema_version": COMPARISON_SCHEMA_VERSION,
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
    except BaseException:
        pass


def _assert_frame_equal(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    sort_by: list[str],
    label: str,
) -> None:
    try:
        pd.testing.assert_frame_equal(
            actual.sort_values(sort_by, ignore_index=True),
            expected.sort_values(sort_by, ignore_index=True),
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-12,
        )
    except AssertionError as error:
        raise PairedComparisonArtifactError(f"{label} differs: {error}") from error


def _assert_json_equal(actual: Any, expected: Any, label: str) -> None:
    if canonical_sha256(actual) != canonical_sha256(expected):
        raise PairedComparisonArtifactError(f"{label} differs.")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PairedComparisonArtifactError(f"Cannot read JSON: {path}.") from error
    if not isinstance(value, dict):
        raise PairedComparisonArtifactError(f"JSON must be a mapping: {path}.")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise PairedComparisonArtifactError(f"Cannot read YAML: {path}.") from error
    if not isinstance(value, dict):
        raise PairedComparisonArtifactError(f"YAML must be a mapping: {path}.")
    _reject_absolute_strings(value, "resolved_comparison_config")
    return value


def _reject_absolute_strings(value: Any, field_path: str) -> None:
    if isinstance(value, str):
        if (
            PurePosixPath(value).is_absolute()
            or PureWindowsPath(value).is_absolute()
            or bool(PureWindowsPath(value).drive)
        ):
            raise PairedComparisonArtifactError(
                f"Non-portable absolute path at {field_path}."
            )
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _reject_absolute_strings(child, f"{field_path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_absolute_strings(child, f"{field_path}[{index}]")
