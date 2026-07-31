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

from src.churn_ml.dataset_comparison_v1 import (
    COMPARISON_OPERATION,
    COMPARISON_SCHEMA_VERSION,
    DEFAULT_TIE_EPSILON,
    PROJECT_CONTRACT,
    DatasetComparisonResult,
    build_comparison_result,
    deterministic_comparison_identity,
    load_completed_research_v2_run,
)
from src.churn_ml.paired_comparison_paths import (
    PairedPathSafetyError,
    assert_pairwise_disjoint_paths,
    prewalk_regular_tree,
    resolve_repository_path,
)
from src.churn_ml.research_data import canonical_sha256

SAFE_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
COMPARISON_SOURCE_PATHS = (
    "scripts/compare_research_v2_datasets.py",
    "src/churn_ml/dataset_comparison_v1.py",
    "src/churn_ml/dataset_comparison_v1_artifacts.py",
    "src/churn_ml/dataset_comparison_v1_cli.py",
    "src/churn_ml/paired_comparison_paths.py",
    "src/churn_ml/research_v2_resolved_config.py",
)
PRIMARY_ARTIFACTS = {
    "resolved_comparison_config.yaml",
    "compatibility_report.json",
    "normalized_identities.json",
    "baseline_run_reference.json",
    "candidate_run_reference.json",
    "summary.json",
    "repeat_metrics.csv",
    "fold_metrics.csv",
    "oof_diagnostics.json",
}
PROVENANCE_ARTIFACT = "provenance/source_provenance.json"
ARTIFACT_DIRECTORIES = {"provenance"}


class DatasetComparisonArtifactError(RuntimeError):
    """Raised when dataset comparison artifacts cannot satisfy their lifecycle."""


def comparison_source_provenance(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    records: list[dict[str, Any]] = []
    for relative in COMPARISON_SOURCE_PATHS:
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise DatasetComparisonArtifactError(
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
        "operation": COMPARISON_OPERATION,
        "hashing_method": "sha256_file_bytes_repository_relative_paths",
        "files": records,
    }
    return {**canonical, "sha256": canonical_sha256(canonical)}


def create_comparison_artifacts(
    *,
    result: DatasetComparisonResult,
    project_root: Path,
    output_root: Path,
    comparison_id: str | None,
    tie_epsilon: float = DEFAULT_TIE_EPSILON,
    before_success: Callable[[], None] | None = None,
) -> Path:
    root = project_root.resolve()
    try:
        output = resolve_repository_path(
            output_root,
            project_root=root,
            role="comparison_output",
            field_path="paths.output_root",
            must_exist=False,
            require_directory=True,
        )
        baseline_root = resolve_repository_path(
            Path(result.baseline_reference.repository_relative_path),
            project_root=root,
            role="baseline_run",
            field_path="paths.baseline_run",
            must_exist=True,
            require_directory=True,
        )
        candidate_root = resolve_repository_path(
            Path(result.candidate_reference.repository_relative_path),
            project_root=root,
            role="candidate_run",
            field_path="paths.candidate_run",
            must_exist=True,
            require_directory=True,
        )
        assert_pairwise_disjoint_paths(
            {
                "baseline_run": baseline_root,
                "candidate_run": candidate_root,
                "output_root": output,
            }
        )
    except PairedPathSafetyError as error:
        raise DatasetComparisonArtifactError(str(error)) from error
    identity = deterministic_comparison_identity(
        result.baseline_reference,
        result.candidate_reference,
    )
    resolved_id = comparison_id or f"dataset_{identity[:20]}"
    if SAFE_SLUG.fullmatch(resolved_id) is None:
        raise DatasetComparisonArtifactError(
            "comparison_id must use the existing safe-slug format and 128-char limit."
        )
    comparison_root = (output / resolved_id).resolve()
    if comparison_root.parent != output:
        raise DatasetComparisonArtifactError(
            "Comparison directory escapes output root."
        )
    try:
        assert_pairwise_disjoint_paths(
            {"baseline_run": baseline_root, "comparison_root": comparison_root}
        )
        assert_pairwise_disjoint_paths(
            {"candidate_run": candidate_root, "comparison_root": comparison_root}
        )
    except PairedPathSafetyError as error:
        raise DatasetComparisonArtifactError(str(error)) from error
    if comparison_root.exists():
        raise DatasetComparisonArtifactError(
            f"Comparison directory already exists: {comparison_root}."
        )
    comparison_root.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    try:
        provenance = comparison_source_provenance(root)
        (comparison_root / "provenance").mkdir()
        _atomic_json(comparison_root / PROVENANCE_ARTIFACT, provenance)
        normalized = result.compatibility.normalized_identities
        if normalized is None:
            raise DatasetComparisonArtifactError(
                "Normalized identities are required to persist comparison artifacts."
            )
        resolved = {
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "operation": COMPARISON_OPERATION,
            "comparison_id": resolved_id,
            "comparison_identity_sha256": identity,
            "created_at_utc": started.isoformat(),
            "project_contract": PROJECT_CONTRACT,
            "baseline_run": result.baseline_reference.to_dict(),
            "candidate_run": result.candidate_reference.to_dict(),
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
            "tie_epsilon": tie_epsilon,
            "tracking_enabled": False,
            "competition_assets_accessed": False,
        }
        _atomic_yaml(comparison_root / "resolved_comparison_config.yaml", resolved)
        _atomic_json(
            comparison_root / "compatibility_report.json",
            result.compatibility.to_dict(),
        )
        _atomic_json(
            comparison_root / "normalized_identities.json",
            normalized.to_dict(),
        )
        _atomic_json(
            comparison_root / "baseline_run_reference.json",
            result.baseline_reference.to_dict(),
        )
        _atomic_json(
            comparison_root / "candidate_run_reference.json",
            result.candidate_reference.to_dict(),
        )
        summary = {
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "operation": COMPARISON_OPERATION,
            "aggregate_summary": result.paired_metrics.aggregate_summary,
            "fit_time_delta": result.fit_time_delta,
        }
        _atomic_json(comparison_root / "summary.json", summary)
        _atomic_csv(
            comparison_root / "repeat_metrics.csv",
            result.paired_metrics.repeat_metrics,
        )
        _atomic_csv(
            comparison_root / "fold_metrics.csv",
            result.paired_metrics.fold_metrics,
        )
        _atomic_json(comparison_root / "oof_diagnostics.json", result.oof_diagnostics)
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
                "operation": COMPARISON_OPERATION,
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
    try:
        root = resolve_repository_path(
            comparison_root,
            project_root=project_root,
            role="comparison_artifact",
            field_path="paths.comparison_root",
            must_exist=True,
            require_directory=True,
        )
        snapshot = prewalk_regular_tree(
            root,
            project_root=project_root,
            role="comparison_artifact",
            field_path="artifacts",
            reject_hardlinks=True,
            reject_forbidden_descendants=True,
        )
    except PairedPathSafetyError as error:
        raise DatasetComparisonArtifactError(str(error)) from error
    success = root / "_SUCCESS"
    failed = root / "_FAILED"
    if failed.exists():
        raise DatasetComparisonArtifactError("Comparison contains _FAILED.")
    if require_success and not success.is_file():
        raise DatasetComparisonArtifactError("Comparison does not contain _SUCCESS.")
    if not require_success and success.exists():
        raise DatasetComparisonArtifactError(
            "_SUCCESS exists before comparison semantic validation."
        )
    expected_files = PRIMARY_ARTIFACTS | {
        "artifact_inventory.json",
        "manifest.json",
        PROVENANCE_ARTIFACT,
    }
    if require_success:
        expected_files.add("_SUCCESS")
    actual_files = {path.as_posix() for path in snapshot.files}
    actual_directories = {path.as_posix() for path in snapshot.directories}
    if actual_files != expected_files or actual_directories != ARTIFACT_DIRECTORIES:
        raise DatasetComparisonArtifactError(
            "Comparison artifact inventory differs; "
            f"missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)}, "
            f"missing_directories={sorted(ARTIFACT_DIRECTORIES - actual_directories)}, "
            f"unexpected_directories={sorted(actual_directories - ARTIFACT_DIRECTORIES)}."
        )
    resolved = _read_yaml(root / "resolved_comparison_config.yaml")
    _assert_json_equal(
        _read_json(root / PROVENANCE_ARTIFACT),
        resolved["source_provenance"],
        "nested source provenance",
    )
    _validate_resolved_config(resolved, root, project_root)
    baseline_reference = _read_json(root / "baseline_run_reference.json")
    candidate_reference = _read_json(root / "candidate_run_reference.json")
    if baseline_reference != resolved["baseline_run"]:
        raise DatasetComparisonArtifactError("Baseline reference differs.")
    if candidate_reference != resolved["candidate_run"]:
        raise DatasetComparisonArtifactError("Candidate reference differs.")
    baseline = load_completed_research_v2_run(
        project_root / baseline_reference["repository_relative_path"],
        project_root=project_root,
        role="baseline_run",
    )
    candidate = load_completed_research_v2_run(
        project_root / candidate_reference["repository_relative_path"],
        project_root=project_root,
        role="candidate_run",
    )
    tie_epsilon = resolved["tie_epsilon"]
    if type(tie_epsilon) is not float or not math.isfinite(tie_epsilon):
        raise DatasetComparisonArtifactError("Resolved tie epsilon is invalid.")
    recomputed = build_comparison_result(
        baseline,
        candidate,
        project_root=project_root,
        tie_epsilon=tie_epsilon,
    )
    _assert_json_equal(
        _read_json(root / "compatibility_report.json"),
        recomputed.compatibility.to_dict(),
        "compatibility report",
    )
    _assert_json_equal(
        _read_json(root / "normalized_identities.json"),
        recomputed.compatibility.normalized_identities.to_dict()
        if recomputed.compatibility.normalized_identities is not None
        else {},
        "normalized identities",
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
    stored_summary = _read_json(root / "summary.json")
    expected_summary = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "operation": COMPARISON_OPERATION,
        "aggregate_summary": recomputed.paired_metrics.aggregate_summary,
        "fit_time_delta": recomputed.fit_time_delta,
    }
    _assert_json_equal(stored_summary, expected_summary, "summary")
    _assert_json_equal(
        _read_json(root / "oof_diagnostics.json"),
        recomputed.oof_diagnostics,
        "oof diagnostics",
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
                "operation": COMPARISON_OPERATION,
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
        "operation": COMPARISON_OPERATION,
        "scope": "dataset_comparison_primary_artifacts",
        "files": files,
    }
    return {**canonical, "inventory_sha256": canonical_sha256(canonical)}


def build_comparison_manifest(root: Path) -> dict[str, Any]:
    files = _file_records(root, excluded={"manifest.json", "_SUCCESS", "_FAILED"})
    canonical = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "operation": COMPARISON_OPERATION,
        "hashing_method": "sha256_raw_artifact_bytes",
        "files": files,
    }
    return {**canonical, "manifest_sha256": canonical_sha256(canonical)}


def _file_records(root: Path, *, excluded: set[str]) -> list[dict[str, Any]]:
    try:
        snapshot = prewalk_regular_tree(
            root,
            project_root=root.parent,
            role="comparison_artifact",
            field_path="artifacts",
            reject_hardlinks=True,
            reject_forbidden_descendants=True,
        )
    except PairedPathSafetyError as error:
        raise DatasetComparisonArtifactError(str(error)) from error
    records: list[dict[str, Any]] = []
    for relative in snapshot.files:
        relative_name = relative.as_posix()
        if relative_name in excluded:
            continue
        raw = (root / relative).read_bytes()
        records.append(
            {
                "path": relative_name,
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
        "operation",
        "comparison_id",
        "comparison_identity_sha256",
        "created_at_utc",
        "project_contract",
        "baseline_run",
        "candidate_run",
        "source_provenance",
        "prediction_alignment_keys",
        "tie_epsilon",
        "tracking_enabled",
        "competition_assets_accessed",
    }
    if set(payload) != expected:
        raise DatasetComparisonArtifactError("Resolved comparison config keys differ.")
    if payload["schema_version"] != COMPARISON_SCHEMA_VERSION:
        raise DatasetComparisonArtifactError("Comparison schema version differs.")
    if payload["operation"] != COMPARISON_OPERATION:
        raise DatasetComparisonArtifactError("Comparison operation differs.")
    if payload["comparison_id"] != comparison_root.name:
        raise DatasetComparisonArtifactError("Comparison ID differs from directory.")
    if (
        not isinstance(payload["comparison_id"], str)
        or SAFE_SLUG.fullmatch(payload["comparison_id"]) is None
    ):
        raise DatasetComparisonArtifactError("Comparison ID is not a safe slug.")
    if payload["project_contract"] != PROJECT_CONTRACT:
        raise DatasetComparisonArtifactError("Comparison project contract differs.")
    created = payload["created_at_utc"]
    if not isinstance(created, str) or not created.endswith("+00:00"):
        raise DatasetComparisonArtifactError("Comparison timestamp is not UTC.")
    if payload["tracking_enabled"] is not False:
        raise DatasetComparisonArtifactError("Tracking must remain disabled.")
    if payload["competition_assets_accessed"] is not False:
        raise DatasetComparisonArtifactError("Competition assets must not be accessed.")
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
        raise DatasetComparisonArtifactError("Run references must be mappings.")
    baseline = load_completed_research_v2_run(
        project_root / str(baseline_reference["repository_relative_path"]),
        project_root=project_root,
        role="baseline_run",
    )
    candidate = load_completed_research_v2_run(
        project_root / str(candidate_reference["repository_relative_path"]),
        project_root=project_root,
        role="candidate_run",
    )
    recomputed = build_comparison_result(
        baseline,
        candidate,
        project_root=project_root,
        tie_epsilon=payload["tie_epsilon"],
    )
    expected_identity = deterministic_comparison_identity(
        recomputed.baseline_reference,
        recomputed.candidate_reference,
    )
    if payload["comparison_identity_sha256"] != expected_identity:
        raise DatasetComparisonArtifactError("Comparison identity differs.")
    _assert_json_equal(
        dict(baseline_reference),
        recomputed.baseline_reference.to_dict(),
        "resolved baseline reference",
    )
    _assert_json_equal(
        dict(candidate_reference),
        recomputed.candidate_reference.to_dict(),
        "resolved candidate reference",
    )


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
                "operation": COMPARISON_OPERATION,
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
        raise DatasetComparisonArtifactError(f"{label} differs: {error}") from error


def _assert_json_equal(actual: Any, expected: Any, label: str) -> None:
    if canonical_sha256(actual) != canonical_sha256(expected):
        raise DatasetComparisonArtifactError(f"{label} differs.")


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
        raise DatasetComparisonArtifactError(f"Cannot read JSON: {path}.") from error
    if not isinstance(value, dict):
        raise DatasetComparisonArtifactError(f"JSON must be a mapping: {path}.")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise DatasetComparisonArtifactError(f"Cannot read YAML: {path}.") from error
    if not isinstance(value, dict):
        raise DatasetComparisonArtifactError(f"YAML must be a mapping: {path}.")
    _reject_absolute_strings(value, "resolved_comparison_config")
    return value


def _reject_absolute_strings(value: Any, field_path: str) -> None:
    if isinstance(value, str):
        if (
            PurePosixPath(value).is_absolute()
            or PureWindowsPath(value).is_absolute()
            or bool(PureWindowsPath(value).drive)
        ):
            raise DatasetComparisonArtifactError(
                f"Non-portable absolute path at {field_path}."
            )
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _reject_absolute_strings(child, f"{field_path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_absolute_strings(child, f"{field_path}[{index}]")
