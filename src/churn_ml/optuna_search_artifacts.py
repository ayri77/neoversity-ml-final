from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from src.churn_ml.optuna_search_authority import (
    LifecycleAuthorityRecorder,
    load_lifecycle_authority_key,
    metric_prediction_evidence_identity,
    validate_report_lifecycle_authority,
)
from src.churn_ml.optuna_search_config import PLAN_KEYS, OptunaSearchConfig
from src.churn_ml.research_data import canonical_sha256
from src.churn_ml.research_v2_config import load_research_v2_config


class OptunaSearchArtifactError(RuntimeError):
    """Raised when an immutable search report is incomplete or corrupted."""


PAYLOAD_FILES = {
    "resolved_search_config.yaml",
    "search_identity.json",
    "dataset_identity.json",
    "assignment_identity.json",
    "search_space.json",
    "study_summary.json",
    "trials.csv",
    "trial_metrics.csv",
    "trial_predictions.csv",
    "best_trial.json",
    "best_candidate_config.yaml",
    "runtime.json",
    "environment.json",
    "source_provenance.json",
    "resume_authentication.json",
    "fold_assignments.csv",
    "threshold_selection_membership.csv",
    "lifecycle_authority.json",
}
TERMINAL_FILES = PAYLOAD_FILES | {
    "recursive_inventory.json",
    "manifest.json",
    "lifecycle_authority_ledger.json",
    "_SUCCESS",
}


@dataclass(frozen=True)
class CompletedSearchResult:
    search_dir: Path
    search_identity: dict[str, Any]
    study_summary: dict[str, Any]
    best_trial: dict[str, Any]
    best_candidate_config: dict[str, Any]


def write_completed_search(
    config: OptunaSearchConfig,
    *,
    dataset_identity: Mapping[str, Any],
    assignment_identity: Mapping[str, Any],
    search_space_payload: Mapping[str, Any],
    study_summary: Mapping[str, Any],
    trials: pd.DataFrame,
    trial_metrics: pd.DataFrame,
    trial_predictions: pd.DataFrame,
    best_trial: Mapping[str, Any],
    best_candidate_config: Mapping[str, Any],
    runtime: Mapping[str, Any],
    environment: Mapping[str, Any],
    source_provenance: Mapping[str, Any],
    resume_authentication: Mapping[str, Any],
    fold_assignments: pd.DataFrame,
    threshold_membership: pd.DataFrame,
    lifecycle_authority: LifecycleAuthorityRecorder,
) -> Path:
    artifact_root = config.artifact_root
    target = (artifact_root / config.search_id).resolve()
    if artifact_root != target.parent:
        raise OptunaSearchArtifactError("Search directory escapes artifact root.")
    if target.exists():
        raise OptunaSearchArtifactError(
            f"Immutable search output already exists: {target}."
        )
    artifact_root.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{config.search_id}.", dir=artifact_root)
    ).resolve()
    try:
        authority_initial = lifecycle_authority.initial_statement["payload"]
        _write_yaml(staging / "resolved_search_config.yaml", config.resolved_payload())
        _write_json(
            staging / "search_identity.json",
            {
                "schema_version": 1,
                "authority_schema_version": authority_initial[
                    "authority_schema_version"
                ],
                "authority_key_fingerprint": authority_initial[
                    "authority_key_fingerprint"
                ],
                "authority_bound_study_identity_sha256": authority_initial[
                    "authority_bound_study_identity_sha256"
                ],
                "study_uuid": authority_initial["study_uuid"],
                "search_id": config.search_id,
                "sha256": config.search_identity_sha256,
                "canonical": deepcopy(config.search_identity),
                "study_identity_sha256": config.study_identity_sha256,
                "dataset_identity_sha256": canonical_sha256(dataset_identity),
                "assignment_identity_sha256": canonical_sha256(assignment_identity),
                "prediction_evidence_identity_sha256": study_summary[
                    "prediction_evidence_identity_sha256"
                ],
                "failure_evidence_identity_sha256": study_summary[
                    "failure_evidence_identity_sha256"
                ],
                "resume_authentication_sha256": resume_authentication[
                    "identity_sha256"
                ],
            },
        )
        _write_json(staging / "dataset_identity.json", dict(dataset_identity))
        _write_json(staging / "assignment_identity.json", dict(assignment_identity))
        _write_json(staging / "search_space.json", dict(search_space_payload))
        _write_json(staging / "study_summary.json", dict(study_summary))
        _write_csv(staging / "trials.csv", trials)
        _write_csv(staging / "trial_metrics.csv", trial_metrics)
        _write_csv(staging / "trial_predictions.csv", trial_predictions)
        _write_json(staging / "best_trial.json", dict(best_trial))
        _write_yaml(
            staging / "best_candidate_config.yaml",
            dict(best_candidate_config),
        )
        _write_json(staging / "runtime.json", dict(runtime))
        _write_json(staging / "environment.json", dict(environment))
        _write_json(
            staging / "source_provenance.json",
            dict(source_provenance),
        )
        _write_json(
            staging / "resume_authentication.json",
            dict(resume_authentication),
        )
        _write_csv(staging / "fold_assignments.csv", fold_assignments)
        _write_csv(
            staging / "threshold_selection_membership.csv",
            threshold_membership,
        )
        _write_json(
            staging / "lifecycle_authority.json",
            lifecycle_authority.report_payload(),
        )
        load_research_v2_config(
            staging / "best_candidate_config.yaml",
            project_root=config.project_root,
        )
        inventory = _build_inventory(staging, PAYLOAD_FILES)
        _write_json(staging / "recursive_inventory.json", inventory)
        manifest_files = _inventory_records(
            staging,
            PAYLOAD_FILES | {"recursive_inventory.json"},
        )
        manifest_body = {"schema_version": 1, "files": manifest_files}
        manifest = {
            **manifest_body,
            "manifest_sha256": canonical_sha256(manifest_body),
        }
        _write_json(staging / "manifest.json", manifest)
        _validate_tree(staging, require_success=False)
        trial_state_universe = [
            {
                "trial_number": int(row.trial_number),
                "state": str(row.state),
            }
            for row in trials.sort_values("trial_number").itertuples(index=False)
        ]
        authority_ledger = lifecycle_authority.finalize(
            trial_state_universe=trial_state_universe,
            interrupted_recovery_trial_numbers=study_summary[
                "interrupted_recovery_trial_numbers"
            ],
            best_trial_number=int(study_summary["best_trial_number"]),
            report_search_identity_sha256=config.search_identity_sha256,
            study_summary_identity_sha256=canonical_sha256(dict(study_summary)),
            metric_prediction_evidence_identity_sha256=(
                metric_prediction_evidence_identity(
                    trial_metrics_bytes=(staging / "trial_metrics.csv").read_bytes(),
                    trial_predictions_bytes=(
                        staging / "trial_predictions.csv"
                    ).read_bytes(),
                )
            ),
            report_manifest_identity_sha256=str(manifest["manifest_sha256"]),
        )
        _write_json(
            staging / "lifecycle_authority_ledger.json",
            authority_ledger,
        )
        final_ledger = authority_ledger["final_ledger"]
        _write_json(
            staging / "_SUCCESS",
            {
                "schema_version": 1,
                "authority_schema_version": authority_initial[
                    "authority_schema_version"
                ],
                "search_id": config.search_id,
                "manifest_sha256": manifest["manifest_sha256"],
                "final_lifecycle_payload_identity_sha256": final_ledger[
                    "payload_identity_sha256"
                ],
                "final_lifecycle_signature_sha256": final_ledger["signature_sha256"],
            },
        )
        _validate_tree(staging, require_success=True)
        staging.rename(target)
        return target
    except BaseException as error:
        try:
            _write_json(
                staging / "_FAILED",
                {
                    "schema_version": 1,
                    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "reason_code": "ARTIFACT_FINALIZATION_FAILED",
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
        except BaseException:
            pass
        raise


def write_failed_attempt(
    config: OptunaSearchConfig,
    error: BaseException,
    *,
    reason_code: str,
) -> Path:
    root = config.artifact_root / "_failed_attempts"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = root / f"{config.search_id}_{stamp}"
    target.mkdir(exist_ok=False)
    _write_yaml(target / "resolved_search_config.yaml", config.resolved_payload())
    _write_json(
        target / "_FAILED",
        {
            "schema_version": 1,
            "search_id": config.search_id,
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "reason_code": reason_code,
            "error_type": type(error).__name__,
            "message": str(error),
        },
    )
    return target


def load_optuna_search_result(
    path: Path,
    *,
    project_root: Path,
) -> CompletedSearchResult:
    root = project_root.resolve()
    unresolved = path if path.is_absolute() else root / path
    search_dir = _contained_real_directory(unresolved, root)
    _validate_tree(search_dir, require_success=True)
    loaded_identity = _load_json(search_dir / "search_identity.json")
    if loaded_identity.get("search_id") != search_dir.name:
        raise OptunaSearchArtifactError(
            "Search directory name differs from authenticated search identity."
        )
    return CompletedSearchResult(
        search_dir=search_dir,
        search_identity=_load_json(search_dir / "search_identity.json"),
        study_summary=_load_json(search_dir / "study_summary.json"),
        best_trial=_load_json(search_dir / "best_trial.json"),
        best_candidate_config=_load_yaml(search_dir / "best_candidate_config.yaml"),
    )


def _validate_tree(root: Path, *, require_success: bool) -> None:
    root_metadata = root.lstat()
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_ISLNK(root_metadata.st_mode)
        or _is_reparse_stat(root_metadata)
    ):
        raise OptunaSearchArtifactError("Search report must be a real directory.")
    actual_files: set[str] = set()
    for item in root.iterdir():
        metadata = item.lstat()
        if stat.S_ISLNK(metadata.st_mode) or _is_reparse_stat(metadata):
            raise OptunaSearchArtifactError(
                "Linked/reparse search artifacts are forbidden."
            )
        if stat.S_ISDIR(metadata.st_mode):
            raise OptunaSearchArtifactError("Unexpected artifact directory.")
        if not stat.S_ISREG(metadata.st_mode):
            raise OptunaSearchArtifactError("Unexpected non-file artifact.")
        if getattr(metadata, "st_nlink", 1) > 1:
            raise OptunaSearchArtifactError(
                "Multiply linked search artifacts are forbidden."
            )
        actual_files.add(item.name)
    expected = (
        TERMINAL_FILES
        if require_success
        else TERMINAL_FILES - {"_SUCCESS", "lifecycle_authority_ledger.json"}
    )
    if actual_files != expected:
        raise OptunaSearchArtifactError(
            f"Artifact files differ; missing={sorted(expected - actual_files)}, "
            f"unknown={sorted(actual_files - expected)}."
        )
    inventory = _load_json(root / "recursive_inventory.json")
    if (
        set(inventory) != {"schema_version", "files"}
        or inventory["schema_version"] != 1
    ):
        raise OptunaSearchArtifactError("Recursive inventory schema differs.")
    if inventory["files"] != _inventory_records(root, PAYLOAD_FILES):
        raise OptunaSearchArtifactError("Recursive inventory does not match bytes.")
    manifest = _load_json(root / "manifest.json")
    if set(manifest) != {"schema_version", "files", "manifest_sha256"}:
        raise OptunaSearchArtifactError("Manifest schema differs.")
    expected_manifest_files = _inventory_records(
        root,
        PAYLOAD_FILES | {"recursive_inventory.json"},
    )
    manifest_body = {"schema_version": 1, "files": expected_manifest_files}
    if (
        manifest["schema_version"] != 1
        or manifest["files"] != expected_manifest_files
        or manifest["manifest_sha256"] != canonical_sha256(manifest_body)
    ):
        raise OptunaSearchArtifactError("Manifest does not match artifact bytes.")
    _validate_semantics(root)
    if require_success:
        success = _load_json(root / "_SUCCESS")
        if set(success) != {
            "schema_version",
            "authority_schema_version",
            "search_id",
            "manifest_sha256",
            "final_lifecycle_payload_identity_sha256",
            "final_lifecycle_signature_sha256",
        }:
            raise OptunaSearchArtifactError("Success marker schema differs.")
        identity = _load_json(root / "search_identity.json")
        if (
            success["schema_version"] != 1
            or success["authority_schema_version"] != 1
            or success["search_id"] != identity["search_id"]
            or success["manifest_sha256"] != manifest["manifest_sha256"]
        ):
            raise OptunaSearchArtifactError("Success marker identity differs.")
        _validate_lifecycle_authority(root, manifest=manifest, success=success)
        success_time = (root / "_SUCCESS").stat().st_mtime_ns
        if any(
            item.stat().st_mtime_ns > success_time
            for item in root.iterdir()
            if item.name != "_SUCCESS"
        ):
            raise OptunaSearchArtifactError("_SUCCESS must be the newest artifact.")


def _validate_lifecycle_authority(
    root: Path,
    *,
    manifest: Mapping[str, Any],
    success: Mapping[str, Any],
) -> None:
    project_root = _project_root_for_artifact(root)
    study_summary = _load_json(root / "study_summary.json")
    search_identity = _load_json(root / "search_identity.json")
    authority_report = _load_json(root / "lifecycle_authority.json")
    authority_ledger = _load_json(root / "lifecycle_authority_ledger.json")
    final_ledger = authority_ledger.get("final_ledger")
    if not isinstance(final_ledger, dict):
        raise OptunaSearchArtifactError("Final lifecycle ledger is malformed.")
    if success["final_lifecycle_payload_identity_sha256"] != final_ledger.get(
        "payload_identity_sha256"
    ) or success["final_lifecycle_signature_sha256"] != final_ledger.get(
        "signature_sha256"
    ):
        raise OptunaSearchArtifactError(
            "Success marker differs from final lifecycle authority."
        )
    storage_value = study_summary.get("storage")
    if type(storage_value) is not str:
        raise OptunaSearchArtifactError("Study storage identity is malformed.")
    database_path = (project_root / storage_value).resolve()
    if (
        database_path == project_root
        or project_root not in database_path.parents
        or database_path.suffix != ".db"
    ):
        raise OptunaSearchArtifactError("Study storage identity is unsafe.")
    validated_database_path = _validated_optional_sqlite_path(
        database_path,
        project_root=project_root,
    )
    trials = pd.read_csv(root / "trials.csv", dtype={"state": str})
    trial_state_universe = [
        {
            "trial_number": int(row.trial_number),
            "state": str(row.state),
        }
        for row in trials.sort_values("trial_number").itertuples(index=False)
    ]
    completed = [
        item["trial_number"]
        for item in trial_state_universe
        if item["state"] == "COMPLETE"
    ]
    failed = [
        item["trial_number"] for item in trial_state_universe if item["state"] == "FAIL"
    ]
    interrupted = study_summary.get("interrupted_recovery_trial_numbers")
    if not isinstance(interrupted, list):
        raise OptunaSearchArtifactError(
            "Interrupted recovery lifecycle summary is malformed."
        )
    expected = {
        "trial_state_universe": trial_state_universe,
        "completed_trial_numbers": completed,
        "failed_trial_numbers": failed,
        "interrupted_trial_numbers": interrupted,
        "interrupted_recovery_trial_numbers": interrupted,
        "best_trial_number": study_summary.get("best_trial_number"),
        "report_search_identity_sha256": search_identity.get("sha256"),
        "study_summary_identity_sha256": canonical_sha256(study_summary),
        "metric_prediction_evidence_identity_sha256": (
            metric_prediction_evidence_identity(
                trial_metrics_bytes=(root / "trial_metrics.csv").read_bytes(),
                trial_predictions_bytes=(root / "trial_predictions.csv").read_bytes(),
            )
        ),
        "report_manifest_identity_sha256": manifest["manifest_sha256"],
    }
    try:
        key = load_lifecycle_authority_key(
            project_root=project_root,
            forbidden_roots=(root, database_path.parent),
        )
        validate_report_lifecycle_authority(
            authority_report=authority_report,
            authority_ledger=authority_ledger,
            key=key,
            expected=expected,
            database_path=validated_database_path,
            study_name=str(study_summary.get("study_name")),
        )
    except Exception as error:
        raise OptunaSearchArtifactError(
            f"External lifecycle authority validation failed: {error}"
        ) from error
    initial = authority_report["initial_statement"]["payload"]
    for artifact in (search_identity, study_summary):
        if (
            artifact.get("authority_schema_version")
            != initial["authority_schema_version"]
            or artifact.get("authority_key_fingerprint")
            != initial["authority_key_fingerprint"]
            or artifact.get("authority_bound_study_identity_sha256")
            != initial["authority_bound_study_identity_sha256"]
            or artifact.get("study_uuid") != initial["study_uuid"]
        ):
            raise OptunaSearchArtifactError("Report authority identity fields diverge.")


def _validated_optional_sqlite_path(
    path: Path,
    *,
    project_root: Path,
) -> Path | None:
    if not path.exists():
        return None
    lexical = Path(os.path.abspath(path))
    current = project_root
    try:
        relative = lexical.relative_to(project_root)
    except ValueError as error:
        raise OptunaSearchArtifactError(
            "SQLite study path escapes the project."
        ) from error
    for part in relative.parts:
        current = current / part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode) or _is_reparse_stat(metadata):
            raise OptunaSearchArtifactError(
                "SQLite study path contains a link or reparse point."
            )
    metadata = lexical.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or getattr(metadata, "st_nlink", 1) != 1
        or lexical.resolve() != lexical
    ):
        raise OptunaSearchArtifactError(
            "SQLite study must be an exact repository-contained regular file."
        )
    return lexical


def _build_inventory(root: Path, names: set[str]) -> dict[str, Any]:
    return {"schema_version": 1, "files": _inventory_records(root, names)}


def _inventory_records(root: Path, names: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name in sorted(names):
        path = root / name
        data = path.read_bytes()
        records.append(
            {
                "path": name,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return records


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, default=_json_default)
        file.write("\n")
    temporary.replace(path)


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(
            deepcopy(dict(payload)),
            file,
            sort_keys=False,
            allow_unicode=True,
        )
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    serialized = frame.copy()
    for name in serialized.columns:
        if pd.api.types.is_float_dtype(serialized[name]):
            serialized[name] = [
                ""
                if value is None or (isinstance(value, float) and math.isnan(value))
                else format(float(value), ".17g")
                for value in serialized[name].tolist()
            ]
    serialized.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OptunaSearchArtifactError(
            f"Invalid JSON artifact: {path.name}."
        ) from error
    if not isinstance(value, dict):
        raise OptunaSearchArtifactError(f"{path.name} must be a mapping.")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise OptunaSearchArtifactError(
            f"Invalid YAML artifact: {path.name}."
        ) from error
    if not isinstance(value, dict):
        raise OptunaSearchArtifactError(f"{path.name} must be a mapping.")
    return value


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _validate_semantics_legacy_unused(root: Path) -> None:
    resolved = _load_yaml(root / "resolved_search_config.yaml")
    expected_resolved_keys = PLAN_KEYS | {
        "base_candidate_config_sha256",
        "search_space_identity",
        "study_identity_sha256",
        "search_identity_sha256",
        "search_id",
    }
    if set(resolved) != expected_resolved_keys:
        raise OptunaSearchArtifactError("Resolved search config schema differs.")
    identity = _load_json(root / "search_identity.json")
    if set(identity) != {
        "schema_version",
        "search_id",
        "sha256",
        "canonical",
        "study_identity_sha256",
    }:
        raise OptunaSearchArtifactError("Search identity schema differs.")
    if (
        identity["schema_version"] != 1
        or not isinstance(identity["canonical"], dict)
        or canonical_sha256(identity["canonical"]) != identity["sha256"]
        or identity["search_id"] != resolved["search_id"]
        or identity["sha256"] != resolved["search_identity_sha256"]
        or identity["study_identity_sha256"] != resolved["study_identity_sha256"]
    ):
        raise OptunaSearchArtifactError("Search identity is inconsistent.")
    assignment = _load_json(root / "assignment_identity.json")
    if set(assignment) != {
        "schema_version",
        "splitter",
        "repeats",
        "folds",
        "assignment_seed",
        "repeat_seed_policy",
        "fold_assignments_sha256",
        "threshold_membership_sha256",
    }:
        raise OptunaSearchArtifactError("Assignment identity schema differs.")
    space = _load_json(root / "search_space.json")
    if set(space) != {"schema_version", "search_space_id", "sha256", "canonical"}:
        raise OptunaSearchArtifactError("Search-space artifact schema differs.")
    if (
        space["schema_version"] != 1
        or not isinstance(space["canonical"], dict)
        or canonical_sha256(space["canonical"]) != space["sha256"]
    ):
        raise OptunaSearchArtifactError("Search-space identity is inconsistent.")
    best = _load_json(root / "best_trial.json")
    if set(best) != {
        "schema_version",
        "trial_number",
        "objective",
        "direction",
        "tie_break",
        "resolved_parameters",
        "prediction_key_coverage",
        "evidence_scope",
    }:
        raise OptunaSearchArtifactError("Best-trial schema differs.")
    if (
        best["schema_version"] != 1
        or type(best["trial_number"]) is not int
        or best["trial_number"] < 0
        or type(best["objective"]) is not float
        or best["direction"] != "maximize"
        or best["tie_break"] != "lowest_trial_number"
        or not isinstance(best["resolved_parameters"], dict)
        or best["evidence_scope"] != "tuning_only_not_unbiased_final_evidence"
    ):
        raise OptunaSearchArtifactError("Best-trial values are invalid.")
    study = _load_json(root / "study_summary.json")
    if set(study) != {
        "schema_version",
        "study_name",
        "search_id",
        "study_identity_sha256",
        "search_identity_sha256",
        "direction",
        "metric",
        "objective_aggregation",
        "requested_trials",
        "actual_trials",
        "state_counts",
        "recovered_interrupted_trials",
        "best_trial_number",
        "best_objective",
        "storage",
        "filesystem_report_authoritative",
        "optuna_storage_role",
        "final_evaluation_status",
    }:
        raise OptunaSearchArtifactError("Study-summary schema differs.")
    if (
        study["search_id"] != identity["search_id"]
        or study["best_trial_number"] != best["trial_number"]
        or study["best_objective"] != best["objective"]
        or study["direction"] != "maximize"
        or study["metric"] != "balanced_accuracy"
        or study["final_evaluation_status"] != "not_run"
    ):
        raise OptunaSearchArtifactError("Study summary is inconsistent.")
    source = _load_json(root / "source_provenance.json")
    if set(source) != {"schema_version", "hashing_method", "files", "sha256"}:
        raise OptunaSearchArtifactError("Source-provenance schema differs.")
    source_body = {
        key: source[key] for key in ("schema_version", "hashing_method", "files")
    }
    if canonical_sha256(source_body) != source["sha256"]:
        raise OptunaSearchArtifactError("Source-provenance identity differs.")
    folds = pd.read_csv(root / "fold_assignments.csv")
    membership = pd.read_csv(root / "threshold_selection_membership.csv")
    if list(folds.columns) != ["repeat", "repeat_seed", "fold", "row_position"]:
        raise OptunaSearchArtifactError("Fold-assignment artifact schema differs.")
    if list(membership.columns) != [
        "repeat",
        "scoring_fold",
        "threshold_source_fold",
    ]:
        raise OptunaSearchArtifactError("Threshold-membership artifact schema differs.")
    trials = pd.read_csv(root / "trials.csv")
    if list(trials.columns) != [
        "trial_number",
        "state",
        "objective",
        "started_at",
        "finished_at",
        "duration_seconds",
        "optuna_parameters_json",
        "resolved_parameters_json",
        "prediction_key_coverage_json",
        "failure_reason_code",
        "failure_message",
    ]:
        raise OptunaSearchArtifactError("Trials artifact schema differs.")
    load_research_v2_config(
        root / "best_candidate_config.yaml",
        project_root=_project_root_for_artifact(root),
    )


def _validate_semantics(root: Path) -> None:
    """Run exact-schema validation and full independent semantic reconstruction."""
    from src.churn_ml.optuna_search_semantics import (
        validate_completed_search_semantics,
    )

    try:
        validate_completed_search_semantics(
            root,
            project_root=_project_root_for_artifact(root),
        )
    except Exception as error:
        raise OptunaSearchArtifactError(
            f"Semantic validation failed: {error}"
        ) from error


def _contained_real_directory(path: Path, root: Path) -> Path:
    lexical = Path(os.path.abspath(path))
    if lexical == root or root not in lexical.parents:
        raise OptunaSearchArtifactError(
            "Search directory must be repository-contained."
        )
    relative = lexical.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as error:
            raise OptunaSearchArtifactError(
                "Search directory path is unavailable."
            ) from error
        if stat.S_ISLNK(metadata.st_mode) or _is_reparse_stat(metadata):
            raise OptunaSearchArtifactError(
                "Search directory path must not contain links/reparse points."
            )
    if not stat.S_ISDIR(current.lstat().st_mode):
        raise OptunaSearchArtifactError("Search report must be a directory.")
    resolved = current.resolve()
    if resolved != lexical or root not in resolved.parents:
        raise OptunaSearchArtifactError(
            "Search directory path changed during non-following validation."
        )
    return resolved


def _is_reparse_stat(metadata: os.stat_result) -> bool:
    return bool(getattr(metadata, "st_file_attributes", 0) & 0x400)


def _project_root_for_artifact(path: Path) -> Path:
    for candidate in path.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise OptunaSearchArtifactError("Artifact is not inside a project root.")
