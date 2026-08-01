"""Immutable blend artifacts and canonical blend candidate materialization."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.churn_ml.blending.compatibility_v1 import CompatibleCandidateSet
from src.churn_ml.blending.diversity_v1 import analyze_diversity
from src.churn_ml.blending.evaluation_v1 import (
    BLEND_SOURCE_KIND,
    OOF_PROTOCOL,
    REPEAT_NORMALIZATION_POLICY,
    BlendEvaluationResult,
    BlendSettings,
    run_blend_evaluation,
)
from src.churn_ml.blending.optimization_v1 import (
    blend_probabilities,
    validate_weights,
)
from src.churn_ml.prediction_candidates.contract_v1 import (
    OOF_COLUMNS,
    POSITIVE_CLASS_LABEL,
    PROBABILITY_SEMANTICS,
    SCHEMA_VERSION,
    TEST_COLUMNS,
    UNAVAILABLE,
    CandidateConflictError,
    CandidatePackage,
    PredictionCandidateError,
    build_candidate_id,
    create_candidate_package,
    file_sha256,
    load_candidate_package,
    repository_relative_path,
    resolve_under_repository,
    utc_now,
    validate_candidate_package,
)
from src.churn_ml.research_data import canonical_sha256


BLEND_ROOT_RELATIVE = "artifacts/prediction_blends"
BLEND_SCHEMA_VERSION = "prediction_blend_v1"
SUCCESS_FILENAME = "_SUCCESS"
MANIFEST_FILENAME = "blend_manifest.json"
REQUIRED_ARTIFACT_FILES = (
    "compatibility.json",
    "candidate_metrics.csv",
    "diversity_matrix.csv",
    "pairwise_analysis.csv",
    "meta_assignments.parquet",
    "fold_results.json",
    "evaluation.json",
    "final_weights.json",
    "cross_fitted_predictions.parquet",
    "held_out_decisions.parquet",
)


class BlendArtifactError(RuntimeError):
    """Raised when blend artifact materialization or validation fails."""

    def __init__(self, message: str, *, reason_code: str = "blend_artifact") -> None:
        self.reason_code = reason_code
        super().__init__(message)


@dataclass(frozen=True)
class BlendMaterialization:
    blend_id: str
    blend_dir: Path
    candidate_package: CandidatePackage
    evaluation: BlendEvaluationResult
    manifest: dict[str, Any]


def build_blend_id(
    pool: CompatibleCandidateSet,
    settings: BlendSettings,
) -> str:
    parent_hashes = [
        file_sha256(package.package_dir / "candidate_manifest.json")
        for package in pool.packages
    ]
    payload = settings.identity_payload(
        parent_candidate_ids=pool.candidate_ids,
        parent_manifest_hashes=parent_hashes,
    )
    return f"pb1_{canonical_sha256(payload)[:16]}"


def normalize_blend_root_relative(blend_root_relative: str | None) -> str:
    relative = blend_root_relative or BLEND_ROOT_RELATIVE
    if not relative or relative.startswith("/") or PurePosixPath(relative).is_absolute():
        raise BlendArtifactError(
            f"Absolute or empty blend root rejected: {relative!r}",
            reason_code="path_traversal",
        )
    if ".." in PurePosixPath(relative).parts:
        raise BlendArtifactError(
            f"Path traversal rejected: {relative!r}",
            reason_code="path_traversal",
        )
    return PurePosixPath(relative).as_posix()


def _atomic_write_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    replaced = False
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        replaced = True
    finally:
        if not replaced:
            temporary_path.unlink(missing_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode(
            "utf-8"
        ),
    )


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_write_bytes(
        path,
        frame.to_csv(index=False, lineterminator="\n").encode("utf-8"),
    )


def _file_ref(
    path: Path,
    repository_root: Path,
    *,
    final_relative: str,
    row_count: int | None = None,
) -> dict[str, Any]:
    return {
        "path": final_relative,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
        "row_count": row_count,
    }


def search_blend(
    pool: CompatibleCandidateSet,
    settings: BlendSettings,
) -> dict[str, Any]:
    """Evaluate without writing immutable artifacts."""
    settings = settings.normalized()
    evaluation = run_blend_evaluation(pool, settings)
    diversity = analyze_diversity(pool)
    blend_id = build_blend_id(pool, settings)
    return {
        "ok": True,
        "materialized": False,
        "blend_id": blend_id,
        "candidate_ids": list(pool.candidate_ids),
        "strategy": settings.strategy,
        "optimizer_backend": settings.optimizer_backend,
        "optimizer_settings": evaluation.search_budget,
        "exploratory": pool.exploratory,
        "deployment": evaluation.deployment,
        "final_deployment_weights": evaluation.deployment["weights"],
        "final_deployment_threshold": evaluation.deployment["threshold"],
        "honest_meta_cv_metrics": evaluation.honest_meta_cv_metrics,
        "cross_fitted_probability_descriptive_metrics": (
            evaluation.cross_fitted_probability_descriptive_metrics
        ),
        "full_oof_descriptive_metrics": evaluation.full_oof_descriptive_metrics,
        # Backward-compatible alias retained as descriptive.
        "cross_fitted_metrics": evaluation.cross_fitted_probability_descriptive_metrics,
        "search_budget": evaluation.search_budget,
        "assignment_hash": evaluation.assignment_hash,
        "candidate_metrics": evaluation.candidate_descriptive_metrics,
        "optuna_study_summaries": evaluation.optuna_study_summaries,
        "pairwise_analysis": diversity["pairwise_analysis"].to_dict(orient="records"),
        "notes": diversity["notes"],
    }


def materialize_blend(
    pool: CompatibleCandidateSet,
    settings: BlendSettings,
    *,
    blend_root_relative: str | None = None,
) -> BlendMaterialization:
    """Write detailed blend artifact and canonical prediction_candidate_v1."""
    settings = settings.normalized()
    root = pool.repository_root
    blend_root_rel = normalize_blend_root_relative(blend_root_relative)
    blend_root = resolve_under_repository(blend_root_rel, root)
    blend_root.mkdir(parents=True, exist_ok=True)

    evaluation = run_blend_evaluation(pool, settings)
    diversity = analyze_diversity(pool)
    blend_id = build_blend_id(pool, settings)
    final_dir = blend_root / blend_id

    parent_hashes = [
        file_sha256(package.package_dir / "candidate_manifest.json")
        for package in pool.packages
    ]
    parent_dataset_ids = [str(package.manifest["dataset_id"]) for package in pool.packages]
    identity_reference_dataset_id = parent_dataset_ids[0]
    identity_settings = settings.identity_payload(
        parent_candidate_ids=pool.candidate_ids,
        parent_manifest_hashes=parent_hashes,
    )

    staging = blend_root / f".staging-{blend_id}-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        _write_json(staging / "compatibility.json", pool.compatibility)
        candidate_metrics_frame = pd.json_normalize(
            evaluation.candidate_descriptive_metrics
        )
        _write_csv(staging / "candidate_metrics.csv", candidate_metrics_frame)
        _write_csv(
            staging / "diversity_matrix.csv",
            diversity["diversity_matrix"].reset_index(names="candidate_id"),
        )
        _write_csv(staging / "pairwise_analysis.csv", diversity["pairwise_analysis"])
        _write_parquet(staging / "meta_assignments.parquet", evaluation.assignments)
        _write_json(staging / "fold_results.json", {"folds": evaluation.fold_results})
        evaluation_payload = {
            "honest_meta_cv_metrics": evaluation.honest_meta_cv_metrics,
            "cross_fitted_probability_descriptive_metrics": (
                evaluation.cross_fitted_probability_descriptive_metrics
            ),
            "full_oof_descriptive_metrics": evaluation.full_oof_descriptive_metrics,
            "repeat_metrics": evaluation.repeat_metrics.to_dict(orient="records"),
            "fold_metrics": evaluation.fold_metrics.to_dict(orient="records"),
            "assignment_hash": evaluation.assignment_hash,
            "search_budget": evaluation.search_budget,
            "optuna_study_summaries": evaluation.optuna_study_summaries,
            "oof_protocol": OOF_PROTOCOL,
            "repeat_normalization_policy": REPEAT_NORMALIZATION_POLICY,
        }
        _write_json(staging / "evaluation.json", evaluation_payload)
        _write_json(staging / "final_weights.json", evaluation.deployment)
        _write_parquet(staging / "held_out_decisions.parquet", evaluation.held_out_decisions)
        cross_frame = pd.DataFrame(
            {
                "row_position": pool.row_positions.astype("int64"),
                "target": pool.target.astype("int64"),
                "probability_positive": evaluation.cross_fitted_oof.astype("float64"),
            }
        )
        _write_parquet(staging / "cross_fitted_predictions.parquet", cross_frame)

        if evaluation.optuna_trial_history is not None:
            # Persist latents/weights as JSON strings for parquet stability.
            history = evaluation.optuna_trial_history.copy()
            for column in ("latents", "weights"):
                if column in history.columns:
                    history[column] = history[column].map(
                        lambda value: json.dumps(value) if value is not None else None
                    )
            _write_parquet(staging / "optuna_trials.parquet", history)
            _write_json(
                staging / "optuna_studies.json",
                {"studies": evaluation.optuna_study_summaries},
            )

        oof_candidate = pd.DataFrame(
            {
                "row_position": pool.row_positions.astype("int64"),
                "target": pool.target.astype("int64"),
                "probability_positive": evaluation.cross_fitted_oof.astype("float64"),
            }
        )[list(OOF_COLUMNS)]
        test_candidate = pd.DataFrame(
            {
                "row_position": pool.test_row_positions.astype("int64"),
                "probability_positive": evaluation.test_probabilities.astype("float64"),
            }
        )[list(TEST_COLUMNS)]

        source_run_path = f"{blend_root_rel}/{blend_id}"
        source_config_sha256 = canonical_sha256(identity_settings)
        candidate_identity = {
            "schema_version": SCHEMA_VERSION,
            "source_kind": BLEND_SOURCE_KIND,
            "dataset_id": identity_reference_dataset_id,
            "source_run_path": source_run_path,
            "source_config_sha256": source_config_sha256,
            "source_model_name": (
                f"blend_{settings.strategy}_{settings.optimizer_backend}"
            ),
            "oof_protocol": OOF_PROTOCOL,
            "positive_class_label": POSITIVE_CLASS_LABEL,
            "probability_semantics": PROBABILITY_SEMANTICS,
        }
        candidate_id = build_candidate_id(candidate_identity)
        honest_ba = float(
            evaluation.honest_meta_cv_metrics["mean_repeat_balanced_accuracy"]
        )
        source_metadata = {
            "schema_version": 1,
            "source_kind": BLEND_SOURCE_KIND,
            "blend_id": blend_id,
            "parent_candidate_ids": list(pool.candidate_ids),
            "parent_manifest_hashes": parent_hashes,
            "identity_reference_dataset_id": identity_reference_dataset_id,
            "parent_dataset_ids": parent_dataset_ids,
            "strategy": settings.strategy,
            "optimizer_backend": settings.optimizer_backend,
            "meta_evaluation_protocol": {
                "folds": settings.folds,
                "repeats": settings.repeats,
                "seed": settings.seed,
                "assignment_hash": evaluation.assignment_hash,
                "repeat_normalization_policy": REPEAT_NORMALIZATION_POLICY,
            },
            "weight_search_settings": evaluation.search_budget,
            "threshold_search_settings": dict(settings.threshold_policy or {}),
            "fold_specific_weights": [
                {
                    "repeat": item["repeat"],
                    "fold": item["fold"],
                    "weights": item["weights"],
                }
                for item in evaluation.fold_results
            ],
            "fold_specific_thresholds": [
                {
                    "repeat": item["repeat"],
                    "fold": item["fold"],
                    "threshold": item["threshold"],
                }
                for item in evaluation.fold_results
            ],
            "final_deployment_weights": evaluation.deployment["weights"],
            "final_deployment_threshold": evaluation.deployment["threshold"],
            "final_weights": evaluation.deployment["weights"],
            "final_threshold": evaluation.deployment["threshold"],
            "threshold_metric": "balanced_accuracy",
            "threshold_selection_scope": "full_canonical_oof_for_deployment",
            "test_probability_generation": (
                "Final deployment weights fitted on all canonical OOF rows; "
                "p_test = sum_i w_i * p_test_i."
            ),
            "honest_meta_cv_metrics": evaluation.honest_meta_cv_metrics,
            "cross_fitted_probability_descriptive_metrics": (
                evaluation.cross_fitted_probability_descriptive_metrics
            ),
            "full_oof_descriptive_metrics": evaluation.full_oof_descriptive_metrics,
            "exploratory": pool.exploratory,
            "oof_generation": (
                "Honest meta-cross-fitted blend probabilities; weights/threshold "
                "for each fold use only meta-train rows. Primary score is "
                "mean_repeat_balanced_accuracy from held-out decisions."
            ),
            "test_generation": (
                "Final deployment weights fitted on all canonical OOF rows; "
                "p_test = sum_i w_i * p_test_i."
            ),
            "competition_asset_identity": {
                "identity_reference_dataset_id": identity_reference_dataset_id,
                "ordered_submission_id_hash_source": "prediction_candidate_v1_manifest",
            },
        }
        from src.churn_ml.competition_assets_v1 import resolve_competition_assets

        competition = resolve_competition_assets(
            root, dataset_version=identity_reference_dataset_id
        )
        readiness_blockers: list[dict[str, str]] = []
        readiness_warnings: list[dict[str, str]] = []
        if pool.exploratory:
            readiness_warnings.append(
                {
                    "code": "exploratory_candidate",
                    "message": (
                        "Candidate is marked exploratory and must remain visibly "
                        "exploratory downstream."
                    ),
                }
            )
        if not competition.ready:
            readiness_blockers.append(
                {
                    "code": "competition_assets_invalid",
                    "message": (
                        "Competition assets are not ready: "
                        + "; ".join(competition.blocking_reasons)
                    ),
                }
            )
        threshold_value = float(evaluation.deployment["threshold"])
        if not np.isfinite(threshold_value) or not (0.0 <= threshold_value <= 1.0):
            readiness_blockers.append(
                {
                    "code": "threshold_invalid",
                    "message": "Final deployment threshold must be finite and in [0, 1].",
                }
            )
        provisional_readiness = {
            "state": "ready" if not readiness_blockers else "blocked",
            "ready": not readiness_blockers,
            "warnings": readiness_warnings,
            "blockers": readiness_blockers,
            "threshold": threshold_value,
            "competition_identity": {
                "ready": competition.ready,
                "ordered_id_sha256": competition.ordered_id_sha256,
                "test_anchor_hash": competition.test_anchor_hash,
            },
        }
        source_metadata["submission_readiness"] = provisional_readiness
        manifest_fields = {
            "candidate_id": candidate_id,
            "created_at_utc": utc_now(),
            "dataset_id": identity_reference_dataset_id,
            "exploratory": pool.exploratory,
            "source_run_path": source_run_path,
            "source_config_path": f"{source_run_path}/{MANIFEST_FILENAME}",
            "source_config_sha256": source_config_sha256,
            "source_model_name": candidate_identity["source_model_name"],
            "source_model_type": "probability_blend",
            "source_predictor_path": UNAVAILABLE,
            "source_autogluon_version": UNAVAILABLE,
            "source_metric_name": "mean_repeat_balanced_accuracy",
            "source_metric_value": honest_ba,
            "source_leaderboard_metadata": {
                "strategy": settings.strategy,
                "optimizer_backend": settings.optimizer_backend,
                "blend_id": blend_id,
                "primary_score": "mean_repeat_balanced_accuracy",
            },
            "provenance": {
                "adapter": "blending.artifact_v1",
                "blend_schema_version": BLEND_SCHEMA_VERSION,
                "parent_candidate_ids": list(pool.candidate_ids),
                "parent_manifest_hashes": parent_hashes,
                "identity_reference_dataset_id": identity_reference_dataset_id,
                "parent_dataset_ids": parent_dataset_ids,
                "settings": identity_settings,
            },
        }
        candidate_package = create_candidate_package(
            repository_root=root,
            identity=candidate_identity,
            oof=oof_candidate,
            test=test_candidate,
            manifest_fields=manifest_fields,
            source_metadata=source_metadata,
            candidates_root_relative=pool.candidates_root_relative,
        )
        # Re-evaluate readiness on the materialized candidate without requiring
        # the blend artifact file that is still being staged.
        from src.churn_ml.prediction_candidates.submission_v1 import (
            evaluate_submission_readiness,
        )

        readiness = evaluate_submission_readiness(
            candidate_package.candidate_id,
            repository_root=root,
            candidates_root_relative=pool.candidates_root_relative,
            blend_root_relative=blend_root_rel,
            require_blend_artifact=False,
        )
        resolved_readiness = {
            "state": readiness["state"],
            "ready": readiness["ready"],
            "warnings": readiness["warnings"],
            "blockers": readiness["blockers"],
            "threshold": readiness.get("threshold"),
            "competition_identity": readiness.get("competition_identity"),
        }

        artifact_files: dict[str, Any] = {}
        for name in REQUIRED_ARTIFACT_FILES:
            path = staging / name
            row_count = None
            if name.endswith(".parquet") or name.endswith(".csv"):
                if name == "candidate_metrics.csv":
                    row_count = len(candidate_metrics_frame)
                elif name == "diversity_matrix.csv":
                    row_count = len(diversity["diversity_matrix"])
                elif name == "pairwise_analysis.csv":
                    row_count = len(diversity["pairwise_analysis"])
                elif name == "meta_assignments.parquet":
                    row_count = len(evaluation.assignments)
                elif name == "cross_fitted_predictions.parquet":
                    row_count = len(cross_frame)
                elif name == "held_out_decisions.parquet":
                    row_count = len(evaluation.held_out_decisions)
            artifact_files[name] = _file_ref(
                path,
                root,
                final_relative=f"{blend_root_rel}/{blend_id}/{name}",
                row_count=row_count,
            )
        if evaluation.optuna_trial_history is not None:
            artifact_files["optuna_trials.parquet"] = _file_ref(
                staging / "optuna_trials.parquet",
                root,
                final_relative=f"{blend_root_rel}/{blend_id}/optuna_trials.parquet",
                row_count=len(evaluation.optuna_trial_history),
            )
            artifact_files["optuna_studies.json"] = _file_ref(
                staging / "optuna_studies.json",
                root,
                final_relative=f"{blend_root_rel}/{blend_id}/optuna_studies.json",
            )

        blend_manifest = {
            "schema_version": BLEND_SCHEMA_VERSION,
            "blend_id": blend_id,
            "created_at_utc": utc_now(),
            "candidate_ids": list(pool.candidate_ids),
            "parent_manifest_hashes": parent_hashes,
            "parent_dataset_ids": parent_dataset_ids,
            "identity_reference_dataset_id": identity_reference_dataset_id,
            "settings": identity_settings,
            "exploratory": pool.exploratory,
            "canonical_candidate_id": candidate_package.candidate_id,
            "canonical_candidate_path": repository_relative_path(
                candidate_package.package_dir, root
            ),
            "assignment_hash": evaluation.assignment_hash,
            "deployment": evaluation.deployment,
            "final_deployment_weights": evaluation.deployment["weights"],
            "final_deployment_threshold": evaluation.deployment["threshold"],
            "honest_meta_cv_metrics": evaluation.honest_meta_cv_metrics,
            "cross_fitted_probability_descriptive_metrics": (
                evaluation.cross_fitted_probability_descriptive_metrics
            ),
            "full_oof_descriptive_metrics": evaluation.full_oof_descriptive_metrics,
            "submission_readiness": resolved_readiness,
            "artifact_files": artifact_files,
            "diversity_notes": diversity["notes"],
            "optuna_study_summaries": evaluation.optuna_study_summaries,
        }
        # Write manifest once in final form. Do not embed a self-hash.
        _write_json(staging / MANIFEST_FILENAME, blend_manifest)
        manifest_sha256 = file_sha256(staging / MANIFEST_FILENAME)

        success_payload = {
            "schema_version": 1,
            "blend_id": blend_id,
            "status": "completed",
            "canonical_candidate_id": candidate_package.candidate_id,
            "manifest_sha256": manifest_sha256,
            "completed_at_utc": utc_now(),
        }
        _write_json(staging / SUCCESS_FILENAME, success_payload)

        if final_dir.exists():
            if not (final_dir / SUCCESS_FILENAME).is_file():
                raise BlendArtifactError(
                    f"Incomplete blend artifact exists: {final_dir}",
                    reason_code="blend_incomplete",
                )
            existing_manifest = json.loads(
                (final_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
            )
            if not _blend_identity_matches(existing_manifest, blend_manifest):
                raise CandidateConflictError(
                    f"Blend {blend_id} already exists with different content."
                )
            shutil.rmtree(staging, ignore_errors=True)
            # Strict-validate the existing artifact before returning.
            load_blend_artifact(
                blend_id,
                repository_root=root,
                blend_root_relative=blend_root_rel,
            )
            return BlendMaterialization(
                blend_id=blend_id,
                blend_dir=final_dir,
                candidate_package=candidate_package,
                evaluation=evaluation,
                manifest=existing_manifest,
            )

        os.replace(staging, final_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    load_blend_artifact(
        blend_id,
        repository_root=root,
        blend_root_relative=blend_root_rel,
    )
    return BlendMaterialization(
        blend_id=blend_id,
        blend_dir=final_dir,
        candidate_package=candidate_package,
        evaluation=evaluation,
        manifest=blend_manifest,
    )


def _blend_identity_matches(
    existing: Mapping[str, Any],
    proposed: Mapping[str, Any],
) -> bool:
    """Compare durable blend identity, ignoring timestamps."""
    keys = (
        "blend_id",
        "candidate_ids",
        "parent_manifest_hashes",
        "settings",
        "exploratory",
        "canonical_candidate_id",
        "assignment_hash",
    )
    for key in keys:
        if existing.get(key) != proposed.get(key):
            return False
    existing_weights = (existing.get("deployment") or {}).get("weight_vector")
    proposed_weights = (proposed.get("deployment") or {}).get("weight_vector")
    if existing_weights != proposed_weights:
        return False
    existing_threshold = (existing.get("deployment") or {}).get("threshold")
    proposed_threshold = (proposed.get("deployment") or {}).get("threshold")
    return existing_threshold == proposed_threshold


def load_blend_artifact(
    blend_id: str,
    *,
    repository_root: Path,
    blend_root_relative: str | None = None,
    candidates_root_relative: str | None = None,
) -> dict[str, Any]:
    """Strictly validate and load an immutable blend artifact."""
    root = repository_root.resolve()
    blend_root_rel = normalize_blend_root_relative(blend_root_relative)
    if ".." in blend_id or "/" in blend_id or "\\" in blend_id:
        raise BlendArtifactError(
            f"Invalid blend_id: {blend_id!r}",
            reason_code="path_traversal",
        )
    blend_dir = resolve_under_repository(f"{blend_root_rel}/{blend_id}", root)
    success_path = blend_dir / SUCCESS_FILENAME
    manifest_path = blend_dir / MANIFEST_FILENAME
    if not success_path.is_file():
        raise BlendArtifactError(
            f"Blend artifact missing _SUCCESS: {blend_id}",
            reason_code="success_missing",
        )
    if not manifest_path.is_file():
        raise BlendArtifactError(
            f"Blend artifact missing {MANIFEST_FILENAME}: {blend_id}",
            reason_code="manifest_missing",
        )

    try:
        success = json.loads(success_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BlendArtifactError(
            f"Blend artifact JSON unreadable: {error}",
            reason_code="json_invalid",
        ) from error

    if not isinstance(success, dict) or not isinstance(manifest, dict):
        raise BlendArtifactError(
            "Blend _SUCCESS and manifest must be JSON objects.",
            reason_code="schema_invalid",
        )
    for key in (
        "schema_version",
        "blend_id",
        "status",
        "manifest_sha256",
        "canonical_candidate_id",
        "completed_at_utc",
    ):
        if key not in success:
            raise BlendArtifactError(
                f"_SUCCESS missing required key {key!r}.",
                reason_code="success_schema_invalid",
            )
    if success.get("status") != "completed":
        raise BlendArtifactError(
            f"_SUCCESS status is not completed: {success.get('status')!r}",
            reason_code="success_status_invalid",
        )
    if success.get("blend_id") != blend_id:
        raise BlendArtifactError(
            "blend_id mismatch between path and _SUCCESS.",
            reason_code="blend_id_mismatch",
        )
    if "manifest_sha256" in manifest:
        raise BlendArtifactError(
            "blend_manifest.json must not contain a self-hash.",
            reason_code="manifest_self_hash_forbidden",
        )
    actual_manifest_hash = file_sha256(manifest_path)
    if success.get("manifest_sha256") != actual_manifest_hash:
        raise BlendArtifactError(
            "blend_manifest.json SHA-256 does not match _SUCCESS.manifest_sha256.",
            reason_code="manifest_hash_mismatch",
        )
    if manifest.get("schema_version") != BLEND_SCHEMA_VERSION:
        raise BlendArtifactError(
            f"Unsupported blend schema: {manifest.get('schema_version')!r}",
            reason_code="schema_invalid",
        )
    if manifest.get("blend_id") != blend_id:
        raise BlendArtifactError(
            "blend_id mismatch between path and manifest.",
            reason_code="blend_id_mismatch",
        )
    if success.get("canonical_candidate_id") != manifest.get("canonical_candidate_id"):
        raise BlendArtifactError(
            "canonical_candidate_id mismatch between _SUCCESS and manifest.",
            reason_code="candidate_mismatch",
        )

    artifact_files = manifest.get("artifact_files")
    if not isinstance(artifact_files, dict):
        raise BlendArtifactError(
            "manifest.artifact_files must be an object.",
            reason_code="schema_invalid",
        )
    required = set(REQUIRED_ARTIFACT_FILES)
    if settings_require_optuna(manifest):
        required.add("optuna_trials.parquet")
        required.add("optuna_studies.json")
    missing = sorted(required - set(artifact_files))
    if missing:
        raise BlendArtifactError(
            f"Blend artifact missing required file references: {missing}",
            reason_code="required_files_missing",
        )

    for name, ref in artifact_files.items():
        if not isinstance(ref, dict):
            raise BlendArtifactError(
                f"Invalid artifact reference for {name}.",
                reason_code="artifact_ref_invalid",
            )
        relative = str(ref.get("path", ""))
        if ".." in PurePosixPath(relative).parts or PurePosixPath(relative).is_absolute():
            raise BlendArtifactError(
                f"Path traversal rejected in artifact ref {name}: {relative!r}",
                reason_code="path_traversal",
            )
        expected_suffix = f"{blend_root_rel}/{blend_id}/{name}"
        if relative != expected_suffix:
            raise BlendArtifactError(
                f"Artifact path for {name} is not the canonical blend path.",
                reason_code="artifact_path_invalid",
            )
        path = resolve_under_repository(relative, root)
        if not path.is_file():
            raise BlendArtifactError(
                f"Missing blend artifact file: {relative}",
                reason_code="artifact_missing",
            )
        actual_sha = file_sha256(path)
        if actual_sha != ref.get("sha256"):
            raise BlendArtifactError(
                f"SHA-256 mismatch for {name}.",
                reason_code="artifact_hash_mismatch",
            )
        if int(ref.get("size_bytes", -1)) != path.stat().st_size:
            raise BlendArtifactError(
                f"Size mismatch for {name}.",
                reason_code="artifact_size_mismatch",
            )
        if ref.get("row_count") is not None:
            if name.endswith(".parquet"):
                actual_rows = len(pd.read_parquet(path))
            elif name.endswith(".csv"):
                actual_rows = len(pd.read_csv(path))
            else:
                actual_rows = None
            if actual_rows is not None and int(ref["row_count"]) != actual_rows:
                raise BlendArtifactError(
                    f"Row count mismatch for {name}.",
                    reason_code="artifact_row_count_mismatch",
                )

    evaluation = json.loads((blend_dir / "evaluation.json").read_text(encoding="utf-8"))
    final_weights = json.loads((blend_dir / "final_weights.json").read_text(encoding="utf-8"))
    deployment = manifest.get("deployment") or final_weights
    weight_vector = deployment.get("weight_vector")
    if not isinstance(weight_vector, list):
        raise BlendArtifactError(
            "Final weights weight_vector missing.",
            reason_code="weights_invalid",
        )
    try:
        weights = validate_weights(
            weight_vector, n_candidates=len(manifest.get("candidate_ids") or [])
        )
    except Exception as error:
        raise BlendArtifactError(
            f"Final weights invalid: {error}",
            reason_code="weights_invalid",
        ) from error
    threshold = deployment.get("threshold")
    if (
        threshold is None
        or not np.isfinite(float(threshold))
        or not (0.0 <= float(threshold) <= 1.0)
    ):
        raise BlendArtifactError(
            "Final threshold must be finite and within [0, 1].",
            reason_code="threshold_invalid",
        )

    parent_ids = list(manifest.get("candidate_ids") or [])
    parent_hashes = list(manifest.get("parent_manifest_hashes") or [])
    if len(parent_ids) != len(parent_hashes):
        raise BlendArtifactError(
            "Parent candidate IDs and manifest hashes length mismatch.",
            reason_code="parent_mismatch",
        )
    for parent_id, expected_hash in zip(parent_ids, parent_hashes, strict=True):
        if ".." in parent_id or "/" in parent_id or "\\" in parent_id:
            raise BlendArtifactError(
                f"Invalid parent candidate_id: {parent_id!r}",
                reason_code="path_traversal",
            )
        parent_dir = _resolve_candidate_dir(
            parent_id,
            repository_root=root,
            candidates_root_relative=candidates_root_relative,
        )
        parent_manifest = parent_dir / "candidate_manifest.json"
        if not parent_manifest.is_file():
            raise BlendArtifactError(
                f"Parent candidate missing: {parent_id}",
                reason_code="parent_missing",
            )
        if file_sha256(parent_manifest) != expected_hash:
            raise BlendArtifactError(
                f"Parent manifest hash mismatch for {parent_id}.",
                reason_code="parent_hash_mismatch",
            )
        try:
            load_candidate_package(
                parent_dir,
                repository_root=root,
                candidates_root_relative=candidates_root_relative,
            )
        except PredictionCandidateError as error:
            raise BlendArtifactError(
                f"Parent candidate invalid: {parent_id}: {error}",
                reason_code="parent_invalid",
            ) from error

    canonical_id = str(manifest["canonical_candidate_id"])
    if ".." in canonical_id or "/" in canonical_id or "\\" in canonical_id:
        raise BlendArtifactError(
            f"Invalid canonical candidate_id: {canonical_id!r}",
            reason_code="path_traversal",
        )
    candidate_dir = _resolve_candidate_dir(
        canonical_id,
        repository_root=root,
        candidates_root_relative=candidates_root_relative,
    )
    try:
        candidate_package = load_candidate_package(
            candidate_dir,
            repository_root=root,
            candidates_root_relative=candidates_root_relative,
        )
        validate_candidate_package(candidate_package, repository_root=root)
    except PredictionCandidateError as error:
        raise BlendArtifactError(
            f"Canonical blend candidate invalid: {error}",
            reason_code="candidate_invalid",
        ) from error
    if candidate_package.manifest.get("source_kind") != BLEND_SOURCE_KIND:
        raise BlendArtifactError(
            "Canonical candidate source_kind is not a blend.",
            reason_code="candidate_mismatch",
        )
    if candidate_package.source_metadata.get("blend_id") != blend_id:
        raise BlendArtifactError(
            "Canonical candidate blend_id mismatch.",
            reason_code="candidate_mismatch",
        )

    # Verify test probabilities reconstruct from final weights and parents.
    parent_packages = [
        load_candidate_package(
            _resolve_candidate_dir(
                parent_id,
                repository_root=root,
                candidates_root_relative=candidates_root_relative,
            ),
            repository_root=root,
            candidates_root_relative=candidates_root_relative,
        )
        for parent_id in parent_ids
    ]
    test_matrix = np.column_stack(
        [
            package.test["probability_positive"].to_numpy(dtype=np.float64)
            for package in parent_packages
        ]
    )
    reconstructed = blend_probabilities(test_matrix, weights)
    stored = candidate_package.test["probability_positive"].to_numpy(dtype=np.float64)
    if reconstructed.shape != stored.shape or not np.allclose(
        reconstructed, stored, rtol=0.0, atol=1.0e-12
    ):
        raise BlendArtifactError(
            "Blend test probabilities do not match final deployment weights.",
            reason_code="test_probability_mismatch",
        )

    return {
        "ok": True,
        "blend_id": blend_id,
        "blend_path": repository_relative_path(blend_dir, root),
        "manifest": manifest,
        "evaluation": evaluation,
        "deployment": final_weights,
        "final_deployment_weights": deployment.get("weights"),
        "final_deployment_threshold": deployment.get("threshold"),
        "honest_meta_cv_metrics": manifest.get("honest_meta_cv_metrics")
        or evaluation.get("honest_meta_cv_metrics"),
        "cross_fitted_probability_descriptive_metrics": manifest.get(
            "cross_fitted_probability_descriptive_metrics"
        )
        or evaluation.get("cross_fitted_probability_descriptive_metrics"),
        "submission_readiness": manifest.get("submission_readiness"),
        "parents": parent_ids,
        "canonical_candidate_id": canonical_id,
        "settings": manifest.get("settings"),
        "optimizer_backend": (manifest.get("settings") or {}).get("optimizer_backend"),
        "manifest_sha256": actual_manifest_hash,
    }


def settings_require_optuna(manifest: Mapping[str, Any]) -> bool:
    settings = manifest.get("settings") or {}
    return str(settings.get("optimizer_backend")) == "optuna"


def _resolve_candidate_dir(
    candidate_id: str,
    *,
    repository_root: Path,
    candidates_root_relative: str | None,
) -> Path:
    relative = candidates_root_relative or "artifacts/prediction_candidates"
    if ".." in PurePosixPath(relative).parts or PurePosixPath(relative).is_absolute():
        raise BlendArtifactError(
            f"Path traversal rejected: {relative!r}",
            reason_code="path_traversal",
        )
    return resolve_under_repository(f"{relative}/{candidate_id}", repository_root)
