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
from src.churn_ml.prediction_candidates.contract_v1 import (
    OOF_COLUMNS,
    POSITIVE_CLASS_LABEL,
    PROBABILITY_SEMANTICS,
    SCHEMA_VERSION,
    TEST_COLUMNS,
    UNAVAILABLE,
    CandidateConflictError,
    CandidatePackage,
    build_candidate_id,
    create_candidate_package,
    file_sha256,
    repository_relative_path,
    resolve_under_repository,
    utc_now,
)
from src.churn_ml.research_data import canonical_sha256


BLEND_ROOT_RELATIVE = "artifacts/prediction_blends"
BLEND_SCHEMA_VERSION = "prediction_blend_v1"
SUCCESS_FILENAME = "_SUCCESS"


class BlendArtifactError(RuntimeError):
    """Raised when blend artifact materialization fails."""

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


def _file_ref(path: Path, repository_root: Path, row_count: int | None = None) -> dict[str, Any]:
    return {
        "path": repository_relative_path(path, repository_root),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
        "row_count": row_count,
    }


def search_blend(
    pool: CompatibleCandidateSet,
    settings: BlendSettings,
) -> dict[str, Any]:
    """Evaluate without writing immutable artifacts."""
    evaluation = run_blend_evaluation(pool, settings)
    diversity = analyze_diversity(pool)
    blend_id = build_blend_id(pool, settings)
    return {
        "ok": True,
        "materialized": False,
        "blend_id": blend_id,
        "candidate_ids": list(pool.candidate_ids),
        "strategy": settings.normalized().strategy,
        "exploratory": pool.exploratory,
        "deployment": evaluation.deployment,
        "cross_fitted_metrics": evaluation.cross_fitted_metrics,
        "full_oof_descriptive_metrics": evaluation.full_oof_descriptive_metrics,
        "search_budget": evaluation.search_budget,
        "assignment_hash": evaluation.assignment_hash,
        "candidate_metrics": evaluation.candidate_descriptive_metrics,
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
    identity_settings = settings.identity_payload(
        parent_candidate_ids=pool.candidate_ids,
        parent_manifest_hashes=parent_hashes,
    )

    staging = blend_root / f".staging-{blend_id}-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        _write_json(staging / "compatibility.json", pool.compatibility)
        candidate_metrics_frame = pd.json_normalize(evaluation.candidate_descriptive_metrics)
        _write_csv(staging / "candidate_metrics.csv", candidate_metrics_frame)
        _write_csv(
            staging / "diversity_matrix.csv",
            diversity["diversity_matrix"].reset_index(names="candidate_id"),
        )
        _write_csv(staging / "pairwise_analysis.csv", diversity["pairwise_analysis"])
        _write_parquet(staging / "meta_assignments.parquet", evaluation.assignments)
        _write_json(staging / "fold_results.json", {"folds": evaluation.fold_results})
        evaluation_payload = {
            "cross_fitted_metrics": evaluation.cross_fitted_metrics,
            "full_oof_descriptive_metrics": evaluation.full_oof_descriptive_metrics,
            "repeat_metrics": evaluation.repeat_metrics.to_dict(orient="records"),
            "fold_metrics": evaluation.fold_metrics.to_dict(orient="records"),
            "assignment_hash": evaluation.assignment_hash,
            "search_budget": evaluation.search_budget,
            "oof_protocol": OOF_PROTOCOL,
            "repeat_normalization_policy": REPEAT_NORMALIZATION_POLICY,
        }
        _write_json(staging / "evaluation.json", evaluation_payload)
        _write_json(staging / "final_weights.json", evaluation.deployment)
        cross_frame = pd.DataFrame(
            {
                "row_position": pool.row_positions.astype("int64"),
                "target": pool.target.astype("int64"),
                "probability_positive": evaluation.cross_fitted_oof.astype("float64"),
            }
        )
        _write_parquet(staging / "cross_fitted_predictions.parquet", cross_frame)

        # Create canonical blend candidate from cross-fitted OOF + deployment test.
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

        representative_dataset_id = pool.packages[0].manifest["dataset_id"]
        source_run_path = f"{blend_root_rel}/{blend_id}"
        source_config_sha256 = canonical_sha256(identity_settings)
        candidate_identity = {
            "schema_version": SCHEMA_VERSION,
            "source_kind": BLEND_SOURCE_KIND,
            "dataset_id": representative_dataset_id,
            "source_run_path": source_run_path,
            "source_config_sha256": source_config_sha256,
            "source_model_name": f"blend_{settings.strategy}",
            "oof_protocol": OOF_PROTOCOL,
            "positive_class_label": POSITIVE_CLASS_LABEL,
            "probability_semantics": PROBABILITY_SEMANTICS,
        }
        candidate_id = build_candidate_id(candidate_identity)
        source_metadata = {
            "schema_version": 1,
            "source_kind": BLEND_SOURCE_KIND,
            "blend_id": blend_id,
            "parent_candidate_ids": list(pool.candidate_ids),
            "parent_manifest_hashes": parent_hashes,
            "strategy": settings.strategy,
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
                {"repeat": item["repeat"], "fold": item["fold"], "weights": item["weights"]}
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
            "final_weights": evaluation.deployment["weights"],
            "final_threshold": evaluation.deployment["threshold"],
            "cross_fitted_metrics": evaluation.cross_fitted_metrics,
            "full_oof_descriptive_metrics": evaluation.full_oof_descriptive_metrics,
            "exploratory": pool.exploratory,
            "oof_generation": (
                "Honest meta-cross-fitted blend probabilities; weights/threshold "
                "for each fold use only meta-train rows."
            ),
            "test_generation": (
                "Final deployment weights fitted on all canonical OOF rows; "
                "p_test = sum_i w_i * p_test_i."
            ),
        }
        manifest_fields = {
            "candidate_id": candidate_id,
            "created_at_utc": utc_now(),
            "dataset_id": representative_dataset_id,
            "exploratory": pool.exploratory,
            "source_run_path": source_run_path,
            "source_config_path": f"{source_run_path}/blend_manifest.json",
            "source_config_sha256": source_config_sha256,
            "source_model_name": f"blend_{settings.strategy}",
            "source_model_type": "probability_blend",
            "source_predictor_path": UNAVAILABLE,
            "source_autogluon_version": UNAVAILABLE,
            "source_metric_name": "balanced_accuracy",
            "source_metric_value": evaluation.cross_fitted_metrics["balanced_accuracy"],
            "source_leaderboard_metadata": {
                "strategy": settings.strategy,
                "blend_id": blend_id,
            },
            "provenance": {
                "adapter": "blending.artifact_v1",
                "blend_schema_version": BLEND_SCHEMA_VERSION,
                "parent_candidate_ids": list(pool.candidate_ids),
                "parent_manifest_hashes": parent_hashes,
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

        artifact_files = {
            "compatibility.json": _file_ref(staging / "compatibility.json", root),
            "candidate_metrics.csv": _file_ref(
                staging / "candidate_metrics.csv",
                root,
                row_count=len(candidate_metrics_frame),
            ),
            "diversity_matrix.csv": _file_ref(
                staging / "diversity_matrix.csv",
                root,
                row_count=len(diversity["diversity_matrix"]),
            ),
            "pairwise_analysis.csv": _file_ref(
                staging / "pairwise_analysis.csv",
                root,
                row_count=len(diversity["pairwise_analysis"]),
            ),
            "meta_assignments.parquet": _file_ref(
                staging / "meta_assignments.parquet",
                root,
                row_count=len(evaluation.assignments),
            ),
            "fold_results.json": _file_ref(staging / "fold_results.json", root),
            "evaluation.json": _file_ref(staging / "evaluation.json", root),
            "final_weights.json": _file_ref(staging / "final_weights.json", root),
            "cross_fitted_predictions.parquet": _file_ref(
                staging / "cross_fitted_predictions.parquet",
                root,
                row_count=len(cross_frame),
            ),
        }
        # Rewrite refs to final paths.
        for name, ref in artifact_files.items():
            ref["path"] = f"{blend_root_rel}/{blend_id}/{name}"

        blend_manifest = {
            "schema_version": BLEND_SCHEMA_VERSION,
            "blend_id": blend_id,
            "created_at_utc": utc_now(),
            "candidate_ids": list(pool.candidate_ids),
            "parent_manifest_hashes": parent_hashes,
            "settings": identity_settings,
            "exploratory": pool.exploratory,
            "canonical_candidate_id": candidate_package.candidate_id,
            "canonical_candidate_path": repository_relative_path(
                candidate_package.package_dir, root
            ),
            "assignment_hash": evaluation.assignment_hash,
            "deployment": evaluation.deployment,
            "cross_fitted_metrics": evaluation.cross_fitted_metrics,
            "full_oof_descriptive_metrics": evaluation.full_oof_descriptive_metrics,
            "artifact_files": artifact_files,
            "diversity_notes": diversity["notes"],
        }
        _write_json(staging / "blend_manifest.json", blend_manifest)
        blend_manifest["artifact_files"]["blend_manifest.json"] = {
            "path": f"{blend_root_rel}/{blend_id}/blend_manifest.json",
            "sha256": file_sha256(staging / "blend_manifest.json"),
            "size_bytes": (staging / "blend_manifest.json").stat().st_size,
            "row_count": None,
        }
        # Rewrite manifest with self-reference hash included.
        _write_json(staging / "blend_manifest.json", blend_manifest)

        success_payload = {
            "schema_version": 1,
            "blend_id": blend_id,
            "status": "completed",
            "canonical_candidate_id": candidate_package.candidate_id,
            "manifest_sha256": file_sha256(staging / "blend_manifest.json"),
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
                (final_dir / "blend_manifest.json").read_text(encoding="utf-8")
            )
            if not _blend_identity_matches(existing_manifest, blend_manifest):
                raise CandidateConflictError(
                    f"Blend {blend_id} already exists with different content."
                )
            shutil.rmtree(staging, ignore_errors=True)
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
    """Compare durable blend identity, ignoring timestamps and self-hashes."""
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
) -> dict[str, Any]:
    root = repository_root.resolve()
    blend_root_rel = normalize_blend_root_relative(blend_root_relative)
    if ".." in blend_id or "/" in blend_id or "\\" in blend_id:
        raise BlendArtifactError(
            f"Invalid blend_id: {blend_id!r}",
            reason_code="path_traversal",
        )
    blend_dir = resolve_under_repository(f"{blend_root_rel}/{blend_id}", root)
    if not (blend_dir / SUCCESS_FILENAME).is_file():
        raise BlendArtifactError(
            f"Blend artifact missing _SUCCESS: {blend_id}",
            reason_code="success_missing",
        )
    manifest = json.loads((blend_dir / "blend_manifest.json").read_text(encoding="utf-8"))
    evaluation = json.loads((blend_dir / "evaluation.json").read_text(encoding="utf-8"))
    final_weights = json.loads((blend_dir / "final_weights.json").read_text(encoding="utf-8"))
    return {
        "ok": True,
        "blend_id": blend_id,
        "blend_path": repository_relative_path(blend_dir, root),
        "manifest": manifest,
        "evaluation": evaluation,
        "deployment": final_weights,
        "parents": manifest.get("candidate_ids"),
        "canonical_candidate_id": manifest.get("canonical_candidate_id"),
        "settings": manifest.get("settings"),
    }
