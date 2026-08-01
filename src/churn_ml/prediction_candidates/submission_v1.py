"""Submission-readiness and local Kaggle submission generation for candidates.

Source-neutral backend over ``prediction_candidate_v1`` packages. Reuses
``competition_assets_v1`` and ``deployment_v1_artifacts.build_submission``. No
training, AutoGluon loading, network, or Kaggle upload.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.churn_ml.competition_assets_v1 import (
    DEFAULT_ID_COLUMN,
    DEFAULT_TARGET_COLUMN,
    SAMPLE_SUBMISSION_PATH,
    ensure_competition_row_identity,
    resolve_competition_assets,
    submission_ids_from_identity,
)
from src.churn_ml.deployment_v1_artifacts import build_submission
from src.churn_ml.prediction_candidates.contract_v1 import (
    POSITIVE_CLASS_LABEL,
    PROBABILITY_SEMANTICS,
    UNAVAILABLE,
    CandidateConflictError,
    PredictionCandidateError,
    file_sha256,
    load_candidate_package,
    repository_relative_path,
    resolve_under_repository,
    utc_now,
    validate_candidate_package,
)
from src.churn_ml.research_data import canonical_sha256


SUBMISSION_ROOT_RELATIVE = "artifacts/candidate_submissions"
SUBMISSION_SCHEMA_VERSION = "candidate_submission_v1"
SUCCESS_FILENAME = "_SUCCESS"
MANIFEST_FILENAME = "submission_manifest.json"
SOURCE_TYPE = "canonical_prediction_candidate"
# Keep local to avoid importing blending during prediction_candidates import.
BLEND_SOURCE_KIND = "canonical_probability_blend_v1"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class CandidateSubmissionError(RuntimeError):
    """Raised when candidate submission readiness or generation fails."""

    def __init__(self, message: str, *, reason_code: str = "candidate_submission") -> None:
        self.reason_code = reason_code
        super().__init__(message)


def normalize_submission_root_relative(root_relative: str | None) -> str:
    relative = root_relative or SUBMISSION_ROOT_RELATIVE
    if not relative or relative.startswith("/") or PurePosixPath(relative).is_absolute():
        raise CandidateSubmissionError(
            f"Absolute or empty submission root rejected: {relative!r}",
            reason_code="path_traversal",
        )
    if ".." in PurePosixPath(relative).parts:
        raise CandidateSubmissionError(
            f"Path traversal rejected: {relative!r}",
            reason_code="path_traversal",
        )
    return PurePosixPath(relative).as_posix()


def _resolve_candidates_root(
    repository_root: Path, candidates_root_relative: str | None
) -> str:
    relative = candidates_root_relative or "artifacts/prediction_candidates"
    if ".." in PurePosixPath(relative).parts or PurePosixPath(relative).is_absolute():
        raise CandidateSubmissionError(
            f"Path traversal rejected: {relative!r}",
            reason_code="path_traversal",
        )
    return PurePosixPath(relative).as_posix()


def _load_package(
    candidate_id: str,
    *,
    repository_root: Path,
    candidates_root_relative: str | None,
):
    if ".." in candidate_id or "/" in candidate_id or "\\" in candidate_id:
        raise CandidateSubmissionError(
            f"Invalid candidate_id: {candidate_id!r}",
            reason_code="path_traversal",
        )
    relative = _resolve_candidates_root(repository_root, candidates_root_relative)
    package_dir = resolve_under_repository(f"{relative}/{candidate_id}", repository_root)
    try:
        return load_candidate_package(
            package_dir,
            repository_root=repository_root,
            candidates_root_relative=relative,
        )
    except PredictionCandidateError as error:
        raise CandidateSubmissionError(
            str(error),
            reason_code=getattr(error, "reason_code", "candidate_invalid"),
        ) from error


def extract_final_deployment_threshold(source_metadata: Mapping[str, Any]) -> float | None:
    for key in ("final_deployment_threshold", "final_threshold"):
        if key in source_metadata and source_metadata[key] is not None:
            try:
                return float(source_metadata[key])
            except (TypeError, ValueError):
                return float("nan")
    return None


def extract_final_deployment_weights(
    source_metadata: Mapping[str, Any],
) -> dict[str, float] | None:
    for key in ("final_deployment_weights", "final_weights"):
        value = source_metadata.get(key)
        if isinstance(value, Mapping) and value:
            try:
                return {str(k): float(v) for k, v in value.items()}
            except (TypeError, ValueError):
                return None
    return None


def evaluate_submission_readiness(
    candidate_id: str,
    *,
    repository_root: Path,
    candidates_root_relative: str | None = None,
    blend_root_relative: str | None = None,
    require_blend_artifact: bool = True,
) -> dict[str, Any]:
    """Evaluate whether a canonical prediction candidate can produce a submission."""
    root = repository_root.resolve()
    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    package = None
    try:
        package = _load_package(
            candidate_id,
            repository_root=root,
            candidates_root_relative=candidates_root_relative,
        )
        validate_candidate_package(package, repository_root=root)
    except (CandidateSubmissionError, PredictionCandidateError) as error:
        blockers.append(
            {
                "code": getattr(error, "reason_code", "candidate_invalid"),
                "message": str(error),
            }
        )
        return {
            "state": "blocked",
            "ready": False,
            "candidate_id": candidate_id,
            "threshold": None,
            "warnings": warnings,
            "blockers": blockers,
            "competition_identity": {},
        }

    if not (package.package_dir / "_SUCCESS").is_file():
        blockers.append(
            {"code": "success_missing", "message": "Candidate _SUCCESS is missing."}
        )

    manifest = package.manifest
    source_metadata = package.source_metadata
    if bool(manifest.get("exploratory")):
        warnings.append(
            {
                "code": "exploratory_candidate",
                "message": (
                    "Candidate is marked exploratory and must remain visibly "
                    "exploratory downstream."
                ),
            }
        )

    if manifest.get("positive_class_label") != POSITIVE_CLASS_LABEL:
        blockers.append(
            {
                "code": "positive_class_label_invalid",
                "message": "positive_class_label must be 1.",
            }
        )
    if manifest.get("probability_semantics") != PROBABILITY_SEMANTICS:
        blockers.append(
            {
                "code": "probability_semantics_invalid",
                "message": "probability_semantics must be P(y=1).",
            }
        )

    probs = package.test["probability_positive"].to_numpy(dtype=np.float64)
    if len(probs) != int(manifest["test_row_count"]):
        blockers.append(
            {
                "code": "test_row_count_mismatch",
                "message": "Test prediction row count is not exact.",
            }
        )
    if not np.isfinite(probs).all() or np.any(probs < 0.0) or np.any(probs > 1.0):
        blockers.append(
            {
                "code": "probabilities_invalid",
                "message": "Test probabilities must be finite and within [0, 1].",
            }
        )

    competition = resolve_competition_assets(
        root, dataset_version=str(manifest["dataset_id"])
    )
    competition_identity = {
        "ready": competition.ready,
        "ordered_id_sha256": competition.ordered_id_sha256,
        "test_anchor_hash": competition.test_anchor_hash,
        "sample_expected_rows": competition.sample_expected_rows,
        "blocking_reasons": list(competition.blocking_reasons),
    }
    if not competition.ready:
        blockers.append(
            {
                "code": "competition_assets_invalid",
                "message": (
                    "Competition assets are not ready: "
                    + "; ".join(competition.blocking_reasons)
                ),
            }
        )
    else:
        if manifest.get("test_anchor_hash") != competition.test_anchor_hash:
            blockers.append(
                {
                    "code": "test_anchor_hash_mismatch",
                    "message": "test_anchor_hash does not match competition assets.",
                }
            )
        ordered = manifest.get("ordered_submission_id_hash")
        if ordered in (None, UNAVAILABLE) or ordered != competition.ordered_id_sha256:
            blockers.append(
                {
                    "code": "ordered_submission_id_hash_mismatch",
                    "message": (
                        "ordered_submission_id_hash does not match competition assets."
                    ),
                }
            )
        if int(manifest["test_row_count"]) != int(competition.sample_expected_rows or -1):
            blockers.append(
                {
                    "code": "test_row_count_mismatch",
                    "message": "Test row count differs from sample submission.",
                }
            )

    threshold = extract_final_deployment_threshold(source_metadata)
    if threshold is None:
        blockers.append(
            {
                "code": "final_threshold_missing",
                "message": (
                    "Final deployment threshold is not recorded on the candidate."
                ),
            }
        )
    elif not np.isfinite(threshold) or not (0.0 <= float(threshold) <= 1.0):
        blockers.append(
            {
                "code": "threshold_invalid",
                "message": "Final deployment threshold must be finite and in [0, 1].",
            }
        )

    if manifest.get("source_kind") == BLEND_SOURCE_KIND:
        weights = extract_final_deployment_weights(source_metadata)
        if weights is None:
            blockers.append(
                {
                    "code": "final_weights_missing",
                    "message": "Blend candidate lacks final deployment weights.",
                }
            )
        parent_ids = list(source_metadata.get("parent_candidate_ids") or [])
        parent_hashes = list(source_metadata.get("parent_manifest_hashes") or [])
        if not parent_ids or len(parent_ids) != len(parent_hashes):
            blockers.append(
                {
                    "code": "parent_mismatch",
                    "message": "Blend parent candidate references are incomplete.",
                }
            )
        else:
            candidates_rel = _resolve_candidates_root(
                root, candidates_root_relative
            )
            for parent_id, expected_hash in zip(parent_ids, parent_hashes, strict=True):
                try:
                    parent = _load_package(
                        str(parent_id),
                        repository_root=root,
                        candidates_root_relative=candidates_rel,
                    )
                    validate_candidate_package(parent, repository_root=root)
                    actual = file_sha256(
                        parent.package_dir / "candidate_manifest.json"
                    )
                    if actual != expected_hash:
                        blockers.append(
                            {
                                "code": "parent_hash_mismatch",
                                "message": (
                                    f"Parent manifest hash mismatch for {parent_id}."
                                ),
                            }
                        )
                except (CandidateSubmissionError, PredictionCandidateError) as error:
                    blockers.append(
                        {
                            "code": "parent_invalid",
                            "message": f"Invalid parent candidate {parent_id}: {error}",
                        }
                    )
        blend_id = source_metadata.get("blend_id")
        if require_blend_artifact:
            if not blend_id:
                blockers.append(
                    {
                        "code": "blend_id_missing",
                        "message": "Blend candidate lacks blend_id.",
                    }
                )
            else:
                try:
                    from src.churn_ml.blending.artifact_v1 import load_blend_artifact

                    load_blend_artifact(
                        str(blend_id),
                        repository_root=root,
                        blend_root_relative=blend_root_relative,
                        candidates_root_relative=candidates_root_relative,
                    )
                except Exception as error:
                    blockers.append(
                        {
                            "code": getattr(
                                error, "reason_code", "blend_artifact_invalid"
                            ),
                            "message": f"Blend artifact validation failed: {error}",
                        }
                    )

    ready = not blockers
    return {
        "state": "ready" if ready else "blocked",
        "ready": ready,
        "candidate_id": candidate_id,
        "threshold": None if threshold is None or not np.isfinite(threshold) else float(threshold),
        "warnings": warnings,
        "blockers": blockers,
        "competition_identity": competition_identity,
        "source_kind": manifest.get("source_kind"),
        "exploratory": bool(manifest.get("exploratory")),
        "blend_id": source_metadata.get("blend_id"),
    }


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


def generate_candidate_submission(
    candidate_id: str,
    *,
    submission_id: str,
    repository_root: Path,
    candidates_root_relative: str | None = None,
    submission_root_relative: str | None = None,
    blend_root_relative: str | None = None,
) -> dict[str, Any]:
    """Generate a validated local submission artifact. No upload."""
    root = repository_root.resolve()
    if not SAFE_ID.match(submission_id):
        raise CandidateSubmissionError(
            f"Unsafe submission_id: {submission_id!r}",
            reason_code="submission_id_invalid",
        )
    readiness = evaluate_submission_readiness(
        candidate_id,
        repository_root=root,
        candidates_root_relative=candidates_root_relative,
        blend_root_relative=blend_root_relative,
        require_blend_artifact=True,
    )
    if not readiness["ready"]:
        raise CandidateSubmissionError(
            "Candidate is not submission-ready: "
            + "; ".join(item["message"] for item in readiness["blockers"]),
            reason_code="submission_not_ready",
        )

    package = _load_package(
        candidate_id,
        repository_root=root,
        candidates_root_relative=candidates_root_relative,
    )
    validate_candidate_package(package, repository_root=root)
    threshold = extract_final_deployment_threshold(package.source_metadata)
    if threshold is None:
        raise CandidateSubmissionError(
            "Final deployment threshold missing.",
            reason_code="final_threshold_missing",
        )
    threshold = float(threshold)

    competition = ensure_competition_row_identity(
        root, dataset_version=str(package.manifest["dataset_id"])
    )
    if not competition.ready:
        raise CandidateSubmissionError(
            "Competition assets are not ready.",
            reason_code="competition_assets_invalid",
        )
    sample_path = resolve_under_repository(SAMPLE_SUBMISSION_PATH, root)
    sample = pd.read_csv(sample_path)
    # Ensure int64 IDs for build_submission contract.
    sample[DEFAULT_ID_COLUMN] = sample[DEFAULT_ID_COLUMN].astype("int64")

    from src.churn_ml.competition_assets_v1 import load_row_identity_artifact
    from src.churn_ml.competition_assets_v1 import ROW_IDENTITY_ARTIFACT_PATH

    identity = load_row_identity_artifact(root, ROW_IDENTITY_ARTIFACT_PATH)
    expected_ids = submission_ids_from_identity(identity)
    if sample[DEFAULT_ID_COLUMN].tolist() != expected_ids:
        raise CandidateSubmissionError(
            "Sample submission ID order does not match row identity.",
            reason_code="submission_id_order_mismatch",
        )

    probs = package.test["probability_positive"].to_numpy(dtype=np.float64)
    if len(probs) != len(sample):
        raise CandidateSubmissionError(
            "Test probability row count differs from sample submission.",
            reason_code="test_row_count_mismatch",
        )
    labels = (probs >= threshold).astype(np.int8)
    submission = build_submission(
        sample,
        labels,
        id_column=DEFAULT_ID_COLUMN,
        target_column=DEFAULT_TARGET_COLUMN,
    )
    if submission[DEFAULT_ID_COLUMN].tolist() != expected_ids:
        raise CandidateSubmissionError(
            "Generated submission ID order is incorrect.",
            reason_code="submission_id_order_mismatch",
        )
    if set(submission[DEFAULT_TARGET_COLUMN].unique().tolist()) - {0, 1}:
        raise CandidateSubmissionError(
            "Submission labels must be binary 0/1.",
            reason_code="labels_invalid",
        )
    if submission[DEFAULT_ID_COLUMN].duplicated().any():
        raise CandidateSubmissionError(
            "Submission contains duplicate IDs.",
            reason_code="duplicate_ids",
        )

    submission_root_rel = normalize_submission_root_relative(submission_root_relative)
    submission_root = resolve_under_repository(submission_root_rel, root)
    submission_root.mkdir(parents=True, exist_ok=True)
    final_dir = submission_root / submission_id
    staging = submission_root / f".staging-{submission_id}-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        csv_bytes = submission.to_csv(index=False, lineterminator="\n").encode("utf-8")
        _atomic_write_bytes(staging / "submission.csv", csv_bytes)
        prediction_hash = canonical_sha256(probs.tolist())
        label_hash = canonical_sha256(labels.astype(int).tolist())
        source_metadata = {
            "schema_version": 1,
            "source_type": SOURCE_TYPE,
            "candidate_id": candidate_id,
            "blend_id": package.source_metadata.get("blend_id"),
            "threshold": threshold,
            "threshold_comparison": "greater_than_or_equal",
            "prediction_probability_sha256": prediction_hash,
            "prediction_label_sha256": label_hash,
            "competition_sample_submission_sha256": competition.sample_submission_sha256,
            "ordered_submission_id_hash": competition.ordered_id_sha256,
            "test_anchor_hash": competition.test_anchor_hash,
            "exploratory": bool(package.manifest.get("exploratory")),
            "exploratory_warning": (
                readiness["warnings"] if readiness["warnings"] else []
            ),
            "network_access": False,
            "kaggle_upload": False,
        }
        _write_json(staging / "source_metadata.json", source_metadata)
        manifest = {
            "schema_version": SUBMISSION_SCHEMA_VERSION,
            "submission_id": submission_id,
            "created_at_utc": utc_now(),
            "source_type": SOURCE_TYPE,
            "candidate_id": candidate_id,
            "blend_id": package.source_metadata.get("blend_id"),
            "threshold": threshold,
            "row_count": len(submission),
            "id_column": DEFAULT_ID_COLUMN,
            "target_column": DEFAULT_TARGET_COLUMN,
            "submission_csv_sha256": file_sha256(staging / "submission.csv"),
            "submission_csv_size_bytes": (staging / "submission.csv").stat().st_size,
            "source_metadata_sha256": file_sha256(staging / "source_metadata.json"),
            "competition_identity": {
                "sample_submission_path": SAMPLE_SUBMISSION_PATH,
                "sample_submission_sha256": competition.sample_submission_sha256,
                "ordered_id_sha256": competition.ordered_id_sha256,
                "test_anchor_hash": competition.test_anchor_hash,
            },
            "readiness_warnings": readiness["warnings"],
            "network_access": False,
            "kaggle_upload": False,
        }
        _write_json(staging / MANIFEST_FILENAME, manifest)
        success = {
            "schema_version": 1,
            "submission_id": submission_id,
            "status": "completed",
            "manifest_sha256": file_sha256(staging / MANIFEST_FILENAME),
            "completed_at_utc": utc_now(),
        }
        _write_json(staging / SUCCESS_FILENAME, success)

        if final_dir.exists():
            if not (final_dir / SUCCESS_FILENAME).is_file():
                raise CandidateSubmissionError(
                    f"Incomplete submission artifact exists: {final_dir}",
                    reason_code="submission_incomplete",
                )
            existing = json.loads(
                (final_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
            )
            if (
                existing.get("candidate_id") != candidate_id
                or existing.get("threshold") != threshold
                or existing.get("submission_csv_sha256")
                != manifest["submission_csv_sha256"]
            ):
                raise CandidateConflictError(
                    f"Submission {submission_id} already exists with different content."
                )
            shutil.rmtree(staging, ignore_errors=True)
            return {
                "ok": True,
                "artifacts_written": False,
                "idempotent": True,
                "submission_id": submission_id,
                "submission_path": repository_relative_path(final_dir, root),
                "candidate_id": candidate_id,
                "threshold": threshold,
                "row_count": len(submission),
                "readiness": readiness,
                "network_access": False,
                "kaggle_upload": False,
            }

        os.replace(staging, final_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "ok": True,
        "artifacts_written": True,
        "idempotent": False,
        "submission_id": submission_id,
        "submission_path": repository_relative_path(final_dir, root),
        "candidate_id": candidate_id,
        "threshold": threshold,
        "row_count": len(submission),
        "readiness": readiness,
        "network_access": False,
        "kaggle_upload": False,
    }
