"""Cheap UI eligibility projection for canonical candidate submission.

This is intentionally weaker than ``evaluate_submission_readiness``. It reads
only lightweight JSON/metadata and filesystem presence so ordinary Streamlit
reruns stay responsive. Strict payload, parent-hash, competition-identity, and
blend-artifact validation remain authoritative inside the generation Job.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from src.churn_ml.competition_assets_v1 import (
    ROW_IDENTITY_ARTIFACT_PATH,
    SAMPLE_SUBMISSION_PATH,
)
from src.churn_ml.control_panel.fs_signatures import (
    cheap_inventory_fingerprint,
    path_stat_mapping,
)
from src.churn_ml.control_panel.performance import (
    note_files_inspected,
    note_files_read,
    performance_stage,
    record_counter,
)
from src.churn_ml.prediction_candidates.contract_v1 import (
    CANDIDATE_ROOT_RELATIVE,
    MANIFEST_FILENAME,
    POSITIVE_CLASS_LABEL,
    PROBABILITY_SEMANTICS,
    SOURCE_METADATA_FILENAME,
    SUCCESS_FILENAME,
)
from src.churn_ml.prediction_candidates.submission_v1 import (
    BLEND_SOURCE_KIND,
    extract_final_deployment_threshold,
)


ELIGIBLE_STATE = "eligible_to_attempt"
BLOCKED_STATE = "blocked_before_generation"
REQUIRES_STRICT_NOTE = (
    "Metadata checks passed. Full candidate, parent, probability, row-identity, "
    "and blend-artifact validation will run inside the generation Job."
)
BLEND_ROOT_RELATIVE = "artifacts/prediction_blends"
BLEND_MANIFEST_FILENAME = "blend_manifest.json"


@dataclass(frozen=True)
class EligibilityIssue:
    code: str
    message: str


@dataclass(frozen=True)
class CandidateSubmissionEligibility:
    candidate_id: str
    state: str
    eligible_to_attempt: bool
    source_kind: str | None
    exploratory: bool
    threshold: float | None
    test_row_count: int | None
    dataset_id: str | None
    blend_id: str | None
    parent_candidate_ids: tuple[str, ...]
    blockers: tuple[EligibilityIssue, ...]
    warnings: tuple[EligibilityIssue, ...]
    strict_validation_required: bool
    note: str

    def to_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["blockers"] = [asdict(item) for item in self.blockers]
        payload["warnings"] = [asdict(item) for item in self.warnings]
        return payload


def eligibility_cache_fingerprint(
    candidate_id: str,
    *,
    repository_root: Path | str,
    candidates_root_relative: str | None = None,
    blend_root_relative: str | None = None,
) -> str:
    """Cheap cache key from metadata path stats only (no payload hashing)."""
    root = Path(repository_root).resolve()
    candidates_rel = _normalize_relative(
        candidates_root_relative or CANDIDATE_ROOT_RELATIVE
    )
    blend_rel = _normalize_relative(blend_root_relative or BLEND_ROOT_RELATIVE)
    package_dir = root / Path(*PurePosixPath(candidates_rel).parts) / candidate_id
    entries: list[dict[str, Any]] = [
        path_stat_mapping(package_dir / MANIFEST_FILENAME, relative="manifest"),
        path_stat_mapping(
            package_dir / SOURCE_METADATA_FILENAME, relative="source_metadata"
        ),
        path_stat_mapping(package_dir / SUCCESS_FILENAME, relative="success"),
        path_stat_mapping(
            root / Path(*PurePosixPath(SAMPLE_SUBMISSION_PATH).parts),
            relative="sample_submission",
        ),
        path_stat_mapping(
            root / Path(*PurePosixPath(ROW_IDENTITY_ARTIFACT_PATH).parts),
            relative="row_identity",
        ),
    ]
    source_metadata = _safe_json_object(package_dir / SOURCE_METADATA_FILENAME)
    blend_id = str(source_metadata.get("blend_id") or "").strip()
    if blend_id:
        blend_dir = root / Path(*PurePosixPath(blend_rel).parts) / blend_id
        entries.append(
            path_stat_mapping(
                blend_dir / BLEND_MANIFEST_FILENAME, relative=f"blend/{blend_id}/manifest"
            )
        )
        entries.append(
            path_stat_mapping(
                blend_dir / SUCCESS_FILENAME, relative=f"blend/{blend_id}/success"
            )
        )
    parent_ids = source_metadata.get("parent_candidate_ids") or []
    if isinstance(parent_ids, Sequence) and not isinstance(parent_ids, (str, bytes)):
        for parent_id in parent_ids:
            parent_dir = (
                root / Path(*PurePosixPath(candidates_rel).parts) / str(parent_id)
            )
            entries.append(
                path_stat_mapping(
                    parent_dir / MANIFEST_FILENAME,
                    relative=f"parent/{parent_id}/manifest",
                )
            )
            entries.append(
                path_stat_mapping(
                    parent_dir / SUCCESS_FILENAME,
                    relative=f"parent/{parent_id}/success",
                )
            )
    return cheap_inventory_fingerprint(
        [{"path": item.get("path"), **item} for item in entries]
    )


def project_candidate_submission_eligibility(
    candidate_id: str,
    *,
    repository_root: Path | str,
    candidates_root_relative: str | None = None,
    blend_root_relative: str | None = None,
) -> CandidateSubmissionEligibility:
    """Metadata-only eligibility projection for UI selectors and ordinary reruns."""
    with performance_stage("run_selected_eligibility"):
        record_counter("selected_eligibility_evaluations", 1)
        return _project_eligibility(
            candidate_id,
            repository_root=repository_root,
            candidates_root_relative=candidates_root_relative,
            blend_root_relative=blend_root_relative,
        )


def _project_eligibility(
    candidate_id: str,
    *,
    repository_root: Path | str,
    candidates_root_relative: str | None,
    blend_root_relative: str | None,
) -> CandidateSubmissionEligibility:
    root = Path(repository_root).resolve()
    blockers: list[EligibilityIssue] = []
    warnings: list[EligibilityIssue] = []
    if ".." in candidate_id or "/" in candidate_id or "\\" in candidate_id:
        return _blocked(
            candidate_id,
            blockers=[
                EligibilityIssue(
                    code="candidate_id_invalid",
                    message=f"Invalid candidate_id: {candidate_id!r}",
                )
            ],
        )

    candidates_rel = _normalize_relative(
        candidates_root_relative or CANDIDATE_ROOT_RELATIVE
    )
    blend_rel = _normalize_relative(blend_root_relative or BLEND_ROOT_RELATIVE)
    package_dir = root / Path(*PurePosixPath(candidates_rel).parts) / candidate_id
    note_files_inspected(3)
    if not package_dir.is_dir():
        blockers.append(
            EligibilityIssue(
                code="candidate_missing",
                message="Candidate package directory does not exist.",
            )
        )
        return _blocked(candidate_id, blockers=blockers)

    manifest_path = package_dir / MANIFEST_FILENAME
    metadata_path = package_dir / SOURCE_METADATA_FILENAME
    success_path = package_dir / SUCCESS_FILENAME
    if not manifest_path.is_file():
        blockers.append(
            EligibilityIssue(
                code="manifest_missing",
                message=f"Missing {MANIFEST_FILENAME}.",
            )
        )
    if not metadata_path.is_file():
        blockers.append(
            EligibilityIssue(
                code="source_metadata_missing",
                message=f"Missing {SOURCE_METADATA_FILENAME}.",
            )
        )
    if not success_path.is_file():
        blockers.append(
            EligibilityIssue(
                code="success_missing",
                message=f"Missing {SUCCESS_FILENAME}.",
            )
        )

    manifest, manifest_ok = _load_json_object(manifest_path)
    source_metadata, metadata_ok = _load_json_object(metadata_path)
    if manifest_path.is_file() and not manifest_ok:
        blockers.append(
            EligibilityIssue(
                code="manifest_invalid",
                message="Candidate manifest is missing or not a JSON object.",
            )
        )
    if metadata_path.is_file() and not metadata_ok:
        blockers.append(
            EligibilityIssue(
                code="source_metadata_invalid",
                message="source_metadata.json is missing or not a JSON object.",
            )
        )
    if manifest_path.is_file():
        note_files_read(1)
    if metadata_path.is_file():
        note_files_read(1)

    recorded_id = str(manifest.get("candidate_id") or "").strip()
    if manifest and recorded_id and recorded_id != candidate_id:
        blockers.append(
            EligibilityIssue(
                code="candidate_id_mismatch",
                message=(
                    f"Manifest candidate_id {recorded_id!r} does not match "
                    f"requested {candidate_id!r}."
                ),
            )
        )

    source_kind = str(manifest.get("source_kind") or "") or None
    dataset_id = str(manifest.get("dataset_id") or "") or None
    exploratory = bool(manifest.get("exploratory", False))
    test_row_count = _as_int(manifest.get("test_row_count"))
    threshold = extract_final_deployment_threshold(source_metadata)
    blend_id = str(source_metadata.get("blend_id") or "").strip() or None
    parent_ids_raw = source_metadata.get("parent_candidate_ids") or []
    parent_hashes = source_metadata.get("parent_manifest_hashes") or []
    parent_ids: tuple[str, ...] = ()
    if isinstance(parent_ids_raw, Sequence) and not isinstance(
        parent_ids_raw, (str, bytes)
    ):
        parent_ids = tuple(str(item) for item in parent_ids_raw)

    if exploratory:
        warnings.append(
            EligibilityIssue(
                code="exploratory_candidate",
                message=(
                    "Candidate is marked exploratory and must remain visibly "
                    "exploratory downstream."
                ),
            )
        )

    if manifest:
        if manifest.get("positive_class_label") != POSITIVE_CLASS_LABEL:
            blockers.append(
                EligibilityIssue(
                    code="positive_class_label_invalid",
                    message="positive_class_label must be 1.",
                )
            )
        if manifest.get("probability_semantics") != PROBABILITY_SEMANTICS:
            blockers.append(
                EligibilityIssue(
                    code="probability_semantics_invalid",
                    message="probability_semantics must be P(y=1).",
                )
            )

    if threshold is None:
        blockers.append(
            EligibilityIssue(
                code="final_threshold_missing",
                message="Final deployment threshold is not recorded on the candidate.",
            )
        )
    else:
        try:
            threshold_value = float(threshold)
        except (TypeError, ValueError):
            threshold_value = float("nan")
            blockers.append(
                EligibilityIssue(
                    code="threshold_invalid",
                    message="Final deployment threshold must be finite and in [0, 1].",
                )
            )
            threshold = None
        else:
            if not (threshold_value == threshold_value) or not (
                0.0 <= threshold_value <= 1.0
            ):
                blockers.append(
                    EligibilityIssue(
                        code="threshold_invalid",
                        message="Final deployment threshold must be finite and in [0, 1].",
                    )
                )
                threshold = None
            else:
                threshold = threshold_value

    if test_row_count is None or test_row_count <= 0:
        blockers.append(
            EligibilityIssue(
                code="test_row_count_invalid",
                message="test_row_count must be a positive integer in metadata.",
            )
        )

    sample_path = root / Path(*PurePosixPath(SAMPLE_SUBMISSION_PATH).parts)
    identity_path = root / Path(*PurePosixPath(ROW_IDENTITY_ARTIFACT_PATH).parts)
    note_files_inspected(2)
    if not sample_path.is_file() or not identity_path.is_file():
        blockers.append(
            EligibilityIssue(
                code="competition_identity_missing",
                message=(
                    "Competition identity metadata is not present "
                    "(sample submission and/or row-identity artifact)."
                ),
            )
        )

    if source_kind == BLEND_SOURCE_KIND:
        if not blend_id:
            blockers.append(
                EligibilityIssue(
                    code="blend_id_missing",
                    message="Blend candidate lacks blend_id in source metadata.",
                )
            )
        if not parent_ids:
            blockers.append(
                EligibilityIssue(
                    code="parent_refs_missing",
                    message="Blend candidate lacks parent_candidate_ids.",
                )
            )
        if not isinstance(parent_hashes, Sequence) or isinstance(
            parent_hashes, (str, bytes)
        ):
            blockers.append(
                EligibilityIssue(
                    code="parent_hashes_missing",
                    message="Blend candidate lacks parent_manifest_hashes.",
                )
            )
            parent_hashes = []
        elif len(parent_ids) != len(parent_hashes):
            blockers.append(
                EligibilityIssue(
                    code="parent_mismatch",
                    message="Blend parent candidate references are incomplete.",
                )
            )
        for parent_id in parent_ids:
            parent_dir = (
                root / Path(*PurePosixPath(candidates_rel).parts) / parent_id
            )
            note_files_inspected(2)
            if not (parent_dir / MANIFEST_FILENAME).is_file():
                blockers.append(
                    EligibilityIssue(
                        code="parent_manifest_missing",
                        message=f"Missing parent manifest for {parent_id}.",
                    )
                )
            if not (parent_dir / SUCCESS_FILENAME).is_file():
                blockers.append(
                    EligibilityIssue(
                        code="parent_success_missing",
                        message=f"Missing parent _SUCCESS for {parent_id}.",
                    )
                )
        if blend_id:
            blend_dir = root / Path(*PurePosixPath(blend_rel).parts) / blend_id
            note_files_inspected(2)
            if not blend_dir.is_dir():
                blockers.append(
                    EligibilityIssue(
                        code="blend_artifact_missing",
                        message=f"Materialized blend directory missing for {blend_id}.",
                    )
                )
            else:
                if not (blend_dir / BLEND_MANIFEST_FILENAME).is_file():
                    blockers.append(
                        EligibilityIssue(
                            code="blend_manifest_missing",
                            message=f"Missing blend_manifest.json for {blend_id}.",
                        )
                    )
                if not (blend_dir / SUCCESS_FILENAME).is_file():
                    blockers.append(
                        EligibilityIssue(
                            code="blend_success_missing",
                            message=f"Missing blend _SUCCESS for {blend_id}.",
                        )
                    )

    eligible = not blockers
    return CandidateSubmissionEligibility(
        candidate_id=candidate_id,
        state=ELIGIBLE_STATE if eligible else BLOCKED_STATE,
        eligible_to_attempt=eligible,
        source_kind=source_kind,
        exploratory=exploratory,
        threshold=threshold,
        test_row_count=test_row_count,
        dataset_id=dataset_id,
        blend_id=blend_id,
        parent_candidate_ids=parent_ids,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        strict_validation_required=True,
        note=REQUIRES_STRICT_NOTE if eligible else (
            "Metadata blockers must be resolved before generation can be attempted."
        ),
    )


def eligibility_label(state: str) -> str:
    mapping = {
        ELIGIBLE_STATE: "Eligible to attempt generation",
        BLOCKED_STATE: "Blocked before generation",
    }
    return mapping.get(str(state), str(state).replace("_", " ").title())


def _blocked(
    candidate_id: str,
    *,
    blockers: Sequence[EligibilityIssue],
    warnings: Sequence[EligibilityIssue] = (),
) -> CandidateSubmissionEligibility:
    return CandidateSubmissionEligibility(
        candidate_id=candidate_id,
        state=BLOCKED_STATE,
        eligible_to_attempt=False,
        source_kind=None,
        exploratory=False,
        threshold=None,
        test_row_count=None,
        dataset_id=None,
        blend_id=None,
        parent_candidate_ids=(),
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        strict_validation_required=True,
        note="Metadata blockers must be resolved before generation can be attempted.",
    )


def _normalize_relative(value: str) -> str:
    text = str(value or "").replace("\\", "/").strip()
    return PurePosixPath(text).as_posix()


def _safe_json_object(path: Path) -> dict[str, Any]:
    payload, ok = _load_json_object(path)
    return payload if ok else {}


def _load_json_object(path: Path) -> tuple[dict[str, Any], bool]:
    if not path.is_file():
        return {}, False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}, False
    if not isinstance(payload, dict):
        return {}, False
    return payload, True


def _as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "BLOCKED_STATE",
    "BLEND_ROOT_RELATIVE",
    "CandidateSubmissionEligibility",
    "ELIGIBLE_STATE",
    "EligibilityIssue",
    "REQUIRES_STRICT_NOTE",
    "eligibility_cache_fingerprint",
    "eligibility_label",
    "project_candidate_submission_eligibility",
]
