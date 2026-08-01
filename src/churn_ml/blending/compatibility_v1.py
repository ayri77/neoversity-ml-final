"""Compatibility gate for multi-candidate probability blending."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import numpy as np

from src.churn_ml.prediction_candidates.contract_v1 import (
    CANDIDATE_ROOT_RELATIVE,
    MANIFEST_FILENAME,
    CandidatePackage,
    PredictionCandidateError,
    discover_candidate_packages,
    file_sha256,
    load_aligned_oof_probabilities,
    load_aligned_test_probabilities,
    load_candidate_package,
    normalize_candidates_root_relative,
    resolve_under_repository,
    validate_candidate_package,
)


MIN_CANDIDATES = 2
DEFAULT_MAX_CANDIDATES = 10

COMPATIBILITY_KEYS = (
    "schema_version",
    "target_hash",
    "train_row_count",
    "test_row_count",
    "train_anchor_hash",
    "train_row_position_hash",
    "test_anchor_hash",
    "test_row_position_hash",
    "positive_class_label",
    "probability_semantics",
)


class BlendCompatibilityError(ValueError):
    """Raised when selected candidates cannot form a valid blend pool."""

    def __init__(self, message: str, *, reason_code: str = "blend_incompatible") -> None:
        self.reason_code = reason_code
        super().__init__(message)


@dataclass(frozen=True)
class CompatibleCandidateSet:
    repository_root: Path
    candidates_root_relative: str
    packages: tuple[CandidatePackage, ...]
    candidate_ids: tuple[str, ...]
    oof_matrix: np.ndarray
    test_matrix: np.ndarray
    target: np.ndarray
    row_positions: np.ndarray
    test_row_positions: np.ndarray
    exploratory: bool
    compatibility: dict[str, Any]

    @property
    def n_candidates(self) -> int:
        return len(self.packages)

    def package_map(self) -> dict[str, CandidatePackage]:
        return {package.candidate_id: package for package in self.packages}


def resolve_candidates_root(
    repository_root: Path,
    candidates_root: str | Path | None,
) -> tuple[str, Path]:
    root = repository_root.resolve()
    if candidates_root is None:
        relative = CANDIDATE_ROOT_RELATIVE
    elif isinstance(candidates_root, Path) and candidates_root.is_absolute():
        relative = PurePosixPath(
            candidates_root.resolve().relative_to(root).as_posix()
        ).as_posix()
        relative = normalize_candidates_root_relative(relative)
    else:
        relative = normalize_candidates_root_relative(
            Path(candidates_root).as_posix().replace("\\", "/")
        )
    absolute = resolve_under_repository(relative, root)
    return relative, absolute


def select_candidate_ids(candidate_ids: Sequence[str]) -> tuple[str, ...]:
    if len(candidate_ids) < MIN_CANDIDATES:
        raise BlendCompatibilityError(
            f"At least {MIN_CANDIDATES} candidates are required.",
            reason_code="too_few_candidates",
        )
    if len(candidate_ids) > DEFAULT_MAX_CANDIDATES:
        raise BlendCompatibilityError(
            f"At most {DEFAULT_MAX_CANDIDATES} candidates are allowed by default.",
            reason_code="too_many_candidates",
        )
    ordered = tuple(str(item) for item in candidate_ids)
    if len(set(ordered)) != len(ordered):
        raise BlendCompatibilityError(
            "Duplicate candidate IDs are rejected.",
            reason_code="duplicate_candidate_ids",
        )
    return ordered


def load_compatible_candidates(
    candidate_ids: Sequence[str],
    *,
    repository_root: Path,
    candidates_root: str | Path | None = None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> CompatibleCandidateSet:
    """Load, validate, and gate an explicit candidate list."""
    if max_candidates < MIN_CANDIDATES:
        raise BlendCompatibilityError(
            "max_candidates must be at least 2.",
            reason_code="invalid_max_candidates",
        )
    if len(candidate_ids) > max_candidates:
        raise BlendCompatibilityError(
            f"Candidate count {len(candidate_ids)} exceeds limit {max_candidates}.",
            reason_code="too_many_candidates",
        )
    ordered_ids = select_candidate_ids(candidate_ids)
    root = repository_root.resolve()
    relative_root, absolute_root = resolve_candidates_root(root, candidates_root)
    if not absolute_root.is_dir():
        raise BlendCompatibilityError(
            f"Candidate root does not exist: {relative_root}",
            reason_code="candidate_root_missing",
        )

    packages: list[CandidatePackage] = []
    for candidate_id in ordered_ids:
        package_dir = absolute_root / candidate_id
        if not package_dir.is_dir():
            raise BlendCompatibilityError(
                f"Candidate package not found: {relative_root}/{candidate_id}",
                reason_code="candidate_missing",
            )
        try:
            package = load_candidate_package(
                package_dir,
                repository_root=root,
                candidates_root_relative=relative_root,
            )
        except PredictionCandidateError as error:
            raise BlendCompatibilityError(
                f"Candidate {candidate_id} failed contract validation: {error}",
                reason_code=getattr(error, "reason_code", "candidate_invalid"),
            ) from error
        validate_candidate_package(package, repository_root=root)
        packages.append(package)

    reference = packages[0].manifest
    for package in packages[1:]:
        for key in COMPATIBILITY_KEYS:
            if package.manifest[key] != reference[key]:
                raise BlendCompatibilityError(
                    f"Compatibility mismatch on {key}: "
                    f"{packages[0].candidate_id}={reference[key]!r} vs "
                    f"{package.candidate_id}={package.manifest[key]!r}",
                    reason_code=f"{key}_mismatch",
                )
        ref_sub = reference["ordered_submission_id_hash"]
        other_sub = package.manifest["ordered_submission_id_hash"]
        if ref_sub != "unavailable" and other_sub != "unavailable" and ref_sub != other_sub:
            raise BlendCompatibilityError(
                "ordered_submission_id_hash mismatch among candidates.",
                reason_code="ordered_submission_id_hash_mismatch",
            )

    target = packages[0].oof["target"].astype("int64").to_numpy()
    row_positions = packages[0].oof["row_position"].astype("int64").to_numpy()
    test_row_positions = packages[0].test["row_position"].astype("int64").to_numpy()
    for package in packages[1:]:
        if not np.array_equal(
            package.oof["target"].astype("int64").to_numpy(), target
        ):
            raise BlendCompatibilityError(
                "OOF target values are not identical across candidates.",
                reason_code="target_values_mismatch",
            )
        if not np.array_equal(
            package.oof["row_position"].astype("int64").to_numpy(), row_positions
        ):
            raise BlendCompatibilityError(
                "OOF row_position alignment mismatch.",
                reason_code="oof_row_position_mismatch",
            )
        if not np.array_equal(
            package.test["row_position"].astype("int64").to_numpy(),
            test_row_positions,
        ):
            raise BlendCompatibilityError(
                "Test row_position alignment mismatch.",
                reason_code="test_row_position_mismatch",
            )

    oof_matrix = np.column_stack(
        [
            load_aligned_oof_probabilities(package).to_numpy(dtype=np.float64)
            for package in packages
        ]
    )
    test_matrix = np.column_stack(
        [
            load_aligned_test_probabilities(package).to_numpy(dtype=np.float64)
            for package in packages
        ]
    )
    exploratory = any(bool(package.manifest.get("exploratory")) for package in packages)

    parent_summaries = [
        {
            "candidate_id": package.candidate_id,
            "dataset_id": package.manifest["dataset_id"],
            "parent_dataset_id": package.manifest["parent_dataset_id"],
            "target_dependency": package.manifest["target_dependency"],
            "exploratory": bool(package.manifest.get("exploratory")),
            "source_kind": package.manifest["source_kind"],
            "source_model_name": package.manifest["source_model_name"],
            "oof_protocol": package.manifest["oof_protocol"],
            "manifest_sha256": _manifest_sha256(package),
        }
        for package in packages
    ]
    compatibility = {
        "schema_version": 1,
        "ok": True,
        "candidate_ids": list(ordered_ids),
        "shared_identity": {key: reference[key] for key in COMPATIBILITY_KEYS},
        "ordered_submission_id_hash": reference["ordered_submission_id_hash"],
        "exploratory": exploratory,
        "parents": parent_summaries,
        "allowed_to_differ": [
            "dataset_id",
            "parent_dataset_id",
            "feature set",
            "model family",
            "source_kind",
            "oof_protocol",
        ],
    }
    return CompatibleCandidateSet(
        repository_root=root,
        candidates_root_relative=relative_root,
        packages=tuple(packages),
        candidate_ids=ordered_ids,
        oof_matrix=oof_matrix,
        test_matrix=test_matrix,
        target=target,
        row_positions=row_positions,
        test_row_positions=test_row_positions,
        exploratory=exploratory,
        compatibility=compatibility,
    )


def list_candidates(
    *,
    repository_root: Path,
    candidates_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    root = repository_root.resolve()
    relative_root, _absolute = resolve_candidates_root(root, candidates_root)
    return discover_candidate_packages(
        root, candidates_root_relative=relative_root
    )


def _manifest_sha256(package: CandidatePackage) -> str:
    return file_sha256(package.package_dir / MANIFEST_FILENAME)
