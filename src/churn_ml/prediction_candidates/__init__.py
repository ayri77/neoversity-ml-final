"""Canonical Prediction Candidate packages for source-neutral blending.

This package defines the immutable ``prediction_candidate_v1`` contract and
source adapters that materialize it. AutoGluon-specific loading stays in
``autogluon_v1`` so the main application environment never imports AutoGluon.

A future Research v2 adapter must produce the same contract. When Research v2
emits repeated OOF rows, that adapter will normalize to one probability per
``row_position`` under a deterministic declared policy (for example mean
probability across repeats) and record that policy in ``oof_protocol``.
AutoGluon and Research v2 do not need shared fold assignments to coexist as
canonical candidates.
"""

from __future__ import annotations

from src.churn_ml.prediction_candidates.contract_v1 import (
    CANDIDATE_ROOT_RELATIVE,
    CANDIDATE_TYPE,
    MANIFEST_FILENAME,
    OOF_FILENAME,
    SCHEMA_VERSION,
    SUCCESS_FILENAME,
    SOURCE_METADATA_FILENAME,
    TEST_FILENAME,
    UNAVAILABLE,
    CandidateConflictError,
    CandidatePackage,
    CandidateValidationError,
    FileReference,
    PredictionCandidateError,
    build_candidate_id,
    candidate_summary,
    create_candidate_package,
    discover_candidate_packages,
    load_aligned_oof_probabilities,
    load_aligned_test_probabilities,
    load_candidate_manifest,
    load_candidate_package,
    validate_candidate_package,
)
__all__ = [
    "CANDIDATE_ROOT_RELATIVE",
    "CANDIDATE_TYPE",
    "MANIFEST_FILENAME",
    "OOF_FILENAME",
    "SCHEMA_VERSION",
    "SUCCESS_FILENAME",
    "SOURCE_METADATA_FILENAME",
    "TEST_FILENAME",
    "UNAVAILABLE",
    "CandidateConflictError",
    "CandidatePackage",
    "CandidateValidationError",
    "FileReference",
    "PredictionCandidateError",
    "build_candidate_id",
    "candidate_summary",
    "create_candidate_package",
    "discover_candidate_packages",
    "load_aligned_oof_probabilities",
    "load_aligned_test_probabilities",
    "load_candidate_manifest",
    "load_candidate_package",
    "validate_candidate_package",
]
