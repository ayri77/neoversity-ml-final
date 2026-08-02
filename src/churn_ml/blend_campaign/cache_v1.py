"""Process-local caches for source-only canonical blend diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.churn_ml.blending import artifact_v1
from src.churn_ml.blending.compatibility_v1 import CompatibleCandidateSet
from src.churn_ml.prediction_candidates.contract_v1 import (
    MANIFEST_FILENAME,
    file_sha256,
)


_DIVERSITY_CACHE: dict[
    tuple[str, tuple[str, ...], tuple[str, ...]], dict[str, Any]
] = {}
_ORIGINAL_ANALYZE_DIVERSITY = artifact_v1.analyze_diversity
_INSTALLED = False


def install_diversity_cache() -> None:
    """Cache diversity by root, ordered IDs, and immutable manifest hashes."""
    global _INSTALLED
    if _INSTALLED:
        return
    artifact_v1.analyze_diversity = _cached_analyze_diversity
    _INSTALLED = True


def _cached_analyze_diversity(pool: CompatibleCandidateSet) -> dict[str, Any]:
    hashes = tuple(
        file_sha256(Path(package.package_dir) / MANIFEST_FILENAME)
        for package in pool.packages
    )
    key = (str(pool.repository_root.resolve()), tuple(pool.candidate_ids), hashes)
    if key not in _DIVERSITY_CACHE:
        _DIVERSITY_CACHE[key] = _ORIGINAL_ANALYZE_DIVERSITY(pool)
    return _DIVERSITY_CACHE[key]


def clear_diversity_cache() -> None:
    """Clear process-local state for tests or a long-lived embedding process."""
    _DIVERSITY_CACHE.clear()


__all__ = ["clear_diversity_cache", "install_diversity_cache"]
