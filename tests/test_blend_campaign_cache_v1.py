"""Cache behavior for repeated immutable candidate sets."""

from __future__ import annotations

from pathlib import Path

from src.churn_ml.blend_campaign import cache_v1


def test_diversity_cache_reuses_exact_manifest_identity(
    monkeypatch, tmp_path: Path
) -> None:
    manifest_a = tmp_path / "a" / "candidate_manifest.json"
    manifest_b = tmp_path / "b" / "candidate_manifest.json"
    manifest_a.parent.mkdir()
    manifest_b.parent.mkdir()
    manifest_a.write_text('{"candidate_id":"a"}', encoding="utf-8")
    manifest_b.write_text('{"candidate_id":"b"}', encoding="utf-8")

    class Package:
        def __init__(self, path: Path) -> None:
            self.package_dir = path

    class Pool:
        repository_root = tmp_path
        candidate_ids = ("a", "b")
        packages = (Package(manifest_a.parent), Package(manifest_b.parent))

    calls = []
    monkeypatch.setattr(
        cache_v1,
        "_ORIGINAL_ANALYZE_DIVERSITY",
        lambda pool: calls.append(pool) or {"value": len(calls)},
    )
    cache_v1.clear_diversity_cache()
    assert cache_v1._cached_analyze_diversity(Pool())["value"] == 1
    assert cache_v1._cached_analyze_diversity(Pool())["value"] == 1
    assert len(calls) == 1

    manifest_b.write_text('{"candidate_id":"b","changed":true}', encoding="utf-8")
    assert cache_v1._cached_analyze_diversity(Pool())["value"] == 2
    assert len(calls) == 2
