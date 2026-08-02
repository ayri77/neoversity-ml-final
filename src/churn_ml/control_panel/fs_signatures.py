"""Cheap filesystem signatures and memoized hashing for Control Panel caches.

Immutable filesystem artifacts remain authoritative. These helpers only avoid
re-scanning or re-hashing unchanged files during UI inventory projections.

Cache contract notes:
- what is cached: path → (size, mtime_ns) → sha256 digest
- cache key: resolved absolute path + size + mtime_ns
- invalidation: any size or mtime change forces recalculation
- running/incomplete artifacts: callers must include status/marker stats so
  incomplete packages are not treated as permanently complete
- manual refresh: ``clear_sha_memo()`` or ``st.cache_data.clear()`` on the
  relevant inventory cache only
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.churn_ml.research_data import canonical_sha256


_SHA_MEMO: dict[tuple[str, int, int], str] = {}
_SHA_MEMO_MAX = 4096


def clear_sha_memo() -> None:
    _SHA_MEMO.clear()


def path_stat_token(path: Path | str) -> tuple[str, int, int] | None:
    """Return ``(resolved_path, size, mtime_ns)`` or ``None`` if missing."""
    candidate = Path(path)
    try:
        resolved = candidate.resolve()
        if not resolved.is_file():
            return None
        stat = resolved.stat()
    except OSError:
        return None
    return (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))


def path_stat_mapping(path: Path | str, *, relative: str | None = None) -> dict[str, Any]:
    token = path_stat_token(path)
    if token is None:
        return {
            "path": relative or str(path),
            "exists": False,
            "size": None,
            "mtime_ns": None,
        }
    resolved, size, mtime_ns = token
    return {
        "path": relative or resolved,
        "exists": True,
        "size": size,
        "mtime_ns": mtime_ns,
    }


def directory_entry_signature(
    directory: Path | str,
    *,
    names: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Cheap per-entry signature for selected files inside a directory."""
    root = Path(directory)
    selected = list(names) if names is not None else None
    entries: list[dict[str, Any]] = []
    if not root.is_dir():
        return entries
    try:
        children = sorted(root.iterdir(), key=lambda item: item.name)
    except OSError:
        return entries
    for child in children:
        if selected is not None and child.name not in selected:
            continue
        if not child.is_file():
            continue
        entries.append(path_stat_mapping(child, relative=child.name))
    return entries


def memoized_file_sha256(path: Path | str) -> str:
    """SHA256 with memoization keyed by resolved path + size + mtime_ns."""
    candidate = Path(path)
    token = path_stat_token(candidate)
    if token is None:
        raise FileNotFoundError(f"File not found for hashing: {candidate}")
    cached = _SHA_MEMO.get(token)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with Path(token[0]).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    hex_digest = digest.hexdigest()
    if len(_SHA_MEMO) >= _SHA_MEMO_MAX:
        # Drop an arbitrary oldest entry; correctness depends only on the key.
        _SHA_MEMO.pop(next(iter(_SHA_MEMO)))
    _SHA_MEMO[token] = hex_digest
    return hex_digest


def cheap_inventory_fingerprint(entries: Iterable[Mapping[str, Any]]) -> str:
    """Stable fingerprint from cheap stat mappings (no file content hashing)."""
    normalized = []
    for item in entries:
        normalized.append(
            {
                "path": item.get("path"),
                "exists": bool(item.get("exists")),
                "size": item.get("size"),
                "mtime_ns": item.get("mtime_ns"),
            }
        )
    normalized.sort(key=lambda row: str(row.get("path") or ""))
    return canonical_sha256({"contract": "cheap_inventory_v1", "entries": normalized})


__all__ = [
    "cheap_inventory_fingerprint",
    "clear_sha_memo",
    "directory_entry_signature",
    "memoized_file_sha256",
    "path_stat_mapping",
    "path_stat_token",
]
