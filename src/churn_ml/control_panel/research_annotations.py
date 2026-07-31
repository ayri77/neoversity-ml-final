"""Versioned Research Workspace annotation overlay.

Annotations are keyed by repository-relative Research v2 run paths and never
mutate authoritative Research v2 artifacts. Persistence follows the same
atomic, path-safe conventions as the Control Panel archive registry.
"""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.churn_ml.control_panel.path_safety import (
    PathSafetyError,
    path_exists_nonfollowing,
    require_regular_file,
    require_safe_directory,
    require_safe_existing_ancestors,
)
from src.churn_ml.control_panel.schemas import SchemaError, validate_relative_path_text


class ResearchAnnotationError(RuntimeError):
    """Raised for annotation registry validation or persistence errors."""


ANNOTATION_SCHEMA_VERSION = "control_panel_research_annotations_v1"
DEFAULT_ANNOTATION_RELATIVE = (
    "artifacts/control_panel_state/research_annotations.json"
)
BUILTIN_TAGS = frozenset(
    {
        "baseline",
        "candidate",
        "shortlist",
        "exploratory",
        "reject",
        "needs_rerun",
        "tuned",
        "kaggle_candidate",
    }
)
USER_TAG_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
NOTE_LIMIT = 2000
_ROOT_KEYS = frozenset({"schema_version", "annotations"})
_ITEM_KEYS = frozenset(
    {
        "relative_path",
        "tags",
        "note",
        "shortlisted",
        "updated_at_utc",
    }
)


def utc_now_text() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def research_annotations_path(
    repository_root: Path,
    *,
    relative: str = DEFAULT_ANNOTATION_RELATIVE,
) -> Path:
    root = repository_root.resolve()
    rel = validate_relative_path_text(relative, "research annotations path")
    path = (root / Path(*Path(rel).parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ResearchAnnotationError(
            "Research annotations path escapes the repository root."
        ) from error
    return path


class ResearchAnnotationRegistry:
    """Strict persisted research annotation overlay with atomic writes."""

    def __init__(
        self,
        repository_root: Path,
        *,
        relative_path: str = DEFAULT_ANNOTATION_RELATIVE,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.relative_path = validate_relative_path_text(
            relative_path, "research annotations path"
        )
        self.path = research_annotations_path(
            self.repository_root, relative=self.relative_path
        )
        self._payload = self._load_or_empty()

    @property
    def annotations(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(item) for item in self._payload["annotations"])

    def get(self, relative_path: str) -> dict[str, Any] | None:
        key = _normalize_run_relative_path(relative_path)
        for item in self._payload["annotations"]:
            if item["relative_path"] == key:
                return deepcopy(item)
        return None

    def list_stale(self, known_relative_paths: set[str]) -> tuple[str, ...]:
        known = {
            _normalize_run_relative_path(path) for path in known_relative_paths
        }
        return tuple(
            sorted(
                item["relative_path"]
                for item in self._payload["annotations"]
                if item["relative_path"] not in known
            )
        )

    def upsert(
        self,
        relative_path: str,
        *,
        tags: list[str] | tuple[str, ...] | None = None,
        note: str | None = None,
        shortlisted: bool | None = None,
        require_existing_artifact: bool = True,
    ) -> dict[str, Any]:
        """Create or update an annotation. Returns the persisted item."""
        key = _normalize_run_relative_path(relative_path)
        if require_existing_artifact:
            self._assert_run_path_safe(key)
        existing = self.get(key)
        next_tags = (
            _normalize_tags(tags)
            if tags is not None
            else list(existing["tags"] if existing else [])
        )
        next_note = (
            _normalize_note(note)
            if note is not None
            else (existing["note"] if existing else "")
        )
        next_shortlisted = (
            bool(shortlisted)
            if shortlisted is not None
            else bool(existing["shortlisted"] if existing else False)
        )
        if next_shortlisted and "shortlist" not in next_tags:
            next_tags.append("shortlist")
            next_tags = sorted(set(next_tags))
        if not next_shortlisted and "shortlist" in next_tags and shortlisted is False:
            next_tags = [tag for tag in next_tags if tag != "shortlist"]
        item = {
            "relative_path": key,
            "tags": next_tags,
            "note": next_note,
            "shortlisted": next_shortlisted,
            "updated_at_utc": utc_now_text(),
        }
        _validate_item(item)
        annotations = [
            entry
            for entry in self._payload["annotations"]
            if entry["relative_path"] != key
        ]
        annotations.append(item)
        self._payload["annotations"] = annotations
        self._sort_items()
        self._persist()
        return deepcopy(item)

    def set_tags(self, relative_path: str, tags: list[str] | tuple[str, ...]) -> dict[str, Any]:
        return self.upsert(relative_path, tags=tags)

    def set_note(self, relative_path: str, note: str) -> dict[str, Any]:
        return self.upsert(relative_path, note=note)

    def promote_to_shortlist(self, relative_path: str) -> dict[str, Any]:
        existing = self.get(relative_path)
        tags = list(existing["tags"] if existing else [])
        if "shortlist" not in tags:
            tags.append("shortlist")
        return self.upsert(relative_path, tags=tags, shortlisted=True)

    def remove_from_shortlist(self, relative_path: str) -> dict[str, Any]:
        existing = self.get(relative_path)
        tags = [tag for tag in (existing["tags"] if existing else []) if tag != "shortlist"]
        return self.upsert(relative_path, tags=tags, shortlisted=False)

    def _assert_run_path_safe(self, relative_path: str) -> None:
        absolute = (self.repository_root / Path(*Path(relative_path).parts)).resolve()
        try:
            absolute.relative_to(self.repository_root)
        except ValueError as error:
            raise ResearchAnnotationError(
                "Annotation relative_path escapes the repository root."
            ) from error
        if not absolute.exists():
            raise ResearchAnnotationError(
                f"Cannot annotate missing Research v2 path: {relative_path}."
            )
        try:
            require_safe_directory(absolute)
        except PathSafetyError as error:
            raise ResearchAnnotationError(
                f"Unsafe Research v2 directory cannot be annotated: {relative_path}."
            ) from error

    def _load_or_empty(self) -> dict[str, Any]:
        if not path_exists_nonfollowing(self.path):
            return {
                "schema_version": ANNOTATION_SCHEMA_VERSION,
                "annotations": [],
            }
        try:
            require_regular_file(self.path, reject_hardlinks=True)
            text = self.path.read_text(encoding="utf-8")
            payload = json.loads(text)
        except (
            OSError,
            json.JSONDecodeError,
            PathSafetyError,
            UnicodeDecodeError,
        ) as error:
            raise ResearchAnnotationError(
                f"Corrupt Control Panel research annotations at "
                f"{self.relative_path}: {error}"
            ) from error
        return _parse_registry_payload(payload)

    def _sort_items(self) -> None:
        self._payload["annotations"] = sorted(
            self._payload["annotations"],
            key=lambda item: (item["relative_path"], item["updated_at_utc"]),
        )

    def _persist(self) -> None:
        self._sort_items()
        payload = {
            "schema_version": ANNOTATION_SCHEMA_VERSION,
            "annotations": deepcopy(self._payload["annotations"]),
        }
        _parse_registry_payload(payload)
        parent = self.path.parent
        try:
            require_safe_existing_ancestors(parent)
            parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            require_safe_directory(parent)
            if path_exists_nonfollowing(self.path):
                require_regular_file(self.path, reject_hardlinks=True)
        except (OSError, PathSafetyError) as error:
            raise ResearchAnnotationError(
                f"Unsafe research annotations path: {self.relative_path}."
            ) from error
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=parent,
        )
        os.chmod(temporary, stat.S_IRUSR | stat.S_IWUSR)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        self._payload = payload


def _normalize_run_relative_path(relative_path: str) -> str:
    try:
        path = validate_relative_path_text(relative_path, "annotation.relative_path")
    except SchemaError as error:
        raise ResearchAnnotationError(str(error)) from error
    if not path.startswith("artifacts/research_v2/"):
        raise ResearchAnnotationError(
            "Annotation relative_path must be under artifacts/research_v2/."
        )
    return path


def _normalize_tags(tags: list[str] | tuple[str, ...]) -> list[str]:
    if not isinstance(tags, (list, tuple)):
        raise ResearchAnnotationError("tags must be a list of strings.")
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in tags:
        if not isinstance(raw, str):
            raise ResearchAnnotationError("Each tag must be a string.")
        tag = raw.strip().lower()
        if not tag:
            continue
        if tag not in BUILTIN_TAGS and USER_TAG_RE.fullmatch(tag) is None:
            raise ResearchAnnotationError(
                f"Unsupported research tag {raw!r}; use a builtin tag or "
                "safe lowercase snake_case user tag."
            )
        if tag not in seen:
            seen.add(tag)
            normalized.append(tag)
    return sorted(normalized)


def _normalize_note(note: str | None) -> str:
    if note is None:
        return ""
    if not isinstance(note, str):
        raise ResearchAnnotationError("note must be a string.")
    if "\x00" in note:
        raise ResearchAnnotationError("note contains a NUL byte.")
    text = note.strip()
    if len(text) > NOTE_LIMIT:
        raise ResearchAnnotationError(f"note exceeds {NOTE_LIMIT} characters.")
    return text


def _exact_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    keys = set(payload)
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected)
    if missing or unknown:
        raise ResearchAnnotationError(
            f"{label} key mismatch; missing={missing}, unknown={unknown}."
        )


def _validate_item(item: Mapping[str, Any]) -> None:
    _exact_keys(item, _ITEM_KEYS, "research annotation item")
    _normalize_run_relative_path(str(item["relative_path"]))
    if not isinstance(item["tags"], list):
        raise ResearchAnnotationError("tags must be a list.")
    _normalize_tags(item["tags"])
    if not isinstance(item["note"], str):
        raise ResearchAnnotationError("note must be a string.")
    if "\x00" in item["note"] or len(item["note"]) > NOTE_LIMIT:
        raise ResearchAnnotationError("note is invalid.")
    if not isinstance(item["shortlisted"], bool):
        raise ResearchAnnotationError("shortlisted must be a boolean.")
    if not isinstance(item["updated_at_utc"], str) or not item["updated_at_utc"]:
        raise ResearchAnnotationError("updated_at_utc must be a non-empty string.")


def _parse_registry_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ResearchAnnotationError("Research annotations must be a JSON object.")
    _exact_keys(payload, _ROOT_KEYS, "research annotations")
    if payload["schema_version"] != ANNOTATION_SCHEMA_VERSION:
        raise ResearchAnnotationError(
            "Unsupported research annotations schema_version "
            f"{payload['schema_version']!r}; expected {ANNOTATION_SCHEMA_VERSION!r}."
        )
    annotations = payload["annotations"]
    if not isinstance(annotations, list):
        raise ResearchAnnotationError("annotations must be a list.")
    parsed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(annotations):
        if not isinstance(raw, dict):
            raise ResearchAnnotationError(f"annotations[{index}] must be an object.")
        item = dict(raw)
        _validate_item(item)
        if item["relative_path"] in seen:
            raise ResearchAnnotationError(
                f"Duplicate research annotation for {item['relative_path']}."
            )
        seen.add(item["relative_path"])
        parsed.append(item)
    parsed.sort(key=lambda item: (item["relative_path"], item["updated_at_utc"]))
    return {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "annotations": parsed,
    }
