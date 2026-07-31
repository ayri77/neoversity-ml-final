"""Versioned Control Panel archive registry (hide without deleting results).

Archive overlay for terminal UI jobs and configured Results artifacts. Physical
deletion in v1 is limited to archived terminal UI job directories under
``artifacts/ui_jobs/<canonical-uuid>/``. Authoritative experiment artifacts are
never moved, renamed, or deleted by this module.
"""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.churn_ml.control_panel.jobs import TERMINAL_STATES, JobError, JobManager
from src.churn_ml.control_panel.path_safety import (
    PathSafetyError,
    path_exists_nonfollowing,
    require_regular_file,
    require_safe_directory,
    require_safe_existing_ancestors,
)
from src.churn_ml.control_panel.schemas import SchemaError, validate_relative_path_text


class ArchiveError(RuntimeError):
    """Raised for archive registry validation, persistence, or lifecycle errors."""


ARCHIVE_SCHEMA_VERSION = "control_panel_archive_v1"
DEFAULT_ARCHIVE_RELATIVE = "artifacts/control_panel_state/archived_items.json"
ITEM_KINDS = frozenset({"ui_job", "artifact"})
REASON_LIMIT = 240
_READER_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

_ROOT_KEYS = frozenset({"schema_version", "items"})
_UI_JOB_KEYS = frozenset({"kind", "job_id", "archived_at_utc", "reason"})
_ARTIFACT_KEYS = frozenset(
    {"kind", "reader_id", "relative_path", "archived_at_utc", "reason"}
)


def utc_now_text() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def archive_registry_path(
    repository_root: Path,
    *,
    relative: str = DEFAULT_ARCHIVE_RELATIVE,
) -> Path:
    root = repository_root.resolve()
    rel = validate_relative_path_text(relative, "archive registry path")
    path = (root / Path(*Path(rel).parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ArchiveError("Archive registry path escapes the repository root.") from error
    return path


class ArchiveRegistry:
    """Strict persisted archive overlay with atomic writes."""

    def __init__(
        self,
        repository_root: Path,
        *,
        relative_path: str = DEFAULT_ARCHIVE_RELATIVE,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.relative_path = validate_relative_path_text(
            relative_path, "archive registry path"
        )
        self.path = archive_registry_path(
            self.repository_root, relative=self.relative_path
        )
        self._payload = self._load_or_empty()

    @property
    def items(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(item) for item in self._payload["items"])

    def is_ui_job_archived(self, job_id: str) -> bool:
        canonical = _canonical_job_id(job_id)
        return any(
            item["kind"] == "ui_job" and item["job_id"] == canonical
            for item in self._payload["items"]
        )

    def is_artifact_archived(self, *, reader_id: str, relative_path: str) -> bool:
        key = _artifact_identity(reader_id=reader_id, relative_path=relative_path)
        return any(
            item["kind"] == "artifact"
            and item["reader_id"] == key["reader_id"]
            and item["relative_path"] == key["relative_path"]
            for item in self._payload["items"]
        )

    def archived_ui_job_ids(self) -> frozenset[str]:
        return frozenset(
            item["job_id"]
            for item in self._payload["items"]
            if item["kind"] == "ui_job"
        )

    def archived_artifact_keys(self) -> frozenset[tuple[str, str]]:
        return frozenset(
            (item["reader_id"], item["relative_path"])
            for item in self._payload["items"]
            if item["kind"] == "artifact"
        )

    def archive_ui_job(
        self,
        job_id: str,
        *,
        state: str,
        reason: str | None = None,
    ) -> bool:
        """Archive a terminal UI job. Returns False when already archived."""
        if state not in TERMINAL_STATES:
            raise ArchiveError(
                "Only terminal UI jobs can be archived "
                f"(state={state!r}; allowed={sorted(TERMINAL_STATES)})."
            )
        canonical = _canonical_job_id(job_id)
        if self.is_ui_job_archived(canonical):
            return False
        item = {
            "kind": "ui_job",
            "job_id": canonical,
            "archived_at_utc": utc_now_text(),
            "reason": _normalize_reason(reason),
        }
        _validate_item(item)
        self._payload["items"].append(item)
        self._sort_items()
        self._persist()
        return True

    def restore_ui_job(self, job_id: str) -> bool:
        """Restore a UI job into normal lists. Returns False when not archived."""
        canonical = _canonical_job_id(job_id)
        before = len(self._payload["items"])
        self._payload["items"] = [
            item
            for item in self._payload["items"]
            if not (item["kind"] == "ui_job" and item["job_id"] == canonical)
        ]
        if len(self._payload["items"]) == before:
            return False
        self._persist()
        return True

    def archive_artifact(
        self,
        *,
        reader_id: str,
        relative_path: str,
        reason: str | None = None,
    ) -> bool:
        """Archive a Results artifact by reader + relative path identity."""
        identity = _artifact_identity(reader_id=reader_id, relative_path=relative_path)
        self._assert_artifact_path_safe(identity["relative_path"])
        if self.is_artifact_archived(
            reader_id=identity["reader_id"],
            relative_path=identity["relative_path"],
        ):
            return False
        item = {
            "kind": "artifact",
            "reader_id": identity["reader_id"],
            "relative_path": identity["relative_path"],
            "archived_at_utc": utc_now_text(),
            "reason": _normalize_reason(reason),
        }
        _validate_item(item)
        self._payload["items"].append(item)
        self._sort_items()
        self._persist()
        return True

    def restore_artifact(self, *, reader_id: str, relative_path: str) -> bool:
        identity = _artifact_identity(reader_id=reader_id, relative_path=relative_path)
        before = len(self._payload["items"])
        self._payload["items"] = [
            item
            for item in self._payload["items"]
            if not (
                item["kind"] == "artifact"
                and item["reader_id"] == identity["reader_id"]
                and item["relative_path"] == identity["relative_path"]
            )
        ]
        if len(self._payload["items"]) == before:
            return False
        self._persist()
        return True

    def permanently_delete_archived_ui_job(
        self,
        job_manager: JobManager,
        job_id: str,
    ) -> Path:
        """Delete only ``artifacts/ui_jobs/<uuid>/`` for an archived terminal job.

        Does not delete Research v2 artifacts, Campaign state, or MLflow runs.
        The job's local ``mlflow_index.json`` sidecar (if present) is removed with
        the UI job directory because it is Control Panel metadata, not the
        authoritative MLflow store.
        """
        canonical = _canonical_job_id(job_id)
        if not self.is_ui_job_archived(canonical):
            raise ArchiveError(
                "Permanent deletion requires an archived terminal UI job."
            )
        try:
            record = job_manager.load(canonical)
        except JobError as error:
            raise ArchiveError(str(error)) from error
        state = str(record.status.get("state") or "")
        if state not in TERMINAL_STATES:
            raise ArchiveError(
                "Permanent deletion requires a terminal job state "
                f"(state={state!r})."
            )
        deleted_root = _delete_canonical_job_directory(
            jobs_root=job_manager.jobs_root,
            job_id=canonical,
        )
        self.restore_ui_job(canonical)
        return deleted_root

    def filter_artifacts(
        self,
        artifacts: list[Any],
        *,
        reader_id: str,
        show_archived: bool,
    ) -> list[Any]:
        """Filter discovered artifact records for Results visibility."""
        if show_archived:
            return list(artifacts)
        archived = self.archived_artifact_keys()
        return [
            item
            for item in artifacts
            if (reader_id, str(item.relative_path)) not in archived
        ]

    def _assert_artifact_path_safe(self, relative_path: str) -> None:
        validate_relative_path_text(relative_path, "artifact.relative_path")
        absolute = (self.repository_root / Path(*Path(relative_path).parts)).resolve()
        try:
            absolute.relative_to(self.repository_root)
        except ValueError as error:
            raise ArchiveError(
                "Artifact relative_path escapes the repository root."
            ) from error
        if not absolute.exists():
            raise ArchiveError(
                f"Cannot archive missing artifact path: {relative_path}."
            )
        try:
            require_safe_directory(absolute)
        except PathSafetyError as error:
            raise ArchiveError(
                f"Unsafe artifact directory cannot be archived: {relative_path}."
            ) from error

    def _load_or_empty(self) -> dict[str, Any]:
        if not path_exists_nonfollowing(self.path):
            return {"schema_version": ARCHIVE_SCHEMA_VERSION, "items": []}
        try:
            require_regular_file(self.path, reject_hardlinks=True)
            text = self.path.read_text(encoding="utf-8")
            payload = json.loads(text)
        except (OSError, json.JSONDecodeError, PathSafetyError, UnicodeDecodeError) as error:
            raise ArchiveError(
                f"Corrupt Control Panel archive registry at "
                f"{self.relative_path}: {error}"
            ) from error
        return _parse_registry_payload(payload)

    def _sort_items(self) -> None:
        self._payload["items"] = sorted(
            self._payload["items"],
            key=_item_sort_key,
        )

    def _persist(self) -> None:
        self._sort_items()
        payload = {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "items": deepcopy(self._payload["items"]),
        }
        _parse_registry_payload(payload)  # re-validate before write
        parent = self.path.parent
        try:
            require_safe_existing_ancestors(parent)
            parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            require_safe_directory(parent)
            if path_exists_nonfollowing(self.path):
                require_regular_file(self.path, reject_hardlinks=True)
        except (OSError, PathSafetyError) as error:
            raise ArchiveError(
                f"Unsafe archive registry path: {self.relative_path}."
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


def preview_job_deletion(job_root: Path) -> list[str]:
    """Return sorted relative names under a job directory for deletion preview."""
    require_safe_directory(job_root)
    names: list[str] = []
    for entry in sorted(os.scandir(job_root), key=lambda item: item.name):
        if entry.name.startswith("."):
            continue
        info = entry.stat(follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode) or bool(
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            raise ArchiveError(
                f"Unsafe link in job directory preview: {entry.name}."
            )
        names.append(entry.name)
    return names


def _delete_canonical_job_directory(*, jobs_root: Path, job_id: str) -> Path:
    canonical = _canonical_job_id(job_id)
    require_safe_directory(jobs_root)
    root = jobs_root / canonical
    if root.parent != jobs_root:
        raise ArchiveError("Invalid job deletion path.")
    if root.resolve().parent != jobs_root.resolve():
        raise ArchiveError("Job deletion path escaped jobs root.")
    require_safe_directory(root)
    _secure_rmtree(root)
    return root


def _secure_rmtree(directory: Path) -> None:
    """Delete a directory tree without following links or reparse points."""
    require_safe_directory(directory)
    for entry in list(os.scandir(directory)):
        path = Path(entry.path)
        info = entry.stat(follow_symlinks=False)
        attributes = getattr(info, "st_file_attributes", 0)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if stat.S_ISLNK(info.st_mode) or bool(attributes & reparse_flag):
            raise ArchiveError(f"Refusing to delete linked path: {path.name}.")
        if stat.S_ISDIR(info.st_mode):
            _secure_rmtree(path)
        elif stat.S_ISREG(info.st_mode):
            path.unlink()
        else:
            raise ArchiveError(f"Refusing to delete non-regular path: {path.name}.")
    directory.rmdir()


def _canonical_job_id(job_id: str) -> str:
    try:
        parsed = uuid.UUID(job_id)
    except (TypeError, ValueError) as error:
        raise ArchiveError("Invalid job id.") from error
    text = str(parsed)
    if text != job_id:
        raise ArchiveError("Job id must use canonical UUID text.")
    return text


def _artifact_identity(*, reader_id: str, relative_path: str) -> dict[str, str]:
    if not isinstance(reader_id, str) or _READER_ID_RE.fullmatch(reader_id) is None:
        raise ArchiveError("reader_id must be a lowercase snake_case identifier.")
    try:
        path = validate_relative_path_text(relative_path, "artifact.relative_path")
    except SchemaError as error:
        raise ArchiveError(str(error)) from error
    return {"reader_id": reader_id, "relative_path": path}


def _normalize_reason(reason: str | None) -> str | None:
    if reason is None:
        return None
    if not isinstance(reason, str):
        raise ArchiveError("Archive reason must be a string or null.")
    text = reason.strip()
    if not text:
        return None
    if len(text) > REASON_LIMIT:
        raise ArchiveError(f"Archive reason exceeds {REASON_LIMIT} characters.")
    if "\x00" in text:
        raise ArchiveError("Archive reason contains a NUL byte.")
    return text


def _item_sort_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    kind = item["kind"]
    if kind == "ui_job":
        return ("ui_job", item["job_id"], item["archived_at_utc"])
    return (
        "artifact",
        item["reader_id"],
        item["relative_path"],
        item["archived_at_utc"],
    )


def _exact_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    keys = set(payload)
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected)
    if missing or unknown:
        raise ArchiveError(
            f"{label} key mismatch; missing={missing}, unknown={unknown}."
        )


def _validate_item(item: Mapping[str, Any]) -> None:
    kind = item.get("kind")
    if kind == "ui_job":
        _exact_keys(item, _UI_JOB_KEYS, "ui_job archive item")
        _canonical_job_id(str(item["job_id"]))
    elif kind == "artifact":
        _exact_keys(item, _ARTIFACT_KEYS, "artifact archive item")
        _artifact_identity(
            reader_id=str(item["reader_id"]),
            relative_path=str(item["relative_path"]),
        )
    else:
        raise ArchiveError(f"Unsupported archive item kind: {kind!r}.")
    if not isinstance(item["archived_at_utc"], str) or not item["archived_at_utc"]:
        raise ArchiveError("archived_at_utc must be a non-empty string.")
    if item["reason"] is not None and not isinstance(item["reason"], str):
        raise ArchiveError("reason must be a string or null.")


def _parse_registry_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ArchiveError("Archive registry must be a JSON object.")
    _exact_keys(payload, _ROOT_KEYS, "archive registry")
    if payload["schema_version"] != ARCHIVE_SCHEMA_VERSION:
        raise ArchiveError(
            "Unsupported archive schema_version "
            f"{payload['schema_version']!r}; expected {ARCHIVE_SCHEMA_VERSION!r}."
        )
    items = payload["items"]
    if not isinstance(items, list):
        raise ArchiveError("archive registry items must be a list.")
    parsed: list[dict[str, Any]] = []
    seen_jobs: set[str] = set()
    seen_artifacts: set[tuple[str, str]] = set()
    for index, raw in enumerate(items):
        if not isinstance(raw, dict):
            raise ArchiveError(f"items[{index}] must be an object.")
        item = dict(raw)
        _validate_item(item)
        if item["kind"] == "ui_job":
            if item["job_id"] in seen_jobs:
                raise ArchiveError(f"Duplicate archived ui_job: {item['job_id']}.")
            seen_jobs.add(item["job_id"])
        else:
            key = (item["reader_id"], item["relative_path"])
            if key in seen_artifacts:
                raise ArchiveError(
                    "Duplicate archived artifact: "
                    f"{item['reader_id']} + {item['relative_path']}."
                )
            seen_artifacts.add(key)
        parsed.append(item)
    parsed.sort(key=_item_sort_key)
    return {"schema_version": ARCHIVE_SCHEMA_VERSION, "items": parsed}
