"""Strict human-readable artifact validation for the local MLflow index."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

import yaml


ArtifactContentType = Literal["json", "yaml", "csv", "terminal_marker"]


class ArtifactContentError(RuntimeError):
    """Raised when an allowlisted artifact is not safe human-readable content."""


@dataclass(frozen=True)
class IndexedArtifact:
    relative_path: str
    content_type: ArtifactContentType
    size_bytes: int
    sha256: str
    max_size_bytes: int


def select_indexed_artifacts(
    run_dir: Path,
    allowlist: tuple[str, ...],
    *,
    enabled: bool,
    size_limit: int,
) -> tuple[IndexedArtifact, ...]:
    """Select and fully validate small allowlisted metadata artifacts."""
    if not enabled:
        return ()
    selected: list[IndexedArtifact] = []
    for relative in allowlist:
        candidate = run_dir / Path(*relative.split("/"))
        if not candidate.exists():
            continue
        size = _safe_file_state(candidate, run_dir, relative)
        if size > size_limit:
            continue
        content_type = _content_type(relative)
        raw = candidate.read_bytes()
        after_size = _safe_file_state(candidate, run_dir, relative)
        if len(raw) != size or after_size != size:
            raise ArtifactContentError(
                f"Artifact size/type changed while selecting: {relative}"
            )
        _validate_content(raw, content_type, relative)
        selected.append(
            IndexedArtifact(
                relative_path=relative,
                content_type=content_type,
                size_bytes=size,
                sha256=hashlib.sha256(raw).hexdigest(),
                max_size_bytes=size_limit,
            )
        )
    return tuple(selected)


def revalidate_indexed_artifact(
    run_dir: Path,
    artifact: IndexedArtifact,
) -> Path:
    """Revalidate type, containment, bytes, format, size, and selected identity."""
    relative = _portable_relative_path(artifact.relative_path)
    candidate = run_dir / Path(*relative.split("/"))
    size = _safe_file_state(candidate, run_dir, relative)
    if size > artifact.max_size_bytes:
        raise ArtifactContentError(
            f"Artifact grew above the configured size cap: {relative}"
        )
    raw = candidate.read_bytes()
    after_size = _safe_file_state(candidate, run_dir, relative)
    if len(raw) != size or after_size != size:
        raise ArtifactContentError(
            f"Artifact size/type changed while reading: {relative}"
        )
    _validate_content(raw, artifact.content_type, relative)
    digest = hashlib.sha256(raw).hexdigest()
    if size != artifact.size_bytes or digest != artifact.sha256:
        raise ArtifactContentError(
            f"Artifact changed after source validation: {relative}"
        )
    return candidate


def _safe_file_state(candidate: Path, run_dir: Path, relative: str) -> int:
    relative = _portable_relative_path(relative)
    if candidate.is_symlink() or _is_reparse_point(candidate):
        raise ArtifactContentError(
            f"Allowlisted artifact is a symlink, junction, or reparse point: {relative}"
        )
    try:
        info = candidate.stat(follow_symlinks=False)
    except OSError as error:
        raise ArtifactContentError(
            f"Cannot stat allowlisted artifact {relative}: {error}"
        ) from error
    if not stat.S_ISREG(info.st_mode):
        raise ArtifactContentError(
            f"Allowlisted artifact is not a regular file: {relative}"
        )
    try:
        resolved_run = run_dir.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_run)
    except (OSError, ValueError) as error:
        raise ArtifactContentError(
            f"Allowlisted artifact escapes the source run: {relative}"
        ) from error
    return info.st_size


def _is_reparse_point(path: Path) -> bool:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError:
        return False
    attributes = getattr(info, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    is_junction = getattr(path, "is_junction", None)
    return bool(attributes & reparse_flag) or bool(
        callable(is_junction) and is_junction()
    )


def _portable_relative_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(value)
    if (
        not normalized
        or normalized.startswith("/")
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or ".." in posix.parts
        or "." in posix.parts
    ):
        raise ArtifactContentError(f"Artifact path is not portable: {value}")
    canonical = posix.as_posix()
    if canonical != normalized:
        raise ArtifactContentError(f"Artifact path is not canonical: {value}")
    return canonical


def _content_type(relative: str) -> ArtifactContentType:
    name = PurePosixPath(relative).name
    suffix = PurePosixPath(relative).suffix.lower()
    if name in {"_SUCCESS", "_FAILED"}:
        return "terminal_marker"
    if suffix == ".json":
        return "json"
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    if suffix == ".csv":
        return "csv"
    raise ArtifactContentError(
        f"Unsupported allowlisted artifact extension/content combination: {relative}"
    )


def _validate_content(
    raw: bytes,
    content_type: ArtifactContentType,
    relative: str,
) -> None:
    if b"\x00" in raw:
        raise ArtifactContentError(f"Artifact contains a NUL byte: {relative}")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ArtifactContentError(
            f"Artifact is not strict UTF-8: {relative}"
        ) from error
    if content_type == "terminal_marker":
        _validate_terminal_marker(text, relative)
        return
    if not text.strip():
        raise ArtifactContentError(f"Artifact is empty: {relative}")
    if content_type == "json":
        try:
            payload = json.loads(
                text,
                parse_constant=lambda token: _reject_json_constant(token),
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ArtifactContentError(
                f"Artifact JSON is invalid: {relative}"
            ) from error
        if type(payload) is not dict:
            raise ArtifactContentError(
                f"Artifact JSON must contain a top-level mapping: {relative}"
            )
        return
    if content_type == "yaml":
        try:
            payload = yaml.safe_load(text)
        except yaml.YAMLError as error:
            raise ArtifactContentError(
                f"Artifact YAML is invalid: {relative}"
            ) from error
        if type(payload) is not dict:
            raise ArtifactContentError(
                f"Artifact YAML must contain a top-level mapping: {relative}"
            )
        return
    if content_type == "csv":
        _validate_csv(text, relative)
        return
    raise ArtifactContentError(f"Unsupported artifact content type: {content_type}")


def _validate_terminal_marker(text: str, relative: str) -> None:
    if not text:
        if PurePosixPath(relative).name == "_FAILED":
            return
        raise ArtifactContentError(f"Success terminal marker is empty: {relative}")
    try:
        payload = json.loads(
            text,
            parse_constant=lambda token: _reject_json_constant(token),
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ArtifactContentError(
            f"Terminal marker is not valid UTF-8 JSON: {relative}"
        ) from error
    if type(payload) is not dict:
        raise ArtifactContentError(
            f"Terminal marker must contain a mapping: {relative}"
        )


def _validate_csv(text: str, relative: str) -> None:
    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as error:
        raise ArtifactContentError(f"Artifact CSV is malformed: {relative}") from error
    if not rows or not rows[0]:
        raise ArtifactContentError(f"Artifact CSV has no header: {relative}")
    header = rows[0]
    if any(not item.strip() for item in header) or len(set(header)) != len(header):
        raise ArtifactContentError(f"Artifact CSV header is invalid: {relative}")
    width = len(header)
    if any(len(row) != width for row in rows[1:]):
        raise ArtifactContentError(
            f"Artifact CSV rows do not match the header width: {relative}"
        )


def _reject_json_constant(token: str) -> Any:
    raise ValueError(f"Non-finite JSON constant is forbidden: {token}")
