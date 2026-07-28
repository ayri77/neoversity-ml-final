from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.churn_ml.control_panel.command_builder import (
    CommandBuildError,
    resolve_safe_path,
)
from src.churn_ml.control_panel.schemas import ReaderSpec


class ArtifactReadError(ValueError):
    """Raised when a configured artifact cannot be read safely."""


@dataclass(frozen=True)
class ArtifactRecord:
    reader_id: str
    root: Path
    relative_path: str
    state: str
    summaries: Mapping[str, Any]
    json_payloads: Mapping[str, Any]


def configured_artifact_file(
    repository_root: Path,
    artifact: ArtifactRecord,
    relative_path: str,
) -> Path:
    try:
        _, path = resolve_safe_path(
            repository_root,
            f"{artifact.relative_path}/{relative_path}",
            allowed_roots=(artifact.relative_path,),
            must_exist=True,
        )
    except CommandBuildError as error:
        raise ArtifactReadError(str(error)) from error
    if not path.is_file():
        raise ArtifactReadError(f"Configured artifact file is not regular: {path}.")
    return path


def safe_json_load(path: Path, *, max_bytes: int = 2_000_000) -> Any:
    _require_regular_file(path)
    if path.stat().st_size > max_bytes:
        raise ArtifactReadError(f"JSON file exceeds {max_bytes} bytes: {path}.")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactReadError(f"Could not read JSON file {path}: {error}") from error


def extract_dot_path(payload: Any, path: str) -> Any:
    current = payload
    for part in path.split("."):
        if not part:
            raise ArtifactReadError("Dot paths must not contain empty segments.")
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return None
            current = current[index]
        else:
            return None
    return current


def csv_preview(
    path: Path,
    *,
    row_limit: int = 100,
    column_limit: int = 40,
    max_bytes: int = 5_000_000,
) -> pd.DataFrame:
    if row_limit < 1 or column_limit < 1:
        raise ArtifactReadError("CSV preview limits must be positive.")
    _require_regular_file(path)
    if path.stat().st_size > max_bytes:
        raise ArtifactReadError(f"CSV file exceeds {max_bytes} bytes: {path}.")
    try:
        frame = pd.read_csv(path, nrows=row_limit)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
        raise ArtifactReadError(
            f"Could not preview CSV file {path}: {error}"
        ) from error
    return frame.iloc[:, :column_limit]


def text_tail(
    path: Path,
    *,
    lines: int,
    max_bytes: int = 2_000_000,
) -> str:
    if lines < 1:
        raise ArtifactReadError("Tail line count must be positive.")
    _require_regular_file(path)
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(max(0, size - max_bytes))
        content = handle.read(max_bytes)
    return "\n".join(content.decode("utf-8", errors="replace").splitlines()[-lines:])


def discover_artifacts(
    repository_root: Path,
    reader: ReaderSpec,
) -> list[ArtifactRecord]:
    root = repository_root.resolve(strict=True)
    results: list[ArtifactRecord] = []
    for configured_root in reader.artifact_roots:
        try:
            _, artifact_root = resolve_safe_path(
                root,
                configured_root,
                allowed_roots=(configured_root,),
                must_exist=True,
            )
        except CommandBuildError:
            continue
        for candidate in artifact_root.glob(reader.discovery_glob):
            if not candidate.is_dir() or candidate.is_symlink():
                continue
            try:
                canonical = candidate.resolve(strict=True)
                canonical.relative_to(artifact_root)
                record = read_artifact(root, reader, canonical)
            except (OSError, ValueError, ArtifactReadError):
                continue
            results.append(record)
    return sorted(results, key=lambda item: item.relative_path, reverse=True)


def read_artifact(
    repository_root: Path,
    reader: ReaderSpec,
    artifact_root: Path,
) -> ArtifactRecord:
    root = repository_root.resolve(strict=True)
    canonical = artifact_root.resolve(strict=True)
    if canonical.is_symlink() or not canonical.is_dir():
        raise ArtifactReadError("Artifact root must be a regular directory.")
    allowed_roots = tuple(reader.artifact_roots)
    try:
        relative, validated = resolve_safe_path(
            root,
            canonical.relative_to(root),
            allowed_roots=allowed_roots,
            must_exist=True,
        )
    except (CommandBuildError, ValueError) as error:
        raise ArtifactReadError(str(error)) from error
    state = _marker_state(validated, reader)
    summaries: dict[str, Any] = {}
    payloads: dict[str, Any] = {}
    for summary in reader.summary_files:
        try:
            _, path = resolve_safe_path(
                root,
                f"{relative}/{summary.path}",
                allowed_roots=(relative,),
                must_exist=True,
            )
        except CommandBuildError:
            continue
        if not path.is_file():
            continue
        payload = safe_json_load(path)
        payloads[summary.path] = payload
        for label, dot_path in summary.fields.items():
            summaries[label] = extract_dot_path(payload, dot_path)
    return ArtifactRecord(
        reader_id=reader.id,
        root=validated,
        relative_path=relative,
        state=state,
        summaries=summaries,
        json_payloads=payloads,
    )


def comparison_rows(
    left: ArtifactRecord,
    right: ArtifactRecord,
    compare_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "field": field,
            "left": left.summaries.get(field),
            "right": right.summaries.get(field),
        }
        for field in compare_fields
    ]


def _marker_state(root: Path, reader: ReaderSpec) -> str:
    if any((root / marker).is_file() for marker in reader.success_markers):
        return "completed"
    if any((root / marker).is_file() for marker in reader.failure_markers):
        return "failed"
    return "running"


def _require_regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ArtifactReadError(f"Expected a regular non-link file: {path}.")
