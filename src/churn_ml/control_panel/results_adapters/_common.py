"""Shared helpers for Results adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.churn_ml.control_panel.artifacts import ArtifactRecord, safe_json_load
from src.churn_ml.control_panel.results_fields import (
    ResultField,
    available,
    from_optional,
    invalid,
    missing_unexpectedly,
    not_applicable,
    unavailable_in_schema,
)


def payload(artifact: ArtifactRecord, relative: str) -> Mapping[str, Any] | None:
    raw = artifact.json_payloads.get(relative)
    return raw if isinstance(raw, dict) else None


def load_json(root: Path, relative: str) -> Mapping[str, Any] | None:
    path = root / relative
    if not path.is_file():
        return None
    try:
        loaded = safe_json_load(path)
    except Exception:
        return None
    return loaded if isinstance(loaded, dict) else None


def state_field(artifact: ArtifactRecord) -> ResultField:
    if artifact.state == "invalid":
        return invalid(
            artifact.diagnostic or "Conflicting or unsafe terminal markers.",
            value=artifact.state,
            source="markers",
        )
    if artifact.state in {"completed", "failed", "running"}:
        return available(artifact.state, source="markers")
    return invalid(
        f"Unrecognized artifact state {artifact.state!r}.",
        value=artifact.state,
        source="markers",
    )


def schema_field(
    value: Any,
    *,
    expected: set[Any] | frozenset[Any] | None = None,
    source: str,
) -> ResultField:
    if value is None:
        return missing_unexpectedly("Schema version is absent.", source=source)
    if expected is not None and value not in expected:
        return invalid(
            f"Unsupported schema version {value!r}.",
            value=value,
            source=source,
        )
    return available(value, source=source)


def mapping_get(
    payload: Mapping[str, Any] | None,
    key: str,
    *,
    source: str,
    missing: ResultField | None = None,
) -> ResultField:
    if payload is None:
        return missing or missing_unexpectedly(
            f"Payload unavailable for {key}.", source=source
        )
    if key not in payload:
        return missing or missing_unexpectedly(
            f"Key {key!r} is absent.", source=source
        )
    return from_optional(payload.get(key), source=f"{source}.{key}", missing=missing)


def nested_number(
    payload: Mapping[str, Any] | None,
    *path: str,
    source: str,
    missing: ResultField | None = None,
) -> ResultField:
    current: Any = payload
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return missing or missing_unexpectedly(
                f"Path {'.'.join(path)} is absent.", source=source
            )
        current = current[part]
    if current is None:
        return missing or missing_unexpectedly(
            f"Path {'.'.join(path)} is null.", source=source
        )
    try:
        return available(float(current), source=source)
    except (TypeError, ValueError):
        return invalid(
            f"Path {'.'.join(path)} is not numeric.",
            value=current,
            source=source,
        )


def bool_field(
    payload: Mapping[str, Any] | None,
    key: str,
    *,
    source: str,
    missing: ResultField | None = None,
) -> ResultField:
    if payload is None or key not in payload:
        return missing or missing_unexpectedly(
            f"Boolean {key!r} is absent.", source=source
        )
    value = payload[key]
    if isinstance(value, bool):
        return available(value, source=f"{source}.{key}")
    return invalid(
        f"Boolean {key!r} has non-boolean type.",
        value=value,
        source=f"{source}.{key}",
    )


def na(message: str) -> ResultField:
    return not_applicable(message)


def unrecorded(message: str) -> ResultField:
    return unavailable_in_schema(message)
