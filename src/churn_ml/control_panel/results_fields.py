"""Normalized field-status semantics for schema-aware Results projections."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class FieldStatus(str, Enum):
    AVAILABLE = "available"
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE_IN_SCHEMA = "unavailable_in_schema"
    MISSING_UNEXPECTEDLY = "missing_unexpectedly"
    INVALID = "invalid"


@dataclass(frozen=True)
class ResultField:
    status: FieldStatus
    value: Any = None
    message: str | None = None
    source: str | None = None

    def is_available(self) -> bool:
        return self.status is FieldStatus.AVAILABLE


def available(value: Any, *, source: str | None = None) -> ResultField:
    return ResultField(status=FieldStatus.AVAILABLE, value=value, source=source)


def not_applicable(
    message: str | None = None, *, source: str | None = None
) -> ResultField:
    return ResultField(
        status=FieldStatus.NOT_APPLICABLE,
        value=None,
        message=message,
        source=source,
    )


def unavailable_in_schema(
    message: str | None = None, *, source: str | None = None
) -> ResultField:
    return ResultField(
        status=FieldStatus.UNAVAILABLE_IN_SCHEMA,
        value=None,
        message=message,
        source=source,
    )


def missing_unexpectedly(
    message: str | None = None, *, source: str | None = None
) -> ResultField:
    return ResultField(
        status=FieldStatus.MISSING_UNEXPECTEDLY,
        value=None,
        message=message,
        source=source,
    )


def invalid(
    message: str | None = None,
    *,
    value: Any = None,
    source: str | None = None,
) -> ResultField:
    return ResultField(
        status=FieldStatus.INVALID,
        value=value,
        message=message,
        source=source,
    )


def from_optional(
    value: Any,
    *,
    source: str | None = None,
    missing: ResultField | None = None,
    treat_empty_string_as_missing: bool = True,
) -> ResultField:
    """Classify an optional payload value without conflating 0/False with missing."""
    if value is None:
        return missing or missing_unexpectedly(
            "Expected value is absent.", source=source
        )
    if treat_empty_string_as_missing and value == "":
        return missing or missing_unexpectedly(
            "Expected value is empty.", source=source
        )
    return available(value, source=source)


DISPLAY_LABELS: dict[FieldStatus, str] = {
    FieldStatus.AVAILABLE: "",
    FieldStatus.NOT_APPLICABLE: "N/A",
    FieldStatus.UNAVAILABLE_IN_SCHEMA: "Not recorded",
    FieldStatus.MISSING_UNEXPECTEDLY: "Missing",
    FieldStatus.INVALID: "Invalid",
}


def render_field(field: ResultField, *, formatter: Any | None = None) -> str:
    """Render a ResultField for compact table/detail display."""
    if field.status is FieldStatus.AVAILABLE:
        value = field.value
        if formatter is not None:
            try:
                return str(formatter(value))
            except Exception:
                return str(value)
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)
    label = DISPLAY_LABELS[field.status]
    if field.message and field.status in {
        FieldStatus.MISSING_UNEXPECTEDLY,
        FieldStatus.INVALID,
    }:
        return f"{label} ({field.message})"
    return label


def field_sort_key(field: ResultField) -> tuple[int, str]:
    """Sort helper: available values first, then explicit statuses."""
    if field.status is FieldStatus.AVAILABLE:
        return (0, str(field.value))
    order = {
        FieldStatus.NOT_APPLICABLE: 1,
        FieldStatus.UNAVAILABLE_IN_SCHEMA: 2,
        FieldStatus.MISSING_UNEXPECTEDLY: 3,
        FieldStatus.INVALID: 4,
    }
    return (order.get(field.status, 9), field.status.value)
