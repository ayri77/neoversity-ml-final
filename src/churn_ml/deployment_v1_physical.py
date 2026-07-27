from __future__ import annotations

import csv
import io
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


CANONICAL_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
CANONICAL_INTEGER = re.compile(r"^(?:0|[1-9]\d*)$")
CANONICAL_FLOAT = re.compile(r"^(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$")


class DeploymentPhysicalError(ValueError):
    """Raised when persisted physical representation is not exact."""


def format_utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise DeploymentPhysicalError("Timestamp must be explicitly UTC.")
    return (
        value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    )


def parse_utc_timestamp(value: Any) -> datetime:
    if type(value) is not str or CANONICAL_TIMESTAMP.fullmatch(value) is None:
        raise DeploymentPhysicalError("Timestamp is not canonical UTC RFC 3339.")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise DeploymentPhysicalError("Timestamp is malformed.") from error
    if format_utc_timestamp(parsed) != value:
        raise DeploymentPhysicalError("Timestamp canonical round-trip differs.")
    return parsed


def canonical_csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(
        buffer,
        index=False,
        sep=",",
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
        encoding=None,
    )
    return buffer.getvalue().encode("utf-8")


def require_columns(frame: pd.DataFrame, expected: list[str], label: str) -> None:
    if frame.columns.tolist() != expected or frame.columns.has_duplicates:
        raise DeploymentPhysicalError(f"{label} columns/order differ.")


def require_dtype(
    frame: pd.DataFrame,
    column: str,
    expected: str,
    label: str,
) -> None:
    actual = str(frame[column].dtype)
    if actual != expected:
        raise DeploymentPhysicalError(
            f"{label}.{column} physical dtype {actual!r} differs from {expected!r}."
        )
    if frame[column].isna().any():
        raise DeploymentPhysicalError(f"{label}.{column} contains nulls.")


def require_exact_integer_values(
    series: pd.Series,
    *,
    allowed: set[int] | None,
    minimum: int | None,
    label: str,
) -> None:
    if str(series.dtype) not in {"int8", "int64"}:
        raise DeploymentPhysicalError(f"{label} is not an exact signed integer dtype.")
    values = series.to_numpy(copy=False)
    if allowed is not None and not set(int(value) for value in values).issubset(
        allowed
    ):
        raise DeploymentPhysicalError(f"{label} contains an invalid integer value.")
    if minimum is not None and bool((values < minimum).any()):
        raise DeploymentPhysicalError(f"{label} is below its allowed minimum.")


def require_float64_probabilities(series: pd.Series, label: str) -> np.ndarray:
    if str(series.dtype) != "float64" or series.isna().any():
        raise DeploymentPhysicalError(f"{label} must be nonnullable float64.")
    values = series.to_numpy(copy=False)
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or bool(((values < 0.0) | (values > 1.0)).any())
    ):
        raise DeploymentPhysicalError(f"{label} contains invalid probabilities.")
    return values


def require_row_id_dtype(
    series: pd.Series,
    expected_values: tuple[Any, ...],
    label: str,
) -> None:
    expected = pd.Series(list(expected_values), name=series.name)
    expected_dtype = str(expected.dtype)
    if expected_dtype == "object":
        if any(type(value) is not str for value in expected_values):
            raise DeploymentPhysicalError(f"{label} has unsupported mixed/object IDs.")
        if str(series.dtype) != "object":
            raise DeploymentPhysicalError(f"{label} string ID dtype differs.")
    elif expected_dtype == "int64":
        if str(series.dtype) != "int64":
            raise DeploymentPhysicalError(f"{label} integer ID dtype differs.")
    else:
        raise DeploymentPhysicalError(f"{label} ID dtype is unsupported.")
    if series.isna().any() or tuple(series.tolist()) != expected_values:
        raise DeploymentPhysicalError(f"{label} values/order differ.")


def read_exact_bag_summary(path: Path, columns: list[str]) -> pd.DataFrame:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf") or b"\r\n" in raw or not raw.endswith(b"\n"):
        raise DeploymentPhysicalError("Bag summary CSV encoding/newlines differ.")
    try:
        text = raw.decode("utf-8")
        rows = list(csv.reader(io.StringIO(text, newline="")))
    except (UnicodeDecodeError, csv.Error) as error:
        raise DeploymentPhysicalError("Bag summary CSV is malformed.") from error
    if not rows or rows[0] != columns or any(len(row) != len(columns) for row in rows):
        raise DeploymentPhysicalError("Bag summary CSV physical schema differs.")
    integer_fields = {
        "bag_index",
        "bag_seed",
        "training_rows",
        "test_rows",
        "probability_bytes",
    }
    boolean_fields = {"early_stopping", "evaluation_set", "model_persisted"}
    for row in rows[1:]:
        record = dict(zip(columns, row, strict=True))
        for field in integer_fields:
            if CANONICAL_INTEGER.fullmatch(record[field]) is None:
                raise DeploymentPhysicalError(
                    f"Bag summary {field} is not a canonical integer."
                )
        for field in boolean_fields:
            if record[field] not in {"True", "False"}:
                raise DeploymentPhysicalError(
                    f"Bag summary {field} is not a canonical boolean."
                )
        if CANONICAL_FLOAT.fullmatch(record["duration_seconds"]) is None:
            raise DeploymentPhysicalError(
                "Bag summary duration_seconds is not a canonical float."
            )
    string_fields = (
        set(columns) - integer_fields - boolean_fields - {"duration_seconds"}
    )
    frame = pd.read_csv(path, dtype={field: str for field in string_fields})
    require_columns(frame, columns, "bag_summary")
    for field in integer_fields:
        require_dtype(frame, field, "int64", "bag_summary")
    for field in boolean_fields:
        require_dtype(frame, field, "bool", "bag_summary")
    require_dtype(frame, "duration_seconds", "float64", "bag_summary")
    for field in string_fields:
        require_dtype(frame, field, "object", "bag_summary")
    return frame


def exact_mapping(
    value: Any,
    *,
    keys: set[str],
    types: Mapping[str, type],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise DeploymentPhysicalError(f"{label} keys differ.")
    for key, expected in types.items():
        if type(value[key]) is not expected:
            raise DeploymentPhysicalError(f"{label}.{key} type differs.")
    return dict(value)
