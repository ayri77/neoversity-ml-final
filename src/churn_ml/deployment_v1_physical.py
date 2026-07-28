from __future__ import annotations

import csv
import io
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


CANONICAL_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
CANONICAL_INTEGER = re.compile(r"^(?:0|[1-9]\d*)$")
CANONICAL_FLOAT = re.compile(r"^(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$")
BAG_SUMMARY_COLUMNS = [
    "component_id",
    "adapter_id",
    "bag_index",
    "bag_seed",
    "training_rows",
    "test_rows",
    "parameter_sha256",
    "probability_sha256",
    "probability_column",
    "row_identity_sha256",
    "probability_bytes",
    "duration_seconds",
    "early_stopping",
    "evaluation_set",
    "model_persisted",
]
BAG_SUMMARY_INTEGER_FIELDS = {
    "bag_index",
    "bag_seed",
    "training_rows",
    "test_rows",
    "probability_bytes",
}
BAG_SUMMARY_BOOLEAN_FIELDS = {
    "early_stopping",
    "evaluation_set",
    "model_persisted",
}
BAG_SUMMARY_FLOAT_FIELDS = {"duration_seconds"}
BAG_SUMMARY_STRING_FIELDS = (
    set(BAG_SUMMARY_COLUMNS)
    - BAG_SUMMARY_INTEGER_FIELDS
    - BAG_SUMMARY_BOOLEAN_FIELDS
    - BAG_SUMMARY_FLOAT_FIELDS
)


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


def canonical_bag_summary_bytes(
    records: Sequence[Mapping[str, Any]],
) -> bytes:
    """Serialize the versioned bag summary with one exact unquoted byte grammar."""
    lines = [",".join(BAG_SUMMARY_COLUMNS)]
    for row_index, source in enumerate(records):
        if type(source) is not dict or set(source) != set(BAG_SUMMARY_COLUMNS):
            raise DeploymentPhysicalError(
                f"Bag summary row {row_index} keys/order source differs."
            )
        tokens: list[str] = []
        for field in BAG_SUMMARY_COLUMNS:
            value = source[field]
            label = f"bag_summary[{row_index}].{field}"
            if field in BAG_SUMMARY_INTEGER_FIELDS:
                if type(value) is not int or value < 0:
                    raise DeploymentPhysicalError(
                        f"{label} must be a nonnegative exact integer."
                    )
                if field == "bag_index" and value < 1:
                    raise DeploymentPhysicalError(
                        f"{label} must be a positive exact integer."
                    )
                token = str(value)
            elif field in BAG_SUMMARY_BOOLEAN_FIELDS:
                if type(value) is not bool:
                    raise DeploymentPhysicalError(f"{label} must be an exact boolean.")
                token = "True" if value else "False"
            elif field in BAG_SUMMARY_FLOAT_FIELDS:
                if type(value) is not float or not math.isfinite(value) or value < 0.0:
                    raise DeploymentPhysicalError(
                        f"{label} must be a nonnegative finite exact float."
                    )
                token = repr(value)
            else:
                if type(value) is not str or not value:
                    raise DeploymentPhysicalError(
                        f"{label} must be a non-empty exact string."
                    )
                token = value
            if any(character in token for character in (",", '"', "\r", "\n")):
                raise DeploymentPhysicalError(
                    f"{label} cannot be represented by the canonical unquoted policy."
                )
            tokens.append(token)
        lines.append(",".join(tokens))
    return ("\n".join(lines) + "\n").encode("utf-8")


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


def read_exact_bag_summary(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    if (
        raw.startswith(b"\xef\xbb\xbf")
        or b"\r" in raw
        or not raw.endswith(b"\n")
        or raw.endswith(b"\n\n")
        or b'"' in raw
    ):
        raise DeploymentPhysicalError("Bag summary CSV encoding/newlines differ.")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DeploymentPhysicalError("Bag summary CSV is malformed.") from error
    lines = text[:-1].split("\n")
    if not lines or lines[0] != ",".join(BAG_SUMMARY_COLUMNS):
        raise DeploymentPhysicalError("Bag summary CSV physical schema differs.")
    records: list[dict[str, Any]] = []
    for line in lines[1:]:
        tokens = line.split(",")
        if len(tokens) != len(BAG_SUMMARY_COLUMNS):
            raise DeploymentPhysicalError("Bag summary CSV physical schema differs.")
        lexical = dict(zip(BAG_SUMMARY_COLUMNS, tokens, strict=True))
        record: dict[str, Any] = {}
        for field in BAG_SUMMARY_INTEGER_FIELDS:
            if CANONICAL_INTEGER.fullmatch(lexical[field]) is None:
                raise DeploymentPhysicalError(
                    f"Bag summary {field} is not a canonical integer."
                )
            record[field] = int(lexical[field])
        for field in BAG_SUMMARY_BOOLEAN_FIELDS:
            if lexical[field] not in {"True", "False"}:
                raise DeploymentPhysicalError(
                    f"Bag summary {field} is not a canonical boolean."
                )
            record[field] = lexical[field] == "True"
        duration_token = lexical["duration_seconds"]
        if CANONICAL_FLOAT.fullmatch(duration_token) is None:
            raise DeploymentPhysicalError(
                "Bag summary duration_seconds is not a canonical float."
            )
        duration = float(duration_token)
        if (
            not math.isfinite(duration)
            or duration < 0.0
            or repr(duration) != duration_token
        ):
            raise DeploymentPhysicalError(
                "Bag summary duration_seconds is not the canonical float token."
            )
        record["duration_seconds"] = duration
        for field in BAG_SUMMARY_STRING_FIELDS:
            token = lexical[field]
            if not token:
                raise DeploymentPhysicalError(
                    f"Bag summary {field} must be a non-empty string."
                )
            record[field] = token
        records.append(record)
    if not records or raw != canonical_bag_summary_bytes(records):
        raise DeploymentPhysicalError("Bag summary raw bytes are not canonical.")
    frame = pd.DataFrame.from_records(records, columns=BAG_SUMMARY_COLUMNS)
    require_columns(frame, BAG_SUMMARY_COLUMNS, "bag_summary")
    for field in BAG_SUMMARY_INTEGER_FIELDS:
        require_dtype(frame, field, "int64", "bag_summary")
    for field in BAG_SUMMARY_BOOLEAN_FIELDS:
        require_dtype(frame, field, "bool", "bag_summary")
    require_dtype(frame, "duration_seconds", "float64", "bag_summary")
    for field in BAG_SUMMARY_STRING_FIELDS:
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


def require_exact_tree_schema(actual: Any, expected: Any, label: str) -> None:
    """Require recursively identical keys, shape, and exact primitive types."""
    if type(actual) is not type(expected):
        raise DeploymentPhysicalError(f"{label} exact type differs.")
    if type(expected) is dict:
        if set(actual) != set(expected):
            raise DeploymentPhysicalError(f"{label} keys differ.")
        for key in expected:
            if type(key) is not str:
                raise DeploymentPhysicalError(f"{label} contains a non-string key.")
            require_exact_tree_schema(actual[key], expected[key], f"{label}.{key}")
        return
    if type(expected) is list:
        if len(actual) != len(expected):
            raise DeploymentPhysicalError(f"{label} list length differs.")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            require_exact_tree_schema(actual_item, expected_item, f"{label}[{index}]")
        return
    if type(expected) is float and not math.isfinite(actual):
        raise DeploymentPhysicalError(f"{label} must be finite.")
    if type(expected) not in {str, int, float, bool, type(None)}:
        raise DeploymentPhysicalError(f"{label} has an unsupported tree type.")


def require_exact_tree(actual: Any, expected: Any, label: str) -> None:
    """Require recursively identical keys, shape, exact primitive types, and values."""
    if type(actual) is not type(expected):
        raise DeploymentPhysicalError(f"{label} exact type differs.")
    if type(expected) is dict:
        if set(actual) != set(expected):
            raise DeploymentPhysicalError(f"{label} keys differ.")
        for key in expected:
            if type(key) is not str:
                raise DeploymentPhysicalError(f"{label} contains a non-string key.")
            require_exact_tree(actual[key], expected[key], f"{label}.{key}")
        return
    if type(expected) is list:
        if len(actual) != len(expected):
            raise DeploymentPhysicalError(f"{label} list length differs.")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            require_exact_tree(actual_item, expected_item, f"{label}[{index}]")
        return
    if type(expected) is float and (
        not math.isfinite(actual) or not math.isfinite(expected)
    ):
        raise DeploymentPhysicalError(f"{label} must be finite.")
    if type(expected) not in {str, int, float, bool, type(None)}:
        raise DeploymentPhysicalError(f"{label} has an unsupported tree type.")
    if actual != expected:
        raise DeploymentPhysicalError(f"{label} value differs.")
