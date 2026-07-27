from __future__ import annotations

from typing import Any, Mapping


class ExperimentV2ContractError(ValueError):
    """Raised when a v2 component violates its exact compatibility contract."""


def first_exact_difference(
    actual: Any,
    expected: Any,
    path: str,
) -> str | None:
    """Return the first recursive type-or-value mismatch with a precise path."""
    if isinstance(expected, dict):
        if not isinstance(actual, Mapping):
            return f"{path} must be a mapping."
        actual_keys = set(actual)
        expected_keys = set(expected)
        if actual_keys != expected_keys:
            return (
                f"{path} keys differ; missing={sorted(expected_keys - actual_keys)}, "
                f"unknown={sorted(actual_keys - expected_keys)}."
            )
        for key, expected_value in expected.items():
            difference = first_exact_difference(
                actual[key],
                expected_value,
                f"{path}.{key}",
            )
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return f"{path} must be a list."
        if len(actual) != len(expected):
            return f"{path} must contain exactly {len(expected)} items."
        for index, (actual_value, expected_value) in enumerate(
            zip(actual, expected, strict=True)
        ):
            difference = first_exact_difference(
                actual_value,
                expected_value,
                f"{path}[{index}]",
            )
            if difference is not None:
                return difference
        return None
    if type(actual) is not type(expected) or actual != expected:
        return (
            f"{path} must be exactly {expected!r} "
            f"({type(expected).__name__}); got {actual!r} "
            f"({type(actual).__name__})."
        )
    return None
