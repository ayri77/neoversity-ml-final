"""Exact worker-completion contract shared by supervision and inspection."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


WORKER_COMPLETION_SCHEMA_VERSION = 1
PREDICTOR_RELATIVE_PATH = "predictor"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SCHEMA: dict[str, type[Any] | tuple[type[Any], ...]] = {
    "schema_version": int,
    "status": str,
    "worker_pid": int,
    "started_at_utc": str,
    "completed_at_utc": str,
    "duration_seconds": float,
    "profile_id": str,
    "profile_sha256": str,
    "dataset_version": str,
    "predictor_relative_path": str,
    "resolved_families": list,
    "model_names": list,
    "best_model": str,
    "decision_threshold": float,
    "autogluon_version": str,
    "python_version": str,
    "requested_seed": int,
    "effective_seed": (int, type(None)),
}


@dataclass(frozen=True)
class CompletionExpectations:
    """Supervisor-owned values that a worker cannot redefine."""

    worker_pid: int
    profile_id: str
    profile_sha256: str
    dataset_version: str
    requested_seed: int
    resolved_families: tuple[str, ...] | None = None


@dataclass(frozen=True)
class CompletionValidation:
    """A parsed completion payload plus stable reason codes."""

    payload: dict[str, Any] | None
    reason_codes: tuple[str, ...]
    details: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.reason_codes and self.payload is not None


def load_and_validate_completion(
    path: Path,
    expectations: CompletionExpectations,
    *,
    inspection_summary: dict[str, Any] | None = None,
) -> CompletionValidation:
    """Load strict JSON and validate identity, timing, and inspection agreement."""
    if not path.is_file():
        return CompletionValidation(None, ("worker_result_missing",), ())
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return CompletionValidation(
            None,
            ("worker_result_invalid_json",),
            (f"{type(error).__name__}: {error}",),
        )
    if type(payload) is not dict:
        return CompletionValidation(
            None,
            ("worker_result_not_object",),
            (f"found {type(payload).__name__}",),
        )
    return validate_completion_payload(
        payload,
        expectations,
        inspection_summary=inspection_summary,
    )


def validate_completion_payload(
    payload: dict[str, Any],
    expectations: CompletionExpectations,
    *,
    inspection_summary: dict[str, Any] | None = None,
) -> CompletionValidation:
    """Validate the exact schema without coercion or bool-as-int acceptance."""
    reasons: list[str] = []
    details: list[str] = []
    expected_keys = set(_SCHEMA)
    actual_keys = set(payload)
    missing = sorted(expected_keys - actual_keys)
    unknown = sorted(actual_keys - expected_keys)
    if missing:
        reasons.append("worker_result_missing_fields")
        details.append(f"missing={','.join(missing)}")
    if unknown:
        reasons.append("worker_result_unknown_fields")
        details.append(f"unknown={','.join(unknown)}")
    for key in sorted(expected_keys & actual_keys):
        accepted = _SCHEMA[key]
        accepted_types = accepted if isinstance(accepted, tuple) else (accepted,)
        if not any(type(payload[key]) is item for item in accepted_types):
            reasons.append("worker_result_wrong_field_type")
            details.append(
                f"{key}={type(payload[key]).__name__}; expected="
                + "|".join(item.__name__ for item in accepted_types)
            )
    if missing or any(reason == "worker_result_wrong_field_type" for reason in reasons):
        return _result(payload, reasons, details)

    if payload["schema_version"] != WORKER_COMPLETION_SCHEMA_VERSION:
        reasons.append("worker_result_schema_version_mismatch")
    if payload["status"] != "completed":
        reasons.append("worker_result_status_not_completed")
    if payload["worker_pid"] <= 0 or payload["worker_pid"] != expectations.worker_pid:
        reasons.append("worker_result_pid_mismatch")
    if payload["profile_id"] != expectations.profile_id:
        reasons.append("worker_result_profile_id_mismatch")
    if (
        not _SHA256_PATTERN.fullmatch(payload["profile_sha256"])
        or payload["profile_sha256"] != expectations.profile_sha256
    ):
        reasons.append("worker_result_profile_hash_mismatch")
    if payload["dataset_version"] != expectations.dataset_version:
        reasons.append("worker_result_dataset_version_mismatch")
    if payload["predictor_relative_path"] != PREDICTOR_RELATIVE_PATH:
        reasons.append("worker_result_predictor_path_mismatch")
    if not payload["autogluon_version"] or not payload["python_version"]:
        reasons.append("worker_result_runtime_version_invalid")
    if payload["requested_seed"] != expectations.requested_seed:
        reasons.append("worker_result_seed_mismatch")
    if (
        payload["effective_seed"] is not None
        and payload["effective_seed"] != payload["requested_seed"]
    ):
        reasons.append("worker_result_effective_seed_mismatch")

    _validate_string_list(payload["resolved_families"], "resolved_families", reasons)
    _validate_string_list(payload["model_names"], "model_names", reasons)
    if expectations.resolved_families is not None and tuple(
        payload["resolved_families"]
    ) != tuple(expectations.resolved_families):
        reasons.append("worker_result_families_mismatch")
    if not payload["best_model"] or payload["best_model"] not in payload["model_names"]:
        reasons.append("worker_result_best_model_invalid")
    threshold = payload["decision_threshold"]
    if not math.isfinite(threshold):
        reasons.append("worker_result_nonfinite_threshold")
    duration = payload["duration_seconds"]
    if not math.isfinite(duration) or duration < 0:
        reasons.append("worker_result_invalid_duration")
    _validate_timing(payload, reasons, details)
    if inspection_summary is not None:
        reasons.extend(completion_inspection_mismatches(payload, inspection_summary))
    return _result(payload, reasons, details)


def completion_inspection_mismatches(
    payload: dict[str, Any], inspection_summary: dict[str, Any]
) -> list[str]:
    """Compare completion declarations with inspection produced from the predictor."""
    mismatches: list[str] = []
    if inspection_summary.get("models") != payload.get("model_names"):
        mismatches.append("worker_result_model_names_mismatch")
    if inspection_summary.get("best_model") != payload.get("best_model"):
        mismatches.append("worker_result_best_model_mismatch")
    if inspection_summary.get("requested_seed") != payload.get("requested_seed"):
        mismatches.append("worker_result_requested_seed_inspection_mismatch")
    if inspection_summary.get("effective_seed") != payload.get("effective_seed"):
        mismatches.append("worker_result_effective_seed_inspection_mismatch")
    observed_threshold = inspection_summary.get("decision_threshold")
    declared_threshold = payload.get("decision_threshold")
    if (
        isinstance(observed_threshold, bool)
        or not isinstance(observed_threshold, (int, float))
        or type(declared_threshold) is not float
        or not math.isclose(
            float(observed_threshold),
            declared_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        mismatches.append("worker_result_threshold_mismatch")
    return mismatches


def parse_utc_timestamp(value: str) -> datetime:
    """Parse an ISO-8601 timestamp that explicitly denotes UTC."""
    if not value.endswith("Z"):
        raise ValueError("timestamp must end in Z")
    parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("timestamp must be UTC")
    return parsed


def _validate_timing(
    payload: dict[str, Any], reasons: list[str], details: list[str]
) -> None:
    try:
        started = parse_utc_timestamp(payload["started_at_utc"])
        completed = parse_utc_timestamp(payload["completed_at_utc"])
    except (TypeError, ValueError) as error:
        reasons.append("worker_result_invalid_timestamp")
        details.append(str(error))
        return
    wall_duration = (completed - started).total_seconds()
    if wall_duration < 0:
        reasons.append("worker_result_timestamp_order_invalid")
        return
    tolerance = max(2.0, wall_duration * 0.05)
    if abs(payload["duration_seconds"] - wall_duration) > tolerance:
        reasons.append("worker_result_duration_inconsistent")


def _validate_string_list(value: list[Any], field: str, reasons: list[str]) -> None:
    if not value or any(type(item) is not str or not item for item in value):
        reasons.append(f"worker_result_{field}_invalid")
    elif len(value) != len(set(value)):
        reasons.append(f"worker_result_{field}_duplicate")


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _result(
    payload: dict[str, Any], reasons: list[str], details: list[str]
) -> CompletionValidation:
    return CompletionValidation(
        payload,
        tuple(dict.fromkeys(reasons)),
        tuple(details),
    )
