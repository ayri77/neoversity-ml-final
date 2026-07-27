from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from src.churn_ml.autogluon_completion import (
    CompletionExpectations,
    load_and_validate_completion,
    validate_completion_payload,
)
from src.churn_ml.autogluon_inspection import predictor_report
from tests.test_autogluon_inspection import FakePredictor, realistic_autogluon_info


def expectations() -> CompletionExpectations:
    return CompletionExpectations(
        worker_pid=1234,
        profile_id="realtabpfn_only_v1",
        profile_sha256="a" * 64,
        dataset_version="v3_targeted_missingness",
        requested_seed=42,
        resolved_families=("REALTABPFN-V2",),
    )


def valid_payload() -> dict[str, Any]:
    started = datetime.now(UTC)
    completed = started + timedelta(seconds=2)
    return {
        "schema_version": 1,
        "status": "completed",
        "worker_pid": 1234,
        "started_at_utc": started.isoformat().replace("+00:00", "Z"),
        "completed_at_utc": completed.isoformat().replace("+00:00", "Z"),
        "duration_seconds": 2.0,
        "profile_id": "realtabpfn_only_v1",
        "profile_sha256": "a" * 64,
        "dataset_version": "v3_targeted_missingness",
        "predictor_relative_path": "predictor",
        "resolved_families": ["REALTABPFN-V2"],
        "model_names": ["FakeModel"],
        "best_model": "FakeModel",
        "decision_threshold": 0.25,
        "autogluon_version": "1.5.0",
        "python_version": "3.12.12",
        "requested_seed": 42,
        "effective_seed": 42,
    }


def inspection_summary() -> dict[str, Any]:
    return {
        "models": ["FakeModel"],
        "best_model": "FakeModel",
        "decision_threshold": 0.25,
        "requested_seed": 42,
        "effective_seed": 42,
    }


def test_valid_completion_payload() -> None:
    result = validate_completion_payload(
        valid_payload(),
        expectations(),
        inspection_summary=inspection_summary(),
    )
    assert result.valid


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda payload: payload.pop("python_version"), "worker_result_missing_fields"),
        (
            lambda payload: payload.update({"unknown": 1}),
            "worker_result_unknown_fields",
        ),
        (
            lambda payload: payload.update({"worker_pid": True}),
            "worker_result_wrong_field_type",
        ),
        (
            lambda payload: payload.update({"worker_pid": 4321}),
            "worker_result_pid_mismatch",
        ),
        (
            lambda payload: payload.update({"profile_id": "wrong"}),
            "worker_result_profile_id_mismatch",
        ),
        (
            lambda payload: payload.update({"profile_sha256": "b" * 64}),
            "worker_result_profile_hash_mismatch",
        ),
        (
            lambda payload: payload.update({"dataset_version": "wrong"}),
            "worker_result_dataset_version_mismatch",
        ),
        (
            lambda payload: payload.update({"predictor_relative_path": "other"}),
            "worker_result_predictor_path_mismatch",
        ),
        (
            lambda payload: payload.update({"decision_threshold": float("nan")}),
            "worker_result_nonfinite_threshold",
        ),
        (
            lambda payload: payload.update({"duration_seconds": float("inf")}),
            "worker_result_invalid_duration",
        ),
        (
            lambda payload: payload.update(
                {"completed_at_utc": "2000-01-01T00:00:00Z"}
            ),
            "worker_result_timestamp_order_invalid",
        ),
        (
            lambda payload: payload.update({"duration_seconds": 100.0}),
            "worker_result_duration_inconsistent",
        ),
        (
            lambda payload: payload.update({"effective_seed": 43}),
            "worker_result_effective_seed_mismatch",
        ),
        (
            lambda payload: payload.update({"autogluon_version": ""}),
            "worker_result_runtime_version_invalid",
        ),
    ],
)
def test_invalid_completion_payloads(
    mutation: Any,
    reason: str,
) -> None:
    payload = valid_payload()
    mutation(payload)
    result = validate_completion_payload(
        payload,
        expectations(),
        inspection_summary=inspection_summary(),
    )
    assert not result.valid
    assert reason in result.reason_codes


def test_completion_must_match_inspection() -> None:
    inspection = inspection_summary()
    inspection["models"] = ["DifferentModel"]
    inspection["best_model"] = "DifferentModel"
    inspection["decision_threshold"] = 0.5
    result = validate_completion_payload(
        valid_payload(),
        expectations(),
        inspection_summary=inspection,
    )
    assert {
        "worker_result_model_names_mismatch",
        "worker_result_best_model_mismatch",
        "worker_result_threshold_mismatch",
    }.issubset(result.reason_codes)


def test_weighted_ensemble_scoped_seed_report_agrees_with_completion() -> None:
    inspection = predictor_report(
        FakePredictor(realistic_autogluon_info([42], auxiliary_seed=0)),
        requested_seed=42,
        configured_families=["REALTABPFN-V2"],
    )
    payload = valid_payload()
    payload["model_names"] = inspection["models"]
    payload["best_model"] = inspection["best_model"]
    payload["decision_threshold"] = inspection["decision_threshold"]
    payload["effective_seed"] = inspection["effective_seed"]
    result = validate_completion_payload(
        payload,
        expectations(),
        inspection_summary=inspection,
    )
    assert result.valid
    assert inspection["effective_seed_status"] == "verified"
    assert inspection["auxiliary_effective_seeds_observed"] == [0]


def test_missing_and_invalid_json_completion_files(tmp_path: Path) -> None:
    path = tmp_path / "worker_result.json"
    missing = load_and_validate_completion(path, expectations())
    assert missing.reason_codes == ("worker_result_missing",)
    path.write_text('{"status":', encoding="utf-8")
    invalid = load_and_validate_completion(path, expectations())
    assert invalid.reason_codes == ("worker_result_invalid_json",)
    path.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    minimal = load_and_validate_completion(path, expectations())
    assert "worker_result_missing_fields" in minimal.reason_codes
