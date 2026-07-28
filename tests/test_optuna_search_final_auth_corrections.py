from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

from src.churn_ml.optuna_search_artifacts import (
    OptunaSearchArtifactError,
    load_optuna_search_result,
)
from src.churn_ml.optuna_search_export import export_best_candidate
from src.churn_ml.optuna_search_lifecycle import build_trial_failure_evidence
from src.churn_ml.research_data import canonical_sha256
from tests.optuna_auth_support import (
    PROJECT_ROOT,
    ensure_train_only_files,
    one_ulp_away,
    reauthenticate,
    run_authorized_completed_search,
)


@pytest.fixture(scope="module")
def completed_search() -> Iterator[tuple[Path, Path]]:
    pytest.importorskip("optuna")
    ensure_train_only_files()
    search_dir, operational = run_authorized_completed_search(n_trials=3)
    try:
        load_optuna_search_result(search_dir, project_root=PROJECT_ROOT)
        yield search_dir, operational
    finally:
        shutil.rmtree(operational, ignore_errors=True)
        shutil.rmtree(search_dir.parent, ignore_errors=True)


def _copy_report(valid: Path, operational: Path, name: str) -> Path:
    parent = operational / "final_auth_corruptions" / name
    target = parent / valid.name
    parent.mkdir(parents=True)
    shutil.copytree(valid, target)
    return target


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        return list(reader.fieldnames), list(reader)


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _canonical_alternate(token: str) -> str:
    return f"{token}0" if "." in token else f"{token}.0"


def _reject_production_paths(corrupted: Path) -> None:
    with pytest.raises(OptunaSearchArtifactError):
        load_optuna_search_result(corrupted, project_root=PROJECT_ROOT)
    output = corrupted.parent / "forged_candidate.yaml"
    with pytest.raises(OptunaSearchArtifactError):
        export_best_candidate(corrupted, output, project_root=PROJECT_ROOT)
    assert not output.exists()


def test_valid_metric_schema_persists_every_required_binding(
    completed_search: tuple[Path, Path],
) -> None:
    valid, _ = completed_search
    fields, rows = _read_csv(valid / "trial_metrics.csv")
    required = {
        "schema_version",
        "trial_number",
        "record_type",
        "repeat",
        "fold",
        "metric_name",
        "metric_value_token",
        "selected_threshold_token",
        "adapter_id",
        "candidate_identity_sha256",
        "search_identity_sha256",
        "assignment_identity_sha256",
        "threshold_policy_identity_sha256",
        "prediction_coverage_identity_sha256",
        "prediction_values_identity_sha256",
    }
    assert required <= set(fields)
    assert rows
    assert {row["schema_version"] for row in rows} == {"1"}
    assert {row["metric_name"] for row in rows} == {"balanced_accuracy"}
    assert all(row["metric_value_token"] == row["balanced_accuracy"] for row in rows)
    assert all(
        row["selected_threshold_token"]
        == (
            row["selected_threshold"]
            if row["record_type"] == "fold"
            else "not_applicable"
        )
        for row in rows
    )


MetricMutation = Callable[[list[str], list[dict[str, str]]], None]


def _mutate_metric_case(
    case: str,
    fields: list[str],
    rows: list[dict[str, str]],
) -> None:
    fold_index = next(i for i, row in enumerate(rows) if row["record_type"] == "fold")
    repeat_index = next(
        i for i, row in enumerate(rows) if row["record_type"] == "repeat"
    )
    fold = rows[fold_index]
    if case == "missing_metric_name_column":
        fields.remove("metric_name")
        for row in rows:
            row.pop("metric_name")
    elif case == "missing_metric_name_value":
        fold["metric_name"] = ""
    elif case == "wrong_metric_name":
        fold["metric_name"] = "roc_auc"
    elif case.startswith("wrong_") and case.endswith("_identity"):
        name = case.removeprefix("wrong_") + "_sha256"
        fold[name] = "0" * 64
    elif case == "extra_metric_row":
        extra = dict(fold)
        extra["fold"] = "99"
        rows.append(extra)
    elif case == "missing_metric_row":
        rows.pop(fold_index)
    elif case == "duplicate_metric_row":
        rows.append(dict(fold))
    elif case == "forged_adapter_identity":
        fold["adapter_id"] = "forged_adapter"
    elif case == "forged_candidate_identity":
        fold["candidate_identity_sha256"] = "0" * 64
    elif case == "wrong_repeat":
        fold["repeat"] = "99"
    elif case == "wrong_fold":
        fold["fold"] = "99"
    elif case == "wrong_trial":
        fold["trial_number"] = "99"
    elif case == "noncanonical_metric_token":
        alternate = _canonical_alternate(fold["balanced_accuracy"])
        fold["metric_value_token"] = alternate
        fold["balanced_accuracy"] = alternate
    elif case == "noncanonical_threshold_token":
        alternate = _canonical_alternate(fold["selected_threshold"])
        fold["selected_threshold_token"] = alternate
        fold["selected_threshold"] = alternate
    elif case == "one_ulp_fold_metric":
        changed = format(one_ulp_away(float(fold["balanced_accuracy"])), ".17g")
        fold["metric_value_token"] = changed
        fold["balanced_accuracy"] = changed
    elif case == "one_ulp_repeat_metric":
        changed = format(
            one_ulp_away(float(rows[repeat_index]["balanced_accuracy"])), ".17g"
        )
        rows[repeat_index]["metric_value_token"] = changed
        rows[repeat_index]["balanced_accuracy"] = changed
    else:
        raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    (
        "missing_metric_name_column",
        "missing_metric_name_value",
        "wrong_metric_name",
        "wrong_search_identity",
        "wrong_assignment_identity",
        "wrong_threshold_policy_identity",
        "wrong_prediction_coverage_identity",
        "wrong_prediction_values_identity",
        "extra_metric_row",
        "missing_metric_row",
        "duplicate_metric_row",
        "forged_adapter_identity",
        "forged_candidate_identity",
        "wrong_repeat",
        "wrong_fold",
        "wrong_trial",
        "noncanonical_metric_token",
        "noncanonical_threshold_token",
        "one_ulp_fold_metric",
        "one_ulp_repeat_metric",
    ),
)
def test_metric_schema_universe_and_identity_mutations_reject_production_paths(
    completed_search: tuple[Path, Path],
    case: str,
) -> None:
    valid, operational = completed_search
    corrupted = _copy_report(valid, operational, f"metric_{case}")
    path = corrupted / "trial_metrics.csv"
    fields, rows = _read_csv(path)
    _mutate_metric_case(case, fields, rows)
    _write_csv(path, fields, rows)
    reauthenticate(corrupted)
    _reject_production_paths(corrupted)


def test_one_ulp_objective_token_rejects_production_paths(
    completed_search: tuple[Path, Path],
) -> None:
    valid, operational = completed_search
    corrupted = _copy_report(valid, operational, "one_ulp_objective_exact")
    path = corrupted / "trials.csv"
    fields, rows = _read_csv(path)
    complete = next(row for row in rows if row["state"] == "COMPLETE")
    complete["objective"] = format(one_ulp_away(float(complete["objective"])), ".17g")
    _write_csv(path, fields, rows)
    reauthenticate(corrupted)
    _reject_production_paths(corrupted)


@pytest.mark.parametrize("case", ("one_ulp", "semantically_equivalent"))
def test_exact_prediction_identity_rejects_metric_preserving_mutations(
    completed_search: tuple[Path, Path],
    case: str,
) -> None:
    valid, operational = completed_search
    corrupted = _copy_report(valid, operational, f"prediction_{case}")
    path = corrupted / "trial_predictions.csv"
    fields, rows = _read_csv(path)
    selected: dict[str, str] | None = None
    replacement = 0.0
    for row in rows:
        probability = float(row["probability"])
        threshold = float(row["selected_threshold"])
        label = int(row["prediction"])
        if case == "one_ulp":
            candidate = one_ulp_away(probability)
        elif label == 1:
            candidate = probability + (1.0 - probability) / 2.0
        else:
            candidate = probability / 2.0
        if (
            0.0 <= candidate <= 1.0
            and candidate != probability
            and int(candidate >= threshold) == label
        ):
            selected = row
            replacement = candidate
            break
    assert selected is not None
    selected["probability"] = format(replacement, ".17g")
    _write_csv(path, fields, rows)
    reauthenticate(corrupted)
    _reject_production_paths(corrupted)


def _load_failure_row(
    path: Path,
) -> tuple[list[str], list[dict[str, str]], dict[str, str], dict[str, Any]]:
    fields, rows = _read_csv(path)
    failed = next(row for row in rows if row["state"] == "FAIL")
    evidence = json.loads(failed["failure_evidence_json"])
    assert isinstance(evidence, dict)
    return fields, rows, failed, evidence


def _store_failure_evidence(row: dict[str, str], evidence: dict[str, Any]) -> None:
    row["failure_evidence_json"] = json.dumps(
        evidence, sort_keys=True, separators=(",", ":")
    )


@pytest.mark.parametrize(
    "case",
    (
        "malformed_digest",
        "uppercase_digest",
        "message_changed_stale_digest",
        "exception_type_changed",
        "missing_message",
        "unknown_key",
        "wrong_primitive_type",
        "bool_trial_number",
        "non_utc_timestamp",
        "naive_timestamp",
        "malformed_timestamp",
        "reversed_timestamps",
        "execution_to_interruption",
        "interruption_to_execution",
        "interrupted_flag_inconsistent",
        "missing_recovery_lifecycle",
        "completed_with_failure_evidence",
        "failed_with_completed_evidence",
        "rebuilt_summary_after_reason_substitution",
    ),
)
def test_strict_failure_evidence_mutations_reject_production_paths(
    completed_search: tuple[Path, Path],
    case: str,
) -> None:
    valid, operational = completed_search
    corrupted = _copy_report(valid, operational, f"failure_{case}")
    trials_path = corrupted / "trials.csv"
    fields, rows, failed, evidence = _load_failure_row(trials_path)
    complete = next(row for row in rows if row["state"] == "COMPLETE")
    if case == "malformed_digest":
        evidence["failure_message_sha256"] = "g" * 64
    elif case == "uppercase_digest":
        evidence["failure_message_sha256"] = evidence["failure_message_sha256"].upper()
    elif case == "message_changed_stale_digest":
        evidence["failure_message"] += " forged"
    elif case == "exception_type_changed":
        evidence["exception_type"] = "ValueError"
        failed["failure_message"] = f"ValueError: {evidence['failure_message']}"
    elif case == "missing_message":
        evidence["failure_message"] = ""
        evidence["failure_message_sha256"] = hashlib.sha256(b"").hexdigest()
        failed["failure_message"] = "RuntimeError: "
    elif case == "unknown_key":
        evidence["unknown"] = "forged"
    elif case == "wrong_primitive_type":
        evidence["schema_version"] = "1"
    elif case == "bool_trial_number":
        evidence["trial_number"] = True
    elif case == "non_utc_timestamp":
        evidence["started_at_utc"] = "2026-07-28T10:00:00.000000+01:00"
    elif case == "naive_timestamp":
        evidence["started_at_utc"] = "2026-07-28T10:00:00.000000"
    elif case == "malformed_timestamp":
        evidence["started_at_utc"] = "not-a-timestamp"
    elif case == "reversed_timestamps":
        evidence["started_at_utc"], evidence["failed_at_utc"] = (
            evidence["failed_at_utc"],
            evidence["started_at_utc"],
        )
    elif case in {
        "execution_to_interruption",
        "interruption_to_execution",
        "missing_recovery_lifecycle",
        "rebuilt_summary_after_reason_substitution",
    }:
        evidence = build_trial_failure_evidence(
            trial_number=int(failed["trial_number"]),
            error=None,
            started_at_utc=failed["started_at"],
            failed_at_utc=failed["finished_at"],
            interrupted_recovery=True,
        )
        failed["failure_message"] = evidence["failure_message"]
        failed["failure_reason_code"] = (
            "TRIAL_EXECUTION_FAILED"
            if case == "interruption_to_execution"
            else "INTERRUPTED_PROCESS_RECOVERY"
        )
        if case == "rebuilt_summary_after_reason_substitution":
            summary_path = corrupted / "study_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["recovered_interrupted_trials"] = 1
            summary["interrupted_recovery_trial_numbers"] = [
                int(failed["trial_number"])
            ]
            summary["failure_evidence_identity_sha256"] = canonical_sha256(
                {"schema_version": 1, "records": [evidence]}
            )
            summary_path.write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
    elif case == "interrupted_flag_inconsistent":
        evidence["failure_stage"] = "interrupted_running_trial_recovery"
        evidence["failure_class"] = "INTERRUPTED_PROCESS_RECOVERY"
        evidence["exception_type"] = None
        evidence["failure_message"] = (
            "Recovered RUNNING trial from an interrupted prior process."
        )
        evidence["failure_message_sha256"] = hashlib.sha256(
            evidence["failure_message"].encode("utf-8")
        ).hexdigest()
        evidence["interrupted_recovery"] = False
        failed["failure_message"] = evidence["failure_message"]
        failed["failure_reason_code"] = "INTERRUPTED_PROCESS_RECOVERY"
    elif case == "completed_with_failure_evidence":
        forged = build_trial_failure_evidence(
            trial_number=int(complete["trial_number"]),
            error=RuntimeError("forged"),
            started_at_utc="2026-07-28T10:00:00.000000Z",
            failed_at_utc="2026-07-28T10:00:01.000000Z",
        )
        complete["failure_reason_code"] = "TRIAL_EXECUTION_FAILED"
        complete["failure_message"] = "RuntimeError: forged"
        _store_failure_evidence(complete, forged)
    elif case == "failed_with_completed_evidence":
        for name in (
            "adapter_id",
            "candidate_identity_sha256",
            "search_identity_sha256",
            "assignment_identity_sha256",
            "threshold_policy_identity_sha256",
            "prediction_coverage_identity_sha256",
            "prediction_values_identity_sha256",
            "objective",
            "resolved_parameters_json",
            "prediction_key_coverage_json",
        ):
            failed[name] = complete[name]
    else:
        raise AssertionError(case)
    if case not in {
        "completed_with_failure_evidence",
        "failed_with_completed_evidence",
    }:
        _store_failure_evidence(failed, evidence)
    _write_csv(trials_path, fields, rows)
    reauthenticate(corrupted)
    _reject_production_paths(corrupted)
