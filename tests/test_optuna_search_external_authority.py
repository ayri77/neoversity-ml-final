from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import uuid
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any, Iterator

import pytest

from src.churn_ml.optuna_search_artifacts import load_optuna_search_result
from src.churn_ml.optuna_search_authority import (
    AUTHORITY_KEY_FILE_ENV,
    EVENT_DOMAIN,
    EVENT_PAYLOAD_KEYS,
    INITIAL_DOMAIN,
    LEDGER_DOMAIN,
    KEY_IDENTIFIER_DOMAIN,
    LifecycleAuthorityKey,
    LifecycleAuthorityRecorder,
    OptunaLifecycleAuthorityError,
    _event_payload_identity,
    _signed_statement,
    authority_bound_study_identity,
    initialize_lifecycle_authority_key,
    load_lifecycle_authority_key,
)
from src.churn_ml.optuna_search_cli import (
    EXIT_ARTIFACT,
    EXIT_SUCCESS,
    execute,
    parse_args,
)
from src.churn_ml.optuna_search_export import export_best_candidate
from src.churn_ml.optuna_search_lifecycle import (
    FAILURE_RECOVERY_MESSAGE,
    FAILURE_STAGE_INTERRUPTED_RECOVERY,
    FAILURE_STAGE_OBJECTIVE_EXECUTION,
)
from src.churn_ml.optuna_search_semantics import (
    validate_completed_search_semantics,
)
from src.churn_ml.research_data import canonical_sha256
from tests.optuna_auth_support import (
    PROJECT_ROOT,
    reauthenticate,
    run_authorized_completed_search,
    run_authorized_interrupted_search,
)


@pytest.fixture(scope="module")
def signed_failure_report() -> Iterator[tuple[Path, Path]]:
    report, operational = run_authorized_completed_search(
        n_trials=3,
        fail_first=True,
    )
    try:
        yield report, operational
    finally:
        shutil.rmtree(operational, ignore_errors=True)
        shutil.rmtree(report.parent, ignore_errors=True)


@pytest.fixture(scope="module")
def signed_interrupted_report() -> Iterator[tuple[Path, Path]]:
    report, operational = run_authorized_interrupted_search()
    try:
        yield report, operational
    finally:
        shutil.rmtree(operational, ignore_errors=True)
        shutil.rmtree(report.parent, ignore_errors=True)


def _copy_report(report: Path, operational: Path, label: str) -> Path:
    target = operational / f"{label}_{uuid.uuid4().hex}"
    shutil.copytree(report, target)
    return target


def _cli_status(arguments: list[str]) -> int:
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        return execute(parse_args(arguments))


def _reject_inspect_and_export(corrupted: Path, operational: Path) -> None:
    assert _cli_status(["inspect", "--search-dir", str(corrupted)]) == EXIT_ARTIFACT
    output = operational / f"rejected_export_{uuid.uuid4().hex}.yaml"
    assert (
        _cli_status(
            [
                "export-best",
                "--search-dir",
                str(corrupted),
                "--output",
                str(output),
            ]
        )
        == EXIT_ARTIFACT
    )
    assert not output.exists()


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        return list(reader.fieldnames), list(reader)


def _write_rows(
    path: Path,
    fields: list[str],
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _coherently_replace_failure(
    root: Path,
    *,
    message: str,
    exception_type: str | None,
    interrupted_recovery: bool,
) -> None:
    fields, rows = _read_rows(root / "trials.csv")
    failed = next(row for row in rows if row["state"] == "FAIL")
    evidence = json.loads(failed["failure_evidence_json"])
    evidence["failure_message"] = message
    evidence["failure_message_sha256"] = hashlib.sha256(
        message.encode("utf-8")
    ).hexdigest()
    evidence["interrupted_recovery"] = interrupted_recovery
    if interrupted_recovery:
        evidence["failure_stage"] = FAILURE_STAGE_INTERRUPTED_RECOVERY
        evidence["failure_class"] = "INTERRUPTED_PROCESS_RECOVERY"
        evidence["exception_type"] = None
        failed["failure_reason_code"] = "INTERRUPTED_PROCESS_RECOVERY"
        failed["failure_message"] = message
    else:
        assert exception_type is not None
        evidence["failure_stage"] = FAILURE_STAGE_OBJECTIVE_EXECUTION
        evidence["failure_class"] = "TRIAL_EXECUTION_FAILED"
        evidence["exception_type"] = exception_type
        failed["failure_reason_code"] = "TRIAL_EXECUTION_FAILED"
        failed["failure_message"] = f"{exception_type}: {message}"
    failed["failure_evidence_json"] = json.dumps(
        evidence,
        sort_keys=True,
        separators=(",", ":"),
    )
    _write_rows(root / "trials.csv", fields, rows)
    failure_records = [
        json.loads(row["failure_evidence_json"])
        for row in rows
        if row["state"] == "FAIL"
    ]
    identity = canonical_sha256({"schema_version": 1, "records": failure_records})
    summary_path = root / "study_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["failure_evidence_identity_sha256"] = identity
    recovery_numbers = [
        int(row["trial_number"])
        for row in rows
        if row["failure_reason_code"] == "INTERRUPTED_PROCESS_RECOVERY"
    ]
    summary["interrupted_recovery_trial_numbers"] = recovery_numbers
    summary["recovered_interrupted_trials"] = len(recovery_numbers)
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    search_path = root / "search_identity.json"
    search = json.loads(search_path.read_text(encoding="utf-8"))
    search["failure_evidence_identity_sha256"] = identity
    search_path.write_text(
        json.dumps(search, indent=2) + "\n",
        encoding="utf-8",
    )
    reauthenticate(root)
    validate_completed_search_semantics(root, project_root=PROJECT_ROOT)


@pytest.mark.parametrize(
    ("message", "exception_type"),
    (
        ("coherently forged failure message", "RuntimeError"),
        ("intentional first-trial failure", "ValueError"),
    ),
)
def test_coherent_failure_message_and_type_forgeries_require_external_key(
    signed_failure_report: tuple[Path, Path],
    message: str,
    exception_type: str,
) -> None:
    report, operational = signed_failure_report
    corrupted = _copy_report(report, operational, "failure_forgery")
    _coherently_replace_failure(
        corrupted,
        message=message,
        exception_type=exception_type,
        interrupted_recovery=False,
    )
    _reject_inspect_and_export(corrupted, operational)


def test_coherent_execution_failure_to_recovery_substitution_is_rejected(
    signed_failure_report: tuple[Path, Path],
) -> None:
    report, operational = signed_failure_report
    corrupted = _copy_report(report, operational, "execution_to_recovery")
    _coherently_replace_failure(
        corrupted,
        message=FAILURE_RECOVERY_MESSAGE,
        exception_type=None,
        interrupted_recovery=True,
    )
    _reject_inspect_and_export(corrupted, operational)


def test_coherent_recovery_to_execution_failure_substitution_is_rejected(
    signed_interrupted_report: tuple[Path, Path],
) -> None:
    report, operational = signed_interrupted_report
    corrupted = _copy_report(report, operational, "recovery_to_execution")
    _coherently_replace_failure(
        corrupted,
        message="coherently substituted execution failure",
        exception_type="RuntimeError",
        interrupted_recovery=False,
    )
    _reject_inspect_and_export(corrupted, operational)


def test_coherently_changed_failure_timestamps_are_rejected(
    signed_failure_report: tuple[Path, Path],
) -> None:
    report, operational = signed_failure_report
    corrupted = _copy_report(report, operational, "coherent_timestamps")
    fields, rows = _read_rows(corrupted / "trials.csv")
    failed = next(row for row in rows if row["state"] == "FAIL")
    evidence = json.loads(failed["failure_evidence_json"])
    evidence["started_at_utc"] = "2026-01-01T00:00:00.000000Z"
    evidence["failed_at_utc"] = "2026-01-01T00:00:01.000000Z"
    failed["started_at"] = evidence["started_at_utc"]
    failed["finished_at"] = evidence["failed_at_utc"]
    failed["duration_seconds"] = "1"
    failed["failure_evidence_json"] = json.dumps(
        evidence,
        sort_keys=True,
        separators=(",", ":"),
    )
    _write_rows(corrupted / "trials.csv", fields, rows)
    identity = canonical_sha256(
        {
            "schema_version": 1,
            "records": [
                json.loads(row["failure_evidence_json"])
                for row in rows
                if row["state"] == "FAIL"
            ],
        }
    )
    for name in ("study_summary.json", "search_identity.json"):
        path = corrupted / name
        value = json.loads(path.read_text(encoding="utf-8"))
        value["failure_evidence_identity_sha256"] = identity
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    reauthenticate(corrupted)
    validate_completed_search_semantics(corrupted, project_root=PROJECT_ROOT)
    _reject_inspect_and_export(corrupted, operational)


@pytest.mark.parametrize(
    "case",
    (
        "delete_failed_event",
        "insert_event",
        "reorder_events",
        "duplicate_event",
        "event_sequence",
        "previous_signature",
        "trial_state",
        "timestamps",
    ),
)
def test_event_chain_corruption_rejects_production_paths(
    signed_failure_report: tuple[Path, Path],
    case: str,
) -> None:
    report, operational = signed_failure_report
    corrupted = _copy_report(report, operational, f"event_{case}")
    path = corrupted / "lifecycle_authority.json"
    authority = json.loads(path.read_text(encoding="utf-8"))
    events = authority["events"]
    failed_index = next(
        index
        for index, event in enumerate(events)
        if event["payload"]["event_type"] == "trial_execution_failed"
    )
    if case == "delete_failed_event":
        del events[failed_index]
    elif case == "insert_event":
        events.insert(failed_index, deepcopy(events[failed_index]))
    elif case == "reorder_events":
        events[1], events[2] = events[2], events[1]
    elif case == "duplicate_event":
        events.append(deepcopy(events[-1]))
    elif case == "event_sequence":
        events[failed_index]["payload"]["event_sequence"] += 1
    elif case == "previous_signature":
        events[failed_index]["payload"]["previous_event_signature_sha256"] = "0" * 64
    elif case == "trial_state":
        events[failed_index]["payload"]["to_state"] = "COMPLETE"
    else:
        events[failed_index]["payload"]["event_at_utc"] = events[failed_index - 1][
            "payload"
        ]["event_at_utc"]
        events[failed_index]["payload"]["started_at_utc"] = events[failed_index - 1][
            "payload"
        ]["event_at_utc"]
    path.write_text(json.dumps(authority, indent=2) + "\n", encoding="utf-8")
    reauthenticate(corrupted)
    validate_completed_search_semantics(corrupted, project_root=PROJECT_ROOT)
    _reject_inspect_and_export(corrupted, operational)


def test_delete_failure_and_claim_all_completed_is_rejected() -> None:
    report, operational = run_authorized_completed_search(
        n_trials=2,
        fail_first=True,
    )
    try:
        authority_path = report / "lifecycle_authority.json"
        authority = json.loads(authority_path.read_text(encoding="utf-8"))
        authority["events"] = [
            event
            for event in authority["events"]
            if event["payload"]["event_type"] != "trial_execution_failed"
        ]
        authority_path.write_text(
            json.dumps(authority, indent=2) + "\n",
            encoding="utf-8",
        )
        fields, rows = _read_rows(report / "trials.csv")
        failed = next(row for row in rows if row["state"] == "FAIL")
        failed["state"] = "COMPLETE"
        failed["failure_reason_code"] = ""
        failed["failure_message"] = ""
        failed["failure_evidence_json"] = "null"
        _write_rows(report / "trials.csv", fields, rows)
        empty_failure_identity = canonical_sha256({"schema_version": 1, "records": []})
        summary_path = report / "study_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["state_counts"]["complete"] = len(rows)
        summary["state_counts"]["fail"] = 0
        summary["failure_evidence_identity_sha256"] = empty_failure_identity
        summary["interrupted_recovery_trial_numbers"] = []
        summary["recovered_interrupted_trials"] = 0
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        search_path = report / "search_identity.json"
        search = json.loads(search_path.read_text(encoding="utf-8"))
        search["failure_evidence_identity_sha256"] = empty_failure_identity
        search_path.write_text(
            json.dumps(search, indent=2) + "\n",
            encoding="utf-8",
        )
        ledger_path = report / "lifecycle_authority_ledger.json"
        ledger_artifact = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger = ledger_artifact["final_ledger"]["payload"]
        ledger["trial_state_universe"] = [
            {"trial_number": int(row["trial_number"]), "state": "COMPLETE"}
            for row in rows
        ]
        ledger["completed_trial_numbers"] = [int(row["trial_number"]) for row in rows]
        ledger["failed_trial_numbers"] = []
        ledger["interrupted_trial_numbers"] = []
        ledger["interrupted_recovery_trial_numbers"] = []
        ledger_path.write_text(
            json.dumps(ledger_artifact, indent=2) + "\n",
            encoding="utf-8",
        )
        combined_events = [
            *authority["events"],
            ledger_artifact["report_finalized_event"],
        ]
        with sqlite3.connect(operational / "study.db") as connection:
            connection.execute("UPDATE trials SET state = 'COMPLETE' WHERE number = 0")
            connection.execute(
                "UPDATE study_user_attributes SET value_json = ? WHERE key = ?",
                (json.dumps(combined_events), "lifecycle_authority_events"),
            )
            connection.execute(
                "UPDATE study_user_attributes SET value_json = ? WHERE key = ?",
                (json.dumps(ledger_artifact), "lifecycle_authority_final_ledger"),
            )
            connection.commit()
        reauthenticate(report)
        _reject_inspect_and_export(report, operational)
    finally:
        shutil.rmtree(operational, ignore_errors=True)
        shutil.rmtree(report.parent, ignore_errors=True)


@pytest.mark.parametrize(
    "case",
    ("replace_final_ledger", "replace_fingerprint"),
)
def test_final_statement_or_fingerprint_replacement_is_rejected(
    signed_failure_report: tuple[Path, Path],
    case: str,
) -> None:
    report, operational = signed_failure_report
    corrupted = _copy_report(report, operational, case)
    if case == "replace_final_ledger":
        path = corrupted / "lifecycle_authority_ledger.json"
        ledger = json.loads(path.read_text(encoding="utf-8"))
        ledger["final_ledger"]["payload"]["best_trial_number"] = None
        path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    else:
        path = corrupted / "lifecycle_authority.json"
        authority = json.loads(path.read_text(encoding="utf-8"))
        authority["initial_statement"]["payload"]["authority_key_fingerprint"] = (
            "1" * 64
        )
        path.write_text(json.dumps(authority, indent=2) + "\n", encoding="utf-8")
        for name in ("search_identity.json", "study_summary.json"):
            identity_path = corrupted / name
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            identity["authority_key_fingerprint"] = "1" * 64
            identity_path.write_text(
                json.dumps(identity, indent=2) + "\n",
                encoding="utf-8",
            )
    reauthenticate(corrupted)
    validate_completed_search_semantics(corrupted, project_root=PROJECT_ROOT)
    _reject_inspect_and_export(corrupted, operational)


def _attacker_resign(root: Path, attacker_key: LifecycleAuthorityKey) -> None:
    report_path = root / "lifecycle_authority.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    initial_payload = report["initial_statement"]["payload"]
    initial_payload["authority_key_fingerprint"] = attacker_key.fingerprint
    initial_payload["authority_bound_study_identity_sha256"] = (
        authority_bound_study_identity(
            base_search_identity_sha256=initial_payload["base_search_identity_sha256"],
            authority_key_fingerprint=attacker_key.fingerprint,
        )
    )
    initial = _signed_statement(
        initial_payload,
        key=attacker_key,
        domain=INITIAL_DOMAIN,
    )
    ledger_path = root / "lifecycle_authority_ledger.json"
    ledger_artifact = json.loads(ledger_path.read_text(encoding="utf-8"))
    combined = [*report["events"], ledger_artifact["report_finalized_event"]]
    previous = initial["signature_sha256"]
    resigned_events: list[dict[str, Any]] = []
    for event in combined:
        payload = event["payload"]
        payload["previous_event_signature_sha256"] = hashlib.sha256(
            bytes.fromhex(previous)
        ).hexdigest()
        event_body = {name: payload[name] for name in EVENT_PAYLOAD_KEYS}
        payload["event_payload_identity_sha256"] = _event_payload_identity(event_body)
        resigned = _signed_statement(
            payload,
            key=attacker_key,
            domain=EVENT_DOMAIN,
        )
        resigned_events.append(resigned)
        previous = resigned["signature_sha256"]
    report["initial_statement"] = initial
    report["events"] = resigned_events[:-1]
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    ledger_artifact["report_finalized_event"] = resigned_events[-1]
    ledger_payload = ledger_artifact["final_ledger"]["payload"]
    ledger_payload["authority_key_fingerprint"] = attacker_key.fingerprint
    ledger_payload["authority_bound_study_identity_sha256"] = initial_payload[
        "authority_bound_study_identity_sha256"
    ]
    ledger_payload["ordered_event_signatures"] = [
        event["signature_sha256"] for event in resigned_events
    ]
    ledger_artifact["final_ledger"] = _signed_statement(
        ledger_payload,
        key=attacker_key,
        domain=LEDGER_DOMAIN,
    )
    ledger_path.write_text(
        json.dumps(ledger_artifact, indent=2) + "\n",
        encoding="utf-8",
    )
    for name in ("search_identity.json", "study_summary.json"):
        path = root / name
        value = json.loads(path.read_text(encoding="utf-8"))
        value["authority_key_fingerprint"] = attacker_key.fingerprint
        value["authority_bound_study_identity_sha256"] = initial_payload[
            "authority_bound_study_identity_sha256"
        ]
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    reauthenticate(root)


def test_complete_resigning_with_attacker_key_is_rejected(
    signed_failure_report: tuple[Path, Path],
) -> None:
    report, operational = signed_failure_report
    corrupted = _copy_report(report, operational, "attacker_resigned")
    material = os.urandom(32)
    attacker_key = LifecycleAuthorityKey(
        material=material,
        fingerprint=hashlib.sha256(KEY_IDENTIFIER_DOMAIN + material).hexdigest(),
    )
    _attacker_resign(corrupted, attacker_key)
    validate_completed_search_semantics(corrupted, project_root=PROJECT_ROOT)
    _reject_inspect_and_export(corrupted, operational)


def test_missing_and_wrong_external_keys_fail_closed(
    signed_failure_report: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report, operational = signed_failure_report
    monkeypatch.delenv(AUTHORITY_KEY_FILE_ENV, raising=False)
    _reject_inspect_and_export(report, operational)
    wrong_key = tmp_path / "wrong.key"
    wrong_key.write_bytes(os.urandom(32))
    monkeypatch.setenv(AUTHORITY_KEY_FILE_ENV, str(wrong_key))
    _reject_inspect_and_export(report, operational)


@pytest.mark.parametrize(
    "case",
    ("missing_initial_statement", "modified_events", "trial_state_divergence"),
)
def test_sqlite_authority_missing_or_divergent_is_rejected(
    case: str,
) -> None:
    report, operational = run_authorized_completed_search(
        n_trials=2,
        fail_first=True,
    )
    try:
        database = operational / "study.db"
        with sqlite3.connect(database) as connection:
            if case == "missing_initial_statement":
                connection.execute(
                    "DELETE FROM study_user_attributes WHERE key = ?",
                    ("lifecycle_authority_initial_statement",),
                )
            elif case == "modified_events":
                row = connection.execute(
                    "SELECT study_user_attribute_id, value_json "
                    "FROM study_user_attributes WHERE key = ?",
                    ("lifecycle_authority_events",),
                ).fetchone()
                assert row is not None
                events = json.loads(row[1])
                events[0]["signature_sha256"] = "0" * 64
                connection.execute(
                    "UPDATE study_user_attributes SET value_json = ? "
                    "WHERE study_user_attribute_id = ?",
                    (json.dumps(events), row[0]),
                )
            else:
                connection.execute("UPDATE trials SET state = 'FAIL' WHERE number = 1")
            connection.commit()
        _reject_inspect_and_export(report, operational)
    finally:
        shutil.rmtree(operational, ignore_errors=True)
        shutil.rmtree(report.parent, ignore_errors=True)


def test_valid_failure_and_interrupted_signed_reports_use_production_paths(
    signed_failure_report: tuple[Path, Path],
    signed_interrupted_report: tuple[Path, Path],
) -> None:
    for report, operational in (
        signed_failure_report,
        signed_interrupted_report,
    ):
        database = operational / "study.db"
        database_before = hashlib.sha256(database.read_bytes()).hexdigest()
        loaded = load_optuna_search_result(report, project_root=PROJECT_ROOT)
        assert loaded.study_summary["authority_schema_version"] == 1
        assert _cli_status(["inspect", "--search-dir", str(report)]) == EXIT_SUCCESS
        assert hashlib.sha256(database.read_bytes()).hexdigest() == database_before
        output = operational / f"valid_export_{uuid.uuid4().hex}.yaml"
        assert (
            _cli_status(
                [
                    "export-best",
                    "--search-dir",
                    str(report),
                    "--output",
                    str(output),
                ]
            )
            == EXIT_SUCCESS
        )
        assert output.is_file()


def test_signed_report_remains_independently_valid_without_sqlite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, operational = run_authorized_completed_search(
        n_trials=2,
        fail_first=True,
    )
    try:
        database = (operational / "study.db").resolve()
        original_exists = Path.exists

        def exists_without_database(path: Path) -> bool:
            return False if path.resolve() == database else original_exists(path)

        monkeypatch.setattr(Path, "exists", exists_without_database)
        loaded = load_optuna_search_result(report, project_root=PROJECT_ROOT)
        assert loaded.study_summary["state_counts"]["fail"] == 1
        assert _cli_status(["inspect", "--search-dir", str(report)]) == EXIT_SUCCESS
    finally:
        shutil.rmtree(operational, ignore_errors=True)
        shutil.rmtree(report.parent, ignore_errors=True)


def test_all_success_study_is_signed_from_creation() -> None:
    report, operational = run_authorized_completed_search(
        n_trials=2,
        fail_first=False,
    )
    try:
        result = load_optuna_search_result(report, project_root=PROJECT_ROOT)
        assert result.study_summary["state_counts"]["fail"] == 0
        authority = json.loads(
            (report / "lifecycle_authority.json").read_text(encoding="utf-8")
        )
        assert authority["events"][0]["payload"]["event_type"] == "study_created"
        assert _cli_status(["inspect", "--search-dir", str(report)]) == EXIT_SUCCESS
        candidate = export_best_candidate(
            report,
            operational / "production_candidate.yaml",
            project_root=PROJECT_ROOT,
        )
        validation = subprocess.run(
            [
                sys.executable,
                "scripts/run_research_v2.py",
                "--config",
                str(candidate),
                "--validate-only",
            ],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert validation.returncode == 0, validation.stderr
        assert "Research v2 validation successful." in validation.stdout
    finally:
        shutil.rmtree(operational, ignore_errors=True)
        shutil.rmtree(report.parent, ignore_errors=True)


def test_authority_init_is_external_binary_non_overwriting_and_path_private(
    tmp_path: Path,
) -> None:
    target = tmp_path / "authority.key"
    fingerprint = initialize_lifecycle_authority_key(
        target,
        project_root=PROJECT_ROOT,
        create_parent=False,
    )
    assert len(target.read_bytes()) == 32
    assert len(fingerprint) == 64
    with pytest.raises(Exception, match="refuses overwrite"):
        initialize_lifecycle_authority_key(
            target,
            project_root=PROJECT_ROOT,
            create_parent=False,
        )
    output = StringIO()
    with redirect_stdout(output), redirect_stderr(StringIO()):
        status = execute(
            parse_args(
                [
                    "authority-init",
                    "--output",
                    str(tmp_path / "second.key"),
                ]
            )
        )
    assert status == EXIT_SUCCESS
    assert str(tmp_path) not in output.getvalue()
    assert "authority_key_fingerprint" in output.getvalue()


@pytest.mark.parametrize("case", ("relative", "directory", "short", "traversal"))
def test_authority_key_provider_rejects_unsafe_inputs(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if case == "relative":
        raw = "relative-authority.key"
    elif case == "directory":
        raw = str(tmp_path)
    elif case == "short":
        key_path = tmp_path / "short.key"
        key_path.write_bytes(b"too-short")
        raw = str(key_path)
    else:
        key_path = tmp_path / "authority.key"
        key_path.write_bytes(os.urandom(32))
        raw = str(tmp_path / "missing" / ".." / "authority.key")
    monkeypatch.setenv(AUTHORITY_KEY_FILE_ENV, raw)
    with pytest.raises(OptunaLifecycleAuthorityError) as caught:
        load_lifecycle_authority_key(project_root=PROJECT_ROOT)
    assert raw not in str(caught.value)


def test_authority_init_parent_creation_is_explicit(tmp_path: Path) -> None:
    target = tmp_path / "missing" / "authority.key"
    with pytest.raises(OptunaLifecycleAuthorityError, match="parent is missing"):
        initialize_lifecycle_authority_key(
            target,
            project_root=PROJECT_ROOT,
            create_parent=False,
        )
    fingerprint = initialize_lifecycle_authority_key(
        target,
        project_root=PROJECT_ROOT,
        create_parent=True,
    )
    assert target.is_file()
    assert len(fingerprint) == 64


def test_authority_key_provider_rejects_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.key"
    target.write_bytes(os.urandom(32))
    linked = tmp_path / "linked.key"
    try:
        linked.symlink_to(target)
    except OSError:
        pytest.skip("Local Windows policy does not allow test symlink creation.")
    monkeypatch.setenv(AUTHORITY_KEY_FILE_ENV, str(linked))
    with pytest.raises(OptunaLifecycleAuthorityError, match="link or reparse"):
        load_lifecycle_authority_key(project_root=PROJECT_ROOT)


class _EmptyUnsignedStudy:
    def __init__(self) -> None:
        self.user_attrs: dict[str, Any] = {}

    def get_trials(self, *, deepcopy: bool = False) -> list[Any]:
        del deepcopy
        return []

    def set_user_attr(self, name: str, value: Any) -> None:
        self.user_attrs[name] = value


def test_preexisting_empty_unsigned_study_fails_closed() -> None:
    material = b"u" * 32
    key = LifecycleAuthorityKey(
        material=material,
        fingerprint=hashlib.sha256(KEY_IDENTIFIER_DOMAIN + material).hexdigest(),
    )
    with pytest.raises(OptunaLifecycleAuthorityError, match="lacks external"):
        LifecycleAuthorityRecorder.initialize_or_load(
            study=_EmptyUnsignedStudy(),
            key=key,
            base_search_identity_sha256="0" * 64,
            dataset_identity_sha256="1" * 64,
            assignment_identity_sha256="2" * 64,
            source_closure_identity_sha256="3" * 64,
            runtime_identity_sha256="4" * 64,
            sampler_pruner_identity_sha256="5" * 64,
            configured_trial_count=1,
            allow_initialize=False,
        )
