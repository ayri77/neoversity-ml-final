from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import stat
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


AUTHORITY_KEY_FILE_ENV = "CHURN_ML_OPTUNA_LIFECYCLE_AUTHORITY_KEY_FILE"
AUTHORITY_SCHEMA_VERSION = 1
EVENT_SCHEMA_VERSION = 1
LEDGER_SCHEMA_VERSION = 1
INITIAL_DOMAIN = b"churn_ml.optuna.lifecycle.initial.v1\x00"
EVENT_DOMAIN = b"churn_ml.optuna.lifecycle.event.v1\x00"
EVENT_PAYLOAD_DOMAIN = b"churn_ml.optuna.lifecycle.event-payload.v1\x00"
LEDGER_DOMAIN = b"churn_ml.optuna.lifecycle.ledger.v1\x00"
KEY_IDENTIFIER_DOMAIN = b"churn_ml.optuna.lifecycle.key-identifier.v1\x00"
BOUND_IDENTITY_DOMAIN = b"churn_ml.optuna.lifecycle.bound-identity.v1\x00"
EVIDENCE_DOMAIN = b"churn_ml.optuna.lifecycle.metric-prediction.v1\x00"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
UTC_PATTERN = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z$"
)
INITIAL_PAYLOAD_KEYS = {
    "authority_schema_version",
    "authority_key_fingerprint",
    "study_uuid",
    "base_search_identity_sha256",
    "authority_bound_study_identity_sha256",
    "dataset_identity_sha256",
    "assignment_identity_sha256",
    "source_closure_identity_sha256",
    "runtime_identity_sha256",
    "sampler_pruner_identity_sha256",
    "configured_trial_count",
    "created_at_utc",
}
EVENT_PAYLOAD_KEYS = {
    "authority_schema_version",
    "event_schema_version",
    "event_type",
    "study_uuid",
    "base_search_identity_sha256",
    "event_sequence",
    "previous_event_signature_sha256",
    "trial_number",
    "from_state",
    "to_state",
    "failure_stage",
    "failure_reason_code",
    "exception_type",
    "failure_message_sha256",
    "started_at_utc",
    "event_at_utc",
    "interrupted_recovery",
}
LEDGER_PAYLOAD_KEYS = {
    "authority_schema_version",
    "ledger_schema_version",
    "authority_key_fingerprint",
    "study_uuid",
    "base_search_identity_sha256",
    "authority_bound_study_identity_sha256",
    "ordered_event_signatures",
    "event_count",
    "trial_state_universe",
    "completed_trial_numbers",
    "failed_trial_numbers",
    "interrupted_trial_numbers",
    "interrupted_recovery_trial_numbers",
    "best_trial_number",
    "report_search_identity_sha256",
    "study_summary_identity_sha256",
    "metric_prediction_evidence_identity_sha256",
    "report_manifest_identity_sha256",
    "completed_at_utc",
}
EVENT_TRANSITIONS = {
    "study_created": (None, "STUDY_CREATED"),
    "trial_allocated": (None, "ALLOCATED"),
    "trial_started": ("ALLOCATED", "RUNNING"),
    "trial_completed": ("RUNNING", "COMPLETE"),
    "trial_execution_failed": ("RUNNING", "FAIL"),
    "interrupted_running_trial_recovered": ("RUNNING", "FAIL"),
    "study_completed": ("STUDY_CREATED", "STUDY_COMPLETED"),
    "report_finalized": ("STUDY_COMPLETED", "REPORT_FINALIZED"),
}


class OptunaLifecycleAuthorityError(RuntimeError):
    """Raised when external lifecycle authority is absent or inconsistent."""


@dataclass(frozen=True)
class LifecycleAuthorityKey:
    material: bytes
    fingerprint: str


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    _validate_json_primitives(value, "authority payload")
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def canonical_authority_timestamp(value: datetime | str) -> str:
    if isinstance(value, datetime):
        parsed = value
    elif type(value) is str and UTC_PATTERN.fullmatch(value):
        try:
            parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError as error:
            raise OptunaLifecycleAuthorityError(
                "Lifecycle authority timestamp is malformed."
            ) from error
    else:
        raise OptunaLifecycleAuthorityError(
            "Lifecycle authority timestamp must be canonical UTC."
        )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise OptunaLifecycleAuthorityError(
            "Lifecycle authority timestamp must be timezone-aware."
        )
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def load_lifecycle_authority_key(
    *,
    project_root: Path,
    forbidden_roots: Sequence[Path] = (),
) -> LifecycleAuthorityKey:
    raw = os.environ.get(AUTHORITY_KEY_FILE_ENV)
    if raw is None or not raw:
        raise OptunaLifecycleAuthorityError(
            "External Optuna lifecycle authority key is required."
        )
    candidate = Path(raw)
    if not candidate.is_absolute() or any(
        part in {".", ".."} for part in candidate.parts
    ):
        raise OptunaLifecycleAuthorityError(
            "External Optuna lifecycle authority key path is unsafe."
        )
    lexical = Path(os.path.abspath(candidate))
    roots = [project_root.resolve(), *(root.resolve() for root in forbidden_roots)]
    if any(lexical == root or root in lexical.parents for root in roots):
        raise OptunaLifecycleAuthorityError(
            "External Optuna lifecycle authority key must be out of band."
        )
    _reject_linked_components(lexical)
    try:
        before = lexical.lstat()
    except OSError as error:
        raise OptunaLifecycleAuthorityError(
            "External Optuna lifecycle authority key is unavailable."
        ) from error
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or _is_reparse_stat(before)
        or getattr(before, "st_nlink", 1) != 1
    ):
        raise OptunaLifecycleAuthorityError(
            "External Optuna lifecycle authority key must be an exact regular file."
        )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lexical, flags)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
                before.st_dev,
                before.st_ino,
            ):
                raise OptunaLifecycleAuthorityError(
                    "External Optuna lifecycle authority key changed while loading."
                )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 64 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            os.close(descriptor)
    except OptunaLifecycleAuthorityError:
        raise
    except OSError as error:
        raise OptunaLifecycleAuthorityError(
            "External Optuna lifecycle authority key could not be read."
        ) from error
    material = b"".join(chunks)
    if len(material) < 32:
        raise OptunaLifecycleAuthorityError(
            "External Optuna lifecycle authority key requires at least 32 bytes."
        )
    fingerprint = hashlib.sha256(KEY_IDENTIFIER_DOMAIN + material).hexdigest()
    return LifecycleAuthorityKey(material=material, fingerprint=fingerprint)


def initialize_lifecycle_authority_key(
    output: Path,
    *,
    project_root: Path,
    create_parent: bool,
) -> str:
    if not output.is_absolute() or any(part in {".", ".."} for part in output.parts):
        raise OptunaLifecycleAuthorityError(
            "Authority initialization output path is unsafe."
        )
    target = Path(os.path.abspath(output))
    root = project_root.resolve()
    if target == root or root in target.parents:
        raise OptunaLifecycleAuthorityError(
            "Authority initialization output must be outside the repository."
        )
    parent = target.parent
    if not parent.exists():
        if not create_parent:
            raise OptunaLifecycleAuthorityError(
                "Authority initialization parent is missing."
            )
        _create_real_parent_chain(parent)
    _reject_linked_components(parent)
    if target.exists() or target.is_symlink():
        raise OptunaLifecycleAuthorityError(
            "Authority initialization refuses overwrite."
        )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    material = secrets.token_bytes(32)
    try:
        descriptor = os.open(target, flags, 0o600)
        try:
            written = 0
            while written < len(material):
                written += os.write(descriptor, material[written:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.chmod(target, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        loaded = load_lifecycle_authority_key_from_path(
            target,
            project_root=root,
        )
    except BaseException:
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return loaded.fingerprint


def load_lifecycle_authority_key_from_path(
    path: Path,
    *,
    project_root: Path,
) -> LifecycleAuthorityKey:
    previous = os.environ.get(AUTHORITY_KEY_FILE_ENV)
    os.environ[AUTHORITY_KEY_FILE_ENV] = os.fspath(path)
    try:
        return load_lifecycle_authority_key(project_root=project_root)
    finally:
        if previous is None:
            os.environ.pop(AUTHORITY_KEY_FILE_ENV, None)
        else:
            os.environ[AUTHORITY_KEY_FILE_ENV] = previous


class LifecycleAuthorityRecorder:
    def __init__(
        self,
        *,
        study: Any,
        key: LifecycleAuthorityKey,
        initial_statement: Mapping[str, Any],
        events: Sequence[Mapping[str, Any]],
    ) -> None:
        self.study = study
        self.key = key
        self.initial_statement = deepcopy(dict(initial_statement))
        self.events = [deepcopy(dict(event)) for event in events]

    @classmethod
    def initialize_or_load(
        cls,
        *,
        study: Any,
        key: LifecycleAuthorityKey,
        base_search_identity_sha256: str,
        dataset_identity_sha256: str,
        assignment_identity_sha256: str,
        source_closure_identity_sha256: str,
        runtime_identity_sha256: str,
        sampler_pruner_identity_sha256: str,
        configured_trial_count: int,
        allow_initialize: bool,
    ) -> LifecycleAuthorityRecorder:
        existing = study.user_attrs.get("lifecycle_authority_initial_statement")
        events = study.user_attrs.get("lifecycle_authority_events")
        if existing is None:
            if not allow_initialize or study.get_trials(deepcopy=False):
                raise OptunaLifecycleAuthorityError(
                    "Existing study lacks external lifecycle authority."
                )
            created_at = canonical_authority_timestamp(datetime.now(timezone.utc))
            study_uuid = str(uuid.uuid4())
            bound = authority_bound_study_identity(
                base_search_identity_sha256=base_search_identity_sha256,
                authority_key_fingerprint=key.fingerprint,
            )
            payload = {
                "authority_schema_version": AUTHORITY_SCHEMA_VERSION,
                "authority_key_fingerprint": key.fingerprint,
                "study_uuid": study_uuid,
                "base_search_identity_sha256": base_search_identity_sha256,
                "authority_bound_study_identity_sha256": bound,
                "dataset_identity_sha256": dataset_identity_sha256,
                "assignment_identity_sha256": assignment_identity_sha256,
                "source_closure_identity_sha256": source_closure_identity_sha256,
                "runtime_identity_sha256": runtime_identity_sha256,
                "sampler_pruner_identity_sha256": sampler_pruner_identity_sha256,
                "configured_trial_count": configured_trial_count,
                "created_at_utc": created_at,
            }
            statement = _signed_statement(payload, key=key, domain=INITIAL_DOMAIN)
            study.set_user_attr("lifecycle_authority_initial_statement", statement)
            study.set_user_attr("lifecycle_authority_events", [])
            recorder = cls(
                study=study,
                key=key,
                initial_statement=statement,
                events=[],
            )
            recorder.append_event(
                event_type="study_created",
                trial_number=None,
                from_state=None,
                to_state="STUDY_CREATED",
                event_at_utc=created_at,
            )
            return recorder
        if not isinstance(events, list):
            raise OptunaLifecycleAuthorityError(
                "Study lifecycle authority events are malformed."
            )
        recorder = cls(
            study=study,
            key=key,
            initial_statement=existing,
            events=events,
        )
        recorder._validate_persisted_final_ledger()
        recorder.validate(
            expected={
                "base_search_identity_sha256": base_search_identity_sha256,
                "dataset_identity_sha256": dataset_identity_sha256,
                "assignment_identity_sha256": assignment_identity_sha256,
                "source_closure_identity_sha256": source_closure_identity_sha256,
                "runtime_identity_sha256": runtime_identity_sha256,
                "sampler_pruner_identity_sha256": sampler_pruner_identity_sha256,
            }
        )
        return recorder

    def _validate_persisted_final_ledger(self) -> None:
        persisted = self.study.user_attrs.get("lifecycle_authority_final_ledger")
        finalized_events = [
            event
            for event in self.events
            if event["payload"]["event_type"] == "report_finalized"
        ]
        if persisted is None:
            if finalized_events:
                raise OptunaLifecycleAuthorityError(
                    "Study final lifecycle ledger is missing."
                )
            return
        if (
            not isinstance(persisted, Mapping)
            or set(persisted)
            != {
                "authority_schema_version",
                "report_finalized_event",
                "final_ledger",
            }
            or not finalized_events
            or persisted["report_finalized_event"] != finalized_events[-1]
        ):
            raise OptunaLifecycleAuthorityError(
                "Study final lifecycle ledger schema differs."
            )
        payload = _validate_signed_statement(
            persisted["final_ledger"],
            key=self.key,
            domain=LEDGER_DOMAIN,
            payload_keys=LEDGER_PAYLOAD_KEYS,
            label="persisted final lifecycle ledger",
        )
        if (
            payload["authority_key_fingerprint"] != self.key.fingerprint
            or payload["study_uuid"] != self.initial_statement["payload"]["study_uuid"]
            or payload["ordered_event_signatures"]
            != [event["signature_sha256"] for event in self.events]
            or payload["event_count"] != len(self.events)
        ):
            raise OptunaLifecycleAuthorityError(
                "Study final lifecycle ledger differs from signed events."
            )

    def validate(self, *, expected: Mapping[str, Any] | None = None) -> None:
        validate_initial_statement(self.initial_statement, key=self.key)
        payload = self.initial_statement["payload"]
        if expected is not None and any(
            payload.get(name) != value for name, value in expected.items()
        ):
            raise OptunaLifecycleAuthorityError(
                "Study lifecycle authority identity differs from the request."
            )
        validate_event_chain(
            self.events,
            initial_statement=self.initial_statement,
            key=self.key,
            require_report_finalized=False,
        )
        self.validate_study_trial_states()

    def validate_study_trial_states(self) -> None:
        signed_states = _signed_trial_states(self.events)
        actual_states = {
            int(trial.number): str(trial.state.name)
            for trial in self.study.get_trials(deepcopy=False)
        }
        if signed_states != actual_states:
            raise OptunaLifecycleAuthorityError(
                "Signed and SQLite trial-state universes differ."
            )

    def append_event(
        self,
        *,
        event_type: str,
        trial_number: int | None,
        from_state: str | None,
        to_state: str,
        failure_stage: str | None = None,
        failure_reason_code: str | None = None,
        exception_type: str | None = None,
        failure_message_sha256: str | None = None,
        started_at_utc: str | None = None,
        event_at_utc: str | None = None,
        interrupted_recovery: bool = False,
        persist: bool = True,
    ) -> dict[str, Any]:
        sequence = len(self.events) + 1
        previous_signature = (
            self.initial_statement["signature_sha256"]
            if not self.events
            else self.events[-1]["signature_sha256"]
        )
        payload = {
            "authority_schema_version": AUTHORITY_SCHEMA_VERSION,
            "event_schema_version": EVENT_SCHEMA_VERSION,
            "event_type": event_type,
            "study_uuid": self.initial_statement["payload"]["study_uuid"],
            "base_search_identity_sha256": self.initial_statement["payload"][
                "base_search_identity_sha256"
            ],
            "event_sequence": sequence,
            "previous_event_signature_sha256": hashlib.sha256(
                bytes.fromhex(previous_signature)
            ).hexdigest(),
            "trial_number": trial_number,
            "from_state": from_state,
            "to_state": to_state,
            "failure_stage": failure_stage,
            "failure_reason_code": failure_reason_code,
            "exception_type": exception_type,
            "failure_message_sha256": failure_message_sha256,
            "started_at_utc": started_at_utc,
            "event_at_utc": event_at_utc
            or canonical_authority_timestamp(datetime.now(timezone.utc)),
            "interrupted_recovery": interrupted_recovery,
        }
        payload_identity = _event_payload_identity(payload)
        event = _signed_statement(
            {
                **payload,
                "event_payload_identity_sha256": payload_identity,
            },
            key=self.key,
            domain=EVENT_DOMAIN,
        )
        candidate = [*self.events, event]
        validate_event_chain(
            candidate,
            initial_statement=self.initial_statement,
            key=self.key,
            require_report_finalized=False,
        )
        self.events.append(event)
        if persist:
            self.study.set_user_attr(
                "lifecycle_authority_events",
                deepcopy(self.events),
            )
        return deepcopy(event)

    def report_payload(self) -> dict[str, Any]:
        return {
            "authority_schema_version": AUTHORITY_SCHEMA_VERSION,
            "initial_statement": deepcopy(self.initial_statement),
            "events": deepcopy(self.events),
        }

    def finalize(
        self,
        *,
        trial_state_universe: Sequence[Mapping[str, Any]],
        interrupted_recovery_trial_numbers: Sequence[int],
        best_trial_number: int | None,
        report_search_identity_sha256: str,
        study_summary_identity_sha256: str,
        metric_prediction_evidence_identity_sha256: str,
        report_manifest_identity_sha256: str,
    ) -> dict[str, Any]:
        if self.events[-1]["payload"]["event_type"] != "study_completed":
            raise OptunaLifecycleAuthorityError(
                "Lifecycle authority cannot finalize before study completion."
            )
        completed_at = canonical_authority_timestamp(datetime.now(timezone.utc))
        final_event = self.append_event(
            event_type="report_finalized",
            trial_number=None,
            from_state="STUDY_COMPLETED",
            to_state="REPORT_FINALIZED",
            event_at_utc=completed_at,
        )
        states = [
            {
                "trial_number": int(item["trial_number"]),
                "state": str(item["state"]),
            }
            for item in trial_state_universe
        ]
        completed = [
            item["trial_number"] for item in states if item["state"] == "COMPLETE"
        ]
        failed = [item["trial_number"] for item in states if item["state"] == "FAIL"]
        interrupted = sorted(int(value) for value in interrupted_recovery_trial_numbers)
        payload = {
            "authority_schema_version": AUTHORITY_SCHEMA_VERSION,
            "ledger_schema_version": LEDGER_SCHEMA_VERSION,
            "authority_key_fingerprint": self.key.fingerprint,
            "study_uuid": self.initial_statement["payload"]["study_uuid"],
            "base_search_identity_sha256": self.initial_statement["payload"][
                "base_search_identity_sha256"
            ],
            "authority_bound_study_identity_sha256": self.initial_statement["payload"][
                "authority_bound_study_identity_sha256"
            ],
            "ordered_event_signatures": [
                event["signature_sha256"] for event in self.events
            ],
            "event_count": len(self.events),
            "trial_state_universe": states,
            "completed_trial_numbers": completed,
            "failed_trial_numbers": failed,
            "interrupted_trial_numbers": interrupted,
            "interrupted_recovery_trial_numbers": interrupted,
            "best_trial_number": best_trial_number,
            "report_search_identity_sha256": report_search_identity_sha256,
            "study_summary_identity_sha256": study_summary_identity_sha256,
            "metric_prediction_evidence_identity_sha256": (
                metric_prediction_evidence_identity_sha256
            ),
            "report_manifest_identity_sha256": report_manifest_identity_sha256,
            "completed_at_utc": completed_at,
        }
        ledger = _signed_statement(payload, key=self.key, domain=LEDGER_DOMAIN)
        result = {
            "authority_schema_version": AUTHORITY_SCHEMA_VERSION,
            "report_finalized_event": final_event,
            "final_ledger": ledger,
        }
        self.study.set_user_attr("lifecycle_authority_final_ledger", result)
        return result


def validate_report_lifecycle_authority(
    *,
    authority_report: Mapping[str, Any],
    authority_ledger: Mapping[str, Any],
    key: LifecycleAuthorityKey,
    expected: Mapping[str, Any],
    database_path: Path | None,
    study_name: str,
) -> None:
    if set(authority_report) != {
        "authority_schema_version",
        "initial_statement",
        "events",
    }:
        raise OptunaLifecycleAuthorityError(
            "Filesystem lifecycle authority report schema differs."
        )
    if authority_report["authority_schema_version"] != AUTHORITY_SCHEMA_VERSION:
        raise OptunaLifecycleAuthorityError(
            "Filesystem lifecycle authority schema differs."
        )
    if set(authority_ledger) != {
        "authority_schema_version",
        "report_finalized_event",
        "final_ledger",
    }:
        raise OptunaLifecycleAuthorityError(
            "Final lifecycle authority ledger schema differs."
        )
    if authority_ledger["authority_schema_version"] != AUTHORITY_SCHEMA_VERSION:
        raise OptunaLifecycleAuthorityError("Final lifecycle authority schema differs.")
    initial = authority_report["initial_statement"]
    validate_initial_statement(initial, key=key)
    events = [*authority_report["events"], authority_ledger["report_finalized_event"]]
    validate_event_chain(
        events,
        initial_statement=initial,
        key=key,
        require_report_finalized=True,
    )
    ledger = authority_ledger["final_ledger"]
    payload = _validate_signed_statement(
        ledger,
        key=key,
        domain=LEDGER_DOMAIN,
        payload_keys=LEDGER_PAYLOAD_KEYS,
        label="final lifecycle ledger",
    )
    if (
        type(payload["authority_schema_version"]) is not int
        or payload["authority_schema_version"] != AUTHORITY_SCHEMA_VERSION
    ):
        raise OptunaLifecycleAuthorityError("Final lifecycle ledger version differs.")
    if (
        type(payload["ledger_schema_version"]) is not int
        or payload["ledger_schema_version"] != LEDGER_SCHEMA_VERSION
    ):
        raise OptunaLifecycleAuthorityError("Final lifecycle ledger schema differs.")
    expected_values = {
        **expected,
        "authority_key_fingerprint": key.fingerprint,
        "study_uuid": initial["payload"]["study_uuid"],
        "base_search_identity_sha256": initial["payload"][
            "base_search_identity_sha256"
        ],
        "authority_bound_study_identity_sha256": initial["payload"][
            "authority_bound_study_identity_sha256"
        ],
        "ordered_event_signatures": [event["signature_sha256"] for event in events],
        "event_count": len(events),
    }
    if any(payload.get(name) != value for name, value in expected_values.items()):
        raise OptunaLifecycleAuthorityError(
            "Final lifecycle ledger differs from reconstructed report evidence."
        )
    if payload["completed_at_utc"] != events[-1]["payload"]["event_at_utc"]:
        raise OptunaLifecycleAuthorityError(
            "Final lifecycle ledger timestamp differs from final event."
        )
    _validate_ledger_types(payload)
    if database_path is not None and database_path.exists():
        _cross_check_sqlite_authority(
            database_path,
            study_name=study_name,
            initial_statement=initial,
            events=events,
            final_ledger=authority_ledger,
            expected_states=payload["trial_state_universe"],
        )


def validate_initial_statement(
    statement: Mapping[str, Any],
    *,
    key: LifecycleAuthorityKey,
) -> None:
    payload = _validate_signed_statement(
        statement,
        key=key,
        domain=INITIAL_DOMAIN,
        payload_keys=INITIAL_PAYLOAD_KEYS,
        label="initial lifecycle authority statement",
    )
    if (
        type(payload["authority_schema_version"]) is not int
        or payload["authority_schema_version"] != AUTHORITY_SCHEMA_VERSION
        or payload["authority_key_fingerprint"] != key.fingerprint
    ):
        raise OptunaLifecycleAuthorityError(
            "External lifecycle authority key fingerprint differs."
        )
    _require_sha256(
        payload["base_search_identity_sha256"],
        "base search identity",
    )
    for name in (
        "authority_bound_study_identity_sha256",
        "dataset_identity_sha256",
        "assignment_identity_sha256",
        "source_closure_identity_sha256",
        "runtime_identity_sha256",
        "sampler_pruner_identity_sha256",
    ):
        _require_sha256(payload[name], name)
    if payload[
        "authority_bound_study_identity_sha256"
    ] != authority_bound_study_identity(
        base_search_identity_sha256=payload["base_search_identity_sha256"],
        authority_key_fingerprint=key.fingerprint,
    ):
        raise OptunaLifecycleAuthorityError("Authority-bound study identity differs.")
    try:
        if str(uuid.UUID(payload["study_uuid"])) != payload["study_uuid"]:
            raise ValueError
    except (AttributeError, TypeError, ValueError) as error:
        raise OptunaLifecycleAuthorityError("Study UUID is malformed.") from error
    if (
        type(payload["configured_trial_count"]) is not int
        or payload["configured_trial_count"] < 1
    ):
        raise OptunaLifecycleAuthorityError(
            "Configured authority trial count is malformed."
        )
    if (
        canonical_authority_timestamp(payload["created_at_utc"])
        != payload["created_at_utc"]
    ):
        raise OptunaLifecycleAuthorityError(
            "Authority creation timestamp is malformed."
        )


def validate_event_chain(
    events: Sequence[Mapping[str, Any]],
    *,
    initial_statement: Mapping[str, Any],
    key: LifecycleAuthorityKey,
    require_report_finalized: bool,
) -> None:
    validate_initial_statement(initial_statement, key=key)
    study_uuid = initial_statement["payload"]["study_uuid"]
    base_identity = initial_statement["payload"]["base_search_identity_sha256"]
    previous_signature = initial_statement["signature_sha256"]
    trial_states: dict[int, str] = {}
    study_state: str | None = None
    previous_time = initial_statement["payload"]["created_at_utc"]
    for expected_sequence, event in enumerate(events, start=1):
        payload = _validate_signed_statement(
            event,
            key=key,
            domain=EVENT_DOMAIN,
            payload_keys=EVENT_PAYLOAD_KEYS | {"event_payload_identity_sha256"},
            label="lifecycle event",
        )
        if payload["event_payload_identity_sha256"] != _event_payload_identity(
            {name: payload[name] for name in EVENT_PAYLOAD_KEYS}
        ):
            raise OptunaLifecycleAuthorityError(
                "Lifecycle event payload identity differs."
            )
        if (
            type(payload["authority_schema_version"]) is not int
            or type(payload["event_schema_version"]) is not int
            or payload["authority_schema_version"] != AUTHORITY_SCHEMA_VERSION
            or payload["event_schema_version"] != EVENT_SCHEMA_VERSION
            or payload["study_uuid"] != study_uuid
            or payload["base_search_identity_sha256"] != base_identity
            or payload["event_sequence"] != expected_sequence
            or payload["previous_event_signature_sha256"]
            != hashlib.sha256(bytes.fromhex(previous_signature)).hexdigest()
        ):
            raise OptunaLifecycleAuthorityError(
                "Lifecycle event sequence or hash-chain differs."
            )
        _validate_event_types(payload)
        event_type = payload["event_type"]
        expected_transition = EVENT_TRANSITIONS.get(event_type)
        if expected_transition != (payload["from_state"], payload["to_state"]):
            raise OptunaLifecycleAuthorityError(
                "Lifecycle event state transition is illegal."
            )
        event_time = canonical_authority_timestamp(payload["event_at_utc"])
        if event_time < previous_time:
            raise OptunaLifecycleAuthorityError(
                "Lifecycle event timestamps are out of order."
            )
        previous_time = event_time
        trial_number = payload["trial_number"]
        if event_type == "study_created":
            if expected_sequence != 1 or study_state is not None:
                raise OptunaLifecycleAuthorityError(
                    "Study-created lifecycle event is misplaced."
                )
            study_state = "STUDY_CREATED"
        elif event_type == "study_completed":
            if study_state != "STUDY_CREATED":
                raise OptunaLifecycleAuthorityError(
                    "Study-completed lifecycle event is misplaced."
                )
            if any(
                state not in {"COMPLETE", "FAIL"} for state in trial_states.values()
            ):
                raise OptunaLifecycleAuthorityError(
                    "Study completed with a nonterminal signed trial."
                )
            study_state = "STUDY_COMPLETED"
        elif event_type == "report_finalized":
            if study_state != "STUDY_COMPLETED":
                raise OptunaLifecycleAuthorityError(
                    "Report-finalized lifecycle event is misplaced."
                )
            study_state = "REPORT_FINALIZED"
        elif event_type == "trial_allocated":
            if study_state not in {"STUDY_CREATED", "REPORT_FINALIZED"} or (
                trial_number in trial_states
            ):
                raise OptunaLifecycleAuthorityError(
                    "Trial allocation lifecycle event is illegal."
                )
            study_state = "STUDY_CREATED"
            trial_states[trial_number] = "ALLOCATED"
        elif event_type == "trial_started":
            if trial_states.get(trial_number) != "ALLOCATED":
                raise OptunaLifecycleAuthorityError(
                    "Trial-start lifecycle event lacks allocation."
                )
            trial_states[trial_number] = "RUNNING"
        else:
            if trial_states.get(trial_number) != "RUNNING":
                raise OptunaLifecycleAuthorityError(
                    "Terminal trial lifecycle event lacks signed RUNNING state."
                )
            trial_states[trial_number] = payload["to_state"]
        previous_signature = event["signature_sha256"]
    if require_report_finalized and study_state != "REPORT_FINALIZED":
        raise OptunaLifecycleAuthorityError(
            "Signed lifecycle report-finalized event is missing."
        )
    del require_report_finalized


def _signed_trial_states(events: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    states: dict[int, str] = {}
    for event in events:
        payload = event["payload"]
        event_type = payload["event_type"]
        trial_number = payload["trial_number"]
        if event_type in {"trial_allocated", "trial_started"}:
            states[int(trial_number)] = "RUNNING"
        elif event_type in {
            "trial_completed",
            "trial_execution_failed",
            "interrupted_running_trial_recovered",
        }:
            states[int(trial_number)] = str(payload["to_state"])
    return states


def authority_bound_study_identity(
    *,
    base_search_identity_sha256: str,
    authority_key_fingerprint: str,
) -> str:
    payload = {
        "authority_schema_version": AUTHORITY_SCHEMA_VERSION,
        "authority_key_fingerprint": authority_key_fingerprint,
        "base_search_identity_sha256": base_search_identity_sha256,
    }
    return hashlib.sha256(
        BOUND_IDENTITY_DOMAIN + canonical_json_bytes(payload)
    ).hexdigest()


def metric_prediction_evidence_identity(
    *,
    trial_metrics_bytes: bytes,
    trial_predictions_bytes: bytes,
) -> str:
    payload = {
        "schema_version": 1,
        "trial_metrics_sha256": hashlib.sha256(trial_metrics_bytes).hexdigest(),
        "trial_predictions_sha256": hashlib.sha256(trial_predictions_bytes).hexdigest(),
    }
    return hashlib.sha256(EVIDENCE_DOMAIN + canonical_json_bytes(payload)).hexdigest()


def sampler_pruner_identity(
    *,
    sampler: Mapping[str, Any],
    pruner: str,
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "sampler": deepcopy(dict(sampler)),
                "pruner": pruner,
            }
        )
    ).hexdigest()


def _signed_statement(
    payload: Mapping[str, Any],
    *,
    key: LifecycleAuthorityKey,
    domain: bytes,
) -> dict[str, Any]:
    payload_bytes = canonical_json_bytes(payload)
    signature = hmac.new(
        key.material,
        domain + payload_bytes,
        hashlib.sha256,
    ).hexdigest()
    return {
        "payload": deepcopy(dict(payload)),
        "payload_identity_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "signature_sha256": signature,
    }


def _validate_signed_statement(
    statement: Mapping[str, Any],
    *,
    key: LifecycleAuthorityKey,
    domain: bytes,
    payload_keys: set[str],
    label: str,
) -> dict[str, Any]:
    if (
        not isinstance(statement, Mapping)
        or set(statement)
        != {
            "payload",
            "payload_identity_sha256",
            "signature_sha256",
        }
        or not isinstance(statement["payload"], Mapping)
        or set(statement["payload"]) != payload_keys
    ):
        raise OptunaLifecycleAuthorityError(f"{label.capitalize()} schema differs.")
    signature = _require_sha256(statement["signature_sha256"], f"{label} signature")
    payload_identity = _require_sha256(
        statement["payload_identity_sha256"],
        f"{label} payload identity",
    )
    payload = deepcopy(dict(statement["payload"]))
    payload_bytes = canonical_json_bytes(payload)
    if payload_identity != hashlib.sha256(payload_bytes).hexdigest():
        raise OptunaLifecycleAuthorityError(
            f"{label.capitalize()} payload identity differs."
        )
    expected = hmac.new(
        key.material,
        domain + payload_bytes,
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise OptunaLifecycleAuthorityError(f"{label.capitalize()} HMAC differs.")
    return payload


def _event_payload_identity(payload: Mapping[str, Any]) -> str:
    if set(payload) != EVENT_PAYLOAD_KEYS:
        raise OptunaLifecycleAuthorityError(
            "Lifecycle event payload identity schema differs."
        )
    return hashlib.sha256(
        EVENT_PAYLOAD_DOMAIN + canonical_json_bytes(payload)
    ).hexdigest()


def _validate_event_types(payload: Mapping[str, Any]) -> None:
    if type(payload["event_type"]) is not str or payload["event_type"] not in (
        EVENT_TRANSITIONS
    ):
        raise OptunaLifecycleAuthorityError("Lifecycle event type is unsupported.")
    if type(payload["event_sequence"]) is not int or payload["event_sequence"] < 1:
        raise OptunaLifecycleAuthorityError("Lifecycle event sequence is malformed.")
    _require_sha256(
        payload["previous_event_signature_sha256"],
        "previous lifecycle event signature",
    )
    _require_sha256(
        payload["event_payload_identity_sha256"],
        "lifecycle event payload identity",
    )
    trial_event = (
        payload["event_type"].startswith("trial_")
        or payload["event_type"] == "interrupted_running_trial_recovered"
    )
    if trial_event != (
        type(payload["trial_number"]) is int and payload["trial_number"] >= 0
    ):
        raise OptunaLifecycleAuthorityError(
            "Lifecycle event trial number is malformed."
        )
    for name in (
        "from_state",
        "to_state",
        "failure_stage",
        "failure_reason_code",
        "exception_type",
        "started_at_utc",
    ):
        if payload[name] is not None and type(payload[name]) is not str:
            raise OptunaLifecycleAuthorityError(f"Lifecycle event {name} is malformed.")
    if type(payload["to_state"]) is not str:
        raise OptunaLifecycleAuthorityError("Lifecycle event to_state is malformed.")
    if type(payload["interrupted_recovery"]) is not bool:
        raise OptunaLifecycleAuthorityError(
            "Lifecycle event interruption flag is malformed."
        )
    if payload["started_at_utc"] is not None:
        started = canonical_authority_timestamp(payload["started_at_utc"])
        if started > payload["event_at_utc"]:
            raise OptunaLifecycleAuthorityError(
                "Lifecycle event start timestamp follows event timestamp."
            )
    failure_event = payload["event_type"] in {
        "trial_execution_failed",
        "interrupted_running_trial_recovered",
    }
    if failure_event:
        if (
            type(payload["failure_stage"]) is not str
            or type(payload["failure_reason_code"]) is not str
            or not payload["failure_reason_code"]
            or not SHA256_PATTERN.fullmatch(str(payload["failure_message_sha256"]))
            or payload["started_at_utc"] is None
        ):
            raise OptunaLifecycleAuthorityError(
                "Failure lifecycle event evidence is incomplete."
            )
        interrupted = payload["event_type"] == "interrupted_running_trial_recovered"
        if payload["interrupted_recovery"] is not interrupted:
            raise OptunaLifecycleAuthorityError(
                "Failure lifecycle event recovery semantics differ."
            )
        if interrupted and payload["exception_type"] is not None:
            raise OptunaLifecycleAuthorityError(
                "Interrupted recovery cannot carry an exception type."
            )
        if not interrupted and (
            type(payload["exception_type"]) is not str or not payload["exception_type"]
        ):
            raise OptunaLifecycleAuthorityError(
                "Execution failure must carry an exception type."
            )
    elif (
        any(
            payload[name] is not None
            for name in (
                "failure_stage",
                "failure_reason_code",
                "exception_type",
                "failure_message_sha256",
            )
        )
        or payload["interrupted_recovery"]
    ):
        raise OptunaLifecycleAuthorityError(
            "Nonfailure lifecycle event carries failure evidence."
        )


def _validate_ledger_types(payload: Mapping[str, Any]) -> None:
    for name in (
        "authority_key_fingerprint",
        "base_search_identity_sha256",
        "authority_bound_study_identity_sha256",
        "report_search_identity_sha256",
        "study_summary_identity_sha256",
        "metric_prediction_evidence_identity_sha256",
        "report_manifest_identity_sha256",
    ):
        _require_sha256(payload[name], name)
    if type(payload["event_count"]) is not int or payload["event_count"] < 3:
        raise OptunaLifecycleAuthorityError("Final lifecycle event count is malformed.")
    if (
        not isinstance(payload["ordered_event_signatures"], list)
        or len(payload["ordered_event_signatures"]) != payload["event_count"]
        or any(
            not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value)
            for value in payload["ordered_event_signatures"]
        )
    ):
        raise OptunaLifecycleAuthorityError(
            "Final lifecycle event signature universe is malformed."
        )
    if (
        canonical_authority_timestamp(payload["completed_at_utc"])
        != payload["completed_at_utc"]
    ):
        raise OptunaLifecycleAuthorityError(
            "Final lifecycle completion timestamp is malformed."
        )


def _cross_check_sqlite_authority(
    database_path: Path,
    *,
    study_name: str,
    initial_statement: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    final_ledger: Mapping[str, Any],
    expected_states: Sequence[Mapping[str, Any]],
) -> None:
    uri = f"file:{database_path.as_posix()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            study_row = connection.execute(
                "SELECT study_id FROM studies WHERE study_name = ?",
                (study_name,),
            ).fetchone()
            if study_row is None:
                raise OptunaLifecycleAuthorityError(
                    "SQLite lifecycle authority study is missing."
                )
            study_id = int(study_row[0])
            attributes = {
                str(key): json.loads(value)
                for key, value in connection.execute(
                    "SELECT key, value_json FROM study_user_attributes "
                    "WHERE study_id = ?",
                    (study_id,),
                )
            }
            expected_attributes = {
                "lifecycle_authority_initial_statement": dict(initial_statement),
                "lifecycle_authority_events": list(events),
                "lifecycle_authority_final_ledger": dict(final_ledger),
            }
            if any(
                attributes.get(name) != value
                for name, value in expected_attributes.items()
            ):
                raise OptunaLifecycleAuthorityError(
                    "SQLite and filesystem lifecycle authority differ."
                )
            states = [
                {"trial_number": int(number), "state": str(state)}
                for number, state in connection.execute(
                    "SELECT number, state FROM trials WHERE study_id = ? "
                    "ORDER BY number",
                    (study_id,),
                )
            ]
            if states != list(expected_states):
                raise OptunaLifecycleAuthorityError(
                    "SQLite and filesystem trial-state universes differ."
                )
    except OptunaLifecycleAuthorityError:
        raise
    except (OSError, sqlite3.Error, json.JSONDecodeError) as error:
        raise OptunaLifecycleAuthorityError(
            "SQLite lifecycle authority validation failed."
        ) from error


def _reject_linked_components(path: Path) -> None:
    anchor = Path(path.anchor)
    current = anchor
    parts = path.parts[1:] if path.anchor else path.parts
    for part in parts:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as error:
            raise OptunaLifecycleAuthorityError(
                "External lifecycle authority path is unavailable."
            ) from error
        if stat.S_ISLNK(metadata.st_mode) or _is_reparse_stat(metadata):
            raise OptunaLifecycleAuthorityError(
                "External lifecycle authority path contains a link or reparse point."
            )


def _create_real_parent_chain(parent: Path) -> None:
    missing: list[Path] = []
    current = parent
    while not current.exists():
        missing.append(current)
        current = current.parent
    _reject_linked_components(current)
    for directory in reversed(missing):
        directory.mkdir()
        metadata = directory.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or _is_reparse_stat(metadata):
            raise OptunaLifecycleAuthorityError(
                "Authority initialization parent is unsafe."
            )


def _is_reparse_stat(metadata: os.stat_result) -> bool:
    return bool(getattr(metadata, "st_file_attributes", 0) & 0x400)


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or not SHA256_PATTERN.fullmatch(value):
        raise OptunaLifecycleAuthorityError(f"{label.capitalize()} is malformed.")
    return value


def _validate_json_primitives(value: Any, label: str) -> None:
    if value is None or type(value) in {str, int, bool}:
        return
    if type(value) is float:
        raise OptunaLifecycleAuthorityError(
            f"{label.capitalize()} must not contain floats."
        )
    if isinstance(value, list):
        for item in value:
            _validate_json_primitives(item, label)
        return
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise OptunaLifecycleAuthorityError(
                f"{label.capitalize()} keys must be strings."
            )
        for item in value.values():
            _validate_json_primitives(item, label)
        return
    raise OptunaLifecycleAuthorityError(
        f"{label.capitalize()} contains a nonprimitive value."
    )
