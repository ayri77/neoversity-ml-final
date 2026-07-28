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
EPOCH_SCHEMA_VERSION = 1
INITIAL_DOMAIN = b"churn_ml.optuna.lifecycle.initial.v1\x00"
EVENT_DOMAIN = b"churn_ml.optuna.lifecycle.event.v1\x00"
EVENT_PAYLOAD_DOMAIN = b"churn_ml.optuna.lifecycle.event-payload.v1\x00"
LEDGER_DOMAIN = b"churn_ml.optuna.lifecycle.ledger.v1\x00"
EPOCH_DOMAIN = b"churn_ml.optuna.lifecycle.epoch.v1\x00"
KEY_IDENTIFIER_DOMAIN = b"churn_ml.optuna.lifecycle.key-identifier.v1\x00"
BOUND_IDENTITY_DOMAIN = b"churn_ml.optuna.lifecycle.bound-identity.v1\x00"
EVIDENCE_DOMAIN = b"churn_ml.optuna.lifecycle.metric-prediction.v1\x00"
EPOCHS_ATTR = "lifecycle_authority_epochs"
LEGACY_FINAL_LEDGER_ATTR = "lifecycle_authority_final_ledger"
AUTHORITY_KEY_INIT_INVALID_DESTINATION = "AUTHORITY_KEY_INIT_INVALID_DESTINATION"
AUTHORITY_KEY_INIT_PARENT_CREATE_FAILED = "AUTHORITY_KEY_INIT_PARENT_CREATE_FAILED"
AUTHORITY_KEY_INIT_ALREADY_EXISTS = "AUTHORITY_KEY_INIT_ALREADY_EXISTS"
AUTHORITY_KEY_INIT_CREATE_FAILED = "AUTHORITY_KEY_INIT_CREATE_FAILED"
AUTHORITY_KEY_INIT_WRITE_FAILED = "AUTHORITY_KEY_INIT_WRITE_FAILED"
AUTHORITY_KEY_INIT_FLUSH_FAILED = "AUTHORITY_KEY_INIT_FLUSH_FAILED"
AUTHORITY_KEY_INIT_PERMISSION_FAILED = "AUTHORITY_KEY_INIT_PERMISSION_FAILED"
AUTHORITY_KEY_INIT_VERIFY_FAILED = "AUTHORITY_KEY_INIT_VERIFY_FAILED"
AUTHORITY_KEY_INIT_CLEANUP_FAILED = "AUTHORITY_KEY_INIT_CLEANUP_FAILED"
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
    "configured_trial_target_before",
    "configured_trial_target_after",
    "previous_epoch_number",
    "opened_epoch_number",
}
EPOCH_PAYLOAD_KEYS = {
    "authority_schema_version",
    "epoch_schema_version",
    "study_uuid",
    "epoch_number",
    "previous_epoch_ledger_signature_sha256",
    "epoch_open_event_sequence",
    "epoch_close_event_sequence",
    "configured_trial_target",
    "starting_trial_universe",
    "ending_trial_universe",
    "event_signature_prefix_count",
    "event_signature_end_count",
    "report_search_identity_sha256",
    "study_summary_identity_sha256",
    "metric_prediction_evidence_identity_sha256",
    "pre_terminal_manifest_identity_sha256",
    "completed_at_utc",
    "epoch_payload_identity_sha256",
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
    "study_target_extended": ("REPORT_FINALIZED", "STUDY_CREATED"),
}


class OptunaLifecycleAuthorityError(RuntimeError):
    """Raised when external lifecycle authority is absent or inconsistent."""


class AuthorityKeyInitError(OptunaLifecycleAuthorityError):
    """Stable, path-independent authority-key initialization failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"AuthorityKeyInitError(code={self.code!r}, message={self.message!r})"


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
    forbidden_roots: Sequence[Path] = (),
) -> str:
    try:
        return _initialize_lifecycle_authority_key(
            output,
            project_root=project_root,
            create_parent=create_parent,
            forbidden_roots=forbidden_roots,
        )
    except AuthorityKeyInitError:
        raise
    except OptunaLifecycleAuthorityError:
        raise AuthorityKeyInitError(
            AUTHORITY_KEY_INIT_INVALID_DESTINATION,
            "Authority initialization destination is invalid.",
        ) from None
    except OSError:
        raise AuthorityKeyInitError(
            AUTHORITY_KEY_INIT_CREATE_FAILED,
            "Authority initialization could not create the key file.",
        ) from None


def _initialize_lifecycle_authority_key(
    output: Path,
    *,
    project_root: Path,
    create_parent: bool,
    forbidden_roots: Sequence[Path],
) -> str:
    if not output.is_absolute() or any(part in {".", ".."} for part in output.parts):
        raise AuthorityKeyInitError(
            AUTHORITY_KEY_INIT_INVALID_DESTINATION,
            "Authority initialization destination is invalid.",
        )
    target = Path(os.path.abspath(output))
    root = project_root.resolve()
    forbidden = [root, *(Path(item).resolve() for item in forbidden_roots)]
    if any(target == item or item in target.parents for item in forbidden):
        raise AuthorityKeyInitError(
            AUTHORITY_KEY_INIT_INVALID_DESTINATION,
            "Authority initialization destination is invalid.",
        )
    parent = target.parent
    if not parent.exists():
        if not create_parent:
            raise AuthorityKeyInitError(
                AUTHORITY_KEY_INIT_INVALID_DESTINATION,
                "Authority initialization parent is missing.",
            )
        try:
            _create_real_parent_chain(parent)
        except AuthorityKeyInitError:
            raise
        except OptunaLifecycleAuthorityError:
            raise AuthorityKeyInitError(
                AUTHORITY_KEY_INIT_PARENT_CREATE_FAILED,
                "Authority initialization parent could not be created.",
            ) from None
        except OSError:
            raise AuthorityKeyInitError(
                AUTHORITY_KEY_INIT_PARENT_CREATE_FAILED,
                "Authority initialization parent could not be created.",
            ) from None
    try:
        _reject_linked_components(parent)
    except OptunaLifecycleAuthorityError:
        raise AuthorityKeyInitError(
            AUTHORITY_KEY_INIT_INVALID_DESTINATION,
            "Authority initialization destination is invalid.",
        ) from None
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            raise AuthorityKeyInitError(
                AUTHORITY_KEY_INIT_INVALID_DESTINATION,
                "Authority initialization destination is invalid.",
            )
        raise AuthorityKeyInitError(
            AUTHORITY_KEY_INIT_ALREADY_EXISTS,
            "Authority initialization refuses overwrite.",
        )

    material = secrets.token_bytes(32)
    temp_path = parent / f".churn-ml-authority-{secrets.token_hex(16)}.tmp"
    temp_identity: tuple[int, int] | None = None
    published_identity: tuple[int, int] | None = None
    try:
        temp_identity = _exclusive_write_key_file(temp_path, material)
        _publish_authority_key(temp_path, target, material)
        try:
            published_meta = target.lstat()
            published_identity = (published_meta.st_dev, published_meta.st_ino)
        except OSError:
            raise AuthorityKeyInitError(
                AUTHORITY_KEY_INIT_VERIFY_FAILED,
                "Authority initialization could not verify the published key.",
            ) from None
        try:
            current_temp = temp_path.lstat()
            if (current_temp.st_dev, current_temp.st_ino) == temp_identity:
                temp_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            raise AuthorityKeyInitError(
                AUTHORITY_KEY_INIT_CLEANUP_FAILED,
                "Authority initialization could not clean up its temporary file.",
            ) from None
        temp_identity = None
        try:
            loaded = load_lifecycle_authority_key_from_path(
                target,
                project_root=root,
            )
            final_meta = target.lstat()
        except (OSError, OptunaLifecycleAuthorityError):
            raise AuthorityKeyInitError(
                AUTHORITY_KEY_INIT_VERIFY_FAILED,
                "Authority initialization could not verify the published key.",
            ) from None
        if (
            not stat.S_ISREG(final_meta.st_mode)
            or stat.S_ISLNK(final_meta.st_mode)
            or _is_reparse_stat(final_meta)
            or getattr(final_meta, "st_nlink", 1) != 1
            or final_meta.st_size != 32
            or (final_meta.st_dev, final_meta.st_ino) != published_identity
            or loaded.fingerprint
            != hashlib.sha256(KEY_IDENTIFIER_DOMAIN + material).hexdigest()
        ):
            raise AuthorityKeyInitError(
                AUTHORITY_KEY_INIT_VERIFY_FAILED,
                "Authority initialization could not verify the published key.",
            )
        return loaded.fingerprint
    except AuthorityKeyInitError:
        _cleanup_authority_init_artifacts(
            temp_path=temp_path if temp_identity is not None else None,
            temp_identity=temp_identity,
            target=None,
            target_identity=None,
            published=published_identity is not None,
        )
        raise
    except OSError:
        _cleanup_authority_init_artifacts(
            temp_path=temp_path if temp_identity is not None else None,
            temp_identity=temp_identity,
            target=None,
            target_identity=None,
            published=published_identity is not None,
        )
        raise AuthorityKeyInitError(
            AUTHORITY_KEY_INIT_CREATE_FAILED,
            "Authority initialization could not create the key file.",
        ) from None


def _exclusive_write_key_file(path: Path, material: bytes) -> tuple[int, int]:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        raise AuthorityKeyInitError(
            AUTHORITY_KEY_INIT_CREATE_FAILED,
            "Authority initialization could not create the key file.",
        ) from None
    except OSError:
        raise AuthorityKeyInitError(
            AUTHORITY_KEY_INIT_CREATE_FAILED,
            "Authority initialization could not create the key file.",
        ) from None
    failure: AuthorityKeyInitError | None = None
    created_identity: tuple[int, int] | None = None
    try:
        try:
            written = 0
            while written < len(material):
                written += os.write(descriptor, material[written:])
        except OSError:
            failure = AuthorityKeyInitError(
                AUTHORITY_KEY_INIT_WRITE_FAILED,
                "Authority initialization could not write the key file.",
            )
        if failure is None:
            try:
                os.fsync(descriptor)
            except OSError:
                failure = AuthorityKeyInitError(
                    AUTHORITY_KEY_INIT_FLUSH_FAILED,
                    "Authority initialization could not flush the key file.",
                )
        if failure is None:
            identity = os.fstat(descriptor)
            created_identity = (identity.st_dev, identity.st_ino)
    finally:
        os.close(descriptor)
    if failure is not None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            raise AuthorityKeyInitError(
                AUTHORITY_KEY_INIT_CLEANUP_FAILED,
                "Authority initialization could not clean up its temporary file.",
            ) from None
        raise failure
    assert created_identity is not None
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    try:
        after = path.lstat()
    except OSError:
        raise AuthorityKeyInitError(
            AUTHORITY_KEY_INIT_VERIFY_FAILED,
            "Authority initialization could not verify the published key.",
        ) from None
    if (after.st_dev, after.st_ino) != created_identity or after.st_size != len(
        material
    ):
        raise AuthorityKeyInitError(
            AUTHORITY_KEY_INIT_VERIFY_FAILED,
            "Authority initialization could not verify the published key.",
        )
    return created_identity


def _publish_authority_key(temp_path: Path, target: Path, material: bytes) -> None:
    if target.exists() or target.is_symlink():
        raise AuthorityKeyInitError(
            AUTHORITY_KEY_INIT_ALREADY_EXISTS,
            "Authority initialization refuses overwrite.",
        )
    try:
        os.link(temp_path, target)
        return
    except FileExistsError:
        raise AuthorityKeyInitError(
            AUTHORITY_KEY_INIT_ALREADY_EXISTS,
            "Authority initialization refuses overwrite.",
        ) from None
    except (AttributeError, OSError):
        pass
    if target.exists() or target.is_symlink():
        raise AuthorityKeyInitError(
            AUTHORITY_KEY_INIT_ALREADY_EXISTS,
            "Authority initialization refuses overwrite.",
        )
    try:
        _exclusive_write_key_file(target, material)
    except AuthorityKeyInitError as error:
        if error.code == AUTHORITY_KEY_INIT_CREATE_FAILED and (
            target.exists() or target.is_symlink()
        ):
            raise AuthorityKeyInitError(
                AUTHORITY_KEY_INIT_ALREADY_EXISTS,
                "Authority initialization refuses overwrite.",
            ) from None
        raise


def _cleanup_authority_init_artifacts(
    *,
    temp_path: Path | None,
    temp_identity: tuple[int, int] | None,
    target: Path | None,
    target_identity: tuple[int, int] | None,
    published: bool,
) -> None:
    del published
    if temp_path is not None and temp_identity is not None:
        try:
            meta = temp_path.lstat()
            if (meta.st_dev, meta.st_ino) == temp_identity:
                temp_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            raise AuthorityKeyInitError(
                AUTHORITY_KEY_INIT_CLEANUP_FAILED,
                "Authority initialization could not clean up its temporary file.",
            ) from None
    if target is not None and target_identity is not None:
        try:
            meta = target.lstat()
            if (meta.st_dev, meta.st_ino) == target_identity:
                target.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            raise AuthorityKeyInitError(
                AUTHORITY_KEY_INIT_CLEANUP_FAILED,
                "Authority initialization could not clean up its temporary file.",
            ) from None


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
        self._epochs: list[dict[str, Any]] | None = None

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
            open_epoch = _build_open_epoch_record(
                study_uuid=study_uuid,
                epoch_number=0,
                previous_epoch_ledger_signature_sha256=None,
                epoch_open_event_sequence=1,
                configured_trial_target=configured_trial_count,
                starting_trial_universe=[],
                event_signature_prefix_count=0,
            )
            study.set_user_attr(EPOCHS_ATTR, [open_epoch])
            recorder._epochs = [open_epoch]
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
        recorder._epochs = recorder._load_and_validate_epochs()
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

    def _load_and_validate_epochs(self) -> list[dict[str, Any]]:
        legacy = self.study.user_attrs.get(LEGACY_FINAL_LEDGER_ATTR)
        persisted = self.study.user_attrs.get(EPOCHS_ATTR)
        if legacy is not None and not isinstance(persisted, list):
            raise OptunaLifecycleAuthorityError(
                "Singleton final ledger schema is unsupported."
            )
        if not isinstance(persisted, list) or not persisted:
            raise OptunaLifecycleAuthorityError(
                "Study lifecycle authority epoch history is missing."
            )
        _validate_epoch_history(
            persisted,
            events=self.events,
            initial_statement=self.initial_statement,
            key=self.key,
        )
        return [deepcopy(item) for item in persisted]

    def _require_epochs(self) -> list[dict[str, Any]]:
        if self._epochs is None:
            self._epochs = self._load_and_validate_epochs()
        return self._epochs

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
        epochs = self._require_epochs()
        _validate_epoch_history(
            epochs,
            events=self.events,
            initial_statement=self.initial_statement,
            key=self.key,
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
        configured_trial_target_before: int | None = None,
        configured_trial_target_after: int | None = None,
        previous_epoch_number: int | None = None,
        opened_epoch_number: int | None = None,
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
            "configured_trial_target_before": configured_trial_target_before,
            "configured_trial_target_after": configured_trial_target_after,
            "previous_epoch_number": previous_epoch_number,
            "opened_epoch_number": opened_epoch_number,
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
        events = deepcopy(self.events)
        if events and events[-1]["payload"]["event_type"] == "report_finalized":
            events = events[:-1]
        return {
            "authority_schema_version": AUTHORITY_SCHEMA_VERSION,
            "initial_statement": deepcopy(self.initial_statement),
            "events": events,
        }

    def extend_configured_trial_target(self, new_target: int) -> None:
        if type(new_target) is not int or new_target < 1:
            raise OptunaLifecycleAuthorityError(
                "Extended lifecycle trial target is malformed."
            )
        epochs = self._require_epochs()
        if not self.events or self.events[-1]["payload"]["event_type"] != (
            "report_finalized"
        ):
            raise OptunaLifecycleAuthorityError(
                "Lifecycle target extension requires a finalized epoch."
            )
        latest = epochs[-1]
        if _epoch_status(latest) != "closed":
            raise OptunaLifecycleAuthorityError(
                "Lifecycle target extension requires a closed prior epoch."
            )
        previous_payload = latest["payload"]
        previous_target = int(previous_payload["configured_trial_target"])
        if new_target <= previous_target:
            raise OptunaLifecycleAuthorityError(
                "Lifecycle target extension requires a strictly larger target."
            )
        previous_number = int(previous_payload["epoch_number"])
        ending_universe = deepcopy(previous_payload["ending_trial_universe"])
        prefix_count = int(previous_payload["event_signature_end_count"])
        self.append_event(
            event_type="study_target_extended",
            trial_number=None,
            from_state="REPORT_FINALIZED",
            to_state="STUDY_CREATED",
            configured_trial_target_before=previous_target,
            configured_trial_target_after=new_target,
            previous_epoch_number=previous_number,
            opened_epoch_number=previous_number + 1,
        )
        open_epoch = _build_open_epoch_record(
            study_uuid=self.initial_statement["payload"]["study_uuid"],
            epoch_number=previous_number + 1,
            previous_epoch_ledger_signature_sha256=latest["signature_sha256"],
            epoch_open_event_sequence=len(self.events),
            configured_trial_target=new_target,
            starting_trial_universe=ending_universe,
            event_signature_prefix_count=prefix_count,
        )
        self._epochs = [*epochs, open_epoch]
        self.persist_epoch_history()

    def persist_epoch_history(self) -> None:
        epochs = self._require_epochs()
        self.study.set_user_attr(EPOCHS_ATTR, deepcopy(epochs))

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
        persist_epochs: bool = False,
    ) -> dict[str, Any]:
        del interrupted_recovery_trial_numbers, best_trial_number
        epochs = self._require_epochs()
        latest = epochs[-1]
        states = [
            {
                "trial_number": int(item["trial_number"]),
                "state": str(item["state"]),
            }
            for item in trial_state_universe
        ]
        if _epoch_status(latest) == "closed":
            closed = latest
            payload = closed["payload"]
            if (
                payload["report_search_identity_sha256"]
                == report_search_identity_sha256
                and payload["study_summary_identity_sha256"]
                == study_summary_identity_sha256
                and payload["metric_prediction_evidence_identity_sha256"]
                == metric_prediction_evidence_identity_sha256
                and payload["pre_terminal_manifest_identity_sha256"]
                == report_manifest_identity_sha256
                and payload["ending_trial_universe"] == states
            ):
                final_event = None
                for event in reversed(self.events):
                    if event["payload"]["event_type"] == "report_finalized":
                        final_event = deepcopy(event)
                        break
                if final_event is None:
                    raise OptunaLifecycleAuthorityError(
                        "Closed lifecycle epoch lacks report-finalized event."
                    )
                return {
                    "authority_schema_version": AUTHORITY_SCHEMA_VERSION,
                    "epoch_schema_version": EPOCH_SCHEMA_VERSION,
                    "epoch_number": int(payload["epoch_number"]),
                    "report_finalized_event": final_event,
                    "epoch_ledger": deepcopy(closed),
                }
            raise OptunaLifecycleAuthorityError(
                "Closed lifecycle epoch diverges from the new report evidence."
            )
        last_type = self.events[-1]["payload"]["event_type"]
        if last_type == "study_completed":
            completed_at = canonical_authority_timestamp(datetime.now(timezone.utc))
            final_event = self.append_event(
                event_type="report_finalized",
                trial_number=None,
                from_state="STUDY_COMPLETED",
                to_state="REPORT_FINALIZED",
                event_at_utc=completed_at,
            )
        elif last_type == "report_finalized":
            final_event = deepcopy(self.events[-1])
            completed_at = final_event["payload"]["event_at_utc"]
        else:
            raise OptunaLifecycleAuthorityError(
                "Lifecycle authority cannot finalize before study completion."
            )
        open_payload = latest["payload"]
        closed_payload = {
            "authority_schema_version": AUTHORITY_SCHEMA_VERSION,
            "epoch_schema_version": EPOCH_SCHEMA_VERSION,
            "study_uuid": self.initial_statement["payload"]["study_uuid"],
            "epoch_number": int(open_payload["epoch_number"]),
            "previous_epoch_ledger_signature_sha256": open_payload[
                "previous_epoch_ledger_signature_sha256"
            ],
            "epoch_open_event_sequence": int(open_payload["epoch_open_event_sequence"]),
            "epoch_close_event_sequence": len(self.events),
            "configured_trial_target": int(open_payload["configured_trial_target"]),
            "starting_trial_universe": deepcopy(
                open_payload["starting_trial_universe"]
            ),
            "ending_trial_universe": states,
            "event_signature_prefix_count": int(
                open_payload["event_signature_prefix_count"]
            ),
            "event_signature_end_count": len(self.events),
            "report_search_identity_sha256": report_search_identity_sha256,
            "study_summary_identity_sha256": study_summary_identity_sha256,
            "metric_prediction_evidence_identity_sha256": (
                metric_prediction_evidence_identity_sha256
            ),
            "pre_terminal_manifest_identity_sha256": report_manifest_identity_sha256,
            "completed_at_utc": completed_at,
        }
        identity = _epoch_payload_identity(closed_payload)
        closed_payload["epoch_payload_identity_sha256"] = identity
        signed = _signed_statement(
            closed_payload,
            key=self.key,
            domain=EPOCH_DOMAIN,
        )
        self._epochs = [*epochs[:-1], signed]
        if persist_epochs:
            self.persist_epoch_history()
        return {
            "authority_schema_version": AUTHORITY_SCHEMA_VERSION,
            "epoch_schema_version": EPOCH_SCHEMA_VERSION,
            "epoch_number": int(closed_payload["epoch_number"]),
            "report_finalized_event": final_event,
            "epoch_ledger": deepcopy(signed),
        }


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
    if "final_ledger" in authority_ledger:
        raise OptunaLifecycleAuthorityError(
            "Singleton final ledger schema is unsupported."
        )
    if set(authority_ledger) != {
        "authority_schema_version",
        "epoch_schema_version",
        "epoch_number",
        "report_finalized_event",
        "epoch_ledger",
    }:
        raise OptunaLifecycleAuthorityError(
            "Final lifecycle authority ledger schema differs."
        )
    if (
        authority_ledger["authority_schema_version"] != AUTHORITY_SCHEMA_VERSION
        or authority_ledger["epoch_schema_version"] != EPOCH_SCHEMA_VERSION
        or type(authority_ledger["epoch_number"]) is not int
        or authority_ledger["epoch_number"] < 0
    ):
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
    ledger = authority_ledger["epoch_ledger"]
    payload = _validate_signed_statement(
        ledger,
        key=key,
        domain=EPOCH_DOMAIN,
        payload_keys=EPOCH_PAYLOAD_KEYS,
        label="lifecycle epoch ledger",
    )
    if payload["epoch_payload_identity_sha256"] != _epoch_payload_identity(
        {
            name: payload[name]
            for name in EPOCH_PAYLOAD_KEYS - {"epoch_payload_identity_sha256"}
        }
    ):
        raise OptunaLifecycleAuthorityError("Lifecycle epoch payload identity differs.")
    if (
        payload["authority_schema_version"] != AUTHORITY_SCHEMA_VERSION
        or payload["epoch_schema_version"] != EPOCH_SCHEMA_VERSION
        or payload["study_uuid"] != initial["payload"]["study_uuid"]
        or payload["epoch_number"] != authority_ledger["epoch_number"]
        or payload["epoch_close_event_sequence"] != len(events)
        or payload["event_signature_end_count"] != len(events)
        or payload["completed_at_utc"] != events[-1]["payload"]["event_at_utc"]
    ):
        raise OptunaLifecycleAuthorityError(
            "Lifecycle epoch ledger differs from reconstructed report evidence."
        )
    expected_values = {
        "ending_trial_universe": expected["ending_trial_universe"],
        "report_search_identity_sha256": expected["report_search_identity_sha256"],
        "study_summary_identity_sha256": expected["study_summary_identity_sha256"],
        "metric_prediction_evidence_identity_sha256": expected[
            "metric_prediction_evidence_identity_sha256"
        ],
        "pre_terminal_manifest_identity_sha256": expected[
            "pre_terminal_manifest_identity_sha256"
        ],
    }
    if any(payload.get(name) != value for name, value in expected_values.items()):
        raise OptunaLifecycleAuthorityError(
            "Lifecycle epoch ledger differs from reconstructed report evidence."
        )
    _validate_epoch_payload_types(payload, open_epoch=False)
    if database_path is not None and database_path.exists():
        _cross_check_sqlite_authority(
            database_path,
            study_name=study_name,
            initial_statement=initial,
            events=events,
            epoch_number=int(authority_ledger["epoch_number"]),
            epoch_ledger=ledger,
            expected_states=payload["ending_trial_universe"],
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
        if event_type == "interrupted_running_trial_recovered":
            if payload["to_state"] != "FAIL" or payload["from_state"] not in {
                "RUNNING",
                "ALLOCATED",
            }:
                raise OptunaLifecycleAuthorityError(
                    "Lifecycle event state transition is illegal."
                )
        else:
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
        elif event_type == "study_target_extended":
            if study_state != "REPORT_FINALIZED":
                raise OptunaLifecycleAuthorityError(
                    "Study target-extension lifecycle event is misplaced."
                )
            if (
                type(payload["configured_trial_target_before"]) is not int
                or type(payload["configured_trial_target_after"]) is not int
                or type(payload["previous_epoch_number"]) is not int
                or type(payload["opened_epoch_number"]) is not int
                or payload["configured_trial_target_before"] < 1
                or payload["configured_trial_target_after"]
                <= payload["configured_trial_target_before"]
                or payload["opened_epoch_number"]
                != payload["previous_epoch_number"] + 1
            ):
                raise OptunaLifecycleAuthorityError(
                    "Study target-extension lifecycle binding is malformed."
                )
            study_state = "STUDY_CREATED"
        elif event_type == "trial_allocated":
            if study_state != "STUDY_CREATED" or (trial_number in trial_states):
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
        elif event_type == "interrupted_running_trial_recovered":
            if trial_states.get(trial_number) not in {"RUNNING", "ALLOCATED"}:
                raise OptunaLifecycleAuthorityError(
                    "Terminal trial lifecycle event lacks signed RUNNING state."
                )
            if payload["from_state"] != trial_states.get(trial_number):
                raise OptunaLifecycleAuthorityError(
                    "Lifecycle event state transition is illegal."
                )
            trial_states[trial_number] = payload["to_state"]
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
        if event_type == "trial_allocated":
            states[int(trial_number)] = "RUNNING"
        elif event_type == "trial_started":
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
    for name in (
        "configured_trial_target_before",
        "configured_trial_target_after",
        "previous_epoch_number",
        "opened_epoch_number",
    ):
        if payload[name] is not None and type(payload[name]) is not int:
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
    if payload["event_type"] == "study_target_extended":
        if (
            any(
                payload[name] is None
                for name in (
                    "configured_trial_target_before",
                    "configured_trial_target_after",
                    "previous_epoch_number",
                    "opened_epoch_number",
                )
            )
            or any(
                payload[name] is not None
                for name in (
                    "failure_stage",
                    "failure_reason_code",
                    "exception_type",
                    "failure_message_sha256",
                    "started_at_utc",
                )
            )
            or payload["interrupted_recovery"]
            or payload["trial_number"] is not None
        ):
            raise OptunaLifecycleAuthorityError(
                "Study target-extension lifecycle binding is malformed."
            )
        return
    if any(
        payload[name] is not None
        for name in (
            "configured_trial_target_before",
            "configured_trial_target_after",
            "previous_epoch_number",
            "opened_epoch_number",
        )
    ):
        raise OptunaLifecycleAuthorityError(
            "Non-extension lifecycle event carries target-extension binding."
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


def _validate_epoch_payload_types(
    payload: Mapping[str, Any],
    *,
    open_epoch: bool,
) -> None:
    if (
        type(payload["authority_schema_version"]) is not int
        or payload["authority_schema_version"] != AUTHORITY_SCHEMA_VERSION
        or type(payload["epoch_schema_version"]) is not int
        or payload["epoch_schema_version"] != EPOCH_SCHEMA_VERSION
        or type(payload["epoch_number"]) is not int
        or payload["epoch_number"] < 0
        or type(payload["epoch_open_event_sequence"]) is not int
        or payload["epoch_open_event_sequence"] < 1
        or type(payload["configured_trial_target"]) is not int
        or payload["configured_trial_target"] < 1
        or type(payload["event_signature_prefix_count"]) is not int
        or payload["event_signature_prefix_count"] < 0
        or not isinstance(payload["starting_trial_universe"], list)
    ):
        raise OptunaLifecycleAuthorityError("Lifecycle epoch payload is malformed.")
    try:
        if str(uuid.UUID(payload["study_uuid"])) != payload["study_uuid"]:
            raise ValueError
    except (AttributeError, TypeError, ValueError) as error:
        raise OptunaLifecycleAuthorityError("Study UUID is malformed.") from error
    if payload["previous_epoch_ledger_signature_sha256"] is not None:
        _require_sha256(
            payload["previous_epoch_ledger_signature_sha256"],
            "previous epoch ledger signature",
        )
    elif payload["epoch_number"] != 0:
        raise OptunaLifecycleAuthorityError("Lifecycle epoch chain is malformed.")
    _validate_trial_universe(payload["starting_trial_universe"])
    _require_sha256(
        payload["epoch_payload_identity_sha256"],
        "epoch payload identity",
    )
    if open_epoch:
        for name in (
            "epoch_close_event_sequence",
            "ending_trial_universe",
            "event_signature_end_count",
            "report_search_identity_sha256",
            "study_summary_identity_sha256",
            "metric_prediction_evidence_identity_sha256",
            "pre_terminal_manifest_identity_sha256",
            "completed_at_utc",
        ):
            if payload[name] is not None:
                raise OptunaLifecycleAuthorityError(
                    "Open lifecycle epoch carries closed fields."
                )
        return
    if (
        type(payload["epoch_close_event_sequence"]) is not int
        or payload["epoch_close_event_sequence"] < payload["epoch_open_event_sequence"]
        or type(payload["event_signature_end_count"]) is not int
        or payload["event_signature_end_count"] != payload["epoch_close_event_sequence"]
        or not isinstance(payload["ending_trial_universe"], list)
    ):
        raise OptunaLifecycleAuthorityError("Closed lifecycle epoch is malformed.")
    _validate_trial_universe(payload["ending_trial_universe"])
    for name in (
        "report_search_identity_sha256",
        "study_summary_identity_sha256",
        "metric_prediction_evidence_identity_sha256",
        "pre_terminal_manifest_identity_sha256",
    ):
        _require_sha256(payload[name], name)
    if (
        canonical_authority_timestamp(payload["completed_at_utc"])
        != payload["completed_at_utc"]
    ):
        raise OptunaLifecycleAuthorityError(
            "Lifecycle epoch completion timestamp is malformed."
        )


def _validate_trial_universe(universe: Sequence[Mapping[str, Any]]) -> None:
    seen: set[int] = set()
    for item in universe:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"trial_number", "state"}
            or type(item["trial_number"]) is not int
            or item["trial_number"] < 0
            or type(item["state"]) is not str
            or item["trial_number"] in seen
        ):
            raise OptunaLifecycleAuthorityError(
                "Lifecycle epoch trial universe is malformed."
            )
        seen.add(item["trial_number"])


def _cross_check_sqlite_authority(
    database_path: Path,
    *,
    study_name: str,
    initial_statement: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    epoch_number: int,
    epoch_ledger: Mapping[str, Any],
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
            if LEGACY_FINAL_LEDGER_ATTR in attributes and EPOCHS_ATTR not in attributes:
                raise OptunaLifecycleAuthorityError(
                    "Singleton final ledger schema is unsupported."
                )
            if attributes.get("lifecycle_authority_initial_statement") != dict(
                initial_statement
            ):
                raise OptunaLifecycleAuthorityError(
                    "SQLite and filesystem lifecycle authority differ."
                )
            sqlite_events = attributes.get("lifecycle_authority_events")
            if not isinstance(sqlite_events, list) or len(sqlite_events) < len(events):
                raise OptunaLifecycleAuthorityError(
                    "SQLite and filesystem lifecycle authority differ."
                )
            if sqlite_events[: len(events)] != list(events):
                raise OptunaLifecycleAuthorityError(
                    "SQLite and filesystem lifecycle authority differ."
                )
            sqlite_epochs = attributes.get(EPOCHS_ATTR)
            if not isinstance(sqlite_epochs, list) or epoch_number >= len(
                sqlite_epochs
            ):
                raise OptunaLifecycleAuthorityError(
                    "SQLite lifecycle epoch history is missing."
                )
            if sqlite_epochs[epoch_number] != dict(epoch_ledger):
                raise OptunaLifecycleAuthorityError(
                    "SQLite and filesystem lifecycle epoch ledgers differ."
                )
            if any(
                _epoch_status(item) == "open" and index != len(sqlite_epochs) - 1
                for index, item in enumerate(sqlite_epochs)
            ):
                raise OptunaLifecycleAuthorityError(
                    "Lifecycle epoch history has a non-terminal open epoch."
                )
            for index, item in enumerate(sqlite_epochs):
                payload = item["payload"]
                if int(payload["epoch_number"]) != index:
                    raise OptunaLifecycleAuthorityError(
                        "Lifecycle epoch history sequence is malformed."
                    )
            states = [
                {"trial_number": int(number), "state": str(state)}
                for number, state in connection.execute(
                    "SELECT number, state FROM trials WHERE study_id = ? "
                    "ORDER BY number",
                    (study_id,),
                )
            ]
            # Report ending universe may be a prefix of the operational study.
            if states[: len(expected_states)] != list(expected_states):
                raise OptunaLifecycleAuthorityError(
                    "SQLite and filesystem trial-state universes differ."
                )
    except OptunaLifecycleAuthorityError:
        raise
    except (OSError, sqlite3.Error, json.JSONDecodeError) as error:
        raise OptunaLifecycleAuthorityError(
            "SQLite lifecycle authority validation failed."
        ) from error


def _epoch_status(record: Mapping[str, Any]) -> str:
    if record.get("epoch_status") == "open":
        return "open"
    if set(record) == {"payload", "payload_identity_sha256", "signature_sha256"}:
        return "closed"
    raise OptunaLifecycleAuthorityError("Lifecycle epoch record schema differs.")


def _build_open_epoch_record(
    *,
    study_uuid: str,
    epoch_number: int,
    previous_epoch_ledger_signature_sha256: str | None,
    epoch_open_event_sequence: int,
    configured_trial_target: int,
    starting_trial_universe: Sequence[Mapping[str, Any]],
    event_signature_prefix_count: int,
) -> dict[str, Any]:
    payload = {
        "authority_schema_version": AUTHORITY_SCHEMA_VERSION,
        "epoch_schema_version": EPOCH_SCHEMA_VERSION,
        "study_uuid": study_uuid,
        "epoch_number": epoch_number,
        "previous_epoch_ledger_signature_sha256": (
            previous_epoch_ledger_signature_sha256
        ),
        "epoch_open_event_sequence": epoch_open_event_sequence,
        "epoch_close_event_sequence": None,
        "configured_trial_target": configured_trial_target,
        "starting_trial_universe": [
            {
                "trial_number": int(item["trial_number"]),
                "state": str(item["state"]),
            }
            for item in starting_trial_universe
        ],
        "ending_trial_universe": None,
        "event_signature_prefix_count": event_signature_prefix_count,
        "event_signature_end_count": None,
        "report_search_identity_sha256": None,
        "study_summary_identity_sha256": None,
        "metric_prediction_evidence_identity_sha256": None,
        "pre_terminal_manifest_identity_sha256": None,
        "completed_at_utc": None,
    }
    identity = _epoch_payload_identity(payload)
    payload["epoch_payload_identity_sha256"] = identity
    return {
        "epoch_status": "open",
        "payload": payload,
        "payload_identity_sha256": hashlib.sha256(
            canonical_json_bytes(payload)
        ).hexdigest(),
    }


def _epoch_payload_identity(payload: Mapping[str, Any]) -> str:
    body = {
        name: payload[name]
        for name in sorted(EPOCH_PAYLOAD_KEYS - {"epoch_payload_identity_sha256"})
    }
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _validate_epoch_history(
    epochs: Sequence[Mapping[str, Any]],
    *,
    events: Sequence[Mapping[str, Any]],
    initial_statement: Mapping[str, Any],
    key: LifecycleAuthorityKey | None,
    prevalidated_closed: Mapping[int, Mapping[str, Any]] | None = None,
) -> None:
    if not epochs:
        raise OptunaLifecycleAuthorityError(
            "Study lifecycle authority epoch history is missing."
        )
    study_uuid = initial_statement["payload"]["study_uuid"]
    previous_signature: str | None = None
    open_seen = False
    for index, record in enumerate(epochs):
        status = _epoch_status(record)
        if open_seen:
            raise OptunaLifecycleAuthorityError(
                "Lifecycle epoch history has a non-terminal open epoch."
            )
        if status == "open":
            if index != len(epochs) - 1:
                raise OptunaLifecycleAuthorityError(
                    "Lifecycle epoch history has a non-terminal open epoch."
                )
            open_seen = True
            payload = deepcopy(dict(record["payload"]))
            if set(payload) != EPOCH_PAYLOAD_KEYS:
                raise OptunaLifecycleAuthorityError(
                    "Open lifecycle epoch schema differs."
                )
            if (
                record.get("payload_identity_sha256")
                != hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
            ):
                raise OptunaLifecycleAuthorityError(
                    "Open lifecycle epoch payload identity differs."
                )
            if payload["epoch_payload_identity_sha256"] != _epoch_payload_identity(
                payload
            ):
                raise OptunaLifecycleAuthorityError(
                    "Open lifecycle epoch payload identity differs."
                )
            _validate_epoch_payload_types(payload, open_epoch=True)
        else:
            if key is None and not (
                prevalidated_closed and index in prevalidated_closed
            ):
                # Cross-check path already compared the report epoch exactly; still
                # require structural checks without recomputing HMAC when key absent.
                payload = deepcopy(dict(record["payload"]))
                if set(record) != {
                    "payload",
                    "payload_identity_sha256",
                    "signature_sha256",
                }:
                    raise OptunaLifecycleAuthorityError(
                        "Closed lifecycle epoch schema differs."
                    )
            else:
                if key is None:
                    payload = deepcopy(dict(record["payload"]))
                else:
                    payload = _validate_signed_statement(
                        record,
                        key=key,
                        domain=EPOCH_DOMAIN,
                        payload_keys=EPOCH_PAYLOAD_KEYS,
                        label="lifecycle epoch ledger",
                    )
            if payload["epoch_payload_identity_sha256"] != _epoch_payload_identity(
                payload
            ):
                raise OptunaLifecycleAuthorityError(
                    "Lifecycle epoch payload identity differs."
                )
            _validate_epoch_payload_types(payload, open_epoch=False)
            end_count = int(payload["event_signature_end_count"])
            if end_count > len(events):
                raise OptunaLifecycleAuthorityError(
                    "Lifecycle epoch event coverage exceeds signed events."
                )
            close_event = events[end_count - 1]
            if close_event["payload"]["event_type"] != "report_finalized":
                raise OptunaLifecycleAuthorityError(
                    "Lifecycle epoch close event is not report-finalized."
                )
        if (
            payload["study_uuid"] != study_uuid
            or payload["epoch_number"] != index
            or payload["previous_epoch_ledger_signature_sha256"] != previous_signature
        ):
            raise OptunaLifecycleAuthorityError(
                "Lifecycle epoch history sequence is malformed."
            )
        if status == "closed":
            previous_signature = str(record["signature_sha256"])
        else:
            open_sequence = int(payload["epoch_open_event_sequence"])
            if open_sequence != len(events) and open_sequence > len(events):
                raise OptunaLifecycleAuthorityError(
                    "Open lifecycle epoch event sequence is malformed."
                )
            if previous_signature is not None:
                # Open suffix must continue after the previous closed end count.
                prefix = int(payload["event_signature_prefix_count"])
                if prefix > len(events):
                    raise OptunaLifecycleAuthorityError(
                        "Open lifecycle epoch event coverage is malformed."
                    )


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
