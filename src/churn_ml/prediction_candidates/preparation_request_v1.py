"""Immutable AutoGluon candidate preparation request contract."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from src.churn_ml.dataset_registry.api import resolve_dataset_package
from src.churn_ml.prediction_candidates.autogluon_v1 import (
    SOURCE_KIND,
    inventory_autogluon_run,
    resolve_run_dir,
    resolve_selected_models,
)
from src.churn_ml.prediction_candidates.contract_v1 import (
    CandidateConflictError,
    PredictionCandidateError,
    file_sha256,
    resolve_under_repository,
)
from src.churn_ml.research_data import canonical_sha256


SCHEMA_VERSION = "autogluon_candidate_preparation_request_v1"
REQUEST_ROOT_RELATIVE = "artifacts/ui_candidate_preparation_requests"
MANAGED_RUNS_ROOT = "artifacts/autogluon_runs"
MIN_MODELS = 1
MAX_MODELS = 8
SAFE_MODEL_NAME = re.compile(r"^[\w.\-+]+$")


class PreparationRequestError(ValueError):
    """Raised when a preparation request is invalid or conflicts."""

    def __init__(self, message: str, *, reason_code: str = "preparation_request") -> None:
        self.reason_code = reason_code
        super().__init__(message)


@dataclass(frozen=True)
class PreparationRequest:
    request_id: str
    payload: dict[str, Any]
    relative_path: str
    absolute_path: Path

    @property
    def selected_models(self) -> list[str]:
        return list(self.payload["selected_models"])

    @property
    def run_path(self) -> str:
        return str(self.payload["run_path"])


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def normalize_root_relative(value: str | None, *, default: str, role: str) -> str:
    relative = value or default
    if not relative or relative.startswith("/") or PurePosixPath(relative).is_absolute():
        raise PreparationRequestError(
            f"Absolute or empty {role} rejected: {relative!r}",
            reason_code="path_traversal",
        )
    if ".." in PurePosixPath(relative).parts:
        raise PreparationRequestError(
            f"Path traversal rejected in {role}: {relative!r}",
            reason_code="path_traversal",
        )
    return PurePosixPath(relative).as_posix()


def validate_selected_models(models: Sequence[str]) -> list[str]:
    ordered = [str(item) for item in models]
    if len(ordered) < MIN_MODELS:
        raise PreparationRequestError(
            f"At least {MIN_MODELS} model must be selected.",
            reason_code="model_count_low",
        )
    if len(ordered) > MAX_MODELS:
        raise PreparationRequestError(
            f"At most {MAX_MODELS} models can be selected.",
            reason_code="model_count_high",
        )
    if len(set(ordered)) != len(ordered):
        raise PreparationRequestError(
            "Duplicate model names are rejected.",
            reason_code="duplicate_models",
        )
    for name in ordered:
        if (
            not name
            or "/" in name
            or "\\" in name
            or ".." in name
            or not SAFE_MODEL_NAME.fullmatch(name)
        ):
            raise PreparationRequestError(
                f"Invalid model name: {name!r}",
                reason_code="model_name_invalid",
            )
    return ordered


def validate_managed_run_path(run_path: str) -> str:
    relative = normalize_root_relative(run_path, default=run_path, role="run_path")
    if not relative.startswith(f"{MANAGED_RUNS_ROOT}/"):
        raise PreparationRequestError(
            f"Run path must be under {MANAGED_RUNS_ROOT}/: {relative}",
            reason_code="run_root_invalid",
        )
    return relative


def collect_run_identity_hashes(
    run_dir: Path,
) -> dict[str, str | None]:
    mapping = {
        "run_metadata_sha256": run_dir / "run_metadata.json",
        "worker_result_sha256": run_dir / "worker_result.json",
        "resolved_config_sha256": run_dir / "resolved_config.yaml",
        "leaderboard_sha256": run_dir / "inspection" / "leaderboard.csv",
        "success_marker_sha256": run_dir / "_SUCCESS",
    }
    hashes: dict[str, str | None] = {}
    for key, path in mapping.items():
        hashes[key] = file_sha256(path) if path.is_file() else None
    return hashes


def build_preparation_request_payload(
    *,
    repository_root: Path,
    run_path: str,
    selected_models: Sequence[str],
    candidate_root: str | None = None,
) -> dict[str, Any]:
    root = repository_root.resolve()
    relative_run = validate_managed_run_path(run_path)
    models = validate_selected_models(selected_models)
    run_dir = resolve_under_repository(relative_run, root)
    inventory = inventory_autogluon_run(run_dir, repository_root=root)
    if inventory.classification != "complete":
        raise PreparationRequestError(
            f"Run is not preparable (classification={inventory.classification}).",
            reason_code="run_incomplete",
        )
    # Ensure selected models exist on this inventory.
    try:
        resolve_selected_models(inventory, best=False, models=models)
    except PredictionCandidateError as error:
        raise PreparationRequestError(
            str(error),
            reason_code=getattr(error, "reason_code", "unknown_model"),
        ) from error
    identity_hashes = collect_run_identity_hashes(run_dir)
    if identity_hashes["resolved_config_sha256"] is None:
        raise PreparationRequestError(
            "resolved_config.yaml is required for preparation requests.",
            reason_code="resolved_config_missing",
        )
    if identity_hashes["run_metadata_sha256"] is None:
        raise PreparationRequestError(
            "run_metadata.json is required for preparation requests.",
            reason_code="run_metadata_missing",
        )
    if identity_hashes["worker_result_sha256"] is None:
        raise PreparationRequestError(
            "worker_result.json is required for preparation requests.",
            reason_code="worker_result_missing",
        )
    candidate_root_rel = normalize_root_relative(
        candidate_root,
        default="artifacts/prediction_candidates",
        role="candidate_root",
    )
    dataset = resolve_dataset_package(root / "data" / "processed", inventory.dataset_id)
    target_dependency = str(dataset.manifest.target_dependency)
    exploratory = target_dependency == "exploratory"
    identity = {
        "schema_version": SCHEMA_VERSION,
        "source_kind": SOURCE_KIND,
        "run_path": inventory.run_path,
        "run_id": _read_run_id(run_dir),
        "dataset_id": inventory.dataset_id,
        "target_dependency": target_dependency,
        "exploratory": exploratory,
        "selected_models": models,
        "config_path": inventory.config_path,
        "config_sha256": inventory.config_sha256,
        "run_metadata_sha256": identity_hashes["run_metadata_sha256"],
        "worker_result_sha256": identity_hashes["worker_result_sha256"],
        "resolved_config_sha256": identity_hashes["resolved_config_sha256"],
        "leaderboard_sha256": identity_hashes["leaderboard_sha256"],
        "success_marker_sha256": identity_hashes["success_marker_sha256"],
        "candidate_root": candidate_root_rel,
        "classification": inventory.classification,
        "autogluon_version": inventory.autogluon_version,
        "best_model": inventory.best_model,
    }
    request_id = f"apr1_{canonical_sha256(identity)[:16]}"
    return {
        **identity,
        "request_id": request_id,
        "selection_fingerprint": canonical_sha256(
            {
                "run_path": inventory.run_path,
                "selected_models": models,
                "config_sha256": inventory.config_sha256,
            }
        ),
        "settings_fingerprint": canonical_sha256(identity),
    }


def materialize_preparation_request(
    *,
    repository_root: Path,
    run_path: str,
    selected_models: Sequence[str],
    candidate_root: str | None = None,
) -> PreparationRequest:
    root = repository_root.resolve()
    payload = build_preparation_request_payload(
        repository_root=root,
        run_path=run_path,
        selected_models=selected_models,
        candidate_root=candidate_root,
    )
    request_id = str(payload["request_id"])
    relative = f"{REQUEST_ROOT_RELATIVE}/{request_id}.json"
    absolute = resolve_under_repository(relative, root)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    body = {**payload, "created_at_utc": utc_now()}
    raw = (json.dumps(body, indent=2, sort_keys=True, default=str) + "\n").encode(
        "utf-8"
    )
    if absolute.is_file():
        existing = json.loads(absolute.read_text(encoding="utf-8"))
        if _identity_view(existing) != _identity_view(body):
            raise CandidateConflictError(
                f"Preparation request {request_id} already exists with different content."
            )
        return PreparationRequest(
            request_id=request_id,
            payload=dict(existing),
            relative_path=relative,
            absolute_path=absolute,
        )
    _atomic_write_bytes(absolute, raw)
    return PreparationRequest(
        request_id=request_id,
        payload=body,
        relative_path=relative,
        absolute_path=absolute,
    )


def load_preparation_request(
    relative_or_id: str,
    *,
    repository_root: Path,
) -> PreparationRequest:
    root = repository_root.resolve()
    text = str(relative_or_id).replace("\\", "/")
    if text.endswith(".json"):
        relative = normalize_root_relative(text, default=text, role="request_path")
    else:
        if ".." in text or "/" in text or "\\" in text:
            raise PreparationRequestError(
                f"Invalid request_id: {text!r}",
                reason_code="path_traversal",
            )
        relative = f"{REQUEST_ROOT_RELATIVE}/{text}.json"
    absolute = resolve_under_repository(relative, root)
    if not absolute.is_file():
        raise PreparationRequestError(
            f"Preparation request not found: {relative}",
            reason_code="request_missing",
        )
    payload = json.loads(absolute.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PreparationRequestError(
            "Request payload must be a JSON object.",
            reason_code="request_schema_invalid",
        )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise PreparationRequestError(
            f"Unsupported request schema: {payload.get('schema_version')!r}",
            reason_code="request_schema_invalid",
        )
    validate_selected_models(payload.get("selected_models") or [])
    validate_managed_run_path(str(payload.get("run_path") or ""))
    request_id = str(payload["request_id"])
    expected_relative = f"{REQUEST_ROOT_RELATIVE}/{request_id}.json"
    if relative != expected_relative:
        raise PreparationRequestError(
            "Request path does not match request_id.",
            reason_code="request_path_mismatch",
        )
    return PreparationRequest(
        request_id=request_id,
        payload=payload,
        relative_path=relative,
        absolute_path=absolute,
    )


def verify_request_run_identity(
    request: PreparationRequest,
    *,
    repository_root: Path,
) -> None:
    """Reject stale requests when the managed run identity changed."""
    root = repository_root.resolve()
    run_dir = resolve_under_repository(request.run_path, root)
    current = collect_run_identity_hashes(run_dir)
    for key in (
        "run_metadata_sha256",
        "worker_result_sha256",
        "resolved_config_sha256",
        "leaderboard_sha256",
        "success_marker_sha256",
    ):
        expected = request.payload.get(key)
        actual = current.get(key)
        if expected != actual:
            raise PreparationRequestError(
                f"Run identity changed after request creation ({key}).",
                reason_code="request_identity_stale",
            )
    inventory = inventory_autogluon_run(run_dir, repository_root=root)
    if inventory.config_sha256 != request.payload.get("config_sha256"):
        raise PreparationRequestError(
            "Run config identity no longer matches the preparation request.",
            reason_code="request_identity_stale",
        )
    if inventory.dataset_id != request.payload.get("dataset_id"):
        raise PreparationRequestError(
            "Run dataset identity no longer matches the preparation request.",
            reason_code="request_identity_stale",
        )
    resolve_selected_models(
        inventory, best=False, models=request.selected_models
    )


def resolve_request_selection(
    request: PreparationRequest,
    *,
    repository_root: Path,
) -> tuple[Path, list[str]]:
    verify_request_run_identity(request, repository_root=repository_root)
    run_dir = resolve_run_dir(Path(request.run_path), repository_root)
    return run_dir, list(request.selected_models)


def _read_run_id(run_dir: Path) -> str | None:
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        value = payload.get("run_id")
        return str(value) if value is not None else None
    return None


def _identity_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    ignored = {"created_at_utc"}
    return {key: payload[key] for key in sorted(payload) if key not in ignored}


def _atomic_write_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


__all__ = [
    "MANAGED_RUNS_ROOT",
    "MAX_MODELS",
    "MIN_MODELS",
    "REQUEST_ROOT_RELATIVE",
    "SCHEMA_VERSION",
    "PreparationRequest",
    "PreparationRequestError",
    "build_preparation_request_payload",
    "collect_run_identity_hashes",
    "load_preparation_request",
    "materialize_preparation_request",
    "resolve_request_selection",
    "validate_managed_run_path",
    "validate_selected_models",
    "verify_request_run_identity",
]
