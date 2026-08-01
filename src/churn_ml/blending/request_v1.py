"""Immutable UI blend request contract (blend_ui_request_v1).

Materializes a validated, repository-relative request file so the Control Panel
can launch search/materialize jobs with a single path placeholder while keeping
dynamic 2–10 candidate selections out of shell-string construction.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from src.churn_ml.blending.evaluation_v1 import (
    DEFAULT_OPTUNA_TRIALS,
    BlendSettings,
)
from src.churn_ml.prediction_candidates.contract_v1 import (
    CandidateConflictError,
    file_sha256,
    resolve_under_repository,
)
from src.churn_ml.research_data import canonical_sha256


SCHEMA_VERSION = "blend_ui_request_v1"
REQUEST_ROOT_RELATIVE = "artifacts/ui_blend_requests"
MIN_CANDIDATES = 2
MAX_CANDIDATES = 10
SAFE_CANDIDATE_ID = re.compile(r"^pc1_[0-9a-f]{16}$")
METHODS = (
    "equal",
    "manual",
    "optimized_native",
    "optimized_optuna",
)


class BlendUIRequestError(ValueError):
    """Raised when a blend UI request is invalid or conflicts."""

    def __init__(self, message: str, *, reason_code: str = "blend_ui_request") -> None:
        self.reason_code = reason_code
        super().__init__(message)


@dataclass(frozen=True)
class BlendUIRequest:
    request_id: str
    payload: dict[str, Any]
    relative_path: str
    absolute_path: Path

    @property
    def candidate_ids(self) -> list[str]:
        return list(self.payload["candidate_ids"])

    def blend_settings(self) -> BlendSettings:
        return settings_from_request_payload(self.payload)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def method_to_strategy_optimizer(method: str) -> tuple[str, str]:
    if method == "equal":
        return "equal", "native"
    if method == "manual":
        return "manual", "native"
    if method == "optimized_native":
        return "optimized", "native"
    if method == "optimized_optuna":
        return "optimized", "optuna"
    raise BlendUIRequestError(
        f"Unsupported blend method {method!r}.",
        reason_code="method_invalid",
    )


def strategy_optimizer_to_method(strategy: str, optimizer: str) -> str:
    if strategy == "equal":
        return "equal"
    if strategy == "manual":
        return "manual"
    if strategy == "optimized" and optimizer == "native":
        return "optimized_native"
    if strategy == "optimized" and optimizer == "optuna":
        return "optimized_optuna"
    raise BlendUIRequestError(
        f"Unsupported strategy/optimizer pair ({strategy!r}, {optimizer!r}).",
        reason_code="strategy_optimizer_invalid",
    )


def normalize_root_relative(value: str | None, *, default: str, role: str) -> str:
    relative = value or default
    if not relative or relative.startswith("/") or PurePosixPath(relative).is_absolute():
        raise BlendUIRequestError(
            f"Absolute or empty {role} rejected: {relative!r}",
            reason_code="path_traversal",
        )
    if ".." in PurePosixPath(relative).parts:
        raise BlendUIRequestError(
            f"Path traversal rejected in {role}: {relative!r}",
            reason_code="path_traversal",
        )
    return PurePosixPath(relative).as_posix()


def validate_candidate_ids(candidate_ids: Sequence[str]) -> list[str]:
    ordered = [str(item) for item in candidate_ids]
    if len(ordered) < MIN_CANDIDATES:
        raise BlendUIRequestError(
            f"At least {MIN_CANDIDATES} candidates are required.",
            reason_code="candidate_count_low",
        )
    if len(ordered) > MAX_CANDIDATES:
        raise BlendUIRequestError(
            f"At most {MAX_CANDIDATES} candidates are allowed.",
            reason_code="candidate_count_high",
        )
    if len(set(ordered)) != len(ordered):
        raise BlendUIRequestError(
            "Duplicate candidate IDs are rejected.",
            reason_code="duplicate_candidates",
        )
    for candidate_id in ordered:
        if (
            ".." in candidate_id
            or "/" in candidate_id
            or "\\" in candidate_id
            or not SAFE_CANDIDATE_ID.match(candidate_id)
        ):
            raise BlendUIRequestError(
                f"Invalid candidate_id: {candidate_id!r}",
                reason_code="candidate_id_invalid",
            )
    return ordered


def build_request_payload(
    *,
    candidate_ids: Sequence[str],
    method: str,
    folds: int = 5,
    repeats: int = 2,
    seed: int = 42,
    max_active_models: int | None = None,
    manual_weights: Mapping[str, float] | Sequence[str] | None = None,
    dirichlet_draws: int = 32,
    pairwise_grid_step: float = 0.1,
    optuna_trials: int = DEFAULT_OPTUNA_TRIALS,
    optuna_timeout_seconds: float | None = None,
    optuna_seed: int | None = None,
    candidate_root: str | None = None,
    blend_root: str | None = None,
) -> dict[str, Any]:
    ordered = validate_candidate_ids(candidate_ids)
    strategy, optimizer = method_to_strategy_optimizer(method)
    weight_specs = _normalize_manual_weights(manual_weights, ordered, strategy=strategy)
    settings = BlendSettings(
        strategy=strategy,
        folds=int(folds),
        repeats=int(repeats),
        seed=int(seed),
        max_active_models=max_active_models,
        manual_weights=tuple(weight_specs),
        pairwise_grid_step=float(pairwise_grid_step),
        dirichlet_draws=int(dirichlet_draws),
        optimizer_backend=optimizer,
        optuna_trials=int(optuna_trials),
        optuna_timeout_seconds=optuna_timeout_seconds,
        optuna_seed=optuna_seed,
        optuna_options_provided=optimizer == "optuna",
        native_search_options_provided=(
            optimizer == "native" and strategy == "optimized"
        ),
    ).normalized()
    candidate_root_rel = normalize_root_relative(
        candidate_root,
        default="artifacts/prediction_candidates",
        role="candidate_root",
    )
    blend_root_rel = normalize_root_relative(
        blend_root,
        default="artifacts/prediction_blends",
        role="blend_root",
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "candidate_ids": ordered,
        "strategy": settings.strategy,
        "optimizer_backend": settings.optimizer_backend,
        "folds": settings.folds,
        "repeats": settings.repeats,
        "seed": settings.seed,
        "max_active_models": settings.max_active_models,
        "manual_weights": list(settings.manual_weights),
        "dirichlet_draws": (
            settings.dirichlet_draws if settings.optimizer_backend == "native" else None
        ),
        "pairwise_grid_step": (
            settings.pairwise_grid_step
            if settings.optimizer_backend == "native"
            else None
        ),
        "optuna_trials": (
            settings.optuna_trials if settings.optimizer_backend == "optuna" else None
        ),
        "optuna_timeout_seconds": (
            settings.optuna_timeout_seconds
            if settings.optimizer_backend == "optuna"
            else None
        ),
        "optuna_seed": (
            settings.optuna_seed if settings.optimizer_backend == "optuna" else None
        ),
        "candidate_root": candidate_root_rel,
        "blend_root": blend_root_rel,
    }
    request_id = f"br1_{canonical_sha256(identity)[:16]}"
    return {
        **identity,
        "request_id": request_id,
        "method": method,
        "selection_fingerprint": canonical_sha256(
            {"candidate_ids": ordered, "candidate_root": candidate_root_rel}
        ),
        "settings_fingerprint": canonical_sha256(identity),
    }


def settings_from_request_payload(payload: Mapping[str, Any]) -> BlendSettings:
    strategy = str(payload["strategy"])
    optimizer = str(payload["optimizer_backend"])
    return BlendSettings(
        strategy=strategy,
        folds=int(payload["folds"]),
        repeats=int(payload["repeats"]),
        seed=int(payload["seed"]),
        max_active_models=payload.get("max_active_models"),
        manual_weights=tuple(payload.get("manual_weights") or ()),
        pairwise_grid_step=float(payload.get("pairwise_grid_step") or 0.1),
        dirichlet_draws=int(payload.get("dirichlet_draws") or 32),
        optimizer_backend=optimizer,
        optuna_trials=int(payload.get("optuna_trials") or DEFAULT_OPTUNA_TRIALS),
        optuna_timeout_seconds=payload.get("optuna_timeout_seconds"),
        optuna_seed=payload.get("optuna_seed"),
        optuna_options_provided=optimizer == "optuna",
        native_search_options_provided=(
            optimizer == "native" and strategy == "optimized"
        ),
    ).normalized()


def materialize_blend_ui_request(
    *,
    repository_root: Path,
    candidate_ids: Sequence[str],
    method: str,
    folds: int = 5,
    repeats: int = 2,
    seed: int = 42,
    max_active_models: int | None = None,
    manual_weights: Mapping[str, float] | Sequence[str] | None = None,
    dirichlet_draws: int = 32,
    pairwise_grid_step: float = 0.1,
    optuna_trials: int = DEFAULT_OPTUNA_TRIALS,
    optuna_timeout_seconds: float | None = None,
    optuna_seed: int | None = None,
    candidate_root: str | None = None,
    blend_root: str | None = None,
) -> BlendUIRequest:
    root = repository_root.resolve()
    payload = build_request_payload(
        candidate_ids=candidate_ids,
        method=method,
        folds=folds,
        repeats=repeats,
        seed=seed,
        max_active_models=max_active_models,
        manual_weights=manual_weights,
        dirichlet_draws=dirichlet_draws,
        pairwise_grid_step=pairwise_grid_step,
        optuna_trials=optuna_trials,
        optuna_timeout_seconds=optuna_timeout_seconds,
        optuna_seed=optuna_seed,
        candidate_root=candidate_root,
        blend_root=blend_root,
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
                f"Blend UI request {request_id} already exists with different content."
            )
        return BlendUIRequest(
            request_id=request_id,
            payload=dict(existing),
            relative_path=relative,
            absolute_path=absolute,
        )
    _atomic_write_bytes(absolute, raw)
    return BlendUIRequest(
        request_id=request_id,
        payload=body,
        relative_path=relative,
        absolute_path=absolute,
    )


def load_blend_ui_request(
    relative_or_id: str,
    *,
    repository_root: Path,
) -> BlendUIRequest:
    root = repository_root.resolve()
    text = str(relative_or_id).replace("\\", "/")
    if text.endswith(".json"):
        relative = normalize_root_relative(text, default=text, role="request_path")
    else:
        if ".." in text or "/" in text or "\\" in text:
            raise BlendUIRequestError(
                f"Invalid request_id: {text!r}",
                reason_code="path_traversal",
            )
        relative = f"{REQUEST_ROOT_RELATIVE}/{text}.json"
    absolute = resolve_under_repository(relative, root)
    if not absolute.is_file():
        raise BlendUIRequestError(
            f"Blend UI request not found: {relative}",
            reason_code="request_missing",
        )
    payload = json.loads(absolute.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BlendUIRequestError(
            "Request payload must be a JSON object.",
            reason_code="request_schema_invalid",
        )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise BlendUIRequestError(
            f"Unsupported request schema: {payload.get('schema_version')!r}",
            reason_code="request_schema_invalid",
        )
    validate_candidate_ids(payload.get("candidate_ids") or [])
    # Re-validate settings through BlendSettings.
    settings_from_request_payload(payload)
    request_id = str(payload["request_id"])
    expected_relative = f"{REQUEST_ROOT_RELATIVE}/{request_id}.json"
    if relative != expected_relative:
        raise BlendUIRequestError(
            "Request path does not match request_id.",
            reason_code="request_path_mismatch",
        )
    return BlendUIRequest(
        request_id=request_id,
        payload=payload,
        relative_path=relative,
        absolute_path=absolute,
    )


def _normalize_manual_weights(
    manual_weights: Mapping[str, float] | Sequence[str] | None,
    candidate_ids: Sequence[str],
    *,
    strategy: str,
) -> list[str]:
    if strategy != "manual":
        return []
    if manual_weights is None:
        raise BlendUIRequestError(
            "Manual strategy requires weights.",
            reason_code="manual_weights_missing",
        )
    if isinstance(manual_weights, Mapping):
        specs = [f"{candidate_id}={manual_weights[candidate_id]}" for candidate_id in candidate_ids]
        missing = [item for item in candidate_ids if item not in manual_weights]
        if missing:
            raise BlendUIRequestError(
                f"Missing manual weights for: {missing}",
                reason_code="manual_weights_missing",
            )
        return specs
    return [str(item) for item in manual_weights]


def _identity_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "request_id",
        "candidate_ids",
        "strategy",
        "optimizer_backend",
        "folds",
        "repeats",
        "seed",
        "max_active_models",
        "manual_weights",
        "dirichlet_draws",
        "pairwise_grid_step",
        "optuna_trials",
        "optuna_timeout_seconds",
        "optuna_seed",
        "candidate_root",
        "blend_root",
        "selection_fingerprint",
        "settings_fingerprint",
        "method",
    )
    return {key: payload.get(key) for key in keys}


def _atomic_write_bytes(path: Path, raw: bytes) -> None:
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    replaced = False
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        replaced = True
    finally:
        if not replaced:
            temporary_path.unlink(missing_ok=True)


def request_file_sha256(request: BlendUIRequest) -> str:
    return file_sha256(request.absolute_path)


__all__ = [
    "METHODS",
    "MIN_CANDIDATES",
    "MAX_CANDIDATES",
    "REQUEST_ROOT_RELATIVE",
    "SCHEMA_VERSION",
    "BlendUIRequest",
    "BlendUIRequestError",
    "build_request_payload",
    "load_blend_ui_request",
    "materialize_blend_ui_request",
    "method_to_strategy_optimizer",
    "request_file_sha256",
    "settings_from_request_payload",
    "strategy_optimizer_to_method",
    "validate_candidate_ids",
]
