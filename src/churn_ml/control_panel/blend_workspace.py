"""Blend Workspace helpers for the Control Panel.

Pure presentation, discovery, fingerprinting, and job-recovery helpers. No
Streamlit imports and no blending mathematics — all strict behavior delegates
to the canonical prediction-candidate and blending backends.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.churn_ml.blending.artifact_v1 import (
    BlendArtifactError,
    build_blend_id,
    load_blend_artifact,
    normalize_blend_root_relative,
)
from src.churn_ml.blending.compatibility_v1 import (
    BlendCompatibilityError,
    list_candidates,
    load_compatible_candidates,
    resolve_candidates_root,
)
from src.churn_ml.blending.diversity_v1 import analyze_diversity
from src.churn_ml.control_panel.blend_ui_request_v1 import (
    BlendUIRequest,
    materialize_blend_ui_request,
    validate_candidate_ids,
)
from src.churn_ml.control_panel.candidate_display import (
    resolve_candidate_display,
)
from src.churn_ml.control_panel.command_builder import SAFE_STRING
from src.churn_ml.prediction_candidates.contract_v1 import (
    MANIFEST_FILENAME,
    SUCCESS_FILENAME,
    PredictionCandidateError,
    file_sha256,
    load_candidate_package,
    resolve_under_repository,
)
from src.churn_ml.prediction_candidates.submission_v1 import (
    MANIFEST_FILENAME as SUBMISSION_MANIFEST_FILENAME,
    CandidateSubmissionError,
    evaluate_submission_readiness,
    extract_final_deployment_threshold,
    normalize_submission_root_relative,
)
from src.churn_ml.research_data import canonical_sha256


CACHE_CONTRACT_VERSION = "blend_workspace_candidate_cache_v1"
CANONICAL_SUBMISSION_HANDOFF_KEY = "canonical_submission_handoff"

_SOURCE_KIND_LABELS = {
    "autogluon_standalone_v1": "AutoGluon",
    "canonical_probability_blend_v1": "Canonical blend",
}

_METHOD_LABELS = {
    "equal": "Equal weights",
    "manual": "Manual weights",
    "optimized_native": "Optimized — Native",
    "optimized_optuna": "Optimized — Optuna",
}

_READINESS_LABELS = {
    "ready": "Ready",
    "blocked": "Blocked",
}

_SETTINGS_FINGERPRINT_KEYS = (
    "schema_version",
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
)

_SLUG_CLEAN = re.compile(r"[^A-Za-z0-9._-]+")


class BlendWorkspaceError(ValueError):
    """Raised when Blend Workspace helper validation fails."""

    def __init__(self, message: str, *, reason_code: str = "blend_workspace") -> None:
        self.reason_code = reason_code
        super().__init__(message)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def source_kind_label(source_kind: str) -> str:
    return _SOURCE_KIND_LABELS.get(source_kind, source_kind.replace("_", " ").title())


def optimizer_label(strategy: str, optimizer: str) -> str:
    if strategy == "equal":
        return _METHOD_LABELS["equal"]
    if strategy == "manual":
        return _METHOD_LABELS["manual"]
    if strategy == "optimized" and optimizer == "native":
        return _METHOD_LABELS["optimized_native"]
    if strategy == "optimized" and optimizer == "optuna":
        return _METHOD_LABELS["optimized_optuna"]
    return f"{strategy} / {optimizer}"


def method_label(method: str) -> str:
    return _METHOD_LABELS.get(method, method.replace("_", " ").title())


def readiness_label(state: str) -> str:
    return _READINESS_LABELS.get(state, state.replace("_", " ").title())


def candidate_display_label(summary: Mapping[str, Any]) -> str:
    return resolve_candidate_display(summary).primary_label


def blend_display_label(
    manifest_or_summary: Mapping[str, Any],
    *,
    repository_root: Path | str | None = None,
) -> str:
    display = resolve_candidate_display(
        {
            "candidate_id": manifest_or_summary.get("canonical_candidate_id")
            or manifest_or_summary.get("candidate_id")
            or "",
            "source_model_name": manifest_or_summary.get("source_model_name")
            or (
                f"blend_"
                f"{manifest_or_summary.get('strategy') or (manifest_or_summary.get('settings') or {}).get('strategy') or 'optimized'}_"
                f"{manifest_or_summary.get('optimizer_backend') or (manifest_or_summary.get('settings') or {}).get('optimizer_backend') or 'native'}"
            ),
            "source_kind": manifest_or_summary.get("source_kind")
            or "canonical_probability_blend_v1",
            "dataset_id": manifest_or_summary.get("dataset_id") or "",
            "candidate_ids": manifest_or_summary.get("candidate_ids")
            or manifest_or_summary.get("parent_candidate_ids")
            or [],
            "final_deployment_weights": manifest_or_summary.get(
                "final_deployment_weights"
            ),
            "settings": manifest_or_summary.get("settings") or {},
            "strategy": manifest_or_summary.get("strategy"),
            "optimizer_backend": manifest_or_summary.get("optimizer_backend"),
            "exploratory": manifest_or_summary.get("exploratory"),
        },
        repository_root=repository_root,
    )
    return display.primary_label


def candidate_inventory_fingerprint(
    repository_root: Path | str,
    candidates_root_relative: str | None = None,
) -> str:
    root = Path(repository_root).resolve()
    summaries = list_candidates(
        repository_root=root,
        candidates_root=candidates_root_relative,
    )
    relative_root, absolute_root = resolve_candidates_root(root, candidates_root_relative)
    entries: list[tuple[str, str, str]] = []
    for summary in summaries:
        candidate_id = str(summary["candidate_id"])
        package_dir = absolute_root / candidate_id
        manifest_path = package_dir / MANIFEST_FILENAME
        success_path = package_dir / SUCCESS_FILENAME
        if not manifest_path.is_file() or not success_path.is_file():
            continue
        entries.append(
            (
                candidate_id,
                file_sha256(manifest_path),
                file_sha256(success_path),
            )
        )
    entries.sort()
    return canonical_sha256(
        [
            {"candidate_id": item[0], "manifest_sha256": item[1], "success_sha256": item[2]}
            for item in entries
        ]
    )


def discover_candidate_rows(
    repository_root: Path | str,
    *,
    candidates_root_relative: str | None = None,
    include_readiness: bool = True,
) -> list[dict[str, Any]]:
    root = Path(repository_root).resolve()
    relative_root, absolute_root = resolve_candidates_root(root, candidates_root_relative)
    if not absolute_root.is_dir():
        return []
    # Canonical discover skips unreadable packages; the UI still surfaces them
    # as invalid so operators can see tampered/incomplete inventory entries.
    summaries_by_id = {
        str(item["candidate_id"]): item
        for item in list_candidates(
            repository_root=root,
            candidates_root=candidates_root_relative,
        )
    }
    directory_ids: list[str] = []
    for child in sorted(absolute_root.iterdir(), key=lambda item: item.name):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if not (child / SUCCESS_FILENAME).is_file() and not (
            child / MANIFEST_FILENAME
        ).is_file():
            continue
        directory_ids.append(child.name)

    rows: list[dict[str, Any]] = []
    for candidate_id in directory_ids:
        summary = summaries_by_id.get(candidate_id) or {
            "candidate_id": candidate_id,
            "source_model_name": candidate_id,
            "source_kind": None,
            "dataset_id": None,
        }
        package_dir = absolute_root / candidate_id
        manifest_path = package_dir / MANIFEST_FILENAME
        success_path = package_dir / SUCCESS_FILENAME
        manifest_sha256 = (
            file_sha256(manifest_path) if manifest_path.is_file() else None
        )
        success_sha256 = file_sha256(success_path) if success_path.is_file() else None
        package_status = "completed" if success_path.is_file() else "incomplete"
        readiness_state: str | None = None
        readiness_blockers: list[dict[str, str]] = []
        final_threshold: float | None = None
        parent_dataset_id = summary.get("parent_dataset_id")
        target_dependency = summary.get("target_dependency")
        exploratory = summary.get("exploratory")
        probability_min: float | None = None
        probability_max: float | None = None
        loaded_package = None
        try:
            loaded_package = load_candidate_package(
                package_dir,
                repository_root=root,
                candidates_root_relative=relative_root,
            )
            manifest = loaded_package.manifest
            parent_dataset_id = manifest.get("parent_dataset_id", parent_dataset_id)
            target_dependency = manifest.get("target_dependency", target_dependency)
            exploratory = bool(manifest.get("exploratory", exploratory))
            probs = loaded_package.test["probability_positive"].to_numpy(dtype=np.float64)
            if probs.size:
                probability_min = float(np.min(probs))
                probability_max = float(np.max(probs))
            threshold = extract_final_deployment_threshold(
                loaded_package.source_metadata
            )
            if threshold is not None and np.isfinite(threshold):
                final_threshold = float(threshold)
            if include_readiness:
                readiness = evaluate_submission_readiness(
                    candidate_id,
                    repository_root=root,
                    candidates_root_relative=relative_root,
                )
                readiness_state = str(readiness["state"])
                readiness_blockers = list(readiness.get("blockers") or [])
                if final_threshold is None and readiness.get("threshold") is not None:
                    final_threshold = float(readiness["threshold"])
        except (PredictionCandidateError, OSError, json.JSONDecodeError, ValueError):
            package_status = "invalid"
            loaded_package = None
            if include_readiness:
                readiness_state = "blocked"
                readiness_blockers = [
                    {
                        "code": "candidate_invalid",
                        "message": "Candidate package failed strict validation.",
                    }
                ]
        if loaded_package is not None:
            label_payload: dict[str, Any] = {
                "candidate_id": candidate_id,
                "source_model_name": loaded_package.manifest.get(
                    "source_model_name", summary.get("source_model_name")
                ),
                "source_kind": loaded_package.manifest.get(
                    "source_kind", summary.get("source_kind")
                ),
                "dataset_id": loaded_package.manifest.get(
                    "dataset_id", summary.get("dataset_id")
                ),
                "source_metadata": loaded_package.source_metadata,
                "exploratory": loaded_package.manifest.get("exploratory"),
            }
        else:
            label_payload = {
                "candidate_id": candidate_id,
                "source_model_name": summary.get("source_model_name"),
                "source_kind": summary.get("source_kind"),
                "dataset_id": summary.get("dataset_id"),
            }
        display = resolve_candidate_display(label_payload, repository_root=root)
        label = display.primary_label
        rows.append(
            {
                "candidate_id": candidate_id,
                "label": label,
                "compact_label": display.compact_label,
                "source_model_name": summary.get("source_model_name"),
                "source_kind": summary.get("source_kind"),
                "source_kind_label": source_kind_label(
                    str(summary.get("source_kind") or "")
                ),
                "dataset_id": summary.get("dataset_id"),
                "dataset_label": display.dataset_label,
                "parent_dataset_id": parent_dataset_id,
                "target_dependency": target_dependency,
                "exploratory": exploratory,
                "source_metric_name": summary.get("source_metric_name"),
                "source_metric_value": summary.get("source_metric_value"),
                "oof_protocol": summary.get("oof_protocol"),
                "readiness_state": readiness_state,
                "readiness_blockers": readiness_blockers,
                "final_threshold": final_threshold,
                "package_status": package_status,
                "manifest_sha256": manifest_sha256,
                "success_sha256": success_sha256,
                "train_row_count": summary.get("train_row_count"),
                "test_row_count": summary.get("test_row_count"),
                "probability_min": probability_min,
                "probability_max": probability_max,
            }
        )
    return rows


def filter_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: str | None = None,
    dataset: str | None = None,
    model_query: str | None = None,
    exploratory: bool | None = None,
    readiness: str | None = None,
    search_text: str | None = None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    query = (search_text or "").strip().casefold()
    model_needle = (model_query or "").strip().casefold()
    for row in rows:
        item = dict(row)
        if source is not None and str(item.get("source_kind") or "") != source:
            continue
        if dataset is not None and str(item.get("dataset_id") or "") != dataset:
            continue
        if model_needle:
            haystack = str(item.get("source_model_name") or "").casefold()
            if model_needle not in haystack:
                continue
        if exploratory is not None and bool(item.get("exploratory")) != exploratory:
            continue
        if readiness is not None and str(item.get("readiness_state") or "") != readiness:
            continue
        if query:
            searchable = " ".join(
                str(item.get(key) or "")
                for key in (
                    "label",
                    "candidate_id",
                    "source_model_name",
                    "dataset_id",
                    "source_kind_label",
                )
            ).casefold()
            if query not in searchable:
                continue
        filtered.append(item)
    return filtered


def selection_fingerprint(candidate_ids: Sequence[str]) -> str:
    ordered = [str(item) for item in candidate_ids]
    return canonical_sha256({"candidate_ids": ordered})


def settings_fingerprint_from_request_payload(payload: Mapping[str, Any]) -> str:
    identity = {key: payload.get(key) for key in _SETTINGS_FINGERPRINT_KEYS}
    return canonical_sha256(identity)


def validate_selection_count(candidate_ids: Sequence[str]) -> None:
    validate_candidate_ids(candidate_ids)


def run_compatibility(
    candidate_ids: Sequence[str],
    repository_root: Path | str,
    candidates_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    try:
        pool = load_compatible_candidates(
            candidate_ids,
            repository_root=root,
            candidates_root=candidates_root,
        )
        shared = pool.compatibility["shared_identity"]
        summary = {
            "status": "compatible",
            "status_label": "Compatible",
            "candidate_count": pool.n_candidates,
            "candidate_ids": list(pool.candidate_ids),
            "train_row_count": shared["train_row_count"],
            "test_row_count": shared["test_row_count"],
            "target_hash": shared["target_hash"],
            "train_anchor_hash": shared["train_anchor_hash"],
            "test_anchor_hash": shared["test_anchor_hash"],
            "positive_class_label": shared["positive_class_label"],
            "probability_semantics": shared["probability_semantics"],
            "exploratory": pool.exploratory,
            "allowed_differences": list(pool.compatibility["allowed_to_differ"]),
        }
        return {"ok": True, "summary": summary, "raw": pool.compatibility}
    except BlendCompatibilityError as error:
        summary = {
            "status": "blocked",
            "status_label": "Blocked",
            "reason": str(error),
            "reason_code": error.reason_code,
            "candidate_count": len(candidate_ids),
        }
        return {"ok": False, "summary": summary, "raw": None}


def run_diversity(
    candidate_ids: Sequence[str],
    repository_root: Path | str,
    candidates_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    pool = load_compatible_candidates(
        candidate_ids,
        repository_root=root,
        candidates_root=candidates_root,
    )
    diversity = analyze_diversity(pool)
    matrices = {
        name: frame.reset_index(names="candidate_id").to_dict(orient="records")
        for name, frame in diversity["matrices"].items()
    }
    return {
        "ok": True,
        "candidate_metrics": diversity["candidate_metrics"],
        "pairwise_analysis": diversity["pairwise_analysis"].to_dict(orient="records"),
        "diversity_matrix": diversity["diversity_matrix"]
        .reset_index(names="candidate_id")
        .to_dict(orient="records"),
        "matrices": matrices,
        "notes": diversity["notes"],
        "raw": {
            "compatibility": pool.compatibility,
            "candidate_metrics": diversity["candidate_metrics"],
            "pairwise_analysis": diversity["pairwise_analysis"].to_dict(
                orient="records"
            ),
            "notes": diversity["notes"],
        },
    }


def prepare_request(
    *,
    repository_root: Path | str,
    candidate_ids: Sequence[str],
    method: str,
    folds: int = 5,
    repeats: int = 2,
    seed: int = 42,
    max_active_models: int | None = None,
    manual_weights: Mapping[str, float] | Sequence[str] | None = None,
    dirichlet_draws: int = 32,
    pairwise_grid_step: float = 0.1,
    optuna_trials: int | None = None,
    optuna_timeout_seconds: float | None = None,
    optuna_seed: int | None = None,
    candidate_root: str | None = None,
    blend_root: str | None = None,
) -> BlendUIRequest:
    kwargs: dict[str, Any] = {
        "repository_root": Path(repository_root).resolve(),
        "candidate_ids": candidate_ids,
        "method": method,
        "folds": folds,
        "repeats": repeats,
        "seed": seed,
        "max_active_models": max_active_models,
        "manual_weights": manual_weights,
        "dirichlet_draws": dirichlet_draws,
        "pairwise_grid_step": pairwise_grid_step,
        "optuna_timeout_seconds": optuna_timeout_seconds,
        "optuna_seed": optuna_seed,
        "candidate_root": candidate_root,
        "blend_root": blend_root,
    }
    if optuna_trials is not None:
        kwargs["optuna_trials"] = optuna_trials
    return materialize_blend_ui_request(**kwargs)


def expected_blend_id(request: BlendUIRequest, repository_root: Path | str) -> str:
    root = Path(repository_root).resolve()
    pool = load_compatible_candidates(
        request.candidate_ids,
        repository_root=root,
        candidates_root=request.payload.get("candidate_root"),
    )
    return build_blend_id(pool, request.blend_settings())


def parse_job_stdout_json(stdout_text: str) -> dict[str, Any]:
    text = stdout_text.strip()
    if not text:
        raise BlendWorkspaceError(
            "Job stdout is empty.",
            reason_code="stdout_empty",
        )
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    last_payload: dict[str, Any] | None = None
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        try:
            payload, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            index += 1
            continue
        if isinstance(payload, dict):
            last_payload = payload
        index = end
    if last_payload is None:
        raise BlendWorkspaceError(
            "No complete JSON object found in job stdout.",
            reason_code="stdout_json_invalid",
        )
    return last_payload


def verify_search_result(
    payload: Mapping[str, Any],
    *,
    request_id: str,
    expected_blend_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise BlendWorkspaceError(
            "Search result payload must be a mapping.",
            reason_code="search_result_invalid",
        )
    if payload.get("ok") is not True:
        message = str(payload.get("error") or "Blend search failed.")
        reason_code = str(payload.get("reason_code") or "search_failed")
        raise BlendWorkspaceError(message, reason_code=reason_code)
    command = payload.get("command")
    if command not in {None, "search"}:
        raise BlendWorkspaceError(
            f"Unexpected CLI command in search result: {command!r}.",
            reason_code="search_command_invalid",
        )
    actual_request_id = payload.get("request_id")
    if actual_request_id is not None and str(actual_request_id) != request_id:
        raise BlendWorkspaceError(
            "Search result request_id does not match the prepared request.",
            reason_code="request_id_mismatch",
        )
    blend_id = payload.get("blend_id")
    if expected_blend_id is not None and blend_id != expected_blend_id:
        raise BlendWorkspaceError(
            "Search result blend_id does not match the expected deterministic identity.",
            reason_code="blend_id_mismatch",
        )
    if blend_id is None:
        raise BlendWorkspaceError(
            "Search result is missing blend_id.",
            reason_code="blend_id_missing",
        )
    return dict(payload)


def list_jobs_for_request(
    jobs_root: Path,
    request_id: str,
    *,
    command_id: str = "prediction_blend_v1",
) -> list[dict[str, Any]]:
    if not jobs_root.is_dir():
        return []
    matches: list[dict[str, Any]] = []
    for entry in sorted(jobs_root.iterdir(), key=lambda item: item.name):
        if not entry.is_dir():
            continue
        job_path = entry / "job.json"
        if not job_path.is_file():
            continue
        try:
            job = json.loads(job_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if job.get("schema_version") != 2:
            continue
        if job.get("command_id") != command_id:
            continue
        references = job.get("references")
        if not isinstance(references, dict):
            continue
        if str(references.get("request_id") or "") != request_id:
            continue
        status_payload: dict[str, Any] = {}
        status_path = entry / "status.json"
        if status_path.is_file():
            try:
                loaded_status = json.loads(status_path.read_text(encoding="utf-8"))
                if isinstance(loaded_status, dict):
                    status_payload = loaded_status
            except (OSError, json.JSONDecodeError):
                status_payload = {}
        matches.append(
            {
                "job_id": str(job.get("job_id") or entry.name),
                "command_id": job.get("command_id"),
                "action_id": job.get("action_id"),
                "created_at_utc": job.get("created_at_utc"),
                "references": dict(references),
                "state": status_payload.get("state"),
                "exit_code": status_payload.get("exit_code"),
                "started_at_utc": status_payload.get("started_at_utc"),
                "finished_at_utc": status_payload.get("finished_at_utc"),
            }
        )
    matches.sort(
        key=lambda item: str(item.get("created_at_utc") or ""),
        reverse=True,
    )
    return matches


def discover_materialized_blends(
    repository_root: Path | str,
    blend_root_relative: str | None = None,
) -> list[dict[str, Any]]:
    root = Path(repository_root).resolve()
    blend_root_rel = normalize_blend_root_relative(blend_root_relative)
    blend_root = resolve_under_repository(blend_root_rel, root)
    if not blend_root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for child in sorted(blend_root.iterdir(), key=lambda item: item.name):
        if not child.is_dir() or child.name.startswith("."):
            continue
        blend_id = child.name
        try:
            loaded = load_blend_artifact(
                blend_id,
                repository_root=root,
                blend_root_relative=blend_root_rel,
            )
            manifest = loaded["manifest"]
            readiness = manifest.get("submission_readiness") or {}
            rows.append(
                {
                    "blend_id": blend_id,
                    "status": "completed",
                    "label": blend_display_label(
                        {
                            **manifest,
                            "optimizer_backend": loaded.get("optimizer_backend"),
                            "final_deployment_weights": loaded.get(
                                "final_deployment_weights"
                            ),
                            "canonical_candidate_id": loaded.get(
                                "canonical_candidate_id"
                            ),
                        },
                        repository_root=root,
                    ),
                    "candidate_ids": list(manifest.get("candidate_ids") or []),
                    "optimizer_backend": loaded.get("optimizer_backend"),
                    "honest_meta_cv_metrics": loaded.get("honest_meta_cv_metrics"),
                    "final_deployment_threshold": loaded.get(
                        "final_deployment_threshold"
                    ),
                    "exploratory": bool(manifest.get("exploratory")),
                    "submission_readiness": readiness,
                    "created_at_utc": manifest.get("created_at_utc"),
                    "canonical_candidate_id": loaded.get("canonical_candidate_id"),
                    "manifest_sha256": loaded.get("manifest_sha256"),
                    "loaded": loaded,
                }
            )
        except BlendArtifactError as error:
            rows.append(
                {
                    "blend_id": blend_id,
                    "status": "blocked",
                    "label": blend_id,
                    "reason": str(error),
                    "reason_code": error.reason_code,
                }
            )
    return rows


def apply_canonical_submission_handoff(
    session_state: Any,
    *,
    candidate_id: str,
    manifest_sha256: str,
    blend_id: str | None = None,
) -> None:
    session_state[CANONICAL_SUBMISSION_HANDOFF_KEY] = {
        "candidate_id": candidate_id,
        "manifest_sha256": manifest_sha256,
        "blend_id": blend_id,
        "selected_at_utc": _utc_now(),
    }


def resolve_canonical_submission_handoff(
    session_state: Any,
    repository_root: Path | str,
) -> dict[str, Any] | None:
    handoff = session_state.get(CANONICAL_SUBMISSION_HANDOFF_KEY)
    if not isinstance(handoff, Mapping):
        return None
    candidate_id = handoff.get("candidate_id")
    manifest_sha256 = handoff.get("manifest_sha256")
    if not isinstance(candidate_id, str) or not isinstance(manifest_sha256, str):
        return None
    root = Path(repository_root).resolve()
    relative_root, absolute_root = resolve_candidates_root(root, None)
    package_dir = absolute_root / candidate_id
    try:
        package = load_candidate_package(
            package_dir,
            repository_root=root,
            candidates_root_relative=relative_root,
        )
    except PredictionCandidateError:
        return None
    current_hash = file_sha256(package.package_dir / MANIFEST_FILENAME)
    if current_hash != manifest_sha256:
        return None
    summary = {
        "candidate_id": candidate_id,
        "manifest_sha256": manifest_sha256,
        "blend_id": handoff.get("blend_id"),
        "selected_at_utc": handoff.get("selected_at_utc"),
        "label": candidate_display_label(package.manifest),
        "source_kind": package.manifest.get("source_kind"),
        "source_kind_label": source_kind_label(
            str(package.manifest.get("source_kind") or "")
        ),
        "dataset_id": package.manifest.get("dataset_id"),
        "exploratory": bool(package.manifest.get("exploratory")),
    }
    return summary


def suggest_submission_id(candidate_label: str, *, candidate_id: str) -> str:
    slug = _SLUG_CLEAN.sub("-", candidate_label.strip()).strip("-._")
    slug = slug[:48] or "candidate"
    suffix = candidate_id.removeprefix("pc1_")[:8] or "00000000"
    candidate = f"sub_{slug}_{suffix}"
    if SAFE_STRING.fullmatch(candidate):
        return candidate
    fallback = f"sub_{suffix}"
    if SAFE_STRING.fullmatch(fallback):
        return fallback
    raise BlendWorkspaceError(
        f"Unable to derive a safe submission id from {candidate_id!r}.",
        reason_code="submission_id_invalid",
    )


def authorized_blend_argv_values(request_relative_path: str) -> dict[str, str]:
    relative = str(request_relative_path).replace("\\", "/")
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise BlendWorkspaceError(
            f"Unsafe request path: {request_relative_path!r}",
            reason_code="path_traversal",
        )
    return {"request": relative}


def authorized_submission_argv_values(
    candidate_id: str,
    submission_id: str | None = None,
) -> dict[str, str]:
    if ".." in candidate_id or "/" in candidate_id or "\\" in candidate_id:
        raise BlendWorkspaceError(
            f"Invalid candidate_id: {candidate_id!r}",
            reason_code="candidate_id_invalid",
        )
    values: dict[str, str] = {"candidate_id": candidate_id}
    if submission_id is not None:
        if not SAFE_STRING.fullmatch(submission_id):
            raise BlendWorkspaceError(
                f"Unsafe submission_id: {submission_id!r}",
                reason_code="submission_id_invalid",
            )
        values["submission_id"] = submission_id
    return values


def load_validated_submission_csv_bytes(
    repository_root: Path | str,
    submission_id: str,
) -> bytes:
    root = Path(repository_root).resolve()
    if not SAFE_STRING.fullmatch(submission_id):
        raise CandidateSubmissionError(
            f"Unsafe submission_id: {submission_id!r}",
            reason_code="submission_id_invalid",
        )
    submission_root_rel = normalize_submission_root_relative(None)
    submission_dir = resolve_under_repository(
        f"{submission_root_rel}/{submission_id}",
        root,
    )
    success_path = submission_dir / SUCCESS_FILENAME
    manifest_path = submission_dir / SUBMISSION_MANIFEST_FILENAME
    csv_path = submission_dir / "submission.csv"
    if not success_path.is_file():
        raise CandidateSubmissionError(
            f"Submission artifact missing _SUCCESS: {submission_id}",
            reason_code="success_missing",
        )
    if not manifest_path.is_file() or not csv_path.is_file():
        raise CandidateSubmissionError(
            f"Submission artifact is incomplete: {submission_id}",
            reason_code="artifact_missing",
        )
    try:
        success = json.loads(success_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CandidateSubmissionError(
            f"Submission artifact JSON unreadable: {error}",
            reason_code="json_invalid",
        ) from error
    if not isinstance(success, dict) or not isinstance(manifest, dict):
        raise CandidateSubmissionError(
            "Submission _SUCCESS and manifest must be JSON objects.",
            reason_code="schema_invalid",
        )
    if success.get("status") != "completed":
        raise CandidateSubmissionError(
            f"Submission _SUCCESS status is not completed: {success.get('status')!r}",
            reason_code="success_status_invalid",
        )
    if success.get("submission_id") != submission_id:
        raise CandidateSubmissionError(
            "submission_id mismatch between path and _SUCCESS.",
            reason_code="submission_id_mismatch",
        )
    if manifest.get("submission_id") != submission_id:
        raise CandidateSubmissionError(
            "submission_id mismatch between path and manifest.",
            reason_code="submission_id_mismatch",
        )
    expected_csv_sha = manifest.get("submission_csv_sha256")
    actual_csv_sha = file_sha256(csv_path)
    if expected_csv_sha != actual_csv_sha:
        raise CandidateSubmissionError(
            "submission.csv SHA-256 does not match submission_manifest.json.",
            reason_code="artifact_hash_mismatch",
        )
    if success.get("manifest_sha256") != file_sha256(manifest_path):
        raise CandidateSubmissionError(
            "submission_manifest.json SHA-256 does not match _SUCCESS.manifest_sha256.",
            reason_code="manifest_hash_mismatch",
        )
    expected_size = manifest.get("submission_csv_size_bytes")
    actual_size = csv_path.stat().st_size
    if expected_size is not None and int(expected_size) != actual_size:
        raise CandidateSubmissionError(
            "submission.csv size does not match submission_manifest.json.",
            reason_code="artifact_size_mismatch",
        )
    manifest_csv_path = f"{submission_root_rel}/{submission_id}/submission.csv"
    if manifest.get("submission_csv_sha256") and manifest_csv_path:
        # Manifest does not store csv path; ensure artifact stays under canonical root.
        if not str(submission_dir.relative_to(root)).replace("\\", "/").startswith(
            submission_root_rel
        ):
            raise CandidateSubmissionError(
                "Submission artifact path escapes the canonical submission root.",
                reason_code="path_traversal",
            )
    return csv_path.read_bytes()


__all__ = [
    "CACHE_CONTRACT_VERSION",
    "CANONICAL_SUBMISSION_HANDOFF_KEY",
    "BlendWorkspaceError",
    "apply_canonical_submission_handoff",
    "authorized_blend_argv_values",
    "authorized_submission_argv_values",
    "blend_display_label",
    "candidate_display_label",
    "candidate_inventory_fingerprint",
    "discover_candidate_rows",
    "discover_materialized_blends",
    "expected_blend_id",
    "filter_candidate_rows",
    "list_jobs_for_request",
    "load_validated_submission_csv_bytes",
    "method_label",
    "optimizer_label",
    "parse_job_stdout_json",
    "prepare_request",
    "readiness_label",
    "resolve_canonical_submission_handoff",
    "run_compatibility",
    "run_diversity",
    "selection_fingerprint",
    "settings_fingerprint_from_request_payload",
    "source_kind_label",
    "suggest_submission_id",
    "validate_selection_count",
    "verify_search_result",
]
