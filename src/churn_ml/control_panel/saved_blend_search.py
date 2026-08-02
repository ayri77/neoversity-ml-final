"""Strict projection of completed blend searches over durable Jobs + requests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.churn_ml.control_panel.blend_ui_request_v1 import (
    BlendUIRequest,
    BlendUIRequestError,
    load_blend_ui_request,
    strategy_optimizer_to_method,
)
from src.churn_ml.control_panel.blend_workspace import (
    BlendWorkspaceError,
    expected_blend_id,
    list_jobs_for_request,
    parse_job_stdout_json,
    verify_search_result,
)
from src.churn_ml.control_panel.candidate_display import (
    compact_model_name,
    resolve_candidate_display,
)
from src.churn_ml.control_panel.selection_state import (
    get_durable_value,
    set_durable_value,
)
from src.churn_ml.prediction_candidates.contract_v1 import (
    MANIFEST_FILENAME,
    PredictionCandidateError,
    file_sha256,
    load_candidate_package,
    resolve_under_repository,
)
from src.churn_ml.research_data import canonical_sha256


BLEND_COMMAND_ID = "prediction_blend_v1"
DURABLE_LOADED_CONFIG_KEY = "prediction_blend_v1::loaded_config"
DURABLE_LOAD_PENDING_KEY = "prediction_blend_v1::load_pending"
NUMERIC_COMPARE_TOLERANCE = 1.0e-12

@dataclass(frozen=True)
class SavedBlendSearch:
    job_id: str
    state: str
    request_id: str
    request_path: str
    expected_blend_id: str | None
    candidate_ids: tuple[str, ...]
    candidate_labels: tuple[str, ...]
    strategy: str
    optimizer_backend: str
    folds: int | None
    repeats: int | None
    seed: int | None
    max_active_models: int | None
    search_budget: Mapping[str, Any]
    honest_mean_balanced_accuracy: float | None
    honest_std_balanced_accuracy: float | None
    final_threshold: float | None
    final_weights: Mapping[str, float]
    created_at_utc: str | None
    started_at_utc: str | None
    completed_at_utc: str | None
    reusable: bool
    blocker_codes: tuple[str, ...]
    experiment_label: str
    attempt_count: int = 1
    attempt_job_ids: tuple[str, ...] = ()
    method: str | None = None
    payload_snapshot: Mapping[str, Any] = field(default_factory=dict)
    search_result: Mapping[str, Any] | None = None
    candidate_manifest_sha256: tuple[str, ...] = ()
    candidate_datasets: tuple[str, ...] = ()
    technical: Mapping[str, Any] = field(default_factory=dict)

    @property
    def status_label(self) -> str:
        if self.reusable:
            return "Reusable"
        if self.state in {"failed", "stopped", "orphaned"}:
            return self.state.replace("_", " ").title()
        if self.blocker_codes:
            return "Blocked"
        return (self.state or "unknown").replace("_", " ").title()


def short_model_label(model_name: str, *, dataset_id: str | None = None) -> str:
    """History/table model fragment backed by the central display resolver."""
    if "WeightedEnsemble" in str(model_name or ""):
        version = ""
        if dataset_id and str(dataset_id).startswith("v"):
            version = str(dataset_id).split("_", 1)[0]
        return f"WeightedEnsemble {version}".strip()
    display = resolve_candidate_display(
        {
            "candidate_id": "pc1_0000000000000000",
            "source_model_name": model_name,
            "source_kind": "autogluon_standalone_v1",
            "dataset_id": dataset_id or "",
        }
    )
    return compact_model_name(display.model_name or model_name)


def experiment_label_for(
    *,
    optimizer_backend: str,
    candidate_labels: Sequence[str],
) -> str:
    optimizer = str(optimizer_backend or "unknown").title()
    if optimizer_backend == "native":
        optimizer = "Native"
    elif optimizer_backend == "optuna":
        optimizer = "Optuna"
    models = " + ".join(candidate_labels) if candidate_labels else "no models"
    return f"{optimizer} · {models}"


def jobs_search_fingerprint(jobs_root: Path | str) -> str:
    root = Path(jobs_root)
    entries: list[dict[str, str | None]] = []
    if not root.is_dir():
        return canonical_sha256({"contract": "saved_blend_search_v1", "jobs": []})
    for entry in sorted(root.iterdir(), key=lambda item: item.name):
        if not entry.is_dir():
            continue
        job_path = entry / "job.json"
        if not job_path.is_file():
            continue
        try:
            job = json.loads(job_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if job.get("command_id") != BLEND_COMMAND_ID:
            continue
        if job.get("action_id") != "search":
            continue
        status_path = entry / "status.json"
        stdout_path = entry / "stdout.log"
        entries.append(
            {
                "job_id": str(job.get("job_id") or entry.name),
                "job_sha256": file_sha256(job_path),
                "status_sha256": file_sha256(status_path) if status_path.is_file() else None,
                "stdout_sha256": file_sha256(stdout_path) if stdout_path.is_file() else None,
            }
        )
    return canonical_sha256({"contract": "saved_blend_search_v1", "jobs": entries})


def discover_saved_blend_searches(
    *,
    repository_root: Path | str,
    jobs_root: Path | str,
) -> list[SavedBlendSearch]:
    root = Path(repository_root).resolve()
    jobs = Path(jobs_root)
    if not jobs.is_absolute():
        jobs = root / jobs
    attempts = _discover_search_attempts(root, jobs)
    return _dedupe_attempts(attempts)


def _discover_search_attempts(
    repository_root: Path,
    jobs_root: Path,
) -> list[SavedBlendSearch]:
    if not jobs_root.is_dir():
        return []
    rows: list[SavedBlendSearch] = []
    for entry in sorted(jobs_root.iterdir(), key=lambda item: item.name):
        if not entry.is_dir():
            continue
        projected = _project_search_job(
            entry,
            repository_root=repository_root,
            jobs_root=jobs_root,
        )
        if projected is not None:
            rows.append(projected)
    rows.sort(
        key=lambda item: (
            item.honest_mean_balanced_accuracy is not None,
            item.honest_mean_balanced_accuracy or float("-inf"),
            str(item.created_at_utc or ""),
        ),
        reverse=True,
    )
    return rows


def _project_search_job(
    job_dir: Path,
    *,
    repository_root: Path,
    jobs_root: Path,
) -> SavedBlendSearch | None:
    del jobs_root
    job_path = job_dir / "job.json"
    if not job_path.is_file():
        return None
    blockers: list[str] = []
    technical: dict[str, Any] = {"job_dir": job_dir.name}
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return _blocked_shell(
            job_id=job_dir.name,
            state="unknown",
            blocker_codes=("job_invalid",),
            technical={"error": str(error)},
        )
    if job.get("schema_version") != 2:
        return None
    if job.get("command_id") != BLEND_COMMAND_ID or job.get("action_id") != "search":
        return None
    references = job.get("references") if isinstance(job.get("references"), dict) else {}
    job_id = str(job.get("job_id") or job_dir.name)
    request_id = str(references.get("request_id") or "")
    request_path = str(references.get("request_path") or "")
    status_payload: dict[str, Any] = {}
    status_path = job_dir / "status.json"
    if status_path.is_file():
        try:
            loaded = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                status_payload = loaded
        except (OSError, json.JSONDecodeError):
            blockers.append("status_invalid")
    state = str(status_payload.get("state") or "unknown")
    created = job.get("created_at_utc")
    started = status_payload.get("started_at_utc")
    finished = status_payload.get("finished_at_utc")

    if not request_id:
        blockers.append("request_missing")
        return _blocked_shell(
            job_id=job_id,
            state=state,
            request_id=request_id,
            request_path=request_path,
            created_at_utc=str(created) if created else None,
            started_at_utc=str(started) if started else None,
            completed_at_utc=str(finished) if finished else None,
            blocker_codes=tuple(blockers or ("request_missing",)),
            technical=technical,
        )

    if state in {"failed", "stopped", "orphaned"}:
        blockers.append(f"job_{state}")

    request: BlendUIRequest | None = None
    try:
        request = load_blend_ui_request(
            request_path or request_id,
            repository_root=repository_root,
        )
    except BlendUIRequestError as error:
        code = getattr(error, "reason_code", "request_invalid")
        blockers.append(str(code) if code else "request_invalid")
        technical["request_error"] = str(error)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        blockers.append("request_invalid")
        technical["request_error"] = str(error)

    if request is not None:
        if request.request_id != request_id:
            blockers.append("request_id_mismatch")
        if request_path and request.relative_path != request_path.replace("\\", "/"):
            blockers.append("request_path_mismatch")
        request_path = request.relative_path
        request_id = request.request_id

    candidate_ids: tuple[str, ...] = ()
    candidate_labels: list[str] = []
    candidate_datasets: list[str] = []
    manifest_hashes: list[str] = []
    expected_id: str | None = None
    strategy = ""
    optimizer = ""
    folds = None
    repeats = None
    seed = None
    max_active = None
    method = None
    budget: dict[str, Any] = {}
    payload_snapshot: dict[str, Any] = {}

    if request is not None:
        payload_snapshot = dict(request.payload)
        candidate_ids = tuple(request.candidate_ids)
        strategy = str(request.payload.get("strategy") or "")
        optimizer = str(request.payload.get("optimizer_backend") or "")
        folds = _as_int(request.payload.get("folds"))
        repeats = _as_int(request.payload.get("repeats"))
        seed = _as_int(request.payload.get("seed"))
        max_active = _as_int(request.payload.get("max_active_models"))
        try:
            method = strategy_optimizer_to_method(strategy, optimizer)
        except BlendUIRequestError:
            method = str(request.payload.get("method") or None)
        budget = _search_budget_from_payload(request.payload)
        for candidate_id in candidate_ids:
            try:
                package = load_candidate_package(
                    resolve_under_repository(
                        f"artifacts/prediction_candidates/{candidate_id}",
                        repository_root,
                    ),
                    repository_root=repository_root,
                    candidates_root_relative=str(
                        request.payload.get("candidate_root")
                        or "artifacts/prediction_candidates"
                    ),
                )
            except (PredictionCandidateError, OSError, ValueError) as error:
                blockers.append("candidate_missing")
                technical.setdefault("candidate_errors", {})[candidate_id] = str(error)
                candidate_labels.append(candidate_id)
                candidate_datasets.append("")
                manifest_hashes.append("")
                continue
            display = resolve_candidate_display(
                {
                    "candidate_id": candidate_id,
                    "source_model_name": package.manifest.get("source_model_name"),
                    "source_kind": package.manifest.get("source_kind"),
                    "dataset_id": package.manifest.get("dataset_id"),
                    "source_metadata": package.source_metadata,
                    "exploratory": package.manifest.get("exploratory"),
                }
            )
            label = short_model_label(
                display.model_name or candidate_id,
                dataset_id=display.dataset_id or None,
            )
            candidate_labels.append(label)
            candidate_datasets.append(str(display.dataset_id or ""))
            package_dir = resolve_under_repository(
                f"artifacts/prediction_candidates/{candidate_id}",
                repository_root,
            )
            manifest_hashes.append(file_sha256(package_dir / MANIFEST_FILENAME))
        if "candidate_missing" not in blockers:
            try:
                expected_id = expected_blend_id(request, repository_root)
            except Exception as error:  # noqa: BLE001
                blockers.append("candidate_manifest_changed")
                technical["expected_blend_error"] = str(error)

    search_result: dict[str, Any] | None = None
    honest_mean = None
    honest_std = None
    threshold = None
    weights: dict[str, float] = {}
    stdout_path = job_dir / "stdout.log"
    if state == "succeeded":
        if not stdout_path.is_file():
            blockers.append("stdout_missing")
        else:
            try:
                payload = parse_job_stdout_json(
                    stdout_path.read_text(encoding="utf-8", errors="replace")
                )
                verified = verify_search_result(
                    payload,
                    request_id=request_id,
                    expected_blend_id=expected_id,
                )
                search_result = verified
                honest = verified.get("honest_meta_cv_metrics") or {}
                honest_mean = _as_float(honest.get("mean_repeat_balanced_accuracy"))
                honest_std = _as_float(honest.get("std_repeat_balanced_accuracy"))
                threshold = _as_float(
                    verified.get("final_deployment_threshold")
                    or (verified.get("deployment") or {}).get("threshold")
                )
                raw_weights = verified.get("final_deployment_weights") or (
                    (verified.get("deployment") or {}).get("weights")
                )
                if isinstance(raw_weights, Mapping):
                    weights = {
                        str(key): float(value) for key, value in raw_weights.items()
                    }
                if expected_id is not None and verified.get("blend_id") != expected_id:
                    blockers.append("blend_id_mismatch")
                    blockers.append("candidate_manifest_changed")
            except BlendWorkspaceError as error:
                code = str(
                    getattr(error, "reason_code", None) or "search_result_invalid"
                )
                if code in {"stdout_json_invalid", "stdout_empty"}:
                    code = "stdout_malformed"
                blockers.append(code)
                if code == "blend_id_mismatch":
                    blockers.append("candidate_manifest_changed")
                technical["stdout_error"] = str(error)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                blockers.append("stdout_malformed")
                technical["stdout_error"] = str(error)
    elif state not in {"failed", "stopped", "orphaned"} and state != "succeeded":
        # running/queued/unknown remain non-reusable without extra codes unless already blocked
        if not blockers:
            blockers.append("job_incomplete")

    reusable = state == "succeeded" and not blockers and search_result is not None
    short_labels = tuple(candidate_labels)
    return SavedBlendSearch(
        job_id=job_id,
        state=state,
        request_id=request_id,
        request_path=request_path,
        expected_blend_id=expected_id
        or (str(search_result.get("blend_id")) if search_result else None),
        candidate_ids=candidate_ids,
        candidate_labels=short_labels,
        strategy=strategy,
        optimizer_backend=optimizer,
        folds=folds,
        repeats=repeats,
        seed=seed,
        max_active_models=max_active,
        search_budget=budget,
        honest_mean_balanced_accuracy=honest_mean,
        honest_std_balanced_accuracy=honest_std,
        final_threshold=threshold,
        final_weights=weights,
        created_at_utc=str(created) if created else None,
        started_at_utc=str(started) if started else None,
        completed_at_utc=str(finished) if finished else None,
        reusable=reusable,
        blocker_codes=tuple(dict.fromkeys(blockers)),
        experiment_label=experiment_label_for(
            optimizer_backend=optimizer or "unknown",
            candidate_labels=short_labels,
        ),
        attempt_count=1,
        attempt_job_ids=(job_id,),
        method=method,
        payload_snapshot=payload_snapshot,
        search_result=search_result,
        candidate_manifest_sha256=tuple(manifest_hashes),
        candidate_datasets=tuple(candidate_datasets),
        technical=technical,
    )


def _dedupe_attempts(attempts: Sequence[SavedBlendSearch]) -> list[SavedBlendSearch]:
    by_request: dict[str, list[SavedBlendSearch]] = {}
    for item in attempts:
        by_request.setdefault(item.request_id or item.job_id, []).append(item)
    rows: list[SavedBlendSearch] = []
    for request_id, group in by_request.items():
        del request_id
        ordered = sorted(
            group,
            key=lambda item: str(item.created_at_utc or ""),
            reverse=True,
        )
        preferred = next((item for item in ordered if item.reusable), ordered[0])
        attempt_ids = tuple(item.job_id for item in ordered)
        rows.append(
            SavedBlendSearch(
                job_id=preferred.job_id,
                state=preferred.state,
                request_id=preferred.request_id,
                request_path=preferred.request_path,
                expected_blend_id=preferred.expected_blend_id,
                candidate_ids=preferred.candidate_ids,
                candidate_labels=preferred.candidate_labels,
                strategy=preferred.strategy,
                optimizer_backend=preferred.optimizer_backend,
                folds=preferred.folds,
                repeats=preferred.repeats,
                seed=preferred.seed,
                max_active_models=preferred.max_active_models,
                search_budget=dict(preferred.search_budget),
                honest_mean_balanced_accuracy=preferred.honest_mean_balanced_accuracy,
                honest_std_balanced_accuracy=preferred.honest_std_balanced_accuracy,
                final_threshold=preferred.final_threshold,
                final_weights=dict(preferred.final_weights),
                created_at_utc=preferred.created_at_utc,
                started_at_utc=preferred.started_at_utc,
                completed_at_utc=preferred.completed_at_utc,
                reusable=preferred.reusable,
                blocker_codes=preferred.blocker_codes,
                experiment_label=preferred.experiment_label,
                attempt_count=len(ordered),
                attempt_job_ids=attempt_ids,
                method=preferred.method,
                payload_snapshot=dict(preferred.payload_snapshot),
                search_result=preferred.search_result,
                candidate_manifest_sha256=preferred.candidate_manifest_sha256,
                candidate_datasets=preferred.candidate_datasets,
                technical={
                    **dict(preferred.technical),
                    "attempts": [
                        {
                            "job_id": item.job_id,
                            "state": item.state,
                            "reusable": item.reusable,
                            "created_at_utc": item.created_at_utc,
                            "blocker_codes": list(item.blocker_codes),
                        }
                        for item in ordered
                    ],
                },
            )
        )
    rows.sort(
        key=lambda item: (
            item.honest_mean_balanced_accuracy is not None,
            item.honest_mean_balanced_accuracy or float("-inf"),
            str(item.created_at_utc or ""),
        ),
        reverse=True,
    )
    return rows


def filter_saved_searches(
    rows: Sequence[SavedBlendSearch],
    *,
    status: str | None = None,
    optimizer: str | None = None,
    search_text: str | None = None,
) -> list[SavedBlendSearch]:
    query = (search_text or "").strip().casefold()
    filtered: list[SavedBlendSearch] = []
    for row in rows:
        if status == "reusable" and not row.reusable:
            continue
        if status == "blocked" and row.reusable:
            continue
        if optimizer is not None and row.optimizer_backend != optimizer:
            continue
        if query:
            haystack = " ".join(
                [
                    row.experiment_label,
                    row.request_id,
                    row.expected_blend_id or "",
                    " ".join(row.candidate_labels),
                    " ".join(row.candidate_ids),
                    row.optimizer_backend,
                ]
            ).casefold()
            if query not in haystack:
                continue
        filtered.append(row)
    return filtered


def configuration_from_saved_search(saved: SavedBlendSearch) -> dict[str, Any]:
    payload = dict(saved.payload_snapshot)
    method = saved.method or str(payload.get("method") or "optimized_native")
    return {
        "candidate_ids": list(saved.candidate_ids),
        "method": method,
        "folds": saved.folds if saved.folds is not None else 5,
        "repeats": saved.repeats if saved.repeats is not None else 2,
        "seed": saved.seed if saved.seed is not None else 42,
        "max_active_models": saved.max_active_models,
        "manual_weights": list(payload.get("manual_weights") or []),
        "dirichlet_draws": payload.get("dirichlet_draws", 32),
        "pairwise_grid_step": payload.get("pairwise_grid_step", 0.1),
        "optuna_trials": payload.get("optuna_trials"),
        "optuna_timeout_seconds": payload.get("optuna_timeout_seconds"),
        "optuna_seed": payload.get("optuna_seed"),
        "request_id": saved.request_id,
        "request_path": saved.request_path,
        "expected_blend_id": saved.expected_blend_id,
        "source_search_job_id": saved.job_id,
    }


def persist_loaded_blend_configuration(
    session_state: Any,
    config: Mapping[str, Any],
    *,
    pending_apply: bool = False,
) -> None:
    set_durable_value(session_state, DURABLE_LOADED_CONFIG_KEY, dict(config))
    set_durable_value(
        session_state,
        DURABLE_LOAD_PENDING_KEY,
        True if pending_apply else None,
    )


def load_persisted_blend_configuration(session_state: Any) -> dict[str, Any] | None:
    value = get_durable_value(session_state, DURABLE_LOADED_CONFIG_KEY)
    if not isinstance(value, dict):
        return None
    return dict(value)


def consume_loaded_blend_configuration_pending(session_state: Any) -> bool:
    pending = bool(get_durable_value(session_state, DURABLE_LOAD_PENDING_KEY))
    if pending:
        set_durable_value(session_state, DURABLE_LOAD_PENDING_KEY, None)
    return pending


def clear_persisted_blend_configuration(session_state: Any) -> None:
    set_durable_value(session_state, DURABLE_LOADED_CONFIG_KEY, None)
    set_durable_value(session_state, DURABLE_LOAD_PENDING_KEY, None)


def compare_materialized_to_saved(
    *,
    saved: SavedBlendSearch,
    materialized_payload: Mapping[str, Any],
    loaded_artifact: Mapping[str, Any],
) -> list[str]:
    mismatches: list[str] = []
    if str(materialized_payload.get("request_id") or "") not in {"", saved.request_id}:
        if str(materialized_payload.get("request_id")) != saved.request_id:
            mismatches.append("request_id")
    blend_id = str(
        materialized_payload.get("blend_id")
        or loaded_artifact.get("blend_id")
        or ""
    )
    if saved.expected_blend_id and blend_id != saved.expected_blend_id:
        mismatches.append("blend_id")
    actual_candidates = [
        str(item)
        for item in (
            materialized_payload.get("candidate_ids")
            or loaded_artifact.get("candidate_ids")
            or []
        )
    ]
    if actual_candidates and actual_candidates != list(saved.candidate_ids):
        mismatches.append("candidate_ids")
    optimizer = str(
        materialized_payload.get("optimizer_backend")
        or loaded_artifact.get("optimizer_backend")
        or ""
    )
    if optimizer and optimizer != saved.optimizer_backend:
        mismatches.append("optimizer")
    honest = loaded_artifact.get("honest_meta_cv_metrics") or materialized_payload.get(
        "honest_meta_cv_metrics"
    ) or {}
    mean = _as_float(honest.get("mean_repeat_balanced_accuracy"))
    if (
        saved.honest_mean_balanced_accuracy is not None
        and mean is not None
        and abs(mean - saved.honest_mean_balanced_accuracy) > NUMERIC_COMPARE_TOLERANCE
    ):
        mismatches.append("honest_mean_balanced_accuracy")
    threshold = _as_float(
        loaded_artifact.get("final_deployment_threshold")
        or materialized_payload.get("final_deployment_threshold")
    )
    if (
        saved.final_threshold is not None
        and threshold is not None
        and abs(threshold - saved.final_threshold) > NUMERIC_COMPARE_TOLERANCE
    ):
        mismatches.append("final_threshold")
    weights = loaded_artifact.get("final_deployment_weights") or (
        materialized_payload.get("final_deployment_weights")
    )
    if isinstance(weights, Mapping) and saved.final_weights:
        for key, expected in saved.final_weights.items():
            actual = weights.get(key)
            if actual is None or abs(float(actual) - float(expected)) > NUMERIC_COMPARE_TOLERANCE:
                mismatches.append("final_weights")
                break
    return mismatches


def recover_search_for_request(
    *,
    jobs_root: Path,
    request: BlendUIRequest,
    expected_blend_id_value: str | None,
) -> dict[str, Any] | None:
    related = list_jobs_for_request(
        jobs_root, request.request_id, command_id=BLEND_COMMAND_ID
    )
    search_jobs = [item for item in related if item.get("action_id") == "search"]
    if not search_jobs:
        return None
    latest = search_jobs[0]
    state = latest.get("state")
    if state != "succeeded":
        return {
            "recovery_state": state or "unknown",
            "job_id": latest.get("job_id"),
            "payload": None,
        }
    stdout_path = jobs_root / str(latest["job_id"]) / "stdout.log"
    payload = verify_search_result(
        parse_job_stdout_json(
            stdout_path.read_text(encoding="utf-8", errors="replace")
        ),
        request_id=request.request_id,
        expected_blend_id=expected_blend_id_value,
    )
    return {
        "recovery_state": "succeeded",
        "job_id": latest.get("job_id"),
        "payload": payload,
    }


def resolve_displayed_search_job_id(
    *,
    session_job_id: str | None,
    session_request_id: str | None,
    current_request_id: str,
    request_search_job_ids: Sequence[str],
) -> str | None:
    """Return a session search job ID only when it belongs to the current request."""
    if not session_job_id:
        return None
    if session_request_id != current_request_id:
        return None
    allowed = set(request_search_job_ids)
    if session_job_id not in allowed:
        return None
    return session_job_id


def historical_materialize_launch_plan(saved: SavedBlendSearch) -> Mapping[str, Any]:
    """Pure plan for materializing a saved search without consulting Build form state."""
    return {
        "command_id": BLEND_COMMAND_ID,
        "action_id": "materialize",
        "request_path": saved.request_path,
        "request_id": saved.request_id,
        "requires_confirmation": True,
        "references": {
            "source_search_job_id": saved.job_id,
            "request_id": saved.request_id,
            "request_path": saved.request_path,
            "expected_blend_id": str(saved.expected_blend_id or ""),
            "candidate_ids": ",".join(saved.candidate_ids),
            "strategy": saved.strategy,
            "optimizer": saved.optimizer_backend,
        },
    }


def filter_materialize_jobs_for_saved_search(
    jobs: Sequence[Mapping[str, Any]],
    saved: SavedBlendSearch,
) -> list[Mapping[str, Any]]:
    matched: list[Mapping[str, Any]] = []
    for item in jobs:
        if item.get("action_id") != "materialize":
            continue
        references = item.get("references") if isinstance(item.get("references"), dict) else {}
        source_job = str(references.get("source_search_job_id") or "")
        expected = str(references.get("expected_blend_id") or "")
        request_id = str(references.get("request_id") or "")
        if request_id and request_id != saved.request_id:
            continue
        if source_job and source_job != saved.job_id:
            continue
        if (
            expected
            and saved.expected_blend_id
            and expected != saved.expected_blend_id
            and source_job != saved.job_id
        ):
            continue
        matched.append(item)
    return matched


def _search_budget_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    optimizer = str(payload.get("optimizer_backend") or "")
    if optimizer == "optuna":
        return {
            "optuna_trials": payload.get("optuna_trials"),
            "optuna_timeout_seconds": payload.get("optuna_timeout_seconds"),
            "optuna_seed": payload.get("optuna_seed"),
        }
    return {
        "dirichlet_draws": payload.get("dirichlet_draws"),
        "pairwise_grid_step": payload.get("pairwise_grid_step"),
    }


def _blocked_shell(
    *,
    job_id: str,
    state: str,
    blocker_codes: tuple[str, ...],
    request_id: str = "",
    request_path: str = "",
    created_at_utc: str | None = None,
    started_at_utc: str | None = None,
    completed_at_utc: str | None = None,
    technical: Mapping[str, Any] | None = None,
) -> SavedBlendSearch:
    return SavedBlendSearch(
        job_id=job_id,
        state=state,
        request_id=request_id,
        request_path=request_path,
        expected_blend_id=None,
        candidate_ids=(),
        candidate_labels=(),
        strategy="",
        optimizer_backend="",
        folds=None,
        repeats=None,
        seed=None,
        max_active_models=None,
        search_budget={},
        honest_mean_balanced_accuracy=None,
        honest_std_balanced_accuracy=None,
        final_threshold=None,
        final_weights={},
        created_at_utc=created_at_utc,
        started_at_utc=started_at_utc,
        completed_at_utc=completed_at_utc,
        reusable=False,
        blocker_codes=blocker_codes,
        experiment_label=f"Blocked · {job_id[:8]}",
        technical=dict(technical or {}),
    )


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "BLEND_COMMAND_ID",
    "DURABLE_LOADED_CONFIG_KEY",
    "DURABLE_LOAD_PENDING_KEY",
    "NUMERIC_COMPARE_TOLERANCE",
    "SavedBlendSearch",
    "clear_persisted_blend_configuration",
    "compare_materialized_to_saved",
    "configuration_from_saved_search",
    "consume_loaded_blend_configuration_pending",
    "discover_saved_blend_searches",
    "experiment_label_for",
    "filter_materialize_jobs_for_saved_search",
    "filter_saved_searches",
    "historical_materialize_launch_plan",
    "jobs_search_fingerprint",
    "load_persisted_blend_configuration",
    "persist_loaded_blend_configuration",
    "recover_search_for_request",
    "resolve_displayed_search_job_id",
    "short_model_label",
]
