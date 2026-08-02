"""Managed AutoGluon candidate preparation helpers for Blend Workspace."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.churn_ml.control_panel.blend_workspace import parse_job_stdout_json
from src.churn_ml.control_panel.selection_state import (
    get_durable_value,
    set_durable_value,
)
from src.churn_ml.dataset_registry.api import resolve_dataset_package
from src.churn_ml.prediction_candidates.autogluon_v1 import (
    OOF_PROTOCOL,
    SOURCE_KIND,
    inventory_autogluon_run,
)
from src.churn_ml.prediction_candidates.contract_v1 import (
    MANIFEST_FILENAME,
    POSITIVE_CLASS_LABEL,
    PROBABILITY_SEMANTICS,
    SCHEMA_VERSION,
    SUCCESS_FILENAME,
    PredictionCandidateError,
    build_candidate_id,
    file_sha256,
    load_candidate_package,
    resolve_under_repository,
)
from src.churn_ml.prediction_candidates.preparation_request_v1 import (
    MANAGED_RUNS_ROOT,
    MAX_MODELS,
    MIN_MODELS,
    PreparationRequest,
    PreparationRequestPreview,
    collect_run_identity_hashes,
    load_preparation_request,
    materialize_preparation_request,
    preview_preparation_request,
    resolve_existing_preparation_request,
    validate_selected_models,
    verify_request_run_identity,
)
from src.churn_ml.prediction_candidates.submission_v1 import (
    evaluate_submission_readiness,
)
from src.churn_ml.research_data import canonical_sha256


CACHE_CONTRACT_VERSION = "candidate_preparation_cache_v1"
ENSEMBLE_PREFIX = re.compile(r"^WeightedEnsemble_")
PREPARATION_COMMAND_ID = "autogluon_candidate_preparation_v1"
DURABLE_CONTEXT_KEY = "autogluon_candidate_preparation_v1::context"
INDEPENDENT_CANDIDATE_COPY = (
    "Each selected model is prepared as a separate canonical candidate. "
    "Selecting multiple models only groups the work into one Job."
)


class CandidatePreparationError(ValueError):
    def __init__(self, message: str, *, reason_code: str = "candidate_preparation") -> None:
        self.reason_code = reason_code
        super().__init__(message)


def managed_runs_root(repository_root: Path | str) -> Path:
    root = Path(repository_root).resolve()
    return resolve_under_repository(MANAGED_RUNS_ROOT, root)


def discover_managed_run_directories(repository_root: Path | str) -> list[Path]:
    root = Path(repository_root).resolve()
    runs_root = managed_runs_root(root)
    if not runs_root.is_dir():
        return []
    found: set[Path] = set()
    for name in ("_SUCCESS", "_FAILED", "execution_status.json"):
        for artifact in runs_root.rglob(name):
            if artifact.is_file():
                try:
                    found.add(artifact.parent.resolve(strict=True))
                except OSError:
                    continue
    return sorted(found, key=lambda item: item.as_posix())


def managed_runs_inventory_fingerprint(repository_root: Path | str) -> str:
    root = Path(repository_root).resolve()
    entries: list[dict[str, str | None]] = []
    for run_dir in discover_managed_run_directories(root):
        hashes = collect_run_identity_hashes(run_dir)
        entries.append(
            {
                "run_path": run_dir.relative_to(root).as_posix().replace("\\", "/"),
                **hashes,
            }
        )
    return canonical_sha256(
        {"contract": CACHE_CONTRACT_VERSION, "runs": entries}
    )


def discover_managed_autogluon_runs(
    repository_root: Path | str,
    *,
    include_incomplete: bool = False,
) -> list[dict[str, Any]]:
    root = Path(repository_root).resolve()
    prepared_index = _prepared_candidate_index(root)
    rows: list[dict[str, Any]] = []
    for run_dir in discover_managed_run_directories(root):
        row = summarize_managed_run(
            run_dir,
            repository_root=root,
            prepared_index=prepared_index,
        )
        if row["preparable"] or include_incomplete:
            rows.append(row)
    return rows


def summarize_managed_run(
    run_dir: Path,
    *,
    repository_root: Path,
    prepared_index: Mapping[tuple[str, str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    blockers: list[dict[str, str]] = []
    run_path = run_dir.relative_to(root).as_posix().replace("\\", "/")
    try:
        inventory = inventory_autogluon_run(run_dir, repository_root=root)
    except (PredictionCandidateError, OSError, ValueError, json.JSONDecodeError) as error:
        reason_code = getattr(error, "reason_code", "run_inventory_failed")
        return {
            "run_path": run_path,
            "run_id": None,
            "dataset_id": None,
            "target_dependency": None,
            "exploratory": None,
            "classification": "blocked",
            "config_sha256": None,
            "autogluon_version": None,
            "best_model": None,
            "model_count": 0,
            "leaderboard_available": False,
            "predictor_present": (run_dir / "predictor").is_dir(),
            "prepared_candidate_count": 0,
            "preparable": False,
            "blocking_reasons": [
                {"code": str(reason_code), "message": str(error)}
            ],
            "label": run_path,
            "models": [],
        }

    metadata = _read_json(run_dir / "run_metadata.json")
    run_id = metadata.get("run_id") if isinstance(metadata, dict) else None
    profile_id = metadata.get("profile_id") if isinstance(metadata, dict) else None
    leaderboard_available = (run_dir / "inspection" / "leaderboard.csv").is_file()
    predictor_present = (run_dir / "predictor" / "predictor.pkl").is_file()
    target_dependency = None
    exploratory = None
    try:
        dataset = resolve_dataset_package(
            root / "data" / "processed", inventory.dataset_id
        )
        target_dependency = str(dataset.manifest.target_dependency)
        exploratory = target_dependency == "exploratory"
    except Exception as error:  # noqa: BLE001
        blockers.append(
            {
                "code": "dataset_unresolved",
                "message": f"Dataset Package unresolved: {error}",
            }
        )

    if inventory.classification != "complete":
        blockers.append(
            {
                "code": "run_incomplete",
                "message": f"Run classification is {inventory.classification!r}.",
            }
        )
    if not predictor_present:
        blockers.append(
            {
                "code": "predictor_missing",
                "message": "Predictor files are missing.",
            }
        )
    if not leaderboard_available:
        blockers.append(
            {
                "code": "leaderboard_missing",
                "message": "inspection/leaderboard.csv is missing.",
            }
        )
    if not inventory.model_names:
        blockers.append(
            {
                "code": "model_inventory_missing",
                "message": "No model names recorded in worker_result.",
            }
        )
    smoke_token = f"{profile_id or ''} {run_path}".casefold()
    if "smoke" in smoke_token:
        blockers.append(
            {
                "code": "smoke_run_blocked",
                "message": "Smoke AutoGluon runs are not preparable for blending.",
            }
        )

    index = prepared_index or {}
    models = build_model_rows(
        inventory,
        prepared_index=index,
        repository_root=root,
    )
    prepared_count = sum(1 for item in models if item.get("preparation_state") == "already_prepared")
    preparable = not blockers
    label_parts = [
        inventory.dataset_id,
        str(profile_id or "managed"),
        f"{len(inventory.model_names)} models",
    ]
    return {
        "run_path": inventory.run_path,
        "run_id": run_id,
        "dataset_id": inventory.dataset_id,
        "target_dependency": target_dependency,
        "exploratory": exploratory,
        "classification": inventory.classification,
        "config_path": inventory.config_path,
        "config_sha256": inventory.config_sha256,
        "autogluon_version": inventory.autogluon_version,
        "best_model": inventory.best_model,
        "model_count": len(inventory.model_names),
        "leaderboard_available": leaderboard_available,
        "predictor_present": predictor_present,
        "prepared_candidate_count": prepared_count,
        "preparable": preparable,
        "blocking_reasons": blockers,
        "profile_id": profile_id,
        "label": " · ".join(label_parts),
        "models": models,
        "identity_hashes": collect_run_identity_hashes(run_dir),
    }


def build_model_rows(
    inventory: Any,
    *,
    prepared_index: Mapping[tuple[str, str, str], dict[str, Any]],
    repository_root: Path,
) -> list[dict[str, Any]]:
    del repository_root
    rows: list[dict[str, Any]] = []
    leaderboard_by_name = {
        str(row.get("model")): row for row in inventory.leaderboard if row.get("model")
    }
    for model_name in inventory.model_names:
        entry = leaderboard_by_name.get(model_name) or {}
        score = entry.get("score_val")
        model_type = entry.get("model_type") or entry.get("child_model_type")
        stack = entry.get("stack_level")
        fit_time = entry.get("fit_time")
        pred_time = entry.get("pred_time_val") or entry.get("pred_time_test")
        key = (
            str(inventory.run_path),
            str(model_name),
            str(inventory.config_sha256),
        )
        prepared = prepared_index.get(key)
        if prepared is None:
            # Deterministic expected ID even when not prepared.
            expected_id = build_candidate_id(
                {
                    "schema_version": SCHEMA_VERSION,
                    "source_kind": SOURCE_KIND,
                    "dataset_id": inventory.dataset_id,
                    "source_run_path": inventory.run_path,
                    "source_config_sha256": inventory.config_sha256,
                    "source_model_name": model_name,
                    "oof_protocol": OOF_PROTOCOL,
                    "positive_class_label": POSITIVE_CLASS_LABEL,
                    "probability_semantics": PROBABILITY_SEMANTICS,
                }
            )
            preparation_state = "ready_to_validate"
            candidate_id = expected_id
            package_status = None
        else:
            preparation_state = str(prepared["preparation_state"])
            candidate_id = prepared.get("candidate_id")
            package_status = prepared.get("package_status")
        score_text = None
        if score is not None:
            try:
                score_text = f"{float(score):.4f}"
            except (TypeError, ValueError):
                score_text = str(score)
        if model_name == inventory.best_model:
            label = f"{model_name} · best ensemble" if ENSEMBLE_PREFIX.match(model_name) else (
                f"{model_name} · best model"
            )
            if score_text is not None and "best" not in label:
                label = f"{model_name} · validation BA {score_text}"
        elif score_text is not None:
            label = f"{model_name} · validation BA {score_text}"
        else:
            label = model_name
        rows.append(
            {
                "model_name": model_name,
                "label": label,
                "model_type": model_type,
                "validation_score": score,
                "stack_level": stack,
                "fit_time": fit_time,
                "prediction_time": pred_time,
                "is_best": model_name == inventory.best_model,
                "is_ensemble": bool(ENSEMBLE_PREFIX.match(model_name)),
                "already_prepared": preparation_state == "already_prepared",
                "preparation_state": preparation_state,
                "candidate_id": candidate_id,
                "package_status": package_status,
            }
        )
    return rows


def _prepared_candidate_index(
    repository_root: Path,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    root = Path(repository_root).resolve()
    candidates_root = resolve_under_repository("artifacts/prediction_candidates", root)
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not candidates_root.is_dir():
        return index
    for child in sorted(candidates_root.iterdir(), key=lambda item: item.name):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if not (child / SUCCESS_FILENAME).is_file():
            continue
        try:
            package = load_candidate_package(
                child,
                repository_root=root,
                candidates_root_relative="artifacts/prediction_candidates",
            )
        except PredictionCandidateError:
            # Try to read raw manifest for invalid marker.
            manifest_path = child / MANIFEST_FILENAME
            if not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(manifest, dict):
                continue
            key = (
                str(manifest.get("source_run_path") or ""),
                str(manifest.get("source_model_name") or ""),
                str(manifest.get("source_config_sha256") or ""),
            )
            if not all(key):
                continue
            index[key] = {
                "preparation_state": "invalid_existing_candidate",
                "candidate_id": child.name,
                "package_status": "invalid",
            }
            continue
        manifest = package.manifest
        if manifest.get("source_kind") != SOURCE_KIND:
            continue
        key = (
            str(manifest.get("source_run_path") or ""),
            str(manifest.get("source_model_name") or ""),
            str(manifest.get("source_config_sha256") or ""),
        )
        if not all(key):
            continue
        index[key] = {
            "preparation_state": "already_prepared",
            "candidate_id": package.candidate_id,
            "package_status": "completed",
            "exploratory": bool(manifest.get("exploratory")),
            "dataset_id": manifest.get("dataset_id"),
        }
    return index


def filter_managed_runs(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str | None = None,
    target_dependency: str | None = None,
    completion: str | None = None,
    search_text: str | None = None,
) -> list[dict[str, Any]]:
    query = (search_text or "").strip().casefold()
    filtered: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if dataset is not None and str(item.get("dataset_id") or "") != dataset:
            continue
        if target_dependency is not None and str(item.get("target_dependency") or "") != target_dependency:
            continue
        if completion == "completed" and not item.get("preparable"):
            continue
        if completion == "blocked" and item.get("preparable"):
            continue
        if query:
            haystack = " ".join(
                str(item.get(key) or "")
                for key in ("label", "run_path", "run_id", "dataset_id", "profile_id")
            ).casefold()
            if query not in haystack:
                continue
        filtered.append(item)
    return filtered


def ensemble_component_warning(selected_models: Sequence[str]) -> str | None:
    names = list(selected_models)
    ensembles = [name for name in names if ENSEMBLE_PREFIX.match(name)]
    components = [name for name in names if not ENSEMBLE_PREFIX.match(name)]
    if ensembles and components:
        ensemble_label = ensembles[0]
        return (
            f"{ensemble_label} and some of its same-run components are selected. "
            "Preparing them together is allowed: each becomes a separate candidate. "
            "For the first blend, avoid using the ensemble together with all of its "
            "own components unless there is an explicit reason. "
            "This warning concerns later blend selection only. "
            "Preparation still creates independent candidates and is safe."
        )
    return None


def package_status_label(preparation_state: str | None) -> str:
    mapping = {
        "already_prepared": "Prepared",
        "invalid_existing_candidate": "Invalid",
        "blocked": "Blocked",
        "ready_to_validate": "Not prepared",
    }
    return mapping.get(str(preparation_state or ""), "Not prepared")


def preparation_button_label(count: int) -> str:
    if count == 1:
        return "Prepare 1 independent candidate"
    return f"Prepare {count} independent candidates"


def independent_candidate_summary(count: int) -> str:
    return f"{count} selected models → {count} independent prediction candidates"


def preview_independent_candidate_rows(
    preview: PreparationRequestPreview,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for model_name, candidate_id in zip(
        preview.selected_models, preview.candidate_ids, strict=True
    ):
        rows.append(
            {
                "Model": model_name,
                "Future candidate ID": candidate_id,
                "Output type": "Independent prediction_candidate_v1 package",
            }
        )
    return rows


def prepare_preparation_request(
    *,
    repository_root: Path | str,
    run_path: str,
    selected_models: Sequence[str],
) -> PreparationRequest:
    validate_selected_models(selected_models)
    return materialize_preparation_request(
        repository_root=Path(repository_root).resolve(),
        run_path=run_path,
        selected_models=selected_models,
    )


def preview_request_for_selection(
    *,
    repository_root: Path | str,
    run_path: str,
    selected_models: Sequence[str],
) -> PreparationRequestPreview:
    validate_selected_models(selected_models)
    return preview_preparation_request(
        repository_root=Path(repository_root).resolve(),
        run_path=run_path,
        selected_models=selected_models,
    )


def resolve_request_for_selection(
    *,
    repository_root: Path | str,
    run_path: str,
    selected_models: Sequence[str],
) -> PreparationRequest | None:
    validate_selected_models(selected_models)
    return resolve_existing_preparation_request(
        repository_root=Path(repository_root).resolve(),
        run_path=run_path,
        selected_models=selected_models,
    )


def authorized_preparation_argv_values(request_relative_path: str) -> dict[str, str]:
    relative = str(request_relative_path).replace("\\", "/")
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise CandidatePreparationError(
            f"Unsafe request path: {request_relative_path!r}",
            reason_code="path_traversal",
        )
    return {"request": relative}


def verify_validation_result(
    payload: Mapping[str, Any],
    *,
    request_id: str,
    run_path: str,
    selected_models: Sequence[str],
) -> dict[str, Any]:
    if payload.get("ok") is not True:
        raise CandidatePreparationError(
            str(payload.get("error") or "Validation failed."),
            reason_code=str(payload.get("reason_code") or "validation_failed"),
        )
    if payload.get("command") not in {None, "validate"}:
        raise CandidatePreparationError(
            f"Unexpected command in validation result: {payload.get('command')!r}.",
            reason_code="validation_command_invalid",
        )
    if payload.get("artifacts_written") is not False:
        raise CandidatePreparationError(
            "Validation result must report artifacts_written=false.",
            reason_code="validation_artifacts_unexpected",
        )
    if payload.get("all_models_valid") is not True:
        raise CandidatePreparationError(
            "Validation result must report all_models_valid=true.",
            reason_code="all_models_valid_false",
        )
    if str(payload.get("request_id") or "") != request_id:
        raise CandidatePreparationError(
            "Validation result request_id mismatch.",
            reason_code="request_id_mismatch",
        )
    if str(payload.get("run_path") or "") != run_path:
        raise CandidatePreparationError(
            "Validation result run_path mismatch.",
            reason_code="run_path_mismatch",
        )
    actual_models = [str(item) for item in (payload.get("selected_models") or [])]
    expected_models = list(selected_models)
    if actual_models != expected_models:
        raise CandidatePreparationError(
            "Validation result selected model list mismatch.",
            reason_code="selected_models_mismatch",
        )
    results = payload.get("results") or []
    if not isinstance(results, list):
        raise CandidatePreparationError(
            "Validation results must be a list.",
            reason_code="validation_result_malformed",
        )
    if len(results) != len(expected_models):
        raise CandidatePreparationError(
            "Validation result model count mismatch.",
            reason_code="validation_result_incomplete",
        )
    seen_models: list[str] = []
    for index, item in enumerate(results):
        if not isinstance(item, Mapping):
            raise CandidatePreparationError(
                "Malformed per-model validation result.",
                reason_code="validation_result_malformed",
            )
        if item.get("ok") is not True:
            raise CandidatePreparationError(
                "One or more model validations failed.",
                reason_code="model_validation_failed",
            )
        model_name = str(item.get("model_name") or "")
        if model_name != expected_models[index]:
            raise CandidatePreparationError(
                "Per-model validation result order/name mismatch.",
                reason_code="selected_models_mismatch",
            )
        if model_name in seen_models:
            raise CandidatePreparationError(
                "Duplicate model validation result.",
                reason_code="duplicate_model_result",
            )
        seen_models.append(model_name)
        if item.get("train_row_count") is None or item.get("test_row_count") is None:
            raise CandidatePreparationError(
                "Validation result missing row counts.",
                reason_code="validation_row_counts_missing",
            )
    return dict(payload)


def recover_validation_for_request(
    *,
    jobs_root: Path,
    request: PreparationRequest,
) -> dict[str, Any] | None:
    """Recover a succeeded validation result for an authenticated request."""
    related = list_jobs_for_preparation_request(jobs_root, request.request_id)
    validate_jobs = [
        item
        for item in related
        if item.get("action_id") == "validate"
        and str((item.get("references") or {}).get("source_type") or "")
        == "managed_autogluon"
        and str((item.get("references") or {}).get("run_path") or "")
        == request.run_path
    ]
    if not validate_jobs:
        return None
    latest = validate_jobs[0]
    state = latest.get("state")
    if state != "succeeded":
        return {
            "recovery_state": state or "unknown",
            "job_id": latest.get("job_id"),
            "payload": None,
        }
    stdout_path = jobs_root / str(latest["job_id"]) / "stdout.log"
    try:
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        payload = parse_job_stdout_json(stdout)
        verified = verify_validation_result(
            payload,
            request_id=request.request_id,
            run_path=request.run_path,
            selected_models=request.selected_models,
        )
    except (CandidatePreparationError, OSError, ValueError, json.JSONDecodeError) as error:
        raise CandidatePreparationError(
            f"Validation job recovery failed: {error}",
            reason_code=getattr(error, "reason_code", "validation_recovery_failed"),
        ) from error
    return {
        "recovery_state": "succeeded",
        "job_id": latest.get("job_id"),
        "payload": verified,
    }


def durable_preparation_context_from_session(session_state: Any) -> dict[str, Any] | None:
    value = get_durable_value(session_state, DURABLE_CONTEXT_KEY)
    if not isinstance(value, dict):
        return None
    run_path = str(value.get("run_path") or "")
    models = value.get("selected_models")
    if not run_path or not isinstance(models, list) or not models:
        return None
    return {
        "run_path": run_path,
        "selected_models": [str(item) for item in models],
        "request_id": str(value.get("request_id") or "") or None,
    }


def persist_preparation_context(
    session_state: Any,
    *,
    run_path: str,
    selected_models: Sequence[str],
    request_id: str | None = None,
) -> None:
    set_durable_value(
        session_state,
        DURABLE_CONTEXT_KEY,
        {
            "run_path": run_path,
            "selected_models": list(selected_models),
            "request_id": request_id,
        },
    )


def restore_selection_from_jobs(
    *,
    jobs_root: Path,
    repository_root: Path,
    available_run_paths: Sequence[str],
    preferred_run_path: str | None = None,
) -> dict[str, Any] | None:
    """Restore ordered models from the latest non-stale preparation Job/request."""
    root = Path(repository_root).resolve()
    available = set(available_run_paths)
    jobs = list_preparation_jobs(
        jobs_root,
        run_path=preferred_run_path,
        source_type="managed_autogluon",
    )
    for job in jobs:
        references = job.get("references") or {}
        run_path = str(references.get("run_path") or "")
        request_id = str(references.get("request_id") or "")
        if not run_path or run_path not in available or not request_id:
            continue
        try:
            request = load_preparation_request(request_id, repository_root=root)
            verify_request_run_identity(request, repository_root=root)
        except Exception:  # noqa: BLE001 - skip stale/invalid recovery candidates
            continue
        if request.run_path != run_path:
            continue
        inventory = inventory_autogluon_run(
            resolve_under_repository(run_path, root),
            repository_root=root,
        )
        if any(name not in inventory.model_names for name in request.selected_models):
            continue
        return {
            "run_path": request.run_path,
            "selected_models": list(request.selected_models),
            "request_id": request.request_id,
            "job_id": job.get("job_id"),
            "action_id": job.get("action_id"),
        }
    return None


def list_preparation_jobs(
    jobs_root: Path,
    *,
    run_path: str | None = None,
    request_id: str | None = None,
    source_type: str | None = "managed_autogluon",
    command_id: str = PREPARATION_COMMAND_ID,
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
        if source_type is not None and str(references.get("source_type") or "") != source_type:
            continue
        if request_id is not None and str(references.get("request_id") or "") != request_id:
            continue
        if run_path is not None and str(references.get("run_path") or "") != run_path:
            continue
        status_payload: dict[str, Any] = {}
        status_path = entry / "status.json"
        if status_path.is_file():
            try:
                loaded = json.loads(status_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    status_payload = loaded
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
    matches.sort(key=lambda item: str(item.get("created_at_utc") or ""), reverse=True)
    return matches


def verify_preparation_result(
    payload: Mapping[str, Any],
    *,
    request_id: str,
    run_path: str,
    selected_models: Sequence[str],
    repository_root: Path | str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    if payload.get("ok") is not True:
        raise CandidatePreparationError(
            str(payload.get("error") or "Preparation failed."),
            reason_code=str(payload.get("reason_code") or "preparation_failed"),
        )
    if payload.get("command") not in {None, "import"}:
        raise CandidatePreparationError(
            f"Unexpected command in preparation result: {payload.get('command')!r}.",
            reason_code="preparation_command_invalid",
        )
    if str(payload.get("request_id") or "") != request_id:
        raise CandidatePreparationError(
            "Preparation result request_id mismatch.",
            reason_code="request_id_mismatch",
        )
    if str(payload.get("run_path") or "") != run_path:
        raise CandidatePreparationError(
            "Preparation result run_path mismatch.",
            reason_code="run_path_mismatch",
        )
    imported = payload.get("imported") or []
    if not isinstance(imported, list) or not imported:
        raise CandidatePreparationError(
            "Preparation result missing imported candidates.",
            reason_code="imported_missing",
        )
    actual_models = [str(item.get("source_model_name")) for item in imported]
    if sorted(actual_models) != sorted(selected_models):
        raise CandidatePreparationError(
            "Preparation result imported model set mismatch.",
            reason_code="selected_models_mismatch",
        )
    summaries: list[dict[str, Any]] = []
    for item in imported:
        candidate_id = str(item.get("candidate_id") or "")
        if not candidate_id:
            raise CandidatePreparationError(
                "Imported candidate missing candidate_id.",
                reason_code="candidate_id_missing",
            )
        package_dir = resolve_under_repository(
            f"artifacts/prediction_candidates/{candidate_id}",
            root,
        )
        if not (package_dir / SUCCESS_FILENAME).is_file():
            raise CandidatePreparationError(
                f"Imported candidate missing _SUCCESS: {candidate_id}",
                reason_code="success_missing",
            )
        package = load_candidate_package(
            package_dir,
            repository_root=root,
            candidates_root_relative="artifacts/prediction_candidates",
        )
        if package.candidate_id != candidate_id:
            raise CandidatePreparationError(
                "Candidate ID mismatch after import.",
                reason_code="candidate_id_mismatch",
            )
        readiness = evaluate_submission_readiness(
            candidate_id, repository_root=root
        )
        oof = package.oof["probability_positive"]
        test = package.test["probability_positive"]
        summaries.append(
            {
                "model": package.manifest.get("source_model_name"),
                "candidate_id": candidate_id,
                "dataset_id": package.manifest.get("dataset_id"),
                "oof_rows": int(len(package.oof)),
                "test_rows": int(len(package.test)),
                "oof_probability_range": [float(oof.min()), float(oof.max())],
                "test_probability_range": [float(test.min()), float(test.max())],
                "source_metric": package.manifest.get("source_metric_value"),
                "exploratory": bool(package.manifest.get("exploratory")),
                "blend_ready": True,
                "submission_readiness": readiness.get("state"),
                "submission_blockers": readiness.get("blockers") or [],
                "manifest_sha256": file_sha256(package_dir / MANIFEST_FILENAME),
            }
        )
    return {"payload": dict(payload), "candidates": summaries}


def list_jobs_for_preparation_request(
    jobs_root: Path,
    request_id: str,
    *,
    command_id: str = "autogluon_candidate_preparation_v1",
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
                loaded = json.loads(status_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    status_payload = loaded
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
    matches.sort(key=lambda item: str(item.get("created_at_utc") or ""), reverse=True)
    return matches


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "CACHE_CONTRACT_VERSION",
    "DURABLE_CONTEXT_KEY",
    "INDEPENDENT_CANDIDATE_COPY",
    "MAX_MODELS",
    "MIN_MODELS",
    "PREPARATION_COMMAND_ID",
    "CandidatePreparationError",
    "authorized_preparation_argv_values",
    "build_model_rows",
    "discover_managed_autogluon_runs",
    "discover_managed_run_directories",
    "durable_preparation_context_from_session",
    "ensemble_component_warning",
    "filter_managed_runs",
    "independent_candidate_summary",
    "list_jobs_for_preparation_request",
    "list_preparation_jobs",
    "managed_runs_inventory_fingerprint",
    "package_status_label",
    "persist_preparation_context",
    "prepare_preparation_request",
    "preparation_button_label",
    "preview_independent_candidate_rows",
    "preview_request_for_selection",
    "recover_validation_for_request",
    "resolve_request_for_selection",
    "restore_selection_from_jobs",
    "summarize_managed_run",
    "verify_preparation_result",
    "verify_validation_result",
]
