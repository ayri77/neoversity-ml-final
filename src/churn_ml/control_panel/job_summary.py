"""Structured Control Panel Job summaries for known workflows.

Reuses existing stdout parsers and verifiers. Malformed or partial results never
crash the Jobs page; they return an unavailable/pending summary while logs remain
accessible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.churn_ml.control_panel.blend_workspace import (
    BlendWorkspaceError,
    parse_job_stdout_json,
    verify_search_result,
)
from src.churn_ml.control_panel.candidate_display import (
    CandidateDisplay,
    candidate_display_map,
    project_weight_rows,
    resolve_candidate_displays,
)
from src.churn_ml.control_panel.candidate_preparation import (
    CandidatePreparationError,
    verify_preparation_result,
    verify_validation_result,
)
from src.churn_ml.control_panel.job_logs import inspect_job_log, read_job_log_full


@dataclass(frozen=True)
class JobSummary:
    title: str
    status: str
    rows: tuple[tuple[str, Any], ...]
    tables: tuple[tuple[str, tuple[dict[str, Any], ...]], ...] = ()
    notes: tuple[str, ...] = ()
    technical: Mapping[str, Any] = field(default_factory=dict)
    available: bool = True
    pending: bool = False


def _split_csv_refs(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _duration_seconds(status: Mapping[str, Any]) -> Any:
    return status.get("elapsed_seconds")


def generic_job_summary(
    job: Mapping[str, Any],
    status: Mapping[str, Any],
) -> JobSummary:
    rows = (
        ("Command", job.get("command_id")),
        ("Action", job.get("action_id")),
        ("State", status.get("state")),
        ("Created", job.get("created_at_utc")),
        ("Started", status.get("started_at_utc") or job.get("started_at_utc")),
        ("Completed", status.get("completed_at_utc") or job.get("completed_at_utc")),
        ("Duration", _duration_seconds(status)),
        ("Exit code", status.get("exit_code")),
    )
    return JobSummary(
        title="Job summary",
        status=str(status.get("state") or "unknown"),
        rows=rows,
        technical={"job": dict(job), "status": dict(status)},
    )


def lightweight_job_list_label(job: Mapping[str, Any]) -> str | None:
    """Cheap selectbox label from job.json references only (no artifact I/O)."""
    command_id = str(job.get("command_id") or "")
    action_id = str(job.get("action_id") or "")
    references = job.get("references")
    if not isinstance(references, Mapping):
        references = {}

    if command_id == "prediction_blend_v1" and action_id == "search":
        optimizer = str(references.get("optimizer") or "search").title()
        if references.get("optimizer") == "native":
            optimizer = "Native"
        elif references.get("optimizer") == "optuna":
            optimizer = "Optuna"
        count = len(_split_csv_refs(references.get("candidate_ids")))
        return f"Blend search · {optimizer} · {count} models"

    if command_id == "prediction_blend_v1" and action_id == "materialize":
        optimizer = str(references.get("optimizer") or "blend").title()
        if references.get("optimizer") == "native":
            optimizer = "Native"
        elif references.get("optimizer") == "optuna":
            optimizer = "Optuna"
        count = len(_split_csv_refs(references.get("candidate_ids")))
        return f"Materialize blend · {optimizer} · {count} models"

    if command_id == "autogluon_candidate_preparation_v1" and action_id == "validate":
        count = len(_split_csv_refs(references.get("selected_models")))
        return f"Validate AutoGluon candidates · {count} models"

    if command_id == "autogluon_candidate_preparation_v1" and action_id in {
        "prepare",
        "import",
    }:
        count = len(_split_csv_refs(references.get("selected_models")))
        return f"Prepare AutoGluon candidates · {count} models"

    if command_id == "candidate_submission_v1":
        source = references.get("source_label") or references.get("candidate_id")
        if source:
            return f"Generate submission · {source}"
        return "Generate submission"

    return None


def _try_parse_stdout(stdout_text: str) -> tuple[dict[str, Any] | None, str | None]:
    text = stdout_text.strip()
    if not text:
        return None, "pending"
    try:
        payload = parse_job_stdout_json(stdout_text)
        return payload, None
    except BlendWorkspaceError as error:
        code = getattr(error, "reason_code", "") or ""
        if code in {"stdout_empty"}:
            return None, "pending"
        if code in {"stdout_json_invalid"}:
            # Incomplete trailing JSON while the job is still writing.
            if not text.endswith("}"):
                return None, "pending"
            return None, "unavailable"
        return None, "unavailable"
    except Exception:
        return None, "unavailable"


def _display_map_for_ids(
    candidate_ids: Sequence[str],
    *,
    repository_root: Path | str | None,
) -> dict[str, CandidateDisplay]:
    if repository_root is None:
        return {
            candidate_id: CandidateDisplay(
                candidate_id=candidate_id,
                primary_label=candidate_id,
                compact_label=candidate_id,
                model_name="",
                source_label="",
                dataset_label="",
                dataset_id="",
                source_kind="",
                exploratory=False,
                missing_reason="Repository root unavailable",
            )
            for candidate_id in candidate_ids
        }
    return candidate_display_map(candidate_ids, repository_root=repository_root)


def _summarize_blend_search(
    *,
    job: Mapping[str, Any],
    status: Mapping[str, Any],
    payload: Mapping[str, Any],
    repository_root: Path | str | None,
) -> JobSummary:
    references = job.get("references") if isinstance(job.get("references"), Mapping) else {}
    request_id = str(references.get("request_id") or payload.get("request_id") or "")
    expected_blend_id = references.get("expected_blend_id")
    try:
        verified = verify_search_result(
            payload,
            request_id=request_id,
            expected_blend_id=str(expected_blend_id) if expected_blend_id else None,
        )
    except BlendWorkspaceError as error:
        return JobSummary(
            title="Structured summary unavailable",
            status="unavailable",
            rows=(("Reason", str(error)),),
            technical={"error": str(error), "payload": dict(payload)},
            available=False,
        )

    candidate_ids = [str(item) for item in (verified.get("candidate_ids") or [])]
    if not candidate_ids:
        candidate_ids = _split_csv_refs(references.get("candidate_ids"))
    displays = _display_map_for_ids(candidate_ids, repository_root=repository_root)
    ordered_models = [
        displays[cid].primary_label if cid in displays else cid for cid in candidate_ids
    ]
    honest = verified.get("honest_meta_cv_metrics") or {}
    pooled = honest.get("pooled_repeated_held_out_confusion_metrics") or {}
    weights = verified.get("final_deployment_weights") or (
        (verified.get("deployment") or {}).get("weights")
    )
    weight_rows = tuple(project_weight_rows(weights, displays))
    optimizer = str(
        verified.get("optimizer_backend") or references.get("optimizer") or ""
    )
    strategy = str(verified.get("strategy") or references.get("strategy") or "")
    settings = verified.get("optimizer_settings") or verified.get("search_budget") or {}
    folds = settings.get("folds") if isinstance(settings, Mapping) else None
    repeats = settings.get("repeats") if isinstance(settings, Mapping) else None
    max_active = settings.get("max_active_models") if isinstance(settings, Mapping) else None
    rows = (
        ("Search status", "Succeeded" if verified.get("ok") is True else status.get("state")),
        ("Strategy", strategy),
        ("Optimizer", optimizer),
        ("Ordered models", " | ".join(ordered_models)),
        ("Meta-CV folds × repeats", f"{folds} × {repeats}" if folds and repeats else None),
        ("Maximum active models", max_active),
        ("Honest mean BA", honest.get("mean_repeat_balanced_accuracy")),
        ("BA std", honest.get("std_repeat_balanced_accuracy")),
        ("Min BA", honest.get("min_repeat_balanced_accuracy")),
        ("Max BA", honest.get("max_repeat_balanced_accuracy")),
        ("Sensitivity", pooled.get("sensitivity") if isinstance(pooled, Mapping) else None),
        ("Specificity", pooled.get("specificity") if isinstance(pooled, Mapping) else None),
        (
            "Final threshold",
            verified.get("final_deployment_threshold")
            or (verified.get("deployment") or {}).get("threshold"),
        ),
        ("Request ID", verified.get("request_id") or request_id),
        ("Blend ID", verified.get("blend_id")),
        (
            "Artifacts written",
            "Yes" if verified.get("artifacts_written") is True else "No",
        ),
    )
    notes = (
        "Honest meta-CV is the primary unbiased estimate.",
        "Descriptive full-OOF evidence is separate and not the primary estimate.",
    )
    return JobSummary(
        title="Blend search",
        status="succeeded",
        rows=rows,
        tables=(("Final weights", weight_rows),),
        notes=notes,
        technical={
            "honest_meta_cv_metrics": honest,
            "cross_fitted_probability_descriptive_metrics": verified.get(
                "cross_fitted_probability_descriptive_metrics"
            ),
            "full_oof_descriptive_metrics": verified.get("full_oof_descriptive_metrics"),
            "final_deployment_weights": weights,
            "optuna_study_summaries": verified.get("optuna_study_summaries"),
            "payload": dict(verified),
        },
    )


def _summarize_blend_materialize(
    *,
    job: Mapping[str, Any],
    payload: Mapping[str, Any],
    repository_root: Path | str | None,
) -> JobSummary:
    if payload.get("ok") is not True:
        return JobSummary(
            title="Structured summary unavailable",
            status="unavailable",
            rows=(("Reason", payload.get("error") or "Materialize failed"),),
            technical={"payload": dict(payload)},
            available=False,
        )
    candidate_ids = [str(item) for item in (payload.get("candidate_ids") or [])]
    references = job.get("references") if isinstance(job.get("references"), Mapping) else {}
    if not candidate_ids:
        candidate_ids = _split_csv_refs(references.get("candidate_ids"))
    displays = _display_map_for_ids(candidate_ids, repository_root=repository_root)
    weights = payload.get("final_deployment_weights") or (
        (payload.get("deployment") or {}).get("weights")
    )
    honest = payload.get("honest_meta_cv_metrics") or {}
    readiness = payload.get("submission_readiness") or {}
    canonical = payload.get("canonical_candidate_id")
    canonical_label = canonical
    if canonical and repository_root is not None:
        canonical_label = resolve_candidate_displays(
            [str(canonical)], repository_root=repository_root
        )[0].primary_label
    rows = (
        ("Blend ID", payload.get("blend_id")),
        ("Canonical candidate", canonical_label),
        (
            "Models",
            " | ".join(
                displays[cid].primary_label if cid in displays else cid
                for cid in candidate_ids
            ),
        ),
        ("Optimizer", payload.get("optimizer_backend") or references.get("optimizer")),
        ("Honest BA", honest.get("mean_repeat_balanced_accuracy")),
        (
            "Final threshold",
            payload.get("final_deployment_threshold")
            or (payload.get("deployment") or {}).get("threshold"),
        ),
        (
            "Submission readiness",
            readiness.get("state") if isinstance(readiness, Mapping) else readiness,
        ),
        (
            "Artifacts written",
            "Yes" if payload.get("artifacts_written") is not False else "No",
        ),
    )
    return JobSummary(
        title="Materialize blend",
        status="succeeded",
        rows=rows,
        tables=(("Final weights", tuple(project_weight_rows(weights, displays))),),
        technical={"payload": dict(payload), "canonical_candidate_id": canonical},
    )


def _summarize_prep_validate(
    *,
    job: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> JobSummary:
    references = job.get("references") if isinstance(job.get("references"), Mapping) else {}
    request_id = str(references.get("request_id") or payload.get("request_id") or "")
    run_path = str(references.get("run_path") or payload.get("run_path") or "")
    selected = _split_csv_refs(
        payload.get("selected_models") or references.get("selected_models")
    )
    try:
        verified = verify_validation_result(
            payload,
            request_id=request_id,
            run_path=run_path,
            selected_models=selected,
        )
    except CandidatePreparationError as error:
        # Still surface readable counts when partial payload exists.
        results = payload.get("results") or []
        valid_count = 0
        if isinstance(results, list):
            valid_count = sum(
                1
                for item in results
                if isinstance(item, Mapping) and item.get("ok") is True
            )
        return JobSummary(
            title="Structured summary unavailable",
            status="unavailable",
            rows=(
                ("Reason", str(error)),
                ("Request", request_id),
                ("Run", run_path),
                ("Models validated", f"{valid_count} of {len(selected)}"),
            ),
            technical={"error": str(error), "payload": dict(payload)},
            available=False,
        )
    results = verified.get("results") or []
    total = len(selected) or (
        len(results) if isinstance(results, list) else 0
    )
    valid_count = total
    if isinstance(results, list) and results:
        valid_count = sum(
            1 for item in results if isinstance(item, Mapping) and item.get("ok") is True
        )
    train_rows = None
    test_rows = None
    positive = None
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, Mapping):
                continue
            train_rows = train_rows or item.get("train_row_count")
            test_rows = test_rows or item.get("test_row_count")
            positive = positive or item.get("positive_class_label")
    rows = (
        ("Request", verified.get("request_id") or request_id),
        ("Run", verified.get("run_path") or run_path),
        ("Models validated", f"{valid_count} of {total}"),
        (
            "Artifacts written",
            "Yes" if verified.get("artifacts_written") is True else "No",
        ),
        ("Train rows", train_rows),
        ("Test rows", test_rows),
        ("Positive class", positive),
        ("Selected models", " | ".join(selected)),
    )
    return JobSummary(
        title="Validate AutoGluon candidates",
        status="succeeded",
        rows=rows,
        technical={"payload": dict(verified)},
    )


def _summarize_prep_prepare(
    *,
    job: Mapping[str, Any],
    payload: Mapping[str, Any],
    repository_root: Path | str | None,
) -> JobSummary:
    references = job.get("references") if isinstance(job.get("references"), Mapping) else {}
    request_id = str(references.get("request_id") or payload.get("request_id") or "")
    run_path = str(references.get("run_path") or payload.get("run_path") or "")
    selected = _split_csv_refs(
        payload.get("selected_models") or references.get("selected_models")
    )
    if repository_root is None:
        verified = dict(payload)
    else:
        try:
            verified = verify_preparation_result(
                payload,
                request_id=request_id,
                run_path=run_path,
                selected_models=selected,
                repository_root=repository_root,
            )
        except CandidatePreparationError as error:
            return JobSummary(
                title="Structured summary unavailable",
                status="unavailable",
                rows=(("Reason", str(error)),),
                technical={"error": str(error), "payload": dict(payload)},
                available=False,
            )
    imported = verified.get("imported") or []
    table: list[dict[str, Any]] = []
    if isinstance(imported, list):
        for item in imported:
            if not isinstance(item, Mapping):
                continue
            table.append(
                {
                    "Model": item.get("source_model_name") or item.get("model"),
                    "Candidate ID": item.get("candidate_id"),
                    "Train rows": item.get("train_row_count"),
                    "Test rows": item.get("test_row_count"),
                    "Success marker": item.get("success")
                    if "success" in item
                    else True,
                }
            )
    rows = (
        ("Request", verified.get("request_id") or request_id),
        ("Independent candidates prepared", len(table)),
        (
            "Artifacts written",
            "Yes" if verified.get("artifacts_written") is not False else "No",
        ),
        ("Run", verified.get("run_path") or run_path),
    )
    return JobSummary(
        title="Prepare AutoGluon candidates",
        status="succeeded",
        rows=rows,
        tables=(("Prepared candidates", tuple(table)),),
        technical={"payload": dict(verified)},
    )


def _summarize_submission(
    *,
    payload: Mapping[str, Any],
    repository_root: Path | str | None,
) -> JobSummary:
    if payload.get("ok") is not True and "submission_path" not in payload:
        return JobSummary(
            title="Structured summary unavailable",
            status="unavailable",
            rows=(("Reason", payload.get("error") or "Submission result unavailable"),),
            technical={"payload": dict(payload)},
            available=False,
        )
    candidate_id = payload.get("candidate_id") or payload.get("source_candidate_id")
    source_label = candidate_id
    if candidate_id and repository_root is not None:
        source_label = resolve_candidate_displays(
            [str(candidate_id)], repository_root=repository_root
        )[0].primary_label
    validation = payload.get("validation")
    validation_state = payload.get("validation_state")
    if validation_state is None and isinstance(validation, Mapping):
        validation_state = validation.get("state")
    rows = (
        ("Source candidate", source_label),
        ("Source blend", payload.get("blend_id") or payload.get("source_blend_id")),
        ("Threshold", payload.get("threshold") or payload.get("final_threshold")),
        ("Rows", payload.get("row_count") or payload.get("rows")),
        (
            "Positive prediction rate",
            payload.get("positive_prediction_rate"),
        ),
        ("Submission path", payload.get("submission_path") or payload.get("path")),
        ("Validation state", validation_state),
    )
    return JobSummary(
        title="Candidate submission",
        status="succeeded" if payload.get("ok") is True else str(payload.get("ok")),
        rows=rows,
        technical={"payload": dict(payload), "candidate_id": candidate_id},
    )


def build_job_summary(
    *,
    job: Mapping[str, Any],
    status: Mapping[str, Any],
    job_root: Path | str,
    repository_root: Path | str | None = None,
    stdout_text: str | None = None,
) -> JobSummary:
    command_id = str(job.get("command_id") or "")
    action_id = str(job.get("action_id") or "")
    state = str(status.get("state") or "")
    root = Path(job_root)
    if stdout_text is None:
        stdout_path = root / "stdout.log"
        info = inspect_job_log(stdout_path)
        if not info.exists:
            if state in {"running", "queued", "starting"}:
                summary = generic_job_summary(job, status)
                return JobSummary(
                    title=summary.title,
                    status=state,
                    rows=summary.rows + (("Structured summary", "not available yet"),),
                    pending=True,
                    technical=summary.technical,
                )
            return JobSummary(
                title="Structured summary unavailable",
                status="unavailable",
                rows=(("Reason", "stdout.log is missing"),),
                available=False,
                technical=dict(generic_job_summary(job, status).technical),
            )
        stdout_text = read_job_log_full(stdout_path)

    payload, parse_state = _try_parse_stdout(stdout_text)
    if payload is None:
        running = state in {"running", "queued", "starting"}
        if running or (parse_state == "pending" and state not in {"succeeded", "failed", "stopped", "orphaned"}):
            summary = generic_job_summary(job, status)
            return JobSummary(
                title=summary.title,
                status=state,
                rows=summary.rows + (("Structured summary", "not available yet"),),
                pending=True,
                technical=summary.technical,
            )
        return JobSummary(
            title="Structured summary unavailable",
            status="unavailable",
            rows=generic_job_summary(job, status).rows
            + (("Reason", "Structured result could not be parsed"),),
            available=False,
            technical=dict(generic_job_summary(job, status).technical),
        )

    try:
        if command_id == "prediction_blend_v1" and action_id == "search":
            return _summarize_blend_search(
                job=job,
                status=status,
                payload=payload,
                repository_root=repository_root,
            )
        if command_id == "prediction_blend_v1" and action_id == "materialize":
            return _summarize_blend_materialize(
                job=job,
                payload=payload,
                repository_root=repository_root,
            )
        if command_id == "autogluon_candidate_preparation_v1" and action_id == "validate":
            return _summarize_prep_validate(job=job, payload=payload)
        if command_id == "autogluon_candidate_preparation_v1" and action_id in {
            "prepare",
            "import",
        }:
            return _summarize_prep_prepare(
                job=job,
                payload=payload,
                repository_root=repository_root,
            )
        if command_id == "candidate_submission_v1":
            return _summarize_submission(
                payload=payload,
                repository_root=repository_root,
            )
    except Exception as error:  # noqa: BLE001 - Jobs UI must never crash
        return JobSummary(
            title="Structured summary unavailable",
            status="unavailable",
            rows=(("Reason", str(error)),),
            available=False,
            technical={"error": str(error), "payload": dict(payload)},
        )

    summary = generic_job_summary(job, status)
    return JobSummary(
        title=summary.title,
        status=summary.status,
        rows=summary.rows,
        technical={"payload": dict(payload), **dict(summary.technical)},
    )


def resolve_job_references(
    job: Mapping[str, Any],
    *,
    repository_root: Path | str | None = None,
) -> dict[str, Any]:
    """Build readable References plus technical raw values."""
    references = job.get("references")
    if not isinstance(references, Mapping):
        references = {}
    candidate_ids = _split_csv_refs(
        references.get("candidate_ids") or references.get("candidate_id")
    )
    parent_ids = _split_csv_refs(references.get("parent_candidate_ids"))
    canonical = references.get("canonical_candidate_id")
    if canonical:
        candidate_ids = list(dict.fromkeys([*candidate_ids, str(canonical)]))
    displays = _display_map_for_ids(
        list(dict.fromkeys([*candidate_ids, *parent_ids])),
        repository_root=repository_root,
    )
    readable_candidates = [
        displays[cid].primary_label if cid in displays else cid for cid in candidate_ids
    ]
    readable_parents = [
        displays[cid].primary_label if cid in displays else cid for cid in parent_ids
    ]
    selected_models = _split_csv_refs(references.get("selected_models"))
    return {
        "candidates": readable_candidates,
        "parents": readable_parents,
        "selected_models": selected_models,
        "source_model_name": references.get("source_model_name"),
        "blend_id": references.get("blend_id") or references.get("expected_blend_id"),
        "request_id": references.get("request_id"),
        "technical": dict(references),
    }


def summary_cache_fingerprint(
    *,
    job: Mapping[str, Any],
    status: Mapping[str, Any],
    stdout_size: int,
    stdout_mtime_ns: int,
) -> str:
    payload = {
        "job_id": job.get("job_id"),
        "command_id": job.get("command_id"),
        "action_id": job.get("action_id"),
        "state": status.get("state"),
        "exit_code": status.get("exit_code"),
        "stdout_size": stdout_size,
        "stdout_mtime_ns": stdout_mtime_ns,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


__all__ = [
    "JobSummary",
    "build_job_summary",
    "generic_job_summary",
    "lightweight_job_list_label",
    "resolve_job_references",
    "summary_cache_fingerprint",
]
