"""Streamlit Blend Workspace page rendering helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from src.churn_ml.blending.artifact_v1 import load_blend_artifact
from src.churn_ml.control_panel.blend_ui_request_v1 import (
    BlendUIRequestError,
    materialize_blend_ui_request,
    method_to_strategy_optimizer,
)
from src.churn_ml.prediction_candidates.contract_v1 import (
    file_sha256,
    resolve_under_repository,
)
from src.churn_ml.control_panel.blend_workspace import (
    CACHE_CONTRACT_VERSION,
    BlendWorkspaceError,
    apply_canonical_submission_handoff,
    authorized_blend_argv_values,
    candidate_inventory_fingerprint,
    discover_candidate_rows,
    discover_materialized_blends,
    expected_blend_id,
    filter_candidate_rows,
    list_jobs_for_request,
    parse_job_stdout_json,
    readiness_label,
    run_compatibility,
    run_diversity,
    selection_fingerprint,
    verify_search_result,
)
from src.churn_ml.control_panel.command_builder import build_command
from src.churn_ml.control_panel.jobs import JobError, JobManager
from src.churn_ml.control_panel.launch import (
    LaunchAuthorizationError,
    RenderedLaunch,
    authorize_launch,
    rendered_launch,
)
from src.churn_ml.control_panel.registry import ControlPanelRegistry
from src.churn_ml.control_panel.selection_state import (
    set_durable_value,
    ui_durable_key,
)


try:
    import plotly.express as px  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    px = None  # type: ignore[assignment]


STATE_PREFIX = "blend_workspace"
METHOD_OPTIONS = {
    "Equal": "Equal weights",
    "manual": "Manual weights",
    "optimized_native": "Optimized — Native",
    "optimized_optuna": "Optimized — Optuna",
}


def _state_key(name: str) -> str:
    return f"{STATE_PREFIX}:{name}"


@st.cache_data(show_spinner="Loading canonical candidates…")
def _load_candidate_rows_cached(
    root: str, fingerprint: str, contract: str
) -> list[dict[str, Any]]:
    del fingerprint, contract
    return discover_candidate_rows(Path(root), include_readiness=True)


def _cached_candidate_rows(
    repository_root: Path,
    inventory_fingerprint: str,
) -> list[dict[str, Any]]:
    return _load_candidate_rows_cached(
        str(repository_root.resolve()),
        inventory_fingerprint,
        CACHE_CONTRACT_VERSION,
    )


def render_blend_workspace_page(
    *,
    repository_root: Path,
    registry: ControlPanelRegistry,
    job_manager: JobManager,
) -> None:
    st.title("Blend Workspace")
    st.caption(
        "Select compatible canonical prediction candidates, analyze diversity, "
        "evaluate a leakage-safe blend, and prepare the result for submission."
    )

    fingerprint = candidate_inventory_fingerprint(repository_root)
    if st.button("Refresh candidates", key=_state_key("refresh")):
        _load_candidate_rows_cached.clear()
        for key in (
            "compatibility",
            "diversity",
            "search_result",
            "request_path",
            "request_id",
            "request_obj",
            "expected_blend_id",
        ):
            st.session_state.pop(_state_key(key), None)
        st.session_state[_state_key("inventory_fp")] = fingerprint

    rows = _cached_candidate_rows(repository_root, fingerprint)
    _render_candidate_inventory(rows)
    selected_ids = _render_selection_controls(rows)
    selected_fp = selection_fingerprint(selected_ids)
    previous_fp = st.session_state.get(_state_key("selection_fp"))
    if previous_fp != selected_fp:
        for key in (
            "compatibility",
            "diversity",
            "search_result",
            "request_path",
            "request_id",
            "request_obj",
            "expected_blend_id",
        ):
            st.session_state.pop(_state_key(key), None)
        st.session_state[_state_key("selection_fp")] = selected_fp

    if len(selected_ids) < 2:
        st.warning(
            "At least two canonical candidates are required. "
            "Import or prepare another candidate before running a blend."
        )
        _render_materialized_blends(repository_root)
        return
    if len(selected_ids) > 10:
        st.error("At most 10 candidates can be selected.")
        return

    _render_selected_details(rows, selected_ids)
    compatibility = _render_compatibility(repository_root, selected_ids)
    if compatibility is None or not compatibility.get("ok"):
        _render_materialized_blends(repository_root)
        return

    _render_diversity(repository_root, selected_ids)
    request = _render_configuration_and_request(repository_root, selected_ids, rows)
    if request is not None:
        _render_search_and_materialize(
            repository_root=repository_root,
            registry=registry,
            job_manager=job_manager,
            request=request,
            rows=rows,
        )
    _render_materialized_blends(repository_root)


def _render_candidate_inventory(rows: list[dict[str, Any]]) -> None:
    st.subheader("Candidate inventory")
    if not rows:
        st.info("No canonical prediction candidates are available yet.")
        return
    source_labels = {
        str(row.get("source_kind_label") or row.get("source_kind") or ""): str(
            row.get("source_kind") or ""
        )
        for row in rows
    }
    sources = sorted(source_labels)
    datasets = sorted({str(row.get("dataset_id") or "") for row in rows})
    cols = st.columns(5)
    source = cols[0].selectbox("Source", ["All", *sources], key=_state_key("filter_source"))
    dataset = cols[1].selectbox("Dataset", ["All", *datasets], key=_state_key("filter_dataset"))
    exploratory = cols[2].selectbox(
        "Exploratory",
        ["All", "Non-exploratory", "Exploratory"],
        key=_state_key("filter_exploratory"),
    )
    readiness = cols[3].selectbox(
        "Submission readiness",
        ["All", "Ready", "Blocked"],
        key=_state_key("filter_readiness"),
    )
    search_text = cols[4].text_input("Search", key=_state_key("filter_search"))
    filtered = filter_candidate_rows(
        rows,
        source=None if source == "All" else source_labels.get(source),
        dataset=None if dataset == "All" else dataset,
        exploratory=(
            None
            if exploratory == "All"
            else exploratory == "Exploratory"
        ),
        readiness=(
            None
            if readiness == "All"
            else ("ready" if readiness == "Ready" else "blocked")
        ),
        search_text=search_text or None,
    )
    table = pd.DataFrame(
        [
            {
                "Model": row.get("label"),
                "Source": row.get("source_kind_label"),
                "Dataset": row.get("dataset_id"),
                "Metric": row.get("source_metric_value"),
                "Exploratory": row.get("exploratory"),
                "OOF protocol": row.get("oof_protocol"),
                "Submission ready": readiness_label(
                    str(row.get("readiness_state") or "unknown")
                ),
                "Candidate ID": row.get("candidate_id"),
            }
            for row in filtered
        ]
    )
    st.dataframe(table, width="stretch", hide_index=True)
    st.session_state[_state_key("filtered_rows")] = filtered


def _render_selection_controls(rows: list[dict[str, Any]]) -> list[str]:
    filtered = st.session_state.get(_state_key("filtered_rows")) or rows
    options = [str(row["candidate_id"]) for row in filtered]
    labels = {
        str(row["candidate_id"]): str(row.get("label") or row["candidate_id"])
        for row in rows
    }
    selected = st.multiselect(
        "Selected candidates (2–10, order preserved)",
        options=options,
        max_selections=10,
        key=_state_key("selected_ids"),
        format_func=lambda value: labels.get(value, value),
    )
    return list(selected)


def _render_selected_details(
    rows: list[dict[str, Any]], selected_ids: list[str]
) -> None:
    by_id = {str(row["candidate_id"]): row for row in rows}
    for candidate_id in selected_ids:
        row = by_id.get(candidate_id)
        if row is None:
            continue
        with st.expander(str(row.get("label") or candidate_id), expanded=False):
            st.write(
                {
                    "Model": row.get("source_model_name"),
                    "Source": row.get("source_kind_label"),
                    "Dataset": row.get("dataset_id"),
                    "Source metric": row.get("source_metric_value"),
                    "Target dependency": row.get("target_dependency"),
                    "OOF protocol": row.get("oof_protocol"),
                    "Probability range": (
                        f"{row.get('probability_min')} – {row.get('probability_max')}"
                    ),
                    "Train rows": row.get("train_row_count"),
                    "Test rows": row.get("test_row_count"),
                    "Exploratory": row.get("exploratory"),
                    "Submission readiness": readiness_label(
                        str(row.get("readiness_state") or "unknown")
                    ),
                }
            )
            with st.expander("Technical details", expanded=False):
                st.json(
                    {
                        "candidate_id": candidate_id,
                        "manifest_sha256": row.get("manifest_sha256"),
                        "success_sha256": row.get("success_sha256"),
                        "readiness_blockers": row.get("readiness_blockers"),
                        "final_threshold": row.get("final_threshold"),
                    }
                )


def _render_compatibility(
    repository_root: Path, selected_ids: list[str]
) -> dict[str, Any] | None:
    st.subheader("Compatibility")
    if st.button("Validate candidates", key=_state_key("validate")):
        try:
            result = run_compatibility(selected_ids, repository_root)
            st.session_state[_state_key("compatibility")] = result
        except (BlendWorkspaceError, BlendUIRequestError, Exception) as error:
            st.error(str(error))
            st.session_state[_state_key("compatibility")] = {
                "ok": False,
                "summary": {"state": "Blocked", "reasons": [str(error)]},
            }
    result = st.session_state.get(_state_key("compatibility"))
    if not result:
        st.info("Validate the current selection before search or materialization.")
        return None
    summary = result.get("summary") or {}
    if result.get("ok"):
        st.success(f"Compatible · {summary.get('candidate_count')} candidates")
    else:
        st.error("Blocked")
        reason = summary.get("reason")
        if reason:
            st.write(f"- {reason}")
        for item in summary.get("reasons") or []:
            st.write(f"- {item}")
    st.write(
        {
            "Shared train identity": summary.get("train_anchor_hash"),
            "Shared test identity": summary.get("test_anchor_hash"),
            "Target identity": summary.get("target_hash"),
            "Positive class": summary.get("positive_class_label"),
            "Probability semantics": summary.get("probability_semantics"),
            "Exploratory": summary.get("exploratory"),
            "Allowed differences": summary.get("allowed_differences"),
        }
    )
    with st.expander("Technical details", expanded=False):
        st.json(result.get("raw") or result)
    return result


def _render_diversity(repository_root: Path, selected_ids: list[str]) -> None:
    st.subheader("Diversity analysis")
    if st.button("Analyze diversity", key=_state_key("analyze")):
        try:
            st.session_state[_state_key("diversity")] = run_diversity(
                selected_ids, repository_root
            )
        except Exception as error:  # noqa: BLE001
            st.error(str(error))
    result = st.session_state.get(_state_key("diversity"))
    if not result:
        return
    st.caption(
        "Candidate metrics below are descriptive. You remain responsible for "
        "candidate selection; highly correlated candidates are not removed automatically."
    )
    quality = result.get("candidate_quality") or result.get("candidate_metrics") or []
    if quality:
        st.markdown("#### Candidate quality")
        st.dataframe(pd.DataFrame(quality), width="stretch", hide_index=True)
    pairwise = result.get("pairwise") or result.get("pairwise_analysis") or []
    if pairwise:
        st.markdown("#### Pairwise diversity")
        frame = pd.DataFrame(pairwise)
        st.dataframe(frame, width="stretch", hide_index=True)
        if px is not None and {"candidate_a", "candidate_b", "pearson_correlation"}.issubset(
            set(frame.columns)
        ):
            try:
                heat = frame.pivot(
                    index="candidate_a",
                    columns="candidate_b",
                    values="pearson_correlation",
                )
                st.plotly_chart(
                    px.imshow(heat, title="Pearson correlation"),
                    width="stretch",
                )
            except Exception:  # noqa: BLE001
                pass
    with st.expander("Technical details", expanded=False):
        st.json(result.get("raw") or result)


def _render_configuration_and_request(
    repository_root: Path,
    selected_ids: list[str],
    rows: list[dict[str, Any]],
):
    st.subheader("Blend configuration")
    with st.form(_state_key("config_form")):
        method = st.radio(
            "Method",
            list(METHOD_OPTIONS),
            format_func=lambda value: METHOD_OPTIONS[value],
            key=_state_key("method"),
        )
        folds = st.number_input("Meta-CV folds", min_value=2, max_value=20, value=5)
        repeats = st.number_input("Meta-CV repeats", min_value=1, max_value=20, value=2)
        seed = st.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=42)
        max_active = st.number_input(
            "Maximum active models (0 = unlimited)",
            min_value=0,
            max_value=10,
            value=0,
        )
        manual_weights: dict[str, float] = {}
        dirichlet_draws = 32
        pairwise_grid_step = 0.1
        optuna_trials = 200
        optuna_timeout: float | None = None
        optuna_seed = int(seed)
        if method == "manual":
            st.caption("Enter non-negative weights that sum to 1. Values are not auto-normalized.")
            total = 0.0
            by_id = {str(row["candidate_id"]): row for row in rows}
            for candidate_id in selected_ids:
                label = (by_id.get(candidate_id) or {}).get("label") or candidate_id
                value = st.number_input(
                    f"Weight · {label}",
                    min_value=0.0,
                    max_value=1.0,
                    value=round(1.0 / len(selected_ids), 6),
                    key=_state_key(f"weight_{candidate_id}"),
                )
                manual_weights[candidate_id] = float(value)
                total += float(value)
            st.write(f"Current sum: {total:.6f}")
        if method == "optimized_optuna":
            st.caption(
                "Optuna searches non-negative weights. Thresholds are selected only "
                "from meta-training rows inside each fold."
            )
            optuna_trials = int(
                st.number_input("Trials", min_value=1, max_value=5000, value=200)
            )
            timeout_raw = st.text_input("Optional timeout seconds", value="")
            optuna_timeout = float(timeout_raw) if timeout_raw.strip() else None
            optuna_seed = int(
                st.number_input("Optuna seed", min_value=0, value=int(seed))
            )
        if method == "optimized_native":
            with st.expander("Advanced settings", expanded=False):
                dirichlet_draws = int(
                    st.number_input("Dirichlet draws", min_value=1, max_value=512, value=32)
                )
                pairwise_grid_step = float(
                    st.number_input(
                        "Pairwise grid step",
                        min_value=0.01,
                        max_value=0.5,
                        value=0.1,
                        step=0.01,
                    )
                )
        submitted = st.form_submit_button("Prepare blend request")
    if not submitted and _state_key("request_obj") not in st.session_state:
        return None
    if submitted:
        try:
            if method == "manual":
                weight_sum = sum(manual_weights.values())
                if abs(weight_sum - 1.0) > 1.0e-9:
                    raise BlendUIRequestError(
                        f"Manual weights must sum to 1; got {weight_sum}.",
                        reason_code="manual_weight_sum_invalid",
                    )
            request = materialize_blend_ui_request(
                repository_root=repository_root,
                candidate_ids=selected_ids,
                method=method,
                folds=int(folds),
                repeats=int(repeats),
                seed=int(seed),
                max_active_models=None if int(max_active) == 0 else int(max_active),
                manual_weights=manual_weights if method == "manual" else None,
                dirichlet_draws=int(dirichlet_draws),
                pairwise_grid_step=float(pairwise_grid_step),
                optuna_trials=int(optuna_trials),
                optuna_timeout_seconds=optuna_timeout,
                optuna_seed=int(optuna_seed),
            )
            blend_id = expected_blend_id(request, repository_root)
            st.session_state[_state_key("request_obj")] = request
            st.session_state[_state_key("request_path")] = request.relative_path
            st.session_state[_state_key("request_id")] = request.request_id
            st.session_state[_state_key("expected_blend_id")] = blend_id
            # Invalidate prior search tied to a different request.
            prior = st.session_state.get(_state_key("search_request_id"))
            if prior != request.request_id:
                st.session_state.pop(_state_key("search_result"), None)
        except (BlendUIRequestError, BlendWorkspaceError, Exception) as error:
            st.error(str(error))
            return None
    request = st.session_state.get(_state_key("request_obj"))
    if request is None:
        return None
    strategy, optimizer = method_to_strategy_optimizer(str(request.payload["method"]))
    st.markdown("#### Prepared request")
    st.write(
        {
            "Candidates": [
                next(
                    (
                        row.get("label")
                        for row in rows
                        if row.get("candidate_id") == candidate_id
                    ),
                    candidate_id,
                )
                for candidate_id in request.candidate_ids
            ],
            "Strategy": strategy,
            "Optimizer": optimizer,
            "Meta-CV": f"{request.payload['folds']} folds × {request.payload['repeats']} repeats",
            "Maximum active models": request.payload.get("max_active_models"),
            "Search budget": (
                f"optuna_trials={request.payload.get('optuna_trials')}"
                if optimizer == "optuna"
                else f"dirichlet_draws={request.payload.get('dirichlet_draws')}"
            ),
            "Expected request ID": request.request_id,
            "Expected blend ID": st.session_state.get(_state_key("expected_blend_id")),
        }
    )
    with st.expander("Exact argv preview", expanded=False):
        st.code(
            "\n".join(
                [
                    "python",
                    "-u",
                    "scripts/run_prediction_blend.py",
                    "search",
                    "--request",
                    request.relative_path,
                ]
            )
        )
    return request


def _render_search_and_materialize(
    *,
    repository_root: Path,
    registry: ControlPanelRegistry,
    job_manager: JobManager,
    request: Any,
    rows: list[dict[str, Any]],
) -> None:
    st.subheader("Search")
    values = authorized_blend_argv_values(request.relative_path)
    built = build_command(
        registry.commands,
        "prediction_blend_v1",
        "search",
        values,
        repository_root=repository_root,
    )
    consumed = st.session_state.setdefault("_consumed_launch_nonces", set())
    previous = st.session_state.get(_state_key("rendered_search"))
    rendered = rendered_launch(
        built,
        previous if isinstance(previous, RenderedLaunch) else None,
        consumed_nonces=consumed,
    )
    st.session_state[_state_key("rendered_search")] = rendered
    confirm = st.checkbox(
        "I confirm this blend search and have reviewed the request.",
        key=_state_key("confirm_search"),
    )
    if st.button("Run blend search", type="primary", disabled=not confirm):
        try:
            authorized = authorize_launch(
                registry.commands,
                command_id="prediction_blend_v1",
                action_id="search",
                values=values,
                repository_root=repository_root,
                confirmed=confirm,
                high_risk_acknowledged=True,
                rendered=rendered,
                consumed_nonces=consumed,
            )
            record = job_manager.start(
                argv=authorized.argv,
                redacted_argv=authorized.redacted_argv,
                command_id="prediction_blend_v1",
                action_id="search",
                references={
                    "request_id": request.request_id,
                    "request_path": request.relative_path,
                    "candidate_ids": ",".join(request.candidate_ids),
                    "strategy": str(request.payload["strategy"]),
                    "optimizer": str(request.payload["optimizer_backend"]),
                    "expected_blend_id": str(
                        st.session_state.get(_state_key("expected_blend_id")) or ""
                    ),
                },
            )
            st.session_state[_state_key("search_job_id")] = record.job_id
            st.session_state[_state_key("search_request_id")] = request.request_id
            st.success(f"Started job {record.job_id}. Open Jobs for live status.")
        except (LaunchAuthorizationError, JobError) as error:
            st.error(str(error))

    job_id = st.session_state.get(_state_key("search_job_id"))
    if job_id:
        st.info(f"Latest search job: `{job_id}`")

    # Recover completed search for this request.
    jobs_root = Path(registry.settings.jobs_root)
    if not jobs_root.is_absolute():
        jobs_root = repository_root / jobs_root
    related = list_jobs_for_request(
        jobs_root, request.request_id, command_id="prediction_blend_v1"
    )
    search_jobs = [item for item in related if item.get("action_id") == "search"]
    if search_jobs:
        latest = search_jobs[0]
        job_state = latest.get("state")
        st.write(
            {
                "Job ID": latest.get("job_id"),
                "Status": job_state,
                "Started": latest.get("started_at_utc") or latest.get("created_at_utc"),
            }
        )
        if job_state == "succeeded":
            try:
                stdout = (
                    jobs_root / str(latest["job_id"]) / "stdout.log"
                ).read_text(encoding="utf-8", errors="replace")
                payload = verify_search_result(
                    parse_job_stdout_json(stdout),
                    request_id=request.request_id,
                    expected_blend_id=st.session_state.get(
                        _state_key("expected_blend_id")
                    ),
                )
                st.session_state[_state_key("search_result")] = payload
                st.session_state[_state_key("search_request_id")] = request.request_id
            except BlendWorkspaceError as error:
                st.error(str(error))
                with st.expander("Technical details", expanded=False):
                    st.code(
                        (jobs_root / str(latest["job_id"]) / "stderr.log")
                        .read_text(encoding="utf-8", errors="replace")[-4000:]
                    )
        elif job_state in {"failed", "stopped", "orphaned"}:
            st.error(f"Search job {job_state}")
            with st.expander("Technical details", expanded=False):
                st.code(
                    (jobs_root / str(latest["job_id"]) / "stderr.log")
                    .read_text(encoding="utf-8", errors="replace")[-4000:]
                )

    result = st.session_state.get(_state_key("search_result"))
    if not result or st.session_state.get(_state_key("search_request_id")) != request.request_id:
        return
    _render_search_result(result, rows)
    _render_materialize_controls(
        repository_root=repository_root,
        registry=registry,
        job_manager=job_manager,
        request=request,
        jobs_root=jobs_root,
    )


def _render_search_result(result: Mapping[str, Any], rows: list[dict[str, Any]]) -> None:
    st.subheader("Search result")
    honest = result.get("honest_meta_cv_metrics") or {}
    st.markdown("#### Honest meta-CV evaluation")
    st.metric(
        "Mean held-out Balanced Accuracy",
        f"{float(honest.get('mean_repeat_balanced_accuracy', float('nan'))):.4f}",
    )
    cols = st.columns(4)
    cols[0].metric("BA std", f"{float(honest.get('std_repeat_balanced_accuracy', 0)):.4f}")
    cols[1].metric("Min BA", f"{float(honest.get('min_repeat_balanced_accuracy', 0)):.4f}")
    cols[2].metric("Max BA", f"{float(honest.get('max_repeat_balanced_accuracy', 0)):.4f}")
    pooled = honest.get("pooled_repeated_held_out_confusion_metrics") or {}
    cols[3].metric("Sensitivity", f"{float(pooled.get('sensitivity', float('nan'))):.4f}")
    st.write(
        {
            "Specificity": pooled.get("specificity"),
            "Repeat metrics": honest.get("repeat_metrics"),
        }
    )
    st.markdown("#### Descriptive evidence")
    st.caption(
        "These scores are descriptive and are not the primary unbiased estimate."
    )
    st.write(
        {
            "Cross-fitted probability descriptive": result.get(
                "cross_fitted_probability_descriptive_metrics"
            ),
            "Full-OOF deployment": result.get("full_oof_descriptive_metrics"),
        }
    )
    st.markdown("#### Deployment parameters")
    weights = result.get("final_deployment_weights") or (
        (result.get("deployment") or {}).get("weights")
    )
    labels = {
        str(row["candidate_id"]): str(row.get("label") or row["candidate_id"])
        for row in rows
    }
    if isinstance(weights, Mapping):
        st.write(
            {
                labels.get(str(key), str(key)): value
                for key, value in weights.items()
            }
        )
    st.write(
        {
            "Final threshold": result.get("final_deployment_threshold")
            or (result.get("deployment") or {}).get("threshold"),
            "Optimizer": result.get("optimizer_backend"),
            "Search budget": result.get("search_budget") or result.get("optimizer_settings"),
            "Exploratory": result.get("exploratory"),
            "Submission readiness": result.get("submission_readiness"),
        }
    )
    if result.get("optuna_study_summaries"):
        with st.expander("Optuna study details", expanded=False):
            summaries = list(result["optuna_study_summaries"])[:20]
            st.dataframe(pd.DataFrame(summaries), width="stretch", hide_index=True)


def _render_materialize_controls(
    *,
    repository_root: Path,
    registry: ControlPanelRegistry,
    job_manager: JobManager,
    request: Any,
    jobs_root: Path,
) -> None:
    st.subheader("Materialization")
    st.write(
        "This writes an immutable detailed blend artifact and a canonical "
        "prediction candidate."
    )
    confirm = st.checkbox(
        "I confirm materialization of this exact search request.",
        key=_state_key("confirm_materialize"),
    )
    values = authorized_blend_argv_values(request.relative_path)
    built = build_command(
        registry.commands,
        "prediction_blend_v1",
        "materialize",
        values,
        repository_root=repository_root,
    )
    consumed = st.session_state.setdefault("_consumed_launch_nonces", set())
    previous = st.session_state.get(_state_key("rendered_materialize"))
    rendered = rendered_launch(
        built,
        previous if isinstance(previous, RenderedLaunch) else None,
        consumed_nonces=consumed,
    )
    st.session_state[_state_key("rendered_materialize")] = rendered
    if st.button("Materialize this blend", disabled=not confirm):
        try:
            authorized = authorize_launch(
                registry.commands,
                command_id="prediction_blend_v1",
                action_id="materialize",
                values=values,
                repository_root=repository_root,
                confirmed=confirm,
                high_risk_acknowledged=True,
                rendered=rendered,
                consumed_nonces=consumed,
            )
            record = job_manager.start(
                argv=authorized.argv,
                redacted_argv=authorized.redacted_argv,
                command_id="prediction_blend_v1",
                action_id="materialize",
                references={
                    "request_id": request.request_id,
                    "request_path": request.relative_path,
                    "candidate_ids": ",".join(request.candidate_ids),
                    "strategy": str(request.payload["strategy"]),
                    "optimizer": str(request.payload["optimizer_backend"]),
                    "expected_blend_id": str(
                        st.session_state.get(_state_key("expected_blend_id")) or ""
                    ),
                },
            )
            st.success(f"Started materialization job {record.job_id}.")
        except (LaunchAuthorizationError, JobError) as error:
            st.error(str(error))

    related = list_jobs_for_request(
        jobs_root, request.request_id, command_id="prediction_blend_v1"
    )
    materialize_jobs = [
        item for item in related if item.get("action_id") == "materialize"
    ]
    if not materialize_jobs:
        return
    latest = materialize_jobs[0]
    if latest.get("state") != "succeeded":
        st.write(
            {
                "Materialize job": latest.get("job_id"),
                "Status": latest.get("state"),
            }
        )
        return
    try:
        stdout = (jobs_root / str(latest["job_id"]) / "stdout.log").read_text(
            encoding="utf-8", errors="replace"
        )
        payload = parse_job_stdout_json(stdout)
        blend_id = str(payload.get("blend_id") or "")
        loaded = load_blend_artifact(blend_id, repository_root=repository_root)
        st.success("Blend materialized and strictly validated.")
        st.write(
            {
                "Blend ID": blend_id,
                "Candidate": payload.get("canonical_candidate_id"),
                "Honest score": (loaded.get("honest_meta_cv_metrics") or {}).get(
                    "mean_repeat_balanced_accuracy"
                ),
                "Final weights": loaded.get("final_deployment_weights"),
                "Final threshold": loaded.get("final_deployment_threshold"),
                "Exploratory": (loaded.get("manifest") or {}).get("exploratory"),
                "Submission readiness": loaded.get("submission_readiness"),
            }
        )
        candidate_id = str(payload.get("canonical_candidate_id") or "")
        if candidate_id and st.button("Prepare submission", key=_state_key("handoff")):
            package_dir = resolve_under_repository(
                f"artifacts/prediction_candidates/{candidate_id}",
                repository_root,
            )
            apply_canonical_submission_handoff(
                st.session_state,
                candidate_id=candidate_id,
                manifest_sha256=file_sha256(package_dir / "candidate_manifest.json"),
                blend_id=blend_id,
            )
            st.session_state["run-command"] = "final_deployment_v1"
            set_durable_value(
                st.session_state, ui_durable_key("run", "command"), "final_deployment_v1"
            )
            st.session_state[_state_key("submission_source")] = (
                "Canonical prediction candidate"
            )
            st.success(
                "Handoff ready. Open Run → Generate submission; the canonical "
                "candidate source is preselected."
            )
            try:
                st.switch_page("Run")
            except Exception:  # noqa: BLE001
                st.info("Open Run → Generate submission to continue.")
    except Exception as error:  # noqa: BLE001
        st.error(f"Materialized blend could not be validated: {error}")


def _render_materialized_blends(repository_root: Path) -> None:
    st.subheader("Materialized blends")
    blends = discover_materialized_blends(repository_root)
    if not blends:
        st.caption("No materialized canonical blends found.")
        return
    frame = pd.DataFrame(
        [
            {
                "Blend": item.get("label") or item.get("blend_id"),
                "Parents": len(item.get("candidate_ids") or []),
                "Optimizer": item.get("optimizer_backend"),
                "Honest BA": (
                    (item.get("honest_meta_cv_metrics") or {}).get(
                        "mean_repeat_balanced_accuracy"
                    )
                ),
                "Final threshold": item.get("final_deployment_threshold"),
                "Exploratory": item.get("exploratory"),
                "Submission readiness": (
                    (item.get("submission_readiness") or {}).get("state")
                ),
                "Created": item.get("created_at_utc"),
                "Status": item.get("status"),
                "Blend ID": item.get("blend_id"),
            }
            for item in blends
        ]
    )
    st.dataframe(frame, width="stretch", hide_index=True)
    options = [
        str(item["blend_id"])
        for item in blends
        if item.get("status") == "completed"
    ]
    if not options:
        return
    selected = st.selectbox(
        "Inspect materialized blend",
        options,
        key=_state_key("inspect_blend"),
        format_func=lambda value: next(
            (
                str(item.get("label") or value)
                for item in blends
                if item.get("blend_id") == value
            ),
            value,
        ),
    )
    chosen = next(item for item in blends if item.get("blend_id") == selected)
    st.write(chosen.get("summary") or chosen)
    if chosen.get("canonical_candidate_id") and st.button(
        "Prepare submission from selected blend",
        key=_state_key("handoff_existing"),
    ):
        candidate_id = str(chosen["canonical_candidate_id"])
        package_dir = resolve_under_repository(
            f"artifacts/prediction_candidates/{candidate_id}",
            repository_root,
        )
        apply_canonical_submission_handoff(
            st.session_state,
            candidate_id=candidate_id,
            manifest_sha256=file_sha256(package_dir / "candidate_manifest.json"),
            blend_id=str(chosen["blend_id"]),
        )
        st.success("Handoff stored. Open Run → Generate submission.")
