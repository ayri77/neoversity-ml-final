"""Streamlit Search history tab for Blend Workspace."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.churn_ml.blending.artifact_v1 import load_blend_artifact
from src.churn_ml.control_panel.blend_ui_request_v1 import load_blend_ui_request
from src.churn_ml.control_panel.blend_workspace import (
    apply_canonical_submission_handoff,
    authorized_blend_argv_values,
    list_jobs_for_request,
    parse_job_stdout_json,
)
from src.churn_ml.control_panel.candidate_display import (
    candidate_display_map,
    project_weight_rows,
    resolve_candidate_display,
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
from src.churn_ml.control_panel.saved_blend_search import (
    SavedBlendSearch,
    compare_materialized_to_saved,
    configuration_from_saved_search,
    discover_saved_blend_searches,
    filter_materialize_jobs_for_saved_search,
    filter_saved_searches,
    historical_materialize_launch_plan,
    jobs_search_fingerprint,
    persist_loaded_blend_configuration,
)
from src.churn_ml.control_panel.selection_state import (
    set_durable_value,
    ui_durable_key,
)
from src.churn_ml.prediction_candidates.contract_v1 import (
    file_sha256,
    resolve_under_repository,
)


STATE_PREFIX = "blend_search_history"


def _state_key(name: str) -> str:
    return f"{STATE_PREFIX}:{name}"


def _jobs_root(repository_root: Path, registry: ControlPanelRegistry) -> Path:
    jobs_root = Path(registry.settings.jobs_root)
    if not jobs_root.is_absolute():
        jobs_root = repository_root / jobs_root
    return jobs_root


@st.cache_data(show_spinner="Loading blend search history…")
def _load_saved_searches_cached(
    root: str, jobs_root: str, fingerprint: str
) -> list[SavedBlendSearch]:
    del fingerprint
    return discover_saved_blend_searches(
        repository_root=Path(root),
        jobs_root=Path(jobs_root),
    )


def render_search_history_tab(
    *,
    repository_root: Path,
    registry: ControlPanelRegistry,
    job_manager: JobManager,
) -> None:
    st.subheader("Search history")
    st.caption(
        "Completed blend searches recovered from durable Jobs and immutable "
        "requests. History does not depend on the current Build blend form."
    )
    jobs_root = _jobs_root(repository_root, registry)
    fingerprint = jobs_search_fingerprint(jobs_root)
    if st.button("Refresh search history", key=_state_key("refresh")):
        _load_saved_searches_cached.clear()

    rows = _load_saved_searches_cached(
        str(repository_root.resolve()),
        str(jobs_root.resolve()),
        fingerprint,
    )
    if not rows:
        st.info("No prediction_blend_v1 search Jobs were found.")
        return

    cols = st.columns(3)
    status = cols[0].selectbox(
        "Status",
        ["All", "Reusable", "Blocked"],
        key=_state_key("filter_status"),
    )
    optimizers = sorted(
        {row.optimizer_backend for row in rows if row.optimizer_backend}
    )
    optimizer = cols[1].selectbox(
        "Optimizer",
        ["All", *optimizers],
        key=_state_key("filter_optimizer"),
    )
    search_text = cols[2].text_input("Search", key=_state_key("filter_search"))
    filtered = filter_saved_searches(
        rows,
        status=(
            None
            if status == "All"
            else ("reusable" if status == "Reusable" else "blocked")
        ),
        optimizer=None if optimizer == "All" else optimizer,
        search_text=search_text or None,
    )
    frame = pd.DataFrame(
        [
            {
                "Experiment": row.experiment_label,
                "Models": " · ".join(row.candidate_labels),
                "Optimizer": row.optimizer_backend,
                "Meta-CV": (
                    f"{row.folds}×{row.repeats}"
                    if row.folds is not None and row.repeats is not None
                    else "—"
                ),
                "Maximum active": (
                    "unlimited"
                    if row.max_active_models is None
                    else row.max_active_models
                ),
                "Honest BA": row.honest_mean_balanced_accuracy,
                "BA std": row.honest_std_balanced_accuracy,
                "Threshold": row.final_threshold,
                "Status": row.status_label,
                "Created": row.created_at_utc,
                "Request ID": row.request_id,
                "Blend ID": row.expected_blend_id,
                "Attempts": row.attempt_count,
            }
            for row in filtered
        ]
    )
    st.dataframe(frame, width="stretch", hide_index=True)
    if not filtered:
        st.info("No searches match the current filters.")
        return

    options = [row.request_id for row in filtered]
    labels = {row.request_id: row.experiment_label for row in filtered}
    selected_id = st.selectbox(
        "Inspect saved search",
        options,
        key=_state_key("selected_request"),
        format_func=lambda value: labels.get(value, value),
    )
    saved = next(row for row in filtered if row.request_id == selected_id)
    _render_saved_search_details(saved)
    _render_saved_search_actions(
        repository_root=repository_root,
        registry=registry,
        job_manager=job_manager,
        jobs_root=jobs_root,
        saved=saved,
    )


def _render_saved_search_details(saved: SavedBlendSearch) -> None:
    st.markdown("#### Identity")
    st.write(
        {
            "Request ID": saved.request_id,
            "Blend ID": saved.expected_blend_id,
            "Job ID": saved.job_id,
            "Status": saved.status_label,
            "Created": saved.created_at_utc,
            "Completed": saved.completed_at_utc,
            "Attempts": saved.attempt_count,
        }
    )
    st.markdown("#### Exact ordered candidates")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Order": index + 1,
                    "Model": label,
                    "Dataset": (
                        saved.candidate_datasets[index]
                        if index < len(saved.candidate_datasets)
                        else ""
                    ),
                    "Candidate ID": candidate_id,
                    "Manifest SHA256": (
                        saved.candidate_manifest_sha256[index]
                        if index < len(saved.candidate_manifest_sha256)
                        else ""
                    ),
                }
                for index, (candidate_id, label) in enumerate(
                    zip(saved.candidate_ids, saved.candidate_labels, strict=False)
                )
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    st.markdown("#### Configuration")
    st.write(
        {
            "Strategy": saved.strategy,
            "Optimizer": saved.optimizer_backend,
            "Folds": saved.folds,
            "Repeats": saved.repeats,
            "Seed": saved.seed,
            "Maximum active models": saved.max_active_models,
            "Search budget": dict(saved.search_budget),
            "Method": saved.method,
        }
    )
    if saved.search_result is not None:
        honest = saved.search_result.get("honest_meta_cv_metrics") or {}
        st.markdown("#### Honest evaluation")
        st.metric(
            "Mean held-out Balanced Accuracy",
            f"{float(saved.honest_mean_balanced_accuracy or float('nan')):.6f}",
        )
        cols = st.columns(4)
        cols[0].metric(
            "BA std",
            f"{float(saved.honest_std_balanced_accuracy or 0):.6f}",
        )
        cols[1].metric(
            "Min BA",
            f"{float(honest.get('min_repeat_balanced_accuracy', 0)):.6f}",
        )
        cols[2].metric(
            "Max BA",
            f"{float(honest.get('max_repeat_balanced_accuracy', 0)):.6f}",
        )
        pooled = honest.get("pooled_repeated_held_out_confusion_metrics") or {}
        cols[3].metric(
            "Sensitivity",
            f"{float(pooled.get('sensitivity', float('nan'))):.4f}",
        )
        st.write(
            {
                "Specificity": pooled.get("specificity"),
                "Repeat metrics": honest.get("repeat_metrics"),
            }
        )
        st.markdown("#### Deployment parameters")
        weight_rows = []
        for candidate_id, label in zip(
            saved.candidate_ids, saved.candidate_labels, strict=False
        ):
            weight_rows.append(
                {
                    "Model": label,
                    "Weight": saved.final_weights.get(candidate_id),
                    "Candidate ID": candidate_id,
                }
            )
        st.dataframe(pd.DataFrame(weight_rows), width="stretch", hide_index=True)
        st.write({"Final threshold": saved.final_threshold})
        st.markdown("#### Descriptive evidence")
        st.caption(
            "These scores are descriptive and are not the primary unbiased estimate."
        )
        st.write(
            {
                "Cross-fitted probability descriptive": saved.search_result.get(
                    "cross_fitted_probability_descriptive_metrics"
                ),
                "Full-OOF deployment": saved.search_result.get(
                    "full_oof_descriptive_metrics"
                ),
            }
        )
    if not saved.reusable:
        st.warning(
            "This saved search is not reusable: "
            + ", ".join(saved.blocker_codes or ("unknown",))
        )
    with st.expander("Technical details", expanded=False):
        st.json(
            {
                "blocker_codes": list(saved.blocker_codes),
                "request_path": saved.request_path,
                "attempt_job_ids": list(saved.attempt_job_ids),
                "technical": dict(saved.technical),
            }
        )


def _render_saved_search_actions(
    *,
    repository_root: Path,
    registry: ControlPanelRegistry,
    job_manager: JobManager,
    jobs_root: Path,
    saved: SavedBlendSearch,
) -> None:
    st.markdown("#### Actions")
    if st.button(
        "Load configuration in Build blend",
        key=_state_key(f"load_{saved.request_id}"),
    ):
        config = configuration_from_saved_search(saved)
        # Durable-only write on this run. Widget keys are seeded on the next run
        # before Blend Workspace widgets are instantiated.
        persist_loaded_blend_configuration(
            st.session_state, config, pending_apply=True
        )
        st.rerun()

    if not saved.reusable:
        return

    st.write(
        "Materialization uses the exact saved immutable request. "
        "The current backend deterministically re-evaluates that request before "
        "writing the immutable blend artifact and canonical candidate. "
        "It does not retrain the source models."
    )
    st.write(
        {
            "Exact request": saved.request_path,
            "Expected blend ID": saved.expected_blend_id,
            "Source search job": saved.job_id,
        }
    )
    confirm = st.checkbox(
        "I confirm materialization of this exact saved search request.",
        key=_state_key(f"confirm_mat_{saved.request_id}"),
    )
    if st.button(
        "Materialize this saved search",
        type="primary",
        disabled=not confirm,
        key=_state_key(f"materialize_{saved.request_id}"),
    ):
        try:
            plan = historical_materialize_launch_plan(saved)
            request = load_blend_ui_request(
                str(plan["request_path"]), repository_root=repository_root
            )
            values = authorized_blend_argv_values(request.relative_path)
            built = build_command(
                registry.commands,
                str(plan["command_id"]),
                str(plan["action_id"]),
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
            authorized = authorize_launch(
                registry.commands,
                command_id=str(plan["command_id"]),
                action_id=str(plan["action_id"]),
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
                command_id=str(plan["command_id"]),
                action_id=str(plan["action_id"]),
                references=dict(plan["references"]),
            )
            st.session_state[_state_key(f"mat_job_{saved.request_id}")] = record.job_id
            st.success(f"Started materialization job `{record.job_id}`.")
        except (LaunchAuthorizationError, JobError, Exception) as error:  # noqa: BLE001
            st.error(str(error))

    related = list_jobs_for_request(
        jobs_root, saved.request_id, command_id="prediction_blend_v1"
    )
    materialize_jobs = list(filter_materialize_jobs_for_saved_search(related, saved))
    if not materialize_jobs:
        return
    latest = materialize_jobs[0]
    st.write(
        {
            "Materialization Job": latest.get("job_id"),
            "Status": latest.get("state"),
            "Started": latest.get("started_at_utc") or latest.get("created_at_utc"),
            "Completed": latest.get("finished_at_utc"),
        }
    )
    if latest.get("state") != "succeeded":
        return
    try:
        stdout = (jobs_root / str(latest["job_id"]) / "stdout.log").read_text(
            encoding="utf-8", errors="replace"
        )
        payload = parse_job_stdout_json(stdout)
        blend_id = str(payload.get("blend_id") or "")
        loaded = load_blend_artifact(blend_id, repository_root=repository_root)
        mismatches = compare_materialized_to_saved(
            saved=saved,
            materialized_payload=payload,
            loaded_artifact=loaded,
        )
        if mismatches:
            st.error(
                "Materialized result differs from the selected saved search: "
                + ", ".join(mismatches)
            )
            with st.expander("Technical details", expanded=False):
                st.json(
                    {
                        "mismatches": mismatches,
                        "saved_blend_id": saved.expected_blend_id,
                        "materialized_blend_id": blend_id,
                        "saved_ba": saved.honest_mean_balanced_accuracy,
                        "materialized_ba": (
                            (loaded.get("honest_meta_cv_metrics") or {}).get(
                                "mean_repeat_balanced_accuracy"
                            )
                        ),
                    }
                )
            return
        st.success("Blend materialized and strictly validated.")
        candidate_id = str(payload.get("canonical_candidate_id") or "")
        displays = candidate_display_map(
            list(saved.candidate_ids), repository_root=repository_root
        )
        weights = loaded.get("final_deployment_weights")
        st.write(
            {
                "Blend ID": blend_id,
                "Canonical candidate": (
                    resolve_candidate_display(
                        candidate_id, repository_root=repository_root
                    ).primary_label
                    if candidate_id
                    else None
                ),
                "Honest BA": (loaded.get("honest_meta_cv_metrics") or {}).get(
                    "mean_repeat_balanced_accuracy"
                ),
                "Final threshold": loaded.get("final_deployment_threshold"),
                "Submission readiness": loaded.get("submission_readiness"),
            }
        )
        st.dataframe(
            pd.DataFrame(project_weight_rows(weights, displays)),
            width="stretch",
            hide_index=True,
        )
        with st.expander("Technical details", expanded=False):
            st.json(
                {
                    "canonical_candidate_id": candidate_id,
                    "final_deployment_weights": weights,
                }
            )
        if candidate_id and st.button(
            "Prepare submission",
            key=_state_key(f"handoff_{saved.request_id}"),
        ):
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
            st.success(
                "Handoff ready. Open Run → Generate submission; the canonical "
                "candidate source is preselected."
            )
    except Exception as error:  # noqa: BLE001
        st.error(f"Materialized blend could not be validated: {error}")
