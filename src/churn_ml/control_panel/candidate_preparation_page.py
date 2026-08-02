"""Streamlit Prepare candidates tab for Blend Workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.churn_ml.control_panel.blend_workspace import parse_job_stdout_json
from src.churn_ml.control_panel.candidate_preparation import (
    CACHE_CONTRACT_VERSION,
    MAX_MODELS,
    CandidatePreparationError,
    authorized_preparation_argv_values,
    discover_managed_autogluon_runs,
    ensemble_component_warning,
    filter_managed_runs,
    list_jobs_for_preparation_request,
    managed_runs_inventory_fingerprint,
    prepare_preparation_request,
    verify_preparation_result,
    verify_validation_result,
)
from src.churn_ml.control_panel.command_builder import build_command
from src.churn_ml.control_panel.jobs import JobError, JobManager
from src.churn_ml.control_panel.launch import (
    LaunchAuthorizationError,
    RenderedLaunch,
    authorize_launch,
    rendered_launch,
)
from src.churn_ml.control_panel.python_runtimes import (
    PythonRuntimeError,
    require_autogluon_python,
    resolve_autogluon_python,
)
from src.churn_ml.control_panel.registry import ControlPanelRegistry
from src.churn_ml.prediction_candidates.preparation_request_v1 import (
    PreparationRequestError,
)


STATE_PREFIX = "candidate_prep"


def _state_key(name: str) -> str:
    return f"{STATE_PREFIX}:{name}"


@st.cache_data(show_spinner="Loading managed AutoGluon runs…")
def _load_managed_runs_cached(
    root: str, fingerprint: str, contract: str, include_incomplete: bool
) -> list[dict[str, Any]]:
    del fingerprint, contract
    return discover_managed_autogluon_runs(
        Path(root), include_incomplete=include_incomplete
    )


def render_prepare_candidates_tab(
    *,
    repository_root: Path,
    registry: ControlPanelRegistry,
    job_manager: JobManager,
) -> None:
    st.subheader("Prepare candidates")
    st.caption(
        "Discover completed managed AutoGluon runs, validate selected models, "
        "and prepare immutable prediction candidates for blending."
    )
    st.info(
        "Source: Managed AutoGluon runs. Research v2 and historical candidates "
        "are not supported by this preparation workflow yet."
    )

    runtime = resolve_autogluon_python(repository_root)
    if runtime.available:
        st.success(f"AutoGluon interpreter: `{runtime.relative_path}`")
    else:
        st.error(
            runtime.reason
            or "AutoGluon interpreter (.venv-autogluon) is unavailable."
        )
        with st.expander("Technical details", expanded=False):
            st.json(
                {
                    "reason_code": runtime.reason_code,
                    "relative_path": runtime.relative_path,
                }
            )

    fingerprint = managed_runs_inventory_fingerprint(repository_root)
    include_incomplete = st.checkbox(
        "Show incomplete / blocked runs",
        key=_state_key("include_incomplete"),
        value=False,
    )
    if st.button("Refresh AutoGluon runs", key=_state_key("refresh")):
        _load_managed_runs_cached.clear()
        for key in (
            "validation_result",
            "preparation_result",
            "request_obj",
            "request_id",
            "selected_models",
        ):
            st.session_state.pop(_state_key(key), None)

    runs = _load_managed_runs_cached(
        str(repository_root.resolve()),
        fingerprint,
        CACHE_CONTRACT_VERSION,
        include_incomplete,
    )
    if not runs:
        st.warning("No managed AutoGluon runs were found under artifacts/autogluon_runs/.")
        return

    datasets = sorted({str(row.get("dataset_id") or "") for row in runs if row.get("dataset_id")})
    deps = sorted(
        {
            str(row.get("target_dependency") or "")
            for row in runs
            if row.get("target_dependency")
        }
    )
    cols = st.columns(4)
    dataset = cols[0].selectbox("Dataset", ["All", *datasets], key=_state_key("filter_dataset"))
    dependency = cols[1].selectbox(
        "Target dependency", ["All", *deps], key=_state_key("filter_dep")
    )
    completion = cols[2].selectbox(
        "Completed / blocked",
        ["All", "Completed", "Blocked"],
        key=_state_key("filter_completion"),
    )
    search = cols[3].text_input("Search", key=_state_key("filter_search"))
    filtered = filter_managed_runs(
        runs,
        dataset=None if dataset == "All" else dataset,
        target_dependency=None if dependency == "All" else dependency,
        completion=(
            None
            if completion == "All"
            else ("completed" if completion == "Completed" else "blocked")
        ),
        search_text=search or None,
    )
    if not filtered:
        st.info("No runs match the current filters.")
        return

    options = [str(row["run_path"]) for row in filtered]
    labels = {str(row["run_path"]): str(row.get("label") or row["run_path"]) for row in filtered}
    selected_run_path = st.selectbox(
        "Managed AutoGluon run",
        options,
        key=_state_key("run_path"),
        format_func=lambda value: labels.get(value, value),
    )
    run = next(row for row in filtered if row["run_path"] == selected_run_path)
    st.write(
        {
            "Dataset": run.get("dataset_id"),
            "Target dependency": run.get("target_dependency"),
            "Exploratory": run.get("exploratory"),
            "Classification": run.get("classification"),
            "Best model": run.get("best_model"),
            "Model count": run.get("model_count"),
            "Prepared candidates": run.get("prepared_candidate_count"),
            "Preparable": run.get("preparable"),
        }
    )
    if run.get("exploratory"):
        st.warning(
            "This Dataset Package is exploratory. Prepared candidates will remain "
            "visibly exploratory."
        )
    if run.get("blocking_reasons"):
        st.error("Run blockers")
        for blocker in run["blocking_reasons"]:
            st.write(f"- `{blocker.get('code')}`: {blocker.get('message')}")
    with st.expander("Technical details", expanded=False):
        st.json(
            {
                "run_path": run.get("run_path"),
                "run_id": run.get("run_id"),
                "config_sha256": run.get("config_sha256"),
                "identity_hashes": run.get("identity_hashes"),
                "blocking_reasons": run.get("blocking_reasons"),
            }
        )

    if not run.get("preparable"):
        st.info("Select a preparable completed run to continue.")
        return

    st.caption(
        "AutoGluon validation scores use the source run's own validation protocol. "
        "They are descriptive selection evidence, not the final blend evaluation."
    )
    models = list(run.get("models") or [])
    model_frame = pd.DataFrame(
        [
            {
                "Model": item.get("label"),
                "Type": item.get("model_type"),
                "Validation score": item.get("validation_score"),
                "Stack": item.get("stack_level"),
                "Fit time": item.get("fit_time"),
                "Prediction time": item.get("prediction_time"),
                "Best": item.get("is_best"),
                "Ensemble": item.get("is_ensemble"),
                "Already prepared": item.get("already_prepared"),
                "Preparation state": item.get("preparation_state"),
                "Candidate ID": item.get("candidate_id"),
            }
            for item in models
        ]
    )
    st.dataframe(model_frame, width="stretch", hide_index=True)

    # Never auto-select models; the user must choose explicitly.
    include_prepared = st.checkbox(
        "Include already prepared models in selection",
        key=_state_key("include_prepared"),
        value=False,
    )
    selectable = [
        str(item["model_name"])
        for item in models
        if include_prepared or not item.get("already_prepared")
    ]
    model_labels = {
        str(item["model_name"]): str(item.get("label") or item["model_name"])
        for item in models
    }
    previous_run = st.session_state.get(_state_key("models_for_run"))
    if previous_run != selected_run_path:
        st.session_state[_state_key("selected_models")] = []
        st.session_state[_state_key("models_for_run")] = selected_run_path
        for key in ("validation_result", "preparation_result", "request_obj", "request_id"):
            st.session_state.pop(_state_key(key), None)
    elif _state_key("selected_models") not in st.session_state:
        st.session_state[_state_key("selected_models")] = []
    # Drop selections that are no longer selectable after filter changes.
    current = [
        name
        for name in list(st.session_state.get(_state_key("selected_models")) or [])
        if name in selectable
    ]
    st.session_state[_state_key("selected_models")] = current

    selected_models = st.multiselect(
        "Selected models (1–8, order preserved)",
        options=selectable,
        max_selections=MAX_MODELS,
        key=_state_key("selected_models"),
        format_func=lambda value: model_labels.get(value, value),
    )
    selection_fp = ",".join(selected_models)
    if st.session_state.get(_state_key("selection_fp")) != selection_fp:
        for key in ("validation_result", "preparation_result", "request_obj", "request_id"):
            st.session_state.pop(_state_key(key), None)
        st.session_state[_state_key("selection_fp")] = selection_fp

    warning = ensemble_component_warning(selected_models)
    if warning:
        st.warning(warning)
    if not selected_models:
        st.info("Select at least one model to validate.")
        return
    if len(selected_models) > MAX_MODELS:
        st.error(f"At most {MAX_MODELS} models can be selected.")
        return

    _render_validate_and_prepare(
        repository_root=repository_root,
        registry=registry,
        job_manager=job_manager,
        run=run,
        selected_models=list(selected_models),
        runtime_available=runtime.available,
    )


def _render_validate_and_prepare(
    *,
    repository_root: Path,
    registry: ControlPanelRegistry,
    job_manager: JobManager,
    run: dict[str, Any],
    selected_models: list[str],
    runtime_available: bool,
) -> None:
    st.markdown("#### Validation")
    can_act = runtime_available and bool(selected_models)
    if st.button(
        "Validate selected models",
        type="primary",
        disabled=not can_act,
        key=_state_key("validate"),
    ):
        try:
            runtime = require_autogluon_python(repository_root)
            request = prepare_preparation_request(
                repository_root=repository_root,
                run_path=str(run["run_path"]),
                selected_models=selected_models,
            )
            st.session_state[_state_key("request_obj")] = request
            st.session_state[_state_key("request_id")] = request.request_id
            values = authorized_preparation_argv_values(request.relative_path)
            built = build_command(
                registry.commands,
                "autogluon_candidate_preparation_v1",
                "validate",
                values,
                repository_root=repository_root,
                python_executable=str(runtime.absolute_path),
            )
            consumed = st.session_state.setdefault("_consumed_launch_nonces", set())
            rendered = rendered_launch(built, None, consumed_nonces=consumed)
            authorized = authorize_launch(
                registry.commands,
                command_id="autogluon_candidate_preparation_v1",
                action_id="validate",
                values=values,
                repository_root=repository_root,
                confirmed=True,
                high_risk_acknowledged=True,
                rendered=rendered,
                consumed_nonces=consumed,
                python_executable=str(runtime.absolute_path),
            )
            if authorized.argv != built.argv:
                raise CandidatePreparationError(
                    "Rendered and authorized argv diverge.",
                    reason_code="argv_mismatch",
                )
            record = job_manager.start(
                argv=authorized.argv,
                redacted_argv=authorized.redacted_argv,
                command_id="autogluon_candidate_preparation_v1",
                action_id="validate",
                references={
                    "request_id": request.request_id,
                    "run_path": request.run_path,
                    "dataset_id": str(request.payload.get("dataset_id") or ""),
                    "selected_models": ",".join(request.selected_models),
                    "source_type": "managed_autogluon",
                    "operation": "validate",
                },
            )
            st.session_state[_state_key("validate_job_id")] = record.job_id
            st.success(f"Started validation job `{record.job_id}`.")
        except (
            PreparationRequestError,
            CandidatePreparationError,
            PythonRuntimeError,
            LaunchAuthorizationError,
            JobError,
        ) as error:
            st.error(str(error))
            with st.expander("Technical details", expanded=False):
                st.write(
                    {
                        "reason_code": getattr(error, "reason_code", None),
                        "error": str(error),
                    }
                )

    request = st.session_state.get(_state_key("request_obj"))
    if request is None:
        return
    if list(request.selected_models) != selected_models or request.run_path != run["run_path"]:
        st.warning("Selection changed after the prepared request. Re-validate.")
        return

    jobs_root = Path(registry.settings.jobs_root)
    if not jobs_root.is_absolute():
        jobs_root = repository_root / jobs_root
    related = list_jobs_for_preparation_request(jobs_root, request.request_id)
    validate_jobs = [item for item in related if item.get("action_id") == "validate"]
    if validate_jobs:
        latest = validate_jobs[0]
        st.write(
            {
                "Validation job": latest.get("job_id"),
                "Status": latest.get("state"),
                "Started": latest.get("started_at_utc") or latest.get("created_at_utc"),
            }
        )
        if latest.get("state") == "succeeded":
            try:
                stdout = (
                    jobs_root / str(latest["job_id"]) / "stdout.log"
                ).read_text(encoding="utf-8", errors="replace")
                payload = verify_validation_result(
                    parse_job_stdout_json(stdout),
                    request_id=request.request_id,
                    run_path=request.run_path,
                    selected_models=request.selected_models,
                )
                st.session_state[_state_key("validation_result")] = payload
                st.success("Validation succeeded for the exact prepared request.")
            except (CandidatePreparationError, Exception) as error:  # noqa: BLE001
                st.error(str(error))
                with st.expander("Technical details", expanded=False):
                    st.code(
                        (jobs_root / str(latest["job_id"]) / "stderr.log")
                        .read_text(encoding="utf-8", errors="replace")[-4000:]
                    )
        elif latest.get("state") in {"failed", "stopped", "orphaned"}:
            st.error(f"Validation job {latest.get('state')}")
            with st.expander("Technical details", expanded=False):
                st.code(
                    (jobs_root / str(latest["job_id"]) / "stderr.log")
                    .read_text(encoding="utf-8", errors="replace")[-4000:]
                )

    validation = st.session_state.get(_state_key("validation_result"))
    if not validation:
        return

    st.markdown("#### Preparation")
    st.write(
        "This loads the existing predictor and exports genuine OOF and test "
        "probabilities. It does not retrain models."
    )
    confirm = st.checkbox(
        "I confirm preparation of the exact validated request.",
        key=_state_key("confirm_prepare"),
    )
    if st.button(
        "Prepare selected candidates",
        type="primary",
        disabled=not confirm or not runtime_available,
        key=_state_key("prepare"),
    ):
        try:
            runtime = require_autogluon_python(repository_root)
            values = authorized_preparation_argv_values(request.relative_path)
            built = build_command(
                registry.commands,
                "autogluon_candidate_preparation_v1",
                "prepare",
                values,
                repository_root=repository_root,
                python_executable=str(runtime.absolute_path),
            )
            consumed = st.session_state.setdefault("_consumed_launch_nonces", set())
            previous = st.session_state.get(_state_key("rendered_prepare"))
            rendered = rendered_launch(
                built,
                previous if isinstance(previous, RenderedLaunch) else None,
                consumed_nonces=consumed,
            )
            st.session_state[_state_key("rendered_prepare")] = rendered
            authorized = authorize_launch(
                registry.commands,
                command_id="autogluon_candidate_preparation_v1",
                action_id="prepare",
                values=values,
                repository_root=repository_root,
                confirmed=confirm,
                high_risk_acknowledged=True,
                rendered=rendered,
                consumed_nonces=consumed,
                python_executable=str(runtime.absolute_path),
            )
            record = job_manager.start(
                argv=authorized.argv,
                redacted_argv=authorized.redacted_argv,
                command_id="autogluon_candidate_preparation_v1",
                action_id="prepare",
                references={
                    "request_id": request.request_id,
                    "run_path": request.run_path,
                    "dataset_id": str(request.payload.get("dataset_id") or ""),
                    "selected_models": ",".join(request.selected_models),
                    "source_type": "managed_autogluon",
                    "operation": "prepare",
                },
            )
            st.session_state[_state_key("prepare_job_id")] = record.job_id
            st.success(f"Started preparation job `{record.job_id}`.")
        except (
            CandidatePreparationError,
            PythonRuntimeError,
            LaunchAuthorizationError,
            JobError,
        ) as error:
            st.error(str(error))

    prepare_jobs = [item for item in related if item.get("action_id") == "prepare"]
    if not prepare_jobs:
        return
    latest_prepare = prepare_jobs[0]
    st.write(
        {
            "Preparation job": latest_prepare.get("job_id"),
            "Status": latest_prepare.get("state"),
        }
    )
    if latest_prepare.get("state") != "succeeded":
        if latest_prepare.get("state") in {"failed", "stopped", "orphaned"}:
            with st.expander("Technical details", expanded=False):
                st.code(
                    (jobs_root / str(latest_prepare["job_id"]) / "stderr.log")
                    .read_text(encoding="utf-8", errors="replace")[-4000:]
                )
        return
    try:
        stdout = (
            jobs_root / str(latest_prepare["job_id"]) / "stdout.log"
        ).read_text(encoding="utf-8", errors="replace")
        verified = verify_preparation_result(
            parse_job_stdout_json(stdout),
            request_id=request.request_id,
            run_path=request.run_path,
            selected_models=request.selected_models,
            repository_root=repository_root,
        )
        st.session_state[_state_key("preparation_result")] = verified
        st.success("Preparation completed and candidates were strictly validated.")
        frame = pd.DataFrame(verified["candidates"])
        st.dataframe(frame, width="stretch", hide_index=True)
        st.caption(
            "Blend-ready means the canonical candidate can participate in blending. "
            "Atomic AutoGluon candidates may still show Submission readiness: "
            "Blocked — final_threshold_missing until a blend or deployment step "
            "records a final threshold."
        )
        if st.button("Refresh canonical candidates", key=_state_key("refresh_canonical")):
            st.session_state["blend_workspace:force_refresh_candidates"] = True
        if st.button("Open Build blend", key=_state_key("open_build")):
            st.session_state["blend_workspace:active_tab"] = "Build blend"
            st.rerun()
    except (CandidatePreparationError, Exception) as error:  # noqa: BLE001
        st.error(str(error))
        with st.expander("Technical details", expanded=False):
            st.code(
                (jobs_root / str(latest_prepare["job_id"]) / "stderr.log")
                .read_text(encoding="utf-8", errors="replace")[-4000:]
            )
