from __future__ import annotations

import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import streamlit as st

try:
    import plotly.express as px  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional at runtime if env is incomplete
    px = None  # type: ignore[assignment]


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.churn_ml.control_panel.artifacts import (  # noqa: E402
    ArtifactRecord,
    ArtifactReadError,
    artifact_selector_options,
    build_experiment_table_rows,
    configured_artifact_file,
    comparison_rows,
    csv_preview,
    default_comparison_id,
    discover_artifacts,
    display_compatibility_summary,
    sanitize_comparison_id,
    text_tail,
)
from src.churn_ml.control_panel.command_builder import (  # noqa: E402
    CommandBuildError,
    build_command,
    display_argv,
)
from src.churn_ml.control_panel.config_editor import (  # noqa: E402
    ConfigEditError,
    parse_config_text,
    read_config,
    save_config_copy,
)
from src.churn_ml.control_panel.dataset_experiment_materializer import (  # noqa: E402
    DatasetExperimentMaterializerError,
    PreparedExperimentBundle,
    RegisteredDatasetView,
    build_dataset_driven_pre_run_summary,
    discover_registry_summaries,
    filter_templates,
    list_base_templates,
    prepare_dataset_driven_experiment,
    selection_fingerprint,
)
from src.churn_ml.control_panel.dataset_identity import (  # noqa: E402
    enrich_experiment_core_references,
    read_dataset_identity_safe,
    read_paired_comparison_datasets,
)
from src.churn_ml.control_panel.launch import (  # noqa: E402
    LaunchAuthorizationError,
    RenderedLaunch,
    authorize_launch,
    rendered_launch,
)
from src.churn_ml.control_panel.formatting import human_duration  # noqa: E402
from src.churn_ml.control_panel.presentation import (  # noqa: E402
    build_cascade_options,
    build_pre_run_summary,
    cascade_available_models,
    cascade_available_modes,
    cascade_available_sources,
    cascade_filter_configs,
    enum_human_label,
    format_date,
    format_time,
    is_raw_run_id,
    job_primary_label,
    mode_badge,
    mode_human_label,
    model_human_label,
    parse_config_metadata,
    readable_config_label,
    readable_path_label,
    source_human_label,
)
from src.churn_ml.control_panel.jobs import JobError, JobManager  # noqa: E402
from src.churn_ml.control_panel.mlflow_post_index import (  # noqa: E402
    DEFAULT_MLFLOW_CONFIG,
    maybe_index_successful_job,
)
from src.churn_ml.control_panel.placeholder_suggestions import (  # noqa: E402
    SuggestionError,
    render_suggested_value_template,
)
from src.churn_ml.control_panel.registry import (  # noqa: E402
    ControlPanelRegistry,
    load_registry,
)
from src.churn_ml.control_panel.schemas import (  # noqa: E402
    ActionSpec,
    PlaceholderSpec,
    SchemaError,
)
from src.churn_ml.control_panel.selection_state import (  # noqa: E402
    apply_cascade_reconciliation,
    get_durable_value,
    remember_durable_value,
    remember_widget_selection,
    reconcile_cascade_selection,
    seed_widget_from_logical,
    set_durable_value,
    set_logical_selection,
    sync_widget_with_durable,
    ui_durable_key,
    widget_selection_key,
)


st.set_page_config(
    page_title="Experiment Control Panel",
    page_icon="🧪",
    layout="wide",
)


@st.cache_resource
def registry() -> ControlPanelRegistry:
    return load_registry(REPOSITORY_ROOT)


def job_manager(loaded: ControlPanelRegistry) -> JobManager:
    return JobManager(
        REPOSITORY_ROOT / loaded.settings.jobs_root,
        working_directory=REPOSITORY_ROOT / loaded.settings.working_directory,
        commands=loaded.commands,
    )


def dashboard_page() -> None:
    loaded = registry()
    st.title("Dashboard")
    st.caption(
        "Local operational control panel. Filesystem experiment artifacts remain "
        "authoritative; UI job records do not."
    )
    jobs = job_manager(loaded).list_jobs()
    counts = Counter(str(item.status["state"]) for item in jobs)
    columns = st.columns(4)
    for column, state in zip(
        columns, ("running", "succeeded", "failed", "stopped"), strict=True
    ):
        column.metric(state.replace("_", " ").title(), counts.get(state, 0))

    st.subheader("Recent jobs")
    if jobs:
        st.dataframe(
            [
                {
                    "label": job_primary_label(
                        item.job,
                        record_commands=loaded.commands,
                        repository_root=REPOSITORY_ROOT,
                    ),
                    "status": item.status["state"],
                    "date": format_date(item.job.get("created_at_utc")),
                    "time": format_time(item.job.get("created_at_utc")),
                }
                for item in jobs[:10]
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No UI jobs have been started.")

    st.subheader("Recent artifacts")
    artifacts = _all_artifacts(loaded)
    if artifacts:
        st.dataframe(
            [
                {
                    "reader": loaded.readers[item.reader_id].title
                    if item.reader_id in loaded.readers
                    else item.reader_id,
                    "artifact": readable_path_label(
                        item.relative_path, REPOSITORY_ROOT
                    ),
                    "path": item.relative_path,
                    "status": item.state,
                }
                for item in artifacts[:10]
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No artifacts match the configured readers.")

    st.subheader("Configured command groups")
    st.dataframe(
        [
            {
                "title": command.title,
                "category": command.category,
                "actions": ", ".join(command.actions),
            }
            for command in loaded.commands.values()
        ],
        width="stretch",
        hide_index=True,
    )
    st.link_button("Open MLflow", loaded.settings.mlflow_url)


def run_page() -> None:
    loaded = registry()
    st.title("Run")
    # Consume explicit Results→Run overrides once so they win over older durable
    # state without permanently locking the Operation selector.
    prefill = st.session_state.pop("run_prefill", None) or {}
    if not isinstance(prefill, dict):
        prefill = {}
    command_ids = list(loaded.commands)
    command_widget_key = "run-command"
    command_durable_key = ui_durable_key("run", "command")
    preferred_command = prefill.get("command_id")
    if preferred_command in command_ids:
        st.session_state[command_widget_key] = preferred_command
        set_durable_value(st.session_state, command_durable_key, preferred_command)
    else:
        sync_widget_with_durable(
            st.session_state,
            widget_key=command_widget_key,
            durable_key=command_durable_key,
            allowed=command_ids,
            default=command_ids[0] if command_ids else None,
        )
    command_id = st.selectbox(
        "Operation",
        command_ids,
        key=command_widget_key,
        format_func=lambda value: loaded.commands[value].title,
    )
    remember_durable_value(
        st.session_state,
        durable_key=command_durable_key,
        value=command_id,
        allowed=command_ids,
    )
    command = loaded.commands[command_id]
    action_ids = list(command.actions)
    action_widget_key = f"run-action-{command_id}"
    action_durable_key = ui_durable_key("run", "action", command_id)
    preferred_action = prefill.get("action_id")
    if preferred_action in action_ids:
        st.session_state[action_widget_key] = preferred_action
        set_durable_value(st.session_state, action_durable_key, preferred_action)
    else:
        sync_widget_with_durable(
            st.session_state,
            widget_key=action_widget_key,
            durable_key=action_durable_key,
            allowed=action_ids,
            default=action_ids[0] if action_ids else None,
        )
    action_id = st.selectbox(
        "Action",
        action_ids,
        key=action_widget_key,
        format_func=lambda value: command.actions[value].title,
    )
    remember_durable_value(
        st.session_state,
        durable_key=action_durable_key,
        value=action_id,
        allowed=action_ids,
    )
    action = command.actions[action_id]
    st.write(action.description)
    risk = (
        "Competition-test / deployment"
        if action.competition_test
        else action.confirmation
    )
    st.caption(f"Safety level: {risk}")
    if not action.enabled:
        st.warning("This action is disabled by the declarative registry.")

    values: dict[str, Any] = {}
    selected_config: Path | None = None
    dataset_driven_summary: dict[str, str] = {}
    dataset_driven_blocks_launch = False
    use_dataset_driven = False
    if command_id == "experiment_core_v2":
        entry_modes = ("Dataset-driven experiment", "Existing config")
        entry_key = "ecv2-entry-mode"
        entry_durable = ui_durable_key("run", "ecv2_entry_mode")
        sync_widget_with_durable(
            st.session_state,
            widget_key=entry_key,
            durable_key=entry_durable,
            allowed=list(entry_modes),
            # Existing config remains the fast default so Run loads without a
            # full Registry scan; Dataset-driven is the recommended workflow.
            default=entry_modes[1],
        )
        entry_mode = st.radio(
            "Experiment Core entry mode",
            entry_modes,
            key=entry_key,
            horizontal=True,
            help=(
                "Prefer Dataset-driven experiment for registered Dataset Packages. "
                "Existing config keeps historical Research v2 launches unchanged."
            ),
        )
        remember_durable_value(
            st.session_state,
            durable_key=entry_durable,
            value=entry_mode,
            allowed=list(entry_modes),
        )
        use_dataset_driven = entry_mode == "Dataset-driven experiment"
        if use_dataset_driven:
            (
                prepared_config,
                dataset_driven_summary,
                dataset_driven_blocks_launch,
            ) = _experiment_core_dataset_driven_controls(loaded, command_id=command_id)
            if prepared_config:
                values["config"] = prepared_config
                selected_config = REPOSITORY_ROOT / prepared_config
                config_widget = widget_selection_key(command_id, "config", "config")
                st.session_state[config_widget] = prepared_config
                set_logical_selection(
                    st.session_state, command_id, "config", "config", prepared_config
                )

    prefill_values = (
        prefill.get("values", {})
        if prefill.get("command_id") == command_id
        and prefill.get("action_id") == action_id
        else {}
    )
    if not use_dataset_driven:
        for name, value in prefill_values.items():
            placeholder = action.placeholders.get(name)
            role = placeholder.role if placeholder is not None else "value"
            shared_key = widget_selection_key(command_id, role, name)
            # Force override older durable/widget state from Results prepare actions.
            st.session_state[shared_key] = value
            set_logical_selection(st.session_state, command_id, role, name, value)
            # Keep legacy per-action key in sync for older session handoffs.
            st.session_state[f"value-{command_id}-{action_id}-{name}"] = value

        for name, placeholder in action.placeholders.items():
            widget_key = widget_selection_key(command_id, placeholder.role, name)
            # Migrate a previous per-action value once when shared key is empty.
            legacy_key = f"value-{command_id}-{action_id}-{name}"
            if st.session_state.get(widget_key) in (None, "") and st.session_state.get(
                legacy_key
            ) not in (None, ""):
                st.session_state[widget_key] = st.session_state[legacy_key]
            value = _placeholder_widget(
                loaded,
                command.allowed_config_globs,
                name,
                placeholder,
                widget_key,
                values,
                operation=command_id,
            )
            if value not in (None, ""):
                values[name] = value
                if placeholder.role == "config":
                    selected_config = REPOSITORY_ROOT / str(value)
    else:
        for name, placeholder in action.placeholders.items():
            if placeholder.role == "config" or name == "config":
                continue
            widget_key = widget_selection_key(command_id, placeholder.role, name)
            value = _placeholder_widget(
                loaded,
                command.allowed_config_globs,
                name,
                placeholder,
                widget_key,
                values,
                operation=command_id,
            )
            if value not in (None, ""):
                values[name] = value

    if selected_config is not None and selected_config.is_file():
        _config_panel(loaded, selected_config)

    st.session_state["_last_command_id"] = command_id
    st.session_state["_last_action_id"] = action_id

    built = None
    pre_run: dict[str, str] = {}
    try:
        if action.enabled and not dataset_driven_blocks_launch:
            built = build_command(
                loaded.commands,
                command_id,
                action_id,
                values,
                repository_root=REPOSITORY_ROOT,
            )
            pre_run = build_pre_run_summary(
                command_id,
                action_id,
                command.title,
                action.title,
                values,
                REPOSITORY_ROOT,
            )
            if dataset_driven_summary:
                merged = {**dataset_driven_summary, **pre_run}
                # Keep dataset-driven identity fields ahead of generic summary.
                ordered: dict[str, str] = {}
                for key in (
                    "Operation",
                    "Action",
                    "Dataset Package",
                    "Parent dataset",
                    "Target dependency",
                    "Pipeline",
                    "Model",
                    "Evaluation mode",
                    "Mode",
                    "Base template",
                    "Prepared config",
                    "Prepared evaluation plan",
                    "Source",
                    "Config",
                    "Plan",
                ):
                    if key in merged:
                        ordered[key] = merged[key]
                for key, value in merged.items():
                    ordered.setdefault(key, value)
                pre_run = ordered
            if pre_run:
                st.markdown("  \n".join(f"**{k}:** {v}" for k, v in pre_run.items()))
            with st.expander("Technical command", expanded=False):
                st.code(display_argv(built.redacted_argv), language="python")
        elif dataset_driven_blocks_launch:
            st.info(
                "Prepare a valid dataset-driven configuration before Validate/Run."
            )
        else:
            with st.expander("Technical command (action disabled)", expanded=False):
                st.code(
                    display_argv(command.argv_prefix + action.argv),
                    language="python",
                )
    except CommandBuildError as error:
        st.error(str(error))

    consumed = st.session_state.setdefault("_consumed_launch_nonces", set())
    if not isinstance(consumed, set):
        consumed = set()
        st.session_state["_consumed_launch_nonces"] = consumed
    rendered: RenderedLaunch | None = None
    if built is not None:
        previous = st.session_state.get("_rendered_launch")
        rendered = rendered_launch(
            built,
            previous if isinstance(previous, RenderedLaunch) else None,
            consumed_nonces=consumed,
        )
        st.session_state["_rendered_launch"] = rendered
    else:
        st.session_state["_rendered_launch"] = None
    if action.competition_test or action.confirmation not in {"none", ""}:
        config_path = values.get("config", "")
        _meta = (
            parse_config_metadata(str(config_path), REPOSITORY_ROOT)
            if config_path
            else {}
        )
        _model = _meta.get("model_family", "—")
        _mode = _meta.get("mode", "—")
        _cfg_base = Path(str(config_path)).name if config_path else "—"
        if action.competition_test:
            _level_badge = "[DEPLOYMENT]"
        elif _meta.get("mode"):
            _level_badge = f"[{mode_badge(_meta['mode'])}]"
        else:
            _level_badge = ""
        st.warning(
            f"⚠️ Review before launch\n\n"
            f"Model: **{_model}** | Mode: **{_mode}** | "
            f"Action: **{action.title}** | Config: `{_cfg_base}` {_level_badge}"
        )

    confirmed = True
    if action.confirmation in {"confirm", "acknowledge"}:
        confirmed = st.checkbox(
            "I confirm this approved action and have reviewed the exact argv.",
            key=f"confirm-{command_id}-{action_id}",
        )
    acknowledged = True
    if action.competition_test:
        acknowledged = st.checkbox(
            "I explicitly acknowledge competition-test/deployment access.",
            key=f"ack-{command_id}-{action_id}",
        )
    if st.button(
        "Start background job",
        type="primary",
        disabled=built is None or not confirmed or not acknowledged,
        key="start-background-job",
    ):
        try:
            authorized = authorize_launch(
                loaded.commands,
                command_id=command_id,
                action_id=action_id,
                values=values,
                repository_root=REPOSITORY_ROOT,
                confirmed=confirmed,
                high_risk_acknowledged=acknowledged,
                rendered=rendered,
                consumed_nonces=consumed,
            )
            references = enrich_experiment_core_references(
                authorized.references,
                repository_root=REPOSITORY_ROOT,
                command_id=command_id,
            )
            record = job_manager(loaded).start(
                argv=authorized.argv,
                redacted_argv=authorized.redacted_argv,
                command_id=command_id,
                action_id=action_id,
                references=references,
            )
            st.success(f"Started job {record.job_id}.")
        except (LaunchAuthorizationError, JobError) as error:
            st.error(str(error))


def jobs_page() -> None:
    loaded = registry()
    manager = job_manager(loaded)
    st.title("Jobs")
    if st.button("Refresh job status"):
        st.cache_data.clear()
    jobs = manager.list_jobs()
    if not jobs:
        st.info("No UI jobs have been started.")
        return
    job_ids = [item.job_id for item in jobs]
    job_widget_key = "jobs-selected"
    job_durable_key = ui_durable_key("jobs", "selected")
    sync_widget_with_durable(
        st.session_state,
        widget_key=job_widget_key,
        durable_key=job_durable_key,
        allowed=job_ids,
        default=job_ids[0],
    )
    selected = st.selectbox(
        "Job",
        job_ids,
        key=job_widget_key,
        format_func=lambda value: _job_label(
            next(item for item in jobs if item.job_id == value),
            commands=loaded.commands,
        ),
    )
    remember_durable_value(
        st.session_state,
        durable_key=job_durable_key,
        value=selected,
        allowed=job_ids,
    )
    st.caption(f"Job ID: `{selected}`")
    record = manager.refresh(selected)
    status = record.status
    columns = st.columns(4)
    columns[0].metric("Status", str(status["state"]))
    columns[1].metric("Elapsed", human_duration(status.get("elapsed_seconds")))
    created = record.job.get("created_at_utc")
    columns[2].metric("Date", format_date(created))
    columns[3].metric("Time", format_time(created))
    references = record.job.get("references")
    if isinstance(references, Mapping) and (
        references.get("dataset_id")
        or references.get("experiment_id")
        or references.get("plan_id")
        or references.get("config")
    ):
        st.markdown(
            "  \n".join(
                [
                    f"**Dataset ID:** `{references.get('dataset_id') or 'Not available'}`",
                    f"**Experiment ID:** `{references.get('experiment_id') or 'Not available'}`",
                    f"**Plan ID:** `{references.get('plan_id') or 'Not available'}`",
                    f"**Config:** `{references.get('config') or 'Not available'}`",
                ]
            )
        )
    if status.get("diagnostic"):
        st.warning(str(status["diagnostic"]))
    index_result = _maybe_post_index_job(loaded, record)
    if index_result is not None and index_result.attempted:
        if index_result.status in {"succeeded", "idempotent", "unchanged"}:
            st.caption(
                f"MLflow index: `{index_result.status}`"
                + (
                    f" · `{index_result.artifact_path}`"
                    if index_result.artifact_path
                    else ""
                )
            )
        elif index_result.status == "failed":
            st.info(
                "Training succeeded; MLflow indexing failed and was recorded "
                "separately without changing job status."
                + (f" ({index_result.message})" if index_result.message else "")
            )
    with st.expander("Technical details", expanded=False):
        st.json(dict(record.job), expanded=False)
        st.caption(
            f"PID: {status.get('pid') or '—'}  |  "
            f"Exit code: {status.get('exit_code') if status.get('exit_code') is not None else '—'}"
        )
    with st.expander("Technical command", expanded=False):
        st.code(display_argv(tuple(record.command["argv"])), language="python")
    stdout, stderr = st.tabs(["stdout", "stderr"])
    with stdout:
        st.code(
            text_tail(
                record.root / "stdout.log",
                lines=loaded.settings.log_tail_lines,
            )
            or "(empty)",
            language="text",
        )
    with stderr:
        st.code(
            text_tail(
                record.root / "stderr.log",
                lines=loaded.settings.log_tail_lines,
            )
            or "(empty)",
            language="text",
        )
    if loaded.settings.allow_process_stop and status["state"] == "running":
        stop_confirmed = st.checkbox(
            "I confirm that I want to stop this process group."
        )
        if st.button("Stop job", disabled=not stop_confirmed):
            try:
                manager.request_stop(selected)
                st.success("Stop requested.")
            except JobError as error:
                st.error(str(error))


def _maybe_post_index_job(loaded: ControlPanelRegistry, record: Any) -> Any:
    """Best-effort MLflow indexing after successful Experiment Core / AutoGluon jobs."""
    try:
        stdout_text = (record.root / "stdout.log").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        stdout_text = ""
    argv = record.command.get("argv") if isinstance(record.command, Mapping) else None
    argv_list = [str(item) for item in argv] if isinstance(argv, (list, tuple)) else []
    return maybe_index_successful_job(
        job_root=record.root,
        command_id=str(record.job.get("command_id") or ""),
        action_id=str(record.job.get("action_id") or ""),
        argv=argv_list,
        job_status=str(record.status.get("state") or ""),
        stdout_text=stdout_text,
        repository_root=REPOSITORY_ROOT,
        mlflow_config_path=DEFAULT_MLFLOW_CONFIG,
    )


def results_page() -> None:
    loaded = registry()
    st.title("Results")
    if loaded.settings.mlflow_url:
        st.link_button("Open MLflow", loaded.settings.mlflow_url)
    experiments_tab, inspect_tab, compare_tab = st.tabs(
        ["Experiments", "Inspect result", "Compare experiments"]
    )
    with experiments_tab:
        _results_experiments_tab(loaded)
    with inspect_tab:
        _results_inspect_tab(loaded)
    with compare_tab:
        _results_compare_tab(loaded)


def _results_reader_select(loaded: ControlPanelRegistry, key: str) -> Any:
    reader_ids = list(loaded.readers)
    durable_key = ui_durable_key("results", "reader", key)
    sync_widget_with_durable(
        st.session_state,
        widget_key=key,
        durable_key=durable_key,
        allowed=reader_ids,
        default=reader_ids[0] if reader_ids else None,
    )
    selected = st.selectbox(
        "Artifact type",
        reader_ids,
        key=key,
        format_func=lambda value: loaded.readers[value].title,
    )
    remember_durable_value(
        st.session_state,
        durable_key=durable_key,
        value=selected,
        allowed=reader_ids,
    )
    return selected


def _results_experiments_tab(loaded: ControlPanelRegistry) -> None:
    reader_id = _results_reader_select(loaded, "results-experiments-reader")
    reader = loaded.readers[reader_id]
    artifacts = discover_artifacts(REPOSITORY_ROOT, reader)
    if not artifacts:
        st.info("No artifacts match this configured reader.")
        return

    rows = build_experiment_table_rows(artifacts, repo_root=REPOSITORY_ROOT)
    datasets = sorted(
        {
            str(row["Dataset"])
            for row in rows
            if row.get("Dataset") not in {None, "", "Not available"}
        }
    )
    models = sorted(
        {str(row["Model"]) for row in rows if row["Model"] != "Not available"}
    )
    modes = sorted({str(row["Mode"]) for row in rows if row["Mode"] != "Not available"})
    statuses = sorted({str(row["Status"]) for row in rows})

    filter_cols = st.columns(5)
    dataset_key = f"results-exp-filter-dataset-{reader_id}"
    model_key = f"results-exp-filter-model-{reader_id}"
    mode_key = f"results-exp-filter-mode-{reader_id}"
    status_key = f"results-exp-filter-status-{reader_id}"
    search_key = f"results-exp-filter-search-{reader_id}"
    dataset_options = ["All", *datasets]
    model_options = ["All", *models]
    mode_options = ["All", *modes]
    status_options = ["All", *statuses]
    sync_widget_with_durable(
        st.session_state,
        widget_key=dataset_key,
        durable_key=ui_durable_key("results", "filter", reader_id, "dataset"),
        allowed=dataset_options,
        default="All",
    )
    sync_widget_with_durable(
        st.session_state,
        widget_key=model_key,
        durable_key=ui_durable_key("results", "filter", reader_id, "model"),
        allowed=model_options,
        default="All",
    )
    sync_widget_with_durable(
        st.session_state,
        widget_key=mode_key,
        durable_key=ui_durable_key("results", "filter", reader_id, "mode"),
        allowed=mode_options,
        default="All",
    )
    sync_widget_with_durable(
        st.session_state,
        widget_key=status_key,
        durable_key=ui_durable_key("results", "filter", reader_id, "status"),
        allowed=status_options,
        default="All",
    )
    sync_widget_with_durable(
        st.session_state,
        widget_key=search_key,
        durable_key=ui_durable_key("results", "filter", reader_id, "search"),
        allowed=None,
        default="",
    )
    selected_dataset = filter_cols[0].selectbox(
        "Dataset",
        dataset_options,
        key=dataset_key,
    )
    selected_model = filter_cols[1].selectbox(
        "Model",
        model_options,
        key=model_key,
    )
    selected_mode = filter_cols[2].selectbox(
        "Mode",
        mode_options,
        key=mode_key,
    )
    selected_status = filter_cols[3].selectbox(
        "Status",
        status_options,
        key=status_key,
    )
    search = filter_cols[4].text_input(
        "Search experiment/config",
        key=search_key,
    )
    remember_durable_value(
        st.session_state,
        durable_key=ui_durable_key("results", "filter", reader_id, "dataset"),
        value=selected_dataset,
        allowed=dataset_options,
    )
    remember_durable_value(
        st.session_state,
        durable_key=ui_durable_key("results", "filter", reader_id, "model"),
        value=selected_model,
        allowed=model_options,
    )
    remember_durable_value(
        st.session_state,
        durable_key=ui_durable_key("results", "filter", reader_id, "mode"),
        value=selected_mode,
        allowed=mode_options,
    )
    remember_durable_value(
        st.session_state,
        durable_key=ui_durable_key("results", "filter", reader_id, "status"),
        value=selected_status,
        allowed=status_options,
    )
    remember_durable_value(
        st.session_state,
        durable_key=ui_durable_key("results", "filter", reader_id, "search"),
        value=search,
        allowed=None,
    )

    filtered = []
    for row in rows:
        if selected_dataset != "All" and row.get("Dataset") != selected_dataset:
            continue
        if selected_model != "All" and row["Model"] != selected_model:
            continue
        if selected_mode != "All" and row["Mode"] != selected_mode:
            continue
        if selected_status != "All" and row["Status"] != selected_status:
            continue
        if search:
            needle = search.lower()
            haystack = " ".join(
                str(row.get(key, ""))
                for key in (
                    "Dataset",
                    "Parent dataset",
                    "Experiment",
                    "Model",
                    "Mode",
                    "Artifact path",
                )
            ).lower()
            if needle not in haystack:
                continue
        filtered.append(row)

    display_columns = [
        "Dataset",
        "Parent dataset",
        "Target dependency",
        "Features",
        "Model",
        "Mode",
        "Experiment",
        "Created date",
        "Created time",
        "Balanced Accuracy",
        "Sensitivity",
        "Specificity",
        "ROC AUC",
        "Average Precision",
        "Brier score",
        "Threshold median",
        "Status",
        "Artifact path",
    ]
    display_rows = [{key: row.get(key) for key in display_columns} for row in filtered]
    if not display_rows:
        st.info("No experiments match the current filters.")
    else:
        st.dataframe(
            display_rows,
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Select one or more rows when Streamlit selection is available. "
            "Artifact path is technical metadata only."
        )

    _render_experiment_charts(filtered)


def _render_experiment_charts(rows: list[dict[str, Any]]) -> None:
    st.subheader("Experiment charts")
    if not rows:
        st.info("Charts appear when at least one experiment row is available.")
        return
    if len(rows) == 1:
        st.caption(
            "Only one experiment is in view; charts still render for inspection."
        )

    chart_frame = []
    for row in rows:
        chart_key = (
            row.get("_chart_key") or row.get("Artifact path") or row.get("Experiment")
        )
        chart_label = row.get("_chart_label") or row.get("Experiment")
        chart_frame.append(
            {
                "Experiment key": chart_key,
                "Experiment": chart_label,
                "Dataset": row.get("Dataset"),
                "Model": row.get("Model"),
                "Mode": row.get("Mode"),
                "Created date": row.get("Created date"),
                "Created time": row.get("Created time"),
                "Balanced Accuracy": row.get("_ba"),
                "Sensitivity": row.get("_sensitivity"),
                "Specificity": row.get("_specificity"),
                "ROC AUC": row.get("_roc_auc"),
                "Average Precision": row.get("_average_precision"),
            }
        )
    frame = pd.DataFrame(chart_frame)
    hover = [
        "Experiment",
        "Dataset",
        "Model",
        "Mode",
        "Created date",
        "Created time",
        "Balanced Accuracy",
    ]

    if px is None:
        st.warning("Plotly is unavailable; showing tabular chart data instead.")
        st.dataframe(frame, width="stretch", hide_index=True)
        return

    ba_frame = frame.dropna(subset=["Balanced Accuracy"])
    if ba_frame.empty:
        st.info("Balanced Accuracy is not available for the filtered experiments.")
    else:
        fig_ba = px.bar(
            ba_frame,
            x="Experiment key",
            y="Balanced Accuracy",
            hover_data=hover,
            title="Balanced Accuracy by experiment",
        )
        tick_text = [str(label) for label in ba_frame["Experiment"].tolist()]
        tick_vals = [str(key) for key in ba_frame["Experiment key"].tolist()]
        fig_ba.update_layout(
            xaxis_title="Experiment",
            yaxis_title="Balanced Accuracy",
            xaxis={"tickmode": "array", "tickvals": tick_vals, "ticktext": tick_text},
        )
        ba_values = [float(value) for value in ba_frame["Balanced Accuracy"].tolist()]
        if ba_values:
            low = min(ba_values)
            high = max(ba_values)
            pad = max(0.02, (high - low) * 0.15) if high > low else 0.05
            fig_ba.update_yaxes(range=[max(0.0, low - pad), min(1.05, high + pad)])
        st.plotly_chart(fig_ba, width="stretch")

    sens_frame = frame.dropna(subset=["Sensitivity", "Specificity"])
    if sens_frame.empty:
        st.info("Sensitivity/Specificity are not available for scatter plotting.")
    else:
        fig_ss = px.scatter(
            sens_frame,
            x="Sensitivity",
            y="Specificity",
            text="Experiment",
            hover_data=hover,
            title="Sensitivity vs Specificity",
        )
        fig_ss.update_traces(textposition="top center")
        st.plotly_chart(fig_ss, width="stretch")

    roc_frame = frame.dropna(subset=["ROC AUC", "Average Precision"])
    if roc_frame.empty:
        st.info("ROC AUC / Average Precision are not available for scatter plotting.")
    else:
        fig_roc = px.scatter(
            roc_frame,
            x="ROC AUC",
            y="Average Precision",
            text="Experiment",
            hover_data=hover,
            title="ROC AUC vs Average Precision",
        )
        fig_roc.update_traces(textposition="top center")
        st.plotly_chart(fig_roc, width="stretch")


def _results_inspect_tab(loaded: ControlPanelRegistry) -> None:
    reader_id = _results_reader_select(loaded, "results-inspect-reader")
    reader = loaded.readers[reader_id]
    artifacts = discover_artifacts(REPOSITORY_ROOT, reader)
    if not artifacts:
        st.info("No artifacts match this configured reader.")
        return
    artifact_paths = [item.relative_path for item in artifacts]
    source_kinds = {
        opt.source_kind
        for opt in build_cascade_options(artifact_paths, REPOSITORY_ROOT)
    }
    inspect_key = f"results-inspect-{reader_id}"
    seed_widget_from_logical(
        st.session_state,
        operation="__results__",
        role="inspect",
        name=reader_id,
        widget_key=inspect_key,
        allowed=artifact_paths,
        cascade_meta=None,
    )
    seeded_inspect = st.session_state.get(inspect_key)
    if seeded_inspect not in (None, "") and seeded_inspect in artifact_paths:
        seed_widget_from_logical(
            st.session_state,
            operation="__results__",
            role="inspect",
            name=reader_id,
            widget_key=inspect_key,
            allowed=artifact_paths,
            cascade_meta=parse_config_metadata(str(seeded_inspect), REPOSITORY_ROOT),
        )
    selected_path = _cascade_item_selector(
        artifact_paths,
        widget_key=inspect_key,
        item_label="Experiment",
        include_source=len(source_kinds) > 1,
        advanced_label="Advanced: raw artifact path",
    )
    remember_widget_selection(
        st.session_state,
        operation="__results__",
        role="inspect",
        name=reader_id,
        value=selected_path,
        allowed=artifact_paths,
        widget_key=inspect_key,
    )
    if not selected_path:
        return
    selected = next(item for item in artifacts if item.relative_path == selected_path)
    st.caption(f"Config basename: `{Path(selected_path).name}`")
    st.code(selected_path, language="text")
    st.write(f"Status: `{selected.state}`")
    if selected.diagnostic:
        st.warning(selected.diagnostic)
    if reader_id == "research_v2":
        _render_dataset_identity_summary(selected.root)
    elif reader_id == "research_v2_comparisons":
        baseline_ds, candidate_ds = read_paired_comparison_datasets(selected.root)
        st.subheader("Comparison datasets")
        st.markdown(
            "  \n".join(
                [
                    f"**Baseline dataset:** `{baseline_ds or 'Not available'}`",
                    f"**Candidate dataset:** `{candidate_ds or 'Not available'}`",
                ]
            )
        )
    if selected.summaries:
        st.dataframe(
            [
                {"field": key, "value": "" if value is None else str(value)}
                for key, value in selected.summaries.items()
            ],
            width="stretch",
            hide_index=True,
        )
    for name, payload in selected.json_payloads.items():
        with st.expander(name):
            st.json(payload)
    for relative in reader.csv_previews:
        try:
            path = configured_artifact_file(REPOSITORY_ROOT, selected, relative)
        except ArtifactReadError:
            continue
        if path.is_file():
            with st.expander(f"CSV preview: {relative}"):
                st.dataframe(csv_preview(path), width="stretch")
    for relative in reader.log_files:
        try:
            path = configured_artifact_file(REPOSITORY_ROOT, selected, relative)
        except ArtifactReadError:
            continue
        if path.is_file():
            with st.expander(f"Log: {relative}"):
                st.code(
                    text_tail(path, lines=loaded.settings.log_tail_lines),
                    language="text",
                )

    if reader_id == "deployment_v1":
        deployment_valid = selected.state != "invalid"
        if (
            st.button(
                "Prepare deployment Inspect action",
                disabled=not deployment_valid,
                key="results-prepare-deployment-inspect",
            )
            and deployment_valid
        ):
            st.session_state["run-command"] = "final_deployment_v1"
            st.session_state["run-action-final_deployment_v1"] = "inspect"
            set_durable_value(
                st.session_state, ui_durable_key("run", "command"), "final_deployment_v1"
            )
            set_durable_value(
                st.session_state,
                ui_durable_key("run", "action", "final_deployment_v1"),
                "inspect",
            )
            deployment_key = widget_selection_key(
                "final_deployment_v1", "input", "deployment_dir"
            )
            st.session_state[deployment_key] = selected.relative_path
            remember_widget_selection(
                st.session_state,
                operation="final_deployment_v1",
                role="input",
                name="deployment_dir",
                value=selected.relative_path,
                allowed=None,
                widget_key=deployment_key,
            )
            st.session_state["run_prefill"] = {
                "command_id": "final_deployment_v1",
                "action_id": "inspect",
                "values": {"deployment_dir": selected.relative_path},
            }
            st.success(
                "Prepared the allowlisted action. Open Run to review argv and start."
            )

    _results_registry_actions(loaded, reader_id)


def _results_compare_tab(loaded: ControlPanelRegistry) -> None:
    reader_id = _results_reader_select(loaded, "results-compare-reader")
    reader = loaded.readers[reader_id]
    artifacts = discover_artifacts(REPOSITORY_ROOT, reader)
    if not artifacts:
        st.info("No artifacts match this configured reader.")
        return
    if len(artifacts) < 2:
        st.info("At least two artifacts are required for comparison.")
        return

    artifact_paths = [item.relative_path for item in artifacts]
    source_kinds = {
        opt.source_kind
        for opt in build_cascade_options(artifact_paths, REPOSITORY_ROOT)
    }
    include_source = len(source_kinds) > 1

    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown("**Left**")
        left_key = f"compare-left-{reader_id}"
        seed_widget_from_logical(
            st.session_state,
            operation="__results__",
            role="compare_left",
            name=reader_id,
            widget_key=left_key,
            allowed=artifact_paths,
            cascade_meta=None,
        )
        seeded_left = st.session_state.get(left_key)
        if seeded_left not in (None, "") and seeded_left in artifact_paths:
            seed_widget_from_logical(
                st.session_state,
                operation="__results__",
                role="compare_left",
                name=reader_id,
                widget_key=left_key,
                allowed=artifact_paths,
                cascade_meta=parse_config_metadata(str(seeded_left), REPOSITORY_ROOT),
            )
        left_path = _cascade_item_selector(
            artifact_paths,
            widget_key=left_key,
            item_label="Experiment",
            include_source=include_source,
            model_label="Model",
            mode_label="Mode",
            show_advanced=False,
            default_index=0,
        )
        remember_widget_selection(
            st.session_state,
            operation="__results__",
            role="compare_left",
            name=reader_id,
            value=left_path,
            allowed=artifact_paths,
            widget_key=left_key,
        )
    with right_col:
        st.markdown("**Right**")
        right_key = f"compare-right-{reader_id}"
        seed_widget_from_logical(
            st.session_state,
            operation="__results__",
            role="compare_right",
            name=reader_id,
            widget_key=right_key,
            allowed=artifact_paths,
            cascade_meta=None,
        )
        seeded_right = st.session_state.get(right_key)
        if seeded_right not in (None, "") and seeded_right in artifact_paths:
            seed_widget_from_logical(
                st.session_state,
                operation="__results__",
                role="compare_right",
                name=reader_id,
                widget_key=right_key,
                allowed=artifact_paths,
                cascade_meta=parse_config_metadata(str(seeded_right), REPOSITORY_ROOT),
            )
        # Prefer a distinct initial right artifact before the widget is created.
        if (
            left_path
            and len(artifact_paths) >= 2
            and st.session_state.get(right_key) in {None, "", left_path}
        ):
            for candidate in artifact_paths:
                if candidate != left_path:
                    st.session_state[right_key] = candidate
                    break
        right_path = _cascade_item_selector(
            artifact_paths,
            widget_key=right_key,
            item_label="Experiment",
            include_source=include_source,
            model_label="Model",
            mode_label="Mode",
            show_advanced=False,
            default_index=1,
        )
        remember_widget_selection(
            st.session_state,
            operation="__results__",
            role="compare_right",
            name=reader_id,
            value=right_path,
            allowed=artifact_paths,
            widget_key=right_key,
        )
    if not left_path or not right_path:
        return

    left_item = next(item for item in artifacts if item.relative_path == left_path)
    right_item = next(item for item in artifacts if item.relative_path == right_path)

    compatibility = display_compatibility_summary(left_item, right_item)
    st.subheader("Compatibility")
    st.write(
        {
            "left dataset": compatibility.get("left_dataset_id"),
            "right dataset": compatibility.get("right_dataset_id"),
            "same evaluation plan": compatibility["same_evaluation_plan"],
            "same dataset fingerprint": compatibility["same_dataset_fingerprint"],
            "same fold assignments": compatibility["same_fold_assignments"],
            "status": compatibility["status"],
        }
    )
    if compatibility.get("descriptive_only"):
        st.warning(
            "Descriptive comparison only: the selected runs use different dataset "
            "fingerprints. The metric delta is not a paired statistical comparison "
            "and the runs are not formally compatible for official Paired Comparison."
        )
    elif compatibility["compatible"]:
        st.success("Display check: compatible fingerprints.")
    else:
        st.warning(
            "Display check: incompatible or incomplete fingerprints. "
            "Official Paired Comparison remains the authoritative gate."
        )

    compare_frame = comparison_rows(left_item, right_item, reader.compare_fields)
    st.dataframe(
        [
            {
                "Metric": row["Metric"],
                "Left": row["Left"],
                "Right": row["Right"],
                "Delta (Right - Left)": row["Delta (Right - Left)"],
            }
            for row in compare_frame
        ],
        width="stretch",
        hide_index=True,
    )
    if any(row["Metric"] == "Brier score" for row in compare_frame):
        st.caption(
            "Brier score delta is numeric only (Right - Left). Lower Brier score is better; "
            "a positive delta is not an improvement."
        )
    if reader_id == "research_v2":
        st.caption(
            "This table is display-only. Use Prepare Paired Comparison for the official "
            "compatibility gate and comparison artifact."
        )

    if reader_id == "research_v2":
        default_id = sanitize_comparison_id(
            default_comparison_id(left_item, right_item, repo_root=REPOSITORY_ROOT)
        )
        id_key = f"results-comparison-id-{reader_id}"
        suggested_key = f"{id_key}__suggested"
        manual_key = f"{id_key}__manual"
        sides_key = f"{id_key}__sides"
        reset_flag = f"{id_key}__do_reset"
        id_durable_key = ui_durable_key("results", "comparison_id", reader_id)
        sides_fp = (left_item.relative_path, right_item.relative_path)
        previous_sides = st.session_state.get(sides_key)
        st.session_state[suggested_key] = default_id
        if st.session_state.pop(reset_flag, False):
            st.session_state[id_key] = default_id
            st.session_state[manual_key] = False
            st.session_state[sides_key] = sides_fp
            set_durable_value(st.session_state, id_durable_key, default_id)
        elif previous_sides != sides_fp:
            if not st.session_state.get(manual_key):
                st.session_state[id_key] = default_id
                set_durable_value(st.session_state, id_durable_key, default_id)
            st.session_state[sides_key] = sides_fp
        elif st.session_state.get(id_key) in (None, "", "ui-paired-comparison"):
            durable_id = get_durable_value(st.session_state, id_durable_key)
            if durable_id not in (None, "", "ui-paired-comparison"):
                st.session_state[id_key] = durable_id
            else:
                st.session_state[id_key] = default_id
        comparison_id = st.text_input("New comparison ID", key=id_key)
        remember_durable_value(
            st.session_state,
            durable_key=id_durable_key,
            value=comparison_id,
            allowed=None,
        )
        if comparison_id != st.session_state.get(suggested_key):
            st.session_state[manual_key] = True
        reset_cols = st.columns([1, 3])
        with reset_cols[0]:
            if st.button("Reset to suggested ID", key=f"{id_key}__reset"):
                st.session_state[reset_flag] = True
                st.rerun()
        with reset_cols[1]:
            st.caption(f"Suggested: `{st.session_state.get(suggested_key)}`")
        comparison_ready = (
            left_item.state == "completed"
            and right_item.state == "completed"
            and bool(compatibility.get("compatible"))
        )
        if not compatibility.get("compatible"):
            st.info(
                "Prepare Paired Comparison action stays disabled until the display "
                "compatibility check passes. The CLI remains the authoritative final gate."
            )
        if (
            st.button(
                "Prepare Paired Comparison action",
                disabled=not comparison_ready,
                key="results-prepare-paired-comparison",
            )
            and comparison_ready
        ):
            st.session_state["run-command"] = "paired_comparison"
            st.session_state["run-action-paired_comparison"] = "run"
            set_durable_value(
                st.session_state, ui_durable_key("run", "command"), "paired_comparison"
            )
            set_durable_value(
                st.session_state,
                ui_durable_key("run", "action", "paired_comparison"),
                "run",
            )
            baseline_key = widget_selection_key(
                "paired_comparison", "input", "baseline_run_dir"
            )
            candidate_key = widget_selection_key(
                "paired_comparison", "input", "candidate_run_dir"
            )
            comparison_value_key = widget_selection_key(
                "paired_comparison", "value", "comparison_id"
            )
            st.session_state[baseline_key] = left_item.relative_path
            st.session_state[candidate_key] = right_item.relative_path
            st.session_state[comparison_value_key] = comparison_id
            remember_widget_selection(
                st.session_state,
                operation="paired_comparison",
                role="input",
                name="baseline_run_dir",
                value=left_item.relative_path,
                allowed=None,
                widget_key=baseline_key,
            )
            remember_widget_selection(
                st.session_state,
                operation="paired_comparison",
                role="input",
                name="candidate_run_dir",
                value=right_item.relative_path,
                allowed=None,
                widget_key=candidate_key,
            )
            remember_widget_selection(
                st.session_state,
                operation="paired_comparison",
                role="value",
                name="comparison_id",
                value=comparison_id,
                allowed=None,
                widget_key=comparison_value_key,
            )
            st.session_state["run_prefill"] = {
                "command_id": "paired_comparison",
                "action_id": "run",
                "values": {
                    "baseline_run_dir": left_item.relative_path,
                    "candidate_run_dir": right_item.relative_path,
                    "comparison_id": comparison_id,
                    "output_root": "artifacts/research_v2_comparisons",
                },
            }
            st.success(
                "Prepared the allowlisted action. Open Run to review argv and confirm."
            )

    _results_registry_actions(loaded, reader_id)


def _results_registry_actions(loaded: ControlPanelRegistry, reader_id: str) -> None:
    st.subheader("Registry-defined actions")
    action_rows = []
    for command in loaded.commands.values():
        if command.result_reader_id != reader_id:
            continue
        for action in command.actions.values():
            if not action.enabled:
                continue
            action_rows.append(
                {
                    "action": action.title,
                    "enabled": action.enabled,
                    "where": "Run page",
                    "additional_input_required": _action_requires_additional_input(
                        action
                    ),
                }
            )
    if action_rows:
        st.dataframe(action_rows, width="stretch", hide_index=True)
        st.info(
            "Launch these allowlisted actions from the Run page after reviewing argv. "
            "Actions that need an output path or config are listed even when they "
            "cannot run from Results directly."
        )
    else:
        st.info("No registry-defined actions are available for this reader.")


def configuration_page() -> None:
    loaded = registry()
    st.title("Configuration")
    st.success("All three registry files passed strict schema validation.")
    if st.button("Reload registry files"):
        registry.clear()
        st.rerun()
    st.subheader("Registry overview")
    st.dataframe(
        [
            {
                "file": path.name,
                "role": role,
            }
            for role, path in (
                ("Settings", loaded.sources.settings),
                ("Commands", loaded.sources.commands),
                ("Readers", loaded.sources.readers),
            )
        ],
        width="stretch",
        hide_index=True,
    )
    with st.expander("Technical registry paths", expanded=False):
        st.code(
            "\n".join(
                str(path.relative_to(REPOSITORY_ROOT))
                for path in (
                    loaded.sources.settings,
                    loaded.sources.commands,
                    loaded.sources.readers,
                )
            ),
            language="text",
        )
    with st.expander("Settings (raw)", expanded=False):
        st.json(loaded.settings.__dict__)
    with st.expander("Commands (raw)", expanded=False):
        st.json(
            {
                command.id: {
                    "title": command.title,
                    "category": command.category,
                    "argv_prefix": command.argv_prefix,
                    "actions": list(command.actions),
                }
                for command in loaded.commands.values()
            }
        )
    with st.expander("Readers (raw)", expanded=False):
        st.json(
            {
                reader.id: {
                    "title": reader.title,
                    "artifact_roots": reader.artifact_roots,
                    "discovery_glob": reader.discovery_glob,
                }
                for reader in loaded.readers.values()
            }
        )
    st.caption("Registry and settings files are read-only in the v1 UI.")


@st.cache_data(show_spinner="Scanning Dataset Registry…")
def _cached_registry_dataset_views(repository_root: str) -> list[dict[str, Any]]:
    """Cache strict Registry discovery for the Run page (serializable views)."""
    summaries = discover_registry_summaries(Path(repository_root))
    return [
        RegisteredDatasetView.from_summary(item).__dict__ for item in summaries
    ]


def _experiment_core_dataset_driven_controls(
    loaded: ControlPanelRegistry,
    *,
    command_id: str,
) -> tuple[str | None, dict[str, str], bool]:
    """Render Registry dataset selectors and prepare a local Experiment Core config.

    Returns ``(prepared_config_relative, pre_run_fields, blocks_launch)``.
    """
    prepared_key = "ecv2_prepared_bundle"
    st.subheader("Dataset Package")
    try:
        view_payloads = _cached_registry_dataset_views(str(REPOSITORY_ROOT))
        views = [RegisteredDatasetView(**item) for item in view_payloads]
    except DatasetExperimentMaterializerError as error:
        st.error(str(error))
        st.session_state.pop(prepared_key, None)
        return None, {}, True
    dataset_ids = [item.dataset_id for item in views]
    dataset_key = "ecv2-dataset-id"
    dataset_durable = ui_durable_key("run", "ecv2_dataset_id")
    sync_widget_with_durable(
        st.session_state,
        widget_key=dataset_key,
        durable_key=dataset_durable,
        allowed=dataset_ids,
        default=dataset_ids[0],
    )
    selected_dataset_id = st.selectbox(
        "Registered Dataset Package",
        dataset_ids,
        key=dataset_key,
        format_func=lambda value: _dataset_option_label(value, views),
    )
    remember_durable_value(
        st.session_state,
        durable_key=dataset_durable,
        value=selected_dataset_id,
        allowed=dataset_ids,
    )
    selected_view = next(
        item for item in views if item.dataset_id == selected_dataset_id
    )
    _render_dataset_package_details(selected_view)

    templates = list_base_templates(REPOSITORY_ROOT)
    if not templates:
        st.error("No Research v2 base templates were found under configs/research_v2.")
        st.session_state.pop(prepared_key, None)
        return None, {}, True

    model_families = sorted({item.model_family for item in templates})
    model_key = "ecv2-model-family"
    model_durable = ui_durable_key("run", "ecv2_model_family")
    sync_widget_with_durable(
        st.session_state,
        widget_key=model_key,
        durable_key=model_durable,
        allowed=model_families,
        default=model_families[0],
    )
    selected_model = st.selectbox("Model", model_families, key=model_key)
    remember_durable_value(
        st.session_state,
        durable_key=model_durable,
        value=selected_model,
        allowed=model_families,
    )

    mode_options = sorted(
        {
            item.mode
            for item in filter_templates(templates, model_family=selected_model)
        }
    )
    if not mode_options:
        st.error("No evaluation modes are available for the selected model.")
        st.session_state.pop(prepared_key, None)
        return None, {}, True
    mode_key = "ecv2-mode"
    mode_durable = ui_durable_key("run", "ecv2_mode")
    sync_widget_with_durable(
        st.session_state,
        widget_key=mode_key,
        durable_key=mode_durable,
        allowed=mode_options,
        default=mode_options[0],
    )
    selected_mode = st.selectbox("Evaluation mode", mode_options, key=mode_key)
    remember_durable_value(
        st.session_state,
        durable_key=mode_durable,
        value=selected_mode,
        allowed=mode_options,
    )

    filtered = filter_templates(
        templates, model_family=selected_model, mode=selected_mode
    )
    if not filtered:
        st.error("No base templates match the selected model and mode.")
        st.session_state.pop(prepared_key, None)
        return None, {}, True
    template_paths = [item.relative_path for item in filtered]
    template_key = "ecv2-base-template"
    template_durable = ui_durable_key("run", "ecv2_base_template")
    sync_widget_with_durable(
        st.session_state,
        widget_key=template_key,
        durable_key=template_durable,
        allowed=template_paths,
        default=template_paths[0],
    )
    selected_template = st.selectbox(
        "Base configuration template",
        template_paths,
        key=template_key,
        format_func=lambda path: readable_config_label(path, REPOSITORY_ROOT),
    )
    remember_durable_value(
        st.session_state,
        durable_key=template_durable,
        value=selected_template,
        allowed=template_paths,
    )
    selected_template_info = next(
        item for item in filtered if item.relative_path == selected_template
    )

    current_fingerprint = selection_fingerprint(
        dataset_id=selected_dataset_id,
        base_config_relative=selected_template,
        model_family=selected_model,
        mode=selected_mode,
    )
    prepared_raw = st.session_state.get(prepared_key)
    prepared = _coerce_prepared_bundle(prepared_raw)
    if prepared is not None and prepared.selection_fingerprint != current_fingerprint:
        st.session_state.pop(prepared_key, None)
        prepared = None
        st.warning(
            "Previously prepared configuration was invalidated because the dataset, "
            "model, mode, or template changed."
        )

    if st.button("Prepare run configuration", key="ecv2-prepare-config"):
        try:
            prepared = prepare_dataset_driven_experiment(
                REPOSITORY_ROOT,
                editable_root=loaded.settings.editable_config_root,
                dataset_id=selected_dataset_id,
                base_config_relative=selected_template,
            )
            st.session_state[prepared_key] = prepared
            if prepared.reused:
                st.success(
                    "Reused an existing identical prepared config/plan pair under "
                    f"`{prepared.config_relative}`."
                )
            else:
                st.success(
                    "Prepared Experiment Core configuration at "
                    f"`{prepared.config_relative}`."
                )
        except DatasetExperimentMaterializerError as error:
            st.session_state.pop(prepared_key, None)
            st.error(str(error))
            return None, {}, True

    prepared = _coerce_prepared_bundle(st.session_state.get(prepared_key))
    if prepared is None:
        return None, {}, True
    if prepared.selection_fingerprint != current_fingerprint:
        st.session_state.pop(prepared_key, None)
        return None, {}, True

    summary = build_dataset_driven_pre_run_summary(
        prepared,
        summary=selected_view,
    )
    st.caption(
        f"Adapter `{selected_template_info.adapter_id}` and evaluation protocol "
        "are copied from the selected base template. Feature pipeline is forced to "
        "`registered_prepared_passthrough_v1`."
    )
    del command_id
    return prepared.config_relative, summary, False


def _dataset_option_label(dataset_id: str, views: list[RegisteredDatasetView]) -> str:
    view = next(item for item in views if item.dataset_id == dataset_id)
    marker = " [exploratory]" if view.target_dependency == "exploratory" else ""
    return f"{dataset_id} ({view.n_features} features){marker}"


def _render_dataset_package_details(view: RegisteredDatasetView) -> None:
    columns = st.columns(4)
    columns[0].metric("Dataset ID", view.dataset_id)
    columns[1].metric("Parent", view.parent_dataset_id or "—")
    columns[2].metric("Features", view.n_features)
    columns[3].metric("Target dependency", view.target_dependency)
    if view.target_dependency == "exploratory":
        st.warning(
            "This Dataset Package has `target_dependency: exploratory`. It is not an "
            "unbiased screening dataset. Results must stay labeled exploratory "
            "(for example `v3_targeted_missingness`)."
        )
    st.write(f"**Hypothesis:** {view.hypothesis}")
    with st.expander("Dataset fingerprints", expanded=False):
        st.code(
            "\n".join(
                [
                    f"schema_hash: {view.schema_hash}",
                    f"train_content_hash: {view.train_content_hash}",
                    f"target_hash: {view.target_hash}",
                    f"train_row_identity_hash: {view.train_row_identity_hash}",
                ]
            ),
            language="text",
        )


def _coerce_prepared_bundle(value: Any) -> PreparedExperimentBundle | None:
    if isinstance(value, PreparedExperimentBundle):
        return value
    return None


def _placeholder_widget(
    loaded: ControlPanelRegistry,
    config_globs: tuple[str, ...],
    name: str,
    spec: PlaceholderSpec,
    widget_key: str,
    current_values: dict[str, Any],
    *,
    operation: str | None = None,
) -> Any:
    label = name.replace("_", " ").title()
    if spec.type == "enum":
        if operation is not None:
            seed_widget_from_logical(
                st.session_state,
                operation=operation,
                role=spec.role,
                name=name,
                widget_key=widget_key,
                allowed=list(spec.choices),
            )
        value = st.selectbox(
            label,
            spec.choices,
            key=widget_key,
            format_func=enum_human_label,
        )
        if operation is not None:
            remember_widget_selection(
                st.session_state,
                operation=operation,
                role=spec.role,
                name=name,
                value=value,
                allowed=list(spec.choices),
                widget_key=widget_key,
            )
        return value
    if spec.type == "integer":
        if operation is not None:
            seed_widget_from_logical(
                st.session_state,
                operation=operation,
                role=spec.role,
                name=name,
                widget_key=widget_key,
                allowed=None,
            )
        number_value = int(st.number_input(label, step=1, key=widget_key))
        if operation is not None:
            remember_widget_selection(
                st.session_state,
                operation=operation,
                role=spec.role,
                name=name,
                value=number_value,
                allowed=None,
                widget_key=widget_key,
            )
        return number_value
    if spec.role == "config":
        if operation == "mlflow_local_index":
            return _mlflow_config_widget(
                loaded,
                widget_key,
                operation=operation,
                name=name,
                role=spec.role,
            )
        return _cascade_config_widget(
            loaded,
            config_globs,
            widget_key,
            operation=operation,
            name=name,
            role=spec.role,
        )
    if spec.artifact_reader_id is not None:
        return _artifact_path_widget(
            loaded,
            name,
            spec,
            widget_key,
            operation=operation,
        )
    if (
        spec.type == "path"
        and spec.role == "input"
        and not spec.external_absolute
        and spec.roots
    ):
        reader = _reader_matching_roots(loaded, spec.roots)
        if reader is not None:
            return _artifact_path_widget(
                loaded,
                name,
                replace(
                    spec,
                    artifact_reader_id=reader.id,
                    artifact_statuses=(),
                    allow_manual_advanced=True,
                ),
                widget_key,
                operation=operation,
            )
        discovered = _discover_input_directories(spec.roots)
        if discovered:
            if operation is not None:
                meta = None
                logical = seed_widget_from_logical(
                    st.session_state,
                    operation=operation,
                    role=spec.role,
                    name=name,
                    widget_key=widget_key,
                    allowed=discovered,
                    cascade_meta=None,
                )
                if logical:
                    meta = parse_config_metadata(str(logical), REPOSITORY_ROOT)
                    seed_widget_from_logical(
                        st.session_state,
                        operation=operation,
                        role=spec.role,
                        name=name,
                        widget_key=widget_key,
                        allowed=discovered,
                        cascade_meta=meta,
                    )
            selected = _cascade_item_selector(
                discovered,
                widget_key=widget_key,
                item_label=label,
                include_source=True,
                advanced_label=f"Advanced: raw {label} path",
            )
            if operation is not None:
                remember_widget_selection(
                    st.session_state,
                    operation=operation,
                    role=spec.role,
                    name=name,
                    value=selected,
                    allowed=discovered,
                    widget_key=widget_key,
                )
            return selected
    if spec.suggested_value_template is not None:
        _apply_suggested_path(spec, widget_key, current_values)
    if spec.type == "path" and spec.external_absolute:
        path_help = (
            "Absolute path outside the repository. The value is redacted when marked "
            "sensitive and is never persisted by the control panel."
            if spec.sensitive
            else "Absolute path outside the repository."
        )
    elif spec.type == "path":
        path_help = f"Repository-relative path within: {', '.join(spec.roots)}"
    else:
        path_help = "Safe identifier"
    if operation is not None and not spec.sensitive:
        seed_widget_from_logical(
            st.session_state,
            operation=operation,
            role=spec.role,
            name=name,
            widget_key=widget_key,
            allowed=None,
        )
    value = st.text_input(
        label,
        help=path_help,
        key=widget_key,
    )
    if operation is not None and not spec.sensitive:
        remember_widget_selection(
            st.session_state,
            operation=operation,
            role=spec.role,
            name=name,
            value=value,
            allowed=None,
            widget_key=widget_key,
        )
    return value


def _artifact_path_widget(
    loaded: ControlPanelRegistry,
    name: str,
    spec: PlaceholderSpec,
    widget_key: str,
    *,
    operation: str | None = None,
) -> Any:
    label = name.replace("_", " ").title()
    reader = loaded.readers[spec.artifact_reader_id or ""]
    options = artifact_selector_options(
        REPOSITORY_ROOT,
        reader,
        roots=spec.roots,
        statuses=spec.artifact_statuses,
    )
    selected: str | None = None
    paths = [path for path, _label in options]
    if not options:
        st.info("No matching completed artifacts were found for this action.")
        st.session_state.pop(widget_key, None)
    else:
        if operation is not None:
            current = st.session_state.get(widget_key)
            meta = (
                parse_config_metadata(str(current), REPOSITORY_ROOT)
                if current not in (None, "")
                else None
            )
            seed_widget_from_logical(
                st.session_state,
                operation=operation,
                role=spec.role,
                name=name,
                widget_key=widget_key,
                allowed=paths,
                cascade_meta=meta,
            )
            seeded = st.session_state.get(widget_key)
            if seeded not in (None, "") and seeded in paths:
                seed_widget_from_logical(
                    st.session_state,
                    operation=operation,
                    role=spec.role,
                    name=name,
                    widget_key=widget_key,
                    allowed=paths,
                    cascade_meta=parse_config_metadata(str(seeded), REPOSITORY_ROOT),
                )
        selected = _cascade_item_selector(
            paths,
            widget_key=widget_key,
            item_label=label,
            include_source=False,
            advanced_label=f"Advanced: raw {label} path",
            show_advanced=True,
        )
        if operation is not None:
            remember_widget_selection(
                st.session_state,
                operation=operation,
                role=spec.role,
                name=name,
                value=selected,
                allowed=paths,
                widget_key=widget_key,
            )
    if spec.allow_manual_advanced:
        with st.expander("Advanced / manual path"):
            manual_toggle = f"{widget_key}__manual_toggle"
            manual_key = f"{widget_key}__manual_path"
            use_manual = st.checkbox(
                "Enter path manually",
                key=manual_toggle,
                help=(
                    "Manual paths still pass root containment, existence, and "
                    "symlink/junction checks."
                ),
            )
            if use_manual:
                selected = st.text_input(
                    f"Manual {label}",
                    help=f"Repository-relative path within: {', '.join(spec.roots)}",
                    key=manual_key,
                )
    return selected


def _reader_matching_roots(
    loaded: ControlPanelRegistry, roots: tuple[str, ...]
) -> Any | None:
    root_set = set(roots)
    for reader in loaded.readers.values():
        if root_set.intersection(reader.artifact_roots):
            return reader
    return None


def _discover_input_directories(roots: tuple[str, ...]) -> list[str]:
    discovered: list[str] = []
    for root in roots:
        base = REPOSITORY_ROOT / root
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            if child.is_dir() and not child.is_symlink():
                discovered.append(child.relative_to(REPOSITORY_ROOT).as_posix())
    return discovered


def _apply_suggested_path(
    spec: PlaceholderSpec,
    widget_key: str,
    current_values: Mapping[str, Any],
) -> None:
    template = spec.suggested_value_template
    if template is None:
        return
    source_key = f"{widget_key}__suggestion_source"
    try:
        suggestion = render_suggested_value_template(template, current_values)
    except SuggestionError:
        return
    fingerprint = tuple(
        sorted((key, str(current_values.get(key, ""))) for key in current_values)
    )
    previous = st.session_state.get(source_key)
    current = st.session_state.get(widget_key)
    if previous != fingerprint or current in (None, ""):
        st.session_state[widget_key] = suggestion
        st.session_state[source_key] = fingerprint


def _action_requires_additional_input(action: ActionSpec) -> bool:
    required = [spec for spec in action.placeholders.values() if spec.required]
    if len(required) > 1:
        return True
    return any(spec.role in {"output", "config", "value"} for spec in required)


def _config_options(
    loaded: ControlPanelRegistry, globs: tuple[str, ...]
) -> list[tuple[str, str]]:
    del loaded
    paths: set[str] = set()
    for pattern in globs:
        for path in REPOSITORY_ROOT.glob(pattern):
            if path.is_file() and not path.is_symlink():
                paths.add(path.relative_to(REPOSITORY_ROOT).as_posix())
    return [(p, readable_config_label(p, REPOSITORY_ROOT)) for p in sorted(paths)]


def _mlflow_config_widget(
    loaded: ControlPanelRegistry,
    widget_key: str,
    *,
    operation: str,
    name: str = "config",
    role: str = "config",
) -> str | None:
    """MLflow Local Index uses only configs/mlflow/local.yaml — no experiment cascade."""
    del loaded
    canonical = DEFAULT_MLFLOW_CONFIG
    allowed = [canonical] if (REPOSITORY_ROOT / canonical).is_file() else []
    if not allowed:
        st.error("MLflow local configuration is missing: configs/mlflow/local.yaml")
        st.session_state.pop(widget_key, None)
        return None
    seed_widget_from_logical(
        st.session_state,
        operation=operation,
        role=role,
        name=name,
        widget_key=widget_key,
        allowed=allowed,
        cascade_meta=None,
    )
    st.session_state[widget_key] = canonical
    # Clear any inherited experiment cascade parent keys from other operations.
    for suffix in ("__src", "__mdl", "__mode", "__flat", "__parent_fp"):
        st.session_state.pop(f"{widget_key}{suffix}", None)
    selected = st.selectbox(
        "Config",
        allowed,
        key=widget_key,
        format_func=lambda path: Path(path).name,
    )
    st.caption(f"Basename: `{Path(selected).name}`")
    st.caption(f"`{selected}`")
    remember_widget_selection(
        st.session_state,
        operation=operation,
        role=role,
        name=name,
        value=selected,
        allowed=allowed,
        widget_key=widget_key,
    )
    return selected


def _cascade_config_widget(
    loaded: ControlPanelRegistry,
    config_globs: tuple[str, ...],
    widget_key: str,
    *,
    operation: str | None = None,
    name: str = "config",
    role: str = "config",
) -> str | None:
    """Four-level cascading config selector: Source → Model → Mode → Config."""
    option_pairs = _config_options(loaded, config_globs)
    if not option_pairs:
        st.warning("No allowed configuration files were found.")
        st.session_state.pop(widget_key, None)
        return None
    raw_paths = [path for path, _ in option_pairs]
    if operation is not None:
        current = st.session_state.get(widget_key)
        meta = (
            parse_config_metadata(str(current), REPOSITORY_ROOT)
            if current not in (None, "")
            else None
        )
        seed_widget_from_logical(
            st.session_state,
            operation=operation,
            role=role,
            name=name,
            widget_key=widget_key,
            allowed=raw_paths,
            cascade_meta=meta,
        )
        seeded = st.session_state.get(widget_key)
        if seeded not in (None, "") and seeded in raw_paths:
            seed_widget_from_logical(
                st.session_state,
                operation=operation,
                role=role,
                name=name,
                widget_key=widget_key,
                allowed=raw_paths,
                cascade_meta=parse_config_metadata(str(seeded), REPOSITORY_ROOT),
            )
    selected = _cascade_item_selector(
        raw_paths,
        widget_key=widget_key,
        item_label="Config",
        include_source=True,
        advanced_label="Advanced: raw config path selector",
        show_advanced=True,
    )
    if operation is not None:
        remember_widget_selection(
            st.session_state,
            operation=operation,
            role=role,
            name=name,
            value=selected,
            allowed=raw_paths,
            widget_key=widget_key,
        )
    return selected


def _cascade_item_selector(
    paths: list[str],
    *,
    widget_key: str,
    item_label: str,
    include_source: bool = True,
    model_label: str = "Model",
    mode_label: str = "Mode",
    advanced_label: str = "Advanced: raw path selector",
    show_advanced: bool = True,
    default_index: int = 0,
) -> str | None:
    """Cascading Source/Model/Mode/Item selector for configs or artifacts."""
    if not paths:
        return None
    cascade_opts = build_cascade_options(paths, REPOSITORY_ROOT)
    path_set = set(paths)

    selected_source: str | None = None
    if include_source:
        sources = cascade_available_sources(cascade_opts)
        src_key = f"{widget_key}__src"
        if st.session_state.get(src_key) not in sources:
            st.session_state[src_key] = sources[0]
        if len(sources) > 1:
            selected_source = st.selectbox(
                "Source",
                sources,
                key=src_key,
                format_func=source_human_label,
            )
        else:
            selected_source = sources[0]
            st.caption(f"Source: **{source_human_label(selected_source)}**")

    models = cascade_available_models(cascade_opts, selected_source)
    mdl_key = f"{widget_key}__mdl"
    selected_model: str | None = None
    if models:
        if st.session_state.get(mdl_key) not in models:
            st.session_state[mdl_key] = models[0]
        if len(models) > 1:
            selected_model = st.selectbox(
                model_label,
                models,
                key=mdl_key,
                format_func=model_human_label,
            )
        else:
            selected_model = models[0]
            st.caption(f"{model_label}: **{model_human_label(selected_model)}**")

    modes = cascade_available_modes(cascade_opts, selected_source, selected_model)
    mode_key = f"{widget_key}__mode"
    selected_mode: str | None = None
    if modes:
        if st.session_state.get(mode_key) not in modes:
            st.session_state[mode_key] = modes[0]
        if len(modes) > 1:
            selected_mode = st.selectbox(
                mode_label,
                modes,
                key=mode_key,
                format_func=mode_human_label,
            )
        else:
            selected_mode = modes[0]
            st.caption(f"{mode_label}: **{mode_human_label(selected_mode)}**")

    matching = cascade_filter_configs(
        cascade_opts,
        selected_source,
        selected_model or None,
        selected_mode or None,
    )
    item_paths = [opt.path for opt in matching if opt.path in path_set]
    parent_fp = (selected_source, selected_model, selected_mode)
    reconciliation = reconcile_cascade_selection(
        allowed_paths=paths,
        matched_paths=item_paths,
        current_path=st.session_state.get(widget_key),
        default_index=default_index,
    )
    canonical = apply_cascade_reconciliation(
        st.session_state,
        widget_key=widget_key,
        reconciliation=reconciliation,
        parent_fingerprint=parent_fp,
    )
    if canonical is None:
        st.error(
            reconciliation.error
            or "Cascade selection could not be reconciled; refusing to use a stale path."
        )
        return None

    labels = {
        opt.path: opt.display_label for opt in cascade_opts if opt.path in item_paths
    }
    for path in item_paths:
        labels.setdefault(path, Path(path).name)
        # Never present raw run IDs as the primary selector label.
        if is_raw_run_id(Path(path).name) and is_raw_run_id(
            str(labels.get(path, "")).split(" · ")[0]
        ):
            labels[path] = readable_path_label(path, REPOSITORY_ROOT)

    selected_value = st.selectbox(
        item_label,
        list(reconciliation.matched_paths),
        key=widget_key,
        format_func=lambda v: labels.get(v, v),
    )
    # Fail closed: diagnostic selector never overrides the visible cascade.
    if selected_value not in reconciliation.matched_paths:
        final_value = canonical
    else:
        final_value = str(selected_value)

    if final_value:
        st.caption(f"Basename: `{Path(final_value).name}`")
        st.caption(f"`{final_value}`")

    if show_advanced:
        with st.expander(advanced_label):
            all_labels = {
                opt.path: readable_path_label(opt.path, REPOSITORY_ROOT)
                for opt in cascade_opts
            }
            flat_key = f"{widget_key}__flat"
            # Keep diagnostic selector aligned with the visible cascade; never
            # feed __flat back into the returned / submitted path.
            st.session_state[flat_key] = final_value
            st.selectbox(
                f"Raw {item_label.lower()} path (diagnostic)",
                paths,
                key=flat_key,
                format_func=lambda v: all_labels.get(v, v),
            )
            st.caption(
                "Diagnostic only. Submitted path follows the cascade selectors above."
            )
    return final_value


def _config_panel(loaded: ControlPanelRegistry, selected: Path) -> None:
    try:
        text, canonical = read_config(
            REPOSITORY_ROOT,
            selected.relative_to(REPOSITORY_ROOT),
            allowed_roots=(
                "configs",
                loaded.settings.editable_config_root,
                "artifacts/optuna_exports",
            ),
        )
    except ConfigEditError as error:
        st.error(str(error))
        return
    with st.expander("Config preview", expanded=False):
        st.code(text, language="yaml" if canonical.suffix != ".json" else "json")
        try:
            parse_config_text(text, canonical.suffix)
            st.success("YAML/JSON syntax is valid.")
        except ConfigEditError as error:
            st.error(str(error))
    if not loaded.settings.allow_config_copy_editing:
        return
    with st.expander("Advanced configuration editor"):
        rel_path = selected.relative_to(REPOSITORY_ROOT).as_posix()
        draft_key = f"config-editor-draft-{rel_path}"
        copy_key = f"config-editor-copy-name-{rel_path}"
        draft_durable = ui_durable_key("config_editor", "draft", rel_path)
        copy_durable = ui_durable_key("config_editor", "copy_name", rel_path)
        if st.session_state.get(draft_key) in (None, ""):
            durable_draft = get_durable_value(st.session_state, draft_durable)
            st.session_state[draft_key] = (
                durable_draft if durable_draft not in (None, "") else text
            )
        default_copy_name = f"{canonical.stem}_copy{canonical.suffix}"
        if st.session_state.get(copy_key) in (None, ""):
            durable_copy = get_durable_value(st.session_state, copy_durable)
            st.session_state[copy_key] = (
                durable_copy
                if durable_copy not in (None, "")
                else default_copy_name
            )
        edited = st.text_area("Configuration copy", height=360, key=draft_key)
        copy_name = st.text_input("New copy filename", key=copy_key)
        remember_durable_value(
            st.session_state,
            durable_key=draft_durable,
            value=edited,
            allowed=None,
        )
        remember_durable_value(
            st.session_state,
            durable_key=copy_durable,
            value=copy_name,
            allowed=None,
        )
        if st.button("Save as new copy"):
            try:
                target = save_config_copy(
                    REPOSITORY_ROOT,
                    editable_root=loaded.settings.editable_config_root,
                    copy_name=copy_name,
                    text=edited,
                )
                st.success(
                    f"Saved {target.relative_to(REPOSITORY_ROOT).as_posix()}. "
                    "The canonical source was not modified."
                )
            except ConfigEditError as error:
                st.error(str(error))


def _all_artifacts(loaded: ControlPanelRegistry) -> list[ArtifactRecord]:
    records: list[ArtifactRecord] = []
    for reader in loaded.readers.values():
        records.extend(discover_artifacts(REPOSITORY_ROOT, reader))
    return records


def _job_label(record: Any, commands: Mapping[str, Any] | None = None) -> str:
    return job_primary_label(
        record.job,
        record_commands=dict(commands) if commands else None,
        repository_root=REPOSITORY_ROOT,
    )


def _render_dataset_identity_summary(run_root: Path) -> None:
    identity = read_dataset_identity_safe(run_root)
    if identity.diagnostic:
        st.error(identity.diagnostic)
        return
    st.subheader("Dataset identity")
    columns = st.columns(4)
    columns[0].metric("Dataset", identity.display("dataset_id"))
    parent = identity.display("parent_dataset_id")
    columns[1].metric("Parent", "—" if parent in {"None", "Not available"} else parent)
    columns[2].metric("Target dependency", identity.display("target_dependency"))
    columns[3].metric("Features", identity.display("n_features"))
    if identity.is_exploratory:
        st.warning(
            "This run used an exploratory Dataset Package "
            "(`target_dependency: exploratory`). Keep results labeled exploratory."
        )
    with st.expander("Dataset fingerprints", expanded=False):
        st.code(
            "\n".join(
                [
                    f"schema_hash: {identity.display('schema_hash')}",
                    f"train_content_hash: {identity.display('train_content_hash')}",
                    f"target_hash: {identity.display('target_hash')}",
                    f"train_row_identity_hash: "
                    f"{identity.display('train_row_identity_hash')}",
                ]
            ),
            language="text",
        )


def main() -> None:
    try:
        registry()
    except SchemaError as error:
        st.error(f"Control-panel registry is invalid: {error}")
        st.stop()
    pages = [
        st.Page(dashboard_page, title="Dashboard", icon=":material/dashboard:"),
        st.Page(run_page, title="Run", icon=":material/play_arrow:"),
        st.Page(jobs_page, title="Jobs", icon=":material/work_history:"),
        st.Page(results_page, title="Results", icon=":material/analytics:"),
        st.Page(configuration_page, title="Configuration", icon=":material/settings:"),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
