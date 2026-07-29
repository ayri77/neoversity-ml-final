from __future__ import annotations

import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import streamlit as st


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.churn_ml.control_panel.artifacts import (  # noqa: E402
    ArtifactRecord,
    ArtifactReadError,
    artifact_selector_options,
    configured_artifact_file,
    comparison_rows,
    csv_preview,
    discover_artifacts,
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
                        item.job, record_commands=loaded.commands
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
    prefill = st.session_state.get("run_prefill", {})
    command_ids = list(loaded.commands)
    preferred_command = prefill.get("command_id")
    command_id = st.selectbox(
        "Operation",
        command_ids,
        index=(
            command_ids.index(preferred_command)
            if preferred_command in command_ids
            else 0
        ),
        key="run-command",
        format_func=lambda value: loaded.commands[value].title,
    )
    command = loaded.commands[command_id]
    action_ids = list(command.actions)
    preferred_action = prefill.get("action_id")
    action_id = st.selectbox(
        "Action",
        action_ids,
        index=(
            action_ids.index(preferred_action) if preferred_action in action_ids else 0
        ),
        key=f"run-action-{command_id}",
        format_func=lambda value: command.actions[value].title,
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
    prefill_values = (
        prefill.get("values", {})
        if prefill.get("command_id") == command_id
        and prefill.get("action_id") == action_id
        else {}
    )
    for name, value in prefill_values.items():
        st.session_state.setdefault(f"value-{command_id}-{action_id}-{name}", value)

    for name, placeholder in action.placeholders.items():
        widget_key = f"value-{command_id}-{action_id}-{name}"
        value = _placeholder_widget(
            loaded,
            command.allowed_config_globs,
            name,
            placeholder,
            widget_key,
            values,
        )
        if value not in (None, ""):
            values[name] = value
            if placeholder.role == "config":
                selected_config = REPOSITORY_ROOT / str(value)

    if selected_config is not None and selected_config.is_file():
        _config_panel(loaded, selected_config)

    # Save last valid selections for UX continuity.
    # Per-command per-action placeholder keys already persist via widget_key scheme.
    st.session_state["_last_command_id"] = command_id
    st.session_state["_last_action_id"] = action_id

    built = None
    pre_run: dict[str, str] = {}
    try:
        if action.enabled:
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
            if pre_run:
                st.markdown("  \n".join(f"**{k}:** {v}" for k, v in pre_run.items()))
            with st.expander("Technical command", expanded=False):
                st.code(display_argv(built.redacted_argv), language="python")
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
            record = job_manager(loaded).start(
                argv=authorized.argv,
                redacted_argv=authorized.redacted_argv,
                command_id=command_id,
                action_id=action_id,
                references=authorized.references,
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
    selected = st.selectbox(
        "Job",
        [item.job_id for item in jobs],
        format_func=lambda value: _job_label(
            next(item for item in jobs if item.job_id == value),
            commands=loaded.commands,
        ),
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
    if status.get("diagnostic"):
        st.warning(str(status["diagnostic"]))
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


def results_page() -> None:
    loaded = registry()
    st.title("Results")
    reader_id = st.selectbox(
        "Artifact type",
        list(loaded.readers),
        format_func=lambda value: loaded.readers[value].title,
    )
    reader = loaded.readers[reader_id]
    artifacts = discover_artifacts(REPOSITORY_ROOT, reader)
    if not artifacts:
        st.info("No artifacts match this configured reader.")
        return
    artifact_paths = [item.relative_path for item in artifacts]

    selected_path = _cascade_item_selector(
        artifact_paths,
        widget_key=f"results-{reader_id}-artifact",
        item_label="Artifact",
        include_source=False,
        advanced_label="Advanced: raw artifact path",
    )
    if not selected_path:
        return
    selected = next(item for item in artifacts if item.relative_path == selected_path)
    st.write(f"Status: `{selected.state}`")
    if selected.diagnostic:
        st.warning(selected.diagnostic)
    if selected.summaries:
        st.dataframe(
            [
                {"field": key, "value": value}
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

    st.subheader("Side-by-side comparison")
    left_item = selected
    right_item = selected
    if len(artifacts) < 2:
        st.info("At least two artifacts are required.")
    else:
        left_col, right_col = st.columns(2)
        with left_col:
            st.markdown("**Left**")
            left_path = _cascade_item_selector(
                artifact_paths,
                widget_key=f"compare-left-{reader_id}",
                item_label="Left artifact",
                include_source=False,
                model_label="Left model",
                mode_label="Left mode",
                show_advanced=False,
                default_index=0,
            )
        with right_col:
            st.markdown("**Right**")
            right_path = _cascade_item_selector(
                artifact_paths,
                widget_key=f"compare-right-{reader_id}",
                item_label="Right artifact",
                include_source=False,
                model_label="Right model",
                mode_label="Right mode",
                show_advanced=False,
                default_index=1,
            )
        if left_path and right_path:
            left_item = next(
                item for item in artifacts if item.relative_path == left_path
            )
            right_item = next(
                item for item in artifacts if item.relative_path == right_path
            )
            st.dataframe(
                comparison_rows(left_item, right_item, reader.compare_fields),
                width="stretch",
                hide_index=True,
            )
        if reader_id == "research_v2":
            st.caption(
                "Use the Paired Comparison action on the Run page for the official "
                "compatibility gate and comparison artifact. This table is display-only."
            )

    if reader_id == "research_v2" and len(artifacts) >= 2:
        comparison_id = st.text_input(
            "New comparison ID",
            value="ui-paired-comparison",
        )
        comparison_ready = (
            left_item.state == "completed" and right_item.state == "completed"
        )
        if (
            st.button("Prepare Paired Comparison action", disabled=not comparison_ready)
            and comparison_ready
        ):
            st.session_state["run-command"] = "paired_comparison"
            st.session_state["run-action-paired_comparison"] = "run"
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
    if reader_id == "deployment_v1":
        deployment_valid = selected.state != "invalid"
        if (
            st.button(
                "Prepare deployment Inspect action", disabled=not deployment_valid
            )
            and deployment_valid
        ):
            st.session_state["run-command"] = "final_deployment_v1"
            st.session_state["run-action-final_deployment_v1"] = "inspect"
            st.session_state["run_prefill"] = {
                "command_id": "final_deployment_v1",
                "action_id": "inspect",
                "values": {"deployment_dir": selected.relative_path},
            }
            st.success(
                "Prepared the allowlisted action. Open Run to review argv and start."
            )

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


def _placeholder_widget(
    loaded: ControlPanelRegistry,
    config_globs: tuple[str, ...],
    name: str,
    spec: PlaceholderSpec,
    widget_key: str,
    current_values: dict[str, Any],
) -> Any:
    label = name.replace("_", " ").title()
    if spec.type == "enum":
        return st.selectbox(
            label,
            spec.choices,
            key=widget_key,
            format_func=enum_human_label,
        )
    if spec.type == "integer":
        return int(st.number_input(label, step=1, key=widget_key))
    if spec.role == "config":
        return _cascade_config_widget(loaded, config_globs, widget_key)
    if spec.artifact_reader_id is not None:
        return _artifact_path_widget(loaded, name, spec, widget_key)
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
            )
        discovered = _discover_input_directories(spec.roots)
        if discovered:
            return _cascade_item_selector(
                discovered,
                widget_key=widget_key,
                item_label=label,
                include_source=True,
                advanced_label=f"Advanced: raw {label} path",
            )
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
    return st.text_input(
        label,
        help=path_help,
        key=widget_key,
    )


def _artifact_path_widget(
    loaded: ControlPanelRegistry,
    name: str,
    spec: PlaceholderSpec,
    widget_key: str,
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
    if not options:
        st.info("No matching completed artifacts were found for this action.")
        st.session_state.pop(widget_key, None)
    else:
        paths = [path for path, _label in options]
        selected = _cascade_item_selector(
            paths,
            widget_key=widget_key,
            item_label=label,
            include_source=False,
            advanced_label=f"Advanced: raw {label} path",
            show_advanced=True,
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


def _cascade_config_widget(
    loaded: ControlPanelRegistry,
    config_globs: tuple[str, ...],
    widget_key: str,
) -> str | None:
    """Four-level cascading config selector: Source → Model → Mode → Config."""
    option_pairs = _config_options(loaded, config_globs)
    if not option_pairs:
        st.warning("No allowed configuration files were found.")
        st.session_state.pop(widget_key, None)
        return None
    raw_paths = [path for path, _ in option_pairs]
    return _cascade_item_selector(
        raw_paths,
        widget_key=widget_key,
        item_label="Config",
        include_source=True,
        advanced_label="Advanced: raw config path selector",
        show_advanced=True,
    )


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
    # When model/mode metadata is sparse, still restrict to selected source.
    if not matching and selected_source is not None:
        matching = [o for o in cascade_opts if o.source_kind == selected_source]
    if not matching:
        matching = cascade_opts

    item_paths = [opt.path for opt in matching if opt.path in path_set]
    if not item_paths:
        item_paths = [p for p in paths]
    labels = {
        opt.path: opt.display_label for opt in cascade_opts if opt.path in item_paths
    }
    for path in item_paths:
        labels.setdefault(path, Path(path).name)

    if st.session_state.get(widget_key) not in item_paths:
        index = min(max(default_index, 0), len(item_paths) - 1)
        st.session_state[widget_key] = item_paths[index]
    selected_value = st.selectbox(
        item_label,
        item_paths,
        key=widget_key,
        format_func=lambda v: labels.get(v, v),
    )
    if selected_value:
        st.caption(f"`{Path(selected_value).name}`")
        st.caption(f"`{selected_value}`")

    if show_advanced:
        with st.expander(advanced_label):
            all_labels = {
                opt.path: readable_path_label(opt.path, REPOSITORY_ROOT)
                for opt in cascade_opts
            }
            flat_key = f"{widget_key}__flat"
            if st.session_state.get(flat_key) not in paths:
                st.session_state[flat_key] = (
                    selected_value if selected_value in paths else paths[0]
                )
            st.selectbox(
                f"Raw {item_label.lower()} path (diagnostic)",
                paths,
                key=flat_key,
                format_func=lambda v: all_labels.get(v, v),
            )
    return selected_value


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
        edited = st.text_area("Configuration copy", value=text, height=360)
        copy_name = st.text_input(
            "New copy filename",
            value=f"{canonical.stem}_copy{canonical.suffix}",
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
        record.job, record_commands=dict(commands) if commands else None
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
