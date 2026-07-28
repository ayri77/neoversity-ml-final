from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

import streamlit as st


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.churn_ml.control_panel.artifacts import (  # noqa: E402
    ArtifactRecord,
    ArtifactReadError,
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
from src.churn_ml.control_panel.jobs import JobError, JobManager  # noqa: E402
from src.churn_ml.control_panel.registry import (  # noqa: E402
    ControlPanelRegistry,
    load_registry,
)
from src.churn_ml.control_panel.schemas import PlaceholderSpec, SchemaError  # noqa: E402


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
                    "job": item.job_id,
                    "status": item.status["state"],
                    "command": item.job["command_id"],
                    "action": item.job["action_id"],
                    "created": item.job["created_at_utc"],
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
                    "reader": item.reader_id,
                    "artifact": item.relative_path,
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
            loaded, command.allowed_config_globs, name, placeholder, widget_key
        )
        if value not in (None, ""):
            values[name] = value
            if placeholder.role == "config":
                selected_config = REPOSITORY_ROOT / str(value)

    if selected_config is not None and selected_config.is_file():
        _config_panel(loaded, selected_config)

    built = None
    try:
        if action.enabled:
            built = build_command(
                loaded.commands,
                command_id,
                action_id,
                values,
                repository_root=REPOSITORY_ROOT,
            )
            st.code(display_argv(built.redacted_argv), language="python")
        else:
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
            next(item for item in jobs if item.job_id == value)
        ),
    )
    record = manager.refresh(selected)
    status = record.status
    columns = st.columns(4)
    columns[0].metric("Status", str(status["state"]))
    columns[1].metric("PID", status.get("pid") or "—")
    columns[2].metric("Elapsed", human_duration(status.get("elapsed_seconds")))
    columns[3].metric(
        "Exit code",
        status.get("exit_code") if status.get("exit_code") is not None else "—",
    )
    st.json(dict(record.job), expanded=False)
    if status.get("diagnostic"):
        st.warning(str(status["diagnostic"]))
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
    selected_path = st.selectbox(
        "Artifact",
        [item.relative_path for item in artifacts],
    )
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
        left, right = st.columns(2)
        left_path = left.selectbox(
            "Left",
            [item.relative_path for item in artifacts],
            key="compare-left",
        )
        right_path = right.selectbox(
            "Right",
            [item.relative_path for item in artifacts],
            index=1,
            key="compare-right",
        )
        left_item = next(item for item in artifacts if item.relative_path == left_path)
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
            if action.id in {"inspect", "run"}:
                action_rows.append(
                    {
                        "command": command.title,
                        "action": action.title,
                        "enabled": action.enabled,
                        "where": "Run page",
                    }
                )
    if action_rows:
        st.dataframe(action_rows, width="stretch", hide_index=True)
        st.info(
            "Launch these allowlisted actions from the Run page after reviewing argv."
        )
    else:
        st.info("No inspect, export, or paired-comparison action is registered.")


def configuration_page() -> None:
    loaded = registry()
    st.title("Configuration")
    st.success("All three registry files passed strict schema validation.")
    if st.button("Reload registry files"):
        registry.clear()
        st.rerun()
    st.subheader("Sources")
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
    st.subheader("Settings")
    st.json(loaded.settings.__dict__)
    st.subheader("Commands")
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
    st.subheader("Readers")
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
) -> Any:
    label = name.replace("_", " ").title()
    if spec.type == "enum":
        return st.selectbox(label, spec.choices, key=widget_key)
    if spec.type == "integer":
        return int(st.number_input(label, step=1, key=widget_key))
    if spec.role == "config":
        options = _config_options(loaded, config_globs)
        if not options:
            st.warning("No allowed configuration files were found.")
            return None
        return st.selectbox(label, options, key=widget_key)
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


def _config_options(loaded: ControlPanelRegistry, globs: tuple[str, ...]) -> list[str]:
    del loaded
    paths: set[str] = set()
    for pattern in globs:
        for path in REPOSITORY_ROOT.glob(pattern):
            if path.is_file() and not path.is_symlink():
                paths.add(path.relative_to(REPOSITORY_ROOT).as_posix())
    return sorted(paths)


def _config_panel(loaded: ControlPanelRegistry, selected: Path) -> None:
    try:
        text, canonical = read_config(
            REPOSITORY_ROOT,
            selected.relative_to(REPOSITORY_ROOT),
            allowed_roots=("configs", loaded.settings.editable_config_root),
        )
    except ConfigEditError as error:
        st.error(str(error))
        return
    st.subheader("Configuration")
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


def _job_label(record: Any) -> str:
    return (
        f"{record.status['state']} · {record.job['command_id']}/"
        f"{record.job['action_id']} · {record.job_id[:8]}"
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
