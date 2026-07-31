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

REGISTRY_DISCOVERY_CACHE_VERSION = "dataset_registry_discovery_v1"
DEFAULT_PROCESSED_ROOT = "data/processed"
RESEARCH_WORKSPACE_CACHE_VERSION = "research_workspace_inventory_v1"

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
from src.churn_ml.control_panel.archive_registry import (  # noqa: E402
    ArchiveError,
    ArchiveRegistry,
    preview_job_deletion,
    utc_now_text as archive_utc_now_text,
)
from src.churn_ml.control_panel.research_annotations import (  # noqa: E402
    BUILTIN_TAGS,
    ResearchAnnotationError,
    ResearchAnnotationRegistry,
)
from src.churn_ml.control_panel.research_export import (  # noqa: E402
    export_annotated_runs_csv,
)
from src.churn_ml.control_panel.research_inventory import (  # noqa: E402
    UNAVAILABLE,
    build_research_inventory,
    inventory_rows_as_mappings,
    inventory_rows_from_mappings,
)
from src.churn_ml.control_panel.research_matrix import (  # noqa: E402
    DEFAULT_BASELINE_DATASET_ID,
    MATRIX_METRICS,
    MatrixFilters,
    annotate_runs,
    build_research_matrix,
    descriptive_comparison_table,
    matrix_display_rows,
)
from src.churn_ml.control_panel.workflow_navigation import (  # noqa: E402
    contract_declaration,
    is_legacy_command,
    visible_command_ids,
    workflow_description,
    workflow_label,
)
from src.churn_ml.control_panel.config_schema_guard import (  # noqa: E402
    ConfigGuardResult,
    guard_config_for_command,
)
from src.churn_ml.control_panel.deployment_candidates import (  # noqa: E402
    CANDIDATE_WIDGET_KEY,
    SUBMISSION_HANDOFF_KEY,
    apply_submission_handoff,
    candidate_durable_key,
    evaluate_deployment_readiness,
    list_deployment_candidates,
)
from src.churn_ml.control_panel.deployment_draft_builder import (  # noqa: E402
    DEFAULT_EXCEPTION_REASON,
    DEFAULT_INTENDED_ROLE,
    DEPLOYMENT_CONFIG_NAME,
    DEPLOYMENT_DRAFT_ROOT,
    DeploymentDraftError,
    prepare_deployment_draft,
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
from src.churn_ml.control_panel.paired_comparison_readiness import (  # noqa: E402
    evaluate_official_paired_readiness,
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
    job_primary_label as _imported_job_primary_label,
    mode_badge,
    mode_human_label,
    model_human_label,
    parse_config_metadata,
    readable_config_label,
    readable_path_label,
    set_presentation_repository_root,
    source_human_label,
)
from src.churn_ml.control_panel.jobs import (  # noqa: E402
    TERMINAL_STATES,
    JobError,
    JobManager,
)
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


# Sole repository-root channel for presentation helpers (not a second store).
set_presentation_repository_root(REPOSITORY_ROOT)

# Live crash class (Dashboard Recent jobs + Jobs selectbox format_func):
# commit 0707fd3 callers pass repository_root=, while a pre-0707fd3
# presentation.job_primary_label rejects that keyword. Bind an app-local
# adapter that accepts the keyword, updates the presentation root override,
# and always calls the imported helper with two positional arguments only.
_IMPORTED_JOB_PRIMARY_LABEL = _imported_job_primary_label
_IMPORTED_JOB_PRIMARY_LABEL_FILE = getattr(
    getattr(_IMPORTED_JOB_PRIMARY_LABEL, "__code__", None),
    "co_filename",
    getattr(_IMPORTED_JOB_PRIMARY_LABEL, "__module__", "?"),
)


def job_primary_label(
    record_job: Mapping[str, Any],
    record_commands: dict | None = None,
    *,
    repository_root: Path | None = None,
) -> str:
    """Label adapter used by Dashboard and Jobs ``_job_label``."""
    if repository_root is not None:
        set_presentation_repository_root(repository_root)
    return _IMPORTED_JOB_PRIMARY_LABEL(record_job, record_commands)


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
                        loaded.commands,
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
    registered_command_ids = list(loaded.commands)
    command_widget_key = "run-command"
    command_durable_key = ui_durable_key("run", "command")
    advanced_widget_key = "run-show-advanced-operations"
    advanced_durable_key = ui_durable_key("run", "show_advanced_operations")
    if advanced_widget_key not in st.session_state:
        st.session_state[advanced_widget_key] = bool(
            get_durable_value(st.session_state, advanced_durable_key)
        )
    # A legacy operation may already be selected (durable state or a historical
    # session). Reveal the legacy section instead of silently rewriting that
    # selection.
    already_selected = (
        prefill.get("command_id")
        or st.session_state.get(command_widget_key)
        or get_durable_value(st.session_state, command_durable_key)
    )
    if already_selected in registered_command_ids and already_selected not in set(
        visible_command_ids(registered_command_ids)
    ):
        st.session_state[advanced_widget_key] = True
    command_ids = visible_command_ids(
        registered_command_ids,
        include_legacy=bool(st.session_state[advanced_widget_key]),
    )
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
        # The label stays "Operation" so existing UI state and UI tests keep
        # working; the options themselves are now the ordered workflow steps.
        "Operation",
        command_ids,
        key=command_widget_key,
        format_func=lambda value: workflow_label(
            value, fallback=loaded.commands[value].title
        ),
    )
    remember_durable_value(
        st.session_state,
        durable_key=command_durable_key,
        value=command_id,
        allowed=command_ids,
    )
    command = loaded.commands[command_id]
    st.caption(
        workflow_description(command_id, fallback=command.description)
    )
    if is_legacy_command(command_id):
        st.warning(
            "Legacy operation. It stays available only so historical runs remain "
            "reproducible. Use Train for new training."
        )
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
    risk = (
        "Competition-test / deployment"
        if action.competition_test
        else action.confirmation
    )
    st.caption(f"Safety level: {risk}")
    if not action.enabled:
        st.warning("This action is disabled by the declarative registry.")
    _render_advanced_operations_toggle(
        widget_key=advanced_widget_key,
        durable_key=advanced_durable_key,
    )
    _render_workflow_technical_details(loaded, command_id, action_id)

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

    deployment_summary: dict[str, str] = {}
    use_candidate_driven = False
    if command_id == "final_deployment_v1" and "config" in action.placeholders:
        (
            prepared_deployment_config,
            deployment_summary,
            deployment_blocks_launch,
        ) = _deployment_candidate_controls(loaded, action_id=action_id)
        use_candidate_driven = True
        dataset_driven_blocks_launch = (
            dataset_driven_blocks_launch or deployment_blocks_launch
        )
        if prepared_deployment_config:
            values["config"] = prepared_deployment_config
            selected_config = REPOSITORY_ROOT / prepared_deployment_config
            config_widget = widget_selection_key(command_id, "config", "config")
            st.session_state[config_widget] = prepared_deployment_config
            set_logical_selection(
                st.session_state,
                command_id,
                "config",
                "config",
                prepared_deployment_config,
            )

    skip_config_placeholder = use_dataset_driven or use_candidate_driven
    prefill_values = (
        prefill.get("values", {})
        if prefill.get("command_id") == command_id
        and prefill.get("action_id") == action_id
        else {}
    )
    for name, value in prefill_values.items():
        if skip_config_placeholder and name == "config":
            continue
        placeholder = action.placeholders.get(name)
        role = placeholder.role if placeholder is not None else "value"
        shared_key = widget_selection_key(command_id, role, name)
        # Force override older durable/widget state from Results prepare actions.
        st.session_state[shared_key] = value
        set_logical_selection(st.session_state, command_id, role, name, value)
        # Keep legacy per-action key in sync for older session handoffs.
        st.session_state[f"value-{command_id}-{action_id}-{name}"] = value

    for name, placeholder in action.placeholders.items():
        if skip_config_placeholder and (
            placeholder.role == "config" or name == "config"
        ):
            continue
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

    config_guard: ConfigGuardResult | None = None
    guarded_config = values.get("config")
    if guarded_config:
        config_guard = guard_config_for_command(
            REPOSITORY_ROOT, command_id, str(guarded_config)
        )
        if not config_guard.ok:
            st.error(
                config_guard.message
                or "The selected configuration does not match this operation."
            )

    if selected_config is not None and selected_config.is_file():
        _config_panel(loaded, selected_config)

    st.session_state["_last_command_id"] = command_id
    st.session_state["_last_action_id"] = action_id

    if deployment_summary:
        dataset_driven_summary = {**deployment_summary, **dataset_driven_summary}
    schema_blocks_launch = config_guard is not None and not config_guard.ok
    built = None
    pre_run: dict[str, str] = {}
    try:
        if action.enabled and not dataset_driven_blocks_launch and (
            not schema_blocks_launch
        ):
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
        elif schema_blocks_launch:
            st.info("Select a configuration that matches this operation's contract.")
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


def _render_advanced_operations_toggle(
    *, widget_key: str, durable_key: str
) -> None:
    """Explicit Advanced / Legacy operations visibility toggle."""
    with st.expander("Advanced / Legacy operations", expanded=False):
        st.caption(
            "The workflow above is the supported path. Superseded operations "
            "stay hidden until you enable them here."
        )
        value = st.checkbox(
            "Show legacy operations",
            key=widget_key,
            help=(
                "Research evaluation v1 is a legacy training backend kept only "
                "so historical runs remain reproducible. Do not use it for new "
                "training."
            ),
        )
        remember_durable_value(
            st.session_state,
            durable_key=durable_key,
            value=bool(value),
            allowed=None,
        )


def _render_workflow_technical_details(
    loaded: ControlPanelRegistry, command_id: str, action_id: str
) -> None:
    """Technical operation identity and declared artifact contracts."""
    command = loaded.commands[command_id]
    action = command.actions[action_id]
    contracts = contract_declaration(command_id)
    accepted = ", ".join(contracts["input"]) or "—"
    produced = ", ".join(contracts["output"]) or "—"
    with st.expander("Technical details", expanded=False):
        lines = [
            f"- **Operation ID:** `{command_id}`",
            f"- **Registry title:** {command.title}",
            f"- **Action ID:** `{action_id}`",
            f"- **Registry description:** {action.description}",
            f"- **Accepted input contract:** `{accepted}`",
        ]
        if contracts["intermediate"]:
            lines.append(
                "- **Intermediate contract:** "
                f"`{', '.join(contracts['intermediate'])}`"
            )
        lines.append(f"- **Produced output contract:** `{produced}`")
        st.markdown("\n".join(lines))


def _archived_research_run_paths() -> frozenset[str]:
    try:
        archive = ArchiveRegistry(REPOSITORY_ROOT)
        return frozenset(
            path
            for reader_id, path in archive.archived_artifact_keys()
            if reader_id == "research_v2"
        )
    except ArchiveError:
        return frozenset()


def _deployment_candidate_controls(
    loaded: ControlPanelRegistry, *, action_id: str
) -> tuple[str | None, dict[str, str], bool]:
    """Canonical completed-run selection and deployment-draft preparation."""
    del loaded
    st.subheader("1. Select a completed run")
    archived = _archived_research_run_paths()
    try:
        rows = inventory_rows_from_mappings(
            _cached_research_inventory_rows(str(REPOSITORY_ROOT))
        )
    except Exception as error:  # noqa: BLE001 - never crash the Run page
        st.error(f"Completed-run inventory is unavailable: {error}")
        return None, {}, True
    all_candidates = list_deployment_candidates(
        rows, archived_paths=archived, include_exploratory=True
    )
    if not all_candidates:
        st.info(
            "No completed canonical run currently satisfies the deployment "
            "candidate contract. Train a run first, then return here."
        )
        return None, {}, True

    handoff = st.session_state.pop(SUBMISSION_HANDOFF_KEY, None)
    exploratory_paths = {
        item.relative_path for item in all_candidates if item.exploratory
    }
    exploratory_key = "deploy-include-exploratory"
    if handoff in exploratory_paths:
        st.session_state[exploratory_key] = True
    include_exploratory = st.checkbox(
        "Include exploratory Dataset Packages",
        key=exploratory_key,
        help=(
            "Exploratory packages used target information during feature "
            "discovery. Any submission built from them stays exploratory."
        ),
    )
    candidates = [
        item for item in all_candidates if include_exploratory or not item.exploratory
    ]
    if not candidates:
        st.info("Enable exploratory packages to see the remaining candidates.")
        return None, {}, True

    paths = [item.relative_path for item in candidates]
    labels = {item.relative_path: item.label for item in candidates}
    durable_key = candidate_durable_key()
    if handoff in paths:
        st.session_state[CANDIDATE_WIDGET_KEY] = handoff
        set_durable_value(st.session_state, durable_key, handoff)
    else:
        sync_widget_with_durable(
            st.session_state,
            widget_key=CANDIDATE_WIDGET_KEY,
            durable_key=durable_key,
            allowed=paths,
            default=paths[0],
        )
    selected_path = st.selectbox(
        "Completed run",
        paths,
        key=CANDIDATE_WIDGET_KEY,
        format_func=lambda value: labels.get(value, value),
    )
    remember_durable_value(
        st.session_state,
        durable_key=durable_key,
        value=selected_path,
        allowed=paths,
    )
    selected = next(
        item for item in candidates if item.relative_path == selected_path
    )
    if selected.exploratory:
        st.warning(
            "This candidate uses an exploratory Dataset Package. It is not a "
            "standard candidate; label every result exploratory."
        )
    duplicates = [
        item
        for item in candidates
        if item.dataset_id == selected.dataset_id
        and item.model_family == selected.model_family
        and item.relative_path != selected.relative_path
    ]
    if duplicates:
        st.info(
            f"{len(duplicates) + 1} completed runs share this dataset and model "
            "family. The exact run above is used; nothing is auto-selected by "
            "maximum Balanced Accuracy."
        )

    st.subheader("2. Deployment readiness")
    readiness = evaluate_deployment_readiness(
        REPOSITORY_ROOT, selected_path, archived_paths=archived
    )
    summary = readiness.summary()
    st.dataframe(
        [{"field": key, "value": value} for key, value in summary.items()],
        width="stretch",
        hide_index=True,
    )
    if readiness.supported:
        st.success("Supported: a deployment draft can be prepared from this run.")
    else:
        st.error("Blocked: required authoritative information is missing.")
        for reason in readiness.blocking_reasons:
            st.write(f"- {reason}")
    for warning in readiness.warnings:
        st.caption(f"⚠ {warning}")

    st.subheader("3. Prepare deployment draft")
    candidate_id = readiness.facts.candidate_id
    existing_config = (
        f"{DEPLOYMENT_DRAFT_ROOT}/{candidate_id}/{DEPLOYMENT_CONFIG_NAME}"
        if candidate_id
        else None
    )
    draft_config: str | None = None
    if existing_config and (REPOSITORY_ROOT / existing_config).is_file():
        draft_config = existing_config
    approver = st.text_input(
        "Approver recorded in the candidate approval",
        key="deploy-approver",
        help=(
            "The deployment contract requires a named manual approval. It cannot "
            "be inferred from a metric."
        ),
    )
    role = st.text_input(
        "Intended deployment role",
        value=DEFAULT_INTENDED_ROLE,
        key="deploy-intended-role",
    )
    reason = st.text_area(
        "Paired-comparison exception reason",
        value=DEFAULT_EXCEPTION_REASON,
        key="deploy-exception-reason",
        help=(
            "A single-run candidate has no paired comparison, so the approval "
            "must record an explicit granted exception and its reason."
        ),
    )
    if st.button(
        "Prepare deployment draft",
        type="primary",
        disabled=not readiness.supported or not str(approver).strip(),
        key="deploy-prepare-draft",
    ):
        try:
            draft = prepare_deployment_draft(
                REPOSITORY_ROOT,
                selected_path,
                approver=str(approver),
                intended_deployment_role=str(role),
                paired_comparison_exception_reason=str(reason),
                archived_paths=archived,
            )
        except DeploymentDraftError as error:
            st.error(str(error))
        else:
            draft_config = draft.deployment_config_path
            if draft.reused:
                st.success(
                    f"Reused the identical draft `{draft.deployment_config_path}`. "
                    "The originally recorded approval evidence is preserved."
                )
            else:
                st.success(f"Prepared `{draft.deployment_config_path}`.")
            st.dataframe(
                [
                    {"field": key, "value": value}
                    for key, value in draft.summary().items()
                ],
                width="stretch",
                hide_index=True,
            )

    st.subheader("4. Validate and rehearse")
    if draft_config is None:
        st.info(
            "Prepare a deployment draft to enable Validate and Synthetic dry run."
        )
    else:
        st.markdown(f"**Deployment draft:** `{draft_config}`")
        st.caption(
            "Validate authenticates the draft, its generated approval, the "
            "completed run, and the threshold evidence without reading "
            "competition data."
        )
        if action_id == "dry_run":
            st.caption(
                "Synthetic dry run needs an approved synthetic fixture directory "
                "under artifacts/deployment_fixtures and a new output directory."
            )
    st.warning(
        "Real competition submission remains blocked: the deployment draft "
        "leaves sample-submission identity unresolved and the real deployment "
        "run stays disabled in the command registry."
    )

    panel_summary = {
        "Selected run": selected_path,
        "Dataset Package": selected.dataset_id,
        "Model family": selected.model_family,
        "Deployment readiness": "supported" if readiness.supported else "blocked",
        "Deployment draft": draft_config or "not prepared",
    }
    return draft_config, panel_summary, draft_config is None


def jobs_page() -> None:
    loaded = registry()
    manager = job_manager(loaded)
    st.title("Jobs")
    st.caption(
        "Deleting a UI job removes only its Control Panel metadata and logs. "
        "Experiment artifacts remain unchanged."
    )
    if st.button("Refresh job status"):
        # Intentionally do not call st.cache_data.clear() — that would wipe the
        # Dataset Registry discovery cache used by the Run page.
        st.rerun()
    try:
        archive = ArchiveRegistry(REPOSITORY_ROOT)
    except ArchiveError as error:
        st.error(f"Archive registry unavailable: {error}")
        return
    show_archived = st.checkbox("Show archived", value=False, key="jobs-show-archived")
    jobs = manager.list_jobs()
    archived_ids = archive.archived_ui_job_ids()
    if show_archived:
        visible = jobs
    else:
        visible = [item for item in jobs if item.job_id not in archived_ids]
    if not visible:
        st.info(
            "No UI jobs match the current visibility filter."
            if jobs
            else "No UI jobs have been started."
        )
        return
    job_ids = [item.job_id for item in visible]
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
        format_func=lambda value: _job_label_with_archive(
            next(item for item in visible if item.job_id == value),
            commands=loaded.commands,
            archived=value in archived_ids,
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
    is_archived = selected in archived_ids
    if is_archived:
        st.info("This job is archived (hidden from the default Jobs list).")
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

    _render_job_archive_controls(
        archive=archive,
        manager=manager,
        record=record,
        is_archived=is_archived,
        visible_job_ids=job_ids,
        job_widget_key=job_widget_key,
        job_durable_key=job_durable_key,
    )


def _job_label_with_archive(
    record: Any,
    *,
    commands: Mapping[str, Any] | None,
    archived: bool,
) -> str:
    label = _job_label(record, commands=commands)
    return f"[archived] {label}" if archived else label


def _render_job_archive_controls(
    *,
    archive: ArchiveRegistry,
    manager: JobManager,
    record: Any,
    is_archived: bool,
    visible_job_ids: list[str],
    job_widget_key: str,
    job_durable_key: str,
) -> None:
    state = str(record.status.get("state") or "")
    terminal = state in TERMINAL_STATES
    st.subheader("Workspace cleanup")
    if not terminal and not is_archived:
        st.caption("Active jobs remain visible and cannot be archived or deleted.")
        return
    if terminal and not is_archived:
        if st.button("Archive job", key="jobs-archive"):
            try:
                archive.archive_ui_job(record.job_id, state=state)
                remaining = [job_id for job_id in visible_job_ids if job_id != record.job_id]
                sync_widget_with_durable(
                    st.session_state,
                    widget_key=job_widget_key,
                    durable_key=job_durable_key,
                    allowed=remaining,
                    default=remaining[0] if remaining else None,
                )
                st.success("Job archived (hidden from the default Jobs list).")
                st.rerun()
            except ArchiveError as error:
                st.error(str(error))
        return
    if is_archived:
        cols = st.columns(2)
        with cols[0]:
            if st.button("Restore job", key="jobs-restore"):
                try:
                    archive.restore_ui_job(record.job_id)
                    st.success("Job restored to the default Jobs list.")
                    st.rerun()
                except ArchiveError as error:
                    st.error(str(error))
        with cols[1]:
            st.caption(
                "Permanent deletion removes only this UI job directory under "
                "`artifacts/ui_jobs/<job-id>/`."
            )
        if terminal:
            try:
                preview = preview_job_deletion(record.root)
            except ArchiveError as error:
                st.error(str(error))
                return
            with st.expander("Permanent deletion preview", expanded=False):
                st.code(
                    "\n".join(
                        [
                            f"Directory: artifacts/ui_jobs/{record.job_id}",
                            "Files:",
                            *[f"  - {name}" for name in preview],
                        ]
                    ),
                    language="text",
                )
            confirm = st.checkbox(
                "I understand this permanently deletes UI job metadata and logs only.",
                key="jobs-delete-confirm",
            )
            typed = st.text_input(
                "Type the exact job ID to confirm permanent deletion",
                key="jobs-delete-typed-id",
            )
            if st.button(
                "Permanently delete UI job",
                disabled=not (confirm and typed == record.job_id),
                key="jobs-delete",
            ):
                try:
                    archive.permanently_delete_archived_ui_job(manager, record.job_id)
                    remaining = [
                        job_id for job_id in visible_job_ids if job_id != record.job_id
                    ]
                    sync_widget_with_durable(
                        st.session_state,
                        widget_key=job_widget_key,
                        durable_key=job_durable_key,
                        allowed=remaining,
                        default=remaining[0] if remaining else None,
                    )
                    st.success(
                        "UI job directory deleted. Experiment artifacts were not modified."
                    )
                    st.rerun()
                except ArchiveError as error:
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
    st.caption(
        "Archive hides an item from the workspace lists and charts without "
        "moving or deleting authoritative experiment artifacts."
    )
    if loaded.settings.mlflow_url:
        st.link_button("Open MLflow", loaded.settings.mlflow_url)
    try:
        archive = ArchiveRegistry(REPOSITORY_ROOT)
    except ArchiveError as error:
        st.error(f"Archive registry unavailable: {error}")
        return
    show_archived = st.checkbox(
        "Show archived", value=False, key="results-show-archived"
    )
    experiments_tab, inspect_tab, compare_tab, research_tab = st.tabs(
        [
            "Experiments",
            "Inspect result",
            "Compare experiments",
            "Research Workspace",
        ]
    )
    with experiments_tab:
        _results_experiments_tab(
            loaded, archive=archive, show_archived=show_archived
        )
    with inspect_tab:
        _results_inspect_tab(loaded, archive=archive, show_archived=show_archived)
    with compare_tab:
        _results_compare_tab(loaded, archive=archive, show_archived=show_archived)
    with research_tab:
        _results_research_workspace_tab(
            loaded, archive=archive, show_archived=show_archived
        )


def _visible_results_artifacts(
    loaded: ControlPanelRegistry,
    reader_id: str,
    *,
    archive: ArchiveRegistry,
    show_archived: bool,
) -> list[ArtifactRecord]:
    reader = loaded.readers[reader_id]
    artifacts = discover_artifacts(REPOSITORY_ROOT, reader)
    return archive.filter_artifacts(
        artifacts, reader_id=reader_id, show_archived=show_archived
    )

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


def _results_experiments_tab(
    loaded: ControlPanelRegistry,
    *,
    archive: ArchiveRegistry,
    show_archived: bool,
) -> None:
    reader_id = _results_reader_select(loaded, "results-experiments-reader")
    artifacts = _visible_results_artifacts(
        loaded, reader_id, archive=archive, show_archived=show_archived
    )
    if not artifacts:
        st.info("No artifacts match this configured reader and visibility filter.")
        return

    rows = build_experiment_table_rows(artifacts, repo_root=REPOSITORY_ROOT)
    archived_keys = archive.archived_artifact_keys()
    for row in rows:
        path = str(row.get("Artifact path") or "")
        if (reader_id, path) in archived_keys:
            status = str(row.get("Status") or "")
            row["Status"] = f"archived · {status}" if status else "archived"
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
    _render_results_archive_controls(
        artifacts=artifacts,
        reader_id=reader_id,
        archive=archive,
        show_archived=show_archived,
        key_prefix="results-exp",
    )


def _render_results_archive_controls(
    *,
    artifacts: list[ArtifactRecord],
    reader_id: str,
    archive: ArchiveRegistry,
    show_archived: bool,
    key_prefix: str,
) -> None:
    if not artifacts:
        return
    st.subheader("Archive result")
    paths = [item.relative_path for item in artifacts]
    archived_keys = archive.archived_artifact_keys()
    selected_path = st.selectbox(
        "Result to archive or restore",
        paths,
        key=f"{key_prefix}-archive-path-{reader_id}",
        format_func=lambda value: (
            f"[archived] {value}"
            if (reader_id, value) in archived_keys
            else value
        ),
    )
    selected = next(item for item in artifacts if item.relative_path == selected_path)
    meta = parse_config_metadata(selected.relative_path, REPOSITORY_ROOT)
    st.markdown(
        "  \n".join(
            [
                f"**Reader:** `{reader_id}`",
                f"**Dataset:** `{meta.get('dataset_id') or meta.get('dataset_version') or 'Not available'}`",
                f"**Model:** `{meta.get('model_family') or 'Not available'}`",
                f"**Mode:** `{meta.get('mode') or 'Not available'}`",
                f"**Status:** `{selected.state}`",
                f"**Artifact path:** `{selected.relative_path}`",
            ]
        )
    )
    is_archived = (reader_id, selected.relative_path) in archived_keys
    if is_archived:
        if st.button("Restore result", key=f"{key_prefix}-restore-{reader_id}"):
            try:
                archive.restore_artifact(
                    reader_id=reader_id, relative_path=selected.relative_path
                )
                st.success("Result restored to default Results visibility.")
                st.rerun()
            except ArchiveError as error:
                st.error(str(error))
        return
    confirm = st.checkbox(
        "Archive this result (hide from lists/charts; do not delete files).",
        key=f"{key_prefix}-archive-confirm-{reader_id}",
    )
    if st.button(
        "Archive result",
        disabled=not confirm,
        key=f"{key_prefix}-archive-{reader_id}",
    ):
        try:
            archive.archive_artifact(
                reader_id=reader_id, relative_path=selected.relative_path
            )
            st.success("Result archived (files unchanged).")
            st.rerun()
        except ArchiveError as error:
            st.error(str(error))


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


def _results_inspect_tab(
    loaded: ControlPanelRegistry,
    *,
    archive: ArchiveRegistry,
    show_archived: bool,
) -> None:
    reader_id = _results_reader_select(loaded, "results-inspect-reader")
    reader = loaded.readers[reader_id]
    artifacts = _visible_results_artifacts(
        loaded, reader_id, archive=archive, show_archived=show_archived
    )
    if not artifacts:
        st.info("No artifacts match this configured reader and visibility filter.")
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
        try:
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
        except Exception as error:
            st.error(f"Comparison dataset identity unavailable: {error}")
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


def _results_compare_tab(
    loaded: ControlPanelRegistry,
    *,
    archive: ArchiveRegistry,
    show_archived: bool,
) -> None:
    reader_id = _results_reader_select(loaded, "results-compare-reader")
    reader = loaded.readers[reader_id]
    artifacts = _visible_results_artifacts(
        loaded, reader_id, archive=archive, show_archived=show_archived
    )
    if not artifacts:
        st.info("No artifacts match this configured reader and visibility filter.")
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
            "display status": compatibility["status"],
        }
    )
    if compatibility.get("descriptive_only"):
        st.warning(
            "Descriptive comparison only: the selected runs use different dataset "
            "fingerprints. The metric delta is not a paired statistical comparison "
            "and the runs are not formally compatible for official Paired Comparison."
        )
    elif compatibility["compatible"]:
        st.caption(
            "Display check: matching plan/fingerprint/fold tokens. "
            "Official Paired Comparison readiness is evaluated separately."
        )
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
        comparison_ready = False
        official = None
        if reader_id == "research_v2":
            official = evaluate_official_paired_readiness(
                left_root=left_item.root,
                right_root=right_item.root,
                repository_root=REPOSITORY_ROOT,
                left_state=left_item.state,
                right_state=right_item.state,
            )
            comparison_ready = bool(official.ready)
            st.subheader("Official Paired Comparison readiness")
            st.write(
                {
                    "ready": official.ready,
                    "compatible": official.compatible,
                    "left dataset": official.left_dataset_version or "Not available",
                    "right dataset": official.right_dataset_version or "Not available",
                    "reason codes": list(official.reason_codes),
                }
            )
            if official.diagnostic:
                st.info(official.diagnostic)
            if not official.ready:
                st.caption(
                    "Prepare Paired Comparison stays disabled until the official "
                    "compatibility contract passes. The CLI remains the final gate."
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


def _results_research_workspace_tab(
    loaded: ControlPanelRegistry,
    *,
    archive: ArchiveRegistry,
    show_archived: bool,
) -> None:
    del loaded  # Research Workspace discovers Research v2 directly.
    st.caption(
        "Research Workspace v1 inventories filesystem Research v2 runs for "
        "descriptive dataset × model comparison. Filesystem artifacts remain "
        "authoritative. Annotations never mutate run artifacts. Baseline "
        "deltas and multi-run tables are descriptive only — not Stage E "
        "paired statistical inference, and not official Paired Comparison v1."
    )
    try:
        annotations = ResearchAnnotationRegistry(REPOSITORY_ROOT)
    except ResearchAnnotationError as error:
        st.error(f"Research annotations unavailable: {error}")
        return

    refresh_cols = st.columns([1, 2, 2])
    with refresh_cols[0]:
        if st.button("Refresh research inventory", key="rw-refresh-inventory"):
            _cached_research_inventory_rows.clear()
            st.session_state["rw_inventory_last_refreshed_utc"] = archive_utc_now_text()
            st.rerun()
    with refresh_cols[1]:
        last_refreshed = st.session_state.get("rw_inventory_last_refreshed_utc")
        if last_refreshed:
            st.caption(f"Inventory last refreshed: `{last_refreshed}`")
        else:
            st.caption("Inventory uses a scoped discovery cache.")
    with refresh_cols[2]:
        st.caption(
            f"Cache version `{RESEARCH_WORKSPACE_CACHE_VERSION}` "
            "(does not call global cache clear)."
        )

    try:
        row_payloads = _cached_research_inventory_rows(
            str(REPOSITORY_ROOT),
            RESEARCH_WORKSPACE_CACHE_VERSION,
        )
        inventory_rows = inventory_rows_from_mappings(row_payloads)
    except Exception as error:  # noqa: BLE001 - surface discovery failures
        st.error(f"Research inventory discovery failed: {error}")
        return

    archived_paths = {
        relative_path
        for reader_id, relative_path in archive.archived_artifact_keys()
        if reader_id == "research_v2"
    }
    annotations_by_path = {
        item["relative_path"]: item for item in annotations.annotations
    }
    stale = annotations.list_stale({row.relative_path for row in inventory_rows})
    if stale:
        with st.expander(f"Stale annotations ({len(stale)})", expanded=False):
            st.warning(
                "These annotation keys no longer resolve to discovered Research "
                "v2 runs. They are retained diagnostically and are not deleted."
            )
            st.code("\n".join(stale))

    annotated = annotate_runs(
        inventory_rows,
        annotations_by_path=annotations_by_path,
        archived_paths=archived_paths,
    )
    st.metric("Discovered Research v2 runs", len(inventory_rows))

    filter_cols = st.columns(4)
    with filter_cols[0]:
        development_only = st.checkbox(
            "Development only", value=True, key="rw-filter-development-only"
        )
        include_exploratory = st.checkbox(
            "Include exploratory", value=False, key="rw-filter-exploratory"
        )
        shortlist_only = st.checkbox(
            "Shortlist only", value=False, key="rw-filter-shortlist"
        )
    with filter_cols[1]:
        model_options = sorted(
            {
                run.row.model_family
                for run in annotated
                if run.row.model_family
            }
        )
        selected_models = st.multiselect(
            "Model family",
            options=model_options,
            default=[],
            key="rw-filter-models",
        )
        dataset_options = sorted(
            {
                run.row.dataset_id
                for run in annotated
                if run.row.dataset_id
            }
        )
        selected_datasets = st.multiselect(
            "Dataset ID",
            options=dataset_options,
            default=[],
            key="rw-filter-datasets",
        )
    with filter_cols[2]:
        adapter_or_config = st.text_input(
            "Adapter / config contains",
            value="",
            key="rw-filter-adapter",
        )
        status_options = sorted({run.row.status for run in annotated})
        selected_statuses = st.multiselect(
            "Status",
            options=status_options,
            default=[],
            key="rw-filter-status",
        )
    with filter_cols[3]:
        tag_options = sorted(BUILTIN_TAGS | {tag for run in annotated for tag in run.tags})
        selected_tags = st.multiselect(
            "Tags",
            options=tag_options,
            default=[],
            key="rw-filter-tags",
        )
        baseline_options = dataset_options or [DEFAULT_BASELINE_DATASET_ID]
        baseline_default = (
            DEFAULT_BASELINE_DATASET_ID
            if DEFAULT_BASELINE_DATASET_ID in baseline_options
            else baseline_options[0]
        )
        baseline_dataset_id = st.selectbox(
            "Baseline Dataset Package",
            options=baseline_options,
            index=baseline_options.index(baseline_default),
            key="rw-baseline-dataset",
        )
        metric_key = st.selectbox(
            "Primary matrix metric",
            options=[item[0] for item in MATRIX_METRICS],
            format_func=lambda value: dict(MATRIX_METRICS)[value],
            key="rw-primary-metric",
        )

    filters = MatrixFilters(
        development_only=development_only,
        include_exploratory=include_exploratory,
        model_families=tuple(selected_models),
        dataset_ids=tuple(selected_datasets),
        adapter_or_config=adapter_or_config,
        statuses=tuple(selected_statuses),
        tags=tuple(selected_tags),
        shortlist_only=shortlist_only,
        show_archived=show_archived,
    )
    explicit_selections = st.session_state.setdefault("rw_explicit_cell_selections", {})
    matrix = build_research_matrix(
        annotated,
        filters=filters,
        baseline_dataset_id=baseline_dataset_id,
        explicit_selections=explicit_selections,
    )
    st.caption(
        "Descriptive baseline delta: "
        f"`delta_BA = BA(candidate) - BA({baseline_dataset_id})`. "
        "Not official paired inference."
    )
    display = matrix_display_rows(matrix, primary_metric=metric_key)
    if display:
        st.dataframe(pd.DataFrame(display), use_container_width=True, hide_index=True)
    else:
        st.info("No Research v2 runs match the current filters.")

    csv_text = export_annotated_runs_csv(matrix.filtered_runs)
    st.download_button(
        "Export visible inventory CSV",
        data=csv_text,
        file_name="research_workspace_inventory.csv",
        mime="text/csv",
        key="rw-export-csv",
    )

    cell_options = [
        f"{dataset_id} × {model_family}"
        for dataset_id in matrix.dataset_ids
        for model_family in matrix.model_families
        if matrix.cells[(dataset_id, model_family)].selected is not None
    ]
    if not cell_options:
        return

    selected_cell_label = st.selectbox(
        "Inspect matrix cell",
        options=cell_options,
        key="rw-selected-cell",
    )
    dataset_id, model_family = selected_cell_label.split(" × ", 1)
    cell = matrix.cells[(dataset_id, model_family)]
    if cell.has_duplicates:
        st.warning(
            f"{cell.duplicate_count} eligible runs for this cell. "
            f"Selection policy: {cell.selection_policy}"
        )
        candidate_paths = [item.row.relative_path for item in cell.candidates]
        current = (
            cell.selected.row.relative_path
            if cell.selected is not None
            else candidate_paths[0]
        )
        chosen = st.selectbox(
            "Choose exact run for this cell",
            options=candidate_paths,
            index=candidate_paths.index(current) if current in candidate_paths else 0,
            key=f"rw-cell-choice-{dataset_id}-{model_family}",
        )
        if chosen != current:
            explicit_selections[(dataset_id, model_family)] = chosen
            st.session_state["rw_explicit_cell_selections"] = explicit_selections
            st.rerun()

    selected_run = cell.selected
    if selected_run is None:
        return

    row = selected_run.row
    st.subheader("Run details")
    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.markdown(
            "\n".join(
                [
                    f"- **Path:** `{row.relative_path}`",
                    f"- **Run ID:** `{row.display('run_id')}`",
                    f"- **Created:** `{row.display('created_at_utc')}`",
                    f"- **Status:** `{row.status}`",
                    f"- **Dataset ID:** `{row.display('dataset_id')}`",
                    f"- **Parent Dataset ID:** `{row.display('parent_dataset_id')}`",
                    f"- **Target dependency:** `{row.display('target_dependency')}`",
                    f"- **Feature count:** `{row.display('n_features')}`",
                    f"- **Model family:** `{row.display('model_family')}`",
                    f"- **Adapter ID:** `{row.display('adapter_id')}`",
                    f"- **Model/config ID:** `{row.display('model_config_id')}`",
                    f"- **Feature pipeline:** `{row.display('feature_pipeline_id')}`",
                ]
            )
        )
    with detail_cols[1]:
        st.markdown(
            "\n".join(
                [
                    f"- **Source config path:** `{row.display('source_config_path')}`",
                    f"- **Source config hash:** `{row.display('source_config_hash')}`",
                    f"- **Resolved config hash:** `{row.display('resolved_config_hash')}`",
                    f"- **Evaluation-plan ID:** `{row.display('evaluation_plan_id')}`",
                    f"- **Evaluation-plan hash:** `{row.display('evaluation_plan_hash')}`",
                    f"- **Evaluation mode:** `{row.display('evaluation_mode')}`",
                    f"- **Repeat seeds:** `{row.display('repeat_seeds')}`",
                    f"- **Outer folds:** `{row.display('outer_fold_count')}`",
                    f"- **Threshold protocol:** `{row.display('threshold_selection_protocol')}`",
                    f"- **Competition assets accessed:** `{row.display('competition_assets_accessed')}`",
                    f"- **Authoritative OOF:** `{row.display('authoritative_oof_path')}`",
                    f"- **Comparability:** `{selected_run.comparability.primary}`",
                ]
            )
        )
    st.write("Comparability reasons:")
    for reason in selected_run.comparability.reasons:
        st.write(f"- {reason}")
    if cell.delta_ba_vs_baseline is not None:
        st.info(
            f"Descriptive delta BA vs `{baseline_dataset_id}`: "
            f"{cell.delta_ba_vs_baseline:.6f}"
        )

    metric_summary = {
        "Balanced Accuracy": row.balanced_accuracy,
        "Sensitivity": row.sensitivity,
        "Specificity": row.specificity,
        "ROC AUC": row.roc_auc,
        "Average Precision": row.average_precision,
        "Brier score": row.brier_score,
        "Threshold median": row.threshold_median,
    }
    st.json({key: (UNAVAILABLE if value is None else value) for key, value in metric_summary.items()})

    st.subheader("Annotations")
    current_annotation = annotations.get(row.relative_path) or {
        "tags": list(selected_run.tags),
        "note": selected_run.note,
        "shortlisted": selected_run.shortlisted,
    }
    tag_selection = st.multiselect(
        "Tags",
        options=sorted(BUILTIN_TAGS | set(current_annotation.get("tags", []))),
        default=list(current_annotation.get("tags", [])),
        key=f"rw-tags-{row.relative_path}",
    )
    note_value = st.text_area(
        "Research note",
        value=str(current_annotation.get("note") or ""),
        key=f"rw-note-{row.relative_path}",
        max_chars=2000,
    )
    ann_cols = st.columns(3)
    with ann_cols[0]:
        if st.button("Save annotation", key=f"rw-save-ann-{row.relative_path}"):
            try:
                annotations.upsert(
                    row.relative_path,
                    tags=tag_selection,
                    note=note_value,
                    shortlisted=bool(current_annotation.get("shortlisted")),
                )
                st.success("Annotation saved.")
                st.rerun()
            except ResearchAnnotationError as error:
                st.error(str(error))
    with ann_cols[1]:
        if st.button("Promote to shortlist", key=f"rw-promote-{row.relative_path}"):
            try:
                annotations.promote_to_shortlist(row.relative_path)
                st.success("Promoted to shortlist.")
                st.rerun()
            except ResearchAnnotationError as error:
                st.error(str(error))
    with ann_cols[2]:
        if st.button("Remove from shortlist", key=f"rw-demote-{row.relative_path}"):
            try:
                annotations.remove_from_shortlist(row.relative_path)
                st.success("Removed from shortlist.")
                st.rerun()
            except ResearchAnnotationError as error:
                st.error(str(error))

    st.subheader("Submission handoff")
    handoff_readiness = evaluate_deployment_readiness(
        REPOSITORY_ROOT, row.relative_path
    )
    if handoff_readiness.supported:
        st.success("This run satisfies the deployment candidate contract.")
    else:
        st.warning("This run cannot be prepared for submission yet.")
        for reason in handoff_readiness.blocking_reasons:
            st.write(f"- {reason}")
    if st.button(
        "Prepare for submission",
        key=f"rw-prepare-submission-{row.relative_path}",
        help=(
            "Transfers only this run's identity to Generate submission. The "
            "Research v2 configuration is never used as a deployment config."
        ),
    ):
        # Run identity only; the draft itself is built by the shared builder
        # inside Generate submission, never duplicated here.
        apply_submission_handoff(st.session_state, row.relative_path)
        st.success(
            "Transferred the selected run to Generate submission. Open Run to "
            "review readiness and prepare the deployment draft."
        )

    st.subheader("Descriptive comparison preview")
    compare_paths = st.multiselect(
        "Select 2–4 runs",
        options=[run.row.relative_path for run in matrix.filtered_runs],
        default=[row.relative_path],
        key="rw-compare-paths",
        max_selections=4,
    )
    if len(compare_paths) >= 2:
        compare_runs = [
            run
            for run in annotated
            if run.row.relative_path in set(compare_paths)
        ]
        reference = st.selectbox(
            "Reference run for descriptive deltas",
            options=compare_paths,
            key="rw-compare-reference",
        )
        table = descriptive_comparison_table(
            compare_runs, reference_relative_path=reference
        )
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
        st.caption(
            "Descriptive aggregate comparison only. Stage E paired inference is "
            "not implemented here."
        )
    elif compare_paths:
        st.info("Select at least two runs for a descriptive comparison table.")


@st.cache_data(show_spinner="Scanning Research v2 inventory…")
def _cached_research_inventory_rows(
    repository_root: str,
    cache_version: str = RESEARCH_WORKSPACE_CACHE_VERSION,
) -> list[dict[str, Any]]:
    """Scoped Research Workspace discovery cache (serializable row mappings)."""
    del cache_version
    rows = build_research_inventory(Path(repository_root))
    return inventory_rows_as_mappings(rows)


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
def _cached_registry_dataset_views(
    repository_root: str,
    processed_root: str = DEFAULT_PROCESSED_ROOT,
    cache_version: str = REGISTRY_DISCOVERY_CACHE_VERSION,
) -> list[dict[str, Any]]:
    """Cache strict Registry discovery for the Run page (serializable views)."""
    del cache_version  # present only to scope the Streamlit cache key
    summaries = discover_registry_summaries(
        Path(repository_root),
        processed_root_relative=processed_root,
    )
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
    refresh_cols = st.columns([1, 2])
    with refresh_cols[0]:
        if st.button("Refresh datasets", key="ecv2-refresh-datasets"):
            _cached_registry_dataset_views.clear()
            st.session_state["ecv2_registry_last_refreshed_utc"] = archive_utc_now_text()
            st.rerun()
    with refresh_cols[1]:
        last_refreshed = st.session_state.get("ecv2_registry_last_refreshed_utc")
        if last_refreshed:
            st.caption(f"Registry last refreshed: `{last_refreshed}`")
        else:
            st.caption("Registry discovery is cached until Refresh datasets.")
    try:
        view_payloads = _cached_registry_dataset_views(
            str(REPOSITORY_ROOT),
            DEFAULT_PROCESSED_ROOT,
            REGISTRY_DISCOVERY_CACHE_VERSION,
        )
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
    try:
        archive = ArchiveRegistry(REPOSITORY_ROOT)
        archived = archive.archived_artifact_keys()
    except ArchiveError:
        archived = frozenset()
    records: list[ArtifactRecord] = []
    for reader in loaded.readers.values():
        for item in discover_artifacts(REPOSITORY_ROOT, reader):
            if (reader.id, item.relative_path) in archived:
                continue
            records.append(item)
    return records


def _job_label(record: Any, commands: Mapping[str, Any] | None = None) -> str:
    return job_primary_label(
        record.job,
        dict(commands) if commands else None,
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
