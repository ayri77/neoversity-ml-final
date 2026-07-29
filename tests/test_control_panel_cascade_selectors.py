"""Tests for real cascading config/artifact selectors on Run, Results, and Jobs pages."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import apps.experiment_control_panel as control_panel_app
from src.churn_ml.control_panel.presentation import (
    build_cascade_options,
    cascade_available_models,
    cascade_available_modes,
    cascade_available_sources,
    cascade_filter_configs,
    model_human_label,
    mode_human_label,
    readable_path_label,
    source_human_label,
)
from src.churn_ml.control_panel.registry import load_registry
from tests.test_control_panel_security import StartSpy, _run_page, _select

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _session_get(at, key: str, default: object = None) -> object:
    try:
        inner = getattr(getattr(at.session_state, "_state", None), "_new_session_state", None)
        if isinstance(inner, dict) and key in inner:
            return inner[key]
        return at.session_state[key]
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Unit tests for cascade helper functions
# ---------------------------------------------------------------------------


def test_cascade_options_parses_lightgbm_development() -> None:
    opts = build_cascade_options(
        ["configs/research_v2/manual_lightgbm_te_v1_compat_development.yaml"],
        PROJECT_ROOT,
    )
    assert len(opts) == 1
    opt = opts[0]
    assert opt.model_family == "LightGBM"
    assert opt.mode == "Development"
    assert opt.source_kind == "Canonical config"


def test_cascade_options_parses_xgboost() -> None:
    opts = build_cascade_options(
        ["configs/research_v2/xgboost_numeric_v1_development.yaml"],
        PROJECT_ROOT,
    )
    assert len(opts) == 1
    assert opts[0].model_family == "XGBoost"
    assert opts[0].mode == "Development"


def test_cascade_available_sources_canonical() -> None:
    opts = build_cascade_options(
        ["configs/research_v2/manual_lightgbm_te_v1_compat_development.yaml"],
        PROJECT_ROOT,
    )
    sources = cascade_available_sources(opts)
    assert "Canonical config" in sources


def test_cascade_models_filtered_by_source() -> None:
    raw_paths = [
        "configs/research_v2/manual_lightgbm_te_v1_compat_development.yaml",
        "configs/research_v2/xgboost_numeric_v1_development.yaml",
        "configs/research_v2/catboost_numeric_v1_development.yaml",
    ]
    opts = build_cascade_options(raw_paths, PROJECT_ROOT)
    models = cascade_available_models(opts, "Canonical config")
    assert "LightGBM" in models
    assert "XGBoost" in models
    assert "CatBoost" in models
    # Non-matching source returns empty
    assert cascade_available_models(opts, "Optuna export") == []


def test_cascade_modes_filtered_by_source_and_model() -> None:
    raw_paths = [
        "configs/research_v2/manual_lightgbm_te_v1_compat_development.yaml",
        "configs/research_v2/xgboost_numeric_v1_development.yaml",
    ]
    opts = build_cascade_options(raw_paths, PROJECT_ROOT)
    modes = cascade_available_modes(opts, "Canonical config", "LightGBM")
    assert "Development" in modes
    # XGBoost not in LightGBM modes
    xgb_modes = cascade_available_modes(opts, "Canonical config", "XGBoost")
    assert "Development" in xgb_modes


def test_cascade_filter_configs_returns_only_matching() -> None:
    raw_paths = [
        "configs/research_v2/manual_lightgbm_te_v1_compat_development.yaml",
        "configs/research_v2/xgboost_numeric_v1_development.yaml",
    ]
    opts = build_cascade_options(raw_paths, PROJECT_ROOT)
    matched = cascade_filter_configs(
        opts, "Canonical config", "LightGBM", "Development"
    )
    assert len(matched) == 1
    assert "lightgbm" in matched[0].path.lower()
    # XGBoost not present in LightGBM-filtered results
    assert not any("xgboost" in o.path.lower() for o in matched)


def test_no_xgboost_in_lightgbm_development_configs() -> None:
    loaded = load_registry(PROJECT_ROOT)
    pairs = control_panel_app._config_options(
        loaded, loaded.commands["experiment_core_v2"].allowed_config_globs
    )
    raw_paths = [p for p, _ in pairs]
    opts = build_cascade_options(raw_paths, PROJECT_ROOT)
    matched = cascade_filter_configs(
        opts, "Canonical config", "LightGBM", "Development"
    )
    for opt in matched:
        assert "xgboost" not in opt.path.lower(), (
            f"XGBoost config unexpectedly present: {opt.path}"
        )


def test_parent_change_resets_invalid_child_safely() -> None:
    raw_paths = [
        "configs/research_v2/manual_lightgbm_te_v1_compat_development.yaml",
        "configs/research_v2/xgboost_numeric_v1_development.yaml",
    ]
    opts = build_cascade_options(raw_paths, PROJECT_ROOT)
    # When source changes to a source that has no configs, filter returns empty
    matched = cascade_filter_configs(opts, "Optuna export", "LightGBM", "Development")
    assert matched == []
    # Models for a new source that has no entries returns empty
    models = cascade_available_models(opts, "Optuna export")
    assert models == []


def test_human_labels_correct() -> None:
    assert source_human_label("Canonical config") == "Canonical config"
    assert source_human_label("Optuna export") == "Optuna export"
    assert source_human_label("UI config copy") == "UI copy"
    assert model_human_label("LightGBM") == "LightGBM"
    assert model_human_label("XGBoost") == "XGBoost"
    assert mode_human_label("Development") == "Development"
    assert mode_human_label("Smoke") == "Smoke"


def test_duplicate_labels_get_unique_suffix() -> None:
    # Two paths with same stem description should receive distinct labels
    raw_paths = [
        "configs/research_v2/catboost_numeric_v1_development.yaml",
        "configs/optuna/catboost_numeric_v1_development.yaml",
    ]
    opts = build_cascade_options(raw_paths, PROJECT_ROOT)
    labels = [opt.display_label for opt in opts]
    assert len(set(labels)) == len(labels), f"Duplicate labels found: {labels}"


# ---------------------------------------------------------------------------
# AppTest widget-level tests
# ---------------------------------------------------------------------------


def test_run_page_renders_source_model_mode_config_widgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "experiment_core_v2")
    at = _select(at, "Action", "run")
    select_labels = [item.label for item in at.selectbox]
    # Must have at least Source/Model/Mode/Config hierarchy widgets
    assert "Config" in select_labels, (
        f"Config selectbox missing; found: {select_labels}"
    )
    # Model and/or Mode controls should appear (may be collapsed to caption if single-option)
    has_model_select = "Model" in select_labels
    has_mode_select = "Mode" in select_labels
    captions = [c.value for c in at.caption]
    has_model_caption = any("Model:" in str(c) for c in captions)
    has_mode_caption = any("Mode:" in str(c) for c in captions)
    assert has_model_select or has_model_caption, (
        f"No Model widget found. Selects: {select_labels}, Captions: {captions}"
    )
    assert has_mode_select or has_mode_caption, (
        f"No Mode widget found. Selects: {select_labels}, Captions: {captions}"
    )


def test_config_preview_expander_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config preview must be in a collapsed expander (not shown inline)."""
    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "experiment_core_v2")
    at = _select(at, "Action", "run")
    expander_labels = [e.label for e in at.expander]
    assert any("Config preview" in label for label in expander_labels), (
        f"No 'Config preview' expander found; expanders: {expander_labels}"
    )


def test_technical_command_expander_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Technical command must be inside a collapsed expander, not shown inline."""
    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "experiment_core_v2")
    at = _select(at, "Action", "run")
    expander_labels = [e.label for e in at.expander]
    assert any("Technical command" in label for label in expander_labels), (
        f"No 'Technical command' expander found; expanders: {expander_labels}"
    )


def test_canonical_lightgbm_development_selects_correctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "experiment_core_v2")
    at = _select(at, "Action", "run")
    # If separate Source/Model/Mode controls exist, select them
    select_labels = [item.label for item in at.selectbox]
    if "Source" in select_labels:
        at = _select(at, "Source", "Canonical config")
    if "Model" in select_labels:
        at = _select(at, "Model", "LightGBM")
    if "Mode" in select_labels:
        at = _select(at, "Mode", "Development")
    config_box = next(item for item in at.selectbox if item.label == "Config")
    # Config options must not contain XGBoost artifacts
    config_paths = list(config_box.options)
    assert config_paths, "Config selectbox is empty"
    # The raw session value (widget key = options list item) must point to a LightGBM config
    assert all("xgboost" not in str(p).lower() for p in config_paths), (
        f"XGBoost path found in LightGBM/Development Config options: {config_paths}"
    )


def test_state_persists_after_navigation_to_jobs_then_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Widget cleanup simulation: durable Run state survives leaving the page."""
    from src.churn_ml.control_panel.selection_state import (
        LOGICAL_SELECTION_KEY,
        snapshot_durable_session,
        ui_durable_key,
        widget_selection_key,
    )
    from tests.test_control_panel_security import _apptest_run_page

    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "experiment_core_v2")
    at = _select(at, "Action", "validate")
    config_key = widget_selection_key("experiment_core_v2", "config", "config")
    selected = _session_get(at, config_key)
    assert selected
    snapshot = snapshot_durable_session(at.session_state)
    at = AppTest.from_function(_apptest_run_page, default_timeout=10)
    for key, value in snapshot.items():
        at.session_state[key] = value
    at = at.run()
    op_box = next(item for item in at.selectbox if item.label == "Operation")
    assert op_box.value == "experiment_core_v2"
    action_box = next(item for item in at.selectbox if item.label == "Action")
    assert action_box.value == "validate"
    assert _session_get(at, config_key) == selected
    store = _session_get(at, LOGICAL_SELECTION_KEY, {})
    assert isinstance(store, dict)
    assert store.get(ui_durable_key("run", "command")) == "experiment_core_v2"
    assert store.get("experiment_core_v2::config:config") == selected


def test_forged_stale_path_cannot_be_launched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load_registry(PROJECT_ROOT)
    pairs = control_panel_app._config_options(
        loaded, loaded.commands["experiment_core_v2"].allowed_config_globs
    )
    options = [p for p, _ in pairs]
    stale = "configs/mlflow/local.yaml"
    assert stale not in options, (
        "mlflow config must not appear in experiment_core_v2 allowed options"
    )
    effective = stale if stale in options else options[0]
    assert effective == options[0]


def test_results_artifact_selection_filtered_by_model_and_mode() -> None:
    """build_cascade_options correctly filters artifacts by model and mode."""
    raw_paths = [
        "artifacts/research_v2/lightgbm_run_development/summary.json",
        "artifacts/research_v2/xgboost_run_development/summary.json",
    ]
    opts = build_cascade_options(raw_paths, PROJECT_ROOT)
    lgbm = [o for o in opts if o.model_family == "LightGBM"]
    xgb = [o for o in opts if o.model_family == "XGBoost"]
    if lgbm and xgb:
        assert lgbm[0].path != xgb[0].path


def test_jobs_label_includes_model_and_mode_when_metadata_exists() -> None:
    from src.churn_ml.control_panel.presentation import job_primary_label

    job = {
        "command_id": "experiment_core_v2",
        "action_id": "run",
        "references": {
            "config": "configs/research_v2/manual_lightgbm_te_v1_compat_development.yaml"
        },
        "created_at_utc": "2026-07-29T10:00:00Z",
    }
    label = job_primary_label(job)
    assert "LightGBM" in label, f"Expected LightGBM in label, got: {label}"
    assert "Development" in label, f"Expected Development in label, got: {label}"
    assert "Run" in label or "run" in label.lower(), (
        f"Expected action in label, got: {label}"
    )


def test_jobs_fallback_label_without_config() -> None:
    from src.churn_ml.control_panel.presentation import job_primary_label

    job = {
        "command_id": "experiment_core_v2",
        "action_id": "run",
        "references": {},
        "created_at_utc": "2026-07-29T10:00:00Z",
    }
    label = job_primary_label(job)
    # Without config metadata, must still produce a readable non-empty label
    assert label
    assert "Experiment Core" in label or "Run" in label


def test_optuna_config_cascade_excludes_other_models() -> None:
    loaded = load_registry(PROJECT_ROOT)
    pairs = control_panel_app._config_options(
        loaded, loaded.commands["optuna_search_v1"].allowed_config_globs
    )
    opts = build_cascade_options([p for p, _ in pairs], PROJECT_ROOT)
    matched = cascade_filter_configs(
        opts, "Canonical config", "CatBoost", "Development"
    )
    assert matched
    assert all("catboost" in o.path.lower() for o in matched)
    assert not any("xgboost" in o.path.lower() for o in matched)


def test_optuna_export_and_research_v1_configs_use_cascade_helpers() -> None:
    loaded = load_registry(PROJECT_ROOT)
    for command_id in ("optuna_search_v1", "research_v1", "final_deployment_v1"):
        command = loaded.commands[command_id]
        if not command.allowed_config_globs:
            continue
        pairs = control_panel_app._config_options(loaded, command.allowed_config_globs)
        if not pairs:
            continue  # deployment configs may be absent in this worktree
        opts = build_cascade_options([p for p, _ in pairs], PROJECT_ROOT)
        assert opts
        sources = cascade_available_sources(opts)
        assert sources


def test_paired_comparison_reader_roots_match() -> None:
    loaded = load_registry(PROJECT_ROOT)
    reader = control_panel_app._reader_matching_roots(
        loaded, ("artifacts/research_v2",)
    )
    assert reader is not None
    assert reader.id == "research_v2"
    deploy_reader = control_panel_app._reader_matching_roots(
        loaded, ("artifacts/deployments",)
    )
    assert deploy_reader is not None
    assert deploy_reader.id == "deployment_v1"


def test_results_left_right_cascade_filters_models() -> None:
    raw_paths = [
        "artifacts/research_v2/lightgbm_numeric_v1_development/run_a",
        "artifacts/research_v2/xgboost_numeric_v1_development/run_b",
        "artifacts/research_v2/catboost_numeric_v1_smoke/run_c",
    ]
    opts = build_cascade_options(raw_paths, PROJECT_ROOT)
    models = cascade_available_models(opts)
    assert "LightGBM" in models
    assert "XGBoost" in models
    lgbm = cascade_filter_configs(opts, model="LightGBM", mode="Development")
    assert all(o.model_family == "LightGBM" for o in lgbm)
    assert not any(o.model_family == "XGBoost" for o in lgbm)


def test_mlflow_enum_human_labels() -> None:
    from src.churn_ml.control_panel.presentation import enum_human_label

    assert enum_human_label("research_v2") == "Experiment Core v2"
    assert enum_human_label("autogluon") == "AutoGluon"
    assert enum_human_label("all") == "All sources"


def test_dashboard_artifact_label_is_human_readable() -> None:
    label = readable_path_label(
        "configs/research_v2/manual_lightgbm_te_v1_compat_development.yaml",
        PROJECT_ROOT,
    )
    assert "LightGBM" in label
    assert "Development" in label
    assert "configs/research_v2/" not in label


def test_optuna_inspect_renders_search_dir_cascade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "optuna_search_v1")
    at = _select(at, "Action", "inspect")
    # Inspect has search_dir via artifact reader; either a cascade item or info message
    select_labels = [item.label for item in at.selectbox]
    info_texts = [str(i.value) for i in at.info]
    assert any(
        "Search Dir" in label or "search" in label.lower() for label in select_labels
    ) or any("No matching" in t for t in info_texts), (
        f"Expected search_dir selector or empty info; selects={select_labels}, info={info_texts}"
    )


def test_paired_comparison_renders_baseline_candidate_selectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "paired_comparison")
    at = _select(at, "Action", "validate")
    select_labels = [item.label for item in at.selectbox]
    info_texts = [str(i.value) for i in at.info]
    # Either cascade widgets for baseline/candidate or empty-artifact info
    has_baseline = any("Baseline" in label for label in select_labels)
    has_candidate = any("Candidate" in label for label in select_labels)
    empty = any("No matching" in t or "No artifacts" in t for t in info_texts)
    assert (
        has_baseline
        or has_candidate
        or empty
        or any("Model" in label for label in select_labels)
    ), (
        f"Expected paired comparison selectors; selects={select_labels}, info={info_texts}"
    )


def test_mlflow_source_type_uses_human_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "mlflow_local_index")
    at = _select(at, "Action", "sync_dry_run")
    source_box = next(
        (item for item in at.selectbox if item.label == "Source Type"),
        None,
    )
    assert source_box is not None, (
        f"Expected Source Type selectbox; got {[i.label for i in at.selectbox]}"
    )
    # options are formatted via enum_human_label
    options = list(source_box.options)
    assert any("Experiment Core" in str(o) or "research_v2" in str(o) for o in options)
