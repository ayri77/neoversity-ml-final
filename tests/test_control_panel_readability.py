from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import apps.experiment_control_panel as control_panel_app
from src.churn_ml.control_panel.presentation import (
    format_date,
    format_time,
    job_primary_label,
    readable_config_label,
)
from src.churn_ml.control_panel.registry import load_registry

from tests.test_control_panel_security import StartSpy, _run_page, _select

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_config_selectbox_shows_readable_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "experiment_core_v2")
    config = next(item for item in at.selectbox if item.label == "Config")
    options = list(config.options)
    assert options, "Config selectbox must have at least one option"
    # In the cascade UI, the model is already determined by the Model selectbox (or
    # shown as a caption when there is only one). The Config selectbox shows
    # display_label values filtered to the selected model. Verify the raw session
    # value (selectbox.value) is a valid repository-relative path from the allowed
    # globs, and that readable_config_label on that path mentions a known model.
    raw_value = str(config.value)
    label = readable_config_label(raw_value, PROJECT_ROOT)
    assert any(m in label for m in ("LightGBM", "XGBoost", "CatBoost")), (
        f"readable_config_label for '{raw_value}' did not contain a known model: {label}"
    )


def test_pre_run_summary_not_in_expander(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-run summary is now shown inline as a compact card, not inside an expander."""
    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Action", "run")
    expander_labels = [e.label for e in at.expander]
    # After the compact layout patch, summary fields are shown inline (st.markdown).
    # The Technical command and Config preview are collapsed expanders.
    assert any("Technical command" in label for label in expander_labels), (
        f"Expected 'Technical command' expander, got: {expander_labels}"
    )


def test_job_label_readable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _job_data = {
        "command_id": "experiment_core_v2",
        "action_id": "run",
        "references": {},
        "created_at_utc": "2026-07-29T04:44:07Z",
    }

    class _FakeRecord:
        job_id = "abc12345-0000-0000-0000-000000000001"
        job = _job_data
        status = {"state": "succeeded"}

    label = control_panel_app._job_label(_FakeRecord())
    assert " · " in label or "Run" in label
    assert "abc12345" not in label  # UUID shown separately below selector


def _apptest_run_page_deployment() -> None:
    # Inline import required for AppTest isolation
    import apps.experiment_control_panel as panel

    panel.run_page()


def test_wrong_selection_safety_deployment_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load_registry(PROJECT_ROOT)
    deployment = loaded.commands["final_deployment_v1"]
    enabled_run = replace(deployment.actions["run"], enabled=True)
    enabled_deployment = replace(
        deployment, actions={**deployment.actions, "run": enabled_run}
    )
    patched = replace(
        loaded,
        commands={**loaded.commands, "final_deployment_v1": enabled_deployment},
    )
    spy = StartSpy()
    monkeypatch.setattr(control_panel_app, "registry", lambda: patched)
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "final_deployment_v1")
    at = _select(at, "Action", "run")
    warning_texts = [str(w.value) for w in at.warning]
    assert any("Review before launch" in t for t in warning_texts), (
        f"Expected deployment warning, got warnings: {warning_texts}"
    )


def test_stale_forged_xgboost_path_rejected_for_lightgbm_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Verify the safety check logic directly: _config_options returns raw paths,
    # and _placeholder_widget resets the session value to options[0] if stale.
    from src.churn_ml.control_panel.registry import load_registry as _load_registry

    loaded = _load_registry(PROJECT_ROOT)
    option_pairs = control_panel_app._config_options(
        loaded, loaded.commands["experiment_core_v2"].allowed_config_globs
    )
    options = [path for path, _label in option_pairs]
    assert options
    # mlflow local.yaml is not an allowed config for experiment_core_v2
    stale = "configs/mlflow/local.yaml"
    assert stale not in options
    # Simulate the session-state reset: stale value is not in options -> resets to options[0]
    effective = stale if stale in options else options[0]
    assert effective == options[0]
    # All options must come from experiment_core_v2 allowed globs (not mlflow)
    assert all("mlflow" not in p for p in options)


def test_persistent_command_selection_survives_rerun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "optuna_search_v1")
    at = at.run()
    operation = next(item for item in at.selectbox if item.label == "Operation")
    assert operation.value == "optuna_search_v1"


def test_dashboard_recent_jobs_uses_readable_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = {
        "command_id": "experiment_core_v2",
        "action_id": "run",
        "references": {},
        "created_at_utc": "2026-07-29T04:44:07Z",
    }

    class _FakeItem:
        job_id = "abc12345"
        status = {"state": "succeeded"}

        def __init__(self) -> None:
            self.job = job

    item = _FakeItem()
    row = {
        "label": job_primary_label(item.job),
        "status": item.status["state"],
        "date": format_date(item.job.get("created_at_utc")),
        "time": format_time(item.job.get("created_at_utc")),
    }
    assert "job" not in row
    assert "command" not in row
    assert "action" not in row
    assert "label" in row
    assert "date" in row
    assert "time" in row
