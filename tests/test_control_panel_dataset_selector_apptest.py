"""AppTest coverage for Experiment Core dataset-driven entry mode."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import apps.experiment_control_panel as control_panel_app
from src.churn_ml.control_panel.command_builder import build_command, display_argv
from src.churn_ml.control_panel.dataset_experiment_materializer import (
    RegisteredDatasetView,
    discover_registry_summaries,
)
from tests.test_control_panel_security import StartSpy, _run_page, _select


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
LIGHTGBM_SMOKE = "configs/research_v2/manual_lightgbm_te_v1_compat_smoke.yaml"


def _has_registry() -> bool:
    return PROCESSED_ROOT.is_dir() and any(
        (path / "dataset_manifest.json").is_file()
        for path in PROCESSED_ROOT.iterdir()
        if path.is_dir()
    )


def _has_v7() -> bool:
    return (
        PROCESSED_ROOT / "v7_compact_zero_indicators" / "dataset_manifest.json"
    ).is_file()


def _set_radio(at: AppTest, label: str, value: str) -> AppTest:
    element = next(item for item in at.radio if item.label == label)
    element.set_value(value)
    return at.run(timeout=90)


def _fast_registry_views(_repository_root: str) -> list[dict]:
    summaries = discover_registry_summaries(PROJECT_ROOT)
    # Use real summaries but avoid repeated full scans inside AppTest reruns by
    # relying on Streamlit cache + this helper being patched once per test.
    return [RegisteredDatasetView.from_summary(item).__dict__ for item in summaries]


@pytest.fixture
def cached_registry_views() -> list[dict]:
    return _fast_registry_views(str(PROJECT_ROOT))


@pytest.mark.skipif(not _has_registry(), reason="local Registry required")
def test_dataset_driven_shows_lineage_and_exploratory_warning(
    monkeypatch: pytest.MonkeyPatch,
    cached_registry_views: list[dict],
) -> None:
    monkeypatch.setattr(
        control_panel_app,
        "_cached_registry_dataset_views",
        lambda _root: cached_registry_views,
    )
    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "experiment_core_v2")
    at = _set_radio(at, "Experiment Core entry mode", "Dataset-driven experiment")
    at = _select(at, "Registered Dataset Package", "v3_targeted_missingness")
    warning_text = "\n".join(str(item.value) for item in at.warning)
    assert "exploratory" in warning_text.lower()
    option_labels = []
    for box in at.selectbox:
        if box.label == "Registered Dataset Package":
            option_labels.extend(str(item) for item in box.options)
    assert any("exploratory" in label for label in option_labels)
    assert spy.calls == []


@pytest.mark.skipif(not _has_registry(), reason="local Registry required")
def test_empty_registry_blocks_dataset_driven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(_root: str) -> list[dict]:
        raise control_panel_app.DatasetExperimentMaterializerError(
            "Dataset Registry contains no strictly validated packages with "
            "canonical dataset_manifest.json."
        )

    monkeypatch.setattr(control_panel_app, "_cached_registry_dataset_views", _boom)
    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "experiment_core_v2")
    at = _set_radio(at, "Experiment Core entry mode", "Dataset-driven experiment")
    assert any("no strictly validated" in str(item.value) for item in at.error)
    assert spy.calls == []


@pytest.mark.skipif(not _has_v7(), reason="v7 package not present")
def test_prepare_then_validate_argv_points_at_prepared_config(
    monkeypatch: pytest.MonkeyPatch,
    cached_registry_views: list[dict],
) -> None:
    monkeypatch.setattr(
        control_panel_app,
        "_cached_registry_dataset_views",
        lambda _root: cached_registry_views,
    )
    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "experiment_core_v2")
    at = _set_radio(at, "Experiment Core entry mode", "Dataset-driven experiment")
    at = _select(at, "Registered Dataset Package", "v7_compact_zero_indicators")
    at = _select(at, "Model", "LightGBM")
    at = _select(at, "Evaluation mode", "Smoke")
    # Sole LightGBM smoke template is selected by default via sync_widget_with_durable.
    button = next(item for item in at.button if item.label == "Prepare run configuration")
    button.click()
    at = at.run(timeout=120)
    try:
        prepared = at.session_state["ecv2_prepared_bundle"]
    except Exception:
        prepared = None
    assert prepared is not None, (
        "Prepare run configuration did not store a prepared bundle; "
        f"errors={[str(item.value) for item in at.error]} "
        f"success={[str(item.value) for item in at.success]}"
    )
    assert prepared.base_config_relative == LIGHTGBM_SMOKE
    config_rel = prepared.config_relative
    at = _select(at, "Action", "validate")
    built = build_command(
        control_panel_app.registry().commands,
        "experiment_core_v2",
        "validate",
        {"config": config_rel},
        repository_root=PROJECT_ROOT,
    )
    argv_text = display_argv(built.redacted_argv)
    assert config_rel in argv_text
    assert "--validate-only" in built.argv
    at = _select(at, "Registered Dataset Package", "v0_raw_minimal")
    try:
        stale = at.session_state["ecv2_prepared_bundle"]
    except Exception:
        stale = None
    assert stale is None
    assert spy.calls == []


@pytest.mark.skipif(not _has_registry(), reason="local Registry required")
def test_existing_config_mode_still_exposes_cascade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "experiment_core_v2")
    at = _set_radio(at, "Experiment Core entry mode", "Existing config")
    labels = [item.label for item in at.selectbox]
    assert "Source" in labels or any("Config" in label for label in labels)
    assert spy.calls == []


def test_materializer_module_import_does_not_load_fitting_runtime() -> None:
    code = (
        "import sys; "
        "import src.churn_ml.control_panel.dataset_experiment_materializer as m; "
        "forbidden=('optuna','lightgbm','xgboost','catboost'); "
        "assert not any("
        "any(x == name or name.endswith('.' + x) for x in forbidden) "
        "for name in sys.modules)"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        check=True,
        timeout=20,
    )
