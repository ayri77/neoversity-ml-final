"""Focused control-panel state bug fixes: cascade, comparison ID, charts, MLflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

import apps.experiment_control_panel as control_panel_app
from src.churn_ml.control_panel.artifacts import (
    build_experiment_table_rows,
    default_comparison_id,
    discover_artifacts,
    sanitize_comparison_id,
)
from src.churn_ml.control_panel.command_builder import build_command
from src.churn_ml.control_panel.mlflow_post_index import (
    extract_run_directory,
    maybe_index_successful_job,
    source_type_for_job,
)
from src.churn_ml.control_panel.presentation import enum_human_label
from src.churn_ml.control_panel.registry import load_registry
from src.churn_ml.control_panel.selection_state import (
    LOGICAL_SELECTION_KEY,
    apply_cascade_reconciliation,
    reconcile_cascade_selection,
    seed_widget_from_logical,
    validate_against_allowed,
    widget_selection_key,
)
from tests.test_control_panel_results_workspace import _write_research_run
from tests.test_control_panel_security import StartSpy, _run_page, _select


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATBOOST_DEV = "configs/optuna/catboost_numeric_v1_development.yaml"
XGBOOST_SMOKE = "configs/optuna/xgboost_numeric_v1_smoke.yaml"


def _session_get(at: AppTest, key: str, default: object = None) -> object:
    try:
        return at.session_state[key]
    except Exception:
        return default


def test_reconcile_cascade_falls_back_atomically() -> None:
    matched = [CATBOOST_DEV, "configs/optuna/catboost_numeric_v1_smoke.yaml"]
    result = reconcile_cascade_selection(
        allowed_paths=[CATBOOST_DEV, XGBOOST_SMOKE, *matched],
        matched_paths=matched,
        current_path=XGBOOST_SMOKE,
        default_index=0,
    )
    assert result.ok
    assert result.path == CATBOOST_DEV


def test_reconcile_cascade_fail_closed_when_empty() -> None:
    result = reconcile_cascade_selection(
        allowed_paths=[XGBOOST_SMOKE],
        matched_paths=[],
        current_path=XGBOOST_SMOKE,
    )
    assert not result.ok
    assert result.path is None
    session: dict[str, Any] = {"cfg": XGBOOST_SMOKE, "cfg__flat": XGBOOST_SMOKE}
    assert (
        apply_cascade_reconciliation(
            session,
            widget_key="cfg",
            reconciliation=result,
            parent_fingerprint=("Canonical config", "CatBoost", "Development"),
        )
        is None
    )
    assert "cfg" not in session
    assert "cfg__flat" not in session


def test_adversarial_parent_change_resolves_catboost_development(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """XGBoost Smoke → Canonical/CatBoost/Development without touching Config."""
    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "optuna_search_v1")
    at = _select(at, "Action", "validate")
    labels = [item.label for item in at.selectbox]
    if "Source" in labels:
        at = _select(at, "Source", "Canonical config")
    if "Model" in labels:
        at = _select(at, "Model", "XGBoost")
    if "Mode" in labels:
        at = _select(at, "Mode", "Smoke")
    config_key = widget_selection_key("optuna_search_v1", "config", "config")
    assert "xgboost" in str(_session_get(at, config_key)).lower()

    if "Model" in [item.label for item in at.selectbox]:
        at = _select(at, "Model", "CatBoost")
    if "Mode" in [item.label for item in at.selectbox]:
        at = _select(at, "Mode", "Development")

    selected = str(_session_get(at, config_key))
    flat = str(_session_get(at, f"{config_key}__flat"))
    assert selected == CATBOOST_DEV
    assert flat == CATBOOST_DEV

    captions = [str(item.value) for item in at.caption]
    assert any(CATBOOST_DEV in text for text in captions)
    assert not any("xgboost_numeric_v1_smoke" in text.lower() for text in captions)

    markdowns = [str(item.value) for item in at.markdown]
    assert any("CatBoost" in text for text in markdowns)
    assert any("Development" in text for text in markdowns)
    assert not any("xgboost_numeric_v1_smoke" in text.lower() for text in markdowns)

    codes = [str(item.value) for item in at.code]
    assert any(CATBOOST_DEV in text for text in codes)
    assert not any("xgboost_numeric_v1_smoke" in text.lower() for text in codes)

    built = build_command(
        load_registry(PROJECT_ROOT).commands,
        "optuna_search_v1",
        "validate",
        {"config": selected},
        repository_root=PROJECT_ROOT,
    )
    assert CATBOOST_DEV in built.argv
    assert XGBOOST_SMOKE not in built.argv


def test_comparison_id_generation_from_left_right(tmp_path: Path) -> None:
    left_path = _write_research_run(
        tmp_path,
        plan="telecom_v3_development_r2x5_t3_v1",
        adapter="manual_lightgbm_te_v1_compat",
        run_id="20260729T070000000000Z_aaaaaaaa",
        ba=0.9,
    )
    right_smoke = _write_research_run(
        tmp_path,
        plan="telecom_v3_smoke_r1x3_t2_v1",
        adapter="manual_lightgbm_te_v1_compat",
        run_id="20260729T070100000000Z_bbbbbbbb",
        ba=0.91,
    )
    right_xgb = _write_research_run(
        tmp_path,
        plan="telecom_v3_development_r2x5_t3_v1",
        adapter="xgboost_numeric_v1",
        run_id="20260729T070200000000Z_cccccccc",
        ba=0.92,
    )
    loaded = load_registry(PROJECT_ROOT)
    artifacts = {
        item.root.resolve(): item
        for item in discover_artifacts(tmp_path, loaded.readers["research_v2"])
    }
    left = artifacts[left_path.resolve()]
    right_a = artifacts[right_smoke.resolve()]
    right_b = artifacts[right_xgb.resolve()]
    first = default_comparison_id(left, right_a, repo_root=tmp_path)
    second = default_comparison_id(left, right_b, repo_root=tmp_path)
    assert first.startswith("lightgbm-vs-lightgbm-")
    assert second == "lightgbm-vs-xgboost-development-r2x5-t3-v1"
    assert first != second
    assert sanitize_comparison_id("Bad ID!!") == "bad-id"


def test_comparison_id_widget_auto_manual_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_research_run(
        tmp_path,
        plan="telecom_v3_development_r2x5_t3_v1",
        adapter="manual_lightgbm_te_v1_compat",
        run_id="20260729T070702417916Z_38bbefe2",
        ba=0.8975,
    )
    _write_research_run(
        tmp_path,
        plan="telecom_v3_development_r2x5_t3_v1",
        adapter="xgboost_numeric_v1",
        run_id="20260729T064407009547Z_55c26d5e",
        ba=0.8975,
    )
    monkeypatch.setattr(control_panel_app, "REPOSITORY_ROOT", tmp_path)
    original_load = control_panel_app.load_registry

    def load_project_registry(root: Path):
        del root
        return original_load(PROJECT_ROOT)

    monkeypatch.setattr(control_panel_app, "load_registry", load_project_registry)
    control_panel_app.registry.clear()

    def _compare_page() -> None:
        import apps.experiment_control_panel as panel

        panel.results_page()

    at = AppTest.from_function(_compare_page, default_timeout=15).run()
    assert not at.exception
    id_key = "results-comparison-id-research_v2"
    at.session_state[id_key] = "lightgbm-vs-lightgbm-smoke-r1x3-t2-v1"
    at.session_state[f"{id_key}__manual"] = False
    at.session_state[f"{id_key}__sides"] = ("old-left", "old-right")
    at = at.run()
    current = str(_session_get(at, id_key) or "")
    assert "smoke" not in current

    at.session_state[id_key] = "my-custom-comparison-id"
    at.session_state[f"{id_key}__manual"] = True
    at.session_state[f"{id_key}__sides"] = ("changed-left", "changed-right")
    at = at.run()
    assert _session_get(at, id_key) == "my-custom-comparison-id"

    reset = next(
        (item for item in at.button if item.label == "Reset to suggested ID"), None
    )
    assert reset is not None
    reset.click()
    at = at.run()
    assert _session_get(at, id_key) == _session_get(at, f"{id_key}__suggested")
    assert not _session_get(at, f"{id_key}__manual")


def test_ba_chart_rows_do_not_sum_duplicate_labels(tmp_path: Path) -> None:
    _write_research_run(
        tmp_path,
        plan="telecom_v3_development_r2x5_t3_v1",
        adapter="manual_lightgbm_te_v1_compat",
        run_id="20260729T070702417916Z_11111111",
        ba=0.8975,
    )
    _write_research_run(
        tmp_path,
        plan="telecom_v3_development_r2x5_t3_v1",
        adapter="manual_lightgbm_te_v1_compat",
        run_id="20260729T070802417916Z_22222222",
        ba=0.8975,
    )
    loaded = load_registry(PROJECT_ROOT)
    artifacts = discover_artifacts(tmp_path, loaded.readers["research_v2"])
    assert len(artifacts) >= 2
    rows = build_experiment_table_rows(artifacts, repo_root=tmp_path)
    ba_rows = [row for row in rows if row.get("_ba") == 0.8975]
    assert len(ba_rows) >= 2
    keys = [row["_chart_key"] for row in ba_rows]
    assert len(keys) == len(set(keys))
    assert all(float(row["_ba"]) == pytest.approx(0.8975) for row in ba_rows[:2])
    assert sum(float(row["_ba"]) for row in ba_rows[:2]) == pytest.approx(1.795)
    assert all("LightGBM" in str(row["_chart_label"]) for row in ba_rows[:2])


def test_persistence_matrix_and_forged_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "optuna_search_v1")
    at = _select(at, "Action", "validate")
    if "Model" in [item.label for item in at.selectbox]:
        at = _select(at, "Model", "CatBoost")
    if "Mode" in [item.label for item in at.selectbox]:
        at = _select(at, "Mode", "Development")
    config_key = widget_selection_key("optuna_search_v1", "config", "config")
    selected = _session_get(at, config_key)
    assert selected == CATBOOST_DEV

    at = at.run()
    assert _session_get(at, config_key) == selected

    at = _select(at, "Action", "run")
    assert _session_get(at, config_key) == selected
    at = _select(at, "Action", "validate")
    assert _session_get(at, config_key) == selected

    # Forge via logical seed path; widget options reject the forged value.
    loaded = load_registry(PROJECT_ROOT)
    pairs = control_panel_app._config_options(
        loaded, loaded.commands["optuna_search_v1"].allowed_config_globs
    )
    allowed = [path for path, _ in pairs]
    assert validate_against_allowed("configs/mlflow/local.yaml", allowed) is None
    session = {
        config_key: "configs/mlflow/local.yaml",
        LOGICAL_SELECTION_KEY: {
            "optuna_search_v1::config:config": "configs/mlflow/local.yaml",
        },
    }
    resolved = seed_widget_from_logical(
        session,
        operation="optuna_search_v1",
        role="config",
        name="config",
        widget_key=config_key,
        allowed=allowed,
    )
    assert resolved in allowed
    assert resolved != "configs/mlflow/local.yaml"
    assert str(resolved).startswith("configs/optuna/")


def test_mlflow_actions_only_use_local_yaml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "experiment_core_v2")
    at = _select(at, "Action", "run")
    if "Model" in [item.label for item in at.selectbox]:
        at = _select(at, "Model", "XGBoost")
    if "Mode" in [item.label for item in at.selectbox]:
        at = _select(at, "Mode", "Smoke")

    at = _select(at, "Operation", "mlflow_local_index")
    at = _select(at, "Action", "sync_dry_run")
    labels = [item.label for item in at.selectbox]
    assert "Model" not in labels
    assert "Mode" not in labels
    config = next(item for item in at.selectbox if item.label == "Config")
    assert config.value == "configs/mlflow/local.yaml"
    assert "local.yaml" in [str(option) for option in config.options]

    for text in (
        [str(item.value) for item in at.caption]
        + [str(item.value) for item in at.markdown]
        + [str(item.value) for item in at.code]
    ):
        assert "xgboost" not in text.lower()
        assert "catboost" not in text.lower()

    source = next(item for item in at.selectbox if item.label == "Source Type")
    assert "All sources" in list(source.options)
    assert enum_human_label("all") == "All sources"
    assert enum_human_label("research_v2") == "Experiment Core v2"
    assert enum_human_label("autogluon") == "AutoGluon"
    at = _select(at, "Source Type", "all")
    assert (
        _session_get(
            at, widget_selection_key("mlflow_local_index", "value", "source_type")
        )
        == "all"
    )

    built = build_command(
        load_registry(PROJECT_ROOT).commands,
        "mlflow_local_index",
        "sync_dry_run",
        {"config": "configs/mlflow/local.yaml", "source_type": "all"},
        repository_root=PROJECT_ROOT,
    )
    assert "configs/mlflow/local.yaml" in built.argv
    assert "all" in built.argv
    assert not any("xgboost" in part.lower() for part in built.argv)

    sync_all = build_command(
        load_registry(PROJECT_ROOT).commands,
        "mlflow_local_index",
        "sync_all",
        {"config": "configs/mlflow/local.yaml"},
        repository_root=PROJECT_ROOT,
    )
    assert sync_all.argv[-2:] == ("--source-type", "all")


def test_mlflow_post_index_uses_exact_artifact_and_keeps_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_root = tmp_path / "job"
    job_root.mkdir()
    artifact = tmp_path / "artifacts" / "research_v2" / "run_abc"
    artifact.mkdir(parents=True)
    stdout = f"Run directory: {artifact}\nStatus: completed\n"
    calls: list[dict[str, Any]] = []

    class FakeSummary:
        has_failures = False
        counts = {"created": 1, "unchanged": 0, "resumed": 0}

        def to_dict(self) -> dict[str, Any]:
            return {"counts": self.counts, "has_failures": False}

    def fake_sync(**kwargs: Any) -> FakeSummary:
        calls.append(kwargs)
        return FakeSummary()

    monkeypatch.setattr(
        "src.churn_ml.control_panel.mlflow_post_index._sync_one_artifact",
        fake_sync,
    )
    result = maybe_index_successful_job(
        job_root=job_root,
        command_id="experiment_core_v2",
        action_id="run",
        argv=["python", "scripts/run_research_v2.py", "--config", "x.yaml"],
        job_status="succeeded",
        stdout_text=stdout,
        repository_root=tmp_path,
    )
    assert result.status == "succeeded"
    assert result.artifact_path == "artifacts/research_v2/run_abc"
    assert calls and calls[0]["run_dir"] == artifact.resolve()
    assert calls[0]["source_type"] == "research_v2"

    def boom(**kwargs: Any) -> FakeSummary:
        del kwargs
        raise RuntimeError("mlflow down")

    monkeypatch.setattr(
        "src.churn_ml.control_panel.mlflow_post_index._sync_one_artifact",
        boom,
    )
    (job_root / "mlflow_index.json").unlink()
    failed = maybe_index_successful_job(
        job_root=job_root,
        command_id="experiment_core_v2",
        action_id="run",
        argv=["python", "scripts/run_research_v2.py"],
        job_status="succeeded",
        stdout_text=stdout,
        repository_root=tmp_path,
    )
    assert failed.status == "failed"
    assert failed.attempted is True
    assert source_type_for_job("experiment_core_v2", "run") == "research_v2"

    class UnchangedSummary:
        has_failures = False
        counts = {"created": 0, "unchanged": 1, "resumed": 0}

        def to_dict(self) -> dict[str, Any]:
            return {"counts": self.counts, "has_failures": False}

    monkeypatch.setattr(
        "src.churn_ml.control_panel.mlflow_post_index._sync_one_artifact",
        lambda **kwargs: UnchangedSummary(),
    )
    (job_root / "mlflow_index.json").unlink()
    again = maybe_index_successful_job(
        job_root=job_root,
        command_id="experiment_core_v2",
        action_id="run",
        argv=["python", "scripts/run_research_v2.py"],
        job_status="succeeded",
        stdout_text=stdout,
        repository_root=tmp_path,
    )
    assert again.status == "idempotent"
    skipped = maybe_index_successful_job(
        job_root=job_root,
        command_id="experiment_core_v2",
        action_id="run",
        argv=["python", "scripts/run_research_v2.py"],
        job_status="succeeded",
        stdout_text=stdout,
        repository_root=tmp_path,
    )
    assert skipped.status == "skipped_already_indexed"


def test_extract_run_directory_prefers_completion_line(tmp_path: Path) -> None:
    first = tmp_path / "artifacts" / "research_v2" / "a"
    second = tmp_path / "artifacts" / "research_v2" / "b"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    text = f"Run directory: {first}\nRun directory: {second}\n"
    assert extract_run_directory(text, repository_root=tmp_path) == second.resolve()


def test_autogluon_source_type_mapping() -> None:
    assert (
        source_type_for_job(
            "autogluon",
            "train",
            ["python", "scripts/run_autogluon.py", "train", "--config", "x.yaml"],
        )
        == "autogluon"
    )
