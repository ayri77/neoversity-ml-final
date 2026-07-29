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
from src.churn_ml.control_panel.jobs import JobManager
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
from tests.test_control_panel_security import StartSpy, _apptest_run_page, _run_page, _select


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATBOOST_DEV = "configs/optuna/catboost_numeric_v1_development.yaml"
XGBOOST_SMOKE = "configs/optuna/xgboost_numeric_v1_smoke.yaml"


def _session_get(at: AppTest, key: str, default: object = None) -> object:
    try:
        inner = getattr(
            getattr(at.session_state, "_state", None), "_new_session_state", None
        )
        if isinstance(inner, dict) and key in inner:
            return inner[key]
        return at.session_state[key]
    except Exception:
        return default


def _make_run_apptest() -> AppTest:
    return AppTest.from_function(_apptest_run_page, default_timeout=10)


def _rerun_with_durable_only(at: AppTest, make_apptest) -> AppTest:
    """Start a fresh page render with only durable backing state (simulate navigation)."""
    from src.churn_ml.control_panel.selection_state import snapshot_durable_session

    snapshot = snapshot_durable_session(at.session_state)
    rebuilt = make_apptest()
    for key, value in snapshot.items():
        rebuilt.session_state[key] = value
    return rebuilt.run()


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
    labels = [item.label for item in at.selectbox]
    if "Source" in labels:
        at = _select(at, "Source", "Canonical config")
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
    assert "mlflow" not in str(resolved)


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


def test_valid_widget_config_wins_over_stale_logical_example(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug 1: newly selected working.yaml must not be overwritten by .example logical."""
    example = (
        "configs/blend_evaluation/lightgbm_xgboost_blend_v1.example.yaml"
    )
    working = "configs/blend_evaluation/lightgbm_xgboost_blend_v1.yaml"
    cfg_dir = tmp_path / "configs" / "blend_evaluation"
    cfg_dir.mkdir(parents=True)
    for relative in (example, working):
        target = tmp_path / relative
        target.write_text(
            "schema_version: 1\n"
            "experiment:\n"
            "  id: lightgbm_xgboost_blend_v1\n"
            "components: {}\n"
            "blend: {}\n"
            "deployment_parameters: {}\n"
            "artifacts:\n"
            "  root: artifacts/blend_evaluations\n"
            "  evaluation_id: lightgbm_xgboost_blend_v1\n",
            encoding="utf-8",
        )

    allowed = [example, working]
    config_key = widget_selection_key("blend_evaluation_v1", "config", "config")
    session: dict[str, Any] = {
        config_key: working,
        LOGICAL_SELECTION_KEY: {
            "blend_evaluation_v1::config:config": example,
        },
    }
    resolved = seed_widget_from_logical(
        session,
        operation="blend_evaluation_v1",
        role="config",
        name="config",
        widget_key=config_key,
        allowed=allowed,
    )
    assert resolved == working
    assert session[config_key] == working
    assert (
        session[LOGICAL_SELECTION_KEY]["blend_evaluation_v1::config:config"] == working
    )

    monkeypatch.setattr(control_panel_app, "REPOSITORY_ROOT", tmp_path)
    original_load = control_panel_app.load_registry

    def load_project_registry(root: Path):
        del root
        return original_load(PROJECT_ROOT)

    monkeypatch.setattr(control_panel_app, "load_registry", load_project_registry)
    control_panel_app.registry.clear()

    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "blend_evaluation_v1")
    at = _select(at, "Action", "validate")
    existing_store = dict(_session_get(at, LOGICAL_SELECTION_KEY, {}) or {})
    existing_store["blend_evaluation_v1::config:config"] = example
    at.session_state[LOGICAL_SELECTION_KEY] = existing_store
    at.session_state[config_key] = working
    at = at.run()
    assert _session_get(at, config_key) == working
    store = _session_get(at, LOGICAL_SELECTION_KEY, {})
    assert isinstance(store, dict)
    assert store.get("blend_evaluation_v1::config:config") == working

    captions = [str(item.value) for item in at.caption]
    assert any(working in text for text in captions)
    codes = [str(item.value) for item in at.code]
    assert any(working in text for text in codes)

    built = build_command(
        load_registry(PROJECT_ROOT).commands,
        "blend_evaluation_v1",
        "validate",
        {"config": working},
        repository_root=tmp_path,
    )
    assert working in built.argv
    assert example not in built.argv

    at = _select(at, "Action", "run")
    assert _session_get(at, config_key) == working
    at = _select(at, "Action", "validate")
    assert _session_get(at, config_key) == working
    store = _session_get(at, LOGICAL_SELECTION_KEY, {})
    assert isinstance(store, dict)
    assert store.get("blend_evaluation_v1::config:config") == working


def test_run_state_survives_widget_key_cleanup_navigation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bug 2: Operation/Action/Config/enum survive simulated page navigation."""
    from src.churn_ml.control_panel.selection_state import ui_durable_key

    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "mlflow_local_index")
    at = _select(at, "Action", "sync_dry_run")
    at = _select(at, "Source Type", "autogluon")
    config_key = widget_selection_key("mlflow_local_index", "config", "config")
    enum_key = widget_selection_key("mlflow_local_index", "value", "source_type")
    assert _session_get(at, config_key) == "configs/mlflow/local.yaml"
    assert _session_get(at, enum_key) == "autogluon"

    at = _rerun_with_durable_only(at, _make_run_apptest)
    assert next(item for item in at.selectbox if item.label == "Operation").value == (
        "mlflow_local_index"
    )
    assert next(item for item in at.selectbox if item.label == "Action").value == (
        "sync_dry_run"
    )
    assert _session_get(at, config_key) == "configs/mlflow/local.yaml"
    assert _session_get(at, enum_key) == "autogluon"
    store = _session_get(at, LOGICAL_SELECTION_KEY, {})
    assert isinstance(store, dict)
    assert store.get(ui_durable_key("run", "command")) == "mlflow_local_index"
    assert store.get("mlflow_local_index::value:source_type") == "autogluon"


def test_results_filters_and_compare_survive_navigation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.churn_ml.control_panel.selection_state import ui_durable_key

    _write_research_run(
        tmp_path,
        plan="telecom_v3_development_r2x5_t3_v1",
        adapter="manual_lightgbm_te_v1_compat",
        run_id="20260729T070000000000Z_aaaaaaaa",
        ba=0.9,
    )
    _write_research_run(
        tmp_path,
        plan="telecom_v3_development_r2x5_t3_v1",
        adapter="xgboost_numeric_v1",
        run_id="20260729T070100000000Z_bbbbbbbb",
        ba=0.91,
    )
    monkeypatch.setattr(control_panel_app, "REPOSITORY_ROOT", tmp_path)
    original_load = control_panel_app.load_registry

    def load_project_registry(root: Path):
        del root
        return original_load(PROJECT_ROOT)

    monkeypatch.setattr(control_panel_app, "load_registry", load_project_registry)
    control_panel_app.registry.clear()

    def _results_page() -> None:
        import apps.experiment_control_panel as panel

        panel.results_page()

    def _make_results_apptest() -> AppTest:
        return AppTest.from_function(_results_page, default_timeout=15)

    at = _make_results_apptest().run()
    assert not at.exception
    at = _select(at, "Artifact type", "research_v2")
    model_key = "results-exp-filter-model-research_v2"
    search_key = "results-exp-filter-search-research_v2"
    if "Model" in [item.label for item in at.selectbox]:
        model_box = next(item for item in at.selectbox if item.label == "Model")
        if "LightGBM" in list(model_box.options):
            at = _select(at, "Model", "LightGBM")
    search_inputs = [item for item in at.text_input if item.label.startswith("Search")]
    if search_inputs:
        search_inputs[0].set_value("lightgbm")
        at = at.run()

    selected_model = _session_get(at, model_key)
    selected_search = _session_get(at, search_key)
    at = _rerun_with_durable_only(at, _make_results_apptest)
    if selected_model not in (None, "", "All"):
        assert _session_get(at, model_key) == selected_model
    if selected_search not in (None, ""):
        assert _session_get(at, search_key) == selected_search
    store = _session_get(at, LOGICAL_SELECTION_KEY, {})
    assert isinstance(store, dict)
    assert store.get(ui_durable_key("results", "reader", "results-experiments-reader"))


def test_selected_job_survives_navigation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.churn_ml.control_panel.selection_state import (
        set_durable_value,
        ui_durable_key,
    )
    from tests.test_control_panel_jobs import FakeBackend

    backend = FakeBackend()
    manager = JobManager(
        tmp_path / "jobs",
        working_directory=tmp_path,
        commands=load_registry(PROJECT_ROOT).commands,
        backend=backend,
    )
    first = manager.start(
        argv=["safe", "a"],
        redacted_argv=["safe", "a"],
        command_id="experiment_core_v2",
        action_id="validate",
        references={"config": "configs/a.yaml"},
    )
    second = manager.start(
        argv=["safe", "b"],
        redacted_argv=["safe", "b"],
        command_id="experiment_core_v2",
        action_id="validate",
        references={"config": "configs/b.yaml"},
    )
    monkeypatch.setattr(control_panel_app, "job_manager", lambda _loaded: manager)

    def _jobs_page() -> None:
        import apps.experiment_control_panel as panel

        panel.jobs_page()

    def _make_jobs_apptest() -> AppTest:
        return AppTest.from_function(_jobs_page, default_timeout=15)

    at = _make_jobs_apptest().run()
    assert not at.exception
    job_ids = [first.job_id, second.job_id]
    # Prefer the non-default job so restoration is observable.
    job_box = next(item for item in at.selectbox if item.label == "Job")
    default_job = job_box.value
    target_job = next(job_id for job_id in job_ids if job_id != default_job)
    at.session_state["jobs-selected"] = target_job
    set_durable_value(
        at.session_state,
        ui_durable_key("jobs", "selected"),
        target_job,
    )
    at = at.run()
    assert _session_get(at, "jobs-selected") == target_job
    at = _rerun_with_durable_only(at, _make_jobs_apptest)
    assert _session_get(at, "jobs-selected") == target_job
    store = _session_get(at, LOGICAL_SELECTION_KEY, {})
    assert isinstance(store, dict)
    assert store.get(ui_durable_key("jobs", "selected")) == target_job


def test_stale_durable_options_fall_back_safely() -> None:
    from src.churn_ml.control_panel.selection_state import (
        sync_widget_with_durable,
        ui_durable_key,
    )

    session: dict[str, Any] = {
        LOGICAL_SELECTION_KEY: {
            ui_durable_key("run", "command"): "removed_operation",
            "experiment_core_v2::config:config": "configs/gone.yaml",
        }
    }
    resolved_command = sync_widget_with_durable(
        session,
        widget_key="run-command",
        durable_key=ui_durable_key("run", "command"),
        allowed=["experiment_core_v2", "optuna_search_v1"],
        default="experiment_core_v2",
    )
    assert resolved_command == "experiment_core_v2"
    config_key = widget_selection_key("experiment_core_v2", "config", "config")
    allowed = [
        "configs/research_v2/manual_lightgbm_te_v1_compat_development.yaml",
        "configs/research_v2/xgboost_numeric_v1_development.yaml",
    ]
    resolved_config = seed_widget_from_logical(
        session,
        operation="experiment_core_v2",
        role="config",
        name="config",
        widget_key=config_key,
        allowed=allowed,
    )
    assert resolved_config == allowed[0]


def test_safety_confirmations_are_not_restored_from_durable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "experiment_core_v2")
    at = _select(at, "Action", "run")
    confirm = next(item for item in at.checkbox if item.label.startswith("I confirm"))
    confirm.check()
    at = at.run()
    assert any(item.value for item in at.checkbox if item.label.startswith("I confirm"))
    at = _rerun_with_durable_only(at, _make_run_apptest)
    assert all(
        not item.value for item in at.checkbox if item.label.startswith("I confirm")
    )
