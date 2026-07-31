"""Results workspace tabs, experiment charts, labels, and selection persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import apps.experiment_control_panel as control_panel_app
from src.churn_ml.control_panel.artifacts import (
    build_experiment_table_rows,
    comparison_rows,
    discover_artifacts,
    display_compatibility_summary,
)
from src.churn_ml.control_panel.presentation import (
    build_cascade_options,
    cascade_filter_configs,
    is_raw_run_id,
    job_primary_label,
    parse_run_id_timestamp,
    readable_path_label,
)
from src.churn_ml.control_panel.registry import load_registry
from src.churn_ml.control_panel.selection_state import (
    LOGICAL_SELECTION_KEY,
    remember_widget_selection,
    seed_widget_from_logical,
    validate_against_allowed,
    widget_selection_key,
)
from tests.test_control_panel_security import StartSpy, _run_page, _select


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_research_run(
    root: Path,
    *,
    plan: str,
    adapter: str,
    run_id: str,
    ba: float,
    sensitivity: float = 0.9,
    specificity: float = 0.8,
    roc_auc: float = 0.95,
    average_precision: float = 0.85,
    brier: float = 0.05,
    plan_hash: str = "planhash",
    dataset_sha: str = "datasetsha",
    fold_payload: bytes = b"fold-a",
) -> Path:
    artifact = (
        root
        / "artifacts"
        / "research_v2"
        / f"{plan}_{plan_hash}"
        / f"pipeline__{adapter}"
        / run_id
    )
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "_SUCCESS").write_text("", encoding="utf-8")
    (artifact / "run_metadata.json").write_text(
        json.dumps(
            {
                "experiment_id": f"{adapter}_development",
                "plan_id": plan,
                "candidate_adapter_id": adapter,
                "started_at_utc": parse_run_id_timestamp(run_id)
                or "2026-07-29T07:07:02+00:00",
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )
    (artifact / "metrics").mkdir(exist_ok=True)
    (artifact / "metrics" / "aggregate.json").write_text(
        json.dumps(
            {
                "primary_metric": "balanced_accuracy",
                "metrics": {
                    "balanced_accuracy": {"mean": ba},
                    "sensitivity": {"mean": sensitivity},
                    "specificity": {"mean": specificity},
                    "roc_auc": {"mean": roc_auc},
                    "average_precision": {"mean": average_precision},
                    "brier_score": {"mean": brier},
                },
            }
        ),
        encoding="utf-8",
    )
    (artifact / "thresholds").mkdir(exist_ok=True)
    (artifact / "thresholds" / "threshold_summary.json").write_text(
        json.dumps({"median": 0.11, "interquartile_range": 0.03}),
        encoding="utf-8",
    )
    (artifact / "identities").mkdir(exist_ok=True)
    (artifact / "identities" / "evaluation_plan.json").write_text(
        json.dumps({"sha256": plan_hash, "canonical": {"plan_id": plan}}),
        encoding="utf-8",
    )
    (artifact / "dataset_fingerprints.json").write_text(
        json.dumps(
            {
                "dataset_version": "v3",
                "files": {"train_features": {"sha256": dataset_sha}},
                "row_position_identity": {"sha256": "rowsha"},
            }
        ),
        encoding="utf-8",
    )
    (artifact / "dataset_provenance.json").write_text(
        json.dumps(
            {
                "dataset_id": "v3",
                "parent_dataset_id": "v2",
                "hypothesis": "fixture",
                "n_features": 42,
                "schema_hash": "a" * 64,
                "train_content_hash": "b" * 64,
                "target_hash": "c" * 64,
                "target_dependency": "none",
                "train_row_identity_hash": "d" * 64,
                "registry_schema_version": "dataset_package_v1",
            }
        ),
        encoding="utf-8",
    )
    splits = artifact / "splits"
    splits.mkdir(exist_ok=True)
    (splits / "outer_assignments.parquet").write_bytes(fold_payload)
    (splits / "threshold_selection_assignments.parquet").write_bytes(fold_payload)
    return artifact


def _session_get(at: AppTest, key: str, default: object = None) -> object:
    try:
        return at.session_state[key]
    except Exception:
        return default


def _apptest_results_page() -> None:
    import apps.experiment_control_panel as panel

    panel.results_page()


def _load_results(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AppTest:
    monkeypatch.setattr(control_panel_app, "REPOSITORY_ROOT", tmp_path)
    original_load = control_panel_app.load_registry

    def load_project_registry(root: Path):
        del root
        return original_load(PROJECT_ROOT)

    monkeypatch.setattr(control_panel_app, "load_registry", load_project_registry)
    control_panel_app.registry.clear()
    return AppTest.from_function(_apptest_results_page, default_timeout=15).run()


def test_results_page_has_three_tabs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_research_run(
        tmp_path,
        plan="telecom_v3_development_r2x5_t3_v1",
        adapter="manual_lightgbm_te_v1_compat",
        run_id="20260729T070702417916Z_38bbefe2",
        ba=0.897453,
    )
    at = _load_results(monkeypatch, tmp_path)
    assert not at.exception
    tab_labels = [tab.label for tab in at.tabs]
    assert tab_labels == ["Experiments", "Inspect result", "Compare experiments"]


def test_experiment_table_has_required_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_research_run(
        tmp_path,
        plan="telecom_v3_development_r2x5_t3_v1",
        adapter="manual_lightgbm_te_v1_compat",
        run_id="20260729T070702417916Z_38bbefe2",
        ba=0.897453,
    )
    loaded = load_registry(PROJECT_ROOT)
    artifacts = discover_artifacts(tmp_path, loaded.readers["research_v2"])
    rows = build_experiment_table_rows(artifacts, repo_root=tmp_path)
    assert rows
    required = {
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
    }
    assert required.issubset(rows[0].keys())
    assert rows[0]["Dataset"] == "v3"
    assert rows[0]["Parent dataset"] == "v2"
    assert rows[0]["Target dependency"] == "none"
    assert rows[0]["Features"] == "42"
    assert rows[0]["Model"] == "LightGBM"
    assert rows[0]["Mode"] == "Development"
    assert rows[0]["Balanced Accuracy"] == pytest.approx(0.897453)
    assert "v3" in str(rows[0]["_chart_label"])
    assert "Artifact path" in rows[0]
    assert list(rows[0].keys()).index("Artifact path") >= list(rows[0].keys()).index(
        "Status"
    )


def test_experiment_charts_render_and_empty_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_research_run(
        tmp_path,
        plan="telecom_v3_development_r2x5_t3_v1",
        adapter="manual_lightgbm_te_v1_compat",
        run_id="20260729T070702417916Z_38bbefe2",
        ba=0.897453,
    )
    _write_research_run(
        tmp_path,
        plan="telecom_v3_development_r2x5_t3_v1",
        adapter="xgboost_numeric_v1",
        run_id="20260729T064407009547Z_55c26d5e",
        ba=0.897987,
    )
    at = _load_results(monkeypatch, tmp_path)
    assert not at.exception
    titles = []
    for attr in ("plotly_chart", "caption", "subheader", "markdown"):
        for item in getattr(at, attr, []):
            titles.append(str(getattr(item, "value", item)))
    assert any("Balanced Accuracy" in text for text in titles) or len(at.dataframe) >= 1


def test_inspect_cascade_filters_model_mode(tmp_path: Path) -> None:
    paths = [
        str(
            Path("artifacts/research_v2")
            / "telecom_v3_development_r2x5_t3_v1_plan"
            / "pipeline__manual_lightgbm_te_v1_compat"
            / "20260729T070702417916Z_38bbefe2"
        ),
        str(
            Path("artifacts/research_v2")
            / "telecom_v3_development_r2x5_t3_v1_plan"
            / "pipeline__xgboost_numeric_v1"
            / "20260729T064407009547Z_55c26d5e"
        ),
        str(
            Path("artifacts/research_v2")
            / "telecom_v3_smoke_r1x3_t2_v1_plan"
            / "pipeline__catboost_numeric_v1"
            / "20260729T050000000000Z_aaaaaaaa"
        ),
    ]
    # Create real files so metadata parsing can read adapters from parent names.
    for path in paths:
        full = tmp_path / path
        full.mkdir(parents=True, exist_ok=True)
        (full / "_SUCCESS").write_text("", encoding="utf-8")
        (full / "run_metadata.json").write_text(
            json.dumps(
                {
                    "candidate_adapter_id": path.split("__")[1].split("/")[0],
                    "plan_id": "telecom_v3_development_r2x5_t3_v1",
                    "started_at_utc": "2026-07-29T07:07:02+00:00",
                }
            ),
            encoding="utf-8",
        )
        (full / "metrics").mkdir()
        (full / "metrics" / "aggregate.json").write_text(
            json.dumps({"metrics": {"balanced_accuracy": {"mean": 0.9}}}),
            encoding="utf-8",
        )
    rels = [Path(p).as_posix() for p in paths]
    opts = build_cascade_options(rels, tmp_path)
    matched = cascade_filter_configs(opts, model="LightGBM", mode="Development")
    assert matched
    assert all(opt.model_family == "LightGBM" for opt in matched)
    assert not any(opt.model_family == "XGBoost" for opt in matched)
    assert not any(opt.mode == "Smoke" for opt in matched)


def test_compare_left_right_cascades_filter_independently(tmp_path: Path) -> None:
    _write_research_run(
        tmp_path,
        plan="telecom_v3_development_r2x5_t3_v1",
        adapter="manual_lightgbm_te_v1_compat",
        run_id="20260729T070702417916Z_38bbefe2",
        ba=0.897453,
    )
    _write_research_run(
        tmp_path,
        plan="telecom_v3_development_r2x5_t3_v1",
        adapter="xgboost_numeric_v1",
        run_id="20260729T064407009547Z_55c26d5e",
        ba=0.897987,
    )
    loaded = load_registry(PROJECT_ROOT)
    artifacts = discover_artifacts(tmp_path, loaded.readers["research_v2"])
    paths = [item.relative_path for item in artifacts]
    opts = build_cascade_options(paths, tmp_path)
    left = cascade_filter_configs(opts, model="LightGBM", mode="Development")
    right = cascade_filter_configs(opts, model="XGBoost", mode="Development")
    assert left and right
    assert {opt.path for opt in left}.isdisjoint({opt.path for opt in right})


def test_raw_run_ids_are_not_primary_labels(tmp_path: Path) -> None:
    artifact = _write_research_run(
        tmp_path,
        plan="telecom_v3_development_r2x5_t3_v1",
        adapter="xgboost_numeric_v1",
        run_id="20260729T064407009547Z_55c26d5e",
        ba=0.897987,
    )
    relative = artifact.relative_to(tmp_path).as_posix()
    label = readable_path_label(relative, tmp_path)
    assert "20260729T064407009547Z_55c26d5e" not in label
    assert is_raw_run_id("20260729T064407009547Z_55c26d5e")
    assert "XGBoost" in label
    assert "BA" in label
    opts = build_cascade_options([relative], tmp_path)
    assert not is_raw_run_id(opts[0].display_label.split(" · ")[0])


def test_readable_labels_include_datetime_and_ba(tmp_path: Path) -> None:
    artifact = _write_research_run(
        tmp_path,
        plan="telecom_v3_development_r2x5_t3_v1",
        adapter="manual_lightgbm_te_v1_compat",
        run_id="20260729T070702417916Z_38bbefe2",
        ba=0.897453,
    )
    relative = artifact.relative_to(tmp_path).as_posix()
    label = readable_path_label(relative, tmp_path)
    assert "29 Jul 2026" in label
    assert "07:07" in label
    assert "0.897453" in label


def test_compare_table_contains_delta(tmp_path: Path) -> None:
    left = _write_research_run(
        tmp_path,
        plan="telecom_v3_development_r2x5_t3_v1",
        adapter="manual_lightgbm_te_v1_compat",
        run_id="20260729T070702417916Z_38bbefe2",
        ba=0.897453,
    )
    right = _write_research_run(
        tmp_path,
        plan="telecom_v3_development_r2x5_t3_v1",
        adapter="xgboost_numeric_v1",
        run_id="20260729T064407009547Z_55c26d5e",
        ba=0.897987,
    )
    loaded = load_registry(PROJECT_ROOT)
    artifacts = {
        item.root.name: item
        for item in discover_artifacts(tmp_path, loaded.readers["research_v2"])
    }
    rows = comparison_rows(
        artifacts[left.name],
        artifacts[right.name],
        loaded.readers["research_v2"].compare_fields,
    )
    assert any(row["Metric"] == "Balanced Accuracy mean" for row in rows)
    ba_row = next(row for row in rows if row["Metric"] == "Balanced Accuracy mean")
    assert "Delta (Right - Left)" in ba_row
    assert ba_row["Delta (Right - Left)"] == pytest.approx(0.897987 - 0.897453)
    assert any(row["Metric"] == "Brier score" for row in rows)


def test_compatibility_shown_before_paired_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_research_run(
        tmp_path,
        plan="telecom_v3_development_r2x5_t3_v1",
        adapter="manual_lightgbm_te_v1_compat",
        run_id="20260729T070702417916Z_38bbefe2",
        ba=0.897453,
    )
    _write_research_run(
        tmp_path,
        plan="telecom_v3_development_r2x5_t3_v1",
        adapter="xgboost_numeric_v1",
        run_id="20260729T064407009547Z_55c26d5e",
        ba=0.897987,
    )
    loaded = load_registry(PROJECT_ROOT)
    artifacts = discover_artifacts(tmp_path, loaded.readers["research_v2"])
    summary = display_compatibility_summary(artifacts[0], artifacts[1])
    assert summary["same_evaluation_plan"] is True
    assert summary["same_dataset_fingerprint"] is True
    assert summary["same_fold_assignments"] is True
    assert summary["compatible"] is True
    assert summary["status"] == "compatible"
    assert summary["left_dataset_id"] == "v3"
    assert summary["right_dataset_id"] == "v3"
    assert summary["descriptive_only"] is False

    at = _load_results(monkeypatch, tmp_path)
    assert not at.exception
    subheaders = [str(getattr(item, "value", item)) for item in at.subheader]
    assert (
        any("Compatibility" in text for text in subheaders)
        or any("compatible" in str(item).lower() for item in at.success)
        or any(
            button.label == "Prepare Paired Comparison action" for button in at.button
        )
    )
    assert any(
        button.label == "Prepare Paired Comparison action" for button in at.button
    )
    prepare = next(
        button
        for button in at.button
        if button.label == "Prepare Paired Comparison action"
    )
    assert prepare.disabled is False


def test_cross_dataset_compare_is_descriptive_only(tmp_path: Path) -> None:
    left = _write_research_run(
        tmp_path,
        plan="telecom_v3_development_r2x5_t3_v1",
        adapter="manual_lightgbm_te_v1_compat",
        run_id="20260729T070702417916Z_38bbefe2",
        ba=0.897453,
        dataset_sha="dataset-a",
    )
    right = _write_research_run(
        tmp_path,
        plan="telecom_v3_development_r2x5_t3_v1",
        adapter="xgboost_numeric_v1",
        run_id="20260729T064407009547Z_55c26d5e",
        ba=0.897987,
        dataset_sha="dataset-b",
    )
    (right / "dataset_provenance.json").write_text(
        json.dumps(
            {
                "dataset_id": "v7_compact_zero_indicators",
                "parent_dataset_id": "v4",
                "n_features": 10,
                "target_dependency": "none",
                "schema_hash": "a" * 64,
                "train_content_hash": "b" * 64,
                "target_hash": "c" * 64,
                "train_row_identity_hash": "d" * 64,
                "registry_schema_version": "dataset_package_v1",
            }
        ),
        encoding="utf-8",
    )
    (right / "dataset_fingerprints.json").write_text(
        json.dumps(
            {
                "dataset_version": "v7_compact_zero_indicators",
                "files": {"train_features": {"sha256": "dataset-b"}},
                "row_position_identity": {"sha256": "rowsha"},
            }
        ),
        encoding="utf-8",
    )
    loaded = load_registry(PROJECT_ROOT)
    artifacts = discover_artifacts(tmp_path, loaded.readers["research_v2"])
    by_path = {item.root: item for item in artifacts}
    summary = display_compatibility_summary(by_path[left], by_path[right])
    assert summary["same_dataset_fingerprint"] is False
    assert summary["compatible"] is False
    assert summary["descriptive_only"] is True
    assert summary["left_dataset_id"] == "v3"
    assert summary["right_dataset_id"] == "v7_compact_zero_indicators"


def test_run_jobs_run_restores_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "experiment_core_v2")
    at = _select(at, "Action", "run")
    config_key = widget_selection_key("experiment_core_v2", "config", "config")
    selected = _session_get(at, config_key)
    assert selected
    source = _session_get(at, f"{config_key}__src")
    model = _session_get(at, f"{config_key}__mdl")
    mode = _session_get(at, f"{config_key}__mode")
    at = at.run()
    assert _session_get(at, config_key) == selected
    assert _session_get(at, f"{config_key}__src") == source
    assert _session_get(at, f"{config_key}__mdl") == model
    assert _session_get(at, f"{config_key}__mode") == mode


def test_validate_run_preserves_same_valid_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "experiment_core_v2")
    at = _select(at, "Action", "validate")
    config_key = widget_selection_key("experiment_core_v2", "config", "config")
    if "Source" in [item.label for item in at.selectbox]:
        at = _select(at, "Source", "Canonical config")
    if "Model" in [item.label for item in at.selectbox]:
        at = _select(at, "Model", "LightGBM")
    if "Mode" in [item.label for item in at.selectbox]:
        at = _select(at, "Mode", "Development")
    selected = _session_get(at, config_key)
    assert selected
    assert "lightgbm" in str(selected).lower()
    at = _select(at, "Action", "run")
    assert _session_get(at, config_key) == selected
    store = _session_get(at, LOGICAL_SELECTION_KEY, {})
    assert isinstance(store, dict)
    assert store.get("experiment_core_v2::config:config") == selected


def test_run_validate_preserves_same_valid_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "experiment_core_v2")
    at = _select(at, "Action", "run")
    config_key = widget_selection_key("experiment_core_v2", "config", "config")
    if "Model" in [item.label for item in at.selectbox]:
        at = _select(at, "Model", "XGBoost")
    if "Mode" in [item.label for item in at.selectbox]:
        at = _select(at, "Mode", "Development")
    selected = _session_get(at, config_key)
    assert selected
    at = _select(at, "Action", "validate")
    assert _session_get(at, config_key) == selected


def test_stale_forged_paths_are_rejected() -> None:
    session: dict = {}
    allowed = [
        "configs/research_v2/manual_lightgbm_te_v1_compat_development.yaml",
        "configs/research_v2/xgboost_numeric_v1_development.yaml",
    ]
    forged = "configs/mlflow/local.yaml"
    assert validate_against_allowed(forged, allowed) is None
    session[widget_selection_key("experiment_core_v2", "config", "config")] = forged
    session[LOGICAL_SELECTION_KEY] = {
        "experiment_core_v2::config:config": forged,
    }
    resolved = seed_widget_from_logical(
        session,
        operation="experiment_core_v2",
        role="config",
        name="config",
        widget_key=widget_selection_key("experiment_core_v2", "config", "config"),
        allowed=allowed,
    )
    assert resolved == allowed[0]
    assert (
        session[widget_selection_key("experiment_core_v2", "config", "config")]
        in allowed
    )
    remembered = remember_widget_selection(
        session,
        operation="experiment_core_v2",
        role="config",
        name="config",
        value=forged,
        allowed=allowed,
    )
    assert remembered is None


def test_jobs_label_includes_date_time() -> None:
    job = {
        "command_id": "experiment_core_v2",
        "action_id": "run",
        "references": {
            "config": "configs/research_v2/manual_lightgbm_te_v1_compat_development.yaml"
        },
        "created_at_utc": "2026-07-29T10:00:00Z",
    }
    label = job_primary_label(job)
    assert "LightGBM" in label
    assert "Development" in label
    assert "29 Jul 2026" in label
    assert "10:00:00" in label
