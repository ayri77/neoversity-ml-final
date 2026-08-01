"""Cheap AppTest smoke for action-aware Compare placeholder routing."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import apps.experiment_control_panel as control_panel_app
from src.churn_ml.control_panel.compare_workflow import (
    COMPARE_SCOPE_DIFFERENT_DATASETS,
    COMPARE_SCOPE_SAME_DATASET,
    DatasetComparisonReadiness,
)
from src.churn_ml.control_panel.paired_comparison_readiness import (
    OfficialPairedReadiness,
)
from src.churn_ml.control_panel.research_inventory import ResearchRunInventoryRow
from src.churn_ml.control_panel.selection_state import ui_durable_key


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LIGHTGBM_V0 = "artifacts/research_v2/v0/lgbm/run-a"
XGBOOST_V0 = "artifacts/research_v2/v0/xgb/run-b"
LIGHTGBM_V1 = "artifacts/research_v2/v1/lgbm/run-c"


def _session_get(at: AppTest, key: str, default: object = None) -> object:
    try:
        return at.session_state[key]
    except KeyError:
        return default


def _row(
    *,
    relative_path: str,
    dataset_id: str,
    model_family: str,
    adapter_id: str,
    parent_dataset_id: str | None = None,
) -> ResearchRunInventoryRow:
    return ResearchRunInventoryRow(
        relative_path=relative_path,
        run_id=Path(relative_path).name,
        created_at_utc="2026-07-31T10:00:00+00:00",
        status="completed",
        dataset_id=dataset_id,
        parent_dataset_id=parent_dataset_id,
        target_dependency="none",
        n_features=205,
        model_family=model_family,
        adapter_id=adapter_id,
        model_config_id="cfg",
        source_config_path=None,
        source_config_hash=None,
        resolved_config_hash=None,
        evaluation_plan_id="plan",
        evaluation_plan_hash="plan-hash",
        evaluation_mode="Development",
        repeat_seeds=(0, 17),
        outer_fold_count=5,
        threshold_selection_protocol="grid_v1",
        feature_pipeline_id="registered_prepared_passthrough_v1",
        balanced_accuracy=0.89,
        sensitivity=0.9,
        specificity=0.8,
        roc_auc=0.95,
        average_precision=0.85,
        brier_score=0.05,
        threshold_median=0.1,
        competition_assets_accessed=False,
        authoritative_oof_path=None,
        target_hash="target-a",
        train_row_identity_hash="row-a",
        candidate_adapter_hash=None,
        candidate_hash="cfg",
        train_content_hash=None,
        schema_hash=None,
        has_search_provenance=False,
        identity_complete=True,
    )


def _inventory_rows() -> list[ResearchRunInventoryRow]:
    return [
        _row(
            relative_path=LIGHTGBM_V0,
            dataset_id="v0_raw_minimal",
            model_family="LightGBM",
            adapter_id="manual_lightgbm_te_v1_compat",
        ),
        _row(
            relative_path=XGBOOST_V0,
            dataset_id="v0_raw_minimal",
            model_family="XGBoost",
            adapter_id="xgboost_numeric_v1",
        ),
        _row(
            relative_path=LIGHTGBM_V1,
            dataset_id="v1_missingness_summary",
            model_family="LightGBM",
            adapter_id="manual_lightgbm_te_v1_compat",
            parent_dataset_id="v0_raw_minimal",
        ),
    ]


def _select(at: AppTest, label: str, value: str) -> AppTest:
    box = next(item for item in at.selectbox if item.label == label)
    return box.set_value(value).run()


def _load_run_page(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AppTest:
    for relative in (LIGHTGBM_V0, XGBOOST_V0, LIGHTGBM_V1):
        path = tmp_path / relative
        path.mkdir(parents=True, exist_ok=True)
        (path / "_SUCCESS").write_text("ok\n", encoding="utf-8")
        (path / "dataset_provenance.json").write_text(
            '{"dataset_id":"v0_raw_minimal"}',
            encoding="utf-8",
        )
        (path / "run_metadata.json").write_text(
            '{"candidate_adapter_id":"manual_lightgbm_te_v1_compat"}',
            encoding="utf-8",
        )
        (path / "resolved_config.yaml").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(control_panel_app, "REPOSITORY_ROOT", tmp_path)
    original_load = control_panel_app.load_registry

    def load_project_registry(root: Path):
        del root
        return original_load(PROJECT_ROOT)

    monkeypatch.setattr(control_panel_app, "load_registry", load_project_registry)
    monkeypatch.setattr(
        control_panel_app,
        "_cached_research_inventory_rows",
        lambda *_args, **_kwargs: [row.to_mapping() for row in _inventory_rows()],
    )
    monkeypatch.setattr(
        control_panel_app,
        "evaluate_official_paired_readiness",
        lambda **_kwargs: OfficialPairedReadiness(
            ready=True,
            compatible=True,
            reason_codes=(),
            field_paths=(),
            diagnostic=None,
            left_dataset_version="v0_raw_minimal",
            right_dataset_version="v0_raw_minimal",
        ),
    )
    monkeypatch.setattr(
        control_panel_app,
        "evaluate_dataset_comparison_readiness",
        lambda **_kwargs: DatasetComparisonReadiness(
            ready=True,
            compatible=True,
            summary=None,
            diagnostic=None,
        ),
    )
    monkeypatch.setattr(
        control_panel_app,
        "default_same_dataset_comparison_id",
        lambda *_args, **_kwargs: "v0_raw_minimal__lightgbm__vs__xgboost",
    )
    control_panel_app.registry.clear()

    def _page() -> None:
        import apps.experiment_control_panel as panel

        panel.run_page()

    return AppTest.from_function(_page, default_timeout=30).run()


def test_same_dataset_validate_builds_without_unknown_placeholders(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    at = _load_run_page(monkeypatch, tmp_path)
    assert not at.exception
    at.session_state["run-command"] = "paired_comparison"
    at.session_state[ui_durable_key("run", "command")] = "paired_comparison"
    at.session_state["run-action-paired_comparison"] = "validate"
    at.session_state[ui_durable_key("run", "action", "paired_comparison")] = "validate"
    at.session_state["compare-workflow-scope"] = COMPARE_SCOPE_SAME_DATASET
    at.session_state[ui_durable_key("run", "compare_scope")] = COMPARE_SCOPE_SAME_DATASET
    at.session_state["compare-baseline-run"] = LIGHTGBM_V0
    at.session_state[ui_durable_key("run", "compare_baseline_run")] = LIGHTGBM_V0
    at.session_state["compare-candidate-run"] = XGBOOST_V0
    at.session_state[ui_durable_key("run", "compare_candidate_run")] = XGBOOST_V0
    at = at.run()
    assert not at.exception
    assert not any(
        "Unknown placeholder values" in str(item.value) for item in at.error
    )
    assert any(
        "Ready for official paired comparison." in str(item.value) for item in at.success
    )
    start = next(item for item in at.button if item.label == "Start background job")
    assert start.disabled is False


def test_dataset_comparison_validate_builds_without_unknown_placeholders(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    at = _load_run_page(monkeypatch, tmp_path)
    assert not at.exception
    at.session_state["run-command"] = "paired_comparison"
    at.session_state[ui_durable_key("run", "command")] = "paired_comparison"
    at.session_state["run-action-paired_comparison"] = "validate"
    at.session_state[ui_durable_key("run", "action", "paired_comparison")] = "validate"
    at.session_state["compare-workflow-scope"] = COMPARE_SCOPE_DIFFERENT_DATASETS
    at.session_state[ui_durable_key("run", "compare_scope")] = (
        COMPARE_SCOPE_DIFFERENT_DATASETS
    )
    at.session_state["compare-baseline-run"] = LIGHTGBM_V0
    at.session_state[ui_durable_key("run", "compare_baseline_run")] = LIGHTGBM_V0
    at.session_state["compare-candidate-run"] = LIGHTGBM_V1
    at.session_state[ui_durable_key("run", "compare_candidate_run")] = LIGHTGBM_V1
    at = at.run()
    assert not at.exception
    assert not any(
        "Unknown placeholder values" in str(item.value) for item in at.error
    )
    assert any("Ready for dataset comparison." in str(item.value) for item in at.success)
    start = next(item for item in at.button if item.label == "Start background job")
    assert start.disabled is False


def test_manual_comparison_id_survives_action_switch_in_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    at = _load_run_page(monkeypatch, tmp_path)
    at.session_state["run-command"] = "paired_comparison"
    at.session_state[ui_durable_key("run", "command")] = "paired_comparison"
    at.session_state["run-action-paired_comparison"] = "run"
    at.session_state[ui_durable_key("run", "action", "paired_comparison")] = "run"
    at.session_state["compare-workflow-scope"] = COMPARE_SCOPE_SAME_DATASET
    at.session_state[ui_durable_key("run", "compare_scope")] = COMPARE_SCOPE_SAME_DATASET
    at.session_state["compare-baseline-run"] = LIGHTGBM_V0
    at.session_state[ui_durable_key("run", "compare_baseline_run")] = LIGHTGBM_V0
    at.session_state["compare-candidate-run"] = XGBOOST_V0
    at.session_state[ui_durable_key("run", "compare_candidate_run")] = XGBOOST_V0
    at = at.run()

    id_key = "run-compare-id-paired_comparison"
    at.session_state[id_key] = "my-manual-compare-id"
    at.session_state[f"{id_key}__manual"] = True
    at = at.run()
    assert _session_get(at, id_key) == "my-manual-compare-id"

    at.session_state["run-action-paired_comparison"] = "validate"
    at.session_state[ui_durable_key("run", "action", "paired_comparison")] = "validate"
    at = at.run()
    assert _session_get(at, id_key) == "my-manual-compare-id"
    assert not any(
        "Unknown placeholder values" in str(item.value) for item in at.error
    )

    at.session_state["run-action-paired_comparison"] = "run"
    at.session_state[ui_durable_key("run", "action", "paired_comparison")] = "run"
    at = at.run()
    assert _session_get(at, id_key) == "my-manual-compare-id"
    reset = next(
        (item for item in at.button if item.label == "Reset to suggested ID"),
        None,
    )
    assert reset is not None
    reset.click()
    at = at.run()
    assert _session_get(at, id_key) == _session_get(at, f"{id_key}__suggested")
    assert not _session_get(at, f"{id_key}__manual")
