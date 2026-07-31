from __future__ import annotations

import json
from pathlib import Path

from src.churn_ml.control_panel.compare_workflow import (
    COMPARE_SCOPE_DESCRIPTIVE,
    COMPARE_SCOPE_DIFFERENT_DATASETS,
    COMPARE_SCOPE_LABELS,
    COMPARE_SCOPE_SAME_DATASET,
    comparison_scope_label,
    default_dataset_comparison_id_for_runs,
    default_dataset_comparison_output_root,
    default_same_dataset_comparison_id,
    default_same_dataset_output_root,
    exploratory_comparison_warning,
    filter_dataset_comparison_candidates,
    filter_same_dataset_candidates,
    is_descriptive_comparison_scope,
    list_completed_development_runs,
)
from src.churn_ml.control_panel.registry import load_registry
from src.churn_ml.control_panel.research_inventory import ResearchRunInventoryRow
from src.churn_ml.control_panel.workflow_navigation import (
    ADVANCED_ONLY_COMMAND_IDS,
    DATASET_COMPARISON_CONTRACT,
    advanced_command_ids,
    visible_command_ids,
    workflow_command_ids,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _row(
    *,
    relative_path: str,
    dataset_id: str,
    model_family: str = "LightGBM",
    model_config_id: str = "cfg-a",
    parent_dataset_id: str | None = None,
    target_dependency: str | None = "none",
    evaluation_mode: str = "Development",
    adapter_id: str = "manual_lightgbm_te_v1_compat",
    target_hash: str = "target-a",
    train_row_identity_hash: str = "row-a",
    evaluation_plan_hash: str = "plan-a",
    repeat_seeds: tuple[int, ...] = (0, 17),
    outer_fold_count: int = 5,
    threshold_selection_protocol: str = "grid_v1",
    feature_pipeline_id: str = "registered_prepared_passthrough_v1",
) -> ResearchRunInventoryRow:
    return ResearchRunInventoryRow(
        relative_path=relative_path,
        run_id=Path(relative_path).name,
        created_at_utc="2026-07-29T07:00:00+00:00",
        status="completed",
        dataset_id=dataset_id,
        parent_dataset_id=parent_dataset_id,
        target_dependency=target_dependency,
        n_features=10,
        model_family=model_family,
        adapter_id=adapter_id,
        model_config_id=model_config_id,
        source_config_path=None,
        source_config_hash=None,
        resolved_config_hash=None,
        evaluation_plan_id="plan",
        evaluation_plan_hash=evaluation_plan_hash,
        evaluation_mode=evaluation_mode,
        repeat_seeds=repeat_seeds,
        outer_fold_count=outer_fold_count,
        threshold_selection_protocol=threshold_selection_protocol,
        feature_pipeline_id=feature_pipeline_id,
        balanced_accuracy=0.89,
        sensitivity=0.9,
        specificity=0.8,
        roc_auc=0.95,
        average_precision=0.85,
        brier_score=0.05,
        threshold_median=0.1,
        competition_assets_accessed=False,
        authoritative_oof_path=None,
        target_hash=target_hash,
        train_row_identity_hash=train_row_identity_hash,
        candidate_adapter_hash=None,
        candidate_hash=model_config_id,
        train_content_hash=None,
        schema_hash=None,
        has_search_provenance=False,
        identity_complete=True,
    )


def test_three_comparison_scopes_exist() -> None:
    assert len(COMPARE_SCOPE_LABELS) == 3
    assert COMPARE_SCOPE_SAME_DATASET in dict(COMPARE_SCOPE_LABELS)
    assert COMPARE_SCOPE_DIFFERENT_DATASETS in dict(COMPARE_SCOPE_LABELS)
    assert COMPARE_SCOPE_DESCRIPTIVE in dict(COMPARE_SCOPE_LABELS)
    assert is_descriptive_comparison_scope(COMPARE_SCOPE_DESCRIPTIVE) is True
    assert comparison_scope_label(COMPARE_SCOPE_SAME_DATASET).startswith("Same dataset")


def test_workflow_still_five_steps_and_dataset_comparison_registered() -> None:
    registered = list(load_registry(PROJECT_ROOT).commands)
    assert workflow_command_ids() == (
        "experiment_core_v2",
        "paired_comparison",
        "optuna_search_v1",
        "blend_evaluation_v1",
        "final_deployment_v1",
    )
    assert "dataset_comparison_v1" in registered
    assert "dataset_comparison_v1" in advanced_command_ids()
    assert "dataset_comparison_v1" in ADVANCED_ONLY_COMMAND_IDS
    standard = visible_command_ids(registered)
    assert "dataset_comparison_v1" not in standard
    advanced = visible_command_ids(registered, include_legacy=True)
    assert "dataset_comparison_v1" in advanced
    assert DATASET_COMPARISON_CONTRACT == "dataset_comparison_v1_artifact"


def test_registry_loads_dataset_comparison_reader_and_command() -> None:
    loaded = load_registry(PROJECT_ROOT)
    command = loaded.commands["dataset_comparison_v1"]
    reader = loaded.readers["dataset_comparison_v1"]
    assert command.public_cli == "scripts/compare_research_v2_datasets.py"
    assert command.result_reader_id == "dataset_comparison_v1"
    assert reader.title == "Dataset Comparison v1 results"
    assert "artifacts/research_v2_dataset_comparisons" in reader.artifact_roots


def test_inventory_filters_for_dataset_comparison() -> None:
    baseline = _row(
        relative_path="artifacts/research_v2/plan/pipeline/run-a",
        dataset_id="v0_raw_minimal",
    )
    # Full plan/config/row hashes may differ across datasets; prefilter must allow.
    compatible = _row(
        relative_path="artifacts/research_v2/plan/pipeline/run-b",
        dataset_id="v1_missingness_summary",
        parent_dataset_id="v0_raw_minimal",
        evaluation_plan_hash="plan-b",
        train_row_identity_hash="row-b",
        model_config_id="cfg-b",
    )
    same_dataset = _row(
        relative_path="artifacts/research_v2/plan/pipeline/run-c",
        dataset_id="v0_raw_minimal",
    )
    different_model = _row(
        relative_path="artifacts/research_v2/plan/pipeline/run-d",
        dataset_id="v1_missingness_summary",
        model_family="XGBoost",
        adapter_id="xgboost_numeric_v1",
    )
    rows = [baseline, compatible, same_dataset, different_model]
    listed = list_completed_development_runs(rows)
    assert {item.relative_path for item in listed} == {item.relative_path for item in rows}
    same = filter_same_dataset_candidates(baseline, rows)
    assert [item.relative_path for item in same] == [same_dataset.relative_path]
    cross = filter_dataset_comparison_candidates(baseline, rows)
    assert [item.relative_path for item in cross] == [compatible.relative_path]


def test_default_ids_and_output_roots(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    for run_dir, dataset_id, adapter, search in (
        (baseline, "v0_raw_minimal", "manual_lightgbm_te_v1_compat", False),
        (candidate, "v0_raw_minimal", "xgboost_numeric_v1", False),
    ):
        run_dir.mkdir()
        (run_dir / "run_metadata.json").write_text(
            json.dumps({"candidate_adapter_id": adapter}),
            encoding="utf-8",
        )
        resolved: dict[str, object] = {}
        if search:
            resolved["search_provenance"] = {"study": "demo"}
        (run_dir / "resolved_config.yaml").write_text(
            json.dumps(resolved),
            encoding="utf-8",
        )
        (run_dir / "dataset_provenance.json").write_text(
            json.dumps({"dataset_id": dataset_id}),
            encoding="utf-8",
        )

    same_dataset_id = default_same_dataset_comparison_id(
        baseline,
        candidate,
        tmp_path,
    )
    assert same_dataset_id == "v0_raw_minimal__lightgbm__vs__xgboost"

    tuned = tmp_path / "tuned"
    tuned.mkdir()
    (tuned / "run_metadata.json").write_text(
        json.dumps({"candidate_adapter_id": "manual_lightgbm_te_v1_compat"}),
        encoding="utf-8",
    )
    (tuned / "resolved_config.yaml").write_text(
        json.dumps({"search_provenance": {"study": "x"}}),
        encoding="utf-8",
    )
    (tuned / "dataset_provenance.json").write_text(
        json.dumps({"dataset_id": "v0_raw_minimal"}),
        encoding="utf-8",
    )
    tuned_id = default_same_dataset_comparison_id(baseline, tuned, tmp_path)
    assert tuned_id == "v0_raw_minimal__lightgbm_base__vs__lightgbm_tuned"

    assert default_same_dataset_output_root() == "artifacts/research_v2_comparisons"
    assert (
        default_dataset_comparison_output_root()
        == "artifacts/research_v2_dataset_comparisons"
    )
    baseline_row = _row(
        relative_path="a",
        dataset_id="v0_raw_minimal",
    )
    candidate_row = _row(
        relative_path="b",
        dataset_id="v1_missingness_summary",
    )
    assert default_dataset_comparison_id_for_runs(baseline_row, candidate_row) == (
        "lightgbm__v0_raw_minimal__vs__v1_missingness_summary"
    )


def test_exploratory_warning_helper() -> None:
    assert exploratory_comparison_warning(
        baseline_dataset_id="v0_raw_minimal",
        candidate_dataset_id="v3_targeted_missingness",
        baseline_target_dependency="none",
        candidate_target_dependency="exploratory",
    )
    assert (
        exploratory_comparison_warning(
            baseline_dataset_id="v0_raw_minimal",
            candidate_dataset_id="v1_missingness_summary",
            baseline_target_dependency="none",
            candidate_target_dependency="none",
        )
        is None
    )
