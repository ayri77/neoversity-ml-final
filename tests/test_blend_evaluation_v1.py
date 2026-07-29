from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from src.churn_ml.blend_evaluation_v1 import (
    BlendEvaluationError,
    apply_fixed_blend_probabilities,
    build_weight_grid,
    run_cross_fit_blend_evaluation,
    select_blend_weight_and_threshold,
    validate_blend_inputs,
)
from src.churn_ml.blend_evaluation_v1_cli import parse_args
from src.churn_ml.blend_evaluation_v1_deployment import (
    VARIANT_SPECS,
    generate_submission_variants,
)
from src.churn_ml.control_panel.presentation import (
    normalize_model_family,
    normalize_source_kind,
)
from src.churn_ml.paired_comparison import CompletedResearchV2Run
from src.churn_ml.research_data import canonical_sha256
from src.churn_ml.research_protocol import (
    build_repeat_metrics,
    select_balanced_accuracy_threshold,
)
from src.churn_ml.research_v2_config import ResearchV2Config


THRESHOLD_POLICY = {
    "id": "grid_balanced_accuracy_v1",
    "metric": "balanced_accuracy",
    "minimum": 0.010,
    "maximum": 0.990,
    "step": 0.001,
    "comparison": "greater_than_or_equal",
    "maximizer_absolute_tolerance": 1.0e-12,
    "tie_break": "median_maximizer_lower_on_even",
    "constant_probability_fallback": 0.500,
}


def _plan() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "plan": {"id": "synthetic_blend_plan_v1"},
        "dataset": {
            "version": "synthetic_v1",
            "target": {"negative_label": 0, "positive_label": 1},
        },
        "outer_evaluation": {
            "splitter": "stratified_kfold",
            "n_splits": 2,
            "shuffle": True,
            "repeat_seeds": [0, 17],
        },
        "threshold_selection": {
            "splitter": "stratified_kfold",
            "n_splits": 2,
            "shuffle": True,
            "random_state": 314159,
        },
        "threshold_policy": THRESHOLD_POLICY,
        "metrics": {
            "primary": "balanced_accuracy",
            "secondary": [
                "sensitivity",
                "specificity",
                "roc_auc",
                "average_precision",
                "brier_score",
            ],
            "aggregation": {"unit": "repeat", "summary": ["mean", "median"]},
        },
    }


def _synthetic_run(
    project_root: Path,
    *,
    run_id: str,
    adapter_id: str,
    probability_shift: float = 0.0,
) -> CompletedResearchV2Run:
    root = project_root / "runs" / run_id
    root.mkdir(parents=True)
    manifest = {
        "schema_version": 2,
        "hashing_method": "synthetic_test_fixture",
        "files": [],
        "manifest_sha256": canonical_sha256({"run_id": run_id}),
    }
    (root / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    target = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    fold_rows = {1: [0, 1, 4, 5], 2: [2, 3, 6, 7]}
    outer_prediction_records: list[dict[str, Any]] = []
    outer_records: list[dict[str, int]] = []
    threshold_prediction_records: list[dict[str, Any]] = []
    threshold_records: list[dict[str, int]] = []
    selected_records: list[dict[str, Any]] = []
    fold_metric_records: list[dict[str, Any]] = []
    for repeat, repeat_seed in ((1, 0), (2, 17)):
        for outer_fold, validation_rows in fold_rows.items():
            training_rows = sorted(set(range(8)) - set(validation_rows))
            selected_records.append(
                {
                    "repeat": repeat,
                    "repeat_seed": repeat_seed,
                    "outer_fold": outer_fold,
                    "selected_threshold": 0.5,
                    "threshold_selection_balanced_accuracy": 1.0,
                    "status": "selected",
                    "degenerate": False,
                }
            )
            for row in validation_rows:
                probability = float(
                    np.clip(0.2 + 0.6 * int(target[row]) + probability_shift, 0.0, 1.0)
                )
                prediction = int(probability >= 0.5)
                outer_records.append(
                    {
                        "repeat": repeat,
                        "repeat_seed": repeat_seed,
                        "outer_fold": outer_fold,
                        "row_position": row,
                    }
                )
                outer_prediction_records.append(
                    {
                        "repeat": repeat,
                        "repeat_seed": repeat_seed,
                        "outer_fold": outer_fold,
                        "row_position": row,
                        "target": int(target[row]),
                        "probability": probability,
                        "selected_threshold": 0.5,
                        "prediction": prediction,
                    }
                )
            for threshold_fold, row in enumerate(training_rows[:2], start=1):
                probability = float(
                    np.clip(0.2 + 0.6 * int(target[row]) + probability_shift, 0.0, 1.0)
                )
                threshold_records.append(
                    {
                        "repeat": repeat,
                        "repeat_seed": repeat_seed,
                        "outer_fold": outer_fold,
                        "threshold_selection_fold": threshold_fold,
                        "row_position": row,
                    }
                )
                threshold_prediction_records.append(
                    {
                        "repeat": repeat,
                        "repeat_seed": repeat_seed,
                        "outer_fold": outer_fold,
                        "threshold_selection_fold": threshold_fold,
                        "row_position": row,
                        "target": int(target[row]),
                        "probability": probability,
                        "selected_threshold": 0.5,
                        "prediction": int(probability >= 0.5),
                    }
                )
            fold_metric_records.append(
                {
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "balanced_accuracy": 1.0,
                }
            )
    outer_predictions = pd.DataFrame(outer_prediction_records)
    plan = _plan()
    plan_identity = {"sha256": "e" * 64}
    candidate_identity = {"sha256": canonical_sha256({"adapter": adapter_id})}
    fingerprints = {
        "files": {"train": "b" * 64},
        "row_count": 8,
        "row_position_identity": {
            "kind": "contiguous_zero_based",
            "sha256": canonical_sha256(list(range(8))),
        },
        "source_schema": {
            "ordered_names": ["x"],
            "ordered_names_sha256": "d" * 64,
        },
        "target": {
            "name": "y",
            "dtype": "int8",
            "negative_count": 4,
            "positive_count": 4,
            "values_sha256": canonical_sha256(target.tolist()),
        },
    }
    payload = {
        "schema_version": 2,
        "experiment": {"id": f"experiment_{run_id}"},
        "dataset": {"version": "synthetic_v1"},
        "evaluation_plan_path": "configs/synthetic_plan.yaml",
        "feature_pipeline": {"id": "manual_v3_pipeline_v1_compat", "contract": {}},
        "candidate_adapter": {"id": adapter_id, "contract": {}},
        "artifacts": {"root": "runs"},
        "persistence": {},
        "tracking": {"enabled": False},
    }
    config = ResearchV2Config(
        payload=payload,
        plan_payload=plan,
        source_path=project_root / "synthetic_config.yaml",
        plan_path=project_root / "configs/synthetic_plan.yaml",
        project_root=project_root,
    )
    return CompletedResearchV2Run(
        root=root,
        config=config,
        metadata={
            "schema_version": 2,
            "run_id": run_id,
            "hashes": {
                "plan": plan_identity["sha256"],
                "candidate": candidate_identity["sha256"],
            },
        },
        manifest=manifest,
        dataset_fingerprints=fingerprints,
        evaluation_plan_identity=plan_identity,
        candidate_identity=candidate_identity,
        outer_assignments=pd.DataFrame(outer_records),
        threshold_assignments=pd.DataFrame(threshold_records),
        outer_predictions=outer_predictions,
        threshold_predictions=pd.DataFrame(threshold_prediction_records),
        selected_thresholds=pd.DataFrame(selected_records),
        fold_metrics=pd.DataFrame(fold_metric_records),
        repeat_metrics=build_repeat_metrics(outer_predictions),
    )


def test_weight_grid_and_blend_formula() -> None:
    grid = build_weight_grid()
    assert grid[0] == 0.0
    assert grid[-1] == 1.0
    assert len(grid) == 41
    blend = apply_fixed_blend_probabilities(
        np.asarray([0.2, 0.8]),
        np.asarray([0.4, 0.6]),
        lightgbm_weight=0.25,
    )
    np.testing.assert_allclose(blend, np.asarray([0.35, 0.65]))


def test_weight_threshold_tie_breaks_prefer_half_then_lower() -> None:
    y = np.asarray([0, 1, 0, 1], dtype=np.int8)
    p = np.asarray([0.1, 0.9, 0.2, 0.8], dtype=float)
    selection = select_blend_weight_and_threshold(
        y,
        p,
        p,
        threshold_policy=THRESHOLD_POLICY,
        weight_grid=np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float),
    )
    assert selection.lightgbm_weight == 0.5


def test_cross_fit_excludes_evaluation_fold(tmp_path: Path) -> None:
    lightgbm = _synthetic_run(
        tmp_path,
        run_id="lgbm",
        adapter_id="manual_lightgbm_te_v1_compat",
    )
    xgboost = _synthetic_run(
        tmp_path,
        run_id="xgb",
        adapter_id="xgboost_numeric_v1",
        probability_shift=0.05,
    )
    training_sizes: list[int] = []
    original = select_blend_weight_and_threshold

    def _spy(y_true, lightgbm_probability, xgboost_probability, **kwargs):
        training_sizes.append(len(y_true))
        return original(
            y_true,
            lightgbm_probability,
            xgboost_probability,
            **kwargs,
        )

    import src.churn_ml.blend_evaluation_v1 as module

    monkey = pytest.MonkeyPatch()
    monkey.setattr(module, "select_blend_weight_and_threshold", _spy)
    try:
        result = run_cross_fit_blend_evaluation(
            lightgbm,
            xgboost,
            weight_grid=np.asarray([0.0, 0.5, 1.0], dtype=float),
        )
    finally:
        monkey.undo()

    assert result.summary["tuning_excludes_evaluation_fold"] is True
    assert result.summary["pooled_oof_fallback_used"] is False
    assert result.summary["competition_test_used"] is False
    assert training_sizes
    assert all(size == 4 for size in training_sizes)
    assert len(result.held_out_predictions) == len(lightgbm.outer_predictions)
    assert result.deployment_parameters["weight_aggregation"] == "median"
    assert (
        result.summary["metrics"]["balanced_accuracy"]["confidence_intervals"] is None
    )


def test_compatibility_rejection(tmp_path: Path) -> None:
    lightgbm = _synthetic_run(
        tmp_path,
        run_id="lgbm",
        adapter_id="manual_lightgbm_te_v1_compat",
    )
    xgboost = _synthetic_run(
        tmp_path,
        run_id="xgb",
        adapter_id="xgboost_numeric_v1",
    )
    object.__setattr__(
        xgboost,
        "dataset_fingerprints",
        {**xgboost.dataset_fingerprints, "files": {"train": "f" * 64}},
    )
    with pytest.raises(BlendEvaluationError, match="pair-compatible"):
        validate_blend_inputs(lightgbm, xgboost)


def test_no_optimistic_fallback(tmp_path: Path) -> None:
    lightgbm = _synthetic_run(
        tmp_path,
        run_id="lgbm",
        adapter_id="manual_lightgbm_te_v1_compat",
    )
    xgboost = _synthetic_run(
        tmp_path,
        run_id="xgb",
        adapter_id="xgboost_numeric_v1",
    )
    with pytest.raises(BlendEvaluationError, match="Pooled OOF fallback"):
        run_cross_fit_blend_evaluation(
            lightgbm,
            xgboost,
            forbid_pooled_oof_fallback=False,
            weight_grid=np.asarray([0.5], dtype=float),
        )


def test_metrics_deltas_and_median_parameters(tmp_path: Path) -> None:
    lightgbm = _synthetic_run(
        tmp_path,
        run_id="lgbm",
        adapter_id="manual_lightgbm_te_v1_compat",
    )
    xgboost = _synthetic_run(
        tmp_path,
        run_id="xgb",
        adapter_id="xgboost_numeric_v1",
    )
    result = run_cross_fit_blend_evaluation(
        lightgbm,
        xgboost,
        weight_grid=np.asarray([0.25, 0.5, 0.75], dtype=float),
    )
    assert "blend_minus_lightgbm_balanced_accuracy" in result.fold_metrics.columns
    assert "blend_minus_xgboost_balanced_accuracy" in result.fold_metrics.columns
    weights = result.fold_selections["selected_lightgbm_weight"].to_numpy(dtype=float)
    thresholds = result.fold_selections["selected_threshold"].to_numpy(dtype=float)
    assert result.deployment_parameters["deployment_lightgbm_weight"] == float(
        np.median(weights)
    )
    assert result.deployment_parameters["deployment_threshold"] == float(
        np.median(thresholds)
    )
    assert set(result.sensitivity_table["weight_offset"]) >= {
        0.0,
        0.05,
        -0.05,
        0.1,
        -0.1,
    }


def test_alignment_rejects_missing_rows(tmp_path: Path) -> None:
    lightgbm = _synthetic_run(
        tmp_path,
        run_id="lgbm",
        adapter_id="manual_lightgbm_te_v1_compat",
    )
    xgboost = _synthetic_run(
        tmp_path,
        run_id="xgb",
        adapter_id="xgboost_numeric_v1",
    )
    object.__setattr__(
        xgboost,
        "outer_predictions",
        xgboost.outer_predictions.iloc[:-1].copy(),
    )
    with pytest.raises(BlendEvaluationError, match="pair-compatible|not aligned"):
        validate_blend_inputs(lightgbm, xgboost)


def test_submission_variants_without_retraining(tmp_path: Path) -> None:
    project_root = tmp_path
    deployment = project_root / "artifacts" / "deployments" / "blend_dep"
    deployment.mkdir(parents=True)
    (deployment / "_SUCCESS").write_text("ok", encoding="utf-8")
    pd.DataFrame(
        {
            "row_position": [0, 1, 2, 3],
            "row_id": [10, 11, 12, 13],
            "manual_lightgbm_te_v1_compat": [0.1, 0.2, 0.8, 0.9],
            "xgboost_numeric_v1": [0.2, 0.3, 0.7, 0.85],
        }
    ).to_parquet(deployment / "component_probabilities.parquet", index=False)
    (deployment / "prediction_summary.json").write_text(
        json.dumps({"threshold": 0.5}),
        encoding="utf-8",
    )
    resolved = {
        "deployment_id": "blend_dep",
        "components": [
            {
                "component_id": "manual_lightgbm_te_v1_compat",
                "adapter_id": "manual_lightgbm_te_v1_compat",
                "component_weight": 0.5,
            },
            {
                "component_id": "xgboost_numeric_v1",
                "adapter_id": "xgboost_numeric_v1",
                "component_weight": 0.5,
            },
        ],
        "sample_submission": {
            "path": "data/sample.csv",
            "id_column": "index",
            "target_column": "y",
        },
    }
    (deployment / "resolved_deployment_config.yaml").write_text(
        yaml.safe_dump(resolved),
        encoding="utf-8",
    )
    sample_dir = project_root / "data"
    sample_dir.mkdir()
    pd.DataFrame({"index": [10, 11, 12, 13], "y": [0, 0, 0, 0]}).to_csv(
        sample_dir / "sample.csv",
        index=False,
    )
    variants_root = generate_submission_variants(deployment, project_root=project_root)
    for spec in VARIANT_SPECS:
        frame = pd.read_csv(variants_root / f"{spec.id}.csv")
        assert frame.columns.tolist() == ["index", "y"]
        assert frame["index"].tolist() == [10, 11, 12, 13]
        assert set(frame["y"].tolist()).issubset({0, 1})
    manifest = json.loads((variants_root / "variants_manifest.json").read_text())
    assert manifest["retrained"] is False
    assert all(item["retrained"] is False for item in manifest["variants"])
    probes = {item["id"]: item["leaderboard_probe"] for item in manifest["variants"]}
    assert probes == {
        "blend_robust": False,
        "blend_lgbm_plus": True,
        "blend_xgb_plus": True,
    }


def test_ui_labels_for_blend() -> None:
    assert (
        normalize_model_family("lightgbm_xgboost_blend_v1")
        == "LightGBM + XGBoost Blend"
    )
    assert (
        normalize_source_kind("artifacts/blend_evaluations/lightgbm_xgboost_blend_v1")
        == "Blend evaluation"
    )


def test_cli_parse_commands() -> None:
    args = parse_args(["validate", "--config", "configs/blend_evaluation/x.yaml"])
    assert args.command == "validate"
    args = parse_args(
        [
            "generate-submission-variants",
            "--deployment-dir",
            "artifacts/deployments/x",
        ]
    )
    assert args.command == "generate-submission-variants"


def test_example_config_parses() -> None:
    path = Path("configs/blend_evaluation/lightgbm_xgboost_blend_v1.example.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["experiment"]["id"] == "lightgbm_xgboost_blend_v1"
    assert payload["blend"]["cross_fitting"]["forbid_pooled_oof_fallback"] is True
    grid = build_weight_grid(
        minimum=float(payload["blend"]["lightgbm_weight_grid"]["minimum"]),
        maximum=float(payload["blend"]["lightgbm_weight_grid"]["maximum"]),
        step=float(payload["blend"]["lightgbm_weight_grid"]["step"]),
    )
    assert len(grid) == 41


def test_deterministic_selection_matches_manual_threshold() -> None:
    y = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8)
    p_lgbm = np.asarray([0.1, 0.8, 0.2, 0.7, 0.15, 0.85], dtype=float)
    p_xgb = np.asarray([0.2, 0.7, 0.25, 0.65, 0.2, 0.8], dtype=float)
    selection = select_blend_weight_and_threshold(
        y,
        p_lgbm,
        p_xgb,
        threshold_policy=THRESHOLD_POLICY,
        weight_grid=np.asarray([0.0, 0.5, 1.0], dtype=float),
    )
    blended = apply_fixed_blend_probabilities(
        p_lgbm,
        p_xgb,
        lightgbm_weight=selection.lightgbm_weight,
    )
    manual = select_balanced_accuracy_threshold(y, blended, THRESHOLD_POLICY)
    assert selection.threshold == manual.threshold


def test_registry_exposes_blend_operation() -> None:
    commands = yaml.safe_load(
        Path("configs/ui/ui_commands.yaml").read_text(encoding="utf-8")
    )
    ids = {item["id"] for item in commands["commands"]}
    assert "blend_evaluation_v1" in ids
    assert "final_deployment_v1" in ids
    readers = yaml.safe_load(
        Path("configs/ui/ui_readers.yaml").read_text(encoding="utf-8")
    )
    reader_ids = {item["id"] for item in readers["readers"]}
    assert "blend_evaluation_v1" in reader_ids
