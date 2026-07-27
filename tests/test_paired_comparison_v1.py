from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.churn_ml.paired_comparison import (
    ComparisonPolicy,
    CompletedResearchV2Run,
    PairedComparisonError,
    build_comparison_result,
    build_compatibility_summary,
    build_fixed_blend_diagnostic,
    build_paired_metrics,
    build_prediction_comparison,
    deterministic_comparison_identity,
    load_comparison_policy,
    load_completed_research_v2_run,
)
from src.churn_ml.paired_comparison_artifacts import (
    PairedComparisonArtifactError,
    create_comparison_artifacts,
    validate_comparison_artifacts,
)
from src.churn_ml.paired_comparison_cli import execute
from src.churn_ml.research_data import canonical_sha256
from src.churn_ml.research_protocol import (
    build_repeat_metrics,
    calculate_outer_metrics,
    select_balanced_accuracy_threshold,
)
from src.churn_ml.research_v2_artifact_validation import (
    ResearchV2SemanticValidationError,
)
from src.churn_ml.research_v2_config import ResearchV2Config


POLICY = ComparisonPolicy(
    schema_version=1,
    policy_id="test_policy_v1",
    tie_epsilon=1.0e-12,
    promising_min_mean_balanced_accuracy_delta=0.0001,
    promising_min_repeat_win_fraction=0.5,
    threshold_stability_max_sample_standard_deviation=0.05,
)


def _synthetic_run(
    project_root: Path,
    run_id: str,
    *,
    improved: bool = False,
    constant: bool = False,
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
    outer_records: list[dict[str, int]] = []
    threshold_records: list[dict[str, int]] = []
    outer_prediction_records: list[dict[str, Any]] = []
    threshold_prediction_records: list[dict[str, Any]] = []
    selected_records: list[dict[str, Any]] = []
    fold_metric_records: list[dict[str, Any]] = []
    fold_rows = {1: [0, 1, 4, 5], 2: [2, 3, 6, 7]}
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
            fold_predictions: list[dict[str, Any]] = []
            for row in validation_rows:
                probability = _probability(
                    row, int(target[row]), improved=improved, constant=constant
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
                record = {
                    "repeat": repeat,
                    "repeat_seed": repeat_seed,
                    "outer_fold": outer_fold,
                    "row_position": row,
                    "target": int(target[row]),
                    "probability": probability,
                    "selected_threshold": 0.5,
                    "prediction": prediction,
                }
                outer_prediction_records.append(record)
                fold_predictions.append(record)
            fold_frame = _typed_outer(pd.DataFrame(fold_predictions))
            metrics = calculate_outer_metrics(
                fold_frame["target"],
                fold_frame["probability"].to_numpy(),
                fold_frame["prediction"].to_numpy(),
                selected_threshold=0.5,
            )
            fold_metric_records.append(
                {
                    "repeat": repeat,
                    "repeat_seed": repeat_seed,
                    "outer_fold": outer_fold,
                    **metrics,
                }
            )
            for threshold_index, row in enumerate(training_rows):
                threshold_fold = 1 + threshold_index % 2
                probability = _probability(
                    row, int(target[row]), improved=improved, constant=constant
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
    outer_predictions = _typed_outer(pd.DataFrame(outer_prediction_records))
    threshold_predictions = _typed_threshold(pd.DataFrame(threshold_prediction_records))
    outer_assignments = pd.DataFrame(outer_records).astype("int64")
    threshold_assignments = pd.DataFrame(threshold_records).astype("int64")
    selected_thresholds = pd.DataFrame(selected_records).astype(
        {
            "repeat": "int64",
            "repeat_seed": "int64",
            "outer_fold": "int64",
            "selected_threshold": "float64",
            "threshold_selection_balanced_accuracy": "float64",
            "status": "object",
            "degenerate": "bool",
        }
    )
    plan = _plan()
    plan_identity = {
        "schema_version": 1,
        "sha256": canonical_sha256(plan),
        "canonical": plan,
    }
    candidate_canonical = {
        "schema_version": 2,
        "dataset_version": "synthetic_v1",
        "feature_pipeline": {"id": f"pipeline_{run_id}", "sha256": "1" * 64},
        "candidate_adapter": {"id": f"adapter_{run_id}", "sha256": "2" * 64},
        "probability_semantics": "binary_positive_class_label_1",
        "evaluation_boundary": "adapter_receives_fold_local_data_only",
    }
    candidate_identity = {
        "sha256": canonical_sha256(candidate_canonical),
        "canonical": candidate_canonical,
    }
    fingerprints = {
        "dataset_version": "synthetic_v1",
        "files": {
            "train_features": {"sha256": "a" * 64},
            "target": {"sha256": "b" * 64},
            "metadata": {"sha256": "c" * 64},
        },
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
        "feature_pipeline": {"id": f"pipeline_{run_id}", "contract": {}},
        "candidate_adapter": {"id": f"adapter_{run_id}", "contract": {}},
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
    metadata = {
        "schema_version": 2,
        "run_id": run_id,
        "hashes": {
            "plan": plan_identity["sha256"],
            "candidate": candidate_identity["sha256"],
        },
    }
    return CompletedResearchV2Run(
        root=root,
        config=config,
        metadata=metadata,
        manifest=manifest,
        dataset_fingerprints=fingerprints,
        evaluation_plan_identity=plan_identity,
        candidate_identity=candidate_identity,
        outer_assignments=outer_assignments,
        threshold_assignments=threshold_assignments,
        outer_predictions=outer_predictions,
        threshold_predictions=threshold_predictions,
        selected_thresholds=selected_thresholds,
        fold_metrics=pd.DataFrame(fold_metric_records),
        repeat_metrics=build_repeat_metrics(outer_predictions),
    )


def _plan() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "plan": {"id": "synthetic_paired_v1"},
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
        "threshold_policy": {
            "id": "grid_balanced_accuracy_v1",
            "metric": "balanced_accuracy",
            "minimum": 0.1,
            "maximum": 0.9,
            "step": 0.1,
            "comparison": "greater_than_or_equal",
            "maximizer_absolute_tolerance": 1e-12,
            "tie_break": "median_maximizer_lower_on_even",
            "constant_probability_fallback": 0.5,
        },
        "metrics": {
            "primary": "balanced_accuracy",
            "secondary": [
                "sensitivity",
                "specificity",
                "roc_auc",
                "average_precision",
                "brier_score",
            ],
        },
        "aggregation": {
            "repeat_method": "pooled_predictions",
            "aggregate_unit": "repeat",
            "standard_deviation": "sample",
            "fold_standard_deviation": "descriptive_only",
            "confidence_interval": "none",
        },
    }


def _probability(row: int, target: int, *, improved: bool, constant: bool) -> float:
    if constant:
        return 0.5
    if improved:
        return 0.9 if target else 0.1
    return (0.2, 0.4, 0.6, 0.8)[row % 4]


def _typed_outer(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.astype(
        {
            "repeat": "int64",
            "repeat_seed": "int64",
            "outer_fold": "int64",
            "row_position": "int64",
            "target": "int8",
            "probability": "float64",
            "selected_threshold": "float64",
            "prediction": "int8",
        }
    )


def _typed_threshold(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.astype(
        {
            "repeat": "int64",
            "repeat_seed": "int64",
            "outer_fold": "int64",
            "threshold_selection_fold": "int64",
            "row_position": "int64",
            "target": "int8",
            "probability": "float64",
            "selected_threshold": "float64",
            "prediction": "int8",
        }
    )


@pytest.fixture
def runs(tmp_path: Path) -> tuple[CompletedResearchV2Run, CompletedResearchV2Run]:
    return (
        _synthetic_run(tmp_path, "baseline"),
        _synthetic_run(tmp_path, "candidate", improved=True),
    )


def test_identical_run_self_comparison_has_zero_deltas_and_perfect_agreement(
    tmp_path: Path,
) -> None:
    run = _synthetic_run(tmp_path, "self")
    result = build_comparison_result(run, run, POLICY)
    delta_columns = [
        name for name in result.paired_metrics.repeat_metrics if name.endswith("_delta")
    ]
    assert (result.paired_metrics.repeat_metrics[delta_columns] == 0.0).all().all()
    assert result.prediction_comparison["prediction_label_agreement_rate"] == 1.0
    assert result.prediction_comparison["pearson_correlation"]["coefficient"] == 1.0


def test_candidate_improvement_and_wins_ties_losses(
    runs: tuple[CompletedResearchV2Run, CompletedResearchV2Run],
) -> None:
    baseline, candidate = runs
    result = build_comparison_result(baseline, candidate, POLICY)
    balanced = result.paired_metrics.aggregate_summary["metrics"]["balanced_accuracy"]
    assert balanced["candidate_minus_baseline"]["mean"] > 0
    assert balanced["repeat_wins_ties_losses"] == {
        "wins": 2,
        "ties": 0,
        "losses": 0,
    }
    assert result.decision_report["every_repeat_improved"] is True


@pytest.mark.parametrize(
    ("mutation", "reason_code"),
    [
        ("plan", "EVALUATION_PLAN_IDENTITY_MISMATCH"),
        ("dataset", "DATASET_CONTENT_MISMATCH"),
        ("assignments", "OUTER_ASSIGNMENTS_MISMATCH"),
        ("target", "TARGET_IDENTITY_MISMATCH"),
        ("threshold", "THRESHOLD_POLICY_MISMATCH"),
    ],
)
def test_compatibility_gate_reports_exact_reason_codes(
    tmp_path: Path,
    mutation: str,
    reason_code: str,
) -> None:
    baseline = _synthetic_run(tmp_path, "baseline")
    candidate = _synthetic_run(tmp_path, "candidate")
    if mutation == "plan":
        identity = deepcopy(candidate.evaluation_plan_identity)
        identity["sha256"] = "f" * 64
        candidate = replace(candidate, evaluation_plan_identity=identity)
    elif mutation == "dataset":
        fingerprints = deepcopy(candidate.dataset_fingerprints)
        fingerprints["files"]["target"]["sha256"] = "f" * 64
        candidate = replace(candidate, dataset_fingerprints=fingerprints)
    elif mutation == "assignments":
        assignments = candidate.outer_assignments.copy()
        assignments.loc[0, "outer_fold"] = 2
        candidate = replace(candidate, outer_assignments=assignments)
    elif mutation == "target":
        fingerprints = deepcopy(candidate.dataset_fingerprints)
        fingerprints["target"]["values_sha256"] = "f" * 64
        candidate = replace(candidate, dataset_fingerprints=fingerprints)
    else:
        plan = deepcopy(candidate.config.plan_payload)
        plan["threshold_policy"]["maximum"] = 0.8
        candidate = replace(
            candidate, config=replace(candidate.config, plan_payload=plan)
        )
    report = build_compatibility_summary(baseline, candidate)
    assert report.compatible is False
    assert reason_code in {issue.reason_code for issue in report.issues}
    assert all(issue.field_path for issue in report.issues)


def test_duplicate_or_missing_fold_and_prediction_keys_are_rejected(
    tmp_path: Path,
) -> None:
    baseline = _synthetic_run(tmp_path, "baseline")
    candidate = _synthetic_run(tmp_path, "candidate")
    duplicate = pd.concat(
        [candidate.selected_thresholds, candidate.selected_thresholds.iloc[[0]]],
        ignore_index=True,
    )
    assert not build_compatibility_summary(
        baseline, replace(candidate, selected_thresholds=duplicate)
    ).compatible
    assert not build_compatibility_summary(
        baseline,
        replace(candidate, outer_predictions=candidate.outer_predictions.iloc[1:]),
    ).compatible


def test_exact_threshold_equality_and_leakage_safe_blend(
    runs: tuple[CompletedResearchV2Run, CompletedResearchV2Run],
) -> None:
    baseline, candidate = runs
    diagnostic = build_fixed_blend_diagnostic(baseline, candidate, tie_epsilon=1.0e-12)
    baseline_threshold = baseline.threshold_predictions
    candidate_threshold = candidate.threshold_predictions
    mask = (baseline_threshold["repeat"] == 1) & (baseline_threshold["outer_fold"] == 1)
    probability = (
        0.5 * baseline_threshold.loc[mask, "probability"].to_numpy()
        + 0.5 * candidate_threshold.loc[mask, "probability"].to_numpy()
    )
    expected = select_balanced_accuracy_threshold(
        baseline_threshold.loc[mask, "target"],
        probability,
        baseline.config.plan_payload["threshold_policy"],
    )
    actual = diagnostic.fold_metrics.loc[
        (diagnostic.fold_metrics["repeat"] == 1)
        & (diagnostic.fold_metrics["outer_fold"] == 1),
        "blend_threshold",
    ].item()
    assert actual == expected.threshold
    assert (np.asarray([actual - 0.01, actual]) >= actual).astype("int8").tolist() == [
        0,
        1,
    ]
    assert (
        diagnostic.summary["outer_validation_target_used_for_threshold_selection"]
        is False
    )


@pytest.mark.parametrize("bad_value", [0, True, np.float64(0.0), float("nan")])
def test_tie_epsilon_requires_exact_finite_float(
    tmp_path: Path, bad_value: Any
) -> None:
    run = _synthetic_run(tmp_path, "self")
    with pytest.raises(PairedComparisonError, match="exact finite float"):
        build_paired_metrics(run, run, tie_epsilon=bad_value)


def test_brier_direction_and_sensitivity_specificity_trade_off(
    runs: tuple[CompletedResearchV2Run, CompletedResearchV2Run],
) -> None:
    baseline, candidate = runs
    paired = build_paired_metrics(baseline, candidate, tie_epsilon=1.0e-12)
    brier = paired.aggregate_summary["metrics"]["brier_score"]
    assert brier["optimization_direction"] == "lower_is_better"
    assert brier["candidate_minus_baseline"]["mean"] < 0
    assert brier["repeat_wins_ties_losses"]["wins"] == 2
    assert (
        paired.aggregate_summary["sensitivity_specificity_trade_off"][
            "both_improved_repeats"
        ]
        == 2
    )


def test_prediction_disagreement_matrix_and_correlation_edges(tmp_path: Path) -> None:
    baseline = _synthetic_run(tmp_path, "baseline")
    candidate = _synthetic_run(tmp_path, "candidate", improved=True)
    comparison = build_prediction_comparison(baseline, candidate)
    matrix = comparison["correctness_matrix"]
    assert sum(item["count"] for item in matrix.values()) == 16
    assert matrix["candidate_only_correct"]["count"] > 0
    constant = _synthetic_run(tmp_path, "constant", constant=True)
    undefined = build_prediction_comparison(constant, candidate)
    assert undefined["pearson_correlation"] == {
        "coefficient": None,
        "status": "undefined_constant_vector",
    }
    identical_constant = build_prediction_comparison(constant, constant)
    assert identical_constant["spearman_correlation"]["coefficient"] == 1.0


def test_blend_alignment_rejection(tmp_path: Path) -> None:
    baseline = _synthetic_run(tmp_path, "baseline")
    candidate = _synthetic_run(tmp_path, "candidate")
    candidate = replace(
        candidate,
        threshold_predictions=candidate.threshold_predictions.iloc[1:].copy(),
    )
    with pytest.raises(PairedComparisonError):
        build_fixed_blend_diagnostic(baseline, candidate, tie_epsilon=1.0e-12)


def test_deterministic_identity(
    runs: tuple[CompletedResearchV2Run, CompletedResearchV2Run],
) -> None:
    baseline, candidate = runs
    first = deterministic_comparison_identity(
        baseline.reference("baseline"), candidate.reference("candidate"), POLICY
    )
    second = deterministic_comparison_identity(
        baseline.reference("baseline"), candidate.reference("candidate"), POLICY
    )
    assert first == second
    assert len(first) == 64


def test_policy_loader_rejects_non_float_tie_epsilon(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        """
schema_version: 1
policy:
  id: invalid
  tie_epsilon: 0
  promising_min_mean_balanced_accuracy_delta: 0.0
  promising_min_repeat_win_fraction: 0.5
  threshold_stability_max_sample_standard_deviation: 0.05
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(PairedComparisonError, match="exact finite float"):
        load_comparison_policy(policy_path)


def test_failed_input_delegates_to_production_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "input"
    root.mkdir()
    (root / "resolved_config.yaml").write_text(
        "schema_version: 2\nevaluation_plan_path: configs/plan.yaml\n"
        "evaluation_plan: {}\n",
        encoding="utf-8",
    )
    (root / "run_metadata.json").write_text(
        json.dumps({"hashes": {"plan": "a"}}), encoding="utf-8"
    )

    def reject(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise ResearchV2SemanticValidationError("Run contains _FAILED.")

    monkeypatch.setattr(
        "src.churn_ml.paired_comparison.validate_research_v2_run", reject
    )
    with pytest.raises(ResearchV2SemanticValidationError, match="_FAILED"):
        load_completed_research_v2_run(root, project_root=tmp_path)


def test_path_and_symlink_escape_are_rejected_where_supported(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}_outside"
    outside.mkdir(exist_ok=True)
    with pytest.raises(PairedComparisonError, match="escapes"):
        load_completed_research_v2_run(outside, project_root=tmp_path)
    link = tmp_path / "escaped_link"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except OSError:
        return
    with pytest.raises(PairedComparisonError, match="escapes"):
        load_completed_research_v2_run(link, project_root=tmp_path)


def _patch_artifact_validation(
    monkeypatch: pytest.MonkeyPatch,
    baseline: CompletedResearchV2Run,
    candidate: CompletedResearchV2Run,
) -> None:
    def load(path: Path, *, project_root: Path) -> CompletedResearchV2Run:
        del project_root
        return baseline if path.resolve().name == "baseline" else candidate

    provenance = {
        "schema_version": 1,
        "hashing_method": "synthetic",
        "files": [],
        "sha256": canonical_sha256({"synthetic": True}),
    }
    monkeypatch.setattr(
        "src.churn_ml.paired_comparison_artifacts.load_completed_research_v2_run",
        load,
    )
    monkeypatch.setattr(
        "src.churn_ml.paired_comparison_artifacts.comparison_source_provenance",
        lambda project_root: provenance,
    )


def test_artifact_lifecycle_manifest_success_and_existing_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = _synthetic_run(tmp_path, "baseline")
    candidate = _synthetic_run(tmp_path, "candidate", improved=True)
    _patch_artifact_validation(monkeypatch, baseline, candidate)
    result = build_comparison_result(baseline, candidate, POLICY)
    comparison_root = create_comparison_artifacts(
        result=result,
        policy=POLICY,
        project_root=tmp_path,
        output_root=Path("comparisons"),
        comparison_id="synthetic-improved",
    )
    validate_comparison_artifacts(
        comparison_root,
        project_root=tmp_path,
        require_success=True,
        verify_manifest=True,
    )
    assert (comparison_root / "_SUCCESS").is_file()
    assert not (comparison_root / "_FAILED").exists()
    success_time = (comparison_root / "_SUCCESS").stat().st_mtime_ns
    assert all(
        success_time >= path.stat().st_mtime_ns
        for path in comparison_root.iterdir()
        if path.name != "_SUCCESS"
    )
    manifest = json.loads(
        (comparison_root / "manifest.json").read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (comparison_root / "artifact_inventory.json").read_text(encoding="utf-8")
    )
    assert manifest["manifest_sha256"]
    assert inventory["inventory_sha256"]
    with pytest.raises(PairedComparisonArtifactError, match="already exists"):
        create_comparison_artifacts(
            result=result,
            policy=POLICY,
            project_root=tmp_path,
            output_root=Path("comparisons"),
            comparison_id="synthetic-improved",
        )


def test_failure_creates_failed_without_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = _synthetic_run(tmp_path, "baseline")
    candidate = _synthetic_run(tmp_path, "candidate", improved=True)
    _patch_artifact_validation(monkeypatch, baseline, candidate)
    result = build_comparison_result(baseline, candidate, POLICY)

    def fail() -> None:
        raise RuntimeError("last fallible operation failed")

    with pytest.raises(RuntimeError, match="last fallible"):
        create_comparison_artifacts(
            result=result,
            policy=POLICY,
            project_root=tmp_path,
            output_root=Path("comparisons"),
            comparison_id="synthetic-failed",
            before_success=fail,
        )
    root = tmp_path / "comparisons/synthetic-failed"
    assert (root / "_FAILED").is_file()
    assert not (root / "_SUCCESS").exists()


def test_validate_only_allocates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = _synthetic_run(tmp_path, "baseline")
    candidate = _synthetic_run(tmp_path, "candidate")
    calls = iter((baseline, candidate))
    monkeypatch.setattr(
        "src.churn_ml.paired_comparison_cli.load_completed_research_v2_run",
        lambda *args, **kwargs: next(calls),
    )
    monkeypatch.setattr(
        "src.churn_ml.paired_comparison_cli.load_comparison_policy",
        lambda path: POLICY,
    )
    output = tmp_path / "must_not_exist"
    args = argparse.Namespace(
        baseline_run_dir=baseline.root,
        candidate_run_dir=candidate.root,
        comparison_id=None,
        output_root=output,
        policy=Path("configs/research_v2/comparison_policy_v1.yaml"),
        validate_only=True,
    )
    assert execute(args) == 0
    assert not output.exists()


def test_outputs_have_no_formal_claims_or_deployment_action(
    runs: tuple[CompletedResearchV2Run, CompletedResearchV2Run],
) -> None:
    baseline, candidate = runs
    result = build_comparison_result(baseline, candidate, POLICY)
    serialized = json.dumps(
        {
            "aggregate": result.paired_metrics.aggregate_summary,
            "decision": result.decision_report,
            "blend": result.blend.summary,
        }
    ).lower()
    assert "p_value" not in serialized
    assert "confidence_interval" not in serialized
    assert result.decision_report["formal_inference_performed"] is False
    assert result.decision_report["deployment_recommendation"] is None
    assert result.blend.summary["weight_search_performed"] is False
