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
import yaml

from src.churn_ml.paired_comparison import (
    build_comparison_result,
    _wins_ties_losses,
    build_compatibility_summary,
)
from src.churn_ml.paired_comparison_artifacts import (
    PROVENANCE_ARTIFACT,
    PairedComparisonArtifactError,
    create_comparison_artifacts,
    validate_comparison_artifacts,
)
from src.churn_ml.paired_comparison_cli import execute
from src.churn_ml.paired_comparison_paths import (
    PairedPathSafetyError,
    assert_pairwise_disjoint_paths,
    prewalk_regular_tree,
    resolve_repository_path,
)
from src.churn_ml.research_v2_resolved_config import (
    ResolvedResearchV2ConfigurationError,
    load_resolved_research_v2_config,
)
from tests.test_paired_comparison_v1 import (
    POLICY,
    _patch_artifact_validation,
    _synthetic_run,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUN_CONFIG = (
    REPOSITORY_ROOT / "configs/research_v2/manual_lightgbm_te_v1_compat_smoke.yaml"
)
PLAN_CONFIG = (
    REPOSITORY_ROOT / "configs/research_v2/plans/telecom_v3_smoke_r1x3_t2_v1.yaml"
)


def _resolved_payload() -> dict[str, Any]:
    payload = yaml.safe_load(RUN_CONFIG.read_text(encoding="utf-8"))
    payload["evaluation_plan"] = yaml.safe_load(PLAN_CONFIG.read_text(encoding="utf-8"))
    return payload


def _write_resolved(root: Path, payload: dict[str, Any]) -> Path:
    path = root / "resolved_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _strict_error(
    tmp_path: Path, payload: dict[str, Any]
) -> ResolvedResearchV2ConfigurationError:
    path = _write_resolved(tmp_path, payload)
    with pytest.raises(ResolvedResearchV2ConfigurationError) as caught:
        load_resolved_research_v2_config(path, project_root=tmp_path)
    return caught.value


def test_strict_resolved_loader_accepts_self_contained_production_contract(
    tmp_path: Path,
) -> None:
    config = load_resolved_research_v2_config(
        _write_resolved(tmp_path, _resolved_payload()),
        project_root=tmp_path,
    )
    assert config.payload["schema_version"] == 2
    assert config.dataset_version == "v3_targeted_missingness"
    assert not config.plan_path.exists()


@pytest.mark.parametrize(
    ("mutation", "reason_code", "field_path"),
    [
        ("unknown_nested", "RESOLVED_CONFIG_KEYS_INVALID", "resolved_config.tracking"),
        (
            "missing_nested",
            "RESOLVED_CONFIG_KEYS_INVALID",
            "resolved_config.persistence",
        ),
        (
            "bool_as_int",
            "RESOLVED_PLAN_CONTRACT_INVALID",
            "resolved_config.evaluation_plan.dataset.expected_rows",
        ),
        (
            "float_as_int",
            "RESOLVED_PLAN_CONTRACT_INVALID",
            "resolved_config.evaluation_plan.outer_evaluation.n_splits",
        ),
        (
            "string_as_int",
            "RESOLVED_PLAN_CONTRACT_INVALID",
            "resolved_config.evaluation_plan.threshold_selection.n_splits",
        ),
        (
            "invalid_invariant",
            "RESOLVED_PLAN_CONTRACT_INVALID",
            "resolved_config.evaluation_plan.outer_evaluation.n_splits",
        ),
        ("tracking", "RESOLVED_TRACKING_ENABLED", "resolved_config.tracking.enabled"),
        (
            "models",
            "RESOLVED_MODEL_PERSISTENCE_ENABLED",
            "resolved_config.persistence.models",
        ),
    ],
)
def test_strict_resolved_loader_exact_corruption_errors(
    tmp_path: Path,
    mutation: str,
    reason_code: str,
    field_path: str,
) -> None:
    payload = _resolved_payload()
    if mutation == "unknown_nested":
        payload["tracking"]["unexpected"] = False
    elif mutation == "missing_nested":
        del payload["persistence"]["models"]
    elif mutation == "bool_as_int":
        payload["evaluation_plan"]["dataset"]["expected_rows"] = True
    elif mutation == "float_as_int":
        payload["evaluation_plan"]["outer_evaluation"]["n_splits"] = 3.0
    elif mutation == "string_as_int":
        payload["evaluation_plan"]["threshold_selection"]["n_splits"] = "2"
    elif mutation == "invalid_invariant":
        payload["evaluation_plan"]["outer_evaluation"]["n_splits"] = 1
    elif mutation == "tracking":
        payload["tracking"]["enabled"] = True
    else:
        payload["persistence"]["models"] = True
    error = _strict_error(tmp_path, payload)
    assert (error.reason_code, error.field_path) == (reason_code, field_path)


@pytest.mark.parametrize(
    "bad_path",
    [
        "../escape",
        "/absolute/path",
        r"C:\absolute\path",
        r"C:drive-relative",
        r"\\server\share\path",
        r"\rooted\path",
    ],
)
def test_strict_resolved_loader_rejects_all_nonportable_paths(
    tmp_path: Path,
    bad_path: str,
) -> None:
    payload = _resolved_payload()
    payload["artifacts"]["root"] = bad_path
    error = _strict_error(tmp_path, payload)
    assert (error.reason_code, error.field_path) == (
        "RESOLVED_CONFIG_PATH_INVALID",
        "resolved_config.artifacts.root",
    )


@pytest.mark.parametrize(
    ("relative", "role", "reason_code"),
    [
        ("submissions", "baseline_run", "BASELINE_RUN_FORBIDDEN_NAMESPACE"),
        ("submissions/one/deep", "candidate_run", "CANDIDATE_RUN_FORBIDDEN_NAMESPACE"),
        ("models", "comparison_output", "COMPARISON_OUTPUT_FORBIDDEN_NAMESPACE"),
        ("data/raw", "baseline_run", "BASELINE_RUN_FORBIDDEN_NAMESPACE"),
        (
            "artifacts/final_submissions/deep",
            "comparison_output",
            "COMPARISON_OUTPUT_FORBIDDEN_NAMESPACE",
        ),
        (
            r"artifacts\autogluon_runs\deep",
            "candidate_run",
            "CANDIDATE_RUN_FORBIDDEN_NAMESPACE",
        ),
        ("kaggle_assets/deep", "baseline_run", "BASELINE_RUN_FORBIDDEN_NAMESPACE"),
        (
            "sample_submission/deep",
            "comparison_output",
            "COMPARISON_OUTPUT_FORBIDDEN_NAMESPACE",
        ),
    ],
)
def test_exact_forbidden_namespaces_for_inputs_and_outputs(
    tmp_path: Path,
    relative: str,
    role: str,
    reason_code: str,
) -> None:
    with pytest.raises(PairedPathSafetyError) as caught:
        resolve_repository_path(
            Path(relative),
            project_root=tmp_path,
            role=role,
            field_path=f"paths.{role}",
            must_exist=False,
            require_directory=True,
        )
    assert caught.value.reason_code == reason_code


@pytest.mark.parametrize(
    "relative", ["models_analysis", "submission_notes", "raw_analysis"]
)
def test_similar_legal_namespace_names_are_allowed(
    tmp_path: Path, relative: str
) -> None:
    assert (
        resolve_repository_path(
            Path(relative),
            project_root=tmp_path,
            role="comparison_output",
            field_path="paths.output_root",
            must_exist=False,
            require_directory=True,
        )
        == tmp_path / relative
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("runs/baseline", "runs/baseline"),
        ("runs/baseline", "runs/baseline/output"),
        ("runs/baseline/nested", "runs/baseline"),
        ("runs/candidate", "runs/candidate/output"),
        ("runs/candidate/nested", "runs/candidate"),
    ],
)
def test_pairwise_overlap_is_rejected_with_exact_error(
    tmp_path: Path, left: str, right: str
) -> None:
    with pytest.raises(PairedPathSafetyError) as caught:
        assert_pairwise_disjoint_paths(
            {"baseline_run": tmp_path / left, "output_root": tmp_path / right}
        )
    assert (caught.value.reason_code, caught.value.field_path) == (
        "COMPARISON_PATHS_OVERLAP",
        "paths.baseline_run|paths.output_root",
    )


def test_sibling_paths_are_disjoint(tmp_path: Path) -> None:
    assert_pairwise_disjoint_paths(
        {
            "baseline_run": tmp_path / "runs/baseline",
            "candidate_run": tmp_path / "runs/candidate",
            "output_root": tmp_path / "comparisons",
        }
    )


def test_overlap_refusal_occurs_before_output_allocation(tmp_path: Path) -> None:
    baseline = _synthetic_run(tmp_path, "baseline")
    candidate = _synthetic_run(tmp_path, "candidate")
    result = build_comparison_result(baseline, candidate, POLICY)
    proposed = baseline.root / "nested-output"
    with pytest.raises(PairedComparisonArtifactError, match="COMPARISON_PATHS_OVERLAP"):
        create_comparison_artifacts(
            result=result,
            policy=POLICY,
            project_root=tmp_path,
            output_root=proposed,
            comparison_id="must-not-exist",
        )
    assert not proposed.exists()
    assert not (baseline.root / "_FAILED").exists()


def test_baseline_and_candidate_must_be_distinct(tmp_path: Path) -> None:
    run = _synthetic_run(tmp_path, "same")
    result = build_comparison_result(run, run, POLICY)
    with pytest.raises(PairedComparisonArtifactError, match="COMPARISON_PATHS_OVERLAP"):
        create_comparison_artifacts(
            result=result,
            policy=POLICY,
            project_root=tmp_path,
            output_root=Path("comparisons"),
            comparison_id="same-run",
        )


def test_input_tree_prewalk_accepts_regular_files(tmp_path: Path) -> None:
    root = tmp_path / "run"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (nested / "ordinary.json").write_text("{}", encoding="utf-8")
    snapshot = prewalk_regular_tree(
        root,
        project_root=tmp_path,
        role="baseline_run",
        field_path="paths.baseline_run",
        reject_hardlinks=True,
        reject_forbidden_descendants=True,
    )
    assert snapshot.files == (Path("nested/ordinary.json"),)


def _created_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, comparison_id: str
) -> Path:
    baseline = _synthetic_run(tmp_path, "baseline")
    candidate = _synthetic_run(tmp_path, "candidate", improved=True)
    _patch_artifact_validation(monkeypatch, baseline, candidate)
    return create_comparison_artifacts(
        result=build_comparison_result(baseline, candidate, POLICY),
        policy=POLICY,
        project_root=tmp_path,
        output_root=Path("comparisons"),
        comparison_id=comparison_id,
    )


@pytest.mark.parametrize(
    "tamper",
    [
        "nested_model",
        "nested_submission",
        "nested_text",
        "empty_directory",
        "changed_allowed",
        "deleted_allowed",
        "extra_top_level",
    ],
)
def test_recursive_inventory_rejects_all_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    root = _created_comparison(tmp_path, monkeypatch, f"tamper-{tamper}")
    if tamper == "nested_model":
        path = root / "models/copied.joblib"
        path.parent.mkdir()
        path.write_bytes(b"model")
    elif tamper == "nested_submission":
        path = root / "submissions/nested/result.csv"
        path.parent.mkdir(parents=True)
        path.write_text("prediction\n1\n", encoding="utf-8")
    elif tamper == "nested_text":
        path = root / "notes/unlisted.txt"
        path.parent.mkdir()
        path.write_text("unexpected", encoding="utf-8")
    elif tamper == "empty_directory":
        (root / "unexpected-empty").mkdir()
    elif tamper == "changed_allowed":
        (root / PROVENANCE_ARTIFACT).write_text("{}", encoding="utf-8")
    elif tamper == "deleted_allowed":
        (root / PROVENANCE_ARTIFACT).unlink()
    else:
        (root / "extra.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(PairedComparisonArtifactError):
        validate_comparison_artifacts(
            root,
            project_root=tmp_path,
            require_success=True,
            verify_manifest=True,
        )


def test_recursive_manifest_records_nested_path_size_and_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _created_comparison(tmp_path, monkeypatch, "recursive-manifest")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    record = next(
        item for item in manifest["files"] if item["path"] == PROVENANCE_ARTIFACT
    )
    assert record["size_bytes"] == (root / PROVENANCE_ARTIFACT).stat().st_size
    assert len(record["sha256"]) == 64
    record["size_bytes"] += 1
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PairedComparisonArtifactError, match="manifest"):
        validate_comparison_artifacts(
            root,
            project_root=tmp_path,
            require_success=True,
            verify_manifest=True,
        )


def test_all_link_boundaries_are_rejected_with_one_privilege_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside_file = tmp_path.parent / f"{tmp_path.name}-outside.json"
    outside_file.write_text("{}", encoding="utf-8")
    outside_directory = tmp_path.parent / f"{tmp_path.name}-outside-directory"
    outside_directory.mkdir()
    linked_root = tmp_path / "linked-root"
    try:
        os.symlink(outside_directory, linked_root, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symbolic-link creation unavailable: {error}")
    with pytest.raises(PairedPathSafetyError) as caught:
        resolve_repository_path(
            linked_root,
            project_root=tmp_path,
            role="baseline_run",
            field_path="paths.baseline_run",
            must_exist=True,
            require_directory=True,
        )
    assert caught.value.reason_code == "BASELINE_RUN_PATH_LINK"
    linked_root.unlink()

    run = tmp_path / "input-run"
    nested = run / "nested"
    nested.mkdir(parents=True)
    linked_file = nested / "expected.json"
    os.symlink(outside_file, linked_file)
    with pytest.raises(PairedPathSafetyError, match="TREE_LINK"):
        prewalk_regular_tree(
            run,
            project_root=tmp_path,
            role="baseline_run",
            field_path="paths.baseline_run",
            reject_hardlinks=True,
            reject_forbidden_descendants=True,
        )
    linked_file.unlink()
    linked_directory = nested / "linked-directory"
    os.symlink(outside_directory, linked_directory, target_is_directory=True)
    with pytest.raises(PairedPathSafetyError, match="TREE_LINK"):
        prewalk_regular_tree(
            run,
            project_root=tmp_path,
            role="baseline_run",
            field_path="paths.baseline_run",
            reject_hardlinks=True,
            reject_forbidden_descendants=True,
        )
    linked_directory.unlink()

    root = _created_comparison(tmp_path, monkeypatch, "linked-artifacts")
    for relative in ("manifest.json", "artifact_inventory.json"):
        path = root / relative
        original = path.read_bytes()
        path.unlink()
        os.symlink(outside_file, path)
        with pytest.raises(PairedComparisonArtifactError, match="TREE_LINK"):
            validate_comparison_artifacts(
                root,
                project_root=tmp_path,
                require_success=True,
                verify_manifest=True,
            )
        path.unlink()
        path.write_bytes(original)
    nested_link = root / "provenance/linked.txt"
    os.symlink(outside_file, nested_link)
    with pytest.raises(PairedComparisonArtifactError, match="TREE_LINK"):
        validate_comparison_artifacts(
            root,
            project_root=tmp_path,
            require_success=True,
            verify_manifest=True,
        )


def _compatibility_mutation(candidate: Any, mutation: str) -> Any:
    if mutation == "plan_hash":
        identity = deepcopy(candidate.evaluation_plan_identity)
        identity["sha256"] = "f" * 64
        return replace(candidate, evaluation_plan_identity=identity)
    if mutation in {
        "schema",
        "protocol",
        "plan_id",
        "metrics",
        "aggregation",
        "threshold_policy",
        "threshold_grid",
        "positive_class",
        "label",
    }:
        payload = deepcopy(candidate.config.payload)
        plan = deepcopy(candidate.config.plan_payload)
        if mutation == "schema":
            payload["schema_version"] = 3
        elif mutation == "protocol":
            plan["schema_version"] = 2
        elif mutation == "plan_id":
            plan["plan"]["id"] = "different_plan"
        elif mutation == "metrics":
            plan["metrics"]["primary"] = "roc_auc"
        elif mutation == "aggregation":
            plan["aggregation"]["aggregate_unit"] = "fold"
        elif mutation == "threshold_policy":
            plan["threshold_policy"]["id"] = "different_policy"
        elif mutation == "threshold_grid":
            plan["threshold_policy"]["maximum"] = 0.8
        elif mutation == "positive_class":
            plan["dataset"]["target"]["positive_label"] = 2
        else:
            plan["threshold_policy"]["comparison"] = "greater_than"
        return replace(
            candidate,
            config=replace(candidate.config, payload=payload, plan_payload=plan),
        )
    if mutation in {"dataset_version", "dataset_content", "target", "row_order"}:
        if mutation == "dataset_version":
            payload = deepcopy(candidate.config.payload)
            payload["dataset"]["version"] = "different_dataset"
            return replace(candidate, config=replace(candidate.config, payload=payload))
        fingerprints = deepcopy(candidate.dataset_fingerprints)
        if mutation == "dataset_content":
            fingerprints["files"]["target"]["sha256"] = "f" * 64
        elif mutation == "target":
            fingerprints["target"]["values_sha256"] = "f" * 64
        else:
            fingerprints["row_position_identity"]["sha256"] = "f" * 64
        return replace(candidate, dataset_fingerprints=fingerprints)
    if mutation == "outer_assignments":
        frame = candidate.outer_assignments.copy()
        frame.loc[0, "outer_fold"] = 99
        return replace(candidate, outer_assignments=frame)
    if mutation == "threshold_assignments":
        frame = candidate.threshold_assignments.copy()
        frame.loc[0, "threshold_selection_fold"] = 99
        return replace(candidate, threshold_assignments=frame)
    if mutation == "probability":
        identity = deepcopy(candidate.candidate_identity)
        identity["canonical"]["probability_semantics"] = "different"
        return replace(candidate, candidate_identity=identity)
    if mutation == "fold_product":
        return replace(
            candidate, selected_thresholds=candidate.selected_thresholds.iloc[1:]
        )
    if mutation == "prediction_keys_missing":
        return replace(
            candidate, outer_predictions=candidate.outer_predictions.iloc[1:]
        )
    if mutation == "threshold_keys_missing":
        return replace(
            candidate,
            threshold_predictions=candidate.threshold_predictions.iloc[1:],
        )
    if mutation == "outer_prediction_target":
        frame = candidate.outer_predictions.copy()
        frame.loc[0, "target"] = 1 - frame.loc[0, "target"]
        return replace(candidate, outer_predictions=frame)
    if mutation == "threshold_prediction_target":
        frame = candidate.threshold_predictions.copy()
        frame.loc[0, "target"] = 1 - frame.loc[0, "target"]
        return replace(candidate, threshold_predictions=frame)
    if mutation == "threshold_keys_duplicate":
        frame = pd.concat(
            [
                candidate.threshold_predictions,
                candidate.threshold_predictions.iloc[[0]],
            ],
            ignore_index=True,
        )
        return replace(candidate, threshold_predictions=frame)
    frame = pd.concat(
        [candidate.outer_predictions, candidate.outer_predictions.iloc[[0]]],
        ignore_index=True,
    )
    return replace(candidate, outer_predictions=frame)


@pytest.mark.parametrize(
    ("mutation", "reason_code", "field_path"),
    [
        ("schema", "SCHEMA_VERSION_MISMATCH", "resolved_config.schema_version"),
        (
            "protocol",
            "PROTOCOL_VERSION_MISMATCH",
            "resolved_config.evaluation_plan.schema_version",
        ),
        (
            "plan_id",
            "EVALUATION_PLAN_ID_MISMATCH",
            "resolved_config.evaluation_plan.plan",
        ),
        (
            "plan_hash",
            "EVALUATION_PLAN_IDENTITY_MISMATCH",
            "identities.evaluation_plan.sha256",
        ),
        (
            "dataset_version",
            "DATASET_VERSION_MISMATCH",
            "resolved_config.dataset.version",
        ),
        ("dataset_content", "DATASET_CONTENT_MISMATCH", "dataset_fingerprints.files"),
        ("target", "TARGET_IDENTITY_MISMATCH", "dataset_fingerprints.target"),
        (
            "row_order",
            "SAMPLE_ORDER_IDENTITY_MISMATCH",
            "dataset_fingerprints.row_position_identity",
        ),
        ("outer_assignments", "OUTER_ASSIGNMENTS_MISMATCH", "splits.outer_assignments"),
        (
            "threshold_assignments",
            "THRESHOLD_ASSIGNMENTS_MISMATCH",
            "splits.threshold_selection_assignments",
        ),
        (
            "metrics",
            "METRIC_CONTRACT_MISMATCH",
            "resolved_config.evaluation_plan.metrics",
        ),
        (
            "aggregation",
            "METRIC_AGGREGATION_MISMATCH",
            "resolved_config.evaluation_plan.aggregation",
        ),
        (
            "threshold_policy",
            "THRESHOLD_POLICY_MISMATCH",
            "resolved_config.evaluation_plan.threshold_policy",
        ),
        (
            "threshold_grid",
            "THRESHOLD_GRID_MISMATCH",
            "resolved_config.evaluation_plan.threshold_policy.grid",
        ),
        (
            "positive_class",
            "POSITIVE_CLASS_SEMANTICS_MISMATCH",
            "resolved_config.evaluation_plan.dataset.target.positive_label",
        ),
        (
            "probability",
            "PROBABILITY_SEMANTICS_MISMATCH",
            "identities.candidate.canonical.probability_semantics",
        ),
        (
            "label",
            "LABEL_CONVENTION_MISMATCH",
            "resolved_config.evaluation_plan.threshold_policy.comparison",
        ),
        (
            "fold_product",
            "FOLD_PRODUCT_MISMATCH",
            "repeat_outer_fold_cartesian_product",
        ),
        (
            "prediction_keys_missing",
            "OUTER_PREDICTION_KEY_COVERAGE_MISMATCH",
            "predictions.outer_validation.keys",
        ),
        (
            "prediction_keys_duplicate",
            "OUTER_PREDICTION_KEY_COVERAGE_MISMATCH",
            "predictions.outer_validation.keys",
        ),
        (
            "threshold_keys_missing",
            "THRESHOLD_PREDICTION_KEY_COVERAGE_MISMATCH",
            "predictions.threshold_selection_oof.keys",
        ),
        (
            "threshold_keys_duplicate",
            "THRESHOLD_PREDICTION_KEY_COVERAGE_MISMATCH",
            "predictions.threshold_selection_oof.keys",
        ),
        (
            "outer_prediction_target",
            "OUTER_TARGET_ORDER_MISMATCH",
            "predictions.outer_validation.keys_and_target",
        ),
        (
            "threshold_prediction_target",
            "THRESHOLD_TARGET_ORDER_MISMATCH",
            "predictions.threshold_selection_oof.keys_and_target",
        ),
    ],
)
def test_complete_compatibility_matrix_has_exact_reason_and_field(
    tmp_path: Path, mutation: str, reason_code: str, field_path: str
) -> None:
    baseline = _synthetic_run(tmp_path, "baseline")
    candidate = _compatibility_mutation(_synthetic_run(tmp_path, "candidate"), mutation)
    issues = build_compatibility_summary(baseline, candidate).issues
    assert (reason_code, field_path) in {
        (issue.reason_code, issue.field_path) for issue in issues
    }


def test_allowed_identity_and_metadata_differences_remain_compatible(
    tmp_path: Path,
) -> None:
    baseline = _synthetic_run(tmp_path, "baseline")
    candidate = _synthetic_run(tmp_path, "candidate")
    metadata = deepcopy(candidate.metadata)
    metadata["created_at_utc"] = "2099-01-01T00:00:00+00:00"
    candidate = replace(candidate, metadata=metadata)
    report = build_compatibility_summary(baseline, candidate)
    assert report.compatible is True
    assert set(report.intentionally_ignored_contracts) == {
        "feature_pipeline_identity",
        "candidate_adapter_identity",
        "complete_candidate_identity",
        "source_provenance",
        "run_id",
        "timestamps",
    }


def test_all_decision_labels_and_policy_boundaries(tmp_path: Path) -> None:
    baseline = _synthetic_run(tmp_path, "baseline")
    candidate = _synthetic_run(tmp_path, "candidate", improved=True)
    promising = build_comparison_result(baseline, candidate, POLICY)
    assert promising.decision_report["status"] == "promising"
    mixed_policy = replace(POLICY, promising_min_mean_balanced_accuracy_delta=1.0)
    mixed = build_comparison_result(baseline, candidate, mixed_policy)
    assert mixed.decision_report["status"] == "mixed"
    unchanged_candidate = _synthetic_run(tmp_path, "unchanged")
    unchanged = build_comparison_result(baseline, unchanged_candidate, POLICY)
    assert unchanged.decision_report["status"] == "not_improved"
    assert unchanged.decision_report["repeat_wins_ties_losses"] == {
        "wins": 0,
        "ties": 2,
        "losses": 0,
    }
    boundary_policy = replace(
        POLICY,
        promising_min_mean_balanced_accuracy_delta=promising.decision_report[
            "candidate_mean_repeat_balanced_accuracy_delta"
        ],
        promising_min_repeat_win_fraction=1.0,
    )
    assert (
        build_comparison_result(baseline, candidate, boundary_policy).decision_report[
            "status"
        ]
        == "promising"
    )


def test_exact_numeric_regression_matrix(tmp_path: Path) -> None:
    baseline = _synthetic_run(tmp_path, "baseline")
    candidate = _synthetic_run(tmp_path, "candidate", improved=True)
    result = build_comparison_result(baseline, candidate, POLICY)
    repeat = result.paired_metrics.repeat_metrics
    assert repeat["balanced_accuracy_delta"].tolist() == [0.5, 0.5]
    assert repeat["sensitivity_delta"].tolist() == [0.5, 0.5]
    assert repeat["specificity_delta"].tolist() == [0.5, 0.5]
    assert repeat["roc_auc_delta"].tolist() == [0.25, 0.25]
    assert repeat["average_precision_delta"].tolist() == [
        0.16666666666666674,
        0.16666666666666674,
    ]
    assert repeat["brier_score_delta"].tolist() == [
        -0.18999999999999997,
        -0.18999999999999997,
    ]
    fold = result.paired_metrics.fold_metrics
    assert fold["balanced_accuracy_delta"].tolist() == [0.5, 0.5, 0.5, 0.5]
    assert fold["sensitivity_delta"].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert fold["specificity_delta"].tolist() == [0.0, 1.0, 0.0, 1.0]
    prediction = result.prediction_comparison
    assert prediction["pearson_correlation"]["coefficient"] == 0.447213595499958
    assert prediction["spearman_correlation"]["coefficient"] == 0.44721359549995804
    assert prediction["correctness_matrix"] == {
        "both_correct": {"count": 8, "rate": 0.5},
        "baseline_only_correct": {"count": 0, "rate": 0.0},
        "candidate_only_correct": {"count": 8, "rate": 0.5},
        "both_wrong": {"count": 0, "rate": 0.0},
    }
    assert prediction["disagreement_by_true_class"]["0"]["prediction_disagreement"] == {
        "count": 4,
        "rate": 0.5,
    }
    assert prediction["disagreement_by_true_class"]["1"]["prediction_disagreement"] == {
        "count": 4,
        "rate": 0.5,
    }
    blend_fold = result.blend.fold_metrics
    assert blend_fold["blend_threshold"].tolist() == [0.6, 0.4, 0.6, 0.4]
    assert blend_fold["blend_balanced_accuracy"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert result.blend.repeat_metrics["blend_brier_score"].tolist() == [
        0.07249999999999998,
        0.07249999999999998,
    ]
    assert (np.asarray([0.59, 0.6]) >= 0.6).astype("int8").tolist() == [0, 1]


def test_tie_epsilon_boundaries_use_production_classification() -> None:
    epsilon = 1.0e-12
    values = np.asarray(
        [
            epsilon,
            -epsilon,
            np.nextafter(epsilon, np.inf),
            np.nextafter(-epsilon, -np.inf),
        ]
    )
    assert _wins_ties_losses(values, epsilon, higher_is_better=True) == {
        "wins": 1,
        "ties": 2,
        "losses": 1,
    }


def test_cli_relative_paths_are_repository_rooted_from_different_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = _synthetic_run(tmp_path, "baseline")
    candidate = _synthetic_run(tmp_path, "candidate")
    policy = tmp_path / "policy.yaml"
    policy.write_text("synthetic", encoding="utf-8")
    seen: list[Path] = []

    def load(path: Path, *, project_root: Path, role: str) -> Any:
        resolved = resolve_repository_path(
            path,
            project_root=project_root,
            role=role,
            field_path=f"paths.{role}",
            must_exist=True,
            require_directory=True,
        )
        seen.append(resolved)
        return baseline if role == "baseline_run" else candidate

    monkeypatch.setattr("src.churn_ml.paired_comparison_cli.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "src.churn_ml.paired_comparison_cli.load_completed_research_v2_run", load
    )
    monkeypatch.setattr(
        "src.churn_ml.paired_comparison_cli.load_comparison_policy", lambda path: POLICY
    )
    outside_cwd = tmp_path / "outside-cwd"
    outside_cwd.mkdir()
    monkeypatch.chdir(outside_cwd)
    args = argparse.Namespace(
        baseline_run_dir=Path("runs/baseline"),
        candidate_run_dir=Path("runs/candidate"),
        comparison_id=None,
        output_root=Path("comparisons"),
        policy=Path("policy.yaml"),
        validate_only=True,
    )
    assert execute(args) == 0
    assert seen == [baseline.root, candidate.root]
    assert not (tmp_path / "comparisons").exists()


def test_documented_decision_contract_matches_versioned_policy() -> None:
    document = (REPOSITORY_ROOT / "docs/research-v2-paired-comparison.md").read_text(
        encoding="utf-8"
    )
    for required in (
        "mean_delta >= 0.0001",
        "wins / (wins + ties + losses) >= 0.5",
        "abs(delta) <= 1.0e-12",
        "mean_delta > 1.0e-12 or wins > losses",
        "threshold stability is advisory only",
        "does not affect the status label",
        "no significance interpretation",
    ):
        assert required in document
