from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.churn_ml.research_protocol import (
    ResearchProtocolError,
    aggregate_repeat_metrics,
    build_evaluation_assignments,
    build_evaluation_plan_identity,
    calculate_outer_metrics,
    select_balanced_accuracy_threshold,
)


def _plan() -> dict:
    return {
        "plan": {"id": "unit-plan"},
        "outer_evaluation": {
            "splitter": "stratified_kfold",
            "n_splits": 5,
            "shuffle": True,
            "repeat_seeds": [0, 17],
        },
        "threshold_selection": {
            "splitter": "stratified_kfold",
            "n_splits": 4,
            "shuffle": True,
            "random_state": 314159,
        },
        "threshold_policy": _policy(),
        "metrics": {
            "primary": "balanced_accuracy",
            "secondary": ["roc_auc"],
            "diagnostic": ["tn"],
        },
        "aggregation": {
            "repeat_method": "pooled_predictions",
            "aggregate_unit": "repeat",
            "standard_deviation": "sample",
            "fold_standard_deviation": "descriptive_only",
            "confidence_interval": "none",
        },
    }


def _policy() -> dict:
    return {
        "id": "grid_balanced_accuracy_v1",
        "metric": "balanced_accuracy",
        "minimum": 0.01,
        "maximum": 0.99,
        "step": 0.001,
        "comparison": "greater_than_or_equal",
        "maximizer_absolute_tolerance": 1e-12,
        "tie_break": "median_maximizer_lower_on_even",
        "constant_probability_fallback": 0.5,
    }


def _target() -> pd.Series:
    return pd.Series(([0] * 80) + ([1] * 20), name="y")


def test_repeated_outer_and_threshold_assignments_are_deterministic() -> None:
    first = build_evaluation_assignments(_target(), _plan())
    second = build_evaluation_assignments(_target(), _plan())

    pd.testing.assert_frame_equal(first.outer, second.outer)
    pd.testing.assert_frame_equal(
        first.threshold_selection,
        second.threshold_selection,
    )
    assert len(first.outer) == 200
    assert len(first.threshold_selection) == 800


def test_every_row_is_outer_validation_once_per_repeat_without_overlap() -> None:
    assignments = build_evaluation_assignments(_target(), _plan())
    all_rows = set(range(100))
    for repeat in (1, 2):
        repeat_frame = assignments.outer.loc[assignments.outer["repeat"] == repeat]
        assert repeat_frame["row_position"].value_counts().eq(1).all()
        for outer_fold in range(1, 6):
            validation = set(
                repeat_frame.loc[
                    repeat_frame["outer_fold"] == outer_fold,
                    "row_position",
                ]
            )
            inner = set(
                assignments.threshold_selection.loc[
                    (assignments.threshold_selection["repeat"] == repeat)
                    & (assignments.threshold_selection["outer_fold"] == outer_fold),
                    "row_position",
                ]
            )
            assert not validation & inner
            assert inner == all_rows - validation


def test_threshold_selection_uses_lower_median_maximizer() -> None:
    policy = _policy()
    policy.update({"minimum": 0.1, "maximum": 0.9, "step": 0.1})
    result = select_balanced_accuracy_threshold(
        np.array([0, 1]),
        np.array([0.2, 0.8]),
        policy,
    )

    assert result.threshold == 0.5
    assert result.balanced_accuracy == 1.0


def test_threshold_equality_is_inclusive() -> None:
    threshold = 0.5
    metrics = calculate_outer_metrics(
        np.array([0, 1, 1]),
        np.array([0.49, threshold, 0.51]),
        (np.array([0.49, threshold, 0.51]) >= threshold).astype("int8"),
        selected_threshold=threshold,
    )

    assert metrics["fn"] == 0
    assert metrics["tp"] == 2


def test_constant_probabilities_use_explicit_fallback() -> None:
    result = select_balanced_accuracy_threshold(
        np.array([0, 0, 1, 1]),
        np.full(4, 0.2),
        _policy(),
    )

    assert result.threshold == 0.5
    assert result.degenerate is True
    assert result.status == "degenerate_constant_probabilities"


@pytest.mark.parametrize(
    ("targets", "probabilities"),
    [
        ([0, 0], [0.1, 0.2]),
        ([0, 1], [np.nan, 0.2]),
        ([0, 1], [np.inf, 0.2]),
        ([0, 1], [-0.1, 0.2]),
        ([0, 1], [0.1, 1.1]),
        ([0, None], [0.1, 0.2]),
    ],
)
def test_invalid_threshold_inputs_fail(targets: list, probabilities: list) -> None:
    with pytest.raises(ResearchProtocolError):
        select_balanced_accuracy_threshold(
            np.asarray(targets, dtype=object),
            np.asarray(probabilities, dtype=float),
            _policy(),
        )


def test_repeat_aggregation_uses_sample_standard_deviation() -> None:
    repeat_metrics = pd.DataFrame(
        {
            "balanced_accuracy": [0.8, 0.9],
            "sensitivity": [0.7, 0.9],
            "specificity": [0.9, 0.9],
            "tn": [90, 90],
            "fp": [10, 10],
            "fn": [30, 10],
            "tp": [70, 90],
            "predicted_positive_rate": [0.4, 0.5],
            "roc_auc": [0.85, 0.95],
            "average_precision": [0.75, 0.85],
            "brier_score": [0.15, 0.1],
        }
    )
    fold_metrics = pd.concat([repeat_metrics] * 2, ignore_index=True)

    aggregate = aggregate_repeat_metrics(repeat_metrics, fold_metrics)

    assert aggregate["metrics"]["balanced_accuracy"]["mean"] == pytest.approx(0.85)
    assert aggregate["metrics"]["balanced_accuracy"][
        "sample_standard_deviation"
    ] == pytest.approx(np.std([0.8, 0.9], ddof=1))
    assert aggregate["confidence_interval"] is None


def test_plan_hash_is_stable_and_sensitive_to_plan_data_and_assignments() -> None:
    plan = _plan()
    assignments = build_evaluation_assignments(_target(), plan)
    fingerprints = {
        "dataset_version": "unit",
        "files": {"train": {"sha256": "a" * 64}},
        "row_count": 100,
        "row_position_identity": {"kind": "contiguous", "sha256": "b" * 64},
        "source_schema": {
            "ordered_names": ["feature"],
            "ordered_names_sha256": "c" * 64,
            "ordered_dtypes": [{"name": "feature", "dtype": "float64"}],
            "ordered_dtypes_sha256": "d" * 64,
        },
        "target": {"name": "y", "values_sha256": "e" * 64},
        "candidate_feature_schema": {
            "drop": ["candidate_only"],
            "categorical": ["candidate_only"],
        },
    }
    canonical, digest = build_evaluation_plan_identity(
        plan,
        fingerprints,
        assignments,
    )
    canonical_again, digest_again = build_evaluation_plan_identity(
        deepcopy(plan),
        deepcopy(fingerprints),
        assignments,
    )
    candidate_changed = deepcopy(fingerprints)
    candidate_changed["candidate_feature_schema"] = {
        "drop": ["different"],
        "categorical": [],
        "model": ["different"],
        "transformed": ["different__te"],
    }
    _, candidate_digest = build_evaluation_plan_identity(
        plan, candidate_changed, assignments
    )
    changed_plan = deepcopy(plan)
    changed_plan["outer_evaluation"]["repeat_seeds"][0] = 1
    _, seed_digest = build_evaluation_plan_identity(
        changed_plan,
        fingerprints,
        assignments,
    )
    changed_fingerprints = deepcopy(fingerprints)
    changed_fingerprints["row_count"] = 101
    _, data_digest = build_evaluation_plan_identity(
        plan,
        changed_fingerprints,
        assignments,
    )
    changed_assignments = deepcopy(assignments)
    changed_assignments.outer.loc[0, "outer_fold"] = 99
    _, assignment_digest = build_evaluation_plan_identity(
        plan,
        fingerprints,
        changed_assignments,
    )
    changed_policy = deepcopy(plan)
    changed_policy["threshold_policy"]["minimum"] = 0.02
    _, policy_digest = build_evaluation_plan_identity(
        changed_policy,
        fingerprints,
        assignments,
    )

    changed_source = deepcopy(fingerprints)
    changed_source["source_schema"]["ordered_dtypes"][0]["dtype"] = "int64"
    _, source_digest = build_evaluation_plan_identity(plan, changed_source, assignments)
    changed_target = deepcopy(fingerprints)
    changed_target["target"]["values_sha256"] = "f" * 64
    _, target_digest = build_evaluation_plan_identity(plan, changed_target, assignments)
    changed_metrics = deepcopy(plan)
    changed_metrics["metrics"]["secondary"] = ["average_precision"]
    _, metrics_digest = build_evaluation_plan_identity(
        changed_metrics, fingerprints, assignments
    )
    changed_aggregation = deepcopy(plan)
    changed_aggregation["aggregation"]["standard_deviation"] = "population"
    _, aggregation_digest = build_evaluation_plan_identity(
        changed_aggregation, fingerprints, assignments
    )

    assert canonical == canonical_again
    assert digest == digest_again
    assert candidate_digest == digest
    assert (
        len({digest, seed_digest, data_digest, assignment_digest, policy_digest}) == 5
    )
    assert all(
        changed != digest
        for changed in (
            source_digest,
            target_digest,
            metrics_digest,
            aggregation_digest,
        )
    )
