from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.churn_ml.dataset_comparison_v1 import (
    DEFAULT_OUTPUT_ROOT,
    EXPLORATORY_PARENT_CHILD_PAIRS,
    UNBIASED_PARENT_CHILD_PAIRS,
    CompletedResearchV2Run,
    DatasetCompatibilityError,
    build_comparison_result,
    build_compatibility_summary,
    build_evaluation_protocol_identity,
    build_independent_row_identity,
    build_model_configuration_identity,
    default_dataset_comparison_id,
    is_exploratory_dataset,
    resolve_dataset_package_manifest,
)
from src.churn_ml.dataset_comparison_v1_artifacts import (
    DatasetComparisonArtifactError,
    create_comparison_artifacts,
    validate_comparison_artifacts,
)
from src.churn_ml.dataset_comparison_v1_cli import execute
from src.churn_ml.dataset_registry.constants import MANIFEST_FILENAME, SCHEMA_VERSION
from src.churn_ml.research_data import canonical_sha256
from src.churn_ml.research_protocol import (
    build_repeat_metrics,
    calculate_outer_metrics,
)
from src.churn_ml.research_v2_config import ResearchV2Config

from tests.test_paired_comparison_v1 import (
    _plan,
    _probability,
    _typed_outer,
    _typed_threshold,
)


ANCHOR_HASH = "a" * 64
POSITION_IDENTITY = {
    "kind": "contiguous_zero_based",
    "sha256": canonical_sha256(list(range(8))),
}
TARGET_FP = {
    "name": "y",
    "dtype": "int8",
    "negative_count": 4,
    "positive_count": 4,
    "values_sha256": canonical_sha256([0, 1, 0, 1, 0, 1, 0, 1]),
}
TARGET_HASH = "b" * 64
ADAPTER_ID = "manual_lightgbm_te_v1_compat"
PIPELINE_ID = "registered_prepared_passthrough_v1"


def _hex(seed: str) -> str:
    return (seed * 64)[:64]


def _manifest_payload(
    dataset_id: str,
    *,
    parent_dataset_id: str | None,
    train_hash: str,
    target_dependency: str = "none",
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "parent_dataset_id": parent_dataset_id,
        "hypothesis": f"test package {dataset_id}",
        "files": {
            "X_train": "X_train.parquet",
            "y_train": "y_train.parquet",
            "X_test": "X_test.parquet",
            "metadata": "metadata.json",
        },
        "train_row_count": 8,
        "test_row_count": 3,
        "n_features": 1,
        "features": [{"name": "x", "dtype": "float64", "role": "numeric"}],
        "transformations": [{"type": "synthetic"}],
        "target_dependency": target_dependency,
        "schema_hash": _hex("c"),
        "content_hashes": {
            "X_train": train_hash,
            "y_train": _hex("d"),
            "X_test": _hex("e"),
            "metadata": _hex("f"),
        },
        "target": {
            "name": "y",
            "dtype": "int8",
            "class_counts": {"0": 4, "1": 4},
            "hash": TARGET_HASH,
        },
        "row_identity": {
            "train_hash": train_hash,
            "test_hash": _hex("0"),
            "alignment_status": "proven",
            "alignment_method": "ordered_v0_raw_minimal_feature_projection",
            "train_anchor_hash": ANCHOR_HASH,
            "test_anchor_hash": _hex("1"),
        },
    }


def _write_manifest(
    project_root: Path, dataset_id: str, payload: dict[str, Any]
) -> None:
    package_dir = project_root / "data" / "processed" / dataset_id
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / MANIFEST_FILENAME).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _candidate_canonical(
    *, adapter_id: str = ADAPTER_ID, pipeline_id: str = PIPELINE_ID
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "dataset_version": "ignored_for_model_hash",
        "feature_pipeline": {"id": pipeline_id, "sha256": "1" * 64},
        "candidate_adapter": {"id": adapter_id, "sha256": "2" * 64},
        "probability_semantics": "binary_positive_class_label_1",
        "evaluation_boundary": "adapter_receives_fold_local_data_only",
    }


def _synthetic_dataset_run(
    project_root: Path,
    run_id: str,
    *,
    dataset_id: str,
    parent_dataset_id: str | None,
    train_hash: str,
    improved: bool = False,
    adapter_id: str = ADAPTER_ID,
    pipeline_id: str = PIPELINE_ID,
    adapter_contract: dict[str, Any] | None = None,
    target_dependency: str = "none",
) -> CompletedResearchV2Run:
    _write_manifest(
        project_root,
        dataset_id,
        _manifest_payload(
            dataset_id,
            parent_dataset_id=parent_dataset_id,
            train_hash=train_hash,
            target_dependency=target_dependency,
        ),
    )
    root = project_root / "runs" / run_id
    root.mkdir(parents=True)
    manifest = {
        "schema_version": 2,
        "hashing_method": "synthetic_test_fixture",
        "files": [],
        "manifest_sha256": canonical_sha256(
            {"run_id": run_id, "dataset_id": dataset_id}
        ),
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
                    row, int(target[row]), improved=improved, constant=False
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
                    row, int(target[row]), improved=improved, constant=False
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
    plan["dataset"]["version"] = dataset_id
    plan["plan"]["id"] = f"{dataset_id}__development_plan"
    plan_identity = {
        "schema_version": 1,
        "sha256": canonical_sha256(plan),
        "canonical": plan,
    }
    candidate_canonical = _candidate_canonical(
        adapter_id=adapter_id, pipeline_id=pipeline_id
    )
    candidate_identity = {
        "sha256": canonical_sha256(candidate_canonical),
        "canonical": candidate_canonical,
    }
    default_adapter_contract = {
        "target_encoder": {
            "implementation": "custom_autogluon_compatible_binary_oof_target_encoder",
            "inner_splits": 5,
            "shuffle": True,
            "random_state": 42,
            "alpha": 10.0,
        },
        "lightgbm": {
            "estimator": "LGBMClassifier",
            "parameters": {"n_estimators": 100, "learning_rate": 0.05},
        },
    }
    fingerprints = {
        "dataset_version": dataset_id,
        "files": {
            "train_features": {"sha256": train_hash},
            "target": {"sha256": TARGET_FP["values_sha256"]},
            "metadata": {"sha256": _hex("2")},
        },
        "row_count": 8,
        "row_position_identity": deepcopy(POSITION_IDENTITY),
        "source_schema": {
            "ordered_names": ["x"],
            "ordered_names_sha256": _hex("3"),
        },
        "target": deepcopy(TARGET_FP),
    }
    payload = {
        "schema_version": 2,
        "experiment": {"id": f"{dataset_id}__{adapter_id}__development"},
        "dataset": {"version": dataset_id},
        "evaluation_plan_path": "configs/synthetic_plan.yaml",
        "feature_pipeline": {
            "id": pipeline_id,
            "contract": {"prepared_passthrough": True},
        },
        "candidate_adapter": {
            "id": adapter_id,
            "contract": adapter_contract or default_adapter_contract,
        },
        "artifacts": {"root": "runs"},
        "persistence": {
            "resolved_config": True,
            "identities": True,
            "assignments": True,
            "predictions": True,
            "metrics": True,
            "threshold_curves": True,
            "models": False,
        },
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


@pytest.fixture
def dataset_runs(
    tmp_path: Path,
) -> tuple[CompletedResearchV2Run, CompletedResearchV2Run]:
    baseline = _synthetic_dataset_run(
        tmp_path,
        "baseline",
        dataset_id="v0_raw_minimal",
        parent_dataset_id=None,
        train_hash=_hex("4"),
    )
    candidate = _synthetic_dataset_run(
        tmp_path,
        "candidate",
        dataset_id="v1_missingness_summary",
        parent_dataset_id="v0_raw_minimal",
        train_hash=_hex("5"),
        improved=True,
    )
    return baseline, candidate


def test_default_id_and_pair_constants() -> None:
    comparison_id = default_dataset_comparison_id(
        "lightgbm",
        "v0_raw_minimal",
        "v1_missingness_summary",
    )
    assert comparison_id == "lightgbm__v0_raw_minimal__vs__v1_missingness_summary"
    assert DEFAULT_OUTPUT_ROOT.as_posix() == (
        "artifacts/research_v2_dataset_comparisons"
    )
    assert len(UNBIASED_PARENT_CHILD_PAIRS) == 6
    assert EXPLORATORY_PARENT_CHILD_PAIRS == (
        ("v1_missingness_summary", "v3_targeted_missingness"),
    )
    assert is_exploratory_dataset("v3_targeted_missingness", "exploratory") is True
    assert is_exploratory_dataset("v0_raw_minimal", "none") is False


def test_independent_row_identity_uses_anchor_not_content_hash(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        "v0_raw_minimal",
        _manifest_payload(
            "v0_raw_minimal",
            parent_dataset_id=None,
            train_hash=_hex("4"),
        ),
    )
    manifest = resolve_dataset_package_manifest(
        "v0_raw_minimal",
        project_root=tmp_path,
    )
    _write_manifest(
        tmp_path,
        "v1_missingness_summary",
        _manifest_payload(
            "v1_missingness_summary",
            parent_dataset_id="v0_raw_minimal",
            train_hash=_hex("9"),
        ),
    )
    child_manifest = resolve_dataset_package_manifest(
        "v1_missingness_summary",
        project_root=tmp_path,
    )
    baseline_identity = build_independent_row_identity(
        manifest,
        POSITION_IDENTITY,
        8,
    )
    child_identity = build_independent_row_identity(
        child_manifest,
        POSITION_IDENTITY,
        8,
    )
    assert baseline_identity["train_anchor_hash"] == child_identity["train_anchor_hash"]
    assert manifest.row_identity.train_hash != child_manifest.row_identity.train_hash


def test_normalized_protocol_and_model_hashes_ignore_dataset_fields(
    dataset_runs: tuple[CompletedResearchV2Run, CompletedResearchV2Run],
) -> None:
    baseline, candidate = dataset_runs
    baseline_protocol, baseline_hash = build_evaluation_protocol_identity(
        baseline.config.plan_payload,
        baseline.outer_assignments,
        baseline.threshold_assignments,
        fold_product=[(1, 1), (1, 2), (2, 1), (2, 2)],
    )
    candidate_protocol, candidate_hash = build_evaluation_protocol_identity(
        candidate.config.plan_payload,
        candidate.outer_assignments,
        candidate.threshold_assignments,
        fold_product=[(1, 1), (1, 2), (2, 1), (2, 2)],
    )
    assert baseline_hash == candidate_hash
    assert baseline_protocol["threshold_grid"]
    baseline_model, baseline_model_hash = build_model_configuration_identity(baseline)
    candidate_model, candidate_model_hash = build_model_configuration_identity(
        candidate
    )
    assert baseline_model_hash == candidate_model_hash
    assert "dataset_version" not in baseline_model
    assert baseline_model["model_family"] == "lightgbm"


def test_compatible_parent_child_summary(
    tmp_path: Path,
    dataset_runs: tuple[CompletedResearchV2Run, CompletedResearchV2Run],
) -> None:
    baseline, candidate = dataset_runs
    summary = build_compatibility_summary(
        baseline,
        candidate,
        project_root=tmp_path,
    )
    assert summary.compatible is True
    assert summary.parent_child_relation == "baseline_is_parent"
    assert summary.exploratory is False
    assert summary.normalized_identities is not None
    assert "dataset_id" in summary.expected_differences


@pytest.mark.parametrize(
    ("mutation", "reason_code"),
    [
        ("same_dataset", "SAME_DATASET_ID"),
        ("anchor", "INDEPENDENT_ROW_IDENTITY_MISMATCH"),
        ("target", "TARGET_IDENTITY_MISMATCH"),
        ("adapter", "MODEL_CONFIGURATION_MISMATCH"),
        ("assignments", "OUTER_ASSIGNMENTS_MISMATCH"),
    ],
)
def test_compatibility_blockers(
    tmp_path: Path,
    dataset_runs: tuple[CompletedResearchV2Run, CompletedResearchV2Run],
    mutation: str,
    reason_code: str,
) -> None:
    baseline, candidate = dataset_runs
    if mutation == "same_dataset":
        candidate = _synthetic_dataset_run(
            tmp_path,
            "candidate_same",
            dataset_id="v0_raw_minimal",
            parent_dataset_id=None,
            train_hash=_hex("4"),
        )
    elif mutation == "anchor":
        _write_manifest(
            tmp_path,
            "v1_missingness_summary",
            _manifest_payload(
                "v1_missingness_summary",
                parent_dataset_id="v0_raw_minimal",
                train_hash=_hex("5"),
            ),
        )
        payload = _manifest_payload(
            "v1_missingness_summary",
            parent_dataset_id="v0_raw_minimal",
            train_hash=_hex("5"),
        )
        payload["row_identity"]["train_anchor_hash"] = _hex("9")
        _write_manifest(tmp_path, "v1_missingness_summary", payload)
    elif mutation == "target":
        candidate = deepcopy(candidate)
        fingerprints = dict(candidate.dataset_fingerprints)
        target = dict(fingerprints["target"])
        target["values_sha256"] = _hex("8")
        fingerprints["target"] = target
        candidate = CompletedResearchV2Run(
            root=candidate.root,
            config=candidate.config,
            metadata=candidate.metadata,
            manifest=candidate.manifest,
            dataset_fingerprints=fingerprints,
            evaluation_plan_identity=candidate.evaluation_plan_identity,
            candidate_identity=candidate.candidate_identity,
            outer_assignments=candidate.outer_assignments,
            threshold_assignments=candidate.threshold_assignments,
            outer_predictions=candidate.outer_predictions,
            threshold_predictions=candidate.threshold_predictions,
            selected_thresholds=candidate.selected_thresholds,
            fold_metrics=candidate.fold_metrics,
            repeat_metrics=candidate.repeat_metrics,
        )
    elif mutation == "adapter":
        candidate = _synthetic_dataset_run(
            tmp_path,
            "candidate_other_adapter",
            dataset_id="v1_missingness_summary",
            parent_dataset_id="v0_raw_minimal",
            train_hash=_hex("5"),
            adapter_id="xgboost_numeric_v1",
            adapter_contract={
                "numeric_features": {"random_state": 42},
                "xgboost": {"parameters": {"n_estimators": 100}},
            },
        )
    elif mutation == "assignments":
        assignments = candidate.outer_assignments.copy()
        assignments.loc[0, "row_position"] = 99
        candidate = CompletedResearchV2Run(
            root=candidate.root,
            config=candidate.config,
            metadata=candidate.metadata,
            manifest=candidate.manifest,
            dataset_fingerprints=candidate.dataset_fingerprints,
            evaluation_plan_identity=candidate.evaluation_plan_identity,
            candidate_identity=candidate.candidate_identity,
            outer_assignments=assignments,
            threshold_assignments=candidate.threshold_assignments,
            outer_predictions=candidate.outer_predictions,
            threshold_predictions=candidate.threshold_predictions,
            selected_thresholds=candidate.selected_thresholds,
            fold_metrics=candidate.fold_metrics,
            repeat_metrics=candidate.repeat_metrics,
        )
    summary = build_compatibility_summary(
        baseline,
        candidate,
        project_root=tmp_path,
    )
    assert summary.compatible is False
    assert any(issue.reason_code == reason_code for issue in summary.issues)


def test_row_identity_unavailable_when_manifest_missing(
    tmp_path: Path,
    dataset_runs: tuple[CompletedResearchV2Run, CompletedResearchV2Run],
) -> None:
    baseline, candidate = dataset_runs
    manifest_path = (
        tmp_path / "data/processed/v1_missingness_summary" / MANIFEST_FILENAME
    )
    manifest_path.unlink()
    summary = build_compatibility_summary(
        baseline,
        candidate,
        project_root=tmp_path,
    )
    assert summary.compatible is False
    assert any(
        issue.reason_code == "ROW_IDENTITY_UNAVAILABLE" for issue in summary.issues
    )


def test_exploratory_flag_for_v3(
    tmp_path: Path,
    dataset_runs: tuple[CompletedResearchV2Run, CompletedResearchV2Run],
) -> None:
    baseline, _ = dataset_runs
    candidate = _synthetic_dataset_run(
        tmp_path,
        "candidate_v3",
        dataset_id="v3_targeted_missingness",
        parent_dataset_id="v1_missingness_summary",
        train_hash=_hex("6"),
        target_dependency="exploratory",
    )
    summary = build_compatibility_summary(
        baseline,
        candidate,
        project_root=tmp_path,
    )
    assert summary.exploratory is True


def test_comparison_metrics_and_oof_diagnostics(
    tmp_path: Path,
    dataset_runs: tuple[CompletedResearchV2Run, CompletedResearchV2Run],
) -> None:
    baseline, candidate = dataset_runs
    result = build_comparison_result(
        baseline,
        candidate,
        project_root=tmp_path,
    )
    balanced = result.paired_metrics.aggregate_summary["metrics"]["balanced_accuracy"]
    assert balanced["candidate_minus_baseline"]["mean"] > 0
    assert result.oof_diagnostics["prediction_label_agreement_rate"] <= 1.0
    assert result.fit_time_delta["available"] is False


def test_incompatible_runs_raise(
    tmp_path: Path,
    dataset_runs: tuple[CompletedResearchV2Run, CompletedResearchV2Run],
) -> None:
    baseline, candidate = dataset_runs
    same = _synthetic_dataset_run(
        tmp_path,
        "candidate_same",
        dataset_id="v0_raw_minimal",
        parent_dataset_id=None,
        train_hash=_hex("4"),
    )
    with pytest.raises(DatasetCompatibilityError):
        build_comparison_result(baseline, same, project_root=tmp_path)


def _patch_artifact_validation(
    monkeypatch: pytest.MonkeyPatch,
    baseline: CompletedResearchV2Run,
    candidate: CompletedResearchV2Run,
) -> None:
    def load(
        path: Path, *, project_root: Path, role: str = "input_run"
    ) -> CompletedResearchV2Run:
        del project_root, role
        return baseline if path.resolve().name == "baseline" else candidate

    provenance = {
        "schema_version": 1,
        "operation": "dataset_comparison_v1",
        "hashing_method": "synthetic",
        "files": [],
        "sha256": canonical_sha256({"synthetic": True}),
    }
    monkeypatch.setattr(
        "src.churn_ml.dataset_comparison_v1_artifacts.load_completed_research_v2_run",
        load,
    )
    monkeypatch.setattr(
        "src.churn_ml.dataset_comparison_v1_artifacts.comparison_source_provenance",
        lambda project_root: provenance,
    )


def test_artifact_lifecycle_and_existing_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset_runs: tuple[CompletedResearchV2Run, CompletedResearchV2Run],
) -> None:
    baseline, candidate = dataset_runs
    _patch_artifact_validation(monkeypatch, baseline, candidate)
    result = build_comparison_result(
        baseline,
        candidate,
        project_root=tmp_path,
    )
    comparison_root = create_comparison_artifacts(
        result=result,
        project_root=tmp_path,
        output_root=Path("dataset_comparisons"),
        comparison_id="lightgbm__v0__vs__v1",
    )
    validate_comparison_artifacts(
        comparison_root,
        project_root=tmp_path,
        require_success=True,
        verify_manifest=True,
    )
    assert (comparison_root / "_SUCCESS").is_file()
    assert (comparison_root / "oof_diagnostics.json").is_file()
    with pytest.raises(DatasetComparisonArtifactError, match="already exists"):
        create_comparison_artifacts(
            result=result,
            project_root=tmp_path,
            output_root=Path("dataset_comparisons"),
            comparison_id="lightgbm__v0__vs__v1",
        )


def test_validate_only_allocates_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset_runs: tuple[CompletedResearchV2Run, CompletedResearchV2Run],
) -> None:
    baseline, candidate = dataset_runs
    calls = iter((baseline, candidate))
    monkeypatch.setattr(
        "src.churn_ml.dataset_comparison_v1_cli.load_completed_research_v2_run",
        lambda *args, **kwargs: next(calls),
    )
    monkeypatch.setattr(
        "src.churn_ml.dataset_comparison_v1_cli.PROJECT_ROOT",
        tmp_path,
    )
    output = tmp_path / "must_not_exist"
    args = argparse.Namespace(
        baseline_run_dir=baseline.root,
        candidate_run_dir=candidate.root,
        comparison_id=None,
        output_root=output,
        validate_only=True,
    )
    assert execute(args) == 0
    assert not output.exists()
