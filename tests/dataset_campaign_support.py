"""Synthetic fixtures for Dataset Campaign Runner v1 tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.churn_ml.dataset_campaign.constants import UNBIASED_SCREENING_DATASET_IDS
from src.churn_ml.dataset_campaign.runner import CellRunResult
from tests.dataset_registry_support import (
    make_v0_frames,
    register_package,
    write_package_files,
)


SMOKE_PLAN = {
    "schema_version": 1,
    "plan": {"id": "synthetic_smoke_r1x3_t2_v1"},
    "dataset": {
        "processed_dir": "data/processed",
        "version": "v0_raw_minimal",
        "files": {
            "train_features": {"name": "X_train.parquet", "sha256": "0" * 64},
            "target": {"name": "y_train.parquet", "sha256": "0" * 64},
            "metadata": {"name": "metadata.json", "sha256": "0" * 64},
        },
        "expected_rows": 6,
        "expected_source_features": 3,
        "ordered_source_schema_sha256": "0" * 64,
        "ordered_dtype_schema_sha256": "0" * 64,
        "target": {
            "name": "y",
            "dtype": "int64",
            "negative_label": 0,
            "positive_label": 1,
            "expected_negative_rows": 3,
            "expected_positive_rows": 3,
            "values_sha256": "0" * 64,
        },
    },
    "outer_evaluation": {
        "splitter": "stratified_kfold",
        "n_splits": 3,
        "shuffle": True,
        "repeat_seeds": [0],
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
        "minimum": 0.01,
        "maximum": 0.99,
        "step": 0.001,
        "comparison": "greater_than_or_equal",
        "maximizer_absolute_tolerance": 1.0e-12,
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
        "diagnostic": [
            "tn",
            "fp",
            "fn",
            "tp",
            "predicted_positive_rate",
            "selected_threshold",
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


def write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def build_model_config(
    *,
    family: str,
    adapter_id: str,
    plan_path: str,
    stem: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "experiment": {"id": stem},
        "dataset": {"version": "v0_raw_minimal"},
        "evaluation_plan_path": plan_path,
        "feature_pipeline": {
            "id": "registered_prepared_passthrough_v1",
            "contract": {
                "mode": "registry_prepared_passthrough_v1",
                "drop": [],
                "keep_all_features": True,
            },
        },
        "candidate_adapter": {
            "id": adapter_id,
            "contract": {"synthetic": True, "family": family},
        },
        "artifacts": {"root": "artifacts/research_v2"},
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


def create_campaign_project(root: Path) -> dict[str, str]:
    """Create a synthetic repository with 7 unbiased packages + 3 model configs."""
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    X_train, y_train, X_test = make_v0_frames()

    parents: dict[str, str | None] = {
        "v0_raw_minimal": None,
        "v1_missingness_summary": "v0_raw_minimal",
        "v2_missingness_indicators": "v0_raw_minimal",
        "v4_zero_value_summary": "v0_raw_minimal",
        "v5_joint_missingness_pattern": "v1_missingness_summary",
        "v6_compact_missingness_indicators": "v1_missingness_summary",
        "v7_compact_zero_indicators": "v4_zero_value_summary",
    }

    frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
        "v0_raw_minimal": (X_train, X_test),
    }
    for dataset_id in UNBIASED_SCREENING_DATASET_IDS:
        if dataset_id == "v0_raw_minimal":
            continue
        parent_id = parents[dataset_id]
        assert parent_id is not None
        parent_train, parent_test = frames[parent_id]
        train = parent_train.copy()
        test = parent_test.copy()
        extra = f"{dataset_id}_flag"
        train[extra] = 0
        test[extra] = 0
        frames[dataset_id] = (train, test)

    order = [
        "v0_raw_minimal",
        "v1_missingness_summary",
        "v2_missingness_indicators",
        "v4_zero_value_summary",
        "v5_joint_missingness_pattern",
        "v6_compact_missingness_indicators",
        "v7_compact_zero_indicators",
    ]
    for dataset_id in order:
        train, test = frames[dataset_id]
        write_package_files(
            processed / dataset_id,
            dataset_id=dataset_id,
            X_train=train,
            y_train=y_train,
            X_test=test,
        )
        register_package(
            processed,
            dataset_id,
            parent_dataset_id=parents[dataset_id],
            hypothesis=f"synthetic {dataset_id}",
            target_dependency="none",
            summary_features=(),
            binary_indicator_features=(),
        )

    v1_train, v1_test = frames["v1_missingness_summary"]
    v3_train = v1_train.copy()
    v3_test = v1_test.copy()
    v3_train["Var217_is_missing"] = 0
    v3_test["Var217_is_missing"] = 0
    write_package_files(
        processed / "v3_targeted_missingness",
        dataset_id="v3_targeted_missingness",
        X_train=v3_train,
        y_train=y_train,
        X_test=v3_test,
    )
    register_package(
        processed,
        "v3_targeted_missingness",
        parent_dataset_id="v1_missingness_summary",
        hypothesis="synthetic exploratory v3",
        target_dependency="exploratory",
        binary_indicator_features=("Var217_is_missing",),
    )

    plan_rel = "configs/research_v2/plans/synthetic_smoke_r1x3_t2_v1.yaml"
    write_yaml(root / plan_rel, SMOKE_PLAN)

    configs = {
        "LightGBM": (
            "configs/research_v2/manual_lightgbm_te_v1_compat_smoke.yaml",
            "manual_lightgbm_te_v1_compat",
        ),
        "XGBoost": (
            "configs/research_v2/xgboost_numeric_v1_smoke.yaml",
            "xgboost_numeric_v1",
        ),
        "CatBoost": (
            "configs/research_v2/catboost_numeric_v1_smoke.yaml",
            "catboost_numeric_v1",
        ),
    }
    config_paths: dict[str, str] = {}
    for family, (rel, adapter_id) in configs.items():
        write_yaml(
            root / rel,
            build_model_config(
                family=family,
                adapter_id=adapter_id,
                plan_path=plan_rel,
                stem=Path(rel).stem,
            ),
        )
        config_paths[family] = rel

    (root / "artifacts" / "research_v2").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "dataset_campaigns").mkdir(parents=True, exist_ok=True)
    return config_paths


def campaign_payload(
    *,
    campaign_id: str = "test_screening_v1",
    campaign_type: str = "smoke",
    classification: str = "unbiased",
    dataset_ids: list[str] | None = None,
    preset: str | None = None,
    config_paths: dict[str, str] | None = None,
    evaluation_plan_path: str | None = (
        "configs/research_v2/plans/synthetic_smoke_r1x3_t2_v1.yaml"
    ),
    families: list[str] | None = None,
) -> dict[str, Any]:
    paths = config_paths or {
        "LightGBM": "configs/research_v2/manual_lightgbm_te_v1_compat_smoke.yaml",
        "XGBoost": "configs/research_v2/xgboost_numeric_v1_smoke.yaml",
        "CatBoost": "configs/research_v2/catboost_numeric_v1_smoke.yaml",
    }
    selected_families = families or list(paths.keys())
    models = [
        {"family": family, "config_path": paths[family]}
        for family in selected_families
    ]
    if preset is not None:
        datasets: dict[str, Any] = {"preset": preset}
    else:
        datasets = {
            "ids": dataset_ids
            or [
                "v0_raw_minimal",
                "v7_compact_zero_indicators",
            ]
        }
    return {
        "schema_version": "dataset_campaign_v1",
        "campaign": {
            "id": campaign_id,
            "name": "Synthetic campaign",
            "type": campaign_type,
            "classification": classification,
        },
        "datasets": datasets,
        "models": models,
        "evaluation_plan": {"path": evaluation_plan_path},
        "execution": {
            "policy": "sequential",
            "processed_root": "data/processed",
            "artifacts_root": "artifacts/dataset_campaigns",
            "index_mlflow": False,
            "mlflow_config_path": "configs/mlflow/local.yaml",
        },
    }


class FakeCellRunner:
    """Test double that never fits models."""

    def __init__(self) -> None:
        self.validated: list[str] = []
        self.runs: list[str] = []
        self.fail_prefixes: set[str] = set()
        self.overwrite_guard: set[str] = set()

    def validate(self, config_path: Path, *, project_root: Path) -> None:
        del project_root
        self.validated.append(str(config_path))

    def run(
        self,
        config_path: Path,
        *,
        project_root: Path,
        run_id: str | None = None,
    ) -> CellRunResult:
        key = run_id or str(config_path)
        if key in self.overwrite_guard:
            return CellRunResult(
                success=False,
                run_directory=None,
                metrics=None,
                threshold_summary=None,
                oof_relative=None,
                error="Refusing to overwrite existing run identity.",
            )
        self.overwrite_guard.add(key)
        self.runs.append(key)

        should_fail = any(prefix and prefix in key for prefix in self.fail_prefixes)
        if should_fail:
            run_dir = project_root / "artifacts" / "research_v2" / f"failed_{run_id}"
            run_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / "_FAILED").write_text("failed\n", encoding="utf-8")
            return CellRunResult(
                success=False,
                run_directory=str(run_dir.relative_to(project_root).as_posix()),
                metrics=None,
                threshold_summary=None,
                oof_relative=None,
                error="synthetic failure",
            )

        run_dir = project_root / "artifacts" / "research_v2" / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "predictions").mkdir()
        (run_dir / "metrics").mkdir()
        (run_dir / "thresholds").mkdir()
        oof = pd.DataFrame(
            {
                "repeat": [1, 1],
                "repeat_seed": [0, 0],
                "outer_fold": [1, 2],
                "row_position": [0, 1],
                "target": [0, 1],
                "probability": [0.1, 0.8],
            }
        )
        oof_path = run_dir / "predictions" / "outer_validation.parquet"
        oof.to_parquet(oof_path, index=False)
        metrics = {
            "balanced_accuracy": {
                "mean": 0.75,
                "sample_standard_deviation": 0.0,
                "minimum": 0.75,
                "maximum": 0.75,
            },
            "sensitivity": {
                "mean": 0.7,
                "sample_standard_deviation": None,
                "minimum": 0.7,
                "maximum": 0.7,
            },
            "specificity": {
                "mean": 0.8,
                "sample_standard_deviation": None,
                "minimum": 0.8,
                "maximum": 0.8,
            },
            "roc_auc": {
                "mean": 0.82,
                "sample_standard_deviation": None,
                "minimum": 0.82,
                "maximum": 0.82,
            },
            "average_precision": {
                "mean": 0.6,
                "sample_standard_deviation": None,
                "minimum": 0.6,
                "maximum": 0.6,
            },
            "brier_score": {
                "mean": 0.15,
                "sample_standard_deviation": None,
                "minimum": 0.15,
                "maximum": 0.15,
            },
        }
        (run_dir / "metrics" / "aggregate.json").write_text(
            json.dumps({"metrics": metrics}, indent=2) + "\n",
            encoding="utf-8",
        )
        threshold = {
            "count": 2,
            "minimum": 0.1,
            "median": 0.2,
            "maximum": 0.3,
        }
        (run_dir / "thresholds" / "threshold_summary.json").write_text(
            json.dumps(threshold, indent=2) + "\n",
            encoding="utf-8",
        )
        (run_dir / "_SUCCESS").write_text("ok\n", encoding="utf-8")
        return CellRunResult(
            success=True,
            run_directory=str(run_dir.relative_to(project_root).as_posix()),
            metrics=metrics,
            threshold_summary=threshold,
            oof_relative=str(oof_path.relative_to(project_root).as_posix()),
            error=None,
            mlflow_index_status=None,
        )
