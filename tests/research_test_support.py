from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.churn_ml.manual_lightgbm import ordered_feature_schema_sha256
from src.churn_ml.research_artifacts import ResearchArtifactStore
from src.churn_ml.research_config import ResearchConfig
from src.churn_ml.research_data import canonical_sha256, load_research_training_data
from src.churn_ml.research_evaluation import (
    ResearchEvaluationResult,
    run_research_evaluation,
)
from src.churn_ml.research_protocol import (
    build_candidate_contract_identity,
    build_evaluation_assignments,
    build_evaluation_plan_identity,
)
from src.churn_ml.research_provenance import (
    candidate_source_provenance,
    run_implementation_provenance,
)
from src.churn_ml.run_artifacts import fingerprint_file
from src.churn_ml.research_runtime_provenance import (
    build_invocation_provenance,
    build_loaded_module_provenance,
    collect_research_environment_versions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_persisted_test_run(
    tmp_path: Path,
) -> tuple[ResearchArtifactStore, ResearchEvaluationResult, dict[str, Any]]:
    config = _build_test_config(tmp_path)
    data = load_research_training_data(config)
    assignments = build_evaluation_assignments(data.y, config.plan_payload)
    plan_identity, plan_hash = build_evaluation_plan_identity(
        config.plan_payload,
        data.fingerprints,
        assignments,
    )
    candidate_sources, candidate_source_hash = candidate_source_provenance(PROJECT_ROOT)
    environment = collect_research_environment_versions()
    candidate_identity, candidate_hash = build_candidate_contract_identity(
        config.candidate_contract,
        feature_schema=data.feature_schema.to_dict(),
        source_provenance={
            **candidate_sources,
            "manifest_sha256": candidate_source_hash,
        },
        runtime_dependencies={
            name: environment[name]
            for name in (
                "numpy",
                "pandas",
                "scikit_learn",
                "lightgbm",
                "pyarrow",
            )
        },
    )
    source = config.candidate_contract["source"]
    run_identity, run_hash = run_implementation_provenance(
        PROJECT_ROOT,
        config_paths={
            "research_run": config.source_path,
            "evaluation_plan": config.plan_path,
            "historical_candidate": Path(source["historical_config_path"]),
            "baseline_manifest": Path(source["manifest_path"]),
        },
    )
    from src.churn_ml import research_cli  # noqa: F401

    process_started = research_cli.PROCESS_STARTED_AT_UTC
    loaded_identity, loaded_hash = build_loaded_module_provenance(
        PROJECT_ROOT,
        (candidate_sources, run_identity),
        entrypoint_module_name=None,
    )
    store = ResearchArtifactStore(
        config,
        plan_hash=plan_hash,
        candidate_hash=candidate_hash,
        candidate_source_manifest_hash=candidate_source_hash,
        run_implementation_hash=run_hash,
        loaded_module_hash=loaded_hash,
        run_id="unit-run",
    )
    metadata = {
        "experiment_id": config.experiment_id,
        "candidate_id": config.candidate_id,
        "plan_id": config.plan_id,
        "plan_sha256": plan_hash,
        "candidate_sha256": candidate_hash,
        "candidate_source_manifest_sha256": candidate_source_hash,
        "run_implementation_sha256": run_hash,
        "loaded_module_sha256": loaded_hash,
        "status": "running",
        "started_at_utc": store.started_at.isoformat(),
        "run_id": store.run_id,
        "config_source_path": str(config.source_path),
        "plan_source_path": str(config.plan_path),
        "environment": environment,
        "invocation": build_invocation_provenance(
            entry_point="tests/research_test_support.py",
            process_started_at_utc=process_started,
        ),
        "git": {
            "branch": "test",
            "commit": "0" * 40,
            "dirty": True,
            "status_porcelain": [],
            "untracked_implementation_files": [],
        },
        "upstream_feature_selection_limitation": "unit test fixture",
        "competition_assets_accessed": False,
    }
    store.save_initial_contract(
        metadata=metadata,
        plan_identity=plan_identity,
        candidate_identity=candidate_identity,
        run_implementation_identity=run_identity,
        loaded_module_identity=loaded_identity,
        fingerprints=data.fingerprints,
        feature_schema=data.feature_schema.to_dict(),
        assignments=assignments,
    )

    def feature_only_predictor(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_prediction: pd.DataFrame,
        candidate_contract: dict[str, Any],
        *,
        model_training_positions: np.ndarray,
        prediction_positions: np.ndarray,
        audit_callback: Any,
    ) -> np.ndarray:
        del X_train, y_train, candidate_contract
        if audit_callback is not None:
            audit_callback(model_training_positions, prediction_positions)
        row_ids = X_prediction["row_id"].to_numpy(dtype=int)
        return np.where(row_ids % 2 == 1, 0.8, 0.2)

    result = run_research_evaluation(
        data.X,
        data.y,
        assignments,
        config.plan_payload,
        config.candidate_contract,
        fit_predict=feature_only_predictor,
        on_outer_fold_complete=store.save_completed_fold,
    )
    store.save_final_outputs(result)
    return store, result, metadata


def _build_test_config(tmp_path: Path) -> ResearchConfig:
    version = "unit_v1"
    dataset_dir = tmp_path / "processed" / version
    dataset_dir.mkdir(parents=True)
    X = pd.DataFrame({"row_id": np.arange(8, dtype=float)})
    y = pd.Series(np.arange(8) % 2, name="y", dtype="int64")
    train_path = dataset_dir / "X_train.parquet"
    target_path = dataset_dir / "y_train.parquet"
    metadata_path = dataset_dir / "metadata.json"
    X.to_parquet(train_path, index=False)
    y.to_frame().to_parquet(target_path, index=False)
    metadata_path.write_text(
        json.dumps({"version": version}),
        encoding="utf-8",
    )

    schema_hash = ordered_feature_schema_sha256(X.columns.tolist())
    transformed_hash = ordered_feature_schema_sha256(["row_id"])
    dtype_hash = canonical_sha256(
        [{"name": name, "dtype": str(dtype)} for name, dtype in X.dtypes.items()]
    )
    target_hash = canonical_sha256(
        {"name": "y", "dtype": "int64", "values": y.tolist()}
    )
    plan = {
        "schema_version": 1,
        "plan": {"id": "unit-plan"},
        "dataset": {
            "processed_dir": str(tmp_path / "processed"),
            "version": version,
            "files": {
                "train_features": {
                    "name": train_path.name,
                    "sha256": fingerprint_file(train_path)["sha256"],
                },
                "target": {
                    "name": target_path.name,
                    "sha256": fingerprint_file(target_path)["sha256"],
                },
                "metadata": {
                    "name": metadata_path.name,
                    "sha256": fingerprint_file(metadata_path)["sha256"],
                },
            },
            "expected_rows": 8,
            "expected_source_features": 1,
            "ordered_source_schema_sha256": schema_hash,
            "ordered_dtype_schema_sha256": dtype_hash,
            "target": {
                "name": "y",
                "dtype": "int64",
                "negative_label": 0,
                "positive_label": 1,
                "expected_negative_rows": 4,
                "expected_positive_rows": 4,
                "values_sha256": target_hash,
            },
        },
        "outer_evaluation": {
            "splitter": "stratified_kfold",
            "n_splits": 2,
            "shuffle": True,
            "repeat_seeds": [0],
        },
        "threshold_selection": {
            "splitter": "stratified_kfold",
            "n_splits": 2,
            "shuffle": True,
            "random_state": 17,
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
    historical_path = tmp_path / "historical.yaml"
    manifest_path = tmp_path / "manifest.yaml"
    run_path = tmp_path / "run.yaml"
    plan_path = tmp_path / "plan.yaml"
    historical_path.write_text("kind: unit\n", encoding="utf-8")
    manifest_path.write_text("kind: unit\n", encoding="utf-8")
    run_path.write_text("kind: unit\n", encoding="utf-8")
    plan_path.write_text("kind: unit\n", encoding="utf-8")

    payload = {
        "schema_version": 1,
        "experiment": {"id": "unit-experiment"},
        "evaluation_plan_path": str(plan_path),
        "candidate": {
            "id": "unit-candidate",
            "implementation": "manual_lightgbm_r31",
            "historical_config_path": str(historical_path),
            "manifest_path": str(manifest_path),
            "manifest_key": "unit",
        },
        "artifacts": {
            "root": str(tmp_path / "runs"),
            "save_threshold_curves": True,
            "save_models": False,
        },
    }
    candidate_contract = {
        "candidate_id": "unit-candidate",
        "implementation": "manual_lightgbm_r31",
        "source": {
            "historical_config_path": str(historical_path),
            "manifest_path": str(manifest_path),
            "manifest_key": "unit",
        },
        "features": {
            "drop": [],
            "expected_model_feature_count": 1,
            "expected_categorical_count": 0,
            "expected_source_schema_sha256": schema_hash,
            "expected_model_input_schema_sha256": schema_hash,
            "expected_transformed_schema_sha256": transformed_hash,
            "categorical": [],
        },
        "target_encoder": {
            "implementation": "unit",
            "inner_splits": 2,
            "shuffle": True,
            "random_state": 42,
            "alpha": 10.0,
            "prior": "unit",
            "keep_original_categorical_features": False,
        },
        "lightgbm": {
            "estimator": "LGBMClassifier",
            "parameters": {"n_estimators": 1},
        },
    }
    return ResearchConfig(
        payload=payload,
        plan_payload=plan,
        candidate_contract=candidate_contract,
        source_path=run_path,
        plan_path=plan_path,
        project_root=PROJECT_ROOT,
    )
