from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.churn_ml.features import load_dataset
from src.churn_ml.config import (
    ManualExperimentConfig,
    load_manual_experiment_config,
    validate_config_against_manifest,
)
from src.churn_ml.manual_lightgbm import (
    FeatureSchema,
    ParityReport,
    build_submission,
    evaluate_parity,
    evaluate_predictions,
    prepare_model_features,
    run_manual_lightgbm_cross_validation,
)
from src.churn_ml.run_artifacts import (
    RunArtifactStore,
    RunFailureFinalizationError,
    collect_environment_versions,
    collect_git_state,
    fingerprint_file,
    utc_now,
    validate_environment_contract,
)


class RequiredParityError(RuntimeError):
    """Raised after artifacts are saved when required parity fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the manual LightGBMPrep_r31 reproduction."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the experiment YAML configuration.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration, environment, and local inputs without training.",
    )
    parser.add_argument(
        "--run-id",
        help="Optional unique run identifier; existing directories are rejected.",
    )
    return parser.parse_args()


def preflight(
    config_path: Path,
) -> tuple[
    ManualExperimentConfig,
    dict[str, Any],
    Any,
    pd.DataFrame,
    pd.DataFrame,
    FeatureSchema,
    pd.DataFrame,
    pd.DataFrame | None,
    dict[str, str],
]:
    config = load_manual_experiment_config(
        config_path,
        project_root=PROJECT_ROOT,
    )
    manifest_contract = validate_config_against_manifest(config)
    expected_environment = dict(manifest_contract["current_parity_environment"])
    actual_environment = collect_environment_versions()
    environment_mismatches = validate_environment_contract(
        actual_environment,
        expected_environment,
    )
    if environment_mismatches and config.payload["parity"]["required"]:
        raise RuntimeError(
            "Required parity environment mismatch:\n- "
            + "\n- ".join(environment_mismatches)
        )

    prepared = load_dataset(
        version=config.dataset_version,
        processed_dir=config.processed_dir,
    )
    X_train_model, X_test_model, schema = prepare_model_features(
        prepared.X_train,
        prepared.y_train,
        prepared.X_test,
        config,
    )

    sample_submission = pd.read_csv(config.sample_submission_path)
    if list(sample_submission.columns) != ["index", "y"]:
        raise ValueError(
            "The official sample submission must contain columns ['index', 'y'] "
            "in that order."
        )
    if len(sample_submission) != config.payload["dataset"]["expected_test_rows"]:
        raise ValueError("The official sample submission row count is invalid.")

    reference_submission: pd.DataFrame | None = None
    reference_path = config.reference_submission_path
    if config.payload["parity"]["enabled"]:
        if reference_path is not None and reference_path.exists():
            reference_submission = pd.read_csv(reference_path)
        elif config.payload["parity"]["required"]:
            raise FileNotFoundError(
                f"Required parity reference submission is missing: {reference_path}"
            )

    return (
        config,
        manifest_contract,
        prepared,
        X_train_model,
        X_test_model,
        schema,
        sample_submission,
        reference_submission,
        actual_environment,
    )


def build_fold_assignments(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ManualExperimentConfig,
) -> pd.DataFrame:
    outer = config.payload["outer_cv"]
    splitter = StratifiedKFold(
        n_splits=int(outer["n_splits"]),
        shuffle=bool(outer["shuffle"]),
        random_state=int(outer["random_state"]),
    )
    assignments = np.zeros(len(X_train), dtype=np.int16)
    for fold, (_, validation_index) in enumerate(
        splitter.split(X_train, y_train),
        start=1,
    ):
        assignments[validation_index] = fold
    return pd.DataFrame(
        {
            "row_position": np.arange(len(X_train), dtype=np.int64),
            "row_index": X_train.index.to_numpy(),
            "fold": assignments,
        }
    )


def build_metadata(
    config: ManualExperimentConfig,
    manifest_contract: dict[str, Any],
    prepared: Any,
    schema: FeatureSchema,
    environment: dict[str, str],
) -> dict[str, Any]:
    dataset_dir = config.processed_dir / config.dataset_version
    return {
        "experiment_id": config.experiment_id,
        "run_id": None,
        "status": "running",
        "implementation": config.payload["experiment"]["implementation"],
        "known_implementation_difference": manifest_contract[
            "known_implementation_difference"
        ],
        "started_at_utc": utc_now().isoformat(),
        "finished_at_utc": None,
        "duration_seconds": None,
        "config_source_path": str(config.source_path),
        "manifest_path": str(config.manifest_path),
        "dataset": {
            "version": prepared.version,
            "processed_dir": str(config.processed_dir),
            "train_rows": len(prepared.X_train),
            "test_rows": len(prepared.X_test),
            "source_feature_count": prepared.X_train.shape[1],
            "files": {
                name: fingerprint_file(dataset_dir / filename)
                for name, filename in {
                    "X_train": "X_train.parquet",
                    "y_train": "y_train.parquet",
                    "X_test": "X_test.parquet",
                    "metadata": "metadata.json",
                }.items()
            },
        },
        "git": collect_git_state(PROJECT_ROOT),
        "environment": {
            "actual": environment,
            "expected_parity_environment": manifest_contract[
                "current_parity_environment"
            ],
        },
        "cv": {
            "outer": config.payload["outer_cv"],
            "inner_target_encoding": {
                "n_splits": config.payload["target_encoder"]["inner_splits"],
                "shuffle": config.payload["target_encoder"]["shuffle"],
                "random_state": config.payload["target_encoder"]["random_state"],
            },
        },
        "target_encoder": config.payload["target_encoder"],
        "lightgbm_parameters": config.model_parameters,
        "prediction_policy": config.payload["prediction"],
        "thresholds": config.payload["thresholds"],
        "features": schema.to_dict(),
        "paths": {
            "sample_submission": str(config.sample_submission_path),
            "reference_submission": (
                str(config.reference_submission_path)
                if config.reference_submission_path is not None
                else None
            ),
            "artifact_root": str(config.artifact_root),
        },
        "parity": {
            "enabled": config.payload["parity"]["enabled"],
            "required": config.payload["parity"]["required"],
        },
    }


def execute(args: argparse.Namespace) -> int:
    config_path = (
        args.config
        if args.config.is_absolute()
        else (Path.cwd() / args.config)
    )
    (
        config,
        manifest_contract,
        prepared,
        X_train_model,
        X_test_model,
        schema,
        sample_submission,
        reference_submission,
        environment,
    ) = preflight(config_path)

    if args.validate_only:
        print("Validation successful.")
        print(f"Experiment: {config.experiment_id}")
        print(f"Dataset: {config.dataset_version}")
        print(
            "Features: "
            f"{len(schema.source_feature_names)} source -> "
            f"{len(schema.transformed_feature_names)} transformed"
        )
        print(f"Categorical features: {len(schema.categorical_features)}")
        print(
            "Parity: "
            f"enabled={config.payload['parity']['enabled']}, "
            f"required={config.payload['parity']['required']}"
        )
        return 0

    metadata = build_metadata(
        config,
        manifest_contract,
        prepared,
        schema,
        environment,
    )
    store: RunArtifactStore | None = None
    try:
        store = RunArtifactStore(
            config,
            train_index=X_train_model.index,
            test_index=X_test_model.index,
            run_id=args.run_id,
        )
        metadata["run_id"] = store.run_id
        fold_assignments = build_fold_assignments(
            X_train_model,
            prepared.y_train,
            config,
        )
        store.save_initial_contract(metadata, schema, fold_assignments)
        print(f"Run directory: {store.root}")

        def persist_fold(fold: Any) -> None:
            store.save_fold(fold)
            print(
                f"Fold {fold.fold}: "
                f"BA@{config.payload['thresholds']['submission']:.3f} = "
                f"{fold.balanced_accuracy:.6f}"
            )

        result = run_manual_lightgbm_cross_validation(
            X_train_model,
            prepared.y_train,
            X_test_model,
            config,
            on_fold_complete=persist_fold,
        )
        if not result.fold_assignments.equals(fold_assignments):
            raise RuntimeError(
                "Training fold assignments differ from the preflight assignments."
            )

        evaluation = evaluate_predictions(result, config)
        submission = build_submission(
            sample_submission,
            result.test_predictions,
        )
        parity_report: ParityReport | None = None
        if (
            config.payload["parity"]["enabled"]
            and reference_submission is not None
        ):
            parity_report = evaluate_parity(
                config,
                schema,
                evaluation,
                sample_submission,
                submission,
                reference_submission,
            )

        store.save_final_outputs(
            result,
            evaluation,
            submission,
            parity_report,
        )
        if (
            config.payload["parity"]["required"]
            and (parity_report is None or not parity_report.passed)
        ):
            failed_gates = (
                []
                if parity_report is None
                else [
                    name
                    for name, gate in parity_report.primary_gates.items()
                    if not gate["passed"]
                ]
            )
            raise RequiredParityError(
                f"Required parity failed: {failed_gates or ['unavailable']}"
            )

        metadata["parity_result"] = (
            parity_report.to_dict() if parity_report is not None else None
        )
        store.complete(metadata)
    except BaseException as error:
        final_error: BaseException = error
        if store is not None and not (store.root / "_SUCCESS").exists():
            try:
                store.fail_preserving(error, metadata)
            except RunFailureFinalizationError as finalization_error:
                final_error = finalization_error
        print(
            f"Status: failed ({type(final_error).__name__}: {final_error})",
            file=sys.stderr,
        )
        if store is not None:
            print(f"Failed run directory: {store.root}", file=sys.stderr)
        return 1

    print(
        "BA@0.117: "
        f"{evaluation.metrics['primary']['balanced_accuracy']:.12f}"
    )
    print(
        "Global optimized OOF (optimistic): "
        f"threshold={evaluation.metrics['global_optimized_oof_optimistic']['threshold']:.12f}, "
        f"BA={evaluation.metrics['global_optimized_oof_optimistic']['balanced_accuracy']:.12f}"
    )
    print(
        "Legacy cross-fitted (not fully nested) BA: "
        f"{evaluation.metrics['legacy_cross_fitted_not_fully_nested']['balanced_accuracy']:.12f}"
    )
    print(f"Positive predictions: {int(submission['y'].sum())}")
    print("Status: completed")
    return 0

def main() -> int:
    return execute(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
