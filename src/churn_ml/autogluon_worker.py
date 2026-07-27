"""Crash-isolated AutoGluon fitting worker.

This module is launched by the supervisor with the current Python interpreter.
AutoGluon and pandas are intentionally imported only inside ``run_worker``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from src.churn_ml.autogluon_artifacts import sha256_file, utc_now, write_json
from src.churn_ml.autogluon_config import load_config
from src.churn_ml.autogluon_inspection import export_worker_inspection
from src.churn_ml.autogluon_profiles import (
    SUPPORTED_AUTOGLUON_VERSION,
    get_profile,
    resolve_profile_hyperparameters,
)


def _stage(name: str) -> None:
    print(f"AUTOGLUON_STAGE:{name}", flush=True)


def run_worker(run_dir: Path, repository_root: Path) -> dict[str, Any]:
    """Fit one configured predictor using train-only processed inputs."""
    started_at = utc_now()
    _stage("worker_started")
    config = load_config(
        run_dir / "resolved_config.yaml",
        repository_root,
        require_data_files=True,
    )
    _stage("config_validated")

    installed_version = importlib.metadata.version("autogluon.tabular")
    if installed_version != SUPPORTED_AUTOGLUON_VERSION:
        raise RuntimeError(
            f"AutoGluon {SUPPORTED_AUTOGLUON_VERSION} is required; "
            f"found {installed_version}"
        )

    import pandas as pd
    from autogluon.tabular import TabularPredictor  # type: ignore[import-not-found]

    _stage("autogluon_imported")
    profile = get_profile(config.profile_id)
    hyperparameters = resolve_profile_hyperparameters(profile)
    write_json(
        run_dir / "profile_resolution.json",
        {
            "created_at_utc": utc_now(),
            "profile_id": profile.profile_id,
            "autogluon_version": installed_version,
            "portfolio": profile.portfolio,
            "model_families": list(hyperparameters),
            "resolved_hyperparameters": hyperparameters,
            "fit_strategy": config.resources.fit_strategy,
            "fold_fitting_strategy": config.resources.fold_fitting_strategy,
            "test_predictions": "not_applicable_prohibited_by_benchmark_contract",
        },
    )
    _stage("profile_resolved")

    features = pd.read_parquet(config.paths.train_features)
    target_frame = pd.read_parquet(config.paths.train_target)
    if (
        len(target_frame.columns) != 1
        or config.dataset.label not in target_frame.columns
    ):
        raise ValueError(
            "The target Parquet must contain exactly the configured label column "
            f"{config.dataset.label!r}."
        )
    if len(features) != len(target_frame):
        raise ValueError("Training features and target row counts do not match.")
    if config.dataset.label in features.columns:
        raise ValueError("The configured label already exists in the feature frame.")
    train_data = features.copy()
    train_data[config.dataset.label] = target_frame[config.dataset.label].to_numpy()
    write_json(
        run_dir / "dataset_manifest.json",
        {
            "created_at_utc": utc_now(),
            "dataset_version": config.dataset.version,
            "row_count": len(features),
            "feature_count": len(features.columns),
            "ordered_features": list(features.columns),
            "feature_dtypes": {
                name: str(dtype) for name, dtype in features.dtypes.items()
            },
            "label": config.dataset.label,
            "target_value_counts": {
                str(key): int(value)
                for key, value in target_frame[config.dataset.label]
                .value_counts(dropna=False)
                .items()
            },
            "train_features_sha256": sha256_file(config.paths.train_features),
            "train_target_sha256": sha256_file(config.paths.train_target),
        },
    )
    _stage("train_data_loaded")

    predictor_dir = run_dir / "predictor"
    predictor = TabularPredictor(
        label=config.dataset.label,
        problem_type=config.predictor.problem_type,
        eval_metric=config.predictor.eval_metric,
        positive_class=config.predictor.positive_class,
        verbosity=config.predictor.verbosity,
        path=str(predictor_dir),
    )
    _stage("predictor_created")
    fit_kwargs: dict[str, Any] = {
        "train_data": train_data,
        "presets": config.fit.presets,
        "hyperparameters": hyperparameters,
        "time_limit": config.resources.time_limit_seconds,
        "num_cpus": config.resources.num_cpus,
        "num_gpus": config.resources.num_gpus,
        "fit_strategy": config.resources.fit_strategy,
        "ag_args_ensemble": {
            "fold_fitting_strategy": config.resources.fold_fitting_strategy,
        },
        "calibrate_decision_threshold": config.fit.calibrate_decision_threshold,
    }
    predictor.fit(**fit_kwargs)
    _stage("fit_completed")
    inspection = export_worker_inspection(run_dir, predictor)
    _stage("inspection_exported")

    result = {
        "status": "completed",
        "worker_pid": os.getpid(),
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "profile_id": config.profile_id,
        "model_families": list(hyperparameters),
        "best_model": inspection.get("best_model"),
        "decision_threshold": inspection.get("decision_threshold"),
    }
    write_json(run_dir / "worker_result.json", result)
    _stage("worker_result_written")
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        run_worker(
            args.run_dir.resolve(strict=True), args.repository_root.resolve(strict=True)
        )
    except Exception as error:
        traceback.print_exc()
        try:
            write_json(
                args.run_dir / "worker_result.json",
                {
                    "status": "python_exception",
                    "worker_pid": os.getpid(),
                    "completed_at_utc": utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        except Exception:
            traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
