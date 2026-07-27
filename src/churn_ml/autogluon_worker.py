"""Crash-isolated AutoGluon fitting worker.

This module is launched by the supervisor with the current Python interpreter.
AutoGluon and pandas are intentionally imported only inside ``run_worker``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from src.churn_ml.autogluon_artifacts import sha256_file, utc_now, write_json
from src.churn_ml.autogluon_completion import (
    PREDICTOR_RELATIVE_PATH,
    WORKER_COMPLETION_SCHEMA_VERSION,
)
from src.churn_ml.autogluon_config import AutoGluonConfig, load_config
from src.churn_ml.autogluon_inspection import export_worker_inspection
from src.churn_ml.autogluon_profiles import (
    SUPPORTED_AUTOGLUON_VERSION,
    effective_family_resources,
    get_profile,
    profile_sha256,
    profile_summary,
    resolve_profile_hyperparameters,
)


def _stage(name: str) -> None:
    print(f"AUTOGLUON_STAGE:{name}", flush=True)


def load_training_data(
    config: AutoGluonConfig,
    parquet_reader: Any,
) -> tuple[Any, Any, Any]:
    """Read exactly the two validated train-only Parquet files."""
    features = parquet_reader(config.paths.train_features)
    target_frame = parquet_reader(config.paths.train_target)
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
    return features, target_frame, train_data


def build_fit_kwargs(
    config: AutoGluonConfig,
    hyperparameters: dict[str, Any],
    train_data: Any,
) -> dict[str, Any]:
    """Build supported fit arguments without a mixed-profile global GPU value."""
    return {
        "train_data": train_data,
        "presets": config.fit.presets,
        "hyperparameters": hyperparameters,
        "time_limit": config.resources.time_limit_seconds,
        "num_cpus": config.resources.num_cpus,
        "fit_strategy": config.resources.fit_strategy,
        "ag_args_ensemble": {
            "fold_fitting_strategy": config.resources.fold_fitting_strategy,
        },
        "calibrate_decision_threshold": config.fit.calibrate_decision_threshold,
    }


def validate_effective_seed_report(
    inspection: dict[str, Any],
    *,
    requested_seed: int,
) -> None:
    """Fail only on configured base-model seed mismatch or invalid evidence."""
    status = inspection.get("effective_seed_status")
    effective_seed = inspection.get("effective_seed")
    if status == "mismatch":
        raise RuntimeError(
            "AutoGluon public metadata exposed a configured base-model seed that "
            f"differs from requested seed {requested_seed}: "
            f"{inspection.get('effective_seeds_observed')}"
        )
    if status == "verified":
        if effective_seed != requested_seed:
            raise RuntimeError(
                "Effective seed report is internally inconsistent: verified seed "
                f"{effective_seed!r} does not equal requested seed {requested_seed}."
            )
        return
    if status == "unavailable" and effective_seed is None:
        return
    raise RuntimeError(
        "Effective seed report has an invalid status/value combination: "
        f"status={status!r}, effective_seed={effective_seed!r}."
    )


def run_worker(run_dir: Path, repository_root: Path) -> dict[str, Any]:
    """Fit one configured predictor using train-only processed inputs."""
    started_at = utc_now()
    monotonic_start = time.monotonic()
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
    profile_hash = profile_sha256(profile, config.seed, config.resources.num_gpus)
    hyperparameters = resolve_profile_hyperparameters(
        profile,
        seed=config.seed,
        gpu_budget=config.resources.num_gpus,
    )
    family_resources = effective_family_resources(hyperparameters)
    resolved_families = list(hyperparameters)
    write_json(
        run_dir / "profile_resolution.json",
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "profile_id": profile.profile_id,
            "profile_sha256": profile_hash,
            "profile_identity": profile_summary(
                profile, config.seed, config.resources.num_gpus
            ),
            "dataset_version": config.dataset.version,
            "requested_seed": config.seed,
            "autogluon_version": installed_version,
            "portfolio": profile.portfolio,
            "resolved_families": resolved_families,
            "family_resources": family_resources,
            "resolved_hyperparameters": hyperparameters,
            "fit_strategy": config.resources.fit_strategy,
            "fold_fitting_strategy": config.resources.fold_fitting_strategy,
            "top_level_num_gpus_passed_to_fit": False,
        },
    )
    _stage("profile_resolved")

    features, target_frame, train_data = load_training_data(config, pd.read_parquet)
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

    predictor_dir = run_dir / PREDICTOR_RELATIVE_PATH
    predictor = TabularPredictor(
        label=config.dataset.label,
        problem_type=config.predictor.problem_type,
        eval_metric=config.predictor.eval_metric,
        positive_class=config.predictor.positive_class,
        verbosity=config.predictor.verbosity,
        path=str(predictor_dir),
    )
    _stage("predictor_created")
    predictor.fit(**build_fit_kwargs(config, hyperparameters, train_data))
    _stage("fit_completed")
    inspection = export_worker_inspection(
        run_dir,
        predictor,
        requested_seed=config.seed,
        configured_families=resolved_families,
    )
    validate_effective_seed_report(inspection, requested_seed=config.seed)
    _stage("inspection_exported")

    completed_at = utc_now()
    result = {
        "schema_version": WORKER_COMPLETION_SCHEMA_VERSION,
        "status": "completed",
        "worker_pid": os.getpid(),
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "duration_seconds": float(time.monotonic() - monotonic_start),
        "profile_id": config.profile_id,
        "profile_sha256": profile_hash,
        "dataset_version": config.dataset.version,
        "predictor_relative_path": PREDICTOR_RELATIVE_PATH,
        "resolved_families": resolved_families,
        "model_names": inspection["models"],
        "best_model": inspection["best_model"],
        "decision_threshold": float(inspection["decision_threshold"]),
        "autogluon_version": installed_version,
        "python_version": platform.python_version(),
        "requested_seed": config.seed,
        "effective_seed": inspection["effective_seed"],
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
                args.run_dir / "worker_error.json",
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
