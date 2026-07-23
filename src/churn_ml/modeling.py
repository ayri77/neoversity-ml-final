from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class ExperimentConfig:
    random_state: int
    n_splits: int
    shuffle: bool

    primary_metric: str
    scoring_metrics: list[str]
    decision_threshold: float

    @classmethod
    def default(cls) -> "ExperimentConfig":
        return cls(
            random_state=42,
            n_splits=5,
            shuffle=True,
            primary_metric="balanced_accuracy",
            scoring_metrics=[
                "balanced_accuracy",
                "roc_auc",
                "average_precision",
                "log_loss",
            ],
            decision_threshold=0.5,
        )


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    dataset_version: str
    model_name: str
    validation_strategy: dict[str, Any]
    metrics: dict[str, float]
    parameters: dict[str, Any]
    random_state: int
    n_features: int
    training_time_seconds: float
    created_at_utc: str


@dataclass(frozen=True)
class CrossValidationOutput:
    fold_metrics: pd.DataFrame
    oof_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    metrics: dict[str, float]
    fitted_models: list[Any]
    training_time_seconds: float


@dataclass(frozen=True)
class ExperimentOutput:
    result: ExperimentResult
    fold_metrics: pd.DataFrame
    oof_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    fitted_models: list[Any]


@dataclass(frozen=True)
class LoadedExperiment:
    result: ExperimentResult
    fold_metrics: pd.DataFrame
    oof_predictions: pd.DataFrame
    test_predictions: pd.DataFrame


def load_experiment(
    experiment_id: str,
    experiments_dir: Path,
) -> LoadedExperiment:
    """Load a saved experiment and its prediction artifacts."""
    experiment_dir = experiments_dir / experiment_id

    required_files = {
        "result": experiment_dir / "result.json",
        "fold_metrics": experiment_dir / "fold_metrics.csv",
        "oof_predictions": experiment_dir / "oof_predictions.parquet",
        "test_predictions": experiment_dir / "test_predictions.parquet",
    }

    missing_files = [path.name for path in required_files.values() if not path.exists()]

    if missing_files:
        raise FileNotFoundError(
            f"Experiment '{experiment_id}' is incomplete. "
            f"Missing files: {missing_files}"
        )

    with required_files["result"].open(
        "r",
        encoding="utf-8",
    ) as file:
        result_payload = json.load(file)

    result = ExperimentResult(**result_payload)

    return LoadedExperiment(
        result=result,
        fold_metrics=pd.read_csv(required_files["fold_metrics"]),
        oof_predictions=pd.read_parquet(required_files["oof_predictions"]),
        test_predictions=pd.read_parquet(required_files["test_predictions"]),
    )


def create_stratified_cv(
    config: ExperimentConfig,
) -> StratifiedKFold:
    """Create a reproducible stratified cross-validation splitter."""
    if config.n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    return StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=config.shuffle,
        random_state=config.random_state,
    )


def summarize_target(y: pd.Series) -> pd.DataFrame:
    """Return class counts and proportions for a target variable."""
    summary = (
        y.value_counts(dropna=False).rename_axis("class").reset_index(name="count")
    )

    summary["proportion"] = summary["count"] / len(y)

    return summary


def save_experiment_result(
    result: ExperimentResult,
    experiments_dir: Path,
    *,
    fold_metrics: pd.DataFrame | None = None,
    oof_predictions: pd.DataFrame | None = None,
    test_predictions: pd.DataFrame | None = None,
    model: Any | None = None,
    overwrite: bool = False,
) -> Path:
    """Save experiment metadata, metrics, predictions, and model."""
    experiment_dir = experiments_dir / result.experiment_id

    if experiment_dir.exists() and any(experiment_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Experiment already exists: {experiment_dir}. "
            "Use overwrite=True to replace it."
        )

    experiment_dir.mkdir(parents=True, exist_ok=True)

    with (experiment_dir / "result.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            asdict(result),
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    if fold_metrics is not None:
        fold_metrics.to_csv(
            experiment_dir / "fold_metrics.csv",
            index=False,
        )

    if oof_predictions is not None:
        oof_predictions.to_parquet(
            experiment_dir / "oof_predictions.parquet",
            index=False,
        )

    if test_predictions is not None:
        test_predictions.to_parquet(
            experiment_dir / "test_predictions.parquet",
            index=False,
        )

    if model is not None:
        import joblib

        joblib.dump(
            model,
            experiment_dir / "model.joblib",
        )

    return experiment_dir


def load_experiment_results(
    experiments_dir: Path,
) -> pd.DataFrame:
    """Load all available experiment summaries into one table."""
    records: list[dict[str, Any]] = []

    if not experiments_dir.exists():
        return pd.DataFrame()

    for result_path in sorted(experiments_dir.glob("*/result.json")):
        with result_path.open("r", encoding="utf-8") as file:
            record = json.load(file)

        metrics = record.pop("metrics", {})
        validation = record.pop("validation_strategy", {})
        parameters = record.pop("parameters", {})

        record.update({f"metric_{name}": value for name, value in metrics.items()})
        record["validation_strategy"] = validation
        record["parameters"] = parameters

        records.append(record)

    return pd.DataFrame(records)


def utc_timestamp() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def build_experiment_summary(
    result: ExperimentResult,
) -> pd.DataFrame:
    """Build a compact experiment summary table."""
    metrics = result.metrics

    rows = [
        {
            "metric": "Balanced Accuracy @ default threshold",
            "value": metrics["balanced_accuracy"],
        },
        {
            "metric": "Optimized OOF Balanced Accuracy",
            "value": metrics["optimized_balanced_accuracy_oof"],
        },
        {
            "metric": "Optimized OOF threshold",
            "value": metrics["optimized_threshold_oof"],
        },
        {
            "metric": "ROC-AUC",
            "value": metrics["roc_auc"],
        },
        {
            "metric": "Average Precision",
            "value": metrics["average_precision"],
        },
        {
            "metric": "Log Loss",
            "value": metrics["log_loss"],
        },
        {
            "metric": "Training time, minutes",
            "value": result.training_time_seconds / 60,
        },
    ]

    return pd.DataFrame(rows)
