from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from src.churn_ml.features import PreparedDataset
from src.churn_ml.metrics import (
    PredictionResult,
    calculate_binary_metrics,
    optimize_balanced_accuracy_threshold,
)
from src.churn_ml.modeling import (
    CrossValidationOutput,
    ExperimentConfig,
    ExperimentOutput,
    ExperimentResult,
    create_stratified_cv,
    utc_timestamp,
)

@dataclass(frozen=True)
class CatBoostConfig:
    iterations: int = 1000
    learning_rate: float = 0.05
    depth: int = 6
    loss_function: str = "Logloss"
    eval_metric: str = "AUC"

    task_type: str = "GPU"
    devices: str = "0"

    metric_period: int = 20
    early_stopping_rounds: int = 100

    allow_writing_files: bool = False

    @classmethod
    def default(cls) -> "CatBoostConfig":
        return cls()

    def model_parameters(
        self,
        random_state: int,
    ) -> dict[str, Any]:
        parameters = asdict(self)

        parameters.pop("early_stopping_rounds")

        parameters["random_seed"] = random_state

        return parameters

def get_catboost_categorical_features(
    X: pd.DataFrame,
) -> list[str]:
    """Return categorical feature names for CatBoost."""
    return X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

def prepare_catboost_data(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare feature frames for CatBoost.

    CatBoost does not accept missing values in categorical columns,
    therefore categorical missing values are replaced with a string token.
    """
    X_train_prepared = X_train.copy()
    X_valid_prepared = X_valid.copy()
    X_test_prepared = X_test.copy()

    for column in categorical_features:
        X_train_prepared[column] = (
            X_train_prepared[column]
            .astype("string")
            .fillna("__MISSING__")
        )
        X_valid_prepared[column] = (
            X_valid_prepared[column]
            .astype("string")
            .fillna("__MISSING__")
        )
        X_test_prepared[column] = (
            X_test_prepared[column]
            .astype("string")
            .fillna("__MISSING__")
        )

    return X_train_prepared, X_valid_prepared, X_test_prepared


def create_catboost_model(
    parameters: dict[str, Any],
) -> CatBoostClassifier:
    """Create a CatBoost classifier from experiment parameters."""
    return CatBoostClassifier(**parameters)


def fit_catboost_fold(
    model: CatBoostClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical_features: list[str],
    *,
    early_stopping_rounds: int,
) -> CatBoostClassifier:
    """Fit CatBoost on one validation fold."""
    model.fit(
        X_train,
        y_train,
        cat_features=categorical_features,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
    )

    return model

def run_catboost_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    cv: StratifiedKFold,
    model_config: CatBoostConfig,
    experiment_config: ExperimentConfig,
) -> CrossValidationOutput:
    """Run CatBoost cross-validation and generate OOF/test predictions."""

    categorical_features = get_catboost_categorical_features(X)

    oof_probabilities = np.zeros(len(X), dtype=float)
    test_fold_probabilities = np.zeros(
        (len(X_test), cv.get_n_splits()),
        dtype=float,
    )

    fold_records: list[dict[str, float | int]] = []
    fitted_models: list[CatBoostClassifier] = []

    start_time = perf_counter()

    n_splits = cv.get_n_splits()

    print(
        (
            "Starting CatBoost cross-validation: "
            f"{n_splits} folds, {len(X):,} rows, "
            f"{X.shape[1]} features"
        ),
        flush=True,
    )

    parameters = model_config.model_parameters(
        random_state=experiment_config.random_state,
    )

    fold_iterator = tqdm(
        cv.split(X, y),
        total=n_splits,
        desc="CatBoost CV",
        unit="fold",
    )

    oof_folds = np.zeros(len(X), dtype=int)

    for fold_number, (train_idx, valid_idx) in enumerate(
        fold_iterator,
        start=1,
    ):        

        oof_folds[valid_idx] = fold_number

        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]

        X_valid_fold = X.iloc[valid_idx]
        y_valid_fold = y.iloc[valid_idx]

        (
            X_train_prepared,
            X_valid_prepared,
            X_test_prepared,
        ) = prepare_catboost_data(
            X_train=X_train_fold,
            X_valid=X_valid_fold,
            X_test=X_test,
            categorical_features=categorical_features,
        )

        model = create_catboost_model(parameters)

        fold_start_time = perf_counter()

        model = fit_catboost_fold(
            model=model,
            X_train=X_train_prepared,
            y_train=y_train_fold,
            X_valid=X_valid_prepared,
            y_valid=y_valid_fold,
            categorical_features=categorical_features,
            early_stopping_rounds=(
                model_config.early_stopping_rounds
            ),
        )

        fold_training_time = perf_counter() - fold_start_time

        valid_probabilities = model.predict_proba(
            X_valid_prepared
        )[:, 1]

        test_probabilities = model.predict_proba(
            X_test_prepared
        )[:, 1]

        oof_probabilities[valid_idx] = valid_probabilities
        test_fold_probabilities[:, fold_number - 1] = (
            test_probabilities
        )

        fold_prediction_result = (
            PredictionResult.from_probabilities(
                probabilities=valid_probabilities,
                threshold=(
                    experiment_config.decision_threshold
                ),
            )
        )

        fold_metrics = calculate_binary_metrics(
            y_true=y_valid_fold,
            prediction_result=fold_prediction_result,
        )

        best_iteration = int(model.get_best_iteration())

        fold_records.append(
            {
                "fold": fold_number,
                **fold_metrics,
                "training_time_seconds": fold_training_time,
                "best_iteration": best_iteration,
            }
        )

        fitted_models.append(model)

        fold_iterator.set_postfix(
            balanced_accuracy=(
                f"{fold_metrics['balanced_accuracy']:.4f}"
            ),
            roc_auc=f"{fold_metrics['roc_auc']:.4f}",
            best_iteration=best_iteration,
        )


    total_training_time = perf_counter() - start_time

    oof_prediction_result = (
        PredictionResult.from_probabilities(
            probabilities=oof_probabilities,
            threshold=(
                experiment_config.decision_threshold
            ),
        )
    )

    overall_metrics = calculate_binary_metrics(
        y_true=y,
        prediction_result=oof_prediction_result,
    )

    threshold_optimization = (
        optimize_balanced_accuracy_threshold(
            y_true=y,
            probabilities=oof_probabilities,
        )
    )

    overall_metrics.update(
        {
            "optimized_threshold_oof": (
                threshold_optimization.threshold
            ),
            "optimized_balanced_accuracy_oof": (
                threshold_optimization.balanced_accuracy
            ),
        }
    )

    fold_metrics_frame = pd.DataFrame(fold_records)

    overall_metrics.update(
        {
            "balanced_accuracy_mean": float(
                fold_metrics_frame[
                    "balanced_accuracy"
                ].mean()
            ),
            "balanced_accuracy_std": float(
                fold_metrics_frame[
                    "balanced_accuracy"
                ].std(ddof=1)
            ),
            "roc_auc_mean": float(
                fold_metrics_frame["roc_auc"].mean()
            ),
            "roc_auc_std": float(
                fold_metrics_frame["roc_auc"].std(ddof=1)
            ),
            "average_precision_mean": float(
                fold_metrics_frame["average_precision"].mean()
            ),
            "average_precision_std": float(
                fold_metrics_frame["average_precision"].std(ddof=1)
            ),
            "log_loss_mean": float(
                fold_metrics_frame["log_loss"].mean()
            ),
            "log_loss_std": float(
                fold_metrics_frame["log_loss"].std(ddof=1)
            ),
        }
    )

    test_probabilities = test_fold_probabilities.mean(axis=1)

    test_default_predictions = (
        PredictionResult.from_probabilities(
            probabilities=test_probabilities,
            threshold=experiment_config.decision_threshold,
        )
    )

    test_optimized_predictions = (
        PredictionResult.from_probabilities(
            probabilities=test_probabilities,
            threshold=threshold_optimization.threshold,
        )
    )    

    return CrossValidationOutput(
        fold_metrics=fold_metrics_frame,
        oof_predictions=pd.DataFrame(
            {
                "row_index": X.index,
                "fold": oof_folds,
                "target": y.to_numpy(),
                "probability": oof_probabilities,
                "prediction_default": (
                    oof_prediction_result.predictions
                ),
                "prediction_optimized_oof": (
                    PredictionResult.from_probabilities(
                        probabilities=oof_probabilities,
                        threshold=threshold_optimization.threshold,
                    ).predictions
                ),
            }
        ),
        test_predictions=pd.DataFrame(
            {
                "row_index": X_test.index,
                "probability": test_probabilities,
                "prediction_default": (
                    test_default_predictions.predictions
                ),
                "prediction_optimized_oof": (
                    test_optimized_predictions.predictions
                ),
            }
        ),
        metrics=overall_metrics,
        fitted_models=fitted_models,
        training_time_seconds=total_training_time,
    )

def run_catboost_experiment(
    dataset: PreparedDataset,
    config: ExperimentConfig,
    model_config: CatBoostConfig,
    experiment_id: str,
) -> ExperimentOutput:
    """Run a complete CatBoost cross-validation experiment."""
    cv = create_stratified_cv(config)

    cv_output = run_catboost_cross_validation(
        X=dataset.X_train,
        y=dataset.y_train,
        X_test=dataset.X_test,
        cv=cv,
        model_config=model_config,
        experiment_config=config,
    )

    result = ExperimentResult(
        experiment_id=experiment_id,
        dataset_version=dataset.version,
        model_name="CatBoostClassifier",
        validation_strategy={
            "strategy": type(cv).__name__,
            "n_splits": config.n_splits,
            "shuffle": config.shuffle,
            "random_state": config.random_state,
        },
        metrics=cv_output.metrics,
        parameters=asdict(model_config),
        random_state=config.random_state,
        n_features=dataset.X_train.shape[1],
        training_time_seconds=cv_output.training_time_seconds,
        created_at_utc=utc_timestamp(),
    )

    return ExperimentOutput(
        result=result,
        fold_metrics=cv_output.fold_metrics,
        oof_predictions=cv_output.oof_predictions,
        test_predictions=cv_output.test_predictions,
        fitted_models=cv_output.fitted_models,
    )