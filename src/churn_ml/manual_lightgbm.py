from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from src.churn_ml.config import ManualExperimentConfig
from src.churn_ml.metrics import (
    PredictionResult,
    calculate_binary_metrics,
    evaluate_cross_fitted_thresholds,
    optimize_balanced_accuracy_threshold,
)
from src.churn_ml.target_encoding import AutoGluonBinaryOOFTargetEncoder


class FeatureContractError(ValueError):
    """Raised when the loaded dataset violates the manual feature contract."""


def ordered_feature_schema_sha256(feature_names: list[str]) -> str:
    """Hash ordered feature names using the frozen UTF-8 JSON representation."""
    encoded = json.dumps(
        feature_names,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FeatureSchema:
    source_feature_names: list[str]
    dropped_features: list[str]
    model_feature_names: list[str]
    categorical_features: list[str]
    numerical_features: list[str]
    transformed_feature_names: list[str]

    @property
    def source_schema_sha256(self) -> str:
        return ordered_feature_schema_sha256(self.source_feature_names)

    @property
    def model_input_schema_sha256(self) -> str:
        return ordered_feature_schema_sha256(self.model_feature_names)

    @property
    def transformed_schema_sha256(self) -> str:
        return ordered_feature_schema_sha256(self.transformed_feature_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_hash_canonicalization": (
                "sha256(utf8(json(names, ensure_ascii=false, separators=(',', ':'))))"
            ),
            "source_feature_count": len(self.source_feature_names),
            "source_feature_names": self.source_feature_names,
            "source_schema_sha256": self.source_schema_sha256,
            "dropped_features": self.dropped_features,
            "model_input_feature_count": len(self.model_feature_names),
            "model_input_feature_names": self.model_feature_names,
            "model_input_schema_sha256": self.model_input_schema_sha256,
            "categorical_feature_count": len(self.categorical_features),
            "categorical_features": self.categorical_features,
            "numerical_feature_count": len(self.numerical_features),
            "numerical_features": self.numerical_features,
            "transformed_feature_count": len(self.transformed_feature_names),
            "transformed_feature_names": self.transformed_feature_names,
            "transformed_schema_sha256": self.transformed_schema_sha256,
            "transformed_order_rule": (
                "source-order numerical passthrough columns followed by "
                "source-order categorical __te columns"
            ),
        }

@dataclass
class FoldResult:
    fold: int
    train_positions: np.ndarray
    validation_positions: np.ndarray
    validation_targets: np.ndarray
    validation_probabilities: np.ndarray
    test_probabilities: np.ndarray
    balanced_accuracy: float
    duration_seconds: float
    model: Any
    encoder: AutoGluonBinaryOOFTargetEncoder


@dataclass(frozen=True)
class CrossValidationResult:
    fold_metrics: pd.DataFrame
    fold_assignments: pd.DataFrame
    oof_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    test_probabilities_by_fold: np.ndarray
    duration_seconds: float


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, Any]
    global_threshold_curve: pd.DataFrame
    legacy_cross_fitted_fold_metrics: pd.DataFrame
    legacy_cross_fitted_predictions: pd.DataFrame


@dataclass(frozen=True)
class ParityReport:
    enabled: bool
    required: bool
    passed: bool
    primary_gates: dict[str, dict[str, Any]]
    secondary_diagnostics: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "required": self.required,
            "passed": self.passed,
            "primary_gates": self.primary_gates,
            "secondary_diagnostics": self.secondary_diagnostics,
        }


def prepare_model_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    config: ManualExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, FeatureSchema]:
    """Validate and apply the notebook's exact feature-selection behavior."""
    dataset = config.payload["dataset"]
    features = config.payload["features"]

    if len(X_train) != dataset["expected_train_rows"]:
        raise FeatureContractError(
            f"Expected {dataset['expected_train_rows']} training rows, got {len(X_train)}."
        )
    if len(X_test) != dataset["expected_test_rows"]:
        raise FeatureContractError(
            f"Expected {dataset['expected_test_rows']} test rows, got {len(X_test)}."
        )
    if not X_train.index.equals(y_train.index):
        raise FeatureContractError("Training feature and target indices differ.")
    if not X_train.index.is_unique or not X_test.index.is_unique:
        raise FeatureContractError("Train and test indices must be unique.")
    if list(X_train.columns) != list(X_test.columns):
        raise FeatureContractError("Train and test source feature order differs.")
    if X_train.shape[1] != dataset["expected_source_feature_count"]:
        raise FeatureContractError(
            "Source feature count mismatch: "
            f"expected {dataset['expected_source_feature_count']}, "
            f"got {X_train.shape[1]}."
        )

    dropped_features = list(features["drop"])
    missing_drops = [
        column for column in dropped_features if column not in X_train.columns
    ]
    if missing_drops:
        raise FeatureContractError(
            f"Expected dropped features are missing: {missing_drops}"
        )

    X_train_model = X_train.drop(columns=dropped_features)
    X_test_model = X_test.drop(columns=dropped_features)

    categorical_features = X_train_model.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical_features = X_train_model.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()
    unsupported_features = [
        column
        for column in X_train_model.columns
        if column not in categorical_features
        and column not in numerical_features
    ]

    expected_categorical = list(features["categorical"])
    if categorical_features != expected_categorical:
        raise FeatureContractError(
            "Categorical feature names or order differ from the contract: "
            f"actual={categorical_features}"
        )
    if len(categorical_features) != features["expected_categorical_count"]:
        raise FeatureContractError("Categorical feature count differs from contract.")
    if X_train_model.shape[1] != features["expected_model_feature_count"]:
        raise FeatureContractError("Final model input feature count differs from contract.")
    if list(X_train_model.columns) != list(X_test_model.columns):
        raise FeatureContractError("Train and test model feature order differs.")
    if unsupported_features:
        raise FeatureContractError(
            f"Unsupported model feature dtypes: {unsupported_features}"
        )

    transformed_feature_names = numerical_features + [
        f"{column}__te" for column in categorical_features
    ]
    if len(transformed_feature_names) != features["expected_model_feature_count"]:
        raise FeatureContractError("Transformed feature count differs from contract.")

    schema = FeatureSchema(
        source_feature_names=X_train.columns.tolist(),
        dropped_features=dropped_features,
        model_feature_names=X_train_model.columns.tolist(),
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        transformed_feature_names=transformed_feature_names,
    )
    schema_hashes = {
        "source": (
            schema.source_schema_sha256,
            features["expected_source_schema_sha256"],
        ),
        "model input": (
            schema.model_input_schema_sha256,
            features["expected_model_input_schema_sha256"],
        ),
        "transformed": (
            schema.transformed_schema_sha256,
            features["expected_transformed_schema_sha256"],
        ),
    }
    mismatched_hashes = [
        f"{name}: expected={expected}, actual={actual}"
        for name, (actual, expected) in schema_hashes.items()
        if actual != expected
    ]
    if mismatched_hashes:
        raise FeatureContractError(
            "Ordered feature schema hash mismatch:\n- "
            + "\n- ".join(mismatched_hashes)
        )
    return X_train_model, X_test_model, schema


def run_manual_lightgbm_cross_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    config: ManualExperimentConfig,
    *,
    on_fold_complete: Callable[[FoldResult], None] | None = None,
) -> CrossValidationResult:
    """Run the notebook-compatible outer CV without any parity dependency."""
    outer = config.payload["outer_cv"]
    encoder_config = config.payload["target_encoder"]
    threshold = float(config.payload["thresholds"]["submission"])
    n_folds = int(outer["n_splits"])

    outer_cv = StratifiedKFold(
        n_splits=n_folds,
        shuffle=bool(outer["shuffle"]),
        random_state=int(outer["random_state"]),
    )
    splits = list(outer_cv.split(X_train, y_train))

    oof_probabilities = np.zeros(len(X_train), dtype=float)
    test_probabilities_folds = np.zeros((len(X_test), n_folds), dtype=float)
    fold_assignments_array = np.zeros(len(X_train), dtype=np.int16)
    fold_records: list[dict[str, Any]] = []
    start_time = perf_counter()

    for fold, (train_index, validation_index) in enumerate(splits, start=1):
        fold_start = perf_counter()
        X_fold_train_raw = X_train.iloc[train_index].reset_index(drop=True)
        X_fold_validation_raw = X_train.iloc[validation_index].reset_index(drop=True)
        y_fold_train = y_train.iloc[train_index].reset_index(drop=True)
        y_fold_validation = y_train.iloc[validation_index].reset_index(drop=True)

        fold_encoder = AutoGluonBinaryOOFTargetEncoder(
            n_splits=int(encoder_config["inner_splits"]),
            alpha=float(encoder_config["alpha"]),
            random_state=int(encoder_config["random_state"]),
        )
        X_fold_train = fold_encoder.fit_transform(
            X_fold_train_raw,
            y_fold_train,
        )
        X_fold_validation = fold_encoder.transform(X_fold_validation_raw)
        X_fold_test = fold_encoder.transform(X_test.reset_index(drop=True))

        model = lgb.LGBMClassifier(**config.model_parameters)
        model.fit(X_fold_train, y_fold_train)

        validation_probability_matrix = np.asarray(
            model.predict_proba(X_fold_validation)
        )
        test_probability_matrix = np.asarray(model.predict_proba(X_fold_test))
        validation_probabilities = validation_probability_matrix[:, 1]
        test_probabilities = test_probability_matrix[:, 1]

        oof_probabilities[validation_index] = validation_probabilities
        test_probabilities_folds[:, fold - 1] = test_probabilities
        fold_assignments_array[validation_index] = fold

        fold_predictions = (
            validation_probabilities >= threshold
        ).astype("int8")
        fold_score = float(
            balanced_accuracy_score(y_fold_validation, fold_predictions)
        )
        fold_duration = perf_counter() - fold_start

        fold_result = FoldResult(
            fold=fold,
            train_positions=train_index,
            validation_positions=validation_index,
            validation_targets=y_fold_validation.to_numpy(),
            validation_probabilities=validation_probabilities,
            test_probabilities=test_probabilities,
            balanced_accuracy=fold_score,
            duration_seconds=fold_duration,
            model=model,
            encoder=fold_encoder,
        )
        if on_fold_complete is not None:
            on_fold_complete(fold_result)

        fold_records.append(
            {
                "fold": fold,
                "train_rows": len(train_index),
                "validation_rows": len(validation_index),
                "decision_threshold": threshold,
                "balanced_accuracy": fold_score,
                "predicted_positive_count": int(fold_predictions.sum()),
                "duration_seconds": fold_duration,
            }
        )

    test_probabilities = test_probabilities_folds.mean(axis=1)
    duration = perf_counter() - start_time
    fold_assignments = pd.DataFrame(
        {
            "row_position": np.arange(len(X_train), dtype=np.int64),
            "row_index": X_train.index.to_numpy(),
            "fold": fold_assignments_array,
        }
    )
    oof_predictions = pd.DataFrame(
        {
            "row_position": np.arange(len(X_train), dtype=np.int64),
            "row_index": X_train.index.to_numpy(),
            "fold": fold_assignments_array,
            "target": y_train.to_numpy(),
            "probability": oof_probabilities,
            "prediction_0_117": (
                oof_probabilities >= threshold
            ).astype("int8"),
        }
    )
    test_predictions = pd.DataFrame(
        {
            "row_position": np.arange(len(X_test), dtype=np.int64),
            "row_index": X_test.index.to_numpy(),
            "probability": test_probabilities,
            "prediction_0_117": (
                test_probabilities >= threshold
            ).astype("int8"),
        }
    )
    return CrossValidationResult(
        fold_metrics=pd.DataFrame(fold_records),
        fold_assignments=fold_assignments,
        oof_predictions=oof_predictions,
        test_predictions=test_predictions,
        test_probabilities_by_fold=test_probabilities_folds,
        duration_seconds=duration,
    )


def notebook_threshold_grid(config: ManualExperimentConfig) -> np.ndarray:
    """Build the exact inclusive notebook grid using np.arange."""
    thresholds = config.payload["thresholds"]["legacy_diagnostic_grid"]
    minimum = float(thresholds["minimum"])
    maximum = float(thresholds["maximum"])
    step = float(thresholds["step"])
    return np.arange(minimum, maximum + step, step)


def evaluate_predictions(
    result: CrossValidationResult,
    config: ManualExperimentConfig,
) -> EvaluationResult:
    """Evaluate fixed and historical diagnostic thresholds."""
    y_true = result.oof_predictions["target"].to_numpy()
    probabilities = result.oof_predictions["probability"].to_numpy()
    threshold = float(config.payload["thresholds"]["submission"])
    fixed_prediction = PredictionResult.from_probabilities(probabilities, threshold)
    fixed_metrics = calculate_binary_metrics(y_true, fixed_prediction)
    metrics_at_050 = calculate_binary_metrics(
        y_true,
        PredictionResult.from_probabilities(probabilities, 0.5),
    )

    grid = notebook_threshold_grid(config)
    global_optimization = optimize_balanced_accuracy_threshold(
        y_true,
        probabilities,
        thresholds=grid,
    )
    legacy_cross_fitted = evaluate_cross_fitted_thresholds(
        result.oof_predictions[["fold", "target", "probability"]],
        thresholds=grid,
    )

    metrics: dict[str, Any] = {
        "primary": {
            "name": "balanced_accuracy_at_fixed_submission_threshold",
            **fixed_metrics,
        },
        "balanced_accuracy_at_0_500": metrics_at_050["balanced_accuracy"],
        "fold_balanced_accuracy": {
            "mean": float(result.fold_metrics["balanced_accuracy"].mean()),
            "standard_deviation_population": float(
                result.fold_metrics["balanced_accuracy"].to_numpy().std()
            ),
        },
        "global_optimized_oof_optimistic": {
            "label": "optimistic; threshold selected on all OOF predictions",
            "threshold": global_optimization.threshold,
            "balanced_accuracy": global_optimization.balanced_accuracy,
        },
        "legacy_cross_fitted_not_fully_nested": {
            "label": "legacy cross-fitted estimate; not fully nested",
            "balanced_accuracy": legacy_cross_fitted.balanced_accuracy,
        },
        "training_duration_seconds": result.duration_seconds,
    }
    return EvaluationResult(
        metrics=metrics,
        global_threshold_curve=global_optimization.scores,
        legacy_cross_fitted_fold_metrics=legacy_cross_fitted.fold_metrics,
        legacy_cross_fitted_predictions=legacy_cross_fitted.oof_predictions,
    )


def build_submission(
    sample_submission: pd.DataFrame,
    test_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Apply notebook submission semantics without modifying the source file."""
    submission = sample_submission.copy()
    submission["y"] = test_predictions["prediction_0_117"].to_numpy(dtype="int8")
    return submission


def evaluate_parity(
    config: ManualExperimentConfig,
    schema: FeatureSchema,
    evaluation: EvaluationResult,
    sample_submission: pd.DataFrame,
    generated_submission: pd.DataFrame,
    reference_submission: pd.DataFrame,
) -> ParityReport:
    """Evaluate required gates outside the model-training core."""
    parity = config.payload["parity"]
    features = config.payload["features"]
    dataset = config.payload["dataset"]
    actual_ba = float(evaluation.metrics["primary"]["balanced_accuracy"])
    expected_ba = float(parity["expected_balanced_accuracy"])
    ba_delta = actual_ba - expected_ba
    positive_count = int(generated_submission["y"].sum())

    sample_columns_match = list(generated_submission.columns) == list(
        sample_submission.columns
    )
    sample_row_count_match = len(generated_submission) == len(sample_submission)
    sample_index_match = generated_submission.index.equals(sample_submission.index)
    non_target_columns = [
        column for column in sample_submission.columns if column != "y"
    ]
    sample_row_order_match = generated_submission[non_target_columns].equals(
        sample_submission[non_target_columns]
    )
    reference_structure_match = (
        list(reference_submission.columns) == list(generated_submission.columns)
        and len(reference_submission) == len(generated_submission)
        and reference_submission.index.equals(generated_submission.index)
        and reference_submission[non_target_columns].equals(
            generated_submission[non_target_columns]
        )
    )
    labels_match = (
        reference_structure_match
        and np.array_equal(
            generated_submission["y"].to_numpy(),
            reference_submission["y"].to_numpy(),
        )
    )

    gates: dict[str, dict[str, Any]] = {
        "balanced_accuracy_at_0_117": {
            "expected": expected_ba,
            "actual": actual_ba,
            "delta": ba_delta,
            "absolute_tolerance": float(parity["balanced_accuracy_tolerance"]),
            "passed": abs(ba_delta) <= float(parity["balanced_accuracy_tolerance"]),
        },
        "submission_labels_exact": {
            "expected": "exact match to historical submission",
            "actual": labels_match,
            "passed": labels_match,
        },
        "positive_prediction_count": {
            "expected": int(parity["expected_positive_count"]),
            "actual": positive_count,
            "delta": positive_count - int(parity["expected_positive_count"]),
            "passed": positive_count == int(parity["expected_positive_count"]),
        },
        "sample_submission_columns": {
            "expected": list(sample_submission.columns),
            "actual": list(generated_submission.columns),
            "passed": sample_columns_match,
        },
        "sample_submission_row_count": {
            "expected": len(sample_submission),
            "actual": len(generated_submission),
            "passed": sample_row_count_match,
        },
        "sample_submission_index": {
            "expected": "exact",
            "actual": sample_index_match,
            "passed": sample_index_match,
        },
        "sample_submission_row_order": {
            "expected": "exact",
            "actual": sample_row_order_match,
            "passed": sample_row_order_match,
        },
        "source_schema_and_order": {
            "expected_feature_count": int(dataset["expected_source_feature_count"]),
            "actual_feature_count": len(schema.source_feature_names),
            "expected_sha256": features["expected_source_schema_sha256"],
            "actual_sha256": schema.source_schema_sha256,
            "passed": (
                len(schema.source_feature_names)
                == int(dataset["expected_source_feature_count"])
                and schema.source_schema_sha256
                == features["expected_source_schema_sha256"]
            ),
        },
        "model_input_schema_and_order": {
            "expected_feature_count": int(features["expected_model_feature_count"]),
            "actual_feature_count": len(schema.model_feature_names),
            "expected_sha256": features["expected_model_input_schema_sha256"],
            "actual_sha256": schema.model_input_schema_sha256,
            "passed": (
                len(schema.model_feature_names)
                == int(features["expected_model_feature_count"])
                and schema.model_input_schema_sha256
                == features["expected_model_input_schema_sha256"]
            ),
        },
        "feature_drops": {
            "expected": list(features["drop"]),
            "actual": schema.dropped_features,
            "passed": schema.dropped_features == list(features["drop"]),
        },
        "categorical_features": {
            "expected": list(features["categorical"]),
            "actual": schema.categorical_features,
            "passed": schema.categorical_features == list(features["categorical"]),
        },
        "transformed_schema_and_order": {
            "expected_feature_count": int(features["expected_model_feature_count"]),
            "actual_feature_count": len(schema.transformed_feature_names),
            "expected_sha256": features["expected_transformed_schema_sha256"],
            "actual_sha256": schema.transformed_schema_sha256,
            "passed": (
                len(schema.transformed_feature_names)
                == int(features["expected_model_feature_count"])
                and schema.transformed_schema_sha256
                == features["expected_transformed_schema_sha256"]
            ),
        },
    }

    global_result = evaluation.metrics["global_optimized_oof_optimistic"]
    legacy_result = evaluation.metrics["legacy_cross_fitted_not_fully_nested"]
    expected_secondary = parity["secondary"]
    secondary: dict[str, dict[str, Any]] = {
        "global_optimized_oof_balanced_accuracy": {
            "label": "optimistic",
            "expected_approximately": float(
                expected_secondary["global_optimized_oof_balanced_accuracy"]
            ),
            "actual": float(global_result["balanced_accuracy"]),
            "delta": float(global_result["balanced_accuracy"])
            - float(expected_secondary["global_optimized_oof_balanced_accuracy"]),
        },
        "global_optimized_threshold": {
            "label": "optimistic",
            "expected_approximately": float(
                expected_secondary["global_optimized_threshold"]
            ),
            "actual": float(global_result["threshold"]),
            "delta": float(global_result["threshold"])
            - float(expected_secondary["global_optimized_threshold"]),
        },
        "legacy_cross_fitted_balanced_accuracy": {
            "label": "legacy/not fully nested",
            "expected_approximately": float(
                expected_secondary["legacy_cross_fitted_balanced_accuracy"]
            ),
            "actual": float(legacy_result["balanced_accuracy"]),
            "delta": float(legacy_result["balanced_accuracy"])
            - float(expected_secondary["legacy_cross_fitted_balanced_accuracy"]),
        },
    }
    return ParityReport(
        enabled=True,
        required=bool(parity["required"]),
        passed=all(bool(gate["passed"]) for gate in gates.values()),
        primary_gates=gates,
        secondary_diagnostics=secondary,
    )
