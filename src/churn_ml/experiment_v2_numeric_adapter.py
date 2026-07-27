from __future__ import annotations

import importlib
import importlib.metadata

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.churn_ml.experiment_v2_adapter import ExperimentV2AdapterContractError
from src.churn_ml.target_encoding import AutoGluonBinaryOOFTargetEncoder


EXPECTED_NUMERIC_FEATURE_CONTRACT = {
    "implementation": "existing_v2_fold_local_oof_target_encoding",
    "inner_splits": 5,
    "shuffle": True,
    "random_state": 42,
    "alpha": 10.0,
    "prior": "unweighted_mean_of_category_level_target_means",
    "keep_original_categorical_features": False,
    "output": "numeric_matrix",
    "missing_value_policy": "native_nan",
}


ADAPTER_DEPENDENCIES = {
    "xgboost_numeric_v1": ("xgboost", "3.3.0"),
    "catboost_numeric_v1": ("catboost", "1.2.10"),
}


class ExperimentV2AdapterDependencyError(ExperimentV2AdapterContractError):
    """Raised when a selected adapter's locked package is unavailable."""


def adapter_dependency_details(adapter_id: str) -> tuple[str, str]:
    try:
        return ADAPTER_DEPENDENCIES[adapter_id]
    except KeyError as error:
        raise ValueError(
            f"Adapter has no external dependency contract: {adapter_id}."
        ) from error


def load_adapter_class(adapter_id: str, class_name: str) -> Any:
    package, _ = adapter_dependency_details(adapter_id)
    try:
        module = importlib.import_module(package)
    except ModuleNotFoundError as error:
        if error.name != package:
            raise
        raise _dependency_error(adapter_id) from error
    return getattr(module, class_name)


def adapter_dependency_version(adapter_id: str) -> str:
    package, _ = adapter_dependency_details(adapter_id)
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError as error:
        raise _dependency_error(adapter_id) from error


def _dependency_error(adapter_id: str) -> ExperimentV2AdapterDependencyError:
    package, expected_version = adapter_dependency_details(adapter_id)
    return ExperimentV2AdapterDependencyError(
        f"Adapter '{adapter_id}' requires package '{package}' at locked version "
        f"'{expected_version}'; restore the locked project environment before "
        "running this adapter."
    )


@dataclass(frozen=True)
class NumericMatrixDiagnostics:
    label: str
    row_count: int
    column_count: int
    total_missing_count: int
    rows_with_missing_count: int
    columns_with_missing_count: int
    columns_with_missing: tuple[str, ...]
    infinity_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_adapter_inputs(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    prediction_features: pd.DataFrame,
) -> None:
    if not isinstance(train_features, pd.DataFrame) or not isinstance(
        prediction_features, pd.DataFrame
    ):
        raise ExperimentV2AdapterContractError(
            "Adapter features must be pandas DataFrames."
        )
    if not isinstance(train_labels, pd.Series):
        raise ExperimentV2AdapterContractError(
            "Adapter labels must be a pandas Series."
        )
    if len(train_features) != len(train_labels):
        raise ExperimentV2AdapterContractError(
            "Training features and labels are not aligned."
        )
    if list(train_features.columns) != list(prediction_features.columns):
        raise ExperimentV2AdapterContractError(
            "Training and prediction feature schemas differ."
        )
    if train_features.columns.has_duplicates:
        raise ExperimentV2AdapterContractError("Adapter feature names must be unique.")
    if (
        train_labels.isna().any()
        or pd.api.types.is_bool_dtype(train_labels.dtype)
        or set(train_labels.unique()) != {0, 1}
    ):
        raise ExperimentV2AdapterContractError(
            "Adapter target must contain exactly the binary labels 0 and 1."
        )


def build_numeric_matrices(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    prediction_features: pd.DataFrame,
    feature_contract: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if feature_contract.get("missing_value_policy") != "native_nan":
        raise ExperimentV2AdapterContractError(
            "candidate_adapter.contract.numeric_features.missing_value_policy "
            "must be exactly 'native_nan'."
        )
    encoder = AutoGluonBinaryOOFTargetEncoder(
        n_splits=int(feature_contract["inner_splits"]),
        alpha=float(feature_contract["alpha"]),
        random_state=int(feature_contract["random_state"]),
    )
    encoded_train = encoder.fit_transform(
        train_features.reset_index(drop=True),
        train_labels.reset_index(drop=True),
    )
    encoded_prediction = encoder.transform(prediction_features.reset_index(drop=True))
    validate_numeric_matrix(encoded_train, "training")
    validate_numeric_matrix(encoded_prediction, "prediction")
    if list(encoded_train.columns) != list(encoded_prediction.columns):
        raise ExperimentV2AdapterContractError(
            "Encoded training and prediction feature schemas differ."
        )
    return encoded_train, encoded_prediction


def validate_numeric_matrix(
    features: pd.DataFrame,
    label: str,
) -> NumericMatrixDiagnostics:
    if not isinstance(features, pd.DataFrame):
        raise ExperimentV2AdapterContractError(
            f"candidate_adapter.numeric_matrix.{label} must be a pandas DataFrame."
        )
    non_numeric = [
        name
        for name, dtype in features.dtypes.items()
        if not pd.api.types.is_numeric_dtype(dtype)
    ]
    if non_numeric:
        raise ExperimentV2AdapterContractError(
            f"candidate_adapter.numeric_matrix.{label}.non_numeric_columns "
            f"must be empty; got {non_numeric}."
        )
    try:
        values = features.to_numpy(dtype=float, copy=False)
    except (TypeError, ValueError) as error:
        raise ExperimentV2AdapterContractError(
            f"candidate_adapter.numeric_matrix.{label} must contain only numeric "
            "values."
        ) from error
    missing = np.isnan(values)
    infinities = np.isinf(values)
    columns_with_missing = tuple(
        str(name)
        for name, has_missing in zip(
            features.columns,
            missing.any(axis=0),
            strict=True,
        )
        if has_missing
    )
    diagnostics = NumericMatrixDiagnostics(
        label=label,
        row_count=len(features),
        column_count=features.shape[1],
        total_missing_count=int(missing.sum()),
        rows_with_missing_count=int(missing.any(axis=1).sum()),
        columns_with_missing_count=len(columns_with_missing),
        columns_with_missing=columns_with_missing,
        infinity_count=int(infinities.sum()),
    )
    if diagnostics.infinity_count:
        raise ExperimentV2AdapterContractError(
            f"candidate_adapter.numeric_matrix.{label}.infinity_count must be 0; "
            f"got {diagnostics.infinity_count}."
        )
    return diagnostics


def positive_class_probabilities(
    estimator: Any,
    features: pd.DataFrame,
) -> np.ndarray:
    matrix = np.asarray(estimator.predict_proba(features), dtype=float)
    classes = np.asarray(getattr(estimator, "classes_", []))
    positive_columns = np.flatnonzero(classes == 1)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != len(features)
        or matrix.shape[1] != len(classes)
        or len(positive_columns) != 1
    ):
        raise ExperimentV2AdapterContractError(
            "Estimator must expose one predict_proba column for class 1."
        )
    return matrix[:, int(positive_columns[0])]


def numeric_identity_inputs(
    *,
    adapter_id: str,
    contract: Mapping[str, Any],
    estimator: str,
    native_missing_configuration: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "id": adapter_id,
        "contract": deepcopy(dict(contract)),
        "estimator": estimator,
        "feature_semantics": "shared_fold_local_numeric_matrix_native_nan_v1",
        "missing_value_policy": "native_nan",
        "native_missing_configuration": deepcopy(dict(native_missing_configuration)),
        "probability_semantics": "binary_positive_class_label_1",
        "fit_boundary": "supplied_training_fold_only",
        "early_stopping": "disabled",
        "thresholding": "experiment_core_v2_only",
    }
