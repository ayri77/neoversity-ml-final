from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.churn_ml.competition_assets_v1 import (
    load_row_identity_artifact,
    submission_ids_from_identity,
)
from src.churn_ml.deployment_v1_auth import SyntheticFixture
from src.churn_ml.deployment_v1_contracts import (
    DeploymentConfig,
    ValidatedDeployment,
)
from src.churn_ml.control_panel.path_safety import PathSafetyError
from src.churn_ml.deployment_v1_paths import (
    DeploymentPathError,
    validate_regular_file,
)
from src.churn_ml.experiment_v2 import get_feature_pipeline
from src.churn_ml.experiment_v2_numeric_adapter import validate_numeric_matrix
from src.churn_ml.experiment_v2_schema import ordered_feature_schema_sha256
from src.churn_ml.research_data import canonical_sha256
from src.churn_ml.research_v2_data import load_research_v2_training_data
from src.churn_ml.target_encoding import AutoGluonBinaryOOFTargetEncoder


class DeploymentDataError(ValueError):
    """Raised when P4 train, test, encoding, or submission inputs are unsafe."""


@dataclass(frozen=True)
class DeploymentData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    sample_submission: pd.DataFrame
    source_train: pd.DataFrame
    source_test: pd.DataFrame
    train_schema: dict[str, Any]
    test_schema: dict[str, Any]
    dataset_identity: dict[str, Any]
    test_row_keys: tuple[Any, ...]
    test_row_identity_sha256: str
    fixture_identity: dict[str, Any]


@dataclass(frozen=True)
class EncodingResult:
    train: pd.DataFrame
    test: pd.DataFrame
    assignments: pd.DataFrame
    identity: dict[str, Any]


def load_deployment_data(
    validated: ValidatedDeployment,
    *,
    fixture: SyntheticFixture | None = None,
) -> DeploymentData:
    """Load test/sample only after approval and config validation has succeeded."""
    config = validated.config
    if fixture is None:
        training = load_research_v2_training_data(
            validated.approvals[0].research_run.config
        )
        source_train = _load_source_train(validated)
        y_train = training.y.copy()
        try:
            test_path = validate_regular_file(
                config.project_root / config.payload["test_data"]["path"],
                containment_root=config.project_root,
                reject_hardlinks=True,
            )
            sample_path = validate_regular_file(
                config.project_root / config.payload["sample_submission"]["path"],
                containment_root=config.project_root,
                reject_hardlinks=True,
            )
        except DeploymentPathError as error:
            raise DeploymentDataError(str(error)) from error
    else:
        source_train = pd.read_parquet(fixture.paths["train"])
        target = pd.read_parquet(fixture.paths["labels"])
        if target.shape[1] != 1:
            raise DeploymentDataError("Fixture target must have exactly one column.")
        y_train = target.iloc[:, 0]
        test_path = fixture.paths["test"]
        sample_path = fixture.paths["sample_submission"]

    test_file_sha256 = hashlib.sha256(test_path.read_bytes()).hexdigest()
    sample_file_sha256 = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    if fixture is None:
        if test_file_sha256 != config.payload["test_data"]["sha256"]:
            raise DeploymentDataError("Competition test file hash differs.")
        if sample_file_sha256 != config.payload["sample_submission"]["sha256"]:
            raise DeploymentDataError("Sample-submission file hash differs.")
    source_test = pd.read_parquet(test_path)
    sample = pd.read_csv(sample_path)
    _validate_source_frames(source_train, y_train, source_test)
    test_row_keys, test_row_identity = _validate_sample_and_alignment(
        source_test, sample, config, fixture is None
    )
    pipeline = get_feature_pipeline(config.payload["pipeline_id"])
    pipeline_contract = validated.approvals[0].research_run.config.pipeline_contract
    train_output = pipeline.transform(source_train, pipeline_contract)
    test_output = pipeline.transform(source_test, pipeline_contract)
    if train_output.schema.to_dict() != test_output.schema.to_dict():
        raise DeploymentDataError("Train/test pipeline schemas differ.")

    train_schema = schema_record(source_train, train_output.features)
    test_schema = schema_record(source_test, test_output.features)
    expected_schema = config.payload["test_data"]["ordered_schema_sha256"]
    if (
        fixture is None
        and train_schema["source_ordered_names_sha256"] != expected_schema
    ):
        raise DeploymentDataError("Configured competition test schema hash differs.")
    dataset_identity = {
        "schema_version": 1,
        "dataset_version": config.payload["dataset_version"],
        "train_rows": len(source_train),
        "test_rows": len(source_test),
        "test_file_sha256": test_file_sha256,
        "sample_submission_file_sha256": sample_file_sha256,
        "train_row_order_sha256": canonical_sha256(list(range(len(source_train)))),
        "test_row_order_sha256": test_row_identity,
        "target_sha256": canonical_sha256(
            {
                "name": str(y_train.name),
                "dtype": str(y_train.dtype),
                "values": y_train.tolist(),
            }
        ),
        "test_id_sha256": canonical_sha256(list(test_row_keys)),
        "sample_id_sha256": canonical_sha256(
            sample[config.payload["sample_submission"]["id_column"]].tolist()
        ),
    }
    return DeploymentData(
        X_train=train_output.features.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        X_test=test_output.features.reset_index(drop=True),
        sample_submission=sample.reset_index(drop=True),
        source_train=source_train.reset_index(drop=True),
        source_test=source_test.reset_index(drop=True),
        train_schema=train_schema,
        test_schema=test_schema,
        dataset_identity=dataset_identity,
        test_row_keys=test_row_keys,
        test_row_identity_sha256=test_row_identity,
        fixture_identity=(
            fixture.identity
            if fixture is not None
            else {
                "schema_version": 1,
                "mode": "competition",
                "sha256": canonical_sha256(
                    {
                        "test_sha256": test_file_sha256,
                        "sample_sha256": sample_file_sha256,
                    }
                ),
                "canonical": {
                    "test_sha256": test_file_sha256,
                    "sample_sha256": sample_file_sha256,
                },
            }
        ),
    )


def build_full_data_encoding(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    encoder_contract: Mapping[str, Any],
) -> EncodingResult:
    expected_keys = {
        "implementation",
        "inner_splits",
        "shuffle",
        "random_state",
        "alpha",
        "prior",
        "keep_original_categorical_features",
    }
    numeric_extra = {"output", "missing_value_policy"}
    actual = set(encoder_contract)
    if actual not in (expected_keys, expected_keys | numeric_extra):
        raise DeploymentDataError("Target-encoder contract keys differ.")
    n_splits = _exact_int(encoder_contract["inner_splits"], "inner_splits")
    random_state = _exact_int(encoder_contract["random_state"], "random_state")
    alpha = _exact_float(encoder_contract["alpha"], "alpha")
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    fold_values = np.full(len(X_train), -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(
        splitter.split(np.zeros(len(X_train)), y_train),
        start=1,
    ):
        fold_values[validation] = fold
    if (fold_values < 1).any():
        raise DeploymentDataError("OOF encoding assignments are incomplete.")
    assignments = pd.DataFrame(
        {
            "row_position": np.arange(len(X_train), dtype=np.int64),
            "encoding_fold": fold_values,
        }
    )
    encoder = AutoGluonBinaryOOFTargetEncoder(
        n_splits=n_splits,
        alpha=alpha,
        random_state=random_state,
    )
    encoded_train = encoder.fit_transform(
        X_train.reset_index(drop=True),
        y_train.reset_index(drop=True),
    )
    encoded_test = encoder.transform(X_test.reset_index(drop=True))
    validate_numeric_matrix(encoded_train, "deployment_training")
    validate_numeric_matrix(encoded_test, "deployment_test")
    if encoded_train.columns.tolist() != encoded_test.columns.tolist():
        raise DeploymentDataError("Encoded train/test column order differs.")
    mappings = {
        column: {
            "categories_sha256": _array_sha256(state["categories"]),
            "encoded_values_sha256": _array_sha256(state["encoded_values"]),
            "global_mean": float(state["global_mean"]),
        }
        for column, state in encoder.encodings_.items()
    }
    canonical = {
        "schema_version": 1,
        "method": "deterministic_oof_train_full_mapping_test",
        "encoder_contract": dict(encoder_contract),
        "assignment_sha256": frame_sha256(assignments),
        "categorical_columns": list(encoder.categorical_columns_),
        "passthrough_columns": list(encoder.passthrough_columns_),
        "transformed_columns": encoded_train.columns.tolist(),
        "full_data_mappings": mappings,
        "train_matrix_sha256": numeric_frame_sha256(encoded_train),
        "test_matrix_sha256": numeric_frame_sha256(encoded_test),
        "test_fit_performed": False,
        "imputation_performed": False,
    }
    return EncodingResult(
        train=encoded_train,
        test=encoded_test,
        assignments=assignments,
        identity={**canonical, "sha256": canonical_sha256(canonical)},
    )


def _load_source_train(validated: ValidatedDeployment) -> pd.DataFrame:
    first = validated.approvals[0].research_run
    dataset = first.config.plan_payload["dataset"]
    item = dataset["files"]["train_features"]
    path = first.config.dataset_dir / str(item["name"])
    try:
        path = validate_regular_file(
            path,
            containment_root=validated.config.project_root,
            reject_hardlinks=True,
        )
    except DeploymentPathError as error:
        raise DeploymentDataError(str(error)) from error
    source = pd.read_parquet(path)
    for approval in validated.approvals[1:]:
        if approval.research_run.dataset_fingerprints != first.dataset_fingerprints:
            raise DeploymentDataError(
                "Approved components do not share exact dataset identity."
            )
    return source


def _validate_source_frames(
    train: pd.DataFrame,
    labels: pd.Series,
    test: pd.DataFrame,
) -> None:
    if not isinstance(train, pd.DataFrame) or not isinstance(test, pd.DataFrame):
        raise DeploymentDataError("Train/test inputs must be pandas DataFrames.")
    if train.columns.has_duplicates or test.columns.has_duplicates:
        raise DeploymentDataError("Train/test feature names must be unique.")
    if train.columns.tolist() != test.columns.tolist():
        raise DeploymentDataError("Train/test feature names or order differ.")
    train_dtypes = [str(value) for value in train.dtypes]
    test_dtypes = [str(value) for value in test.dtypes]
    if train_dtypes != test_dtypes:
        raise DeploymentDataError("Train/test ordered dtypes differ.")
    if not train.index.equals(pd.RangeIndex(len(train))) or not test.index.equals(
        pd.RangeIndex(len(test))
    ):
        raise DeploymentDataError("Train/test rows must use zero-based RangeIndex.")
    if len(labels) != len(train) or not labels.index.equals(train.index):
        raise DeploymentDataError("Training features and labels are not aligned.")
    if (
        labels.isna().any()
        or pd.api.types.is_bool_dtype(labels.dtype)
        or set(labels.unique()) != {0, 1}
    ):
        raise DeploymentDataError("Training labels must contain exact classes 0 and 1.")


def _validate_sample_and_alignment(
    test: pd.DataFrame,
    sample: pd.DataFrame,
    config: DeploymentConfig,
    enforce_configured_rows: bool,
) -> tuple[tuple[Any, ...], str]:
    sample_config = config.payload["sample_submission"]
    test_config = config.payload["test_data"]
    expected_rows = int(sample_config["expected_rows"])
    if enforce_configured_rows and (
        len(test) != expected_rows or len(sample) != expected_rows
    ):
        raise DeploymentDataError("Competition test/sample row count differs.")
    if len(test) != len(sample):
        raise DeploymentDataError("Test and sample row counts differ.")
    expected_columns = [sample_config["id_column"], sample_config["target_column"]]
    if sample.columns.tolist() != expected_columns:
        raise DeploymentDataError("Sample submission columns or order differ.")
    sample_ids = sample[sample_config["id_column"]].reset_index(drop=True)
    _validate_id_series(sample_ids, "Sample submission")

    row_identity_ref = config.payload.get("submission_row_identity")
    if row_identity_ref is not None:
        # Competition path: IDs come from a separate authenticated artifact.
        # The feature matrix must not contain the submission ID column.
        id_column = str(row_identity_ref["id_column"])
        if id_column in test.columns:
            raise DeploymentDataError(
                "Submission ID column must not appear in the model feature matrix."
            )
        try:
            identity_payload = load_row_identity_artifact(
                config.project_root,
                str(row_identity_ref["path"]),
                expected_sha256=str(row_identity_ref["sha256"]),
            )
            materialised = submission_ids_from_identity(identity_payload)
        except (OSError, ValueError, DeploymentPathError, PathSafetyError) as error:
            raise DeploymentDataError(
                f"Submission row identity failed authentication: {error}"
            ) from error
        if int(identity_payload["expected_rows"]) != len(test):
            raise DeploymentDataError(
                "Feature test row count differs from submission row identity."
            )
        if int(identity_payload["expected_rows"]) != len(sample):
            raise DeploymentDataError(
                "Sample submission row count differs from submission row identity."
            )
        if identity_payload["ordered_id_sha256"] != row_identity_ref["ordered_id_sha256"]:
            raise DeploymentDataError("Ordered submission ID hash differs.")
        if (
            identity_payload["row_position_identity_sha256"]
            != row_identity_ref["row_position_identity_sha256"]
        ):
            raise DeploymentDataError("Row-position identity hash differs.")
        if canonical_sha256(sample_ids.tolist()) != row_identity_ref["ordered_id_sha256"]:
            raise DeploymentDataError(
                "Sample submission ordered IDs disagree with the row-identity artifact."
            )
        if sample_ids.tolist() != materialised:
            raise DeploymentDataError(
                "Sample submission ID order differs from the row-identity artifact."
            )
        # Row-count-only agreement is never enough: the ordered hash must match.
        if canonical_sha256(list(range(len(test)))) != row_identity_ref[
            "row_position_identity_sha256"
        ]:
            raise DeploymentDataError(
                "Feature-matrix row-position identity differs from the authenticated "
                "submission row identity."
            )
        keys = tuple(materialised)
        return keys, canonical_sha256(list(keys))

    # Legacy / synthetic-fixture path: ID column lives on the test frame.
    test_id = str(test_config.get("id_column", sample_config["id_column"]))
    if test_id not in test.columns:
        raise DeploymentDataError("Configured test ID column is absent.")
    test_ids = test[test_id].reset_index(drop=True)
    _validate_id_series(test_ids, "Test")
    if str(test_ids.dtype) != str(sample_ids.dtype):
        raise DeploymentDataError("Test and sample ID dtypes differ.")
    if not test_ids.equals(sample_ids):
        raise DeploymentDataError("Test and sample ID/order alignment differs.")
    keys = tuple(test_ids.tolist())
    identity = canonical_sha256(list(keys))
    return keys, identity


def _validate_id_series(values: pd.Series, label: str) -> None:
    if values.isna().any():
        raise DeploymentDataError(f"{label} IDs contain null values.")
    if values.duplicated().any():
        raise DeploymentDataError(f"{label} IDs contain duplicates.")
    dtype = str(values.dtype)
    if dtype in {"int64", "Int64"}:
        if pd.api.types.is_bool_dtype(values.dtype):
            raise DeploymentDataError(f"{label} IDs cannot be booleans.")
        return
    if dtype in {"object", "string"} and all(
        type(value) is str for value in values.tolist()
    ):
        if any(not value or value.isspace() for value in values.tolist()):
            raise DeploymentDataError(f"{label} IDs contain empty/whitespace values.")
        return
    raise DeploymentDataError(
        f"{label} IDs must use exact int64 or homogeneous string values."
    )


def schema_record(source: pd.DataFrame, model: pd.DataFrame) -> dict[str, Any]:
    source_dtypes = [
        {"name": str(name), "dtype": str(dtype)}
        for name, dtype in source.dtypes.items()
    ]
    return {
        "source_rows": len(source),
        "source_columns": source.columns.tolist(),
        "source_ordered_names_sha256": ordered_feature_schema_sha256(
            source.columns.tolist()
        ),
        "source_ordered_dtypes": source_dtypes,
        "source_ordered_dtypes_sha256": canonical_sha256(source_dtypes),
        "model_columns": model.columns.tolist(),
        "model_ordered_names_sha256": ordered_feature_schema_sha256(
            model.columns.tolist()
        ),
    }


def _array_sha256(values: Any) -> str:
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    if array.dtype.kind in {"O", "U", "S"}:
        digest.update(canonical_sha256(array.tolist()).encode("ascii"))
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def numeric_frame_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_sha256(frame.columns.tolist()).encode("ascii"))
    digest.update(
        np.ascontiguousarray(frame.to_numpy(dtype=np.float64)).tobytes(order="C")
    )
    return digest.hexdigest()


def frame_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def _exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise DeploymentDataError(f"{label} must be an integer.")
    return value


def _exact_float(value: Any, label: str) -> float:
    if type(value) is not float or not np.isfinite(value):
        raise DeploymentDataError(f"{label} must be a finite float.")
    return value
