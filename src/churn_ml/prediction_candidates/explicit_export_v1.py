"""Import aligned, explicit probability exports without loading a predictor.

This adapter is source-neutral. It consumes existing Parquet files only and
normalizes their explicit row-position, target, and probability columns into
the immutable ``prediction_candidate_v1`` contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.churn_ml.dataset_registry.api import DatasetPackage, resolve_dataset_package
from src.churn_ml.prediction_candidates.contract_v1 import (
    OOF_COLUMNS,
    POSITIVE_CLASS_LABEL,
    PROBABILITY_SEMANTICS,
    SCHEMA_VERSION,
    TEST_COLUMNS,
    UNAVAILABLE,
    CandidatePackage,
    CandidateValidationError,
    PredictionCandidateError,
    build_candidate_id,
    create_candidate_package,
    file_sha256,
    repository_relative_path,
    resolve_under_repository,
    validate_oof_frame,
    validate_probability_series,
    validate_test_frame,
)
from src.churn_ml.research_data import canonical_sha256


OOF_PROTOCOL = "explicit_aligned_prediction_export_v1"
ADAPTER_NAME = "prediction_candidates.explicit_export_v1"
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
ROW_ALIASES = ("row_position", "row_index", "index")
PROBABILITY_ALIASES = (
    "probability_positive",
    "probability",
    "positive_probability",
    "proba",
)
TARGET_ALIASES = ("target", "y")


@dataclass(frozen=True)
class ExplicitExportRequest:
    oof_path: Path
    test_path: Path
    dataset_id: str
    source_model_name: str
    source_kind: str
    source_run_path: str
    threshold: float
    exploratory: bool
    validation_balanced_accuracy: float | None = None
    historical_kaggle_public_score: float | None = None
    source_model_type: str = UNAVAILABLE


@dataclass(frozen=True)
class PreparedExplicitExport:
    request: ExplicitExportRequest
    dataset: DatasetPackage
    oof_path: Path
    test_path: Path
    source_run_path: Path
    oof_path_relative: str
    test_path_relative: str
    source_run_path_relative: str
    oof_sha256: str
    test_sha256: str
    oof: pd.DataFrame
    test: pd.DataFrame
    oof_schema: dict[str, Any]
    test_schema: dict[str, Any]
    source_config_sha256: str
    identity: dict[str, Any]

    @property
    def candidate_id(self) -> str:
        return build_candidate_id(self.identity)


def prepare_explicit_export(
    request: ExplicitExportRequest,
    *,
    repository_root: Path,
) -> PreparedExplicitExport:
    """Read and validate existing export files without writing artifacts."""
    root = repository_root.resolve()
    _validate_request_values(request)
    oof_path, oof_relative = _resolve_existing_relative_path(
        request.oof_path, root, label="oof", require_file=True
    )
    test_path, test_relative = _resolve_existing_relative_path(
        request.test_path, root, label="test", require_file=True
    )
    source_path, source_relative = _resolve_existing_relative_path(
        Path(request.source_run_path), root, label="source_run_path", require_file=False
    )
    if oof_path.suffix.lower() != ".parquet" or test_path.suffix.lower() != ".parquet":
        raise PredictionCandidateError(
            "Explicit OOF and test exports must be Parquet files.",
            reason_code="explicit_export_format_invalid",
        )

    dataset = resolve_dataset_package(root / "data" / "processed", request.dataset_id)
    expected_exploratory = dataset.manifest.target_dependency == "exploratory"
    if dataset.manifest.target_dependency not in {"none", "exploratory"}:
        raise PredictionCandidateError(
            "Explicit export v1 supports only target_dependency none or exploratory.",
            reason_code="target_dependency_unsupported",
        )
    if request.exploratory != expected_exploratory:
        raise CandidateValidationError(
            f"Explicit exploratory declaration ({request.exploratory}) does not match "
            f"Dataset Package target_dependency={dataset.manifest.target_dependency!r}.",
            reason_code="exploratory_declaration_mismatch",
        )

    try:
        oof_raw = pd.read_parquet(oof_path)
        test_raw = pd.read_parquet(test_path)
    except Exception as error:
        raise PredictionCandidateError(
            f"Could not read explicit Parquet export: {type(error).__name__}: {error}",
            reason_code="explicit_export_unreadable",
        ) from error
    if not isinstance(oof_raw, pd.DataFrame) or not isinstance(test_raw, pd.DataFrame):
        raise PredictionCandidateError(
            "Explicit exports must deserialize as pandas DataFrames.",
            reason_code="explicit_export_type_invalid",
        )

    oof_schema = _schema_summary(oof_raw)
    test_schema = _schema_summary(test_raw)
    oof = _normalize_oof(oof_raw, dataset)
    test = _normalize_test(test_raw, dataset)
    oof_sha = file_sha256(oof_path)
    test_sha = file_sha256(test_path)
    descriptor = {
        "schema_version": 1,
        "adapter": ADAPTER_NAME,
        "dataset_id": request.dataset_id,
        "target_dependency": dataset.manifest.target_dependency,
        "exploratory": request.exploratory,
        "source_kind": request.source_kind,
        "source_model_name": request.source_model_name,
        "source_model_type": request.source_model_type,
        "source_run_path": source_relative,
        "oof_export_path": oof_relative,
        "oof_export_sha256": oof_sha,
        "test_export_path": test_relative,
        "test_export_sha256": test_sha,
        "threshold": float(request.threshold),
        "validation_balanced_accuracy": request.validation_balanced_accuracy,
        "historical_kaggle_public_score": request.historical_kaggle_public_score,
        "historical_kaggle_score_role": "external_evidence_only_not_validation_or_optimization",
        "oof_protocol": OOF_PROTOCOL,
    }
    source_config_sha = canonical_sha256(descriptor)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "source_kind": request.source_kind,
        "dataset_id": request.dataset_id,
        "source_run_path": source_relative,
        "source_config_sha256": source_config_sha,
        "source_model_name": request.source_model_name,
        "oof_protocol": OOF_PROTOCOL,
        "positive_class_label": POSITIVE_CLASS_LABEL,
        "probability_semantics": PROBABILITY_SEMANTICS,
    }
    return PreparedExplicitExport(
        request=request,
        dataset=dataset,
        oof_path=oof_path,
        test_path=test_path,
        source_run_path=source_path,
        oof_path_relative=oof_relative,
        test_path_relative=test_relative,
        source_run_path_relative=source_relative,
        oof_sha256=oof_sha,
        test_sha256=test_sha,
        oof=oof,
        test=test,
        oof_schema=oof_schema,
        test_schema=test_schema,
        source_config_sha256=source_config_sha,
        identity=identity,
    )


def validate_explicit_export(
    request: ExplicitExportRequest,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    prepared = prepare_explicit_export(request, repository_root=repository_root)
    probabilities = prepared.test["probability_positive"].to_numpy(dtype=np.float64)
    return {
        "ok": True,
        "artifacts_written": False,
        "candidate_id": prepared.candidate_id,
        "dataset_id": request.dataset_id,
        "target_dependency": prepared.dataset.manifest.target_dependency,
        "exploratory": request.exploratory,
        "source_model_name": request.source_model_name,
        "source_kind": request.source_kind,
        "source_run_path": prepared.source_run_path_relative,
        "oof_export": {
            "path": prepared.oof_path_relative,
            "sha256": prepared.oof_sha256,
            "detected_schema": prepared.oof_schema,
            "normalized_columns": list(OOF_COLUMNS),
            "row_count": len(prepared.oof),
        },
        "test_export": {
            "path": prepared.test_path_relative,
            "sha256": prepared.test_sha256,
            "detected_schema": prepared.test_schema,
            "normalized_columns": list(TEST_COLUMNS),
            "row_count": len(prepared.test),
        },
        "dataset_identity": _dataset_identity(prepared.dataset),
        "threshold": float(request.threshold),
        "predicted_positive_count": int(np.sum(probabilities >= request.threshold)),
        "validation_balanced_accuracy": request.validation_balanced_accuracy,
        "historical_kaggle_public_score": request.historical_kaggle_public_score,
        "historical_kaggle_score_role": "external_evidence_only_not_validation_or_optimization",
        "predictor_loaded": False,
        "predictions_generated": False,
    }


def import_explicit_export_candidate(
    request: ExplicitExportRequest,
    *,
    repository_root: Path,
) -> CandidatePackage:
    prepared = prepare_explicit_export(request, repository_root=repository_root)
    dataset_identity = _dataset_identity(prepared.dataset)
    external_evidence: dict[str, Any] = {
        "kaggle_public_score": request.historical_kaggle_public_score,
        "role": "external_evidence_only_not_validation_or_optimization",
    }
    provenance = {
        "adapter": ADAPTER_NAME,
        "legacy_run_path": prepared.source_run_path_relative,
        "oof_export_path": prepared.oof_path_relative,
        "oof_export_sha256": prepared.oof_sha256,
        "test_export_path": prepared.test_path_relative,
        "test_export_sha256": prepared.test_sha256,
        "detected_oof_schema": prepared.oof_schema,
        "detected_test_schema": prepared.test_schema,
        "dataset_identity": dataset_identity,
        "recorded_threshold": float(request.threshold),
        "validation_balanced_accuracy": request.validation_balanced_accuracy,
        "external_evidence": external_evidence,
        "exploratory": request.exploratory,
        "target_dependency": prepared.dataset.manifest.target_dependency,
        "predictor_loaded": False,
        "training_performed": False,
        "predictions_generated": False,
    }
    leaderboard_metadata = {
        "model": request.source_model_name,
        "score_val": request.validation_balanced_accuracy,
        "eval_metric": "balanced_accuracy",
        "historical_kaggle_public_score": request.historical_kaggle_public_score,
        "historical_kaggle_score_role": "external_evidence_only_not_validation_or_optimization",
    }
    manifest_fields = {
        "candidate_id": prepared.candidate_id,
        "dataset_id": request.dataset_id,
        "exploratory": request.exploratory,
        "source_run_path": prepared.source_run_path_relative,
        "source_config_path": UNAVAILABLE,
        "source_config_sha256": prepared.source_config_sha256,
        "source_model_name": request.source_model_name,
        "source_model_type": request.source_model_type,
        "source_predictor_path": UNAVAILABLE,
        "source_autogluon_version": UNAVAILABLE,
        "source_metric_name": (
            "balanced_accuracy"
            if request.validation_balanced_accuracy is not None
            else UNAVAILABLE
        ),
        "source_metric_value": (
            request.validation_balanced_accuracy
            if request.validation_balanced_accuracy is not None
            else UNAVAILABLE
        ),
        "source_leaderboard_metadata": leaderboard_metadata,
        "provenance": provenance,
    }
    source_metadata = {
        "schema_version": 1,
        "source_kind": request.source_kind,
        "adapter": ADAPTER_NAME,
        "legacy_run_path": prepared.source_run_path_relative,
        "model_name": request.source_model_name,
        "model_type": request.source_model_type,
        "dataset_id": request.dataset_id,
        "target_dependency": prepared.dataset.manifest.target_dependency,
        "exploratory": request.exploratory,
        "oof_protocol": OOF_PROTOCOL,
        "positive_class_label": POSITIVE_CLASS_LABEL,
        "probability_semantics": PROBABILITY_SEMANTICS,
        "oof_export": {
            "path": prepared.oof_path_relative,
            "sha256": prepared.oof_sha256,
            "detected_schema": prepared.oof_schema,
        },
        "test_export": {
            "path": prepared.test_path_relative,
            "sha256": prepared.test_sha256,
            "detected_schema": prepared.test_schema,
        },
        "dataset_identity": dataset_identity,
        "final_deployment_threshold": float(request.threshold),
        "threshold_source": "explicit_recorded_historical_threshold",
        "validation_balanced_accuracy": request.validation_balanced_accuracy,
        "external_evidence": external_evidence,
        "predictor_loaded": False,
        "training_performed": False,
        "predictions_generated": False,
    }
    return create_candidate_package(
        repository_root=repository_root.resolve(),
        identity=prepared.identity,
        oof=prepared.oof,
        test=prepared.test,
        manifest_fields=manifest_fields,
        source_metadata=source_metadata,
    )


def _normalize_oof(frame: pd.DataFrame, dataset: DatasetPackage) -> pd.DataFrame:
    mapping = _resolve_schema_columns(frame, label="OOF", allow_target=True)
    _validate_row_evidence(
        frame[mapping["row"]],
        expected_count=dataset.manifest.train_row_count,
        label="OOF",
    )
    probability = frame[mapping["probability"]]
    validate_probability_series(probability, field_name="explicit_oof.probability")
    if mapping["target"] is None:
        target = dataset.artifacts.y_train.astype("int64").reset_index(drop=True)
    else:
        target = frame[mapping["target"]]
    normalized = pd.DataFrame(
        {
            "row_position": frame[mapping["row"]].to_numpy(dtype=np.int64),
            "target": target.to_numpy(dtype=np.int64),
            "probability_positive": probability.to_numpy(dtype=np.float64),
        }
    )
    return validate_oof_frame(
        normalized,
        train_row_count=dataset.manifest.train_row_count,
        expected_target=dataset.artifacts.y_train,
    )


def _normalize_test(frame: pd.DataFrame, dataset: DatasetPackage) -> pd.DataFrame:
    mapping = _resolve_schema_columns(frame, label="test", allow_target=False)
    _validate_row_evidence(
        frame[mapping["row"]],
        expected_count=dataset.manifest.test_row_count,
        label="test",
    )
    probability = frame[mapping["probability"]]
    validate_probability_series(probability, field_name="explicit_test.probability")
    normalized = pd.DataFrame(
        {
            "row_position": frame[mapping["row"]].to_numpy(dtype=np.int64),
            "probability_positive": probability.to_numpy(dtype=np.float64),
        }
    )
    return validate_test_frame(
        normalized, test_row_count=dataset.manifest.test_row_count
    )


def _resolve_schema_columns(
    frame: pd.DataFrame, *, label: str, allow_target: bool
) -> dict[str, str | None]:
    if frame.columns.duplicated().any():
        raise CandidateValidationError(
            f"{label} export has duplicate column names.",
            reason_code="explicit_schema_duplicate_columns",
        )
    row = _one_alias(frame, ROW_ALIASES, label=label, role="row position")
    probability = _one_alias(
        frame, PROBABILITY_ALIASES, label=label, role="probability"
    )
    target_matches = [name for name in TARGET_ALIASES if name in frame.columns]
    if len(target_matches) > 1:
        raise CandidateValidationError(
            f"{label} export has ambiguous target columns: {target_matches}.",
            reason_code="explicit_schema_ambiguous",
        )
    target = target_matches[0] if target_matches else None
    if target is not None and not allow_target:
        raise CandidateValidationError(
            "Test export must not contain target values.",
            reason_code="explicit_test_target_present",
        )
    expected = {row, probability}
    if target is not None:
        expected.add(target)
    unknown = [str(column) for column in frame.columns if column not in expected]
    if unknown:
        raise CandidateValidationError(
            f"{label} export has unsupported columns: {unknown}.",
            reason_code="explicit_schema_unknown_columns",
        )
    return {"row": row, "probability": probability, "target": target}


def _one_alias(
    frame: pd.DataFrame, aliases: tuple[str, ...], *, label: str, role: str
) -> str:
    matches = [name for name in aliases if name in frame.columns]
    if not matches:
        raise CandidateValidationError(
            f"{label} export lacks explicit {role} evidence; accepted columns: {list(aliases)}.",
            reason_code="explicit_identity_evidence_missing",
        )
    if len(matches) > 1:
        raise CandidateValidationError(
            f"{label} export has ambiguous {role} columns: {matches}.",
            reason_code="explicit_schema_ambiguous",
        )
    return matches[0]


def _validate_row_evidence(
    positions: pd.Series, *, expected_count: int, label: str
) -> None:
    if not pd.api.types.is_integer_dtype(positions.dtype):
        raise CandidateValidationError(
            f"{label} row position column must be integer.",
            reason_code="explicit_row_position_dtype",
        )
    values = positions.astype("int64").tolist()
    if positions.duplicated().any():
        raise CandidateValidationError(
            f"{label} export has duplicate row positions.",
            reason_code="explicit_duplicate_rows",
        )
    expected = set(range(expected_count))
    actual = set(values)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        raise CandidateValidationError(
            f"{label} export is missing row positions: {missing[:5]}.",
            reason_code="explicit_missing_rows",
        )
    if unexpected:
        raise CandidateValidationError(
            f"{label} export has unexpected row positions: {unexpected[:5]}.",
            reason_code="explicit_unexpected_rows",
        )
    if len(values) != expected_count:
        raise CandidateValidationError(
            f"{label} export row count {len(values)} != {expected_count}.",
            reason_code="explicit_row_count_mismatch",
        )
    if values != list(range(expected_count)):
        raise CandidateValidationError(
            f"{label} export is not in exact canonical row order.",
            reason_code="explicit_row_order_mismatch",
        )


def _validate_request_values(request: ExplicitExportRequest) -> None:
    for value, label in (
        (request.dataset_id, "dataset_id"),
        (request.source_kind, "source_kind"),
    ):
        if SAFE_IDENTIFIER.fullmatch(str(value)) is None:
            raise PredictionCandidateError(
                f"{label} must be an explicit safe identifier.",
                reason_code="explicit_provenance_invalid",
            )
    if not str(request.source_model_name).strip():
        raise PredictionCandidateError(
            "source_model_name is required.",
            reason_code="explicit_provenance_missing",
        )
    if not str(request.source_run_path).strip():
        raise PredictionCandidateError(
            "source_run_path is required.",
            reason_code="explicit_provenance_missing",
        )
    _validate_score(request.threshold, "threshold", required=True)
    _validate_score(
        request.validation_balanced_accuracy,
        "validation_balanced_accuracy",
        required=False,
    )
    _validate_score(
        request.historical_kaggle_public_score,
        "historical_kaggle_public_score",
        required=False,
    )


def _validate_score(value: float | None, label: str, *, required: bool) -> None:
    if value is None:
        if required:
            raise PredictionCandidateError(
                f"{label} is required.", reason_code="explicit_provenance_missing"
            )
        return
    number = float(value)
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        raise PredictionCandidateError(
            f"{label} must be finite and within [0, 1].",
            reason_code="explicit_provenance_invalid",
        )


def _resolve_existing_relative_path(
    path: Path,
    repository_root: Path,
    *,
    label: str,
    require_file: bool,
) -> tuple[Path, str]:
    if path.is_absolute():
        raise PredictionCandidateError(
            f"{label} must be repository-relative; absolute paths are rejected.",
            reason_code="path_traversal",
        )
    relative = path.as_posix().replace("\\", "/")
    resolved = resolve_under_repository(relative, repository_root)
    if require_file and not resolved.is_file():
        raise PredictionCandidateError(
            f"{label} file not found: {relative}",
            reason_code="explicit_export_missing",
        )
    if not require_file and not resolved.exists():
        raise PredictionCandidateError(
            f"{label} path not found: {relative}",
            reason_code="explicit_provenance_missing",
        )
    return resolved, repository_relative_path(resolved, repository_root)


def _schema_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "columns": [str(column) for column in frame.columns],
        "dtypes": {str(column): str(frame[column].dtype) for column in frame.columns},
        "row_count": len(frame),
        "index_type": type(frame.index).__name__,
        "index_name": frame.index.name,
    }


def _dataset_identity(dataset: DatasetPackage) -> dict[str, Any]:
    return {
        "dataset_id": dataset.dataset_id,
        "target_dependency": dataset.manifest.target_dependency,
        "target_hash": dataset.manifest.target.hash,
        "train_anchor_hash": dataset.manifest.row_identity.train_anchor_hash,
        "test_anchor_hash": dataset.manifest.row_identity.test_anchor_hash,
        "train_row_count": dataset.manifest.train_row_count,
        "test_row_count": dataset.manifest.test_row_count,
        "canonical_train_order": "row_position_0_to_n_minus_1",
        "canonical_test_order": "row_position_0_to_n_minus_1",
    }


__all__ = [
    "ExplicitExportRequest",
    "PreparedExplicitExport",
    "import_explicit_export_candidate",
    "prepare_explicit_export",
    "validate_explicit_export",
]
