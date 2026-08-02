"""Immutable Prediction Candidate contract ``prediction_candidate_v1``.

Source-neutral package for one aligned OOF probability per training row and one
test probability per competition test row, authenticated against a Dataset
Package. AutoGluon-specific extraction belongs in ``autogluon_v1``.

Future Research v2 adapters should normalize repeated OOF rows to a single
probability per ``row_position`` under a deterministic declared policy (for
example mean probability across repeats) and record that policy in
``oof_protocol``. Shared fold assignments with AutoGluon are not required.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.churn_ml.competition_assets_v1 import resolve_competition_assets
from src.churn_ml.control_panel.path_safety import PathSafetyError, is_within
from src.churn_ml.dataset_registry.api import resolve_dataset_package
from src.churn_ml.dataset_registry.errors import DatasetRegistryError
from src.churn_ml.dataset_registry.hashing import target_hash as compute_target_hash
from src.churn_ml.research_data import canonical_sha256


SCHEMA_VERSION = "prediction_candidate_v1"
CANDIDATE_TYPE = "prediction_candidate"
CANDIDATE_ROOT_RELATIVE = "artifacts/prediction_candidates"
MANIFEST_FILENAME = "candidate_manifest.json"
OOF_FILENAME = "oof_predictions.parquet"
TEST_FILENAME = "test_predictions.parquet"
SOURCE_METADATA_FILENAME = "source_metadata.json"
SUCCESS_FILENAME = "_SUCCESS"
UNAVAILABLE = "unavailable"

OOF_COLUMNS = ("row_position", "target", "probability_positive")
TEST_COLUMNS = ("row_position", "probability_positive")
POSITIVE_CLASS_LABEL = 1
PROBABILITY_SEMANTICS = "P(y=1)"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_ID_RE = re.compile(r"^pc1_[0-9a-f]{16}$")

REQUIRED_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "candidate_type",
        "source_kind",
        "created_at_utc",
        "dataset_id",
        "parent_dataset_id",
        "target_dependency",
        "exploratory",
        "source_run_path",
        "source_config_path",
        "source_config_sha256",
        "source_model_name",
        "source_model_type",
        "source_predictor_path",
        "source_autogluon_version",
        "positive_class_label",
        "probability_semantics",
        "train_row_count",
        "test_row_count",
        "target_hash",
        "train_anchor_hash",
        "train_row_position_hash",
        "test_anchor_hash",
        "test_row_position_hash",
        "ordered_submission_id_hash",
        "oof_protocol",
        "oof_prediction_reference",
        "test_prediction_reference",
        "source_metric_name",
        "source_metric_value",
        "source_leaderboard_metadata",
        "provenance",
    }
)


class PredictionCandidateError(ValueError):
    """Base error for prediction-candidate contract failures."""

    def __init__(self, message: str, *, reason_code: str = "candidate_error") -> None:
        self.reason_code = reason_code
        super().__init__(message)


class CandidateValidationError(PredictionCandidateError):
    """Raised when a candidate package fails strict validation."""

    def __init__(self, message: str, *, reason_code: str = "candidate_invalid") -> None:
        super().__init__(message, reason_code=reason_code)


class CandidateConflictError(PredictionCandidateError):
    """Raised when an existing candidate ID has incompatible content."""

    def __init__(self, message: str) -> None:
        super().__init__(message, reason_code="candidate_conflict")


@dataclass(frozen=True)
class FileReference:
    path: str
    sha256: str
    size_bytes: int
    row_count: int
    columns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "row_count": self.row_count,
            "columns": list(self.columns),
        }


@dataclass(frozen=True)
class CandidatePackage:
    root: Path
    package_dir: Path
    candidate_id: str
    manifest: dict[str, Any]
    oof: pd.DataFrame
    test: pd.DataFrame
    source_metadata: dict[str, Any]


def candidate_dir_name(candidate_id: str) -> str:
    if not _CANDIDATE_ID_RE.fullmatch(candidate_id):
        raise CandidateValidationError(
            f"Invalid candidate_id format: {candidate_id!r}",
            reason_code="invalid_candidate_id",
        )
    return candidate_id


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def repository_relative_path(path: Path, repository_root: Path) -> str:
    """Return a POSIX repository-relative path; reject escapes."""
    root = repository_root.resolve()
    resolved = path if path.is_absolute() else (root / path)
    try:
        resolved = resolved.resolve(strict=False)
    except OSError as error:
        raise PredictionCandidateError(
            f"Cannot resolve path: {path}",
            reason_code="path_unresolvable",
        ) from error
    if not is_within(resolved, root):
        raise PredictionCandidateError(
            f"Path escapes repository root: {path}",
            reason_code="path_traversal",
        )
    return resolved.relative_to(root).as_posix()


def resolve_under_repository(relative: str, repository_root: Path) -> Path:
    if not relative or relative.startswith("/") or PurePosixPath(relative).is_absolute():
        raise PredictionCandidateError(
            f"Absolute or empty path rejected: {relative!r}",
            reason_code="path_traversal",
        )
    if ".." in PurePosixPath(relative).parts:
        raise PredictionCandidateError(
            f"Path traversal rejected: {relative!r}",
            reason_code="path_traversal",
        )
    root = repository_root.resolve()
    resolved = (root / Path(*PurePosixPath(relative).parts)).resolve(strict=False)
    if not is_within(resolved, root):
        raise PredictionCandidateError(
            f"Path escapes repository root: {relative!r}",
            reason_code="path_traversal",
        )
    return resolved


def build_candidate_id(identity: Mapping[str, Any]) -> str:
    """Deterministic candidate ID from source/dataset/model identity fields."""
    required = (
        "schema_version",
        "source_kind",
        "dataset_id",
        "source_run_path",
        "source_config_sha256",
        "source_model_name",
        "oof_protocol",
        "positive_class_label",
        "probability_semantics",
    )
    missing = [key for key in required if key not in identity]
    if missing:
        raise PredictionCandidateError(
            f"Candidate identity missing keys: {missing}",
            reason_code="identity_incomplete",
        )
    payload = {key: identity[key] for key in required}
    digest = canonical_sha256(payload)
    return f"pc1_{digest[:16]}"


def row_position_hash(row_count: int) -> str:
    if row_count < 0:
        raise PredictionCandidateError("row_count must be non-negative")
    return canonical_sha256(list(range(row_count)))


_FILE_SHA256_MEMO: dict[tuple[str, int, int], str] = {}
_FILE_SHA256_MEMO_MAX = 4096


def clear_file_sha256_memo() -> None:
    """Clear process-local SHA256 memoization used by ``file_sha256``."""
    _FILE_SHA256_MEMO.clear()


def file_sha256(path: Path) -> str:
    """Return SHA256 of a file, memoized by resolved path + size + mtime_ns.

    The digest is never reused when size or mtime changes. Callers that need a
    forced re-read after an in-place rewrite with identical size/mtime should
    call ``clear_file_sha256_memo()``.
    """
    resolved = path.resolve()
    stat = resolved.stat()
    key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    cached = _FILE_SHA256_MEMO.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    hex_digest = digest.hexdigest()
    if len(_FILE_SHA256_MEMO) >= _FILE_SHA256_MEMO_MAX:
        _FILE_SHA256_MEMO.pop(next(iter(_FILE_SHA256_MEMO)))
    _FILE_SHA256_MEMO[key] = hex_digest
    return hex_digest


def build_file_reference(
    path: Path,
    *,
    repository_root: Path,
    row_count: int,
    columns: Sequence[str],
) -> FileReference:
    relative = repository_relative_path(path, repository_root)
    return FileReference(
        path=relative,
        sha256=file_sha256(path),
        size_bytes=path.stat().st_size,
        row_count=row_count,
        columns=tuple(columns),
    )


def validate_probability_series(
    values: pd.Series,
    *,
    field_name: str,
) -> np.ndarray:
    """Reject non-finite, out-of-range, or hard-label masquerades."""
    if pd.api.types.is_integer_dtype(values.dtype) or pd.api.types.is_bool_dtype(
        values.dtype
    ):
        raise CandidateValidationError(
            f"{field_name} must be floating-point probabilities, not labels "
            f"(dtype={values.dtype}).",
            reason_code="label_masquerading_as_probability",
        )
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise CandidateValidationError(
            f"{field_name} must be one-dimensional.",
            reason_code="probability_shape_invalid",
        )
    if not np.isfinite(array).all():
        raise CandidateValidationError(
            f"{field_name} contains non-finite values.",
            reason_code="probability_nonfinite",
        )
    if ((array < 0.0) | (array > 1.0)).any():
        raise CandidateValidationError(
            f"{field_name} contains values outside [0, 1].",
            reason_code="probability_out_of_range",
        )
    unique = np.unique(array)
    if unique.size > 0 and set(unique.tolist()).issubset({0.0, 1.0}):
        raise CandidateValidationError(
            f"{field_name} contains only hard 0/1 values; label predictions "
            "cannot masquerade as probabilities.",
            reason_code="label_masquerading_as_probability",
        )
    return array


def validate_oof_frame(
    frame: pd.DataFrame,
    *,
    train_row_count: int,
    expected_target: pd.Series,
) -> pd.DataFrame:
    if list(frame.columns) != list(OOF_COLUMNS):
        raise CandidateValidationError(
            f"OOF columns must be exactly {list(OOF_COLUMNS)}; got {list(frame.columns)}.",
            reason_code="oof_schema_invalid",
        )
    if len(frame) != train_row_count:
        raise CandidateValidationError(
            f"OOF row count {len(frame)} != train_row_count {train_row_count}.",
            reason_code="oof_row_count_mismatch",
        )
    positions = frame["row_position"]
    if not pd.api.types.is_integer_dtype(positions.dtype):
        raise CandidateValidationError(
            "OOF row_position must be integer.",
            reason_code="oof_row_position_dtype",
        )
    expected_positions = list(range(train_row_count))
    if positions.tolist() != expected_positions:
        if positions.duplicated().any():
            raise CandidateValidationError(
                "OOF contains duplicate row_position values.",
                reason_code="oof_duplicate_rows",
            )
        missing = sorted(set(expected_positions) - set(positions.tolist()))
        unexpected = sorted(set(positions.tolist()) - set(expected_positions))
        if missing:
            raise CandidateValidationError(
                f"OOF missing row_position values: {missing[:5]}",
                reason_code="oof_missing_rows",
            )
        raise CandidateValidationError(
            f"OOF unexpected row_position values: {unexpected[:5]}",
            reason_code="oof_unexpected_rows",
        )
    target = frame["target"]
    if not set(pd.unique(target)).issubset({0, 1}):
        raise CandidateValidationError(
            "OOF target must be binary in {0, 1}.",
            reason_code="target_not_binary",
        )
    expected_values = expected_target.astype("int64").tolist()
    if target.astype("int64").tolist() != expected_values:
        raise CandidateValidationError(
            "OOF target does not match the Dataset Package authoritative target "
            "order and values.",
            reason_code="target_mismatch",
        )
    validate_probability_series(
        frame["probability_positive"],
        field_name="oof.probability_positive",
    )
    return frame.loc[:, list(OOF_COLUMNS)].reset_index(drop=True)


def validate_test_frame(frame: pd.DataFrame, *, test_row_count: int) -> pd.DataFrame:
    if list(frame.columns) != list(TEST_COLUMNS):
        raise CandidateValidationError(
            f"Test columns must be exactly {list(TEST_COLUMNS)}; got {list(frame.columns)}.",
            reason_code="test_schema_invalid",
        )
    if len(frame) != test_row_count:
        raise CandidateValidationError(
            f"Test row count {len(frame)} != test_row_count {test_row_count}.",
            reason_code="test_row_count_mismatch",
        )
    positions = frame["row_position"]
    if not pd.api.types.is_integer_dtype(positions.dtype):
        raise CandidateValidationError(
            "Test row_position must be integer.",
            reason_code="test_row_position_dtype",
        )
    expected_positions = list(range(test_row_count))
    if positions.tolist() != expected_positions:
        if positions.duplicated().any():
            raise CandidateValidationError(
                "Test predictions contain duplicate row_position values.",
                reason_code="test_duplicate_rows",
            )
        missing = sorted(set(expected_positions) - set(positions.tolist()))
        if missing:
            raise CandidateValidationError(
                f"Test predictions missing row_position values: {missing[:5]}",
                reason_code="test_missing_rows",
            )
        raise CandidateValidationError(
            "Test predictions are not in exact Dataset Package test order.",
            reason_code="test_order_mismatch",
        )
    validate_probability_series(
        frame["probability_positive"],
        field_name="test.probability_positive",
    )
    return frame.loc[:, list(TEST_COLUMNS)].reset_index(drop=True)


def _optional_or_value(value: Any) -> Any:
    if value is None:
        return UNAVAILABLE
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise CandidateValidationError(
            f"{field} must be a 64-character lowercase hex SHA-256 digest.",
            reason_code="hash_invalid",
        )
    return value


def _require_file_reference(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CandidateValidationError(
            f"{field} must be an object.",
            reason_code="file_reference_invalid",
        )
    required = {"path", "sha256", "size_bytes", "row_count", "columns"}
    missing = required - set(value)
    if missing:
        raise CandidateValidationError(
            f"{field} missing keys: {sorted(missing)}",
            reason_code="file_reference_invalid",
        )
    path = value["path"]
    if not isinstance(path, str) or not path or path.startswith("/") or "\\" in path:
        raise CandidateValidationError(
            f"{field}.path must be a repository-relative POSIX path.",
            reason_code="path_invalid",
        )
    if ".." in PurePosixPath(path).parts:
        raise CandidateValidationError(
            f"{field}.path rejects traversal.",
            reason_code="path_traversal",
        )
    _require_sha256(value["sha256"], f"{field}.sha256")
    if type(value["size_bytes"]) is not int or value["size_bytes"] < 0:
        raise CandidateValidationError(
            f"{field}.size_bytes must be a non-negative int.",
            reason_code="file_reference_invalid",
        )
    if type(value["row_count"]) is not int or value["row_count"] < 0:
        raise CandidateValidationError(
            f"{field}.row_count must be a non-negative int.",
            reason_code="file_reference_invalid",
        )
    columns = value["columns"]
    if not isinstance(columns, list) or not all(isinstance(item, str) for item in columns):
        raise CandidateValidationError(
            f"{field}.columns must be a list of strings.",
            reason_code="file_reference_invalid",
        )
    return value


def load_candidate_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CandidateValidationError(
            f"Cannot load candidate manifest: {error}",
            reason_code="manifest_unreadable",
        ) from error
    if not isinstance(payload, dict):
        raise CandidateValidationError(
            "Candidate manifest must be a JSON object.",
            reason_code="manifest_not_object",
        )
    missing = sorted(REQUIRED_MANIFEST_KEYS - set(payload))
    if missing:
        raise CandidateValidationError(
            f"Candidate manifest missing keys: {missing}",
            reason_code="manifest_schema_invalid",
        )
    extra = sorted(set(payload) - REQUIRED_MANIFEST_KEYS)
    if extra:
        raise CandidateValidationError(
            f"Candidate manifest has unexpected keys: {extra}",
            reason_code="manifest_schema_invalid",
        )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise CandidateValidationError(
            f"Unsupported schema_version: {payload['schema_version']!r}",
            reason_code="schema_version_invalid",
        )
    if payload["candidate_type"] != CANDIDATE_TYPE:
        raise CandidateValidationError(
            f"candidate_type must be {CANDIDATE_TYPE!r}",
            reason_code="candidate_type_invalid",
        )
    if not _CANDIDATE_ID_RE.fullmatch(str(payload["candidate_id"])):
        raise CandidateValidationError(
            f"Invalid candidate_id: {payload['candidate_id']!r}",
            reason_code="invalid_candidate_id",
        )
    if payload["positive_class_label"] != POSITIVE_CLASS_LABEL:
        raise CandidateValidationError(
            "positive_class_label must be 1.",
            reason_code="positive_class_invalid",
        )
    if payload["probability_semantics"] != PROBABILITY_SEMANTICS:
        raise CandidateValidationError(
            f"probability_semantics must be {PROBABILITY_SEMANTICS!r}.",
            reason_code="probability_semantics_invalid",
        )
    _require_file_reference(payload["oof_prediction_reference"], "oof_prediction_reference")
    _require_file_reference(payload["test_prediction_reference"], "test_prediction_reference")
    for key in (
        "target_hash",
        "train_anchor_hash",
        "train_row_position_hash",
        "test_anchor_hash",
        "test_row_position_hash",
    ):
        _require_sha256(payload[key], key)
    ordered = payload["ordered_submission_id_hash"]
    if ordered != UNAVAILABLE:
        _require_sha256(ordered, "ordered_submission_id_hash")
    return payload


def normalize_candidates_root_relative(candidates_root_relative: str | None) -> str:
    """Return a repository-relative POSIX candidate root (no traversal)."""
    relative = candidates_root_relative or CANDIDATE_ROOT_RELATIVE
    if not relative or relative.startswith("/") or PurePosixPath(relative).is_absolute():
        raise PredictionCandidateError(
            f"Absolute or empty candidates root rejected: {relative!r}",
            reason_code="path_traversal",
        )
    if ".." in PurePosixPath(relative).parts:
        raise PredictionCandidateError(
            f"Path traversal rejected: {relative!r}",
            reason_code="path_traversal",
        )
    return PurePosixPath(relative).as_posix()


def discover_candidate_packages(
    repository_root: Path,
    *,
    candidates_root_relative: str | None = None,
) -> list[dict[str, Any]]:
    root = repository_root.resolve()
    relative = normalize_candidates_root_relative(candidates_root_relative)
    candidates_root = root / Path(*PurePosixPath(relative).parts)
    if not candidates_root.is_dir():
        return []
    summaries: list[dict[str, Any]] = []
    for child in sorted(candidates_root.iterdir(), key=lambda item: item.name):
        if not child.is_dir() or child.name.startswith("."):
            continue
        success = child / SUCCESS_FILENAME
        manifest_path = child / MANIFEST_FILENAME
        if not success.is_file() or not manifest_path.is_file():
            continue
        try:
            manifest = load_candidate_manifest(manifest_path)
            summaries.append(candidate_summary_from_manifest(manifest, child, root))
        except PredictionCandidateError:
            continue
    return summaries


def candidate_summary_from_manifest(
    manifest: Mapping[str, Any],
    package_dir: Path,
    repository_root: Path,
) -> dict[str, Any]:
    return {
        "candidate_id": manifest["candidate_id"],
        "source_kind": manifest["source_kind"],
        "source_model_name": manifest["source_model_name"],
        "dataset_id": manifest["dataset_id"],
        "train_row_count": manifest["train_row_count"],
        "test_row_count": manifest["test_row_count"],
        "oof_protocol": manifest["oof_protocol"],
        "source_metric_name": manifest["source_metric_name"],
        "source_metric_value": manifest["source_metric_value"],
        "package_path": repository_relative_path(package_dir, repository_root),
    }


def candidate_summary(package: CandidatePackage) -> dict[str, Any]:
    return candidate_summary_from_manifest(
        package.manifest,
        package.package_dir.resolve(),
        package.root.resolve(),
    )


def load_aligned_oof_probabilities(package: CandidatePackage) -> pd.Series:
    series = package.oof["probability_positive"].astype("float64")
    series.index = package.oof["row_position"].astype("int64")
    series.name = "probability_positive"
    return series


def load_aligned_test_probabilities(package: CandidatePackage) -> pd.Series:
    series = package.test["probability_positive"].astype("float64")
    series.index = package.test["row_position"].astype("int64")
    series.name = "probability_positive"
    return series


def load_candidate_package(
    package_dir: Path,
    *,
    repository_root: Path,
    candidates_root_relative: str | None = None,
) -> CandidatePackage:
    root = repository_root.resolve()
    directory = package_dir.resolve(strict=True)
    relative = normalize_candidates_root_relative(candidates_root_relative)
    candidates_root = root / Path(*PurePosixPath(relative).parts)
    if not is_within(directory, candidates_root):
        raise PredictionCandidateError(
            f"Candidate package is outside {relative}.",
            reason_code="path_traversal",
        )
    if not (directory / SUCCESS_FILENAME).is_file():
        raise CandidateValidationError(
            "Candidate package is missing _SUCCESS.",
            reason_code="success_missing",
        )
    manifest = load_candidate_manifest(directory / MANIFEST_FILENAME)
    expected_name = candidate_dir_name(str(manifest["candidate_id"]))
    if directory.name != expected_name:
        raise CandidateValidationError(
            "Package directory name does not match candidate_id.",
            reason_code="candidate_id_mismatch",
        )
    oof = pd.read_parquet(directory / OOF_FILENAME)
    test = pd.read_parquet(directory / TEST_FILENAME)
    source_metadata = json.loads(
        (directory / SOURCE_METADATA_FILENAME).read_text(encoding="utf-8")
    )
    if not isinstance(source_metadata, dict):
        raise CandidateValidationError(
            "source_metadata.json must be a JSON object.",
            reason_code="source_metadata_invalid",
        )
    package = CandidatePackage(
        root=root,
        package_dir=directory,
        candidate_id=str(manifest["candidate_id"]),
        manifest=manifest,
        oof=oof,
        test=test,
        source_metadata=source_metadata,
    )
    validate_candidate_package(package, repository_root=root)
    return package


def validate_candidate_package(
    package: CandidatePackage,
    *,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    root = (repository_root or package.root).resolve()
    manifest = package.manifest
    package_dir = package.package_dir.resolve()

    processed_root = root / "data" / "processed"
    try:
        dataset = resolve_dataset_package(processed_root, str(manifest["dataset_id"]))
    except DatasetRegistryError as error:
        raise CandidateValidationError(
            f"Dataset Package resolution failed: {error}",
            reason_code="dataset_unresolved",
        ) from error

    ds_manifest = dataset.manifest
    if manifest["train_row_count"] != ds_manifest.train_row_count:
        raise CandidateValidationError(
            "train_row_count does not match Dataset Package.",
            reason_code="train_row_count_mismatch",
        )
    if manifest["test_row_count"] != ds_manifest.test_row_count:
        raise CandidateValidationError(
            "test_row_count does not match Dataset Package.",
            reason_code="test_row_count_mismatch",
        )
    if manifest["train_anchor_hash"] != ds_manifest.row_identity.train_anchor_hash:
        raise CandidateValidationError(
            "train_anchor_hash does not match Dataset Package row identity.",
            reason_code="train_anchor_mismatch",
        )
    if manifest["test_anchor_hash"] != ds_manifest.row_identity.test_anchor_hash:
        raise CandidateValidationError(
            "test_anchor_hash does not match Dataset Package row identity.",
            reason_code="test_anchor_mismatch",
        )
    if manifest["target_hash"] != ds_manifest.target.hash:
        raise CandidateValidationError(
            "target_hash does not match Dataset Package target hash.",
            reason_code="target_hash_mismatch",
        )
    if manifest["parent_dataset_id"] not in {
        ds_manifest.parent_dataset_id,
        UNAVAILABLE if ds_manifest.parent_dataset_id is None else ds_manifest.parent_dataset_id,
    }:
        # Allow explicit unavailable only when parent is None.
        if not (
            manifest["parent_dataset_id"] == UNAVAILABLE
            and ds_manifest.parent_dataset_id is None
        ) and manifest["parent_dataset_id"] != ds_manifest.parent_dataset_id:
            raise CandidateValidationError(
                "parent_dataset_id does not match Dataset Package.",
                reason_code="parent_dataset_mismatch",
            )

    expected_train_pos = row_position_hash(ds_manifest.train_row_count)
    expected_test_pos = row_position_hash(ds_manifest.test_row_count)
    if manifest["train_row_position_hash"] != expected_train_pos:
        raise CandidateValidationError(
            "train_row_position_hash is incorrect.",
            reason_code="train_row_position_hash_mismatch",
        )
    if manifest["test_row_position_hash"] != expected_test_pos:
        raise CandidateValidationError(
            "test_row_position_hash is incorrect.",
            reason_code="test_row_position_hash_mismatch",
        )

    recomputed_target = compute_target_hash(dataset.artifacts.y_train)
    if recomputed_target != ds_manifest.target.hash:
        raise CandidateValidationError(
            "Dataset Package target hash failed self-check.",
            reason_code="dataset_target_corrupt",
        )

    oof = validate_oof_frame(
        package.oof,
        train_row_count=ds_manifest.train_row_count,
        expected_target=dataset.artifacts.y_train,
    )
    test = validate_test_frame(
        package.test,
        test_row_count=ds_manifest.test_row_count,
    )

    oof_ref = _require_file_reference(
        manifest["oof_prediction_reference"], "oof_prediction_reference"
    )
    test_ref = _require_file_reference(
        manifest["test_prediction_reference"], "test_prediction_reference"
    )
    oof_path = package_dir / OOF_FILENAME
    test_path = package_dir / TEST_FILENAME
    if oof_path.is_file():
        actual_oof = build_file_reference(
            oof_path,
            repository_root=root,
            row_count=len(oof),
            columns=OOF_COLUMNS,
        )
        if actual_oof.sha256 != oof_ref["sha256"] or actual_oof.row_count != oof_ref["row_count"]:
            raise CandidateValidationError(
                "OOF prediction reference does not match on-disk bytes.",
                reason_code="oof_reference_mismatch",
            )
        if list(oof_ref["columns"]) != list(OOF_COLUMNS):
            raise CandidateValidationError(
                "OOF prediction reference columns are incorrect.",
                reason_code="oof_reference_mismatch",
            )
    if test_path.is_file():
        actual_test = build_file_reference(
            test_path,
            repository_root=root,
            row_count=len(test),
            columns=TEST_COLUMNS,
        )
        if (
            actual_test.sha256 != test_ref["sha256"]
            or actual_test.row_count != test_ref["row_count"]
        ):
            raise CandidateValidationError(
                "Test prediction reference does not match on-disk bytes.",
                reason_code="test_reference_mismatch",
            )

    competition = resolve_competition_assets(
        root, dataset_version=str(manifest["dataset_id"])
    )
    ordered = manifest["ordered_submission_id_hash"]
    if competition.ready and competition.ordered_id_sha256 is not None:
        if ordered != competition.ordered_id_sha256:
            raise CandidateValidationError(
                "ordered_submission_id_hash does not match competition_assets_v1.",
                reason_code="submission_id_hash_mismatch",
            )
    elif ordered != UNAVAILABLE and not competition.ready:
        # Persist explicit unavailable when competition assets are not ready.
        pass

    return {
        "ok": True,
        "candidate_id": manifest["candidate_id"],
        "dataset_id": manifest["dataset_id"],
        "train_row_count": manifest["train_row_count"],
        "test_row_count": manifest["test_row_count"],
        "train_anchor_hash": manifest["train_anchor_hash"],
        "test_anchor_hash": manifest["test_anchor_hash"],
        "target_hash": manifest["target_hash"],
        "positive_class_label": manifest["positive_class_label"],
        "probability_semantics": manifest["probability_semantics"],
        "competition_assets_ready": competition.ready,
    }


def build_manifest(
    *,
    candidate_id: str,
    source_kind: str,
    created_at_utc: str,
    dataset_id: str,
    parent_dataset_id: str | None,
    target_dependency: str,
    exploratory: bool,
    source_run_path: str,
    source_config_path: str,
    source_config_sha256: str,
    source_model_name: str,
    source_model_type: Any,
    source_predictor_path: str,
    source_autogluon_version: Any,
    train_row_count: int,
    test_row_count: int,
    target_hash: str,
    train_anchor_hash: str,
    test_anchor_hash: str,
    ordered_submission_id_hash: Any,
    oof_protocol: str,
    oof_reference: FileReference,
    test_reference: FileReference,
    source_metric_name: Any,
    source_metric_value: Any,
    source_leaderboard_metadata: Any,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "candidate_type": CANDIDATE_TYPE,
        "source_kind": source_kind,
        "created_at_utc": created_at_utc,
        "dataset_id": dataset_id,
        "parent_dataset_id": _optional_or_value(parent_dataset_id),
        "target_dependency": target_dependency,
        "exploratory": bool(exploratory),
        "source_run_path": source_run_path,
        "source_config_path": source_config_path,
        "source_config_sha256": source_config_sha256,
        "source_model_name": source_model_name,
        "source_model_type": _optional_or_value(source_model_type),
        "source_predictor_path": source_predictor_path,
        "source_autogluon_version": _optional_or_value(source_autogluon_version),
        "positive_class_label": POSITIVE_CLASS_LABEL,
        "probability_semantics": PROBABILITY_SEMANTICS,
        "train_row_count": train_row_count,
        "test_row_count": test_row_count,
        "target_hash": target_hash,
        "train_anchor_hash": train_anchor_hash,
        "train_row_position_hash": row_position_hash(train_row_count),
        "test_anchor_hash": test_anchor_hash,
        "test_row_position_hash": row_position_hash(test_row_count),
        "ordered_submission_id_hash": _optional_or_value(ordered_submission_id_hash),
        "oof_protocol": oof_protocol,
        "oof_prediction_reference": oof_reference.to_dict(),
        "test_prediction_reference": test_reference.to_dict(),
        "source_metric_name": _optional_or_value(source_metric_name),
        "source_metric_value": _optional_or_value(source_metric_value),
        "source_leaderboard_metadata": _optional_or_value(source_leaderboard_metadata),
        "provenance": dict(provenance),
    }


def _atomic_write_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    replaced = False
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        replaced = True
    finally:
        if not replaced:
            temporary_path.unlink(missing_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write_bytes(path, raw)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _package_content_fingerprint(
    oof: pd.DataFrame,
    test: pd.DataFrame,
    manifest_without_created: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
) -> str:
    return canonical_sha256(
        {
            "manifest": dict(manifest_without_created),
            "source_metadata": dict(source_metadata),
            "oof": oof.loc[:, list(OOF_COLUMNS)].astype(
                {
                    "row_position": "int64",
                    "target": "int64",
                    "probability_positive": "float64",
                }
            ).to_dict(orient="list"),
            "test": test.loc[:, list(TEST_COLUMNS)].astype(
                {
                    "row_position": "int64",
                    "probability_positive": "float64",
                }
            ).to_dict(orient="list"),
        }
    )


def create_candidate_package(
    *,
    repository_root: Path,
    identity: Mapping[str, Any],
    oof: pd.DataFrame,
    test: pd.DataFrame,
    manifest_fields: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
    candidates_root_relative: str | None = None,
) -> CandidatePackage:
    """Create an immutable candidate package atomically.

    Idempotent when an existing successful package has identical content.
    Conflicts when the same candidate_id exists with different content.
    """
    root = repository_root.resolve()
    relative_root = normalize_candidates_root_relative(candidates_root_relative)
    candidates_root = root / Path(*PurePosixPath(relative_root).parts)
    candidates_root.mkdir(parents=True, exist_ok=True)

    candidate_id = build_candidate_id(identity)
    if "candidate_id" in manifest_fields and manifest_fields["candidate_id"] != candidate_id:
        raise PredictionCandidateError(
            "manifest candidate_id does not match deterministic identity.",
            reason_code="candidate_id_mismatch",
        )

    processed_root = root / "data" / "processed"
    dataset = resolve_dataset_package(processed_root, str(manifest_fields["dataset_id"]))
    oof_valid = validate_oof_frame(
        oof,
        train_row_count=dataset.manifest.train_row_count,
        expected_target=dataset.artifacts.y_train,
    )
    test_valid = validate_test_frame(
        test,
        test_row_count=dataset.manifest.test_row_count,
    )

    final_dir = candidates_root / candidate_id
    created_at = str(manifest_fields.get("created_at_utc") or utc_now())

    staging = candidates_root / f".staging-{candidate_id}-{uuid.uuid4().hex}"
    if staging.exists():
        raise PredictionCandidateError(
            f"Staging directory already exists: {staging}",
            reason_code="staging_exists",
        )
    staging.mkdir(parents=False, exist_ok=False)
    try:
        _write_parquet(staging / OOF_FILENAME, oof_valid)
        _write_parquet(staging / TEST_FILENAME, test_valid)
        oof_ref = build_file_reference(
            staging / OOF_FILENAME,
            repository_root=root,
            row_count=len(oof_valid),
            columns=OOF_COLUMNS,
        )
        # Rewrite references to final relative paths before persisting manifest.
        oof_ref = FileReference(
            path=f"{relative_root}/{candidate_id}/{OOF_FILENAME}",
            sha256=oof_ref.sha256,
            size_bytes=oof_ref.size_bytes,
            row_count=oof_ref.row_count,
            columns=oof_ref.columns,
        )
        test_ref_tmp = build_file_reference(
            staging / TEST_FILENAME,
            repository_root=root,
            row_count=len(test_valid),
            columns=TEST_COLUMNS,
        )
        test_ref = FileReference(
            path=f"{relative_root}/{candidate_id}/{TEST_FILENAME}",
            sha256=test_ref_tmp.sha256,
            size_bytes=test_ref_tmp.size_bytes,
            row_count=test_ref_tmp.row_count,
            columns=test_ref_tmp.columns,
        )

        competition = resolve_competition_assets(
            root, dataset_version=str(manifest_fields["dataset_id"])
        )
        ordered_hash: Any = (
            competition.ordered_id_sha256
            if competition.ready and competition.ordered_id_sha256 is not None
            else UNAVAILABLE
        )

        manifest = build_manifest(
            candidate_id=candidate_id,
            source_kind=str(identity["source_kind"]),
            created_at_utc=created_at,
            dataset_id=str(manifest_fields["dataset_id"]),
            parent_dataset_id=dataset.manifest.parent_dataset_id,
            target_dependency=str(dataset.manifest.target_dependency),
            exploratory=bool(manifest_fields.get("exploratory", False)),
            source_run_path=str(manifest_fields["source_run_path"]),
            source_config_path=str(manifest_fields["source_config_path"]),
            source_config_sha256=str(manifest_fields["source_config_sha256"]),
            source_model_name=str(manifest_fields["source_model_name"]),
            source_model_type=manifest_fields.get("source_model_type"),
            source_predictor_path=str(manifest_fields["source_predictor_path"]),
            source_autogluon_version=manifest_fields.get("source_autogluon_version"),
            train_row_count=dataset.manifest.train_row_count,
            test_row_count=dataset.manifest.test_row_count,
            target_hash=dataset.manifest.target.hash,
            train_anchor_hash=dataset.manifest.row_identity.train_anchor_hash,
            test_anchor_hash=dataset.manifest.row_identity.test_anchor_hash,
            ordered_submission_id_hash=ordered_hash,
            oof_protocol=str(identity["oof_protocol"]),
            oof_reference=oof_ref,
            test_reference=test_ref,
            source_metric_name=manifest_fields.get("source_metric_name"),
            source_metric_value=manifest_fields.get("source_metric_value"),
            source_leaderboard_metadata=manifest_fields.get(
                "source_leaderboard_metadata"
            ),
            provenance=dict(manifest_fields.get("provenance") or {}),
        )
        # Fix staging-local absolute leakage: file refs already relative.
        _write_json(staging / MANIFEST_FILENAME, manifest)
        _write_json(staging / SOURCE_METADATA_FILENAME, dict(source_metadata))
        # Validate against staging paths by temporarily constructing package.
        staging_package = CandidatePackage(
            root=root,
            package_dir=staging,
            candidate_id=candidate_id,
            manifest=manifest,
            oof=oof_valid,
            test=test_valid,
            source_metadata=dict(source_metadata),
        )
        # Point package_dir checks at staging for byte verification.
        _validate_staging_package(staging_package, staging_dir=staging, repository_root=root)

        success_payload = {
            "schema_version": 1,
            "candidate_id": candidate_id,
            "status": "completed",
            "manifest_sha256": file_sha256(staging / MANIFEST_FILENAME),
            "completed_at_utc": utc_now(),
        }
        _write_json(staging / SUCCESS_FILENAME, success_payload)

        if final_dir.exists():
            if not (final_dir / SUCCESS_FILENAME).is_file():
                raise CandidateConflictError(
                    f"Incomplete candidate package already exists: {final_dir}"
                )
            existing = load_candidate_package(
                final_dir,
                repository_root=root,
                candidates_root_relative=relative_root,
            )
            existing_fp = _package_content_fingerprint(
                existing.oof,
                existing.test,
                _manifest_identity_view(existing.manifest),
                existing.source_metadata,
            )
            new_fp = _package_content_fingerprint(
                oof_valid,
                test_valid,
                _manifest_identity_view(manifest),
                dict(source_metadata),
            )
            if existing_fp != new_fp:
                raise CandidateConflictError(
                    f"Candidate {candidate_id} already exists with different content."
                )
            shutil.rmtree(staging, ignore_errors=True)
            return existing

        os.replace(staging, final_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    return load_candidate_package(
        final_dir,
        repository_root=root,
        candidates_root_relative=relative_root,
    )


def _manifest_identity_view(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(manifest)
    payload.pop("created_at_utc", None)
    # File references keep sha256/size which already authenticate bytes.
    return payload


def _validate_staging_package(
    package: CandidatePackage,
    *,
    staging_dir: Path,
    repository_root: Path,
) -> None:
    """Validate prediction tables and dataset identity before _SUCCESS."""
    root = repository_root.resolve()
    processed_root = root / "data" / "processed"
    dataset = resolve_dataset_package(
        processed_root, str(package.manifest["dataset_id"])
    )
    validate_oof_frame(
        package.oof,
        train_row_count=dataset.manifest.train_row_count,
        expected_target=dataset.artifacts.y_train,
    )
    validate_test_frame(
        package.test,
        test_row_count=dataset.manifest.test_row_count,
    )
    if package.manifest["train_anchor_hash"] != dataset.manifest.row_identity.train_anchor_hash:
        raise CandidateValidationError(
            "train_anchor_hash mismatch during package creation.",
            reason_code="train_anchor_mismatch",
        )
    if package.manifest["test_anchor_hash"] != dataset.manifest.row_identity.test_anchor_hash:
        raise CandidateValidationError(
            "test_anchor_hash mismatch during package creation.",
            reason_code="test_anchor_mismatch",
        )
    oof_sha = file_sha256(staging_dir / OOF_FILENAME)
    test_sha = file_sha256(staging_dir / TEST_FILENAME)
    if oof_sha != package.manifest["oof_prediction_reference"]["sha256"]:
        raise CandidateValidationError(
            "Staging OOF digest mismatch.",
            reason_code="oof_reference_mismatch",
        )
    if test_sha != package.manifest["test_prediction_reference"]["sha256"]:
        raise CandidateValidationError(
            "Staging test digest mismatch.",
            reason_code="test_reference_mismatch",
        )


def assert_finite_unit_interval(values: Sequence[float], *, field: str) -> None:
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CandidateValidationError(
                f"{field}[{index}] is not a numeric probability.",
                reason_code="probability_invalid",
            )
        number = float(value)
        if not math.isfinite(number) or number < 0.0 or number > 1.0:
            raise CandidateValidationError(
                f"{field}[{index}] is outside the finite unit interval.",
                reason_code="probability_out_of_range",
            )


# Re-export PathSafetyError for adapters that share path checks.
__all__ = [
    "CANDIDATE_ROOT_RELATIVE",
    "CANDIDATE_TYPE",
    "MANIFEST_FILENAME",
    "OOF_FILENAME",
    "SCHEMA_VERSION",
    "SUCCESS_FILENAME",
    "SOURCE_METADATA_FILENAME",
    "TEST_FILENAME",
    "UNAVAILABLE",
    "CandidateConflictError",
    "CandidatePackage",
    "CandidateValidationError",
    "FileReference",
    "PathSafetyError",
    "PredictionCandidateError",
    "build_candidate_id",
    "build_file_reference",
    "build_manifest",
    "candidate_summary",
    "create_candidate_package",
    "discover_candidate_packages",
    "file_sha256",
    "load_aligned_oof_probabilities",
    "load_aligned_test_probabilities",
    "load_candidate_manifest",
    "load_candidate_package",
    "normalize_candidates_root_relative",
    "repository_relative_path",
    "resolve_under_repository",
    "row_position_hash",
    "utc_now",
    "validate_candidate_package",
    "validate_oof_frame",
    "validate_probability_series",
    "validate_test_frame",
]
