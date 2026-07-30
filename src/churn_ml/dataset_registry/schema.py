"""Strict dataset_manifest.json schema for Dataset Package & Registry v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from src.churn_ml.dataset_registry.constants import (
    FEATURE_ROLE_VALUES,
    SCHEMA_VERSION,
    TARGET_DEPENDENCY_VALUES,
)
from src.churn_ml.dataset_registry.errors import DatasetRegistryError

FeatureRole = Literal["numeric", "categorical", "binary_indicator", "summary"]
TargetDependency = Literal["none", "exploratory", "fold_local"]
AlignmentStatus = Literal["proven", "unverifiable"]

REQUIRED_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "parent_dataset_id",
        "hypothesis",
        "files",
        "train_row_count",
        "test_row_count",
        "n_features",
        "features",
        "transformations",
        "target_dependency",
        "schema_hash",
        "content_hashes",
        "target",
        "row_identity",
    }
)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dtype: str
    role: FeatureRole

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "dtype": self.dtype, "role": self.role}


@dataclass(frozen=True)
class TargetSpec:
    name: str
    dtype: str
    class_counts: dict[str, int]
    hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "class_counts": dict(self.class_counts),
            "hash": self.hash,
        }


@dataclass(frozen=True)
class RowIdentitySpec:
    train_hash: str
    test_hash: str
    alignment_status: AlignmentStatus
    alignment_method: str
    train_anchor_hash: str
    test_anchor_hash: str

    def to_dict(self) -> dict[str, str]:
        return {
            "train_hash": self.train_hash,
            "test_hash": self.test_hash,
            "alignment_status": self.alignment_status,
            "alignment_method": self.alignment_method,
            "train_anchor_hash": self.train_anchor_hash,
            "test_anchor_hash": self.test_anchor_hash,
        }


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: str
    dataset_id: str
    parent_dataset_id: str | None
    hypothesis: str
    files: dict[str, str]
    train_row_count: int
    test_row_count: int
    n_features: int
    features: tuple[FeatureSpec, ...]
    transformations: tuple[Mapping[str, Any], ...]
    target_dependency: TargetDependency
    schema_hash: str
    content_hashes: dict[str, str]
    target: TargetSpec
    row_identity: RowIdentitySpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "parent_dataset_id": self.parent_dataset_id,
            "hypothesis": self.hypothesis,
            "files": dict(self.files),
            "train_row_count": self.train_row_count,
            "test_row_count": self.test_row_count,
            "n_features": self.n_features,
            "features": [feature.to_dict() for feature in self.features],
            "transformations": [dict(item) for item in self.transformations],
            "target_dependency": self.target_dependency,
            "schema_hash": self.schema_hash,
            "content_hashes": dict(self.content_hashes),
            "target": self.target.to_dict(),
            "row_identity": self.row_identity.to_dict(),
        }


def parse_manifest(payload: Mapping[str, Any]) -> DatasetManifest:
    """Strictly parse a manifest mapping. Unknown or missing keys fail."""
    _exact_keys(payload, required=REQUIRED_TOP_LEVEL_KEYS, label="dataset_manifest")
    schema_version = _string(payload["schema_version"], "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise DatasetRegistryError(
            f"Unsupported schema_version: {schema_version!r}."
        )
    dataset_id = _string(payload["dataset_id"], "dataset_id")
    parent_raw = payload["parent_dataset_id"]
    if parent_raw is not None and not isinstance(parent_raw, str):
        raise DatasetRegistryError("parent_dataset_id must be a string or null.")
    if isinstance(parent_raw, str) and not parent_raw:
        raise DatasetRegistryError("parent_dataset_id must not be empty.")
    hypothesis = _string(payload["hypothesis"], "hypothesis")
    files = _string_mapping(
        payload["files"],
        "files",
        required={"X_train", "y_train", "X_test", "metadata"},
    )
    train_row_count = _non_negative_int(payload["train_row_count"], "train_row_count")
    test_row_count = _non_negative_int(payload["test_row_count"], "test_row_count")
    n_features = _non_negative_int(payload["n_features"], "n_features")
    features = _parse_features(payload["features"])
    if len(features) != n_features:
        raise DatasetRegistryError(
            "n_features does not match the length of features."
        )
    transformations = _parse_transformations(payload["transformations"])
    target_dependency = _parse_target_dependency(payload["target_dependency"])
    schema_hash = _sha256_string(payload["schema_hash"], "schema_hash")
    content_hashes = _string_mapping(
        payload["content_hashes"],
        "content_hashes",
        required={"X_train", "y_train", "X_test", "metadata"},
    )
    for key, value in content_hashes.items():
        _sha256_string(value, f"content_hashes.{key}")
    target = _parse_target(payload["target"])
    row_identity = _parse_row_identity(payload["row_identity"])
    return DatasetManifest(
        schema_version=schema_version,
        dataset_id=dataset_id,
        parent_dataset_id=parent_raw,
        hypothesis=hypothesis,
        files=files,
        train_row_count=train_row_count,
        test_row_count=test_row_count,
        n_features=n_features,
        features=features,
        transformations=transformations,
        target_dependency=target_dependency,
        schema_hash=schema_hash,
        content_hashes=content_hashes,
        target=target,
        row_identity=row_identity,
    )


def _parse_features(value: Any) -> tuple[FeatureSpec, ...]:
    if not isinstance(value, list) or not value:
        raise DatasetRegistryError("features must be a non-empty list.")
    features: list[FeatureSpec] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        label = f"features[{index}]"
        if not isinstance(item, dict):
            raise DatasetRegistryError(f"{label} must be a mapping.")
        _exact_keys(
            item,
            required={"name", "dtype", "role"},
            label=label,
        )
        name = _string(item["name"], f"{label}.name")
        if name in seen:
            raise DatasetRegistryError(f"Duplicate feature name: {name!r}.")
        seen.add(name)
        dtype = _string(item["dtype"], f"{label}.dtype")
        role = _string(item["role"], f"{label}.role")
        if role not in FEATURE_ROLE_VALUES:
            raise DatasetRegistryError(
                f"{label}.role must be one of {sorted(FEATURE_ROLE_VALUES)}."
            )
        features.append(FeatureSpec(name=name, dtype=dtype, role=role))  # type: ignore[arg-type]
    return tuple(features)


def _parse_transformations(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        raise DatasetRegistryError("transformations must be a list.")
    transformations: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise DatasetRegistryError(f"transformations[{index}] must be a mapping.")
        if "type" not in item or not isinstance(item["type"], str) or not item["type"]:
            raise DatasetRegistryError(
                f"transformations[{index}].type must be a non-empty string."
            )
        transformations.append(dict(item))
    return tuple(transformations)


def _parse_target_dependency(value: Any) -> TargetDependency:
    if not isinstance(value, str) or value not in TARGET_DEPENDENCY_VALUES:
        raise DatasetRegistryError(
            "target_dependency must be one of "
            f"{sorted(TARGET_DEPENDENCY_VALUES)}."
        )
    return value  # type: ignore[return-value]


def _parse_target(value: Any) -> TargetSpec:
    if not isinstance(value, dict):
        raise DatasetRegistryError("target must be a mapping.")
    _exact_keys(
        value,
        required={"name", "dtype", "class_counts", "hash"},
        label="target",
    )
    name = _string(value["name"], "target.name")
    dtype = _string(value["dtype"], "target.dtype")
    class_counts_raw = value["class_counts"]
    if not isinstance(class_counts_raw, dict) or not class_counts_raw:
        raise DatasetRegistryError("target.class_counts must be a non-empty mapping.")
    class_counts: dict[str, int] = {}
    for key, count in class_counts_raw.items():
        if not isinstance(key, str) or not key:
            raise DatasetRegistryError("target.class_counts keys must be strings.")
        class_counts[key] = _non_negative_int(count, f"target.class_counts[{key}]")
    digest = _sha256_string(value["hash"], "target.hash")
    return TargetSpec(
        name=name,
        dtype=dtype,
        class_counts=class_counts,
        hash=digest,
    )


def _parse_row_identity(value: Any) -> RowIdentitySpec:
    if not isinstance(value, dict):
        raise DatasetRegistryError("row_identity must be a mapping.")
    _exact_keys(
        value,
        required={
            "train_hash",
            "test_hash",
            "alignment_status",
            "alignment_method",
            "train_anchor_hash",
            "test_anchor_hash",
        },
        label="row_identity",
    )
    alignment_status = _string(value["alignment_status"], "row_identity.alignment_status")
    if alignment_status not in {"proven", "unverifiable"}:
        raise DatasetRegistryError(
            "row_identity.alignment_status must be 'proven' or 'unverifiable'."
        )
    return RowIdentitySpec(
        train_hash=_sha256_string(value["train_hash"], "row_identity.train_hash"),
        test_hash=_sha256_string(value["test_hash"], "row_identity.test_hash"),
        alignment_status=alignment_status,  # type: ignore[arg-type]
        alignment_method=_string(
            value["alignment_method"], "row_identity.alignment_method"
        ),
        train_anchor_hash=_sha256_string(
            value["train_anchor_hash"], "row_identity.train_anchor_hash"
        ),
        test_anchor_hash=_sha256_string(
            value["test_anchor_hash"], "row_identity.test_anchor_hash"
        ),
    )


def _exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str] | frozenset[str],
    optional: set[str] | frozenset[str] = frozenset(),
    label: str,
) -> None:
    keys = set(value)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise DatasetRegistryError(f"{label} is missing keys: {sorted(missing)}.")
    if unknown:
        raise DatasetRegistryError(f"{label} has unknown keys: {sorted(unknown)}.")


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DatasetRegistryError(f"{label} must be a non-empty string.")
    return value


def _sha256_string(value: Any, label: str) -> str:
    text = _string(value, label)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise DatasetRegistryError(f"{label} must be a lowercase SHA-256 hex digest.")
    return text


def _non_negative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise DatasetRegistryError(f"{label} must be a non-negative integer.")
    return value


def _string_mapping(
    value: Any,
    label: str,
    *,
    required: set[str] | frozenset[str],
) -> dict[str, str]:
    if not isinstance(value, dict):
        raise DatasetRegistryError(f"{label} must be a mapping.")
    _exact_keys(value, required=required, label=label)
    result: dict[str, str] = {}
    for key in required:
        result[key] = _string(value[key], f"{label}.{key}")
    return result


def feature_names(features: Sequence[FeatureSpec]) -> list[str]:
    return [feature.name for feature in features]
