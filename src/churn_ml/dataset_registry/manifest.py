"""Build and compare dataset manifests deterministically."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.churn_ml.dataset_registry.alignment import (
    AlignmentProof,
    prove_alignment,
    row_identity_from_proof,
)
from src.churn_ml.dataset_registry.constants import SCHEMA_VERSION
from src.churn_ml.dataset_registry.errors import DatasetRegistryError
from src.churn_ml.dataset_registry.hashing import (
    class_counts,
    file_content_sha256,
    schema_hash,
    target_hash,
)
from src.churn_ml.dataset_registry.package import (
    PackageArtifacts,
    validate_basic_shapes,
)
from src.churn_ml.dataset_registry.schema import (
    DatasetManifest,
    FeatureRole,
    FeatureSpec,
    TargetDependency,
    TargetSpec,
    parse_manifest,
)


def build_feature_specs(
    artifacts: PackageArtifacts,
    *,
    engineered_features: Sequence[str] | None = None,
) -> tuple[FeatureSpec, ...]:
    engineered = set(engineered_features or ())
    specs: list[FeatureSpec] = []
    for name, dtype in artifacts.X_train.dtypes.items():
        role: FeatureRole = "engineered" if str(name) in engineered else "feature"
        specs.append(FeatureSpec(name=str(name), dtype=str(dtype), role=role))
    return tuple(specs)


def build_manifest(
    artifacts: PackageArtifacts,
    *,
    hypothesis: str,
    parent_dataset_id: str | None,
    target_dependency: TargetDependency,
    transformations: Sequence[Mapping[str, Any]],
    alignment: AlignmentProof,
    engineered_features: Sequence[str] | None = None,
) -> DatasetManifest:
    validate_basic_shapes(artifacts)
    if alignment.status != "proven":
        raise DatasetRegistryError(
            "Cannot build a registrable manifest without proven alignment.",
            status="unverifiable_alignment",
        )
    features = build_feature_specs(
        artifacts,
        engineered_features=engineered_features,
    )
    content_hashes = {
        "X_train": file_content_sha256(artifacts.paths["X_train"]),
        "y_train": file_content_sha256(artifacts.paths["y_train"]),
        "X_test": file_content_sha256(artifacts.paths["X_test"]),
        "metadata": file_content_sha256(artifacts.paths["metadata"]),
    }
    target = TargetSpec(
        name=str(artifacts.y_train.name),
        dtype=str(artifacts.y_train.dtype),
        class_counts=class_counts(artifacts.y_train),
        hash=target_hash(artifacts.y_train),
    )
    return DatasetManifest(
        schema_version=SCHEMA_VERSION,
        dataset_id=artifacts.dataset_id,
        parent_dataset_id=parent_dataset_id,
        hypothesis=hypothesis,
        files={
            "X_train": "X_train.parquet",
            "y_train": "y_train.parquet",
            "X_test": "X_test.parquet",
            "metadata": "metadata.json",
        },
        train_row_count=len(artifacts.X_train),
        test_row_count=len(artifacts.X_test),
        n_features=len(features),
        features=features,
        transformations=tuple(dict(item) for item in transformations),
        target_dependency=target_dependency,
        schema_hash=schema_hash(features),
        content_hashes=content_hashes,
        target=target,
        row_identity=row_identity_from_proof(alignment),
    )


def recompute_manifest_from_disk(
    artifacts: PackageArtifacts,
    *,
    hypothesis: str,
    parent_dataset_id: str | None,
    target_dependency: TargetDependency,
    transformations: Sequence[Mapping[str, Any]],
    engineered_features: Sequence[str] | None = None,
    anchor_feature_names: tuple[str, ...] | None = None,
    parent_artifacts: PackageArtifacts | None = None,
    root: Path | None = None,
) -> DatasetManifest:
    alignment = prove_alignment(
        artifacts,
        anchor_feature_names=anchor_feature_names,
        parent_artifacts=parent_artifacts,
        root=root,
    )
    return build_manifest(
        artifacts,
        hypothesis=hypothesis,
        parent_dataset_id=parent_dataset_id,
        target_dependency=target_dependency,
        transformations=transformations,
        alignment=alignment,
        engineered_features=engineered_features,
    )


def read_manifest(path: Path) -> DatasetManifest:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise DatasetRegistryError("dataset_manifest.json must be a JSON object.")
    return parse_manifest(payload)


def write_manifest(path: Path, manifest: DatasetManifest) -> None:
    payload = manifest.to_dict()
    # Deterministic JSON: sorted keys, stable separators, UTF-8, trailing newline.
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")


def manifests_equal(left: DatasetManifest, right: DatasetManifest) -> bool:
    return left.to_dict() == right.to_dict()


def protected_property_mismatches(
    expected: DatasetManifest,
    actual: DatasetManifest,
) -> list[str]:
    """Return human-readable mismatches for immutable package properties."""
    mismatches: list[str] = []
    if expected.dataset_id != actual.dataset_id:
        mismatches.append("dataset_id")
    if expected.parent_dataset_id != actual.parent_dataset_id:
        mismatches.append("parent_dataset_id")
    if expected.schema_hash != actual.schema_hash:
        mismatches.append("schema_hash")
    if expected.content_hashes != actual.content_hashes:
        mismatches.append("content_hashes")
    if expected.target.hash != actual.target.hash:
        mismatches.append("target.hash")
    if expected.target.name != actual.target.name:
        mismatches.append("target.name")
    if expected.target.dtype != actual.target.dtype:
        mismatches.append("target.dtype")
    if expected.target.class_counts != actual.target.class_counts:
        mismatches.append("target.class_counts")
    if [f.to_dict() for f in expected.features] != [
        f.to_dict() for f in actual.features
    ]:
        mismatches.append("features")
    if expected.train_row_count != actual.train_row_count:
        mismatches.append("train_row_count")
    if expected.test_row_count != actual.test_row_count:
        mismatches.append("test_row_count")
    if expected.n_features != actual.n_features:
        mismatches.append("n_features")
    if expected.row_identity.to_dict() != actual.row_identity.to_dict():
        mismatches.append("row_identity")
    if expected.files != actual.files:
        mismatches.append("files")
    return mismatches
