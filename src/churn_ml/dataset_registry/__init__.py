"""Dataset Package & Registry v1 public API."""

from __future__ import annotations

from src.churn_ml.dataset_registry.api import (
    DatasetPackage,
    ValidationResult,
    list_dataset_packages,
    load_dataset_manifest,
    resolve_dataset_package,
    validate_dataset_package,
)
from src.churn_ml.dataset_registry.build_spec import (
    DatasetBuildSpec,
    native_build_spec,
    parse_build_spec,
)
from src.churn_ml.dataset_registry.errors import (
    DatasetRegistryError,
    OverwriteRefusedError,
    UnverifiableAlignmentError,
)
from src.churn_ml.dataset_registry.materialize import dataset_manifest_sha256
from src.churn_ml.dataset_registry.schema import (
    SCHEMA_VERSION,
    DatasetManifest,
    FeatureSpec,
    TargetSpec,
)

__all__ = [
    "SCHEMA_VERSION",
    "DatasetBuildSpec",
    "DatasetManifest",
    "DatasetPackage",
    "DatasetRegistryError",
    "FeatureSpec",
    "OverwriteRefusedError",
    "TargetSpec",
    "UnverifiableAlignmentError",
    "ValidationResult",
    "dataset_manifest_sha256",
    "list_dataset_packages",
    "load_dataset_manifest",
    "native_build_spec",
    "parse_build_spec",
    "resolve_dataset_package",
    "validate_dataset_package",
]
