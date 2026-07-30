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
from src.churn_ml.dataset_registry.errors import (
    DatasetRegistryError,
    OverwriteRefusedError,
    UnverifiableAlignmentError,
)
from src.churn_ml.dataset_registry.schema import (
    SCHEMA_VERSION,
    DatasetManifest,
    FeatureSpec,
    TargetSpec,
)

__all__ = [
    "SCHEMA_VERSION",
    "DatasetManifest",
    "DatasetPackage",
    "DatasetRegistryError",
    "FeatureSpec",
    "OverwriteRefusedError",
    "TargetSpec",
    "UnverifiableAlignmentError",
    "ValidationResult",
    "list_dataset_packages",
    "load_dataset_manifest",
    "resolve_dataset_package",
    "validate_dataset_package",
]
