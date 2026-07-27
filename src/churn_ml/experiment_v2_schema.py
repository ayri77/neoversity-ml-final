from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


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
