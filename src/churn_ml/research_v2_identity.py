from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from src.churn_ml.research_data import canonical_sha256


PIPELINE_IMPLEMENTATION_SOURCES = (
    "src/churn_ml/experiment_v2_contract.py",
    "src/churn_ml/experiment_v2_pipeline.py",
    "src/churn_ml/experiment_v2_schema.py",
)
ADAPTER_IMPLEMENTATION_SOURCES = (
    "src/churn_ml/config.py",
    "src/churn_ml/experiment_v2_adapter.py",
    "src/churn_ml/experiment_v2_contract.py",
    "src/churn_ml/research_manual_lightgbm.py",
    "src/churn_ml/target_encoding.py",
)


def build_component_identities(
    *,
    pipeline_inputs: Mapping[str, Any],
    adapter_inputs: Mapping[str, Any],
    resolved_feature_schema: Mapping[str, Any],
    runtime_dependencies: Mapping[str, str],
    dataset_version: str,
    source_records: Mapping[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Build independent portable pipeline, adapter, and complete identities."""
    pipeline_sources = _implementation_source_payload(
        source_records,
        PIPELINE_IMPLEMENTATION_SOURCES,
    )
    adapter_sources = _implementation_source_payload(
        source_records,
        ADAPTER_IMPLEMENTATION_SOURCES,
    )
    pipeline_canonical = {
        "schema_version": 2,
        **deepcopy(dict(pipeline_inputs)),
        "resolved_feature_schema": deepcopy(dict(resolved_feature_schema)),
        "implementation_sources": pipeline_sources,
    }
    pipeline_hash = canonical_sha256(pipeline_canonical)
    adapter_canonical = {
        "schema_version": 2,
        **deepcopy(dict(adapter_inputs)),
        "runtime_dependencies": dict(sorted(runtime_dependencies.items())),
        "implementation_sources": adapter_sources,
    }
    adapter_hash = canonical_sha256(adapter_canonical)
    candidate_canonical = {
        "schema_version": 2,
        "dataset_version": dataset_version,
        "feature_pipeline": {
            "id": pipeline_inputs["id"],
            "sha256": pipeline_hash,
        },
        "candidate_adapter": {
            "id": adapter_inputs["id"],
            "sha256": adapter_hash,
        },
        "probability_semantics": "binary_positive_class_label_1",
        "evaluation_boundary": "adapter_receives_fold_local_data_only",
    }
    candidate_hash = canonical_sha256(candidate_canonical)
    identities = {
        "feature_pipeline": {
            "sha256": pipeline_hash,
            "canonical": pipeline_canonical,
        },
        "candidate_adapter": {
            "sha256": adapter_hash,
            "canonical": adapter_canonical,
        },
        "candidate": {
            "sha256": candidate_hash,
            "canonical": candidate_canonical,
        },
    }
    hashes = {
        "feature_pipeline": pipeline_hash,
        "candidate_adapter": adapter_hash,
        "candidate": candidate_hash,
    }
    return identities, hashes


def source_record_mapping(source_identity: Mapping[str, Any]) -> dict[str, str]:
    records = source_identity.get("files")
    if not isinstance(records, list):
        raise RuntimeError("Source identity files must be a list.")
    result: dict[str, str] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise RuntimeError("Source identity record must be a mapping.")
        path = record.get("path")
        sha256 = record.get("sha256")
        if not isinstance(path, str) or not isinstance(sha256, str):
            raise RuntimeError("Source identity record is malformed.")
        result[path] = sha256
    return result


def _implementation_source_payload(
    source_records: Mapping[str, str],
    required_paths: tuple[str, ...],
) -> dict[str, Any]:
    missing = sorted(set(required_paths) - set(source_records))
    if missing:
        raise RuntimeError(f"Implementation source records are missing: {missing}.")
    return {
        "schema_version": 1,
        "hashing_method": "sha256_file_bytes_repository_relative_paths",
        "files": [
            {"path": path, "sha256": source_records[path]}
            for path in sorted(required_paths)
        ],
    }
