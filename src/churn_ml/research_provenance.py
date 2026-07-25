from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.churn_ml.research_protocol import canonical_sha256


SOURCE_HASHING_METHOD = "sha256_raw_working_tree_bytes_v1"

CANDIDATE_BEHAVIOR_SOURCE_PATHS = (
    "src/churn_ml/manual_lightgbm.py",
    "src/churn_ml/research_data.py",
    "src/churn_ml/research_evaluation.py",
    "src/churn_ml/research_manual_lightgbm.py",
    "src/churn_ml/research_protocol.py",
    "src/churn_ml/target_encoding.py",
)

RUN_IMPLEMENTATION_SOURCE_PATHS = (
    "scripts/run_research_evaluation.py",
    "src/churn_ml/config.py",
    "src/churn_ml/manual_lightgbm.py",
    "src/churn_ml/research_artifacts.py",
    "src/churn_ml/research_artifact_validation.py",
    "src/churn_ml/research_config.py",
    "src/churn_ml/research_data.py",
    "src/churn_ml/research_evaluation.py",
    "src/churn_ml/research_manual_lightgbm.py",
    "src/churn_ml/research_metadata_validation.py",
    "src/churn_ml/research_protocol.py",
    "src/churn_ml/research_cli.py",
    "src/churn_ml/research_provenance.py",
    "src/churn_ml/run_artifacts.py",
    "src/churn_ml/target_encoding.py",
    "src/churn_ml/research_runtime_provenance.py",
)


class ResearchProvenanceError(ValueError):
    """Raised when source provenance cannot be represented safely."""


def build_source_provenance(
    project_root: Path,
    relative_paths: Iterable[str],
) -> tuple[dict[str, Any], str]:
    """Hash ordered source paths and their exact working-tree bytes."""
    root = project_root.resolve()
    normalized = sorted(set(relative_paths))
    files: list[dict[str, Any]] = []
    for relative in normalized:
        portable = relative.replace("\\", "/")
        if portable != relative or Path(portable).is_absolute():
            raise ResearchProvenanceError(
                f"Provenance paths must be portable relative paths: {relative!r}"
            )
        path = (root / Path(*portable.split("/"))).resolve()
        if root not in path.parents:
            raise ResearchProvenanceError(
                f"Provenance path escapes the project root: {relative!r}"
            )
        if not path.is_file():
            raise ResearchProvenanceError(
                f"Provenance source file does not exist: {relative!r}"
            )
        files.append(
            {
                "path": portable,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    canonical = {
        "schema_version": 1,
        "hashing_method": SOURCE_HASHING_METHOD,
        "files": files,
    }
    return canonical, canonical_sha256(canonical)


def candidate_source_provenance(
    project_root: Path,
) -> tuple[dict[str, Any], str]:
    return build_source_provenance(project_root, CANDIDATE_BEHAVIOR_SOURCE_PATHS)


def run_implementation_provenance(
    project_root: Path,
    *,
    config_paths: Mapping[str, Path],
) -> tuple[dict[str, Any], str]:
    root = project_root.resolve()
    static_manifest, _ = build_source_provenance(root, RUN_IMPLEMENTATION_SOURCE_PATHS)
    records = list(static_manifest["files"])
    for role, path in sorted(config_paths.items()):
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError:
            relative = f"external_configuration/{role}.yaml"
        if not resolved.is_file():
            raise ResearchProvenanceError(
                f"Run implementation config does not exist: {resolved}"
            )
        records.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
            }
        )
    paths = [record["path"] for record in records]
    if len(paths) != len(set(paths)):
        raise ResearchProvenanceError("Run provenance contains duplicate paths.")
    canonical = {
        "schema_version": 1,
        "hashing_method": SOURCE_HASHING_METHOD,
        "files": sorted(records, key=lambda record: record["path"]),
    }
    return canonical, canonical_sha256(canonical)
