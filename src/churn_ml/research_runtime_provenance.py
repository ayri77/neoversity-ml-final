from __future__ import annotations

import hashlib
import os
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

from src.churn_ml.research_protocol import canonical_sha256


RUNTIME_SOURCE_HASHING_METHOD = "sha256_loaded_module_raw_source_bytes_v1"
REQUIRED_RESEARCH_DISTRIBUTIONS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit_learn": "scikit-learn",
    "lightgbm": "lightgbm",
    "pyarrow": "pyarrow",
    "pyyaml": "PyYAML",
}


class ResearchRuntimeProvenanceError(RuntimeError):
    """Raised when the loaded runtime cannot prove its executable origins."""


def collect_research_environment_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for key, distribution in REQUIRED_RESEARCH_DISTRIBUTIONS.items():
        try:
            resolved = version(distribution)
        except PackageNotFoundError as error:
            raise ResearchRuntimeProvenanceError(
                f"Required distribution version is unavailable: {distribution}"
            ) from error
        if not resolved:
            raise ResearchRuntimeProvenanceError(
                f"Required distribution has an empty version: {distribution}"
            )
        versions[key] = resolved
    return versions


def build_invocation_provenance(
    *,
    entry_point: str,
    process_started_at_utc: datetime,
) -> dict[str, Any]:
    if (
        process_started_at_utc.tzinfo is None
        or process_started_at_utc.utcoffset()
        != timezone.utc.utcoffset(process_started_at_utc)
    ):
        raise ResearchRuntimeProvenanceError(
            "Process start time must be timezone-aware UTC."
        )
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "sys_argv": list(sys.argv),
        "entry_point": entry_point,
        "current_working_directory": str(Path.cwd().resolve()),
        "process_id": os.getpid(),
        "process_started_at_utc": process_started_at_utc.isoformat(),
    }


def build_loaded_module_provenance(
    project_root: Path,
    source_manifests: Sequence[Mapping[str, Any]],
    *,
    entrypoint_module_name: str | None = "__main__",
    loaded_modules: Mapping[str, ModuleType] | None = None,
) -> tuple[dict[str, Any], str]:
    """Bind expected implementation modules to their loaded source files."""
    root = project_root.resolve()
    modules = sys.modules if loaded_modules is None else loaded_modules
    source_records = _merge_source_records(source_manifests)
    expected_modules: dict[str, dict[str, str]] = {}
    for relative, expected_sha256 in source_records.items():
        module_name = _module_name_for_path(relative, entrypoint_module_name)
        if module_name is None:
            continue
        if module_name in expected_modules:
            raise ResearchRuntimeProvenanceError(
                f"Duplicate expected implementation module: {module_name}"
            )
        expected_modules[module_name] = {
            "path": relative,
            "sha256": expected_sha256,
        }

    canonical_records: list[dict[str, Any]] = []
    observations: list[dict[str, str]] = []
    resolved_paths: dict[Path, str] = {}
    for module_name in sorted(expected_modules):
        expected = expected_modules[module_name]
        module = modules.get(module_name)
        if module is None:
            raise ResearchRuntimeProvenanceError(
                f"Expected implementation module is not loaded: {module_name}"
            )
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str) or not module_file:
            raise ResearchRuntimeProvenanceError(
                f"Loaded implementation module has no source file: {module_name}"
            )
        resolved = Path(module_file).resolve()
        try:
            actual_relative = resolved.relative_to(root).as_posix()
        except ValueError as error:
            raise ResearchRuntimeProvenanceError(
                f"Loaded implementation module is outside the repository: "
                f"{module_name} -> {resolved}"
            ) from error
        if actual_relative != expected["path"]:
            raise ResearchRuntimeProvenanceError(
                f"Loaded implementation module path differs for {module_name}: "
                f"expected={expected['path']}, actual={actual_relative}"
            )
        if not resolved.is_file():
            raise ResearchRuntimeProvenanceError(
                f"Loaded implementation source does not exist: {resolved}"
            )
        actual_sha256 = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if actual_sha256 != expected["sha256"]:
            raise ResearchRuntimeProvenanceError(
                f"Loaded implementation bytes differ for {module_name}: "
                f"expected={expected['sha256']}, actual={actual_sha256}"
            )
        if resolved in resolved_paths:
            raise ResearchRuntimeProvenanceError(
                f"Duplicate loaded implementation source: {resolved}"
            )
        resolved_paths[resolved] = module_name
        canonical_records.append(
            {
                "module_name": module_name,
                "expected_repository_relative_path": expected["path"],
                "actual_repository_relative_path": actual_relative,
                "loaded_source_sha256": actual_sha256,
                "expected_source_sha256": expected["sha256"],
                "match": True,
            }
        )
        observations.append(
            {
                "module_name": module_name,
                "module_file": module_file,
            }
        )

    expected_paths = {
        (root / Path(*record["path"].split("/"))).resolve(): module_name
        for module_name, record in expected_modules.items()
    }
    expected_aliases: list[dict[str, Any]] = []
    unexpected: list[str] = []
    for module_name, module in modules.items():
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str) or not module_file:
            continue
        resolved = Path(module_file).resolve()
        expected_name = expected_paths.get(resolved)
        if expected_name is not None and module_name != expected_name:
            if (
                entrypoint_module_name == "__main__"
                and expected_name == "__main__"
                and module_name == "__mp_main__"
                and module is modules.get("__main__")
            ):
                expected_aliases.append(
                    {
                        "module_name": module_name,
                        "canonical_module_name": expected_name,
                        "actual_repository_relative_path": (
                            resolved.relative_to(root).as_posix()
                        ),
                        "loaded_source_sha256": hashlib.sha256(
                            resolved.read_bytes()
                        ).hexdigest(),
                        "match": True,
                    }
                )
                observations.append(
                    {"module_name": module_name, "module_file": module_file}
                )
            else:
                unexpected.append(f"{module_name} -> {resolved}")
    if unexpected:
        raise ResearchRuntimeProvenanceError(
            "Unexpected implementation module aliases are loaded: "
            + ", ".join(sorted(unexpected))
        )

    canonical = {
        "schema_version": 1,
        "hashing_method": RUNTIME_SOURCE_HASHING_METHOD,
        "modules": canonical_records,
    }
    payload = {
        "canonical": canonical,
        "observations": observations,
    }
    return payload, canonical_sha256(canonical)


def _merge_source_records(
    source_manifests: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    records: dict[str, str] = {}
    for manifest in source_manifests:
        files = manifest.get("files")
        if not isinstance(files, list):
            raise ResearchRuntimeProvenanceError(
                "Source provenance manifest files must be a list."
            )
        for record in files:
            if not isinstance(record, Mapping):
                raise ResearchRuntimeProvenanceError(
                    "Source provenance file record must be a mapping."
                )
            relative = record.get("path")
            sha256 = record.get("sha256")
            if not isinstance(relative, str) or not isinstance(sha256, str):
                raise ResearchRuntimeProvenanceError(
                    "Source provenance records require string path and SHA-256."
                )
            previous = records.get(relative)
            if previous is not None and previous != sha256:
                raise ResearchRuntimeProvenanceError(
                    f"Conflicting source provenance for {relative}."
                )
            records[relative] = sha256
    return records


def _module_name_for_path(
    relative_path: str,
    entrypoint_module_name: str | None,
) -> str | None:
    if relative_path == "scripts/run_research_evaluation.py":
        return entrypoint_module_name
    if relative_path.startswith("src/") and relative_path.endswith(".py"):
        return relative_path[:-3].replace("/", ".")
    return None
