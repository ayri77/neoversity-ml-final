from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from src.churn_ml.research_data import canonical_sha256


class OptunaSearchResumeIdentityError(RuntimeError):
    """Raised when the versioned source/runtime resume identity is invalid."""


RESUME_AUTHENTICATION_SCHEMA_VERSION = 1
SOURCE_CLOSURE_SCHEMA_VERSION = 1
RUNTIME_IDENTITY_SCHEMA_VERSION = 1
_REPARSE_POINT = 0x400

REQUIRED_OPTUNA_MODULES = (
    "src/churn_ml/optuna_search_artifacts.py",
    "src/churn_ml/optuna_search_cli.py",
    "src/churn_ml/optuna_search_config.py",
    "src/churn_ml/optuna_search_export.py",
    "src/churn_ml/optuna_search_lifecycle.py",
    "src/churn_ml/optuna_search_objective.py",
    "src/churn_ml/optuna_search_resume.py",
    "src/churn_ml/optuna_search_semantics.py",
    "src/churn_ml/optuna_search_space.py",
)

BEHAVIOR_CRITICAL_SOURCES = (
    "scripts/run_optuna_search.py",
    "src/churn_ml/experiment_v2.py",
    "src/churn_ml/experiment_v2_adapter.py",
    "src/churn_ml/experiment_v2_catboost_adapter.py",
    "src/churn_ml/experiment_v2_contract.py",
    "src/churn_ml/experiment_v2_model_registry.py",
    "src/churn_ml/experiment_v2_numeric_adapter.py",
    "src/churn_ml/experiment_v2_pipeline.py",
    "src/churn_ml/experiment_v2_schema.py",
    "src/churn_ml/experiment_v2_xgboost_adapter.py",
    "src/churn_ml/metrics.py",
    "src/churn_ml/research_config.py",
    "src/churn_ml/research_data.py",
    "src/churn_ml/research_protocol.py",
    "src/churn_ml/research_v2_config.py",
    "src/churn_ml/research_v2_data.py",
    "src/churn_ml/research_v2_resolved_config.py",
    "src/churn_ml/run_artifacts.py",
    "src/churn_ml/target_encoding.py",
)

COMMON_RUNTIME_DISTRIBUTIONS = {
    "numpy": "numpy",
    "optuna": "optuna",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "pyyaml": "PyYAML",
    "scikit-learn": "scikit-learn",
}
ADAPTER_RUNTIME_DISTRIBUTIONS = {
    "xgboost_numeric_v1": ("xgboost", "xgboost"),
    "catboost_numeric_v1": ("catboost", "catboost"),
}


def build_resume_authentication(
    *,
    project_root: Path,
    adapter_id: str,
    contract_paths: Iterable[str] = (),
) -> dict[str, Any]:
    """Build the exact source/runtime identity that gates every study resume."""
    root = project_root.resolve()
    source_closure = build_source_closure(
        project_root=root,
        contract_paths=contract_paths,
    )
    runtime_dependencies = build_runtime_identity(adapter_id)
    canonical = {
        "schema_version": RESUME_AUTHENTICATION_SCHEMA_VERSION,
        "source_closure": source_closure,
        "runtime_dependencies": runtime_dependencies,
    }
    return {**canonical, "identity_sha256": canonical_sha256(canonical)}


def build_source_closure(
    *,
    project_root: Path,
    contract_paths: Iterable[str] = (),
) -> dict[str, Any]:
    """Hash the stable, versioned behavior-critical repository source closure."""
    root = project_root.resolve()
    discovered = {
        path.relative_to(root).as_posix()
        for path in (root / "src" / "churn_ml").glob("optuna_search_*.py")
    }
    missing_modules = sorted(set(REQUIRED_OPTUNA_MODULES) - discovered)
    if missing_modules:
        raise OptunaSearchResumeIdentityError(
            f"Required Optuna source modules are missing: {missing_modules}."
        )
    relative_paths = {
        *REQUIRED_OPTUNA_MODULES,
        *discovered,
        *BEHAVIOR_CRITICAL_SOURCES,
        *contract_paths,
    }
    records = [
        _source_record(root, _canonical_relative_path(value))
        for value in sorted(relative_paths)
    ]
    canonical = {
        "schema_version": SOURCE_CLOSURE_SCHEMA_VERSION,
        "contract": "optuna_search_v1_behavior_critical_source_closure",
        "hashing_method": "sha256_same_file_bytes_repository_relative_posix_paths",
        "files": records,
    }
    return {**canonical, "identity_sha256": canonical_sha256(canonical)}


def build_runtime_identity(adapter_id: str) -> dict[str, Any]:
    """Capture exact Python and behavior-critical package versions."""
    adapter_distribution = ADAPTER_RUNTIME_DISTRIBUTIONS.get(adapter_id)
    if adapter_distribution is None:
        raise OptunaSearchResumeIdentityError(
            f"Unsupported adapter runtime identity: {adapter_id}."
        )
    distributions = {
        **COMMON_RUNTIME_DISTRIBUTIONS,
        adapter_distribution[0]: adapter_distribution[1],
    }
    versions = {
        key: importlib.metadata.version(distribution)
        for key, distribution in sorted(distributions.items())
    }
    if versions["optuna"] != "4.9.0":
        raise OptunaSearchResumeIdentityError(
            "Optuna Search v1 requires the locked Optuna 4.9.0 API."
        )
    canonical = {
        "schema_version": RUNTIME_IDENTITY_SCHEMA_VERSION,
        "python": platform.python_version(),
        "packages": versions,
        "adapter_id": adapter_id,
    }
    return {**canonical, "identity_sha256": canonical_sha256(canonical)}


def validate_resume_authentication_shape(
    value: Mapping[str, Any],
    *,
    adapter_id: str,
) -> None:
    if set(value) != {
        "schema_version",
        "source_closure",
        "runtime_dependencies",
        "identity_sha256",
    }:
        raise OptunaSearchResumeIdentityError("Resume-authentication schema differs.")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise OptunaSearchResumeIdentityError(
            "Resume-authentication schema version differs."
        )
    source = value["source_closure"]
    runtime = value["runtime_dependencies"]
    if not isinstance(source, Mapping) or not isinstance(runtime, Mapping):
        raise OptunaSearchResumeIdentityError(
            "Resume-authentication components must be mappings."
        )
    _validate_source_closure_shape(source)
    _validate_runtime_identity_shape(runtime, adapter_id=adapter_id)
    canonical = {
        "schema_version": value["schema_version"],
        "source_closure": dict(source),
        "runtime_dependencies": dict(runtime),
    }
    if not _is_sha256(value["identity_sha256"]) or value[
        "identity_sha256"
    ] != canonical_sha256(canonical):
        raise OptunaSearchResumeIdentityError(
            "Resume-authentication identity hash differs."
        )


def _validate_source_closure_shape(value: Mapping[str, Any]) -> None:
    if set(value) != {
        "schema_version",
        "contract",
        "hashing_method",
        "files",
        "identity_sha256",
    }:
        raise OptunaSearchResumeIdentityError("Source-closure schema differs.")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != SOURCE_CLOSURE_SCHEMA_VERSION
        or value["contract"] != "optuna_search_v1_behavior_critical_source_closure"
        or value["hashing_method"]
        != "sha256_same_file_bytes_repository_relative_posix_paths"
        or not isinstance(value["files"], list)
    ):
        raise OptunaSearchResumeIdentityError("Source-closure values differ.")
    paths: list[str] = []
    for record in value["files"]:
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise OptunaSearchResumeIdentityError(
                "Source-closure file record schema differs."
            )
        path = record["path"]
        if (
            type(path) is not str
            or _canonical_relative_path(path) != path
            or type(record["size_bytes"]) is not int
            or record["size_bytes"] < 0
            or not _is_sha256(record["sha256"])
        ):
            raise OptunaSearchResumeIdentityError(
                "Source-closure file record values differ."
            )
        paths.append(path)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise OptunaSearchResumeIdentityError(
            "Source-closure paths must be unique and sorted."
        )
    canonical = {
        key: value[key]
        for key in ("schema_version", "contract", "hashing_method", "files")
    }
    if not _is_sha256(value["identity_sha256"]) or value[
        "identity_sha256"
    ] != canonical_sha256(canonical):
        raise OptunaSearchResumeIdentityError("Source-closure identity differs.")


def _validate_runtime_identity_shape(
    value: Mapping[str, Any],
    *,
    adapter_id: str,
) -> None:
    if set(value) != {
        "schema_version",
        "python",
        "packages",
        "adapter_id",
        "identity_sha256",
    }:
        raise OptunaSearchResumeIdentityError("Runtime identity schema differs.")
    expected_package_keys = set(COMMON_RUNTIME_DISTRIBUTIONS) | {
        ADAPTER_RUNTIME_DISTRIBUTIONS[adapter_id][0]
    }
    packages = value["packages"]
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != RUNTIME_IDENTITY_SCHEMA_VERSION
        or type(value["python"]) is not str
        or not value["python"]
        or value["adapter_id"] != adapter_id
        or not isinstance(packages, Mapping)
        or set(packages) != expected_package_keys
        or any(type(item) is not str or not item for item in packages.values())
    ):
        raise OptunaSearchResumeIdentityError("Runtime identity values differ.")
    canonical = {
        key: value[key]
        for key in ("schema_version", "python", "packages", "adapter_id")
    }
    if not _is_sha256(value["identity_sha256"]) or value[
        "identity_sha256"
    ] != canonical_sha256(canonical):
        raise OptunaSearchResumeIdentityError("Runtime identity hash differs.")


def _source_record(root: Path, relative: str) -> dict[str, Any]:
    path = root.joinpath(*PurePosixPath(relative).parts)
    _reject_linked_path(path, root)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise OptunaSearchResumeIdentityError(
            f"Source closure path is unavailable: {relative}."
        ) from error
    if not stat.S_ISREG(metadata.st_mode) or _is_reparse_stat(metadata):
        raise OptunaSearchResumeIdentityError(
            f"Source closure path is not an exact regular file: {relative}."
        )
    data = path.read_bytes()
    if len(data) != metadata.st_size:
        raise OptunaSearchResumeIdentityError(
            f"Source closure file changed while hashing: {relative}."
        )
    return {
        "path": relative,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _reject_linked_path(path: Path, root: Path) -> None:
    current = path
    while current != root:
        try:
            metadata = current.lstat()
        except OSError:
            current = current.parent
            continue
        if stat.S_ISLNK(metadata.st_mode) or _is_reparse_stat(metadata):
            raise OptunaSearchResumeIdentityError(
                f"Linked/reparse source path is forbidden: {path}."
            )
        current = current.parent
    if current != root:
        raise OptunaSearchResumeIdentityError("Source closure path escapes repository.")


def _canonical_relative_path(value: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise OptunaSearchResumeIdentityError(
            "Source closure paths must be repository-relative POSIX paths."
        )
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise OptunaSearchResumeIdentityError(
            "Source closure paths must not contain traversal."
        )
    canonical = path.as_posix()
    if canonical != value:
        raise OptunaSearchResumeIdentityError("Source closure paths must be canonical.")
    return canonical


def _is_reparse_stat(metadata: os.stat_result) -> bool:
    return bool(getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT)


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
