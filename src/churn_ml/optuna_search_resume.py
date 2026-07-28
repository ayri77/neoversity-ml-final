from __future__ import annotations

import ast
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
SOURCE_CLOSURE_SCHEMA_VERSION = 2
RUNTIME_IDENTITY_SCHEMA_VERSION = 1
_REPARSE_POINT = 0x400
_PACKAGE_ROOT = "src/churn_ml"
_PACKAGE_MODULE_ROOT = "src.churn_ml"

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

CLOSURE_SEED_SOURCES = (
    "scripts/run_optuna_search.py",
    *REQUIRED_OPTUNA_MODULES,
    "src/churn_ml/config.py",
    "src/churn_ml/experiment_v2.py",
    "src/churn_ml/experiment_v2_adapter.py",
    "src/churn_ml/experiment_v2_catboost_adapter.py",
    "src/churn_ml/experiment_v2_contract.py",
    "src/churn_ml/experiment_v2_model_registry.py",
    "src/churn_ml/experiment_v2_numeric_adapter.py",
    "src/churn_ml/experiment_v2_pipeline.py",
    "src/churn_ml/experiment_v2_schema.py",
    "src/churn_ml/experiment_v2_xgboost_adapter.py",
    "src/churn_ml/manual_lightgbm.py",
    "src/churn_ml/metrics.py",
    "src/churn_ml/research_config.py",
    "src/churn_ml/research_data.py",
    "src/churn_ml/research_manual_lightgbm.py",
    "src/churn_ml/research_protocol.py",
    "src/churn_ml/research_v2_config.py",
    "src/churn_ml/research_v2_data.py",
    "src/churn_ml/research_v2_identity.py",
    "src/churn_ml/research_v2_provenance.py",
    "src/churn_ml/research_v2_resolved_config.py",
    "src/churn_ml/run_artifacts.py",
    "src/churn_ml/target_encoding.py",
)

EXCLUDED_CLOSURE_PREFIXES = (
    "tests/",
    "docs/",
    "notebooks/",
    "sandbox/",
    "scripts/test_",
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
    """Hash the complete behavior-critical import-graph source closure."""
    root = project_root.resolve()
    discovered_optuna = {
        path.relative_to(root).as_posix()
        for path in (root / "src" / "churn_ml").glob("optuna_search_*.py")
    }
    missing_modules = sorted(set(REQUIRED_OPTUNA_MODULES) - discovered_optuna)
    if missing_modules:
        raise OptunaSearchResumeIdentityError(
            f"Required Optuna source modules are missing: {missing_modules}."
        )
    seed_paths = {
        *CLOSURE_SEED_SOURCES,
        *discovered_optuna,
        *(_canonical_relative_path(path) for path in contract_paths),
    }
    relative_paths = discover_behavior_critical_source_paths(
        project_root=root,
        seed_paths=seed_paths,
    )
    records = [_source_record(root, path) for path in relative_paths]
    canonical = {
        "schema_version": SOURCE_CLOSURE_SCHEMA_VERSION,
        "contract": "optuna_search_v1_behavior_critical_source_closure",
        "hashing_method": "sha256_same_file_bytes_repository_relative_posix_paths",
        "discovery": {
            "method": "python_ast_import_graph",
            "package_root": _PACKAGE_ROOT,
            "seed_paths": sorted(seed_paths),
        },
        "files": records,
    }
    return {**canonical, "identity_sha256": canonical_sha256(canonical)}


def discover_behavior_critical_source_paths(
    *,
    project_root: Path,
    seed_paths: Iterable[str],
) -> list[str]:
    """Discover sorted internal runtime sources from an explicit seed set."""
    root = project_root.resolve()
    package_root = (root / _PACKAGE_ROOT).resolve()
    if not package_root.is_dir():
        raise OptunaSearchResumeIdentityError(
            f"Allowed package root is missing: {_PACKAGE_ROOT}."
        )
    pending = sorted({_canonical_relative_path(path) for path in seed_paths})
    discovered: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in discovered:
            continue
        if _is_excluded_closure_path(relative):
            raise OptunaSearchResumeIdentityError(
                f"Excluded non-runtime path entered source closure: {relative}."
            )
        path = root.joinpath(*PurePosixPath(relative).parts)
        _reject_linked_path(path, root)
        if not path.is_file():
            raise OptunaSearchResumeIdentityError(
                f"Source closure seed/dependency is missing: {relative}."
            )
        discovered.add(relative)
        if not relative.startswith(f"{_PACKAGE_ROOT}/") and relative != (
            "scripts/run_optuna_search.py"
        ):
            # Contract YAML/JSON seeds contribute identity bytes but are not
            # traversed as Python import roots.
            if relative.endswith((".yaml", ".yml", ".json")):
                continue
        if not relative.endswith(".py"):
            continue
        for dependency in sorted(_internal_python_dependencies(path, root=root)):
            if dependency not in discovered:
                pending.append(dependency)
    return sorted(discovered)


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
        "discovery",
        "files",
        "identity_sha256",
    }:
        raise OptunaSearchResumeIdentityError("Source-closure schema differs.")
    discovery = value["discovery"]
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != SOURCE_CLOSURE_SCHEMA_VERSION
        or value["contract"] != "optuna_search_v1_behavior_critical_source_closure"
        or value["hashing_method"]
        != "sha256_same_file_bytes_repository_relative_posix_paths"
        or not isinstance(value["files"], list)
        or not isinstance(discovery, Mapping)
        or set(discovery)
        != {
            "method",
            "package_root",
            "seed_paths",
        }
        or discovery["method"] != "python_ast_import_graph"
        or discovery["package_root"] != _PACKAGE_ROOT
        or not isinstance(discovery["seed_paths"], list)
        or discovery["seed_paths"] != sorted(discovery["seed_paths"])
        or len(discovery["seed_paths"]) != len(set(discovery["seed_paths"]))
        or any(type(item) is not str for item in discovery["seed_paths"])
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
        for key in (
            "schema_version",
            "contract",
            "hashing_method",
            "discovery",
            "files",
        )
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


def _internal_python_dependencies(path: Path, *, root: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as error:
        raise OptunaSearchResumeIdentityError(
            f"Unable to parse source closure module: {path}."
        ) from error
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module_name = _resolve_from_import(path, root=root, node=node)
            if module_name is not None:
                modules.add(module_name)
                for alias in node.names:
                    if alias.name != "*":
                        modules.add(f"{module_name}.{alias.name}")
    dependencies: set[str] = set()
    for module_name in modules:
        resolved = _module_to_relative_path(module_name, root=root)
        if resolved is not None:
            dependencies.add(resolved)
    return dependencies


def _resolve_from_import(
    path: Path,
    *,
    root: Path,
    node: ast.ImportFrom,
) -> str | None:
    if node.level == 0:
        return node.module
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    if not relative.startswith(f"{_PACKAGE_ROOT}/"):
        return node.module
    package_parts = list(PurePosixPath(relative).parts[:-1])
    if node.level > len(package_parts):
        return None
    base = package_parts[: len(package_parts) - node.level + 1]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base).replace("/", ".")


def _module_to_relative_path(module_name: str, *, root: Path) -> str | None:
    if module_name != _PACKAGE_MODULE_ROOT and not module_name.startswith(
        f"{_PACKAGE_MODULE_ROOT}."
    ):
        return None
    parts = module_name.split(".")
    rest = parts[2:]
    if not rest:
        init_path = root / _PACKAGE_ROOT / "__init__.py"
        if init_path.is_file():
            return f"{_PACKAGE_ROOT}/__init__.py"
        raise OptunaSearchResumeIdentityError(
            f"Internal package root module is missing: {module_name}."
        )
    # Resolve progressively shorter prefixes so attribute imports such as
    # ``from src.churn_ml.research_data import canonical_sha256`` still bind the
    # owning module, but never silently fall back to the package root.
    for length in range(len(rest), 0, -1):
        candidate_parts = rest[:length]
        module_path = root.joinpath(_PACKAGE_ROOT, *candidate_parts[:-1])
        file_path = module_path / f"{candidate_parts[-1]}.py"
        package_init = module_path / candidate_parts[-1] / "__init__.py"
        if file_path.is_file():
            return file_path.relative_to(root).as_posix()
        if package_init.is_file():
            return package_init.relative_to(root).as_posix()
    raise OptunaSearchResumeIdentityError(
        f"Internal source dependency cannot be resolved: {module_name}."
    )


def _is_excluded_closure_path(relative: str) -> bool:
    return any(relative.startswith(prefix) for prefix in EXCLUDED_CLOSURE_PREFIXES)


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
