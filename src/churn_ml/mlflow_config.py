"""Strict local configuration for the optional MLflow artifact index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml


class MLflowConfigError(ValueError):
    """Raised when the local MLflow index configuration is unsafe or invalid."""


@dataclass(frozen=True)
class TrackingConfig:
    backend_store_uri: str
    artifact_root: str


@dataclass(frozen=True)
class ExperimentNames:
    research_v2: str
    autogluon: str


@dataclass(frozen=True)
class SourceRoots:
    research_v2_root: str
    autogluon_root: str


@dataclass(frozen=True)
class SyncOptions:
    log_small_artifacts: bool
    max_artifact_size_bytes: int


@dataclass(frozen=True)
class ResolvedMLflowPaths:
    repository_root: Path
    backend_store: Path
    artifact_root: Path
    research_v2_root: Path
    autogluon_root: Path
    receipts_root: Path

    @property
    def tracking_uri(self) -> str:
        return f"sqlite:///{self.backend_store.as_posix()}"

    @property
    def artifact_uri(self) -> str:
        return self.artifact_root.as_uri()


@dataclass(frozen=True)
class MLflowIndexConfig:
    schema_version: int
    tracking: TrackingConfig
    experiments: ExperimentNames
    sources: SourceRoots
    sync: SyncOptions
    paths: ResolvedMLflowPaths


_SCHEMA: dict[str, Any] = {
    "schema_version": int,
    "tracking": {
        "backend_store_uri": str,
        "artifact_root": str,
    },
    "experiments": {
        "research_v2": str,
        "autogluon": str,
    },
    "sources": {
        "research_v2_root": str,
        "autogluon_root": str,
    },
    "sync": {
        "log_small_artifacts": bool,
        "max_artifact_size_bytes": int,
    },
}


def load_mlflow_config(
    path: Path,
    *,
    repository_root: Path,
) -> MLflowIndexConfig:
    """Load a strict config without allocating any MLflow or filesystem state."""
    root = repository_root.resolve(strict=True)
    source = path.resolve(strict=True)
    _require_contained(source, root, "config path")
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise MLflowConfigError(f"Cannot load MLflow config {path}: {error}") from error
    _validate_node(payload, _SCHEMA, "config")
    assert type(payload) is dict
    if payload["schema_version"] != 1:
        raise MLflowConfigError("config.schema_version must equal 1")

    tracking = payload["tracking"]
    experiments = payload["experiments"]
    sources = payload["sources"]
    sync = payload["sync"]
    for name, value in (
        ("config.experiments.research_v2", experiments["research_v2"]),
        ("config.experiments.autogluon", experiments["autogluon"]),
    ):
        if not value.strip():
            raise MLflowConfigError(f"{name} must not be blank")
    if sync["max_artifact_size_bytes"] <= 0:
        raise MLflowConfigError("config.sync.max_artifact_size_bytes must be positive")

    backend_relative = _sqlite_relative_path(tracking["backend_store_uri"])
    backend_store = _resolve_repository_path(
        root, backend_relative, "config.tracking.backend_store_uri"
    )
    if backend_store.suffix.lower() not in {".db", ".sqlite", ".sqlite3"}:
        raise MLflowConfigError(
            "config.tracking.backend_store_uri must identify a SQLite database file"
        )
    artifact_root = _resolve_repository_path(
        root, tracking["artifact_root"], "config.tracking.artifact_root"
    )
    research_root = _resolve_repository_path(
        root, sources["research_v2_root"], "config.sources.research_v2_root"
    )
    autogluon_root = _resolve_repository_path(
        root, sources["autogluon_root"], "config.sources.autogluon_root"
    )
    for label, resolved in (
        ("config.tracking.backend_store_uri", backend_store),
        ("config.tracking.artifact_root", artifact_root),
    ):
        try:
            resolved.relative_to(root / "artifacts")
        except ValueError as error:
            raise MLflowConfigError(
                f"{label} must remain inside the ignored artifacts area"
            ) from error
    if backend_store == artifact_root or backend_store in artifact_root.parents:
        raise MLflowConfigError(
            "SQLite backend and MLflow artifact root must be separate locations"
        )
    receipts_root = backend_store.parent / "sync_receipts"
    _require_contained(receipts_root.resolve(strict=False), root, "sync receipts")

    return MLflowIndexConfig(
        schema_version=1,
        tracking=TrackingConfig(**tracking),
        experiments=ExperimentNames(**experiments),
        sources=SourceRoots(**sources),
        sync=SyncOptions(**sync),
        paths=ResolvedMLflowPaths(
            repository_root=root,
            backend_store=backend_store,
            artifact_root=artifact_root,
            research_v2_root=research_root,
            autogluon_root=autogluon_root,
            receipts_root=receipts_root,
        ),
    )


def validation_summary(config: MLflowIndexConfig) -> dict[str, Any]:
    """Return a portable validation summary without absolute local paths."""
    return {
        "valid": True,
        "schema_version": config.schema_version,
        "tracking": {
            "backend_store_uri": config.tracking.backend_store_uri,
            "artifact_root": config.tracking.artifact_root,
        },
        "experiments": {
            "research_v2": config.experiments.research_v2,
            "autogluon": config.experiments.autogluon,
        },
        "sources": {
            "research_v2_root": config.sources.research_v2_root,
            "autogluon_root": config.sources.autogluon_root,
        },
        "sync": {
            "log_small_artifacts": config.sync.log_small_artifacts,
            "max_artifact_size_bytes": config.sync.max_artifact_size_bytes,
        },
        "allocates_storage": False,
    }


def _validate_node(value: Any, schema: Any, path: str) -> None:
    if isinstance(schema, dict):
        if type(value) is not dict:
            raise MLflowConfigError(f"{path} must be a mapping")
        expected = set(schema)
        actual = set(value)
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        if missing:
            raise MLflowConfigError(f"{path} is missing keys: {', '.join(missing)}")
        if unknown:
            raise MLflowConfigError(f"{path} has unknown keys: {', '.join(unknown)}")
        for key, child_schema in schema.items():
            _validate_node(value[key], child_schema, f"{path}.{key}")
        return
    if type(value) is not schema:
        raise MLflowConfigError(f"{path} must have exact type {schema.__name__}")


def _sqlite_relative_path(uri: str) -> str:
    prefix = "sqlite:///"
    if not uri.startswith(prefix):
        raise MLflowConfigError(
            "config.tracking.backend_store_uri must use sqlite:/// with a "
            "repository-relative path"
        )
    relative = uri[len(prefix) :]
    if not relative or "?" in relative or "#" in relative:
        raise MLflowConfigError(
            "config.tracking.backend_store_uri must contain one plain relative path"
        )
    return relative


def _resolve_repository_path(root: Path, value: str, label: str) -> Path:
    if not value.strip():
        raise MLflowConfigError(f"{label} must not be blank")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        Path(value).is_absolute()
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or value.startswith(("/", "\\"))
        or "\\" in value
        or ".." in posix.parts
        or ".." in windows.parts
    ):
        raise MLflowConfigError(
            f"{label} must be repository-relative without drive, root, UNC, "
            "backslash, or traversal syntax"
        )
    resolved = (root / Path(*posix.parts)).resolve(strict=False)
    _require_contained(resolved, root, label)
    return resolved


def _require_contained(path: Path, root: Path, label: str) -> None:
    if path != root and root not in path.parents:
        raise MLflowConfigError(f"{label} escapes the repository")
