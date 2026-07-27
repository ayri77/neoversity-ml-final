"""Strict versioned configuration for standalone AutoGluon benchmarks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.churn_ml.autogluon_profiles import get_profile, profile_summary


class ConfigError(ValueError):
    """Raised when an AutoGluon runner configuration is invalid."""


@dataclass(frozen=True)
class DatasetConfig:
    version: str
    directory: str
    train_features_file: str
    train_target_file: str
    label: str


@dataclass(frozen=True)
class PredictorConfig:
    problem_type: str
    eval_metric: str
    positive_class: int
    verbosity: int


@dataclass(frozen=True)
class ResourcesConfig:
    time_limit_seconds: int
    num_cpus: int | str
    num_gpus: int
    fit_strategy: str
    fold_fitting_strategy: str


@dataclass(frozen=True)
class FitConfig:
    presets: str
    calibrate_decision_threshold: bool


@dataclass(frozen=True)
class ArtifactsConfig:
    root: str


@dataclass(frozen=True)
class ResolvedPaths:
    repository_root: Path
    dataset_directory: Path
    train_features: Path
    train_target: Path
    artifacts_root: Path


@dataclass(frozen=True)
class AutoGluonConfig:
    schema_version: int
    profile_id: str
    dataset: DatasetConfig
    predictor: PredictorConfig
    resources: ResourcesConfig
    fit: FitConfig
    artifacts: ArtifactsConfig
    paths: ResolvedPaths
    portable: dict[str, Any]
    identity_sha256: str


_SCHEMA: dict[str, Any] = {
    "schema_version": int,
    "profile_id": str,
    "dataset": {
        "version": str,
        "directory": str,
        "train_features_file": str,
        "train_target_file": str,
        "label": str,
    },
    "predictor": {
        "problem_type": str,
        "eval_metric": str,
        "positive_class": int,
        "verbosity": int,
    },
    "resources": {
        "time_limit_seconds": int,
        "num_cpus": (int, str),
        "num_gpus": int,
        "fit_strategy": str,
        "fold_fitting_strategy": str,
    },
    "fit": {
        "presets": str,
        "calibrate_decision_threshold": bool,
    },
    "artifacts": {"root": str},
}


def load_config(
    config_path: Path,
    repository_root: Path,
    *,
    require_data_files: bool = True,
) -> AutoGluonConfig:
    """Load, strictly validate, and resolve a repository-relative YAML config."""
    repository_root = repository_root.resolve(strict=True)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"Cannot load config {config_path}: {error}") from error
    _validate_node(payload, _SCHEMA, "config")
    assert isinstance(payload, dict)

    if payload["schema_version"] != 1:
        raise ConfigError("config.schema_version must equal 1")
    get_profile(payload["profile_id"])
    _validate_values(payload)

    dataset_directory = _resolve_contained(
        repository_root,
        payload["dataset"]["directory"],
        "config.dataset.directory",
        must_exist=require_data_files,
    )
    train_features = _resolve_contained(
        dataset_directory,
        payload["dataset"]["train_features_file"],
        "config.dataset.train_features_file",
        must_exist=require_data_files,
    )
    train_target = _resolve_contained(
        dataset_directory,
        payload["dataset"]["train_target_file"],
        "config.dataset.train_target_file",
        must_exist=require_data_files,
    )
    artifacts_root = _resolve_contained(
        repository_root,
        payload["artifacts"]["root"],
        "config.artifacts.root",
        must_exist=False,
    )
    if require_data_files:
        if not dataset_directory.is_dir():
            raise ConfigError("config.dataset.directory must be a directory")
        for name, path in (
            ("train_features_file", train_features),
            ("train_target_file", train_target),
        ):
            if not path.is_file():
                raise ConfigError(f"config.dataset.{name} must be a file")

    portable = json.loads(json.dumps(payload))
    identity = hashlib.sha256(
        json.dumps(portable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return AutoGluonConfig(
        schema_version=payload["schema_version"],
        profile_id=payload["profile_id"],
        dataset=DatasetConfig(**payload["dataset"]),
        predictor=PredictorConfig(**payload["predictor"]),
        resources=ResourcesConfig(**payload["resources"]),
        fit=FitConfig(**payload["fit"]),
        artifacts=ArtifactsConfig(**payload["artifacts"]),
        paths=ResolvedPaths(
            repository_root=repository_root,
            dataset_directory=dataset_directory,
            train_features=train_features,
            train_target=train_target,
            artifacts_root=artifacts_root,
        ),
        portable=portable,
        identity_sha256=identity,
    )


def validation_summary(config: AutoGluonConfig) -> dict[str, Any]:
    """Return a portable validation result with no local absolute paths."""
    return {
        "valid": True,
        "schema_version": config.schema_version,
        "config_identity_sha256": config.identity_sha256,
        "dataset_version": config.dataset.version,
        "profile": profile_summary(get_profile(config.profile_id)),
        "train_only_inputs": [
            f"{config.dataset.directory}/{config.dataset.train_features_file}",
            f"{config.dataset.directory}/{config.dataset.train_target_file}",
        ],
        "artifacts_root": config.artifacts.root,
    }


def _validate_node(value: Any, schema: Any, path: str) -> None:
    if isinstance(schema, dict):
        if type(value) is not dict:
            raise ConfigError(f"{path} must be a mapping")
        expected = set(schema)
        actual = set(value)
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        if missing:
            raise ConfigError(f"{path} is missing keys: {', '.join(missing)}")
        if unknown:
            raise ConfigError(f"{path} has unknown keys: {', '.join(unknown)}")
        for key, child_schema in schema.items():
            _validate_node(value[key], child_schema, f"{path}.{key}")
        return

    accepted = schema if isinstance(schema, tuple) else (schema,)
    if not any(type(value) is expected for expected in accepted):
        names = " or ".join(expected.__name__ for expected in accepted)
        raise ConfigError(f"{path} must have exact type {names}")


def _validate_values(payload: dict[str, Any]) -> None:
    dataset = payload["dataset"]
    predictor = payload["predictor"]
    resources = payload["resources"]
    fit = payload["fit"]
    artifacts = payload["artifacts"]

    for path, value in (
        ("config.profile_id", payload["profile_id"]),
        ("config.dataset.version", dataset["version"]),
        ("config.dataset.label", dataset["label"]),
        ("config.fit.presets", fit["presets"]),
    ):
        if not value.strip():
            raise ConfigError(f"{path} must not be blank")
    for file_key in ("train_features_file", "train_target_file"):
        file_name = dataset[file_key]
        if Path(file_name).name != file_name or not file_name.endswith(".parquet"):
            raise ConfigError(f"config.dataset.{file_key} must be a Parquet filename")
    if predictor["problem_type"] != "binary":
        raise ConfigError("config.predictor.problem_type must equal 'binary'")
    if predictor["eval_metric"] != "balanced_accuracy":
        raise ConfigError("config.predictor.eval_metric must equal 'balanced_accuracy'")
    if predictor["positive_class"] != 1:
        raise ConfigError("config.predictor.positive_class must equal 1")
    if not 0 <= predictor["verbosity"] <= 4:
        raise ConfigError("config.predictor.verbosity must be between 0 and 4")
    if resources["time_limit_seconds"] <= 0:
        raise ConfigError("config.resources.time_limit_seconds must be positive")
    if isinstance(resources["num_cpus"], str):
        if resources["num_cpus"] != "auto":
            raise ConfigError(
                "config.resources.num_cpus string value must equal 'auto'"
            )
    elif resources["num_cpus"] <= 0:
        raise ConfigError("config.resources.num_cpus must be positive")
    if resources["num_gpus"] < 0:
        raise ConfigError("config.resources.num_gpus must not be negative")
    if resources["fit_strategy"] != "sequential":
        raise ConfigError("config.resources.fit_strategy must equal 'sequential'")
    if resources["fold_fitting_strategy"] != "sequential_local":
        raise ConfigError(
            "config.resources.fold_fitting_strategy must equal 'sequential_local'"
        )
    if Path(artifacts["root"]).is_absolute():
        raise ConfigError("config.artifacts.root must be repository-relative")

    profile = get_profile(payload["profile_id"])
    if (
        profile.force_num_gpus is not None
        and resources["num_gpus"] != profile.force_num_gpus
    ):
        raise ConfigError(
            f"profile {profile.profile_id} requires resources.num_gpus="
            f"{profile.force_num_gpus}"
        )


def _resolve_contained(
    base: Path,
    configured_path: str,
    field: str,
    *,
    must_exist: bool,
) -> Path:
    candidate = Path(configured_path)
    if candidate.is_absolute():
        raise ConfigError(f"{field} must be relative")
    if not configured_path.strip() or any(part == ".." for part in candidate.parts):
        raise ConfigError(f"{field} must not contain traversal")
    try:
        resolved_base = base.resolve(strict=must_exist)
        resolved = (base / candidate).resolve(strict=must_exist)
        resolved.relative_to(resolved_base)
    except (OSError, ValueError) as error:
        raise ConfigError(
            f"{field} escapes its allowed directory or is invalid"
        ) from error
    return resolved
