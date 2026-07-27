"""Strict versioned configuration for standalone AutoGluon benchmarks."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
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
    seed: int
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
    "seed": int,
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

    processed_root = _resolve_contained(
        repository_root,
        "data/processed",
        "data.processed.root",
        must_exist=require_data_files,
    )
    if processed_root != repository_root / "data" / "processed":
        raise ConfigError("data/processed must not be a symlink or junction redirect")
    dataset_directory = _resolve_contained(
        processed_root,
        payload["dataset"]["version"],
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
    identity = config_identity_sha256(portable)
    return AutoGluonConfig(
        schema_version=payload["schema_version"],
        profile_id=payload["profile_id"],
        seed=payload["seed"],
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


def config_identity_sha256(portable: dict[str, Any]) -> str:
    """Hash a portable resolved config without local absolute paths."""
    return hashlib.sha256(
        json.dumps(portable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validation_summary(config: AutoGluonConfig) -> dict[str, Any]:
    """Return a portable validation result with no local absolute paths."""
    return {
        "valid": True,
        "schema_version": config.schema_version,
        "config_identity_sha256": config.identity_sha256,
        "dataset_version": config.dataset.version,
        "seed": config.seed,
        "profile": profile_summary(
            get_profile(config.profile_id), config.seed, config.resources.num_gpus
        ),
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
    seed = payload["seed"]

    for path, value in (
        ("config.profile_id", payload["profile_id"]),
        ("config.dataset.version", dataset["version"]),
        ("config.dataset.label", dataset["label"]),
        ("config.fit.presets", fit["presets"]),
    ):
        if not value.strip():
            raise ConfigError(f"{path} must not be blank")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", dataset["version"]):
        raise ConfigError("config.dataset.version must be a lowercase safe slug")
    expected_directory = f"data/processed/{dataset['version']}"
    if dataset["directory"] != expected_directory:
        raise ConfigError(
            "config.dataset.directory must equal exactly "
            f"{expected_directory!r} for the configured dataset version"
        )
    if dataset["train_features_file"] != "X_train.parquet":
        raise ConfigError(
            "config.dataset.train_features_file must equal exactly 'X_train.parquet'"
        )
    if dataset["train_target_file"] != "y_train.parquet":
        raise ConfigError(
            "config.dataset.train_target_file must equal exactly 'y_train.parquet'"
        )
    if seed < 0:
        raise ConfigError("config.seed must not be negative")
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
    if fit["presets"] not in {"extreme_quality"}:
        raise ConfigError("config.fit.presets must be one of: 'extreme_quality'")
    _reject_windows_or_absolute_path(artifacts["root"], "config.artifacts.root")

    profile = get_profile(payload["profile_id"])
    if resources["num_gpus"] < profile.minimum_gpu_budget:
        raise ConfigError(
            f"profile {profile.profile_id} requires resources.num_gpus >= "
            f"{profile.minimum_gpu_budget}"
        )


def _resolve_contained(
    base: Path,
    configured_path: str,
    field: str,
    *,
    must_exist: bool,
) -> Path:
    _reject_windows_or_absolute_path(configured_path, field)
    candidate = Path(configured_path)
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


def _reject_windows_or_absolute_path(configured_path: str, field: str) -> None:
    windows_path = PureWindowsPath(configured_path)
    if (
        Path(configured_path).is_absolute()
        or bool(windows_path.drive)
        or bool(windows_path.root)
        or configured_path.startswith(("/", "\\"))
    ):
        raise ConfigError(
            f"{field} must be repository-relative and must not be drive-qualified or rooted"
        )
