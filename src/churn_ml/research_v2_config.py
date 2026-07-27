from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any, Mapping

import yaml

from src.churn_ml.experiment_v2 import (
    get_candidate_adapter,
    get_feature_pipeline,
)
from src.churn_ml.research_config import (
    _validate_plan_invariants,
    _validate_plan_structure,
)


SAFE_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


class ResearchV2ConfigurationError(ValueError):
    """Raised when a v2 configuration is not exact and repository-contained."""


ROOT_KEYS = {
    "schema_version",
    "experiment",
    "dataset",
    "evaluation_plan_path",
    "feature_pipeline",
    "candidate_adapter",
    "artifacts",
    "persistence",
    "tracking",
}
SECTION_KEYS = {
    "experiment": {"id"},
    "dataset": {"version"},
    "feature_pipeline": {"id", "contract"},
    "candidate_adapter": {"id", "contract"},
    "artifacts": {"root"},
    "persistence": {
        "resolved_config",
        "identities",
        "assignments",
        "predictions",
        "metrics",
        "threshold_curves",
        "models",
    },
    "tracking": {"enabled"},
}


@dataclass(frozen=True)
class ResearchV2Config:
    payload: dict[str, Any]
    plan_payload: dict[str, Any]
    source_path: Path
    plan_path: Path
    project_root: Path

    @property
    def experiment_id(self) -> str:
        return str(self.payload["experiment"]["id"])

    @property
    def plan_id(self) -> str:
        return str(self.plan_payload["plan"]["id"])

    @property
    def dataset_version(self) -> str:
        return str(self.payload["dataset"]["version"])

    @property
    def pipeline_id(self) -> str:
        return str(self.payload["feature_pipeline"]["id"])

    @property
    def pipeline_contract(self) -> dict[str, Any]:
        return deepcopy(self.payload["feature_pipeline"]["contract"])

    @property
    def adapter_id(self) -> str:
        return str(self.payload["candidate_adapter"]["id"])

    @property
    def adapter_contract(self) -> dict[str, Any]:
        return deepcopy(self.payload["candidate_adapter"]["contract"])

    @property
    def processed_data_root(self) -> Path:
        return _repo_path(
            str(self.plan_payload["dataset"]["processed_dir"]),
            self.project_root,
            "dataset.processed_dir",
        )

    @property
    def dataset_dir(self) -> Path:
        return self.processed_data_root / self.dataset_version

    @property
    def artifact_root(self) -> Path:
        return _repo_path(
            str(self.payload["artifacts"]["root"]),
            self.project_root,
            "artifacts.root",
        )

    def resolved_payload(self) -> dict[str, Any]:
        payload = deepcopy(self.payload)
        payload["evaluation_plan"] = deepcopy(self.plan_payload)
        return payload


def load_research_v2_config(
    path: Path,
    *,
    project_root: Path,
) -> ResearchV2Config:
    root = project_root.resolve()
    source = path.resolve()
    if source != root and root not in source.parents:
        raise ResearchV2ConfigurationError(
            "V2 configuration must be inside the repository."
        )
    payload = _load_mapping(source, "v2 configuration")
    _exact_keys(payload, ROOT_KEYS, "root")
    for name, expected in SECTION_KEYS.items():
        _exact_keys(_section(payload, name), expected, name)
    if _integer(payload["schema_version"], "schema_version") != 2:
        raise ResearchV2ConfigurationError("schema_version must be 2.")
    _slug(_section(payload, "experiment")["id"], "experiment.id")
    dataset_version = _slug(
        _section(payload, "dataset")["version"],
        "dataset.version",
    )
    plan_path = _repo_path(
        _string(payload["evaluation_plan_path"], "evaluation_plan_path"),
        root,
        "evaluation_plan_path",
    )
    plan = _load_mapping(plan_path, "evaluation plan")
    try:
        _validate_plan_structure(plan)
        _validate_plan_invariants(plan)
    except ValueError as error:
        raise ResearchV2ConfigurationError(str(error)) from error
    if plan["dataset"]["version"] != dataset_version:
        raise ResearchV2ConfigurationError(
            "Run and evaluation-plan dataset versions differ."
        )
    _repo_path(
        _string(plan["dataset"]["processed_dir"], "dataset.processed_dir"),
        root,
        "dataset.processed_dir",
    )
    for label, item in _section(plan["dataset"], "files").items():
        if not isinstance(item, dict):
            raise ResearchV2ConfigurationError(
                f"dataset.files.{label} must be a mapping."
            )
        name = _string(item["name"], f"dataset.files.{label}.name")
        if (
            Path(name).name != name
            or PurePosixPath(name).name != name
            or PureWindowsPath(name).name != name
        ):
            raise ResearchV2ConfigurationError(
                f"dataset.files.{label}.name must be a plain basename."
            )
    pipeline = _section(payload, "feature_pipeline")
    adapter = _section(payload, "candidate_adapter")
    pipeline_id = _slug(pipeline["id"], "feature_pipeline.id")
    adapter_id = _slug(adapter["id"], "candidate_adapter.id")
    pipeline_contract = _section(pipeline, "contract")
    adapter_contract = _section(adapter, "contract")
    try:
        get_feature_pipeline(pipeline_id).validate_contract(pipeline_contract)
        get_candidate_adapter(adapter_id).validate_contract(adapter_contract)
    except ValueError as error:
        raise ResearchV2ConfigurationError(str(error)) from error
    _repo_path(
        _string(_section(payload, "artifacts")["root"], "artifacts.root"),
        root,
        "artifacts.root",
    )
    persistence = _section(payload, "persistence")
    required_true = {
        "resolved_config",
        "identities",
        "assignments",
        "predictions",
        "metrics",
    }
    for key, value in persistence.items():
        if not isinstance(value, bool):
            raise ResearchV2ConfigurationError(f"persistence.{key} must be boolean.")
        if key in required_true and not value:
            raise ResearchV2ConfigurationError(f"persistence.{key} must be true.")
    if persistence["models"]:
        raise ResearchV2ConfigurationError("Model persistence must be disabled.")
    tracking = _section(payload, "tracking")
    if tracking["enabled"] is not False:
        raise ResearchV2ConfigurationError("Optional tracking must remain disabled.")
    return ResearchV2Config(
        payload=deepcopy(dict(payload)),
        plan_payload=deepcopy(dict(plan)),
        source_path=source,
        plan_path=plan_path,
        project_root=root,
    )


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ResearchV2ConfigurationError(f"{label} does not exist: {path}.")
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ResearchV2ConfigurationError(f"{label} must be a mapping.")
    return value


def _section(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    item = value.get(name)
    if not isinstance(item, dict):
        raise ResearchV2ConfigurationError(f"{name} must be a mapping.")
    return item


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise ResearchV2ConfigurationError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}."
        )


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ResearchV2ConfigurationError(f"{label} must be a non-empty string.")
    return value


def _integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ResearchV2ConfigurationError(f"{label} must be an integer.")
    return value


def _slug(value: Any, label: str) -> str:
    result = _string(value, label)
    if SAFE_SLUG.fullmatch(result) is None:
        raise ResearchV2ConfigurationError(f"{label} must be a safe slug.")
    return result


def _repo_path(value: str, root: Path, label: str) -> Path:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or ".." in posix.parts
        or ".." in windows.parts
    ):
        raise ResearchV2ConfigurationError(
            f"{label} must be a repository-relative path without traversal."
        )
    resolved = (root / Path(value)).resolve()
    if resolved != root and root not in resolved.parents:
        raise ResearchV2ConfigurationError(f"{label} escapes the repository.")
    return resolved
