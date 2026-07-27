from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

import yaml

from src.churn_ml.experiment_v2 import get_candidate_adapter
from src.churn_ml.optuna_search_resume import build_resume_authentication
from src.churn_ml.research_data import canonical_sha256
from src.churn_ml.research_v2_config import (
    ResearchV2Config,
    load_research_v2_config,
)


class OptunaSearchConfigurationError(ValueError):
    """Raised when an Optuna search contract is not exact and portable."""


SAFE_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
PLAN_KEYS = {
    "schema_version",
    "search_plan_id",
    "dataset_version",
    "pipeline_id",
    "adapter_id",
    "base_candidate_config",
    "search_space",
    "repeats",
    "folds",
    "assignment_seed",
    "threshold_policy",
    "threshold_grid",
    "metric",
    "direction",
    "n_trials",
    "timeout_seconds",
    "sampler",
    "pruner",
    "study_name",
    "storage",
    "artifacts_root",
}
THRESHOLD_GRID_KEYS = {
    "minimum",
    "maximum",
    "step",
    "comparison",
    "maximizer_absolute_tolerance",
    "tie_break",
    "constant_probability_fallback",
}
SAMPLER_KEYS = {"name", "seed"}
SPACE_KEYS = {"schema_version", "search_space_id", "adapter_id", "parameters"}

XGBOOST_DISTRIBUTIONS = {
    "n_estimators": "int",
    "learning_rate": "float",
    "max_depth": "int",
    "min_child_weight": "float",
    "subsample": "float",
    "colsample_bytree": "float",
    "gamma": "float",
    "reg_alpha": "zero_or_log_float",
    "reg_lambda": "float",
    "max_bin": "int",
}
CATBOOST_DISTRIBUTIONS = {
    "iterations": "int",
    "learning_rate": "float",
    "depth": "int",
    "l2_leaf_reg": "float",
    "random_strength": "float",
    "bagging_temperature": "float",
    "border_count": "int",
}
ADAPTER_DISTRIBUTIONS = {
    "xgboost_numeric_v1": XGBOOST_DISTRIBUTIONS,
    "catboost_numeric_v1": CATBOOST_DISTRIBUTIONS,
}
INT_LIMITS = {
    "n_estimators": (1, 100_000),
    "max_depth": (1, 64),
    "max_bin": (2, 65_536),
    "iterations": (1, 100_000),
    "depth": (1, 16),
    "border_count": (1, 65_535),
}
FLOAT_LIMITS: dict[str, tuple[float, float | None, bool]] = {
    "learning_rate": (0.0, 1.0, True),
    "min_child_weight": (0.0, None, False),
    "subsample": (0.0, 1.0, True),
    "colsample_bytree": (0.0, 1.0, True),
    "gamma": (0.0, None, False),
    "reg_lambda": (0.0, None, False),
    "l2_leaf_reg": (0.0, None, False),
    "random_strength": (0.0, None, False),
    "bagging_temperature": (0.0, None, False),
}


@dataclass(frozen=True)
class SearchSpace:
    payload: dict[str, Any]
    source_path: Path
    sha256: str

    @property
    def search_space_id(self) -> str:
        return str(self.payload["search_space_id"])

    @property
    def adapter_id(self) -> str:
        return str(self.payload["adapter_id"])

    @property
    def parameters(self) -> dict[str, dict[str, Any]]:
        return deepcopy(self.payload["parameters"])


@dataclass(frozen=True)
class OptunaSearchConfig:
    payload: dict[str, Any]
    source_path: Path
    project_root: Path
    base_config: ResearchV2Config
    search_space: SearchSpace
    resume_authentication: dict[str, Any]
    study_identity: dict[str, Any]
    study_identity_sha256: str
    search_identity: dict[str, Any]
    search_identity_sha256: str
    search_id: str

    @property
    def adapter_id(self) -> str:
        return str(self.payload["adapter_id"])

    @property
    def model_section(self) -> str:
        return "xgboost" if self.adapter_id == "xgboost_numeric_v1" else "catboost"

    @property
    def storage_path(self) -> Path:
        return _repo_path(
            self.payload["storage"],
            self.project_root,
            "storage",
        )

    @property
    def artifact_root(self) -> Path:
        return _repo_path(
            self.payload["artifacts_root"],
            self.project_root,
            "artifacts_root",
        )

    @property
    def threshold_policy_payload(self) -> dict[str, Any]:
        grid = self.payload["threshold_grid"]
        return {
            "id": self.payload["threshold_policy"],
            "metric": self.payload["metric"],
            **deepcopy(grid),
        }

    def resolved_payload(self) -> dict[str, Any]:
        payload = deepcopy(self.payload)
        payload["base_candidate_config_sha256"] = canonical_sha256(
            self.base_config.payload
        )
        payload["search_space_identity"] = {
            "id": self.search_space.search_space_id,
            "sha256": self.search_space.sha256,
        }
        payload["resume_authentication"] = deepcopy(self.resume_authentication)
        payload["study_identity_sha256"] = self.study_identity_sha256
        payload["search_identity_sha256"] = self.search_identity_sha256
        payload["search_id"] = self.search_id
        return payload


def load_optuna_search_config(
    path: Path,
    *,
    project_root: Path,
) -> OptunaSearchConfig:
    root = project_root.resolve()
    source = _contained_file(path, root, "search config")
    payload = _load_mapping(source, "search config")
    _exact_keys(payload, PLAN_KEYS, "root")
    if _integer(payload["schema_version"], "schema_version") != 1:
        raise OptunaSearchConfigurationError("schema_version must be 1.")
    _slug(payload["search_plan_id"], "search_plan_id")
    _slug(payload["dataset_version"], "dataset_version")
    _slug(payload["pipeline_id"], "pipeline_id")
    adapter_id = _slug(payload["adapter_id"], "adapter_id")
    if adapter_id not in ADAPTER_DISTRIBUTIONS:
        raise OptunaSearchConfigurationError(
            "adapter_id must be xgboost_numeric_v1 or catboost_numeric_v1."
        )
    base_path = _repo_path(
        _string(payload["base_candidate_config"], "base_candidate_config"),
        root,
        "base_candidate_config",
    )
    space_path = _repo_path(
        _string(payload["search_space"], "search_space"),
        root,
        "search_space",
    )
    try:
        base = load_research_v2_config(base_path, project_root=root)
    except ValueError as error:
        raise OptunaSearchConfigurationError(
            f"Base Experiment Core v2 config is invalid: {error}"
        ) from error
    space = load_search_space(space_path, project_root=root)
    for label, actual, expected in (
        ("dataset_version", payload["dataset_version"], base.dataset_version),
        ("pipeline_id", payload["pipeline_id"], base.pipeline_id),
        ("adapter_id", adapter_id, base.adapter_id),
        ("search_space.adapter_id", space.adapter_id, adapter_id),
    ):
        if actual != expected:
            raise OptunaSearchConfigurationError(
                f"{label} differs from its referenced production contract."
            )
    repeats = _bounded_integer(payload["repeats"], "repeats", 1, 10)
    folds = _bounded_integer(payload["folds"], "folds", 2, 10)
    _bounded_integer(payload["assignment_seed"], "assignment_seed", 0, 2**31 - 1)
    if repeats * folds > 100:
        raise OptunaSearchConfigurationError("repeats * folds must not exceed 100.")
    if payload["threshold_policy"] != "grid_balanced_accuracy_v1":
        raise OptunaSearchConfigurationError(
            "threshold_policy must be grid_balanced_accuracy_v1."
        )
    _validate_threshold_grid(payload["threshold_grid"])
    if payload["metric"] != "balanced_accuracy":
        raise OptunaSearchConfigurationError(
            "metric must be balanced_accuracy for schema v1."
        )
    if payload["direction"] != "maximize":
        raise OptunaSearchConfigurationError("direction must be maximize.")
    search_threshold_policy = {
        "id": payload["threshold_policy"],
        "metric": payload["metric"],
        **deepcopy(payload["threshold_grid"]),
    }
    if search_threshold_policy != base.plan_payload["threshold_policy"]:
        raise OptunaSearchConfigurationError(
            "Search threshold policy/grid differs from Experiment Core v2."
        )
    target_contract = base.plan_payload["dataset"]["target"]
    adapter_identity = get_candidate_adapter(adapter_id).identity_inputs(
        base.adapter_contract
    )
    if (
        target_contract["negative_label"] != 0
        or target_contract["positive_label"] != 1
        or adapter_identity.get("probability_semantics")
        != "binary_positive_class_label_1"
        or adapter_identity.get("early_stopping") != "disabled"
    ):
        raise OptunaSearchConfigurationError(
            "Positive-class, probability, or fit semantics differ from v2."
        )
    _bounded_integer(payload["n_trials"], "n_trials", 1, 500)
    _bounded_integer(payload["timeout_seconds"], "timeout_seconds", 1, 86_400)
    sampler = _mapping(payload["sampler"], "sampler")
    _exact_keys(sampler, SAMPLER_KEYS, "sampler")
    if sampler["name"] != "stateless_random_v1":
        raise OptunaSearchConfigurationError(
            "sampler.name must be stateless_random_v1."
        )
    _bounded_integer(sampler["seed"], "sampler.seed", 0, 2**31 - 1)
    if payload["pruner"] != "nop":
        raise OptunaSearchConfigurationError("pruner must be nop for schema v1.")
    _slug(payload["study_name"], "study_name")
    storage = _repo_path(
        _string(payload["storage"], "storage"),
        root,
        "storage",
    )
    artifact_root = _repo_path(
        _string(payload["artifacts_root"], "artifacts_root"),
        root,
        "artifacts_root",
    )
    if storage.suffix != ".db":
        raise OptunaSearchConfigurationError("storage must use a .db filename.")
    _require_namespace(storage, root / "artifacts" / "optuna", "storage")
    _require_namespace(
        artifact_root,
        root / "artifacts" / "optuna_searches",
        "artifacts_root",
        allow_equal=True,
    )
    resume_authentication = build_resume_authentication(
        project_root=root,
        adapter_id=adapter_id,
        contract_paths=(
            base.source_path.relative_to(root).as_posix(),
            base.plan_path.relative_to(root).as_posix(),
            space.source_path.relative_to(root).as_posix(),
        ),
    )
    study_identity = {
        "schema_version": 1,
        "search_plan_id": payload["search_plan_id"],
        "dataset_version": payload["dataset_version"],
        "pipeline_id": payload["pipeline_id"],
        "adapter_id": adapter_id,
        "base_candidate_config_sha256": canonical_sha256(base.payload),
        "search_space": {
            "id": space.search_space_id,
            "sha256": space.sha256,
        },
        "repeats": repeats,
        "folds": folds,
        "assignment_seed": payload["assignment_seed"],
        "threshold_policy": payload["threshold_policy"],
        "threshold_grid": deepcopy(payload["threshold_grid"]),
        "metric": payload["metric"],
        "direction": payload["direction"],
        "sampler": deepcopy(sampler),
        "pruner": payload["pruner"],
        "study_name": payload["study_name"],
        "positive_class_label": 1,
        "probability_semantics": "binary_positive_class_label_1",
        "label_comparison": "greater_than_or_equal",
        "resume_authentication": deepcopy(resume_authentication),
    }
    study_sha = canonical_sha256(study_identity)
    search_identity = {
        **deepcopy(study_identity),
        "target_n_trials": payload["n_trials"],
        "timeout_seconds": payload["timeout_seconds"],
    }
    search_sha = canonical_sha256(search_identity)
    search_id = f"{payload['search_plan_id']}_{search_sha[:16]}"
    return OptunaSearchConfig(
        payload=deepcopy(payload),
        source_path=source,
        project_root=root,
        base_config=base,
        search_space=space,
        resume_authentication=resume_authentication,
        study_identity=study_identity,
        study_identity_sha256=study_sha,
        search_identity=search_identity,
        search_identity_sha256=search_sha,
        search_id=search_id,
    )


def load_search_space(path: Path, *, project_root: Path) -> SearchSpace:
    root = project_root.resolve()
    source = _contained_file(path, root, "search space")
    payload = _load_mapping(source, "search space")
    _exact_keys(payload, SPACE_KEYS, "search_space")
    if _integer(payload["schema_version"], "search_space.schema_version") != 1:
        raise OptunaSearchConfigurationError("search_space.schema_version must be 1.")
    _slug(payload["search_space_id"], "search_space.search_space_id")
    adapter_id = _slug(payload["adapter_id"], "search_space.adapter_id")
    expected = ADAPTER_DISTRIBUTIONS.get(adapter_id)
    if expected is None:
        raise OptunaSearchConfigurationError(
            "search_space.adapter_id is not supported."
        )
    parameters = _mapping(payload["parameters"], "search_space.parameters")
    _exact_keys(parameters, set(expected), "search_space.parameters")
    for name, distribution_name in expected.items():
        distribution = _mapping(
            parameters[name],
            f"search_space.parameters.{name}",
        )
        if distribution_name == "int":
            _validate_int_distribution(name, distribution)
        elif distribution_name == "float":
            _validate_float_distribution(name, distribution)
        else:
            _validate_zero_or_log_distribution(name, distribution)
    canonical = deepcopy(payload)
    return SearchSpace(
        payload=canonical,
        source_path=source,
        sha256=canonical_sha256(canonical),
    )


def _validate_int_distribution(name: str, value: Mapping[str, Any]) -> None:
    label = f"search_space.parameters.{name}"
    _exact_keys(value, {"distribution", "low", "high", "step", "log"}, label)
    if value["distribution"] != "int":
        raise OptunaSearchConfigurationError(f"{label}.distribution must be int.")
    low = _integer(value["low"], f"{label}.low")
    high = _integer(value["high"], f"{label}.high")
    step = _integer(value["step"], f"{label}.step")
    log = _boolean(value["log"], f"{label}.log")
    minimum, maximum = INT_LIMITS[name]
    if not minimum <= low <= high <= maximum or step <= 0:
        raise OptunaSearchConfigurationError(f"{label} has an invalid bounded range.")
    if (high - low) % step:
        raise OptunaSearchConfigurationError(
            f"{label} range must be exactly divisible by step."
        )
    if log and step != 1:
        raise OptunaSearchConfigurationError(
            f"{label}.step must be 1 for a log integer distribution."
        )


def _validate_float_distribution(name: str, value: Mapping[str, Any]) -> None:
    label = f"search_space.parameters.{name}"
    _exact_keys(value, {"distribution", "low", "high", "log"}, label)
    if value["distribution"] != "float":
        raise OptunaSearchConfigurationError(f"{label}.distribution must be float.")
    low = _float(value["low"], f"{label}.low")
    high = _float(value["high"], f"{label}.high")
    log = _boolean(value["log"], f"{label}.log")
    minimum, maximum, lower_open = FLOAT_LIMITS[name]
    lower_valid = low > minimum if lower_open else low >= minimum
    if not lower_valid or high < low or (maximum is not None and high > maximum):
        raise OptunaSearchConfigurationError(f"{label} has an invalid bounded range.")
    if log and low <= 0.0:
        raise OptunaSearchConfigurationError(
            f"{label}.low must be positive for a log distribution."
        )


def _validate_zero_or_log_distribution(
    name: str,
    value: Mapping[str, Any],
) -> None:
    label = f"search_space.parameters.{name}"
    _exact_keys(
        value,
        {
            "distribution",
            "zero_probability",
            "low_positive",
            "high",
        },
        label,
    )
    if name != "reg_alpha" or value["distribution"] != "zero_or_log_float":
        raise OptunaSearchConfigurationError(
            f"{label}.distribution must be zero_or_log_float."
        )
    probability = _float(value["zero_probability"], f"{label}.zero_probability")
    low = _float(value["low_positive"], f"{label}.low_positive")
    high = _float(value["high"], f"{label}.high")
    if not 0.0 < probability < 1.0 or not 0.0 < low <= high:
        raise OptunaSearchConfigurationError(f"{label} has an invalid bounded range.")


def _validate_threshold_grid(value: Any) -> None:
    grid = _mapping(value, "threshold_grid")
    _exact_keys(grid, THRESHOLD_GRID_KEYS, "threshold_grid")
    expected_strings = {
        "comparison": "greater_than_or_equal",
        "tie_break": "median_maximizer_lower_on_even",
    }
    for key, expected in expected_strings.items():
        if grid[key] != expected:
            raise OptunaSearchConfigurationError(
                f"threshold_grid.{key} must be exactly {expected!r}."
            )
    minimum = _float(grid["minimum"], "threshold_grid.minimum")
    maximum = _float(grid["maximum"], "threshold_grid.maximum")
    step = _float(grid["step"], "threshold_grid.step")
    tolerance = _float(
        grid["maximizer_absolute_tolerance"],
        "threshold_grid.maximizer_absolute_tolerance",
    )
    fallback = _float(
        grid["constant_probability_fallback"],
        "threshold_grid.constant_probability_fallback",
    )
    if (
        not 0.0 <= minimum < maximum <= 1.0
        or step <= 0.0
        or tolerance < 0.0
        or not 0.0 <= fallback <= 1.0
    ):
        raise OptunaSearchConfigurationError("threshold_grid values are invalid.")
    scaled = [minimum * 1000.0, maximum * 1000.0, step * 1000.0]
    if any(abs(item - round(item)) > 1e-9 for item in scaled):
        raise OptunaSearchConfigurationError(
            "threshold_grid must align to integer thousandths."
        )
    low_tick, high_tick, step_tick = (round(item) for item in scaled)
    if step_tick <= 0 or (high_tick - low_tick) % step_tick:
        raise OptunaSearchConfigurationError(
            "threshold_grid must be exactly inclusive."
        )


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise OptunaSearchConfigurationError(f"{label} does not exist: {path}.")
    try:
        with path.open("r", encoding="utf-8") as file:
            value = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise OptunaSearchConfigurationError(f"{label} is invalid YAML.") from error
    if not isinstance(value, dict):
        raise OptunaSearchConfigurationError(f"{label} must be a mapping.")
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OptunaSearchConfigurationError(f"{label} must be a mapping.")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise OptunaSearchConfigurationError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}."
        )


def _string(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise OptunaSearchConfigurationError(f"{label} must be a non-empty string.")
    return value


def _slug(value: Any, label: str) -> str:
    result = _string(value, label)
    if SAFE_SLUG.fullmatch(result) is None:
        raise OptunaSearchConfigurationError(f"{label} must be a safe slug.")
    return result


def _integer(value: Any, label: str) -> int:
    if type(value) is not int:
        raise OptunaSearchConfigurationError(f"{label} must be an integer.")
    return value


def _bounded_integer(
    value: Any,
    label: str,
    minimum: int,
    maximum: int,
) -> int:
    result = _integer(value, label)
    if not minimum <= result <= maximum:
        raise OptunaSearchConfigurationError(
            f"{label} must be in [{minimum}, {maximum}]."
        )
    return result


def _float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise OptunaSearchConfigurationError(f"{label} must be a finite float.")
    return value


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise OptunaSearchConfigurationError(f"{label} must be boolean.")
    return value


def _contained_file(path: Path, root: Path, label: str) -> Path:
    unresolved = path if path.is_absolute() else root / path
    source = unresolved.resolve()
    if source == root or root not in source.parents or not source.is_file():
        raise OptunaSearchConfigurationError(
            f"{label} must be an existing repository-contained file."
        )
    return source


def _repo_path(value: str, root: Path, label: str) -> Path:
    if "\\" in value:
        raise OptunaSearchConfigurationError(
            f"{label} must use POSIX repository-relative separators."
        )
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        not value
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or ".." in posix.parts
        or ".." in windows.parts
        or posix.parts[0] in {"", "."}
    ):
        raise OptunaSearchConfigurationError(
            f"{label} must be a portable repository-relative path without traversal."
        )
    resolved = (root / Path(*posix.parts)).resolve()
    if resolved == root or root not in resolved.parents:
        raise OptunaSearchConfigurationError(f"{label} escapes the repository.")
    return resolved


def _require_namespace(
    path: Path,
    namespace: Path,
    label: str,
    *,
    allow_equal: bool = False,
) -> None:
    root = namespace.resolve()
    if not ((allow_equal and path == root) or root in path.parents):
        raise OptunaSearchConfigurationError(
            f"{label} must be contained under {namespace.as_posix()}."
        )
