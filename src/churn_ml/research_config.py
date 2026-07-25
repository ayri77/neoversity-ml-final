from __future__ import annotations

from copy import deepcopy
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

import yaml

from src.churn_ml.config import EXPECTED_LIGHTGBM_PARAMETERS

SAFE_SLUG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
SUPPORTED_PRIMARY_METRICS = {"balanced_accuracy"}
SUPPORTED_SECONDARY_METRICS = {
    "sensitivity",
    "specificity",
    "roc_auc",
    "average_precision",
    "brier_score",
}
SUPPORTED_DIAGNOSTIC_METRICS = {
    "tn",
    "fp",
    "fn",
    "tp",
    "predicted_positive_rate",
    "selected_threshold",
}
MAX_SEED = 2**32 - 1


class ResearchConfigurationError(ValueError):
    """Raised when a research plan or candidate-run configuration is invalid."""


PLAN_KEYS = {
    "root": {
        "schema_version",
        "plan",
        "dataset",
        "outer_evaluation",
        "threshold_selection",
        "threshold_policy",
        "metrics",
        "aggregation",
    },
    "plan": {"id"},
    "dataset": {
        "processed_dir",
        "version",
        "files",
        "expected_rows",
        "expected_source_features",
        "ordered_source_schema_sha256",
        "ordered_dtype_schema_sha256",
        "target",
    },
    "files": {"train_features", "target", "metadata"},
    "file": {"name", "sha256"},
    "target": {
        "name",
        "dtype",
        "negative_label",
        "positive_label",
        "expected_negative_rows",
        "expected_positive_rows",
        "values_sha256",
    },
    "outer_evaluation": {"splitter", "n_splits", "shuffle", "repeat_seeds"},
    "threshold_selection": {
        "splitter",
        "n_splits",
        "shuffle",
        "random_state",
    },
    "threshold_policy": {
        "id",
        "metric",
        "minimum",
        "maximum",
        "step",
        "comparison",
        "maximizer_absolute_tolerance",
        "tie_break",
        "constant_probability_fallback",
    },
    "metrics": {"primary", "secondary", "diagnostic"},
    "aggregation": {
        "repeat_method",
        "aggregate_unit",
        "standard_deviation",
        "fold_standard_deviation",
        "confidence_interval",
    },
}

RUN_KEYS = {
    "root": {
        "schema_version",
        "experiment",
        "evaluation_plan_path",
        "candidate",
        "artifacts",
    },
    "experiment": {"id"},
    "candidate": {
        "id",
        "implementation",
        "historical_config_path",
        "manifest_path",
        "manifest_key",
    },
    "artifacts": {"root", "save_threshold_curves", "save_models"},
}

FEATURE_KEYS = {
    "drop",
    "expected_model_feature_count",
    "expected_categorical_count",
    "expected_source_schema_sha256",
    "expected_model_input_schema_sha256",
    "expected_transformed_schema_sha256",
    "categorical",
}
ENCODER_KEYS = {
    "implementation",
    "inner_splits",
    "shuffle",
    "random_state",
    "alpha",
    "prior",
    "keep_original_categorical_features",
}
LIGHTGBM_KEYS = {"estimator", "parameters"}


@dataclass(frozen=True)
class ResearchConfig:
    payload: dict[str, Any]
    plan_payload: dict[str, Any]
    candidate_contract: dict[str, Any]
    source_path: Path
    plan_path: Path
    project_root: Path

    @property
    def experiment_id(self) -> str:
        return str(self.payload["experiment"]["id"])

    @property
    def candidate_id(self) -> str:
        return str(self.payload["candidate"]["id"])

    @property
    def plan_id(self) -> str:
        return str(self.plan_payload["plan"]["id"])

    @property
    def processed_dir(self) -> Path:
        value = self.plan_payload["dataset"]["processed_dir"]
        return _resolve_path(value, self.project_root)

    @property
    def dataset_dir(self) -> Path:
        return self.processed_dir / str(self.plan_payload["dataset"]["version"])

    @property
    def artifact_root(self) -> Path:
        return _resolve_path(
            self.payload["artifacts"]["root"],
            self.project_root,
        )

    def resolved_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.payload["schema_version"],
            "experiment": deepcopy(self.payload["experiment"]),
            "evaluation_plan_source_path": str(self.plan_path),
            "evaluation_plan": deepcopy(self.plan_payload),
            "candidate": {
                "id": self.candidate_id,
                "implementation": self.payload["candidate"]["implementation"],
                "historical_config_path": str(
                    _resolve_path(
                        self.payload["candidate"]["historical_config_path"],
                        self.project_root,
                    )
                ),
                "manifest_path": str(
                    _resolve_path(
                        self.payload["candidate"]["manifest_path"],
                        self.project_root,
                    )
                ),
                "manifest_key": self.payload["candidate"]["manifest_key"],
                "contract": deepcopy(self.candidate_contract),
            },
            "artifacts": {
                **deepcopy(self.payload["artifacts"]),
                "root": str(self.artifact_root),
            },
        }


def load_research_config(path: Path, *, project_root: Path) -> ResearchConfig:
    """Load strict run and plan YAML without touching competition assets."""
    root = project_root.resolve()
    source_path = path.resolve()
    run_payload = _load_mapping(source_path, "research run configuration")
    _expect_keys(run_payload, RUN_KEYS["root"], "research run root")
    _expect_keys(
        _section(run_payload, "experiment"), RUN_KEYS["experiment"], "experiment"
    )
    _expect_keys(_section(run_payload, "candidate"), RUN_KEYS["candidate"], "candidate")
    _expect_keys(_section(run_payload, "artifacts"), RUN_KEYS["artifacts"], "artifacts")

    plan_path = _resolve_path(
        _string(run_payload["evaluation_plan_path"], "evaluation_plan_path"),
        root,
    )
    plan_payload = _load_mapping(plan_path, "evaluation plan")
    _validate_plan_structure(plan_payload)
    _validate_run_invariants(run_payload)
    _validate_plan_invariants(plan_payload)

    candidate_contract = _resolve_candidate_contract(
        _section(run_payload, "candidate"),
        project_root=root,
        expected_dataset_version=str(plan_payload["dataset"]["version"]),
    )
    return ResearchConfig(
        payload=deepcopy(dict(run_payload)),
        plan_payload=deepcopy(dict(plan_payload)),
        candidate_contract=candidate_contract,
        source_path=source_path,
        plan_path=plan_path,
        project_root=root,
    )


def _validate_plan_structure(payload: Mapping[str, Any]) -> None:
    _expect_keys(payload, PLAN_KEYS["root"], "evaluation plan root")
    for section in (
        "plan",
        "dataset",
        "outer_evaluation",
        "threshold_selection",
        "threshold_policy",
        "metrics",
        "aggregation",
    ):
        _expect_keys(_section(payload, section), PLAN_KEYS[section], section)
    dataset = _section(payload, "dataset")
    files = _section(dataset, "files")
    _expect_keys(files, PLAN_KEYS["files"], "dataset.files")
    for name in PLAN_KEYS["files"]:
        _expect_keys(_section(files, name), PLAN_KEYS["file"], f"dataset.files.{name}")
    _expect_keys(_section(dataset, "target"), PLAN_KEYS["target"], "dataset.target")


def _validate_run_invariants(payload: Mapping[str, Any]) -> None:
    if _integer(payload["schema_version"], "schema_version") != 1:
        raise ResearchConfigurationError("Run schema_version must be 1.")
    _slug(_section(payload, "experiment")["id"], "experiment.id")
    candidate = _section(payload, "candidate")
    _slug(candidate["id"], "candidate.id")
    if candidate["implementation"] != "manual_lightgbm_r31":
        raise ResearchConfigurationError(
            "The first research slice supports only manual_lightgbm_r31."
        )
    artifacts = _section(payload, "artifacts")
    _string(artifacts["root"], "artifacts.root")
    _boolean(artifacts["save_threshold_curves"], "artifacts.save_threshold_curves")
    if _boolean(artifacts["save_models"], "artifacts.save_models"):
        raise ResearchConfigurationError("Research model persistence must be disabled.")


def _validate_plan_invariants(payload: Mapping[str, Any]) -> None:
    if _integer(payload["schema_version"], "schema_version") != 1:
        raise ResearchConfigurationError("Plan schema_version must be 1.")
    _slug(_section(payload, "plan")["id"], "plan.id")
    dataset = _section(payload, "dataset")
    _string(dataset["processed_dir"], "dataset.processed_dir")
    _slug(dataset["version"], "dataset.version")
    for field in ("expected_rows", "expected_source_features"):
        if _integer(dataset[field], f"dataset.{field}") <= 0:
            raise ResearchConfigurationError(f"dataset.{field} must be positive.")
    for field in (
        "ordered_source_schema_sha256",
        "ordered_dtype_schema_sha256",
    ):
        _sha256(dataset[field], f"dataset.{field}")
    for name, item in _section(dataset, "files").items():
        file_config = _mapping(item, f"dataset.files.{name}")
        _plain_basename(file_config["name"], f"dataset.files.{name}.name")
        _sha256(file_config["sha256"], f"dataset.files.{name}.sha256")

    target = _section(dataset, "target")
    _string(target["name"], "dataset.target.name")
    _string(target["dtype"], "dataset.target.dtype")
    negative = _integer(target["negative_label"], "dataset.target.negative_label")
    positive = _integer(target["positive_label"], "dataset.target.positive_label")
    if (negative, positive) != (0, 1):
        raise ResearchConfigurationError(
            "The first slice requires target labels 0 and 1."
        )
    for field in ("expected_negative_rows", "expected_positive_rows"):
        if _integer(target[field], f"dataset.target.{field}") <= 0:
            raise ResearchConfigurationError(
                f"dataset.target.{field} must be positive."
            )
    if (
        target["expected_negative_rows"] + target["expected_positive_rows"]
        != dataset["expected_rows"]
    ):
        raise ResearchConfigurationError("Target class counts do not equal row count.")
    _sha256(target["values_sha256"], "dataset.target.values_sha256")

    outer = _section(payload, "outer_evaluation")
    if outer["splitter"] != "stratified_kfold":
        raise ResearchConfigurationError("Outer splitter must be stratified_kfold.")
    if not _boolean(outer["shuffle"], "outer_evaluation.shuffle"):
        raise ResearchConfigurationError("Outer evaluation must shuffle.")
    outer_splits = _integer(outer["n_splits"], "outer_evaluation.n_splits")
    if not 2 <= outer_splits <= 100:
        raise ResearchConfigurationError(
            "Outer evaluation fold count must be between 2 and 100."
        )
    repeat_seeds = _integers(outer["repeat_seeds"], "outer_evaluation.repeat_seeds")
    if not repeat_seeds or len(set(repeat_seeds)) != len(repeat_seeds):
        raise ResearchConfigurationError(
            "Outer repeat seeds must be non-empty and unique."
        )

    if any(seed < 0 or seed > MAX_SEED for seed in repeat_seeds):
        raise ResearchConfigurationError("Outer repeat seeds are out of range.")

    threshold_cv = _section(payload, "threshold_selection")
    if threshold_cv["splitter"] != "stratified_kfold":
        raise ResearchConfigurationError(
            "Threshold-selection splitter must be stratified_kfold."
        )
    if not _boolean(threshold_cv["shuffle"], "threshold_selection.shuffle"):
        raise ResearchConfigurationError("Threshold-selection CV must shuffle.")
    threshold_splits = _integer(
        threshold_cv["n_splits"], "threshold_selection.n_splits"
    )
    if not 2 <= threshold_splits <= 100:
        raise ResearchConfigurationError(
            "Threshold-selection fold count must be between 2 and 100."
        )
    threshold_seed = _integer(
        threshold_cv["random_state"], "threshold_selection.random_state"
    )
    if threshold_seed < 0 or threshold_seed > MAX_SEED:
        raise ResearchConfigurationError("Threshold-selection seed is out of range.")

    policy = _section(payload, "threshold_policy")
    expected_strings = {
        "id": "grid_balanced_accuracy_v1",
        "metric": "balanced_accuracy",
        "comparison": "greater_than_or_equal",
        "tie_break": "median_maximizer_lower_on_even",
    }
    for field, expected in expected_strings.items():
        if policy[field] != expected:
            raise ResearchConfigurationError(
                f"threshold_policy.{field} must be {expected!r}."
            )
    minimum = _number(policy["minimum"], "threshold_policy.minimum")
    maximum = _number(policy["maximum"], "threshold_policy.maximum")
    step = _number(policy["step"], "threshold_policy.step")
    fallback = _number(
        policy["constant_probability_fallback"],
        "threshold_policy.constant_probability_fallback",
    )
    tolerance = _number(
        policy["maximizer_absolute_tolerance"],
        "threshold_policy.maximizer_absolute_tolerance",
    )
    if not 0.0 <= minimum <= fallback <= maximum <= 1.0 or step <= 0:
        raise ResearchConfigurationError("Threshold grid and fallback are invalid.")
    if tolerance < 0:
        raise ResearchConfigurationError(
            "Threshold score tolerance must be non-negative."
        )

    metrics = _section(payload, "metrics")
    _validate_metrics(metrics)
    aggregation = _section(payload, "aggregation")
    expected_aggregation = {
        "repeat_method": "pooled_predictions",
        "aggregate_unit": "repeat",
        "standard_deviation": "sample",
        "fold_standard_deviation": "descriptive_only",
        "confidence_interval": "none",
    }
    if dict(aggregation) != expected_aggregation:
        raise ResearchConfigurationError(
            "Aggregation settings must match the research protocol."
        )


def _validate_metrics(metrics: Mapping[str, Any]) -> None:
    primary = _string(metrics["primary"], "metrics.primary")
    secondary = _strings(metrics["secondary"], "metrics.secondary")
    diagnostic = _strings(metrics["diagnostic"], "metrics.diagnostic")
    if primary not in SUPPORTED_PRIMARY_METRICS:
        raise ResearchConfigurationError("Unsupported primary metric.")
    unknown_secondary = sorted(set(secondary) - SUPPORTED_SECONDARY_METRICS)
    unknown_diagnostic = sorted(set(diagnostic) - SUPPORTED_DIAGNOSTIC_METRICS)
    if unknown_secondary or unknown_diagnostic:
        raise ResearchConfigurationError(
            f"Unsupported metrics: {unknown_secondary + unknown_diagnostic}"
        )
    names = [primary, *secondary, *diagnostic]
    if len(names) != len(set(names)):
        raise ResearchConfigurationError("Metric names must not be duplicated.")


def _resolve_candidate_contract(
    candidate: Mapping[str, Any],
    *,
    project_root: Path,
    expected_dataset_version: str,
) -> dict[str, Any]:
    """Validate only candidate sections; ignore all competition-path sections."""
    historical_path = _resolve_path(candidate["historical_config_path"], project_root)
    manifest_path = _resolve_path(candidate["manifest_path"], project_root)
    historical = _load_mapping(historical_path, "historical candidate configuration")
    manifest = _load_mapping(manifest_path, "baseline manifest")
    manifest_key = _string(candidate["manifest_key"], "candidate.manifest_key")
    if manifest_key not in manifest:
        raise ResearchConfigurationError(f"Manifest key not found: {manifest_key}")
    contract = _mapping(manifest[manifest_key], "candidate manifest contract")

    historical_dataset = _section(historical, "dataset")
    if historical_dataset.get("version") != expected_dataset_version:
        raise ResearchConfigurationError(
            "Candidate dataset version differs from the evaluation plan."
        )
    if contract.get("dataset_version") != expected_dataset_version:
        raise ResearchConfigurationError(
            "Manifest candidate dataset version differs from the evaluation plan."
        )

    features = _section(historical, "features")
    encoder = _section(historical, "target_encoder")
    lightgbm = _section(historical, "lightgbm")
    _expect_keys(features, FEATURE_KEYS, "historical features")
    _expect_keys(encoder, ENCODER_KEYS, "historical target_encoder")
    _expect_keys(lightgbm, LIGHTGBM_KEYS, "historical lightgbm")

    manifest_features = _mapping(
        contract["feature_contract"], "manifest feature contract"
    )
    feature_comparisons = {
        "drop": manifest_features["dropped_features"],
        "expected_model_feature_count": manifest_features[
            "final_model_input_feature_count"
        ],
        "expected_categorical_count": manifest_features["categorical_feature_count"],
        "categorical": manifest_features["categorical_features"],
    }
    _require_equal_sections(features, feature_comparisons, "feature")

    manifest_encoder = _mapping(
        contract["target_encoding_contract"],
        "manifest target-encoder contract",
    )
    encoder_comparisons = {
        "implementation": manifest_encoder["implementation"],
        "inner_splits": manifest_encoder["inner_folds"],
        "shuffle": manifest_encoder["shuffle"],
        "random_state": manifest_encoder["random_state"],
        "alpha": float(manifest_encoder["alpha"]),
        "prior": manifest_encoder["prior_definition"],
        "keep_original_categorical_features": manifest_encoder[
            "keep_original_categorical_features"
        ],
    }
    _require_equal_sections(encoder, encoder_comparisons, "target encoder")
    inner_splits = _integer(encoder["inner_splits"], "target_encoder.inner_splits")
    if not 2 <= inner_splits <= 100:
        raise ResearchConfigurationError(
            "Target-encoder fold count must be between 2 and 100."
        )
    encoder_seed = _integer(encoder["random_state"], "target_encoder.random_state")
    if encoder_seed < 0 or encoder_seed > MAX_SEED:
        raise ResearchConfigurationError("Target-encoder seed is out of range.")
    _boolean(encoder["shuffle"], "target_encoder.shuffle")
    _boolean(encoder["keep_original_categorical_features"], "target_encoder.keep")
    if _number(encoder["alpha"], "target_encoder.alpha") <= 0:
        raise ResearchConfigurationError("Target-encoder alpha must be positive.")

    if lightgbm["estimator"] != "LGBMClassifier":
        raise ResearchConfigurationError("Candidate estimator must be LGBMClassifier.")
    parameters = _mapping(lightgbm["parameters"], "historical LightGBM parameters")
    if dict(parameters) != EXPECTED_LIGHTGBM_PARAMETERS:
        raise ResearchConfigurationError(
            "Historical LightGBM parameters differ from the frozen implementation."
        )
    manifest_parameters = _mapping(
        contract["lightgbm_parameters"],
        "manifest LightGBM parameters",
    )
    for name, expected in manifest_parameters.items():
        historical_name = "random_state" if name == "seed" else name
        if parameters.get(historical_name) != expected:
            raise ResearchConfigurationError(
                f"LightGBM parameter mismatch for {historical_name}: "
                f"historical={parameters.get(historical_name)!r}, "
                f"manifest={expected!r}"
            )

    return {
        "candidate_id": _slug(candidate["id"], "candidate.id"),
        "implementation": _string(
            candidate["implementation"],
            "candidate.implementation",
        ),
        "source": {
            "historical_config_path": str(historical_path),
            "manifest_path": str(manifest_path),
            "manifest_key": manifest_key,
        },
        "features": deepcopy(dict(features)),
        "target_encoder": deepcopy(dict(encoder)),
        "lightgbm": deepcopy(dict(lightgbm)),
    }


def _require_equal_sections(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    label: str,
) -> None:
    mismatches = [
        f"{name}: historical={actual.get(name)!r}, manifest={value!r}"
        for name, value in expected.items()
        if actual.get(name) != value
    ]
    if mismatches:
        raise ResearchConfigurationError(
            f"Historical {label} contract differs from manifest:\n- "
            + "\n- ".join(mismatches)
        )


def _load_mapping(path: Path, name: str) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return _mapping(yaml.safe_load(file), name)


def _section(root: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if name not in root:
        raise ResearchConfigurationError(f"Missing required section: {name}")
    return _mapping(root[name], name)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResearchConfigurationError(f"{name} must be a mapping.")
    return value


def _expect_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ResearchConfigurationError(
            f"Invalid keys in {name}: missing={missing}, unknown={unknown}"
        )


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ResearchConfigurationError(f"{name} must be a non-empty string.")
    return value


def _plain_basename(value: Any, name: str) -> str:
    text = _string(value, name)
    if text != text.strip() or not text.strip():
        raise ResearchConfigurationError(f"{name} must not contain whitespace padding.")
    if text in {".", ".."}:
        raise ResearchConfigurationError(f"{name} must be a plain filename.")
    if "/" in text or "\\" in text:
        raise ResearchConfigurationError(f"{name} must not contain path separators.")
    if PurePosixPath(text).is_absolute() or PureWindowsPath(text).is_absolute():
        raise ResearchConfigurationError(f"{name} must not be an absolute path.")
    if Path(text).name != text:
        raise ResearchConfigurationError(f"{name} must be a plain filename.")
    return text


def _slug(value: Any, name: str) -> str:
    text = _string(value, name)
    if SAFE_SLUG_PATTERN.fullmatch(text) is None:
        raise ResearchConfigurationError(
            f"{name} must match {SAFE_SLUG_PATTERN.pattern!r}."
        )
    return text


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResearchConfigurationError(f"{name} must be an integer.")
    return value


def _integers(value: Any, name: str) -> list[int]:
    if not isinstance(value, list):
        raise ResearchConfigurationError(f"{name} must be a list.")
    return [_integer(item, f"{name}[]") for item in value]


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResearchConfigurationError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ResearchConfigurationError(f"{name} must be finite.")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ResearchConfigurationError(f"{name} must be a boolean.")
    return value


def _strings(value: Any, name: str) -> list[str]:
    if not isinstance(value, list):
        raise ResearchConfigurationError(f"{name} must be a list.")
    return [_string(item, f"{name}[]") for item in value]


def _sha256(value: Any, name: str) -> str:
    digest = _string(value, name)
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ResearchConfigurationError(f"{name} must be a lowercase SHA256 digest.")
    return digest


def _resolve_path(value: Any, root: Path) -> Path:
    path = Path(_string(value, "path"))
    return path.resolve() if path.is_absolute() else (root / path).resolve()
