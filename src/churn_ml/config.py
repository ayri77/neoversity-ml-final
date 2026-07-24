from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigurationError(ValueError):
    """Raised when the manual experiment configuration is invalid."""


SCHEMA: dict[str, set[str]] = {
    "root": {
        "schema_version",
        "experiment",
        "contract",
        "dataset",
        "features",
        "outer_cv",
        "target_encoder",
        "lightgbm",
        "thresholds",
        "prediction",
        "artifacts",
        "parity",
    },
    "experiment": {"id", "implementation"},
    "contract": {"manifest_path", "manifest_key"},
    "dataset": {
        "processed_dir",
        "version",
        "sample_submission_path",
        "expected_train_rows",
        "expected_test_rows",
        "expected_source_feature_count",
    },
    "features": {
        "drop",
        "expected_model_feature_count",
        "expected_categorical_count",
        "expected_source_schema_sha256",
        "expected_model_input_schema_sha256",
        "expected_transformed_schema_sha256",
        "categorical",
    },
    "outer_cv": {"splitter", "n_splits", "shuffle", "random_state"},
    "target_encoder": {
        "implementation",
        "inner_splits",
        "shuffle",
        "random_state",
        "alpha",
        "prior",
        "keep_original_categorical_features",
    },
    "lightgbm": {"estimator", "parameters"},
    "thresholds": {"submission", "legacy_diagnostic_grid"},
    "thresholds.legacy_diagnostic_grid": {"minimum", "maximum", "step"},
    "prediction": {"test_aggregation", "full_data_refit"},
    "artifacts": {"root", "save_fold_models", "save_encoder_states"},
    "parity": {
        "enabled",
        "required",
        "reference_submission_path",
        "expected_balanced_accuracy",
        "balanced_accuracy_tolerance",
        "expected_positive_count",
        "secondary",
    },
    "parity.secondary": {
        "global_optimized_oof_balanced_accuracy",
        "global_optimized_threshold",
        "legacy_cross_fitted_balanced_accuracy",
    },
}

EXPECTED_LIGHTGBM_PARAMETERS: dict[str, Any] = {
    "objective": "binary",
    "n_estimators": 376,
    "learning_rate": 0.0397613094741,
    "num_leaves": 16,
    "min_data_in_leaf": 1,
    "feature_fraction": 0.5791844062459,
    "bagging_fraction": 0.9591526242875,
    "bagging_freq": 1,
    "lambda_l1": 0.938461750637,
    "lambda_l2": 0.9899852075056,
    "extra_trees": False,
    "max_cat_to_onehot": 27,
    "cat_l2": 1.8962346412823,
    "cat_smooth": 0.0215219089995,
    "min_data_per_group": 39,
    "random_state": 0,
    "n_jobs": -1,
    "verbosity": -1,
}

INTEGER_LIGHTGBM_PARAMETERS = {
    "n_estimators",
    "num_leaves",
    "min_data_in_leaf",
    "bagging_freq",
    "max_cat_to_onehot",
    "min_data_per_group",
    "random_state",
    "n_jobs",
    "verbosity",
}


LIGHTGBM_PARAMETER_KEYS = {
    "objective",
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "min_data_in_leaf",
    "feature_fraction",
    "bagging_fraction",
    "bagging_freq",
    "lambda_l1",
    "lambda_l2",
    "extra_trees",
    "max_cat_to_onehot",
    "cat_l2",
    "cat_smooth",
    "min_data_per_group",
    "random_state",
    "n_jobs",
    "verbosity",
}


@dataclass(frozen=True)
class ManualExperimentConfig:
    payload: dict[str, Any]
    source_path: Path
    project_root: Path

    @property
    def experiment_id(self) -> str:
        return str(self.payload["experiment"]["id"])

    @property
    def manifest_path(self) -> Path:
        return Path(self.payload["contract"]["manifest_path"])

    @property
    def manifest_key(self) -> str:
        return str(self.payload["contract"]["manifest_key"])

    @property
    def processed_dir(self) -> Path:
        return Path(self.payload["dataset"]["processed_dir"])

    @property
    def dataset_version(self) -> str:
        return str(self.payload["dataset"]["version"])

    @property
    def sample_submission_path(self) -> Path:
        return Path(self.payload["dataset"]["sample_submission_path"])

    @property
    def dropped_features(self) -> list[str]:
        return list(self.payload["features"]["drop"])

    @property
    def categorical_features(self) -> list[str]:
        return list(self.payload["features"]["categorical"])

    @property
    def model_parameters(self) -> dict[str, Any]:
        return dict(self.payload["lightgbm"]["parameters"])

    @property
    def artifact_root(self) -> Path:
        return Path(self.payload["artifacts"]["root"])

    @property
    def reference_submission_path(self) -> Path | None:
        value = self.payload["parity"]["reference_submission_path"]
        return None if value is None else Path(value)

    def resolved_payload(self) -> dict[str, Any]:
        return deepcopy(self.payload)


def load_manual_experiment_config(
    path: Path,
    *,
    project_root: Path,
) -> ManualExperimentConfig:
    """Load the manual experiment config with recursive unknown-key checks."""
    source_path = path.resolve()
    root_path = project_root.resolve()
    with source_path.open("r", encoding="utf-8") as file:
        raw = _mapping(yaml.safe_load(file), "root")

    _expect_keys(raw, SCHEMA["root"], "root")
    sections = {
        name: _section(raw, name)
        for name in (
            "experiment",
            "contract",
            "dataset",
            "features",
            "outer_cv",
            "target_encoder",
            "lightgbm",
            "thresholds",
            "prediction",
            "artifacts",
            "parity",
        )
    }
    for name, section in sections.items():
        _expect_keys(section, SCHEMA[name], name)

    grid = _mapping(
        sections["thresholds"]["legacy_diagnostic_grid"],
        "thresholds.legacy_diagnostic_grid",
    )
    _expect_keys(grid, SCHEMA["thresholds.legacy_diagnostic_grid"], "threshold grid")
    secondary = _mapping(sections["parity"]["secondary"], "parity.secondary")
    _expect_keys(secondary, SCHEMA["parity.secondary"], "parity.secondary")
    parameters = _mapping(
        sections["lightgbm"]["parameters"],
        "lightgbm.parameters",
    )
    _expect_keys(parameters, LIGHTGBM_PARAMETER_KEYS, "lightgbm.parameters")

    payload = deepcopy(dict(raw))
    payload["contract"]["manifest_path"] = str(
        _resolve_path(
            payload["contract"]["manifest_path"],
            "contract.manifest_path",
            root_path,
        )
    )
    payload["dataset"]["processed_dir"] = str(
        _resolve_path(
            payload["dataset"]["processed_dir"],
            "dataset.processed_dir",
            root_path,
        )
    )
    payload["dataset"]["sample_submission_path"] = str(
        _resolve_path(
            payload["dataset"]["sample_submission_path"],
            "dataset.sample_submission_path",
            root_path,
        )
    )
    payload["artifacts"]["root"] = str(
        _resolve_path(
            payload["artifacts"]["root"],
            "artifacts.root",
            root_path,
        )
    )
    reference_path = payload["parity"]["reference_submission_path"]
    if reference_path is not None:
        payload["parity"]["reference_submission_path"] = str(
            _resolve_path(
                reference_path,
                "parity.reference_submission_path",
                root_path,
            )
        )

    config = ManualExperimentConfig(
        payload=payload,
        source_path=source_path,
        project_root=root_path,
    )
    _validate_types_and_invariants(config)
    return config


def validate_config_against_manifest(
    config: ManualExperimentConfig,
) -> dict[str, Any]:
    """Compare duplicated values with the authoritative read-only manifest."""
    with config.manifest_path.open("r", encoding="utf-8") as file:
        manifest = _mapping(yaml.safe_load(file), "manifest")
    if config.manifest_key not in manifest:
        raise ConfigurationError(f"Manifest key not found: {config.manifest_key}")

    contract = _mapping(manifest[config.manifest_key], "manifest contract")
    protocols = _mapping(
        manifest["legacy_validation_protocols"],
        "legacy validation protocols",
    )
    protocol_name = _string(contract["validation_protocol"], "validation protocol")
    if protocol_name not in protocols:
        raise ConfigurationError(f"Validation protocol not found: {protocol_name}")
    protocol = _mapping(protocols[protocol_name], "validation protocol")
    feature = _mapping(contract["feature_contract"], "feature contract")
    encoder = _mapping(contract["target_encoding_contract"], "encoder contract")
    model = _mapping(contract["lightgbm_parameters"], "model contract")
    prediction = _mapping(contract["prediction_policy"], "prediction contract")
    primary = _mapping(
        contract["primary_parity_requirements"],
        "primary parity contract",
    )
    secondary = _mapping(
        contract["secondary_diagnostic_requirements"],
        "secondary parity contract",
    )
    p = config.payload

    comparisons: list[tuple[str, Any, Any]] = [
        ("experiment id", p["experiment"]["id"], contract["experiment_id"]),
        ("dataset version", p["dataset"]["version"], contract["dataset_version"]),
        (
            "source feature count",
            p["dataset"]["expected_source_feature_count"],
            feature["source_dataset_feature_count"],
        ),
        ("dropped features", p["features"]["drop"], feature["dropped_features"]),
        (
            "model feature count",
            p["features"]["expected_model_feature_count"],
            feature["final_model_input_feature_count"],
        ),
        (
            "categorical count",
            p["features"]["expected_categorical_count"],
            feature["categorical_feature_count"],
        ),
        (
            "categorical features",
            p["features"]["categorical"],
            feature["categorical_features"],
        ),
        (
            "encoder implementation",
            p["target_encoder"]["implementation"],
            encoder["implementation"],
        ),
        ("inner folds", p["target_encoder"]["inner_splits"], encoder["inner_folds"]),
        ("inner shuffle", p["target_encoder"]["shuffle"], encoder["shuffle"]),
        (
            "inner random state",
            p["target_encoder"]["random_state"],
            encoder["random_state"],
        ),
        ("encoder alpha", p["target_encoder"]["alpha"], encoder["alpha"]),
        ("encoder prior", p["target_encoder"]["prior"], encoder["prior_definition"]),
        (
            "keep categorical features",
            p["target_encoder"]["keep_original_categorical_features"],
            encoder["keep_original_categorical_features"],
        ),
        ("outer folds", p["outer_cv"]["n_splits"], prediction["outer_fold_model_count"]),
        ("outer splitter", p["outer_cv"]["splitter"], str(protocol["outer_splitter"]).replace("StratifiedKFold", "stratified_kfold")),
        ("outer shuffle", p["outer_cv"]["shuffle"], protocol["outer_shuffle"]),
        ("outer random state", p["outer_cv"]["random_state"], protocol["outer_seed"]),
        (
            "test aggregation",
            p["prediction"]["test_aggregation"],
            "arithmetic_mean",
        ),
        (
            "full-data refit",
            p["prediction"]["full_data_refit"],
            prediction["full_data_refit"],
        ),
        (
            "submission threshold",
            p["thresholds"]["submission"],
            prediction["submission_threshold"],
        ),
        (
            "expected balanced accuracy",
            p["parity"]["expected_balanced_accuracy"],
            primary["balanced_accuracy_at_0_117"]["expected"],
        ),
        (
            "balanced accuracy tolerance",
            p["parity"]["balanced_accuracy_tolerance"],
            primary["balanced_accuracy_at_0_117"]["absolute_tolerance"],
        ),
        (
            "expected positive count",
            p["parity"]["expected_positive_count"],
            primary["positive_prediction_count_at_0_117"]["expected"],
        ),
        (
            "expected global balanced accuracy",
            p["parity"]["secondary"]["global_optimized_oof_balanced_accuracy"],
            secondary["global_optimized_oof_balanced_accuracy"]["expected_approximately"],
        ),
        (
            "expected global threshold",
            p["parity"]["secondary"]["global_optimized_threshold"],
            secondary["global_optimized_threshold"]["expected_approximately"],
        ),
        (
            "expected legacy cross-fitted balanced accuracy",
            p["parity"]["secondary"]["legacy_cross_fitted_balanced_accuracy"],
            secondary["legacy_cross_fitted_balanced_accuracy"]["expected_approximately"],
        ),
    ]
    for name, expected in model.items():
        config_name = "random_state" if name == "seed" else name
        comparisons.append(
            (
                f"LightGBM {config_name}",
                p["lightgbm"]["parameters"][config_name],
                expected,
            )
        )

    mismatches = [
        f"{name}: config={actual!r}, manifest={expected!r}"
        for name, actual, expected in comparisons
        if actual != expected
    ]
    if mismatches:
        raise ConfigurationError(
            "Configuration does not match the baseline manifest:\n- "
            + "\n- ".join(mismatches)
        )
    return dict(contract)


def _validate_types_and_invariants(config: ManualExperimentConfig) -> None:
    p = config.payload
    _integer(p["schema_version"], "schema_version")
    for path in (
        ("experiment", "id"),
        ("experiment", "implementation"),
        ("contract", "manifest_key"),
        ("dataset", "version"),
        ("outer_cv", "splitter"),
        ("target_encoder", "implementation"),
        ("target_encoder", "prior"),
        ("lightgbm", "estimator"),
        ("prediction", "test_aggregation"),
    ):
        _string(p[path[0]][path[1]], ".".join(path))
    for name in (
        "expected_train_rows",
        "expected_test_rows",
        "expected_source_feature_count",
    ):
        value = _integer(p["dataset"][name], f"dataset.{name}")
        if value <= 0:
            raise ConfigurationError(f"dataset.{name} must be positive.")
    for name in ("expected_model_feature_count", "expected_categorical_count"):
        value = _integer(p["features"][name], f"features.{name}")
        if value <= 0:
            raise ConfigurationError(f"features.{name} must be positive.")
    _strings(p["features"]["drop"], "features.drop")
    _strings(p["features"]["categorical"], "features.categorical")
    for name in (
        "expected_source_schema_sha256",
        "expected_model_input_schema_sha256",
        "expected_transformed_schema_sha256",
    ):
        digest = _string(p["features"][name], f"features.{name}")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ConfigurationError(f"features.{name} must be a lowercase SHA256 digest.")

    for section, fields in (
        ("outer_cv", ("n_splits", "random_state")),
        ("target_encoder", ("inner_splits", "random_state")),
    ):
        for field in fields:
            _integer(p[section][field], f"{section}.{field}")
    _boolean(p["outer_cv"]["shuffle"], "outer_cv.shuffle")
    _boolean(p["target_encoder"]["shuffle"], "target_encoder.shuffle")
    _number(p["target_encoder"]["alpha"], "target_encoder.alpha")
    _boolean(
        p["target_encoder"]["keep_original_categorical_features"],
        "target_encoder.keep_original_categorical_features",
    )

    for name, value in p["lightgbm"]["parameters"].items():
        qualified = f"lightgbm.parameters.{name}"
        if name == "objective":
            _string(value, qualified)
        elif name == "extra_trees":
            _boolean(value, qualified)
        elif name in INTEGER_LIGHTGBM_PARAMETERS:
            _integer(value, qualified)
        else:
            _number(value, qualified)

    submission_threshold = _number(
        p["thresholds"]["submission"],
        "thresholds.submission",
    )
    grid = p["thresholds"]["legacy_diagnostic_grid"]
    minimum = _number(grid["minimum"], "thresholds.legacy_diagnostic_grid.minimum")
    maximum = _number(grid["maximum"], "thresholds.legacy_diagnostic_grid.maximum")
    step = _number(grid["step"], "thresholds.legacy_diagnostic_grid.step")
    _boolean(p["prediction"]["full_data_refit"], "prediction.full_data_refit")
    for name in ("save_fold_models", "save_encoder_states"):
        _boolean(p["artifacts"][name], f"artifacts.{name}")
    for name in ("enabled", "required"):
        _boolean(p["parity"][name], f"parity.{name}")
    for name in ("expected_balanced_accuracy", "balanced_accuracy_tolerance"):
        _number(p["parity"][name], f"parity.{name}")
    _integer(p["parity"]["expected_positive_count"], "parity.expected_positive_count")
    for name, value in p["parity"]["secondary"].items():
        _number(value, f"parity.secondary.{name}")

    if p["schema_version"] != 1:
        raise ConfigurationError("schema_version must be 1.")
    expected_outer = {
        "splitter": "stratified_kfold",
        "n_splits": 8,
        "shuffle": True,
        "random_state": 0,
    }
    if p["outer_cv"] != expected_outer:
        raise ConfigurationError(
            f"outer_cv must exactly match the historical contract: {expected_outer}."
        )
    expected_encoder = {
        "implementation": "custom_autogluon_compatible_binary_oof_target_encoder",
        "inner_splits": 5,
        "shuffle": True,
        "random_state": 42,
        "alpha": 10.0,
        "prior": "unweighted_mean_of_category_level_target_means",
        "keep_original_categorical_features": False,
    }
    if p["target_encoder"] != expected_encoder:
        raise ConfigurationError(
            "target_encoder must exactly match the verified implementation contract."
        )
    if p["outer_cv"]["n_splits"] <= 0 or p["target_encoder"]["inner_splits"] <= 0:
        raise ConfigurationError("Outer and inner fold counts must be positive.")
    if p["target_encoder"]["alpha"] < 0:
        raise ConfigurationError("target_encoder.alpha must be non-negative.")

    if p["features"]["expected_categorical_count"] != len(
        p["features"]["categorical"]
    ):
        raise ConfigurationError("Categorical count does not match its list.")
    if p["features"]["expected_model_feature_count"] != (
        p["dataset"]["expected_source_feature_count"] - len(p["features"]["drop"])
    ):
        raise ConfigurationError(
            "Model feature count must equal source count minus dropped features."
        )
    if len(set(p["features"]["drop"])) != len(p["features"]["drop"]):
        raise ConfigurationError("Dropped features contain duplicates.")
    if len(set(p["features"]["categorical"])) != len(p["features"]["categorical"]):
        raise ConfigurationError("Categorical features contain duplicates.")
    if set(p["features"]["drop"]) & set(p["features"]["categorical"]):
        raise ConfigurationError("Dropped and categorical feature lists overlap.")

    if p["lightgbm"]["estimator"] != "LGBMClassifier":
        raise ConfigurationError("The estimator must be LGBMClassifier.")
    if p["lightgbm"]["parameters"] != EXPECTED_LIGHTGBM_PARAMETERS:
        raise ConfigurationError(
            "LightGBM parameters must exactly match the verified baseline contract."
        )
    if not 0.0 <= submission_threshold <= 1.0:
        raise ConfigurationError("thresholds.submission must be within [0, 1].")
    if not 0.0 <= minimum <= maximum <= 1.0 or step <= 0:
        raise ConfigurationError(
            "The diagnostic threshold grid requires 0 <= minimum <= maximum <= 1 and step > 0."
        )
    expected_grid = {"minimum": 0.02, "maximum": 0.30, "step": 0.001}
    if submission_threshold != 0.117 or grid != expected_grid:
        raise ConfigurationError(
            "Threshold settings must exactly match the verified baseline contract."
        )
    if p["prediction"]["test_aggregation"] != "arithmetic_mean":
        raise ConfigurationError("Test aggregation must be arithmetic_mean.")
    if p["prediction"]["full_data_refit"]:
        raise ConfigurationError("This experiment must not refit on full data.")
    if not p["artifacts"]["save_fold_models"] or not p["artifacts"]["save_encoder_states"]:
        raise ConfigurationError(
            "Baseline runs must save every fold model and encoder state."
        )

    parity = p["parity"]
    if parity["required"] and not parity["enabled"]:
        raise ConfigurationError("Required parity cannot be disabled.")
    if parity["required"] and parity["reference_submission_path"] is None:
        raise ConfigurationError("Required parity needs a reference submission.")
    if not 0.0 <= parity["expected_balanced_accuracy"] <= 1.0:
        raise ConfigurationError("Expected balanced accuracy must be within [0, 1].")
    if parity["balanced_accuracy_tolerance"] < 0:
        raise ConfigurationError("Balanced accuracy tolerance must be non-negative.")
    if parity["expected_positive_count"] < 0:
        raise ConfigurationError("Expected positive count must be non-negative.")

def _section(root: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if name not in root:
        raise ConfigurationError(f"Missing required section: {name}")
    return _mapping(root[name], name)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a mapping.")
    return value


def _expect_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ConfigurationError(
            f"Invalid keys in {name}: missing={missing}, unknown={unknown}"
        )


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{name} must be a non-empty string.")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{name} must be an integer.")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be numeric.")
    return float(value)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a boolean.")
    return value


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigurationError(f"{name} must be a list.")
    return tuple(_string(item, f"{name}[]") for item in value)


def _resolve_path(value: Any, name: str, root: Path) -> Path:
    path = Path(_string(value, name))
    return (root / path).resolve() if not path.is_absolute() else path.resolve()
