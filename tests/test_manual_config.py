from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from src.churn_ml.config import (
    ConfigurationError,
    load_manual_experiment_config,
    validate_config_against_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs/experiments/manual_lightgbmprep_r31.yaml"
)


def _baseline_payload() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _write_config(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _set_nested(
    payload: dict[str, Any],
    path: tuple[str, ...],
    value: Any,
) -> None:
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _load_and_validate(path: Path) -> None:
    config = load_manual_experiment_config(path, project_root=PROJECT_ROOT)
    validate_config_against_manifest(config)


def test_baseline_config_matches_read_only_manifest() -> None:
    config = load_manual_experiment_config(
        CONFIG_PATH,
        project_root=PROJECT_ROOT,
    )
    contract = validate_config_against_manifest(config)

    assert config.model_parameters["random_state"] == 0
    assert config.model_parameters["n_estimators"] == 376
    assert contract["lightgbm_parameters"]["seed"] == 0
    assert config.reference_submission_path is not None


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("outer_cv", "random_state"), 1),
        (("outer_cv", "n_splits"), 7),
        (("outer_cv", "shuffle"), False),
        (("outer_cv", "n_splits"), 0),
        (("target_encoder", "implementation"), "different_encoder"),
        (("target_encoder", "inner_splits"), 4),
        (("target_encoder", "inner_splits"), 0),
        (("target_encoder", "shuffle"), False),
        (("target_encoder", "random_state"), 0),
        (("target_encoder", "alpha"), 1.0),
        (("target_encoder", "alpha"), -1.0),
        (("target_encoder", "prior"), "weighted_target_mean"),
        (("target_encoder", "keep_original_categorical_features"), True),
        (("lightgbm", "parameters", "objective"), "regression"),
        (("lightgbm", "parameters", "random_state"), 1),
        (("lightgbm", "parameters", "n_jobs"), 1),
        (("lightgbm", "parameters", "verbosity"), 0),
        (("prediction", "full_data_refit"), True),
        (("thresholds", "submission"), 0.5),
        (("thresholds", "submission"), -0.1),
        (("thresholds", "submission"), 1.1),
        (("thresholds", "legacy_diagnostic_grid", "minimum"), 0.0),
        (("thresholds", "legacy_diagnostic_grid", "minimum"), -0.1),
        (("thresholds", "legacy_diagnostic_grid", "minimum"), 0.31),
        (("thresholds", "legacy_diagnostic_grid", "maximum"), 0.31),
        (("thresholds", "legacy_diagnostic_grid", "maximum"), 1.1),
        (("thresholds", "legacy_diagnostic_grid", "step"), 0.0),
        (("thresholds", "legacy_diagnostic_grid", "step"), 0.002),
        (("artifacts", "save_fold_models"), False),
        (("artifacts", "save_encoder_states"), False),
        (("features", "expected_model_feature_count"), 212),
        (("features", "expected_categorical_count"), 30),
        (("parity", "reference_submission_path"), None),
    ],
)
def test_baseline_contract_mutations_are_rejected(
    tmp_path: Path,
    path: tuple[str, ...],
    value: Any,
) -> None:
    payload = _baseline_payload()
    _set_nested(payload, path, value)

    with pytest.raises(ConfigurationError):
        _load_and_validate(_write_config(tmp_path, payload))


@pytest.mark.parametrize("unknown_key", ["early_stopping_rounds", "callbacks"])
def test_early_stopping_and_callback_keys_are_rejected(
    tmp_path: Path,
    unknown_key: str,
) -> None:
    payload = _baseline_payload()
    payload["lightgbm"]["parameters"][unknown_key] = 10

    with pytest.raises(ConfigurationError, match=unknown_key):
        _load_and_validate(_write_config(tmp_path, payload))


def test_unknown_nested_config_key_fails(tmp_path: Path) -> None:
    payload = _baseline_payload()
    payload["target_encoder"]["unexpected"] = "forbidden"

    with pytest.raises(ConfigurationError, match="unknown=.*unexpected"):
        _load_and_validate(_write_config(tmp_path, payload))


def test_missing_required_parity_field_fails(tmp_path: Path) -> None:
    payload = _baseline_payload()
    del payload["parity"]["expected_positive_count"]

    with pytest.raises(ConfigurationError, match="expected_positive_count"):
        _load_and_validate(_write_config(tmp_path, payload))


def test_required_parity_rejects_disabled_state(tmp_path: Path) -> None:
    payload = _baseline_payload()
    payload["parity"]["enabled"] = False

    with pytest.raises(ConfigurationError, match="cannot be disabled"):
        _load_and_validate(_write_config(tmp_path, payload))


def test_parity_can_be_disabled_without_changing_training_config(
    tmp_path: Path,
) -> None:
    payload = _baseline_payload()
    payload["parity"]["enabled"] = False
    payload["parity"]["required"] = False
    payload["parity"]["reference_submission_path"] = None
    path = _write_config(tmp_path, payload)

    config = load_manual_experiment_config(path, project_root=PROJECT_ROOT)
    validate_config_against_manifest(config)

    assert config.payload["parity"]["enabled"] is False
    assert config.reference_submission_path is None
    assert config.model_parameters["n_estimators"] == 376