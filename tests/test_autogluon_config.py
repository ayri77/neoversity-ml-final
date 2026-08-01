from __future__ import annotations

import copy
import os
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from src.churn_ml.autogluon_cli import EXIT_OK, main
from src.churn_ml.autogluon_config import ConfigError, load_config
from src.churn_ml.autogluon_profiles import get_profile, list_profiles
from src.churn_ml.autogluon_supervisor import (
    RUN_ID_MAX_LENGTH,
    generate_run_id,
    validate_run_id,
)


def valid_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "profile_id": "realtabpfn_only_v1",
        "seed": 42,
        "dataset": {
            "version": "v3_targeted_missingness",
            "directory": "data/processed/v3_targeted_missingness",
            "train_features_file": "X_train.parquet",
            "train_target_file": "y_train.parquet",
            "label": "target",
        },
        "predictor": {
            "problem_type": "binary",
            "eval_metric": "balanced_accuracy",
            "positive_class": 1,
            "verbosity": 2,
        },
        "resources": {
            "time_limit_seconds": 30,
            "num_cpus": "auto",
            "num_gpus": 1,
            "fit_strategy": "sequential",
            "fold_fitting_strategy": "sequential_local",
        },
        "fit": {
            "presets": "extreme_quality",
            "calibrate_decision_threshold": True,
        },
        "artifacts": {"root": "artifacts/autogluon_runs"},
    }


@pytest.fixture
def repository(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repository with spaces"
    data_dir = root / "data" / "processed" / "v3_targeted_missingness"
    data_dir.mkdir(parents=True)
    (data_dir / "X_train.parquet").write_bytes(b"features")
    (data_dir / "y_train.parquet").write_bytes(b"target")
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(valid_payload()), encoding="utf-8")
    return root, config_path


def write_payload(config_path: Path, payload: dict[str, Any]) -> None:
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update({"unknown": 1}), "unknown keys"),
        (lambda value: value["dataset"].pop("label"), "missing keys"),
        (
            lambda value: value["resources"].update({"time_limit_seconds": True}),
            "exact type",
        ),
        (
            lambda value: value["predictor"].update({"positive_class": 1.0}),
            "exact type",
        ),
        (
            lambda value: value["fit"].update({"calibrate_decision_threshold": 1}),
            "exact type",
        ),
        (
            lambda value: value["resources"].update({"num_cpus": "32"}),
            "must equal 'auto'",
        ),
        (lambda value: value.update({"seed": True}), "exact type"),
        (lambda value: value.update({"seed": 42.0}), "exact type"),
        (lambda value: value.update({"seed": "42"}), "exact type"),
    ],
)
def test_strict_recursive_schema(
    repository: tuple[Path, Path],
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    root, config_path = repository
    payload = copy.deepcopy(valid_payload())
    mutate(payload)
    write_payload(config_path, payload)
    with pytest.raises(ConfigError, match=message):
        load_config(config_path, root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["dataset"].update(
                {"train_features_file": "X_test.parquet"}
            ),
            "X_train.parquet",
        ),
        (
            lambda payload: payload["dataset"].update(
                {"directory": "data/raw/v3_targeted_missingness"}
            ),
            "must equal exactly",
        ),
        (
            lambda payload: payload["dataset"].update(
                {"directory": "artifacts/reference/v3_targeted_missingness"}
            ),
            "must equal exactly",
        ),
        (
            lambda payload: payload["dataset"].update(
                {"directory": "artifacts/predictions/v3_targeted_missingness"}
            ),
            "must equal exactly",
        ),
        (
            lambda payload: payload["dataset"].update(
                {"directory": "submissions/v3_targeted_missingness"}
            ),
            "must equal exactly",
        ),
        (
            lambda payload: payload["dataset"].update(
                {"train_target_file": "submission.parquet"}
            ),
            "y_train.parquet",
        ),
        (
            lambda payload: payload["dataset"].update(
                {
                    "version": "v2_missingness_indicators",
                    "directory": "data/processed/v3_targeted_missingness",
                }
            ),
            "configured dataset version",
        ),
    ],
)
def test_production_dataset_isolation_rejects_forbidden_configuration(
    repository: tuple[Path, Path],
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    root, config_path = repository
    payload = valid_payload()
    mutation(payload)
    write_payload(config_path, payload)
    with pytest.raises(ConfigError, match=message):
        load_config(config_path, root)


@pytest.mark.parametrize(
    "configured_path",
    (
        r"C:\absolute",
        "C:/absolute",
        "C:",
        "C:relative",
        r"\\server\share",
        r"\path",
        "/platform/absolute",
    ),
)
def test_windows_and_platform_rooted_artifact_paths_are_rejected(
    repository: tuple[Path, Path],
    configured_path: str,
) -> None:
    root, config_path = repository
    payload = valid_payload()
    payload["artifacts"]["root"] = configured_path
    write_payload(config_path, payload)
    with pytest.raises(ConfigError, match="drive-qualified or rooted"):
        load_config(config_path, root)


@pytest.mark.parametrize(
    "directory",
    ("../outside", "data/processed/../../outside"),
)
def test_rejects_path_traversal(
    repository: tuple[Path, Path],
    directory: str,
) -> None:
    root, config_path = repository
    payload = valid_payload()
    payload["dataset"]["directory"] = directory
    write_payload(config_path, payload)
    with pytest.raises(ConfigError, match="must equal exactly"):
        load_config(config_path, root)


def test_rejects_exact_version_symlink_escape_where_supported(
    repository: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    root, config_path = repository
    version_dir = root / "data" / "processed" / "v3_targeted_missingness"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "X_train.parquet").write_bytes(b"x")
    (outside / "y_train.parquet").write_bytes(b"y")
    shutil.rmtree(version_dir)
    try:
        os.symlink(outside, version_dir, target_is_directory=True)
    except OSError as error:
        pytest.skip(
            f"isolated directory symlink/junction privilege unavailable: {error}"
        )
    with pytest.raises(ConfigError, match="escapes"):
        load_config(config_path, root)


def test_preset_allowlist_rejects_garbage(repository: tuple[Path, Path]) -> None:
    root, config_path = repository
    payload = valid_payload()
    payload["fit"]["presets"] = "garbage"
    write_payload(config_path, payload)
    with pytest.raises(ConfigError, match="must be one of"):
        load_config(config_path, root)


def test_seed_is_identity_bearing(repository: tuple[Path, Path]) -> None:
    root, config_path = repository
    first = load_config(config_path, root)
    payload = valid_payload()
    payload["seed"] = 43
    write_payload(config_path, payload)
    second = load_config(config_path, root)
    assert first.seed == 42
    assert second.seed == 43
    assert first.identity_sha256 != second.identity_sha256


def test_config_only_validation_without_ignored_data(tmp_path: Path) -> None:
    root = tmp_path / "empty-repository"
    root.mkdir()
    config_path = root / "config.yaml"
    write_payload(config_path, valid_payload())
    config = load_config(config_path, root, require_data_files=False)
    assert config.dataset.version == "v3_targeted_missingness"
    assert not (root / "artifacts").exists()


def test_profile_registry_and_unknown_profile() -> None:
    assert {profile.profile_id for profile in list_profiles()} == {
        "realtabpfn_only_v1",
        "tabm_only_gpu_v1",
        "catboost_only_cpu_v1",
        "lightgbmprep_only_cpu_v1",
        "extreme_seqmem_v1",
        "focused_hybrid_v1",
    }
    assert get_profile("realtabpfn_only_v1").included_model_types == ("REALTABPFN-V2",)
    assert get_profile("tabm_only_gpu_v1").included_model_types == ("TABM",)
    with pytest.raises(ValueError, match="Unknown AutoGluon profile"):
        get_profile("arbitrary_python_profile")


def test_tabm_overnight_yaml_loads_without_training() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    config_path = (
        repository_root / "configs" / "autogluon" / "tabm_overnight_gpu_3h.yaml"
    )
    config = load_config(config_path, repository_root, require_data_files=False)
    assert config.profile_id == "tabm_only_gpu_v1"
    assert config.seed == 42
    assert config.resources.time_limit_seconds == 10800
    assert config.resources.num_gpus == 1
    assert config.resources.fit_strategy == "sequential"
    assert config.resources.fold_fitting_strategy == "sequential_local"
    assert config.dataset.version == "v3_targeted_missingness"
    assert config.fit.presets == "extreme_quality"
    assert config.fit.calibrate_decision_threshold is True


@pytest.mark.parametrize(
    "run_id",
    (
        "UPPERCASE",
        "../escape",
        "-leading",
        "trailing-",
        "has space",
        "a" * (RUN_ID_MAX_LENGTH + 1),
        "ab",
    ),
)
def test_run_id_rejects_unsafe_or_out_of_bounds(run_id: str) -> None:
    with pytest.raises(ValueError):
        validate_run_id(run_id)


def test_generated_run_id_is_safe_and_distinct() -> None:
    first = generate_run_id()
    second = generate_run_id()
    assert validate_run_id(first) == first
    assert first != second


def test_validate_only_does_not_allocate(repository: tuple[Path, Path]) -> None:
    root, config_path = repository
    artifacts_root = root / "artifacts"
    assert (
        main(["validate", "--config", str(config_path)], repository_root=root)
        == EXIT_OK
    )
    assert not artifacts_root.exists()
