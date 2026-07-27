from __future__ import annotations

import copy
import os
from pathlib import Path

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


def valid_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "profile_id": "realtabpfn_only_v1",
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
    ],
)
def test_strict_recursive_schema(
    repository: tuple[Path, Path],
    mutate: object,
    message: str,
) -> None:
    root, config_path = repository
    payload = copy.deepcopy(valid_payload())
    mutate(payload)  # type: ignore[operator]
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
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
    payload["dataset"]["directory"] = directory  # type: ignore[index]
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="traversal"):
        load_config(config_path, root)


def test_rejects_symlink_escape_where_supported(
    repository: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    root, config_path = repository
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "X_train.parquet").write_bytes(b"x")
    (outside / "y_train.parquet").write_bytes(b"y")
    link = root / "linked-data"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are not available")
    payload = valid_payload()
    payload["dataset"]["directory"] = "linked-data"  # type: ignore[index]
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="escapes"):
        load_config(config_path, root)


def test_profile_registry_and_unknown_profile() -> None:
    assert {profile.profile_id for profile in list_profiles()} == {
        "realtabpfn_only_v1",
        "catboost_only_cpu_v1",
        "lightgbmprep_only_cpu_v1",
        "extreme_seqmem_v1",
    }
    assert get_profile("realtabpfn_only_v1").included_model_types == ("REALTABPFN-V2",)
    with pytest.raises(ValueError, match="Unknown AutoGluon profile"):
        get_profile("arbitrary_python_profile")


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


def test_validate_only_does_not_allocate(
    repository: tuple[Path, Path],
) -> None:
    root, config_path = repository
    artifacts_root = root / "artifacts"
    assert (
        main(["validate", "--config", str(config_path)], repository_root=root)
        == EXIT_OK
    )
    assert not artifacts_root.exists()
