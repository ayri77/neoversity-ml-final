from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from src.churn_ml.autogluon_config import load_config
from src.churn_ml.autogluon_inspection import predictor_report
from src.churn_ml.autogluon_worker import (
    build_fit_kwargs,
    load_training_data,
    validate_effective_seed_report,
)
from tests.test_autogluon_config import valid_payload
from tests.test_autogluon_inspection import FakePredictor, realistic_autogluon_info


def config_without_data(tmp_path: Path) -> Any:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(valid_payload()), encoding="utf-8")
    return load_config(config_path, tmp_path, require_data_files=False)


def test_production_data_loader_reads_only_validated_train_pair(
    tmp_path: Path,
) -> None:
    config = config_without_data(tmp_path)
    reads: list[Path] = []

    def reader(path: Path) -> pd.DataFrame:
        reads.append(path)
        if path == config.paths.train_features:
            return pd.DataFrame({"feature_a": [1.0, 2.0]})
        if path == config.paths.train_target:
            return pd.DataFrame({"target": [0, 1]})
        raise AssertionError(f"unexpected data read: {path}")

    features, target, train_data = load_training_data(config, reader)
    assert reads == [config.paths.train_features, config.paths.train_target]
    assert list(features.columns) == ["feature_a"]
    assert list(target.columns) == ["target"]
    assert list(train_data.columns) == ["feature_a", "target"]
    assert train_data["target"].tolist() == [0, 1]


def test_fit_kwargs_preserve_seeded_family_config_without_global_gpu(
    tmp_path: Path,
) -> None:
    config = config_without_data(tmp_path)
    hyperparameters = {
        "GBM": [
            {
                "ag_args_fit": {"num_gpus": 0},
                "ag_args_ensemble": {"model_random_seed": config.seed},
            }
        ]
    }
    kwargs = build_fit_kwargs(config, hyperparameters, object())
    assert "num_gpus" not in kwargs
    assert kwargs["hyperparameters"] is hyperparameters
    assert (
        kwargs["hyperparameters"]["GBM"][0]["ag_args_ensemble"]["model_random_seed"]
        == 42
    )


def test_worker_accepts_verified_base_seed_with_independent_auxiliary_seed() -> None:
    report = predictor_report(
        FakePredictor(realistic_autogluon_info([42], auxiliary_seed=0)),
        requested_seed=42,
        configured_families=["REALTABPFN-V2"],
    )
    validate_effective_seed_report(report, requested_seed=42)
    assert report["effective_seed_status"] == "verified"
    assert report["effective_seed"] == 42
    assert report["auxiliary_effective_seeds_observed"] == [0]


def test_worker_rejects_true_configured_base_seed_mismatch() -> None:
    report = predictor_report(
        FakePredictor(realistic_autogluon_info([41], auxiliary_seed=0)),
        requested_seed=42,
        configured_families=["REALTABPFN-V2"],
    )
    with pytest.raises(RuntimeError, match="configured base-model seed"):
        validate_effective_seed_report(report, requested_seed=42)


def test_worker_accepts_unavailable_base_seed_evidence() -> None:
    report = predictor_report(
        FakePredictor(realistic_autogluon_info([None], auxiliary_seed=0)),
        requested_seed=42,
        configured_families=["REALTABPFN-V2"],
    )
    validate_effective_seed_report(report, requested_seed=42)
    assert report["effective_seed_status"] == "unavailable"
    assert report["effective_seed"] is None
