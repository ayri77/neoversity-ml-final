from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.churn_ml.mlflow_config import MLflowConfigError, load_mlflow_config


def valid_payload() -> dict:
    return {
        "schema_version": 1,
        "tracking": {
            "backend_store_uri": "sqlite:///artifacts/mlflow/mlflow.db",
            "artifact_root": "artifacts/mlflow/mlartifacts",
        },
        "experiments": {
            "research_v2": "research",
            "autogluon": "autogluon",
        },
        "sources": {
            "research_v2_root": "artifacts/research_v2",
            "autogluon_root": "artifacts/autogluon_runs",
        },
        "sync": {
            "log_small_artifacts": True,
            "max_artifact_size_bytes": 1024,
        },
    }


def write_config(root: Path, payload: dict) -> Path:
    path = root / "local.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda payload: payload.update({"unknown": True}), "unknown keys"),
        (lambda payload: payload["sync"].pop("log_small_artifacts"), "missing keys"),
        (
            lambda payload: payload["sync"].update({"max_artifact_size_bytes": True}),
            "exact type int",
        ),
        (
            lambda payload: payload.update({"schema_version": True}),
            "exact type int",
        ),
    ],
)
def test_config_rejects_unknown_missing_and_bool_as_int(
    tmp_path: Path,
    mutator,
    message: str,
) -> None:
    payload = valid_payload()
    mutator(payload)
    with pytest.raises(MLflowConfigError, match=message):
        load_mlflow_config(write_config(tmp_path, payload), repository_root=tmp_path)


@pytest.mark.parametrize(
    "bad_path",
    [
        "/tmp/mlflow.db",
        "C:/tmp/mlflow.db",
        "C:tmp/mlflow.db",
        r"\\server\share\mlflow.db",
        r"\rooted\mlflow.db",
        "../mlflow.db",
        "artifacts/../mlflow.db",
        r"artifacts\mlflow\mlflow.db",
    ],
)
def test_config_rejects_posix_windows_and_traversal_paths(
    tmp_path: Path,
    bad_path: str,
) -> None:
    payload = valid_payload()
    payload["tracking"]["backend_store_uri"] = f"sqlite:///{bad_path}"
    with pytest.raises(MLflowConfigError):
        load_mlflow_config(write_config(tmp_path, payload), repository_root=tmp_path)


def test_config_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    try:
        (tmp_path / "artifacts").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Symlink creation is unavailable on this Windows host: {error}")
    with pytest.raises(MLflowConfigError, match="ignored artifacts area"):
        load_mlflow_config(
            write_config(tmp_path, valid_payload()), repository_root=tmp_path
        )


def test_validate_allocates_no_storage(tmp_path: Path) -> None:
    payload = deepcopy(valid_payload())
    config = load_mlflow_config(
        write_config(tmp_path, payload), repository_root=tmp_path
    )
    assert config.paths.backend_store.exists() is False
    assert config.paths.artifact_root.exists() is False
    assert config.paths.receipts_root.exists() is False
