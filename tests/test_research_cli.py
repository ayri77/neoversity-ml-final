from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest

from src.churn_ml import research_cli as cli


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs/research/manual_lightgbmprep_r31_nested_rs5x5.yaml"
)
TRAIN_PATH = PROJECT_ROOT / "data/processed/v3_targeted_missingness/X_train.parquet"

pytestmark = pytest.mark.skipif(
    not TRAIN_PATH.exists(),
    reason="The ignored local v3 training dataset is unavailable.",
)


def test_validate_only_allocates_no_run_and_reads_no_competition_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    original_read_parquet = pd.read_parquet
    original_path_open = Path.open

    def recording_read_parquet(path: object, *args: object, **kwargs: object):
        opened.append(str(path))
        return original_read_parquet(path, *args, **kwargs)

    def recording_path_open(
        path: Path,
        *args: object,
        **kwargs: object,
    ):
        opened.append(str(path))
        return original_path_open(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", recording_read_parquet)
    monkeypatch.setattr(Path, "open", recording_path_open)
    artifact_root = PROJECT_ROOT / "artifacts/research_evaluations"
    before = (
        sorted(path for path in artifact_root.rglob("*") if path.is_dir())
        if artifact_root.exists()
        else []
    )

    return_code = cli.execute(
        argparse.Namespace(
            config=CONFIG_PATH,
            validate_only=True,
            run_id=None,
        )
    )
    after = (
        sorted(path for path in artifact_root.rglob("*") if path.is_dir())
        if artifact_root.exists()
        else []
    )

    assert return_code == 0
    assert after == before
    normalized = [value.replace("\\", "/").lower() for value in opened]
    forbidden = (
        "x_test.parquet",
        "sample_submission",
        "/submissions/",
        "artifacts/reference",
        "artifacts/autogluon",
    )
    assert not any(marker in value for value in normalized for marker in forbidden)
