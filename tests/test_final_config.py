from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.churn_ml.final_config import (
    FinalConfigurationError,
    load_final_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs/final/manual_lightgbmprep_r31_submission_v1.yaml"


def _payload() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "final.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_approved_final_config_is_valid_and_utc() -> None:
    config = load_final_config(CONFIG, project_root=PROJECT_ROOT)

    assert config.promotion_id == "manual_lightgbmprep_r31_submission_v1"
    assert (
        config.payload["promotion"]["approval"]["approved_at_utc"]
        == "2026-07-25T16:49:31.1443662+00:00"
    )
    assert config.research_run.name == "final-provenance-20260725"


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda payload: payload["threshold"].__setitem__("unknown", True),
            "threshold keys differ",
        ),
        (
            lambda payload: payload["promotion"]["approval"].__setitem__(
                "approved_at_utc", "2026-07-25T16:49:31"
            ),
            "timezone-aware UTC",
        ),
        (
            lambda payload: payload["inputs"].__setitem__(
                "processed_dir", "../outside"
            ),
            "contained relative path",
        ),
        (
            lambda payload: payload["artifacts"].__setitem__(
                "submission_filename", "../submission.csv"
            ),
            "plain basename",
        ),
        (
            lambda payload: payload["artifacts"].__setitem__("overwrite", True),
            "overwrite must be false",
        ),
    ],
)
def test_strict_config_rejects_unknown_unsafe_or_mutated_values(
    tmp_path: Path,
    mutator,
    match: str,
) -> None:
    payload = deepcopy(_payload())
    mutator(payload)

    with pytest.raises(FinalConfigurationError, match=match):
        load_final_config(_write(tmp_path, payload), project_root=PROJECT_ROOT)
