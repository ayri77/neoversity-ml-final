"""OOF schema validation for Dataset Campaign Runner v1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.churn_ml.dataset_campaign.constants import REQUIRED_OOF_COLUMNS
from src.churn_ml.dataset_campaign.errors import CampaignExecutionError


def validate_outer_validation_oof(path: Path) -> dict[str, Any]:
    """Validate that an authoritative outer-validation parquet exposes required keys."""
    if not path.is_file():
        raise CampaignExecutionError(f"OOF artifact missing: {path}.")
    try:
        frame = pd.read_parquet(path)
    except Exception as error:  # noqa: BLE001
        raise CampaignExecutionError(
            f"Could not read OOF parquet {path}: {error}"
        ) from error
    missing = [column for column in REQUIRED_OOF_COLUMNS if column not in frame.columns]
    if missing:
        raise CampaignExecutionError(
            f"OOF schema missing required columns {missing} in {path}."
        )
    if frame.empty:
        raise CampaignExecutionError(f"OOF artifact is empty: {path}.")
    for column in ("repeat", "outer_fold", "row_position"):
        if frame[column].isna().any():
            raise CampaignExecutionError(
                f"OOF column {column} contains nulls in {path}."
            )
    # Research v2 authoritative convention: repeat and outer_fold are 1-based.
    # Do not reinterpret or rewrite these to 0-based indices.
    for column in ("repeat", "outer_fold"):
        values = frame[column]
        if (values < 1).any():
            raise CampaignExecutionError(
                f"OOF column {column} must be 1-based (found value < 1) in {path}."
            )
    return {
        "path": str(path),
        "row_count": int(len(frame)),
        "columns": list(frame.columns),
        "required_columns": list(REQUIRED_OOF_COLUMNS),
        "repeat_min": int(frame["repeat"].min()),
        "outer_fold_min": int(frame["outer_fold"].min()),
    }


def resolve_oof_path(run_directory: Path) -> Path:
    return run_directory / "predictions" / "outer_validation.parquet"
