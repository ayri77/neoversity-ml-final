from __future__ import annotations

from pathlib import Path

import pytest

from src.churn_ml.research_data import ResearchDataError, load_research_training_data
from tests.research_test_support import _build_test_config


@pytest.mark.parametrize(
    "unsafe_name",
    ["../X_train.parquet", "nested/X_train.parquet", r"nested\X_train.parquet"],
)
def test_data_loader_requires_resolved_file_parent_to_equal_dataset_directory(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    config = _build_test_config(tmp_path)
    config.plan_payload["dataset"]["files"]["train_features"]["name"] = unsafe_name

    with pytest.raises(ResearchDataError, match="directly beneath"):
        load_research_training_data(config)


def test_data_loader_accepts_plain_basename(tmp_path: Path) -> None:
    config = _build_test_config(tmp_path)

    data = load_research_training_data(config)

    assert len(data.X) == 8
