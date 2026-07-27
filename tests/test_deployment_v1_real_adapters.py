from __future__ import annotations

import importlib.metadata
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.churn_ml.deployment_v1_contracts import CandidateApproval, model_parameters
from src.churn_ml.deployment_v1_features import build_full_data_encoding
from src.churn_ml.deployment_v1_models import fit_component_bags
from src.churn_ml.experiment_v2_adapter import EXPECTED_ADAPTER_CONTRACT
from src.churn_ml.experiment_v2_catboost_adapter import (
    EXPECTED_CATBOOST_NUMERIC_V1_CONTRACT,
)
from src.churn_ml.experiment_v2_model_registry import get_candidate_adapter
from src.churn_ml.experiment_v2_xgboost_adapter import (
    EXPECTED_XGBOOST_NUMERIC_V1_CONTRACT,
)


@pytest.mark.parametrize(
    ("adapter_id", "contract"),
    [
        ("manual_lightgbm_te_v1_compat", EXPECTED_ADAPTER_CONTRACT),
        ("xgboost_numeric_v1", EXPECTED_XGBOOST_NUMERIC_V1_CONTRACT),
        ("catboost_numeric_v1", EXPECTED_CATBOOST_NUMERIC_V1_CONTRACT),
    ],
)
def test_tiny_real_full_data_deployment_adapter(
    tmp_path: Path,
    adapter_id: str,
    contract: dict[str, Any],
) -> None:
    resolved_contract = deepcopy(contract)
    if adapter_id == "xgboost_numeric_v1":
        resolved_contract["xgboost"]["parameters"]["n_estimators"] = 2
    if adapter_id == "catboost_numeric_v1":
        resolved_contract["catboost"]["parameters"]["iterations"] = 2
    get_candidate_adapter(adapter_id).validate_contract(resolved_contract)

    X_train = pd.DataFrame(
        {
            "index": np.arange(20, dtype=np.int64),
            "number": [np.nan, 1.0, 2.0, 3.0, 4.0] * 4,
            "category": ["a", "b", "c", None, "a"] * 4,
        }
    )
    y_train = pd.Series([0, 1] * 10, name="y", dtype="int8")
    X_test = pd.DataFrame(
        {
            "index": [100, 101, 102],
            "number": [np.nan, 5.0, 6.0],
            "category": ["a", "unknown", None],
        }
    )
    encoder_contract = (
        resolved_contract["target_encoder"]
        if adapter_id == "manual_lightgbm_te_v1_compat"
        else resolved_contract["numeric_features"]
    )
    encoding = build_full_data_encoding(
        X_train,
        y_train,
        X_test,
        encoder_contract,
    )
    approval = _approval(tmp_path, adapter_id, resolved_contract)
    component = {
        "component_id": f"{adapter_id}-component",
        "adapter_id": adapter_id,
        "fixed_parameters": model_parameters(adapter_id, resolved_contract),
        "bag_seeds": [7],
        "component_weight": 1.0,
    }
    result = fit_component_bags(component, approval, encoding, y_train)
    assert result.probabilities.shape == (3,)
    assert np.isfinite(result.probabilities).all()
    assert ((result.probabilities >= 0.0) & (result.probabilities <= 1.0)).all()
    assert result.summary["training_rows_per_bag"] == 20
    assert result.summary["all_training_rows_used"] is True
    assert result.summary["model_persistence"] is False
    assert not list(tmp_path.rglob("*.cbm"))
    assert not list(tmp_path.rglob("*.model"))
    assert not list(tmp_path.rglob("*.joblib"))


def _approval(
    root: Path,
    adapter_id: str,
    contract: dict[str, Any],
) -> CandidateApproval:
    run_root = root / adapter_id
    identity_dir = run_root / "identities"
    identity_dir.mkdir(parents=True)
    distribution = {
        "manual_lightgbm_te_v1_compat": "lightgbm",
        "xgboost_numeric_v1": "xgboost",
        "catboost_numeric_v1": "catboost",
    }[adapter_id]
    (identity_dir / "candidate_adapter.json").write_text(
        json.dumps(
            {
                "canonical": {
                    "runtime_dependencies": {
                        distribution: importlib.metadata.version(distribution)
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    source = root / f"{adapter_id}.yaml"
    source.write_text("schema_version: 1\n", encoding="utf-8")
    payload = {
        "approval_id": f"{adapter_id}-approval",
        "component_id": f"{adapter_id}-component",
        "research_run": {"adapter_id": adapter_id},
    }
    run = SimpleNamespace(
        root=run_root,
        config=SimpleNamespace(
            adapter_id=adapter_id,
            adapter_contract=resolved_copy(contract),
        ),
    )
    return CandidateApproval(
        payload=payload,
        source_path=source,
        source_sha256="0" * 64,
        research_run=run,
    )


def resolved_copy(value: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(value)
