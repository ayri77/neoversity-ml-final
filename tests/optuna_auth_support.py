from __future__ import annotations

import json
import math
import shutil
import struct
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.churn_ml.optuna_search_artifacts import (
    PAYLOAD_FILES,
    _inventory_records,
    _write_json,
)
from src.churn_ml.optuna_search_config import load_optuna_search_config
from src.churn_ml.optuna_search_lifecycle import (
    portable_dataset_identity,
    run_optuna_study,
)
from src.churn_ml.optuna_search_objective import build_search_assignments
from src.churn_ml.research_data import canonical_sha256
from src.churn_ml.research_v2_data import load_research_v2_training_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_TRAIN_DIR = Path(
    r"C:\Users\pbori\Documents\Courses\Neoversity\ML Foundations"
    r"\neoversity-ml-final\data\processed\v3_targeted_missingness"
)
LOCAL_TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "v3_targeted_missingness"
TRAIN_ONLY_FILES = ("X_train.parquet", "y_train.parquet", "metadata.json")


class DeterministicAdapter:
    id = "xgboost_numeric_v1"

    def __init__(self, *, fail_first: bool = False) -> None:
        self._fail_first = fail_first
        self._failed = False

    def validate_contract(self, contract: Any) -> None:
        assert isinstance(contract, dict)

    def identity_inputs(self, contract: Any) -> dict[str, Any]:
        del contract
        return {
            "id": self.id,
            "probability_semantics": "binary_positive_class_label_1",
            "early_stopping": "disabled",
        }

    def fit_predict(
        self,
        train_features: pd.DataFrame,
        train_labels: pd.Series,
        prediction_features: pd.DataFrame,
        contract: Any,
        *,
        model_training_positions: np.ndarray,
        prediction_positions: np.ndarray,
        audit_callback: Any = None,
    ) -> np.ndarray:
        del train_features, train_labels, prediction_features, contract
        if self._fail_first and not self._failed:
            self._failed = True
            raise RuntimeError("intentional first-trial failure")
        assert set(model_training_positions).isdisjoint(prediction_positions)
        if audit_callback is not None:
            audit_callback(model_training_positions, prediction_positions)
        return ((prediction_positions % 13) + 1).astype(float) / 14.0


def ensure_train_only_files() -> Path:
    LOCAL_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    for name in TRAIN_ONLY_FILES:
        target = LOCAL_TRAIN_DIR / name
        if target.is_file():
            continue
        source = MAIN_TRAIN_DIR / name
        if not source.is_file():
            raise FileNotFoundError(
                f"Missing train-only file {name}; expected at {source}."
            )
        shutil.copy2(source, target)
    return LOCAL_TRAIN_DIR


def load_authorized_training_bundle(base_config: str):
    ensure_train_only_files()
    config = load_optuna_search_config(
        PROJECT_ROOT / "configs/optuna/xgboost_numeric_v1_smoke.yaml",
        project_root=PROJECT_ROOT,
    )
    # Re-load using the caller-selected base through a temporary search plan when needed.
    del config
    from src.churn_ml.research_v2_config import load_research_v2_config

    research = load_research_v2_config(
        PROJECT_ROOT / base_config,
        project_root=PROJECT_ROOT,
    )
    data = load_research_v2_training_data(research)
    identity = portable_dataset_identity(
        data.fingerprints,
        project_root=PROJECT_ROOT,
    )
    return data, identity


def shrink_space(payload: dict[str, Any]) -> None:
    parameters = payload["parameters"]
    if "n_estimators" in parameters:
        parameters["n_estimators"].update({"low": 2, "high": 4, "step": 2})
    if "iterations" in parameters:
        parameters["iterations"].update({"low": 2, "high": 4, "step": 2})
    for name, specification in parameters.items():
        if name in {"n_estimators", "iterations"}:
            continue
        if specification["distribution"] == "zero_or_log_float":
            continue
        if "high" in specification and "low" in specification:
            specification["high"] = specification["low"]


def write_search_plan(
    *,
    space_path: Path,
    storage_path: Path,
    reports: Path,
    study_name: str,
    adapter_id: str = "xgboost_numeric_v1",
    base_config: str = "configs/research_v2/xgboost_numeric_v1_smoke.yaml",
    n_trials: int = 3,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "search_plan_id": f"{adapter_id}_auth_test_v1",
        "dataset_version": "v3_targeted_missingness",
        "pipeline_id": "manual_v3_pipeline_v1_compat",
        "adapter_id": adapter_id,
        "base_candidate_config": base_config,
        "search_space": space_path.relative_to(PROJECT_ROOT).as_posix(),
        "repeats": 1,
        "folds": 3,
        "assignment_seed": 23,
        "threshold_policy": "grid_balanced_accuracy_v1",
        "threshold_grid": {
            "minimum": 0.01,
            "maximum": 0.99,
            "step": 0.001,
            "comparison": "greater_than_or_equal",
            "maximizer_absolute_tolerance": 1.0e-12,
            "tie_break": "median_maximizer_lower_on_even",
            "constant_probability_fallback": 0.5,
        },
        "metric": "balanced_accuracy",
        "direction": "maximize",
        "n_trials": n_trials,
        "timeout_seconds": 600,
        "sampler": {"name": "stateless_random_v1", "seed": 23},
        "pruner": "nop",
        "study_name": study_name,
        "storage": storage_path.relative_to(PROJECT_ROOT).as_posix(),
        "artifacts_root": reports.relative_to(PROJECT_ROOT).as_posix(),
    }


def run_authorized_completed_search(
    *,
    adapter_id: str = "xgboost_numeric_v1",
    base_config: str = "configs/research_v2/xgboost_numeric_v1_smoke.yaml",
    space_config: str = "configs/optuna/search_spaces/xgboost_numeric_v1.yaml",
    n_trials: int = 3,
    fail_first: bool = True,
) -> tuple[Path, Path]:
    ensure_train_only_files()
    token = uuid.uuid4().hex
    operational = PROJECT_ROOT / "artifacts" / "optuna" / "auth_tests" / token
    reports = PROJECT_ROOT / "artifacts" / "optuna_searches" / "auth_tests" / token
    operational.mkdir(parents=True)
    reports.mkdir(parents=True)
    space = yaml.safe_load((PROJECT_ROOT / space_config).read_text(encoding="utf-8"))
    assert isinstance(space, dict)
    shrink_space(space)
    space_path = operational / "space.yaml"
    config_path = operational / "config.yaml"
    space_path.write_text(yaml.safe_dump(space, sort_keys=False), encoding="utf-8")
    config_path.write_text(
        yaml.safe_dump(
            write_search_plan(
                space_path=space_path,
                storage_path=operational / "study.db",
                reports=reports,
                study_name=f"auth_{token}",
                adapter_id=adapter_id,
                base_config=base_config,
                n_trials=n_trials,
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = load_optuna_search_config(config_path, project_root=PROJECT_ROOT)
    data = load_research_v2_training_data(config.base_config)
    result = run_optuna_study(
        config,
        X=data.X,
        y=data.y,
        dataset_identity=portable_dataset_identity(
            data.fingerprints,
            project_root=PROJECT_ROOT,
        ),
        assignments=build_search_assignments(
            data.y,
            repeats=1,
            folds=3,
            assignment_seed=23,
        ),
        adapter=DeterministicAdapter(fail_first=fail_first),
    )
    return result.search_dir, operational


def reauthenticate(root: Path) -> None:
    inventory = {
        "schema_version": 1,
        "files": _inventory_records(root, PAYLOAD_FILES),
    }
    _write_json(root / "recursive_inventory.json", inventory)
    files = _inventory_records(root, PAYLOAD_FILES | {"recursive_inventory.json"})
    body = {"schema_version": 1, "files": files}
    manifest = {**body, "manifest_sha256": canonical_sha256(body)}
    _write_json(root / "manifest.json", manifest)
    identity = json.loads((root / "search_identity.json").read_text(encoding="utf-8"))
    _write_json(
        root / "_SUCCESS",
        {
            "schema_version": 1,
            "search_id": identity["search_id"],
            "manifest_sha256": manifest["manifest_sha256"],
        },
    )


def one_ulp_away(value: float) -> float:
    packed = struct.pack(">d", float(value))
    as_int = int.from_bytes(packed, "big")
    if math.copysign(1.0, value) >= 0.0:
        as_int += 1
    else:
        as_int -= 1
    return struct.unpack(">d", as_int.to_bytes(8, "big"))[0]


def rewrite_dataset_identity(root: Path, mutator: Any) -> None:
    path = root / "dataset_identity.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    mutator(value)
    body = {
        "schema_version": value["schema_version"],
        "provided_identity": value["provided_identity"],
        "provided_identity_sha256": canonical_sha256(value["provided_identity"]),
        "training_data": value["training_data"],
    }
    value = {**body, "identity_sha256": canonical_sha256(body)}
    _write_json(path, value)
    search_identity = json.loads(
        (root / "search_identity.json").read_text(encoding="utf-8")
    )
    search_identity["dataset_identity_sha256"] = value["identity_sha256"]
    _write_json(root / "search_identity.json", search_identity)
    study = json.loads((root / "study_summary.json").read_text(encoding="utf-8"))
    study["dataset_identity_sha256"] = value["identity_sha256"]
    _write_json(root / "study_summary.json", study)
