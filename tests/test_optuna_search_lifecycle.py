from __future__ import annotations

import shutil
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from src.churn_ml.experiment_v2 import get_candidate_adapter
from src.churn_ml.optuna_search_artifacts import (
    OptunaSearchArtifactError,
    load_optuna_search_result,
)
from src.churn_ml.optuna_search_config import load_optuna_search_config
from src.churn_ml.optuna_search_export import (
    OptunaSearchExportError,
    export_best_candidate,
)
from src.churn_ml.optuna_search_lifecycle import (
    build_source_provenance,
    run_optuna_study,
)
from src.churn_ml.optuna_search_objective import build_search_assignments
from src.churn_ml.research_data import canonical_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _DeterministicAdapter:
    id = "xgboost_numeric_v1"

    def validate_contract(self, contract: Any) -> None:
        assert isinstance(contract, dict)

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
        assert set(model_training_positions).isdisjoint(prediction_positions)
        if audit_callback is not None:
            audit_callback(model_training_positions, prediction_positions)
        return ((prediction_positions % 13) + 1).astype(float) / 14.0

    def identity_inputs(self, contract: Any) -> dict[str, Any]:
        del contract
        return {
            "id": self.id,
            "probability_semantics": "binary_positive_class_label_1",
            "early_stopping": "disabled",
        }


class _FailFirstAdapter(_DeterministicAdapter):
    def __init__(self) -> None:
        self.failed = False

    def fit_predict(self, *args: Any, **kwargs: Any) -> np.ndarray:
        if not self.failed:
            self.failed = True
            raise RuntimeError("intentional synthetic trial failure")
        return super().fit_predict(*args, **kwargs)


@pytest.mark.parametrize(
    ("adapter_id", "base_config", "production_space"),
    (
        (
            "xgboost_numeric_v1",
            "configs/research_v2/xgboost_numeric_v1_smoke.yaml",
            "configs/optuna/search_spaces/xgboost_numeric_v1.yaml",
        ),
        (
            "catboost_numeric_v1",
            "configs/research_v2/catboost_numeric_v1_smoke.yaml",
            "configs/optuna/search_spaces/catboost_numeric_v1.yaml",
        ),
    ),
)
def test_tiny_real_study_resume_artifacts_and_tamper_detection(
    adapter_id: str,
    base_config: str,
    production_space: str,
) -> None:
    package = "xgboost" if adapter_id.startswith("xgboost") else "catboost"
    pytest.importorskip(package)
    pytest.importorskip("optuna")
    token = uuid.uuid4().hex
    operational_root = PROJECT_ROOT / "artifacts" / "optuna" / "tests" / token
    report_root = PROJECT_ROOT / "artifacts" / "optuna_searches" / f"tests_{token}"
    operational_root.mkdir(parents=True)
    report_root.mkdir(parents=True)
    try:
        space_payload = _yaml(PROJECT_ROOT / production_space)
        _shrink_space(space_payload, adapter_id)
        space_path = operational_root / "space.yaml"
        _write_yaml(space_path, space_payload)
        first_path = operational_root / "first.yaml"
        second_path = operational_root / "second.yaml"
        base_plan = _plan(
            adapter_id=adapter_id,
            base_config=base_config,
            space_path=space_path,
            storage_path=operational_root / "study.db",
            report_root=report_root,
            n_trials=1,
            study_name=f"test_{adapter_id}_{token}",
        )
        _write_yaml(first_path, base_plan)
        increased = deepcopy(base_plan)
        increased["n_trials"] = 2
        _write_yaml(second_path, increased)
        X, y = _synthetic_data()
        assignments = build_search_assignments(
            y,
            repeats=1,
            folds=3,
            assignment_seed=23,
        )
        adapter = get_candidate_adapter(adapter_id)
        first_config = load_optuna_search_config(
            first_path,
            project_root=PROJECT_ROOT,
        )
        first = run_optuna_study(
            first_config,
            X=X,
            y=y,
            dataset_identity={"schema_version": 1, "synthetic": True},
            assignments=assignments,
            adapter=adapter,
        )
        assert first.study_summary["actual_trials"] == 1
        first_loaded = load_optuna_search_result(
            first.search_dir,
            project_root=PROJECT_ROOT,
        )
        assert first_loaded.best_trial["evidence_scope"] == (
            "tuning_only_not_unbiased_final_evidence"
        )
        with pytest.raises(OptunaSearchArtifactError, match="already exists"):
            run_optuna_study(
                first_config,
                X=X,
                y=y,
                dataset_identity={"schema_version": 1, "synthetic": True},
                assignments=assignments,
                adapter=adapter,
            )

        second_config = load_optuna_search_config(
            second_path,
            project_root=PROJECT_ROOT,
        )
        assert second_config.study_identity_sha256 == first_config.study_identity_sha256
        assert second_config.search_id != first_config.search_id
        second = run_optuna_study(
            second_config,
            X=X,
            y=y,
            dataset_identity={"schema_version": 1, "synthetic": True},
            assignments=assignments,
            adapter=adapter,
        )
        assert second.study_summary["actual_trials"] == 2
        trials = pd.read_csv(second.search_dir / "trials.csv")
        assert trials["trial_number"].tolist() == [0, 1]
        assert not trials["trial_number"].duplicated().any()
        candidate = _yaml(second.search_dir / "best_candidate_config.yaml")
        assert candidate["candidate_adapter"]["id"] == adapter_id
        assert "search_provenance" in candidate
        assert "storage" not in candidate["search_provenance"]
        assert not (second.search_dir / "model.joblib").exists()
        assert not any(
            "submission" in path.name for path in second.search_dir.iterdir()
        )

        exported_path = operational_root / "exported_candidate.yaml"
        exported = export_best_candidate(
            second.search_dir,
            exported_path,
            project_root=PROJECT_ROOT,
        )
        assert exported == exported_path
        with pytest.raises(OptunaSearchExportError, match="already exists"):
            export_best_candidate(
                second.search_dir,
                exported_path,
                project_root=PROJECT_ROOT,
            )

        with (second.search_dir / "best_trial.json").open(
            "a", encoding="utf-8"
        ) as file:
            file.write(" ")
        with pytest.raises(OptunaSearchArtifactError, match="inventory"):
            load_optuna_search_result(
                second.search_dir,
                project_root=PROJECT_ROOT,
            )
    finally:
        shutil.rmtree(operational_root, ignore_errors=True)
        shutil.rmtree(report_root, ignore_errors=True)


def test_failed_trial_reason_is_persisted_and_later_trial_completes() -> None:
    pytest.importorskip("optuna")
    token = uuid.uuid4().hex
    operational_root = PROJECT_ROOT / "artifacts" / "optuna" / "tests" / token
    report_root = PROJECT_ROOT / "artifacts" / "optuna_searches" / f"tests_{token}"
    operational_root.mkdir(parents=True)
    report_root.mkdir(parents=True)
    try:
        space_payload = _yaml(
            PROJECT_ROOT / "configs/optuna/search_spaces/xgboost_numeric_v1.yaml"
        )
        _shrink_space(space_payload, "xgboost_numeric_v1")
        space_path = operational_root / "space.yaml"
        _write_yaml(space_path, space_payload)
        config_path = operational_root / "failure.yaml"
        payload = _plan(
            adapter_id="xgboost_numeric_v1",
            base_config="configs/research_v2/xgboost_numeric_v1_smoke.yaml",
            space_path=space_path,
            storage_path=operational_root / "study.db",
            report_root=report_root,
            n_trials=2,
            study_name=f"test_failure_{token}",
        )
        _write_yaml(config_path, payload)
        config = load_optuna_search_config(config_path, project_root=PROJECT_ROOT)
        X, y = _synthetic_data()
        result = run_optuna_study(
            config,
            X=X,
            y=y,
            dataset_identity={"schema_version": 1, "synthetic": True},
            assignments=build_search_assignments(
                y,
                repeats=1,
                folds=3,
                assignment_seed=23,
            ),
            adapter=_FailFirstAdapter(),
        )
        trials = pd.read_csv(result.search_dir / "trials.csv")
        assert trials["state"].tolist() == ["FAIL", "COMPLETE"]
        assert trials.loc[0, "failure_reason_code"] == "TRIAL_EXECUTION_FAILED"
        assert result.study_summary["state_counts"]["fail"] == 1
        assert result.study_summary["state_counts"]["complete"] == 1
    finally:
        shutil.rmtree(operational_root, ignore_errors=True)
        shutil.rmtree(report_root, ignore_errors=True)


def test_running_sqlite_trial_is_recovered_after_interruption() -> None:
    optuna = pytest.importorskip("optuna")
    token = uuid.uuid4().hex
    operational_root = PROJECT_ROOT / "artifacts" / "optuna" / "tests" / token
    report_root = PROJECT_ROOT / "artifacts" / "optuna_searches" / f"tests_{token}"
    operational_root.mkdir(parents=True)
    report_root.mkdir(parents=True)
    try:
        space_payload = _yaml(
            PROJECT_ROOT / "configs/optuna/search_spaces/xgboost_numeric_v1.yaml"
        )
        _shrink_space(space_payload, "xgboost_numeric_v1")
        space_path = operational_root / "space.yaml"
        _write_yaml(space_path, space_payload)
        storage_path = operational_root / "study.db"
        config_path = operational_root / "interruption.yaml"
        payload = _plan(
            adapter_id="xgboost_numeric_v1",
            base_config="configs/research_v2/xgboost_numeric_v1_smoke.yaml",
            space_path=space_path,
            storage_path=storage_path,
            report_root=report_root,
            n_trials=2,
            study_name=f"test_interruption_{token}",
        )
        _write_yaml(config_path, payload)
        config = load_optuna_search_config(config_path, project_root=PROJECT_ROOT)
        study = optuna.create_study(
            study_name=config.payload["study_name"],
            storage=f"sqlite:///{storage_path.as_posix()}",
            direction="maximize",
            load_if_exists=True,
        )
        source = build_source_provenance(config)
        config_relative = config.source_path.relative_to(PROJECT_ROOT).as_posix()
        study_source_sha256 = canonical_sha256(
            {
                "schema_version": 1,
                "files": [
                    item for item in source["files"] if item["path"] != config_relative
                ],
            }
        )
        study.set_user_attr("study_identity_sha256", config.study_identity_sha256)
        study.set_user_attr("study_identity", config.study_identity)
        study.set_user_attr("schema_version", 1)
        study.set_user_attr("study_source_sha256", study_source_sha256)
        running = study.ask()
        assert running.number == 0

        X, y = _synthetic_data()
        result = run_optuna_study(
            config,
            X=X,
            y=y,
            dataset_identity={"schema_version": 1, "synthetic": True},
            assignments=build_search_assignments(
                y,
                repeats=1,
                folds=3,
                assignment_seed=23,
            ),
            adapter=_DeterministicAdapter(),
        )
        trials = pd.read_csv(result.search_dir / "trials.csv")
        assert trials["trial_number"].tolist() == [0, 1]
        assert trials["state"].tolist() == ["FAIL", "COMPLETE"]
        assert trials.loc[0, "failure_reason_code"] == ("INTERRUPTED_PROCESS_RECOVERY")
        assert result.study_summary["recovered_interrupted_trials"] == 1
    finally:
        shutil.rmtree(operational_root, ignore_errors=True)
        shutil.rmtree(report_root, ignore_errors=True)


def _plan(
    *,
    adapter_id: str,
    base_config: str,
    space_path: Path,
    storage_path: Path,
    report_root: Path,
    n_trials: int,
    study_name: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "search_plan_id": f"{adapter_id}_tiny_test_v1",
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
        "timeout_seconds": 120,
        "sampler": {"name": "stateless_random_v1", "seed": 23},
        "pruner": "nop",
        "study_name": study_name,
        "storage": storage_path.relative_to(PROJECT_ROOT).as_posix(),
        "artifacts_root": report_root.relative_to(PROJECT_ROOT).as_posix(),
    }


def _shrink_space(payload: dict[str, Any], adapter_id: str) -> None:
    parameters = payload["parameters"]
    count_name = "n_estimators" if adapter_id.startswith("xgboost") else "iterations"
    parameters[count_name].update({"low": 2, "high": 4, "step": 2})
    for name, specification in parameters.items():
        if name == count_name or specification["distribution"] == ("zero_or_log_float"):
            continue
        specification["high"] = specification["low"]


def _synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    rows = 42
    rng = np.random.default_rng(12)
    y = pd.Series(([0, 1] * (rows // 2)), dtype="int64")
    X = pd.DataFrame(
        {
            "a": rng.normal(size=rows) + y.to_numpy() * 0.6,
            "b": rng.normal(size=rows),
            "c": rng.normal(size=rows),
        }
    )
    return X, y


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
