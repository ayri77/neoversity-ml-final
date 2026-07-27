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
from sklearn.metrics import balanced_accuracy_score

from src.churn_ml.experiment_v2 import (
    get_candidate_adapter,
    get_feature_pipeline,
)
from src.churn_ml.optuna_search_config import (
    OptunaSearchConfigurationError,
    load_optuna_search_config,
)
from src.churn_ml.optuna_search_export import build_best_candidate_config
from src.churn_ml.optuna_search_objective import (
    OptunaSearchObjectiveError,
    build_search_assignments,
    evaluate_trial,
    validate_search_assignments,
)
from src.churn_ml.research_protocol import select_balanced_accuracy_threshold
from src.churn_ml.research_v2_config import (
    ResearchV2ConfigurationError,
    load_research_v2_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEARCH_CONFIGS = (
    "configs/optuna/xgboost_numeric_v1_smoke.yaml",
    "configs/optuna/catboost_numeric_v1_smoke.yaml",
    "configs/optuna/xgboost_numeric_v1_development.yaml",
    "configs/optuna/catboost_numeric_v1_development.yaml",
)


@pytest.fixture
def contract_directory() -> Any:
    path = PROJECT_ROOT / "artifacts" / "optuna_tests" / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.mark.parametrize("relative_path", SEARCH_CONFIGS)
def test_all_production_search_configs_validate_without_allocation(
    relative_path: str,
) -> None:
    config = load_optuna_search_config(
        PROJECT_ROOT / relative_path,
        project_root=PROJECT_ROOT,
    )
    assert config.adapter_id in {"xgboost_numeric_v1", "catboost_numeric_v1"}
    assert not config.storage_path.exists()
    assert config.search_identity["metric"] == "balanced_accuracy"
    assert config.search_identity["direction"] == "maximize"
    assert config.search_identity["label_comparison"] == ("greater_than_or_equal")


def test_search_identity_is_deterministic_and_target_is_separate() -> None:
    path = PROJECT_ROOT / SEARCH_CONFIGS[0]
    first = load_optuna_search_config(path, project_root=PROJECT_ROOT)
    second = load_optuna_search_config(path, project_root=PROJECT_ROOT)
    assert first.study_identity_sha256 == second.study_identity_sha256
    assert first.search_identity_sha256 == second.search_identity_sha256
    assert first.search_id == second.search_id
    assert "storage" not in first.study_identity
    assert "target_n_trials" not in first.study_identity
    assert first.search_identity["target_n_trials"] == 3


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("n_trials", True),
        ("n_trials", "3"),
        ("timeout_seconds", 0),
        ("direction", "minimize"),
        ("metric", "roc_auc"),
        ("storage", "C:/absolute/optuna.db"),
        ("artifacts_root", "../outside"),
    ),
)
def test_plan_rejects_wrong_types_values_and_paths(
    contract_directory: Path,
    field: str,
    invalid: Any,
) -> None:
    payload = _yaml(PROJECT_ROOT / SEARCH_CONFIGS[0])
    payload[field] = invalid
    path = contract_directory / "invalid.yaml"
    _write_yaml(path, payload)
    with pytest.raises(OptunaSearchConfigurationError):
        load_optuna_search_config(path, project_root=PROJECT_ROOT)


def test_space_rejects_forbidden_parameter(
    contract_directory: Path,
) -> None:
    plan = _yaml(PROJECT_ROOT / SEARCH_CONFIGS[0])
    space = _yaml(PROJECT_ROOT / "configs/optuna/search_spaces/xgboost_numeric_v1.yaml")
    space["parameters"]["device"] = {
        "distribution": "categorical",
        "choices": ["cuda"],
    }
    space_path = contract_directory / "space.yaml"
    _write_yaml(space_path, space)
    plan["search_space"] = space_path.relative_to(PROJECT_ROOT).as_posix()
    plan_path = contract_directory / "plan.yaml"
    _write_yaml(plan_path, plan)
    with pytest.raises(OptunaSearchConfigurationError, match="unknown"):
        load_optuna_search_config(plan_path, project_root=PROJECT_ROOT)


def test_space_rejects_nonfinite_and_adapter_mismatch(
    contract_directory: Path,
) -> None:
    plan = _yaml(PROJECT_ROOT / SEARCH_CONFIGS[0])
    space = _yaml(PROJECT_ROOT / "configs/optuna/search_spaces/xgboost_numeric_v1.yaml")
    space["parameters"]["learning_rate"]["high"] = float("inf")
    space_path = contract_directory / "space.yaml"
    _write_yaml(space_path, space)
    plan["search_space"] = space_path.relative_to(PROJECT_ROOT).as_posix()
    plan_path = contract_directory / "plan.yaml"
    _write_yaml(plan_path, plan)
    with pytest.raises(OptunaSearchConfigurationError, match="finite float"):
        load_optuna_search_config(plan_path, project_root=PROJECT_ROOT)

    cat_space = _yaml(
        PROJECT_ROOT / "configs/optuna/search_spaces/catboost_numeric_v1.yaml"
    )
    _write_yaml(space_path, cat_space)
    with pytest.raises(OptunaSearchConfigurationError, match="differs"):
        load_optuna_search_config(plan_path, project_root=PROJECT_ROOT)


def test_exported_candidate_is_fixed_and_passes_production_validator(
    contract_directory: Path,
) -> None:
    config = load_optuna_search_config(
        PROJECT_ROOT / SEARCH_CONFIGS[0],
        project_root=PROJECT_ROOT,
    )
    parameters = {
        name: config.base_config.adapter_contract["xgboost"]["parameters"][name]
        for name in config.search_space.parameters
    }
    payload = build_best_candidate_config(
        config,
        best_trial_number=7,
        resolved_parameters=parameters,
    )
    candidate = contract_directory / "candidate.yaml"
    _write_yaml(candidate, payload)
    loaded = load_research_v2_config(candidate, project_root=PROJECT_ROOT)
    assert loaded.adapter_id == "xgboost_numeric_v1"
    assert loaded.payload["search_provenance"]["best_trial_number"] == 7
    assert "storage" not in loaded.payload["search_provenance"]
    assert not any(
        isinstance(value, dict) and "distribution" in value
        for value in loaded.adapter_contract["xgboost"]["parameters"].values()
    )


def test_optional_search_provenance_is_strict_and_identity_neutral(
    contract_directory: Path,
) -> None:
    config = load_optuna_search_config(
        PROJECT_ROOT / SEARCH_CONFIGS[0],
        project_root=PROJECT_ROOT,
    )
    base = config.base_config
    parameters = {
        name: base.adapter_contract["xgboost"]["parameters"][name]
        for name in config.search_space.parameters
    }
    candidate_payload = build_best_candidate_config(
        config,
        best_trial_number=2,
        resolved_parameters=parameters,
    )
    candidate_path = contract_directory / "candidate_with_provenance.yaml"
    _write_yaml(candidate_path, candidate_payload)
    candidate = load_research_v2_config(candidate_path, project_root=PROJECT_ROOT)
    assert get_candidate_adapter(base.adapter_id).identity_inputs(
        base.adapter_contract
    ) == get_candidate_adapter(candidate.adapter_id).identity_inputs(
        candidate.adapter_contract
    )
    assert get_feature_pipeline(base.pipeline_id).identity_inputs(
        base.pipeline_contract
    ) == get_feature_pipeline(candidate.pipeline_id).identity_inputs(
        candidate.pipeline_contract
    )
    assert "search_provenance" not in base.payload

    invalid = deepcopy(candidate_payload)
    invalid["search_provenance"]["unexpected"] = True
    invalid_path = contract_directory / "invalid_provenance.yaml"
    _write_yaml(invalid_path, invalid)
    with pytest.raises(ResearchV2ConfigurationError, match="unknown"):
        load_research_v2_config(invalid_path, project_root=PROJECT_ROOT)

    invalid = deepcopy(candidate_payload)
    invalid["search_provenance"]["best_trial_number"] = True
    _write_yaml(invalid_path, invalid)
    with pytest.raises(ResearchV2ConfigurationError, match="integer"):
        load_research_v2_config(invalid_path, project_root=PROJECT_ROOT)


def test_cross_fitted_objective_matches_independent_recomputation() -> None:
    row_count = 60
    y = pd.Series(([0, 1] * (row_count // 2)), dtype="int64")
    X = pd.DataFrame(
        {
            "signal": np.linspace(-2.0, 2.0, row_count),
            "noise": np.cos(np.arange(row_count)),
        }
    )
    assignments = build_search_assignments(
        y,
        repeats=2,
        folds=3,
        assignment_seed=17,
    )
    fit_calls: list[tuple[set[int], set[int], set[str]]] = []

    def fit_predict(
        train_features: pd.DataFrame,
        train_labels: pd.Series,
        prediction_features: pd.DataFrame,
        contract: dict[str, Any],
        **kwargs: Any,
    ) -> np.ndarray:
        del train_features, train_labels, contract
        train_positions = set(kwargs["model_training_positions"].tolist())
        prediction_positions = set(kwargs["prediction_positions"].tolist())
        fit_calls.append((train_positions, prediction_positions, set(kwargs)))
        values = prediction_features["signal"].to_numpy()
        return 1.0 / (1.0 + np.exp(-values))

    policy = {
        "id": "grid_balanced_accuracy_v1",
        "metric": "balanced_accuracy",
        "minimum": 0.01,
        "maximum": 0.99,
        "step": 0.001,
        "comparison": "greater_than_or_equal",
        "maximizer_absolute_tolerance": 1.0e-12,
        "tie_break": "median_maximizer_lower_on_even",
        "constant_probability_fallback": 0.5,
    }
    result = evaluate_trial(
        X,
        y,
        assignments,
        adapter_contract={},
        threshold_policy=policy,
        fit_predict=fit_predict,
        trial_number=4,
    )
    assert len(fit_calls) == 6
    for training, prediction, keyword_names in fit_calls:
        assert training.isdisjoint(prediction)
        assert keyword_names == {
            "model_training_positions",
            "prediction_positions",
            "audit_callback",
        }
        assert "eval_set" not in keyword_names
        assert "early_stopping_rounds" not in keyword_names

    independent_scores = []
    for (repeat, fold), scoring in result.predictions.groupby(
        ["repeat", "fold"],
        sort=True,
    ):
        selection = result.predictions.loc[
            (result.predictions["repeat"] == repeat)
            & (result.predictions["fold"] != fold)
        ]
        threshold = select_balanced_accuracy_threshold(
            selection["target"],
            selection["probability"].to_numpy(),
            policy,
        )
        labels = (scoring["probability"].to_numpy() >= threshold.threshold).astype(
            "int8"
        )
        independent_scores.append(balanced_accuracy_score(scoring["target"], labels))
        persisted = result.fold_metrics.loc[
            (result.fold_metrics["repeat"] == repeat)
            & (result.fold_metrics["fold"] == fold)
        ].iloc[0]
        assert int(fold) not in {
            int(value) for value in str(persisted["threshold_source_folds"]).split(",")
        }
        assert persisted["comparison"] == "greater_than_or_equal"
        assert persisted["selected_threshold"] == threshold.threshold
    assert result.objective == pytest.approx(float(np.mean(independent_scores)))
    assert result.coverage == {
        "schema_version": 1,
        "key_columns": ["trial_number", "repeat", "row_position"],
        "expected_rows": 120,
        "actual_rows": 120,
        "duplicates": 0,
        "missing": 0,
        "complete": True,
    }


def test_duplicate_assignment_or_threshold_membership_is_rejected() -> None:
    y = pd.Series(([0, 1] * 15), dtype="int64")
    assignments = build_search_assignments(
        y,
        repeats=1,
        folds=3,
        assignment_seed=9,
    )
    duplicate = pd.concat(
        [assignments.folds, assignments.folds.iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(OptunaSearchObjectiveError, match="duplicate"):
        validate_search_assignments(
            duplicate,
            assignments.threshold_membership,
            row_count=len(y),
            repeats=1,
            folds=3,
        )
    leaked = assignments.threshold_membership.copy()
    leaked.loc[0, "threshold_source_fold"] = leaked.loc[0, "scoring_fold"]
    with pytest.raises(OptunaSearchObjectiveError, match="Held-out"):
        validate_search_assignments(
            assignments.folds,
            leaked,
            row_count=len(y),
            repeats=1,
            folds=3,
        )


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return deepcopy(value)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
