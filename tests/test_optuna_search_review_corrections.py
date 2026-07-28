from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
import pytest
import yaml

from src.churn_ml.optuna_search_artifacts import (
    PAYLOAD_FILES,
    OptunaSearchArtifactError,
    _inventory_records,
    _validate_tree,
    _write_json,
    load_optuna_search_result,
)
from src.churn_ml.optuna_search_config import load_optuna_search_config
from src.churn_ml.optuna_search_export import export_best_candidate
from src.churn_ml.optuna_search_lifecycle import (
    _authoritative_dataset_identity,
    _verify_or_initialize_study,
    portable_dataset_identity,
    run_optuna_study,
)
from src.churn_ml.optuna_search_objective import build_search_assignments
from src.churn_ml.research_data import canonical_sha256
from src.churn_ml.research_v2_data import load_research_v2_training_data
from tests.optuna_auth_support import ensure_train_only_files


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


class _DeterministicAdapter:
    id = "xgboost_numeric_v1"

    def __init__(self) -> None:
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
        if not self._failed:
            self._failed = True
            raise RuntimeError("intentional first-trial failure")
        assert set(model_training_positions).isdisjoint(prediction_positions)
        if audit_callback is not None:
            audit_callback(model_training_positions, prediction_positions)
        return ((prediction_positions % 13) + 1).astype(float) / 14.0


@pytest.fixture(scope="module")
def completed_search() -> Iterator[tuple[Path, Path]]:
    pytest.importorskip("optuna")
    ensure_train_only_files()
    token = uuid.uuid4().hex
    operational = PROJECT_ROOT / "artifacts" / "optuna" / "review_tests" / token
    reports = PROJECT_ROOT / "artifacts" / "optuna_searches" / "review_tests" / token
    operational.mkdir(parents=True)
    reports.mkdir(parents=True)
    try:
        space = yaml.safe_load(
            (
                PROJECT_ROOT / "configs/optuna/search_spaces/xgboost_numeric_v1.yaml"
            ).read_text(encoding="utf-8")
        )
        assert isinstance(space, dict)
        _shrink_space(space)
        space_path = operational / "space.yaml"
        config_path = operational / "config.yaml"
        space_path.write_text(
            yaml.safe_dump(space, sort_keys=False),
            encoding="utf-8",
        )
        config_path.write_text(
            yaml.safe_dump(
                _plan(
                    space_path=space_path,
                    storage_path=operational / "study.db",
                    reports=reports,
                    study_name=f"review_{token}",
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
            adapter=_DeterministicAdapter(),
        )
        load_optuna_search_result(result.search_dir, project_root=PROJECT_ROOT)
        yield result.search_dir, operational
    finally:
        shutil.rmtree(operational, ignore_errors=True)
        shutil.rmtree(reports, ignore_errors=True)


Mutation = Callable[[Path], None]


@pytest.mark.parametrize(
    "case",
    (
        "changed_probability",
        "duplicate_prediction",
        "missing_prediction",
        "extra_prediction",
        "wrong_row_key",
        "changed_fold_membership",
        "scored_fold_in_threshold_membership",
        "changed_threshold",
        "changed_prediction_label",
        "changed_fold_balanced_accuracy",
        "changed_repeat_aggregate",
        "changed_final_objective",
        "completed_with_incomplete_evidence",
        "failed_with_completed_evidence",
        "forged_best_trial",
        "forged_winning_parameters",
        "candidate_points_to_nonbest",
        "assignment_hash_mismatch",
        "membership_hash_mismatch",
        "forged_source_identity",
        "forged_search_identity",
        "forged_dataset_identity",
    ),
)
def test_coherently_remanifested_semantic_corruption_is_rejected(
    completed_search: tuple[Path, Path],
    case: str,
) -> None:
    valid, operational = completed_search
    case_parent = operational / "corruptions" / case
    corrupted = case_parent / valid.name
    case_parent.mkdir(parents=True)
    shutil.copytree(valid, corrupted)
    _mutate(corrupted, case)
    _reauthenticate(corrupted)
    with pytest.raises(OptunaSearchArtifactError):
        load_optuna_search_result(corrupted, project_root=PROJECT_ROOT)
    output = case_parent / "candidate.yaml"
    with pytest.raises(OptunaSearchArtifactError):
        export_best_candidate(
            corrupted,
            output,
            project_root=PROJECT_ROOT,
        )
    assert not output.exists()


def test_unexpected_artifact_directory_is_rejected() -> None:
    token = uuid.uuid4().hex
    root = PROJECT_ROOT / "artifacts" / "optuna_searches" / f"tree_{token}"
    (root / "unexpected").mkdir(parents=True)
    try:
        with pytest.raises(
            OptunaSearchArtifactError, match="Unexpected artifact directory"
        ):
            _validate_tree(root, require_success=True)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_linked_or_reparse_search_root_is_rejected() -> None:
    token = uuid.uuid4().hex
    parent = PROJECT_ROOT / "artifacts" / "optuna_searches" / f"link_{token}"
    real = parent / "real"
    linked = parent / "linked"
    real.mkdir(parents=True)
    try:
        try:
            os.symlink(real, linked, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"Directory symlink unavailable: {error}")
        with pytest.raises(OptunaSearchArtifactError, match="links/reparse"):
            load_optuna_search_result(linked, project_root=PROJECT_ROOT)
    finally:
        if linked.is_symlink():
            linked.unlink()
        shutil.rmtree(parent, ignore_errors=True)


def test_post_success_timestamp_mutation_is_rejected(
    completed_search: tuple[Path, Path],
) -> None:
    valid, operational = completed_search
    case_parent = operational / "post_success"
    mutated = case_parent / valid.name
    case_parent.mkdir(parents=True)
    shutil.copytree(valid, mutated)
    success_time = (mutated / "_SUCCESS").stat().st_mtime_ns
    os.utime(
        mutated / "manifest.json",
        ns=(success_time + 1_000_000, success_time + 1_000_000),
    )
    with pytest.raises(OptunaSearchArtifactError, match="newest"):
        load_optuna_search_result(mutated, project_root=PROJECT_ROOT)


@pytest.mark.parametrize(
    "argv",
    (
        (),
        ("validate",),
        ("inspect",),
        ("export-best", "--search-dir", "missing"),
        ("validate", "--unknown"),
        ("unknown-command",),
        ("export-best", "--output", "candidate.yaml"),
    ),
)
def test_cli_usage_failures_are_one_json_object(argv: tuple[str, ...]) -> None:
    completed = subprocess.run(
        [str(PYTHON), "-u", "scripts/run_optuna_search.py", *argv],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    payload = json.loads(completed.stderr)
    assert payload == {
        "error_kind": "usage_error",
        "error_type": "CliUsageError",
        "exit_code": 2,
        "message": payload["message"],
        "status": "failed",
    }
    assert payload["message"]
    assert completed.stderr.count("\n") == 1


def test_documented_search_spaces_exactly_match_yaml() -> None:
    documentation = (PROJECT_ROOT / "docs/optuna-search-v1.md").read_text(
        encoding="utf-8"
    )
    for adapter_id in ("xgboost_numeric_v1", "catboost_numeric_v1"):
        pattern = re.compile(
            rf"<!-- OPTUNA_SEARCH_SPACE:{adapter_id} -->\s*"
            rf"```yaml\s*(.*?)\s*```\s*"
            rf"<!-- /OPTUNA_SEARCH_SPACE:{adapter_id} -->",
            re.DOTALL,
        )
        match = pattern.search(documentation)
        assert match is not None
        documented = yaml.safe_load(match.group(1))
        configured = yaml.safe_load(
            (
                PROJECT_ROOT / f"configs/optuna/search_spaces/{adapter_id}.yaml"
            ).read_text(encoding="utf-8")
        )
        assert documented == configured["parameters"]


@pytest.mark.parametrize(
    "mutation",
    (
        "source_size",
        "source_digest",
        "source_added",
        "source_removed",
        "python",
        "optuna",
        "xgboost",
        "numpy",
        "pandas",
        "scikit-learn",
        "pyarrow",
        "pyyaml",
        "runtime_missing",
        "runtime_malformed",
        "source_missing",
        "source_malformed",
        "different_study_identity",
    ),
)
def test_resume_rejects_every_source_runtime_identity_mutation(
    mutation: str,
) -> None:
    config = load_optuna_search_config(
        PROJECT_ROOT / "configs/optuna/xgboost_numeric_v1_smoke.yaml",
        project_root=PROJECT_ROOT,
    )
    X, y = _synthetic_data()
    dataset = _authoritative_dataset_identity(
        X,
        y,
        {"schema_version": 1, "synthetic": True},
    )
    assignments = build_search_assignments(
        y,
        repeats=1,
        folds=3,
        assignment_seed=23,
    )
    study = _FakeStudy()
    _verify_or_initialize_study(
        study,
        config,
        dataset_identity_sha256=canonical_sha256(dataset),
        assignment_identity_sha256=canonical_sha256(assignments.identity),
    )
    _mutate_study_identity(study.user_attrs, mutation)
    with pytest.raises(Exception, match="identity differs"):
        _verify_or_initialize_study(
            study,
            config,
            dataset_identity_sha256=canonical_sha256(dataset),
            assignment_identity_sha256=canonical_sha256(assignments.identity),
        )


def test_catboost_runtime_version_mutation_is_rejected() -> None:
    config = load_optuna_search_config(
        PROJECT_ROOT / "configs/optuna/catboost_numeric_v1_smoke.yaml",
        project_root=PROJECT_ROOT,
    )
    study = _authenticated_fake_study(config)
    study.user_attrs["runtime_dependencies"]["packages"]["catboost"] = "0.0.0"
    with pytest.raises(Exception, match="identity differs"):
        _verify_or_initialize_study(
            study,
            config,
            dataset_identity_sha256="1" * 64,
            assignment_identity_sha256="2" * 64,
        )


def test_every_direct_source_dependency_is_individually_authenticated() -> None:
    config = load_optuna_search_config(
        PROJECT_ROOT / "configs/optuna/xgboost_numeric_v1_smoke.yaml",
        project_root=PROJECT_ROOT,
    )
    source_files = config.resume_authentication["source_closure"]["files"]
    assert source_files
    for index in range(len(source_files)):
        study = _authenticated_fake_study(config)
        study.user_attrs["source_closure"]["files"][index]["sha256"] = "0" * 64
        with pytest.raises(Exception, match="identity differs"):
            _verify_or_initialize_study(
                study,
                config,
                dataset_identity_sha256="1" * 64,
                assignment_identity_sha256="2" * 64,
            )


def test_valid_unchanged_resume_identity_is_idempotent() -> None:
    config = load_optuna_search_config(
        PROJECT_ROOT / "configs/optuna/xgboost_numeric_v1_smoke.yaml",
        project_root=PROJECT_ROOT,
    )
    study = _authenticated_fake_study(config)
    before = deepcopy(study.user_attrs)
    _verify_or_initialize_study(
        study,
        config,
        dataset_identity_sha256="1" * 64,
        assignment_identity_sha256="2" * 64,
    )
    assert study.user_attrs == before


class _FakeStudy:
    def __init__(self) -> None:
        self.user_attrs: dict[str, Any] = {}

    def get_trials(self, *, deepcopy: bool = False) -> list[Any]:
        del deepcopy
        return []

    def set_user_attr(self, name: str, value: Any) -> None:
        self.user_attrs[name] = deepcopy(value)


def _authenticated_fake_study(config: Any) -> _FakeStudy:
    study = _FakeStudy()
    _verify_or_initialize_study(
        study,
        config,
        dataset_identity_sha256="1" * 64,
        assignment_identity_sha256="2" * 64,
    )
    return study


def _mutate_study_identity(attrs: dict[str, Any], mutation: str) -> None:
    if mutation == "source_size":
        attrs["source_closure"]["files"][0]["size_bytes"] += 1
    elif mutation == "source_digest":
        attrs["source_closure"]["files"][0]["sha256"] = "0" * 64
    elif mutation == "source_added":
        attrs["source_closure"]["files"].append(
            {"path": "src/churn_ml/forged.py", "size_bytes": 1, "sha256": "0" * 64}
        )
    elif mutation == "source_removed":
        attrs["source_closure"]["files"].pop()
    elif mutation == "python":
        attrs["runtime_dependencies"]["python"] = "0.0.0"
    elif mutation in {
        "optuna",
        "xgboost",
        "numpy",
        "pandas",
        "scikit-learn",
        "pyarrow",
        "pyyaml",
    }:
        attrs["runtime_dependencies"]["packages"][mutation] = "0.0.0"
    elif mutation == "runtime_missing":
        attrs.pop("runtime_dependencies")
    elif mutation == "runtime_malformed":
        attrs["runtime_dependencies"] = "invalid"
    elif mutation == "source_missing":
        attrs.pop("source_closure")
    elif mutation == "source_malformed":
        attrs["source_closure"] = "invalid"
    else:
        attrs["study_identity_sha256"] = "0" * 64


def _mutate(root: Path, case: str) -> None:
    predictions_path = root / "trial_predictions.csv"
    metrics_path = root / "trial_metrics.csv"
    trials_path = root / "trials.csv"
    predictions = pd.read_csv(predictions_path)
    metrics = pd.read_csv(metrics_path)
    trials = pd.read_csv(trials_path)
    complete_number = int(
        trials.loc[trials["state"] == "COMPLETE", "trial_number"].iloc[0]
    )
    other_complete = int(
        trials.loc[trials["state"] == "COMPLETE", "trial_number"].iloc[-1]
    )
    if case == "changed_probability":
        predictions.loc[0, "probability"] = (
            0.0 if float(predictions.loc[0, "probability"]) > 0.5 else 1.0
        )
        predictions.to_csv(predictions_path, index=False)
    elif case == "duplicate_prediction":
        pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True).to_csv(
            predictions_path,
            index=False,
        )
    elif case == "missing_prediction":
        predictions.iloc[1:].to_csv(predictions_path, index=False)
    elif case == "extra_prediction":
        extra = predictions.iloc[[0]].copy()
        extra["row_position"] = 999_999
        pd.concat([predictions, extra], ignore_index=True).to_csv(
            predictions_path,
            index=False,
        )
    elif case == "wrong_row_key":
        predictions.loc[0, "row_position"] = 999_999
        predictions.to_csv(predictions_path, index=False)
    elif case == "changed_fold_membership":
        predictions.loc[0, "fold"] = (int(predictions.loc[0, "fold"]) % 3) + 1
        predictions.to_csv(predictions_path, index=False)
    elif case == "scored_fold_in_threshold_membership":
        path = root / "threshold_selection_membership.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "threshold_source_fold"] = frame.loc[0, "scoring_fold"]
        frame.to_csv(path, index=False)
    elif case == "changed_threshold":
        predictions.loc[0, "selected_threshold"] = 0.999
        predictions.to_csv(predictions_path, index=False)
    elif case == "changed_prediction_label":
        predictions.loc[0, "prediction"] = 1 - int(predictions.loc[0, "prediction"])
        predictions.to_csv(predictions_path, index=False)
    elif case == "changed_fold_balanced_accuracy":
        index = metrics.index[metrics["record_type"] == "fold"][0]
        metrics.loc[index, "balanced_accuracy"] = 0.123
        metrics.to_csv(metrics_path, index=False)
    elif case == "changed_repeat_aggregate":
        index = metrics.index[metrics["record_type"] == "repeat"][0]
        metrics.loc[index, "balanced_accuracy"] = 0.123
        metrics.to_csv(metrics_path, index=False)
    elif case == "changed_final_objective":
        index = trials.index[trials["trial_number"] == complete_number][0]
        trials.loc[index, "objective"] = 0.123
        trials.to_csv(trials_path, index=False)
    elif case == "completed_with_incomplete_evidence":
        index = trials.index[trials["trial_number"] == complete_number][0]
        coverage = json.loads(trials.loc[index, "prediction_key_coverage_json"])
        coverage["complete"] = False
        trials.loc[index, "prediction_key_coverage_json"] = json.dumps(
            coverage,
            sort_keys=True,
            separators=(",", ":"),
        )
        trials.to_csv(trials_path, index=False)
    elif case == "failed_with_completed_evidence":
        failed = trials.index[trials["state"] == "FAIL"][0]
        complete = trials.index[trials["state"] == "COMPLETE"][0]
        for name in (
            "adapter_id",
            "candidate_identity_sha256",
            "objective",
            "resolved_parameters_json",
            "prediction_key_coverage_json",
        ):
            trials.loc[failed, name] = trials.loc[complete, name]
        trials.to_csv(trials_path, index=False)
    elif case == "forged_best_trial":
        path = root / "best_trial.json"
        best = json.loads(path.read_text(encoding="utf-8"))
        best["trial_number"] = other_complete
        _write_json(path, best)
    elif case == "forged_winning_parameters":
        path = root / "best_trial.json"
        best = json.loads(path.read_text(encoding="utf-8"))
        name = next(iter(best["resolved_parameters"]))
        best["resolved_parameters"][name] = 999
        _write_json(path, best)
    elif case == "candidate_points_to_nonbest":
        path = root / "best_candidate_config.yaml"
        candidate = yaml.safe_load(path.read_text(encoding="utf-8"))
        candidate["search_provenance"]["best_trial_number"] = other_complete
        path.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
    elif case == "assignment_hash_mismatch":
        path = root / "assignment_identity.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["fold_assignments_sha256"] = "0" * 64
        _write_json(path, value)
    elif case == "membership_hash_mismatch":
        path = root / "assignment_identity.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["threshold_membership_sha256"] = "0" * 64
        _write_json(path, value)
    elif case == "forged_source_identity":
        path = root / "source_provenance.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["files"][0]["sha256"] = "0" * 64
        value["sha256"] = canonical_sha256(
            {
                name: value[name]
                for name in ("schema_version", "hashing_method", "files")
            }
        )
        _write_json(path, value)
    elif case == "forged_search_identity":
        path = root / "search_identity.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["sha256"] = "0" * 64
        _write_json(path, value)
    else:
        path = root / "dataset_identity.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["provided_identity"]["synthetic"] = False
        _write_json(path, value)


def _reauthenticate(root: Path) -> None:
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


def _plan(
    *,
    space_path: Path,
    storage_path: Path,
    reports: Path,
    study_name: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "search_plan_id": "xgboost_numeric_v1_review_test_v1",
        "dataset_version": "v3_targeted_missingness",
        "pipeline_id": "manual_v3_pipeline_v1_compat",
        "adapter_id": "xgboost_numeric_v1",
        "base_candidate_config": "configs/research_v2/xgboost_numeric_v1_smoke.yaml",
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
        "n_trials": 3,
        "timeout_seconds": 120,
        "sampler": {"name": "stateless_random_v1", "seed": 23},
        "pruner": "nop",
        "study_name": study_name,
        "storage": storage_path.relative_to(PROJECT_ROOT).as_posix(),
        "artifacts_root": reports.relative_to(PROJECT_ROOT).as_posix(),
    }


def _shrink_space(payload: dict[str, Any]) -> None:
    parameters = payload["parameters"]
    parameters["n_estimators"].update({"low": 2, "high": 4, "step": 2})
    for name, specification in parameters.items():
        if name == "n_estimators" or specification["distribution"] == (
            "zero_or_log_float"
        ):
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
