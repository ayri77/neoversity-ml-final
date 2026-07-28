from __future__ import annotations

import json
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterator

import pandas as pd
import pytest

from src.churn_ml.optuna_search_artifacts import (
    OptunaSearchArtifactError,
    load_optuna_search_result,
)
from src.churn_ml.optuna_search_export import export_best_candidate
from src.churn_ml.optuna_search_lifecycle import (
    _verify_or_initialize_study,
    build_trial_failure_evidence,
    derive_failure_reason_code,
)
from src.churn_ml.optuna_search_resume import (
    OptunaSearchResumeIdentityError,
    build_resume_authentication,
    build_source_closure,
)
from src.churn_ml.research_data import canonical_sha256
from tests.optuna_auth_support import (
    PROJECT_ROOT,
    ensure_train_only_files,
    one_ulp_away,
    reauthenticate,
    rewrite_dataset_identity,
    run_authorized_completed_search,
)


Mutation = Callable[[Path], None]


@pytest.fixture(scope="module")
def authorized_completed_search() -> Iterator[tuple[Path, Path]]:
    pytest.importorskip("optuna")
    ensure_train_only_files()
    search_dir, operational = run_authorized_completed_search(n_trials=3)
    try:
        load_optuna_search_result(search_dir, project_root=PROJECT_ROOT)
        yield search_dir, operational
    finally:
        shutil.rmtree(operational, ignore_errors=True)
        shutil.rmtree(search_dir.parent, ignore_errors=True)


def _reject_inspect_and_export(corrupted: Path) -> None:
    with pytest.raises(OptunaSearchArtifactError):
        load_optuna_search_result(corrupted, project_root=PROJECT_ROOT)
    output = corrupted.parent / "candidate.yaml"
    with pytest.raises(OptunaSearchArtifactError):
        export_best_candidate(corrupted, output, project_root=PROJECT_ROOT)
    assert not output.exists()


def _corrupt_copy(valid: Path, operational: Path, name: str) -> Path:
    parent = operational / "corruptions" / name
    corrupted = parent / valid.name
    parent.mkdir(parents=True)
    shutil.copytree(valid, corrupted)
    return corrupted


@pytest.mark.parametrize(
    "case",
    (
        "forged_ordered_feature_list",
        "forged_feature_dtype",
        "forged_feature_count",
        "forged_row_count",
        "forged_feature_fingerprint",
        "forged_target_identity",
        "forged_target_value_counts",
        "forged_metadata_fingerprint",
        "forged_complete_dataset_identity_hash",
    ),
)
def test_coherent_dataset_identity_forgery_is_rejected(
    authorized_completed_search: tuple[Path, Path],
    case: str,
) -> None:
    valid, operational = authorized_completed_search
    corrupted = _corrupt_copy(valid, operational, case)

    def mutate(identity: dict[str, Any]) -> None:
        training = identity["training_data"]
        provided = identity["provided_identity"]
        if case == "forged_ordered_feature_list":
            names = list(training["feature_schema"]["ordered_names"])
            names[0] = f"forged_{names[0]}"
            training["feature_schema"]["ordered_names"] = names
            training["feature_schema"]["ordered_dtypes"][0]["name"] = names[0]
            provided["source_schema"]["ordered_names"] = list(names)
        elif case == "forged_feature_dtype":
            training["feature_schema"]["ordered_dtypes"][0]["dtype"] = "forged_dtype"
            provided["source_schema"]["ordered_dtypes"][0]["dtype"] = "forged_dtype"
        elif case == "forged_feature_count":
            training["feature_schema"]["ordered_names"].append("forged_feature")
            training["feature_schema"]["ordered_dtypes"].append(
                {"name": "forged_feature", "dtype": "float64"}
            )
        elif case == "forged_row_count":
            training["row_count"] = int(training["row_count"]) + 1
            training["row_position_identity_sha256"] = canonical_sha256(
                list(range(training["row_count"]))
            )
            provided["row_count"] = training["row_count"]
        elif case == "forged_feature_fingerprint":
            provided["files"]["train_features"]["sha256"] = "0" * 64
        elif case == "forged_target_identity":
            training["target"]["name"] = "forged_target"
            provided["target"]["name"] = "forged_target"
        elif case == "forged_target_value_counts":
            training["target"]["negative_count"] = (
                int(training["target"]["negative_count"]) - 1
            )
            training["target"]["positive_count"] = (
                int(training["target"]["positive_count"]) + 1
            )
            provided["target"]["negative_count"] = training["target"]["negative_count"]
            provided["target"]["positive_count"] = training["target"]["positive_count"]
        elif case == "forged_metadata_fingerprint":
            provided["files"]["metadata"]["sha256"] = "0" * 64
        else:
            # Leave body coherent except final hash after rewrite helper.
            training["target"]["dtype"] = training["target"]["dtype"]

    rewrite_dataset_identity(corrupted, mutate)
    if case == "forged_complete_dataset_identity_hash":
        path = corrupted / "dataset_identity.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["identity_sha256"] = "0" * 64
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        # Keep linked hashes matching the forged identity hash so only the
        # authoritative reconstruction can catch the forgery.
        search_identity = json.loads(
            (corrupted / "search_identity.json").read_text(encoding="utf-8")
        )
        search_identity["dataset_identity_sha256"] = value["identity_sha256"]
        path_si = corrupted / "search_identity.json"
        path_si.write_text(
            json.dumps(search_identity, indent=2) + "\n", encoding="utf-8"
        )
        study = json.loads(
            (corrupted / "study_summary.json").read_text(encoding="utf-8")
        )
        study["dataset_identity_sha256"] = value["identity_sha256"]
        (corrupted / "study_summary.json").write_text(
            json.dumps(study, indent=2) + "\n",
            encoding="utf-8",
        )
    reauthenticate(corrupted)
    _reject_inspect_and_export(corrupted)


@pytest.mark.parametrize(
    "case",
    (
        "extra_fold_metric_row",
        "missing_fold_metric_row",
        "duplicate_metric_row",
        "forged_metric_candidate_identity",
        "forged_adapter_identity",
        "wrong_repeat_fold",
        "one_ulp_objective",
        "one_ulp_fold_metric",
        "one_ulp_repeat_aggregate",
    ),
)
def test_exact_metric_universe_and_ulp_mutations_are_rejected(
    authorized_completed_search: tuple[Path, Path],
    case: str,
) -> None:
    valid, operational = authorized_completed_search
    corrupted = _corrupt_copy(valid, operational, case)
    metrics_path = corrupted / "trial_metrics.csv"
    trials_path = corrupted / "trials.csv"
    metrics = pd.read_csv(metrics_path)
    trials = pd.read_csv(trials_path)
    complete = int(trials.loc[trials["state"] == "COMPLETE", "trial_number"].iloc[0])
    fold_index = metrics.index[
        (metrics["trial_number"] == complete) & (metrics["record_type"] == "fold")
    ][0]
    repeat_index = metrics.index[
        (metrics["trial_number"] == complete) & (metrics["record_type"] == "repeat")
    ][0]
    if case == "extra_fold_metric_row":
        extra = metrics.loc[[fold_index]].copy()
        extra["fold"] = 99
        metrics = pd.concat([metrics, extra], ignore_index=True)
        metrics.to_csv(metrics_path, index=False)
    elif case == "missing_fold_metric_row":
        metrics = metrics.drop(index=fold_index).reset_index(drop=True)
        metrics.to_csv(metrics_path, index=False)
    elif case == "duplicate_metric_row":
        metrics = pd.concat([metrics, metrics.loc[[fold_index]]], ignore_index=True)
        metrics.to_csv(metrics_path, index=False)
    elif case == "forged_metric_candidate_identity":
        metrics.loc[fold_index, "candidate_identity_sha256"] = "0" * 64
        metrics.to_csv(metrics_path, index=False)
    elif case == "forged_adapter_identity":
        metrics.loc[fold_index, "adapter_id"] = "forged_adapter"
        metrics.to_csv(metrics_path, index=False)
    elif case == "wrong_repeat_fold":
        metrics.loc[fold_index, "fold"] = 99
        metrics.to_csv(metrics_path, index=False)
    elif case == "one_ulp_objective":
        trial_index = trials.index[trials["trial_number"] == complete][0]
        trials.loc[trial_index, "objective"] = one_ulp_away(
            float(trials.loc[trial_index, "objective"])
        )
        trials.to_csv(trials_path, index=False)
    elif case == "one_ulp_fold_metric":
        metrics.loc[fold_index, "balanced_accuracy"] = one_ulp_away(
            float(metrics.loc[fold_index, "balanced_accuracy"])
        )
        metrics.to_csv(metrics_path, index=False)
    else:
        metrics.loc[repeat_index, "balanced_accuracy"] = one_ulp_away(
            float(metrics.loc[repeat_index, "balanced_accuracy"])
        )
        metrics.to_csv(metrics_path, index=False)
    reauthenticate(corrupted)
    _reject_inspect_and_export(corrupted)


@pytest.mark.parametrize(
    "case",
    (
        "swap_execution_with_interruption",
        "swap_interruption_with_execution",
        "inconsistent_stage_and_reason",
        "missing_failure_evidence",
        "malformed_failure_evidence",
        "failed_with_completed_objective",
        "completed_with_failure_evidence",
    ),
)
def test_failure_provenance_mutations_are_rejected(
    authorized_completed_search: tuple[Path, Path],
    case: str,
) -> None:
    valid, operational = authorized_completed_search
    corrupted = _corrupt_copy(valid, operational, case)
    trials_path = corrupted / "trials.csv"
    trials = pd.read_csv(trials_path)
    failed = trials.index[trials["state"] == "FAIL"][0]
    complete = trials.index[trials["state"] == "COMPLETE"][0]
    if case == "swap_execution_with_interruption":
        evidence = json.loads(trials.loc[failed, "failure_evidence_json"])
        evidence["failure_stage"] = "interrupted_running_trial_recovery"
        evidence["failure_class"] = "INTERRUPTED_PROCESS_RECOVERY"
        evidence["interrupted_recovery"] = True
        evidence["exception_type"] = None
        evidence["exception_message_sha256"] = None
        trials.loc[failed, "failure_evidence_json"] = json.dumps(
            evidence,
            sort_keys=True,
            separators=(",", ":"),
        )
        trials.loc[failed, "failure_reason_code"] = "INTERRUPTED_PROCESS_RECOVERY"
    elif case == "swap_interruption_with_execution":
        # Force an interruption-coded row onto the failed trial and claim execution.
        evidence = build_trial_failure_evidence(
            trial_number=int(trials.loc[failed, "trial_number"]),
            error=None,
            started_at_utc="2020-01-01T00:00:00+00:00",
            failed_at_utc="2020-01-01T00:00:01+00:00",
            interrupted_recovery=True,
        )
        trials.loc[failed, "failure_evidence_json"] = json.dumps(
            evidence,
            sort_keys=True,
            separators=(",", ":"),
        )
        trials.loc[failed, "failure_reason_code"] = "TRIAL_EXECUTION_FAILED"
    elif case == "inconsistent_stage_and_reason":
        evidence = json.loads(trials.loc[failed, "failure_evidence_json"])
        evidence["failure_stage"] = "objective_execution"
        evidence["failure_class"] = "INTERRUPTED_PROCESS_RECOVERY"
        trials.loc[failed, "failure_evidence_json"] = json.dumps(
            evidence,
            sort_keys=True,
            separators=(",", ":"),
        )
        trials.loc[failed, "failure_reason_code"] = "INTERRUPTED_PROCESS_RECOVERY"
    elif case == "missing_failure_evidence":
        trials.loc[failed, "failure_evidence_json"] = "null"
    elif case == "malformed_failure_evidence":
        trials.loc[failed, "failure_evidence_json"] = "{not-json"
    elif case == "failed_with_completed_objective":
        trials.loc[failed, "objective"] = trials.loc[complete, "objective"]
        trials.loc[failed, "adapter_id"] = trials.loc[complete, "adapter_id"]
        trials.loc[failed, "candidate_identity_sha256"] = trials.loc[
            complete,
            "candidate_identity_sha256",
        ]
    else:
        evidence = build_trial_failure_evidence(
            trial_number=int(trials.loc[complete, "trial_number"]),
            error=RuntimeError("forged"),
            started_at_utc="2020-01-01T00:00:00+00:00",
            failed_at_utc="2020-01-01T00:00:01+00:00",
        )
        trials.loc[complete, "failure_evidence_json"] = json.dumps(
            evidence,
            sort_keys=True,
            separators=(",", ":"),
        )
        trials.loc[complete, "failure_reason_code"] = derive_failure_reason_code(
            evidence
        )
        trials.loc[complete, "failure_message"] = "RuntimeError: forged"
    trials.to_csv(trials_path, index=False)
    reauthenticate(corrupted)
    _reject_inspect_and_export(corrupted)


@pytest.mark.parametrize(
    "relative",
    (
        "src/churn_ml/config.py",
        "src/churn_ml/manual_lightgbm.py",
        "src/churn_ml/research_manual_lightgbm.py",
        "src/churn_ml/experiment_v2_adapter.py",
        "src/churn_ml/research_config.py",
        "src/churn_ml/research_data.py",
        "src/churn_ml/run_artifacts.py",
        "src/churn_ml/optuna_search_objective.py",
        "src/churn_ml/experiment_v2_numeric_adapter.py",
    ),
)
def test_actual_source_byte_mutation_changes_closure_and_rejects_resume(
    relative: str,
) -> None:
    from src.churn_ml.optuna_search_config import load_optuna_search_config

    ensure_train_only_files()
    with tempfile.TemporaryDirectory() as temporary:
        temp_root = Path(temporary)
        for path in (
            "src",
            "scripts/run_optuna_search.py",
            "configs/optuna",
            "configs/research_v2",
            "pyproject.toml",
        ):
            source = PROJECT_ROOT / path
            destination = temp_root / path
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        config = load_optuna_search_config(
            PROJECT_ROOT / "configs/optuna/xgboost_numeric_v1_smoke.yaml",
            project_root=PROJECT_ROOT,
        )
        contract_paths = (
            config.base_config.source_path.relative_to(PROJECT_ROOT).as_posix(),
            config.base_config.plan_path.relative_to(PROJECT_ROOT).as_posix(),
            config.search_space.source_path.relative_to(PROJECT_ROOT).as_posix(),
        )
        before = build_resume_authentication(
            project_root=temp_root,
            adapter_id=config.adapter_id,
            contract_paths=contract_paths,
        )
        study = _FakeStudy()
        config_for_resume = deepcopy(config)
        config_for_resume.resume_authentication.clear()
        config_for_resume.resume_authentication.update(deepcopy(before))
        _verify_or_initialize_study(
            study,
            config_for_resume,
            dataset_identity_sha256="1" * 64,
            assignment_identity_sha256="2" * 64,
        )
        target = temp_root / relative
        assert target.is_file()
        target.write_bytes(target.read_bytes() + b"\n# mutation\n")
        after = build_resume_authentication(
            project_root=temp_root,
            adapter_id=config.adapter_id,
            contract_paths=contract_paths,
        )
        assert (
            after["source_closure"]["identity_sha256"]
            != before["source_closure"]["identity_sha256"]
        )
        config_for_resume.resume_authentication.clear()
        config_for_resume.resume_authentication.update(after)
        with pytest.raises(Exception, match="identity differs"):
            _verify_or_initialize_study(
                study,
                config_for_resume,
                dataset_identity_sha256="1" * 64,
                assignment_identity_sha256="2" * 64,
            )


def test_import_graph_dependency_added_and_removed_change_closure() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        temp_root = Path(temporary)
        shutil.copytree(PROJECT_ROOT / "src", temp_root / "src")
        scripts = temp_root / "scripts"
        scripts.mkdir(parents=True)
        shutil.copy2(
            PROJECT_ROOT / "scripts/run_optuna_search.py",
            scripts / "run_optuna_search.py",
        )
        before = build_source_closure(project_root=temp_root)
        helper = temp_root / "src/churn_ml/optuna_search_auth_helper.py"
        helper.write_text("VALUE = 1\n", encoding="utf-8")
        target = temp_root / "src/churn_ml/optuna_search_lifecycle.py"
        original = target.read_text(encoding="utf-8")
        target.write_text(
            "from src.churn_ml.optuna_search_auth_helper import VALUE\n" + original,
            encoding="utf-8",
        )
        added = build_source_closure(project_root=temp_root)
        assert any(
            record["path"] == "src/churn_ml/optuna_search_auth_helper.py"
            for record in added["files"]
        )
        assert added["identity_sha256"] != before["identity_sha256"]
        helper.unlink()
        with pytest.raises(OptunaSearchResumeIdentityError):
            build_source_closure(project_root=temp_root)
        target.write_text(original, encoding="utf-8")
        restored = build_source_closure(project_root=temp_root)
        assert restored["identity_sha256"] == before["identity_sha256"]


def test_malformed_and_missing_closure_attributes_are_rejected() -> None:
    from src.churn_ml.optuna_search_config import load_optuna_search_config

    config = load_optuna_search_config(
        PROJECT_ROOT / "configs/optuna/xgboost_numeric_v1_smoke.yaml",
        project_root=PROJECT_ROOT,
    )
    study = _FakeStudy()
    _verify_or_initialize_study(
        study,
        config,
        dataset_identity_sha256="1" * 64,
        assignment_identity_sha256="2" * 64,
    )
    study.user_attrs["source_closure"].pop("discovery")
    with pytest.raises(Exception, match="identity differs"):
        _verify_or_initialize_study(
            study,
            config,
            dataset_identity_sha256="1" * 64,
            assignment_identity_sha256="2" * 64,
        )


def test_valid_unchanged_source_closure_resume_is_deterministic() -> None:
    from src.churn_ml.optuna_search_config import load_optuna_search_config

    first = build_source_closure(project_root=PROJECT_ROOT)
    second = build_source_closure(project_root=PROJECT_ROOT)
    assert first == second
    required = {
        "src/churn_ml/config.py",
        "src/churn_ml/manual_lightgbm.py",
        "src/churn_ml/research_manual_lightgbm.py",
    }
    paths = {record["path"] for record in first["files"]}
    assert required <= paths
    config = load_optuna_search_config(
        PROJECT_ROOT / "configs/optuna/xgboost_numeric_v1_smoke.yaml",
        project_root=PROJECT_ROOT,
    )
    study = _FakeStudy()
    _verify_or_initialize_study(
        study,
        config,
        dataset_identity_sha256="1" * 64,
        assignment_identity_sha256="2" * 64,
    )
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
