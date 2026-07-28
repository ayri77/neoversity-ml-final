from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import pandas as pd

from sklearn.metrics import balanced_accuracy_score

from src.churn_ml.experiment_v2 import CandidateAdapter
from src.churn_ml.experiment_v2_adapter import ExperimentV2AdapterContractError
from src.churn_ml.experiment_v2_contract import ExperimentV2ContractError
from src.churn_ml.experiment_v2_numeric_adapter import (
    ExperimentV2AdapterDependencyError,
)
from src.churn_ml.optuna_search_artifacts import _write_csv, write_completed_search
from src.churn_ml.optuna_search_config import OptunaSearchConfig
from src.churn_ml.optuna_search_export import build_best_candidate_config
from src.churn_ml.optuna_search_objective import (
    OptunaSearchObjectiveError,
    SearchAssignments,
    TrialEvaluation,
    evaluate_trial,
)
from src.churn_ml.optuna_search_resume import build_resume_authentication
from src.churn_ml.optuna_search_space import (
    build_resolved_adapter_contract,
    stateless_sample,
    suggest_parameters,
)
from src.churn_ml.research_data import canonical_sha256
from src.churn_ml.research_protocol import ResearchProtocolError
from src.churn_ml.research_v2_config import ResearchV2Config
from src.churn_ml.research_v2_data import load_research_v2_training_data


FAILURE_EVIDENCE_SCHEMA_VERSION = 1
FAILURE_STAGE_OBJECTIVE_EXECUTION = "objective_execution"
FAILURE_STAGE_INTERRUPTED_RECOVERY = "interrupted_running_trial_recovery"
FAILURE_STAGE_CONFIGURATION_AUTHENTICATION = (
    "configuration_source_runtime_authentication"
)

FAILURE_CLASS_TO_REASON = {
    "ADAPTER_CONTRACT_OR_FIT_FAILED": "ADAPTER_CONTRACT_OR_FIT_FAILED",
    "INTERRUPTED_BY_USER": "INTERRUPTED_BY_USER",
    "INTERRUPTED_PROCESS_RECOVERY": "INTERRUPTED_PROCESS_RECOVERY",
    "LEAKAGE_OR_OBJECTIVE_CONTRACT_FAILED": "LEAKAGE_OR_OBJECTIVE_CONTRACT_FAILED",
    "TRIAL_EXECUTION_FAILED": "TRIAL_EXECUTION_FAILED",
    "CONFIGURATION_SOURCE_RUNTIME_AUTHENTICATION_FAILED": (
        "CONFIGURATION_SOURCE_RUNTIME_AUTHENTICATION_FAILED"
    ),
}

FAILURE_STAGE_TO_CLASS = {
    FAILURE_STAGE_OBJECTIVE_EXECUTION: None,
    FAILURE_STAGE_INTERRUPTED_RECOVERY: "INTERRUPTED_PROCESS_RECOVERY",
    FAILURE_STAGE_CONFIGURATION_AUTHENTICATION: (
        "CONFIGURATION_SOURCE_RUNTIME_AUTHENTICATION_FAILED"
    ),
}


class OptunaSearchLifecycleError(RuntimeError):
    """Raised when deterministic study or cache lifecycle guarantees fail."""


@dataclass(frozen=True)
class StudyExecutionResult:
    search_dir: Path
    study_summary: dict[str, Any]
    best_trial: dict[str, Any]


def run_optuna_study(
    config: OptunaSearchConfig,
    *,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_identity: Mapping[str, Any],
    assignments: SearchAssignments,
    adapter: CandidateAdapter,
) -> StudyExecutionResult:
    """Create or resume a local deterministic Optuna study and freeze its report."""
    import optuna
    from optuna.trial import TrialState

    started_at = datetime.now(timezone.utc)
    started = perf_counter()
    authoritative_dataset_identity = _authoritative_dataset_identity(
        X, y, dataset_identity
    )
    source_provenance = build_source_provenance(config)
    storage_path = config.storage_path
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    cache_root = (
        storage_path.parent / "trial_cache" / config.study_identity_sha256
    ).resolve()
    if storage_path.parent not in cache_root.parents:
        raise OptunaSearchLifecycleError("Trial cache escapes Optuna storage root.")
    cache_root.mkdir(parents=True, exist_ok=True)
    sampler = _make_stateless_sampler(
        optuna,
        seed=int(config.payload["sampler"]["seed"]),
        study_name=str(config.payload["study_name"]),
    )
    study = optuna.create_study(
        study_name=str(config.payload["study_name"]),
        storage=f"sqlite:///{storage_path.as_posix()}",
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )
    _verify_or_initialize_study(
        study,
        config,
        dataset_identity_sha256=canonical_sha256(authoritative_dataset_identity),
        assignment_identity_sha256=canonical_sha256(assignments.identity),
    )
    recovered = _recover_interrupted_trials(study, TrialState)
    target = int(config.payload["n_trials"])
    previous_target = int(study.user_attrs.get("maximum_requested_n_trials", 0))
    if previous_target and target < previous_target:
        raise OptunaSearchLifecycleError(
            "Requested n_trials is below the study's previous explicit target."
        )
    if target > previous_target:
        study.set_user_attr("maximum_requested_n_trials", target)
    existing = len(study.get_trials(deepcopy=False))
    if existing > target:
        raise OptunaSearchLifecycleError(
            "Study already contains more trials than the requested target."
        )

    def objective(trial: Any) -> float:
        started_at_utc = datetime.now(timezone.utc).isoformat()
        try:
            tuned = suggest_parameters(trial, config.search_space)
            contract = build_resolved_adapter_contract(
                config.base_config.adapter_contract,
                adapter_id=config.adapter_id,
                tuned_parameters=tuned,
            )
            adapter.validate_contract(contract)
            candidate_identity_sha256 = canonical_sha256(
                {
                    "schema_version": 1,
                    "adapter_id": config.adapter_id,
                    "resolved_parameters": tuned,
                }
            )
            result = evaluate_trial(
                X,
                y,
                assignments,
                adapter_contract=contract,
                threshold_policy=config.threshold_policy_payload,
                fit_predict=adapter.fit_predict,
                trial_number=int(trial.number),
                adapter_id=config.adapter_id,
                candidate_identity_sha256=candidate_identity_sha256,
            )
            cache_identity, authoritative_objective = _write_trial_cache(
                cache_root,
                trial_number=int(trial.number),
                evaluation=result,
            )
            trial.set_user_attr("resolved_parameters", tuned)
            trial.set_user_attr("adapter_id", config.adapter_id)
            trial.set_user_attr(
                "candidate_identity_sha256",
                candidate_identity_sha256,
            )
            trial.set_user_attr("prediction_key_coverage", result.coverage)
            trial.set_user_attr("cache_identity", cache_identity)
            trial.set_user_attr("failure_reason_code", None)
            trial.set_user_attr("failure_message", None)
            trial.set_user_attr("failure_evidence", None)
            return authoritative_objective
        except BaseException as error:
            try:
                evidence = build_trial_failure_evidence(
                    trial_number=int(trial.number),
                    error=error,
                    started_at_utc=started_at_utc,
                    failed_at_utc=datetime.now(timezone.utc).isoformat(),
                )
                trial.set_user_attr(
                    "failure_reason_code",
                    derive_failure_reason_code(evidence),
                )
                trial.set_user_attr(
                    "failure_message",
                    f"{type(error).__name__}: {error}",
                )
                trial.set_user_attr("failure_evidence", evidence)
            except BaseException:
                pass
            raise

    remaining = target - existing
    if remaining:
        study.optimize(
            objective,
            n_trials=remaining,
            timeout=float(config.payload["timeout_seconds"]),
            catch=(Exception,),
            gc_after_trial=True,
            show_progress_bar=False,
        )
    trials = study.get_trials(deepcopy=False)
    if len(trials) != target:
        raise OptunaSearchLifecycleError(
            f"Study stopped before target trials: {len(trials)}/{target}."
        )
    complete = [trial for trial in trials if trial.state == TrialState.COMPLETE]
    if not complete:
        raise OptunaSearchLifecycleError("Study has no completed trial.")
    best = min(
        complete,
        key=lambda trial: (-_required_trial_value(trial), trial.number),
    )
    best_value = _required_trial_value(best)
    trial_table = _build_trial_table(trials)
    trial_metrics, trial_predictions = _load_completed_trial_caches(
        cache_root,
        complete,
    )
    resolved_parameters = best.user_attrs.get("resolved_parameters")
    if not isinstance(resolved_parameters, dict):
        raise OptunaSearchLifecycleError("Best trial lacks strict resolved parameters.")
    best_candidate = build_best_candidate_config(
        config,
        best_trial_number=int(best.number),
        resolved_parameters=resolved_parameters,
    )
    best_payload = {
        "schema_version": 1,
        "trial_number": int(best.number),
        "objective": best_value,
        "direction": "maximize",
        "tie_break": "lowest_trial_number",
        "resolved_parameters": resolved_parameters,
        "candidate_identity_sha256": best.user_attrs["candidate_identity_sha256"],
        "candidate_config_sha256": canonical_sha256(best_candidate),
        "prediction_key_coverage": best.user_attrs["prediction_key_coverage"],
        "evidence_scope": "tuning_only_not_unbiased_final_evidence",
    }
    state_counts = {
        state.name.lower(): sum(trial.state == state for trial in trials)
        for state in TrialState
    }
    study_summary = {
        "schema_version": 1,
        "study_name": config.payload["study_name"],
        "search_id": config.search_id,
        "study_identity_sha256": config.study_identity_sha256,
        "search_identity_sha256": config.search_identity_sha256,
        "resume_authentication_sha256": config.resume_authentication["identity_sha256"],
        "dataset_identity_sha256": canonical_sha256(authoritative_dataset_identity),
        "assignment_identity_sha256": canonical_sha256(assignments.identity),
        "direction": "maximize",
        "metric": "balanced_accuracy",
        "objective_aggregation": "mean_fold_balanced_accuracy_across_repeats",
        "requested_trials": target,
        "actual_trials": len(trials),
        "state_counts": state_counts,
        "recovered_interrupted_trials": recovered,
        "best_trial_number": int(best.number),
        "best_objective": best_value,
        "storage": storage_path.relative_to(config.project_root).as_posix(),
        "filesystem_report_authoritative": True,
        "optuna_storage_role": "resumable_operational_state_only",
        "final_evaluation_status": "not_run",
    }
    current_resume_identity = _current_resume_authentication(config)
    if current_resume_identity != config.resume_authentication:
        raise OptunaSearchLifecycleError(
            "Source/runtime identity changed during the search; report refused."
        )
    runtime = {
        "schema_version": 1,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": perf_counter() - started,
        "network_access": "disabled_during_study",
        "competition_test_accessed": False,
        "submission_created": False,
        "selection_plan_run": False,
        "confirmation_plan_run": False,
    }
    search_dir = write_completed_search(
        config,
        dataset_identity=authoritative_dataset_identity,
        assignment_identity=assignments.identity,
        search_space_payload={
            "schema_version": 1,
            "search_space_id": config.search_space.search_space_id,
            "sha256": config.search_space.sha256,
            "canonical": config.search_space.payload,
        },
        study_summary=study_summary,
        trials=trial_table,
        trial_metrics=trial_metrics,
        trial_predictions=trial_predictions,
        best_trial=best_payload,
        best_candidate_config=best_candidate,
        runtime=runtime,
        environment=environment_identity(config.adapter_id),
        source_provenance=source_provenance,
        resume_authentication=config.resume_authentication,
        fold_assignments=assignments.folds,
        threshold_membership=assignments.threshold_membership,
    )
    return StudyExecutionResult(
        search_dir=search_dir,
        study_summary=study_summary,
        best_trial=best_payload,
    )


def build_source_provenance(config: OptunaSearchConfig) -> dict[str, Any]:
    source = config.resume_authentication["source_closure"]
    canonical = {
        "schema_version": 1,
        "hashing_method": source["hashing_method"],
        "files": deepcopy(source["files"]),
    }
    return {**canonical, "sha256": canonical_sha256(canonical)}


def portable_dataset_identity(
    identity: Mapping[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    result = json.loads(json.dumps(identity))
    files = result.get("files", {})
    if not isinstance(files, dict):
        raise OptunaSearchLifecycleError("Dataset identity files must be a mapping.")
    for item in files.values():
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise OptunaSearchLifecycleError("Dataset identity record is malformed.")
        path = Path(item["path"]).resolve()
        if project_root.resolve() not in path.parents:
            raise OptunaSearchLifecycleError(
                "Dataset identity path escapes repository."
            )
        item["path"] = path.relative_to(project_root.resolve()).as_posix()
    return result


def reconstruct_authoritative_dataset_identity_from_plan(
    base_config: ResearchV2Config,
    *,
    project_root: Path,
) -> tuple[dict[str, Any], pd.Series, pd.DataFrame]:
    """Rebuild trusted dataset identity from the validated plan and train-only files."""
    data = load_research_v2_training_data(base_config)
    provided = portable_dataset_identity(
        data.fingerprints,
        project_root=project_root,
    )
    identity = _authoritative_dataset_identity(data.X, data.y, provided)
    return identity, data.y, data.X


def build_trial_failure_evidence(
    *,
    trial_number: int,
    error: BaseException | None,
    started_at_utc: str,
    failed_at_utc: str,
    failure_stage: str = FAILURE_STAGE_OBJECTIVE_EXECUTION,
    interrupted_recovery: bool = False,
) -> dict[str, Any]:
    """Persist deterministic structured failure provenance for one trial."""
    if interrupted_recovery:
        failure_class = "INTERRUPTED_PROCESS_RECOVERY"
        exception_type = None
        exception_message_sha256 = None
        failure_stage = FAILURE_STAGE_INTERRUPTED_RECOVERY
    elif failure_stage == FAILURE_STAGE_CONFIGURATION_AUTHENTICATION:
        failure_class = "CONFIGURATION_SOURCE_RUNTIME_AUTHENTICATION_FAILED"
        exception_type = None if error is None else type(error).__name__
        exception_message_sha256 = (
            None
            if error is None
            else hashlib.sha256(str(error).encode("utf-8")).hexdigest()
        )
    else:
        if error is None:
            raise OptunaSearchLifecycleError(
                "Objective-execution failure evidence requires an exception."
            )
        failure_class = _failure_class_for_exception(error)
        exception_type = type(error).__name__
        exception_message_sha256 = hashlib.sha256(
            str(error).encode("utf-8")
        ).hexdigest()
        failure_stage = FAILURE_STAGE_OBJECTIVE_EXECUTION
        interrupted_recovery = False
    return {
        "schema_version": FAILURE_EVIDENCE_SCHEMA_VERSION,
        "trial_number": int(trial_number),
        "failure_stage": failure_stage,
        "failure_class": failure_class,
        "exception_type": exception_type,
        "exception_message_sha256": exception_message_sha256,
        "interrupted_recovery": bool(interrupted_recovery),
        "started_at_utc": started_at_utc,
        "failed_at_utc": failed_at_utc,
    }


def derive_failure_reason_code(evidence: Mapping[str, Any]) -> str:
    """Derive the exact persisted reason code from structured failure evidence."""
    if not isinstance(evidence, Mapping):
        raise OptunaSearchLifecycleError("Failure evidence must be a mapping.")
    required = {
        "schema_version",
        "trial_number",
        "failure_stage",
        "failure_class",
        "exception_type",
        "exception_message_sha256",
        "interrupted_recovery",
        "started_at_utc",
        "failed_at_utc",
    }
    if set(evidence) != required:
        raise OptunaSearchLifecycleError("Failure evidence schema differs.")
    if (
        type(evidence["schema_version"]) is not int
        or evidence["schema_version"] != FAILURE_EVIDENCE_SCHEMA_VERSION
        or type(evidence["trial_number"]) is not int
        or evidence["trial_number"] < 0
        or type(evidence["failure_stage"]) is not str
        or type(evidence["failure_class"]) is not str
        or type(evidence["interrupted_recovery"]) is not bool
        or type(evidence["started_at_utc"]) is not str
        or type(evidence["failed_at_utc"]) is not str
        or not evidence["started_at_utc"]
        or not evidence["failed_at_utc"]
    ):
        raise OptunaSearchLifecycleError("Failure evidence values are malformed.")
    stage = evidence["failure_stage"]
    failure_class = evidence["failure_class"]
    if stage not in FAILURE_STAGE_TO_CLASS:
        raise OptunaSearchLifecycleError(f"Unsupported failure stage: {stage}.")
    stage_class = FAILURE_STAGE_TO_CLASS[stage]
    if stage == FAILURE_STAGE_INTERRUPTED_RECOVERY:
        if (
            not evidence["interrupted_recovery"]
            or failure_class != "INTERRUPTED_PROCESS_RECOVERY"
            or evidence["exception_type"] is not None
            or evidence["exception_message_sha256"] is not None
        ):
            raise OptunaSearchLifecycleError(
                "Interrupted recovery evidence is inconsistent."
            )
    elif stage == FAILURE_STAGE_CONFIGURATION_AUTHENTICATION:
        if (
            evidence["interrupted_recovery"]
            or failure_class != stage_class
            or (
                evidence["exception_type"] is not None
                and type(evidence["exception_type"]) is not str
            )
            or (
                evidence["exception_message_sha256"] is not None
                and (
                    type(evidence["exception_message_sha256"]) is not str
                    or len(evidence["exception_message_sha256"]) != 64
                )
            )
        ):
            raise OptunaSearchLifecycleError(
                "Configuration authentication failure evidence is inconsistent."
            )
    else:
        if (
            evidence["interrupted_recovery"]
            or stage_class is not None
            or type(evidence["exception_type"]) is not str
            or not evidence["exception_type"]
            or type(evidence["exception_message_sha256"]) is not str
            or len(evidence["exception_message_sha256"]) != 64
            or failure_class
            not in {
                "ADAPTER_CONTRACT_OR_FIT_FAILED",
                "INTERRUPTED_BY_USER",
                "LEAKAGE_OR_OBJECTIVE_CONTRACT_FAILED",
                "TRIAL_EXECUTION_FAILED",
            }
        ):
            raise OptunaSearchLifecycleError(
                "Objective-execution failure evidence is inconsistent."
            )
        expected_class = _failure_class_for_exception_type(evidence["exception_type"])
        if failure_class != expected_class:
            raise OptunaSearchLifecycleError(
                "Failure class does not match exception type."
            )
    reason = FAILURE_CLASS_TO_REASON.get(failure_class)
    if reason is None:
        raise OptunaSearchLifecycleError(f"Unsupported failure class: {failure_class}.")
    return reason


def _failure_class_for_exception(error: BaseException) -> str:
    return _failure_class_for_exception_type(type(error).__name__, error=error)


def _failure_class_for_exception_type(
    exception_type: str,
    *,
    error: BaseException | None = None,
) -> str:
    if exception_type == "KeyboardInterrupt" or isinstance(error, KeyboardInterrupt):
        return "INTERRUPTED_BY_USER"
    adapter_types = {
        ExperimentV2AdapterContractError.__name__,
        ExperimentV2AdapterDependencyError.__name__,
        ExperimentV2ContractError.__name__,
    }
    objective_types = {
        OptunaSearchObjectiveError.__name__,
        ResearchProtocolError.__name__,
    }
    if error is not None:
        if isinstance(
            error,
            (
                ExperimentV2AdapterContractError,
                ExperimentV2AdapterDependencyError,
                ExperimentV2ContractError,
            ),
        ):
            return "ADAPTER_CONTRACT_OR_FIT_FAILED"
        if isinstance(error, (OptunaSearchObjectiveError, ResearchProtocolError)):
            return "LEAKAGE_OR_OBJECTIVE_CONTRACT_FAILED"
    if exception_type in adapter_types:
        return "ADAPTER_CONTRACT_OR_FIT_FAILED"
    if exception_type in objective_types:
        return "LEAKAGE_OR_OBJECTIVE_CONTRACT_FAILED"
    return "TRIAL_EXECUTION_FAILED"


def _authoritative_dataset_identity(
    X: pd.DataFrame,
    y: pd.Series,
    provided_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if len(X) != len(y) or not X.index.equals(y.index):
        raise OptunaSearchLifecycleError("Training data identity rows are not aligned.")
    if not y.index.equals(pd.RangeIndex(len(y))):
        raise OptunaSearchLifecycleError("Training row identity must be a RangeIndex.")
    provided = json.loads(json.dumps(provided_identity))
    training_data = {
        "row_count": len(y),
        "row_position_identity_sha256": canonical_sha256(list(range(len(y)))),
        "feature_schema": {
            "ordered_names": [str(name) for name in X.columns],
            "ordered_dtypes": [
                {"name": str(name), "dtype": str(dtype)}
                for name, dtype in X.dtypes.items()
            ],
        },
        "target": {
            "name": None if y.name is None else str(y.name),
            "dtype": str(y.dtype),
            "negative_count": int((y == 0).sum()),
            "positive_count": int((y == 1).sum()),
            "values_sha256": canonical_sha256(
                {
                    "name": None if y.name is None else str(y.name),
                    "dtype": str(y.dtype),
                    "values": [int(value) for value in y.tolist()],
                }
            ),
        },
    }
    canonical = {
        "schema_version": 1,
        "provided_identity": provided,
        "provided_identity_sha256": canonical_sha256(provided),
        "training_data": training_data,
    }
    return {**canonical, "identity_sha256": canonical_sha256(canonical)}


def environment_identity(adapter_id: str) -> dict[str, Any]:
    runtime = build_resume_authentication(
        project_root=Path(__file__).resolve().parents[2],
        adapter_id=adapter_id,
    )["runtime_dependencies"]
    return {
        "schema_version": 1,
        "runtime_dependencies": runtime,
        "adapter_id": adapter_id,
        "device_policy": "cpu_only",
        "thread_count": 1,
    }


def _make_stateless_sampler(
    optuna: Any,
    *,
    seed: int,
    study_name: str,
) -> Any:
    class StatelessRandomSampler(optuna.samplers.BaseSampler):
        def infer_relative_search_space(
            self,
            study: Any,
            trial: Any,
        ) -> dict[str, Any]:
            del study, trial
            return {}

        def sample_relative(
            self,
            study: Any,
            trial: Any,
            search_space: Mapping[str, Any],
        ) -> dict[str, Any]:
            del study, trial, search_space
            return {}

        def sample_independent(
            self,
            study: Any,
            trial: Any,
            param_name: str,
            param_distribution: Any,
        ) -> Any:
            del study
            return stateless_sample(
                seed=seed,
                study_name=study_name,
                trial_number=int(trial.number),
                parameter_name=param_name,
                distribution=param_distribution,
            )

    return StatelessRandomSampler()


def _verify_or_initialize_study(
    study: Any,
    config: OptunaSearchConfig,
    *,
    dataset_identity_sha256: str,
    assignment_identity_sha256: str,
) -> None:
    existing = study.user_attrs.get("study_identity_sha256")
    expected_attrs = {
        "study_identity_sha256": config.study_identity_sha256,
        "study_identity": config.study_identity,
        "schema_version": 1,
        "resume_authentication": config.resume_authentication,
        "resume_authentication_sha256": config.resume_authentication["identity_sha256"],
        "source_closure": config.resume_authentication["source_closure"],
        "runtime_dependencies": config.resume_authentication["runtime_dependencies"],
        "dataset_identity_sha256": dataset_identity_sha256,
        "assignment_identity_sha256": assignment_identity_sha256,
    }
    if existing is None:
        if study.get_trials(deepcopy=False):
            raise OptunaSearchLifecycleError(
                "Existing study lacks resume authentication and is unsupported."
            )
        for name, value in expected_attrs.items():
            study.set_user_attr(name, value)
        return
    if any(
        study.user_attrs.get(name) != value for name, value in expected_attrs.items()
    ):
        raise OptunaSearchLifecycleError(
            "Existing study source/runtime/data identity differs from the request."
        )


def _current_resume_authentication(config: OptunaSearchConfig) -> dict[str, Any]:
    root = config.project_root
    return build_resume_authentication(
        project_root=root,
        adapter_id=config.adapter_id,
        contract_paths=(
            config.base_config.source_path.relative_to(root).as_posix(),
            config.base_config.plan_path.relative_to(root).as_posix(),
            config.search_space.source_path.relative_to(root).as_posix(),
        ),
    )


def _recover_interrupted_trials(study: Any, trial_state: Any) -> int:
    recovered = 0
    for trial in study.get_trials(deepcopy=False, states=(trial_state.RUNNING,)):
        started = (
            datetime.now(timezone.utc).isoformat()
            if trial.datetime_start is None
            else trial.datetime_start.astimezone(timezone.utc).isoformat()
        )
        failed_at = datetime.now(timezone.utc).isoformat()
        evidence = build_trial_failure_evidence(
            trial_number=int(trial.number),
            error=None,
            started_at_utc=started,
            failed_at_utc=failed_at,
            interrupted_recovery=True,
        )
        study._storage.set_trial_user_attr(  # noqa: SLF001
            trial._trial_id,  # noqa: SLF001
            "failure_reason_code",
            derive_failure_reason_code(evidence),
        )
        study._storage.set_trial_user_attr(  # noqa: SLF001
            trial._trial_id,  # noqa: SLF001
            "failure_message",
            "Recovered RUNNING trial from an interrupted prior process.",
        )
        study._storage.set_trial_user_attr(  # noqa: SLF001
            trial._trial_id,  # noqa: SLF001
            "failure_evidence",
            evidence,
        )
        changed = study._storage.set_trial_state_values(  # noqa: SLF001
            trial._trial_id,  # noqa: SLF001
            trial_state.FAIL,
        )
        if changed:
            recovered += 1
    return recovered


def _write_trial_cache(
    cache_root: Path,
    *,
    trial_number: int,
    evaluation: TrialEvaluation,
) -> tuple[dict[str, Any], float]:
    directory = cache_root / f"trial_{trial_number:06d}"
    directory.mkdir(exist_ok=False)
    paths = {
        "fold_metrics": directory / "fold_metrics.csv",
        "repeat_metrics": directory / "repeat_metrics.csv",
        "predictions": directory / "predictions.csv",
    }
    # Persist predictions first, then reload the exact CSV float tokens and bind
    # metrics/objective to those authoritative prediction floats.
    _write_csv(paths["predictions"], evaluation.predictions)
    with paths["predictions"].open("r", encoding="utf-8", newline="") as handle:
        persisted_predictions = pd.DataFrame(list(csv.DictReader(handle)))
    for column in ("probability", "selected_threshold", "target", "prediction"):
        if column in persisted_predictions.columns:
            persisted_predictions[column] = [
                float(value)
                if column in {"probability", "selected_threshold"}
                else int(value)
                for value in persisted_predictions[column].tolist()
            ]
    for column in ("repeat", "fold", "row_position", "repeat_seed", "trial_number"):
        if column in persisted_predictions.columns:
            persisted_predictions[column] = [
                int(value) for value in persisted_predictions[column].tolist()
            ]
    fold_metrics = evaluation.fold_metrics.copy()
    for index, row in fold_metrics.iterrows():
        scoring = persisted_predictions.loc[
            (persisted_predictions["repeat"] == row["repeat"])
            & (persisted_predictions["fold"] == row["fold"])
        ]
        labels = (
            scoring["probability"].to_numpy(dtype=float)
            >= float(scoring["selected_threshold"].iloc[0])
        ).astype("int8")
        targets = scoring["target"].to_numpy(dtype="int8")
        fold_metrics.at[index, "balanced_accuracy"] = float(
            format(float(balanced_accuracy_score(targets, labels)), ".17g")
        )
        fold_metrics.at[index, "selected_threshold"] = float(
            format(float(scoring["selected_threshold"].iloc[0]), ".17g")
        )
    repeat_metrics = (
        fold_metrics.groupby("repeat", sort=True, as_index=False)
        .agg(
            repeat_seed=("repeat_seed", "first"),
            fold_count=("fold", "count"),
            balanced_accuracy=("balanced_accuracy", "mean"),
        )
        .assign(
            trial_number=int(evaluation.fold_metrics["trial_number"].iloc[0]),
            adapter_id=str(evaluation.fold_metrics["adapter_id"].iloc[0]),
            candidate_identity_sha256=str(
                evaluation.fold_metrics["candidate_identity_sha256"].iloc[0]
            ),
        )
    )
    repeat_metrics["balanced_accuracy"] = [
        float(format(float(value), ".17g"))
        for value in repeat_metrics["balanced_accuracy"].tolist()
    ]
    repeat_metrics = repeat_metrics[
        [
            "trial_number",
            "adapter_id",
            "candidate_identity_sha256",
            "repeat",
            "repeat_seed",
            "fold_count",
            "balanced_accuracy",
        ]
    ]
    _write_csv(paths["fold_metrics"], fold_metrics)
    _write_csv(paths["repeat_metrics"], repeat_metrics)
    authoritative_objective = float(
        format(float(repeat_metrics["balanced_accuracy"].mean()), ".17g")
    )
    return (
        {
            "schema_version": 1,
            "files": {
                name: {
                    "path": path.relative_to(cache_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for name, path in sorted(paths.items())
            },
        },
        authoritative_objective,
    )


def _load_completed_trial_caches(
    cache_root: Path,
    complete_trials: list[Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for trial in sorted(complete_trials, key=lambda item: item.number):
        identity = trial.user_attrs.get("cache_identity")
        if not isinstance(identity, dict) or identity.get("schema_version") != 1:
            raise OptunaSearchLifecycleError(
                f"Completed trial {trial.number} lacks cache identity."
            )
        files = identity.get("files")
        if not isinstance(files, dict):
            raise OptunaSearchLifecycleError("Trial cache file mapping is invalid.")
        loaded: dict[str, pd.DataFrame] = {}
        for name in ("fold_metrics", "repeat_metrics", "predictions"):
            item = files.get(name)
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise OptunaSearchLifecycleError("Trial cache identity is malformed.")
            path = (cache_root / item["path"]).resolve()
            if cache_root not in path.parents or not path.is_file():
                raise OptunaSearchLifecycleError("Trial cache path is invalid.")
            data = path.read_bytes()
            if len(data) != item.get("size_bytes") or hashlib.sha256(
                data
            ).hexdigest() != item.get("sha256"):
                raise OptunaSearchLifecycleError("Trial cache bytes were modified.")
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                raise OptunaSearchLifecycleError(f"Trial cache {name} is empty.")
            frame = pd.DataFrame(rows)
            for column in frame.columns:
                sample = next(
                    (
                        value
                        for value in frame[column].tolist()
                        if value not in {"", None}
                    ),
                    "",
                )
                if sample in {"True", "False"}:
                    frame[column] = frame[column].map(
                        {"True": True, "False": False, "": None}
                    )
                elif sample == "" or sample is None:
                    continue
                else:
                    try:
                        float(sample)
                    except ValueError:
                        continue
                    else:
                        if sample.isdigit() or (
                            sample.startswith("-") and sample[1:].isdigit()
                        ):
                            frame[column] = [
                                None if value == "" else int(value)
                                for value in frame[column].tolist()
                            ]
                        else:
                            frame[column] = [
                                None if value == "" else float(value)
                                for value in frame[column].tolist()
                            ]
            loaded[name] = frame
        fold = loaded["fold_metrics"]
        repeat = loaded["repeat_metrics"].copy()
        repeat["record_type"] = "repeat"
        fold["record_type"] = "fold"
        metric_frames.extend([fold, repeat])
        prediction_frames.append(loaded["predictions"])
    metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
    nullable_integer_columns = (
        "fold",
        "training_rows",
        "validation_rows",
        "threshold_selection_rows",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "fold_count",
    )
    for name in nullable_integer_columns:
        if name in metrics.columns:
            metrics[name] = metrics[name].astype("Int64")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return metrics, predictions


def _build_trial_table(trials: list[Any]) -> pd.DataFrame:
    records = []
    for trial in sorted(trials, key=lambda item: item.number):
        duration = (
            None
            if trial.datetime_start is None or trial.datetime_complete is None
            else (trial.datetime_complete - trial.datetime_start).total_seconds()
        )
        failure_evidence = trial.user_attrs.get("failure_evidence")
        records.append(
            {
                "trial_number": int(trial.number),
                "state": trial.state.name,
                "adapter_id": trial.user_attrs.get("adapter_id"),
                "candidate_identity_sha256": trial.user_attrs.get(
                    "candidate_identity_sha256"
                ),
                "objective": None if trial.value is None else float(trial.value),
                "started_at": (
                    None
                    if trial.datetime_start is None
                    else trial.datetime_start.isoformat()
                ),
                "finished_at": (
                    None
                    if trial.datetime_complete is None
                    else trial.datetime_complete.isoformat()
                ),
                "duration_seconds": duration,
                "optuna_parameters_json": json.dumps(
                    trial.params,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "resolved_parameters_json": json.dumps(
                    trial.user_attrs.get("resolved_parameters"),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "prediction_key_coverage_json": json.dumps(
                    trial.user_attrs.get("prediction_key_coverage"),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "failure_reason_code": trial.user_attrs.get("failure_reason_code"),
                "failure_message": trial.user_attrs.get("failure_message"),
                "failure_evidence_json": json.dumps(
                    failure_evidence,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    return pd.DataFrame(records)


def _required_trial_value(trial: Any) -> float:
    value = trial.value
    if value is None:
        raise OptunaSearchLifecycleError(
            f"Completed trial {trial.number} has no objective value."
        )
    return float(value)
