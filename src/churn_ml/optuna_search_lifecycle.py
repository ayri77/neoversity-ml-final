from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import pandas as pd

from src.churn_ml.experiment_v2 import CandidateAdapter
from src.churn_ml.optuna_search_artifacts import write_completed_search
from src.churn_ml.optuna_search_config import OptunaSearchConfig
from src.churn_ml.optuna_search_export import build_best_candidate_config
from src.churn_ml.optuna_search_objective import (
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
            cache_identity = _write_trial_cache(
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
            return result.objective
        except BaseException as error:
            try:
                trial.set_user_attr(
                    "failure_reason_code",
                    _failure_reason_code(error),
                )
                trial.set_user_attr(
                    "failure_message",
                    f"{type(error).__name__}: {error}",
                )
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
        study._storage.set_trial_user_attr(  # noqa: SLF001
            trial._trial_id,  # noqa: SLF001
            "failure_reason_code",
            "INTERRUPTED_PROCESS_RECOVERY",
        )
        study._storage.set_trial_user_attr(  # noqa: SLF001
            trial._trial_id,  # noqa: SLF001
            "failure_message",
            "Recovered RUNNING trial from an interrupted prior process.",
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
) -> dict[str, Any]:
    directory = cache_root / f"trial_{trial_number:06d}"
    directory.mkdir(exist_ok=False)
    paths = {
        "fold_metrics": directory / "fold_metrics.csv",
        "repeat_metrics": directory / "repeat_metrics.csv",
        "predictions": directory / "predictions.csv",
    }
    evaluation.fold_metrics.to_csv(paths["fold_metrics"], index=False)
    evaluation.repeat_metrics.to_csv(paths["repeat_metrics"], index=False)
    evaluation.predictions.to_csv(paths["predictions"], index=False)
    return {
        "schema_version": 1,
        "files": {
            name: {
                "path": path.relative_to(cache_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in sorted(paths.items())
        },
    }


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
            loaded[name] = pd.read_csv(path)
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
            }
        )
    return pd.DataFrame(records)


def _failure_reason_code(error: BaseException) -> str:
    name = type(error).__name__
    if isinstance(error, KeyboardInterrupt):
        return "INTERRUPTED_BY_USER"
    if "Adapter" in name:
        return "ADAPTER_CONTRACT_OR_FIT_FAILED"
    if "Objective" in name or "Protocol" in name:
        return "LEAKAGE_OR_OBJECTIVE_CONTRACT_FAILED"
    return "TRIAL_EXECUTION_FAILED"


def _required_trial_value(trial: Any) -> float:
    value = trial.value
    if value is None:
        raise OptunaSearchLifecycleError(
            f"Completed trial {trial.number} has no objective value."
        )
    return float(value)
