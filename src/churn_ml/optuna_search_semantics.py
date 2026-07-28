from __future__ import annotations

import csv
import json
import math
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, NoReturn

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score

from src.churn_ml.optuna_search_config import (
    PLAN_KEYS,
    load_search_space,
)
from src.churn_ml.optuna_search_lifecycle import (
    derive_failure_reason_code,
    reconstruct_authoritative_dataset_identity_from_plan,
)
from src.churn_ml.optuna_search_objective import build_search_assignments
from src.churn_ml.optuna_search_resume import (
    OptunaSearchResumeIdentityError,
    build_resume_authentication,
    validate_resume_authentication_shape,
)
from src.churn_ml.optuna_search_space import (
    build_resolved_adapter_contract,
    stateless_sample,
)
from src.churn_ml.research_data import canonical_sha256
from src.churn_ml.research_protocol import select_balanced_accuracy_threshold
from src.churn_ml.research_v2_config import load_research_v2_config


class OptunaSearchSemanticError(RuntimeError):
    """Raised when completed search evidence cannot be independently rebuilt."""


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
INTEGER_PATTERN = re.compile(r"^(0|[1-9][0-9]*)$")
TRIAL_STATES = {"COMPLETE", "FAIL"}
FAILURE_REASON_CODES = {
    "ADAPTER_CONTRACT_OR_FIT_FAILED",
    "CONFIGURATION_SOURCE_RUNTIME_AUTHENTICATION_FAILED",
    "INTERRUPTED_BY_USER",
    "INTERRUPTED_PROCESS_RECOVERY",
    "LEAKAGE_OR_OBJECTIVE_CONTRACT_FAILED",
    "TRIAL_EXECUTION_FAILED",
}
REQUIRED_METRIC_NAME = "balanced_accuracy"
FORBIDDEN_TEXT = (
    "x_test",
    "sample_submission",
    "sample-submission",
    "kaggle",
    "submission",
    "p2",
    "p3",
)

TRIAL_COLUMNS = [
    "trial_number",
    "state",
    "adapter_id",
    "candidate_identity_sha256",
    "objective",
    "started_at",
    "finished_at",
    "duration_seconds",
    "optuna_parameters_json",
    "resolved_parameters_json",
    "prediction_key_coverage_json",
    "failure_reason_code",
    "failure_message",
    "failure_evidence_json",
]
PREDICTION_COLUMNS = [
    "trial_number",
    "adapter_id",
    "candidate_identity_sha256",
    "repeat",
    "repeat_seed",
    "fold",
    "row_position",
    "target",
    "probability",
    "selected_threshold",
    "prediction",
]
METRIC_COLUMNS = [
    "trial_number",
    "adapter_id",
    "candidate_identity_sha256",
    "repeat",
    "repeat_seed",
    "fold",
    "training_rows",
    "validation_rows",
    "threshold_selection_rows",
    "threshold_source_folds",
    "selected_threshold",
    "threshold_selection_balanced_accuracy",
    "threshold_status",
    "threshold_degenerate",
    "balanced_accuracy",
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
    "comparison",
    "record_type",
    "fold_count",
]
ASSIGNMENT_COLUMNS = ["repeat", "repeat_seed", "fold", "row_position"]
MEMBERSHIP_COLUMNS = ["repeat", "scoring_fold", "threshold_source_fold"]


def validate_completed_search_semantics(root: Path, *, project_root: Path) -> None:
    """Reconstruct every authoritative completed-search claim from raw evidence."""
    resolved = _load_yaml(root / "resolved_search_config.yaml")
    expected_resolved_keys = PLAN_KEYS | {
        "base_candidate_config_sha256",
        "search_space_identity",
        "resume_authentication",
        "study_identity_sha256",
        "search_identity_sha256",
        "search_id",
    }
    _exact_keys(resolved, expected_resolved_keys, "resolved search config")
    adapter_id = _required_string(resolved["adapter_id"], "adapter_id")
    if adapter_id not in {"xgboost_numeric_v1", "catboost_numeric_v1"}:
        _fail("Resolved adapter is not supported.")

    base_path = _repository_path(
        resolved["base_candidate_config"],
        project_root,
        "base_candidate_config",
    )
    space_path = _repository_path(
        resolved["search_space"],
        project_root,
        "search_space",
    )
    base = load_research_v2_config(base_path, project_root=project_root)
    space = load_search_space(space_path, project_root=project_root)
    if (
        base.adapter_id != adapter_id
        or base.dataset_version != resolved["dataset_version"]
        or base.pipeline_id != resolved["pipeline_id"]
        or space.adapter_id != adapter_id
    ):
        _fail("Resolved search references disagree with production contracts.")

    expected_resume = build_resume_authentication(
        project_root=project_root,
        adapter_id=adapter_id,
        contract_paths=(
            base.source_path.relative_to(project_root).as_posix(),
            base.plan_path.relative_to(project_root).as_posix(),
            space.source_path.relative_to(project_root).as_posix(),
        ),
    )
    persisted_resume = _load_json(root / "resume_authentication.json")
    try:
        validate_resume_authentication_shape(
            persisted_resume,
            adapter_id=adapter_id,
        )
    except OptunaSearchResumeIdentityError as error:
        raise OptunaSearchSemanticError(str(error)) from error
    if persisted_resume != expected_resume or resolved["resume_authentication"] != (
        expected_resume
    ):
        _fail("Persisted source/runtime closure differs from the current contract.")

    expected_study_identity = _study_identity(
        resolved,
        base_payload=base.payload,
        search_space_id=space.search_space_id,
        search_space_sha256=space.sha256,
        resume_authentication=expected_resume,
    )
    expected_study_sha256 = canonical_sha256(expected_study_identity)
    expected_search_identity = {
        **deepcopy(expected_study_identity),
        "target_n_trials": resolved["n_trials"],
        "timeout_seconds": resolved["timeout_seconds"],
    }
    expected_search_sha256 = canonical_sha256(expected_search_identity)
    expected_search_id = f"{resolved['search_plan_id']}_{expected_search_sha256[:16]}"
    if (
        resolved["base_candidate_config_sha256"] != canonical_sha256(base.payload)
        or resolved["search_space_identity"]
        != {"id": space.search_space_id, "sha256": space.sha256}
        or resolved["study_identity_sha256"] != expected_study_sha256
        or resolved["search_identity_sha256"] != expected_search_sha256
        or resolved["search_id"] != expected_search_id
    ):
        _fail("Resolved search identity cannot be independently reconstructed.")

    search_identity = _load_json(root / "search_identity.json")
    _exact_keys(
        search_identity,
        {
            "schema_version",
            "search_id",
            "sha256",
            "canonical",
            "study_identity_sha256",
            "dataset_identity_sha256",
            "assignment_identity_sha256",
            "resume_authentication_sha256",
        },
        "search identity",
    )
    if (
        _required_int(search_identity["schema_version"], "schema_version") != 1
        or search_identity["search_id"] != expected_search_id
        or search_identity["sha256"] != expected_search_sha256
        or search_identity["canonical"] != expected_search_identity
        or search_identity["study_identity_sha256"] != expected_study_sha256
        or search_identity["resume_authentication_sha256"]
        != expected_resume["identity_sha256"]
    ):
        _fail("Search identity artifact differs from reconstructed identity.")
    _require_sha256(
        search_identity["dataset_identity_sha256"],
        "dataset_identity_sha256",
    )
    _require_sha256(
        search_identity["assignment_identity_sha256"],
        "assignment_identity_sha256",
    )

    _validate_source_and_environment(
        root,
        adapter_id=adapter_id,
        resume_authentication=expected_resume,
    )
    _validate_search_space_artifact(root, space.payload, space.sha256)

    trials = _load_trials(root / "trials.csv", adapter_id=adapter_id)
    predictions = _load_predictions(
        root / "trial_predictions.csv",
        adapter_id=adapter_id,
    )
    metrics = _load_metrics(root / "trial_metrics.csv", adapter_id=adapter_id)
    assignments = _load_integer_frame(
        root / "fold_assignments.csv",
        ASSIGNMENT_COLUMNS,
    )
    membership = _load_integer_frame(
        root / "threshold_selection_membership.csv",
        MEMBERSHIP_COLUMNS,
    )
    dataset_identity, y = _validate_dataset_identity(
        root / "dataset_identity.json",
        predictions=predictions,
        base_config=base,
        project_root=project_root,
    )
    if canonical_sha256(dataset_identity) != search_identity["dataset_identity_sha256"]:
        _fail("Dataset identity is not linked to the search identity.")

    expected_assignments = build_search_assignments(
        y,
        repeats=_required_int(resolved["repeats"], "repeats"),
        folds=_required_int(resolved["folds"], "folds"),
        assignment_seed=_required_int(
            resolved["assignment_seed"],
            "assignment_seed",
        ),
    )
    if not assignments.equals(expected_assignments.folds):
        _fail("Fold assignments differ from repeated-stratified reconstruction.")
    if not membership.equals(expected_assignments.threshold_membership):
        _fail("Threshold membership differs from exact cross-fitted reconstruction.")
    assignment_identity = _load_json(root / "assignment_identity.json")
    if assignment_identity != expected_assignments.identity:
        _fail("Assignment identity differs from reconstructed assignments.")
    if (
        canonical_sha256(assignment_identity)
        != search_identity["assignment_identity_sha256"]
    ):
        _fail("Assignment identity is not linked to the search identity.")

    reconstructed_objectives = _reconstruct_trials(
        trials=trials,
        predictions=predictions,
        metrics=metrics,
        assignments=expected_assignments.folds,
        membership=expected_assignments.threshold_membership,
        resolved=resolved,
        search_space=space.payload,
        adapter_id=adapter_id,
        row_count=len(y),
    )
    best_number = min(
        reconstructed_objectives,
        key=lambda number: (-reconstructed_objectives[number], number),
    )
    _validate_best_and_candidate(
        root,
        resolved=resolved,
        base_payload=base.payload,
        adapter_id=adapter_id,
        search_space_id=space.search_space_id,
        search_space_sha256=space.sha256,
        trials=trials,
        predictions=predictions,
        best_number=best_number,
        best_objective=reconstructed_objectives[best_number],
        project_root=project_root,
    )
    _validate_study_summary(
        root,
        resolved=resolved,
        trials=trials,
        best_number=best_number,
        best_objective=reconstructed_objectives[best_number],
        search_identity=search_identity,
        resume_authentication=expected_resume,
        dataset_identity=dataset_identity,
        assignment_identity=assignment_identity,
    )
    _validate_runtime(root / "runtime.json")


def _load_trials(path: Path, *, adapter_id: str) -> list[dict[str, Any]]:
    raw = _read_csv(path, TRIAL_COLUMNS)
    if not raw:
        _fail("Trials artifact must contain at least one row.")
    trials: list[dict[str, Any]] = []
    for item in raw:
        number = _csv_int(item["trial_number"], "trial_number")
        state = item["state"]
        if state not in TRIAL_STATES:
            _fail(f"Trial {number} has invalid state {state!r}.")
        _parse_timestamp(item["started_at"], "started_at", require_timezone=False)
        _parse_timestamp(item["finished_at"], "finished_at", require_timezone=False)
        duration = _csv_float(item["duration_seconds"], "duration_seconds")
        if duration < 0.0:
            _fail("Trial duration must be non-negative.")
        optuna_parameters = _csv_json_mapping(
            item["optuna_parameters_json"],
            "optuna_parameters_json",
        )
        if state == "COMPLETE":
            objective = _csv_float(item["objective"], "objective")
            if item["adapter_id"] != adapter_id:
                _fail("Completed trial adapter identity differs.")
            candidate_identity = _require_sha256(
                item["candidate_identity_sha256"],
                "candidate_identity_sha256",
            )
            resolved_parameters = _csv_json_mapping(
                item["resolved_parameters_json"],
                "resolved_parameters_json",
            )
            coverage = _csv_json_mapping(
                item["prediction_key_coverage_json"],
                "prediction_key_coverage_json",
            )
            if (
                item["failure_reason_code"]
                or item["failure_message"]
                or item["failure_evidence_json"] not in {"", "null"}
            ):
                _fail("Completed trial carries failure evidence.")
            failure_evidence = None
        else:
            objective = None
            candidate_identity = None
            resolved_parameters = None
            coverage = None
            if (
                item["objective"]
                or item["adapter_id"]
                or item["candidate_identity_sha256"]
            ):
                _fail("Failed trial carries completed objective or identity evidence.")
            if (
                item["resolved_parameters_json"] != "null"
                or item["prediction_key_coverage_json"] != "null"
            ):
                _fail("Failed trial carries completed parameter/coverage evidence.")
            if item["failure_reason_code"] not in FAILURE_REASON_CODES:
                _fail("Failed trial reason code is invalid.")
            if not item["failure_message"]:
                _fail("Failed trial must carry an exact failure message.")
            if item["failure_evidence_json"] in {"", "null"}:
                _fail("Failed trial must carry structured failure evidence.")
            try:
                failure_evidence = json.loads(item["failure_evidence_json"])
            except json.JSONDecodeError as error:
                raise OptunaSearchSemanticError(
                    "Failed trial failure evidence is malformed."
                ) from error
            if not isinstance(failure_evidence, dict):
                _fail("Failed trial failure evidence must be a mapping.")
            try:
                derived = derive_failure_reason_code(failure_evidence)
            except Exception as error:
                raise OptunaSearchSemanticError(
                    f"Failed trial failure evidence is invalid: {error}"
                ) from error
            if failure_evidence.get("trial_number") != number:
                _fail("Failure evidence trial number differs.")
            if derived != item["failure_reason_code"]:
                _fail("Persisted failure reason code differs from derived evidence.")
        trials.append(
            {
                "trial_number": number,
                "state": state,
                "objective": objective,
                "objective_token": item["objective"] if state == "COMPLETE" else None,
                "optuna_parameters": optuna_parameters,
                "resolved_parameters": resolved_parameters,
                "coverage": coverage,
                "candidate_identity_sha256": candidate_identity,
                "failure_reason_code": item["failure_reason_code"] or None,
                "failure_evidence": failure_evidence,
                "started_at": item["started_at"],
                "finished_at": item["finished_at"],
            }
        )
    numbers = [item["trial_number"] for item in trials]
    if numbers != list(range(len(trials))):
        _fail("Trial numbers must be unique, sorted, and contiguous from zero.")
    if not any(item["state"] == "COMPLETE" for item in trials):
        _fail("Completed search must contain at least one completed trial.")
    return trials


def _load_predictions(path: Path, *, adapter_id: str) -> pd.DataFrame:
    raw = _read_csv(path, PREDICTION_COLUMNS)
    records: list[dict[str, Any]] = []
    for item in raw:
        if item["adapter_id"] != adapter_id:
            _fail("Prediction adapter identity differs.")
        records.append(
            {
                "trial_number": _csv_int(item["trial_number"], "trial_number"),
                "adapter_id": item["adapter_id"],
                "candidate_identity_sha256": _require_sha256(
                    item["candidate_identity_sha256"],
                    "candidate_identity_sha256",
                ),
                "repeat": _csv_int(item["repeat"], "repeat"),
                "repeat_seed": _csv_int(item["repeat_seed"], "repeat_seed"),
                "fold": _csv_int(item["fold"], "fold"),
                "row_position": _csv_int(item["row_position"], "row_position"),
                "target": _csv_binary(item["target"], "target"),
                "probability": _csv_probability(item["probability"]),
                "selected_threshold": _csv_probability(
                    item["selected_threshold"],
                    label="selected_threshold",
                ),
                "prediction": _csv_binary(item["prediction"], "prediction"),
            }
        )
    frame = pd.DataFrame(records, columns=PREDICTION_COLUMNS)
    if frame.empty:
        _fail("Completed search predictions must not be empty.")
    keys = ["trial_number", "repeat", "row_position"]
    if frame.duplicated(keys).any():
        _fail("Prediction keys contain duplicate rows.")
    return frame


def _load_metrics(path: Path, *, adapter_id: str) -> pd.DataFrame:
    raw = _read_csv(path, METRIC_COLUMNS)
    records: list[dict[str, Any]] = []
    for item in raw:
        if item["adapter_id"] != adapter_id:
            _fail("Metric adapter identity differs.")
        record_type = item["record_type"]
        if record_type not in {"fold", "repeat"}:
            _fail("Metric record_type is invalid.")
        common: dict[str, Any] = {
            "trial_number": _csv_int(item["trial_number"], "trial_number"),
            "adapter_id": item["adapter_id"],
            "candidate_identity_sha256": _require_sha256(
                item["candidate_identity_sha256"],
                "candidate_identity_sha256",
            ),
            "repeat": _csv_int(item["repeat"], "repeat"),
            "repeat_seed": _csv_int(item["repeat_seed"], "repeat_seed"),
            "balanced_accuracy": _csv_float(
                item["balanced_accuracy"],
                "balanced_accuracy",
            ),
            "balanced_accuracy_token": item["balanced_accuracy"],
            "record_type": record_type,
            "metric_name": REQUIRED_METRIC_NAME,
        }
        if not 0.0 <= common["balanced_accuracy"] <= 1.0:
            _fail("Balanced Accuracy must be in [0, 1].")
        if record_type == "fold":
            required_empty = {"fold_count"}
            if any(item[name] for name in required_empty):
                _fail("Fold metric contains repeat-only fields.")
            record = {
                **common,
                "fold": _csv_int(item["fold"], "fold"),
                "training_rows": _csv_int(item["training_rows"], "training_rows"),
                "validation_rows": _csv_int(
                    item["validation_rows"],
                    "validation_rows",
                ),
                "threshold_selection_rows": _csv_int(
                    item["threshold_selection_rows"],
                    "threshold_selection_rows",
                ),
                "threshold_source_folds": item["threshold_source_folds"],
                "selected_threshold": _csv_probability(
                    item["selected_threshold"],
                    label="selected_threshold",
                ),
                "selected_threshold_token": item["selected_threshold"],
                "threshold_selection_balanced_accuracy": _csv_float(
                    item["threshold_selection_balanced_accuracy"],
                    "threshold_selection_balanced_accuracy",
                ),
                "threshold_selection_balanced_accuracy_token": item[
                    "threshold_selection_balanced_accuracy"
                ],
                "threshold_status": item["threshold_status"],
                "threshold_degenerate": _csv_bool(
                    item["threshold_degenerate"],
                    "threshold_degenerate",
                ),
                "true_negative": _csv_int(item["true_negative"], "true_negative"),
                "false_positive": _csv_int(
                    item["false_positive"],
                    "false_positive",
                ),
                "false_negative": _csv_int(
                    item["false_negative"],
                    "false_negative",
                ),
                "true_positive": _csv_int(item["true_positive"], "true_positive"),
                "comparison": item["comparison"],
                "fold_count": None,
            }
        else:
            fold_only = set(METRIC_COLUMNS) - {
                "trial_number",
                "adapter_id",
                "candidate_identity_sha256",
                "repeat",
                "repeat_seed",
                "balanced_accuracy",
                "record_type",
                "fold_count",
            }
            if any(item[name] for name in fold_only):
                _fail("Repeat metric contains fold-only fields.")
            record = {
                **common,
                "fold": None,
                "fold_count": _csv_int(item["fold_count"], "fold_count"),
            }
        records.append(record)
    frame = pd.DataFrame(records)
    if frame.empty:
        _fail("Metric evidence is empty.")
    key_columns = ["trial_number", "record_type", "repeat", "fold", "metric_name"]
    if frame.duplicated(key_columns, keep=False).any():
        _fail("Metric evidence contains duplicate keys.")
    return frame


def _validate_dataset_identity(
    path: Path,
    *,
    predictions: pd.DataFrame,
    base_config: Any,
    project_root: Path,
) -> tuple[dict[str, Any], pd.Series]:
    identity = _load_json(path)
    expected_identity, trusted_y, _trusted_x = (
        reconstruct_authoritative_dataset_identity_from_plan(
            base_config,
            project_root=project_root,
        )
    )
    if identity != expected_identity:
        _fail("Persisted dataset identity differs from plan/train-only reconstruction.")
    _reject_forbidden_paths(identity["provided_identity"])
    training = identity["training_data"]
    row_count = _required_int(training["row_count"], "row_count")
    if len(trusted_y) != row_count:
        _fail("Trusted target length differs from training row count.")
    targets = [int(value) for value in trusted_y.tolist()]
    for _, repeat_frame in predictions.groupby(
        ["trial_number", "repeat"],
        sort=True,
    ):
        ordered = repeat_frame.sort_values("row_position")
        if (
            ordered["row_position"].tolist() != list(range(row_count))
            or ordered["target"].astype("int64").tolist() != targets
        ):
            _fail("Prediction targets differ from trusted train-only target identity.")
    return identity, trusted_y.copy()


def _expected_metric_key_universe(
    *,
    complete_numbers: set[int],
    repeats: int,
    folds: int,
) -> set[tuple[int, str, int, int | None, str]]:
    keys: set[tuple[int, str, int, int | None, str]] = set()
    for trial_number in complete_numbers:
        for repeat in range(1, repeats + 1):
            keys.add(
                (
                    trial_number,
                    "repeat",
                    repeat,
                    None,
                    REQUIRED_METRIC_NAME,
                )
            )
            for fold in range(1, folds + 1):
                keys.add(
                    (
                        trial_number,
                        "fold",
                        repeat,
                        fold,
                        REQUIRED_METRIC_NAME,
                    )
                )
    return keys


def _metric_key_tuple(row: Mapping[str, Any]) -> tuple[int, str, int, int | None, str]:
    return (
        int(row["trial_number"]),
        str(row["record_type"]),
        int(row["repeat"]),
        _optional_fold_key(row["fold"]),
        REQUIRED_METRIC_NAME,
    )


def _optional_fold_key(fold: Any) -> int | None:
    if fold is None or fold is pd.NA:
        return None
    try:
        if bool(pd.isna(fold)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(fold, bool):
        _fail("Metric fold key is invalid.")
    if isinstance(fold, (int, np.integer)):
        value = int(fold)
    elif isinstance(fold, (float, np.floating)):
        number = float(fold)
        if math.isnan(number):
            return None
        if not number.is_integer():
            _fail("Metric fold key is invalid.")
        value = int(number)
    else:
        _fail("Metric fold key is invalid.")
    if value < 0:
        _fail("Metric fold key is invalid.")
    return value


def _assert_exact_metric_universe(
    metrics: pd.DataFrame,
    *,
    complete_numbers: set[int],
    repeats: int,
    folds: int,
) -> None:
    expected = _expected_metric_key_universe(
        complete_numbers=complete_numbers,
        repeats=repeats,
        folds=folds,
    )
    actual = {_metric_key_tuple(row) for row in metrics.to_dict("records")}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        _fail(f"Metric key universe differs; missing={missing[:5]}, extra={extra[:5]}.")


def _reconstruct_trials(
    *,
    trials: list[dict[str, Any]],
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    assignments: pd.DataFrame,
    membership: pd.DataFrame,
    resolved: Mapping[str, Any],
    search_space: Mapping[str, Any],
    adapter_id: str,
    row_count: int,
) -> dict[int, float]:
    complete = [item for item in trials if item["state"] == "COMPLETE"]
    complete_numbers = {item["trial_number"] for item in complete}
    if (
        set(predictions["trial_number"]) != complete_numbers
        or set(metrics["trial_number"]) != complete_numbers
    ):
        _fail(
            "Failed/incomplete trials carry evidence or completed evidence is missing."
        )
    repeats = _required_int(resolved["repeats"], "repeats")
    folds = _required_int(resolved["folds"], "folds")
    if resolved["metric"] != REQUIRED_METRIC_NAME:
        _fail("Resolved metric schema is not the v1 Balanced Accuracy contract.")
    _assert_exact_metric_universe(
        metrics,
        complete_numbers=complete_numbers,
        repeats=repeats,
        folds=folds,
    )
    expected_prediction_rows = row_count * repeats
    objectives: dict[int, float] = {}
    threshold_policy_identity = {
        "id": resolved["threshold_policy"],
        "metric": resolved["metric"],
        **resolved["threshold_grid"],
    }
    for trial in complete:
        number = trial["trial_number"]
        expected_parameters, expected_optuna = _expected_parameters(
            search_space,
            seed=resolved["sampler"]["seed"],
            study_name=resolved["study_name"],
            trial_number=number,
        )
        if (
            trial["resolved_parameters"] != expected_parameters
            or trial["optuna_parameters"] != expected_optuna
        ):
            _fail(f"Trial {number} parameters differ from deterministic sampling.")
        candidate_identity = canonical_sha256(
            {
                "schema_version": 1,
                "adapter_id": adapter_id,
                "resolved_parameters": expected_parameters,
            }
        )
        if trial["candidate_identity_sha256"] != candidate_identity:
            _fail(f"Trial {number} candidate identity differs.")
        trial_predictions = predictions.loc[predictions["trial_number"] == number]
        if (
            len(trial_predictions) != expected_prediction_rows
            or set(trial_predictions["adapter_id"]) != {adapter_id}
            or set(trial_predictions["candidate_identity_sha256"])
            != {candidate_identity}
        ):
            _fail(f"Trial {number} prediction Cartesian coverage differs.")
        expected_keys = {
            (repeat, row)
            for repeat in range(1, repeats + 1)
            for row in range(row_count)
        }
        if (
            set(
                trial_predictions[["repeat", "row_position"]].itertuples(
                    index=False,
                    name=None,
                )
            )
            != expected_keys
        ):
            _fail(f"Trial {number} prediction key universe differs.")
        merged = trial_predictions.merge(
            assignments,
            on=["repeat", "repeat_seed", "fold", "row_position"],
            how="outer",
            indicator=True,
        )
        if len(merged) != expected_prediction_rows or set(merged["_merge"]) != {"both"}:
            _fail(f"Trial {number} prediction assignment membership differs.")
        expected_coverage = {
            "schema_version": 1,
            "key_columns": ["trial_number", "repeat", "row_position"],
            "expected_rows": expected_prediction_rows,
            "actual_rows": expected_prediction_rows,
            "duplicates": 0,
            "missing": 0,
            "complete": True,
        }
        if trial["coverage"] != expected_coverage:
            _fail(f"Trial {number} prediction coverage summary differs.")
        fold_scores: list[dict[str, Any]] = []
        for repeat in range(1, repeats + 1):
            repeat_predictions = trial_predictions.loc[
                trial_predictions["repeat"] == repeat
            ]
            for fold in range(1, folds + 1):
                scoring = repeat_predictions.loc[repeat_predictions["fold"] == fold]
                allowed_sources = membership.loc[
                    (membership["repeat"] == repeat)
                    & (membership["scoring_fold"] == fold),
                    "threshold_source_fold",
                ].tolist()
                if fold in allowed_sources:
                    _fail("Scored fold entered threshold selection.")
                selection = repeat_predictions.loc[
                    repeat_predictions["fold"].isin(allowed_sources)
                ]
                threshold = select_balanced_accuracy_threshold(
                    selection["target"],
                    selection["probability"].to_numpy(dtype=float),
                    {
                        "id": resolved["threshold_policy"],
                        "metric": resolved["metric"],
                        **resolved["threshold_grid"],
                    },
                )
                selected_threshold = _canonical_csv_float(threshold.threshold)
                labels = (
                    scoring["probability"].to_numpy(dtype=float) >= selected_threshold
                ).astype("int8")
                targets = scoring["target"].to_numpy(dtype="int8")
                score = _canonical_csv_float(
                    float(balanced_accuracy_score(targets, labels))
                )
                expected_prediction = scoring.assign(
                    expected_threshold=selected_threshold,
                    expected_prediction=labels,
                )
                if (
                    not (
                        expected_prediction["selected_threshold"]
                        == expected_prediction["expected_threshold"]
                    ).all()
                    or not (
                        expected_prediction["prediction"]
                        == expected_prediction["expected_prediction"]
                    ).all()
                ):
                    _fail("Persisted threshold or predicted label differs.")
                fold_metric = metrics.loc[
                    (metrics["trial_number"] == number)
                    & (metrics["record_type"] == "fold")
                    & (metrics["repeat"] == repeat)
                    & (metrics["fold"] == fold)
                ]
                if len(fold_metric) != 1:
                    _fail("Fold metric coverage differs.")
                item = fold_metric.iloc[0]
                if (
                    item["adapter_id"] != adapter_id
                    or item["candidate_identity_sha256"] != candidate_identity
                    or item["comparison"] != threshold_policy_identity["comparison"]
                    or item["metric_name"] != REQUIRED_METRIC_NAME
                ):
                    _fail(f"Trial {number} fold metric identity bindings differ.")
                expected_values = {
                    "repeat_seed": int(scoring["repeat_seed"].iloc[0]),
                    "training_rows": row_count - len(scoring),
                    "validation_rows": len(scoring),
                    "threshold_selection_rows": len(selection),
                    "threshold_source_folds": ",".join(
                        str(value) for value in sorted(allowed_sources)
                    ),
                    "selected_threshold": selected_threshold,
                    "threshold_selection_balanced_accuracy": _canonical_csv_float(
                        threshold.balanced_accuracy
                    ),
                    "threshold_status": threshold.status,
                    "threshold_degenerate": threshold.degenerate,
                    "balanced_accuracy": score,
                    "true_negative": int(((targets == 0) & (labels == 0)).sum()),
                    "false_positive": int(((targets == 0) & (labels == 1)).sum()),
                    "false_negative": int(((targets == 1) & (labels == 0)).sum()),
                    "true_positive": int(((targets == 1) & (labels == 1)).sum()),
                    "comparison": "greater_than_or_equal",
                }
                for name, value in expected_values.items():
                    if not _exact_persisted_equal(
                        item[name], value, row=item, field=name
                    ):
                        _fail(
                            f"Trial {number} fold {repeat}/{fold} metric {name} differs: "
                            f"persisted={item[name]!r}, reconstructed={value!r}."
                        )
                fold_scores.append(
                    {
                        "repeat": repeat,
                        "repeat_seed": int(scoring["repeat_seed"].iloc[0]),
                        "balanced_accuracy": score,
                    }
                )
        fold_frame = pd.DataFrame(fold_scores)
        repeat_frame = fold_frame.groupby("repeat", sort=True, as_index=False).agg(
            repeat_seed=("repeat_seed", "first"),
            fold_count=("balanced_accuracy", "count"),
            balanced_accuracy=("balanced_accuracy", "mean"),
        )
        repeat_frame["balanced_accuracy"] = [
            _canonical_csv_float(value)
            for value in repeat_frame["balanced_accuracy"].tolist()
        ]
        trial_repeat_metrics = metrics.loc[
            (metrics["trial_number"] == number) & (metrics["record_type"] == "repeat")
        ].sort_values("repeat")
        if len(trial_repeat_metrics) != repeats:
            _fail(f"Trial {number} repeat aggregate coverage differs.")
        for expected, (_, actual) in zip(
            repeat_frame.to_dict("records"),
            trial_repeat_metrics.iterrows(),
            strict=True,
        ):
            if (
                actual["adapter_id"] != adapter_id
                or actual["candidate_identity_sha256"] != candidate_identity
                or actual["metric_name"] != REQUIRED_METRIC_NAME
            ):
                _fail(f"Trial {number} repeat metric identity bindings differ.")
            for name, value in expected.items():
                if not _exact_persisted_equal(
                    actual[name],
                    value,
                    row=actual,
                    field=name,
                ):
                    _fail(f"Trial {number} repeat aggregate {name} differs.")
        objective = _canonical_csv_float(
            float(repeat_frame["balanced_accuracy"].mean())
        )
        if not _exact_persisted_equal(
            trial["objective"],
            objective,
            token=trial.get("objective_token"),
        ):
            _fail(f"Trial {number} final objective differs.")
        objectives[number] = objective
    return objectives


def _validate_best_and_candidate(
    root: Path,
    *,
    resolved: Mapping[str, Any],
    base_payload: Mapping[str, Any],
    adapter_id: str,
    search_space_id: str,
    search_space_sha256: str,
    trials: list[dict[str, Any]],
    predictions: pd.DataFrame,
    best_number: int,
    best_objective: float,
    project_root: Path,
) -> None:
    best = _load_json(root / "best_trial.json")
    _exact_keys(
        best,
        {
            "schema_version",
            "trial_number",
            "objective",
            "direction",
            "tie_break",
            "resolved_parameters",
            "candidate_identity_sha256",
            "candidate_config_sha256",
            "prediction_key_coverage",
            "evidence_scope",
        },
        "best trial",
    )
    winner = trials[best_number]
    if (
        _required_int(best["schema_version"], "best schema_version") != 1
        or best["trial_number"] != best_number
        or type(best["objective"]) is not float
        or not math.isfinite(best["objective"])
        or not _exact_persisted_equal(best["objective"], best_objective)
        or best["direction"] != "maximize"
        or best["tie_break"] != "lowest_trial_number"
        or best["resolved_parameters"] != winner["resolved_parameters"]
        or best["candidate_identity_sha256"] != winner["candidate_identity_sha256"]
        or best["prediction_key_coverage"] != winner["coverage"]
        or best["evidence_scope"] != "tuning_only_not_unbiased_final_evidence"
    ):
        _fail("Best-trial artifact differs from reconstructed winner.")
    candidate = _load_yaml(root / "best_candidate_config.yaml")
    expected = deepcopy(dict(base_payload))
    expected["experiment"]["id"] = (
        f"{adapter_id}_optuna_{resolved['search_identity_sha256'][:12]}_t{best_number}"
    )
    expected["candidate_adapter"]["contract"] = build_resolved_adapter_contract(
        expected["candidate_adapter"]["contract"],
        adapter_id=adapter_id,
        tuned_parameters=winner["resolved_parameters"],
    )
    expected["search_provenance"] = {
        "schema_version": 1,
        "search_id": resolved["search_id"],
        "study_name": resolved["study_name"],
        "best_trial_number": best_number,
        "search_identity_sha256": resolved["search_identity_sha256"],
        "search_space_id": search_space_id,
        "search_space_sha256": search_space_sha256,
        "evidence_scope": "tuning_only_not_unbiased_final_evidence",
    }
    if candidate != expected or best["candidate_config_sha256"] != canonical_sha256(
        candidate
    ):
        _fail("Exported candidate linkage or identity differs from winner.")
    _reject_forbidden_candidate(candidate)
    load_research_v2_config(
        root / "best_candidate_config.yaml",
        project_root=project_root,
    )
    winning_predictions = predictions.loc[predictions["trial_number"] == best_number]
    if set(winning_predictions["candidate_identity_sha256"]) != {
        winner["candidate_identity_sha256"]
    }:
        _fail("Winning predictions point to another candidate identity.")


def _validate_study_summary(
    root: Path,
    *,
    resolved: Mapping[str, Any],
    trials: list[dict[str, Any]],
    best_number: int,
    best_objective: float,
    search_identity: Mapping[str, Any],
    resume_authentication: Mapping[str, Any],
    dataset_identity: Mapping[str, Any],
    assignment_identity: Mapping[str, Any],
) -> None:
    study = _load_json(root / "study_summary.json")
    _exact_keys(
        study,
        {
            "schema_version",
            "study_name",
            "search_id",
            "study_identity_sha256",
            "search_identity_sha256",
            "resume_authentication_sha256",
            "dataset_identity_sha256",
            "assignment_identity_sha256",
            "direction",
            "metric",
            "objective_aggregation",
            "requested_trials",
            "actual_trials",
            "state_counts",
            "recovered_interrupted_trials",
            "best_trial_number",
            "best_objective",
            "storage",
            "filesystem_report_authoritative",
            "optuna_storage_role",
            "final_evaluation_status",
        },
        "study summary",
    )
    state_counts = {
        "running": 0,
        "complete": sum(item["state"] == "COMPLETE" for item in trials),
        "pruned": 0,
        "fail": sum(item["state"] == "FAIL" for item in trials),
        "waiting": 0,
    }
    recovered = sum(
        item["failure_reason_code"] == "INTERRUPTED_PROCESS_RECOVERY" for item in trials
    )
    storage = _required_string(study["storage"], "study storage")
    if (
        _required_int(study["schema_version"], "study schema_version") != 1
        or study["study_name"] != resolved["study_name"]
        or study["search_id"] != resolved["search_id"]
        or study["study_identity_sha256"] != resolved["study_identity_sha256"]
        or study["search_identity_sha256"] != resolved["search_identity_sha256"]
        or study["resume_authentication_sha256"]
        != resume_authentication["identity_sha256"]
        or study["dataset_identity_sha256"] != canonical_sha256(dataset_identity)
        or study["assignment_identity_sha256"] != canonical_sha256(assignment_identity)
        or study["dataset_identity_sha256"]
        != search_identity["dataset_identity_sha256"]
        or study["assignment_identity_sha256"]
        != search_identity["assignment_identity_sha256"]
        or study["direction"] != "maximize"
        or study["metric"] != "balanced_accuracy"
        or study["objective_aggregation"]
        != "mean_fold_balanced_accuracy_across_repeats"
        or study["requested_trials"] != resolved["n_trials"]
        or study["actual_trials"] != len(trials)
        or study["state_counts"] != state_counts
        or study["recovered_interrupted_trials"] != recovered
        or study["best_trial_number"] != best_number
        or type(study["best_objective"]) is not float
        or not _exact_persisted_equal(study["best_objective"], best_objective)
        or not _safe_storage_path(storage)
        or study["filesystem_report_authoritative"] is not True
        or study["optuna_storage_role"] != "resumable_operational_state_only"
        or study["final_evaluation_status"] != "not_run"
    ):
        _fail("Study summary differs from reconstructed lifecycle state.")


def _validate_source_and_environment(
    root: Path,
    *,
    adapter_id: str,
    resume_authentication: Mapping[str, Any],
) -> None:
    source = _load_json(root / "source_provenance.json")
    _exact_keys(
        source,
        {"schema_version", "hashing_method", "files", "sha256"},
        "source provenance",
    )
    expected_source = {
        "schema_version": 1,
        "hashing_method": resume_authentication["source_closure"]["hashing_method"],
        "files": resume_authentication["source_closure"]["files"],
    }
    expected_source["sha256"] = canonical_sha256(expected_source)
    if source != expected_source:
        _fail("Source provenance differs from resume source closure.")
    environment = _load_json(root / "environment.json")
    expected_environment = {
        "schema_version": 1,
        "runtime_dependencies": resume_authentication["runtime_dependencies"],
        "adapter_id": adapter_id,
        "device_policy": "cpu_only",
        "thread_count": 1,
    }
    if environment != expected_environment:
        _fail("Environment identity differs from resume runtime identity.")


def _validate_search_space_artifact(
    root: Path,
    payload: Mapping[str, Any],
    sha256: str,
) -> None:
    artifact = _load_json(root / "search_space.json")
    expected = {
        "schema_version": 1,
        "search_space_id": payload["search_space_id"],
        "sha256": sha256,
        "canonical": dict(payload),
    }
    if artifact != expected:
        _fail("Search-space artifact differs from validated YAML.")


def _validate_runtime(path: Path) -> None:
    runtime = _load_json(path)
    _exact_keys(
        runtime,
        {
            "schema_version",
            "started_at_utc",
            "finished_at_utc",
            "duration_seconds",
            "network_access",
            "competition_test_accessed",
            "submission_created",
            "selection_plan_run",
            "confirmation_plan_run",
        },
        "runtime",
    )
    started = _parse_timestamp(runtime["started_at_utc"], "started_at_utc")
    finished = _parse_timestamp(runtime["finished_at_utc"], "finished_at_utc")
    if (
        _required_int(runtime["schema_version"], "runtime schema_version") != 1
        or finished < started
        or type(runtime["duration_seconds"]) is not float
        or not math.isfinite(runtime["duration_seconds"])
        or runtime["duration_seconds"] < 0.0
        or runtime["network_access"] != "disabled_during_study"
        or runtime["competition_test_accessed"] is not False
        or runtime["submission_created"] is not False
        or runtime["selection_plan_run"] is not False
        or runtime["confirmation_plan_run"] is not False
    ):
        _fail("Runtime artifact values differ from the production boundary.")


def _expected_parameters(
    search_space: Mapping[str, Any],
    *,
    seed: int,
    study_name: str,
    trial_number: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from optuna.distributions import FloatDistribution, IntDistribution

    resolved: dict[str, Any] = {}
    optuna_parameters: dict[str, Any] = {}
    for name, specification in search_space["parameters"].items():
        distribution = specification["distribution"]
        if distribution == "int":
            value = stateless_sample(
                seed=seed,
                study_name=study_name,
                trial_number=trial_number,
                parameter_name=name,
                distribution=IntDistribution(
                    low=specification["low"],
                    high=specification["high"],
                    step=specification["step"],
                    log=specification["log"],
                ),
            )
            resolved[name] = value
            optuna_parameters[name] = value
        elif distribution == "float":
            value = stateless_sample(
                seed=seed,
                study_name=study_name,
                trial_number=trial_number,
                parameter_name=name,
                distribution=FloatDistribution(
                    low=specification["low"],
                    high=specification["high"],
                    log=specification["log"],
                ),
            )
            resolved[name] = value
            optuna_parameters[name] = value
        else:
            draw_name = f"{name}__zero_draw"
            zero_draw = stateless_sample(
                seed=seed,
                study_name=study_name,
                trial_number=trial_number,
                parameter_name=draw_name,
                distribution=FloatDistribution(low=0.0, high=1.0),
            )
            optuna_parameters[draw_name] = zero_draw
            if zero_draw < specification["zero_probability"]:
                resolved[name] = 0.0
            else:
                value = stateless_sample(
                    seed=seed,
                    study_name=study_name,
                    trial_number=trial_number,
                    parameter_name=name,
                    distribution=FloatDistribution(
                        low=specification["low_positive"],
                        high=specification["high"],
                        log=True,
                    ),
                )
                resolved[name] = value
                optuna_parameters[name] = value
    for name, specification in search_space["parameters"].items():
        if (
            specification["distribution"] in {"int", "float"}
            and specification["low"] == specification["high"]
        ):
            resolved[name] = specification["low"]
            optuna_parameters[name] = specification["low"]
    return resolved, optuna_parameters


def _study_identity(
    resolved: Mapping[str, Any],
    *,
    base_payload: Mapping[str, Any],
    search_space_id: str,
    search_space_sha256: str,
    resume_authentication: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "search_plan_id": resolved["search_plan_id"],
        "dataset_version": resolved["dataset_version"],
        "pipeline_id": resolved["pipeline_id"],
        "adapter_id": resolved["adapter_id"],
        "base_candidate_config_sha256": canonical_sha256(base_payload),
        "search_space": {
            "id": search_space_id,
            "sha256": search_space_sha256,
        },
        "repeats": resolved["repeats"],
        "folds": resolved["folds"],
        "assignment_seed": resolved["assignment_seed"],
        "threshold_policy": resolved["threshold_policy"],
        "threshold_grid": resolved["threshold_grid"],
        "metric": resolved["metric"],
        "direction": resolved["direction"],
        "sampler": resolved["sampler"],
        "pruner": resolved["pruner"],
        "study_name": resolved["study_name"],
        "positive_class_label": 1,
        "probability_semantics": "binary_positive_class_label_1",
        "label_comparison": "greater_than_or_equal",
        "resume_authentication": dict(resume_authentication),
    }


def _load_integer_frame(path: Path, columns: list[str]) -> pd.DataFrame:
    raw = _read_csv(path, columns)
    records = [{name: _csv_int(item[name], name) for name in columns} for item in raw]
    frame = pd.DataFrame(records, columns=columns, dtype="int64")
    if frame.empty or frame.duplicated().any():
        _fail(f"{path.name} is empty or contains duplicate rows.")
    return frame


def _read_csv(path: Path, columns: list[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != columns:
                _fail(f"{path.name} exact column schema differs.")
            rows = [dict(item) for item in reader]
    except (OSError, csv.Error) as error:
        raise OptunaSearchSemanticError(
            f"Invalid CSV artifact: {path.name}."
        ) from error
    if any(set(item) != set(columns) or None in item for item in rows):
        _fail(f"{path.name} contains malformed rows.")
    row_values = [tuple(item[name] for name in columns) for item in rows]
    if len(row_values) != len(set(row_values)):
        _fail(f"{path.name} contains duplicate rows.")
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OptunaSearchSemanticError(f"Invalid JSON: {path.name}.") from error
    if not isinstance(value, dict):
        _fail(f"{path.name} must be a mapping.")
    _validate_json_primitives(value, path.name)
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise OptunaSearchSemanticError(f"Invalid YAML: {path.name}.") from error
    if not isinstance(value, dict):
        _fail(f"{path.name} must be a mapping.")
    _validate_json_primitives(value, path.name)
    return value


def _validate_json_primitives(value: Any, label: str) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            _fail(f"{label} contains a non-finite float.")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_primitives(item, label)
        return
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            _fail(f"{label} contains a non-string mapping key.")
        for item in value.values():
            _validate_json_primitives(item, label)
        return
    _fail(f"{label} contains non-primitive value {type(value).__name__}.")


def _csv_int(value: str, label: str) -> int:
    if not INTEGER_PATTERN.fullmatch(value):
        _fail(f"{label} must be a canonical non-negative integer.")
    return int(value)


def _csv_binary(value: str, label: str) -> int:
    result = _csv_int(value, label)
    if result not in {0, 1}:
        _fail(f"{label} must be exactly 0 or 1.")
    return result


def _csv_float(value: str, label: str) -> float:
    if value != value.strip() or not value:
        _fail(f"{label} must be a canonical finite float token.")
    try:
        result = float(value)
    except ValueError as error:
        raise OptunaSearchSemanticError(f"{label} must be a float.") from error
    if not math.isfinite(result):
        _fail(f"{label} must be finite.")
    return result


def _csv_probability(value: str, label: str = "probability") -> float:
    result = _csv_float(value, label)
    if not 0.0 <= result <= 1.0:
        _fail(f"{label} must be in [0, 1].")
    return result


def _csv_bool(value: str, label: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    _fail(f"{label} must be exactly True or False.")


def _csv_json_mapping(value: str, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise OptunaSearchSemanticError(f"{label} must be valid JSON.") from error
    if not isinstance(parsed, dict):
        _fail(f"{label} must be a JSON mapping.")
    _validate_json_primitives(parsed, label)
    return parsed


def _parse_timestamp(
    value: Any,
    label: str,
    *,
    require_timezone: bool = True,
) -> datetime:
    if type(value) is not str or not value:
        _fail(f"{label} must be an ISO timestamp.")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as error:
        raise OptunaSearchSemanticError(f"{label} is not ISO-8601.") from error
    if require_timezone and result.tzinfo is None:
        _fail(f"{label} must include a timezone.")
    return result


def _repository_path(value: Any, root: Path, label: str) -> Path:
    text = _required_string(value, label)
    if "\\" in text:
        _fail(f"{label} must use POSIX separators.")
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or windows.root
        or ".." in posix.parts
    ):
        _fail(f"{label} must be repository-relative without traversal.")
    path = root.joinpath(*posix.parts).resolve()
    if root not in path.parents or not path.is_file():
        _fail(f"{label} must be a contained regular file.")
    return path


def _safe_storage_path(value: str) -> bool:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    return (
        "\\" not in value
        and not posix.is_absolute()
        and not windows.is_absolute()
        and not windows.drive
        and not windows.root
        and ".." not in posix.parts
        and posix.parts[:2] == ("artifacts", "optuna")
        and posix.suffix == ".db"
    )


def _reject_forbidden_paths(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_forbidden_paths(key)
            _reject_forbidden_paths(item)
    elif isinstance(value, list):
        for item in value:
            _reject_forbidden_paths(item)
    elif isinstance(value, str):
        lowered = value.lower()
        if any(token in lowered for token in FORBIDDEN_TEXT):
            _fail("Dataset identity references forbidden competition evidence.")


def _reject_forbidden_candidate(candidate: Mapping[str, Any]) -> None:
    serialized = json.dumps(candidate, sort_keys=True).lower()
    if "sqlite" in serialized or ".db" in serialized or "distribution" in serialized:
        _fail("Exported candidate contains search/database objects.")
    parameters = candidate["candidate_adapter"]["contract"][
        "xgboost"
        if candidate["candidate_adapter"]["id"].startswith("xgboost")
        else "catboost"
    ]["parameters"]
    _validate_json_primitives(parameters, "candidate parameters")


def _required_int(value: Any, label: str) -> int:
    if type(value) is not int:
        _fail(f"{label} must be an integer, not bool/string.")
    return value


def _exact_persisted_equal(
    actual: Any,
    expected: Any,
    *,
    row: Mapping[str, Any] | None = None,
    field: str | None = None,
    token: str | None = None,
) -> bool:
    if isinstance(actual, (float, np.floating)) or isinstance(
        expected,
        (float, np.floating),
    ):
        expected_canonical = _canonical_csv_float(float(expected))
        if token is not None:
            return token == format(expected_canonical, ".17g")
        if row is not None and field is not None:
            token_field = f"{field}_token"
            if token_field in row and row[token_field] is not None:
                return str(row[token_field]) == format(expected_canonical, ".17g")
        return _canonical_csv_float(float(actual)) == expected_canonical
    if isinstance(actual, (np.integer,)):
        actual = int(actual)
    if isinstance(expected, (np.integer,)):
        expected = int(expected)
    if isinstance(actual, (np.bool_,)):
        actual = bool(actual)
    if isinstance(expected, (np.bool_,)):
        expected = bool(expected)
    return actual == expected


def _canonical_csv_float(value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        _fail("Canonical float token requires a finite value.")
    return float(format(number, ".17g"))


def _canonical_float_token(value: float) -> str:
    """Exact Optuna Search v1 CSV float token."""
    return format(_canonical_csv_float(value), ".17g")


def _required_string(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        _fail(f"{label} must be a non-empty string.")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or SHA256_PATTERN.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256.")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        _fail(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}."
        )


def _fail(message: str) -> NoReturn:
    raise OptunaSearchSemanticError(message)
