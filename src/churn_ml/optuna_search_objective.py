from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from src.churn_ml.research_data import canonical_sha256
from src.churn_ml.research_protocol import select_balanced_accuracy_threshold


class OptunaSearchObjectiveError(RuntimeError):
    """Raised when fold-local search evaluation violates its strict boundary."""


FitPredict = Callable[..., np.ndarray]


@dataclass(frozen=True)
class SearchAssignments:
    folds: pd.DataFrame
    threshold_membership: pd.DataFrame
    identity: dict[str, Any]


@dataclass(frozen=True)
class TrialEvaluation:
    objective: float
    fold_metrics: pd.DataFrame
    repeat_metrics: pd.DataFrame
    predictions: pd.DataFrame
    coverage: dict[str, Any]


def build_search_assignments(
    y: pd.Series,
    *,
    repeats: int,
    folds: int,
    assignment_seed: int,
) -> SearchAssignments:
    if (
        not isinstance(y, pd.Series)
        or y.isna().any()
        or set(y.unique()) != {0, 1}
        or not y.index.equals(pd.RangeIndex(len(y)))
    ):
        raise OptunaSearchObjectiveError(
            "Search targets must be aligned binary labels on a RangeIndex."
        )
    row_positions = np.arange(len(y), dtype=np.int64)
    frames: list[pd.DataFrame] = []
    memberships: list[dict[str, int]] = []
    for repeat in range(1, repeats + 1):
        repeat_seed = assignment_seed + repeat - 1
        if repeat_seed > 2**31 - 1:
            raise OptunaSearchObjectiveError("Derived repeat seed exceeds int32.")
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=repeat_seed,
        )
        for fold, (_, validation_local) in enumerate(
            splitter.split(row_positions, y.to_numpy()),
            start=1,
        ):
            frames.append(
                pd.DataFrame(
                    {
                        "repeat": repeat,
                        "repeat_seed": repeat_seed,
                        "fold": fold,
                        "row_position": row_positions[validation_local],
                    }
                )
            )
            memberships.extend(
                {
                    "repeat": repeat,
                    "scoring_fold": fold,
                    "threshold_source_fold": source_fold,
                }
                for source_fold in range(1, folds + 1)
                if source_fold != fold
            )
    assignments = pd.concat(frames, ignore_index=True).astype("int64")
    membership = pd.DataFrame(memberships).astype("int64")
    validate_search_assignments(
        assignments,
        membership,
        row_count=len(y),
        repeats=repeats,
        folds=folds,
    )
    identity = {
        "schema_version": 1,
        "splitter": "repeated_stratified_kfold",
        "repeats": repeats,
        "folds": folds,
        "assignment_seed": assignment_seed,
        "repeat_seed_policy": "assignment_seed_plus_zero_based_repeat",
        "fold_assignments_sha256": _integer_frame_sha256(assignments),
        "threshold_membership_sha256": _integer_frame_sha256(membership),
    }
    return SearchAssignments(
        folds=assignments,
        threshold_membership=membership,
        identity=identity,
    )


def validate_search_assignments(
    assignments: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    row_count: int,
    repeats: int,
    folds: int,
) -> None:
    expected_assignment_columns = [
        "repeat",
        "repeat_seed",
        "fold",
        "row_position",
    ]
    expected_membership_columns = [
        "repeat",
        "scoring_fold",
        "threshold_source_fold",
    ]
    if list(assignments.columns) != expected_assignment_columns:
        raise OptunaSearchObjectiveError("Fold-assignment schema differs.")
    if list(membership.columns) != expected_membership_columns:
        raise OptunaSearchObjectiveError("Threshold-membership schema differs.")
    expected_rows = set(range(row_count))
    for repeat in range(1, repeats + 1):
        frame = assignments.loc[assignments["repeat"] == repeat]
        if (
            len(frame) != row_count
            or frame["row_position"].duplicated().any()
            or set(frame["row_position"]) != expected_rows
            or set(frame["fold"]) != set(range(1, folds + 1))
        ):
            raise OptunaSearchObjectiveError(
                "Fold assignments have incomplete or duplicate prediction keys."
            )
        for scoring_fold in range(1, folds + 1):
            sources = membership.loc[
                (membership["repeat"] == repeat)
                & (membership["scoring_fold"] == scoring_fold),
                "threshold_source_fold",
            ]
            expected_sources = set(range(1, folds + 1)) - {scoring_fold}
            if (
                len(sources) != folds - 1
                or sources.duplicated().any()
                or set(sources) != expected_sources
            ):
                raise OptunaSearchObjectiveError(
                    "Held-out fold entered threshold selection membership."
                )


def evaluate_trial(
    X: pd.DataFrame,
    y: pd.Series,
    assignments: SearchAssignments,
    *,
    adapter_contract: Mapping[str, Any],
    threshold_policy: dict[str, Any],
    fit_predict: FitPredict,
    trial_number: int,
    fit_audit_callback: Any = None,
) -> TrialEvaluation:
    if len(X) != len(y) or not X.index.equals(y.index):
        raise OptunaSearchObjectiveError("Search features and targets are not aligned.")
    all_positions = np.arange(len(y), dtype=np.int64)
    prediction_frames: list[pd.DataFrame] = []
    repeats = sorted(assignments.folds["repeat"].unique())
    for repeat in repeats:
        repeat_assignments = assignments.folds.loc[
            assignments.folds["repeat"] == repeat
        ]
        repeat_seed = int(repeat_assignments["repeat_seed"].iloc[0])
        for fold in sorted(repeat_assignments["fold"].unique()):
            validation_positions = repeat_assignments.loc[
                repeat_assignments["fold"] == fold,
                "row_position",
            ].to_numpy(dtype=np.int64)
            training_positions = np.setdiff1d(
                all_positions,
                validation_positions,
                assume_unique=True,
            )
            probabilities = fit_predict(
                X.iloc[training_positions],
                y.iloc[training_positions],
                X.iloc[validation_positions],
                deepcopy(dict(adapter_contract)),
                model_training_positions=training_positions,
                prediction_positions=validation_positions,
                audit_callback=fit_audit_callback,
            )
            probabilities = np.asarray(probabilities, dtype=float)
            if (
                probabilities.ndim != 1
                or len(probabilities) != len(validation_positions)
                or not np.isfinite(probabilities).all()
                or ((probabilities < 0.0) | (probabilities > 1.0)).any()
            ):
                raise OptunaSearchObjectiveError(
                    "Adapter returned invalid class-1 probabilities."
                )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "trial_number": trial_number,
                        "repeat": int(repeat),
                        "repeat_seed": repeat_seed,
                        "fold": int(fold),
                        "row_position": validation_positions,
                        "target": y.iloc[validation_positions].to_numpy(dtype="int8"),
                        "probability": probabilities,
                    }
                )
            )
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["repeat", "row_position"],
        ignore_index=True,
    )
    _validate_prediction_coverage(predictions, len(y), len(repeats), trial_number)
    fold_records: list[dict[str, Any]] = []
    for repeat in repeats:
        repeat_predictions = predictions.loc[predictions["repeat"] == repeat]
        for fold in sorted(repeat_predictions["fold"].unique()):
            scoring = repeat_predictions.loc[repeat_predictions["fold"] == fold]
            allowed_sources = assignments.threshold_membership.loc[
                (assignments.threshold_membership["repeat"] == repeat)
                & (assignments.threshold_membership["scoring_fold"] == fold),
                "threshold_source_fold",
            ].tolist()
            if int(fold) in allowed_sources:
                raise OptunaSearchObjectiveError(
                    "Held-out fold entered threshold selection."
                )
            selection = repeat_predictions.loc[
                repeat_predictions["fold"].isin(allowed_sources)
            ]
            expected_selection_rows = len(repeat_predictions) - len(scoring)
            if len(selection) != expected_selection_rows or set(
                selection["fold"]
            ) != set(allowed_sources):
                raise OptunaSearchObjectiveError(
                    "Threshold-selection prediction coverage differs."
                )
            threshold = select_balanced_accuracy_threshold(
                selection["target"],
                selection["probability"].to_numpy(),
                threshold_policy,
            )
            labels = (scoring["probability"].to_numpy() >= threshold.threshold).astype(
                "int8"
            )
            score = float(balanced_accuracy_score(scoring["target"], labels))
            fold_records.append(
                {
                    "trial_number": trial_number,
                    "repeat": int(repeat),
                    "repeat_seed": int(scoring["repeat_seed"].iloc[0]),
                    "fold": int(fold),
                    "training_rows": len(y) - len(scoring),
                    "validation_rows": len(scoring),
                    "threshold_selection_rows": len(selection),
                    "threshold_source_folds": ",".join(
                        str(value) for value in sorted(allowed_sources)
                    ),
                    "selected_threshold": threshold.threshold,
                    "threshold_selection_balanced_accuracy": (
                        threshold.balanced_accuracy
                    ),
                    "threshold_status": threshold.status,
                    "threshold_degenerate": threshold.degenerate,
                    "balanced_accuracy": score,
                    "comparison": "greater_than_or_equal",
                }
            )
            predictions.loc[scoring.index, "selected_threshold"] = threshold.threshold
            predictions.loc[scoring.index, "prediction"] = labels
    predictions["prediction"] = predictions["prediction"].astype("int8")
    fold_metrics = pd.DataFrame(fold_records).sort_values(
        ["repeat", "fold"],
        ignore_index=True,
    )
    repeat_metrics = (
        fold_metrics.groupby("repeat", sort=True, as_index=False)
        .agg(
            repeat_seed=("repeat_seed", "first"),
            fold_count=("fold", "count"),
            balanced_accuracy=("balanced_accuracy", "mean"),
        )
        .assign(trial_number=trial_number)
    )
    repeat_metrics = repeat_metrics[
        [
            "trial_number",
            "repeat",
            "repeat_seed",
            "fold_count",
            "balanced_accuracy",
        ]
    ]
    objective = float(repeat_metrics["balanced_accuracy"].mean())
    coverage = {
        "schema_version": 1,
        "key_columns": ["trial_number", "repeat", "row_position"],
        "expected_rows": len(y) * len(repeats),
        "actual_rows": len(predictions),
        "duplicates": int(
            predictions.duplicated(["trial_number", "repeat", "row_position"]).sum()
        ),
        "missing": 0,
        "complete": True,
    }
    return TrialEvaluation(
        objective=objective,
        fold_metrics=fold_metrics,
        repeat_metrics=repeat_metrics,
        predictions=predictions,
        coverage=coverage,
    )


def _validate_prediction_coverage(
    predictions: pd.DataFrame,
    row_count: int,
    repeats: int,
    trial_number: int,
) -> None:
    keys = ["trial_number", "repeat", "row_position"]
    if predictions.duplicated(keys).any():
        raise OptunaSearchObjectiveError("Duplicate prediction keys were produced.")
    expected = {
        (trial_number, repeat, row_position)
        for repeat in range(1, repeats + 1)
        for row_position in range(row_count)
    }
    actual = set(predictions[keys].itertuples(index=False, name=None))
    if actual != expected:
        raise OptunaSearchObjectiveError(
            "Missing or unexpected prediction keys were produced."
        )


def _integer_frame_sha256(frame: pd.DataFrame) -> str:
    payload = {
        "columns": frame.columns.tolist(),
        "rows": [
            [int(value) for value in row]
            for row in frame.itertuples(index=False, name=None)
        ],
    }
    return canonical_sha256(payload)
