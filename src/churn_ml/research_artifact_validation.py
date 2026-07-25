from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.churn_ml.research_config import ResearchConfig
from src.churn_ml.research_data import load_research_training_data
from src.churn_ml.research_protocol import (
    EvaluationAssignments,
    aggregate_repeat_metrics,
    build_candidate_contract_identity,
    build_evaluation_assignments,
    build_evaluation_plan_identity,
    build_repeat_metrics,
    calculate_outer_metrics,
    canonical_sha256,
    select_balanced_accuracy_threshold,
    summarize_thresholds,
    validate_evaluation_assignments,
)
from src.churn_ml.research_provenance import (
    candidate_source_provenance,
    run_implementation_provenance,
)
from src.churn_ml.research_runtime_provenance import (
    collect_research_environment_versions,
)


NUMERIC_ABSOLUTE_TOLERANCE = 1e-12
ARTIFACT_HASHING_METHOD = "sha256_raw_artifact_bytes_v1"

OUTER_ASSIGNMENT_SCHEMA = (
    ("repeat", "int64"),
    ("repeat_seed", "int64"),
    ("outer_fold", "int64"),
    ("row_position", "int64"),
)
THRESHOLD_ASSIGNMENT_SCHEMA = (
    ("repeat", "int64"),
    ("repeat_seed", "int64"),
    ("outer_fold", "int64"),
    ("threshold_selection_fold", "int64"),
    ("row_position", "int64"),
)
OUTER_PREDICTION_SCHEMA = (
    ("repeat", "int64"),
    ("repeat_seed", "int64"),
    ("outer_fold", "int64"),
    ("row_position", "int64"),
    ("target", "int8"),
    ("probability", "float64"),
    ("selected_threshold", "float64"),
    ("prediction", "int8"),
)
THRESHOLD_PREDICTION_SCHEMA = (
    ("repeat", "int64"),
    ("repeat_seed", "int64"),
    ("outer_fold", "int64"),
    ("threshold_selection_fold", "int64"),
    ("row_position", "int64"),
    ("target", "int8"),
    ("probability", "float64"),
    ("selected_threshold", "float64"),
    ("prediction", "int8"),
)
SELECTED_THRESHOLD_SCHEMA = (
    ("repeat", "int64"),
    ("repeat_seed", "int64"),
    ("outer_fold", "int64"),
    ("selected_threshold", "float64"),
    ("threshold_selection_balanced_accuracy", "float64"),
    ("status", "object"),
    ("degenerate", "bool"),
)
OUTER_FOLD_METRIC_SCHEMA = (
    ("repeat", "int64"),
    ("repeat_seed", "int64"),
    ("outer_fold", "int64"),
    ("training_rows", "int64"),
    ("validation_rows", "int64"),
    ("threshold_selection_balanced_accuracy", "float64"),
    ("threshold_status", "object"),
    ("threshold_degenerate", "bool"),
    ("duration_seconds", "float64"),
    ("balanced_accuracy", "float64"),
    ("sensitivity", "float64"),
    ("specificity", "float64"),
    ("tn", "int64"),
    ("fp", "int64"),
    ("fn", "int64"),
    ("tp", "int64"),
    ("predicted_positive_rate", "float64"),
    ("roc_auc", "float64"),
    ("average_precision", "float64"),
    ("brier_score", "float64"),
    ("selected_threshold", "float64"),
)
REPEAT_METRIC_SCHEMA = (
    ("repeat", "int64"),
    ("validation_rows", "int64"),
    ("balanced_accuracy", "float64"),
    ("sensitivity", "float64"),
    ("specificity", "float64"),
    ("tn", "int64"),
    ("fp", "int64"),
    ("fn", "int64"),
    ("tp", "int64"),
    ("predicted_positive_rate", "float64"),
    ("roc_auc", "float64"),
    ("average_precision", "float64"),
    ("brier_score", "float64"),
)
THRESHOLD_CURVE_SCHEMA = (
    ("repeat", "int64"),
    ("outer_fold", "int64"),
    ("threshold", "float64"),
    ("balanced_accuracy", "float64"),
)


class PersistedArtifactValidationError(RuntimeError):
    """Raised when finalized run artifacts do not prove a valid run."""


def validate_persisted_run(
    root: Path,
    config: ResearchConfig,
    *,
    expected_plan_hash: str,
    expected_candidate_hash: str,
    expected_candidate_source_manifest_hash: str,
    expected_run_implementation_hash: str,
    expected_loaded_module_hash: str,
) -> None:
    """Recompute the run contract from persisted files before terminal success."""
    required = _required_artifact_paths(config)
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise PersistedArtifactValidationError(f"Missing required artifacts: {missing}")
    if (root / "_FAILED").exists() or (root / "_SUCCESS").exists():
        raise PersistedArtifactValidationError(
            "A terminal marker exists before completion validation."
        )

    outer_assignments = pd.read_parquet(root / "splits" / "outer_assignments.parquet")
    threshold_assignments = pd.read_parquet(
        root / "splits" / "threshold_selection_assignments.parquet"
    )
    outer_predictions = pd.read_parquet(
        root / "predictions" / "outer_validation.parquet"
    )
    threshold_predictions = pd.read_parquet(
        root / "predictions" / "threshold_selection_oof.parquet"
    )
    selected_thresholds = pd.read_csv(root / "thresholds" / "selected_thresholds.csv")
    fold_metrics = pd.read_csv(root / "metrics" / "outer_folds.csv")
    repeat_metrics = pd.read_csv(root / "metrics" / "repeats.csv")

    _assert_schema(outer_assignments, OUTER_ASSIGNMENT_SCHEMA, "outer assignments")
    _assert_schema(
        threshold_assignments,
        THRESHOLD_ASSIGNMENT_SCHEMA,
        "threshold assignments",
    )
    _assert_schema(outer_predictions, OUTER_PREDICTION_SCHEMA, "outer predictions")
    _assert_schema(
        threshold_predictions,
        THRESHOLD_PREDICTION_SCHEMA,
        "threshold-selection predictions",
    )
    _assert_schema(
        selected_thresholds,
        SELECTED_THRESHOLD_SCHEMA,
        "selected thresholds",
    )
    _assert_schema(fold_metrics, OUTER_FOLD_METRIC_SCHEMA, "outer-fold metrics")
    _assert_schema(repeat_metrics, REPEAT_METRIC_SCHEMA, "repeat metrics")

    threshold_curves: pd.DataFrame | None = None
    if config.payload["artifacts"]["save_threshold_curves"]:
        threshold_curves = pd.read_parquet(
            root / "thresholds" / "threshold_curves.parquet"
        )
        _assert_schema(
            threshold_curves,
            THRESHOLD_CURVE_SCHEMA,
            "threshold curves",
        )

    data = load_research_training_data(config)
    saved_fingerprints = _read_json(root / "dataset_fingerprints.json")
    _assert_json_equal(saved_fingerprints, data.fingerprints, "dataset fingerprints")
    saved_feature_schema = _read_json(root / "feature_schema.json")
    _assert_json_equal(
        saved_feature_schema,
        data.feature_schema.to_dict(),
        "candidate feature schema",
    )

    generated_assignments = build_evaluation_assignments(data.y, config.plan_payload)
    _assert_frame_equal(
        outer_assignments,
        generated_assignments.outer,
        ["repeat", "outer_fold", "row_position"],
        "outer assignments",
    )
    _assert_frame_equal(
        threshold_assignments,
        generated_assignments.threshold_selection,
        [
            "repeat",
            "outer_fold",
            "threshold_selection_fold",
            "row_position",
        ],
        "threshold assignments",
    )
    persisted_assignments = EvaluationAssignments(
        outer=outer_assignments,
        threshold_selection=threshold_assignments,
    )
    validate_evaluation_assignments(
        persisted_assignments,
        len(data.y),
        config.plan_payload,
    )
    _validate_identity_artifacts(
        root,
        config,
        data.fingerprints,
        data.feature_schema.to_dict(),
        persisted_assignments,
        expected_plan_hash=expected_plan_hash,
        expected_candidate_hash=expected_candidate_hash,
        expected_candidate_source_manifest_hash=(
            expected_candidate_source_manifest_hash
        ),
        expected_run_implementation_hash=expected_run_implementation_hash,
        expected_loaded_module_hash=expected_loaded_module_hash,
    )
    _validate_prediction_assignments(
        outer_assignments,
        threshold_assignments,
        outer_predictions,
        threshold_predictions,
        selected_thresholds,
        data.y,
    )
    _validate_metrics_and_thresholds(
        root,
        config,
        outer_predictions,
        threshold_predictions,
        selected_thresholds,
        fold_metrics,
        repeat_metrics,
        threshold_curves,
    )


def build_artifact_manifest(root: Path) -> tuple[dict[str, Any], str]:
    excluded = {"artifact_manifest.json", "_SUCCESS", "_FAILED"}
    records: list[dict[str, Any]] = []
    for path in sorted(
        (
            item
            for item in root.rglob("*")
            if item.is_file() and item.name not in excluded
        ),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        raw = path.read_bytes()
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    canonical = {
        "schema_version": 1,
        "hashing_method": ARTIFACT_HASHING_METHOD,
        "artifacts": records,
    }
    return canonical, canonical_sha256(canonical)


def validate_artifact_manifest(root: Path, payload: dict[str, Any]) -> None:
    expected_canonical, expected_hash = build_artifact_manifest(root)
    actual_canonical = {
        name: payload.get(name)
        for name in ("schema_version", "hashing_method", "artifacts")
    }
    if set(payload) != {
        "schema_version",
        "hashing_method",
        "artifacts",
        "manifest_sha256",
    }:
        raise PersistedArtifactValidationError(
            "Artifact manifest has missing or extra keys."
        )
    _assert_json_equal(
        actual_canonical,
        expected_canonical,
        "artifact manifest records",
    )
    if payload["manifest_sha256"] != expected_hash:
        raise PersistedArtifactValidationError(
            "Artifact manifest canonical hash does not match its records."
        )


def _validate_identity_artifacts(
    root: Path,
    config: ResearchConfig,
    fingerprints: dict[str, Any],
    feature_schema: dict[str, Any],
    assignments: EvaluationAssignments,
    *,
    expected_plan_hash: str,
    expected_candidate_hash: str,
    expected_candidate_source_manifest_hash: str,
    expected_run_implementation_hash: str,
    expected_loaded_module_hash: str,
) -> None:
    plan_identity, plan_hash = build_evaluation_plan_identity(
        config.plan_payload,
        fingerprints,
        assignments,
    )
    _validate_identity_file(
        root / "evaluation_plan.json",
        plan_identity,
        plan_hash,
        expected_plan_hash,
        "evaluation plan",
    )

    candidate_sources, candidate_source_hash = candidate_source_provenance(
        config.project_root
    )
    if candidate_source_hash != expected_candidate_source_manifest_hash:
        raise PersistedArtifactValidationError(
            "Candidate source manifest hash does not match."
        )
    environment = collect_research_environment_versions()
    candidate_identity, candidate_hash = build_candidate_contract_identity(
        config.candidate_contract,
        feature_schema=feature_schema,
        source_provenance={
            **candidate_sources,
            "manifest_sha256": candidate_source_hash,
        },
        runtime_dependencies={
            name: environment[name]
            for name in (
                "numpy",
                "pandas",
                "scikit_learn",
                "lightgbm",
                "pyarrow",
            )
        },
    )
    _validate_identity_file(
        root / "candidate_contract.json",
        candidate_identity,
        candidate_hash,
        expected_candidate_hash,
        "candidate contract",
    )

    source = config.candidate_contract["source"]
    run_identity, run_hash = run_implementation_provenance(
        config.project_root,
        config_paths={
            "research_run": config.source_path,
            "evaluation_plan": config.plan_path,
            "historical_candidate": Path(str(source["historical_config_path"])),
            "baseline_manifest": Path(str(source["manifest_path"])),
        },
    )
    _validate_identity_file(
        root / "run_implementation.json",
        run_identity,
        run_hash,
        expected_run_implementation_hash,
        "run implementation",
    )

    loaded = _read_json(root / "loaded_modules.json")
    if set(loaded) != {"sha256", "canonical", "observations"}:
        raise PersistedArtifactValidationError(
            "Loaded-module identity has missing or extra keys."
        )
    if (
        loaded["sha256"] != expected_loaded_module_hash
        or canonical_sha256(loaded["canonical"]) != expected_loaded_module_hash
    ):
        raise PersistedArtifactValidationError(
            "Loaded-module provenance hash does not match."
        )
    modules = loaded["canonical"].get("modules")
    if (
        not isinstance(modules, list)
        or not modules
        or any(
            not isinstance(record, Mapping) or record.get("match") is not True
            for record in modules
        )
    ):
        raise PersistedArtifactValidationError(
            "Loaded-module provenance does not contain verified modules."
        )


def _validate_identity_file(
    path: Path,
    expected_canonical: dict[str, Any],
    recomputed_hash: str,
    expected_hash: str,
    label: str,
) -> None:
    payload = _read_json(path)
    if set(payload) != {"sha256", "canonical"}:
        raise PersistedArtifactValidationError(
            f"{label} identity has missing or extra keys."
        )
    if recomputed_hash != expected_hash or payload["sha256"] != recomputed_hash:
        raise PersistedArtifactValidationError(f"{label} hash does not match.")
    _assert_json_equal(payload["canonical"], expected_canonical, f"{label} canonical")


def _validate_prediction_assignments(
    outer_assignments: pd.DataFrame,
    threshold_assignments: pd.DataFrame,
    outer_predictions: pd.DataFrame,
    threshold_predictions: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
    y: pd.Series,
) -> None:
    outer_keys = ["repeat", "repeat_seed", "outer_fold", "row_position"]
    threshold_keys = [
        "repeat",
        "repeat_seed",
        "outer_fold",
        "threshold_selection_fold",
        "row_position",
    ]
    _assert_unique(outer_predictions, outer_keys, "outer prediction")
    _assert_unique(threshold_predictions, threshold_keys, "threshold prediction")
    _assert_unique(
        selected_thresholds,
        ["repeat", "outer_fold"],
        "selected-threshold",
    )
    _assert_frame_equal(
        outer_predictions[outer_keys],
        outer_assignments[outer_keys],
        outer_keys,
        "outer prediction keys",
    )
    _assert_frame_equal(
        threshold_predictions[threshold_keys],
        threshold_assignments[threshold_keys],
        threshold_keys,
        "threshold prediction keys",
    )

    for label, frame in (
        ("outer", outer_predictions),
        ("threshold", threshold_predictions),
    ):
        expected_targets = y.iloc[frame["row_position"].to_numpy()].to_numpy(
            dtype="int8"
        )
        if not np.array_equal(frame["target"].to_numpy(), expected_targets):
            raise PersistedArtifactValidationError(
                f"{label} prediction targets do not match row identity."
            )
        probabilities = frame["probability"].to_numpy(dtype=float)
        thresholds = frame["selected_threshold"].to_numpy(dtype=float)
        if (
            not np.isfinite(probabilities).all()
            or ((probabilities < 0.0) | (probabilities > 1.0)).any()
            or not np.isfinite(thresholds).all()
        ):
            raise PersistedArtifactValidationError(
                f"{label} probabilities or thresholds are malformed."
            )
        expected_labels = (probabilities >= thresholds).astype("int8")
        if not np.array_equal(frame["prediction"].to_numpy(), expected_labels):
            raise PersistedArtifactValidationError(
                f"{label} labels do not use probability >= selected_threshold."
            )

    threshold_lookup = selected_thresholds.set_index(["repeat", "outer_fold"])[
        "selected_threshold"
    ]
    for label, frame in (
        ("outer", outer_predictions),
        ("threshold", threshold_predictions),
    ):
        mapped = np.asarray(
            [
                threshold_lookup.loc[(repeat, outer_fold)]
                for repeat, outer_fold in frame[["repeat", "outer_fold"]].itertuples(
                    index=False, name=None
                )
            ],
            dtype=float,
        )
        if not np.allclose(
            frame["selected_threshold"].to_numpy(dtype=float),
            mapped,
            rtol=0.0,
            atol=NUMERIC_ABSOLUTE_TOLERANCE,
        ):
            raise PersistedArtifactValidationError(
                f"{label} selected thresholds do not match threshold records."
            )


def _validate_metrics_and_thresholds(
    root: Path,
    config: ResearchConfig,
    outer_predictions: pd.DataFrame,
    threshold_predictions: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    repeat_metrics: pd.DataFrame,
    threshold_curves: pd.DataFrame | None,
) -> None:
    _assert_unique(fold_metrics, ["repeat", "outer_fold"], "outer-fold metric")
    recomputed_curves: list[pd.DataFrame] = []
    for selected in selected_thresholds.itertuples(index=False):
        key = (int(selected.repeat), int(selected.outer_fold))
        threshold_frame = threshold_predictions.loc[
            (threshold_predictions["repeat"] == key[0])
            & (threshold_predictions["outer_fold"] == key[1])
        ]
        outer_frame = outer_predictions.loc[
            (outer_predictions["repeat"] == key[0])
            & (outer_predictions["outer_fold"] == key[1])
        ]
        repeat_seeds = pd.concat(
            [threshold_frame["repeat_seed"], outer_frame["repeat_seed"]],
            ignore_index=True,
        ).unique()
        if len(repeat_seeds) != 1:
            raise PersistedArtifactValidationError(f"Repeat seed differs for {key}.")
        repeat_seed = int(repeat_seeds[0])

        selection = select_balanced_accuracy_threshold(
            threshold_frame["target"],
            threshold_frame["probability"].to_numpy(),
            config.plan_payload["threshold_policy"],
        )
        expected_selected = {
            "repeat": key[0],
            "repeat_seed": repeat_seed,
            "outer_fold": key[1],
            "selected_threshold": selection.threshold,
            "threshold_selection_balanced_accuracy": selection.balanced_accuracy,
            "status": selection.status,
            "degenerate": selection.degenerate,
        }
        _assert_json_equal(
            selected._asdict(),
            expected_selected,
            f"selected threshold {key}",
        )

        metrics = calculate_outer_metrics(
            outer_frame["target"],
            outer_frame["probability"].to_numpy(),
            outer_frame["prediction"].to_numpy(),
            selected_threshold=selection.threshold,
        )
        actual_metric_rows = fold_metrics.loc[
            (fold_metrics["repeat"] == key[0]) & (fold_metrics["outer_fold"] == key[1])
        ]
        if len(actual_metric_rows) != 1:
            raise PersistedArtifactValidationError(
                f"Expected one outer-fold metric record for {key}."
            )
        actual_metric = actual_metric_rows.iloc[0].to_dict()
        duration = float(actual_metric["duration_seconds"])
        if not math.isfinite(duration) or duration < 0:
            raise PersistedArtifactValidationError(
                f"Fold duration is invalid for {key}."
            )
        expected_metric = {
            "repeat": key[0],
            "repeat_seed": repeat_seed,
            "outer_fold": key[1],
            "training_rows": len(threshold_frame),
            "validation_rows": len(outer_frame),
            "threshold_selection_balanced_accuracy": selection.balanced_accuracy,
            "threshold_status": selection.status,
            "threshold_degenerate": selection.degenerate,
            "duration_seconds": duration,
            **metrics,
        }
        _assert_json_equal(actual_metric, expected_metric, f"outer-fold metrics {key}")

        curve = selection.scores.copy()
        curve.insert(0, "outer_fold", key[1])
        curve.insert(0, "repeat", key[0])
        recomputed_curves.append(curve)
        _validate_progress_fold(
            root,
            config,
            key,
            threshold_frame,
            outer_frame,
            expected_selected,
            expected_metric,
            curve,
        )

    expected_repeat_metrics = build_repeat_metrics(outer_predictions)
    _assert_frame_equal(
        repeat_metrics,
        expected_repeat_metrics,
        ["repeat"],
        "repeat metrics",
    )
    expected_aggregate = aggregate_repeat_metrics(
        expected_repeat_metrics,
        fold_metrics,
    )
    _assert_json_equal(
        _read_json(root / "metrics" / "aggregate.json"),
        expected_aggregate,
        "aggregate metrics",
    )
    expected_threshold_summary = summarize_thresholds(selected_thresholds)
    _assert_json_equal(
        _read_json(root / "thresholds" / "threshold_summary.json"),
        expected_threshold_summary,
        "threshold summary",
    )
    if threshold_curves is not None:
        _assert_frame_equal(
            threshold_curves,
            pd.concat(recomputed_curves, ignore_index=True),
            ["repeat", "outer_fold", "threshold"],
            "consolidated threshold curves",
        )


def _validate_progress_fold(
    root: Path,
    config: ResearchConfig,
    key: tuple[int, int],
    threshold_frame: pd.DataFrame,
    outer_frame: pd.DataFrame,
    selected: dict[str, Any],
    metrics: dict[str, Any],
    curve: pd.DataFrame,
) -> None:
    directory = root / "fold_progress" / f"repeat_{key[0]:02d}_outer_fold_{key[1]:02d}"
    required = {
        "threshold_selection_oof.parquet",
        "outer_validation.parquet",
        "selected_threshold.json",
        "fold_metrics.json",
    }
    if config.payload["artifacts"]["save_threshold_curves"]:
        required.add("threshold_curve.parquet")
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    if actual != required:
        raise PersistedArtifactValidationError(
            f"Progress artifact set differs for {key}: {sorted(actual)}"
        )
    progress_threshold = pd.read_parquet(directory / "threshold_selection_oof.parquet")
    progress_outer = pd.read_parquet(directory / "outer_validation.parquet")
    _assert_schema(
        progress_threshold,
        THRESHOLD_PREDICTION_SCHEMA,
        f"progress threshold predictions {key}",
    )
    _assert_schema(
        progress_outer,
        OUTER_PREDICTION_SCHEMA,
        f"progress outer predictions {key}",
    )
    _assert_frame_equal(
        progress_threshold,
        threshold_frame,
        ["row_position"],
        f"progress threshold predictions {key}",
    )
    _assert_frame_equal(
        progress_outer,
        outer_frame,
        ["row_position"],
        f"progress outer predictions {key}",
    )
    _assert_json_equal(
        _read_json(directory / "selected_threshold.json"),
        selected,
        f"progress selected threshold {key}",
    )
    _assert_json_equal(
        _read_json(directory / "fold_metrics.json"),
        metrics,
        f"progress fold metrics {key}",
    )
    if config.payload["artifacts"]["save_threshold_curves"]:
        progress_curve = pd.read_parquet(directory / "threshold_curve.parquet")
        _assert_schema(
            progress_curve,
            THRESHOLD_CURVE_SCHEMA,
            f"progress threshold curve {key}",
        )
        _assert_frame_equal(
            progress_curve,
            curve,
            ["threshold"],
            f"progress threshold curve {key}",
        )


def _required_artifact_paths(config: ResearchConfig) -> list[str]:
    required = [
        "resolved_config.yaml",
        "run_metadata.json",
        "execution_status.json",
        "evaluation_plan.json",
        "candidate_contract.json",
        "run_implementation.json",
        "loaded_modules.json",
        "dataset_fingerprints.json",
        "feature_schema.json",
        "splits/outer_assignments.parquet",
        "splits/threshold_selection_assignments.parquet",
        "predictions/threshold_selection_oof.parquet",
        "predictions/outer_validation.parquet",
        "thresholds/selected_thresholds.csv",
        "thresholds/threshold_summary.json",
        "metrics/outer_folds.csv",
        "metrics/repeats.csv",
        "metrics/aggregate.json",
    ]
    if config.payload["artifacts"]["save_threshold_curves"]:
        required.append("thresholds/threshold_curves.parquet")
    return required


def _assert_schema(
    frame: pd.DataFrame,
    schema: Sequence[tuple[str, str]],
    label: str,
) -> None:
    actual = tuple((name, str(dtype)) for name, dtype in frame.dtypes.items())
    expected = tuple(schema)
    if actual != expected:
        raise PersistedArtifactValidationError(
            f"{label} schema differs: expected={expected}, actual={actual}"
        )


def _assert_unique(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    if frame.duplicated(keys).any():
        raise PersistedArtifactValidationError(
            f"{label} records contain duplicate keys: {keys}"
        )


def _assert_frame_equal(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    sort_by: list[str],
    label: str,
) -> None:
    left = actual.sort_values(sort_by, ignore_index=True)
    right = expected.sort_values(sort_by, ignore_index=True)
    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=True,
            check_exact=False,
            rtol=0.0,
            atol=NUMERIC_ABSOLUTE_TOLERANCE,
        )
    except AssertionError as error:
        raise PersistedArtifactValidationError(
            f"{label} differs from its recomputed value: {error}"
        ) from error


def _assert_json_equal(actual: Any, expected: Any, label: str) -> None:
    difference = _first_json_difference(actual, expected, label)
    if difference is not None:
        raise PersistedArtifactValidationError(difference)


def _first_json_difference(actual: Any, expected: Any, path: str) -> str | None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return f"{path} must be a mapping."
        if set(actual) != set(expected):
            return (
                f"{path} keys differ: expected={sorted(expected)}, "
                f"actual={sorted(actual)}"
            )
        for key in expected:
            difference = _first_json_difference(
                actual[key],
                expected[key],
                f"{path}.{key}",
            )
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return f"{path} list shape differs."
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            difference = _first_json_difference(
                actual_item,
                expected_item,
                f"{path}[{index}]",
            )
            if difference is not None:
                return difference
        return None
    if isinstance(expected, (bool, np.bool_)):
        if not isinstance(actual, (bool, np.bool_)) or bool(actual) != bool(expected):
            return f"{path} differs: expected={expected!r}, actual={actual!r}"
        return None

    if (
        isinstance(expected, (int, float, np.number))
        and not isinstance(expected, (bool, np.bool_))
        and isinstance(actual, (int, float, np.number))
        and not isinstance(actual, (bool, np.bool_))
    ):
        actual_number = float(actual)
        expected_number = float(expected)
        if not math.isfinite(actual_number) or not math.isclose(
            actual_number,
            expected_number,
            rel_tol=0.0,
            abs_tol=NUMERIC_ABSOLUTE_TOLERANCE,
        ):
            return f"{path} differs: expected={expected!r}, actual={actual!r}"
        return None
    if actual != expected or type(actual) is not type(expected):
        return f"{path} differs: expected={expected!r}, actual={actual!r}"
    return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise PersistedArtifactValidationError(
            f"Could not read valid JSON artifact {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise PersistedArtifactValidationError(
            f"JSON artifact must contain an object: {path}"
        )
    return payload
