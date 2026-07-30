from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.churn_ml.experiment_v2_schema import ordered_feature_schema_sha256
from src.churn_ml.research_data import canonical_sha256
from src.churn_ml.research_protocol import (
    EvaluationAssignments,
    aggregate_repeat_metrics,
    build_evaluation_assignments,
    build_evaluation_plan_identity,
    build_repeat_metrics,
    calculate_outer_metrics,
    select_balanced_accuracy_threshold,
    summarize_thresholds,
    validate_evaluation_assignments,
)
from src.churn_ml.research_v2_config import ResearchV2Config
from src.churn_ml.research_v2_data import load_research_v2_training_data


NUMERIC_ABSOLUTE_TOLERANCE = 1e-12
ARTIFACT_HASHING_METHOD = "sha256_raw_artifact_bytes_v2"

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


class ResearchV2SemanticValidationError(RuntimeError):
    """Raised when persisted v2 artifacts do not prove a valid run."""


def validate_portable_payload_paths(
    payload: Any,
    *,
    label: str,
    project_root: Path,
) -> None:
    """Reject host-absolute path strings anywhere in a portable payload."""
    _validate_portable_value(
        payload,
        field_path=label,
        project_root=project_root.resolve(),
    )


def validate_research_v2_run(
    root: Path,
    config: ResearchV2Config,
    *,
    expected_hashes: Mapping[str, str],
    require_success: bool,
    verify_manifest: bool,
) -> None:
    """Recompute v2 run semantics for completion or independent read-only use."""
    root = root.resolve()
    if not root.is_dir():
        raise ResearchV2SemanticValidationError(f"Run directory is missing: {root}.")
    if (root / "_FAILED").exists():
        raise ResearchV2SemanticValidationError("Run contains _FAILED.")
    if require_success:
        if not (root / "_SUCCESS").is_file():
            raise ResearchV2SemanticValidationError("Run does not contain _SUCCESS.")
    elif (root / "_SUCCESS").exists():
        raise ResearchV2SemanticValidationError(
            "_SUCCESS exists before semantic completion validation."
        )

    expected_paths = _required_artifact_paths(config)
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {"artifact_manifest.json", "_SUCCESS", "_FAILED"}
    }
    forbidden = sorted(path for path in actual_paths if _forbidden_path(path))
    if forbidden:
        raise ResearchV2SemanticValidationError(
            f"Forbidden artifacts or paths are present: {forbidden}."
        )
    if actual_paths != expected_paths:
        raise ResearchV2SemanticValidationError(
            "Ordinary artifact inventory differs; "
            f"missing={sorted(expected_paths - actual_paths)}, "
            f"unexpected={sorted(actual_paths - expected_paths)}."
        )

    resolved = _read_yaml(root / "resolved_config.yaml")
    validate_portable_payload_paths(
        resolved,
        label="resolved_config",
        project_root=config.project_root,
    )
    _assert_json_equal(resolved, config.resolved_payload(), "resolved configuration")
    metadata = _read_json(root / "run_metadata.json")
    status = _read_json(root / "execution_status.json")
    _validate_terminal_records(root, config, metadata, status, expected_hashes)

    data = load_research_v2_training_data(config)
    _assert_json_equal(
        _read_json(root / "dataset_fingerprints.json"),
        data.fingerprints,
        "dataset fingerprints",
    )
    provenance = _read_json(root / "dataset_provenance.json")
    _assert_json_equal(
        provenance,
        data.dataset_provenance,
        "dataset provenance",
    )
    _validate_dataset_provenance(provenance, data.fingerprints, metadata)
    feature_schema = _read_json(root / "feature_schema.json")
    _assert_json_equal(
        feature_schema,
        data.pipeline_output.schema.to_dict(),
        "feature schema",
    )
    _validate_feature_schema(feature_schema)

    outer_assignments = pd.read_parquet(root / "splits" / "outer_assignments.parquet")
    threshold_assignments = pd.read_parquet(
        root / "splits" / "threshold_selection_assignments.parquet"
    )
    _assert_schema(outer_assignments, OUTER_ASSIGNMENT_SCHEMA, "outer assignments")
    _assert_schema(
        threshold_assignments,
        THRESHOLD_ASSIGNMENT_SCHEMA,
        "threshold assignments",
    )
    generated = build_evaluation_assignments(data.y, config.plan_payload)
    _assert_frame_equal(
        outer_assignments,
        generated.outer,
        ["repeat", "outer_fold", "row_position"],
        "outer assignments",
    )
    _assert_frame_equal(
        threshold_assignments,
        generated.threshold_selection,
        [
            "repeat",
            "outer_fold",
            "threshold_selection_fold",
            "row_position",
        ],
        "threshold assignments",
    )
    assignments = EvaluationAssignments(
        outer=outer_assignments,
        threshold_selection=threshold_assignments,
    )
    validate_evaluation_assignments(assignments, len(data.y), config.plan_payload)
    _validate_identities(
        root,
        config,
        data.fingerprints,
        assignments,
        expected_hashes,
        metadata,
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
    _assert_schema(outer_predictions, OUTER_PREDICTION_SCHEMA, "outer predictions")
    _assert_schema(
        threshold_predictions,
        THRESHOLD_PREDICTION_SCHEMA,
        "threshold predictions",
    )
    _assert_schema(
        selected_thresholds,
        SELECTED_THRESHOLD_SCHEMA,
        "selected thresholds",
    )
    _assert_schema(fold_metrics, OUTER_FOLD_METRIC_SCHEMA, "outer-fold metrics")
    _assert_schema(repeat_metrics, REPEAT_METRIC_SCHEMA, "repeat metrics")
    threshold_curves: pd.DataFrame | None = None
    if config.payload["persistence"]["threshold_curves"]:
        threshold_curves = pd.read_parquet(
            root / "thresholds" / "threshold_curves.parquet"
        )
        _assert_schema(threshold_curves, THRESHOLD_CURVE_SCHEMA, "threshold curves")

    expected_fold_keys = _expected_fold_keys(config)
    _assert_exact_fold_keys(selected_thresholds, expected_fold_keys, "threshold")
    _assert_exact_fold_keys(fold_metrics, expected_fold_keys, "metric")
    _validate_prediction_assignments(
        outer_assignments,
        threshold_assignments,
        outer_predictions,
        threshold_predictions,
        selected_thresholds,
        data.y,
    )
    _validate_metrics_thresholds_and_progress(
        root,
        config,
        expected_fold_keys,
        outer_predictions,
        threshold_predictions,
        selected_thresholds,
        fold_metrics,
        repeat_metrics,
        threshold_curves,
    )
    if verify_manifest:
        manifest = _read_json(root / "artifact_manifest.json")
        validate_artifact_manifest(root, manifest)
        success = _read_json(root / "_SUCCESS")
        if set(success) != {"schema_version", "artifact_manifest_sha256"}:
            raise ResearchV2SemanticValidationError("_SUCCESS schema differs.")
        if (
            success["schema_version"] != 2
            or success["artifact_manifest_sha256"] != manifest["manifest_sha256"]
        ):
            raise ResearchV2SemanticValidationError(
                "_SUCCESS does not identify the verified manifest."
            )


def build_artifact_manifest(root: Path) -> dict[str, Any]:
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
        "schema_version": 2,
        "hashing_method": ARTIFACT_HASHING_METHOD,
        "files": records,
    }
    return {**canonical, "manifest_sha256": canonical_sha256(canonical)}


def validate_artifact_manifest(root: Path, payload: Mapping[str, Any]) -> None:
    if set(payload) != {
        "schema_version",
        "hashing_method",
        "files",
        "manifest_sha256",
    }:
        raise ResearchV2SemanticValidationError(
            "Artifact manifest has missing or extra keys."
        )
    expected = build_artifact_manifest(root)
    _assert_json_equal(dict(payload), expected, "artifact manifest")


def _validate_terminal_records(
    root: Path,
    config: ResearchV2Config,
    metadata: Mapping[str, Any],
    status: Mapping[str, Any],
    expected_hashes: Mapping[str, str],
) -> None:
    expected_folds = len(_expected_fold_keys(config))
    required_metadata = {
        "schema_version",
        "experiment_id",
        "plan_id",
        "feature_pipeline_id",
        "candidate_adapter_id",
        "hashes",
        "status",
        "started_at_utc",
        "environment",
        "runtime",
        "git",
        "competition_assets_accessed",
        "tracking_enabled",
        "run_id",
        "finished_at_utc",
        "evaluation_duration_seconds",
        "dataset_provenance",
    }
    if set(metadata) != required_metadata:
        raise ResearchV2SemanticValidationError(
            "Run metadata keys differ; "
            f"missing={sorted(required_metadata - set(metadata))}, "
            f"unknown={sorted(set(metadata) - required_metadata)}."
        )
    expected_values = {
        "schema_version": 2,
        "experiment_id": config.experiment_id,
        "plan_id": config.plan_id,
        "feature_pipeline_id": config.pipeline_id,
        "candidate_adapter_id": config.adapter_id,
        "status": "completed",
        "run_id": root.name,
        "competition_assets_accessed": False,
        "tracking_enabled": False,
    }
    for name, expected in expected_values.items():
        if metadata.get(name) != expected or type(metadata.get(name)) is not type(
            expected
        ):
            raise ResearchV2SemanticValidationError(f"run_metadata.{name} differs.")
    _assert_json_equal(metadata["hashes"], dict(expected_hashes), "metadata hashes")
    duration = metadata["evaluation_duration_seconds"]
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or float(duration) < 0
    ):
        raise ResearchV2SemanticValidationError("Evaluation duration is invalid.")
    for name in ("started_at_utc", "finished_at_utc"):
        value = metadata[name]
        if not isinstance(value, str) or not value.endswith("+00:00"):
            raise ResearchV2SemanticValidationError(f"run_metadata.{name} must be UTC.")
    expected_status = {
        "schema_version": 2,
        "run_id": root.name,
        "status": "completed",
        "started_at_utc": metadata["started_at_utc"],
        "finished_at_utc": metadata["finished_at_utc"],
        "completed_outer_folds": expected_folds,
        "expected_outer_folds": expected_folds,
        "failure": None,
    }
    _assert_json_equal(status, expected_status, "execution status")


def _validate_identities(
    root: Path,
    config: ResearchV2Config,
    fingerprints: dict[str, Any],
    assignments: EvaluationAssignments,
    expected_hashes: Mapping[str, str],
    metadata: Mapping[str, Any],
) -> None:
    plan, plan_hash = build_evaluation_plan_identity(
        config.plan_payload,
        fingerprints,
        assignments,
    )
    if plan_hash != expected_hashes["plan"]:
        raise ResearchV2SemanticValidationError("Plan hash changed.")
    _validate_identity_file(
        root / "identities" / "evaluation_plan.json",
        expected_hashes["plan"],
        project_root=config.project_root,
        expected_canonical=plan,
    )
    for name in (
        "feature_pipeline",
        "candidate_adapter",
        "candidate",
        "source",
        "loaded_modules",
    ):
        _validate_identity_file(
            root / "identities" / f"{name}.json",
            expected_hashes[name],
            project_root=config.project_root,
        )
    candidate = _read_json(root / "identities" / "candidate.json")["canonical"]
    if (
        candidate["feature_pipeline"]["id"] != config.pipeline_id
        or candidate["feature_pipeline"]["sha256"]
        != expected_hashes["feature_pipeline"]
        or candidate["candidate_adapter"]["id"] != config.adapter_id
        or candidate["candidate_adapter"]["sha256"]
        != expected_hashes["candidate_adapter"]
        or candidate["dataset_version"] != config.dataset_version
    ):
        raise ResearchV2SemanticValidationError(
            "Complete-candidate component identities are inconsistent."
        )
    _assert_json_equal(metadata["hashes"], dict(expected_hashes), "identity hashes")


def _validate_identity_file(
    path: Path,
    expected_hash: str,
    *,
    project_root: Path,
    expected_canonical: Mapping[str, Any] | None = None,
) -> None:
    payload = _read_json(path)
    if set(payload) != {"sha256", "canonical"}:
        raise ResearchV2SemanticValidationError(
            f"Identity schema differs: {path.name}."
        )
    if (
        payload["sha256"] != expected_hash
        or canonical_sha256(payload["canonical"]) != expected_hash
    ):
        raise ResearchV2SemanticValidationError(f"Identity hash differs: {path.name}.")
    if expected_canonical is not None:
        _assert_json_equal(
            payload["canonical"],
            dict(expected_canonical),
            path.stem,
        )
    validate_portable_payload_paths(
        payload["canonical"],
        label=f"identities.{path.stem}.canonical",
        project_root=project_root,
    )


def _validate_portable_value(
    value: Any,
    *,
    field_path: str,
    project_root: Path,
) -> None:
    if isinstance(value, str):
        _reject_nonportable_path(
            value,
            field_path=field_path,
            project_root=project_root,
        )
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str):
                _reject_nonportable_path(
                    key,
                    field_path=f"{field_path}.<key>",
                    project_root=project_root,
                )
            _validate_portable_value(
                child,
                field_path=f"{field_path}.{key}",
                project_root=project_root,
            )
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for index, child in enumerate(value):
            _validate_portable_value(
                child,
                field_path=f"{field_path}[{index}]",
                project_root=project_root,
            )


def _reject_nonportable_path(
    value: str,
    *,
    field_path: str,
    project_root: Path,
) -> None:
    windows_path = PureWindowsPath(value)
    posix_path = PurePosixPath(value)
    repository_path = PureWindowsPath(str(project_root))
    is_repository_path = (
        windows_path == repository_path or repository_path in windows_path.parents
    )
    if (
        is_repository_path
        or bool(windows_path.drive)
        or bool(windows_path.root)
        or posix_path.is_absolute()
    ):
        raise ResearchV2SemanticValidationError(
            f"Non-portable path string at {field_path}: {value!r}."
        )


def _validate_feature_schema(schema: Mapping[str, Any]) -> None:
    checks = {
        "source": ("source_feature_names", "source_schema_sha256"),
        "model input": (
            "model_input_feature_names",
            "model_input_schema_sha256",
        ),
        "transformed": (
            "transformed_feature_names",
            "transformed_schema_sha256",
        ),
    }
    for label, (names_key, hash_key) in checks.items():
        names = schema.get(names_key)
        if not isinstance(names, list) or not all(
            isinstance(name, str) for name in names
        ):
            raise ResearchV2SemanticValidationError(
                f"{label} ordered feature schema is malformed."
            )
        if ordered_feature_schema_sha256(names) != schema.get(hash_key):
            raise ResearchV2SemanticValidationError(
                f"{label} ordered feature schema hash differs."
            )


def _validate_dataset_provenance(
    provenance: Mapping[str, Any],
    fingerprints: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    required = {
        "dataset_id",
        "parent_dataset_id",
        "hypothesis",
        "n_features",
        "schema_hash",
        "train_content_hash",
        "target_hash",
        "target_dependency",
        "train_row_identity_hash",
        "registry_schema_version",
    }
    missing = sorted(required - set(provenance))
    if missing:
        raise ResearchV2SemanticValidationError(
            f"dataset_provenance is missing keys: {missing}."
        )
    if provenance.get("dataset_id") != fingerprints.get("dataset_version"):
        raise ResearchV2SemanticValidationError(
            "dataset_provenance.dataset_id disagrees with fingerprints."
        )
    embedded = fingerprints.get("dataset_provenance")
    if embedded != dict(provenance):
        raise ResearchV2SemanticValidationError(
            "dataset_provenance disagrees with fingerprint embedding."
        )
    metadata_provenance = metadata.get("dataset_provenance")
    if metadata_provenance != dict(provenance):
        raise ResearchV2SemanticValidationError(
            "run_metadata.dataset_provenance disagrees with dataset_provenance.json."
        )
    row_identity = fingerprints.get("row_position_identity")
    if not isinstance(row_identity, Mapping):
        raise ResearchV2SemanticValidationError(
            "fingerprints.row_position_identity is malformed."
        )
    bound = row_identity.get("bound_train_row_identity_hash")
    train_hash = provenance.get("train_row_identity_hash")
    if train_hash is not None and bound != train_hash:
        raise ResearchV2SemanticValidationError(
            "row_position identity is not bound to train_row_identity_hash."
        )


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
    threshold_lookup = selected_thresholds.set_index(["repeat", "outer_fold"])[
        "selected_threshold"
    ]
    for label, frame in (
        ("outer", outer_predictions),
        ("threshold", threshold_predictions),
    ):
        positions = frame["row_position"].to_numpy(dtype=np.int64)
        expected_targets = y.iloc[positions].to_numpy(dtype="int8")
        if not np.array_equal(frame["target"].to_numpy(), expected_targets):
            raise ResearchV2SemanticValidationError(
                f"{label} prediction targets differ from row keys."
            )
        probabilities = frame["probability"].to_numpy(dtype=float)
        thresholds = frame["selected_threshold"].to_numpy(dtype=float)
        if (
            not np.isfinite(probabilities).all()
            or ((probabilities < 0.0) | (probabilities > 1.0)).any()
            or not np.isfinite(thresholds).all()
        ):
            raise ResearchV2SemanticValidationError(
                f"{label} probabilities or thresholds are invalid."
            )
        expected_labels = (probabilities >= thresholds).astype("int8")
        if not np.array_equal(frame["prediction"].to_numpy(), expected_labels):
            raise ResearchV2SemanticValidationError(
                f"{label} labels do not use probability >= selected_threshold."
            )
        mapped = np.asarray(
            [
                threshold_lookup.loc[(repeat, fold)]
                for repeat, fold in frame[["repeat", "outer_fold"]].itertuples(
                    index=False,
                    name=None,
                )
            ],
            dtype=float,
        )
        if not np.allclose(
            thresholds,
            mapped,
            rtol=0.0,
            atol=NUMERIC_ABSOLUTE_TOLERANCE,
        ):
            raise ResearchV2SemanticValidationError(
                f"{label} threshold assignment differs."
            )


def _validate_metrics_thresholds_and_progress(
    root: Path,
    config: ResearchV2Config,
    expected_fold_keys: set[tuple[int, int]],
    outer_predictions: pd.DataFrame,
    threshold_predictions: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    repeat_metrics: pd.DataFrame,
    threshold_curves: pd.DataFrame | None,
) -> None:
    recomputed_curves: list[pd.DataFrame] = []
    for repeat, outer_fold in sorted(expected_fold_keys):
        key = (repeat, outer_fold)
        threshold_frame = threshold_predictions.loc[
            (threshold_predictions["repeat"] == repeat)
            & (threshold_predictions["outer_fold"] == outer_fold)
        ]
        outer_frame = outer_predictions.loc[
            (outer_predictions["repeat"] == repeat)
            & (outer_predictions["outer_fold"] == outer_fold)
        ]
        selected_row = selected_thresholds.loc[
            (selected_thresholds["repeat"] == repeat)
            & (selected_thresholds["outer_fold"] == outer_fold)
        ]
        metric_row = fold_metrics.loc[
            (fold_metrics["repeat"] == repeat)
            & (fold_metrics["outer_fold"] == outer_fold)
        ]
        if len(selected_row) != 1 or len(metric_row) != 1:
            raise ResearchV2SemanticValidationError(
                f"Fold records are not unique for {key}."
            )
        repeat_seeds = pd.concat(
            [threshold_frame["repeat_seed"], outer_frame["repeat_seed"]],
            ignore_index=True,
        ).unique()
        if len(repeat_seeds) != 1:
            raise ResearchV2SemanticValidationError(f"Repeat seed differs for {key}.")
        repeat_seed = int(repeat_seeds[0])
        selection = select_balanced_accuracy_threshold(
            threshold_frame["target"],
            threshold_frame["probability"].to_numpy(),
            config.plan_payload["threshold_policy"],
        )
        expected_selected = {
            "repeat": repeat,
            "repeat_seed": repeat_seed,
            "outer_fold": outer_fold,
            "selected_threshold": selection.threshold,
            "threshold_selection_balanced_accuracy": selection.balanced_accuracy,
            "status": selection.status,
            "degenerate": selection.degenerate,
        }
        _assert_json_equal(
            selected_row.iloc[0].to_dict(),
            expected_selected,
            f"selected threshold {key}",
        )
        metrics = calculate_outer_metrics(
            outer_frame["target"],
            outer_frame["probability"].to_numpy(),
            outer_frame["prediction"].to_numpy(),
            selected_threshold=selection.threshold,
        )
        actual_metric = metric_row.iloc[0].to_dict()
        duration = actual_metric["duration_seconds"]
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) < 0
        ):
            raise ResearchV2SemanticValidationError(
                f"Fold duration is invalid for {key}."
            )
        expected_metric = {
            "repeat": repeat,
            "repeat_seed": repeat_seed,
            "outer_fold": outer_fold,
            "training_rows": len(threshold_frame),
            "validation_rows": len(outer_frame),
            "threshold_selection_balanced_accuracy": selection.balanced_accuracy,
            "threshold_status": selection.status,
            "threshold_degenerate": selection.degenerate,
            "duration_seconds": float(duration),
            **metrics,
        }
        _assert_json_equal(
            actual_metric,
            expected_metric,
            f"outer-fold metrics {key}",
        )
        curve = selection.scores.copy()
        curve.insert(0, "outer_fold", outer_fold)
        curve.insert(0, "repeat", repeat)
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
    expected_repeats = build_repeat_metrics(outer_predictions)
    _assert_frame_equal(
        repeat_metrics,
        expected_repeats,
        ["repeat"],
        "repeat metrics",
    )
    expected_aggregate = aggregate_repeat_metrics(expected_repeats, fold_metrics)
    _assert_json_equal(
        _read_json(root / "metrics" / "aggregate.json"),
        expected_aggregate,
        "aggregate metrics",
    )
    _assert_json_equal(
        _read_json(root / "thresholds" / "threshold_summary.json"),
        summarize_thresholds(selected_thresholds),
        "threshold summary",
    )
    if threshold_curves is not None:
        _assert_frame_equal(
            threshold_curves,
            pd.concat(recomputed_curves, ignore_index=True),
            ["repeat", "outer_fold", "threshold"],
            "threshold curves",
        )


def _validate_progress_fold(
    root: Path,
    config: ResearchV2Config,
    key: tuple[int, int],
    threshold_frame: pd.DataFrame,
    outer_frame: pd.DataFrame,
    selected: dict[str, Any],
    metrics: dict[str, Any],
    curve: pd.DataFrame,
) -> None:
    directory = root / "fold_progress" / f"repeat_{key[0]:02d}_outer_fold_{key[1]:02d}"
    expected_names = {
        "threshold_selection_oof.parquet",
        "outer_validation.parquet",
        "selected_threshold.json",
        "fold_metrics.json",
    }
    if config.payload["persistence"]["threshold_curves"]:
        expected_names.add("threshold_curve.parquet")
    actual_names = {path.name for path in directory.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise ResearchV2SemanticValidationError(
            f"Progress artifact inventory differs for {key}."
        )
    progress_threshold = pd.read_parquet(directory / "threshold_selection_oof.parquet")
    progress_outer = pd.read_parquet(directory / "outer_validation.parquet")
    _assert_schema(
        progress_threshold,
        THRESHOLD_PREDICTION_SCHEMA,
        f"progress threshold {key}",
    )
    _assert_schema(progress_outer, OUTER_PREDICTION_SCHEMA, f"progress outer {key}")
    _assert_frame_equal(
        progress_threshold,
        threshold_frame,
        ["row_position"],
        f"progress threshold {key}",
    )
    _assert_frame_equal(
        progress_outer,
        outer_frame,
        ["row_position"],
        f"progress outer {key}",
    )
    _assert_json_equal(
        _read_json(directory / "selected_threshold.json"),
        selected,
        f"progress selected threshold {key}",
    )
    _assert_json_equal(
        _read_json(directory / "fold_metrics.json"),
        metrics,
        f"progress metrics {key}",
    )
    if config.payload["persistence"]["threshold_curves"]:
        progress_curve = pd.read_parquet(directory / "threshold_curve.parquet")
        _assert_schema(
            progress_curve,
            THRESHOLD_CURVE_SCHEMA,
            f"progress curve {key}",
        )
        _assert_frame_equal(
            progress_curve,
            curve,
            ["threshold"],
            f"progress curve {key}",
        )


def _expected_fold_keys(config: ResearchV2Config) -> set[tuple[int, int]]:
    repeats = len(config.plan_payload["outer_evaluation"]["repeat_seeds"])
    folds = config.plan_payload["outer_evaluation"]["n_splits"]
    return {
        (repeat, fold)
        for repeat in range(1, repeats + 1)
        for fold in range(1, folds + 1)
    }


def _assert_exact_fold_keys(
    frame: pd.DataFrame,
    expected: set[tuple[int, int]],
    label: str,
) -> None:
    keys = list(frame[["repeat", "outer_fold"]].itertuples(index=False, name=None))
    actual = set(keys)
    if len(keys) != len(actual) or actual != expected:
        raise ResearchV2SemanticValidationError(
            f"{label} fold keys differ; expected={sorted(expected)}, "
            f"actual={sorted(actual)}, record_count={len(keys)}."
        )


def _required_artifact_paths(config: ResearchV2Config) -> set[str]:
    paths = {
        "resolved_config.yaml",
        "run_metadata.json",
        "execution_status.json",
        "dataset_fingerprints.json",
        "dataset_provenance.json",
        "feature_schema.json",
        "identities/evaluation_plan.json",
        "identities/feature_pipeline.json",
        "identities/candidate_adapter.json",
        "identities/candidate.json",
        "identities/source.json",
        "identities/loaded_modules.json",
        "splits/outer_assignments.parquet",
        "splits/threshold_selection_assignments.parquet",
        "predictions/threshold_selection_oof.parquet",
        "predictions/outer_validation.parquet",
        "thresholds/selected_thresholds.csv",
        "thresholds/threshold_summary.json",
        "metrics/outer_folds.csv",
        "metrics/repeats.csv",
        "metrics/aggregate.json",
    }
    if config.payload["persistence"]["threshold_curves"]:
        paths.add("thresholds/threshold_curves.parquet")
    for repeat, fold in _expected_fold_keys(config):
        prefix = f"fold_progress/repeat_{repeat:02d}_outer_fold_{fold:02d}"
        paths.update(
            {
                f"{prefix}/threshold_selection_oof.parquet",
                f"{prefix}/outer_validation.parquet",
                f"{prefix}/selected_threshold.json",
                f"{prefix}/fold_metrics.json",
            }
        )
        if config.payload["persistence"]["threshold_curves"]:
            paths.add(f"{prefix}/threshold_curve.parquet")
    return paths


def _forbidden_path(relative: str) -> bool:
    normalized = relative.lower().replace("\\", "/")
    forbidden = (
        "competition",
        "sample_submission",
        "x_test",
        "kaggle",
        "autogluon",
        "submission",
        "/models/",
        ".joblib",
        ".pkl",
    )
    return any(marker in normalized for marker in forbidden)


def _assert_schema(
    frame: pd.DataFrame,
    schema: Sequence[tuple[str, str]],
    label: str,
) -> None:
    actual = tuple((name, str(dtype)) for name, dtype in frame.dtypes.items())
    if actual != tuple(schema):
        raise ResearchV2SemanticValidationError(
            f"{label} schema differs: expected={tuple(schema)}, actual={actual}."
        )


def _assert_unique(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    if frame.duplicated(keys).any():
        raise ResearchV2SemanticValidationError(
            f"{label} records contain duplicate keys: {keys}."
        )


def _assert_frame_equal(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    sort_by: list[str],
    label: str,
) -> None:
    try:
        pd.testing.assert_frame_equal(
            actual.sort_values(sort_by, ignore_index=True),
            expected.sort_values(sort_by, ignore_index=True),
            check_dtype=True,
            check_exact=False,
            rtol=0.0,
            atol=NUMERIC_ABSOLUTE_TOLERANCE,
        )
    except AssertionError as error:
        raise ResearchV2SemanticValidationError(
            f"{label} differs from its recomputed value: {error}"
        ) from error


def _assert_json_equal(actual: Any, expected: Any, label: str) -> None:
    difference = _first_json_difference(actual, expected, label)
    if difference is not None:
        raise ResearchV2SemanticValidationError(difference)


def _first_json_difference(actual: Any, expected: Any, path: str) -> str | None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return f"{path} must be a mapping."
        if set(actual) != set(expected):
            return (
                f"{path} keys differ: expected={sorted(expected)}, "
                f"actual={sorted(actual)}."
            )
        for key, value in expected.items():
            difference = _first_json_difference(actual[key], value, f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return f"{path} list shape differs."
        for index, (actual_value, expected_value) in enumerate(
            zip(actual, expected, strict=True)
        ):
            difference = _first_json_difference(
                actual_value,
                expected_value,
                f"{path}[{index}]",
            )
            if difference is not None:
                return difference
        return None
    if isinstance(expected, bool):
        if not isinstance(actual, bool) or actual is not expected:
            return f"{path} differs: expected={expected!r}, actual={actual!r}."
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
            return f"{path} differs: expected={expected!r}, actual={actual!r}."
        return None
    if type(actual) is not type(expected) or actual != expected:
        return f"{path} differs: expected={expected!r}, actual={actual!r}."
    return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ResearchV2SemanticValidationError(
            f"Could not read valid JSON artifact {path}: {error}."
        ) from error
    if not isinstance(payload, dict):
        raise ResearchV2SemanticValidationError(
            f"JSON artifact must contain an object: {path}."
        )
    return payload


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file)
    except (OSError, yaml.YAMLError) as error:
        raise ResearchV2SemanticValidationError(
            f"Could not read valid YAML artifact {path}: {error}."
        ) from error
    if not isinstance(payload, dict):
        raise ResearchV2SemanticValidationError(
            f"YAML artifact must contain a mapping: {path}."
        )
    return payload
