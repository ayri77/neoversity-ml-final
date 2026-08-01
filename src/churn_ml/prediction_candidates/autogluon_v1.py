"""AutoGluon adapter that materializes ``prediction_candidate_v1`` packages.

AutoGluon is imported lazily so the main project environment and ordinary tests
never require it. Only genuine bagged OOF probabilities from the public
``predict_proba_oof`` API are accepted; in-sample ``predict_proba`` on training
rows is never used as a substitute.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from src.churn_ml.autogluon_config import config_identity_sha256, load_config
from src.churn_ml.autogluon_inspection import inspect_run
from src.churn_ml.dataset_registry.api import resolve_dataset_package
from src.churn_ml.prediction_candidates.contract_v1 import (
    OOF_COLUMNS,
    POSITIVE_CLASS_LABEL,
    PROBABILITY_SEMANTICS,
    SCHEMA_VERSION,
    TEST_COLUMNS,
    UNAVAILABLE,
    CandidatePackage,
    CandidateValidationError,
    PredictionCandidateError,
    build_candidate_id,
    create_candidate_package,
    repository_relative_path,
    resolve_under_repository,
    utc_now,
    validate_oof_frame,
    validate_probability_series,
    validate_test_frame,
)


SOURCE_KIND = "autogluon_standalone_v1"
OOF_PROTOCOL = "autogluon_bagged_predict_proba_oof"
# Future Research v2 adapters will declare a repeat-normalization policy such as
# ``research_v2_mean_probability_across_repeats_per_row_position`` instead.


@dataclass(frozen=True)
class AutoGluonRunInventory:
    run_dir: Path
    run_path: str
    dataset_id: str
    predictor_path: str
    config_path: str
    config_sha256: str
    autogluon_version: Any
    best_model: str | None
    model_names: tuple[str, ...]
    leaderboard: tuple[dict[str, Any], ...]
    classification: str
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class ExtractedModelPredictions:
    model_name: str
    model_type: Any
    oof: pd.DataFrame
    test: pd.DataFrame
    positive_class_label: int
    class_labels: tuple[Any, ...]
    source_metric_name: Any
    source_metric_value: Any
    source_leaderboard_metadata: dict[str, Any]
    limitations: tuple[str, ...]


def _lazy_load_tabular_predictor() -> Any:
    try:
        from autogluon.tabular import TabularPredictor  # type: ignore[import-not-found]
    except ImportError as error:
        raise PredictionCandidateError(
            "AutoGluon is not available in this environment. Use .venv-autogluon "
            "for import operations.",
            reason_code="autogluon_unavailable",
        ) from error
    return TabularPredictor


def load_predictor(
    predictor_dir: Path,
    *,
    loader: Callable[[str], Any] | None = None,
) -> Any:
    if loader is not None:
        return loader(str(predictor_dir))
    tabular_predictor = _lazy_load_tabular_predictor()
    if not hasattr(tabular_predictor, "load"):
        raise PredictionCandidateError(
            "Installed AutoGluon TabularPredictor.load is unavailable.",
            reason_code="autogluon_api_unsupported",
        )
    return tabular_predictor.load(str(predictor_dir))


def require_predict_proba_oof(predictor: Any) -> Callable[..., Any]:
    method = getattr(predictor, "predict_proba_oof", None)
    if not callable(method):
        raise PredictionCandidateError(
            "Genuine OOF export requires public predict_proba_oof(); it is "
            "unavailable on this predictor. In-sample predict_proba() on train "
            "rows is prohibited.",
            reason_code="oof_api_unsupported",
        )
    return method


def resolve_positive_probability_column(
    frame_or_series: pd.DataFrame | pd.Series,
    *,
    class_labels: Sequence[Any] | None,
    positive_class_label: int = POSITIVE_CLASS_LABEL,
) -> pd.Series:
    """Select P(y=1) from AutoGluon probability output."""
    if isinstance(frame_or_series, pd.Series):
        # Single-column Series is ambiguous unless class_labels length is 1.
        raise CandidateValidationError(
            "Probability Series without explicit class columns is ambiguous for "
            "positive-class selection.",
            reason_code="positive_class_ambiguous",
        )

    frame = frame_or_series
    if frame.shape[1] < 1:
        raise CandidateValidationError(
            "Probability frame has no class columns.",
            reason_code="positive_class_missing",
        )

    columns = list(frame.columns)
    # Prefer exact label match, including numpy scalar labels.
    matched: list[Any] = []
    for column in columns:
        if column == positive_class_label:
            matched.append(column)
            continue
        try:
            if int(column) == positive_class_label and float(column) == float(
                positive_class_label
            ):
                matched.append(column)
        except (TypeError, ValueError):
            continue
    if len(matched) == 1:
        series = frame[matched[0]]
        validate_probability_series(series, field_name="probability_positive")
        return series.astype("float64")
    if len(matched) > 1:
        raise CandidateValidationError(
            f"Ambiguous positive-class columns for label {positive_class_label}: "
            f"{matched}",
            reason_code="positive_class_ambiguous",
        )

    if class_labels is not None:
        labels = list(class_labels)
        if positive_class_label not in labels and not any(
            _label_equals(label, positive_class_label) for label in labels
        ):
            raise CandidateValidationError(
                f"Positive class label {positive_class_label} is absent from "
                f"predictor class_labels {labels}.",
                reason_code="positive_class_missing",
            )
    raise CandidateValidationError(
        f"Positive class label {positive_class_label} is absent from probability "
        f"columns {columns}.",
        reason_code="positive_class_missing",
    )


def _label_equals(label: Any, expected: int) -> bool:
    try:
        return int(label) == expected and float(label) == float(expected)
    except (TypeError, ValueError):
        return False


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PredictionCandidateError(
            f"Expected JSON object at {path}",
            reason_code="json_invalid",
        )
    return payload


def inventory_autogluon_run(
    run_dir: Path,
    *,
    repository_root: Path,
    predictor_loader: Callable[[str], Any] | None = None,
    attempt_load: bool = False,
) -> AutoGluonRunInventory:
    root = repository_root.resolve()
    directory = run_dir.resolve(strict=True)
    run_path = repository_relative_path(directory, root)
    report = inspect_run(
        directory,
        attempt_load=False,
        predictor_loader=None,
    )
    metadata = _read_json(directory / "run_metadata.json")
    worker = _read_json(directory / "worker_result.json")
    resolved_config = directory / "resolved_config.yaml"
    config = load_config(resolved_config, root, require_data_files=False)
    config_sha = config.identity_sha256 or config_identity_sha256(config.portable)
    recorded = metadata.get("config_identity_sha256")
    if recorded is not None and recorded != config_sha:
        raise PredictionCandidateError(
            "resolved_config.yaml identity does not match run_metadata "
            "config_identity_sha256.",
            reason_code="config_identity_mismatch",
        )

    dataset_id = str(
        worker.get("dataset_version")
        or metadata.get("dataset_version")
        or config.dataset.version
    )
    best_model = worker.get("best_model")
    model_names = tuple(str(name) for name in worker.get("model_names") or [])
    if not model_names and report.get("predictor"):
        model_names = tuple(str(name) for name in report["predictor"].get("models") or [])

    leaderboard_rows: list[dict[str, Any]] = []
    leaderboard_csv = directory / "inspection" / "leaderboard.csv"
    if leaderboard_csv.is_file():
        frame = pd.read_csv(leaderboard_csv)
        leaderboard_rows = frame.to_dict(orient="records")

    limitations: list[str] = []
    if report.get("classification") != "complete":
        limitations.append(
            f"run_classification={report.get('classification')}: "
            f"{report.get('reason_codes')}"
        )

    if attempt_load:
        predictor = load_predictor(
            directory / "predictor",
            loader=predictor_loader,
        )
        if not callable(getattr(predictor, "predict_proba_oof", None)):
            limitations.append(
                "predict_proba_oof unavailable; genuine OOF import unsupported"
            )
        if not callable(getattr(predictor, "predict_proba", None)):
            limitations.append("predict_proba unavailable; test import unsupported")

    return AutoGluonRunInventory(
        run_dir=directory,
        run_path=run_path,
        dataset_id=dataset_id,
        predictor_path=f"{run_path}/predictor",
        config_path=f"{run_path}/resolved_config.yaml",
        config_sha256=str(config_sha),
        autogluon_version=worker.get("autogluon_version", UNAVAILABLE),
        best_model=str(best_model) if best_model is not None else None,
        model_names=model_names,
        leaderboard=tuple(leaderboard_rows),
        classification=str(report.get("classification")),
        limitations=tuple(limitations),
    )


def resolve_selected_models(
    inventory: AutoGluonRunInventory,
    *,
    best: bool,
    models: Sequence[str],
) -> tuple[str, ...]:
    selected: list[str] = []
    if best:
        if not inventory.best_model:
            raise PredictionCandidateError(
                "Run does not record a best/final model.",
                reason_code="best_model_missing",
            )
        selected.append(inventory.best_model)
    for name in models:
        if name not in inventory.model_names:
            raise PredictionCandidateError(
                f"Unknown model {name!r}. Available: {list(inventory.model_names)}",
                reason_code="unknown_model",
            )
        selected.append(name)
    # Preserve order, drop duplicates.
    ordered: list[str] = []
    seen: set[str] = set()
    for name in selected:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    if not ordered:
        raise PredictionCandidateError(
            "Select --best and/or one or more --model values.",
            reason_code="no_model_selected",
        )
    return tuple(ordered)


def _leaderboard_entry(
    inventory: AutoGluonRunInventory,
    model_name: str,
) -> dict[str, Any]:
    for row in inventory.leaderboard:
        if str(row.get("model")) == model_name:
            return dict(row)
    return {}


def _model_type(predictor: Any, model_name: str) -> Any:
    info_fn = getattr(predictor, "model_info", None)
    if callable(info_fn):
        try:
            info = info_fn(model_name)
            if isinstance(info, dict):
                return info.get("model_type") or info.get("type") or UNAVAILABLE
        except Exception:
            pass
    info_all = getattr(predictor, "info", None)
    if callable(info_all):
        try:
            payload = info_all()
            model_info = payload.get("model_info") if isinstance(payload, dict) else None
            if isinstance(model_info, dict) and model_name in model_info:
                entry = model_info[model_name]
                if isinstance(entry, dict):
                    return entry.get("model_type") or UNAVAILABLE
        except Exception:
            pass
    return UNAVAILABLE


def extract_model_predictions(
    inventory: AutoGluonRunInventory,
    model_name: str,
    *,
    repository_root: Path,
    predictor: Any | None = None,
    predictor_loader: Callable[[str], Any] | None = None,
) -> ExtractedModelPredictions:
    root = repository_root.resolve()
    if model_name not in inventory.model_names:
        raise PredictionCandidateError(
            f"Unknown model {model_name!r}.",
            reason_code="unknown_model",
        )
    loaded = predictor or load_predictor(
        inventory.run_dir / "predictor",
        loader=predictor_loader,
    )
    predict_proba_oof = require_predict_proba_oof(loaded)
    predict_proba = getattr(loaded, "predict_proba", None)
    if not callable(predict_proba):
        raise PredictionCandidateError(
            "Public predict_proba() is unavailable; test probabilities cannot "
            "be exported.",
            reason_code="test_api_unsupported",
        )

    class_labels = tuple(getattr(loaded, "class_labels", None) or ())
    positive = getattr(loaded, "positive_class", POSITIVE_CLASS_LABEL)
    if not _label_equals(positive, POSITIVE_CLASS_LABEL):
        raise CandidateValidationError(
            f"Predictor positive_class is {positive!r}, expected 1.",
            reason_code="positive_class_invalid",
        )

    try:
        oof_raw = predict_proba_oof(model=model_name)
    except TypeError:
        # Some stubs/APIs accept model as positional.
        oof_raw = predict_proba_oof(model_name)
    except Exception as error:
        raise PredictionCandidateError(
            f"Genuine OOF export failed for model {model_name!r}: "
            f"{type(error).__name__}: {error}. Individual-model OOF may be "
            "unsupported for non-bagged/refit models; import only models with "
            "public predict_proba_oof support (typically bagged models or the "
            "best ensemble).",
            reason_code="oof_export_failed",
        ) from error

    if isinstance(oof_raw, pd.Series):
        raise CandidateValidationError(
            "OOF probabilities returned as a Series without class columns.",
            reason_code="positive_class_ambiguous",
        )
    if not isinstance(oof_raw, pd.DataFrame):
        raise PredictionCandidateError(
            f"Unexpected OOF type: {type(oof_raw)!r}",
            reason_code="oof_type_invalid",
        )

    oof_proba = resolve_positive_probability_column(
        oof_raw,
        class_labels=class_labels,
        positive_class_label=POSITIVE_CLASS_LABEL,
    )
    if len(oof_proba) != len(oof_raw):
        raise CandidateValidationError(
            "Positive-class OOF column length mismatch.",
            reason_code="oof_length_mismatch",
        )

    dataset = resolve_dataset_package(root / "data" / "processed", inventory.dataset_id)
    train_count = dataset.manifest.train_row_count
    if len(oof_proba) != train_count:
        raise CandidateValidationError(
            f"OOF length {len(oof_proba)} != Dataset Package train_row_count "
            f"{train_count}.",
            reason_code="oof_row_count_mismatch",
        )
    # Preserve canonical train order only after proving positional alignment.
    if not oof_raw.index.equals(pd.RangeIndex(train_count)):
        if list(oof_raw.index) != list(range(train_count)):
            raise CandidateValidationError(
                "AutoGluon OOF index is not aligned to contiguous canonical "
                "train row positions.",
                reason_code="oof_index_misaligned",
            )

    oof = pd.DataFrame(
        {
            "row_position": np.arange(train_count, dtype=np.int64),
            "target": dataset.artifacts.y_train.astype("int64").to_numpy(),
            "probability_positive": oof_proba.to_numpy(dtype=np.float64),
        }
    )
    validate_oof_frame(
        oof,
        train_row_count=train_count,
        expected_target=dataset.artifacts.y_train,
    )

    x_test = dataset.artifacts.X_test.copy()
    if "index" in x_test.columns or "id" in x_test.columns:
        raise CandidateValidationError(
            "Dataset Package X_test unexpectedly contains a submission ID "
            "column; IDs must remain outside the model feature matrix.",
            reason_code="submission_id_in_features",
        )
    feature_columns = list(x_test.columns)
    try:
        test_raw = predict_proba(x_test, model=model_name)
    except TypeError:
        test_raw = predict_proba(x_test, model_name)
    except Exception as error:
        raise PredictionCandidateError(
            f"Test probability export failed for model {model_name!r}: "
            f"{type(error).__name__}: {error}",
            reason_code="test_export_failed",
        ) from error

    if not isinstance(test_raw, pd.DataFrame):
        raise PredictionCandidateError(
            f"Unexpected test probability type: {type(test_raw)!r}",
            reason_code="test_type_invalid",
        )
    if list(x_test.columns) != feature_columns:
        raise CandidateValidationError(
            "Test feature column order changed during prediction.",
            reason_code="test_feature_order_changed",
        )
    test_proba = resolve_positive_probability_column(
        test_raw,
        class_labels=class_labels,
        positive_class_label=POSITIVE_CLASS_LABEL,
    )
    if len(test_proba) != dataset.manifest.test_row_count:
        raise CandidateValidationError(
            f"Test probability length {len(test_proba)} != test_row_count "
            f"{dataset.manifest.test_row_count}.",
            reason_code="test_row_count_mismatch",
        )
    test = pd.DataFrame(
        {
            "row_position": np.arange(dataset.manifest.test_row_count, dtype=np.int64),
            "probability_positive": test_proba.to_numpy(dtype=np.float64),
        }
    )
    validate_test_frame(test, test_row_count=dataset.manifest.test_row_count)

    entry = _leaderboard_entry(inventory, model_name)
    metric_name = entry.get("eval_metric", UNAVAILABLE)
    metric_value = entry.get("score_val", UNAVAILABLE)
    limitations: list[str] = []
    if model_name != inventory.best_model:
        limitations.append(
            "Individual model imported via public predict_proba_oof/predict_proba; "
            "unsupported models will fail explicitly rather than falling back to "
            "in-sample predictions."
        )

    return ExtractedModelPredictions(
        model_name=model_name,
        model_type=_model_type(loaded, model_name),
        oof=oof.loc[:, list(OOF_COLUMNS)],
        test=test.loc[:, list(TEST_COLUMNS)],
        positive_class_label=POSITIVE_CLASS_LABEL,
        class_labels=class_labels,
        source_metric_name=metric_name,
        source_metric_value=metric_value,
        source_leaderboard_metadata=entry or {"status": UNAVAILABLE},
        limitations=tuple(limitations),
    )


def import_autogluon_model_candidate(
    inventory: AutoGluonRunInventory,
    model_name: str,
    *,
    repository_root: Path,
    predictor: Any | None = None,
    predictor_loader: Callable[[str], Any] | None = None,
) -> CandidatePackage:
    root = repository_root.resolve()
    extracted = extract_model_predictions(
        inventory,
        model_name,
        repository_root=root,
        predictor=predictor,
        predictor_loader=predictor_loader,
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "source_kind": SOURCE_KIND,
        "dataset_id": inventory.dataset_id,
        "source_run_path": inventory.run_path,
        "source_config_sha256": inventory.config_sha256,
        "source_model_name": extracted.model_name,
        "oof_protocol": OOF_PROTOCOL,
        "positive_class_label": POSITIVE_CLASS_LABEL,
        "probability_semantics": PROBABILITY_SEMANTICS,
    }
    candidate_id = build_candidate_id(identity)
    dataset = resolve_dataset_package(root / "data" / "processed", inventory.dataset_id)
    exploratory = dataset.manifest.target_dependency == "exploratory"
    provenance = {
        "adapter": "prediction_candidates.autogluon_v1",
        "oof_api": "TabularPredictor.predict_proba_oof",
        "test_api": "TabularPredictor.predict_proba",
        "in_sample_train_predict_proba_prohibited": True,
        "dataset_package_id": inventory.dataset_id,
        "run_classification": inventory.classification,
        "limitations": list(extracted.limitations) + list(inventory.limitations),
        "research_v2_repeat_normalization_note": (
            "Future Research v2 adapters must normalize repeated OOF rows to one "
            "probability per row_position under a declared deterministic policy "
            "(for example mean across repeats) and record that policy in "
            "oof_protocol. Shared fold assignments with AutoGluon are not required."
        ),
    }
    manifest_fields = {
        "candidate_id": candidate_id,
        "created_at_utc": utc_now(),
        "dataset_id": inventory.dataset_id,
        "exploratory": exploratory,
        "source_run_path": inventory.run_path,
        "source_config_path": inventory.config_path,
        "source_config_sha256": inventory.config_sha256,
        "source_model_name": extracted.model_name,
        "source_model_type": extracted.model_type,
        "source_predictor_path": inventory.predictor_path,
        "source_autogluon_version": inventory.autogluon_version,
        "source_metric_name": extracted.source_metric_name,
        "source_metric_value": extracted.source_metric_value,
        "source_leaderboard_metadata": extracted.source_leaderboard_metadata,
        "provenance": provenance,
    }
    source_metadata = {
        "schema_version": 1,
        "source_kind": SOURCE_KIND,
        "run_path": inventory.run_path,
        "model_name": extracted.model_name,
        "model_type": extracted.model_type,
        "class_labels": [_json_safe_label(label) for label in extracted.class_labels],
        "positive_class_label": extracted.positive_class_label,
        "oof_protocol": OOF_PROTOCOL,
        "best_model": inventory.best_model,
        "selected_as_best": model_name == inventory.best_model,
        "autogluon_version": inventory.autogluon_version,
        "leaderboard_entry": extracted.source_leaderboard_metadata,
        "limitations": list(extracted.limitations),
    }
    return create_candidate_package(
        repository_root=root,
        identity=identity,
        oof=extracted.oof,
        test=extracted.test,
        manifest_fields=manifest_fields,
        source_metadata=source_metadata,
    )


def validate_autogluon_model_import(
    inventory: AutoGluonRunInventory,
    model_name: str,
    *,
    repository_root: Path,
    predictor: Any | None = None,
    predictor_loader: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Validate extractability without writing artifacts."""
    extracted = extract_model_predictions(
        inventory,
        model_name,
        repository_root=repository_root,
        predictor=predictor,
        predictor_loader=predictor_loader,
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "source_kind": SOURCE_KIND,
        "dataset_id": inventory.dataset_id,
        "source_run_path": inventory.run_path,
        "source_config_sha256": inventory.config_sha256,
        "source_model_name": extracted.model_name,
        "oof_protocol": OOF_PROTOCOL,
        "positive_class_label": POSITIVE_CLASS_LABEL,
        "probability_semantics": PROBABILITY_SEMANTICS,
    }
    return {
        "ok": True,
        "candidate_id": build_candidate_id(identity),
        "model_name": extracted.model_name,
        "dataset_id": inventory.dataset_id,
        "train_row_count": int(len(extracted.oof)),
        "test_row_count": int(len(extracted.test)),
        "positive_class_label": extracted.positive_class_label,
        "source_metric_name": extracted.source_metric_name,
        "source_metric_value": extracted.source_metric_value,
        "oof_protocol": OOF_PROTOCOL,
        "artifacts_written": False,
        "limitations": list(extracted.limitations),
    }


def resolve_run_dir(run_dir: Path, repository_root: Path) -> Path:
    root = repository_root.resolve()
    if run_dir.is_absolute():
        resolved = run_dir.resolve(strict=True)
        repository_relative_path(resolved, root)
        return resolved
    return resolve_under_repository(run_dir.as_posix().replace("\\", "/"), root)


def _json_safe_label(label: Any) -> Any:
    if isinstance(label, (bool, np.bool_)):
        return bool(label)
    if isinstance(label, (int, np.integer)):
        return int(label)
    if isinstance(label, (float, np.floating)):
        number = float(label)
        if number.is_integer():
            return int(number)
        return number
    return label
