"""AutoGluon standalone run → Model run adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.churn_ml.control_panel.artifacts import ArtifactRecord
from src.churn_ml.control_panel.results_adapters._common import (
    load_json,
    mapping_get,
    na,
    payload,
    schema_field,
    state_field,
    unrecorded,
)
from src.churn_ml.control_panel.results_entities import ResultEntity, ResultEntityKind
from src.churn_ml.control_panel.results_fields import (
    ResultField,
    available,
    from_optional,
    invalid,
    missing_unexpectedly,
)


class AutoGluonRunAdapter:
    entity_kind = ResultEntityKind.MODEL_RUN
    reader_ids = ("autogluon_v1",)
    supported_schema_versions = frozenset({1})

    def adapt(
        self,
        artifact: ArtifactRecord,
        *,
        repository_root: Path,
        archived: bool = False,
    ) -> ResultEntity:
        del repository_root
        meta = payload(artifact, "run_metadata.json") or load_json(
            artifact.root, "run_metadata.json"
        )
        status = payload(artifact, "execution_status.json") or load_json(
            artifact.root, "execution_status.json"
        )
        # Lightweight inspection only — never open predictor directories.
        inspection = payload(artifact, "inspection/summary.json") or load_json(
            artifact.root, "inspection/summary.json"
        )
        profile_resolution = load_json(artifact.root, "profile_resolution.json")

        schema = schema_field(
            None if meta is None else meta.get("schema_version"),
            expected=self.supported_schema_versions,
            source="run_metadata.json.schema_version",
        )
        run_id = _text(meta, "run_id") or Path(artifact.relative_path).name
        dataset = mapping_get(
            meta,
            "dataset_version",
            source="run_metadata.json",
            missing=missing_unexpectedly("dataset_version missing."),
        )
        profile = mapping_get(
            meta,
            "profile_id",
            source="run_metadata.json",
            missing=missing_unexpectedly("profile_id missing."),
        )
        best_model, metric_name, metric_value = _leaderboard_fields(
            inspection, artifact.state
        )
        created = mapping_get(
            meta,
            "started_at_utc",
            source="run_metadata.json",
            missing=unrecorded("Started timestamp not recorded."),
        )
        diagnostics: list[str] = []
        if artifact.diagnostic:
            diagnostics.append(artifact.diagnostic)
        if isinstance(status, dict) and status.get("failure_reason"):
            diagnostics.append(str(status["failure_reason"]))

        fields: dict[str, ResultField] = {
            "run_type": available("autogluon", source="adapter"),
            "framework": available("AutoGluon", source="adapter"),
            "run_id": available(run_id, source="run_metadata.json.run_id"),
            "model_family": available("AutoGluon", source="adapter"),
            "mode": na("Research evaluation mode does not apply to AutoGluon runs."),
            "configuration": profile,
            "profile": profile,
            "feature_pipeline": na("Feature pipeline IDs are Research-specific."),
            "feature_count": unrecorded(
                "Feature count is not a top-level AutoGluon run_metadata field."
            ),
            "target_dependency": na("Target dependency is Research-dataset-specific."),
            "parent_dataset_id": na("Parent dataset lineage is Research-specific."),
            "primary_metric_name": metric_name,
            "primary_metric_value": metric_value,
            "balanced_accuracy": (
                metric_value
                if metric_name.is_available()
                and str(metric_name.value) == "balanced_accuracy"
                else na(
                    "Leaderboard metric is not balanced_accuracy; "
                    "Research repeated-CV BA does not apply."
                )
            ),
            "sensitivity": na("Repeated-CV sensitivity is Research-specific."),
            "specificity": na("Repeated-CV specificity is Research-specific."),
            "roc_auc": na("Repeated-CV ROC AUC is Research-specific."),
            "average_precision": na(
                "Repeated-CV average precision is Research-specific."
            ),
            "brier_score": na("Repeated-CV Brier score is Research-specific."),
            "threshold": mapping_get(
                inspection if isinstance(inspection, dict) else None,
                "decision_threshold",
                source="inspection/summary.json",
                missing=unrecorded("Decision threshold not recorded.")
                if artifact.state == "completed"
                else unrecorded("Inspection summary unavailable."),
            ),
            "cv_mean": na("Repeated-CV means are Research-specific."),
            "cv_std": na("Repeated-CV std is Research-specific."),
            "cv_min": na("Repeated-CV min is Research-specific."),
            "cv_max": na("Repeated-CV max is Research-specific."),
            "fold_count": _bag_folds(inspection),
            "repeat_count": na("Repeat count is Research-specific."),
            "seed": mapping_get(
                meta,
                "requested_seed",
                source="run_metadata.json",
                missing=unrecorded("Seed not recorded."),
            ),
            "duration_seconds": mapping_get(
                meta,
                "duration_seconds",
                source="run_metadata.json",
                missing=unrecorded("Duration not recorded."),
            ),
            "best_model": best_model,
            "autogluon_version": mapping_get(
                profile_resolution,
                "autogluon_version",
                source="profile_resolution.json",
                missing=unrecorded("AutoGluon version not recorded."),
            ),
            "mlflow_status": unrecorded("Resolved by Results MLflow projection."),
        }

        label = " · ".join(
            part
            for part in (
                str(dataset.value) if dataset.is_available() else None,
                "AutoGluon",
                str(profile.value) if profile.is_available() else None,
                run_id,
            )
            if part
        )
        return ResultEntity(
            kind=self.entity_kind,
            entity_id=run_id,
            display_label=label,
            reader_id=artifact.reader_id,
            source_path=artifact.relative_path,
            schema_version=schema,
            state=state_field(artifact),
            created_at=created,
            dataset_id=dataset,
            exploratory=available(False, source="default_non_exploratory"),
            fields=fields,
            lineage=(),
            diagnostics=tuple(diagnostics),
            archived=archived,
        )


def _text(payload: Mapping[str, Any] | None, key: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    return str(value) if isinstance(value, (str, int)) and value != "" else None


def _leaderboard_fields(
    inspection: Mapping[str, Any] | None, state: str
) -> tuple[ResultField, ResultField, ResultField]:
    if not isinstance(inspection, dict):
        if state == "completed":
            return (
                missing_unexpectedly("inspection/summary.json missing."),
                missing_unexpectedly("eval metric missing."),
                missing_unexpectedly("score_val missing."),
            )
        return (
            unrecorded("Inspection summary unavailable for failed/incomplete runs."),
            unrecorded("Eval metric unavailable."),
            unrecorded("Validation score unavailable."),
        )
    best = inspection.get("best_model")
    best_field = from_optional(
        best,
        source="inspection/summary.json.best_model",
        missing=missing_unexpectedly("best_model missing."),
    )
    info = inspection.get("predictor_info")
    metric_name = None
    metric_value = None
    if isinstance(info, dict):
        metric_name = info.get("eval_metric")
        metric_value = info.get("best_model_score_val")
    leaderboard = inspection.get("leaderboard")
    if metric_value is None and isinstance(leaderboard, list) and best:
        for row in leaderboard:
            if isinstance(row, dict) and row.get("model") == best:
                metric_name = row.get("eval_metric") or metric_name
                metric_value = row.get("score_val")
                break
    name_field = from_optional(
        metric_name,
        source="inspection/summary.json",
        missing=missing_unexpectedly("eval_metric missing.")
        if state == "completed"
        else unrecorded("eval_metric unavailable."),
    )
    if metric_value is None:
        value_field = (
            missing_unexpectedly("score_val missing.")
            if state == "completed"
            else unrecorded("score_val unavailable.")
        )
    else:
        try:
            value_field = available(float(metric_value), source="inspection/summary.json")
        except (TypeError, ValueError):
            value_field = invalid(
                "score_val is not numeric.",
                value=metric_value,
                source="inspection/summary.json",
            )
    return best_field, name_field, value_field


def _bag_folds(inspection: Mapping[str, Any] | None) -> ResultField:
    if not isinstance(inspection, dict):
        return unrecorded("Bag folds unavailable.")
    info = inspection.get("predictor_info")
    if not isinstance(info, dict) or "num_bag_folds" not in info:
        return unrecorded("Bag folds not recorded.")
    value = info.get("num_bag_folds")
    if isinstance(value, int) and not isinstance(value, bool):
        return available(value, source="predictor_info.num_bag_folds")
    return invalid("num_bag_folds is malformed.", value=value)
