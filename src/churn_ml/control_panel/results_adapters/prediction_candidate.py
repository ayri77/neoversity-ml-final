"""prediction_candidate_v1 → Prediction candidate adapter."""

from __future__ import annotations

from pathlib import Path

from src.churn_ml.control_panel.artifacts import ArtifactRecord
from src.churn_ml.control_panel.results_adapters._common import (
    bool_field,
    load_json,
    mapping_get,
    na,
    payload,
    schema_field,
    state_field,
    unrecorded,
)
from src.churn_ml.control_panel.results_entities import (
    ResultEntity,
    ResultEntityKind,
    ResultLink,
)
from src.churn_ml.control_panel.results_fields import (
    ResultField,
    available,
    from_optional,
    missing_unexpectedly,
)


class PredictionCandidateV1Adapter:
    entity_kind = ResultEntityKind.PREDICTION_CANDIDATE
    reader_ids = ("prediction_candidate_v1",)
    supported_schema_versions = frozenset({"prediction_candidate_v1"})

    def adapt(
        self,
        artifact: ArtifactRecord,
        *,
        repository_root: Path,
        archived: bool = False,
    ) -> ResultEntity:
        del repository_root
        manifest = payload(artifact, "candidate_manifest.json") or load_json(
            artifact.root, "candidate_manifest.json"
        )
        schema = schema_field(
            None if manifest is None else manifest.get("schema_version"),
            expected=self.supported_schema_versions,
            source="candidate_manifest.json.schema_version",
        )
        candidate_id = (
            str(manifest.get("candidate_id"))
            if isinstance(manifest, dict) and manifest.get("candidate_id")
            else Path(artifact.relative_path).name
        )
        oof_ref = (
            manifest.get("oof_prediction_reference")
            if isinstance(manifest, dict)
            else None
        )
        oof_rows = (
            oof_ref.get("row_count")
            if isinstance(oof_ref, dict)
            else None
        )
        fields: dict[str, ResultField] = {
            "candidate_id": available(candidate_id, source="candidate_manifest.json"),
            "source_kind": mapping_get(
                manifest, "source_kind", source="candidate_manifest.json"
            ),
            "source_model": mapping_get(
                manifest, "source_model_name", source="candidate_manifest.json"
            ),
            "source_metric_name": mapping_get(
                manifest,
                "source_metric_name",
                source="candidate_manifest.json",
                missing=unrecorded("Source metric name not recorded."),
            ),
            "source_metric_value": mapping_get(
                manifest, "source_metric_value", source="candidate_manifest.json"
            ),
            "oof_row_count": from_optional(
                oof_rows,
                source="candidate_manifest.json.oof_prediction_reference.row_count",
                missing=unrecorded("OOF row count not recorded."),
            ),
            "test_row_count": mapping_get(
                manifest,
                "test_row_count",
                source="candidate_manifest.json",
                missing=unrecorded("Test row count not recorded."),
            ),
            "train_row_count": mapping_get(
                manifest,
                "train_row_count",
                source="candidate_manifest.json",
                missing=unrecorded("Train row count not recorded."),
            ),
            "positive_class_label": mapping_get(
                manifest,
                "positive_class_label",
                source="candidate_manifest.json",
                missing=unrecorded("Positive class label not recorded."),
            ),
            "probability_semantics": mapping_get(
                manifest,
                "probability_semantics",
                source="candidate_manifest.json",
                missing=unrecorded("Probability semantics not recorded."),
            ),
            "oof_protocol": mapping_get(
                manifest,
                "oof_protocol",
                source="candidate_manifest.json",
                missing=unrecorded("OOF protocol not recorded."),
            ),
            "balanced_accuracy": na(
                "Research repeated-CV Balanced Accuracy does not apply to candidates."
            ),
            "sensitivity": na("Research sensitivity does not apply to candidates."),
            "specificity": na("Research specificity does not apply to candidates."),
            "roc_auc": na("Research ROC AUC does not apply to candidates."),
            "brier_score": na("Research Brier score does not apply to candidates."),
            "threshold": unrecorded(
                "Candidates do not record a deployment threshold; blends/submissions do."
            ),
            "model_family": na(
                "Candidates are prediction packages, not training-run model families."
            ),
        }
        lineage: list[ResultLink] = []
        if isinstance(manifest, dict):
            source_run = manifest.get("source_run_path")
            if isinstance(source_run, str) and source_run:
                lineage.append(
                    ResultLink(
                        relation="source_run",
                        target_kind=ResultEntityKind.MODEL_RUN,
                        target_id=Path(source_run).name,
                        target_path=source_run.replace("\\", "/"),
                        resolved=True,
                    )
                )
            blend_id = manifest.get("source_blend_id") or (
                (manifest.get("provenance") or {}).get("blend_id")
                if isinstance(manifest.get("provenance"), dict)
                else None
            )
            if isinstance(blend_id, str) and blend_id:
                lineage.append(
                    ResultLink(
                        relation="parent_blend",
                        target_kind=ResultEntityKind.PROBABILITY_BLEND,
                        target_id=blend_id,
                        resolved=False,
                        message="Blend ID recorded; path resolution deferred.",
                    )
                )
        dataset = mapping_get(manifest, "dataset_id", source="candidate_manifest.json")
        label = " · ".join(
            part
            for part in (
                candidate_id,
                str(dataset.value) if dataset.is_available() else None,
                str(fields["source_model"].value)
                if fields["source_model"].is_available()
                else None,
            )
            if part
        )
        return ResultEntity(
            kind=self.entity_kind,
            entity_id=candidate_id,
            display_label=label,
            reader_id=artifact.reader_id,
            source_path=artifact.relative_path,
            schema_version=schema,
            state=state_field(artifact),
            created_at=mapping_get(
                manifest,
                "created_at_utc",
                source="candidate_manifest.json",
                missing=unrecorded("Created timestamp not recorded."),
            ),
            dataset_id=dataset,
            exploratory=bool_field(
                manifest,
                "exploratory",
                source="candidate_manifest.json",
                missing=missing_unexpectedly("exploratory flag missing."),
            ),
            fields=fields,
            lineage=tuple(lineage),
            diagnostics=tuple([artifact.diagnostic] if artifact.diagnostic else []),
            archived=archived,
        )
