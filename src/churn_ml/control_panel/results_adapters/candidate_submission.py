"""candidate_submission_v1 → Candidate submission adapter."""

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
    missing_unexpectedly,
)


class CandidateSubmissionV1Adapter:
    entity_kind = ResultEntityKind.CANDIDATE_SUBMISSION
    reader_ids = ("candidate_submission_v1",)
    supported_schema_versions = frozenset({"candidate_submission_v1"})

    def adapt(
        self,
        artifact: ArtifactRecord,
        *,
        repository_root: Path,
        archived: bool = False,
    ) -> ResultEntity:
        del repository_root
        manifest = payload(artifact, "submission_manifest.json") or load_json(
            artifact.root, "submission_manifest.json"
        )
        source_meta = payload(artifact, "source_metadata.json") or load_json(
            artifact.root, "source_metadata.json"
        )
        schema = schema_field(
            None if manifest is None else manifest.get("schema_version"),
            expected=self.supported_schema_versions,
            source="submission_manifest.json.schema_version",
        )
        submission_id = (
            str(manifest.get("submission_id"))
            if isinstance(manifest, dict) and manifest.get("submission_id")
            else Path(artifact.relative_path).name
        )
        candidate_id = mapping_get(
            manifest, "candidate_id", source="submission_manifest.json"
        )
        blend_id = mapping_get(
            manifest,
            "blend_id",
            source="submission_manifest.json",
            missing=unrecorded("Blend ID not recorded for this submission."),
        )
        fields: dict[str, ResultField] = {
            "submission_id": available(submission_id, source="submission_manifest.json"),
            "candidate_id": candidate_id,
            "blend_id": blend_id,
            "row_count": mapping_get(
                manifest, "row_count", source="submission_manifest.json"
            ),
            "threshold": mapping_get(
                manifest, "threshold", source="submission_manifest.json"
            ),
            "network_access": bool_field(
                manifest,
                "network_access",
                source="submission_manifest.json",
                missing=missing_unexpectedly("network_access missing."),
            ),
            "kaggle_upload": bool_field(
                manifest,
                "kaggle_upload",
                source="submission_manifest.json",
                missing=missing_unexpectedly("kaggle_upload missing."),
            ),
            "kaggle_public_score": unrecorded(
                "Kaggle public score is not recorded by candidate_submission_v1."
            ),
            "readiness": available(
                "generated_locally",
                source="adapter",
            ),
            "model_family": na(
                "Submissions are delivery packages, not training-run model families."
            ),
            "balanced_accuracy": na(
                "Research metrics do not apply to submission packages."
            ),
        }
        lineage: list[ResultLink] = []
        if candidate_id.is_available():
            lineage.append(
                ResultLink(
                    relation="source_candidate",
                    target_kind=ResultEntityKind.PREDICTION_CANDIDATE,
                    target_id=str(candidate_id.value),
                    resolved=False,
                )
            )
        if blend_id.is_available():
            lineage.append(
                ResultLink(
                    relation="source_blend",
                    target_kind=ResultEntityKind.PROBABILITY_BLEND,
                    target_id=str(blend_id.value),
                    resolved=False,
                )
            )
        label = " · ".join(
            part
            for part in (
                submission_id,
                str(candidate_id.value) if candidate_id.is_available() else None,
                f"rows={fields['row_count'].value}"
                if fields["row_count"].is_available()
                else None,
            )
            if part
        )
        return ResultEntity(
            kind=self.entity_kind,
            entity_id=submission_id,
            display_label=label,
            reader_id=artifact.reader_id,
            source_path=artifact.relative_path,
            schema_version=schema,
            state=state_field(artifact),
            created_at=mapping_get(
                manifest,
                "created_at_utc",
                source="submission_manifest.json",
                missing=unrecorded("Created timestamp not recorded."),
            ),
            dataset_id=unrecorded(
                "Submission packages do not embed dataset_id; inherit via candidate lineage."
            ),
            exploratory=bool_field(
                source_meta,
                "exploratory",
                source="source_metadata.json",
                missing=available(False, source="default_non_exploratory"),
            ),
            fields=fields,
            lineage=tuple(lineage),
            diagnostics=tuple([artifact.diagnostic] if artifact.diagnostic else []),
            archived=archived,
        )
