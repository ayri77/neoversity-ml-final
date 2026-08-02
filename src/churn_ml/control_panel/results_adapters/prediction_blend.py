"""prediction_blend_v1 → Probability blend adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.churn_ml.control_panel.artifacts import ArtifactRecord
from src.churn_ml.control_panel.results_adapters._common import (
    bool_field,
    load_json,
    mapping_get,
    na,
    nested_number,
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


class PredictionBlendV1Adapter:
    entity_kind = ResultEntityKind.PROBABILITY_BLEND
    reader_ids = ("prediction_blend_v1",)
    supported_schema_versions = frozenset({"prediction_blend_v1"})

    def adapt(
        self,
        artifact: ArtifactRecord,
        *,
        repository_root: Path,
        archived: bool = False,
    ) -> ResultEntity:
        del repository_root
        manifest = payload(artifact, "blend_manifest.json") or load_json(
            artifact.root, "blend_manifest.json"
        )
        evaluation = payload(artifact, "evaluation.json") or load_json(
            artifact.root, "evaluation.json"
        )
        schema = schema_field(
            None if manifest is None else manifest.get("schema_version"),
            expected=self.supported_schema_versions,
            source="blend_manifest.json.schema_version",
        )
        blend_id = (
            str(manifest.get("blend_id"))
            if isinstance(manifest, dict) and manifest.get("blend_id")
            else Path(artifact.relative_path).name
        )
        settings = (
            manifest.get("settings") if isinstance(manifest, dict) else None
        )
        if not isinstance(settings, dict):
            settings = {}
        honest = (
            manifest.get("honest_meta_cv_metrics")
            if isinstance(manifest, dict)
            else None
        )
        if not isinstance(honest, dict):
            honest = (
                evaluation.get("honest_meta_cv_metrics")
                if isinstance(evaluation, dict)
                else None
            )
        weights = (
            manifest.get("final_deployment_weights")
            if isinstance(manifest, dict)
            else None
        )
        nonzero = _nonzero_weights(weights)
        parent_ids = (
            list(manifest.get("candidate_ids") or settings.get("parent_candidate_ids") or [])
            if isinstance(manifest, dict)
            else []
        )
        readiness = (
            manifest.get("submission_readiness")
            if isinstance(manifest, dict)
            else None
        )
        readiness_state = (
            readiness.get("state")
            if isinstance(readiness, dict)
            else None
        )
        fields: dict[str, ResultField] = {
            "blend_id": available(blend_id, source="blend_manifest.json"),
            "optimizer": mapping_get(
                settings,
                "optimizer_backend",
                source="blend_manifest.json.settings",
                missing=missing_unexpectedly("optimizer_backend missing."),
            ),
            "strategy": mapping_get(
                settings,
                "strategy",
                source="blend_manifest.json.settings",
                missing=unrecorded("Strategy not recorded."),
            ),
            "parent_count": available(
                len(parent_ids), source="blend_manifest.json.candidate_ids"
            ),
            "parent_candidate_ids": available(
                tuple(str(item) for item in parent_ids),
                source="blend_manifest.json.candidate_ids",
            ),
            "nonzero_weights": available(nonzero, source="final_deployment_weights"),
            "honest_mean_ba": mapping_get(
                honest if isinstance(honest, dict) else None,
                "mean_repeat_balanced_accuracy",
                source="honest_meta_cv_metrics",
                missing=missing_unexpectedly("Honest mean BA missing."),
            ),
            "honest_ba_std": mapping_get(
                honest if isinstance(honest, dict) else None,
                "std_repeat_balanced_accuracy",
                source="honest_meta_cv_metrics",
                missing=unrecorded("Honest BA std not recorded."),
            ),
            "honest_ba_min": mapping_get(
                honest if isinstance(honest, dict) else None,
                "min_repeat_balanced_accuracy",
                source="honest_meta_cv_metrics",
                missing=unrecorded("Honest BA min not recorded."),
            ),
            "honest_ba_max": mapping_get(
                honest if isinstance(honest, dict) else None,
                "max_repeat_balanced_accuracy",
                source="honest_meta_cv_metrics",
                missing=unrecorded("Honest BA max not recorded."),
            ),
            "full_oof_descriptive_ba": nested_number(
                manifest if isinstance(manifest, dict) else None,
                "full_oof_descriptive_metrics",
                "balanced_accuracy",
                source="blend_manifest.json",
                missing=unrecorded("Full-OOF descriptive BA not recorded."),
            ),
            "cross_fitted_descriptive_ba": nested_number(
                manifest if isinstance(manifest, dict) else None,
                "cross_fitted_probability_descriptive_metrics",
                "balanced_accuracy",
                source="blend_manifest.json",
                missing=unrecorded("Cross-fitted descriptive BA not recorded."),
            ),
            "threshold": mapping_get(
                manifest,
                "final_deployment_threshold",
                source="blend_manifest.json",
                missing=missing_unexpectedly("Final threshold missing."),
            ),
            "folds": mapping_get(
                settings,
                "folds",
                source="blend_manifest.json.settings",
                missing=unrecorded("Folds not recorded."),
            ),
            "repeats": mapping_get(
                settings,
                "repeats",
                source="blend_manifest.json.settings",
                missing=unrecorded("Repeats not recorded."),
            ),
            "seed": mapping_get(
                settings,
                "seed",
                source="blend_manifest.json.settings",
                missing=unrecorded("Seed not recorded."),
            ),
            "submission_readiness": from_optional(
                readiness_state,
                source="submission_readiness.state",
                missing=unrecorded("Submission readiness not recorded."),
            ),
            "model_family": na(
                "Probability blends are ensembles of candidates, not a training-run model family."
            ),
            "balanced_accuracy": na(
                "Use honest_mean_ba for blend evaluation; Research CV BA does not apply."
            ),
        }
        lineage = [
            ResultLink(
                relation="parent_candidate",
                target_kind=ResultEntityKind.PREDICTION_CANDIDATE,
                target_id=str(item),
                resolved=False,
            )
            for item in parent_ids
        ]
        if isinstance(manifest, dict) and manifest.get("canonical_candidate_id"):
            lineage.append(
                ResultLink(
                    relation="canonical_candidate",
                    target_kind=ResultEntityKind.PREDICTION_CANDIDATE,
                    target_id=str(manifest["canonical_candidate_id"]),
                    target_path=(
                        str(manifest["canonical_candidate_path"])
                        if manifest.get("canonical_candidate_path")
                        else None
                    ),
                    resolved=bool(manifest.get("canonical_candidate_path")),
                )
            )
        dataset = mapping_get(
            manifest,
            "identity_reference_dataset_id",
            source="blend_manifest.json",
            missing=unrecorded("Dataset identity reference not recorded."),
        )
        honest_ba = fields["honest_mean_ba"]
        label = " · ".join(
            part
            for part in (
                blend_id,
                str(fields["optimizer"].value)
                if fields["optimizer"].is_available()
                else None,
                f"BA={honest_ba.value:.6f}"
                if honest_ba.is_available() and isinstance(honest_ba.value, float)
                else None,
            )
            if part
        )
        return ResultEntity(
            kind=self.entity_kind,
            entity_id=blend_id,
            display_label=label,
            reader_id=artifact.reader_id,
            source_path=artifact.relative_path,
            schema_version=schema,
            state=state_field(artifact),
            created_at=mapping_get(
                manifest,
                "created_at_utc",
                source="blend_manifest.json",
                missing=unrecorded("Created timestamp not recorded."),
            ),
            dataset_id=dataset,
            exploratory=bool_field(
                manifest,
                "exploratory",
                source="blend_manifest.json",
                missing=missing_unexpectedly("exploratory flag missing."),
            ),
            fields=fields,
            lineage=tuple(lineage),
            diagnostics=tuple([artifact.diagnostic] if artifact.diagnostic else []),
            archived=archived,
        )


def _nonzero_weights(weights: Any) -> tuple[tuple[str, float], ...]:
    if not isinstance(weights, dict):
        return ()
    items: list[tuple[str, float]] = []
    for key, value in weights.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number != 0.0:
            items.append((str(key), number))
    return tuple(sorted(items, key=lambda item: (-item[1], item[0])))
