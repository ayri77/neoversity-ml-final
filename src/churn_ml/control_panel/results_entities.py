"""First-class Result entity contracts for schema-aware Results views."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from src.churn_ml.control_panel.results_fields import ResultField


class ResultEntityKind(str, Enum):
    MODEL_RUN = "model_run"
    COMPARISON = "comparison"
    PREDICTION_CANDIDATE = "prediction_candidate"
    PROBABILITY_BLEND = "probability_blend"
    CANDIDATE_SUBMISSION = "candidate_submission"


@dataclass(frozen=True)
class ResultLink:
    """Structural lineage reference to another Results entity or artifact."""

    relation: str
    target_kind: ResultEntityKind | None
    target_id: str | None
    target_path: str | None = None
    resolved: bool = False
    message: str | None = None


@dataclass(frozen=True)
class ResultEntity:
    """Normalized projection of one Results artifact."""

    kind: ResultEntityKind
    entity_id: str
    display_label: str
    reader_id: str
    source_path: str
    schema_version: ResultField
    state: ResultField
    created_at: ResultField
    dataset_id: ResultField
    exploratory: ResultField
    fields: Mapping[str, ResultField] = field(default_factory=dict)
    lineage: tuple[ResultLink, ...] = ()
    diagnostics: tuple[str, ...] = ()
    archived: bool = False

    def field(self, key: str) -> ResultField | None:
        return self.fields.get(key)

    def require_field(self, key: str) -> ResultField:
        value = self.fields.get(key)
        if value is None:
            raise KeyError(f"Unknown ResultEntity field: {key}")
        return value
