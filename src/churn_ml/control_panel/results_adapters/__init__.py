"""Schema-aware Results adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.churn_ml.control_panel.artifacts import ArtifactRecord
from src.churn_ml.control_panel.results_adapters.autogluon import AutoGluonRunAdapter
from src.churn_ml.control_panel.results_adapters.candidate_submission import (
    CandidateSubmissionV1Adapter,
)
from src.churn_ml.control_panel.results_adapters.comparison import ComparisonAdapter
from src.churn_ml.control_panel.results_adapters.prediction_blend import (
    PredictionBlendV1Adapter,
)
from src.churn_ml.control_panel.results_adapters.prediction_candidate import (
    PredictionCandidateV1Adapter,
)
from src.churn_ml.control_panel.results_adapters.research_v2 import ResearchV2Adapter
from src.churn_ml.control_panel.results_entities import ResultEntity, ResultEntityKind


class ResultsAdapter(Protocol):
    entity_kind: ResultEntityKind
    reader_ids: tuple[str, ...]

    def adapt(
        self,
        artifact: ArtifactRecord,
        *,
        repository_root: Path,
        archived: bool = False,
    ) -> ResultEntity: ...


_ADAPTERS: tuple[ResultsAdapter, ...] = (
    ResearchV2Adapter(),
    AutoGluonRunAdapter(),
    ComparisonAdapter(),
    PredictionCandidateV1Adapter(),
    PredictionBlendV1Adapter(),
    CandidateSubmissionV1Adapter(),
)


def adapter_for_reader(reader_id: str) -> ResultsAdapter | None:
    for adapter in _ADAPTERS:
        if reader_id in adapter.reader_ids:
            return adapter
    return None


def adapt_artifact(
    artifact: ArtifactRecord,
    *,
    repository_root: Path,
    archived: bool = False,
) -> ResultEntity | None:
    adapter = adapter_for_reader(artifact.reader_id)
    if adapter is None:
        return None
    return adapter.adapt(
        artifact, repository_root=repository_root, archived=archived
    )


def supported_reader_ids() -> tuple[str, ...]:
    ids: list[str] = []
    for adapter in _ADAPTERS:
        ids.extend(adapter.reader_ids)
    return tuple(ids)


ENTITY_VIEW_READERS: dict[str, tuple[str, ...]] = {
    "Model runs": ("research_v2", "autogluon_v1"),
    "Comparisons": ("paired_comparison", "dataset_comparison_v1"),
    "Prediction candidates": ("prediction_candidate_v1",),
    "Blends": ("prediction_blend_v1",),
    "Submissions": ("candidate_submission_v1",),
}

__all__ = [
    "ENTITY_VIEW_READERS",
    "adapt_artifact",
    "adapter_for_reader",
    "supported_reader_ids",
    "AutoGluonRunAdapter",
    "CandidateSubmissionV1Adapter",
    "ComparisonAdapter",
    "PredictionBlendV1Adapter",
    "PredictionCandidateV1Adapter",
    "ResearchV2Adapter",
]
