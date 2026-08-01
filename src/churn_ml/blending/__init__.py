"""Source-neutral multi-candidate probability blending (v1).

Consumes only ``prediction_candidate_v1`` packages. Independent of AutoGluon,
Research v2 adapters, and UI. Parent origin may differ so long as row/target
identity and probability semantics are compatible.
"""

from __future__ import annotations

from src.churn_ml.blending.compatibility_v1 import (
    BlendCompatibilityError,
    CompatibleCandidateSet,
    load_compatible_candidates,
)
from src.churn_ml.blending.evaluation_v1 import BlendEvaluationResult, run_blend_evaluation

__all__ = [
    "BlendCompatibilityError",
    "BlendEvaluationResult",
    "CompatibleCandidateSet",
    "load_compatible_candidates",
    "run_blend_evaluation",
]
