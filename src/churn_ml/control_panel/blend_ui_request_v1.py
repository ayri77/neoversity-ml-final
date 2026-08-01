"""Control Panel re-export of the blend UI request contract."""

from __future__ import annotations

from src.churn_ml.blending.request_v1 import (
    METHODS,
    MAX_CANDIDATES,
    MIN_CANDIDATES,
    REQUEST_ROOT_RELATIVE,
    SCHEMA_VERSION,
    BlendUIRequest,
    BlendUIRequestError,
    build_request_payload,
    load_blend_ui_request,
    materialize_blend_ui_request,
    method_to_strategy_optimizer,
    request_file_sha256,
    settings_from_request_payload,
    strategy_optimizer_to_method,
    validate_candidate_ids,
)

__all__ = [
    "METHODS",
    "MAX_CANDIDATES",
    "MIN_CANDIDATES",
    "REQUEST_ROOT_RELATIVE",
    "SCHEMA_VERSION",
    "BlendUIRequest",
    "BlendUIRequestError",
    "build_request_payload",
    "load_blend_ui_request",
    "materialize_blend_ui_request",
    "method_to_strategy_optimizer",
    "request_file_sha256",
    "settings_from_request_payload",
    "strategy_optimizer_to_method",
    "validate_candidate_ids",
]
