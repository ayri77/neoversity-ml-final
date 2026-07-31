"""Dataset Campaign / Matrix Runner v1."""

from __future__ import annotations

from src.churn_ml.dataset_campaign.constants import (
    CAMPAIGN_CONTRACT_VERSION,
    UNBIASED_SCREENING_DATASET_IDS,
)
from src.churn_ml.dataset_campaign.execute import (
    inspect_campaign,
    run_campaign,
    validate_only,
)
from src.churn_ml.dataset_campaign.matrix import expand_matrix
from src.churn_ml.dataset_campaign.schema import (
    CampaignSpec,
    load_campaign_spec,
    parse_campaign_spec,
)

__all__ = [
    "CAMPAIGN_CONTRACT_VERSION",
    "UNBIASED_SCREENING_DATASET_IDS",
    "CampaignSpec",
    "expand_matrix",
    "inspect_campaign",
    "load_campaign_spec",
    "parse_campaign_spec",
    "run_campaign",
    "validate_only",
]
