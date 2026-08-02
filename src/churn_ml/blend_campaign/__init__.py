"""CLI-first orchestration for immutable probability-blend campaigns."""

from src.churn_ml.blend_campaign.cache_v1 import install_diversity_cache
from src.churn_ml.blend_campaign.config_v1 import (
    BlendCampaignConfig,
    BlendCampaignConfigurationError,
    ExperimentConfig,
    load_campaign_config,
    parse_campaign_config,
)


# The wrapper is source-diagnostic-only and keyed by immutable manifest hashes.
# It changes no blend evaluation or optimization mathematics.
install_diversity_cache()

__all__ = [
    "BlendCampaignConfig",
    "BlendCampaignConfigurationError",
    "ExperimentConfig",
    "load_campaign_config",
    "parse_campaign_config",
]
