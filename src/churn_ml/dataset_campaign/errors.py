"""Errors and CLI exit codes for Dataset Campaign Runner v1."""

from __future__ import annotations


class DatasetCampaignError(ValueError):
    """Base error for campaign specification, resolution, or execution failures."""


class CampaignConfigurationError(DatasetCampaignError):
    """Raised when a campaign specification is invalid or unsafe."""


class CampaignManifestError(DatasetCampaignError):
    """Raised when a frozen campaign manifest is missing, corrupt, or drifted."""


class CampaignExecutionError(DatasetCampaignError):
    """Raised when campaign execution or resume cannot proceed safely."""


class CampaignValidationError(DatasetCampaignError):
    """Raised when validate-only collects one or more blocking failures."""

    def __init__(self, message: str, *, failures: list[str] | None = None) -> None:
        super().__init__(message)
        self.failures = list(failures or [])


EXIT_SUCCESS = 0
EXIT_UNEXPECTED = 1
EXIT_CONFIG = 2
EXIT_VALIDATION = 3
EXIT_MANIFEST = 4
EXIT_EXECUTION = 5
