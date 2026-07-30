"""Errors and status codes for Dataset Package & Registry v1."""

from __future__ import annotations

from typing import Literal

PackageStatus = Literal[
    "valid",
    "invalid",
    "unregistered",
    "unverifiable_alignment",
    "refused_overwrite",
]

EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_INVALID = 2
EXIT_UNREGISTERED = 3
EXIT_UNVERIFIABLE = 4
EXIT_REFUSED_OVERWRITE = 5

STATUS_EXIT_CODES: dict[PackageStatus, int] = {
    "valid": EXIT_SUCCESS,
    "invalid": EXIT_INVALID,
    "unregistered": EXIT_UNREGISTERED,
    "unverifiable_alignment": EXIT_UNVERIFIABLE,
    "refused_overwrite": EXIT_REFUSED_OVERWRITE,
}


class DatasetRegistryError(ValueError):
    """Base error for dataset registry operations."""

    def __init__(self, message: str, *, status: PackageStatus = "invalid") -> None:
        super().__init__(message)
        self.status: PackageStatus = status
        self.reason_code = status.upper()


class UnverifiableAlignmentError(DatasetRegistryError):
    """Raised when row alignment cannot be proven from verifiable identities."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status="unverifiable_alignment")


class OverwriteRefusedError(DatasetRegistryError):
    """Raised when writing would overwrite an existing immutable manifest."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status="refused_overwrite")
