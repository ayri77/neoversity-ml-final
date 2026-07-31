"""Path safety helpers for Dataset Campaign Runner v1."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from src.churn_ml.control_panel.path_safety import (
    PathSafetyError,
    require_safe_directory,
    require_safe_existing_ancestors,
)
from src.churn_ml.dataset_campaign.errors import CampaignConfigurationError


def safe_repo_relative_path(
    repository_root: Path,
    relative: str,
    *,
    label: str,
) -> Path:
    """Resolve a repository-relative path without following unsafe links."""
    root = repository_root.resolve()
    posix = PurePosixPath(str(relative).replace("\\", "/"))
    if (
        posix.is_absolute()
        or ".." in posix.parts
        or any(part in {"", "."} for part in posix.parts)
    ):
        raise CampaignConfigurationError(f"{label} path is unsafe: {relative!r}.")
    path = root / Path(*posix.parts)
    try:
        require_safe_existing_ancestors(path)
    except PathSafetyError as error:
        raise CampaignConfigurationError(str(error)) from error
    try:
        resolved = path.resolve()
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise CampaignConfigurationError(
            f"{label} escapes the repository root: {relative!r}."
        ) from error
    return resolved


def safe_repo_relative_file(
    repository_root: Path,
    relative: str,
    *,
    label: str,
) -> Path:
    path = safe_repo_relative_path(repository_root, relative, label=label)
    try:
        require_safe_existing_ancestors(path)
        if path.is_symlink():
            raise CampaignConfigurationError(
                f"{label} must not be a symlink: {relative!r}."
            )
    except PathSafetyError as error:
        raise CampaignConfigurationError(str(error)) from error
    if not path.is_file():
        raise CampaignConfigurationError(f"{label} does not exist: {relative}.")
    return path


def safe_repo_relative_dir(
    repository_root: Path,
    relative: str,
    *,
    label: str,
    must_exist: bool = True,
) -> Path:
    path = safe_repo_relative_path(repository_root, relative, label=label)
    if not must_exist and not path.exists():
        return path
    try:
        require_safe_directory(path)
    except PathSafetyError as error:
        raise CampaignConfigurationError(str(error)) from error
    return path


def to_repo_relative(repository_root: Path, path: Path) -> str:
    root = repository_root.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise CampaignConfigurationError(
            f"Path escapes repository root: {path}."
        ) from error
