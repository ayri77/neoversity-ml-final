from __future__ import annotations

import os
import stat
from pathlib import Path


class PathSafetyError(ValueError):
    """Raised when a filesystem object is unsafe to traverse or use."""


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def require_safe_existing_ancestors(path: Path) -> None:
    """Inspect every existing component without following links or reparse points."""
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise PathSafetyError(
                f"Could not inspect path component: {current}."
            ) from error
        _reject_link(info, current)


def require_safe_directory(path: Path) -> Path:
    require_safe_existing_ancestors(path)
    try:
        info = path.lstat()
    except OSError as error:
        raise PathSafetyError(f"Could not inspect directory: {path}.") from error
    _reject_link(info, path)
    if not stat.S_ISDIR(info.st_mode):
        raise PathSafetyError(f"Expected a regular directory: {path}.")
    return path


def require_regular_file(
    path: Path,
    *,
    reject_hardlinks: bool = True,
) -> Path:
    require_safe_existing_ancestors(path)
    try:
        info = path.lstat()
    except OSError as error:
        raise PathSafetyError(f"Could not inspect file: {path}.") from error
    _reject_link(info, path)
    if not stat.S_ISREG(info.st_mode):
        raise PathSafetyError(f"Expected a regular file: {path}.")
    if reject_hardlinks and info.st_nlink != 1:
        raise PathSafetyError(f"Hard-linked files are not allowed: {path}.")
    return path


def path_exists_nonfollowing(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise PathSafetyError(f"Could not inspect path: {path}.") from error
    return True


def canonical_executable(path: str | Path) -> str:
    try:
        resolved = Path(path).resolve(strict=True)
        require_regular_file(resolved, reject_hardlinks=False)
    except (OSError, PathSafetyError) as error:
        raise PathSafetyError("Executable identity could not be resolved.") from error
    value = str(resolved)
    return os.path.normcase(value) if os.name == "nt" else value


def _reject_link(info: os.stat_result, path: Path) -> None:
    attributes = getattr(info, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if stat.S_ISLNK(info.st_mode) or bool(attributes & reparse_flag):
        raise PathSafetyError(
            f"Symlink, junction, or reparse paths are not allowed: {path}."
        )
