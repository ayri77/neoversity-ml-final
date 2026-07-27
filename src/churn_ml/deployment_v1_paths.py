from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class DeploymentPathError(ValueError):
    """Raised when a deployment path can escape or alias an authenticated tree."""


@dataclass(frozen=True)
class SafeTree:
    root: Path
    files: tuple[Path, ...]
    directories: tuple[Path, ...]


def prewalk_regular_tree(
    root: Path,
    *,
    containment_roots: Iterable[Path] = (),
    reject_hardlinks: bool,
) -> SafeTree:
    """Enumerate without following links, junctions, or reparse points."""
    unresolved = root.absolute()
    _reject_special(unresolved, unresolved)
    try:
        canonical_root = unresolved.resolve(strict=True)
    except OSError as error:
        raise DeploymentPathError(f"Tree root does not exist: {root}") from error
    if not canonical_root.is_dir():
        raise DeploymentPathError(f"Tree root is not a directory: {root}")
    canonical_containment = tuple(
        item.resolve(strict=True) for item in containment_roots
    )
    _require_containment(canonical_root, canonical_containment, "tree root")
    files: list[Path] = []
    directories: list[Path] = []

    def visit(directory: Path) -> None:
        try:
            entries = os.scandir(directory)
        except OSError as error:
            raise DeploymentPathError(
                f"Cannot enumerate directory: {directory}"
            ) from error
        with entries:
            for entry in entries:
                path = Path(entry.path)
                relative = path.relative_to(canonical_root)
                metadata = path.stat(follow_symlinks=False)
                _reject_special(path, canonical_root, metadata=metadata)
                try:
                    resolved = path.resolve(strict=True)
                except OSError as error:
                    raise DeploymentPathError(
                        f"Tree descendant cannot be resolved: {relative.as_posix()}"
                    ) from error
                if canonical_root not in resolved.parents:
                    raise DeploymentPathError(
                        f"Tree descendant escapes root: {relative.as_posix()}"
                    )
                _require_containment(
                    resolved,
                    canonical_containment,
                    relative.as_posix(),
                )
                if entry.is_dir(follow_symlinks=False):
                    directories.append(relative)
                    visit(path)
                elif entry.is_file(follow_symlinks=False):
                    if reject_hardlinks and int(metadata.st_nlink) > 1:
                        raise DeploymentPathError(
                            f"Multiply linked file is prohibited: {relative.as_posix()}"
                        )
                    files.append(relative)
                else:
                    raise DeploymentPathError(
                        f"Only regular files/directories are allowed: {relative.as_posix()}"
                    )

    visit(canonical_root)
    return SafeTree(
        root=canonical_root,
        files=tuple(sorted(files, key=lambda item: item.as_posix())),
        directories=tuple(sorted(directories, key=lambda item: item.as_posix())),
    )


def validate_regular_file(
    path: Path,
    *,
    containment_root: Path | None = None,
    reject_hardlinks: bool,
) -> Path:
    """Validate every existing component and the leaf without following links."""
    unresolved = path.absolute()
    root = (
        containment_root.resolve(strict=True)
        if containment_root is not None
        else unresolved.parent.resolve(strict=True)
    )
    absolute = unresolved.absolute()
    try:
        relative = absolute.relative_to(root)
    except ValueError as error:
        raise DeploymentPathError("File path is outside its permitted root.") from error
    current = root
    for part in relative.parts:
        current = current / part
        if not current.exists() and not current.is_symlink():
            raise DeploymentPathError(
                f"File path does not exist: {relative.as_posix()}"
            )
        _reject_special(current, root)
    resolved = absolute.resolve(strict=True)
    if root not in resolved.parents or not resolved.is_file():
        raise DeploymentPathError("File is not a contained regular file.")
    metadata = resolved.stat(follow_symlinks=False)
    if reject_hardlinks and int(metadata.st_nlink) > 1:
        raise DeploymentPathError("Multiply linked file is prohibited.")
    return resolved


def validate_new_path(path: Path) -> Path:
    """Reject links/reparse points in every existing output-path component."""
    absolute = path.absolute()
    anchor = Path(absolute.anchor)
    current = anchor
    for part in absolute.parts[1:]:
        current = current / part
        if not current.exists() and not current.is_symlink():
            break
        _reject_special(current, anchor)
    return absolute


def _reject_special(
    path: Path,
    root: Path,
    *,
    metadata: os.stat_result | None = None,
) -> None:
    try:
        info = metadata if metadata is not None else path.lstat()
    except OSError as error:
        raise DeploymentPathError(f"Cannot inspect path: {path}") from error
    is_junction = bool(getattr(path, "is_junction", lambda: False)())
    attributes = int(getattr(info, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    is_reparse = bool(reparse_flag and attributes & reparse_flag)
    if path.is_symlink() or is_junction or is_reparse:
        label = "." if path == root else path.name
        raise DeploymentPathError(
            f"Links, junctions, and reparse points are prohibited: {label}"
        )


def _require_containment(
    path: Path,
    roots: tuple[Path, ...],
    label: str,
) -> None:
    for root in roots:
        if path != root and root not in path.parents:
            raise DeploymentPathError(f"{label} escapes required containment root.")
