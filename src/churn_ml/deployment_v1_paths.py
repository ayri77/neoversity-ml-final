from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal


PathKind = Literal["file", "directory", "either"]


class DeploymentPathError(ValueError):
    """Raised when a deployment path can escape or alias an authenticated tree."""


@dataclass(frozen=True)
class SafePath:
    requested: Path
    normalized: Path
    containment_root: Path
    validated_chain: tuple[Path, ...]
    terminal_kind: Literal["file", "directory", "missing"]
    canonical: Path


@dataclass(frozen=True)
class SafeTree:
    path: SafePath
    files: tuple[Path, ...]
    directories: tuple[Path, ...]

    @property
    def root(self) -> Path:
        return self.path.canonical


def validate_path_chain(
    *,
    containment_root: Path,
    requested_path: Path,
    require_exists: bool,
    expected_kind: PathKind,
    reject_hardlinks: bool = False,
) -> SafePath:
    """Validate every ancestor without following it, then resolve once."""
    requested = requested_path
    containment = containment_root.absolute()
    normalized = (
        requested_path.absolute()
        if requested_path.is_absolute()
        else (containment / requested_path).absolute()
    )
    try:
        normalized.relative_to(containment)
    except ValueError as error:
        raise DeploymentPathError(
            "Requested path is outside its permitted root."
        ) from error

    # Validate the containment root's own ancestors so a linked ancestor above the
    # requested subtree cannot disappear through a later resolve().
    anchor = Path(containment.anchor)
    chain: list[Path] = []
    current = anchor
    containment_parts = containment.parts[1:]
    for part in containment_parts:
        current = current / part
        _inspect_existing_component(
            current,
            expected_directory=True,
            reject_hardlinks=False,
            allow_mount=False,
        )
        chain.append(current)

    relative = normalized.relative_to(containment)
    current = containment
    missing_seen = False
    for index, part in enumerate(relative.parts):
        current = current / part
        terminal = index == len(relative.parts) - 1
        try:
            info = current.lstat()
        except FileNotFoundError:
            missing_seen = True
            if require_exists:
                raise DeploymentPathError(
                    f"Path component does not exist: {relative.as_posix()}"
                )
            break
        except OSError as error:
            raise DeploymentPathError(
                f"Cannot inspect path component: {current}"
            ) from error
        _reject_special(current, containment, metadata=info, allow_mount=False)
        if terminal:
            _require_kind(current, info, expected_kind)
            if (
                reject_hardlinks
                and stat.S_ISREG(info.st_mode)
                and int(info.st_nlink) > 1
            ):
                raise DeploymentPathError("Multiply linked file is prohibited.")
        elif not stat.S_ISDIR(info.st_mode):
            raise DeploymentPathError("A path ancestor is not a directory.")
        chain.append(current)

    if require_exists and missing_seen:
        raise DeploymentPathError("Requested path does not exist.")
    if require_exists:
        canonical = normalized.resolve(strict=True)
        terminal_kind: Literal["file", "directory", "missing"] = (
            "file" if canonical.is_file() else "directory"
        )
    else:
        nearest = containment
        remaining: tuple[str, ...] = relative.parts
        for index, part in enumerate(relative.parts):
            candidate = nearest / part
            try:
                candidate.lstat()
            except FileNotFoundError:
                remaining = relative.parts[index:]
                break
            nearest = candidate
        else:
            remaining = ()
        canonical_nearest = nearest.resolve(strict=True)
        canonical = canonical_nearest.joinpath(*remaining)
        terminal_kind = (
            "file"
            if normalized.is_file()
            else "directory"
            if normalized.is_dir()
            else "missing"
        )
    canonical_containment = containment.resolve(strict=True)
    if (
        canonical != canonical_containment
        and canonical_containment not in canonical.parents
    ):
        raise DeploymentPathError("Validated path escapes its permitted root.")
    return SafePath(
        requested=requested,
        normalized=normalized,
        containment_root=containment,
        validated_chain=tuple(chain),
        terminal_kind=terminal_kind,
        canonical=canonical,
    )


def validate_existing_root(root: Path) -> Path:
    """Authenticate an existing directory from its filesystem anchor."""
    absolute = root.absolute()
    return validate_path_chain(
        containment_root=Path(absolute.anchor),
        requested_path=absolute,
        require_exists=True,
        expected_kind="directory",
    ).canonical


def prewalk_regular_tree(
    root: Path,
    *,
    containment_roots: Iterable[Path] = (),
    reject_hardlinks: bool,
) -> SafeTree:
    """Enumerate a validated tree without following links or reparse points."""
    roots = tuple(containment_roots)
    containment = roots[0] if roots else Path(root.absolute().anchor)
    safe_root = validate_path_chain(
        containment_root=containment,
        requested_path=root,
        require_exists=True,
        expected_kind="directory",
    )
    for extra in roots[1:]:
        absolute_extra = extra.absolute()
        canonical_extra = validate_path_chain(
            containment_root=Path(absolute_extra.anchor),
            requested_path=absolute_extra,
            require_exists=True,
            expected_kind="directory",
        ).canonical
        if (
            safe_root.canonical != canonical_extra
            and canonical_extra not in safe_root.canonical.parents
        ):
            raise DeploymentPathError("Tree root escapes required containment root.")
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
                child = Path(entry.path)
                relative = child.relative_to(safe_root.canonical)
                try:
                    metadata = child.lstat()
                except OSError as error:
                    raise DeploymentPathError(
                        f"Cannot inspect tree descendant: {relative.as_posix()}"
                    ) from error
                _reject_special(
                    child, safe_root.canonical, metadata=metadata, allow_mount=False
                )
                if stat.S_ISDIR(metadata.st_mode):
                    directories.append(relative)
                    visit(child)
                elif stat.S_ISREG(metadata.st_mode):
                    if reject_hardlinks and int(metadata.st_nlink) > 1:
                        raise DeploymentPathError(
                            f"Multiply linked file is prohibited: {relative.as_posix()}"
                        )
                    files.append(relative)
                else:
                    raise DeploymentPathError(
                        f"Only regular files/directories are allowed: {relative.as_posix()}"
                    )
                resolved = child.resolve(strict=True)
                if safe_root.canonical not in resolved.parents:
                    raise DeploymentPathError(
                        f"Tree descendant escapes root: {relative.as_posix()}"
                    )

    visit(safe_root.canonical)
    return SafeTree(
        path=safe_root,
        files=tuple(sorted(files, key=lambda item: item.as_posix())),
        directories=tuple(sorted(directories, key=lambda item: item.as_posix())),
    )


def validate_regular_file(
    path: Path,
    *,
    containment_root: Path | None = None,
    reject_hardlinks: bool,
) -> Path:
    root = containment_root or Path(path.absolute().anchor)
    return validate_path_chain(
        containment_root=root,
        requested_path=path,
        require_exists=True,
        expected_kind="file",
        reject_hardlinks=reject_hardlinks,
    ).canonical


def validate_new_path(path: Path, *, containment_root: Path | None = None) -> Path:
    root = containment_root or Path(path.absolute().anchor)
    return validate_path_chain(
        containment_root=root,
        requested_path=path,
        require_exists=False,
        expected_kind="either",
    ).canonical


def _inspect_existing_component(
    path: Path,
    *,
    expected_directory: bool,
    reject_hardlinks: bool,
    allow_mount: bool,
) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise DeploymentPathError(
            f"Containment ancestor does not exist: {path}"
        ) from error
    except OSError as error:
        raise DeploymentPathError(f"Cannot inspect path component: {path}") from error
    _reject_special(path, path, metadata=metadata, allow_mount=allow_mount)
    if expected_directory and not stat.S_ISDIR(metadata.st_mode):
        raise DeploymentPathError("Containment ancestor is not a directory.")
    if (
        reject_hardlinks
        and stat.S_ISREG(metadata.st_mode)
        and int(metadata.st_nlink) > 1
    ):
        raise DeploymentPathError("Multiply linked file is prohibited.")


def _require_kind(path: Path, metadata: os.stat_result, expected: PathKind) -> None:
    if expected == "file" and not stat.S_ISREG(metadata.st_mode):
        raise DeploymentPathError(f"Expected a regular file: {path}")
    if expected == "directory" and not stat.S_ISDIR(metadata.st_mode):
        raise DeploymentPathError(f"Expected a directory: {path}")
    if expected == "either" and not (
        stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode)
    ):
        raise DeploymentPathError(f"Expected a regular file or directory: {path}")


def _reject_special(
    path: Path,
    root: Path,
    *,
    metadata: os.stat_result,
    allow_mount: bool,
) -> None:
    is_junction = bool(getattr(path, "is_junction", lambda: False)())
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    is_reparse = bool(attributes & reparse_flag)
    is_link = stat.S_ISLNK(metadata.st_mode) or path.is_symlink()
    is_mount = False
    if not allow_mount:
        try:
            is_mount = path.is_mount()
        except OSError:
            is_mount = False
    if is_link or is_junction or is_reparse or is_mount:
        label = "." if path == root else path.name
        raise DeploymentPathError(
            f"Links, junctions, reparse points, and mounts are prohibited: {label}"
        )
