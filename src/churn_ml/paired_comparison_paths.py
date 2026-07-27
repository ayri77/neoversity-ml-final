from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable, Mapping


FORBIDDEN_NAMESPACE_COMPONENTS = {
    "autogluon",
    "autogluon_runs",
    "competition",
    "competition-test",
    "competition_test",
    "competition_test_assets",
    "final-submission",
    "final-submissions",
    "final_submission",
    "final_submissions",
    "kaggle",
    "kaggle_assets",
    "kaggle_submissions",
    "model-artifact",
    "model-artifacts",
    "model_artifact",
    "model_artifacts",
    "models",
    "sample-submission",
    "sample-submissions",
    "sample_submission",
    "sample_submissions",
    "submission",
    "submissions",
    "x_test",
}
FORBIDDEN_NAMESPACE_SEQUENCES = {
    ("artifacts", "autogluon_runs"),
    ("artifacts", "final_submissions"),
    ("data", "raw"),
    ("data", "test"),
}
FORBIDDEN_BINARY_SUFFIXES = {
    ".bin",
    ".cbm",
    ".joblib",
    ".model",
    ".pickle",
    ".pkl",
}


class PairedPathSafetyError(ValueError):
    """Raised when a comparison path violates a stable safety contract."""

    def __init__(self, message: str, *, reason_code: str, field_path: str) -> None:
        self.reason_code = reason_code
        self.field_path = field_path
        self.detail = message
        super().__init__(f"{reason_code} at {field_path}: {message}")


@dataclass(frozen=True)
class TreeSnapshot:
    files: tuple[Path, ...]
    directories: tuple[Path, ...]


def resolve_repository_path(
    path: Path,
    *,
    project_root: Path,
    role: str,
    field_path: str,
    must_exist: bool,
    require_directory: bool,
) -> Path:
    root = project_root.resolve()
    text = str(path)
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
    ):
        unresolved = path
    else:
        if ".." in posix.parts or ".." in windows.parts:
            _raise(
                "path traversal is prohibited",
                f"{role.upper()}_PATH_TRAVERSAL",
                field_path,
            )
        unresolved = root / path
    _reject_linked_existing_components(
        unresolved,
        root,
        reason_code=f"{role.upper()}_PATH_LINK",
        field_path=field_path,
    )
    try:
        resolved = unresolved.resolve(strict=must_exist)
    except OSError as error:
        raise PairedPathSafetyError(
            "path does not exist",
            reason_code=f"{role.upper()}_PATH_MISSING",
            field_path=field_path,
        ) from error
    if resolved == root or root not in resolved.parents:
        _raise(
            "path is outside the repository",
            f"{role.upper()}_PATH_OUTSIDE_REPOSITORY",
            field_path,
        )
    if must_exist:
        if require_directory and not resolved.is_dir():
            _raise(
                "path must be a directory",
                f"{role.upper()}_PATH_TYPE_INVALID",
                field_path,
            )
        if not require_directory and not resolved.is_file():
            _raise(
                "path must be a file",
                f"{role.upper()}_PATH_TYPE_INVALID",
                field_path,
            )
    relative_parts = tuple(part.casefold() for part in resolved.relative_to(root).parts)
    forbidden = forbidden_namespace(relative_parts)
    if forbidden is not None:
        _raise(
            f"forbidden namespace: {'/'.join(forbidden)}",
            f"{role.upper()}_FORBIDDEN_NAMESPACE",
            field_path,
        )
    return resolved


def prewalk_regular_tree(
    root: Path,
    *,
    project_root: Path,
    role: str,
    field_path: str,
    reject_hardlinks: bool,
    reject_forbidden_descendants: bool,
) -> TreeSnapshot:
    canonical_root = root.resolve(strict=True)
    canonical_project = project_root.resolve()
    files: list[Path] = []
    directories: list[Path] = []

    def visit(directory: Path) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative = path.relative_to(canonical_root)
                child_field = f"{field_path}.{relative.as_posix()}"
                metadata = entry.stat(follow_symlinks=False)
                if entry.is_symlink() or _metadata_is_reparse(metadata):
                    _raise(
                        "links, junctions, and reparse points are prohibited",
                        f"{role.upper()}_TREE_LINK",
                        child_field,
                    )
                try:
                    resolved = path.resolve(strict=True)
                except OSError as error:
                    raise PairedPathSafetyError(
                        "descendant cannot be resolved",
                        reason_code=f"{role.upper()}_TREE_ENTRY_INVALID",
                        field_path=child_field,
                    ) from error
                if (
                    canonical_root not in resolved.parents
                    or canonical_project not in resolved.parents
                ):
                    _raise(
                        "descendant escapes its permitted roots",
                        f"{role.upper()}_TREE_ESCAPE",
                        child_field,
                    )
                parts = tuple(part.casefold() for part in relative.parts)
                if reject_forbidden_descendants:
                    forbidden = forbidden_namespace(parts)
                    if forbidden is not None or forbidden_artifact_file(relative):
                        _raise(
                            "forbidden descendant artifact",
                            f"{role.upper()}_TREE_FORBIDDEN_ARTIFACT",
                            child_field,
                        )
                if entry.is_dir(follow_symlinks=False):
                    directories.append(relative)
                    visit(path)
                elif entry.is_file(follow_symlinks=False):
                    if reject_hardlinks and metadata.st_nlink > 1:
                        _raise(
                            "multiply linked regular files are prohibited",
                            f"{role.upper()}_TREE_HARDLINK",
                            child_field,
                        )
                    files.append(relative)
                else:
                    _raise(
                        "only regular files and directories are permitted",
                        f"{role.upper()}_TREE_ENTRY_INVALID",
                        child_field,
                    )

    visit(canonical_root)
    return TreeSnapshot(
        files=tuple(sorted(files, key=lambda item: item.as_posix())),
        directories=tuple(sorted(directories, key=lambda item: item.as_posix())),
    )


def assert_pairwise_disjoint_paths(
    paths: Mapping[str, Path],
) -> None:
    canonical = {name: path.resolve(strict=False) for name, path in paths.items()}
    names = list(canonical)
    for index, left_name in enumerate(names):
        left = canonical[left_name]
        for right_name in names[index + 1 :]:
            right = canonical[right_name]
            if left == right or left in right.parents or right in left.parents:
                _raise(
                    f"{left_name} and {right_name} overlap",
                    "COMPARISON_PATHS_OVERLAP",
                    f"paths.{left_name}|paths.{right_name}",
                )


def forbidden_namespace(parts: Iterable[str]) -> tuple[str, ...] | None:
    normalized = tuple(part.casefold() for part in parts)
    for part in normalized:
        if part in FORBIDDEN_NAMESPACE_COMPONENTS:
            return (part,)
    for sequence in FORBIDDEN_NAMESPACE_SEQUENCES:
        width = len(sequence)
        for index in range(len(normalized) - width + 1):
            if normalized[index : index + width] == sequence:
                return sequence
    return None


def forbidden_artifact_file(relative: Path) -> bool:
    name = relative.name.casefold()
    return relative.suffix.casefold() in FORBIDDEN_BINARY_SUFFIXES or name in {
        "_failed.tmp",
        "_success.tmp",
    }


def _reject_linked_existing_components(
    path: Path,
    project_root: Path,
    *,
    reason_code: str,
    field_path: str,
) -> None:
    absolute = path.absolute()
    try:
        relative = absolute.relative_to(project_root)
    except ValueError:
        return
    current = project_root
    for part in relative.parts:
        current = current / part
        if not current.exists() and not current.is_symlink():
            break
        try:
            metadata = current.stat(follow_symlinks=False)
        except OSError as error:
            raise PairedPathSafetyError(
                "cannot inspect path component",
                reason_code=reason_code,
                field_path=field_path,
            ) from error
        if current.is_symlink() or _metadata_is_reparse(metadata):
            _raise(
                "linked path components are prohibited",
                reason_code,
                field_path,
            )


def _metadata_is_reparse(metadata: os.stat_result) -> bool:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return bool(reparse_flag and attributes & reparse_flag)


def _raise(message: str, reason_code: str, field_path: str) -> None:
    raise PairedPathSafetyError(
        message,
        reason_code=reason_code,
        field_path=field_path,
    )
