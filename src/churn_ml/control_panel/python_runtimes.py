"""Repository-local Python runtime resolution for Control Panel jobs."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from src.churn_ml.control_panel.path_safety import is_within


AUTOGLUON_RUNTIME_ID = "venv_autogluon"
_WINDOWS_RELATIVE = ".venv-autogluon/Scripts/python.exe"
_POSIX_RELATIVE = ".venv-autogluon/bin/python"


class PythonRuntimeError(ValueError):
    """Raised when a required repository-local interpreter is unavailable."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(message)


@dataclass(frozen=True)
class ResolvedPythonRuntime:
    runtime_id: str
    relative_path: str
    absolute_path: Path
    available: bool
    reason_code: str | None = None
    reason: str | None = None


def autogluon_interpreter_relative() -> str:
    if os.name == "nt":
        return _WINDOWS_RELATIVE
    return _POSIX_RELATIVE


def resolve_autogluon_python(
    repository_root: Path | str,
) -> ResolvedPythonRuntime:
    """Resolve ``.venv-autogluon`` interpreter inside the repository only."""
    root = Path(repository_root).resolve(strict=True)
    relative = autogluon_interpreter_relative()
    return _resolve_runtime(
        root,
        runtime_id=AUTOGLUON_RUNTIME_ID,
        relative_path=relative,
    )


def require_autogluon_python(repository_root: Path | str) -> ResolvedPythonRuntime:
    resolved = resolve_autogluon_python(repository_root)
    if not resolved.available:
        raise PythonRuntimeError(
            resolved.reason or "AutoGluon interpreter unavailable.",
            reason_code=resolved.reason_code or "autogluon_runtime_unavailable",
        )
    return resolved


def _resolve_runtime(
    root: Path,
    *,
    runtime_id: str,
    relative_path: str,
) -> ResolvedPythonRuntime:
    try:
        relative = _normalize_relative(relative_path)
    except PythonRuntimeError as error:
        return ResolvedPythonRuntime(
            runtime_id=runtime_id,
            relative_path=relative_path,
            absolute_path=root,
            available=False,
            reason_code=error.reason_code,
            reason=str(error),
        )
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    try:
        _reject_symlink_ancestors(candidate)
    except PythonRuntimeError as error:
        return ResolvedPythonRuntime(
            runtime_id=runtime_id,
            relative_path=relative,
            absolute_path=candidate,
            available=False,
            reason_code=error.reason_code,
            reason=str(error),
        )
    try:
        info = candidate.lstat()
    except FileNotFoundError:
        return ResolvedPythonRuntime(
            runtime_id=runtime_id,
            relative_path=relative,
            absolute_path=candidate,
            available=False,
            reason_code="autogluon_runtime_missing",
            reason=(
                f"AutoGluon interpreter not found at `{relative}`. "
                "Create `.venv-autogluon` before validating or preparing candidates."
            ),
        )
    except OSError as error:
        return ResolvedPythonRuntime(
            runtime_id=runtime_id,
            relative_path=relative,
            absolute_path=candidate,
            available=False,
            reason_code="autogluon_runtime_unsafe",
            reason=str(error),
        )
    if stat.S_ISLNK(info.st_mode):
        return ResolvedPythonRuntime(
            runtime_id=runtime_id,
            relative_path=relative,
            absolute_path=candidate,
            available=False,
            reason_code="autogluon_runtime_symlink",
            reason=f"Interpreter path must not be a symlink: {relative}",
        )
    attributes = getattr(info, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if bool(attributes & reparse_flag):
        return ResolvedPythonRuntime(
            runtime_id=runtime_id,
            relative_path=relative,
            absolute_path=candidate,
            available=False,
            reason_code="autogluon_runtime_reparse",
            reason=f"Interpreter path must not be a reparse point: {relative}",
        )
    if not stat.S_ISREG(info.st_mode):
        return ResolvedPythonRuntime(
            runtime_id=runtime_id,
            relative_path=relative,
            absolute_path=candidate,
            available=False,
            reason_code="autogluon_runtime_not_file",
            reason=f"Interpreter path is not a regular file: {relative}",
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        return ResolvedPythonRuntime(
            runtime_id=runtime_id,
            relative_path=relative,
            absolute_path=candidate,
            available=False,
            reason_code="autogluon_runtime_unresolvable",
            reason=f"Interpreter path could not be resolved: {error}",
        )
    if not is_within(resolved, root):
        return ResolvedPythonRuntime(
            runtime_id=runtime_id,
            relative_path=relative,
            absolute_path=resolved,
            available=False,
            reason_code="autogluon_runtime_escape",
            reason="Interpreter path escapes the repository root.",
        )
    if not os.access(resolved, os.X_OK):
        return ResolvedPythonRuntime(
            runtime_id=runtime_id,
            relative_path=relative,
            absolute_path=resolved,
            available=False,
            reason_code="autogluon_runtime_not_executable",
            reason=f"Interpreter is not executable: {relative}",
        )
    return ResolvedPythonRuntime(
        runtime_id=runtime_id,
        relative_path=relative,
        absolute_path=resolved,
        available=True,
    )


def _normalize_relative(relative_path: str) -> str:
    text = str(relative_path).replace("\\", "/").strip()
    if not text or text.startswith("/") or PurePosixPath(text).is_absolute():
        raise PythonRuntimeError(
            f"Absolute interpreter path rejected: {relative_path!r}",
            reason_code="path_traversal",
        )
    parts = PurePosixPath(text).parts
    if ".." in parts or any(part in {"", "."} for part in parts):
        raise PythonRuntimeError(
            f"Path traversal rejected in interpreter path: {relative_path!r}",
            reason_code="path_traversal",
        )
    return PurePosixPath(text).as_posix()


def _reject_symlink_ancestors(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise PythonRuntimeError(
                f"Could not inspect path component: {current}.",
                reason_code="autogluon_runtime_unsafe",
            ) from error
        if stat.S_ISLNK(info.st_mode):
            raise PythonRuntimeError(
                f"Symlink rejected in interpreter path: {current}",
                reason_code="autogluon_runtime_symlink",
            )
        attributes = getattr(info, "st_file_attributes", 0)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if bool(attributes & reparse_flag):
            raise PythonRuntimeError(
                f"Reparse point rejected in interpreter path: {current}",
                reason_code="autogluon_runtime_reparse",
            )


__all__ = [
    "AUTOGLUON_RUNTIME_ID",
    "PythonRuntimeError",
    "ResolvedPythonRuntime",
    "autogluon_interpreter_relative",
    "require_autogluon_python",
    "resolve_autogluon_python",
]
