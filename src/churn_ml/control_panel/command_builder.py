from __future__ import annotations

import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.churn_ml.control_panel.schemas import ActionSpec, CommandSpec, PlaceholderSpec


class CommandBuildError(ValueError):
    """Raised when a requested allowlisted command cannot be rendered safely."""


SAFE_STRING = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class BuiltCommand:
    command_id: str
    action_id: str
    argv: tuple[str, ...]
    redacted_argv: tuple[str, ...]
    references: Mapping[str, str | int]
    input_paths: tuple[Path, ...]
    output_paths: tuple[Path, ...]


def build_command(
    commands: Mapping[str, CommandSpec],
    command_id: str,
    action_id: str,
    values: Mapping[str, Any],
    *,
    repository_root: Path,
    python_executable: str | None = None,
) -> BuiltCommand:
    try:
        command = commands[command_id]
    except KeyError as error:
        raise CommandBuildError(f"Unknown command id: {command_id}.") from error
    try:
        action = command.actions[action_id]
    except KeyError as error:
        raise CommandBuildError(
            f"Unknown action id {action_id!r} for command {command_id!r}."
        ) from error
    if not action.enabled:
        raise CommandBuildError(
            f"Action {command_id}/{action_id} is disabled by the registry."
        )
    expected = set(action.placeholders)
    unknown = set(values) - expected
    if unknown:
        raise CommandBuildError(f"Unknown placeholder values: {sorted(unknown)}.")

    root = repository_root.resolve(strict=True)
    rendered: dict[str, str] = {}
    redacted: dict[str, str] = {}
    references: dict[str, str | int] = {}
    inputs: list[Path] = []
    outputs: list[Path] = []
    for name, spec in action.placeholders.items():
        value = values.get(name)
        if value in (None, ""):
            if spec.required:
                raise CommandBuildError(f"Missing required placeholder: {name}.")
            continue
        text, reference, canonical = _render_value(root, name, value, spec)
        if spec.role == "config" and not any(
            Path(text).match(pattern) for pattern in command.allowed_config_globs
        ):
            raise CommandBuildError(
                f"Configuration path is outside allowed globs: {text}."
            )
        rendered[name] = text
        redacted[name] = "<redacted>" if spec.sensitive else text
        references[name] = "<redacted>" if spec.sensitive else reference
        if canonical is not None:
            if spec.role == "output":
                outputs.append(canonical)
            else:
                inputs.append(canonical)

    _reject_path_overlap(inputs, outputs)
    prefix = tuple(
        (python_executable or sys.executable) if token == "$PYTHON" else token
        for token in command.argv_prefix
    )
    argv = prefix + _substitute(action, rendered)
    redacted_argv = prefix + _substitute(action, redacted)
    return BuiltCommand(
        command_id=command_id,
        action_id=action_id,
        argv=argv,
        redacted_argv=redacted_argv,
        references=references,
        input_paths=tuple(inputs),
        output_paths=tuple(outputs),
    )


def display_argv(argv: tuple[str, ...] | list[str]) -> str:
    """Render argv for human review without producing shell-executable text."""
    return "argv = " + repr(list(argv))


def resolve_safe_path(
    repository_root: Path,
    raw: str | Path,
    *,
    allowed_roots: tuple[str, ...],
    must_exist: bool,
) -> tuple[str, Path]:
    raw_path = Path(raw)
    if raw_path.is_absolute() or raw_path.drive or raw_path.anchor:
        raise CommandBuildError("Absolute or drive-qualified paths are not allowed.")
    if any(part in {"", ".", ".."} for part in raw_path.parts):
        raise CommandBuildError("Dot segments and path traversal are not allowed.")
    root = repository_root.resolve(strict=True)
    requested = root.joinpath(*raw_path.parts)
    _reject_linked_components(root, requested, include_leaf=True)
    try:
        canonical = requested.resolve(strict=must_exist)
    except OSError as error:
        raise CommandBuildError(f"Path cannot be resolved: {raw}.") from error
    if not _is_within(canonical, root):
        raise CommandBuildError("Path escapes the repository root.")
    allowed = [
        root.joinpath(*Path(item).parts).resolve(strict=False) for item in allowed_roots
    ]
    if not any(_is_within(canonical, allowed_root) for allowed_root in allowed):
        raise CommandBuildError(
            f"Path is outside declared roots: {list(allowed_roots)}."
        )
    if must_exist and not canonical.exists():
        raise CommandBuildError(f"Required path does not exist: {raw}.")
    relative = canonical.relative_to(root).as_posix()
    return relative, canonical


def _render_value(
    root: Path,
    name: str,
    value: Any,
    spec: PlaceholderSpec,
) -> tuple[str, str | int, Path | None]:
    if spec.type == "path":
        if not isinstance(value, (str, Path)):
            raise CommandBuildError(f"{name} must be a path string.")
        relative, canonical = resolve_safe_path(
            root,
            value,
            allowed_roots=spec.roots,
            must_exist=spec.must_exist,
        )
        if spec.must_exist and spec.role in {"config", "input"}:
            if not (canonical.is_file() or canonical.is_dir()):
                raise CommandBuildError(f"{name} is not a regular file or directory.")
        return relative, relative, canonical
    if spec.type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise CommandBuildError(f"{name} must be an integer.")
        return str(value), value, None
    if not isinstance(value, str):
        raise CommandBuildError(f"{name} must be a string.")
    if spec.type == "enum":
        if value not in spec.choices:
            raise CommandBuildError(f"{name} must be one of {list(spec.choices)}.")
    elif SAFE_STRING.fullmatch(value) is None:
        raise CommandBuildError(
            f"{name} must use the safe identifier format {SAFE_STRING.pattern}."
        )
    return value, value, None


def _substitute(action: ActionSpec, values: Mapping[str, str]) -> tuple[str, ...]:
    result: list[str] = []
    for token in action.argv:
        if token.startswith("{") and token.endswith("}"):
            name = token[1:-1]
            if name not in values:
                continue
            result.append(values[name])
        else:
            result.append(token)
    return tuple(result)


def _reject_path_overlap(inputs: list[Path], outputs: list[Path]) -> None:
    for input_path in inputs:
        for output_path in outputs:
            if _is_within(input_path, output_path) or _is_within(
                output_path, input_path
            ):
                raise CommandBuildError(
                    "Input and output paths must not overlap or contain one another."
                )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _reject_linked_components(
    root: Path, requested: Path, *, include_leaf: bool
) -> None:
    relative = requested.relative_to(root)
    current = root
    parts = relative.parts if include_leaf else relative.parts[:-1]
    for part in parts:
        current = current / part
        if not current.exists():
            continue
        try:
            info = current.lstat()
        except OSError as error:
            raise CommandBuildError(
                f"Could not inspect path component: {current}."
            ) from error
        attributes = getattr(info, "st_file_attributes", 0)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if current.is_symlink() or bool(attributes & reparse_flag):
            raise CommandBuildError(
                f"Symlink, junction, or reparse paths are not allowed: {current}."
            )
        if os.name != "nt" and stat.S_ISLNK(info.st_mode):
            raise CommandBuildError(f"Symbolic links are not allowed: {current}.")
