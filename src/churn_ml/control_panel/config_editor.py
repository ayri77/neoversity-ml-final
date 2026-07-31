from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from src.churn_ml.control_panel.command_builder import (
    CommandBuildError,
    resolve_safe_path,
)
from src.churn_ml.control_panel.path_safety import (
    PathSafetyError,
    require_safe_directory,
    require_safe_existing_ancestors,
)


class ConfigEditError(ValueError):
    """Raised when a configuration copy cannot be validated or saved safely."""


SAFE_COPY_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
CONFIG_SUFFIXES = {".json", ".yaml", ".yml"}


def parse_config_text(text: str, suffix: str) -> dict[str, Any]:
    normalized = suffix.lower()
    if normalized not in CONFIG_SUFFIXES:
        raise ConfigEditError("Only JSON and YAML configuration files are supported.")
    try:
        payload = json.loads(text) if normalized == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as error:
        raise ConfigEditError(f"Configuration syntax is invalid: {error}") from error
    if not isinstance(payload, dict):
        raise ConfigEditError("Configuration must contain a top-level mapping.")
    _validate_primitive_tree(payload, "configuration")
    return payload


def read_config(
    repository_root: Path,
    source: str | Path,
    *,
    allowed_roots: tuple[str, ...],
    max_bytes: int = 1_000_000,
) -> tuple[str, Path]:
    try:
        _, canonical = resolve_safe_path(
            repository_root,
            source,
            allowed_roots=allowed_roots,
            must_exist=True,
        )
    except CommandBuildError as error:
        raise ConfigEditError(str(error)) from error
    if canonical.suffix.lower() not in CONFIG_SUFFIXES or not canonical.is_file():
        raise ConfigEditError("Selected configuration must be a JSON or YAML file.")
    if canonical.stat().st_size > max_bytes:
        raise ConfigEditError(f"Configuration exceeds {max_bytes} bytes.")
    try:
        text = canonical.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ConfigEditError(f"Could not read configuration: {error}") from error
    parse_config_text(text, canonical.suffix)
    return text, canonical


def save_config_copy(
    repository_root: Path,
    *,
    editable_root: str,
    copy_name: str,
    text: str,
) -> Path:
    return publish_editable_text_file(
        repository_root,
        editable_root=editable_root,
        relative_path=copy_name,
        text=text,
        allow_identical_reuse=False,
    )


@dataclass(frozen=True)
class PublishedEditableFile:
    """Result of an exclusive editable-root publication."""

    path: Path
    reused: bool


def publish_editable_text_file(
    repository_root: Path,
    *,
    editable_root: str,
    relative_path: str,
    text: str,
    allow_identical_reuse: bool = False,
) -> Path | PublishedEditableFile:
    """Publish YAML/JSON text under the editable root with create-if-absent safety.

    ``relative_path`` may be a bare filename or one safe subdirectory such as
    ``plans/example.yaml``. Existing targets are never overwritten. When
    ``allow_identical_reuse`` is true, an existing file with identical parsed
    content is returned instead of raising.
    """
    root = repository_root.resolve(strict=True)
    relative = _validate_editable_relative_path(relative_path)
    parse_config_text(text, Path(relative).suffix)
    try:
        _, editable_directory = resolve_safe_path(
            root,
            editable_root,
            allowed_roots=(editable_root,),
            must_exist=False,
        )
    except CommandBuildError as error:
        raise ConfigEditError(str(error)) from error
    target = editable_directory.joinpath(*PurePosixPath(relative).parts)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        require_safe_directory(editable_directory)
        require_safe_directory(target.parent)
    except (OSError, PathSafetyError) as error:
        raise ConfigEditError("Editable configuration directory is unsafe.") from error

    desired = text if text.endswith("\n") else f"{text}\n"
    if target.exists():
        try:
            existing = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise ConfigEditError(
                f"Could not read existing editable file: {relative}."
            ) from error
        if allow_identical_reuse and _parsed_config_equal(
            existing, desired, Path(relative).suffix
        ):
            return PublishedEditableFile(path=target, reused=True)
        raise ConfigEditError(
            f"Configuration copy already exists with different content: {relative}."
            if allow_identical_reuse
            else f"Configuration copy already exists: {Path(relative).name}."
        )

    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.chmod(temporary, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(desired)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_new_file(Path(temporary), target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    if allow_identical_reuse:
        return PublishedEditableFile(path=target, reused=False)
    return target


def _publish_new_file(temporary: Path, target: Path) -> None:
    """Atomically publish a validated copy only when the target is absent."""
    claimed = False
    try:
        require_safe_existing_ancestors(target)
        require_safe_directory(target.parent)
        if os.name == "nt":
            # Exclusive create claims the name; replace then publishes content.
            # Concurrent writers: only one O_EXCL claim succeeds.
            claim = os.open(
                target, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_BINARY, 0o600
            )
            os.close(claim)
            claimed = True
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
            temporary.unlink()
    except FileExistsError as error:
        raise ConfigEditError(
            f"Configuration copy already exists: {target.name}."
        ) from error
    except PathSafetyError as error:
        raise ConfigEditError("Editable configuration directory is unsafe.") from error
    except BaseException:
        if claimed:
            try:
                target.unlink()
            except OSError:
                pass
        raise


def _validate_primitive_tree(value: Any, path: str) -> None:
    if value is None or type(value) in {str, int, float, bool}:
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_primitive_tree(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConfigEditError(f"{path} contains a non-string mapping key.")
            _validate_primitive_tree(item, f"{path}.{key}")
        return
    raise ConfigEditError(f"{path} contains unsupported type {type(value).__name__}.")


def _validate_editable_relative_path(relative_path: str) -> str:
    posix = PurePosixPath(relative_path.replace("\\", "/"))
    if (
        not relative_path
        or posix.is_absolute()
        or ".." in posix.parts
        or any(part in {"", ".", ".."} for part in posix.parts)
        or len(posix.parts) > 2
    ):
        raise ConfigEditError(
            "Editable relative path must be a safe filename or one safe subdirectory."
        )
    for part in posix.parts:
        if SAFE_COPY_NAME.fullmatch(part) is None:
            raise ConfigEditError(
                "Editable relative path contains an unsafe path segment."
            )
    if posix.name != posix.parts[-1] or Path(posix.name).suffix.lower() not in CONFIG_SUFFIXES:
        raise ConfigEditError(
            "Editable file must end in .json, .yaml, or .yml."
        )
    return posix.as_posix()


def _parsed_config_equal(left_text: str, right_text: str, suffix: str) -> bool:
    try:
        return parse_config_text(left_text, suffix) == parse_config_text(
            right_text, suffix
        )
    except ConfigEditError:
        return False
