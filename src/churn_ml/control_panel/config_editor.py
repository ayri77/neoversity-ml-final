from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import yaml

from src.churn_ml.control_panel.command_builder import (
    CommandBuildError,
    resolve_safe_path,
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
    root = repository_root.resolve(strict=True)
    name = Path(copy_name)
    if (
        name.name != copy_name
        or SAFE_COPY_NAME.fullmatch(copy_name) is None
        or name.suffix.lower() not in CONFIG_SUFFIXES
    ):
        raise ConfigEditError(
            "Copy name must be a safe filename ending in .json, .yaml, or .yml."
        )
    parse_config_text(text, name.suffix)
    try:
        _, directory = resolve_safe_path(
            root,
            editable_root,
            allowed_roots=(editable_root,),
            must_exist=False,
        )
    except CommandBuildError as error:
        raise ConfigEditError(str(error)) from error
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name.name
    if target.exists():
        raise ConfigEditError(f"Configuration copy already exists: {target.name}.")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            if text and not text.endswith("\n"):
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return target


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
