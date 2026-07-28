from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from src.churn_ml.control_panel.command_builder import (
    CommandBuildError,
    resolve_safe_path,
)
from src.churn_ml.control_panel.path_safety import (
    PathSafetyError,
    require_regular_file,
)

from src.churn_ml.control_panel.schemas import (
    CommandSpec,
    ReaderSpec,
    SchemaError,
    UISettings,
    parse_commands,
    parse_readers,
)


@dataclass(frozen=True)
class RegistrySources:
    settings: Path
    commands: Path
    readers: Path


@dataclass(frozen=True)
class ControlPanelRegistry:
    repository_root: Path
    settings: UISettings
    commands: dict[str, CommandSpec]
    readers: dict[str, ReaderSpec]
    sources: RegistrySources


def load_registry(
    repository_root: Path,
    *,
    settings_path: Path | None = None,
    commands_path: Path | None = None,
    readers_path: Path | None = None,
) -> ControlPanelRegistry:
    root = repository_root.resolve(strict=True)
    sources = RegistrySources(
        settings=settings_path or root / "configs/ui/ui_settings.json",
        commands=commands_path or root / "configs/ui/ui_commands.yaml",
        readers=readers_path or root / "configs/ui/ui_readers.yaml",
    )
    settings = UISettings.from_dict(_load_json(sources.settings))
    _validate_settings_paths(root, settings)
    commands = dict(parse_commands(_load_yaml(sources.commands)))
    readers = dict(parse_readers(_load_yaml(sources.readers)))
    for public_cli in sorted({command.public_cli for command in commands.values()}):
        try:
            _, cli_path = resolve_safe_path(
                root,
                public_cli,
                allowed_roots=("scripts",),
                must_exist=True,
            )
            require_regular_file(cli_path, reject_hardlinks=True)
        except (CommandBuildError, PathSafetyError) as error:
            raise SchemaError(
                f"Invalid approved public CLI {public_cli}: {error}"
            ) from error
    unknown_readers = sorted(
        {
            command.result_reader_id
            for command in commands.values()
            if command.result_reader_id is not None
            and command.result_reader_id not in readers
        }
    )
    if unknown_readers:
        raise SchemaError(
            f"Commands reference unknown result readers: {unknown_readers}."
        )
    for command in commands.values():
        input_roots = set(command.allowed_input_roots)
        output_roots = set(command.allowed_output_roots)
        for action in command.actions.values():
            for name, placeholder in action.placeholders.items():
                if placeholder.type != "path":
                    continue
                declared = output_roots if placeholder.role == "output" else input_roots
                unknown_roots = set(placeholder.roots) - declared
                if unknown_roots:
                    raise SchemaError(
                        f"{command.id}/{action.id}/{name} uses undeclared roots: "
                        f"{sorted(unknown_roots)}."
                    )
    return ControlPanelRegistry(
        repository_root=root,
        settings=settings,
        commands=commands,
        readers=readers,
        sources=sources,
    )


def _validate_settings_paths(root: Path, settings: UISettings) -> None:
    if settings.working_directory != ".":
        try:
            _, working = resolve_safe_path(
                root,
                settings.working_directory,
                allowed_roots=(settings.working_directory,),
                must_exist=True,
            )
        except CommandBuildError as error:
            raise SchemaError(f"Invalid settings working directory: {error}") from error
        if not working.is_dir():
            raise SchemaError("settings.working_directory must be a directory.")
    for value in (settings.jobs_root, settings.editable_config_root):
        try:
            resolve_safe_path(root, value, allowed_roots=(value,), must_exist=False)
        except CommandBuildError as error:
            raise SchemaError(f"Invalid settings path {value}: {error}") from error


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise SchemaError(f"Could not load JSON registry {path}: {error}") from error


def _load_yaml(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as error:
        raise SchemaError(f"Could not load YAML registry {path}: {error}") from error
