from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping


class SchemaError(ValueError):
    """Raised when a control-panel registry does not match its strict schema."""


Primitive = str | int | float | bool | None
Confirmation = Literal["none", "confirm", "acknowledge"]
PlaceholderType = Literal["path", "string", "integer", "enum"]
PathRole = Literal["config", "input", "output", "value"]
ENVIRONMENT_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
DOT_PATH_SEGMENT = re.compile(r"^(?:[A-Za-z_][A-Za-z0-9_-]*|0|[1-9][0-9]*)$")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise SchemaError(f"{label} must be a mapping.")
    if not all(isinstance(key, str) for key in value):
        raise SchemaError(f"{label} keys must be strings.")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | frozenset[str] = frozenset(),
    label: str,
) -> None:
    keys = set(value)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise SchemaError(f"{label} is missing keys: {sorted(missing)}.")
    if unknown:
        raise SchemaError(f"{label} has unknown keys: {sorted(unknown)}.")


def _typed(value: Any, expected: type, label: str) -> Any:
    if expected is int and (not isinstance(value, int) or isinstance(value, bool)):
        raise SchemaError(f"{label} must be an integer.")
    if expected is float and (
        not isinstance(value, (int, float)) or isinstance(value, bool)
    ):
        raise SchemaError(f"{label} must be a number.")
    if expected is not int and not isinstance(value, expected):
        raise SchemaError(f"{label} must be {expected.__name__}.")
    return value


def _string_list(
    value: Any, label: str, *, allow_empty: bool = True
) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SchemaError(f"{label} must be a list of strings.")
    if not allow_empty and not value:
        raise SchemaError(f"{label} must not be empty.")
    return tuple(value)


def validate_relative_path_text(value: str, label: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or "\x00" in value
        or "\\" in value
        or value.startswith("/")
        or value.endswith("/")
        or "//" in value
    ):
        raise SchemaError(f"{label} must be a non-empty POSIX-style relative path.")
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or path.as_posix() != value
    ):
        raise SchemaError(f"{label} must be a safe repository-relative path.")
    if ":" in path.parts[0]:
        raise SchemaError(f"{label} must not be drive-qualified.")
    return value


def validate_dot_path(value: str, label: str) -> str:
    if not value or value.startswith(".") or value.endswith("."):
        raise SchemaError(f"{label} must be a non-empty dot path.")
    segments = value.split(".")
    if any(DOT_PATH_SEGMENT.fullmatch(segment) is None for segment in segments):
        raise SchemaError(
            f"{label} must contain only mapping-key or canonical numeric-index "
            "segments separated by single dots."
        )
    return value


def validate_public_cli_path(value: str, label: str) -> str:
    validate_relative_path_text(value, label)
    path = PurePosixPath(value)
    if len(path.parts) != 2 or path.parts[0] != "scripts" or path.suffix != ".py":
        raise SchemaError(f"{label} must have the form scripts/<public-cli>.py.")
    return value


@dataclass(frozen=True)
class UISettings:
    schema_version: int
    working_directory: str
    jobs_root: str
    editable_config_root: str
    poll_interval_seconds: int
    log_tail_lines: int
    allow_process_stop: bool
    allow_config_copy_editing: bool
    mlflow_url: str

    @classmethod
    def from_dict(cls, raw: Any) -> UISettings:
        value = _mapping(raw, "settings")
        required = {
            "schema_version",
            "working_directory",
            "jobs_root",
            "editable_config_root",
            "poll_interval_seconds",
            "log_tail_lines",
            "allow_process_stop",
            "allow_config_copy_editing",
            "mlflow_url",
        }
        _exact_keys(value, required=required, label="settings")
        version = _typed(value["schema_version"], int, "settings.schema_version")
        if version != 1:
            raise SchemaError(f"Unsupported settings schema_version: {version}.")
        working = _typed(value["working_directory"], str, "settings.working_directory")
        if working != ".":
            validate_relative_path_text(working, "settings.working_directory")
        jobs = validate_relative_path_text(
            _typed(value["jobs_root"], str, "settings.jobs_root"),
            "settings.jobs_root",
        )
        editable = validate_relative_path_text(
            _typed(
                value["editable_config_root"],
                str,
                "settings.editable_config_root",
            ),
            "settings.editable_config_root",
        )
        poll = _typed(
            value["poll_interval_seconds"], int, "settings.poll_interval_seconds"
        )
        tail = _typed(value["log_tail_lines"], int, "settings.log_tail_lines")
        if poll < 1 or poll > 60:
            raise SchemaError("settings.poll_interval_seconds must be in [1, 60].")
        if tail < 1 or tail > 5000:
            raise SchemaError("settings.log_tail_lines must be in [1, 5000].")
        url = _typed(value["mlflow_url"], str, "settings.mlflow_url")
        if not (url.startswith("http://") or url.startswith("https://")):
            raise SchemaError("settings.mlflow_url must be an HTTP(S) URL.")
        return cls(
            schema_version=version,
            working_directory=working,
            jobs_root=jobs,
            editable_config_root=editable,
            poll_interval_seconds=poll,
            log_tail_lines=tail,
            allow_process_stop=_typed(
                value["allow_process_stop"], bool, "settings.allow_process_stop"
            ),
            allow_config_copy_editing=_typed(
                value["allow_config_copy_editing"],
                bool,
                "settings.allow_config_copy_editing",
            ),
            mlflow_url=url,
        )


@dataclass(frozen=True)
class PlaceholderSpec:
    type: PlaceholderType
    role: PathRole
    required: bool
    roots: tuple[str, ...]
    choices: tuple[str, ...]
    sensitive: bool
    must_exist: bool

    @classmethod
    def from_dict(cls, raw: Any, label: str) -> PlaceholderSpec:
        value = _mapping(raw, label)
        _exact_keys(
            value,
            required={"type", "role", "required"},
            optional={"roots", "choices", "sensitive", "must_exist"},
            label=label,
        )
        kind = _typed(value["type"], str, f"{label}.type")
        role = _typed(value["role"], str, f"{label}.role")
        if kind not in {"path", "string", "integer", "enum"}:
            raise SchemaError(f"{label}.type is unsupported: {kind}.")
        if role not in {"config", "input", "output", "value"}:
            raise SchemaError(f"{label}.role is unsupported: {role}.")
        roots = _string_list(value.get("roots", []), f"{label}.roots")
        for index, root in enumerate(roots):
            validate_relative_path_text(root, f"{label}.roots[{index}]")
        choices = _string_list(value.get("choices", []), f"{label}.choices")
        if kind == "path" and not roots:
            raise SchemaError(f"{label}.roots is required for path placeholders.")
        if kind == "enum" and not choices:
            raise SchemaError(f"{label}.choices is required for enum placeholders.")
        if kind != "path" and roots:
            raise SchemaError(f"{label}.roots is only valid for path placeholders.")
        if kind != "enum" and choices:
            raise SchemaError(f"{label}.choices is only valid for enum placeholders.")
        return cls(
            type=kind,  # type: ignore[arg-type]
            role=role,  # type: ignore[arg-type]
            required=_typed(value["required"], bool, f"{label}.required"),
            roots=roots,
            choices=choices,
            sensitive=_typed(value.get("sensitive", False), bool, f"{label}.sensitive"),
            must_exist=_typed(
                value.get("must_exist", role != "output"),
                bool,
                f"{label}.must_exist",
            ),
        )


@dataclass(frozen=True)
class EnvironmentVariableSpec:
    name: str
    required: bool
    sensitive: bool

    @classmethod
    def from_dict(cls, raw: Any, label: str) -> EnvironmentVariableSpec:
        value = _mapping(raw, label)
        _exact_keys(
            value,
            required={"name", "required", "sensitive"},
            label=label,
        )
        name = _typed(value["name"], str, f"{label}.name")
        if ENVIRONMENT_NAME.fullmatch(name) is None:
            raise SchemaError(f"{label}.name must match {ENVIRONMENT_NAME.pattern}.")
        return cls(
            name=name,
            required=_typed(value["required"], bool, f"{label}.required"),
            sensitive=_typed(value["sensitive"], bool, f"{label}.sensitive"),
        )


@dataclass(frozen=True)
class ActionSpec:
    id: str
    title: str
    description: str
    argv: tuple[str, ...]
    placeholders: Mapping[str, PlaceholderSpec]
    confirmation: Confirmation
    enabled: bool
    competition_test: bool
    success_markers: tuple[str, ...]
    failure_markers: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: Any, label: str) -> ActionSpec:
        value = _mapping(raw, label)
        _exact_keys(
            value,
            required={
                "id",
                "title",
                "description",
                "argv",
                "placeholders",
                "confirmation",
                "enabled",
                "competition_test",
                "success_markers",
                "failure_markers",
            },
            label=label,
        )
        action_id = _typed(value["id"], str, f"{label}.id")
        confirmation = _typed(value["confirmation"], str, f"{label}.confirmation")
        if confirmation not in {"none", "confirm", "acknowledge"}:
            raise SchemaError(f"{label}.confirmation is unsupported.")
        placeholders_raw = _mapping(value["placeholders"], f"{label}.placeholders")
        placeholders = {
            name: PlaceholderSpec.from_dict(spec, f"{label}.placeholders.{name}")
            for name, spec in placeholders_raw.items()
        }
        argv = _string_list(value["argv"], f"{label}.argv")
        referenced: set[str] = set()
        for token in argv:
            if token.startswith("{") and token.endswith("}") and token.count("{") == 1:
                referenced.add(token[1:-1])
            elif "{" in token or "}" in token:
                raise SchemaError(
                    f"{label}.argv placeholders must occupy a complete argv token."
                )
        if referenced != set(placeholders):
            raise SchemaError(
                f"{label}.placeholders must exactly match argv placeholders: "
                f"{sorted(referenced)}."
            )
        return cls(
            id=action_id,
            title=_typed(value["title"], str, f"{label}.title"),
            description=_typed(value["description"], str, f"{label}.description"),
            argv=argv,
            placeholders=placeholders,
            confirmation=confirmation,  # type: ignore[arg-type]
            enabled=_typed(value["enabled"], bool, f"{label}.enabled"),
            competition_test=_typed(
                value["competition_test"], bool, f"{label}.competition_test"
            ),
            success_markers=_string_list(
                value["success_markers"], f"{label}.success_markers"
            ),
            failure_markers=_string_list(
                value["failure_markers"], f"{label}.failure_markers"
            ),
        )


@dataclass(frozen=True)
class CommandSpec:
    id: str
    title: str
    description: str
    category: str
    argv_prefix: tuple[str, ...]
    actions: Mapping[str, ActionSpec]
    allowed_config_globs: tuple[str, ...]
    allowed_input_roots: tuple[str, ...]
    allowed_output_roots: tuple[str, ...]
    result_reader_id: str | None
    artifact_url: str | None
    public_cli: str
    environment: Mapping[str, EnvironmentVariableSpec]

    @classmethod
    def from_dict(
        cls,
        raw: Any,
        label: str,
        *,
        approved_public_clis: frozenset[str],
    ) -> CommandSpec:
        value = _mapping(raw, label)
        _exact_keys(
            value,
            required={
                "id",
                "title",
                "description",
                "category",
                "argv_prefix",
                "actions",
                "allowed_config_globs",
                "allowed_input_roots",
                "allowed_output_roots",
                "result_reader_id",
                "artifact_url",
                "environment",
            },
            label=label,
        )
        prefix = _string_list(
            value["argv_prefix"], f"{label}.argv_prefix", allow_empty=False
        )
        if len(prefix) not in {2, 3} or prefix[0] != "$PYTHON":
            raise SchemaError(
                f"{label}.argv_prefix must be $PYTHON, optional -u, and one "
                "approved scripts/<public-cli>.py path."
            )
        if len(prefix) == 3 and prefix[1] != "-u":
            raise SchemaError(f"{label}.argv_prefix permits only the -u flag.")
        public_cli = prefix[-1]
        validate_public_cli_path(public_cli, f"{label}.argv_prefix script")
        if public_cli not in approved_public_clis:
            raise SchemaError(
                f"{label}.argv_prefix script is not an approved public CLI: "
                f"{public_cli}."
            )
        actions_raw = value["actions"]
        if not isinstance(actions_raw, list) or not actions_raw:
            raise SchemaError(f"{label}.actions must be a non-empty list.")
        actions_list = [
            ActionSpec.from_dict(item, f"{label}.actions[{index}]")
            for index, item in enumerate(actions_raw)
        ]
        actions = _unique_by_id(actions_list, f"{label}.actions")
        inputs = _string_list(
            value["allowed_input_roots"], f"{label}.allowed_input_roots"
        )
        outputs = _string_list(
            value["allowed_output_roots"], f"{label}.allowed_output_roots"
        )
        for name, roots in (
            ("allowed_input_roots", inputs),
            ("allowed_output_roots", outputs),
        ):
            for index, root in enumerate(roots):
                validate_relative_path_text(root, f"{label}.{name}[{index}]")
        reader = value["result_reader_id"]
        if reader is not None and not isinstance(reader, str):
            raise SchemaError(f"{label}.result_reader_id must be a string or null.")
        artifact_url = value["artifact_url"]
        if artifact_url is not None and not isinstance(artifact_url, str):
            raise SchemaError(f"{label}.artifact_url must be a string or null.")
        environment_raw = value["environment"]
        if not isinstance(environment_raw, list):
            raise SchemaError(f"{label}.environment must be a list.")
        environment_items = [
            EnvironmentVariableSpec.from_dict(item, f"{label}.environment[{index}]")
            for index, item in enumerate(environment_raw)
        ]
        environment = _unique_by_name(environment_items, f"{label}.environment")
        return cls(
            id=_typed(value["id"], str, f"{label}.id"),
            title=_typed(value["title"], str, f"{label}.title"),
            description=_typed(value["description"], str, f"{label}.description"),
            category=_typed(value["category"], str, f"{label}.category"),
            argv_prefix=prefix,
            actions=actions,
            allowed_config_globs=_string_list(
                value["allowed_config_globs"], f"{label}.allowed_config_globs"
            ),
            allowed_input_roots=inputs,
            allowed_output_roots=outputs,
            result_reader_id=reader,
            artifact_url=artifact_url,
            public_cli=public_cli,
            environment=environment,
        )


@dataclass(frozen=True)
class SummaryFileSpec:
    path: str
    fields: Mapping[str, str]

    @classmethod
    def from_dict(cls, raw: Any, label: str) -> SummaryFileSpec:
        value = _mapping(raw, label)
        _exact_keys(value, required={"path", "fields"}, label=label)
        path = validate_relative_path_text(
            _typed(value["path"], str, f"{label}.path"), f"{label}.path"
        )
        fields_raw = _mapping(value["fields"], f"{label}.fields")
        fields: dict[str, str] = {}
        for field_label, dot_path in fields_raw.items():
            if not isinstance(dot_path, str):
                raise SchemaError(f"{label}.fields values must be non-empty strings.")
            fields[field_label] = validate_dot_path(
                dot_path, f"{label}.fields.{field_label}"
            )
        return cls(path=path, fields=fields)


@dataclass(frozen=True)
class ReaderSpec:
    id: str
    title: str
    artifact_roots: tuple[str, ...]
    discovery_glob: str
    success_markers: tuple[str, ...]
    failure_markers: tuple[str, ...]
    summary_files: tuple[SummaryFileSpec, ...]
    csv_previews: tuple[str, ...]
    log_files: tuple[str, ...]
    compare_fields: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: Any, label: str) -> ReaderSpec:
        value = _mapping(raw, label)
        _exact_keys(
            value,
            required={
                "id",
                "title",
                "artifact_roots",
                "discovery_glob",
                "success_markers",
                "failure_markers",
                "summary_files",
                "csv_previews",
                "log_files",
                "compare_fields",
            },
            label=label,
        )
        roots = _string_list(value["artifact_roots"], f"{label}.artifact_roots")
        for index, root in enumerate(roots):
            validate_relative_path_text(root, f"{label}.artifact_roots[{index}]")
        glob = _typed(value["discovery_glob"], str, f"{label}.discovery_glob")
        if Path(glob).is_absolute() or ".." in PurePosixPath(glob).parts:
            raise SchemaError(f"{label}.discovery_glob is unsafe.")
        summaries_raw = value["summary_files"]
        if not isinstance(summaries_raw, list):
            raise SchemaError(f"{label}.summary_files must be a list.")
        summaries = tuple(
            SummaryFileSpec.from_dict(item, f"{label}.summary_files[{index}]")
            for index, item in enumerate(summaries_raw)
        )
        success_markers = _safe_relative_list(
            value["success_markers"], f"{label}.success_markers"
        )
        failure_markers = _safe_relative_list(
            value["failure_markers"], f"{label}.failure_markers"
        )
        return cls(
            id=_typed(value["id"], str, f"{label}.id"),
            title=_typed(value["title"], str, f"{label}.title"),
            artifact_roots=roots,
            discovery_glob=glob,
            success_markers=success_markers,
            failure_markers=failure_markers,
            summary_files=summaries,
            csv_previews=_safe_relative_list(
                value["csv_previews"], f"{label}.csv_previews"
            ),
            log_files=_safe_relative_list(value["log_files"], f"{label}.log_files"),
            compare_fields=_string_list(
                value["compare_fields"], f"{label}.compare_fields"
            ),
        )


def parse_commands(raw: Any) -> Mapping[str, CommandSpec]:
    value = _mapping(raw, "command registry")
    _exact_keys(
        value,
        required={"schema_version", "approved_public_clis", "commands"},
        label="command registry",
    )
    version = _typed(value["schema_version"], int, "command registry.schema_version")
    if version != 1:
        raise SchemaError(f"Unsupported command registry schema_version: {version}.")
    commands_raw = value["commands"]
    if not isinstance(commands_raw, list) or not commands_raw:
        raise SchemaError("command registry.commands must be a non-empty list.")
    approved_values = _string_list(
        value["approved_public_clis"],
        "command registry.approved_public_clis",
        allow_empty=False,
    )
    approved_public_clis: set[str] = set()
    for index, public_cli in enumerate(approved_values):
        validate_public_cli_path(
            public_cli, f"command registry.approved_public_clis[{index}]"
        )
        if public_cli in approved_public_clis:
            raise SchemaError(
                "command registry.approved_public_clis contains duplicate path: "
                f"{public_cli}."
            )
        approved_public_clis.add(public_cli)
    return _unique_by_id(
        [
            CommandSpec.from_dict(
                item,
                f"command registry.commands[{index}]",
                approved_public_clis=frozenset(approved_public_clis),
            )
            for index, item in enumerate(commands_raw)
        ],
        "command registry.commands",
    )


def parse_readers(raw: Any) -> Mapping[str, ReaderSpec]:
    value = _mapping(raw, "reader registry")
    _exact_keys(value, required={"schema_version", "readers"}, label="reader registry")
    version = _typed(value["schema_version"], int, "reader registry.schema_version")
    if version != 1:
        raise SchemaError(f"Unsupported reader registry schema_version: {version}.")
    readers_raw = value["readers"]
    if not isinstance(readers_raw, list) or not readers_raw:
        raise SchemaError("reader registry.readers must be a non-empty list.")
    return _unique_by_id(
        [
            ReaderSpec.from_dict(item, f"reader registry.readers[{index}]")
            for index, item in enumerate(readers_raw)
        ],
        "reader registry.readers",
    )


def _unique_by_id(items: list[Any], label: str) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        if item.id in result:
            raise SchemaError(f"{label} contains duplicate id: {item.id}.")
        result[item.id] = item
    return result


def _unique_by_name(items: list[Any], label: str) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        if item.name in result:
            raise SchemaError(f"{label} contains duplicate name: {item.name}.")
        result[item.name] = item
    return result


def _safe_relative_list(value: Any, label: str) -> tuple[str, ...]:
    values = _string_list(value, label)
    for index, item in enumerate(values):
        validate_relative_path_text(item, f"{label}[{index}]")
    return values
