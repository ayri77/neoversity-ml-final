from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import AbstractSet, Any, Mapping, MutableSet

from src.churn_ml.control_panel.command_builder import BuiltCommand, build_command
from src.churn_ml.control_panel.process import argv_fingerprint
from src.churn_ml.control_panel.schemas import CommandSpec


class LaunchAuthorizationError(ValueError):
    """Raised when current authoritative state does not authorize a launch."""


@dataclass(frozen=True)
class RenderedLaunch:
    command_id: str
    action_id: str
    argv_sha256: str
    nonce: str


def rendered_launch(
    built: BuiltCommand,
    previous: RenderedLaunch | None,
    *,
    consumed_nonces: AbstractSet[str] | None = None,
) -> RenderedLaunch:
    fingerprint = argv_fingerprint(built.argv)
    if (
        previous is not None
        and previous.command_id == built.command_id
        and previous.action_id == built.action_id
        and previous.argv_sha256 == fingerprint
        and (consumed_nonces is None or previous.nonce not in consumed_nonces)
    ):
        return previous
    return RenderedLaunch(
        command_id=built.command_id,
        action_id=built.action_id,
        argv_sha256=fingerprint,
        nonce=str(uuid.uuid4()),
    )


def authorize_launch(
    commands: Mapping[str, CommandSpec],
    *,
    command_id: str,
    action_id: str,
    values: Mapping[str, Any],
    repository_root: Path,
    confirmed: bool,
    high_risk_acknowledged: bool,
    rendered: RenderedLaunch | None,
    consumed_nonces: MutableSet[str],
    python_executable: str | None = None,
) -> BuiltCommand:
    """Revalidate and consume one launch authorization immediately before spawn."""
    command = commands.get(command_id)
    if command is None:
        raise LaunchAuthorizationError("The selected command no longer exists.")
    action = command.actions.get(action_id)
    if action is None:
        raise LaunchAuthorizationError("The selected action no longer exists.")
    if not action.enabled:
        raise LaunchAuthorizationError("The selected action is disabled.")
    if action.confirmation in {"confirm", "acknowledge"} and not confirmed:
        raise LaunchAuthorizationError("Required launch confirmation is missing.")
    if action.competition_test and not high_risk_acknowledged:
        raise LaunchAuthorizationError("Required high-risk acknowledgement is missing.")
    try:
        built = build_command(
            commands,
            command_id,
            action_id,
            values,
            repository_root=repository_root,
            python_executable=python_executable,
        )
    except ValueError as error:
        raise LaunchAuthorizationError(f"Command build failed: {error}") from error
    if rendered is None:
        raise LaunchAuthorizationError("No matching rendered launch is available.")
    if (
        rendered.command_id != command_id
        or rendered.action_id != action_id
        or rendered.argv_sha256 != argv_fingerprint(built.argv)
    ):
        raise LaunchAuthorizationError(
            "The rendered command is stale; review the current argv again."
        )
    try:
        uuid.UUID(rendered.nonce)
    except ValueError as error:
        raise LaunchAuthorizationError("The launch nonce is invalid.") from error
    if rendered.nonce in consumed_nonces:
        raise LaunchAuthorizationError("This launch event was already consumed.")
    consumed_nonces.add(rendered.nonce)
    return built
