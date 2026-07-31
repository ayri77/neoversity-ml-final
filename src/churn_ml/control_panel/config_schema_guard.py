"""Configuration-type validation before a Control Panel launch.

Several operations historically shared one untyped YAML selector rooted at
``artifacts/ui_configs``. That allowed an Experiment Core training configuration
to be offered to Final Deployment even though the two schemas are unrelated.

Classification uses document content and, for deployment, the authoritative
production parser. It never relies on a filename or a directory name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

from src.churn_ml.control_panel.path_safety import (
    PathSafetyError,
    path_exists_nonfollowing,
    require_regular_file,
)
from src.churn_ml.deployment_v1_contracts import (
    CONFIG_KEYS as DEPLOYMENT_CONFIG_KEYS,
    DeploymentContractError,
    load_deployment_config,
)


DEPLOYMENT_KIND = "deployment_v1"
EXPERIMENT_CORE_KIND = "experiment_core_v2"
BLEND_KIND = "blend_evaluation_v1"
OPTUNA_KIND = "optuna_search_v1"
MLFLOW_KIND = "mlflow_local_index"
UNKNOWN_KIND = "unknown"

EXPERIMENT_CORE_MESSAGE = (
    "This file is an Experiment Core training configuration.\n"
    "Generate a deployment draft from a completed run instead."
)
DEPLOYMENT_MESSAGE = (
    "This file is a Final Deployment configuration.\n"
    "Select an Experiment Core training configuration instead."
)

_EXPECTED_KIND_BY_COMMAND = {
    "final_deployment_v1": DEPLOYMENT_KIND,
    "experiment_core_v2": EXPERIMENT_CORE_KIND,
}
_MAX_CONFIG_BYTES = 2_000_000


@dataclass(frozen=True)
class ConfigGuardResult:
    """Outcome of configuration-type validation for one command."""

    ok: bool
    kind: str
    expected_kind: str | None
    message: str | None = None


def classify_config_document(payload: Any) -> str:
    """Classify a parsed configuration document by its schema content."""
    if not isinstance(payload, Mapping):
        return UNKNOWN_KIND
    keys = set(payload)
    if DEPLOYMENT_CONFIG_KEYS.issubset(keys):
        return DEPLOYMENT_KIND
    if {"deployment_id", "components", "test_data", "sample_submission"}.issubset(keys):
        return DEPLOYMENT_KIND
    if "evaluation_plan_path" in keys or {
        "candidate_adapter",
        "feature_pipeline",
    }.issubset(keys):
        return EXPERIMENT_CORE_KIND
    if {"lightgbm_run", "xgboost_run"}.issubset(keys) or "blend" in keys:
        return BLEND_KIND
    if "search" in keys or "study" in keys:
        return OPTUNA_KIND
    if "mlflow" in keys or "tracking_uri" in keys:
        return MLFLOW_KIND
    return UNKNOWN_KIND


def guard_config_for_command(
    repository_root: Path, command_id: str, relative_path: str
) -> ConfigGuardResult:
    """Validate that a selected configuration matches the command's contract."""
    expected = _EXPECTED_KIND_BY_COMMAND.get(command_id)
    if expected is None:
        return ConfigGuardResult(ok=True, kind=UNKNOWN_KIND, expected_kind=None)
    payload, error = _load_document(repository_root, relative_path)
    if error is not None:
        return ConfigGuardResult(
            ok=False, kind=UNKNOWN_KIND, expected_kind=expected, message=error
        )
    kind = classify_config_document(payload)
    if kind == expected:
        if expected == DEPLOYMENT_KIND:
            parse_error = _authoritative_deployment_error(
                repository_root, relative_path
            )
            if parse_error is not None:
                return ConfigGuardResult(
                    ok=False,
                    kind=kind,
                    expected_kind=expected,
                    message=(
                        "This deployment configuration is rejected by the Final "
                        f"Deployment parser: {parse_error}"
                    ),
                )
        return ConfigGuardResult(ok=True, kind=kind, expected_kind=expected)
    if expected == DEPLOYMENT_KIND and kind == EXPERIMENT_CORE_KIND:
        message = EXPERIMENT_CORE_MESSAGE
    elif expected == EXPERIMENT_CORE_KIND and kind == DEPLOYMENT_KIND:
        message = DEPLOYMENT_MESSAGE
    elif expected == DEPLOYMENT_KIND:
        message = (
            "This file is not a deployment configuration "
            f"(detected schema: {kind}).\n"
            "Generate a deployment draft from a completed run instead."
        )
    else:
        message = (
            "This file is not an Experiment Core training configuration "
            f"(detected schema: {kind})."
        )
    return ConfigGuardResult(
        ok=False, kind=kind, expected_kind=expected, message=message
    )


def _authoritative_deployment_error(
    repository_root: Path, relative_path: str
) -> str | None:
    try:
        load_deployment_config(
            Path(*PurePosixPath(relative_path.replace("\\", "/")).parts),
            project_root=repository_root,
        )
    except DeploymentContractError as error:
        return str(error)
    except (OSError, ValueError) as error:  # pragma: no cover - defensive
        return str(error)
    return None


def _load_document(
    repository_root: Path, relative_path: str
) -> tuple[Any, str | None]:
    text = str(relative_path).strip().replace("\\", "/")
    if not text:
        return None, "No configuration file is selected."
    posix = PurePosixPath(text)
    if posix.is_absolute() or ".." in posix.parts or "." in posix.parts:
        return None, "The configuration path is not repository-relative."
    path = repository_root.resolve() / Path(*posix.parts)
    if not path_exists_nonfollowing(path):
        return None, f"The configuration file does not exist: {text}"
    try:
        require_regular_file(path, reject_hardlinks=True)
        if path.stat().st_size > _MAX_CONFIG_BYTES:
            return None, "The configuration file is too large to validate."
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, PathSafetyError) as error:
        return None, f"The configuration file cannot be read: {error}"
    try:
        return yaml.safe_load(raw), None
    except yaml.YAMLError as error:
        return None, f"The configuration file is not valid YAML/JSON: {error}"


__all__ = [
    "BLEND_KIND",
    "DEPLOYMENT_KIND",
    "DEPLOYMENT_MESSAGE",
    "EXPERIMENT_CORE_KIND",
    "EXPERIMENT_CORE_MESSAGE",
    "MLFLOW_KIND",
    "OPTUNA_KIND",
    "UNKNOWN_KIND",
    "ConfigGuardResult",
    "classify_config_document",
    "guard_config_for_command",
]
