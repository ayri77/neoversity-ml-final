from __future__ import annotations

from copy import deepcopy
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

import yaml

from src.churn_ml.experiment_v2 import (
    get_candidate_adapter,
    get_feature_pipeline,
)
from src.churn_ml.research_config import (
    _validate_plan_invariants,
    _validate_plan_structure,
)
from src.churn_ml.research_v2_config import (
    ROOT_KEYS,
    SAFE_SLUG,
    SECTION_KEYS,
    ResearchV2Config,
    _validate_search_provenance,
)


class ResolvedResearchV2ConfigurationError(ValueError):
    """Raised before semantic validation when a resolved v2 config is invalid."""

    def __init__(self, message: str, *, reason_code: str, field_path: str) -> None:
        self.reason_code = reason_code
        self.field_path = field_path
        self.detail = message
        super().__init__(f"{reason_code} at {field_path}: {message}")


def load_resolved_research_v2_config(
    path: Path,
    *,
    project_root: Path,
) -> ResearchV2Config:
    """Strictly load a self-contained persisted Experiment Core v2 config."""
    root = project_root.resolve()
    source = _contained_existing_file(
        path,
        root,
        reason_code="RESOLVED_CONFIG_PATH_INVALID",
        field_path="resolved_config_path",
    )
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ResolvedResearchV2ConfigurationError(
            "cannot read resolved YAML",
            reason_code="RESOLVED_CONFIG_YAML_INVALID",
            field_path="resolved_config",
        ) from error
    if not isinstance(value, dict):
        raise ResolvedResearchV2ConfigurationError(
            "must be a mapping",
            reason_code="RESOLVED_CONFIG_TYPE_INVALID",
            field_path="resolved_config",
        )
    resolved = dict(value)
    actual_root_keys = set(resolved)
    base_resolved_keys = ROOT_KEYS | {"evaluation_plan"}
    if actual_root_keys not in (
        base_resolved_keys,
        base_resolved_keys | {"search_provenance"},
    ):
        _exact_keys(resolved, base_resolved_keys, "resolved_config")
    payload = deepcopy(resolved)
    plan = _mapping(
        payload.pop("evaluation_plan"),
        "resolved_config.evaluation_plan",
    )
    for name, expected in SECTION_KEYS.items():
        _exact_keys(
            _mapping(payload.get(name), f"resolved_config.{name}"),
            expected,
            f"resolved_config.{name}",
        )
    schema_version = _integer(
        payload.get("schema_version"),
        "resolved_config.schema_version",
    )
    if schema_version != 2:
        _raise(
            "must be exactly 2",
            "RESOLVED_CONFIG_VALUE_INVALID",
            "resolved_config.schema_version",
        )
    experiment = _mapping(payload["experiment"], "resolved_config.experiment")
    _slug(experiment.get("id"), "resolved_config.experiment.id")
    dataset = _mapping(payload["dataset"], "resolved_config.dataset")
    dataset_version = _slug(
        dataset.get("version"),
        "resolved_config.dataset.version",
    )
    plan_path = _repository_relative_path(
        payload.get("evaluation_plan_path"),
        root,
        "resolved_config.evaluation_plan_path",
    )
    try:
        _validate_plan_structure(plan)
        _validate_plan_invariants(plan)
    except ValueError as error:
        raise ResolvedResearchV2ConfigurationError(
            str(error),
            reason_code="RESOLVED_PLAN_CONTRACT_INVALID",
            field_path=_plan_error_path(str(error)),
        ) from error
    if plan["dataset"]["version"] != dataset_version:
        _raise(
            "run and plan dataset versions differ",
            "RESOLVED_CONFIG_CROSS_CONTRACT_INVALID",
            "resolved_config.dataset.version",
        )
    plan_dataset = _mapping(
        plan["dataset"],
        "resolved_config.evaluation_plan.dataset",
    )
    _repository_relative_path(
        plan_dataset.get("processed_dir"),
        root,
        "resolved_config.evaluation_plan.dataset.processed_dir",
    )
    files = _mapping(
        plan_dataset.get("files"),
        "resolved_config.evaluation_plan.dataset.files",
    )
    for label, item in files.items():
        file_config = _mapping(
            item,
            f"resolved_config.evaluation_plan.dataset.files.{label}",
        )
        _plain_basename(
            file_config.get("name"),
            f"resolved_config.evaluation_plan.dataset.files.{label}.name",
        )
    pipeline = _mapping(
        payload["feature_pipeline"],
        "resolved_config.feature_pipeline",
    )
    adapter = _mapping(
        payload["candidate_adapter"],
        "resolved_config.candidate_adapter",
    )
    pipeline_id = _slug(
        pipeline.get("id"),
        "resolved_config.feature_pipeline.id",
    )
    adapter_id = _slug(
        adapter.get("id"),
        "resolved_config.candidate_adapter.id",
    )
    pipeline_contract = _mapping(
        pipeline.get("contract"),
        "resolved_config.feature_pipeline.contract",
    )
    adapter_contract = _mapping(
        adapter.get("contract"),
        "resolved_config.candidate_adapter.contract",
    )
    try:
        get_feature_pipeline(pipeline_id).validate_contract(pipeline_contract)
    except ValueError as error:
        raise ResolvedResearchV2ConfigurationError(
            str(error),
            reason_code="RESOLVED_PIPELINE_CONTRACT_INVALID",
            field_path=_component_error_path(
                str(error),
                "resolved_config.feature_pipeline.contract",
            ),
        ) from error
    try:
        get_candidate_adapter(adapter_id).validate_contract(adapter_contract)
    except ValueError as error:
        raise ResolvedResearchV2ConfigurationError(
            str(error),
            reason_code="RESOLVED_ADAPTER_CONTRACT_INVALID",
            field_path=_component_error_path(
                str(error),
                "resolved_config.candidate_adapter.contract",
            ),
        ) from error
    artifacts = _mapping(payload["artifacts"], "resolved_config.artifacts")
    _repository_relative_path(
        artifacts.get("root"),
        root,
        "resolved_config.artifacts.root",
    )
    persistence = _mapping(
        payload["persistence"],
        "resolved_config.persistence",
    )
    required_true = {
        "resolved_config",
        "identities",
        "assignments",
        "predictions",
        "metrics",
    }
    for key, item in persistence.items():
        value_bool = _boolean(
            item,
            f"resolved_config.persistence.{key}",
        )
        if key in required_true and not value_bool:
            _raise(
                "must be true",
                "RESOLVED_PERSISTENCE_CONTRACT_INVALID",
                f"resolved_config.persistence.{key}",
            )
    if persistence["models"]:
        _raise(
            "model persistence must be disabled",
            "RESOLVED_MODEL_PERSISTENCE_ENABLED",
            "resolved_config.persistence.models",
        )
    tracking = _mapping(payload["tracking"], "resolved_config.tracking")
    if _boolean(
        tracking.get("enabled"),
        "resolved_config.tracking.enabled",
    ):
        _raise(
            "tracking must be disabled",
            "RESOLVED_TRACKING_ENABLED",
            "resolved_config.tracking.enabled",
        )
    if "search_provenance" in payload:
        try:
            _validate_search_provenance(payload["search_provenance"])
        except ValueError as error:
            raise ResolvedResearchV2ConfigurationError(
                str(error),
                reason_code="RESOLVED_SEARCH_PROVENANCE_INVALID",
                field_path="resolved_config.search_provenance",
            ) from error
    return ResearchV2Config(
        payload=deepcopy(payload),
        plan_payload=deepcopy(dict(plan)),
        source_path=source,
        plan_path=plan_path,
        project_root=root,
    )


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    field_path: str,
) -> None:
    actual = set(value)
    if actual != expected:
        _raise(
            f"keys differ; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}",
            "RESOLVED_CONFIG_KEYS_INVALID",
            field_path,
        )


def _mapping(value: Any, field_path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _raise(
            "must be a mapping",
            "RESOLVED_CONFIG_TYPE_INVALID",
            field_path,
        )
    return dict(value)


def _integer(value: Any, field_path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _raise(
            "must be an integer",
            "RESOLVED_CONFIG_TYPE_INVALID",
            field_path,
        )
    return value


def _boolean(value: Any, field_path: str) -> bool:
    if type(value) is not bool:
        _raise(
            "must be a boolean",
            "RESOLVED_CONFIG_TYPE_INVALID",
            field_path,
        )
    return value


def _string(value: Any, field_path: str) -> str:
    if not isinstance(value, str) or not value:
        _raise(
            "must be a non-empty string",
            "RESOLVED_CONFIG_TYPE_INVALID",
            field_path,
        )
    return value


def _slug(value: Any, field_path: str) -> str:
    text = _string(value, field_path)
    if SAFE_SLUG.fullmatch(text) is None:
        _raise(
            "must be a safe slug",
            "RESOLVED_CONFIG_VALUE_INVALID",
            field_path,
        )
    return text


def _plain_basename(value: Any, field_path: str) -> str:
    text = _string(value, field_path)
    if (
        text in {".", ".."}
        or "/" in text
        or "\\" in text
        or PurePosixPath(text).is_absolute()
        or PureWindowsPath(text).is_absolute()
        or bool(PureWindowsPath(text).drive)
        or bool(PureWindowsPath(text).root)
        or Path(text).name != text
    ):
        _raise(
            "must be a plain basename",
            "RESOLVED_CONFIG_PATH_INVALID",
            field_path,
        )
    return text


def _repository_relative_path(
    value: Any,
    root: Path,
    field_path: str,
) -> Path:
    text = _string(value, field_path)
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or ".." in posix.parts
        or ".." in windows.parts
    ):
        _raise(
            "must be repository-relative without traversal",
            "RESOLVED_CONFIG_PATH_INVALID",
            field_path,
        )
    resolved = (root / Path(text)).resolve()
    if resolved == root or root not in resolved.parents:
        _raise(
            "escapes the repository",
            "RESOLVED_CONFIG_PATH_INVALID",
            field_path,
        )
    return resolved


def _contained_existing_file(
    path: Path,
    root: Path,
    *,
    reason_code: str,
    field_path: str,
) -> Path:
    unresolved = path if path.is_absolute() else root / path
    try:
        resolved = unresolved.resolve(strict=True)
    except OSError as error:
        raise ResolvedResearchV2ConfigurationError(
            "does not exist",
            reason_code=reason_code,
            field_path=field_path,
        ) from error
    if not resolved.is_file() or root not in resolved.parents:
        _raise("must be a contained file", reason_code, field_path)
    return resolved


def _plan_error_path(message: str) -> str:
    invariant_paths = {
        "Outer splitter": "outer_evaluation.splitter",
        "Outer evaluation fold count": "outer_evaluation.n_splits",
        "Outer repeat seeds": "outer_evaluation.repeat_seeds",
        "Threshold-selection splitter": "threshold_selection.splitter",
        "Threshold-selection fold count": "threshold_selection.n_splits",
        "Threshold score tolerance": ("threshold_policy.maximizer_absolute_tolerance"),
        "Aggregation settings": "aggregation",
        "Plan schema_version": "schema_version",
    }
    for prefix, relative in invariant_paths.items():
        if message.startswith(prefix):
            return f"resolved_config.evaluation_plan.{relative}"
    candidates = (
        "dataset.files.",
        "dataset.target.",
        "outer_evaluation.",
        "threshold_selection.",
        "threshold_policy.",
        "metrics.",
        "aggregation.",
        "dataset.",
        "plan.",
        "schema_version",
    )
    for candidate in candidates:
        position = message.find(candidate)
        if position >= 0:
            suffix = message[position:].split()[0].rstrip(":.,")
            return f"resolved_config.evaluation_plan.{suffix}"
    return "resolved_config.evaluation_plan"


def _component_error_path(message: str, default: str) -> str:
    for marker in ("contract.", "target_encoder.", "lightgbm.", "features."):
        position = message.find(marker)
        if position >= 0:
            suffix = message[position:].split()[0].rstrip(":.,")
            return f"{default}.{suffix}"
    return default


def _raise(message: str, reason_code: str, field_path: str) -> None:
    raise ResolvedResearchV2ConfigurationError(
        message,
        reason_code=reason_code,
        field_path=field_path,
    )
