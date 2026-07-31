"""Strict versioned Dataset Campaign specification parsing."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.churn_ml.control_panel.presentation import normalize_model_family
from src.churn_ml.dataset_campaign.constants import (
    CAMPAIGN_CONTRACT_VERSION,
    CAMPAIGN_TYPES,
    CLASSIFICATIONS,
    DATASET_PRESETS,
    DEFAULT_CAMPAIGN_ARTIFACTS_ROOT,
    DEFAULT_PROCESSED_ROOT,
    EXECUTION_POLICIES,
    EXPLORATORY_DATASET_IDS,
    MODEL_FAMILIES,
    UNBIASED_SCREENING_DATASET_IDS,
)
from src.churn_ml.dataset_campaign.errors import CampaignConfigurationError
from src.churn_ml.dataset_campaign.paths import safe_repo_relative_file
from src.churn_ml.research_data import canonical_sha256


SAFE_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")

ROOT_KEYS = frozenset(
    {
        "schema_version",
        "campaign",
        "datasets",
        "models",
        "evaluation_plan",
        "execution",
    }
)
CAMPAIGN_KEYS = frozenset({"id", "name", "type", "classification"})
DATASETS_KEYS = frozenset({"ids", "preset"})
MODEL_ENTRY_KEYS = frozenset({"family", "config_path"})
EVALUATION_PLAN_KEYS = frozenset({"path"})
EXECUTION_KEYS = frozenset(
    {
        "policy",
        "processed_root",
        "artifacts_root",
        "index_mlflow",
        "mlflow_config_path",
    }
)


@dataclass(frozen=True)
class CampaignModelEntry:
    family: str
    config_path: str


@dataclass(frozen=True)
class CampaignSpec:
    """Immutable parsed user-facing campaign specification."""

    schema_version: str
    campaign_id: str
    campaign_name: str
    campaign_type: str
    classification: str
    dataset_ids: tuple[str, ...]
    models: tuple[CampaignModelEntry, ...]
    evaluation_plan_path: str | None
    execution_policy: str
    processed_root: str
    artifacts_root: str
    index_mlflow: bool
    mlflow_config_path: str
    source_path: Path | None
    source_sha256: str
    payload: dict[str, Any]

    @property
    def is_unbiased(self) -> bool:
        return self.classification == "unbiased"


def load_campaign_spec(
    path: Path,
    *,
    project_root: Path,
) -> CampaignSpec:
    """Load and strictly validate a campaign specification YAML file."""
    root = project_root.resolve()
    spec_path = path if path.is_absolute() else (Path.cwd() / path).resolve()
    if not spec_path.is_file():
        raise CampaignConfigurationError(f"Campaign specification not found: {path}.")
    try:
        text = spec_path.read_text(encoding="utf-8")
        payload = yaml.safe_load(text)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise CampaignConfigurationError(
            f"Could not load campaign specification: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise CampaignConfigurationError(
            "Campaign specification must be a top-level mapping."
        )
    return parse_campaign_spec(payload, project_root=root, source_path=spec_path)


def parse_campaign_spec(
    payload: Mapping[str, Any],
    *,
    project_root: Path,
    source_path: Path | None = None,
) -> CampaignSpec:
    """Parse an in-memory campaign specification with exact-key rejection."""
    data = dict(payload)
    _exact_keys(data, ROOT_KEYS, "campaign specification")

    schema_version = str(data["schema_version"])
    if schema_version != CAMPAIGN_CONTRACT_VERSION:
        raise CampaignConfigurationError(
            f"Unsupported campaign schema_version {schema_version!r}; "
            f"expected {CAMPAIGN_CONTRACT_VERSION!r}."
        )

    campaign = _require_mapping(data["campaign"], "campaign")
    _exact_keys(campaign, CAMPAIGN_KEYS, "campaign")
    campaign_id = _require_slug(str(campaign["id"]), "campaign.id")
    campaign_name = str(campaign["name"]).strip()
    if not campaign_name:
        raise CampaignConfigurationError("campaign.name must be non-empty.")
    campaign_type = str(campaign["type"]).strip().lower()
    if campaign_type not in CAMPAIGN_TYPES:
        raise CampaignConfigurationError(
            f"campaign.type must be one of {sorted(CAMPAIGN_TYPES)}; "
            f"got {campaign_type!r}."
        )
    classification = str(campaign["classification"]).strip().lower()
    if classification not in CLASSIFICATIONS:
        raise CampaignConfigurationError(
            f"campaign.classification must be one of {sorted(CLASSIFICATIONS)}; "
            f"got {classification!r}."
        )

    datasets = _require_mapping(data["datasets"], "datasets")
    unknown_dataset_keys = set(datasets) - DATASETS_KEYS
    if unknown_dataset_keys:
        raise CampaignConfigurationError(
            f"datasets has unknown keys: {sorted(unknown_dataset_keys)}."
        )
    if not datasets:
        raise CampaignConfigurationError(
            "datasets must provide either ids or preset."
        )
    dataset_ids = _resolve_dataset_ids(datasets, classification=classification)

    models_raw = data["models"]
    if not isinstance(models_raw, list) or not models_raw:
        raise CampaignConfigurationError("models must be a non-empty list.")
    models: list[CampaignModelEntry] = []
    seen_families: set[str] = set()
    for index, entry in enumerate(models_raw):
        if not isinstance(entry, Mapping):
            raise CampaignConfigurationError(
                f"models[{index}] must be a mapping."
            )
        _exact_keys(dict(entry), MODEL_ENTRY_KEYS, f"models[{index}]")
        family = normalize_model_family(str(entry["family"]))
        if family not in MODEL_FAMILIES:
            raise CampaignConfigurationError(
                f"models[{index}].family must be one of {sorted(MODEL_FAMILIES)}; "
                f"got {family!r}."
            )
        if family in seen_families:
            raise CampaignConfigurationError(
                f"Duplicate model family in campaign specification: {family}."
            )
        seen_families.add(family)
        config_path = PurePosix(str(entry["config_path"]))
        safe_repo_relative_file(
            project_root, config_path, label=f"models[{index}].config_path"
        )
        models.append(CampaignModelEntry(family=family, config_path=config_path))

    evaluation_plan = _require_mapping(data["evaluation_plan"], "evaluation_plan")
    _exact_keys(evaluation_plan, EVALUATION_PLAN_KEYS, "evaluation_plan")
    plan_path_raw = evaluation_plan["path"]
    evaluation_plan_path: str | None
    if plan_path_raw is None:
        evaluation_plan_path = None
    else:
        evaluation_plan_path = PurePosix(str(plan_path_raw))
        safe_repo_relative_file(
            project_root, evaluation_plan_path, label="evaluation_plan.path"
        )

    execution = _require_mapping(data["execution"], "execution")
    unknown_execution = set(execution) - EXECUTION_KEYS
    if unknown_execution:
        raise CampaignConfigurationError(
            f"execution has unknown keys: {sorted(unknown_execution)}."
        )
    required_execution = {"policy", "processed_root", "artifacts_root", "index_mlflow"}
    missing_execution = required_execution - set(execution)
    if missing_execution:
        raise CampaignConfigurationError(
            f"execution is missing keys: {sorted(missing_execution)}."
        )
    policy = str(execution["policy"]).strip().lower()
    if policy not in EXECUTION_POLICIES:
        raise CampaignConfigurationError(
            f"execution.policy must be one of {sorted(EXECUTION_POLICIES)}; "
            f"got {policy!r}."
        )
    processed_root = PurePosix(str(execution.get("processed_root", DEFAULT_PROCESSED_ROOT)))
    artifacts_root = PurePosix(
        str(execution.get("artifacts_root", DEFAULT_CAMPAIGN_ARTIFACTS_ROOT))
    )
    index_mlflow = bool(execution["index_mlflow"])
    mlflow_config_path = PurePosix(
        str(execution.get("mlflow_config_path", "configs/mlflow/local.yaml"))
    )

    if classification == "unbiased":
        _reject_exploratory_mix(dataset_ids)
        if campaign_type == "exploratory":
            raise CampaignConfigurationError(
                "Unbiased campaigns cannot use campaign.type=exploratory."
            )
    elif campaign_type != "exploratory" and any(
        dataset_id in EXPLORATORY_DATASET_IDS for dataset_id in dataset_ids
    ):
        raise CampaignConfigurationError(
            "Exploratory datasets such as v3_targeted_missingness require "
            "campaign.classification=exploratory and campaign.type=exploratory."
        )

    source_sha256 = canonical_sha256(deepcopy(data))
    return CampaignSpec(
        schema_version=schema_version,
        campaign_id=campaign_id,
        campaign_name=campaign_name,
        campaign_type=campaign_type,
        classification=classification,
        dataset_ids=tuple(dataset_ids),
        models=tuple(models),
        evaluation_plan_path=evaluation_plan_path,
        execution_policy=policy,
        processed_root=processed_root,
        artifacts_root=artifacts_root,
        index_mlflow=index_mlflow,
        mlflow_config_path=mlflow_config_path,
        source_path=source_path,
        source_sha256=source_sha256,
        payload=deepcopy(data),
    )


def _resolve_dataset_ids(
    datasets: Mapping[str, Any],
    *,
    classification: str,
) -> list[str]:
    has_ids = "ids" in datasets
    has_preset = "preset" in datasets
    if has_ids and has_preset:
        raise CampaignConfigurationError(
            "datasets must not set both ids and preset."
        )
    if has_preset:
        preset = str(datasets["preset"])
        if preset not in DATASET_PRESETS:
            raise CampaignConfigurationError(
                f"Unknown datasets.preset {preset!r}; "
                f"known presets: {sorted(DATASET_PRESETS)}."
            )
        if classification != "unbiased" and preset == "unbiased_screening_v1":
            raise CampaignConfigurationError(
                "datasets.preset=unbiased_screening_v1 requires "
                "campaign.classification=unbiased."
            )
        return list(DATASET_PRESETS[preset])
    ids_raw = datasets.get("ids")
    if not isinstance(ids_raw, list) or not ids_raw:
        raise CampaignConfigurationError("datasets.ids must be a non-empty list.")
    ids: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(ids_raw):
        dataset_id = _require_slug(str(value), f"datasets.ids[{index}]")
        if dataset_id in seen:
            raise CampaignConfigurationError(
                f"Duplicate dataset id in campaign specification: {dataset_id}."
            )
        seen.add(dataset_id)
        ids.append(dataset_id)
    if classification == "unbiased" and set(ids) == set(UNBIASED_SCREENING_DATASET_IDS):
        # Preserve canonical screening order when the full set is selected.
        return list(UNBIASED_SCREENING_DATASET_IDS)
    return ids


def _reject_exploratory_mix(dataset_ids: list[str]) -> None:
    mixed = sorted(set(dataset_ids) & EXPLORATORY_DATASET_IDS)
    if mixed:
        raise CampaignConfigurationError(
            "Unbiased campaigns must not include exploratory datasets "
            f"{mixed}; run them in a separate exploratory campaign."
        )


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CampaignConfigurationError(f"{label} must be a mapping.")
    return dict(value)


def _exact_keys(payload: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    keys = set(payload)
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected)
    if missing or unknown:
        raise CampaignConfigurationError(
            f"{label} key mismatch: missing={missing}, unknown={unknown}."
        )


def _require_slug(value: str, label: str) -> str:
    if SAFE_SLUG.fullmatch(value) is None:
        raise CampaignConfigurationError(
            f"{label} must be a safe slug matching {SAFE_SLUG.pattern}."
        )
    return value


def PurePosix(value: str) -> str:
    return value.replace("\\", "/")
