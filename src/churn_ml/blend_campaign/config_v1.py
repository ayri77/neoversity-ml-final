"""Strict configuration contract for Blend Campaign Runner v1."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

from src.churn_ml.blending.evaluation_v1 import BlendSettings
from src.churn_ml.blending.optimization_v1 import DEFAULT_THRESHOLD_POLICY
from src.churn_ml.research_data import canonical_sha256


SCHEMA_VERSION = "blend_campaign_v1"
SAFE_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
ROOT_KEYS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "candidate_root",
        "blend_root",
        "submission_root",
        "candidates",
        "experiments",
        "materialize_policy",
        "submission_policy",
    }
)
EXPERIMENT_KEYS = frozenset(
    {
        "experiment_id",
        "candidates",
        "strategy",
        "optimizer",
        "folds",
        "repeats",
        "seed",
        "max_active_models",
        "manual_weights",
        "native_search",
        "optuna",
        "threshold_policy",
        "exploratory",
    }
)
NATIVE_KEYS = frozenset({"pairwise_grid_step", "dirichlet_draws"})
OPTUNA_KEYS = frozenset({"trials", "timeout_seconds", "seed"})
THRESHOLD_KEYS = frozenset(DEFAULT_THRESHOLD_POLICY)
MATERIALIZE_KEYS = frozenset({"top_k", "experiment_ids"})
SUBMISSION_KEYS = frozenset({"enabled", "experiment_ids", "id_prefix"})


class BlendCampaignConfigurationError(ValueError):
    """Raised when a campaign YAML does not satisfy the v1 contract."""


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    aliases: tuple[str, ...]
    strategy: str
    optimizer: str
    folds: int
    repeats: int
    seed: int
    max_active_models: int | None
    manual_weights: tuple[tuple[str, float], ...]
    pairwise_grid_step: float
    dirichlet_draws: int
    optuna_trials: int
    optuna_timeout_seconds: float | None
    optuna_seed: int
    threshold_policy: dict[str, Any]
    exploratory: bool

    def settings(self, candidate_ids: tuple[str, ...]) -> BlendSettings:
        alias_to_id = dict(zip(self.aliases, candidate_ids, strict=True))
        weights = tuple(
            f"{alias_to_id[alias]}={value:.17g}" for alias, value in self.manual_weights
        )
        return BlendSettings(
            strategy=self.strategy,
            folds=self.folds,
            repeats=self.repeats,
            seed=self.seed,
            max_active_models=self.max_active_models,
            manual_weights=weights,
            threshold_policy=dict(self.threshold_policy),
            pairwise_grid_step=self.pairwise_grid_step,
            dirichlet_draws=self.dirichlet_draws,
            optimizer_backend=self.optimizer,
            optuna_trials=self.optuna_trials,
            optuna_timeout_seconds=self.optuna_timeout_seconds,
            optuna_seed=self.optuna_seed,
        ).normalized()


@dataclass(frozen=True)
class BlendCampaignConfig:
    campaign_id: str
    candidate_root: str
    blend_root: str
    submission_root: str
    aliases: dict[str, str]
    experiments: tuple[ExperimentConfig, ...]
    materialize_top_k: int
    materialize_experiment_ids: tuple[str, ...]
    submission_enabled: bool
    submission_experiment_ids: tuple[str, ...]
    submission_id_prefix: str
    payload: dict[str, Any]
    source_path: Path | None
    config_hash: str


def load_campaign_config(path: Path, *, repository_root: Path) -> BlendCampaignConfig:
    config_path = path if path.is_absolute() else repository_root / path
    if not config_path.is_file():
        raise BlendCampaignConfigurationError(f"Campaign config not found: {path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise BlendCampaignConfigurationError(
            f"Could not read campaign YAML: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise BlendCampaignConfigurationError("Campaign config must be a mapping.")
    return parse_campaign_config(payload, source_path=config_path.resolve())


def parse_campaign_config(
    payload: Mapping[str, Any], *, source_path: Path | None = None
) -> BlendCampaignConfig:
    data = deepcopy(dict(payload))
    _exact_keys(data, ROOT_KEYS, "campaign")
    if data["schema_version"] != SCHEMA_VERSION:
        raise BlendCampaignConfigurationError(
            f"schema_version must be {SCHEMA_VERSION!r}."
        )
    campaign_id = _slug(data["campaign_id"], "campaign_id")
    candidate_root = _relative_path(data["candidate_root"], "candidate_root")
    blend_root = _relative_path(data["blend_root"], "blend_root")
    submission_root = _relative_path(data["submission_root"], "submission_root")

    aliases_raw = _mapping(data["candidates"], "candidates")
    if len(aliases_raw) < 2:
        raise BlendCampaignConfigurationError(
            "candidates must define at least two aliases."
        )
    aliases: dict[str, str] = {}
    for alias, candidate_id in aliases_raw.items():
        alias_value = _slug(alias, f"candidate alias {alias!r}")
        value = str(candidate_id).strip()
        if not value:
            raise BlendCampaignConfigurationError(
                f"Candidate ID for {alias_value} is empty."
            )
        aliases[alias_value] = value
    if len(set(aliases.values())) != len(aliases):
        raise BlendCampaignConfigurationError(
            "Candidate aliases must map to distinct candidate IDs."
        )

    experiments_raw = data["experiments"]
    if not isinstance(experiments_raw, list) or not experiments_raw:
        raise BlendCampaignConfigurationError(
            "experiments must be a non-empty ordered list."
        )
    experiments: list[ExperimentConfig] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(experiments_raw):
        item = _mapping(raw, f"experiments[{index}]")
        _exact_keys(item, EXPERIMENT_KEYS, f"experiments[{index}]")
        experiment_id = _slug(
            item["experiment_id"], f"experiments[{index}].experiment_id"
        )
        if experiment_id in seen_ids:
            raise BlendCampaignConfigurationError(
                f"Duplicate experiment_id: {experiment_id}."
            )
        seen_ids.add(experiment_id)
        selected = item["candidates"]
        if not isinstance(selected, list) or len(selected) < 2:
            raise BlendCampaignConfigurationError(
                f"{experiment_id}: candidates must contain at least two aliases."
            )
        selected_aliases = tuple(str(value) for value in selected)
        if len(set(selected_aliases)) != len(selected_aliases):
            raise BlendCampaignConfigurationError(
                f"{experiment_id}: duplicate candidate aliases are rejected."
            )
        unknown = sorted(set(selected_aliases) - set(aliases))
        if unknown:
            raise BlendCampaignConfigurationError(
                f"{experiment_id}: unknown candidate aliases {unknown}."
            )
        strategy = str(item["strategy"])
        optimizer = str(item["optimizer"])
        native = _mapping(item["native_search"], f"{experiment_id}.native_search")
        _exact_keys(native, NATIVE_KEYS, f"{experiment_id}.native_search")
        optuna = _mapping(item["optuna"], f"{experiment_id}.optuna")
        _exact_keys(optuna, OPTUNA_KEYS, f"{experiment_id}.optuna")
        threshold = _mapping(
            item["threshold_policy"], f"{experiment_id}.threshold_policy"
        )
        _exact_keys(threshold, THRESHOLD_KEYS, f"{experiment_id}.threshold_policy")
        _validate_threshold(threshold, experiment_id)

        manual_raw = _mapping(item["manual_weights"], f"{experiment_id}.manual_weights")
        manual_weights = tuple(
            (str(key), float(value)) for key, value in manual_raw.items()
        )
        if strategy == "manual" and set(manual_raw) != set(selected_aliases):
            raise BlendCampaignConfigurationError(
                f"{experiment_id}: manual_weights must exactly match selected aliases."
            )
        if strategy != "manual" and manual_raw:
            raise BlendCampaignConfigurationError(
                f"{experiment_id}: manual_weights are only valid for strategy=manual."
            )

        experiment = ExperimentConfig(
            experiment_id=experiment_id,
            aliases=selected_aliases,
            strategy=strategy,
            optimizer=optimizer,
            folds=int(item["folds"]),
            repeats=int(item["repeats"]),
            seed=int(item["seed"]),
            max_active_models=(
                None
                if item["max_active_models"] is None
                else int(item["max_active_models"])
            ),
            manual_weights=manual_weights,
            pairwise_grid_step=float(native["pairwise_grid_step"]),
            dirichlet_draws=int(native["dirichlet_draws"]),
            optuna_trials=int(optuna["trials"]),
            optuna_timeout_seconds=(
                None
                if optuna["timeout_seconds"] is None
                else float(optuna["timeout_seconds"])
            ),
            optuna_seed=int(optuna["seed"]),
            threshold_policy=dict(threshold),
            exploratory=_strict_bool(
                item["exploratory"], f"{experiment_id}.exploratory"
            ),
        )
        try:
            experiment.settings(tuple(aliases[value] for value in selected_aliases))
        except (TypeError, ValueError) as error:
            raise BlendCampaignConfigurationError(
                f"{experiment_id}: invalid blend settings: {error}"
            ) from error
        experiments.append(experiment)

    materialize = _mapping(data["materialize_policy"], "materialize_policy")
    _exact_keys(materialize, MATERIALIZE_KEYS, "materialize_policy")
    top_k = int(materialize["top_k"])
    if top_k < 0:
        raise BlendCampaignConfigurationError("materialize_policy.top_k must be >= 0.")
    materialize_ids = _id_list(
        materialize["experiment_ids"], seen_ids, "materialize_policy.experiment_ids"
    )

    submission = _mapping(data["submission_policy"], "submission_policy")
    _exact_keys(submission, SUBMISSION_KEYS, "submission_policy")
    submission_ids = _id_list(
        submission["experiment_ids"], seen_ids, "submission_policy.experiment_ids"
    )
    prefix = _slug(submission["id_prefix"], "submission_policy.id_prefix")
    enabled = _strict_bool(submission["enabled"], "submission_policy.enabled")
    if submission_ids and not enabled:
        raise BlendCampaignConfigurationError(
            "submission_policy.experiment_ids requires enabled=true."
        )

    return BlendCampaignConfig(
        campaign_id=campaign_id,
        candidate_root=candidate_root,
        blend_root=blend_root,
        submission_root=submission_root,
        aliases=aliases,
        experiments=tuple(experiments),
        materialize_top_k=top_k,
        materialize_experiment_ids=materialize_ids,
        submission_enabled=enabled,
        submission_experiment_ids=submission_ids,
        submission_id_prefix=prefix,
        payload=data,
        source_path=source_path,
        config_hash=canonical_sha256(data),
    )


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BlendCampaignConfigurationError(f"{label} must be a mapping.")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise BlendCampaignConfigurationError(
            f"{label} key mismatch: missing={missing}, unknown={unknown}."
        )


def _slug(value: Any, label: str) -> str:
    text = str(value).strip()
    if SAFE_SLUG.fullmatch(text) is None:
        raise BlendCampaignConfigurationError(
            f"{label} is not a safe identifier: {text!r}."
        )
    return text


def _relative_path(value: Any, label: str) -> str:
    text = str(value).replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise BlendCampaignConfigurationError(
            f"{label} must be a safe repository-relative path."
        )
    return path.as_posix()


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise BlendCampaignConfigurationError(f"{label} must be true or false.")
    return value


def _id_list(value: Any, known: set[str], label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise BlendCampaignConfigurationError(f"{label} must be a list.")
    items = tuple(str(item) for item in value)
    if len(set(items)) != len(items):
        raise BlendCampaignConfigurationError(f"{label} contains duplicates.")
    unknown = sorted(set(items) - known)
    if unknown:
        raise BlendCampaignConfigurationError(
            f"{label} contains unknown IDs: {unknown}."
        )
    return items


def _validate_threshold(policy: Mapping[str, Any], experiment_id: str) -> None:
    minimum = float(policy["minimum"])
    maximum = float(policy["maximum"])
    step = float(policy["step"])
    fallback = float(policy["constant_probability_fallback"])
    tolerance = float(policy["maximizer_absolute_tolerance"])
    if not 0.0 <= minimum <= maximum <= 1.0:
        raise BlendCampaignConfigurationError(
            f"{experiment_id}: threshold bounds must satisfy 0 <= minimum <= maximum <= 1."
        )
    if step <= 0.0 or tolerance < 0.0 or not 0.0 <= fallback <= 1.0:
        raise BlendCampaignConfigurationError(
            f"{experiment_id}: invalid threshold step, tolerance, or fallback."
        )


__all__ = [
    "BlendCampaignConfig",
    "BlendCampaignConfigurationError",
    "ExperimentConfig",
    "SCHEMA_VERSION",
    "load_campaign_config",
    "parse_campaign_config",
]
