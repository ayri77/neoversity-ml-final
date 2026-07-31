"""Matrix expansion and controlled-comparison invariants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.churn_ml.control_panel.presentation import normalize_model_family
from src.churn_ml.dataset_campaign.constants import FAMILY_TO_ADAPTER_PREFIX
from src.churn_ml.dataset_campaign.errors import CampaignConfigurationError
from src.churn_ml.dataset_campaign.paths import safe_repo_relative_file
from src.churn_ml.dataset_campaign.schema import CampaignModelEntry, CampaignSpec
from src.churn_ml.research_data import canonical_sha256


@dataclass(frozen=True)
class MatrixCellRef:
    """One unresolved matrix cell (dataset × model family)."""

    execution_order: int
    dataset_id: str
    model_family: str
    config_path: str


@dataclass(frozen=True)
class ModelConfigSnapshot:
    family: str
    config_path: str
    config_sha256: str
    adapter_id: str
    adapter_contract: dict[str, Any]
    evaluation_plan_path: str
    evaluation_plan_sha256: str
    protocol_identity: dict[str, Any]
    protocol_sha256: str


def expand_matrix(spec: CampaignSpec) -> tuple[MatrixCellRef, ...]:
    """Expand the campaign into a deterministic ordered matrix."""
    cells: list[MatrixCellRef] = []
    order = 0
    for model in spec.models:
        for dataset_id in spec.dataset_ids:
            cells.append(
                MatrixCellRef(
                    execution_order=order,
                    dataset_id=dataset_id,
                    model_family=model.family,
                    config_path=model.config_path,
                )
            )
            order += 1
    return tuple(cells)


def load_model_config_snapshots(
    spec: CampaignSpec,
    *,
    project_root: Path,
) -> dict[str, ModelConfigSnapshot]:
    """Load referenced model configs and enforce cross-family protocol identity."""
    snapshots: dict[str, ModelConfigSnapshot] = {}
    protocol_hashes: dict[str, str] = {}
    plan_paths: dict[str, str] = {}
    for model in spec.models:
        snapshot = _load_model_snapshot(model, project_root=project_root)
        if spec.evaluation_plan_path is not None:
            if snapshot.evaluation_plan_path != spec.evaluation_plan_path:
                raise CampaignConfigurationError(
                    f"Model {model.family} evaluation_plan_path "
                    f"{snapshot.evaluation_plan_path!r} differs from campaign "
                    f"evaluation_plan.path {spec.evaluation_plan_path!r}."
                )
        snapshots[model.family] = snapshot
        protocol_hashes[model.family] = snapshot.protocol_sha256
        plan_paths[model.family] = snapshot.evaluation_plan_path

    unique_protocols = set(protocol_hashes.values())
    if len(unique_protocols) != 1:
        raise CampaignConfigurationError(
            "Controlled-comparison invariant failed: model families reference "
            f"conflicting evaluation protocols: {protocol_hashes}."
        )
    unique_plans = set(plan_paths.values())
    if len(unique_plans) != 1:
        raise CampaignConfigurationError(
            "Controlled-comparison invariant failed: model families reference "
            f"different evaluation_plan_path values: {plan_paths}."
        )
    return snapshots


def assert_family_adapter_match(*, family: str, adapter_id: str) -> None:
    prefixes = FAMILY_TO_ADAPTER_PREFIX.get(family)
    if prefixes is None:
        raise CampaignConfigurationError(f"Unsupported model family: {family}.")
    lower = adapter_id.lower()
    if not any(lower.startswith(prefix) for prefix in prefixes):
        raise CampaignConfigurationError(
            f"Model-family/config mismatch: family {family} is incompatible with "
            f"adapter id {adapter_id!r}."
        )


def cell_identity_payload(
    *,
    campaign_id: str,
    dataset_id: str,
    model_family: str,
    adapter_id: str,
    config_sha256: str,
    protocol_sha256: str,
    package_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "campaign_id": campaign_id,
        "dataset_id": dataset_id,
        "model_family": model_family,
        "adapter_id": adapter_id,
        "config_sha256": config_sha256,
        "protocol_sha256": protocol_sha256,
        "package_identity": dict(package_identity),
    }


def cell_id_for(payload: Mapping[str, Any]) -> str:
    digest = canonical_sha256(dict(payload))
    return f"cell_{digest[:16]}"


def _load_model_snapshot(
    model: CampaignModelEntry,
    *,
    project_root: Path,
) -> ModelConfigSnapshot:
    path = safe_repo_relative_file(
        project_root, model.config_path, label="model config"
    )
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise CampaignConfigurationError(
            f"Could not load model config {model.config_path}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise CampaignConfigurationError(
            f"Model config must be a mapping: {model.config_path}."
        )
    if int(payload.get("schema_version", -1)) != 2:
        raise CampaignConfigurationError(
            f"Model config must be Research v2 (schema_version=2): "
            f"{model.config_path}."
        )
    adapter = payload.get("candidate_adapter")
    if not isinstance(adapter, Mapping) or "id" not in adapter or "contract" not in adapter:
        raise CampaignConfigurationError(
            f"Model config missing candidate_adapter: {model.config_path}."
        )
    adapter_id = str(adapter["id"])
    inferred_family = normalize_model_family(adapter_id)
    if inferred_family != model.family:
        raise CampaignConfigurationError(
            f"Model-family/config mismatch for {model.config_path}: "
            f"declared family {model.family}, adapter implies {inferred_family}."
        )
    assert_family_adapter_match(family=model.family, adapter_id=adapter_id)

    plan_rel = str(payload.get("evaluation_plan_path", "")).replace("\\", "/")
    if not plan_rel:
        raise CampaignConfigurationError(
            f"Model config missing evaluation_plan_path: {model.config_path}."
        )
    plan_path = safe_repo_relative_file(
        project_root, plan_rel, label="evaluation plan"
    )
    try:
        plan_payload = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise CampaignConfigurationError(
            f"Could not load evaluation plan {plan_rel}: {error}"
        ) from error
    if not isinstance(plan_payload, Mapping):
        raise CampaignConfigurationError(
            f"Evaluation plan must be a mapping: {plan_rel}."
        )

    protocol_identity = {
        "outer_evaluation": plan_payload.get("outer_evaluation"),
        "threshold_selection": plan_payload.get("threshold_selection"),
        "threshold_policy": plan_payload.get("threshold_policy"),
        "metrics": plan_payload.get("metrics"),
        "aggregation": plan_payload.get("aggregation"),
    }
    return ModelConfigSnapshot(
        family=model.family,
        config_path=model.config_path,
        config_sha256=canonical_sha256(dict(payload)),
        adapter_id=adapter_id,
        adapter_contract=dict(adapter["contract"]),
        evaluation_plan_path=plan_rel,
        evaluation_plan_sha256=canonical_sha256(dict(plan_payload)),
        protocol_identity=protocol_identity,
        protocol_sha256=canonical_sha256(protocol_identity),
    )
