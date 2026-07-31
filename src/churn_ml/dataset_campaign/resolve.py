"""Resolve and freeze immutable Dataset Campaign manifests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from src.churn_ml.control_panel.dataset_experiment_materializer import (
    REGISTERED_PREPARED_PASSTHROUGH_CONTRACT,
    REGISTERED_PREPARED_PASSTHROUGH_V1,
    build_evaluation_plan_payload,
    build_experiment_config_payload,
    build_plan_dataset_section,
    make_experiment_id,
    make_plan_id,
)
from src.churn_ml.dataset_campaign.constants import (
    CAMPAIGN_CONTRACT_VERSION,
    EXPLORATORY_DATASET_IDS,
    MANIFEST_SCHEMA_VERSION,
    PREPARED_SUBDIR,
    UNBIASED_SCREENING_DATASET_IDS,
)
from src.churn_ml.dataset_campaign.errors import (
    CampaignConfigurationError,
    CampaignValidationError,
)
from src.churn_ml.dataset_campaign.matrix import (
    MatrixCellRef,
    ModelConfigSnapshot,
    cell_id_for,
    cell_identity_payload,
    expand_matrix,
    load_model_config_snapshots,
)
from src.churn_ml.dataset_campaign.paths import (
    safe_repo_relative_dir,
    to_repo_relative,
)
from src.churn_ml.dataset_campaign.schema import CampaignSpec
from src.churn_ml.dataset_registry import (
    build_dataset_provenance,
    discover_registered_datasets,
    resolve_dataset_package,
)
from src.churn_ml.dataset_registry.errors import DatasetRegistryError
from src.churn_ml.dataset_registry.materialize import dataset_manifest_sha256
from src.churn_ml.research_data import canonical_sha256


@dataclass(frozen=True)
class ResolvedCampaign:
    """Fully resolved campaign with frozen identity separate from runtime status."""

    spec: CampaignSpec
    manifest: dict[str, Any]
    manifest_hash: str
    prepared_configs: dict[str, Path]
    prepared_plans: dict[str, Path]
    validation_failures: tuple[str, ...]


def resolve_campaign(
    spec: CampaignSpec,
    *,
    project_root: Path,
    campaign_dir: Path | None = None,
    materialize: bool = True,
    validate_prepared: bool = True,
) -> ResolvedCampaign:
    """Discover datasets, expand the matrix, and freeze an immutable manifest.

    When ``materialize`` is True, prepared Research v2 config/plan pairs are
    written under the campaign directory. This does not allocate Experiment Core
    run directories or job records.
    """
    root = project_root.resolve()
    failures: list[str] = []
    cells = expand_matrix(spec)

    try:
        snapshots = load_model_config_snapshots(spec, project_root=root)
    except CampaignConfigurationError as error:
        failures.append(str(error))
        snapshots = {}

    processed = safe_repo_relative_dir(
        root, spec.processed_root, label="execution.processed_root"
    )
    try:
        summaries = {
            item.dataset_id: item for item in discover_registered_datasets(processed)
        }
    except (DatasetRegistryError, FileNotFoundError, OSError) as error:
        raise CampaignConfigurationError(
            f"Dataset Registry discovery failed: {error}"
        ) from error

    for dataset_id in spec.dataset_ids:
        if dataset_id not in summaries:
            failures.append(
                f"Unknown Registry dataset id (not discovered): {dataset_id}."
            )

    if spec.is_unbiased:
        missing_screening = [
            dataset_id
            for dataset_id in UNBIASED_SCREENING_DATASET_IDS
            if dataset_id in set(spec.dataset_ids)
            and dataset_id not in summaries
        ]
        for dataset_id in missing_screening:
            failures.append(f"Unbiased screening dataset missing from Registry: {dataset_id}.")

    package_records: dict[str, dict[str, Any]] = {}
    for dataset_id in spec.dataset_ids:
        if dataset_id not in summaries:
            continue
        try:
            package = resolve_dataset_package(processed, dataset_id)
        except DatasetRegistryError as error:
            failures.append(f"Dataset package invalid for {dataset_id}: {error}")
            continue
        manifest = package.manifest
        provenance = build_dataset_provenance(manifest)
        if (
            spec.is_unbiased
            and str(manifest.target_dependency) != "none"
        ):
            failures.append(
                f"Unbiased campaign dataset {dataset_id} has "
                f"target_dependency={manifest.target_dependency!r}; expected 'none'."
            )
        if (
            not spec.is_unbiased
            and dataset_id in EXPLORATORY_DATASET_IDS
            and str(manifest.target_dependency) != "exploratory"
        ):
            failures.append(
                f"Exploratory dataset {dataset_id} must declare "
                "target_dependency=exploratory."
            )
        package_records[dataset_id] = {
            "dataset_id": dataset_id,
            "parent_dataset_id": manifest.parent_dataset_id,
            "target_dependency": manifest.target_dependency,
            "n_features": manifest.n_features,
            "hypothesis": manifest.hypothesis,
            "schema_hash": manifest.schema_hash,
            "train_content_hash": manifest.content_hashes["X_train"],
            "target_hash": manifest.target.hash,
            "train_row_identity_hash": manifest.row_identity.train_hash,
            "dataset_manifest_sha256": dataset_manifest_sha256(package.package_dir),
            "package_dir": to_repo_relative(root, package.package_dir),
            "provenance": provenance,
        }

    if campaign_dir is None:
        campaign_dir = (
            root / Path(*PurePosixPath(spec.artifacts_root).parts) / spec.campaign_id
        )
    campaign_dir = campaign_dir.resolve()

    prepared_configs: dict[str, Path] = {}
    prepared_plans: dict[str, Path] = {}
    resolved_cells: list[dict[str, Any]] = []

    for cell in cells:
        snapshot = snapshots.get(cell.model_family)
        package = package_records.get(cell.dataset_id)
        if snapshot is None or package is None:
            continue
        identity = cell_identity_payload(
            campaign_id=spec.campaign_id,
            dataset_id=cell.dataset_id,
            model_family=cell.model_family,
            adapter_id=snapshot.adapter_id,
            config_sha256=snapshot.config_sha256,
            protocol_sha256=snapshot.protocol_sha256,
            package_identity={
                "schema_hash": package["schema_hash"],
                "train_content_hash": package["train_content_hash"],
                "target_hash": package["target_hash"],
                "train_row_identity_hash": package["train_row_identity_hash"],
                "dataset_manifest_sha256": package["dataset_manifest_sha256"],
            },
        )
        cell_id = cell_id_for(identity)
        prepared_config_rel: str | None = None
        prepared_plan_rel: str | None = None
        if materialize:
            try:
                config_path, plan_path, rels = _materialize_cell(
                    root=root,
                    campaign_dir=campaign_dir,
                    cell=cell,
                    snapshot=snapshot,
                    package_dir=root / package["package_dir"],
                    processed_root=spec.processed_root,
                    validate_prepared=validate_prepared,
                )
                prepared_configs[cell_id] = config_path
                prepared_plans[cell_id] = plan_path
                prepared_config_rel, prepared_plan_rel = rels
            except Exception as error:  # noqa: BLE001 - collect all failures
                failures.append(
                    f"Cell {cell.dataset_id} × {cell.model_family} materialization "
                    f"failed: {error}"
                )
                continue

        resolved_cells.append(
            {
                "cell_id": cell_id,
                "execution_order": cell.execution_order,
                "dataset_id": cell.dataset_id,
                "parent_dataset_id": package["parent_dataset_id"],
                "target_dependency": package["target_dependency"],
                "package": deepcopy(package),
                "model_family": cell.model_family,
                "adapter_id": snapshot.adapter_id,
                "base_config_path": snapshot.config_path,
                "base_config_sha256": snapshot.config_sha256,
                "evaluation_plan_path": snapshot.evaluation_plan_path,
                "evaluation_plan_sha256": snapshot.evaluation_plan_sha256,
                "protocol_sha256": snapshot.protocol_sha256,
                "protocol_identity": deepcopy(snapshot.protocol_identity),
                "cell_identity": identity,
                "prepared_config_relative": prepared_config_rel,
                "prepared_plan_relative": prepared_plan_rel,
                "pipeline_id": REGISTERED_PREPARED_PASSTHROUGH_V1,
                "pipeline_contract": deepcopy(REGISTERED_PREPARED_PASSTHROUGH_CONTRACT),
            }
        )

    expected_count = len(spec.dataset_ids) * len(spec.models)
    if not failures and len(resolved_cells) != expected_count:
        failures.append(
            f"Resolved cell count {len(resolved_cells)} != expected {expected_count}."
        )

    if (
        spec.is_unbiased
        and set(spec.dataset_ids) == set(UNBIASED_SCREENING_DATASET_IDS)
        and len(spec.models) == 3
        and not failures
        and len(resolved_cells) != 21
    ):
        failures.append(
            f"Primary unbiased matrix must resolve to exactly 21 cells; "
            f"got {len(resolved_cells)}."
        )

    manifest_body = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "campaign_contract_version": CAMPAIGN_CONTRACT_VERSION,
        "campaign_id": spec.campaign_id,
        "campaign_name": spec.campaign_name,
        "campaign_type": spec.campaign_type,
        "classification": spec.classification,
        "source_spec_sha256": spec.source_sha256,
        "execution_policy": spec.execution_policy,
        "processed_root": spec.processed_root,
        "artifacts_root": spec.artifacts_root,
        "index_mlflow": spec.index_mlflow,
        "mlflow_config_path": spec.mlflow_config_path,
        "dataset_ids": list(spec.dataset_ids),
        "model_families": [model.family for model in spec.models],
        "evaluation_plan_path": (
            next(iter(snapshots.values())).evaluation_plan_path if snapshots else None
        ),
        "protocol_sha256": (
            next(iter(snapshots.values())).protocol_sha256 if snapshots else None
        ),
        "packages": package_records,
        "cells": resolved_cells,
        "cell_count": len(resolved_cells),
    }
    manifest_hash = canonical_sha256(manifest_body)
    manifest = {
        **manifest_body,
        "manifest_hash": manifest_hash,
    }

    if failures:
        raise CampaignValidationError(
            "Campaign resolution failed with "
            f"{len(failures)} error(s):\n- " + "\n- ".join(failures),
            failures=failures,
        )

    return ResolvedCampaign(
        spec=spec,
        manifest=manifest,
        manifest_hash=manifest_hash,
        prepared_configs=prepared_configs,
        prepared_plans=prepared_plans,
        validation_failures=tuple(failures),
    )


def _materialize_cell(
    *,
    root: Path,
    campaign_dir: Path,
    cell: MatrixCellRef,
    snapshot: ModelConfigSnapshot,
    package_dir: Path,
    processed_root: str,
    validate_prepared: bool,
) -> tuple[Path, Path, tuple[str, str]]:
    base_config = yaml.safe_load(
        (root / PurePosixPath(snapshot.config_path)).read_text(encoding="utf-8")
    )
    base_plan = yaml.safe_load(
        (root / PurePosixPath(snapshot.evaluation_plan_path)).read_text(
            encoding="utf-8"
        )
    )
    mode = _infer_mode(snapshot.config_path)
    experiment_id = make_experiment_id(
        dataset_id=cell.dataset_id,
        adapter_id=snapshot.adapter_id,
        mode=mode,
    )
    plan_id = make_plan_id(
        dataset_id=cell.dataset_id,
        base_plan=base_plan,
        mode=mode,
    )
    dataset_section = build_plan_dataset_section(
        package_dir=package_dir,
        dataset_id=cell.dataset_id,
        processed_dir_relative=processed_root,
    )
    plan_payload = build_evaluation_plan_payload(
        base_plan=base_plan,
        dataset_section=dataset_section,
        plan_id=plan_id,
    )
    prepared_root = campaign_dir / PREPARED_SUBDIR
    plan_rel = f"{PREPARED_SUBDIR}/plans/{plan_id}.yaml"
    config_rel = f"{PREPARED_SUBDIR}/{experiment_id}.yaml"
    plan_path = campaign_dir / PurePosixPath(plan_rel)
    config_path = campaign_dir / PurePosixPath(config_rel)

    try:
        campaign_repo_rel = to_repo_relative(root, campaign_dir)
        plan_repo_rel = f"{campaign_repo_rel}/{plan_rel}".replace("\\", "/")
        config_repo_rel = f"{campaign_repo_rel}/{config_rel}".replace("\\", "/")
    except CampaignConfigurationError:
        # Validate-only may stage under a temporary directory outside the repo.
        plan_repo_rel = str(plan_path.resolve())
        config_repo_rel = str(config_path.resolve())

    config_payload = build_experiment_config_payload(
        base_config=base_config,
        dataset_id=cell.dataset_id,
        experiment_id=experiment_id,
        evaluation_plan_relative=(
            plan_repo_rel
            if not Path(plan_repo_rel).is_absolute()
            else plan_path.resolve().as_posix()
        ),
    )

    # Research v2 configs require repository-relative evaluation_plan_path.
    # When staging outside the repository, rewrite the plan path after we know
    # whether the campaign directory is inside the repo.
    if Path(plan_repo_rel).is_absolute() or PurePosixPath(plan_repo_rel).is_absolute():
        # Cannot satisfy Research v2 repo-relative plan path outside the repo.
        # Write an absolute path and skip load_research_v2_config; structural
        # YAML validity is still checked below.
        config_payload["evaluation_plan_path"] = plan_path.resolve().as_posix()
        skip_research_load = True
    else:
        skip_research_load = False

    plan_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_root.mkdir(parents=True, exist_ok=True)
    plan_text = yaml.safe_dump(
        plan_payload, sort_keys=False, allow_unicode=True, default_flow_style=False
    )
    config_text = yaml.safe_dump(
        config_payload, sort_keys=False, allow_unicode=True, default_flow_style=False
    )
    _write_if_absent_or_identical(plan_path, plan_text)
    _write_if_absent_or_identical(config_path, config_text)

    if validate_prepared and not skip_research_load:
        from src.churn_ml.research_v2_config import (
            ResearchV2ConfigurationError,
            load_research_v2_config,
        )

        try:
            loaded = load_research_v2_config(config_path, project_root=root)
        except ResearchV2ConfigurationError as error:
            raise CampaignConfigurationError(
                f"Prepared config failed Research v2 validation: {error}"
            ) from error
        if loaded.pipeline_id != REGISTERED_PREPARED_PASSTHROUGH_V1:
            raise CampaignConfigurationError(
                "Prepared configuration must use registered_prepared_passthrough_v1."
            )
        if loaded.adapter_id != snapshot.adapter_id:
            raise CampaignConfigurationError(
                f"Prepared adapter {loaded.adapter_id} != {snapshot.adapter_id}."
            )
    elif validate_prepared and skip_research_load:
        # Structural checks only for out-of-repo staging.
        if config_payload["feature_pipeline"]["id"] != REGISTERED_PREPARED_PASSTHROUGH_V1:
            raise CampaignConfigurationError(
                "Prepared configuration must use registered_prepared_passthrough_v1."
            )
        if config_payload["candidate_adapter"]["id"] != snapshot.adapter_id:
            raise CampaignConfigurationError(
                f"Prepared adapter {config_payload['candidate_adapter']['id']} "
                f"!= {snapshot.adapter_id}."
            )

    return config_path, plan_path, (config_repo_rel, plan_repo_rel)


def _write_if_absent_or_identical(path: Path, text: str) -> None:
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != text:
            raise CampaignConfigurationError(
                f"Refusing to overwrite differing prepared file: {path}."
            )
        return
    path.write_text(text, encoding="utf-8")


def _infer_mode(config_path: str) -> str:
    stem = PurePosixPath(config_path).stem.lower()
    for token in ("smoke", "development", "deployment"):
        if token in stem.split("_"):
            return token
    raise CampaignConfigurationError(
        f"Could not infer evaluation mode from config path {config_path!r}."
    )
