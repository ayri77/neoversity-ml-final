"""Prepare dataset-driven Experiment Core v2 configs without Streamlit coupling.

This module discovers Registry packages, clones evaluation protocol from a base
Research v2 template, rebuilds the plan dataset section for the selected package,
and publishes config/plan pairs under the UI editable root with create-if-absent
semantics. Fitting runtimes are imported only inside explicit prepare/validate
calls so Control Panel package import stays free of Experiment Core internals.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import pandas as pd
import yaml

from src.churn_ml.control_panel.config_editor import (
    ConfigEditError,
    PublishedEditableFile,
    publish_editable_text_file,
)
from src.churn_ml.control_panel.presentation import (
    normalize_mode,
    normalize_model_family,
    parse_config_metadata,
)


class DatasetExperimentMaterializerError(ValueError):
    """Raised when a dataset-driven experiment cannot be prepared safely."""


SAFE_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
REGISTERED_PREPARED_PASSTHROUGH_V1 = "registered_prepared_passthrough_v1"
REGISTERED_PREPARED_PASSTHROUGH_CONTRACT: dict[str, Any] = {
    "mode": "registry_prepared_passthrough_v1",
    "drop": [],
    "keep_all_features": True,
}
DEFAULT_PROCESSED_ROOT = "data/processed"
DEFAULT_TEMPLATE_GLOB = "configs/research_v2/*.yaml"
EXCLUDED_TEMPLATE_NAMES = frozenset({"comparison_policy_v1.yaml"})
PLAN_SUBDIR = "plans"


@dataclass(frozen=True)
class BaseTemplateInfo:
    """One canonical Research v2 config usable as a dataset-driven template."""

    relative_path: str
    experiment_id: str
    adapter_id: str
    model_family: str
    mode: str
    evaluation_plan_path: str
    dataset_version: str


@dataclass(frozen=True)
class RegisteredDatasetView:
    """UI-friendly subset of RegisteredDatasetSummary."""

    dataset_id: str
    parent_dataset_id: str | None
    n_features: int
    hypothesis: str
    target_dependency: str
    schema_hash: str
    train_content_hash: str
    target_hash: str
    train_row_identity_hash: str

    @classmethod
    def from_summary(cls, summary: Any) -> RegisteredDatasetView:
        return cls(
            dataset_id=str(summary.dataset_id),
            parent_dataset_id=summary.parent_dataset_id,
            n_features=int(summary.n_features),
            hypothesis=str(summary.hypothesis),
            target_dependency=str(summary.target_dependency),
            schema_hash=str(summary.schema_hash),
            train_content_hash=str(summary.train_content_hash),
            target_hash=str(summary.target_hash),
            train_row_identity_hash=str(summary.train_row_identity_hash),
        )


@dataclass(frozen=True)
class PreparedExperimentBundle:
    """Repository-relative paths for one prepared config/plan pair."""

    config_relative: str
    plan_relative: str
    experiment_id: str
    plan_id: str
    dataset_id: str
    base_config_relative: str
    model_family: str
    mode: str
    adapter_id: str
    selection_fingerprint: str
    reused: bool


def selection_fingerprint(
    *,
    dataset_id: str,
    base_config_relative: str,
    model_family: str,
    mode: str,
) -> str:
    payload = {
        "dataset_id": dataset_id,
        "base_config_relative": PurePosixPath(base_config_relative).as_posix(),
        "model_family": model_family,
        "mode": mode,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def discover_registry_summaries(
    repository_root: Path,
    *,
    processed_root_relative: str = DEFAULT_PROCESSED_ROOT,
) -> list[Any]:
    """Return sorted RegisteredDatasetSummary rows or raise a readable error."""
    from src.churn_ml.dataset_registry import discover_registered_datasets
    from src.churn_ml.dataset_registry.errors import DatasetRegistryError

    root = repository_root.resolve()
    processed = _safe_repo_relative_dir(root, processed_root_relative, "processed root")
    try:
        summaries = discover_registered_datasets(processed)
    except FileNotFoundError as error:
        raise DatasetExperimentMaterializerError(
            f"Dataset Registry root is missing: {processed_root_relative}."
        ) from error
    except DatasetRegistryError as error:
        raise DatasetExperimentMaterializerError(
            f"Dataset Registry discovery failed: {error}"
        ) from error
    except OSError as error:
        raise DatasetExperimentMaterializerError(
            f"Dataset Registry is unreadable: {error}"
        ) from error
    if not summaries:
        raise DatasetExperimentMaterializerError(
            "Dataset Registry contains no strictly validated packages with "
            "canonical dataset_manifest.json."
        )
    return summaries


def list_base_templates(repository_root: Path) -> list[BaseTemplateInfo]:
    """List Research v2 run templates (excludes comparison-policy files)."""
    root = repository_root.resolve()
    templates: list[BaseTemplateInfo] = []
    for path in sorted(root.glob(DEFAULT_TEMPLATE_GLOB)):
        if not path.is_file() or path.name in EXCLUDED_TEMPLATE_NAMES:
            continue
        relative = path.relative_to(root).as_posix()
        try:
            payload = _load_mapping(path)
        except DatasetExperimentMaterializerError:
            continue
        if int(payload.get("schema_version", -1)) != 2:
            continue
        experiment = payload.get("experiment")
        dataset = payload.get("dataset")
        adapter = payload.get("candidate_adapter")
        plan_path = payload.get("evaluation_plan_path")
        if not isinstance(experiment, Mapping) or not isinstance(dataset, Mapping):
            continue
        if not isinstance(adapter, Mapping) or not isinstance(plan_path, str):
            continue
        experiment_id = str(experiment.get("id", ""))
        adapter_id = str(adapter.get("id", ""))
        dataset_version = str(dataset.get("version", ""))
        if not experiment_id or not adapter_id or not dataset_version:
            continue
        meta = parse_config_metadata(relative, root)
        mode = meta.get("mode") or _infer_mode_from_stem(path.stem)
        model_family = meta.get("model_family") or normalize_model_family(adapter_id)
        templates.append(
            BaseTemplateInfo(
                relative_path=relative,
                experiment_id=experiment_id,
                adapter_id=adapter_id,
                model_family=model_family,
                mode=mode,
                evaluation_plan_path=PurePosixPath(plan_path).as_posix(),
                dataset_version=dataset_version,
            )
        )
    return templates


def filter_templates(
    templates: list[BaseTemplateInfo],
    *,
    model_family: str | None = None,
    mode: str | None = None,
) -> list[BaseTemplateInfo]:
    selected = templates
    if model_family:
        selected = [item for item in selected if item.model_family == model_family]
    if mode:
        selected = [item for item in selected if item.mode == mode]
    return selected


def build_plan_dataset_section(
    *,
    package_dir: Path,
    dataset_id: str,
    processed_dir_relative: str,
) -> dict[str, Any]:
    """Rebuild evaluation-plan dataset identity from a validated package."""
    from src.churn_ml.research_data import canonical_sha256
    from src.churn_ml.run_artifacts import fingerprint_file

    train_path = package_dir / "X_train.parquet"
    target_path = package_dir / "y_train.parquet"
    metadata_path = package_dir / "metadata.json"
    for path in (train_path, target_path, metadata_path):
        if not path.is_file():
            raise DatasetExperimentMaterializerError(
                f"Registry package is missing required file: {path.name}."
            )
    X_train = pd.read_parquet(train_path)
    y = pd.read_parquet(target_path).iloc[:, 0]
    feature_names = X_train.columns.tolist()
    return {
        "processed_dir": PurePosixPath(processed_dir_relative).as_posix(),
        "version": dataset_id,
        "files": {
            "train_features": {
                "name": "X_train.parquet",
                "sha256": str(fingerprint_file(train_path)["sha256"]),
            },
            "target": {
                "name": "y_train.parquet",
                "sha256": str(fingerprint_file(target_path)["sha256"]),
            },
            "metadata": {
                "name": "metadata.json",
                "sha256": str(fingerprint_file(metadata_path)["sha256"]),
            },
        },
        "expected_rows": int(len(X_train)),
        "expected_source_features": int(X_train.shape[1]),
        "ordered_source_schema_sha256": _ordered_feature_schema_sha256(feature_names),
        "ordered_dtype_schema_sha256": canonical_sha256(
            [
                {"name": name, "dtype": str(dtype)}
                for name, dtype in X_train.dtypes.items()
            ]
        ),
        "target": {
            "name": str(y.name),
            "dtype": str(y.dtype),
            "negative_label": 0,
            "positive_label": 1,
            "expected_negative_rows": int((y == 0).sum()),
            "expected_positive_rows": int((y == 1).sum()),
            "values_sha256": canonical_sha256(
                {
                    "name": str(y.name),
                    "dtype": str(y.dtype),
                    "values": y.tolist(),
                }
            ),
        },
    }


def build_evaluation_plan_payload(
    *,
    base_plan: Mapping[str, Any],
    dataset_section: Mapping[str, Any],
    plan_id: str,
) -> dict[str, Any]:
    _require_slug(plan_id, "plan.id")
    return {
        "schema_version": base_plan["schema_version"],
        "plan": {"id": plan_id},
        "dataset": deepcopy(dict(dataset_section)),
        "outer_evaluation": deepcopy(base_plan["outer_evaluation"]),
        "threshold_selection": deepcopy(base_plan["threshold_selection"]),
        "threshold_policy": deepcopy(base_plan["threshold_policy"]),
        "metrics": deepcopy(base_plan["metrics"]),
        "aggregation": deepcopy(base_plan["aggregation"]),
    }


def build_experiment_config_payload(
    *,
    base_config: Mapping[str, Any],
    dataset_id: str,
    experiment_id: str,
    evaluation_plan_relative: str,
) -> dict[str, Any]:
    _require_slug(dataset_id, "dataset.version")
    _require_slug(experiment_id, "experiment.id")
    plan_rel = PurePosixPath(evaluation_plan_relative.replace("\\", "/")).as_posix()
    return {
        "schema_version": 2,
        "experiment": {"id": experiment_id},
        "dataset": {"version": dataset_id},
        "evaluation_plan_path": plan_rel,
        "feature_pipeline": {
            "id": REGISTERED_PREPARED_PASSTHROUGH_V1,
            "contract": deepcopy(REGISTERED_PREPARED_PASSTHROUGH_CONTRACT),
        },
        "candidate_adapter": {
            "id": str(base_config["candidate_adapter"]["id"]),
            "contract": deepcopy(base_config["candidate_adapter"]["contract"]),
        },
        "artifacts": deepcopy(base_config["artifacts"]),
        "persistence": deepcopy(base_config["persistence"]),
        "tracking": deepcopy(base_config["tracking"]),
    }


def make_experiment_id(*, dataset_id: str, adapter_id: str, mode: str) -> str:
    mode_token = _mode_token(mode)
    return _require_slug(
        f"{dataset_id}__{adapter_id}__{mode_token}",
        "experiment.id",
    )


def make_plan_id(*, dataset_id: str, base_plan: Mapping[str, Any], mode: str) -> str:
    mode_token = _mode_token(mode)
    outer = base_plan["outer_evaluation"]
    threshold = base_plan["threshold_selection"]
    n_repeats = len(outer["repeat_seeds"])
    n_splits = int(outer["n_splits"])
    t_splits = int(threshold["n_splits"])
    return _require_slug(
        f"{dataset_id}__{mode_token}_r{n_repeats}x{n_splits}_t{t_splits}_v1",
        "plan.id",
    )


def prepare_dataset_driven_experiment(
    repository_root: Path,
    *,
    editable_root: str,
    dataset_id: str,
    base_config_relative: str,
    processed_root_relative: str = DEFAULT_PROCESSED_ROOT,
    validate: bool = True,
) -> PreparedExperimentBundle:
    """Materialize a UI-local config/plan pair for one Registry dataset."""
    root = repository_root.resolve()
    _require_slug(dataset_id, "dataset_id")
    base_rel = PurePosixPath(base_config_relative.replace("\\", "/")).as_posix()
    base_path = _safe_repo_relative_file(root, base_rel, "base config")
    base_config = _load_mapping(base_path)
    if int(base_config.get("schema_version", -1)) != 2:
        raise DatasetExperimentMaterializerError(
            "Base template must be a Research v2 configuration."
        )

    meta = parse_config_metadata(base_rel, root)
    adapter_id = str(base_config["candidate_adapter"]["id"])
    model_family = meta.get("model_family") or normalize_model_family(adapter_id)
    mode = meta.get("mode") or _infer_mode_from_stem(base_path.stem)

    base_plan_rel = PurePosixPath(
        str(base_config["evaluation_plan_path"]).replace("\\", "/")
    ).as_posix()
    base_plan_path = _safe_repo_relative_file(
        root, base_plan_rel, "base evaluation plan"
    )
    base_plan = _load_mapping(base_plan_path)

    from src.churn_ml.dataset_registry import resolve_dataset_package
    from src.churn_ml.dataset_registry.errors import DatasetRegistryError

    processed = _safe_repo_relative_dir(root, processed_root_relative, "processed root")
    try:
        package = resolve_dataset_package(processed, dataset_id)
    except DatasetRegistryError as error:
        raise DatasetExperimentMaterializerError(
            f"Selected Dataset Package is not a valid Registry package: {error}"
        ) from error

    experiment_id = make_experiment_id(
        dataset_id=dataset_id, adapter_id=adapter_id, mode=mode
    )
    plan_id = make_plan_id(dataset_id=dataset_id, base_plan=base_plan, mode=mode)
    fingerprint = selection_fingerprint(
        dataset_id=dataset_id,
        base_config_relative=base_rel,
        model_family=model_family,
        mode=mode,
    )

    dataset_section = build_plan_dataset_section(
        package_dir=package.package_dir,
        dataset_id=dataset_id,
        processed_dir_relative=processed_root_relative,
    )
    plan_payload = build_evaluation_plan_payload(
        base_plan=base_plan,
        dataset_section=dataset_section,
        plan_id=plan_id,
    )
    plan_relative = f"{editable_root.rstrip('/')}/{PLAN_SUBDIR}/{plan_id}.yaml"
    config_payload = build_experiment_config_payload(
        base_config=base_config,
        dataset_id=dataset_id,
        experiment_id=experiment_id,
        evaluation_plan_relative=plan_relative,
    )
    config_relative = f"{editable_root.rstrip('/')}/{experiment_id}.yaml"

    plan_result = _publish_or_reuse(
        root,
        editable_root=editable_root,
        relative_path=f"{PLAN_SUBDIR}/{plan_id}.yaml",
        text=_dump_yaml(plan_payload),
    )
    config_result = _publish_or_reuse(
        root,
        editable_root=editable_root,
        relative_path=f"{experiment_id}.yaml",
        text=_dump_yaml(config_payload),
    )
    if validate:
        _validate_prepared_pair(root, config_path=config_result.path)

    return PreparedExperimentBundle(
        config_relative=PurePosixPath(config_relative).as_posix(),
        plan_relative=PurePosixPath(plan_relative).as_posix(),
        experiment_id=experiment_id,
        plan_id=plan_id,
        dataset_id=dataset_id,
        base_config_relative=base_rel,
        model_family=model_family,
        mode=mode,
        adapter_id=adapter_id,
        selection_fingerprint=fingerprint,
        reused=bool(plan_result.reused and config_result.reused),
    )


def build_dataset_driven_pre_run_summary(
    bundle: PreparedExperimentBundle,
    *,
    summary: RegisteredDatasetView | None = None,
    pipeline_id: str = REGISTERED_PREPARED_PASSTHROUGH_V1,
) -> dict[str, str]:
    parent = "—"
    target_dependency = "—"
    if summary is not None:
        parent = summary.parent_dataset_id or "—"
        target_dependency = summary.target_dependency
    return {
        "Dataset Package": bundle.dataset_id,
        "Parent dataset": parent,
        "Target dependency": target_dependency,
        "Pipeline": pipeline_id,
        "Model": bundle.model_family,
        "Evaluation mode": bundle.mode,
        "Base template": bundle.base_config_relative,
        "Prepared config": bundle.config_relative,
        "Prepared evaluation plan": bundle.plan_relative,
    }


def _validate_prepared_pair(repository_root: Path, *, config_path: Path) -> None:
    from src.churn_ml.research_v2_config import (
        ResearchV2ConfigurationError,
        load_research_v2_config,
    )

    try:
        loaded = load_research_v2_config(config_path, project_root=repository_root)
    except ResearchV2ConfigurationError as error:
        raise DatasetExperimentMaterializerError(
            f"Prepared configuration failed Research v2 validation: {error}"
        ) from error
    if loaded.dataset_version != loaded.plan_payload["dataset"]["version"]:
        raise DatasetExperimentMaterializerError(
            "Prepared config and evaluation plan dataset versions differ."
        )
    if loaded.pipeline_id != REGISTERED_PREPARED_PASSTHROUGH_V1:
        raise DatasetExperimentMaterializerError(
            "Prepared configuration must use registered_prepared_passthrough_v1."
        )


def _publish_or_reuse(
    repository_root: Path,
    *,
    editable_root: str,
    relative_path: str,
    text: str,
) -> PublishedEditableFile:
    try:
        result = publish_editable_text_file(
            repository_root,
            editable_root=editable_root,
            relative_path=relative_path,
            text=text,
            allow_identical_reuse=True,
        )
    except ConfigEditError as error:
        raise DatasetExperimentMaterializerError(str(error)) from error
    if isinstance(result, PublishedEditableFile):
        return result
    return PublishedEditableFile(path=result, reused=False)


def _dump_yaml(payload: Mapping[str, Any]) -> str:
    return yaml.safe_dump(
        dict(payload),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise DatasetExperimentMaterializerError(
            f"Could not load YAML mapping from {path.name}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise DatasetExperimentMaterializerError(
            f"Expected a top-level mapping in {path.name}."
        )
    return payload


def _ordered_feature_schema_sha256(feature_names: list[str]) -> str:
    encoded = json.dumps(
        feature_names,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_slug(value: str, label: str) -> str:
    if not isinstance(value, str) or SAFE_SLUG.fullmatch(value) is None:
        raise DatasetExperimentMaterializerError(
            f"{label} must be a safe slug matching {SAFE_SLUG.pattern}."
        )
    return value


def _mode_token(mode: str) -> str:
    normalized = normalize_mode(mode).lower()
    if normalized not in {"smoke", "development", "deployment"}:
        raise DatasetExperimentMaterializerError(
            f"Unsupported evaluation mode for materialization: {mode!r}."
        )
    return normalized


def _infer_mode_from_stem(stem: str) -> str:
    tokens = stem.lower().split("_")
    for token in ("smoke", "development", "deployment"):
        if token in tokens:
            return normalize_mode(token)
    raise DatasetExperimentMaterializerError(
        f"Could not infer evaluation mode from template stem {stem!r}."
    )


def _safe_repo_relative_file(root: Path, relative: str, label: str) -> Path:
    path = _safe_repo_relative_path(root, relative, label)
    if not path.is_file():
        raise DatasetExperimentMaterializerError(f"{label} does not exist: {relative}.")
    return path


def _safe_repo_relative_dir(root: Path, relative: str, label: str) -> Path:
    path = _safe_repo_relative_path(root, relative, label)
    if not path.is_dir():
        raise DatasetExperimentMaterializerError(f"{label} does not exist: {relative}.")
    return path


def _safe_repo_relative_path(root: Path, relative: str, label: str) -> Path:
    posix = PurePosixPath(relative.replace("\\", "/"))
    if (
        posix.is_absolute()
        or ".." in posix.parts
        or any(part in {"", "."} for part in posix.parts)
    ):
        raise DatasetExperimentMaterializerError(f"{label} path is unsafe: {relative!r}.")
    path = (root / Path(*posix.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise DatasetExperimentMaterializerError(
            f"{label} escapes the repository root."
        ) from error
    return path
