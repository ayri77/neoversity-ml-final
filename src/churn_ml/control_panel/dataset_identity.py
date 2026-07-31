"""Run-local and config-local dataset identity for Control Panel presentation.

Historical Experiment Core results must be labeled from artifacts that travel
with the run. This module never consults the live data/processed Registry to
identify a past run.

Unsafe preferred metadata fails visibly and must not silently downgrade to a
weaker source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

from src.churn_ml.control_panel.path_safety import (
    PathSafetyError,
    path_exists_nonfollowing,
    require_regular_file,
    require_safe_directory,
)
from src.churn_ml.control_panel.presentation import (
    normalize_mode,
    normalize_model_family,
    parse_config_metadata,
)


_NA = "Not available"
_MAX_BYTES = 32 * 1024


class DatasetIdentityConflictError(ValueError):
    """Raised when run-local dataset identities contradict each other."""


class DatasetIdentityUnsafeError(ValueError):
    """Raised when preferred run-local metadata is present but unsafe."""


class DatasetIdentityMalformedError(ValueError):
    """Raised when preferred run-local metadata is present but unreadable."""


class MetadataReadStatus(str, Enum):
    ABSENT = "absent"
    OK = "ok"
    UNSAFE = "unsafe"
    MALFORMED = "malformed"
    OVERSIZED = "oversized"


@dataclass(frozen=True)
class MetadataReadResult:
    status: MetadataReadStatus
    payload: dict[str, Any] | None = None
    diagnostic: str | None = None


@dataclass(frozen=True)
class DatasetIdentityView:
    """Presentation-facing dataset identity extracted from local metadata."""

    dataset_id: str | None
    parent_dataset_id: str | None
    target_dependency: str | None
    n_features: int | None
    hypothesis: str | None
    schema_hash: str | None
    train_content_hash: str | None
    target_hash: str | None
    train_row_identity_hash: str | None
    registry_schema_version: str | None
    source: str
    diagnostic: str | None = None

    @property
    def is_exploratory(self) -> bool:
        return self.target_dependency == "exploratory"

    def display(self, field: str) -> str:
        value = getattr(self, field, None)
        if value is None or value == "":
            return _NA
        return str(value)


@dataclass(frozen=True)
class ExperimentCoreLaunchIdentity:
    """Stable identity persisted into Experiment Core UI job references."""

    dataset_id: str
    experiment_id: str
    plan_id: str
    model_family: str
    mode: str
    config: str


def read_dataset_identity(run_root: Path) -> DatasetIdentityView:
    """Extract dataset identity from a completed (or partial) run directory.

    Preference order when consistent and safe:
    1. ``dataset_provenance.json``
    2. ``run_metadata.json → dataset_provenance``
    3. ``resolved_config.yaml → dataset.version``
    4. ``dataset_fingerprints.json → dataset_version``

    Conflicting non-empty identities raise ``DatasetIdentityConflictError``.
    Unsafe or malformed preferred provenance raises explicitly and must not
    silently fall back to a weaker source.
    """
    root = _require_run_root(run_root)
    provenance = _read_json_mapping(root / "dataset_provenance.json")
    metadata = _read_json_mapping(root / "run_metadata.json")
    fingerprints = _read_json_mapping(root / "dataset_fingerprints.json")
    resolved = _read_yaml_mapping(root / "resolved_config.yaml")

    _raise_if_preferred_unusable(
        "dataset_provenance.json",
        provenance,
        preferred=True,
    )

    embedded = None
    if metadata.status == MetadataReadStatus.OK and isinstance(metadata.payload, dict):
        candidate = metadata.payload.get("dataset_provenance")
        if isinstance(candidate, dict):
            embedded = candidate
    elif metadata.status in {
        MetadataReadStatus.UNSAFE,
        MetadataReadStatus.MALFORMED,
        MetadataReadStatus.OVERSIZED,
    }:
        # run_metadata is a preferred cross-check source when provenance is
        # absent; if provenance exists and is OK we already use it. When
        # provenance is absent, unsafe run_metadata must not be ignored.
        if provenance.status == MetadataReadStatus.ABSENT:
            raise DatasetIdentityUnsafeError(
                f"Unsafe or unreadable run_metadata.json: {metadata.diagnostic}"
            )

    fingerprint_embedded = None
    if fingerprints.status == MetadataReadStatus.OK and isinstance(
        fingerprints.payload, dict
    ):
        candidate = fingerprints.payload.get("dataset_provenance")
        if isinstance(candidate, dict):
            fingerprint_embedded = candidate

    if provenance.status == MetadataReadStatus.ABSENT and embedded is None:
        # Prefer provenance file when present; otherwise embedded provenance in
        # run_metadata is the preferred Registry-backed form.
        pass

    ids: dict[str, str] = {}
    if provenance.status == MetadataReadStatus.OK and provenance.payload:
        dataset_id = provenance.payload.get("dataset_id")
        if dataset_id:
            ids["dataset_provenance.json"] = str(dataset_id)
    if isinstance(embedded, dict) and embedded.get("dataset_id"):
        ids["run_metadata.dataset_provenance"] = str(embedded["dataset_id"])
    if isinstance(fingerprint_embedded, dict) and fingerprint_embedded.get(
        "dataset_id"
    ):
        ids["dataset_fingerprints.dataset_provenance"] = str(
            fingerprint_embedded["dataset_id"]
        )

    resolved_version = None
    if resolved.status == MetadataReadStatus.OK:
        resolved_version = _resolved_dataset_version(resolved.payload)
        if resolved_version:
            ids["resolved_config.dataset.version"] = resolved_version
    elif resolved.status in {
        MetadataReadStatus.UNSAFE,
        MetadataReadStatus.MALFORMED,
        MetadataReadStatus.OVERSIZED,
    }:
        if provenance.status == MetadataReadStatus.ABSENT and embedded is None:
            raise DatasetIdentityUnsafeError(
                f"Unsafe or unreadable resolved_config.yaml: {resolved.diagnostic}"
            )

    fingerprint_version = None
    if fingerprints.status == MetadataReadStatus.OK and fingerprints.payload:
        if fingerprints.payload.get("dataset_version"):
            fingerprint_version = str(fingerprints.payload["dataset_version"])
            ids["dataset_fingerprints.dataset_version"] = fingerprint_version
    elif fingerprints.status in {
        MetadataReadStatus.UNSAFE,
        MetadataReadStatus.MALFORMED,
        MetadataReadStatus.OVERSIZED,
    }:
        if (
            provenance.status == MetadataReadStatus.ABSENT
            and embedded is None
            and resolved_version is None
        ):
            raise DatasetIdentityUnsafeError(
                "Unsafe or unreadable dataset_fingerprints.json: "
                f"{fingerprints.diagnostic}"
            )

    unique_ids = sorted({value for value in ids.values() if value})
    if len(unique_ids) > 1:
        detail = ", ".join(f"{key}={value!r}" for key, value in sorted(ids.items()))
        raise DatasetIdentityConflictError(
            "Conflicting run-local dataset identities: " + detail
        )

    primary = _first_mapping(
        provenance.payload if provenance.status == MetadataReadStatus.OK else None,
        embedded,
        fingerprint_embedded,
    )
    dataset_id = unique_ids[0] if unique_ids else None
    if primary is not None:
        return DatasetIdentityView(
            dataset_id=str(primary.get("dataset_id") or dataset_id or "") or None,
            parent_dataset_id=_optional_str(primary.get("parent_dataset_id")),
            target_dependency=_optional_str(primary.get("target_dependency")),
            n_features=_optional_int(primary.get("n_features")),
            hypothesis=_optional_str(primary.get("hypothesis")),
            schema_hash=_optional_str(primary.get("schema_hash")),
            train_content_hash=_optional_str(primary.get("train_content_hash")),
            target_hash=_optional_str(primary.get("target_hash")),
            train_row_identity_hash=_optional_str(
                primary.get("train_row_identity_hash")
            ),
            registry_schema_version=_optional_str(
                primary.get("registry_schema_version")
            ),
            source="dataset_provenance",
            diagnostic=None,
        )

    if dataset_id:
        source = "resolved_config" if resolved_version == dataset_id else "fingerprints"
        return DatasetIdentityView(
            dataset_id=dataset_id,
            parent_dataset_id=None,
            target_dependency=None,
            n_features=None,
            hypothesis=None,
            schema_hash=None,
            train_content_hash=None,
            target_hash=None,
            train_row_identity_hash=None,
            registry_schema_version=None,
            source=source,
            diagnostic=None,
        )

    return DatasetIdentityView(
        dataset_id=None,
        parent_dataset_id=None,
        target_dependency=None,
        n_features=None,
        hypothesis=None,
        schema_hash=None,
        train_content_hash=None,
        target_hash=None,
        train_row_identity_hash=None,
        registry_schema_version=None,
        source="missing",
        diagnostic=None,
    )


def read_dataset_identity_safe(run_root: Path) -> DatasetIdentityView:
    """Like ``read_dataset_identity`` but returns a diagnostic view on failures."""
    try:
        return read_dataset_identity(run_root)
    except (
        DatasetIdentityConflictError,
        DatasetIdentityUnsafeError,
        DatasetIdentityMalformedError,
        PathSafetyError,
    ) as error:
        source = "conflict"
        if isinstance(error, DatasetIdentityUnsafeError | PathSafetyError):
            source = "unsafe"
        elif isinstance(error, DatasetIdentityMalformedError):
            source = "malformed"
        return DatasetIdentityView(
            dataset_id=None,
            parent_dataset_id=None,
            target_dependency=None,
            n_features=None,
            hypothesis=None,
            schema_hash=None,
            train_content_hash=None,
            target_hash=None,
            train_row_identity_hash=None,
            registry_schema_version=None,
            source=source,
            diagnostic=str(error),
        )


def read_config_dataset_version(
    repository_root: Path, config_relative: str
) -> str | None:
    """Read ``dataset.version`` from a repository-relative Research v2 config."""
    path = _safe_repo_file(repository_root, config_relative)
    if path is None:
        return None
    payload = _read_yaml_mapping(path)
    if payload.status != MetadataReadStatus.OK:
        return None
    return _resolved_dataset_version(payload.payload)


def read_paired_comparison_datasets(
    comparison_root: Path,
) -> tuple[str | None, str | None]:
    """Return (baseline_dataset_version, candidate_dataset_version)."""
    root = _require_run_root(comparison_root)
    baseline = _read_json_mapping(root / "baseline_run_reference.json")
    candidate = _read_json_mapping(root / "candidate_run_reference.json")
    for label, result in (
        ("baseline_run_reference.json", baseline),
        ("candidate_run_reference.json", candidate),
    ):
        if result.status in {
            MetadataReadStatus.UNSAFE,
            MetadataReadStatus.MALFORMED,
            MetadataReadStatus.OVERSIZED,
        }:
            raise DatasetIdentityUnsafeError(
                f"Unsafe or unreadable {label}: {result.diagnostic}"
            )
    left = None
    right = None
    if baseline.status == MetadataReadStatus.OK and baseline.payload:
        if baseline.payload.get("dataset_version"):
            left = str(baseline.payload["dataset_version"])
    if candidate.status == MetadataReadStatus.OK and candidate.payload:
        if candidate.payload.get("dataset_version"):
            right = str(candidate.payload["dataset_version"])
    return left, right


def extract_experiment_core_launch_identity(
    repository_root: Path,
    *,
    config_relative: str,
) -> ExperimentCoreLaunchIdentity | None:
    """Derive persistable job identity from an authorized Experiment Core config."""
    root = Path(repository_root).resolve()
    relative = PurePosixPath(str(config_relative).replace("\\", "/")).as_posix()
    path = _safe_repo_file(root, relative)
    if path is None:
        return None
    payload_result = _read_yaml_mapping(path)
    if payload_result.status != MetadataReadStatus.OK:
        return None
    payload = payload_result.payload
    if not isinstance(payload, dict):
        return None
    dataset = payload.get("dataset")
    experiment = payload.get("experiment")
    adapter = payload.get("candidate_adapter")
    if not isinstance(dataset, Mapping) or not dataset.get("version"):
        return None
    if not isinstance(experiment, Mapping) or not experiment.get("id"):
        return None
    dataset_id = str(dataset["version"])
    experiment_id = str(experiment["id"])
    adapter_id = ""
    if isinstance(adapter, Mapping) and adapter.get("id"):
        adapter_id = str(adapter["id"])
    meta = parse_config_metadata(relative, root)
    model_family = meta.get("model_family") or (
        normalize_model_family(adapter_id) if adapter_id else "Unknown"
    )
    mode = meta.get("mode") or _infer_mode(relative, experiment_id)
    plan_id = _plan_id_from_config(root, payload) or _NA
    return ExperimentCoreLaunchIdentity(
        dataset_id=dataset_id,
        experiment_id=experiment_id,
        plan_id=plan_id,
        model_family=model_family,
        mode=mode,
        config=relative,
    )


def enrich_experiment_core_references(
    references: Mapping[str, str | int],
    *,
    repository_root: Path,
    command_id: str,
) -> dict[str, str | int]:
    """Add dataset identity keys to Experiment Core job references."""
    enriched = dict(references)
    if command_id != "experiment_core_v2":
        return enriched
    config = enriched.get("config")
    if not isinstance(config, str) or not config:
        return enriched
    identity = extract_experiment_core_launch_identity(
        repository_root, config_relative=config
    )
    if identity is None:
        return enriched
    enriched["dataset_id"] = identity.dataset_id
    enriched["experiment_id"] = identity.experiment_id
    enriched["plan_id"] = identity.plan_id
    enriched["model_family"] = identity.model_family
    enriched["mode"] = identity.mode
    return enriched


def _plan_id_from_config(
    repository_root: Path, payload: Mapping[str, Any]
) -> str | None:
    plan_path = payload.get("evaluation_plan_path")
    if isinstance(plan_path, str) and plan_path:
        plan_file = _safe_repo_file(repository_root, plan_path)
        if plan_file is not None:
            plan = _read_yaml_mapping(plan_file)
            if plan.status == MetadataReadStatus.OK and isinstance(plan.payload, dict):
                section = plan.payload.get("plan")
                if isinstance(section, Mapping) and section.get("id"):
                    return str(section["id"])
    embedded = payload.get("evaluation_plan")
    if isinstance(embedded, Mapping):
        section = embedded.get("plan")
        if isinstance(section, Mapping) and section.get("id"):
            return str(section["id"])
    return None


def _infer_mode(config_relative: str, experiment_id: str) -> str:
    tokens = (
        PurePosixPath(config_relative).stem.lower().split("_")
        + experiment_id.lower().split("_")
    )
    for token in ("smoke", "development", "deployment"):
        if token in tokens:
            return normalize_mode(token)
    return "Unknown"


def _resolved_dataset_version(payload: Mapping[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    dataset = payload.get("dataset")
    if isinstance(dataset, Mapping) and dataset.get("version"):
        return str(dataset["version"])
    return None


def _first_mapping(*candidates: Any) -> dict[str, Any] | None:
    for item in candidates:
        if isinstance(item, dict) and item:
            return item
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _raise_if_preferred_unusable(
    label: str, result: MetadataReadResult, *, preferred: bool
) -> None:
    if not preferred:
        return
    if result.status == MetadataReadStatus.UNSAFE:
        raise DatasetIdentityUnsafeError(
            f"Unsafe preferred metadata {label}: {result.diagnostic}"
        )
    if result.status in {
        MetadataReadStatus.MALFORMED,
        MetadataReadStatus.OVERSIZED,
    }:
        raise DatasetIdentityMalformedError(
            f"Unreadable preferred metadata {label}: {result.diagnostic}"
        )


def _require_run_root(run_root: Path) -> Path:
    root = Path(run_root)
    try:
        return require_safe_directory(root)
    except PathSafetyError as error:
        raise DatasetIdentityUnsafeError(str(error)) from error


def _safe_repo_file(repository_root: Path, relative: str) -> Path | None:
    posix = PurePosixPath(str(relative).replace("\\", "/"))
    if (
        not relative
        or posix.is_absolute()
        or ".." in posix.parts
        or any(part in {"", "."} for part in posix.parts)
    ):
        return None
    root = Path(repository_root).resolve()
    path = root.joinpath(*posix.parts)
    try:
        path.resolve().relative_to(root)
    except ValueError:
        return None
    if not path_exists_nonfollowing(path):
        return None
    try:
        require_regular_file(path, reject_hardlinks=True)
    except PathSafetyError:
        return None
    return path


def _read_json_mapping(path: Path) -> MetadataReadResult:
    return _read_mapping(path, kind="json")


def _read_yaml_mapping(path: Path) -> MetadataReadResult:
    return _read_mapping(path, kind="yaml")


def _read_mapping(path: Path, *, kind: str) -> MetadataReadResult:
    try:
        if not path_exists_nonfollowing(path):
            return MetadataReadResult(status=MetadataReadStatus.ABSENT)
        require_regular_file(path, reject_hardlinks=True)
    except PathSafetyError as error:
        return MetadataReadResult(
            status=MetadataReadStatus.UNSAFE,
            diagnostic=str(error),
        )
    except OSError as error:
        return MetadataReadResult(
            status=MetadataReadStatus.UNSAFE,
            diagnostic=str(error),
        )
    try:
        size = path.lstat().st_size
    except OSError as error:
        return MetadataReadResult(
            status=MetadataReadStatus.UNSAFE,
            diagnostic=str(error),
        )
    if size > _MAX_BYTES:
        return MetadataReadResult(
            status=MetadataReadStatus.OVERSIZED,
            diagnostic=f"{path.name} exceeds {_MAX_BYTES} bytes",
        )
    try:
        text = path.read_text(encoding="utf-8")
        if kind == "json":
            payload = json.loads(text)
        else:
            payload = yaml.safe_load(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as error:
        return MetadataReadResult(
            status=MetadataReadStatus.MALFORMED,
            diagnostic=str(error),
        )
    if not isinstance(payload, dict):
        return MetadataReadResult(
            status=MetadataReadStatus.MALFORMED,
            diagnostic=f"{path.name} must contain a mapping",
        )
    return MetadataReadResult(status=MetadataReadStatus.OK, payload=payload)
