"""Canonical completed-run candidates and deployment readiness for the UI.

The Generate submission workflow selects a completed canonical Research v2 run,
not a raw YAML path. This module reads only authoritative run sidecars and the
immutable Dataset Package manifest, never fits a model, and never reads
competition test data or a sample submission.

Readiness is deterministic: the same run always produces the same supported or
blocked result with the same ordered reasons. Missing authoritative information
blocks preparation instead of being guessed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

import yaml

from src.churn_ml.control_panel.path_safety import (
    PathSafetyError,
    path_exists_nonfollowing,
    require_regular_file,
    require_safe_directory,
)
from src.churn_ml.control_panel.research_inventory import (
    RESEARCH_V2_ROOT,
    ResearchRunInventoryRow,
)
from src.churn_ml.control_panel.selection_state import (
    set_durable_value,
    ui_durable_key,
)
from src.churn_ml.deployment_v1_contracts import (
    DEPLOYMENT_SCHEMA_VERSION,
    SUPPORTED_ADAPTERS,
    DeploymentContractError,
    model_parameters,
)
from src.churn_ml.experiment_v2_schema import ordered_feature_schema_sha256
from src.churn_ml.research_data import canonical_sha256


DEPLOYMENT_SCHEMA_LABEL = f"deployment_v1 (schema_version {DEPLOYMENT_SCHEMA_VERSION})"
PROCESSED_ROOT = "data/processed"
THRESHOLD_SOURCE_TYPE = "experiment_core_v2_manual_threshold_v1"
UNRESOLVED_SAMPLE_SUBMISSION_PATH = (
    "data/competition/UNRESOLVED_sample_submission.csv"
)
DEFAULT_SUBMISSION_ID_COLUMN = "index"
DEFAULT_SUBMISSION_TARGET_COLUMN = "y"
SUBMISSION_COMMAND_ID = "final_deployment_v1"
SUBMISSION_ACTION_ID = "validate"
SUBMISSION_HANDOFF_KEY = "deployment_candidate_handoff"
CANDIDATE_WIDGET_KEY = "deploy-candidate-run"

_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_SIDECAR_BYTES = 4_000_000


@dataclass(frozen=True)
class CanonicalRunFacts:
    """Authoritative deployment-relevant facts derived from one completed run."""

    run_relative_path: str
    run_id: str | None
    manifest_sha256: str | None
    plan_id: str | None
    plan_sha256: str | None
    pipeline_id: str | None
    pipeline_sha256: str | None
    adapter_id: str | None
    adapter_sha256: str | None
    candidate_sha256: str | None
    dataset_identity_sha256: str | None
    dataset_version: str | None
    dataset_target_dependency: str | None
    dataset_schema_hash: str | None
    experiment_id: str | None
    threshold_value: float | None
    threshold_policy_id: str | None
    fixed_parameters: dict[str, Any] | None
    test_data_path: str | None
    test_data_sha256: str | None
    test_expected_rows: int | None
    test_ordered_schema_sha256: str | None
    status: str
    competition_test_assets_accessed: bool = False
    missing: tuple[str, ...] = field(default_factory=tuple)

    @property
    def candidate_id(self) -> str | None:
        """Deterministic safe candidate identity for draft storage and IDs."""
        if self.run_id is None or self.manifest_sha256 is None:
            return None
        candidate = f"{self.run_id}-{self.manifest_sha256[:12]}"
        return candidate if _SLUG.fullmatch(candidate) else None


@dataclass(frozen=True)
class DeploymentReadiness:
    """Deterministic deployment-capability report for one selected run."""

    supported: bool
    run_relative_path: str
    blocking_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    facts: CanonicalRunFacts
    deployment_schema: str = DEPLOYMENT_SCHEMA_LABEL
    competition_test_assets_accessed: bool = False

    def summary(self) -> dict[str, str]:
        """Ordered report fields for the readiness panel."""
        facts = self.facts
        return {
            "Readiness": "supported" if self.supported else "blocked",
            "Run": self.run_relative_path,
            "Run ID": _display(facts.run_id),
            "Run manifest": _display(facts.manifest_sha256),
            "Candidate ID": _display(facts.candidate_id),
            "Dataset ID": _display(facts.dataset_version),
            "Dataset identity": _display(facts.dataset_identity_sha256),
            "Dataset target dependency": _display(facts.dataset_target_dependency),
            "Model adapter": _display(facts.adapter_id),
            "Model candidate identity": _display(facts.candidate_sha256),
            "Feature pipeline": _display(facts.pipeline_id),
            "Source experiment": _display(facts.experiment_id),
            "Resolved evaluation plan": _display(facts.plan_sha256),
            "Selected threshold": _display(facts.threshold_value),
            "Threshold policy": _display(facts.threshold_policy_id),
            "Threshold evidence source": THRESHOLD_SOURCE_TYPE,
            "Competition-test assets accessed": (
                "true" if self.competition_test_assets_accessed else "false"
            ),
            "Deployment schema to produce": self.deployment_schema,
        }


@dataclass(frozen=True)
class DeploymentCandidate:
    """One canonical completed run offered as a deployment candidate."""

    relative_path: str
    label: str
    dataset_id: str
    model_family: str
    model_config_id: str
    evaluation_mode: str
    balanced_accuracy: float | None
    run_identity: str
    exploratory: bool


def read_canonical_run_facts(
    repository_root: Path, run_relative_path: str
) -> CanonicalRunFacts:
    """Read deployment-relevant facts from one Research v2 run directory."""
    root = repository_root.resolve()
    relative = _normalized_relative(run_relative_path)
    missing: list[str] = []
    if relative is None:
        return _empty_facts(run_relative_path, ("Run path is not repository-safe.",))
    run_dir = root / Path(*PurePosixPath(relative).parts)
    try:
        require_safe_directory(run_dir)
    except PathSafetyError as error:
        return _empty_facts(relative, (f"Run directory is unsafe: {error}",))
    if not relative.startswith(f"{RESEARCH_V2_ROOT}/"):
        return _empty_facts(
            relative,
            (f"Run must be located under {RESEARCH_V2_ROOT}.",),
        )

    status = _status_from_markers(run_dir)
    metadata = _read_json(run_dir / "run_metadata.json") or {}
    manifest = _read_json(run_dir / "artifact_manifest.json") or {}
    resolved = _read_yaml(run_dir / "resolved_config.yaml") or {}
    fingerprints = _read_json(run_dir / "dataset_fingerprints.json")
    thresholds = _read_json(run_dir / "thresholds" / "threshold_summary.json") or {}
    identities = {
        name: _read_json(run_dir / "identities" / f"{name}.json") or {}
        for name in (
            "evaluation_plan",
            "feature_pipeline",
            "candidate_adapter",
            "candidate",
        )
    }

    run_id = _slug_or_none(metadata.get("run_id"))
    if run_id is None:
        missing.append("Run identity (run_metadata.run_id) is unavailable.")
    manifest_sha256 = _sha_or_none(manifest.get("manifest_sha256"))
    if manifest_sha256 is None:
        missing.append("Run artifact manifest hash is unavailable.")

    declared = metadata.get("hashes")
    declared_hashes = declared if isinstance(declared, Mapping) else {}
    identity_hashes: dict[str, str | None] = {}
    for name, key in (
        ("evaluation_plan", "plan"),
        ("feature_pipeline", "feature_pipeline"),
        ("candidate_adapter", "candidate_adapter"),
        ("candidate", "candidate"),
    ):
        value = _sha_or_none(identities[name].get("sha256"))
        if value is None:
            missing.append(f"Identity artifact identities/{name}.json is unavailable.")
        elif _sha_or_none(declared_hashes.get(key)) not in (None, value):
            missing.append(
                f"Identity artifact identities/{name}.json disagrees with "
                "run_metadata.hashes."
            )
            value = None
        identity_hashes[name] = value

    plan_id = _slug_or_none(metadata.get("plan_id")) or _slug_or_none(
        _dig(resolved, "evaluation_plan", "plan", "id")
    )
    if plan_id is None:
        missing.append("Evaluation-plan ID is unavailable.")
    pipeline_id = _slug_or_none(_dig(resolved, "feature_pipeline", "id"))
    if pipeline_id is None:
        missing.append("Feature-pipeline ID is unavailable.")
    adapter_id = _slug_or_none(_dig(resolved, "candidate_adapter", "id"))
    if adapter_id is None:
        missing.append("Candidate-adapter ID is unavailable.")
    elif adapter_id not in SUPPORTED_ADAPTERS:
        missing.append(
            f"Adapter {adapter_id} is not supported by the deployment contract."
        )

    dataset_version = _slug_or_none(_dig(resolved, "dataset", "version"))
    if dataset_version is None:
        missing.append("Dataset version is unavailable in the resolved configuration.")
    dataset_identity_sha256 = (
        canonical_sha256(fingerprints) if isinstance(fingerprints, dict) else None
    )
    if dataset_identity_sha256 is None:
        missing.append("Dataset fingerprints are unavailable.")

    experiment_id = _as_str(_dig(resolved, "experiment", "id"))
    threshold_policy_id = _slug_or_none(
        _dig(resolved, "evaluation_plan", "threshold_policy", "id")
    )
    if threshold_policy_id is None:
        missing.append("Threshold-policy identity is unavailable.")
    threshold_value = _threshold_or_none(thresholds.get("median"))
    if threshold_value is None:
        missing.append(
            "Selected threshold evidence (thresholds/threshold_summary.json median) "
            "is unavailable."
        )

    fixed_parameters: dict[str, Any] | None = None
    contract = _dig(resolved, "candidate_adapter", "contract")
    if adapter_id in SUPPORTED_ADAPTERS and isinstance(contract, Mapping):
        try:
            fixed_parameters = model_parameters(str(adapter_id), contract)
        except (DeploymentContractError, KeyError):
            fixed_parameters = None
    if fixed_parameters is None:
        missing.append("Resolved fixed model parameters are unavailable.")

    package = _dataset_package_facts(root, dataset_version)
    missing.extend(package.pop("missing"))

    if status != "completed":
        missing.append(f"Run status is {status}; only completed runs are supported.")

    return CanonicalRunFacts(
        run_relative_path=relative,
        run_id=run_id,
        manifest_sha256=manifest_sha256,
        plan_id=plan_id,
        plan_sha256=identity_hashes["evaluation_plan"],
        pipeline_id=pipeline_id,
        pipeline_sha256=identity_hashes["feature_pipeline"],
        adapter_id=adapter_id,
        adapter_sha256=identity_hashes["candidate_adapter"],
        candidate_sha256=identity_hashes["candidate"],
        dataset_identity_sha256=dataset_identity_sha256,
        dataset_version=dataset_version,
        dataset_target_dependency=package["target_dependency"],
        dataset_schema_hash=package["schema_hash"],
        experiment_id=experiment_id,
        threshold_value=threshold_value,
        threshold_policy_id=threshold_policy_id,
        fixed_parameters=fixed_parameters,
        test_data_path=package["test_data_path"],
        test_data_sha256=package["test_data_sha256"],
        test_expected_rows=package["test_expected_rows"],
        test_ordered_schema_sha256=package["test_ordered_schema_sha256"],
        status=status,
        missing=tuple(dict.fromkeys(missing)),
    )


def evaluate_deployment_readiness(
    repository_root: Path,
    run_relative_path: str,
    *,
    archived_paths: frozenset[str] = frozenset(),
) -> DeploymentReadiness:
    """Deterministic supported/blocked deployment report for one run."""
    facts = read_canonical_run_facts(repository_root, run_relative_path)
    blocking = list(facts.missing)
    warnings: list[str] = []
    if facts.run_relative_path in archived_paths:
        blocking.append("Run is archived in the Control Panel workspace.")
    if _looks_like_smoke(facts):
        blocking.append("Smoke runs are not deployment candidates.")
    if facts.candidate_id is None and facts.run_id is not None:
        blocking.append("A safe deterministic candidate ID cannot be derived.")
    if facts.dataset_target_dependency == "exploratory":
        warnings.append(
            "The Dataset Package is exploratory (target_dependency: exploratory). "
            "Any submission built from it stays exploratory evidence."
        )
    if facts.test_data_sha256 is not None:
        warnings.append(
            "Competition test identity is taken from the immutable Dataset Package "
            "manifest; the feature matrix itself is not read here."
        )
    warnings.append(
        "Competition submission readiness is evaluated separately from run "
        "readiness: authenticated sample-submission and test-row identity must "
        "resolve before Generate submission can run."
    )
    return DeploymentReadiness(
        supported=not blocking,
        run_relative_path=facts.run_relative_path,
        blocking_reasons=tuple(dict.fromkeys(blocking)),
        warnings=tuple(dict.fromkeys(warnings)),
        facts=facts,
        competition_test_assets_accessed=False,
    )


def list_deployment_candidates(
    rows: list[ResearchRunInventoryRow],
    *,
    archived_paths: frozenset[str] = frozenset(),
    include_exploratory: bool = True,
) -> list[DeploymentCandidate]:
    """Default-filtered canonical candidates in deterministic order."""
    candidates: list[DeploymentCandidate] = []
    for row in rows:
        if row.status != "completed" or not row.identity_complete:
            continue
        if row.relative_path in archived_paths:
            continue
        if _mentions_smoke(
            row.evaluation_mode, row.evaluation_plan_id, row.relative_path
        ):
            continue
        if row.adapter_id not in SUPPORTED_ADAPTERS:
            continue
        if row.run_id is None or row.dataset_id is None:
            continue
        exploratory = row.target_dependency == "exploratory"
        if exploratory and not include_exploratory:
            continue
        candidates.append(
            DeploymentCandidate(
                relative_path=row.relative_path,
                label=candidate_label(row),
                dataset_id=row.dataset_id,
                model_family=row.model_family or row.adapter_id or "unavailable",
                model_config_id=row.model_config_id or "unavailable",
                evaluation_mode=row.evaluation_mode or "unavailable",
                balanced_accuracy=row.balanced_accuracy,
                run_identity=row.run_id,
                exploratory=exploratory,
            )
        )
    candidates.sort(key=lambda item: (item.dataset_id, item.model_family, item.label))
    return candidates


def candidate_label(row: ResearchRunInventoryRow) -> str:
    """Human-readable candidate label; never only a maximum metric."""
    metric = (
        f"BA {row.balanced_accuracy:.6f}"
        if row.balanced_accuracy is not None
        else "BA unavailable"
    )
    identity = (row.model_config_id or row.run_id or "unavailable")[:12]
    parts = [
        row.dataset_id or "unavailable",
        row.model_family or row.adapter_id or "unavailable",
        f"cfg {identity}",
        row.evaluation_mode or "unavailable",
        metric,
        f"run {(row.run_id or 'unavailable')[:20]}",
    ]
    label = " · ".join(parts)
    if row.target_dependency == "exploratory":
        label = f"⚠ exploratory · {label}"
    return label


def candidate_durable_key() -> str:
    """Durable key holding the selected deployment candidate run path."""
    return ui_durable_key("run", "deployment_candidate")


def apply_submission_handoff(session_state: Any, run_relative_path: str) -> None:
    """Transfer one completed-run identity to the Generate submission workflow.

    Only the canonical run path travels. The Research v2 configuration of that
    run is never handed over as a deployment config, so ``run_prefill`` carries
    no ``config`` value and the draft is always rebuilt by the shared builder.
    """
    relative = _normalized_relative(run_relative_path)
    if relative is None:
        raise ValueError("A deployment handoff needs a repository-relative run path.")
    session_state[SUBMISSION_HANDOFF_KEY] = relative
    set_durable_value(session_state, candidate_durable_key(), relative)
    session_state["run-command"] = SUBMISSION_COMMAND_ID
    session_state[f"run-action-{SUBMISSION_COMMAND_ID}"] = SUBMISSION_ACTION_ID
    set_durable_value(
        session_state, ui_durable_key("run", "command"), SUBMISSION_COMMAND_ID
    )
    set_durable_value(
        session_state,
        ui_durable_key("run", "action", SUBMISSION_COMMAND_ID),
        SUBMISSION_ACTION_ID,
    )
    session_state["run_prefill"] = {
        "command_id": SUBMISSION_COMMAND_ID,
        "action_id": SUBMISSION_ACTION_ID,
        "values": {},
    }


def _dataset_package_facts(
    root: Path, dataset_version: str | None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "target_dependency": None,
        "schema_hash": None,
        "test_data_path": None,
        "test_data_sha256": None,
        "test_expected_rows": None,
        "test_ordered_schema_sha256": None,
        "missing": [],
    }
    if dataset_version is None:
        return result
    package_dir = root / PROCESSED_ROOT / dataset_version
    manifest = _read_json(package_dir / "dataset_manifest.json")
    if not isinstance(manifest, dict):
        result["missing"].append(
            f"Dataset Package manifest is unavailable for {dataset_version}."
        )
        return result
    result["target_dependency"] = _as_str(manifest.get("target_dependency"))
    result["schema_hash"] = _sha_or_none(manifest.get("schema_hash"))
    test_name = _as_str(_dig(manifest, "files", "X_test")) or "X_test.parquet"
    test_sha = _sha_or_none(_dig(manifest, "content_hashes", "X_test"))
    rows = manifest.get("test_row_count")
    features = manifest.get("features")
    if test_sha is None:
        result["missing"].append(
            "Dataset Package manifest does not authenticate X_test content."
        )
    if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
        result["missing"].append(
            "Dataset Package manifest does not record a positive test row count."
        )
        rows = None
    names: list[str] = []
    if isinstance(features, list):
        for item in features:
            name = _as_str(item.get("name")) if isinstance(item, Mapping) else None
            if name is None:
                names = []
                break
            names.append(name)
    if not names:
        result["missing"].append(
            "Dataset Package manifest does not record an ordered feature schema."
        )
    else:
        result["test_ordered_schema_sha256"] = ordered_feature_schema_sha256(names)
    result["test_data_path"] = (
        f"{PROCESSED_ROOT}/{dataset_version}/{test_name}"
        if test_sha is not None
        else None
    )
    result["test_data_sha256"] = test_sha
    result["test_expected_rows"] = rows
    return result


def _looks_like_smoke(facts: CanonicalRunFacts) -> bool:
    return _mentions_smoke(facts.plan_id, facts.experiment_id, facts.run_relative_path)


def _mentions_smoke(*values: str | None) -> bool:
    """Smoke protocols are named explicitly; candidates and readiness agree."""
    return "smoke" in " ".join(filter(None, values)).casefold()


def _empty_facts(relative: str, missing: tuple[str, ...]) -> CanonicalRunFacts:
    return CanonicalRunFacts(
        run_relative_path=relative,
        run_id=None,
        manifest_sha256=None,
        plan_id=None,
        plan_sha256=None,
        pipeline_id=None,
        pipeline_sha256=None,
        adapter_id=None,
        adapter_sha256=None,
        candidate_sha256=None,
        dataset_identity_sha256=None,
        dataset_version=None,
        dataset_target_dependency=None,
        dataset_schema_hash=None,
        experiment_id=None,
        threshold_value=None,
        threshold_policy_id=None,
        fixed_parameters=None,
        test_data_path=None,
        test_data_sha256=None,
        test_expected_rows=None,
        test_ordered_schema_sha256=None,
        status="invalid",
        missing=missing,
    )


def _normalized_relative(value: str) -> str | None:
    text = str(value).strip().replace("\\", "/")
    if not text:
        return None
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        posix.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or ".." in posix.parts
        or "." in posix.parts
    ):
        return None
    return posix.as_posix()


def _status_from_markers(run_dir: Path) -> str:
    success = path_exists_nonfollowing(run_dir / "_SUCCESS")
    failed = path_exists_nonfollowing(run_dir / "_FAILED")
    if success and not failed:
        return "completed"
    if failed and not success:
        return "failed"
    if success and failed:
        return "invalid"
    return "running"


def _read_json(path: Path) -> Any:
    raw = _read_text(path)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _read_yaml(path: Path) -> Any:
    raw = _read_text(path)
    if raw is None:
        return None
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return None


def _read_text(path: Path) -> str | None:
    if not path_exists_nonfollowing(path):
        return None
    try:
        require_regular_file(path, reject_hardlinks=True)
        if path.stat().st_size > _MAX_SIDECAR_BYTES:
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, PathSafetyError):
        return None


def _dig(payload: Any, *keys: str) -> Any:
    current = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _as_str(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _slug_or_none(value: Any) -> str | None:
    text = _as_str(value)
    if text is None or _SLUG.fullmatch(text) is None:
        return None
    return text


def _sha_or_none(value: Any) -> str | None:
    text = _as_str(value)
    if text is None or _SHA256.fullmatch(text) is None:
        return None
    return text


def _threshold_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not 0.0 <= number <= 1.0:
        return None
    return number


def _display(value: Any) -> str:
    if value is None:
        return "unavailable"
    return str(value)


__all__ = [
    "CANDIDATE_WIDGET_KEY",
    "DEFAULT_SUBMISSION_ID_COLUMN",
    "DEFAULT_SUBMISSION_TARGET_COLUMN",
    "DEPLOYMENT_SCHEMA_LABEL",
    "SUBMISSION_ACTION_ID",
    "SUBMISSION_COMMAND_ID",
    "SUBMISSION_HANDOFF_KEY",
    "THRESHOLD_SOURCE_TYPE",
    "UNRESOLVED_SAMPLE_SUBMISSION_PATH",
    "CanonicalRunFacts",
    "DeploymentCandidate",
    "DeploymentReadiness",
    "apply_submission_handoff",
    "candidate_durable_key",
    "candidate_label",
    "evaluate_deployment_readiness",
    "list_deployment_candidates",
    "read_canonical_run_facts",
]
