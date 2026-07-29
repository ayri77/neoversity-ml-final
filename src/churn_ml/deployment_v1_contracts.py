from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Mapping

import yaml

from src.churn_ml.deployment_v1_auth import (
    AuthenticatedSource,
    authenticate_source,
    THRESHOLD_REFERENCE_KEYS,
    approval_identity,
    derive_approval_id,
    load_threshold_evidence,
)
from src.churn_ml.deployment_v1_paths import (
    DeploymentPathError,
    prewalk_regular_tree,
    validate_existing_root,
    validate_path_chain,
    validate_regular_file,
)
from src.churn_ml.experiment_v2_model_registry import get_candidate_adapter
from src.churn_ml.paired_comparison import (
    CompletedResearchV2Run,
    load_completed_research_v2_run,
)
from src.churn_ml.paired_comparison_artifacts import validate_comparison_artifacts
from src.churn_ml.research_data import canonical_sha256


APPROVAL_SCHEMA_VERSION = 1
DEPLOYMENT_SCHEMA_VERSION = 1
WEIGHT_SUM_TOLERANCE = 1e-12
SUPPORTED_ADAPTERS = {
    "manual_lightgbm_te_v1_compat",
    "xgboost_numeric_v1",
    "catboost_numeric_v1",
}
SAFE_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")

APPROVAL_KEYS = {
    "schema_version",
    "approval_id",
    "component_id",
    "component_name",
    "research_run",
    "fixed_resolved_model_parameters",
    "threshold_evidence",
    "paired_comparison",
    "manual_approval",
    "intended_deployment_role",
}
RESEARCH_REFERENCE_KEYS = {
    "path",
    "run_id",
    "manifest_sha256",
    "plan_id",
    "plan_sha256",
    "dataset_identity_sha256",
    "pipeline_id",
    "pipeline_sha256",
    "adapter_id",
    "adapter_sha256",
    "candidate_sha256",
}
PAIRED_KEYS = {"reference", "manifest_sha256", "exception"}
PAIRED_EXCEPTION_KEYS = {"granted", "reason"}
MANUAL_APPROVAL_KEYS = {"status", "approver", "approved_at_utc"}

CONFIG_KEYS = {
    "schema_version",
    "deployment_id",
    "dataset_version",
    "pipeline_id",
    "components",
    "blend",
    "threshold",
    "bagging",
    "test_data",
    "sample_submission",
    "output",
    "runtime",
}
COMPONENT_KEYS = {
    "component_id",
    "approval_artifact_path",
    "adapter_id",
    "fixed_parameters",
    "bag_seeds",
    "component_weight",
}
BLEND_KEYS = {"method", "weight_sum_tolerance"}
THRESHOLD_KEYS = {"value", "evidence", "comparison"}
BAGGING_KEYS = {"method", "aggregation", "model_persistence"}
TEST_DATA_KEYS = {"path", "sha256", "expected_rows", "ordered_schema_sha256"}
SAMPLE_KEYS = {
    "path",
    "sha256",
    "expected_rows",
    "id_column",
    "target_column",
}
OUTPUT_KEYS = {"root", "submission_filename"}
RUNTIME_KEYS = {"tracking_enabled", "network_enabled"}


class DeploymentContractError(ValueError):
    """Raised when a P4 approval or deployment config is not exact and safe."""


@dataclass(frozen=True)
class CandidateApproval:
    payload: dict[str, Any]
    source_path: Path
    source_sha256: str
    research_run: CompletedResearchV2Run
    threshold_identity: dict[str, Any] = field(default_factory=dict)
    identity: dict[str, Any] = field(default_factory=dict)
    approval_source: AuthenticatedSource | None = None
    research_run_source: AuthenticatedSource | None = None
    threshold_evidence_source: AuthenticatedSource | None = None
    paired_comparison_source: AuthenticatedSource | None = None

    @property
    def approval_id(self) -> str:
        return str(self.payload["approval_id"])

    @property
    def component_id(self) -> str:
        return str(self.payload["component_id"])

    @property
    def adapter_id(self) -> str:
        return str(self.payload["research_run"]["adapter_id"])


@dataclass(frozen=True)
class DeploymentConfig:
    payload: dict[str, Any]
    source_path: Path
    project_root: Path

    @property
    def deployment_id(self) -> str:
        return str(self.payload["deployment_id"])

    @property
    def output_root(self) -> Path:
        return resolve_repository_path(
            self.payload["output"]["root"],
            self.project_root,
            "output.root",
        )

    def resolved_payload(self) -> dict[str, Any]:
        return deepcopy(self.payload)


@dataclass(frozen=True)
class ValidatedDeployment:
    config: DeploymentConfig
    approvals: tuple[CandidateApproval, ...]
    identity: dict[str, Any]
    identity_sha256: str


ResearchLoader = Callable[..., CompletedResearchV2Run]


def load_candidate_approval(
    path: Path,
    *,
    project_root: Path,
    research_loader: ResearchLoader = load_completed_research_v2_run,
) -> CandidateApproval:
    try:
        root = validate_existing_root(project_root)
    except DeploymentPathError as error:
        raise DeploymentContractError(str(error)) from error
    source = _contained_file(path, root, "approval artifact")
    payload = _load_yaml(source, "approval artifact")
    _exact_keys(payload, APPROVAL_KEYS, "approval")
    if _integer(payload["schema_version"], "approval.schema_version") != 1:
        raise DeploymentContractError("approval.schema_version must be 1.")
    _sha(payload["approval_id"], "approval.approval_id")
    _slug(payload["component_id"], "approval.component_id")
    _string(payload["component_name"], "approval.component_name")
    _string(payload["intended_deployment_role"], "intended deployment role")

    reference = _mapping(payload["research_run"], "approval.research_run")
    _exact_keys(reference, RESEARCH_REFERENCE_KEYS, "approval.research_run")
    run_path = portable_repository_path(
        reference["path"], root, "approval.research_run.path"
    )
    try:
        unresolved_run_path = (
            root / Path(*PurePosixPath(str(reference["path"])).parts)
        ).absolute()
        prewalk_regular_tree(
            unresolved_run_path,
            containment_roots=(root,),
            reject_hardlinks=True,
        )
    except DeploymentPathError as error:
        raise DeploymentContractError(str(error)) from error
    for key in ("run_id", "plan_id", "pipeline_id", "adapter_id"):
        _slug(reference[key], f"approval.research_run.{key}")
    for key in (
        "manifest_sha256",
        "plan_sha256",
        "dataset_identity_sha256",
        "pipeline_sha256",
        "adapter_sha256",
        "candidate_sha256",
    ):
        _sha(reference[key], f"approval.research_run.{key}")
    if reference["adapter_id"] not in SUPPORTED_ADAPTERS:
        raise DeploymentContractError("Approval adapter is not supported by P4 v1.")

    parameters = _mapping(
        payload["fixed_resolved_model_parameters"],
        "approval.fixed_resolved_model_parameters",
    )
    _validate_primitive_tree(parameters, "approval.fixed_resolved_model_parameters")
    paired = _mapping(payload["paired_comparison"], "approval.paired_comparison")
    _exact_keys(paired, PAIRED_KEYS, "approval.paired_comparison")
    exception = _mapping(paired["exception"], "approval.paired_comparison.exception")
    _exact_keys(
        exception,
        PAIRED_EXCEPTION_KEYS,
        "approval.paired_comparison.exception",
    )
    granted = _boolean(
        exception["granted"], "approval.paired_comparison.exception.granted"
    )
    comparison_source: AuthenticatedSource | None = None
    if paired["reference"] is None:
        if paired["manifest_sha256"] is not None:
            raise DeploymentContractError(
                "Paired-comparison manifest must be null when its reference is null."
            )
        if not granted:
            raise DeploymentContractError(
                "Missing paired comparison requires an explicit granted exception."
            )
        _string(exception["reason"], "paired-comparison exception reason")
    else:
        comparison_path = portable_repository_path(
            paired["reference"], root, "approval.paired_comparison.reference"
        )
        try:
            unresolved_comparison_path = (
                root / Path(*PurePosixPath(str(paired["reference"])).parts)
            ).absolute()
            prewalk_regular_tree(
                unresolved_comparison_path,
                containment_roots=(root,),
                reject_hardlinks=True,
            )
        except DeploymentPathError as error:
            raise DeploymentContractError(str(error)) from error
        _sha(
            paired["manifest_sha256"],
            "approval.paired_comparison.manifest_sha256",
        )
        if granted or exception["reason"] is not None:
            raise DeploymentContractError(
                "A referenced paired comparison cannot also claim an exception."
            )
        validate_comparison_artifacts(
            comparison_path,
            project_root=root,
            require_success=True,
            verify_manifest=True,
        )
        comparison_manifest = _read_json(comparison_path / "manifest.json")
        comparison_source = authenticate_source(
            comparison_path,
            project_root=root,
            source_type="paired_comparison",
            semantic_identity=str(comparison_manifest.get("manifest_sha256")),
            expected_kind="directory",
        )
        if comparison_manifest.get("manifest_sha256") != paired["manifest_sha256"]:
            raise DeploymentContractError(
                "Paired-comparison manifest authentication failed."
            )
        candidate_refs = (
            _read_json(comparison_path / "baseline_run_reference.json"),
            _read_json(comparison_path / "candidate_run_reference.json"),
        )
        if not any(
            item.get("repository_relative_path") == reference["path"]
            and item.get("artifact_manifest_sha256") == reference["manifest_sha256"]
            for item in candidate_refs
        ):
            raise DeploymentContractError(
                "Paired comparison does not authenticate the approved research run."
            )

    manual = _mapping(payload["manual_approval"], "approval.manual_approval")
    _exact_keys(manual, MANUAL_APPROVAL_KEYS, "approval.manual_approval")
    if manual["status"] != "approved":
        raise DeploymentContractError(
            "Candidate approval status must be exactly 'approved'."
        )
    _string(manual["approver"], "approval.manual_approval.approver")
    _utc(manual["approved_at_utc"], "approval.manual_approval.approved_at_utc")

    run = research_loader(
        run_path,
        project_root=root,
        role="approved_research_run",
    )
    _authenticate_research_reference(reference, parameters, run)
    try:
        threshold_evidence = load_threshold_evidence(
            payload["threshold_evidence"], project_root=root, research_run=run
        )
    except ValueError as error:
        raise DeploymentContractError(str(error)) from error
    if payload["approval_id"] != derive_approval_id(payload):
        raise DeploymentContractError(
            "approval_id is not the deterministic immutable identity."
        )
    identity = approval_identity(payload)
    raw = source.read_bytes()
    return CandidateApproval(
        payload=deepcopy(payload),
        source_path=source,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        research_run=run,
        threshold_identity=threshold_evidence.identity,
        identity=identity,
        approval_source=authenticate_source(
            source,
            project_root=root,
            source_type="candidate_approval",
            semantic_identity=str(payload["approval_id"]),
            expected_kind="file",
        ),
        research_run_source=authenticate_source(
            run_path,
            project_root=root,
            source_type="completed_research_run",
            semantic_identity=str(reference["manifest_sha256"]),
            expected_kind="directory",
        ),
        threshold_evidence_source=threshold_evidence.source,
        paired_comparison_source=comparison_source,
    )


def load_deployment_config(
    path: Path,
    *,
    project_root: Path,
    allow_external_source: bool = False,
) -> DeploymentConfig:
    try:
        root = validate_existing_root(project_root)
    except DeploymentPathError as error:
        raise DeploymentContractError(str(error)) from error
    if allow_external_source:
        try:
            source = validate_regular_file(
                path,
                containment_root=Path(path.absolute().anchor),
                reject_hardlinks=True,
            )
        except DeploymentPathError as error:
            raise DeploymentContractError(str(error)) from error
    else:
        source = _contained_file(path, root, "deployment config")
    payload = _load_yaml(source, "deployment config")
    _exact_keys(payload, CONFIG_KEYS, "deployment")
    if _integer(payload["schema_version"], "schema_version") != 1:
        raise DeploymentContractError("schema_version must be 1.")
    _slug(payload["deployment_id"], "deployment_id")
    _slug(payload["dataset_version"], "dataset_version")
    _slug(payload["pipeline_id"], "pipeline_id")

    components = payload["components"]
    if not isinstance(components, list) or not components:
        raise DeploymentContractError("components must be a non-empty list.")
    seen: set[str] = set()
    weights: list[float] = []
    for index, item in enumerate(components):
        label = f"components[{index}]"
        component = _mapping(item, label)
        _exact_keys(component, COMPONENT_KEYS, label)
        component_id = _slug(component["component_id"], f"{label}.component_id")
        if component_id in seen:
            raise DeploymentContractError("Duplicate component IDs are prohibited.")
        seen.add(component_id)
        portable_repository_path(
            component["approval_artifact_path"],
            root,
            f"{label}.approval_artifact_path",
        )
        adapter_id = _slug(component["adapter_id"], f"{label}.adapter_id")
        if adapter_id not in SUPPORTED_ADAPTERS:
            raise DeploymentContractError(f"{label}.adapter_id is unsupported.")
        fixed = _mapping(component["fixed_parameters"], f"{label}.fixed_parameters")
        _validate_primitive_tree(fixed, f"{label}.fixed_parameters")
        seeds = component["bag_seeds"]
        if not isinstance(seeds, list) or not seeds:
            raise DeploymentContractError(f"{label}.bag_seeds must be non-empty.")
        parsed_seeds = [
            _nonnegative_integer(value, f"{label}.bag_seeds[{seed_index}]")
            for seed_index, value in enumerate(seeds)
        ]
        if len(set(parsed_seeds)) != len(parsed_seeds):
            raise DeploymentContractError(f"{label}.bag_seeds contains duplicates.")
        payload["components"][index]["bag_seeds"] = sorted(parsed_seeds)
        payload["components"][index]["bag_seeds"] = sorted(parsed_seeds)
        weight = _finite_float(component["component_weight"], f"{label}.weight")
        if weight < 0.0:
            raise DeploymentContractError("Component weights must be nonnegative.")
        weights.append(weight)

    blend = _mapping(payload["blend"], "blend")
    _exact_keys(blend, BLEND_KEYS, "blend")
    if blend["method"] != "fixed_weighted_mean":
        raise DeploymentContractError("blend.method must be fixed_weighted_mean.")
    tolerance = _finite_float(
        blend["weight_sum_tolerance"], "blend.weight_sum_tolerance"
    )
    if tolerance != WEIGHT_SUM_TOLERANCE:
        raise DeploymentContractError(
            f"blend.weight_sum_tolerance must be exactly {WEIGHT_SUM_TOLERANCE}."
        )
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=tolerance):
        raise DeploymentContractError("Component weights must sum to exactly 1.")

    threshold = _mapping(payload["threshold"], "threshold")
    _exact_keys(threshold, THRESHOLD_KEYS, "threshold")
    threshold_value = _finite_float(threshold["value"], "threshold.value")
    if not 0.0 <= threshold_value <= 1.0:
        raise DeploymentContractError("threshold.value must be within [0, 1].")
    evidence = _mapping(threshold["evidence"], "threshold.evidence")
    _exact_keys(evidence, THRESHOLD_REFERENCE_KEYS, "threshold.evidence")
    if _integer(evidence["schema_version"], "threshold.evidence.schema_version") != 1:
        raise DeploymentContractError("threshold evidence schema must be 1.")
    portable_repository_path(evidence["path"], root, "threshold.evidence.path")
    _positive_integer(evidence["size_bytes"], "threshold.evidence.size_bytes")
    for key in ("sha256", "source_manifest_sha256", "plan_sha256", "candidate_sha256"):
        _sha(evidence[key], f"threshold.evidence.{key}")
    if evidence["source_type"] not in {
        "experiment_core_v2_manual_threshold_v1",
        "blend_evaluation_v1_cross_fit_threshold_v1",
    }:
        raise DeploymentContractError("threshold evidence source_type differs.")
    _slug(evidence["source_run_id"], "threshold.evidence.source_run_id")
    _slug(evidence["threshold_policy_id"], "threshold.evidence.threshold_policy_id")
    evidence_threshold = _finite_float(
        evidence["threshold"], "threshold.evidence.threshold"
    )
    if evidence_threshold != threshold_value:
        raise DeploymentContractError(
            "threshold.value differs from threshold evidence."
        )
    if threshold["comparison"] != "greater_than_or_equal":
        raise DeploymentContractError("threshold.comparison must encode exact >=.")

    bagging = _mapping(payload["bagging"], "bagging")
    _exact_keys(bagging, BAGGING_KEYS, "bagging")
    expected_bagging = {
        "method": "full_data_seed_bagging",
        "aggregation": "arithmetic_mean",
        "model_persistence": False,
    }
    if dict(bagging) != expected_bagging:
        raise DeploymentContractError("bagging contract differs from P4 v1.")

    test_data = _mapping(payload["test_data"], "test_data")
    _exact_keys(test_data, TEST_DATA_KEYS, "test_data")
    portable_repository_path(test_data["path"], root, "test_data.path")
    _sha(test_data["sha256"], "test_data.sha256")
    _positive_integer(test_data["expected_rows"], "test_data.expected_rows")
    _sha(test_data["ordered_schema_sha256"], "test_data.ordered_schema_sha256")

    sample = _mapping(payload["sample_submission"], "sample_submission")
    _exact_keys(sample, SAMPLE_KEYS, "sample_submission")
    portable_repository_path(sample["path"], root, "sample_submission.path")
    _sha(sample["sha256"], "sample_submission.sha256")
    _positive_integer(sample["expected_rows"], "sample_submission.expected_rows")
    _string(sample["id_column"], "sample_submission.id_column")
    _string(sample["target_column"], "sample_submission.target_column")
    if sample["expected_rows"] != test_data["expected_rows"]:
        raise DeploymentContractError("Test and sample expected row counts differ.")
    if sample["id_column"] == sample["target_column"]:
        raise DeploymentContractError("Submission ID and target columns must differ.")

    output = _mapping(payload["output"], "output")
    _exact_keys(output, OUTPUT_KEYS, "output")
    output_root = portable_repository_path(output["root"], root, "output.root")
    expected_root = portable_repository_path(
        "artifacts/deployments", root, "expected output root"
    )
    if output_root != expected_root:
        raise DeploymentContractError(
            "output.root must be exactly artifacts/deployments."
        )
    _basename(output["submission_filename"], "output.submission_filename")
    if output["submission_filename"] != "submission.csv":
        raise DeploymentContractError(
            "output.submission_filename must be exactly submission.csv."
        )
    runtime = _mapping(payload["runtime"], "runtime")
    _exact_keys(runtime, RUNTIME_KEYS, "runtime")
    if runtime != {"tracking_enabled": False, "network_enabled": False}:
        raise DeploymentContractError(
            "P4 v1 tracking and network access must remain disabled."
        )
    return DeploymentConfig(deepcopy(payload), source, root)


def validate_deployment(
    config_path: Path,
    *,
    project_root: Path,
    research_loader: ResearchLoader = load_completed_research_v2_run,
) -> ValidatedDeployment:
    config = load_deployment_config(config_path, project_root=project_root)
    return validate_loaded_deployment(config, research_loader=research_loader)


def validate_loaded_deployment(
    config: DeploymentConfig,
    *,
    research_loader: ResearchLoader = load_completed_research_v2_run,
) -> ValidatedDeployment:
    approvals: list[CandidateApproval] = []
    approval_ids: set[str] = set()
    for component in config.payload["components"]:
        approval = load_candidate_approval(
            Path(component["approval_artifact_path"]),
            project_root=config.project_root,
            research_loader=research_loader,
        )
        if approval.approval_id in approval_ids:
            raise DeploymentContractError("Duplicate approval IDs are prohibited.")
        approval_ids.add(approval.approval_id)
        if approval.component_id != component["component_id"]:
            raise DeploymentContractError("Component and approval IDs differ.")
        if approval.adapter_id != component["adapter_id"]:
            raise DeploymentContractError("Component and approval adapters differ.")
        if (
            approval.payload["fixed_resolved_model_parameters"]
            != component["fixed_parameters"]
        ):
            raise DeploymentContractError(
                "Component parameters differ from the immutable approval."
            )
        run = approval.research_run
        if run.config.dataset_version != config.payload["dataset_version"]:
            raise DeploymentContractError("Deployment dataset identity differs.")
        if run.config.pipeline_id != config.payload["pipeline_id"]:
            raise DeploymentContractError("Deployment pipeline identity differs.")
        get_candidate_adapter(component["adapter_id"]).validate_contract(
            run.config.adapter_contract
        )
        approvals.append(approval)
    dataset_hashes = {
        item.payload["research_run"]["dataset_identity_sha256"] for item in approvals
    }
    pipeline_hashes = {
        item.payload["research_run"]["pipeline_sha256"] for item in approvals
    }
    threshold_references = {
        canonical_sha256(item.payload["threshold_evidence"]) for item in approvals
    }
    if len(dataset_hashes) != 1:
        raise DeploymentContractError(
            "Approved components do not share exact dataset identity."
        )
    if len(pipeline_hashes) != 1:
        raise DeploymentContractError(
            "Approved components do not share exact pipeline identity."
        )
    if threshold_references != {
        canonical_sha256(config.payload["threshold"]["evidence"])
    }:
        raise DeploymentContractError(
            "Deployment threshold evidence differs from component approvals."
        )
    canonical = {
        "schema_version": DEPLOYMENT_SCHEMA_VERSION,
        "resolved_config": config.resolved_payload(),
        "approvals": [
            {
                "approval_id": item.approval_id,
                "approval_sha256": item.source_sha256,
                "research_manifest_sha256": item.payload["research_run"][
                    "manifest_sha256"
                ],
                "approval_identity_sha256": item.identity["sha256"],
                "threshold_identity_sha256": item.threshold_identity["sha256"],
            }
            for item in approvals
        ],
        "semantics": {
            "phase": "P4_deployment_only",
            "model_selection": "manual_preapproved_inputs_only",
            "bagging": "all_train_rows_seed_only",
            "blend": "fixed_weight_arithmetic_mean",
            "threshold": "fixed_greater_than_or_equal",
        },
    }
    return ValidatedDeployment(
        config=config,
        approvals=tuple(approvals),
        identity=canonical,
        identity_sha256=canonical_sha256(canonical),
    )


def model_parameters(
    adapter_id: str, adapter_contract: Mapping[str, Any]
) -> dict[str, Any]:
    key = {
        "manual_lightgbm_te_v1_compat": "lightgbm",
        "xgboost_numeric_v1": "xgboost",
        "catboost_numeric_v1": "catboost",
    }[adapter_id]
    section = _mapping(adapter_contract[key], f"candidate_adapter.contract.{key}")
    return deepcopy(
        _mapping(section["parameters"], f"candidate_adapter.contract.{key}.parameters")
    )


def portable_repository_path(value: Any, root: Path, label: str) -> Path:
    text = _string(value, label)
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        "\\" in text
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or ".." in posix.parts
        or "." in posix.parts
    ):
        raise DeploymentContractError(
            f"{label} must be a normalized repository-relative portable path."
        )
    try:
        safe = validate_path_chain(
            containment_root=root,
            requested_path=root / Path(*posix.parts),
            require_exists=False,
            expected_kind="either",
        )
    except DeploymentPathError as error:
        raise DeploymentContractError(f"{label} is unsafe: {error}") from error
    if safe.canonical == root:
        raise DeploymentContractError(f"{label} escapes the repository.")
    return safe.canonical


def resolve_repository_path(value: Any, root: Path, label: str) -> Path:
    return portable_repository_path(value, root, label)


def _authenticate_research_reference(
    reference: Mapping[str, Any],
    approved_parameters: Mapping[str, Any],
    run: CompletedResearchV2Run,
) -> None:
    identity_dir = run.root / "identities"
    plan = _read_json(identity_dir / "evaluation_plan.json")
    pipeline = _read_json(identity_dir / "feature_pipeline.json")
    adapter = _read_json(identity_dir / "candidate_adapter.json")
    candidate = _read_json(identity_dir / "candidate.json")
    actual = {
        "run_id": run.metadata.get("run_id"),
        "manifest_sha256": run.manifest.get("manifest_sha256"),
        "plan_id": run.config.plan_id,
        "plan_sha256": plan.get("sha256"),
        "dataset_identity_sha256": canonical_sha256(run.dataset_fingerprints),
        "pipeline_id": run.config.pipeline_id,
        "pipeline_sha256": pipeline.get("sha256"),
        "adapter_id": run.config.adapter_id,
        "adapter_sha256": adapter.get("sha256"),
        "candidate_sha256": candidate.get("sha256"),
    }
    for key, actual_value in actual.items():
        if reference[key] != actual_value:
            raise DeploymentContractError(
                f"Approved research identity mismatch at research_run.{key}."
            )
    if model_parameters(run.config.adapter_id, run.config.adapter_contract) != dict(
        approved_parameters
    ):
        raise DeploymentContractError(
            "Approved fixed parameters differ from the completed research run."
        )


def _validate_primitive_tree(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise DeploymentContractError(
                    f"{label} keys must be non-empty strings."
                )
            lowered = key.casefold()
            if any(
                marker in lowered
                for marker in ("optuna", "distribution", "search_space", "trial")
            ):
                raise DeploymentContractError(
                    f"{label}.{key} contains a forbidden optimization field."
                )
            _validate_primitive_tree(child, f"{label}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_primitive_tree(child, f"{label}[{index}]")
        return
    if value is None or type(value) in {str, int, bool}:
        return
    if type(value) is float and math.isfinite(value):
        return
    raise DeploymentContractError(f"{label} contains an invalid primitive value.")


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise DeploymentContractError(f"Cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise DeploymentContractError(f"{label} must be a mapping.")
    return dict(value)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DeploymentContractError(f"Cannot read JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise DeploymentContractError(f"JSON artifact must be an object: {path}")
    return value


def _contained_file(path: Path, root: Path, label: str) -> Path:
    source = path if path.is_absolute() else root / path
    try:
        return validate_regular_file(
            source, containment_root=root, reject_hardlinks=True
        )
    except DeploymentPathError as error:
        raise DeploymentContractError(f"{label} is unsafe: {error}") from error


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DeploymentContractError(f"{label} must be a mapping.")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise DeploymentContractError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}."
        )


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DeploymentContractError(f"{label} must be a non-empty string.")
    return value


def _slug(value: Any, label: str) -> str:
    text = _string(value, label)
    if SAFE_SLUG.fullmatch(text) is None:
        raise DeploymentContractError(f"{label} must be a safe slug.")
    return text


def _sha(value: Any, label: str) -> str:
    text = _string(value, label)
    if SHA256.fullmatch(text) is None:
        raise DeploymentContractError(f"{label} must be a lowercase SHA-256.")
    return text


def _basename(value: Any, label: str) -> str:
    text = _string(value, label)
    if (
        text in {".", ".."}
        or "/" in text
        or "\\" in text
        or PurePosixPath(text).name != text
        or PureWindowsPath(text).name != text
    ):
        raise DeploymentContractError(f"{label} must be a plain basename.")
    return text


def _integer(value: Any, label: str) -> int:
    if type(value) is not int:
        raise DeploymentContractError(f"{label} must be an integer.")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    parsed = _integer(value, label)
    if not 0 <= parsed <= 2**31 - 1:
        raise DeploymentContractError(f"{label} must be in [0, 2^31 - 1].")
    return parsed


def _positive_integer(value: Any, label: str) -> int:
    parsed = _integer(value, label)
    if parsed <= 0:
        raise DeploymentContractError(f"{label} must be positive.")
    return parsed


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise DeploymentContractError(f"{label} must be boolean.")
    return value


def _finite_float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise DeploymentContractError(f"{label} must be a finite float.")
    return value


def _utc(value: Any, label: str) -> datetime:
    text = _string(value, label)
    try:
        result = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise DeploymentContractError(f"{label} must be ISO-8601.") from error
    if result.tzinfo is None or result.utcoffset() != timezone.utc.utcoffset(result):
        raise DeploymentContractError(f"{label} must be timezone-aware UTC.")
    return result
