"""Canonical completed run to deployment-draft builder.

One completed canonical Research v2 run becomes a validated deployment draft
payload and then a deployment-schema YAML. The builder derives every field it
can from authoritative run artifacts and Dataset Package identity, and refuses
to guess model parameters, dataset paths, target semantics, threshold evidence,
or test-row identity.

Drafts live under a dedicated ignored deployment root, never in the generic
mixed-schema ``artifacts/ui_configs`` root. Writes are atomic, repository
relative, and never silently replace a different draft.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

from src.churn_ml.competition_assets_v1 import (
    DEFAULT_ID_COLUMN,
    DEFAULT_TARGET_COLUMN,
    ID_SEMANTICS,
    ensure_competition_row_identity,
    write_competition_asset_registry,
)
from src.churn_ml.control_panel.deployment_candidates import (
    DEFAULT_SUBMISSION_ID_COLUMN,
    DEFAULT_SUBMISSION_TARGET_COLUMN,
    THRESHOLD_SOURCE_TYPE,
    UNRESOLVED_SAMPLE_SUBMISSION_PATH,
    CanonicalRunFacts,
    evaluate_deployment_readiness,
)
from src.churn_ml.control_panel.path_safety import (
    PathSafetyError,
    path_exists_nonfollowing,
    require_regular_file,
)
from src.churn_ml.deployment_v1_auth import derive_approval_id
from src.churn_ml.deployment_v1_contracts import (
    DEPLOYMENT_SCHEMA_VERSION,
    WEIGHT_SUM_TOLERANCE,
)


BUILDER_ID = "canonical_run_deployment_draft_builder"
BUILDER_VERSION = 1
DEPLOYMENT_DRAFT_ROOT = "artifacts/deployment_drafts"
DEPLOYMENT_CONFIG_NAME = "deployment_config.yaml"
THRESHOLD_EVIDENCE_NAME = "threshold_evidence.yaml"
APPROVAL_NAME = "candidate_approval.yaml"
PROVENANCE_NAME = "draft_provenance.json"
DEFAULT_INTENDED_ROLE = "single-run-canonical-candidate"
DEFAULT_EXCEPTION_REASON = (
    "Single completed canonical run promoted without a paired comparison; "
    "comparison evidence is recorded separately in the Research Workspace."
)

_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


class DeploymentDraftError(ValueError):
    """Raised when a deployment draft cannot be derived safely."""


class DeploymentDraftConflictError(DeploymentDraftError):
    """Raised when an existing draft would be silently replaced."""


@dataclass(frozen=True)
class DeploymentDraftPayload:
    """Fully derived deployment-draft content before it is persisted."""

    candidate_id: str
    facts: CanonicalRunFacts
    deployment_config: dict[str, Any]
    threshold_evidence: dict[str, Any]
    threshold_reference: dict[str, Any]
    candidate_approval: dict[str, Any]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class DeploymentDraft:
    """Persisted deployment draft with repository-relative locations."""

    candidate_id: str
    draft_dir: str
    deployment_config_path: str
    threshold_evidence_path: str
    candidate_approval_path: str
    provenance_path: str
    payload: DeploymentDraftPayload
    reused: bool

    def summary(self) -> dict[str, str]:
        return {
            "Candidate ID": self.candidate_id,
            "Deployment draft": self.deployment_config_path,
            "Threshold evidence": self.threshold_evidence_path,
            "Candidate approval": self.candidate_approval_path,
            "Draft provenance": self.provenance_path,
            "Regenerated identically": "true" if self.reused else "false",
        }


def approval_timestamp_now() -> str:
    """Timezone-aware UTC approval timestamp accepted by the deployment loader."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_deployment_draft_payload(
    repository_root: Path,
    run_relative_path: str,
    *,
    approver: str,
    approved_at_utc: str,
    intended_deployment_role: str = DEFAULT_INTENDED_ROLE,
    paired_comparison_exception_reason: str = DEFAULT_EXCEPTION_REASON,
    archived_paths: frozenset[str] = frozenset(),
) -> DeploymentDraftPayload:
    """Derive the complete deployment draft payload for one completed run."""
    readiness = evaluate_deployment_readiness(
        repository_root, run_relative_path, archived_paths=archived_paths
    )
    if not readiness.supported:
        raise DeploymentDraftError(
            "The selected run is not deployment ready: "
            + " ".join(readiness.blocking_reasons)
        )
    approver_text = _required_text(approver, "approver")
    role_text = _required_text(intended_deployment_role, "intended deployment role")
    reason_text = _required_text(
        paired_comparison_exception_reason, "paired-comparison exception reason"
    )
    approved_at = _required_utc(approved_at_utc)

    facts = readiness.facts
    base_candidate_id = facts.candidate_id
    if base_candidate_id is None or _SLUG.fullmatch(base_candidate_id) is None:
        raise DeploymentDraftError("A safe deterministic candidate ID is unavailable.")

    assets = ensure_competition_row_identity(
        repository_root, dataset_version=facts.dataset_version
    )
    if assets.ready and assets.asset_fingerprint:
        candidate_id = f"{base_candidate_id}-r{assets.asset_fingerprint[:8]}"
        try:
            write_competition_asset_registry(repository_root, assets)
        except OSError as error:
            raise DeploymentDraftError(
                f"Competition asset registry could not be written: {error}"
            ) from error
    else:
        candidate_id = base_candidate_id
    if _SLUG.fullmatch(candidate_id) is None:
        raise DeploymentDraftError("A safe deterministic candidate ID is unavailable.")

    threshold_evidence = {
        "schema_version": 1,
        "source_type": THRESHOLD_SOURCE_TYPE,
        "source_run_id": facts.run_id,
        "source_manifest_sha256": facts.manifest_sha256,
        "threshold": float(facts.threshold_value or 0.0),
        "threshold_policy_id": facts.threshold_policy_id,
        "plan_sha256": facts.plan_sha256,
        "candidate_sha256": facts.candidate_sha256,
    }
    evidence_bytes = _dump_yaml(threshold_evidence)
    evidence_relative = (
        f"{DEPLOYMENT_DRAFT_ROOT}/{candidate_id}/{THRESHOLD_EVIDENCE_NAME}"
    )
    threshold_reference = {
        "schema_version": 1,
        "path": evidence_relative,
        "size_bytes": len(evidence_bytes),
        "sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        "source_type": threshold_evidence["source_type"],
        "source_run_id": threshold_evidence["source_run_id"],
        "source_manifest_sha256": threshold_evidence["source_manifest_sha256"],
        "threshold": threshold_evidence["threshold"],
        "threshold_policy_id": threshold_evidence["threshold_policy_id"],
        "plan_sha256": threshold_evidence["plan_sha256"],
        "candidate_sha256": threshold_evidence["candidate_sha256"],
    }

    research_reference = {
        "path": facts.run_relative_path,
        "run_id": facts.run_id,
        "manifest_sha256": facts.manifest_sha256,
        "plan_id": facts.plan_id,
        "plan_sha256": facts.plan_sha256,
        "dataset_identity_sha256": facts.dataset_identity_sha256,
        "pipeline_id": facts.pipeline_id,
        "pipeline_sha256": facts.pipeline_sha256,
        "adapter_id": facts.adapter_id,
        "adapter_sha256": facts.adapter_sha256,
        "candidate_sha256": facts.candidate_sha256,
    }
    component_id = str(facts.adapter_id)
    approval = {
        "schema_version": 1,
        "approval_id": "0" * 64,
        "component_id": component_id,
        "component_name": (
            f"{facts.dataset_version} · {component_id} · run {facts.run_id}"
        ),
        "research_run": research_reference,
        "fixed_resolved_model_parameters": dict(facts.fixed_parameters or {}),
        "threshold_evidence": threshold_reference,
        "paired_comparison": {
            "reference": None,
            "manifest_sha256": None,
            "exception": {"granted": True, "reason": reason_text},
        },
        "manual_approval": {
            "status": "approved",
            "approver": approver_text,
            "approved_at_utc": approved_at,
        },
        "intended_deployment_role": role_text,
    }
    approval["approval_id"] = derive_approval_id(approval)
    approval_relative = f"{DEPLOYMENT_DRAFT_ROOT}/{candidate_id}/{APPROVAL_NAME}"

    deployment_config = {
        "schema_version": DEPLOYMENT_SCHEMA_VERSION,
        "deployment_id": candidate_id,
        "dataset_version": facts.dataset_version,
        "pipeline_id": facts.pipeline_id,
        "components": [
            {
                "component_id": component_id,
                "approval_artifact_path": approval_relative,
                "adapter_id": facts.adapter_id,
                "fixed_parameters": dict(facts.fixed_parameters or {}),
                "bag_seeds": [0],
                "component_weight": 1.0,
            }
        ],
        "blend": {
            "method": "fixed_weighted_mean",
            "weight_sum_tolerance": WEIGHT_SUM_TOLERANCE,
        },
        "threshold": {
            "value": threshold_reference["threshold"],
            "evidence": threshold_reference,
            "comparison": "greater_than_or_equal",
        },
        "bagging": {
            "method": "full_data_seed_bagging",
            "aggregation": "arithmetic_mean",
            "model_persistence": False,
        },
        "test_data": {
            "path": facts.test_data_path,
            "sha256": facts.test_data_sha256,
            "expected_rows": facts.test_expected_rows,
            "ordered_schema_sha256": facts.test_ordered_schema_sha256,
        },
        "sample_submission": (
            {
                "path": assets.sample_submission_path,
                "sha256": assets.sample_submission_sha256,
                "expected_rows": assets.sample_expected_rows,
                "id_column": assets.id_column or DEFAULT_ID_COLUMN,
                "target_column": assets.target_column or DEFAULT_TARGET_COLUMN,
            }
            if assets.ready
            else {
                "path": UNRESOLVED_SAMPLE_SUBMISSION_PATH,
                "sha256": "0" * 64,
                "expected_rows": facts.test_expected_rows,
                "id_column": DEFAULT_SUBMISSION_ID_COLUMN,
                "target_column": DEFAULT_SUBMISSION_TARGET_COLUMN,
            }
        ),
        "output": {
            "root": "artifacts/deployments",
            "submission_filename": "submission.csv",
        },
        "runtime": {"tracking_enabled": False, "network_enabled": False},
    }
    if assets.ready:
        deployment_config["submission_row_identity"] = {
            "path": assets.row_identity_path,
            "sha256": assets.row_identity_sha256,
            "expected_rows": assets.sample_expected_rows,
            "id_column": assets.id_column or DEFAULT_ID_COLUMN,
            "id_dtype": "int64",
            "id_semantics": ID_SEMANTICS,
            "ordered_id_sha256": assets.ordered_id_sha256,
            "row_position_identity_sha256": assets.ordered_id_sha256,
            "test_anchor_hash": assets.test_anchor_hash,
        }

    provenance = {
        "schema_version": 1,
        "builder_id": BUILDER_ID,
        "builder_version": BUILDER_VERSION,
        "deployment_schema_version": DEPLOYMENT_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "source": {
            "kind": "research_v2_completed_run",
            "run_path": facts.run_relative_path,
            "run_id": facts.run_id,
            "run_manifest_sha256": facts.manifest_sha256,
            "experiment_id": facts.experiment_id,
        },
        "dataset": {
            "dataset_id": facts.dataset_version,
            "dataset_identity_sha256": facts.dataset_identity_sha256,
            "schema_hash": facts.dataset_schema_hash,
            "target_dependency": facts.dataset_target_dependency,
            "test_expected_rows": facts.test_expected_rows,
            "test_content_sha256": facts.test_data_sha256,
            "test_ordered_schema_sha256": facts.test_ordered_schema_sha256,
        },
        "model": {
            "adapter_id": facts.adapter_id,
            "adapter_sha256": facts.adapter_sha256,
            "candidate_sha256": facts.candidate_sha256,
        },
        "protocol": {
            "plan_id": facts.plan_id,
            "plan_sha256": facts.plan_sha256,
            "pipeline_id": facts.pipeline_id,
            "pipeline_sha256": facts.pipeline_sha256,
        },
        "threshold": {
            "value": threshold_reference["threshold"],
            "policy_id": facts.threshold_policy_id,
            "evidence_path": evidence_relative,
            "evidence_sha256": threshold_reference["sha256"],
            "source_type": THRESHOLD_SOURCE_TYPE,
        },
        "generated_artifacts": {
            "deployment_config": (
                f"{DEPLOYMENT_DRAFT_ROOT}/{candidate_id}/{DEPLOYMENT_CONFIG_NAME}"
            ),
            "threshold_evidence": evidence_relative,
            "candidate_approval": approval_relative,
        },
        "safety": {
            "competition_test_assets_accessed": False,
            "sample_submission_identity_resolved": bool(assets.ready),
            "real_competition_run_ready": bool(assets.ready),
            "competition_asset_fingerprint": assets.asset_fingerprint,
            "competition_blocking_reasons": list(assets.blocking_reasons),
        },
        "competition_assets": {
            "ready": bool(assets.ready),
            "sample_submission_path": assets.sample_submission_path,
            "row_identity_path": assets.row_identity_path,
            "ordered_id_sha256": assets.ordered_id_sha256,
            "test_anchor_hash": assets.test_anchor_hash,
            "asset_fingerprint": assets.asset_fingerprint,
        },
    }
    return DeploymentDraftPayload(
        candidate_id=candidate_id,
        facts=facts,
        deployment_config=deployment_config,
        threshold_evidence=threshold_evidence,
        threshold_reference=threshold_reference,
        candidate_approval=approval,
        provenance=provenance,
    )


def write_deployment_draft(
    repository_root: Path, payload: DeploymentDraftPayload
) -> DeploymentDraft:
    """Persist one deployment draft atomically without silent replacement."""
    root = repository_root.resolve()
    draft_relative = f"{DEPLOYMENT_DRAFT_ROOT}/{payload.candidate_id}"
    draft_dir = root / Path(*PurePosixPath(draft_relative).parts)
    if draft_dir.resolve().parent != (root / DEPLOYMENT_DRAFT_ROOT).resolve():
        raise DeploymentDraftError("Deployment draft directory escapes its root.")
    deterministic = {
        DEPLOYMENT_CONFIG_NAME: _dump_yaml(payload.deployment_config),
        THRESHOLD_EVIDENCE_NAME: _dump_yaml(payload.threshold_evidence),
        PROVENANCE_NAME: _dump_json(payload.provenance),
    }
    approval_bytes = _dump_yaml(payload.candidate_approval)

    reused = True
    if path_exists_nonfollowing(draft_dir) and not draft_dir.is_dir():
        raise DeploymentDraftError(
            f"Deployment draft path is not a directory: {draft_relative}"
        )
    for name, content in deterministic.items():
        existing = _read_existing(draft_dir / name)
        if existing is None:
            reused = False
        elif existing != content:
            raise DeploymentDraftConflictError(
                f"A different deployment draft already exists at "
                f"{draft_relative}/{name}. Remove or archive it before "
                "regenerating this candidate."
            )
    existing_approval = _read_existing(draft_dir / APPROVAL_NAME)
    if existing_approval is None:
        reused = False
    elif existing_approval != approval_bytes:
        recorded = _approval_id(existing_approval)
        if recorded != payload.candidate_approval["approval_id"]:
            raise DeploymentDraftConflictError(
                f"A different candidate approval already exists at "
                f"{draft_relative}/{APPROVAL_NAME}. Historical approval "
                "evidence is never rewritten."
            )
        # Same immutable approval identity: keep the recorded audit metadata.
        approval_bytes = existing_approval

    draft_dir.mkdir(parents=True, exist_ok=True)
    for name, content in {**deterministic, APPROVAL_NAME: approval_bytes}.items():
        _atomic_write(draft_dir / name, content)
    return DeploymentDraft(
        candidate_id=payload.candidate_id,
        draft_dir=draft_relative,
        deployment_config_path=f"{draft_relative}/{DEPLOYMENT_CONFIG_NAME}",
        threshold_evidence_path=f"{draft_relative}/{THRESHOLD_EVIDENCE_NAME}",
        candidate_approval_path=f"{draft_relative}/{APPROVAL_NAME}",
        provenance_path=f"{draft_relative}/{PROVENANCE_NAME}",
        payload=payload,
        reused=reused,
    )


def prepare_deployment_draft(
    repository_root: Path,
    run_relative_path: str,
    *,
    approver: str,
    approved_at_utc: str | None = None,
    intended_deployment_role: str = DEFAULT_INTENDED_ROLE,
    paired_comparison_exception_reason: str = DEFAULT_EXCEPTION_REASON,
    archived_paths: frozenset[str] = frozenset(),
) -> DeploymentDraft:
    """Derive and persist a deployment draft for one completed canonical run."""
    payload = build_deployment_draft_payload(
        repository_root,
        run_relative_path,
        approver=approver,
        approved_at_utc=approved_at_utc or approval_timestamp_now(),
        intended_deployment_role=intended_deployment_role,
        paired_comparison_exception_reason=paired_comparison_exception_reason,
        archived_paths=archived_paths,
    )
    return write_deployment_draft(repository_root, payload)


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _read_existing(path: Path) -> bytes | None:
    if not path_exists_nonfollowing(path):
        return None
    try:
        require_regular_file(path, reject_hardlinks=True)
        return path.read_bytes()
    except (OSError, PathSafetyError) as error:
        raise DeploymentDraftError(
            f"Existing draft artifact cannot be authenticated: {path.name} ({error})"
        ) from error


def _approval_id(raw: bytes) -> str | None:
    try:
        payload = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError):
        return None
    if not isinstance(payload, Mapping):
        return None
    value = payload.get("approval_id")
    return value if isinstance(value, str) else None


def _dump_yaml(payload: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(
        dict(payload), sort_keys=False, allow_unicode=False
    ).encode("utf-8")


def _dump_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DeploymentDraftError(f"A non-empty {label} is required.")
    return value.strip()


def _required_utc(value: Any) -> str:
    text = _required_text(value, "approval timestamp")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise DeploymentDraftError(
            "The approval timestamp must be ISO-8601."
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise DeploymentDraftError("The approval timestamp must be UTC.")
    return text


__all__ = [
    "APPROVAL_NAME",
    "BUILDER_ID",
    "BUILDER_VERSION",
    "DEPLOYMENT_CONFIG_NAME",
    "DEPLOYMENT_DRAFT_ROOT",
    "DEFAULT_EXCEPTION_REASON",
    "DEFAULT_INTENDED_ROLE",
    "PROVENANCE_NAME",
    "THRESHOLD_EVIDENCE_NAME",
    "DeploymentDraft",
    "DeploymentDraftConflictError",
    "DeploymentDraftError",
    "DeploymentDraftPayload",
    "approval_timestamp_now",
    "build_deployment_draft_payload",
    "prepare_deployment_draft",
    "write_deployment_draft",
]
