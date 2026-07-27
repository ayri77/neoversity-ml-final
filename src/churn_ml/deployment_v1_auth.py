from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

import yaml

from src.churn_ml.deployment_v1_paths import (
    DeploymentPathError,
    prewalk_regular_tree,
    validate_existing_root,
    validate_regular_file,
    validate_path_chain,
)
from src.churn_ml.research_data import canonical_sha256


SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
THRESHOLD_SOURCE_TYPES = {"experiment_core_v2_manual_threshold_v1"}
THRESHOLD_REFERENCE_KEYS = {
    "schema_version",
    "path",
    "size_bytes",
    "sha256",
    "source_type",
    "source_run_id",
    "source_manifest_sha256",
    "threshold",
    "threshold_policy_id",
    "plan_sha256",
    "candidate_sha256",
}
THRESHOLD_ARTIFACT_KEYS = THRESHOLD_REFERENCE_KEYS - {"path", "size_bytes", "sha256"}
FIXTURE_KEYS = {
    "schema_version",
    "fixture_id",
    "fixture_type",
    "files",
    "generation",
}
FIXTURE_ROLES = {"train", "labels", "test", "sample_submission"}
FIXTURE_FILE_KEYS = {"path", "size_bytes", "sha256"}
FIXTURE_GENERATION_KEYS = {"generator_id", "seed"}
FORBIDDEN_FIXTURE_COMPONENTS = {
    "competition",
    "data",
    "raw",
    "submission",
    "submissions",
    "kaggle",
    "x_test",
    "sample_submission",
}


class DeploymentAuthenticationError(ValueError):
    """Raised when authenticated P4 evidence or fixtures cannot be proven."""


@dataclass(frozen=True)
class AuthenticatedSource:
    source_type: str
    repository_relative_path: str
    kind: str
    size_bytes: int
    sha256: str
    semantic_identity: str
    path_chain_sha256: str


@dataclass(frozen=True)
class ThresholdEvidence:
    identity: dict[str, Any]
    source: AuthenticatedSource


def authenticate_source(
    path: Path,
    *,
    project_root: Path,
    source_type: str,
    semantic_identity: str,
    expected_kind: str,
) -> AuthenticatedSource:
    try:
        root = validate_existing_root(project_root)
        if expected_kind == "file":
            safe = validate_path_chain(
                containment_root=root,
                requested_path=path,
                require_exists=True,
                expected_kind="file",
                reject_hardlinks=True,
            )
            before_chain = _path_chain_sha256(safe.validated_chain)
            raw = safe.canonical.read_bytes()
            after = validate_path_chain(
                containment_root=root,
                requested_path=path,
                require_exists=True,
                expected_kind="file",
                reject_hardlinks=True,
            )
            if before_chain != _path_chain_sha256(after.validated_chain):
                raise DeploymentAuthenticationError(
                    "Authenticated file path changed while it was read."
                )
            safe = after
            size = len(raw)
            digest = hashlib.sha256(raw).hexdigest()
        elif expected_kind == "directory":
            tree = prewalk_regular_tree(
                path, containment_roots=(root,), reject_hardlinks=True
            )
            safe = tree.path
            before_chain = _path_chain_sha256(safe.validated_chain)
            digest, size = _tree_content_identity(tree)
            after_tree = prewalk_regular_tree(
                path, containment_roots=(root,), reject_hardlinks=True
            )
            after_digest, after_size = _tree_content_identity(after_tree)
            if (
                tree.files != after_tree.files
                or tree.directories != after_tree.directories
                or digest != after_digest
                or size != after_size
                or before_chain != _path_chain_sha256(after_tree.path.validated_chain)
            ):
                raise DeploymentAuthenticationError(
                    "Authenticated directory changed while it was read."
                )
            safe = after_tree.path
        else:
            raise DeploymentAuthenticationError("Authenticated source kind differs.")
    except DeploymentPathError as error:
        raise DeploymentAuthenticationError(str(error)) from error
    relative_path = safe.canonical.relative_to(root).as_posix()
    chain_identity = _path_chain_sha256(safe.validated_chain)
    return AuthenticatedSource(
        source_type=source_type,
        repository_relative_path=relative_path,
        kind=expected_kind,
        size_bytes=size,
        sha256=digest,
        semantic_identity=semantic_identity,
        path_chain_sha256=chain_identity,
    )


def _tree_content_identity(tree: Any) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for relative in tree.directories:
        digest.update(b"D\0")
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
    for relative in tree.files:
        raw = (tree.root / relative).read_bytes()
        size += len(raw)
        digest.update(b"F\0")
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest(), size


def _path_chain_sha256(chain: tuple[Path, ...]) -> str:
    records = []
    for item in chain:
        metadata = item.lstat()
        records.append(
            {
                "path": str(item.absolute()),
                "device": int(metadata.st_dev),
                "inode": int(metadata.st_ino),
                "mode": int(metadata.st_mode),
                "attributes": int(getattr(metadata, "st_file_attributes", 0)),
            }
        )
    return canonical_sha256(records)


def reauthenticate_source(
    expected: AuthenticatedSource, *, project_root: Path
) -> AuthenticatedSource:
    actual = authenticate_source(
        project_root / Path(*PurePosixPath(expected.repository_relative_path).parts),
        project_root=project_root,
        source_type=expected.source_type,
        semantic_identity=expected.semantic_identity,
        expected_kind=expected.kind,
    )
    if actual != expected:
        raise DeploymentAuthenticationError(
            f"Authenticated source changed: {expected.repository_relative_path}"
        )
    return actual


@dataclass(frozen=True)
class SyntheticFixture:
    root: Path
    manifest_path: Path
    payload: dict[str, Any]
    paths: dict[str, Path]
    identity: dict[str, Any]
    source: AuthenticatedSource


def load_threshold_evidence(
    reference_value: Any,
    *,
    project_root: Path,
    research_run: Any,
) -> ThresholdEvidence:
    reference = _mapping(reference_value, "threshold_evidence")
    _exact_keys(reference, THRESHOLD_REFERENCE_KEYS, "threshold_evidence")
    _schema_one(reference["schema_version"], "threshold_evidence.schema_version")
    path = _portable_path(reference["path"], project_root, "threshold_evidence.path")
    try:
        source = validate_regular_file(
            path,
            containment_root=project_root,
            reject_hardlinks=True,
        )
    except DeploymentPathError as error:
        raise DeploymentAuthenticationError(str(error)) from error
    raw = source.read_bytes()
    _positive_int(reference["size_bytes"], "threshold_evidence.size_bytes")
    if len(raw) != reference["size_bytes"]:
        raise DeploymentAuthenticationError("Threshold evidence size differs.")
    _sha(reference["sha256"], "threshold_evidence.sha256")
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise DeploymentAuthenticationError("Threshold evidence hash differs.")
    artifact = _load_mapping_bytes(raw, source)
    _exact_keys(artifact, THRESHOLD_ARTIFACT_KEYS, "threshold evidence artifact")
    expected_artifact = {
        key: value
        for key, value in reference.items()
        if key not in {"path", "size_bytes", "sha256"}
    }
    if artifact != expected_artifact:
        raise DeploymentAuthenticationError(
            "Threshold evidence reference differs from authenticated artifact."
        )
    if reference["source_type"] not in THRESHOLD_SOURCE_TYPES:
        raise DeploymentAuthenticationError(
            "Threshold evidence source_type is invalid."
        )
    _safe_id(reference["source_run_id"], "threshold_evidence.source_run_id")
    _safe_id(
        reference["threshold_policy_id"],
        "threshold_evidence.threshold_policy_id",
    )
    for key in ("source_manifest_sha256", "plan_sha256", "candidate_sha256"):
        _sha(reference[key], f"threshold_evidence.{key}")
    threshold = _float(reference["threshold"], "threshold_evidence.threshold")
    if not 0.0 <= threshold <= 1.0:
        raise DeploymentAuthenticationError("Threshold evidence value is out of range.")
    run_plan = _read_json(research_run.root / "identities/evaluation_plan.json")
    run_candidate = _read_json(research_run.root / "identities/candidate.json")
    expected = {
        "source_run_id": research_run.metadata["run_id"],
        "source_manifest_sha256": research_run.manifest["manifest_sha256"],
        "plan_sha256": run_plan["sha256"],
        "candidate_sha256": run_candidate["sha256"],
    }
    for key, value in expected.items():
        if reference[key] != value:
            raise DeploymentAuthenticationError(
                f"Threshold evidence {key} differs from completed research run."
            )
    canonical = deepcopy(reference)
    identity = {
        "schema_version": 1,
        "sha256": canonical_sha256(canonical),
        "canonical": canonical,
    }
    source_identity = canonical_sha256(identity)
    return ThresholdEvidence(
        identity=identity,
        source=authenticate_source(
            source,
            project_root=project_root,
            source_type="threshold_evidence",
            semantic_identity=source_identity,
            expected_kind="file",
        ),
    )


def derive_approval_id(payload: Mapping[str, Any]) -> str:
    manual = _mapping(payload.get("manual_approval"), "manual_approval")
    canonical = {
        "schema_version": payload.get("schema_version"),
        "component_id": payload.get("component_id"),
        "research_run": deepcopy(payload.get("research_run")),
        "fixed_resolved_model_parameters": deepcopy(
            payload.get("fixed_resolved_model_parameters")
        ),
        "threshold_evidence": deepcopy(payload.get("threshold_evidence")),
        "paired_comparison": deepcopy(payload.get("paired_comparison")),
        "manual_approval_status": manual.get("status"),
        "intended_deployment_role": payload.get("intended_deployment_role"),
    }
    return canonical_sha256(canonical)


def approval_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    derived = derive_approval_id(payload)
    canonical = {
        "schema_version": 1,
        "approval_id": derived,
        "identity_policy": {
            "included": [
                "schema_version",
                "component_id",
                "research_run",
                "fixed_resolved_model_parameters",
                "threshold_evidence",
                "paired_comparison",
                "manual_approval.status",
                "intended_deployment_role",
            ],
            "excluded_audit_fields": [
                "component_name",
                "manual_approval.approver",
                "manual_approval.approved_at_utc",
            ],
        },
    }
    return {"sha256": canonical_sha256(canonical), "canonical": canonical}


def load_synthetic_fixture(
    fixture_dir: Path,
    *,
    project_root: Path,
    forbidden_hashes: set[str],
) -> SyntheticFixture:
    try:
        root = validate_existing_root(project_root)
    except DeploymentPathError as error:
        raise DeploymentAuthenticationError(str(error)) from error
    unresolved = fixture_dir if fixture_dir.is_absolute() else root / fixture_dir
    try:
        tree = prewalk_regular_tree(
            unresolved,
            containment_roots=(root,),
            reject_hardlinks=True,
        )
    except DeploymentPathError as error:
        raise DeploymentAuthenticationError(str(error)) from error
    relative_root_parts = {
        part.casefold() for part in tree.root.relative_to(root).parts
    }
    if relative_root_parts.intersection(FORBIDDEN_FIXTURE_COMPONENTS):
        raise DeploymentAuthenticationError(
            "Synthetic fixture uses a forbidden competition namespace."
        )
    if tree.directories:
        raise DeploymentAuthenticationError(
            "Synthetic fixture must not contain nested directories."
        )
    manifest_path = tree.root / "fixture_manifest.yaml"
    if Path("fixture_manifest.yaml") not in tree.files:
        raise DeploymentAuthenticationError("Synthetic fixture manifest is missing.")
    payload = _load_mapping_bytes(manifest_path.read_bytes(), manifest_path)
    _exact_keys(payload, FIXTURE_KEYS, "fixture manifest")
    _schema_one(payload["schema_version"], "fixture.schema_version")
    if payload["fixture_type"] != "synthetic_noncompetition":
        raise DeploymentAuthenticationError(
            "fixture_type must be synthetic_noncompetition."
        )
    files = _mapping(payload["files"], "fixture.files")
    _exact_keys(files, FIXTURE_ROLES, "fixture.files")
    generation = _mapping(payload["generation"], "fixture.generation")
    _exact_keys(generation, FIXTURE_GENERATION_KEYS, "fixture.generation")
    _safe_id(generation["generator_id"], "fixture.generation.generator_id")
    if type(generation["seed"]) is not int:
        raise DeploymentAuthenticationError("fixture generation seed must be integer.")
    paths: dict[str, Path] = {}
    expected_names = {"fixture_manifest.yaml"}
    for role in sorted(FIXTURE_ROLES):
        record = _mapping(files[role], f"fixture.files.{role}")
        _exact_keys(record, FIXTURE_FILE_KEYS, f"fixture.files.{role}")
        relative = _plain_basename(record["path"], f"fixture.files.{role}.path")
        expected_names.add(relative)
        path = tree.root / relative
        if Path(relative) not in tree.files:
            raise DeploymentAuthenticationError(f"Fixture file is missing: {role}.")
        raw = path.read_bytes()
        _positive_int(record["size_bytes"], f"fixture.files.{role}.size_bytes")
        if len(raw) != record["size_bytes"]:
            raise DeploymentAuthenticationError(f"Fixture size differs: {role}.")
        _sha(record["sha256"], f"fixture.files.{role}.sha256")
        actual_hash = hashlib.sha256(raw).hexdigest()
        if actual_hash != record["sha256"]:
            raise DeploymentAuthenticationError(f"Fixture hash differs: {role}.")
        if actual_hash in forbidden_hashes:
            raise DeploymentAuthenticationError(
                "Synthetic fixture matches an authenticated competition asset."
            )
        paths[role] = path
    actual_names = {item.as_posix() for item in tree.files}
    if actual_names != expected_names:
        raise DeploymentAuthenticationError(
            "Synthetic fixture contains missing or extra files."
        )
    identity_inputs = {
        key: deepcopy(value) for key, value in payload.items() if key != "fixture_id"
    }
    derived_id = canonical_sha256(identity_inputs)
    if payload["fixture_id"] != derived_id:
        raise DeploymentAuthenticationError("Synthetic fixture_id is not canonical.")
    canonical = {
        **deepcopy(payload),
        "repository_relative_path": tree.root.relative_to(root).as_posix(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    return SyntheticFixture(
        root=tree.root,
        manifest_path=manifest_path,
        payload=deepcopy(payload),
        paths=paths,
        identity={
            "schema_version": 1,
            "mode": "synthetic",
            "sha256": canonical_sha256(canonical),
            "canonical": canonical,
        },
        source=authenticate_source(
            tree.root,
            project_root=root,
            source_type="synthetic_fixture",
            semantic_identity=canonical_sha256(canonical),
            expected_kind="directory",
        ),
    )


def _load_mapping_bytes(raw: bytes, path: Path) -> dict[str, Any]:
    try:
        value = (
            json.loads(raw.decode("utf-8"))
            if path.suffix.casefold() == ".json"
            else yaml.safe_load(raw.decode("utf-8"))
        )
    except (UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise DeploymentAuthenticationError(
            f"Authenticated mapping is malformed: {path}"
        ) from error
    if not isinstance(value, dict):
        raise DeploymentAuthenticationError("Authenticated artifact must be a mapping.")
    return dict(value)


def _portable_path(value: Any, root: Path, label: str) -> Path:
    text = _string(value, label)
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        "\\" in text
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in posix.parts
        or "." in posix.parts
    ):
        raise DeploymentAuthenticationError(f"{label} must be repository-relative.")
    try:
        return validate_path_chain(
            containment_root=root,
            requested_path=root / Path(*posix.parts),
            require_exists=False,
            expected_kind="either",
        ).canonical
    except DeploymentPathError as error:
        raise DeploymentAuthenticationError(str(error)) from error


def _plain_basename(value: Any, label: str) -> str:
    text = _string(value, label)
    if "/" in text or "\\" in text or text in {".", ".."}:
        raise DeploymentAuthenticationError(f"{label} must be a plain basename.")
    return text


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DeploymentAuthenticationError(f"Cannot read identity: {path}") from error
    if not isinstance(value, dict):
        raise DeploymentAuthenticationError("Identity artifact must be an object.")
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DeploymentAuthenticationError(f"{label} must be a mapping.")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise DeploymentAuthenticationError(f"{label} keys differ.")


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DeploymentAuthenticationError(f"{label} must be a non-empty string.")
    return value


def _safe_id(value: Any, label: str) -> str:
    text = _string(value, label)
    if SAFE_ID.fullmatch(text) is None:
        raise DeploymentAuthenticationError(f"{label} must be a safe identifier.")
    return text


def _sha(value: Any, label: str) -> str:
    text = _string(value, label)
    if SHA256.fullmatch(text) is None:
        raise DeploymentAuthenticationError(f"{label} must be a lowercase SHA-256.")
    return text


def _schema_one(value: Any, label: str) -> None:
    if type(value) is not int or value != 1:
        raise DeploymentAuthenticationError(f"{label} must be integer 1.")


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise DeploymentAuthenticationError(f"{label} must be a positive integer.")
    return value


def _float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise DeploymentAuthenticationError(f"{label} must be a finite float.")
    return value
