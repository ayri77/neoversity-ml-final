"""Authenticated competition submission assets and test-row identity.

Submission IDs are not model features. They live in a separate authenticated
row-identity contract linked to Dataset Package alignment proofs:

    row_position → submission_id

The original competition files under ``data/raw/`` are never modified. This
module only authenticates them and derives a portable identity artifact.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import pandas as pd
import yaml

from src.churn_ml.control_panel.path_safety import (
    PathSafetyError,
    path_exists_nonfollowing,
    require_regular_file,
)
from src.churn_ml.research_data import canonical_sha256


COMPETITION_ASSET_SCHEMA_VERSION = 1
SAMPLE_SUBMISSION_PATH = "data/raw/final_proj_sample_submission.csv"
RAW_TEST_PATH = "data/raw/final_proj_test.csv"
ROW_IDENTITY_ARTIFACT_PATH = "data/competition/test_row_identity_v1.json"
COMPETITION_ASSET_REGISTRY_PATH = "configs/competition/competition_assets_v1.yaml"
DEFAULT_ID_COLUMN = "index"
DEFAULT_TARGET_COLUMN = "y"
ID_SEMANTICS = "zero_based_row_position"
ALIGNMENT_METHOD = "ordered_v0_raw_minimal_feature_projection"
PROCESSED_ROOT = "data/processed"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_BYTES = 8_000_000

ROW_IDENTITY_KEYS = {
    "schema_version",
    "asset_id",
    "path",
    "size_bytes",
    "sha256",
    "expected_rows",
    "id_column",
    "id_dtype",
    "id_semantics",
    "ordered_id_sha256",
    "row_position_identity_sha256",
    "source_sample_submission",
    "source_test",
    "dataset_package_link",
}
_ROW_IDENTITY_HASH_EXCLUDED = frozenset({"sha256", "size_bytes"})


@dataclass(frozen=True)
class CompetitionAssetResolution:
    """Resolved, authenticated competition submission assets."""

    ready: bool
    blocking_reasons: tuple[str, ...]
    sample_submission_path: str | None
    sample_submission_sha256: str | None
    sample_expected_rows: int | None
    id_column: str | None
    target_column: str | None
    ordered_id_sha256: str | None
    row_identity_path: str | None
    row_identity_sha256: str | None
    test_anchor_hash: str | None
    asset_fingerprint: str | None

    def summary(self) -> dict[str, str]:
        return {
            "Competition submission readiness": (
                "ready" if self.ready else "blocked"
            ),
            "Sample submission": self.sample_submission_path or "unavailable",
            "Submission ID column": self.id_column or "unavailable",
            "Target column": self.target_column or "unavailable",
            "Expected rows": (
                str(self.sample_expected_rows)
                if self.sample_expected_rows is not None
                else "unavailable"
            ),
            "Ordered ID hash": self.ordered_id_sha256 or "unavailable",
            "Row-identity artifact": self.row_identity_path or "unavailable",
            "Dataset Package test anchor": self.test_anchor_hash or "unavailable",
        }


def resolve_competition_assets(
    repository_root: Path,
    *,
    dataset_version: str | None = None,
) -> CompetitionAssetResolution:
    """Authenticate local competition assets and Dataset Package alignment."""
    root = repository_root.resolve()
    blocking: list[str] = []
    sample_path = SAMPLE_SUBMISSION_PATH
    sample_file = root / Path(*PurePosixPath(sample_path).parts)
    if not path_exists_nonfollowing(sample_file):
        blocking.append(f"Sample submission is missing: {sample_path}")
        return _blocked(blocking)

    try:
        require_regular_file(sample_file, reject_hardlinks=True)
        sample_raw = sample_file.read_bytes()
    except (OSError, PathSafetyError) as error:
        blocking.append(f"Sample submission cannot be read: {error}")
        return _blocked(blocking)
    if len(sample_raw) > _MAX_BYTES:
        blocking.append("Sample submission exceeds the authenticated size bound.")
        return _blocked(blocking)

    sample_sha = hashlib.sha256(sample_raw).hexdigest()
    try:
        sample = pd.read_csv(sample_file)
    except Exception as error:  # noqa: BLE001
        blocking.append(f"Sample submission is unreadable as CSV: {error}")
        return _blocked(blocking)

    expected_columns = [DEFAULT_ID_COLUMN, DEFAULT_TARGET_COLUMN]
    if sample.columns.tolist() != expected_columns:
        blocking.append(
            "Sample submission columns must be exactly "
            f"{expected_columns}; got {sample.columns.tolist()}."
        )
        return _blocked(blocking)
    if len(sample) <= 0:
        blocking.append("Sample submission has no rows.")
        return _blocked(blocking)

    ids = sample[DEFAULT_ID_COLUMN]
    if str(ids.dtype) != "int64" or ids.isna().any() or ids.duplicated().any():
        blocking.append("Sample submission IDs must be unique non-null int64 values.")
        return _blocked(blocking)
    if ids.tolist() != list(range(len(sample))):
        blocking.append(
            "Sample submission IDs are not exact zero-based row positions; "
            "alignment cannot use the positional identity contract."
        )
        return _blocked(blocking)
    targets = sample[DEFAULT_TARGET_COLUMN]
    if set(targets.unique().tolist()) - {0, 1}:
        blocking.append("Sample submission target placeholders must be 0/1 only.")
        return _blocked(blocking)

    ordered_id_sha = canonical_sha256(ids.tolist())
    row_position_sha = canonical_sha256(list(range(len(sample))))
    if ordered_id_sha != row_position_sha:
        blocking.append("Ordered submission ID hash differs from row-position identity.")
        return _blocked(blocking)

    test_path = RAW_TEST_PATH
    test_file = root / Path(*PurePosixPath(test_path).parts)
    test_sha: str | None = None
    test_rows: int | None = None
    if not path_exists_nonfollowing(test_file):
        blocking.append(f"Raw competition test file is missing: {test_path}")
    else:
        try:
            require_regular_file(test_file, reject_hardlinks=True)
            test_raw = test_file.read_bytes()
            test_sha = hashlib.sha256(test_raw).hexdigest()
            test_frame = pd.read_csv(test_file)
            test_rows = len(test_frame)
            if test_rows != len(sample):
                blocking.append(
                    "Raw competition test row count differs from sample submission."
                )
            if DEFAULT_ID_COLUMN in test_frame.columns:
                blocking.append(
                    "Raw competition test unexpectedly contains the submission ID "
                    "column; IDs must remain outside the feature matrix."
                )
        except (OSError, PathSafetyError, UnicodeDecodeError, ValueError) as error:
            blocking.append(f"Raw competition test cannot be authenticated: {error}")

    test_anchor_hash: str | None = None
    if dataset_version:
        package = _package_row_identity(root, dataset_version)
        if package is None:
            blocking.append(
                f"Dataset Package row identity is unavailable for {dataset_version}."
            )
        else:
            if package.get("alignment_status") != "proven":
                blocking.append(
                    f"Dataset Package {dataset_version} row alignment is not proven."
                )
            test_anchor_hash = _sha_or_none(package.get("test_anchor_hash"))
            if test_anchor_hash is None:
                blocking.append(
                    f"Dataset Package {dataset_version} lacks a test_anchor_hash."
                )
            expected_rows = package.get("test_row_count")
            if expected_rows != len(sample):
                blocking.append(
                    "Dataset Package test_row_count differs from sample submission."
                )
    else:
        # Without a selected package, still require the shared proven anchor
        # across registered packages when available.
        test_anchor_hash = _shared_proven_test_anchor(root)
        if test_anchor_hash is None:
            blocking.append(
                "No shared proven Dataset Package test_anchor_hash is available."
            )

    if blocking:
        return _blocked(blocking)

    identity_payload = build_row_identity_payload(
        sample_submission_path=sample_path,
        sample_submission_sha256=sample_sha,
        sample_size_bytes=len(sample_raw),
        expected_rows=len(sample),
        ordered_id_sha256=ordered_id_sha,
        row_position_identity_sha256=row_position_sha,
        source_test_path=test_path,
        source_test_sha256=str(test_sha),
        source_test_rows=int(test_rows or 0),
        test_anchor_hash=str(test_anchor_hash),
    )
    identity_sha = str(identity_payload["sha256"])
    identity_file = root / Path(*PurePosixPath(ROW_IDENTITY_ARTIFACT_PATH).parts)
    if path_exists_nonfollowing(identity_file):
        try:
            loaded = load_row_identity_artifact(
                root, ROW_IDENTITY_ARTIFACT_PATH, expected_sha256=identity_sha
            )
            identity_sha = str(loaded["sha256"])
        except (OSError, ValueError, PathSafetyError) as error:
            blocking.append(
                f"Existing row-identity artifact failed authentication: {error}"
            )
            return _blocked(blocking)
    fingerprint = canonical_sha256(
        {
            "sample_submission_sha256": sample_sha,
            "row_identity_sha256": identity_sha,
            "ordered_id_sha256": ordered_id_sha,
            "test_anchor_hash": test_anchor_hash,
        }
    )
    return CompetitionAssetResolution(
        ready=True,
        blocking_reasons=(),
        sample_submission_path=sample_path,
        sample_submission_sha256=sample_sha,
        sample_expected_rows=len(sample),
        id_column=DEFAULT_ID_COLUMN,
        target_column=DEFAULT_TARGET_COLUMN,
        ordered_id_sha256=ordered_id_sha,
        row_identity_path=ROW_IDENTITY_ARTIFACT_PATH,
        row_identity_sha256=identity_sha,
        test_anchor_hash=test_anchor_hash,
        asset_fingerprint=fingerprint,
    )


def build_row_identity_payload(
    *,
    sample_submission_path: str,
    sample_submission_sha256: str,
    sample_size_bytes: int,
    expected_rows: int,
    ordered_id_sha256: str,
    row_position_identity_sha256: str,
    source_test_path: str,
    source_test_sha256: str,
    source_test_rows: int,
    test_anchor_hash: str,
) -> dict[str, Any]:
    """Build the canonical submission/test-row identity document."""
    body = {
        "schema_version": COMPETITION_ASSET_SCHEMA_VERSION,
        "path": ROW_IDENTITY_ARTIFACT_PATH,
        "expected_rows": expected_rows,
        "id_column": DEFAULT_ID_COLUMN,
        "id_dtype": "int64",
        "id_semantics": ID_SEMANTICS,
        "ordered_id_sha256": ordered_id_sha256,
        "row_position_identity_sha256": row_position_identity_sha256,
        "source_sample_submission": {
            "path": sample_submission_path,
            "sha256": sample_submission_sha256,
            "size_bytes": sample_size_bytes,
            "expected_rows": expected_rows,
            "columns": [DEFAULT_ID_COLUMN, DEFAULT_TARGET_COLUMN],
            "id_column": DEFAULT_ID_COLUMN,
            "target_column": DEFAULT_TARGET_COLUMN,
            "target_placeholder_values": [0],
        },
        "source_test": {
            "path": source_test_path,
            "sha256": source_test_sha256,
            "expected_rows": source_test_rows,
            "contains_submission_id_column": False,
        },
        "dataset_package_link": {
            "alignment_method": ALIGNMENT_METHOD,
            "alignment_status": "proven",
            "test_anchor_hash": test_anchor_hash,
        },
    }
    asset_id = canonical_sha256(body)
    core = {**body, "asset_id": asset_id}
    digest = hashlib.sha256(_dump_json(core)).hexdigest()
    # size_bytes is a file property checked against the written bytes; it is not
    # part of the identity digest. Placeholder zero is replaced at write time.
    return {**core, "size_bytes": 0, "sha256": digest}


def write_row_identity_artifact(
    repository_root: Path, payload: Mapping[str, Any]
) -> str:
    """Persist the row-identity artifact atomically; return repository-relative path."""
    root = repository_root.resolve()
    relative = ROW_IDENTITY_ARTIFACT_PATH
    path = root / Path(*PurePosixPath(relative).parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    core = {
        key: value
        for key, value in dict(payload).items()
        if key not in _ROW_IDENTITY_HASH_EXCLUDED
    }
    digest = hashlib.sha256(_dump_json(core)).hexdigest()
    provisional = {**core, "sha256": digest, "size_bytes": 0}
    size = len(_dump_json(provisional))
    document = {**core, "sha256": digest, "size_bytes": size}
    # If digit growth changed the length, adjust once more.
    size = len(_dump_json(document))
    document = {**core, "sha256": digest, "size_bytes": size}
    final_bytes = _dump_json(document)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_bytes(final_bytes)
    temporary.replace(path)
    return relative


def ensure_competition_row_identity(
    repository_root: Path, *, dataset_version: str | None
) -> CompetitionAssetResolution:
    """Resolve assets and ensure the row-identity artifact exists on disk."""
    resolution = resolve_competition_assets(
        repository_root, dataset_version=dataset_version
    )
    if not resolution.ready:
        return resolution
    root = repository_root.resolve()
    payload = build_row_identity_payload(
        sample_submission_path=str(resolution.sample_submission_path),
        sample_submission_sha256=str(resolution.sample_submission_sha256),
        sample_size_bytes=(
            root / Path(*PurePosixPath(str(resolution.sample_submission_path)).parts)
        ).stat().st_size,
        expected_rows=int(resolution.sample_expected_rows or 0),
        ordered_id_sha256=str(resolution.ordered_id_sha256),
        row_position_identity_sha256=str(resolution.ordered_id_sha256),
        source_test_path=RAW_TEST_PATH,
        source_test_sha256=hashlib.sha256(
            (root / Path(*PurePosixPath(RAW_TEST_PATH).parts)).read_bytes()
        ).hexdigest(),
        source_test_rows=int(resolution.sample_expected_rows or 0),
        test_anchor_hash=str(resolution.test_anchor_hash),
    )
    write_row_identity_artifact(root, payload)
    # Re-resolve so sha/fingerprint match the written artifact.
    return resolve_competition_assets(
        repository_root, dataset_version=dataset_version
    )


def load_row_identity_artifact(
    repository_root: Path, relative_path: str, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    """Load and authenticate one submission/test-row identity artifact."""
    root = repository_root.resolve()
    text = str(relative_path).replace("\\", "/")
    posix = PurePosixPath(text)
    if posix.is_absolute() or ".." in posix.parts or "." in posix.parts:
        raise ValueError("Row-identity path must be repository-relative.")
    path = root / Path(*posix.parts)
    require_regular_file(path, reject_hardlinks=True)
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Row-identity artifact must be a JSON object.")
    missing = ROW_IDENTITY_KEYS - set(payload)
    unknown = set(payload) - ROW_IDENTITY_KEYS
    if missing or unknown:
        raise ValueError(
            f"Row-identity keys differ; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}."
        )
    if int(payload["size_bytes"]) != len(raw):
        raise ValueError("Row-identity artifact size differs.")
    core = {
        key: value
        for key, value in payload.items()
        if key not in _ROW_IDENTITY_HASH_EXCLUDED
    }
    actual = hashlib.sha256(_dump_json(core)).hexdigest()
    if payload["sha256"] != actual:
        raise ValueError("Row-identity artifact hash differs.")
    if expected_sha256 is not None and payload["sha256"] != expected_sha256:
        raise ValueError(
            "Row-identity artifact hash differs from the deployment config."
        )
    if int(payload["expected_rows"]) <= 0:
        raise ValueError("Row-identity expected_rows must be positive.")
    if payload["id_semantics"] != ID_SEMANTICS:
        raise ValueError("Row-identity id_semantics differs.")
    if payload["id_column"] != DEFAULT_ID_COLUMN:
        raise ValueError("Row-identity id_column differs.")
    return dict(payload)


def submission_ids_from_identity(payload: Mapping[str, Any]) -> list[int]:
    """Materialize ordered submission IDs from an authenticated identity payload."""
    rows = int(payload["expected_rows"])
    if payload["id_semantics"] != ID_SEMANTICS:
        raise ValueError("Unsupported submission ID semantics.")
    ids = list(range(rows))
    if canonical_sha256(ids) != payload["ordered_id_sha256"]:
        raise ValueError("Materialized submission IDs disagree with ordered_id_sha256.")
    return ids


def require_resolved_competition_draft(
    repository_root: Path, config_relative: str
) -> None:
    """Fail closed unless the deployment config carries authenticated IDs.

    Generate submission must not launch against an unresolved placeholder draft.
    """
    root = repository_root.resolve()
    text = str(config_relative).replace("\\", "/")
    posix = PurePosixPath(text)
    if posix.is_absolute() or ".." in posix.parts or "." in posix.parts:
        raise ValueError("Deployment config path must be repository-relative.")
    path = root / Path(*posix.parts)
    require_regular_file(path, reject_hardlinks=True)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"Deployment config cannot be read: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Deployment config must be a mapping.")
    sample = payload.get("sample_submission")
    if not isinstance(sample, Mapping):
        raise ValueError("Deployment config lacks sample_submission.")
    sample_path = str(sample.get("path") or "")
    if "UNRESOLVED" in sample_path:
        raise ValueError(
            "Generate submission requires a resolved sample-submission identity."
        )
    row_identity = payload.get("submission_row_identity")
    if not isinstance(row_identity, Mapping):
        raise ValueError(
            "Generate submission requires authenticated submission_row_identity."
        )
    load_row_identity_artifact(
        root,
        str(row_identity["path"]),
        expected_sha256=str(row_identity.get("sha256")),
    )
    assets = resolve_competition_assets(
        root, dataset_version=str(payload.get("dataset_version") or "") or None
    )
    if not assets.ready:
        raise ValueError(
            "Competition submission assets are not ready: "
            + " ".join(assets.blocking_reasons)
        )


def write_competition_asset_registry(
    repository_root: Path, resolution: CompetitionAssetResolution
) -> str:
    """Write the version-controlled competition asset registry summary."""
    if not resolution.ready:
        raise ValueError("Cannot write a registry for unresolved competition assets.")
    root = repository_root.resolve()
    relative = COMPETITION_ASSET_REGISTRY_PATH
    path = root / Path(*PurePosixPath(relative).parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": COMPETITION_ASSET_SCHEMA_VERSION,
        "sample_submission": {
            "path": resolution.sample_submission_path,
            "sha256": resolution.sample_submission_sha256,
            "expected_rows": resolution.sample_expected_rows,
            "columns": [DEFAULT_ID_COLUMN, DEFAULT_TARGET_COLUMN],
            "id_column": DEFAULT_ID_COLUMN,
            "target_column": DEFAULT_TARGET_COLUMN,
            "ordered_id_sha256": resolution.ordered_id_sha256,
            "target_placeholder_semantics": "all_zeros_allowed",
        },
        "test_row_identity": {
            "path": resolution.row_identity_path,
            "sha256": resolution.row_identity_sha256,
            "expected_rows": resolution.sample_expected_rows,
            "id_column": DEFAULT_ID_COLUMN,
            "id_dtype": "int64",
            "id_semantics": ID_SEMANTICS,
            "test_anchor_hash": resolution.test_anchor_hash,
        },
        "asset_fingerprint": resolution.asset_fingerprint,
        "notes": [
            "Submission IDs are separate from model features.",
            "Dataset Packages link through the proven test_anchor_hash.",
            "Original competition files under data/raw are never modified.",
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return relative


def _package_row_identity(root: Path, dataset_version: str) -> dict[str, Any] | None:
    manifest_path = root / PROCESSED_ROOT / dataset_version / "dataset_manifest.json"
    if not path_exists_nonfollowing(manifest_path):
        return None
    try:
        require_regular_file(manifest_path, reject_hardlinks=True)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, PathSafetyError):
        return None
    if not isinstance(payload, Mapping):
        return None
    identity = payload.get("row_identity")
    if not isinstance(identity, Mapping):
        return None
    result = dict(identity)
    result["test_row_count"] = payload.get("test_row_count")
    return result


def _shared_proven_test_anchor(root: Path) -> str | None:
    processed = root / PROCESSED_ROOT
    if not processed.is_dir():
        return None
    anchors: set[str] = set()
    for child in processed.iterdir():
        if not child.is_dir():
            continue
        identity = _package_row_identity(root, child.name)
        if identity is None:
            continue
        if identity.get("alignment_status") != "proven":
            continue
        sha = _sha_or_none(identity.get("test_anchor_hash"))
        if sha is not None:
            anchors.add(sha)
    if len(anchors) == 1:
        return next(iter(anchors))
    return None


def _blocked(reasons: list[str]) -> CompetitionAssetResolution:
    return CompetitionAssetResolution(
        ready=False,
        blocking_reasons=tuple(dict.fromkeys(reasons)),
        sample_submission_path=None,
        sample_submission_sha256=None,
        sample_expected_rows=None,
        id_column=None,
        target_column=None,
        ordered_id_sha256=None,
        row_identity_path=None,
        row_identity_sha256=None,
        test_anchor_hash=None,
        asset_fingerprint=None,
    )


def _sha_or_none(value: Any) -> str | None:
    if isinstance(value, str) and _SHA256.fullmatch(value):
        return value
    return None


def _dump_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


__all__ = [
    "ALIGNMENT_METHOD",
    "COMPETITION_ASSET_REGISTRY_PATH",
    "COMPETITION_ASSET_SCHEMA_VERSION",
    "DEFAULT_ID_COLUMN",
    "DEFAULT_TARGET_COLUMN",
    "ID_SEMANTICS",
    "RAW_TEST_PATH",
    "ROW_IDENTITY_ARTIFACT_PATH",
    "SAMPLE_SUBMISSION_PATH",
    "CompetitionAssetResolution",
    "build_row_identity_payload",
    "ensure_competition_row_identity",
    "load_row_identity_artifact",
    "require_resolved_competition_draft",
    "resolve_competition_assets",
    "submission_ids_from_identity",
    "write_competition_asset_registry",
    "write_row_identity_artifact",
]
