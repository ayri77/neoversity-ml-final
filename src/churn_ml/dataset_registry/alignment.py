"""Row-identity and alignment proof for dataset packages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.churn_ml.dataset_registry.constants import (
    ROW_IDENTITY_ANCHOR_DATASET_ID,
    ROW_IDENTITY_METHOD,
)
from src.churn_ml.dataset_registry.errors import UnverifiableAlignmentError
from src.churn_ml.dataset_registry.hashing import dataframe_content_sha256
from src.churn_ml.dataset_registry.package import (
    PackageArtifacts,
    load_package_artifacts,
    package_dir_for,
)
from src.churn_ml.dataset_registry.schema import RowIdentitySpec


@dataclass(frozen=True)
class AlignmentProof:
    status: str
    method: str
    train_hash: str
    test_hash: str
    train_anchor_hash: str
    test_anchor_hash: str
    anchor_feature_names: tuple[str, ...]


def project_ordered_features(
    frame: pd.DataFrame,
    feature_names: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise UnverifiableAlignmentError(
            "Cannot prove row alignment; missing anchor features: "
            f"{missing}."
        )
    projected = frame.loc[:, list(feature_names)]
    if list(projected.columns) != list(feature_names):
        raise UnverifiableAlignmentError(
            "Anchor feature projection does not preserve the required order."
        )
    return projected


def load_anchor_feature_names(root: Path) -> tuple[str, ...]:
    """
    Load the ordered v0_raw_minimal feature names used as the row-identity anchor.

    Assumption: every implemented legacy package under root retains these columns
    with unchanged values and row order. Legacy metadata.json stores
    parent/base_version/source lineage but no explicit row-identity key, so the
    ordered v0 projection is the verifiable identity anchor.
    """
    anchor_dir = package_dir_for(root, ROW_IDENTITY_ANCHOR_DATASET_ID)
    artifacts = load_package_artifacts(
        anchor_dir,
        dataset_id=ROW_IDENTITY_ANCHOR_DATASET_ID,
    )
    return tuple(artifacts.X_train.columns.tolist())


def prove_alignment(
    artifacts: PackageArtifacts,
    *,
    anchor_feature_names: tuple[str, ...] | None = None,
    parent_artifacts: PackageArtifacts | None = None,
    root: Path | None = None,
) -> AlignmentProof:
    """
    Prove train/test row identities and parent/child row-order continuity.

    Train identity is derived from the ordered anchor feature projection of
    X_train, never from y alone. Parent continuity requires identical train and
    test anchor hashes between parent and child when a parent is supplied.
    """
    names = anchor_feature_names
    if names is None:
        if root is None:
            raise UnverifiableAlignmentError(
                "Cannot prove row alignment without anchor features or registry root."
            )
        names = load_anchor_feature_names(root)

    if not names:
        raise UnverifiableAlignmentError("Anchor feature list is empty.")

    try:
        train_anchor = project_ordered_features(artifacts.X_train, names)
        test_anchor = project_ordered_features(artifacts.X_test, names)
    except UnverifiableAlignmentError:
        raise
    except Exception as error:  # noqa: BLE001 - convert to unverifiable
        raise UnverifiableAlignmentError(
            f"Cannot prove row alignment: {error}"
        ) from error

    train_anchor_hash = dataframe_content_sha256(train_anchor)
    test_anchor_hash = dataframe_content_sha256(test_anchor)
    train_hash = dataframe_content_sha256(artifacts.X_train)
    test_hash = dataframe_content_sha256(artifacts.X_test)

    if parent_artifacts is not None:
        parent_train = project_ordered_features(parent_artifacts.X_train, names)
        parent_test = project_ordered_features(parent_artifacts.X_test, names)
        parent_train_hash = dataframe_content_sha256(parent_train)
        parent_test_hash = dataframe_content_sha256(parent_test)
        if parent_train_hash != train_anchor_hash:
            raise UnverifiableAlignmentError(
                "Train row identity does not match the parent package on the "
                "ordered v0 feature projection."
            )
        if parent_test_hash != test_anchor_hash:
            raise UnverifiableAlignmentError(
                "Test row identity does not match the parent package on the "
                "ordered v0 feature projection."
            )
        if len(parent_artifacts.X_train) != len(artifacts.X_train):
            raise UnverifiableAlignmentError(
                "Train row count differs from the parent package."
            )
        if len(parent_artifacts.X_test) != len(artifacts.X_test):
            raise UnverifiableAlignmentError(
                "Test row count differs from the parent package."
            )

    return AlignmentProof(
        status="proven",
        method=ROW_IDENTITY_METHOD,
        train_hash=train_hash,
        test_hash=test_hash,
        train_anchor_hash=train_anchor_hash,
        test_anchor_hash=test_anchor_hash,
        anchor_feature_names=names,
    )


def row_identity_from_proof(proof: AlignmentProof) -> RowIdentitySpec:
    return RowIdentitySpec(
        train_hash=proof.train_hash,
        test_hash=proof.test_hash,
        alignment_status="proven",
        alignment_method=proof.method,
        train_anchor_hash=proof.train_anchor_hash,
        test_anchor_hash=proof.test_anchor_hash,
    )
