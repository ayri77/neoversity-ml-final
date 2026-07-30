"""Safe legacy dataset manifest backfill."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from src.churn_ml.dataset_registry.alignment import (
    load_anchor_feature_names,
    prove_alignment,
)
from src.churn_ml.dataset_registry.constants import (
    LEGACY_CANONICAL_IDS,
    LEGACY_CATALOG,
    LEGACY_UNREGISTERED_IDS,
    V1_ENGINEERED_FEATURES,
    V2_SUMMARY_FEATURES,
    V3_ENGINEERED_FEATURES,
)
from src.churn_ml.dataset_registry.errors import (
    DatasetRegistryError,
    OverwriteRefusedError,
    PackageStatus,
    UnverifiableAlignmentError,
)
from src.churn_ml.dataset_registry.manifest import build_manifest, write_manifest
from src.churn_ml.dataset_registry.package import (
    has_manifest,
    has_package_files,
    load_package_artifacts,
    manifest_path,
    package_dir_for,
    validate_basic_shapes,
)
from src.churn_ml.dataset_registry.schema import (
    DatasetManifest,
    TargetDependency,
)


@dataclass(frozen=True)
class BackfillResult:
    status: PackageStatus
    dataset_id: str
    package_dir: Path
    message: str
    wrote: bool
    dry_run: bool
    manifest: DatasetManifest | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "dataset_id": self.dataset_id,
            "package_dir": str(self.package_dir),
            "message": self.message,
            "wrote": self.wrote,
            "dry_run": self.dry_run,
        }
        if self.manifest is not None:
            payload["manifest"] = self.manifest.to_dict()
        return payload


def engineered_features_for(dataset_id: str, feature_names: Sequence[str]) -> tuple[str, ...]:
    names = set(feature_names)
    if dataset_id == "v0_raw_minimal":
        return ()
    if dataset_id == "v1_missingness_summary":
        return tuple(name for name in V1_ENGINEERED_FEATURES if name in names)
    if dataset_id == "v2_missingness_indicators":
        engineered = [name for name in V2_SUMMARY_FEATURES if name in names]
        engineered.extend(
            name for name in feature_names if str(name).endswith("_is_missing")
        )
        return tuple(dict.fromkeys(engineered))
    if dataset_id == "v3_targeted_missingness":
        engineered = [name for name in V1_ENGINEERED_FEATURES if name in names]
        engineered.extend(name for name in V3_ENGINEERED_FEATURES if name in names)
        return tuple(dict.fromkeys(engineered))
    return ()


def backfill_legacy_dataset(
    root: Path,
    dataset_id: str,
    *,
    write: bool = False,
) -> BackfillResult:
    root = root.resolve()
    package_dir = package_dir_for(root, dataset_id)
    dry_run = not write

    if dataset_id in LEGACY_UNREGISTERED_IDS:
        return BackfillResult(
            status="unregistered",
            dataset_id=dataset_id,
            package_dir=package_dir,
            message="Obsolete or unimplemented dataset name; backfill refused.",
            wrote=False,
            dry_run=dry_run,
        )
    if dataset_id not in LEGACY_CATALOG:
        return BackfillResult(
            status="unregistered",
            dataset_id=dataset_id,
            package_dir=package_dir,
            message=(
                "Dataset is not in the canonical legacy catalog "
                f"{list(LEGACY_CANONICAL_IDS)}."
            ),
            wrote=False,
            dry_run=dry_run,
        )
    if not package_dir.is_dir() or not has_package_files(package_dir):
        return BackfillResult(
            status="invalid",
            dataset_id=dataset_id,
            package_dir=package_dir,
            message="Legacy package files are missing or incomplete.",
            wrote=False,
            dry_run=dry_run,
        )
    if has_manifest(package_dir):
        raise OverwriteRefusedError(
            f"Refusing to overwrite existing manifest: {manifest_path(package_dir)}"
        )

    catalog = LEGACY_CATALOG[dataset_id]
    parent_id = catalog["parent_dataset_id"]
    try:
        artifacts = load_package_artifacts(package_dir, dataset_id=dataset_id)
        validate_basic_shapes(artifacts)
        _validate_legacy_lineage_metadata(artifacts.metadata, dataset_id, parent_id)
        parent_artifacts = None
        if parent_id is not None:
            parent_dir = package_dir_for(root, parent_id)
            if not has_package_files(parent_dir):
                raise DatasetRegistryError(
                    f"Parent package {parent_id!r} is missing or incomplete."
                )
            parent_artifacts = load_package_artifacts(parent_dir, dataset_id=parent_id)
            validate_basic_shapes(parent_artifacts)
        anchor_names = load_anchor_feature_names(root)
        alignment = prove_alignment(
            artifacts,
            anchor_feature_names=anchor_names,
            parent_artifacts=parent_artifacts,
            root=root,
        )
        engineered = engineered_features_for(
            dataset_id,
            artifacts.X_train.columns.tolist(),
        )
        manifest = build_manifest(
            artifacts,
            hypothesis=str(catalog["hypothesis"]),
            parent_dataset_id=parent_id,
            target_dependency=catalog["target_dependency"],  # type: ignore[arg-type]
            transformations=list(catalog["transformations"]),
            alignment=alignment,
            engineered_features=engineered,
        )
    except UnverifiableAlignmentError as error:
        return BackfillResult(
            status="unverifiable_alignment",
            dataset_id=dataset_id,
            package_dir=package_dir,
            message=str(error),
            wrote=False,
            dry_run=dry_run,
        )
    except DatasetRegistryError as error:
        return BackfillResult(
            status="invalid",
            dataset_id=dataset_id,
            package_dir=package_dir,
            message=str(error),
            wrote=False,
            dry_run=dry_run,
        )

    if not write:
        return BackfillResult(
            status="valid",
            dataset_id=dataset_id,
            package_dir=package_dir,
            message="Dry-run backfill succeeded; manifest was not written.",
            wrote=False,
            dry_run=True,
            manifest=manifest,
        )

    write_manifest(manifest_path(package_dir), manifest)
    return BackfillResult(
        status="valid",
        dataset_id=dataset_id,
        package_dir=package_dir,
        message="Legacy manifest written.",
        wrote=True,
        dry_run=False,
        manifest=manifest,
    )


def backfill_legacy_registry(
    root: Path,
    *,
    dataset_ids: Sequence[str] | None = None,
    write: bool = False,
) -> list[BackfillResult]:
    ids = list(dataset_ids) if dataset_ids is not None else list(LEGACY_CANONICAL_IDS)
    results: list[BackfillResult] = []
    for dataset_id in ids:
        try:
            results.append(backfill_legacy_dataset(root, dataset_id, write=write))
        except OverwriteRefusedError as error:
            results.append(
                BackfillResult(
                    status="refused_overwrite",
                    dataset_id=dataset_id,
                    package_dir=package_dir_for(root, dataset_id),
                    message=str(error),
                    wrote=False,
                    dry_run=not write,
                )
            )
    return results


def _validate_legacy_lineage_metadata(
    metadata: dict[str, Any],
    dataset_id: str,
    parent_id: str | None,
) -> None:
    """
    Validate lineage hints present in legacy metadata.json.

    Legacy packages wrap fields under {"version": ..., "metadata": {...}}.
    Nested lineage uses source/base_version (not parent_dataset_id).
    """
    if metadata.get("version") not in {None, dataset_id}:
        raise DatasetRegistryError(
            f"metadata version {metadata.get('version')!r} does not match {dataset_id!r}."
        )
    nested = metadata.get("metadata")
    if not isinstance(nested, dict):
        # Synthetic fixtures may use a flat metadata object; lineage is then
        # taken from the registry catalog only.
        return
    if parent_id is None:
        return
    hints = {
        value
        for key in ("parent", "base_version", "source")
        if isinstance((value := nested.get(key)), str)
    }
    if hints and parent_id not in hints and "raw" not in hints:
        raise DatasetRegistryError(
            f"Legacy metadata lineage hints {sorted(hints)} do not include "
            f"expected parent {parent_id!r}."
        )


def assert_target_dependency(dataset_id: str, value: TargetDependency) -> None:
    expected = LEGACY_CATALOG[dataset_id]["target_dependency"]
    if value != expected:
        raise DatasetRegistryError(
            f"Unexpected target_dependency for {dataset_id}: {value!r}."
        )
