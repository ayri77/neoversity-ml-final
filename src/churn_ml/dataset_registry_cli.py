"""CLI for Dataset Package & Registry v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from src.churn_ml.dataset_registry.backfill import backfill_legacy_registry
from src.churn_ml.dataset_registry.constants import LEGACY_CANONICAL_IDS
from src.churn_ml.dataset_registry.discovery import scan_registry
from src.churn_ml.dataset_registry.errors import (
    EXIT_ERROR,
    EXIT_SUCCESS,
    STATUS_EXIT_CODES,
    DatasetRegistryError,
    OverwriteRefusedError,
    PackageStatus,
)
from src.churn_ml.dataset_registry.package import package_dir_for
from src.churn_ml.dataset_registry.validation import validate_package_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dataset Package & Registry v1 (scan/validate/inspect/backfill)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser(
        "scan",
        help="Report dataset directories as registered, unregistered, or invalid.",
    )
    scan_parser.add_argument("--root", type=Path, required=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Strict read-only validation of one package or the whole root.",
    )
    validate_parser.add_argument("--root", type=Path, required=True)
    validate_parser.add_argument("--dataset-id")

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Show manifest, schema, lineage, hashes, and validation result.",
    )
    inspect_parser.add_argument("--root", type=Path, required=True)
    inspect_parser.add_argument("--dataset-id", required=True)

    backfill_parser = subparsers.add_parser(
        "backfill",
        help=(
            "Generate legacy manifests after validating files, lineage, schema, "
            "and row identity. Dry-run by default; pass --write to persist."
        ),
    )
    backfill_parser.add_argument("--root", type=Path, required=True)
    backfill_parser.add_argument(
        "--dataset-id",
        action="append",
        dest="dataset_ids",
        help=(
            "Legacy dataset id to backfill. May be repeated. "
            f"Defaults to {list(LEGACY_CANONICAL_IDS)}."
        ),
    )
    backfill_parser.add_argument(
        "--write",
        action="store_true",
        help="Persist dataset_manifest.json. Never overwrites an existing manifest.",
    )
    return parser.parse_args(argv)


def execute(args: argparse.Namespace) -> int:
    try:
        root = _resolve_root(args.root)
        if args.command == "scan":
            return _scan(root)
        if args.command == "validate":
            return _validate(root, dataset_id=args.dataset_id)
        if args.command == "inspect":
            return _inspect(root, dataset_id=args.dataset_id)
        if args.command == "backfill":
            return _backfill(root, dataset_ids=args.dataset_ids, write=args.write)
        raise DatasetRegistryError(f"Unknown command: {args.command}")
    except OverwriteRefusedError as error:
        _emit({"ok": False, "status": error.status, "error": str(error)})
        return STATUS_EXIT_CODES[error.status]
    except DatasetRegistryError as error:
        _emit({"ok": False, "status": error.status, "error": str(error)})
        return STATUS_EXIT_CODES[error.status]
    except FileNotFoundError as error:
        _emit({"ok": False, "status": "invalid", "error": str(error)})
        return STATUS_EXIT_CODES["invalid"]
    except Exception as error:  # noqa: BLE001 - CLI boundary
        _emit(
            {
                "ok": False,
                "status": "invalid",
                "error": f"{type(error).__name__}: {error}",
            }
        )
        return EXIT_ERROR


def main(argv: Sequence[str] | None = None) -> int:
    return execute(parse_args(argv))


def _scan(root: Path) -> int:
    entries = scan_registry(root)
    payload = {
        "ok": True,
        "command": "scan",
        "root": str(root),
        "wrote": False,
        "packages": [entry.to_dict() for entry in entries],
        "counts": _count_statuses([entry.status for entry in entries]),
    }
    _emit(payload)
    return _aggregate_exit([entry.status for entry in entries])


def _validate(root: Path, *, dataset_id: str | None) -> int:
    if dataset_id is None:
        entries = scan_registry(root)
        results = []
        statuses: list[PackageStatus] = []
        for entry in entries:
            result = validate_package_dir(entry.package_dir, root=root)
            results.append(result.to_dict())
            statuses.append(result.status)
        _emit(
            {
                "ok": all(status == "valid" for status in statuses)
                if statuses
                else True,
                "command": "validate",
                "root": str(root),
                "wrote": False,
                "results": results,
                "counts": _count_statuses(statuses),
            }
        )
        return _aggregate_exit(statuses) if statuses else EXIT_SUCCESS

    result = validate_package_dir(package_dir_for(root, dataset_id), root=root)
    _emit(
        {
            "ok": result.status == "valid",
            "command": "validate",
            "root": str(root),
            "wrote": False,
            "result": result.to_dict(),
        }
    )
    return STATUS_EXIT_CODES[result.status]


def _inspect(root: Path, *, dataset_id: str) -> int:
    package_dir = package_dir_for(root, dataset_id)
    result = validate_package_dir(package_dir, root=root)
    payload: dict[str, Any] = {
        "ok": result.status == "valid",
        "command": "inspect",
        "root": str(root),
        "wrote": False,
        "dataset_id": dataset_id,
        "validation": result.to_dict(),
    }
    if result.manifest is not None:
        payload["manifest"] = result.manifest.to_dict()
        payload["lineage"] = {
            "dataset_id": result.manifest.dataset_id,
            "parent_dataset_id": result.manifest.parent_dataset_id,
            "transformations": [
                dict(item) for item in result.manifest.transformations
            ],
            "target_dependency": result.manifest.target_dependency,
        }
        payload["schema"] = {
            "n_features": result.manifest.n_features,
            "schema_hash": result.manifest.schema_hash,
            "features": [feature.to_dict() for feature in result.manifest.features],
        }
        payload["hashes"] = {
            "content_hashes": dict(result.manifest.content_hashes),
            "target_hash": result.manifest.target.hash,
            "row_identity": result.manifest.row_identity.to_dict(),
        }
    _emit(payload)
    return STATUS_EXIT_CODES[result.status]


def _backfill(
    root: Path,
    *,
    dataset_ids: Sequence[str] | None,
    write: bool,
) -> int:
    results = backfill_legacy_registry(root, dataset_ids=dataset_ids, write=write)
    statuses = [result.status for result in results]
    _emit(
        {
            "ok": all(status == "valid" for status in statuses) if statuses else True,
            "command": "backfill",
            "root": str(root),
            "wrote": any(result.wrote for result in results),
            "dry_run": not write,
            "results": [result.to_dict() for result in results],
            "counts": _count_statuses(statuses),
        }
    )
    return _aggregate_exit(statuses) if statuses else EXIT_SUCCESS


def _resolve_root(root: Path) -> Path:
    resolved = root.expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    else:
        resolved = resolved.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Registry root does not exist: {resolved}")
    return resolved


def _count_statuses(statuses: Sequence[PackageStatus]) -> dict[str, int]:
    counts = {
        "valid": 0,
        "invalid": 0,
        "unregistered": 0,
        "unverifiable_alignment": 0,
        "refused_overwrite": 0,
    }
    for status in statuses:
        counts[status] = counts.get(status, 0) + 1
    return counts


def _aggregate_exit(statuses: Sequence[PackageStatus]) -> int:
    priority: tuple[PackageStatus, ...] = (
        "refused_overwrite",
        "unverifiable_alignment",
        "invalid",
        "unregistered",
        "valid",
    )
    for status in priority:
        if status in statuses and status != "valid":
            return STATUS_EXIT_CODES[status]
    return EXIT_SUCCESS


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
