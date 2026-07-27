"""CLI for the optional local MLflow metadata index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.churn_ml.mlflow_config import (
    MLflowConfigError,
    load_mlflow_config,
    validation_summary,
)
from src.churn_ml.mlflow_sources import SourceError, default_source_registry
from src.churn_ml.mlflow_sync import sync_sources


EXIT_OK = 0
EXIT_SYNC_ERRORS = 1
EXIT_INVALID = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="validate configuration without allocating local storage"
    )
    validate.add_argument("--config", required=True, type=Path)

    sync = subparsers.add_parser(
        "sync", help="validate source artifacts and optionally index them"
    )
    sync.add_argument("--config", required=True, type=Path)
    sync.add_argument(
        "--source-type",
        required=True,
        choices=("research_v2", "autogluon", "all"),
    )
    sync.add_argument(
        "--run-dir",
        type=Path,
        help="sync one run directory (requires one explicit source type)",
    )
    sync.add_argument(
        "--dry-run",
        action="store_true",
        help="fully validate sources without importing MLflow or allocating storage",
    )
    sync.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop after the first rejected or errored source",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    repository_root: Path | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = (repository_root or Path(__file__).resolve().parents[2]).resolve(strict=True)
    try:
        config = load_mlflow_config(args.config, repository_root=root)
        if args.command == "validate":
            _print_json(validation_summary(config))
            return EXIT_OK
        if args.command == "sync":
            if args.run_dir is not None and args.source_type == "all":
                raise MLflowConfigError(
                    "--run-dir requires --source-type research_v2 or autogluon"
                )
            registry = default_source_registry()
            source_types = (
                ("research_v2", "autogluon")
                if args.source_type == "all"
                else (args.source_type,)
            )
            summary = sync_sources(
                config,
                registry=registry,
                source_types=source_types,
                run_dir=args.run_dir,
                dry_run=args.dry_run,
                fail_fast=args.fail_fast,
            )
            _print_json(summary.to_dict())
            return EXIT_SYNC_ERRORS if summary.has_failures else EXIT_OK
    except (MLflowConfigError, SourceError, OSError, ValueError) as error:
        _print_json(
            {
                "schema_version": 1,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
        )
        return EXIT_INVALID
    return EXIT_INVALID


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
