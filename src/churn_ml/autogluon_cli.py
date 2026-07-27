"""Command-line interface for standalone AutoGluon benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.churn_ml.autogluon_config import ConfigError, load_config, validation_summary
from src.churn_ml.autogluon_inspection import inspect_run
from src.churn_ml.autogluon_supervisor import (
    RunAlreadyExistsError,
    train_supervised,
)


EXIT_OK = 0
EXIT_INVALID = 2
EXIT_RUN_EXISTS = 3
EXIT_WORKER_FAILED = 4
EXIT_INSPECTION_FAILED = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="validate without allocating a run"
    )
    validate.add_argument("--config", required=True, type=Path)

    train = subparsers.add_parser("train", help="launch a supervised worker process")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--run-id")
    train.add_argument(
        "--quiet-progress",
        action="store_true",
        help="keep stage progress only in durable logs",
    )

    inspect = subparsers.add_parser("inspect", help="read-only run inspection")
    inspect.add_argument("--run-dir", required=True, type=Path)
    loading = inspect.add_mutually_exclusive_group()
    loading.add_argument("--attempt-load", action="store_true")
    loading.add_argument("--no-load", action="store_true")
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
        if args.command == "validate":
            config = load_config(args.config, root, require_data_files=True)
            print(json.dumps(validation_summary(config), indent=2, sort_keys=True))
            return EXIT_OK
        if args.command == "train":
            result = train_supervised(
                args.config,
                root,
                run_id=args.run_id,
                tee_progress=not args.quiet_progress,
            )
            return EXIT_OK if result.succeeded else EXIT_WORKER_FAILED
        if args.command == "inspect":
            attempt_load = (
                True if args.attempt_load else False if args.no_load else None
            )
            report = inspect_run(args.run_dir, attempt_load=attempt_load)
            print(json.dumps(report, indent=2, sort_keys=True))
            if (
                report["predictor_loading_attempted"]
                and not report["predictor_loading_succeeded"]
            ):
                return EXIT_INSPECTION_FAILED
            return EXIT_OK
    except RunAlreadyExistsError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return EXIT_RUN_EXISTS
    except (ConfigError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return EXIT_INVALID
    return EXIT_INVALID
