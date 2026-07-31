"""CLI for Dataset Campaign / Matrix Runner v1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Never

from src.churn_ml.dataset_campaign.errors import (
    EXIT_CONFIG,
    EXIT_EXECUTION,
    EXIT_MANIFEST,
    EXIT_SUCCESS,
    EXIT_UNEXPECTED,
    EXIT_VALIDATION,
    CampaignConfigurationError,
    CampaignExecutionError,
    CampaignManifestError,
    CampaignValidationError,
    DatasetCampaignError,
)
from src.churn_ml.dataset_campaign.execute import (
    inspect_campaign,
    run_campaign,
    validate_only,
)
from src.churn_ml.dataset_campaign.schema import load_campaign_spec


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class CliUsageError(ValueError):
    """Deterministic CLI usage failures."""


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        raise CliUsageError(message)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = JsonArgumentParser(
        description=(
            "Dataset Campaign / Matrix Runner v1: validate, freeze, execute, "
            "and resume Registry × model screening matrices."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate",
        help="Validate the full matrix without allocating Experiment Core runs.",
    )
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument(
        "--freeze",
        action="store_true",
        help="Persist the immutable campaign manifest after successful validation.",
    )
    validate.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON summary on success.",
    )

    inspect_cmd = subparsers.add_parser(
        "inspect",
        help="Inspect a frozen campaign manifest and optional status/summary.",
    )
    inspect_cmd.add_argument("--campaign-dir", type=Path, required=True)
    inspect_cmd.add_argument("--json", action="store_true")

    run = subparsers.add_parser(
        "run",
        help="Validate, freeze if needed, and execute the campaign sequentially.",
    )
    run.add_argument("--config", type=Path, required=True)
    run.add_argument(
        "--campaign-dir",
        type=Path,
        help="Optional existing campaign directory (must match frozen identity).",
    )

    resume = subparsers.add_parser(
        "resume",
        help="Resume from an exact frozen campaign manifest.",
    )
    resume.add_argument("--campaign-dir", type=Path, required=True)

    status = subparsers.add_parser(
        "status",
        help="Print campaign status and descriptive summary.",
    )
    status.add_argument("--campaign-dir", type=Path, required=True)
    status.add_argument("--json", action="store_true")

    return parser.parse_args(argv)


def execute(args: argparse.Namespace, *, project_root: Path = PROJECT_ROOT) -> int:
    try:
        if args.command == "validate":
            return _cmd_validate(args, project_root=project_root)
        if args.command == "inspect":
            return _cmd_inspect(args, project_root=project_root)
        if args.command == "run":
            return _cmd_run(args, project_root=project_root)
        if args.command == "resume":
            return _cmd_resume(args, project_root=project_root)
        if args.command == "status":
            return _cmd_status(args, project_root=project_root)
        raise CliUsageError(f"Unknown command: {args.command}")
    except CliUsageError as error:
        _emit_error("usage", str(error))
        return EXIT_CONFIG
    except CampaignConfigurationError as error:
        _emit_error("config", str(error))
        return EXIT_CONFIG
    except CampaignValidationError as error:
        _emit_error("validation", str(error), failures=error.failures)
        return EXIT_VALIDATION
    except CampaignManifestError as error:
        _emit_error("manifest", str(error))
        return EXIT_MANIFEST
    except CampaignExecutionError as error:
        _emit_error("execution", str(error))
        return EXIT_EXECUTION
    except DatasetCampaignError as error:
        _emit_error("campaign", str(error))
        return EXIT_UNEXPECTED
    except Exception as error:  # noqa: BLE001
        _emit_error("unexpected", f"{type(error).__name__}: {error}")
        return EXIT_UNEXPECTED


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except CliUsageError as error:
        _emit_error("usage", str(error))
        return EXIT_CONFIG
    return execute(args)


def _cmd_validate(args: argparse.Namespace, *, project_root: Path) -> int:
    spec = load_campaign_spec(args.config, project_root=project_root)
    resolved = validate_only(
        spec,
        project_root=project_root,
        freeze=bool(args.freeze),
    )
    payload = {
        "status": "validated",
        "campaign_id": spec.campaign_id,
        "classification": spec.classification,
        "campaign_type": spec.campaign_type,
        "cell_count": resolved.manifest["cell_count"],
        "manifest_hash": resolved.manifest_hash,
        "frozen": bool(args.freeze),
        "cells": [
            {
                "execution_order": cell["execution_order"],
                "cell_id": cell["cell_id"],
                "dataset_id": cell["dataset_id"],
                "model_family": cell["model_family"],
                "adapter_id": cell["adapter_id"],
            }
            for cell in resolved.manifest["cells"]
        ],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("Dataset campaign validation successful.")
        print(f"Campaign: {spec.campaign_id}")
        print(f"Type: {spec.campaign_type}")
        print(f"Classification: {spec.classification}")
        print(f"Cells: {resolved.manifest['cell_count']}")
        print(f"Manifest SHA256: {resolved.manifest_hash}")
        if args.freeze:
            print(
                "Frozen manifest written under "
                f"{spec.artifacts_root}/{spec.campaign_id}/"
            )
        else:
            print("No campaign freeze requested; no Experiment Core runs allocated.")
        for cell in resolved.manifest["cells"]:
            print(
                f"  [{cell['execution_order']:02d}] "
                f"{cell['dataset_id']} × {cell['model_family']} "
                f"({cell['cell_id']})"
            )
    return EXIT_SUCCESS


def _cmd_inspect(args: argparse.Namespace, *, project_root: Path) -> int:
    del project_root
    payload = inspect_campaign(args.campaign_dir)
    if args.json:
        print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    else:
        manifest = payload["manifest"]
        print(f"Campaign: {manifest['campaign_id']}")
        print(f"Manifest SHA256: {manifest['manifest_hash']}")
        print(f"Classification: {manifest['classification']}")
        print(f"Type: {manifest['campaign_type']}")
        print(f"Cells: {manifest['cell_count']}")
        status = payload.get("status")
        if status:
            print(f"Status: {status.get('state')}")
        for cell in manifest["cells"]:
            print(
                f"  [{cell['execution_order']:02d}] "
                f"{cell['dataset_id']} × {cell['model_family']}"
            )
    return EXIT_SUCCESS


def _cmd_run(args: argparse.Namespace, *, project_root: Path) -> int:
    spec = load_campaign_spec(args.config, project_root=project_root)
    result = run_campaign(
        project_root=project_root,
        spec=spec,
        campaign_dir=args.campaign_dir,
        resume=False,
    )
    _print_run_result(result)
    return EXIT_SUCCESS if result["status"]["state"] == "completed" else EXIT_EXECUTION


def _cmd_resume(args: argparse.Namespace, *, project_root: Path) -> int:
    result = run_campaign(
        project_root=project_root,
        campaign_dir=args.campaign_dir,
        resume=True,
    )
    _print_run_result(result)
    return EXIT_SUCCESS if result["status"]["state"] == "completed" else EXIT_EXECUTION


def _cmd_status(args: argparse.Namespace, *, project_root: Path) -> int:
    del project_root
    payload = inspect_campaign(args.campaign_dir)
    summary = payload["summary"]
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Campaign: {summary.get('campaign_id')}")
        print(f"Manifest SHA256: {summary.get('manifest_hash')}")
        print(f"State: {summary.get('campaign_state')}")
        counts = summary.get("counts") or {}
        print(
            "Counts: "
            + ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        )
        for row in summary.get("rows") or []:
            ba = row.get("balanced_accuracy")
            ba_text = "n/a" if ba is None else f"{ba:.6f}"
            print(
                f"  [{row['execution_order']:02d}] {row['state']:9} "
                f"{row['dataset_id']} × {row['model_family']} BA={ba_text}"
            )
        print(summary.get("note"))
    return EXIT_SUCCESS


def _print_run_result(result: dict[str, Any]) -> None:
    status = result["status"]
    summary = result["summary"]
    print(f"Campaign directory: {result['campaign_dir']}")
    print(f"Manifest SHA256: {result['manifest_hash']}")
    print(f"Campaign state: {status['state']}")
    counts = summary.get("counts") or {}
    print(
        "Counts: "
        + ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    )


def _emit_error(code: str, message: str, *, failures: list[str] | None = None) -> None:
    payload: dict[str, Any] = {"status": "error", "code": code, "message": message}
    if failures:
        payload["failures"] = failures
    print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
