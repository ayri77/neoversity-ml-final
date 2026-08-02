"""Command-line interface for Blend Campaign Runner v1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from src.churn_ml.blend_campaign.artifact_v1 import BlendCampaignArtifactError
from src.churn_ml.blend_campaign.config_v1 import (
    BlendCampaignConfigurationError,
    load_campaign_config,
)
from src.churn_ml.blend_campaign.runner_v1 import (
    BlendCampaignRunner,
    BlendCampaignRunnerError,
)


EXIT_OK = 0
EXIT_INVALID = 2
EXIT_FAILED = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate, search, rank, and selectively materialize canonical probability "
            "blend campaigns. No training, prediction generation, network access, or Kaggle upload."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        (
            "validate",
            "Validate config, exact candidate aliases, hashes, and experiment sets (read-only).",
        ),
        ("plan", "Freeze the validated campaign configuration and deterministic plan."),
        (
            "run",
            "Plan and execute all searches, then configured materialization/submissions.",
        ),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, required=True)

    for name, help_text in (
        ("resume", "Resume an interrupted or partially failed frozen campaign."),
        ("inspect", "Inspect frozen plan, status, results, failures, and handoffs."),
        (
            "materialize-top",
            "Materialize configured explicit selections and top-K successful searches.",
        ),
        (
            "generate-submissions",
            "Generate validated local submissions from materialized candidates only.",
        ),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--campaign-dir", type=Path, required=True)
        if name == "materialize-top":
            command.add_argument("--top-k", type=int, default=None)
    return parser


def main(
    argv: Sequence[str] | None = None, *, repository_root: Path | None = None
) -> int:
    root = (repository_root or Path(__file__).resolve().parents[3]).resolve()
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    runner = BlendCampaignRunner(root)
    try:
        if args.command in {"validate", "plan", "run"}:
            config = load_campaign_config(args.config, repository_root=root)
            if args.command == "validate":
                plan = runner.validate(config)
                result: dict[str, Any] = {
                    "ok": True,
                    "command": "validate",
                    "artifacts_written": False,
                    "campaign_id": config.campaign_id,
                    "plan_hash": plan["plan_hash"],
                    "experiment_count": len(plan["experiments"]),
                }
            elif args.command == "plan":
                planned = runner.plan(config)
                result = {
                    "ok": True,
                    "command": "plan",
                    "artifacts_written": True,
                    "campaign_id": config.campaign_id,
                    "campaign_dir": str(planned["campaign_dir"]),
                    "plan_hash": planned["plan"]["plan_hash"],
                }
            else:
                result = runner.run(config)
                result = _summary("run", result)
        elif args.command == "resume":
            result = _summary(
                "resume", runner.resume(_resolve(root, args.campaign_dir))
            )
        elif args.command == "inspect":
            result = runner.inspect(_resolve(root, args.campaign_dir))
            result = {"ok": True, "command": "inspect", **result}
        elif args.command == "materialize-top":
            payload = runner.materialize_top(
                _resolve(root, args.campaign_dir), top_k=args.top_k
            )
            result = {"ok": True, "command": "materialize-top", **payload}
        elif args.command == "generate-submissions":
            payload = runner.generate_submissions(_resolve(root, args.campaign_dir))
            result = {"ok": True, "command": "generate-submissions", **payload}
        else:  # pragma: no cover - argparse owns this branch
            raise BlendCampaignRunnerError(f"Unsupported command: {args.command}")
    except (
        BlendCampaignConfigurationError,
        BlendCampaignArtifactError,
        BlendCampaignRunnerError,
        OSError,
        ValueError,
    ) as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "command": getattr(args, "command", None),
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                sort_keys=True,
            )
        )
        print(f"STATUS: failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    print(json.dumps(result, sort_keys=True, default=str))
    print(f"STATUS: {args.command} complete", file=sys.stderr)
    if result.get("campaign_state") == "completed_with_failures":
        return EXIT_FAILED
    return EXIT_OK


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _summary(command: str, result: dict[str, Any]) -> dict[str, Any]:
    status = result["status"]
    results = result["results"].get("results") or []
    return {
        "ok": status["state"] == "completed",
        "command": command,
        "campaign_dir": result["campaign_dir"],
        "campaign_id": result["plan"]["campaign_id"],
        "plan_hash": result["plan"]["plan_hash"],
        "campaign_state": status["state"],
        "succeeded_searches": sum(row.get("status") == "succeeded" for row in results),
        "failed_searches": sum(row.get("status") == "failed" for row in results),
        "network_access": False,
        "kaggle_upload": False,
    }


if __name__ == "__main__":
    raise SystemExit(main())
