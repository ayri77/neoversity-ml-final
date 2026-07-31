"""Read-only dataset-identity audit CLI (no apply/migration mode)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.churn_ml.control_panel.historical_identity_audit import (  # noqa: E402
    audit_historical_dataset_identity,
    render_audit_text,
    write_audit_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only audit of persisted dataset identity across UI jobs and "
            "configured artifact roots. Never mutates historical evidence."
        )
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="Repository root (default: repository containing this script).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for deterministic JSON report output.",
    )
    parser.add_argument(
        "--skip-jobs",
        action="store_true",
        help="Skip artifacts/ui_jobs scan.",
    )
    parser.add_argument(
        "--skip-artifacts",
        action="store_true",
        help="Skip configured reader artifact roots.",
    )
    args = parser.parse_args(argv)
    report = audit_historical_dataset_identity(
        args.repository_root.resolve(),
        include_jobs=not args.skip_jobs,
        include_artifacts=not args.skip_artifacts,
    )
    sys.stdout.buffer.write(render_audit_text(report).encode("utf-8"))
    if args.json_out is not None:
        write_audit_json(report, args.json_out)
        sys.stdout.buffer.write(
            f"Wrote JSON report: {args.json_out}\n".encode("utf-8")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
