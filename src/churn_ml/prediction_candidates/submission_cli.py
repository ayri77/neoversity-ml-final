"""CLI for canonical prediction-candidate local submission generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from src.churn_ml.prediction_candidates.contract_v1 import (
    CandidateConflictError,
    PredictionCandidateError,
)
from src.churn_ml.prediction_candidates.submission_v1 import (
    CandidateSubmissionError,
    evaluate_submission_readiness,
    generate_candidate_submission,
)


EXIT_OK = 0
EXIT_INVALID = 2
EXIT_CONFLICT = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate submission readiness and generate a local Kaggle submission "
            "from a canonical prediction_candidate_v1 package. No training, "
            "AutoGluon loading, network, or upload."
        )
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--candidate-root", default=None)
    common.add_argument("--blend-root", default=None)
    common.add_argument("--submission-root", default=None)

    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser(
        "validate",
        parents=[common],
        help="read-only submission-readiness evaluation",
    )
    validate.add_argument("--candidate-id", required=True)

    generate = sub.add_parser(
        "generate",
        parents=[common],
        help="generate a validated local submission artifact",
    )
    generate.add_argument("--candidate-id", required=True)
    generate.add_argument("--submission-id", required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    repository_root: Path | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = (repository_root or Path(__file__).resolve().parents[3]).resolve()
    try:
        if args.command == "validate":
            return _cmd_validate(args, root)
        if args.command == "generate":
            return _cmd_generate(args, root)
    except CandidateConflictError as error:
        _emit_error("conflict", str(error), reason_code=error.reason_code)
        return EXIT_CONFLICT
    except (
        CandidateSubmissionError,
        PredictionCandidateError,
        ValueError,
    ) as error:
        reason = getattr(error, "reason_code", None)
        _emit_error("invalid", str(error), reason_code=reason)
        return EXIT_INVALID
    except OSError as error:
        _emit_error("invalid", f"{type(error).__name__}: {error}")
        return EXIT_INVALID
    return EXIT_INVALID


def _cmd_validate(args: argparse.Namespace, root: Path) -> int:
    readiness = evaluate_submission_readiness(
        args.candidate_id,
        repository_root=root,
        candidates_root_relative=args.candidate_root,
        blend_root_relative=args.blend_root,
    )
    _emit(
        {
            "ok": True,
            "command": "validate",
            "artifacts_written": False,
            **readiness,
        }
    )
    print(
        f"STATUS: validate readiness={readiness['state']} for "
        f"{args.candidate_id}; read-only; no upload",
        file=sys.stderr,
    )
    return EXIT_OK


def _cmd_generate(args: argparse.Namespace, root: Path) -> int:
    result = generate_candidate_submission(
        args.candidate_id,
        submission_id=args.submission_id,
        repository_root=root,
        candidates_root_relative=args.candidate_root,
        submission_root_relative=args.submission_root,
        blend_root_relative=args.blend_root,
    )
    result["command"] = "generate"
    _emit(result)
    print(
        f"STATUS: submission {args.submission_id} from {args.candidate_id}; "
        f"local only; no upload",
        file=sys.stderr,
    )
    return EXIT_OK


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _emit_error(status: str, message: str, *, reason_code: str | None = None) -> None:
    payload: dict[str, Any] = {"ok": False, "status": status, "error": message}
    if reason_code is not None:
        payload["reason_code"] = reason_code
    print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
