"""CLI for source-neutral multi-candidate probability blending."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from src.churn_ml.blending.artifact_v1 import (
    BlendArtifactError,
    load_blend_artifact,
    materialize_blend,
    search_blend,
)
from src.churn_ml.blending.compatibility_v1 import (
    BlendCompatibilityError,
    list_candidates,
    load_compatible_candidates,
    resolve_candidates_root,
)
from src.churn_ml.blending.diversity_v1 import analyze_diversity
from src.churn_ml.blending.evaluation_v1 import BlendSettings
from src.churn_ml.blending.optimization_v1 import BlendOptimizationError
from src.churn_ml.prediction_candidates.contract_v1 import (
    CandidateConflictError,
    PredictionCandidateError,
    candidate_summary,
    load_candidate_package,
)


EXIT_OK = 0
EXIT_INVALID = 2
EXIT_CONFLICT = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Source-neutral multi-candidate probability blending over "
            "prediction_candidate_v1 packages. No training, AutoGluon, network, or UI."
        )
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--candidate-root",
        default=None,
        help="Repository-relative candidate root (default artifacts/prediction_candidates)",
    )
    common.add_argument(
        "--blend-root",
        default=None,
        help="Repository-relative blend artifact root (default artifacts/prediction_blends)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-candidates", parents=[common], help="list candidate packages")

    validate = sub.add_parser("validate", parents=[common], help="compatibility check")
    validate.add_argument("--candidate", action="append", default=[], dest="candidates")

    analyze = sub.add_parser("analyze", parents=[common], help="descriptive diversity")
    analyze.add_argument("--candidate", action="append", default=[], dest="candidates")

    search = sub.add_parser("search", parents=[common], help="evaluate without writing")
    _add_search_args(search)

    materialize = sub.add_parser(
        "materialize", parents=[common], help="write blend artifact and candidate"
    )
    _add_search_args(materialize)

    inspect = sub.add_parser("inspect", parents=[common], help="inspect blend artifact")
    inspect.add_argument("--blend-id", required=True)

    inspect_candidate = sub.add_parser(
        "inspect-candidate",
        parents=[common],
        help="inspect one prediction_candidate_v1 package",
    )
    inspect_candidate.add_argument("--candidate-id", required=True)
    return parser


def _add_search_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidate", action="append", default=[], dest="candidates")
    parser.add_argument(
        "--strategy",
        choices=("equal", "manual", "optimized"),
        default="optimized",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-active-models", type=int, default=None)
    parser.add_argument("--weight", action="append", default=[], dest="weights")
    parser.add_argument("--dirichlet-draws", type=int, default=32)
    parser.add_argument("--pairwise-grid-step", type=float, default=0.1)


def main(
    argv: Sequence[str] | None = None,
    *,
    repository_root: Path | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = (repository_root or Path(__file__).resolve().parents[3]).resolve()
    try:
        if args.command == "list-candidates":
            return _cmd_list(args, root)
        if args.command == "validate":
            return _cmd_validate(args, root)
        if args.command == "analyze":
            return _cmd_analyze(args, root)
        if args.command == "search":
            return _cmd_search(args, root)
        if args.command == "materialize":
            return _cmd_materialize(args, root)
        if args.command == "inspect":
            return _cmd_inspect(args, root)
        if args.command == "inspect-candidate":
            return _cmd_inspect_candidate(args, root)
    except CandidateConflictError as error:
        _emit_error("conflict", str(error), reason_code=error.reason_code)
        return EXIT_CONFLICT
    except (
        BlendCompatibilityError,
        BlendOptimizationError,
        BlendArtifactError,
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


def _settings_from_args(args: argparse.Namespace) -> BlendSettings:
    return BlendSettings(
        strategy=args.strategy,
        folds=args.folds,
        repeats=args.repeats,
        seed=args.seed,
        max_active_models=args.max_active_models,
        manual_weights=tuple(args.weights),
        pairwise_grid_step=args.pairwise_grid_step,
        dirichlet_draws=args.dirichlet_draws,
    )


def _cmd_list(args: argparse.Namespace, root: Path) -> int:
    summaries = list_candidates(
        repository_root=root, candidates_root=args.candidate_root
    )
    _emit(
        {
            "ok": True,
            "command": "list-candidates",
            "artifacts_written": False,
            "candidates": summaries,
            "count": len(summaries),
        }
    )
    print(f"STATUS: listed {len(summaries)} candidate(s); read-only", file=sys.stderr)
    return EXIT_OK


def _cmd_validate(args: argparse.Namespace, root: Path) -> int:
    pool = load_compatible_candidates(
        args.candidates,
        repository_root=root,
        candidates_root=args.candidate_root,
    )
    _emit(
        {
            "ok": True,
            "command": "validate",
            "artifacts_written": False,
            "compatibility": pool.compatibility,
        }
    )
    print(
        f"STATUS: validated {pool.n_candidates} candidates; compatible; read-only",
        file=sys.stderr,
    )
    return EXIT_OK


def _cmd_analyze(args: argparse.Namespace, root: Path) -> int:
    pool = load_compatible_candidates(
        args.candidates,
        repository_root=root,
        candidates_root=args.candidate_root,
    )
    diversity = analyze_diversity(pool)
    _emit(
        {
            "ok": True,
            "command": "analyze",
            "artifacts_written": False,
            "compatibility": pool.compatibility,
            "candidate_metrics": diversity["candidate_metrics"],
            "pairwise_analysis": diversity["pairwise_analysis"].to_dict(orient="records"),
            "notes": diversity["notes"],
        }
    )
    print(
        f"STATUS: analyzed {pool.n_candidates} candidates; read-only",
        file=sys.stderr,
    )
    return EXIT_OK


def _cmd_search(args: argparse.Namespace, root: Path) -> int:
    pool = load_compatible_candidates(
        args.candidates,
        repository_root=root,
        candidates_root=args.candidate_root,
    )
    payload = search_blend(pool, _settings_from_args(args))
    payload["command"] = "search"
    payload["artifacts_written"] = False
    _emit(payload)
    print(
        f"STATUS: search complete for blend_id={payload['blend_id']}; not materialized",
        file=sys.stderr,
    )
    return EXIT_OK


def _cmd_materialize(args: argparse.Namespace, root: Path) -> int:
    pool = load_compatible_candidates(
        args.candidates,
        repository_root=root,
        candidates_root=args.candidate_root,
    )
    result = materialize_blend(
        pool,
        _settings_from_args(args),
        blend_root_relative=args.blend_root,
    )
    _emit(
        {
            "ok": True,
            "command": "materialize",
            "artifacts_written": True,
            "blend_id": result.blend_id,
            "blend_path": result.blend_dir.as_posix(),
            "canonical_candidate_id": result.candidate_package.candidate_id,
            "canonical_candidate_path": candidate_summary(result.candidate_package)[
                "package_path"
            ],
            "deployment": result.evaluation.deployment,
            "cross_fitted_metrics": result.evaluation.cross_fitted_metrics,
            "exploratory": pool.exploratory,
        }
    )
    print(
        f"STATUS: materialized blend {result.blend_id} and candidate "
        f"{result.candidate_package.candidate_id}; _SUCCESS written last",
        file=sys.stderr,
    )
    return EXIT_OK


def _cmd_inspect(args: argparse.Namespace, root: Path) -> int:
    payload = load_blend_artifact(
        args.blend_id,
        repository_root=root,
        blend_root_relative=args.blend_root,
    )
    payload["command"] = "inspect"
    payload["artifacts_written"] = False
    _emit(payload)
    print(
        f"STATUS: inspected blend {args.blend_id}; "
        f"parents={payload.get('parents')}",
        file=sys.stderr,
    )
    return EXIT_OK


def _cmd_inspect_candidate(args: argparse.Namespace, root: Path) -> int:
    relative, absolute = resolve_candidates_root(root, args.candidate_root)
    package = load_candidate_package(
        absolute / args.candidate_id,
        repository_root=root,
        candidates_root_relative=relative,
    )
    summary = candidate_summary(package)
    _emit(
        {
            "ok": True,
            "command": "inspect-candidate",
            "artifacts_written": False,
            "summary": summary,
            "identity": {
                "dataset_id": package.manifest["dataset_id"],
                "target_hash": package.manifest["target_hash"],
                "train_anchor_hash": package.manifest["train_anchor_hash"],
                "test_anchor_hash": package.manifest["test_anchor_hash"],
                "positive_class_label": package.manifest["positive_class_label"],
                "probability_semantics": package.manifest["probability_semantics"],
            },
            "source": {
                "source_kind": package.manifest["source_kind"],
                "source_model_name": package.manifest["source_model_name"],
                "oof_protocol": package.manifest["oof_protocol"],
            },
        }
    )
    print(
        f"STATUS: inspected candidate {package.candidate_id}; read-only",
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
