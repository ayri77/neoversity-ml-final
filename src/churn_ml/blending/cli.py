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
from src.churn_ml.blending.evaluation_v1 import (
    DEFAULT_OPTUNA_TRIALS,
    BlendSettings,
)
from src.churn_ml.blending.optimization_v1 import BlendOptimizationError
from src.churn_ml.blending.request_v1 import (
    BlendUIRequestError,
    load_blend_ui_request,
)
from src.churn_ml.prediction_candidates.contract_v1 import (
    CandidateConflictError,
    PredictionCandidateError,
    candidate_summary,
    load_candidate_package,
)
from src.churn_ml.prediction_candidates.submission_v1 import (
    CandidateSubmissionError,
    evaluate_submission_readiness,
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

    readiness = sub.add_parser(
        "submission-readiness",
        parents=[common],
        help="evaluate submission readiness for one candidate (read-only)",
    )
    readiness.add_argument("--candidate-id", required=True)
    return parser


def _add_search_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--request",
        default=None,
        help=(
            "Repository-relative blend_ui_request_v1 JSON. Incompatible with "
            "direct candidate/settings arguments."
        ),
    )
    parser.add_argument("--candidate", action="append", default=[], dest="candidates")
    parser.add_argument(
        "--strategy",
        choices=("equal", "manual", "optimized"),
        default=None,
    )
    parser.add_argument(
        "--optimizer",
        choices=("native", "optuna"),
        default=None,
        help="Weight optimizer backend for strategy=optimized (default: native)",
    )
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-active-models", type=int, default=None)
    parser.add_argument("--weight", action="append", default=[], dest="weights")
    parser.add_argument("--dirichlet-draws", type=int, default=None)
    parser.add_argument("--pairwise-grid-step", type=float, default=None)
    parser.add_argument("--optuna-trials", type=int, default=None)
    parser.add_argument("--optuna-timeout-seconds", type=float, default=None)
    parser.add_argument("--optuna-seed", type=int, default=None)


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
        if args.command == "submission-readiness":
            return _cmd_submission_readiness(args, root)
    except CandidateConflictError as error:
        _emit_error("conflict", str(error), reason_code=error.reason_code)
        return EXIT_CONFLICT
    except (
        BlendCompatibilityError,
        BlendOptimizationError,
        BlendArtifactError,
        BlendUIRequestError,
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


def _direct_settings_provided(args: argparse.Namespace) -> bool:
    return any(
        [
            bool(args.candidates),
            args.strategy is not None,
            args.optimizer is not None,
            args.folds is not None,
            args.repeats is not None,
            args.seed is not None,
            args.max_active_models is not None,
            bool(args.weights),
            args.dirichlet_draws is not None,
            args.pairwise_grid_step is not None,
            args.optuna_trials is not None,
            args.optuna_timeout_seconds is not None,
            args.optuna_seed is not None,
        ]
    )


def _resolve_search_inputs(
    args: argparse.Namespace, root: Path
) -> tuple[list[str], BlendSettings, str | None, str | None, str | None]:
    """Return candidates, settings, candidate_root, blend_root, request_id."""
    if getattr(args, "request", None):
        if _direct_settings_provided(args):
            raise BlendUIRequestError(
                "--request cannot be mixed with direct candidate/settings arguments.",
                reason_code="request_direct_conflict",
            )
        if args.candidate_root is not None or args.blend_root is not None:
            raise BlendUIRequestError(
                "--request cannot be mixed with --candidate-root/--blend-root.",
                reason_code="request_direct_conflict",
            )
        request = load_blend_ui_request(args.request, repository_root=root)
        return (
            request.candidate_ids,
            request.blend_settings(),
            str(request.payload["candidate_root"]),
            str(request.payload["blend_root"]),
            request.request_id,
        )
    if not args.candidates:
        raise BlendOptimizationError(
            "Provide --candidate entries or --request.",
            reason_code="candidates_missing",
        )
    optuna_options_provided = (
        args.optuna_trials is not None
        or args.optuna_timeout_seconds is not None
        or args.optuna_seed is not None
    )
    native_search_options_provided = (
        args.dirichlet_draws is not None or args.pairwise_grid_step is not None
    )
    settings = BlendSettings(
        strategy=args.strategy or "optimized",
        folds=5 if args.folds is None else int(args.folds),
        repeats=2 if args.repeats is None else int(args.repeats),
        seed=42 if args.seed is None else int(args.seed),
        max_active_models=args.max_active_models,
        manual_weights=tuple(args.weights),
        pairwise_grid_step=(
            0.1 if args.pairwise_grid_step is None else float(args.pairwise_grid_step)
        ),
        dirichlet_draws=(
            32 if args.dirichlet_draws is None else int(args.dirichlet_draws)
        ),
        optimizer_backend=args.optimizer or "native",
        optuna_trials=(
            DEFAULT_OPTUNA_TRIALS
            if args.optuna_trials is None
            else int(args.optuna_trials)
        ),
        optuna_timeout_seconds=args.optuna_timeout_seconds,
        optuna_seed=args.optuna_seed,
        optuna_options_provided=optuna_options_provided,
        native_search_options_provided=native_search_options_provided,
    )
    return (
        list(args.candidates),
        settings,
        args.candidate_root,
        args.blend_root,
        None,
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
    candidates, settings, candidate_root, blend_root, request_id = _resolve_search_inputs(
        args, root
    )
    del blend_root
    pool = load_compatible_candidates(
        candidates,
        repository_root=root,
        candidates_root=candidate_root,
    )
    payload = search_blend(pool, settings)
    payload["command"] = "search"
    payload["artifacts_written"] = False
    if request_id is not None:
        payload["request_id"] = request_id
    _emit(payload)
    print(
        f"STATUS: search complete for blend_id={payload['blend_id']}; not materialized",
        file=sys.stderr,
    )
    return EXIT_OK


def _cmd_materialize(args: argparse.Namespace, root: Path) -> int:
    candidates, settings, candidate_root, blend_root, request_id = _resolve_search_inputs(
        args, root
    )
    pool = load_compatible_candidates(
        candidates,
        repository_root=root,
        candidates_root=candidate_root,
    )
    result = materialize_blend(
        pool,
        settings,
        blend_root_relative=blend_root,
    )
    _emit(
        {
            "ok": True,
            "command": "materialize",
            "artifacts_written": True,
            "request_id": request_id,
            "blend_id": result.blend_id,
            "blend_path": result.blend_dir.as_posix(),
            "canonical_candidate_id": result.candidate_package.candidate_id,
            "canonical_candidate_path": candidate_summary(result.candidate_package)[
                "package_path"
            ],
            "optimizer_backend": result.evaluation.settings.optimizer_backend,
            "optimizer_settings": result.evaluation.search_budget,
            "deployment": result.evaluation.deployment,
            "final_deployment_weights": result.evaluation.deployment["weights"],
            "final_deployment_threshold": result.evaluation.deployment["threshold"],
            "honest_meta_cv_metrics": result.evaluation.honest_meta_cv_metrics,
            "cross_fitted_probability_descriptive_metrics": (
                result.evaluation.cross_fitted_probability_descriptive_metrics
            ),
            "submission_readiness": result.manifest.get("submission_readiness"),
            "cross_fitted_metrics": (
                result.evaluation.cross_fitted_probability_descriptive_metrics
            ),
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
        candidates_root_relative=args.candidate_root,
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
    readiness = evaluate_submission_readiness(
        package.candidate_id,
        repository_root=root,
        candidates_root_relative=relative,
        blend_root_relative=args.blend_root,
    )
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
            "final_deployment_threshold": package.source_metadata.get(
                "final_deployment_threshold",
                package.source_metadata.get("final_threshold"),
            ),
            "final_deployment_weights": package.source_metadata.get(
                "final_deployment_weights",
                package.source_metadata.get("final_weights"),
            ),
            "honest_meta_cv_metrics": package.source_metadata.get(
                "honest_meta_cv_metrics"
            ),
            "submission_readiness": readiness,
        }
    )
    print(
        f"STATUS: inspected candidate {package.candidate_id}; read-only",
        file=sys.stderr,
    )
    return EXIT_OK


def _cmd_submission_readiness(args: argparse.Namespace, root: Path) -> int:
    readiness = evaluate_submission_readiness(
        args.candidate_id,
        repository_root=root,
        candidates_root_relative=args.candidate_root,
        blend_root_relative=args.blend_root,
    )
    _emit(
        {
            "ok": True,
            "command": "submission-readiness",
            "artifacts_written": False,
            **readiness,
        }
    )
    state = readiness["state"]
    print(
        f"STATUS: submission-readiness for {args.candidate_id}: {state}; read-only",
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
