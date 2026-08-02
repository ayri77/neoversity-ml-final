"""CLI for AutoGluon → prediction_candidate_v1 import (no UI)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from src.churn_ml.prediction_candidates.autogluon_v1 import (
    import_autogluon_model_candidate,
    inventory_autogluon_run,
    load_predictor,
    resolve_run_dir,
    resolve_selected_models,
    validate_autogluon_model_import,
)
from src.churn_ml.prediction_candidates.contract_v1 import (
    CANDIDATE_ROOT_RELATIVE,
    CandidateConflictError,
    CandidateValidationError,
    PredictionCandidateError,
    candidate_summary,
    load_candidate_package,
    resolve_under_repository,
    validate_candidate_package,
)
from src.churn_ml.prediction_candidates.explicit_export_v1 import (
    ExplicitExportRequest,
    import_explicit_export_candidate,
    validate_explicit_export,
)
from src.churn_ml.prediction_candidates.preparation_request_v1 import (
    PreparationRequestError,
    load_preparation_request,
    resolve_request_selection,
)


EXIT_OK = 0
EXIT_INVALID = 2
EXIT_CONFLICT = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import selected AutoGluon models or aligned explicit exports into "
            "immutable prediction_candidate_v1 packages. No training, network, or UI."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list", help="list models in a completed AutoGluon run (read-only)"
    )
    list_parser.add_argument("--run-dir", required=True, type=Path)

    validate_parser = subparsers.add_parser(
        "validate",
        help="validate that selected models can be imported (read-only)",
    )
    _add_selection_args(validate_parser)

    import_parser = subparsers.add_parser(
        "import",
        help="import selected models as canonical prediction candidates",
    )
    _add_selection_args(import_parser)

    validate_explicit_parser = subparsers.add_parser(
        "validate-explicit",
        help="validate aligned explicit Parquet exports without loading a predictor",
    )
    _add_explicit_export_args(validate_explicit_parser)

    import_explicit_parser = subparsers.add_parser(
        "import-explicit",
        help="import aligned explicit Parquet exports without loading a predictor",
    )
    _add_explicit_export_args(import_explicit_parser)

    inspect_parser = subparsers.add_parser(
        "inspect", help="inspect an existing canonical prediction candidate"
    )
    inspect_parser.add_argument("--candidate-dir", type=Path)
    inspect_parser.add_argument("--candidate-id", type=str)
    return parser


def _add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--request",
        default=None,
        help=(
            "Repository-relative autogluon_candidate_preparation_request_v1 JSON. "
            "Incompatible with direct --run-dir/--model/--best arguments."
        ),
    )
    parser.add_argument("--run-dir", default=None, type=Path)
    parser.add_argument("--best", action="store_true")
    parser.add_argument("--model", action="append", default=[], dest="models")


def _add_explicit_export_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--oof", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--source-model-name", required=True)
    parser.add_argument("--source-kind", required=True)
    parser.add_argument("--source-run-path", required=True)
    parser.add_argument("--threshold", required=True, type=float)
    declaration = parser.add_mutually_exclusive_group(required=True)
    declaration.add_argument("--exploratory", dest="exploratory", action="store_true")
    declaration.add_argument(
        "--non-exploratory", dest="exploratory", action="store_false"
    )
    parser.add_argument("--validation-balanced-accuracy", type=float)
    parser.add_argument("--historical-kaggle-public-score", type=float)
    parser.add_argument("--source-model-type", default="unavailable")


def main(
    argv: Sequence[str] | None = None,
    *,
    repository_root: Path | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = (repository_root or Path(__file__).resolve().parents[3]).resolve()

    try:
        if args.command == "list":
            return _cmd_list(args, root)
        if args.command == "validate":
            return _cmd_validate(args, root)
        if args.command == "import":
            return _cmd_import(args, root)
        if args.command == "validate-explicit":
            return _cmd_validate_explicit(args, root)
        if args.command == "import-explicit":
            return _cmd_import_explicit(args, root)
        if args.command == "inspect":
            return _cmd_inspect(args, root)
    except CandidateConflictError as error:
        _emit_error("conflict", str(error), reason_code=error.reason_code)
        return EXIT_CONFLICT
    except (
        CandidateValidationError,
        PredictionCandidateError,
        PreparationRequestError,
    ) as error:
        reason = getattr(error, "reason_code", None)
        _emit_error("invalid", str(error), reason_code=reason)
        return EXIT_INVALID
    except (OSError, ValueError, json.JSONDecodeError) as error:
        _emit_error("invalid", f"{type(error).__name__}: {error}")
        return EXIT_INVALID
    return EXIT_INVALID


def _direct_selection_provided(args: argparse.Namespace) -> bool:
    return any(
        [
            args.run_dir is not None,
            bool(getattr(args, "best", False)),
            bool(getattr(args, "models", None)),
        ]
    )


def _resolve_selection(
    args: argparse.Namespace, root: Path
) -> tuple[Path, list[str], str | None]:
    """Return run_dir, selected_models, request_id."""
    if getattr(args, "request", None):
        if _direct_selection_provided(args):
            raise PreparationRequestError(
                "--request cannot be mixed with direct --run-dir/--model/--best.",
                reason_code="request_direct_conflict",
            )
        request = load_preparation_request(args.request, repository_root=root)
        run_dir, models = resolve_request_selection(request, repository_root=root)
        return run_dir, models, request.request_id
    if args.run_dir is None:
        raise PredictionCandidateError(
            "Provide --run-dir or --request.",
            reason_code="run_dir_missing",
        )
    run_dir = resolve_run_dir(args.run_dir, root)
    inventory = inventory_autogluon_run(run_dir, repository_root=root)
    selected = resolve_selected_models(
        inventory, best=bool(args.best), models=list(args.models)
    )
    return run_dir, list(selected), None


def _cmd_list(args: argparse.Namespace, root: Path) -> int:
    run_dir = resolve_run_dir(args.run_dir, root)
    inventory = inventory_autogluon_run(run_dir, repository_root=root)
    payload = {
        "ok": True,
        "command": "list",
        "artifacts_written": False,
        "run_path": inventory.run_path,
        "dataset_id": inventory.dataset_id,
        "classification": inventory.classification,
        "best_model": inventory.best_model,
        "model_names": list(inventory.model_names),
        "config_path": inventory.config_path,
        "config_sha256": inventory.config_sha256,
        "autogluon_version": inventory.autogluon_version,
        "leaderboard": list(inventory.leaderboard),
        "limitations": list(inventory.limitations),
    }
    _emit(payload)
    print(
        f"STATUS: listed {len(inventory.model_names)} models; "
        f"best={inventory.best_model}; read-only",
        file=sys.stderr,
    )
    return EXIT_OK


def _all_models_valid(results: list[Any]) -> bool:
    return bool(results) and all(
        isinstance(item, dict) and item.get("ok") is True for item in results
    )


def _cmd_validate(args: argparse.Namespace, root: Path) -> int:
    run_dir, selected, request_id = _resolve_selection(args, root)
    inventory = inventory_autogluon_run(run_dir, repository_root=root)
    predictor = load_predictor(inventory.run_dir / "predictor")
    results = [
        validate_autogluon_model_import(
            inventory,
            model_name,
            repository_root=root,
            predictor=predictor,
        )
        for model_name in selected
    ]
    all_models_valid = _all_models_valid(results)
    payload = {
        "ok": all_models_valid,
        "command": "validate",
        "artifacts_written": False,
        "request_id": request_id,
        "run_path": inventory.run_path,
        "selected_models": list(selected),
        "results": results,
        "all_models_valid": all_models_valid,
    }
    _emit(payload)
    if not all_models_valid:
        print(
            f"STATUS: validation failed for {len(results)} model(s); no artifacts written",
            file=sys.stderr,
        )
        return EXIT_INVALID
    print(
        f"STATUS: validated {len(results)} model(s); no artifacts written",
        file=sys.stderr,
    )
    return EXIT_OK


def _cmd_import(args: argparse.Namespace, root: Path) -> int:
    run_dir, selected, request_id = _resolve_selection(args, root)
    inventory = inventory_autogluon_run(run_dir, repository_root=root)
    predictor = load_predictor(inventory.run_dir / "predictor")
    imported: list[dict[str, Any]] = []
    for model_name in selected:
        package = import_autogluon_model_candidate(
            inventory,
            model_name,
            repository_root=root,
            predictor=predictor,
        )
        imported.append(
            {
                "candidate_id": package.candidate_id,
                "source_model_name": model_name,
                "package_path": candidate_summary(package)["package_path"],
                "train_row_count": package.manifest["train_row_count"],
                "test_row_count": package.manifest["test_row_count"],
                "success_marker": "_SUCCESS",
            }
        )
    payload = {
        "ok": True,
        "command": "import",
        "artifacts_written": True,
        "request_id": request_id,
        "run_path": inventory.run_path,
        "selected_models": list(selected),
        "imported": imported,
    }
    _emit(payload)
    print(
        f"STATUS: imported {len(imported)} candidate package(s); _SUCCESS written last",
        file=sys.stderr,
    )
    return EXIT_OK


def _explicit_request(args: argparse.Namespace) -> ExplicitExportRequest:
    return ExplicitExportRequest(
        oof_path=args.oof,
        test_path=args.test,
        dataset_id=str(args.dataset_id),
        source_model_name=str(args.source_model_name),
        source_kind=str(args.source_kind),
        source_run_path=str(args.source_run_path),
        threshold=float(args.threshold),
        exploratory=bool(args.exploratory),
        validation_balanced_accuracy=args.validation_balanced_accuracy,
        historical_kaggle_public_score=args.historical_kaggle_public_score,
        source_model_type=str(args.source_model_type),
    )


def _cmd_validate_explicit(args: argparse.Namespace, root: Path) -> int:
    result = validate_explicit_export(_explicit_request(args), repository_root=root)
    payload = {"command": "validate-explicit", **result}
    _emit(payload)
    print(
        "STATUS: explicit exports validated; no predictor loaded; read-only",
        file=sys.stderr,
    )
    return EXIT_OK


def _cmd_import_explicit(args: argparse.Namespace, root: Path) -> int:
    package = import_explicit_export_candidate(
        _explicit_request(args), repository_root=root
    )
    source_metadata = package.source_metadata
    threshold = float(source_metadata["final_deployment_threshold"])
    payload = {
        "ok": True,
        "command": "import-explicit",
        "artifacts_written": True,
        "candidate_id": package.candidate_id,
        "package_path": candidate_summary(package)["package_path"],
        "dataset_id": package.manifest["dataset_id"],
        "target_dependency": package.manifest["target_dependency"],
        "exploratory": bool(package.manifest["exploratory"]),
        "source_model_name": package.manifest["source_model_name"],
        "source_kind": package.manifest["source_kind"],
        "oof_export": source_metadata["oof_export"],
        "test_export": source_metadata["test_export"],
        "train_row_count": package.manifest["train_row_count"],
        "test_row_count": package.manifest["test_row_count"],
        "threshold": threshold,
        "predicted_positive_count": int(
            (package.test["probability_positive"] >= threshold).sum()
        ),
        "predictor_loaded": False,
        "predictions_generated": False,
        "success_marker": "_SUCCESS",
    }
    _emit(payload)
    print(
        f"STATUS: imported explicit candidate {package.candidate_id}; "
        "no predictor loaded; _SUCCESS written last",
        file=sys.stderr,
    )
    return EXIT_OK


def _cmd_inspect(args: argparse.Namespace, root: Path) -> int:
    if bool(args.candidate_dir) == bool(args.candidate_id):
        raise PredictionCandidateError(
            "Inspect requires exactly one of --candidate-dir or --candidate-id.",
            reason_code="inspect_target_invalid",
        )
    if args.candidate_id:
        package_dir = resolve_under_repository(
            f"{CANDIDATE_ROOT_RELATIVE}/{args.candidate_id}",
            root,
        )
    else:
        package_dir = (
            args.candidate_dir.resolve(strict=True)
            if args.candidate_dir.is_absolute()
            else resolve_under_repository(
                args.candidate_dir.as_posix().replace("\\", "/"),
                root,
            )
        )
    package = load_candidate_package(package_dir, repository_root=root)
    validation = validate_candidate_package(package, repository_root=root)
    summary = candidate_summary(package)
    oof_probs = package.oof["probability_positive"]
    payload = {
        "ok": True,
        "command": "inspect",
        "artifacts_written": False,
        "summary": summary,
        "validation": validation,
        "source": {
            "source_kind": package.manifest["source_kind"],
            "source_model_name": package.manifest["source_model_name"],
            "source_model_type": package.manifest["source_model_type"],
            "source_run_path": package.manifest["source_run_path"],
            "source_metric_name": package.manifest["source_metric_name"],
            "source_metric_value": package.manifest["source_metric_value"],
            "oof_protocol": package.manifest["oof_protocol"],
        },
        "identity": {
            "dataset_id": package.manifest["dataset_id"],
            "target_hash": package.manifest["target_hash"],
            "train_anchor_hash": package.manifest["train_anchor_hash"],
            "test_anchor_hash": package.manifest["test_anchor_hash"],
            "train_row_position_hash": package.manifest["train_row_position_hash"],
            "test_row_position_hash": package.manifest["test_row_position_hash"],
            "ordered_submission_id_hash": package.manifest[
                "ordered_submission_id_hash"
            ],
            "positive_class_label": package.manifest["positive_class_label"],
            "probability_semantics": package.manifest["probability_semantics"],
        },
        "probability_range": {
            "oof_min": float(oof_probs.min()),
            "oof_max": float(oof_probs.max()),
            "test_min": float(package.test["probability_positive"].min()),
            "test_max": float(package.test["probability_positive"].max()),
        },
        "source_metadata": package.source_metadata,
    }
    _emit(payload)
    print(
        f"STATUS: inspected {package.candidate_id}; "
        f"model={package.manifest['source_model_name']}",
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
