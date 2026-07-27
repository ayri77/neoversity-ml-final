from __future__ import annotations

import argparse
import json
import socket
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src.churn_ml.experiment_v2 import get_candidate_adapter
from src.churn_ml.optuna_search_artifacts import (
    OptunaSearchArtifactError,
    load_optuna_search_result,
    write_failed_attempt,
)
from src.churn_ml.optuna_search_config import (
    OptunaSearchConfigurationError,
    load_optuna_search_config,
)
from src.churn_ml.optuna_search_export import (
    OptunaSearchExportError,
    export_best_candidate,
)
from src.churn_ml.optuna_search_lifecycle import (
    OptunaSearchLifecycleError,
    portable_dataset_identity,
    run_optuna_study,
)
from src.churn_ml.optuna_search_objective import build_search_assignments
from src.churn_ml.research_v2_data import load_research_v2_training_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXIT_SUCCESS = 0
EXIT_UNEXPECTED = 1
EXIT_CONFIG = 2
EXIT_ARTIFACT = 3
EXIT_STUDY = 4
EXIT_EXPORT = 5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic train-only Optuna Search v1."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser(
        "validate",
        help="Validate config and referenced contracts without allocation.",
    )
    validate.add_argument("--config", type=Path, required=True)
    run = subparsers.add_parser("run", help="Create or resume a train-only study.")
    run.add_argument("--config", type=Path, required=True)
    inspect = subparsers.add_parser(
        "inspect",
        help="Read and validate one immutable completed search.",
    )
    inspect.add_argument("--search-dir", type=Path, required=True)
    export = subparsers.add_parser(
        "export-best",
        help="Copy the fixed best candidate config without overwriting.",
    )
    export.add_argument("--search-dir", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def execute(args: argparse.Namespace) -> int:
    config = None
    try:
        if args.command == "validate":
            config = load_optuna_search_config(
                _project_path(args.config),
                project_root=PROJECT_ROOT,
            )
            _emit(
                {
                    "status": "valid",
                    "schema_version": 1,
                    "search_plan_id": config.payload["search_plan_id"],
                    "search_id": config.search_id,
                    "study_name": config.payload["study_name"],
                    "study_identity_sha256": config.study_identity_sha256,
                    "search_identity_sha256": config.search_identity_sha256,
                    "search_space_sha256": config.search_space.sha256,
                    "adapter_id": config.adapter_id,
                    "allocation_performed": False,
                }
            )
            return EXIT_SUCCESS
        if args.command == "inspect":
            inspection_result = load_optuna_search_result(
                _project_path(args.search_dir),
                project_root=PROJECT_ROOT,
            )
            _emit(
                {
                    "status": "completed",
                    "search_dir": inspection_result.search_dir.relative_to(
                        PROJECT_ROOT
                    ).as_posix(),
                    "search_identity": inspection_result.search_identity,
                    "study_summary": inspection_result.study_summary,
                    "best_trial": inspection_result.best_trial,
                    "read_only": True,
                }
            )
            return EXIT_SUCCESS
        if args.command == "export-best":
            target = export_best_candidate(
                _project_path(args.search_dir),
                args.output,
                project_root=PROJECT_ROOT,
            )
            _emit(
                {
                    "status": "exported",
                    "output": target.relative_to(PROJECT_ROOT).as_posix(),
                    "overwrite": False,
                    "final_evaluation_run": False,
                }
            )
            return EXIT_SUCCESS
        if args.command == "run":
            config = load_optuna_search_config(
                _project_path(args.config),
                project_root=PROJECT_ROOT,
            )
            final_dir = config.artifact_root / config.search_id
            if final_dir.exists():
                raise OptunaSearchArtifactError(
                    f"Immutable search output already exists: {final_dir}."
                )
            data = load_research_v2_training_data(config.base_config)
            assignments = build_search_assignments(
                data.y,
                repeats=int(config.payload["repeats"]),
                folds=int(config.payload["folds"]),
                assignment_seed=int(config.payload["assignment_seed"]),
            )
            adapter = get_candidate_adapter(config.adapter_id)
            identity_inputs = adapter.identity_inputs(
                config.base_config.adapter_contract
            )
            if (
                identity_inputs.get("probability_semantics")
                != "binary_positive_class_label_1"
                or identity_inputs.get("early_stopping") != "disabled"
            ):
                raise OptunaSearchLifecycleError(
                    "Adapter probability or early-stopping semantics differ."
                )
            with network_disabled():
                study_result = run_optuna_study(
                    config,
                    X=data.X,
                    y=data.y,
                    dataset_identity=portable_dataset_identity(
                        data.fingerprints,
                        project_root=PROJECT_ROOT,
                    ),
                    assignments=assignments,
                    adapter=adapter,
                )
            _emit(
                {
                    "status": "completed",
                    "search_dir": study_result.search_dir.relative_to(
                        PROJECT_ROOT
                    ).as_posix(),
                    "study_summary": study_result.study_summary,
                    "best_trial": study_result.best_trial,
                    "selection_plan_run": False,
                    "confirmation_plan_run": False,
                    "submission_created": False,
                }
            )
            return EXIT_SUCCESS
        raise RuntimeError(f"Unknown command: {args.command}.")
    except OptunaSearchConfigurationError as error:
        _emit_error("configuration_error", error, EXIT_CONFIG)
        return EXIT_CONFIG
    except OptunaSearchArtifactError as error:
        _emit_error("artifact_error", error, EXIT_ARTIFACT)
        return EXIT_ARTIFACT
    except OptunaSearchLifecycleError as error:
        if config is not None:
            _persist_failure(config, error, "STUDY_LIFECYCLE_FAILED")
        _emit_error("study_error", error, EXIT_STUDY)
        return EXIT_STUDY
    except OptunaSearchExportError as error:
        _emit_error("export_error", error, EXIT_EXPORT)
        return EXIT_EXPORT
    except BaseException as error:
        if config is not None and getattr(args, "command", None) == "run":
            _persist_failure(config, error, "UNEXPECTED_SEARCH_FAILURE")
        _emit_error("unexpected_error", error, EXIT_UNEXPECTED)
        return EXIT_UNEXPECTED


def _persist_failure(config: Any, error: BaseException, reason_code: str) -> None:
    try:
        write_failed_attempt(config, error, reason_code=reason_code)
    except BaseException:
        pass


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str))


def _emit_error(kind: str, error: BaseException, exit_code: int) -> None:
    print(
        json.dumps(
            {
                "status": "failed",
                "error_kind": kind,
                "error_type": type(error).__name__,
                "message": str(error),
                "exit_code": exit_code,
            },
            sort_keys=True,
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )


@contextmanager
def network_disabled() -> Iterator[None]:
    original_socket = socket.socket
    original_connection = socket.create_connection

    def deny(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("Network access is prohibited during Optuna Search v1.")

    socket.socket = deny  # type: ignore[misc,assignment]
    socket.create_connection = deny  # type: ignore[misc,assignment]
    try:
        yield
    finally:
        socket.socket = original_socket  # type: ignore[misc,assignment]
        socket.create_connection = original_connection


def main() -> int:
    return execute(parse_args())
