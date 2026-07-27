from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from src.churn_ml.deployment_v1 import (
    execute_deployment,
    network_disabled,
    validate_only,
)
from src.churn_ml.deployment_v1_artifacts import (
    DeploymentArtifactError,
    load_completed_deployment,
    load_failed_deployment,
)
from src.churn_ml.deployment_v1_contracts import DeploymentContractError
from src.churn_ml.deployment_v1_paths import (
    DeploymentPathError,
    validate_path_chain,
)


PROJECT_ROOT = Path(__file__).absolute().parents[2]
EXIT_SUCCESS = 0
EXIT_VALIDATION = 2
EXIT_SAFETY = 3
EXIT_EXECUTION = 4


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict P4 deployment framework v1.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--config", type=Path, required=True)

    dry_parser = subparsers.add_parser("dry-run")
    dry_parser.add_argument("--config", type=Path, required=True)
    dry_parser.add_argument("--fixture-dir", type=Path, required=True)
    dry_parser.add_argument("--output-dir", type=Path, required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--allow-competition-test", action="store_true")

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--deployment-dir", type=Path, required=True)
    return parser.parse_args(argv)


def execute(args: argparse.Namespace) -> int:
    try:
        if args.command == "inspect":
            root = _resolve_path(
                args.deployment_dir, require_exists=True, expected_kind="directory"
            )
            if (root / "_SUCCESS").is_file():
                loaded = load_completed_deployment(root, project_root=PROJECT_ROOT)
                _emit(
                    {
                        "ok": True,
                        "command": "inspect",
                        "status": "completed",
                        "deployment_dir": str(loaded.root),
                        "deployment_identity_sha256": loaded.deployment_identity[
                            "sha256"
                        ],
                        "prediction_summary": loaded.prediction_summary,
                        "manifest_sha256": loaded.manifest["manifest_sha256"],
                    }
                )
            else:
                loaded_failed = load_failed_deployment(root)
                _emit(
                    {
                        "ok": True,
                        "command": "inspect",
                        "status": "failed",
                        "deployment_dir": str(loaded_failed.root),
                        "failure": loaded_failed.failure,
                    }
                )
            return EXIT_SUCCESS

        config_path = _resolve_project_path(args.config)
        validated = validate_only(config_path, project_root=PROJECT_ROOT)
        if args.command == "validate":
            _emit(
                {
                    "ok": True,
                    "command": "validate",
                    "deployment_id": validated.config.deployment_id,
                    "deployment_identity_sha256": validated.identity_sha256,
                    "component_count": len(validated.approvals),
                    "competition_test_accessed": False,
                    "output_allocated": False,
                }
            )
            return EXIT_SUCCESS
        if args.command == "run" and not args.allow_competition_test:
            _emit_error(
                "competition_test_confirmation_required",
                "run requires --allow-competition-test",
            )
            return EXIT_SAFETY
        with network_disabled():
            if args.command == "dry-run":
                result = execute_deployment(
                    validated,
                    mode="dry-run",
                    fixture_dir=_resolve_path(
                        args.fixture_dir,
                        require_exists=True,
                        expected_kind="directory",
                    ),
                    output_dir=_resolve_path(
                        args.output_dir,
                        require_exists=False,
                        expected_kind="either",
                    ),
                )
            else:
                result = execute_deployment(validated, mode="run")
        _emit(
            {
                "ok": True,
                "command": args.command,
                "deployment_dir": str(result.root),
                "deployment_identity_sha256": result.deployment_identity_sha256,
                "positive_count": result.positive_count,
                "positive_rate": result.positive_rate,
                "uploaded": False,
            }
        )
        return EXIT_SUCCESS
    except (DeploymentContractError, DeploymentArtifactError) as error:
        _emit_error("validation_failed", str(error))
        return EXIT_VALIDATION
    except (FileExistsError, PermissionError, DeploymentPathError) as error:
        _emit_error("safety_refusal", str(error))
        return EXIT_SAFETY
    except BaseException as error:
        _emit_error("execution_failed", f"{type(error).__name__}: {error}")
        return EXIT_EXECUTION


def main(argv: Sequence[str] | None = None) -> int:
    return execute(parse_args(argv))


def _resolve_project_path(path: Path) -> Path:
    return validate_path_chain(
        containment_root=PROJECT_ROOT,
        requested_path=path,
        require_exists=True,
        expected_kind="file",
        reject_hardlinks=True,
    ).canonical


def _resolve_path(path: Path, *, require_exists: bool, expected_kind: str) -> Path:
    absolute = path if path.is_absolute() else PROJECT_ROOT / path
    return validate_path_chain(
        containment_root=Path(absolute.absolute().anchor),
        requested_path=absolute,
        require_exists=require_exists,
        expected_kind=expected_kind,  # type: ignore[arg-type]
        reject_hardlinks=require_exists and expected_kind == "file",
    ).canonical


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def _emit_error(code: str, message: str) -> None:
    print(
        json.dumps(
            {"ok": False, "error": {"code": code, "message": message}},
            sort_keys=True,
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )
