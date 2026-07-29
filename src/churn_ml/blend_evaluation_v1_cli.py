from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.churn_ml.blend_evaluation_v1 import (
    BlendEvaluationError,
    load_blend_component_run,
    run_cross_fit_blend_evaluation,
    validate_blend_inputs,
)
from src.churn_ml.blend_evaluation_v1_artifacts import (
    BlendEvaluationArtifactError,
    create_blend_evaluation_artifacts,
    load_completed_blend_evaluation,
)
from src.churn_ml.blend_evaluation_v1_config import (
    BlendEvaluationConfig,
    load_blend_evaluation_config,
)
from src.churn_ml.blend_evaluation_v1_deployment import (
    generate_submission_variants,
    prepare_deployment_package,
)
from src.churn_ml.paired_comparison_paths import (
    PairedPathSafetyError,
    assert_pairwise_disjoint_paths,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-safe LightGBM + XGBoost Blend Evaluation v1 "
            "(second-level outer-fold cross-fitting)."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate config, base artifacts, and compatibility without writing output.",
    )
    validate_parser.add_argument("--config", type=Path, required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run leakage-safe blend evaluation and write artifacts.",
    )
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--evaluation-id")

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect a completed blend evaluation artifact.",
    )
    inspect_parser.add_argument("--evaluation-dir", type=Path, required=True)

    prepare_parser = subparsers.add_parser(
        "prepare-deployment",
        help="Prepare draft Final Deployment inputs from a completed blend evaluation.",
    )
    prepare_parser.add_argument("--evaluation-dir", type=Path, required=True)

    variants_parser = subparsers.add_parser(
        "generate-submission-variants",
        help=(
            "Build at most three submission CSVs from one completed deployment's "
            "component probabilities without retraining."
        ),
    )
    variants_parser.add_argument("--deployment-dir", type=Path, required=True)
    variants_parser.add_argument("--output-dir", type=Path)

    return parser.parse_args(argv)


def execute(args: argparse.Namespace) -> int:
    try:
        if args.command == "validate":
            return _validate(args.config)
        if args.command == "run":
            return _run(args.config, evaluation_id=args.evaluation_id)
        if args.command == "inspect":
            return _inspect(args.evaluation_dir)
        if args.command == "prepare-deployment":
            return _prepare_deployment(args.evaluation_dir)
        if args.command == "generate-submission-variants":
            return _generate_variants(args.deployment_dir, args.output_dir)
        raise BlendEvaluationError(f"Unknown command: {args.command}")
    except PairedPathSafetyError as error:
        _emit_error(error.reason_code, str(error))
        return 2
    except BlendEvaluationError as error:
        _emit_error(error.reason_code, str(error))
        return 2
    except BlendEvaluationArtifactError as error:
        _emit_error("BLEND_ARTIFACT_ERROR", str(error))
        return 1
    except RuntimeError as error:
        _emit_error("RUNTIME_ERROR", str(error))
        return 1


def main(argv: list[str] | None = None) -> int:
    return execute(parse_args(argv))


def _validate(config_path: Path) -> int:
    config = load_blend_evaluation_config(config_path, project_root=PROJECT_ROOT)
    lightgbm_run, xgboost_run = _load_components(config)
    compatibility = validate_blend_inputs(lightgbm_run, xgboost_run)
    assert_pairwise_disjoint_paths(
        {
            "lightgbm_run": lightgbm_run.root,
            "xgboost_run": xgboost_run.root,
            "output_root": config.artifact_root,
        }
    )
    payload = {
        "ok": True,
        "compatibility": compatibility.to_dict(),
        "dataset_version": lightgbm_run.config.dataset_version,
        "evaluation_plan_sha256": lightgbm_run.evaluation_plan_identity["sha256"],
        "lightgbm_manifest_sha256": lightgbm_run.manifest["manifest_sha256"],
        "xgboost_manifest_sha256": xgboost_run.manifest["manifest_sha256"],
        "outer_prediction_contract": "sufficient_aligned_oof_probabilities",
        "reruns_required": False,
    }
    print(json.dumps(payload, indent=2))
    print("Blend evaluation input validation successful.")
    print("Compatibility: compatible")
    return 0


def _run(config_path: Path, *, evaluation_id: str | None) -> int:
    config = load_blend_evaluation_config(config_path, project_root=PROJECT_ROOT)
    lightgbm_run, xgboost_run = _load_components(config)
    assert_pairwise_disjoint_paths(
        {
            "lightgbm_run": lightgbm_run.root,
            "xgboost_run": xgboost_run.root,
            "output_root": config.artifact_root,
        }
    )
    result = run_cross_fit_blend_evaluation(
        lightgbm_run,
        xgboost_run,
        weight_grid=config.weight_grid(),
        maximizer_absolute_tolerance=config.maximizer_absolute_tolerance,
        prefer_closest_to=config.prefer_closest_to,
        sensitivity_weight_deltas=config.sensitivity_weight_deltas,
        forbid_pooled_oof_fallback=config.forbid_pooled_oof_fallback,
    )
    evaluation_root = create_blend_evaluation_artifacts(
        config=config,
        result=result,
        evaluation_id=evaluation_id,
    )
    params = result.deployment_parameters
    print(f"Blend evaluation directory: {evaluation_root}")
    print(
        "Deployment LightGBM weight (median): "
        f"{params['deployment_lightgbm_weight']:.6f}"
    )
    print(f"Deployment threshold (median): {params['deployment_threshold']:.6f}")
    print(
        "Aggregate mean Balanced Accuracy: "
        f"{result.summary['metrics']['balanced_accuracy']['aggregate_mean']:.6f}"
    )
    print("Status: completed")
    return 0


def _inspect(evaluation_dir: Path) -> int:
    root = load_completed_blend_evaluation(
        evaluation_dir,
        project_root=PROJECT_ROOT,
    )
    summary = _read_json(root / "aggregate_summary.json")
    parameters = _read_json(root / "deployment_parameters.json")
    payload = {
        "ok": True,
        "evaluation_dir": root.relative_to(PROJECT_ROOT).as_posix(),
        "deployment_parameters": parameters,
        "aggregate_balanced_accuracy": summary["metrics"]["balanced_accuracy"],
        "weight_summary": summary["weight_summary"],
        "threshold_summary": summary["threshold_summary"],
        "protocol": summary["protocol"],
        "pooled_oof_fallback_used": summary["pooled_oof_fallback_used"],
    }
    print(json.dumps(payload, indent=2))
    return 0


def _prepare_deployment(evaluation_dir: Path) -> int:
    payload = prepare_deployment_package(
        evaluation_dir,
        project_root=PROJECT_ROOT,
    )
    print(json.dumps({"ok": True, **payload}, indent=2))
    print("Status: completed")
    return 0


def _generate_variants(deployment_dir: Path, output_dir: Path | None) -> int:
    variants_root = generate_submission_variants(
        deployment_dir,
        project_root=PROJECT_ROOT,
        output_dir=output_dir,
    )
    print(f"Submission variants directory: {variants_root}")
    print("Status: completed")
    return 0


def _load_components(config: BlendEvaluationConfig) -> tuple[Any, Any]:
    lightgbm_run = load_blend_component_run(
        config.lightgbm_run_dir,
        project_root=PROJECT_ROOT,
        role="lightgbm",
        expected_adapter_id=config.lightgbm_adapter_id,
    )
    xgboost_run = load_blend_component_run(
        config.xgboost_run_dir,
        project_root=PROJECT_ROOT,
        role="xgboost",
        expected_adapter_id=config.xgboost_adapter_id,
    )
    return lightgbm_run, xgboost_run


def _emit_error(code: str, message: str) -> None:
    print(json.dumps({"ok": False, "error": code, "message": message}, indent=2))
    print(f"Status: failed ({code}: {message})", file=sys.stderr)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BlendEvaluationArtifactError(f"JSON root must be an object: {path}")
    return payload
