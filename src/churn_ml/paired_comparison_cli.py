from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.churn_ml.paired_comparison import (
    PairedCompatibilityError,
    PairedComparisonError,
    build_comparison_result,
    build_compatibility_summary,
    load_comparison_policy,
    load_completed_research_v2_run,
)
from src.churn_ml.paired_comparison_artifacts import (
    PairedComparisonArtifactError,
    create_comparison_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path("artifacts/research_v2_comparisons")
DEFAULT_POLICY = Path("configs/research_v2/comparison_policy_v1.yaml")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two completed, compatible Experiment Core v2 research runs."
        )
    )
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--comparison-id")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def execute(args: argparse.Namespace) -> int:
    try:
        policy_path = _resolve_repository_path(args.policy)
        policy = load_comparison_policy(policy_path)
        baseline = load_completed_research_v2_run(
            args.baseline_run_dir,
            project_root=PROJECT_ROOT,
        )
        candidate = load_completed_research_v2_run(
            args.candidate_run_dir,
            project_root=PROJECT_ROOT,
        )
        compatibility = build_compatibility_summary(baseline, candidate)
        if not compatibility.compatible:
            print(json.dumps(compatibility.to_dict(), indent=2), file=sys.stderr)
            return 2
        if args.validate_only:
            print("Paired comparison input validation successful.")
            print(
                f"Evaluation plan SHA256: {baseline.evaluation_plan_identity['sha256']}"
            )
            print(f"Baseline manifest SHA256: {baseline.manifest['manifest_sha256']}")
            print(f"Candidate manifest SHA256: {candidate.manifest['manifest_sha256']}")
            print("Compatibility: compatible")
            return 0
        result = build_comparison_result(baseline, candidate, policy)
        output_root = args.output_root
        comparison_root = create_comparison_artifacts(
            result=result,
            policy=policy,
            project_root=PROJECT_ROOT,
            output_root=output_root,
            comparison_id=args.comparison_id,
        )
        decision = result.decision_report
        balanced = result.paired_metrics.aggregate_summary["metrics"][
            "balanced_accuracy"
        ]
        print(f"Comparison directory: {comparison_root}")
        print(
            "Mean repeat Balanced Accuracy delta: "
            f"{balanced['candidate_minus_baseline']['mean']:.6f}"
        )
        print(f"Decision status: {decision['status']}")
        print("Status: completed")
        return 0
    except (PairedCompatibilityError, PairedComparisonError) as error:
        print(f"Status: failed ({type(error).__name__}: {error})", file=sys.stderr)
        return 2
    except PairedComparisonArtifactError as error:
        print(f"Status: failed ({type(error).__name__}: {error})", file=sys.stderr)
        return 1
    except RuntimeError as error:
        print(f"Status: failed ({type(error).__name__}: {error})", file=sys.stderr)
        return 1


def _resolve_repository_path(path: Path) -> Path:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    resolved = resolved.resolve()
    if resolved == PROJECT_ROOT or PROJECT_ROOT not in resolved.parents:
        raise PairedComparisonError("Policy path must be inside the repository.")
    return resolved


def main() -> int:
    return execute(parse_args())
