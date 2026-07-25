from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.churn_ml.research_artifacts import ResearchArtifactStore
from src.churn_ml.research_config import ResearchConfig, load_research_config
from src.churn_ml.research_data import (
    ResearchTrainingData,
    load_research_training_data,
)
from src.churn_ml.research_evaluation import run_research_evaluation
from src.churn_ml.research_protocol import (
    EvaluationAssignments,
    build_candidate_contract_identity,
    build_evaluation_assignments,
    build_evaluation_plan_identity,
)
from src.churn_ml.research_provenance import (
    candidate_source_provenance,
    run_implementation_provenance,
)
from src.churn_ml.research_runtime_provenance import (
    build_invocation_provenance,
    build_loaded_module_provenance,
    collect_research_environment_versions,
)
from src.churn_ml.run_artifacts import collect_git_state


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESS_STARTED_AT_UTC = datetime.now(timezone.utc)


@dataclass(frozen=True)
class ResearchPreflight:
    config: ResearchConfig
    data: ResearchTrainingData
    assignments: EvaluationAssignments
    plan_identity: dict[str, Any]
    plan_hash: str
    candidate_identity: dict[str, Any]
    candidate_hash: str
    candidate_source_manifest_hash: str
    run_implementation_identity: dict[str, Any]
    run_implementation_hash: str
    loaded_module_identity: dict[str, Any]
    loaded_module_hash: str
    environment: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run nested repeated train-only research evaluation."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--run-id")
    return parser.parse_args()


def preflight(
    config_path: Path,
    *,
    entrypoint_module_name: str | None = None,
) -> ResearchPreflight:
    """Validate data, identities, dependencies, and loaded code before allocation."""
    config = load_research_config(config_path, project_root=PROJECT_ROOT)
    data = load_research_training_data(config)
    assignments = build_evaluation_assignments(data.y, config.plan_payload)
    plan_identity, plan_hash = build_evaluation_plan_identity(
        config.plan_payload,
        data.fingerprints,
        assignments,
    )
    environment = collect_research_environment_versions()
    candidate_sources, candidate_source_hash = candidate_source_provenance(
        config.project_root
    )
    candidate_sources = {
        **candidate_sources,
        "manifest_sha256": candidate_source_hash,
    }
    candidate_identity, candidate_hash = build_candidate_contract_identity(
        config.candidate_contract,
        feature_schema=data.feature_schema.to_dict(),
        source_provenance=candidate_sources,
        runtime_dependencies={
            name: environment[name]
            for name in (
                "numpy",
                "pandas",
                "scikit_learn",
                "lightgbm",
                "pyarrow",
            )
        },
    )
    candidate_source = config.candidate_contract["source"]
    run_identity, run_hash = run_implementation_provenance(
        config.project_root,
        config_paths={
            "research_run": config.source_path,
            "evaluation_plan": config.plan_path,
            "historical_candidate": Path(
                str(candidate_source["historical_config_path"])
            ),
            "baseline_manifest": Path(str(candidate_source["manifest_path"])),
        },
    )
    loaded_identity, loaded_hash = build_loaded_module_provenance(
        config.project_root,
        (candidate_sources, run_identity),
        entrypoint_module_name=entrypoint_module_name,
    )
    return ResearchPreflight(
        config=config,
        data=data,
        assignments=assignments,
        plan_identity=plan_identity,
        plan_hash=plan_hash,
        candidate_identity=candidate_identity,
        candidate_hash=candidate_hash,
        candidate_source_manifest_hash=candidate_source_hash,
        run_implementation_identity=run_identity,
        run_implementation_hash=run_hash,
        loaded_module_identity=loaded_identity,
        loaded_module_hash=loaded_hash,
        environment=environment,
    )


def execute(
    args: argparse.Namespace,
    *,
    entrypoint_module_name: str | None = None,
    entry_point: str = "src/churn_ml/research_cli.py",
    process_started_at_utc: datetime = PROCESS_STARTED_AT_UTC,
) -> int:
    config_path = args.config
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    prepared = preflight(
        config_path,
        entrypoint_module_name=entrypoint_module_name,
    )
    invocation = build_invocation_provenance(
        entry_point=entry_point,
        process_started_at_utc=process_started_at_utc,
    )
    if args.validate_only:
        print("Research validation successful.")
        print(f"Experiment: {prepared.config.experiment_id}")
        print(f"Plan: {prepared.config.plan_id}")
        print(f"Plan SHA256: {prepared.plan_hash}")
        print(f"Candidate SHA256: {prepared.candidate_hash}")
        print(
            "Candidate source manifest SHA256: "
            f"{prepared.candidate_source_manifest_hash}"
        )
        print(f"Run implementation SHA256: {prepared.run_implementation_hash}")
        print(f"Loaded module SHA256: {prepared.loaded_module_hash}")
        print(f"Training rows: {len(prepared.data.X)}")
        repeat_count = len(
            prepared.config.plan_payload["outer_evaluation"]["repeat_seeds"]
        )
        outer_fold_count = int(
            prepared.config.plan_payload["outer_evaluation"]["n_splits"]
        )
        print(f"Expected outer folds: {repeat_count * outer_fold_count}")
        return 0

    git_state = _validated_git_state(
        prepared.run_implementation_identity,
    )
    metadata = {
        "experiment_id": prepared.config.experiment_id,
        "candidate_id": prepared.config.candidate_id,
        "plan_id": prepared.config.plan_id,
        "plan_sha256": prepared.plan_hash,
        "candidate_sha256": prepared.candidate_hash,
        "candidate_source_manifest_sha256": (prepared.candidate_source_manifest_hash),
        "run_implementation_sha256": prepared.run_implementation_hash,
        "loaded_module_sha256": prepared.loaded_module_hash,
        "status": "running",
        "started_at_utc": None,
        "config_source_path": str(prepared.config.source_path),
        "plan_source_path": str(prepared.config.plan_path),
        "environment": prepared.environment,
        "invocation": invocation,
        "git": git_state,
        "upstream_feature_selection_limitation": (
            "v3_targeted_missingness contains targeted indicators selected using "
            "a prior full-training-data experiment; this evaluation is conditional "
            "on the frozen processed dataset."
        ),
        "competition_assets_accessed": False,
    }
    store: ResearchArtifactStore | None = None
    try:
        store = ResearchArtifactStore(
            prepared.config,
            plan_hash=prepared.plan_hash,
            candidate_hash=prepared.candidate_hash,
            candidate_source_manifest_hash=(prepared.candidate_source_manifest_hash),
            run_implementation_hash=prepared.run_implementation_hash,
            loaded_module_hash=prepared.loaded_module_hash,
            run_id=args.run_id,
        )
        metadata["run_id"] = store.run_id
        metadata["started_at_utc"] = store.started_at.isoformat()
        store.save_initial_contract(
            metadata=metadata,
            plan_identity=prepared.plan_identity,
            candidate_identity=prepared.candidate_identity,
            fingerprints=prepared.data.fingerprints,
            feature_schema=prepared.data.feature_schema.to_dict(),
            run_implementation_identity=prepared.run_implementation_identity,
            loaded_module_identity=prepared.loaded_module_identity,
            assignments=prepared.assignments,
        )
        print(f"Run directory: {store.root}")

        def persist_fold(fold: Any) -> None:
            store.save_completed_fold(fold)
            print(
                f"Repeat {fold.repeat}, outer fold {fold.outer_fold}: "
                f"threshold={fold.selected_threshold['selected_threshold']:.3f}, "
                f"BA={fold.fold_metrics['balanced_accuracy']:.6f}"
            )

        result = run_research_evaluation(
            prepared.data.X,
            prepared.data.y,
            prepared.assignments,
            prepared.config.plan_payload,
            prepared.config.candidate_contract,
            on_outer_fold_complete=persist_fold,
        )
        store.save_final_outputs(result)
        summary_lines = _build_summary_lines(result)
        store.complete(metadata, result)
    except BaseException as error:
        if store is not None and not (store.root / "_SUCCESS").exists():
            try:
                store.fail(error, metadata)
            except BaseException as persistence_error:
                print(
                    "Failure persistence also failed: "
                    f"{type(persistence_error).__name__}: {persistence_error}",
                    file=sys.stderr,
                )
        print(
            f"Status: failed ({type(error).__name__}: {error})",
            file=sys.stderr,
        )
        if store is not None:
            print(f"Failed run directory: {store.root}", file=sys.stderr)
        return 1

    for line in summary_lines:
        print(line)
    print("Status: completed")
    return 0


def _validated_git_state(
    run_implementation_identity: dict[str, Any],
) -> dict[str, Any]:
    git_state = collect_git_state(PROJECT_ROOT)
    required = {"branch", "commit", "dirty", "status_porcelain"}
    if set(git_state) != required:
        raise RuntimeError(f"Could not resolve complete Git provenance: {git_state}")
    implementation_paths = {
        item["path"] for item in run_implementation_identity["files"]
    }
    git_state["untracked_implementation_files"] = sorted(
        line[3:].replace("\\", "/")
        for line in git_state["status_porcelain"]
        if line.startswith("?? ")
        and line[3:].replace("\\", "/") in implementation_paths
    )
    return git_state


def _build_summary_lines(result: Any) -> list[str]:
    aggregate = result.aggregate_metrics["metrics"]["balanced_accuracy"]
    sample_sd_value = aggregate["sample_standard_deviation"]
    sample_sd = "N/A" if sample_sd_value is None else f"{sample_sd_value:.6f}"
    return [
        (
            "Nested Balanced Accuracy across repeat-level pooled results: "
            f"mean={aggregate['mean']:.6f}, sample_sd={sample_sd}, "
            f"min={aggregate['minimum']:.6f}, max={aggregate['maximum']:.6f}"
        ),
        (
            "Threshold distribution: "
            f"median={result.threshold_summary['median']:.3f}, "
            f"IQR={result.threshold_summary['interquartile_range']:.3f}, "
            f"range=[{result.threshold_summary['minimum']:.3f}, "
            f"{result.threshold_summary['maximum']:.3f}]"
        ),
    ]


def main() -> int:
    return execute(
        parse_args(),
        entrypoint_module_name="__main__",
        entry_point="scripts/run_research_evaluation.py",
        process_started_at_utc=PROCESS_STARTED_AT_UTC,
    )
