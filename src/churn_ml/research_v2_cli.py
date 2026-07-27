from __future__ import annotations

import argparse
import socket
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.churn_ml.experiment_v2 import (
    CandidateAdapter,
    get_candidate_adapter,
    get_feature_pipeline,
)
from src.churn_ml.research_evaluation import run_research_evaluation
from src.churn_ml.research_protocol import (
    EvaluationAssignments,
    build_evaluation_assignments,
    build_evaluation_plan_identity,
)
from src.churn_ml.research_v2_artifacts import ResearchV2ArtifactStore
from src.churn_ml.research_v2_config import (
    ResearchV2Config,
    load_research_v2_config,
)
from src.churn_ml.research_v2_data import (
    ResearchV2TrainingData,
    load_research_v2_training_data,
)
from src.churn_ml.research_v2_identity import (
    build_component_identities,
    source_record_mapping,
)
from src.churn_ml.research_v2_provenance import (
    SOURCE_PATHS,
    environment_identity,
    file_identity,
    loaded_module_identity,
    runtime_identity,
)
from src.churn_ml.run_artifacts import collect_git_state


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESS_STARTED_AT_UTC = datetime.now(timezone.utc)


@dataclass(frozen=True)
class ResearchV2Preflight:
    config: ResearchV2Config
    data: ResearchV2TrainingData
    assignments: EvaluationAssignments
    adapter: CandidateAdapter
    identities: dict[str, dict[str, Any]]
    hashes: dict[str, str]
    environment: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run experiment-core v2 train-only research evaluation."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--run-id")
    return parser.parse_args()


def preflight(config_path: Path) -> ResearchV2Preflight:
    config = load_research_v2_config(config_path, project_root=PROJECT_ROOT)
    data = load_research_v2_training_data(config)
    assignments = build_evaluation_assignments(data.y, config.plan_payload)
    plan_identity, plan_hash = build_evaluation_plan_identity(
        config.plan_payload,
        data.fingerprints,
        assignments,
    )
    sources, source_hash = file_identity(
        PROJECT_ROOT,
        (
            *SOURCE_PATHS,
            config.source_path.relative_to(PROJECT_ROOT).as_posix(),
            config.plan_path.relative_to(PROJECT_ROOT).as_posix(),
        ),
    )
    pipeline = get_feature_pipeline(config.pipeline_id)
    adapter = get_candidate_adapter(config.adapter_id)
    environment = environment_identity(config.adapter_id)
    component_identities, component_hashes = build_component_identities(
        pipeline_inputs=pipeline.identity_inputs(config.pipeline_contract),
        adapter_inputs=adapter.identity_inputs(config.adapter_contract),
        resolved_feature_schema=data.pipeline_output.schema.to_dict(),
        runtime_dependencies=environment,
        dataset_version=config.dataset_version,
        source_records=source_record_mapping(sources),
    )
    loaded, loaded_hash = loaded_module_identity(PROJECT_ROOT)
    hashes = {
        "plan": plan_hash,
        **component_hashes,
        "source": source_hash,
        "loaded_modules": loaded_hash,
    }
    identities = {
        "evaluation_plan": {"sha256": plan_hash, "canonical": plan_identity},
        **component_identities,
        "source": {"sha256": source_hash, "canonical": sources},
        "loaded_modules": {"sha256": loaded_hash, "canonical": loaded},
    }
    return ResearchV2Preflight(
        config=config,
        data=data,
        assignments=assignments,
        adapter=adapter,
        identities=identities,
        hashes=hashes,
        environment=environment,
    )


def execute(
    args: argparse.Namespace,
    *,
    entry_point: str = "scripts/run_research_v2.py",
    process_started_at_utc: datetime = PROCESS_STARTED_AT_UTC,
) -> int:
    config_path = args.config
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    prepared = preflight(config_path)
    if args.validate_only:
        print("Research v2 validation successful.")
        print(f"Experiment: {prepared.config.experiment_id}")
        print(f"Plan: {prepared.config.plan_id}")
        print(f"Feature pipeline: {prepared.config.pipeline_id}")
        print(f"Candidate adapter: {prepared.config.adapter_id}")
        for name, value in prepared.hashes.items():
            print(f"{name.replace('_', ' ').title()} SHA256: {value}")
        print(f"Training rows: {len(prepared.data.X)}")
        return 0

    metadata = {
        "schema_version": 2,
        "experiment_id": prepared.config.experiment_id,
        "plan_id": prepared.config.plan_id,
        "feature_pipeline_id": prepared.config.pipeline_id,
        "candidate_adapter_id": prepared.config.adapter_id,
        "hashes": prepared.hashes,
        "status": "running",
        "started_at_utc": None,
        "environment": prepared.environment,
        "runtime": runtime_identity(
            process_started_at_utc=process_started_at_utc,
            entry_point=entry_point,
        ),
        "git": collect_git_state(PROJECT_ROOT),
        "competition_assets_accessed": False,
        "tracking_enabled": False,
    }
    store: ResearchV2ArtifactStore | None = None
    try:
        store = ResearchV2ArtifactStore(
            prepared.config,
            plan_hash=prepared.hashes["plan"],
            candidate_hash=prepared.hashes["candidate"],
            expected_hashes=prepared.hashes,
            run_id=args.run_id,
        )
        metadata["run_id"] = store.run_id
        metadata["started_at_utc"] = store.started_at.isoformat()
        store.save_initial(
            metadata=metadata,
            identities=prepared.identities,
            fingerprints=prepared.data.fingerprints,
            feature_schema=prepared.data.pipeline_output.schema.to_dict(),
            assignments=prepared.assignments,
        )
        print(f"Run directory: {store.root}")

        def fit_predict(
            train_features: Any,
            train_labels: Any,
            prediction_features: Any,
            candidate_contract: Any,
            **fit_kwargs: Any,
        ) -> Any:
            del candidate_contract
            return prepared.adapter.fit_predict(
                train_features,
                train_labels,
                prediction_features,
                prepared.config.adapter_contract,
                **fit_kwargs,
            )

        def persist_fold(fold: Any) -> None:
            assert store is not None
            store.save_fold(fold)
            print(
                f"Repeat {fold.repeat}, outer fold {fold.outer_fold}: "
                f"threshold={fold.selected_threshold['selected_threshold']:.3f}, "
                f"BA={fold.fold_metrics['balanced_accuracy']:.6f}"
            )

        with network_disabled():
            result = run_research_evaluation(
                prepared.data.X,
                prepared.data.y,
                prepared.assignments,
                prepared.config.plan_payload,
                prepared.config.adapter_contract,
                fit_predict=fit_predict,
                on_outer_fold_complete=persist_fold,
            )
        store.save_result(result)
        balanced = result.aggregate_metrics["metrics"]["balanced_accuracy"]
        sample_sd = balanced["sample_standard_deviation"]
        sample_sd_text = "N/A" if sample_sd is None else f"{sample_sd:.6f}"
        completion_lines = (
            (
                "Nested Balanced Accuracy: "
                f"mean={balanced['mean']:.6f}, sample_sd={sample_sd_text}, "
                f"min={balanced['minimum']:.6f}, max={balanced['maximum']:.6f}"
            ),
            (
                "Threshold distribution: "
                f"median={result.threshold_summary['median']:.3f}, "
                f"range=[{result.threshold_summary['minimum']:.3f}, "
                f"{result.threshold_summary['maximum']:.3f}]"
            ),
            "Status: completed",
        )

        def emit_success_report() -> None:
            for line in completion_lines:
                print(line)

        store.complete(metadata, result, pre_success_output=emit_success_report)
        return 0
    except BaseException as error:
        if store is not None and not (store.root / "_SUCCESS").exists():
            store.fail(error, metadata)
        print(
            f"Status: failed ({type(error).__name__}: {error})",
            file=sys.stderr,
        )
        if store is not None:
            print(f"Failed run directory: {store.root}", file=sys.stderr)
        return 1


@contextmanager
def network_disabled() -> Iterator[None]:
    original_socket = socket.socket
    original_connection = socket.create_connection

    def deny(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("Network access is prohibited during research v2.")

    socket.socket = deny  # type: ignore[misc,assignment]
    socket.create_connection = deny  # type: ignore[misc,assignment]
    try:
        yield
    finally:
        socket.socket = original_socket  # type: ignore[misc,assignment]
        socket.create_connection = original_connection


def main() -> int:
    return execute(parse_args())
