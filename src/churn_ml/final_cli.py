from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.churn_ml.final_artifact_validation import verify_reloaded_inference
from src.churn_ml.final_artifacts import FinalArtifactStore
from src.churn_ml.final_config import FinalConfig, load_final_config
from src.churn_ml.final_data import FinalData, build_final_predictions, load_final_data
from src.churn_ml.final_manual_lightgbm import (
    FinalFitResult,
    fit_final_manual_lightgbm,
)
from src.churn_ml.final_promotion import (
    OperationalThreshold,
    PromotionEvidence,
    derive_operational_threshold,
    verify_approved_research_run,
)
from src.churn_ml.final_provenance import (
    build_final_loaded_module_provenance,
    build_final_model_identity,
    build_final_source_provenance,
    build_invocation,
    build_promotion_identity,
    collect_final_environment,
    collect_validated_git_state,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESS_STARTED_AT_UTC = datetime.now(timezone.utc)


@dataclass(frozen=True)
class FinalPreflight:
    config: FinalConfig
    promotion: PromotionEvidence
    threshold: OperationalThreshold
    data: FinalData
    promotion_identity: dict[str, Any]
    promotion_hash: str
    final_model_identity: dict[str, Any]
    final_model_hash: str
    source_manifest: dict[str, Any]
    source_hash: str
    loaded_modules: dict[str, Any]
    loaded_hash: str
    environment: dict[str, str]
    invocation: dict[str, Any]
    git: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote one approved research candidate to one final submission."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--run-id")
    return parser.parse_args()


def preflight(
    config_path: Path,
    *,
    process_started_at_utc: datetime,
    entrypoint_module_name: str = "__main__",
) -> FinalPreflight:
    if Path.cwd().resolve() != PROJECT_ROOT:
        raise RuntimeError("Final CLI must execute from the repository root.")
    config = load_final_config(config_path, project_root=PROJECT_ROOT)

    # Competition test and sample assets are intentionally unavailable until
    # the approved research evidence and operational threshold pass.
    promotion = verify_approved_research_run(config)
    threshold = derive_operational_threshold(config, promotion)
    data = load_final_data(config)

    environment = collect_final_environment()
    source_manifest, source_hash = build_final_source_provenance(PROJECT_ROOT)
    loaded_modules, loaded_hash = build_final_loaded_module_provenance(
        PROJECT_ROOT,
        source_manifest,
        entrypoint_module_name=entrypoint_module_name,
    )
    promotion_identity, promotion_hash = build_promotion_identity(
        config,
        promotion,
        threshold,
    )
    final_model_identity, final_model_hash = build_final_model_identity(
        config,
        promotion_hash,
        data,
        source_hash,
        loaded_hash,
        environment,
        promotion.research_config.candidate_contract,
    )
    invocation = build_invocation(
        process_started_at_utc=process_started_at_utc,
    )
    git = collect_validated_git_state(PROJECT_ROOT, source_manifest)
    return FinalPreflight(
        config=config,
        promotion=promotion,
        threshold=threshold,
        data=data,
        promotion_identity=promotion_identity,
        promotion_hash=promotion_hash,
        final_model_identity=final_model_identity,
        final_model_hash=final_model_hash,
        source_manifest=source_manifest,
        source_hash=source_hash,
        loaded_modules=loaded_modules,
        loaded_hash=loaded_hash,
        environment=environment,
        invocation=invocation,
        git=git,
    )


def execute(
    args: argparse.Namespace,
    *,
    process_started_at_utc: datetime = PROCESS_STARTED_AT_UTC,
    entrypoint_module_name: str = "__main__",
) -> int:
    config_path = args.config
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    store: FinalArtifactStore | None = None
    try:
        prepared = preflight(
            config_path,
            process_started_at_utc=process_started_at_utc,
            entrypoint_module_name=entrypoint_module_name,
        )
        if args.validate_only:
            print("Final promotion validation successful.")
            print(f"Promotion: {prepared.config.promotion_id}")
            print(f"Promotion SHA256: {prepared.promotion_hash}")
            print(f"Final-model SHA256: {prepared.final_model_hash}")
            print(
                "Unbiased nested research estimate: "
                f"mean={prepared.config.payload['promotion']['research']['approved_metrics']['balanced_accuracy_mean']:.12f}, "
                f"sample_sd={prepared.config.payload['promotion']['research']['approved_metrics']['balanced_accuracy_sample_sd']:.12f}"
            )
            print(
                "Operational threshold-selection diagnostic: "
                f"threshold={prepared.threshold.selected_threshold:.3f}, "
                f"balanced_accuracy={prepared.threshold.diagnostic_balanced_accuracy:.12f}"
            )
            print("No artifact run was allocated.")
            return 0

        store = FinalArtifactStore(
            prepared.config,
            promotion_hash=prepared.promotion_hash,
            final_model_hash=prepared.final_model_hash,
            run_id=args.run_id,
        )
        store.save_initial_contract(
            promotion=prepared.promotion,
            threshold=prepared.threshold,
            data=prepared.data,
            promotion_identity=prepared.promotion_identity,
            final_model_identity=prepared.final_model_identity,
            source_manifest=prepared.source_manifest,
            source_hash=prepared.source_hash,
            loaded_modules=prepared.loaded_modules,
            loaded_hash=prepared.loaded_hash,
            environment=prepared.environment,
            invocation=prepared.invocation,
            git=prepared.git,
        )
        fit: FinalFitResult = fit_final_manual_lightgbm(
            prepared.data,
            prepared.promotion.research_config.candidate_contract,
        )
        predictions, submission = build_final_predictions(
            prepared.data.sample_submission,
            fit.test_probabilities,
            prepared.threshold.selected_threshold,
        )
        probabilities = fit.test_probabilities
        positive_count = int(predictions["prediction"].sum())
        diagnostics = {
            "schema_version": 1,
            "research_metric": {
                **prepared.config.payload["promotion"]["research"]["approved_metrics"],
                "claim": "unbiased_nested_research_estimate",
            },
            "threshold_selection": prepared.threshold.selection_record,
            "test_probability_summary": {
                "count": len(probabilities),
                "minimum": float(np.min(probabilities)),
                "maximum": float(np.max(probabilities)),
                "mean": float(np.mean(probabilities, dtype=np.float64)),
                "standard_deviation_population": float(
                    np.std(probabilities, dtype=np.float64)
                ),
            },
            "test_predicted_positive_count": positive_count,
            "test_predicted_positive_rate": positive_count / len(probabilities),
            "test_metric_claims": "none",
            "model_count": 1,
            "encoder_count": 1,
            "fit_duration_seconds": fit.duration_seconds,
            "network_calls": False,
        }
        store.save_fit_outputs(fit, predictions, submission, diagnostics)
        verification = verify_reloaded_inference(
            store.root,
            prepared.data,
            fit,
            prepared.threshold,
            predictions,
            submission,
        )
        store.save_inference_verification(verification)

        # All strings and fallible formatting are completed before _SUCCESS.
        summary = [
            f"Final run directory: {store.root}",
            (
                "Operational threshold-selection diagnostic: "
                f"threshold={prepared.threshold.selected_threshold:.3f}, "
                f"balanced_accuracy={prepared.threshold.diagnostic_balanced_accuracy:.12f}"
            ),
            (
                f"Test predictions: positives={positive_count}, "
                f"rate={positive_count / len(probabilities):.6f}"
            ),
            (
                "Submission: "
                f"{store.root / 'submission' / prepared.config.payload['artifacts']['submission_filename']}"
            ),
            "Finalizing validated artifact manifest and terminal success marker.",
        ]
        for line in summary:
            print(line)
        store.complete(
            data=prepared.data,
            fit=fit,
            threshold=prepared.threshold,
            predictions=predictions,
            submission=submission,
        )
        return 0
    except BaseException as error:
        if store is not None and not (store.root / "_SUCCESS").exists():
            try:
                store.fail(error)
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


def main() -> int:
    return execute(
        parse_args(),
        process_started_at_utc=PROCESS_STARTED_AT_UTC,
        entrypoint_module_name="__main__",
    )
