from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.churn_ml.research_evaluation import (
    ResearchEvaluationResult,
    run_research_evaluation,
)
from src.churn_ml.research_v2_artifacts import ResearchV2ArtifactStore
from src.churn_ml.research_v2_cli import ResearchV2Preflight, preflight


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = (
    PROJECT_ROOT / "configs/research_v2/manual_lightgbm_te_v1_compat_smoke.yaml"
)


class DeterministicV2TestAdapter:
    id: str = "deterministic_v2_test_adapter"

    def validate_contract(self, contract: Mapping[str, Any]) -> None:
        del contract

    def fit_predict(
        self,
        train_features: pd.DataFrame,
        train_labels: pd.Series,
        prediction_features: pd.DataFrame,
        contract: Mapping[str, Any],
        *,
        model_training_positions: np.ndarray,
        prediction_positions: np.ndarray,
        audit_callback: Any = None,
    ) -> np.ndarray:
        del train_features, train_labels, prediction_features, contract
        if audit_callback is not None:
            audit_callback(model_training_positions, prediction_positions)
        positions = np.asarray(prediction_positions, dtype=np.int64)
        return 0.05 + (positions % 97).astype(float) / 107.0

    def identity_inputs(self, contract: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "id": self.id,
            "contract": deepcopy(dict(contract)),
            "probability_semantics": "binary_positive_class_label_1",
        }


def build_persisted_v2_test_run(
    tmp_path: Path,
) -> tuple[
    ResearchV2ArtifactStore,
    ResearchEvaluationResult,
    dict[str, Any],
    ResearchV2Preflight,
]:
    prepared = build_v2_test_preflight(tmp_path)
    store = ResearchV2ArtifactStore(
        prepared.config,
        plan_hash=prepared.hashes["plan"],
        candidate_hash=prepared.hashes["candidate"],
        expected_hashes=prepared.hashes,
        run_id="unit-run",
    )
    metadata = build_v2_metadata(prepared, store)
    store.save_initial(
        metadata=metadata,
        identities=prepared.identities,
        fingerprints=prepared.data.fingerprints,
        feature_schema=prepared.data.pipeline_output.schema.to_dict(),
        assignments=prepared.assignments,
        dataset_provenance=prepared.data.dataset_provenance,
    )
    adapter = DeterministicV2TestAdapter()

    def fit_predict(
        train_features: pd.DataFrame,
        train_labels: pd.Series,
        prediction_features: pd.DataFrame,
        contract: dict[str, Any],
        **kwargs: Any,
    ) -> np.ndarray:
        return adapter.fit_predict(
            train_features,
            train_labels,
            prediction_features,
            contract,
            **kwargs,
        )

    result = run_research_evaluation(
        prepared.data.X,
        prepared.data.y,
        prepared.assignments,
        prepared.config.plan_payload,
        prepared.config.adapter_contract,
        fit_predict=fit_predict,
        on_outer_fold_complete=store.save_fold,
    )
    store.save_result(result)
    return store, result, metadata, prepared


def build_v2_test_preflight(tmp_path: Path) -> ResearchV2Preflight:
    prepared = preflight(SMOKE_CONFIG)
    payload = deepcopy(prepared.config.payload)
    artifact_root = PROJECT_ROOT / "artifacts/research_v2_tests" / tmp_path.name
    payload["artifacts"]["root"] = artifact_root.relative_to(PROJECT_ROOT).as_posix()
    config = type(prepared.config)(
        payload=payload,
        plan_payload=deepcopy(prepared.config.plan_payload),
        source_path=prepared.config.source_path,
        plan_path=prepared.config.plan_path,
        project_root=prepared.config.project_root,
    )
    return ResearchV2Preflight(
        config=config,
        data=prepared.data,
        assignments=prepared.assignments,
        adapter=DeterministicV2TestAdapter(),
        identities=prepared.identities,
        hashes=prepared.hashes,
        environment=prepared.environment,
    )


def build_v2_metadata(
    prepared: ResearchV2Preflight,
    store: ResearchV2ArtifactStore,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "experiment_id": prepared.config.experiment_id,
        "plan_id": prepared.config.plan_id,
        "feature_pipeline_id": prepared.config.pipeline_id,
        "candidate_adapter_id": prepared.config.adapter_id,
        "hashes": prepared.hashes,
        "status": "running",
        "started_at_utc": store.started_at.isoformat(),
        "environment": prepared.environment,
        "runtime": {
            "process_started_at_utc": store.started_at.isoformat(),
            "preflight_at_utc": store.started_at.isoformat(),
            "entry_point": "tests/research_v2_test_support.py",
            "fresh_process_required": True,
            "network_access": "disabled_during_execution",
            "competition_assets_accessed": False,
        },
        "git": {
            "branch": "test",
            "commit": "0" * 40,
            "dirty": True,
            "status_porcelain": [],
        },
        "competition_assets_accessed": False,
        "tracking_enabled": False,
        "run_id": store.run_id,
        "dataset_provenance": prepared.data.dataset_provenance,
    }
