from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from src.churn_ml.deployment_v1 import execute_deployment
from src.churn_ml.deployment_v1_artifacts import (
    CompletedDeployment,
    load_completed_deployment,
)
from src.churn_ml.deployment_v1_auth import derive_approval_id
from src.churn_ml.deployment_v1_contracts import validate_deployment
from src.churn_ml.experiment_v2 import get_candidate_adapter, get_feature_pipeline
from src.churn_ml.experiment_v2_schema import (
    FeatureSchema,
    ordered_feature_schema_sha256,
)
from src.churn_ml.research_data import canonical_sha256
from src.churn_ml.research_evaluation import run_research_evaluation
from src.churn_ml.research_protocol import (
    build_evaluation_assignments,
    build_evaluation_plan_identity,
)
from src.churn_ml.research_v2_artifacts import ResearchV2ArtifactStore
from src.churn_ml.research_v2_config import load_research_v2_config
from src.churn_ml.research_v2_data import load_research_v2_training_data
from src.churn_ml.research_v2_identity import (
    build_component_identities,
    source_record_mapping,
)
from src.churn_ml.research_v2_provenance import (
    SOURCE_PATHS,
    environment_identity,
    file_identity,
    loaded_module_identity,
)
from src.churn_ml.paired_comparison import load_completed_research_v2_run


PROJECT_ROOT = Path(__file__).absolute().parents[1]
SMOKE_CONFIG = (
    PROJECT_ROOT / "configs/research_v2/manual_lightgbm_te_v1_compat_smoke.yaml"
)
SMOKE_PLAN = PROJECT_ROOT / "configs/research_v2/plans/telecom_v3_smoke_r1x3_t2_v1.yaml"


@dataclass(frozen=True)
class ProductionLoaderFixture:
    root: Path
    output: Path
    config_path: Path
    approval_path: Path
    threshold_path: Path
    research_root: Path
    fixture_root: Path
    loaded: CompletedDeployment

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


class TinyPipeline:
    id = "manual_v3_pipeline_v1_compat"

    def validate_contract(self, contract: Mapping[str, Any]) -> None:
        get_feature_pipeline(self.id).validate_contract(contract)

    def transform(self, features: pd.DataFrame, contract: Mapping[str, Any]) -> Any:
        self.validate_contract(contract)
        columns = features.columns.tolist()
        categorical = features.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        numerical = [name for name in columns if name not in categorical]
        schema = FeatureSchema(
            source_feature_names=columns,
            dropped_features=[],
            model_feature_names=columns,
            categorical_features=categorical,
            numerical_features=numerical,
            transformed_feature_names=numerical
            + [f"{name}__te" for name in categorical],
        )
        from src.churn_ml.experiment_v2 import PipelineOutput

        return PipelineOutput(features=features.copy(), schema=schema)

    def identity_inputs(self, contract: Mapping[str, Any]) -> dict[str, Any]:
        return get_feature_pipeline(self.id).identity_inputs(contract)


class TinyEstimator:
    classes_ = np.array([0, 1])

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> None:
        if len(features) != len(labels):
            raise AssertionError("Tiny estimator inputs are misaligned.")

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        positive = np.linspace(0.25, 0.75, len(features), dtype=np.float64)
        return np.column_stack([1.0 - positive, positive])


def build_production_loader_fixture(
    tmp_name: str, monkeypatch: Any
) -> ProductionLoaderFixture:
    root = PROJECT_ROOT / "artifacts/deployment_v1_loader_tests" / tmp_name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    pipeline = TinyPipeline()
    monkeypatch.setattr(
        "src.churn_ml.research_v2_data.get_feature_pipeline", lambda _: pipeline
    )
    monkeypatch.setattr(
        "src.churn_ml.deployment_v1_features.get_feature_pipeline", lambda _: pipeline
    )

    source_train = pd.DataFrame(
        {
            "index": np.arange(12, dtype=np.int64),
            "number": np.arange(12, dtype=np.float64),
            "category": ["a", "b"] * 6,
        }
    )
    labels = pd.Series([0, 1] * 6, name="y", dtype="int64")
    dataset_dir = root / "data/synthetic_deployment_v1"
    dataset_dir.mkdir(parents=True)
    train_path = dataset_dir / "X_train.parquet"
    target_path = dataset_dir / "y_train.parquet"
    metadata_path = dataset_dir / "metadata.json"
    source_train.to_parquet(train_path, index=False)
    labels.to_frame().to_parquet(target_path, index=False)
    metadata_path.write_text(
        json.dumps({"version": "synthetic_deployment_v1"}), encoding="utf-8"
    )

    plan = yaml.safe_load(SMOKE_PLAN.read_text(encoding="utf-8"))
    plan["plan"]["id"] = "synthetic_deployment_v1_plan"
    dataset = plan["dataset"]
    dataset["processed_dir"] = (
        root.relative_to(PROJECT_ROOT).joinpath("data").as_posix()
    )
    dataset["version"] = "synthetic_deployment_v1"
    for role, path in {
        "train_features": train_path,
        "target": target_path,
        "metadata": metadata_path,
    }.items():
        raw = path.read_bytes()
        dataset["files"][role] = {
            "name": path.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    dataset["expected_rows"] = len(source_train)
    dataset["expected_source_features"] = source_train.shape[1]
    dataset["ordered_source_schema_sha256"] = ordered_feature_schema_sha256(
        source_train.columns.tolist()
    )
    dataset["ordered_dtype_schema_sha256"] = canonical_sha256(
        [
            {"name": name, "dtype": str(dtype)}
            for name, dtype in source_train.dtypes.items()
        ]
    )
    dataset["target"] = {
        "name": "y",
        "dtype": "int64",
        "negative_label": 0,
        "positive_label": 1,
        "expected_negative_rows": 6,
        "expected_positive_rows": 6,
        "values_sha256": canonical_sha256(
            {"name": "y", "dtype": "int64", "values": labels.tolist()}
        ),
    }
    plan_path = root / "contracts/research_plan.yaml"
    _write_yaml(plan_path, plan)

    config_payload = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
    config_payload["experiment"]["id"] = "synthetic_deployment_v1_experiment"
    config_payload["dataset"]["version"] = "synthetic_deployment_v1"
    config_payload["evaluation_plan_path"] = plan_path.relative_to(
        PROJECT_ROOT
    ).as_posix()
    config_payload["artifacts"]["root"] = (
        root.relative_to(PROJECT_ROOT).joinpath("research").as_posix()
    )
    config_path = root / "contracts/research_config.yaml"
    _write_yaml(config_path, config_payload)
    config = load_research_v2_config(config_path, project_root=PROJECT_ROOT)
    training = load_research_v2_training_data(config)
    assignments = build_evaluation_assignments(training.y, config.plan_payload)
    plan_identity, plan_hash = build_evaluation_plan_identity(
        config.plan_payload, training.fingerprints, assignments
    )
    relative_config = config_path.relative_to(PROJECT_ROOT).as_posix()
    relative_plan = plan_path.relative_to(PROJECT_ROOT).as_posix()
    source_identity, source_hash = file_identity(
        PROJECT_ROOT, (*SOURCE_PATHS, relative_config, relative_plan)
    )
    environment = environment_identity(config.adapter_id)
    adapter = get_candidate_adapter(config.adapter_id)
    component_identities, component_hashes = build_component_identities(
        pipeline_inputs=get_feature_pipeline(config.pipeline_id).identity_inputs(
            config.pipeline_contract
        ),
        adapter_inputs=adapter.identity_inputs(config.adapter_contract),
        resolved_feature_schema=training.pipeline_output.schema.to_dict(),
        runtime_dependencies=environment,
        dataset_version=config.dataset_version,
        source_records=source_record_mapping(source_identity),
    )
    loaded_modules, loaded_modules_hash = loaded_module_identity(PROJECT_ROOT)
    hashes = {
        "plan": plan_hash,
        **component_hashes,
        "source": source_hash,
        "loaded_modules": loaded_modules_hash,
    }
    identities = {
        "evaluation_plan": {"sha256": plan_hash, "canonical": plan_identity},
        **component_identities,
        "source": {"sha256": source_hash, "canonical": source_identity},
        "loaded_modules": {
            "sha256": loaded_modules_hash,
            "canonical": loaded_modules,
        },
    }
    store = ResearchV2ArtifactStore(
        config,
        plan_hash=plan_hash,
        candidate_hash=hashes["candidate"],
        expected_hashes=hashes,
        run_id="synthetic_deployment_v1_run",
    )
    metadata = {
        "schema_version": 2,
        "experiment_id": config.experiment_id,
        "plan_id": config.plan_id,
        "feature_pipeline_id": config.pipeline_id,
        "candidate_adapter_id": config.adapter_id,
        "hashes": hashes,
        "status": "running",
        "started_at_utc": store.started_at.isoformat(),
        "environment": environment,
        "runtime": {
            "process_started_at_utc": store.started_at.isoformat(),
            "preflight_at_utc": store.started_at.isoformat(),
            "entry_point": "tests/deployment_v1_loader_support.py",
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
    }
    store.save_initial(
        metadata=metadata,
        identities=identities,
        fingerprints=training.fingerprints,
        feature_schema=training.pipeline_output.schema.to_dict(),
        assignments=assignments,
    )

    def fit_predict(
        train_features: pd.DataFrame,
        train_labels: pd.Series,
        prediction_features: pd.DataFrame,
        contract: dict[str, Any],
        *,
        model_training_positions: np.ndarray,
        prediction_positions: np.ndarray,
        audit_callback: Any = None,
    ) -> np.ndarray:
        del train_features, train_labels, prediction_features, contract
        if audit_callback is not None:
            audit_callback(model_training_positions, prediction_positions)
        positions = np.asarray(prediction_positions, dtype=np.int64)
        return 0.1 + (positions % 7).astype(np.float64) / 10.0

    result = run_research_evaluation(
        training.X,
        training.y,
        assignments,
        config.plan_payload,
        config.adapter_contract,
        fit_predict=fit_predict,
        on_outer_fold_complete=store.save_fold,
    )
    store.save_result(result)
    store.complete(metadata, result)
    research = load_completed_research_v2_run(store.root, project_root=PROJECT_ROOT)

    threshold_artifact = {
        "schema_version": 1,
        "source_type": "experiment_core_v2_manual_threshold_v1",
        "source_run_id": research.metadata["run_id"],
        "source_manifest_sha256": research.manifest["manifest_sha256"],
        "threshold": 0.5,
        "threshold_policy_id": "fixed_manual_threshold_v1",
        "plan_sha256": identities["evaluation_plan"]["sha256"],
        "candidate_sha256": identities["candidate"]["sha256"],
    }
    threshold_path = root / "evidence/threshold.yaml"
    _write_yaml(threshold_path, threshold_artifact)
    threshold_raw = threshold_path.read_bytes()
    threshold_reference = {
        **threshold_artifact,
        "path": threshold_path.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": len(threshold_raw),
        "sha256": hashlib.sha256(threshold_raw).hexdigest(),
    }
    reference = {
        "path": store.root.relative_to(PROJECT_ROOT).as_posix(),
        "run_id": research.metadata["run_id"],
        "manifest_sha256": research.manifest["manifest_sha256"],
        "plan_id": research.config.plan_id,
        "plan_sha256": identities["evaluation_plan"]["sha256"],
        "dataset_identity_sha256": canonical_sha256(research.dataset_fingerprints),
        "pipeline_id": research.config.pipeline_id,
        "pipeline_sha256": identities["feature_pipeline"]["sha256"],
        "adapter_id": research.config.adapter_id,
        "adapter_sha256": identities["candidate_adapter"]["sha256"],
        "candidate_sha256": identities["candidate"]["sha256"],
    }
    parameters = deepcopy(config.adapter_contract["lightgbm"]["parameters"])
    approval = {
        "schema_version": 1,
        "approval_id": "0" * 64,
        "component_id": "manual-component",
        "component_name": "Synthetic authenticated component",
        "research_run": reference,
        "fixed_resolved_model_parameters": parameters,
        "threshold_evidence": threshold_reference,
        "paired_comparison": {
            "reference": None,
            "manifest_sha256": None,
            "exception": {
                "granted": True,
                "reason": "Synthetic production-loader fixture uses an explicit exception.",
            },
        },
        "manual_approval": {
            "status": "approved",
            "approver": "production-loader-test",
            "approved_at_utc": "2026-01-01T00:00:00+00:00",
        },
        "intended_deployment_role": "synthetic-test-only",
    }
    approval["approval_id"] = derive_approval_id(approval)
    approval_path = root / "approvals/manual.yaml"
    _write_yaml(approval_path, approval)

    fixture_root = root / "fixture"
    fixture_root.mkdir()
    test = pd.DataFrame(
        {
            "index": pd.Series([100, 101, 102], dtype="int64"),
            "number": pd.Series([12.0, 13.0, np.nan], dtype="float64"),
            "category": pd.Series(["a", "unknown", None], dtype="object"),
        }
    )
    sample = pd.DataFrame(
        {"index": pd.Series([100, 101, 102], dtype="int64"), "y": [0, 0, 0]}
    )
    source_train.to_parquet(fixture_root / "X_train.parquet", index=False)
    labels.to_frame().to_parquet(fixture_root / "y_train.parquet", index=False)
    test.to_parquet(fixture_root / "X_test.parquet", index=False)
    sample.to_csv(
        fixture_root / "sample_submission.csv", index=False, lineterminator="\n"
    )
    fixture_files = {}
    for role, name in {
        "train": "X_train.parquet",
        "labels": "y_train.parquet",
        "test": "X_test.parquet",
        "sample_submission": "sample_submission.csv",
    }.items():
        raw = (fixture_root / name).read_bytes()
        fixture_files[role] = {
            "path": name,
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    fixture_manifest = {
        "schema_version": 1,
        "fixture_id": "0" * 64,
        "fixture_type": "synthetic_noncompetition",
        "files": fixture_files,
        "generation": {"generator_id": "production_loader_fixture_v1", "seed": 42},
    }
    fixture_manifest["fixture_id"] = canonical_sha256(
        {key: value for key, value in fixture_manifest.items() if key != "fixture_id"}
    )
    _write_yaml(fixture_root / "fixture_manifest.yaml", fixture_manifest)

    deployment = {
        "schema_version": 1,
        "deployment_id": "synthetic-loader-deployment",
        "dataset_version": "synthetic_deployment_v1",
        "pipeline_id": "manual_v3_pipeline_v1_compat",
        "components": [
            {
                "component_id": "manual-component",
                "approval_artifact_path": approval_path.relative_to(
                    PROJECT_ROOT
                ).as_posix(),
                "adapter_id": "manual_lightgbm_te_v1_compat",
                "fixed_parameters": parameters,
                "bag_seeds": [0],
                "component_weight": 1.0,
            }
        ],
        "blend": {"method": "fixed_weighted_mean", "weight_sum_tolerance": 1e-12},
        "threshold": {
            "value": 0.5,
            "evidence": threshold_reference,
            "comparison": "greater_than_or_equal",
        },
        "bagging": {
            "method": "full_data_seed_bagging",
            "aggregation": "arithmetic_mean",
            "model_persistence": False,
        },
        "test_data": {
            "path": "competition/never-read.parquet",
            "sha256": "0" * 64,
            "expected_rows": 3,
            "ordered_schema_sha256": "0" * 64,
        },
        "sample_submission": {
            "path": "competition/never-read.csv",
            "sha256": "1" * 64,
            "expected_rows": 3,
            "id_column": "index",
            "target_column": "y",
        },
        "output": {
            "root": "artifacts/deployments",
            "submission_filename": "submission.csv",
        },
        "runtime": {"tracking_enabled": False, "network_enabled": False},
    }
    deployment_path = root / "deployment.yaml"
    _write_yaml(deployment_path, deployment)
    validated = validate_deployment(deployment_path, project_root=PROJECT_ROOT)
    output = root / "completed"
    execute_deployment(
        validated,
        mode="dry-run",
        output_dir=output,
        fixture_dir=fixture_root,
        estimator_factory=lambda *_: TinyEstimator(),
    )
    loaded = load_completed_deployment(output, project_root=PROJECT_ROOT)
    return ProductionLoaderFixture(
        root=root,
        output=output,
        config_path=deployment_path,
        approval_path=approval_path,
        threshold_path=threshold_path,
        research_root=store.root,
        fixture_root=fixture_root,
        loaded=loaded,
    )


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False),
        encoding="utf-8",
    )
