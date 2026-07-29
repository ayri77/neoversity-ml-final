from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from src.churn_ml.blend_evaluation_v1 import (
    BlendEvaluationError,
    apply_fixed_blend_probabilities,
    blend_candidate_identity,
    clip_weight,
)
from src.churn_ml.blend_evaluation_v1_artifacts import (
    BlendEvaluationArtifactError,
    load_completed_blend_evaluation,
)
from src.churn_ml.deployment_v1_models import classify_fixed
from src.churn_ml.paired_comparison_paths import resolve_repository_path


@dataclass(frozen=True)
class SubmissionVariantSpec:
    id: str
    lightgbm_weight_delta: float
    leaderboard_probe: bool


VARIANT_SPECS = (
    SubmissionVariantSpec(
        id="blend_robust",
        lightgbm_weight_delta=0.0,
        leaderboard_probe=False,
    ),
    SubmissionVariantSpec(
        id="blend_lgbm_plus",
        lightgbm_weight_delta=0.05,
        leaderboard_probe=True,
    ),
    SubmissionVariantSpec(
        id="blend_xgb_plus",
        lightgbm_weight_delta=-0.05,
        leaderboard_probe=True,
    ),
)


def prepare_deployment_package(
    evaluation_dir: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    root = load_completed_blend_evaluation(
        evaluation_dir,
        project_root=project_root,
    )
    parameters = _read_json(root / "deployment_parameters.json")
    resolved = _read_yaml(root / "resolved_config.yaml")
    lightgbm_ref = _read_json(root / "lightgbm_run_reference.json")
    xgboost_ref = _read_json(root / "xgboost_run_reference.json")
    summary = _read_json(root / "aggregate_summary.json")
    manifest = _read_json(root / "manifest.json")
    evaluation_id = root.name

    package_root = (
        project_root / "artifacts" / "blend_deployment_packages" / evaluation_id
    ).resolve()
    if package_root.exists():
        raise BlendEvaluationArtifactError(
            f"Deployment package already exists: {package_root}."
        )
    package_root.mkdir(parents=True, exist_ok=False)

    weight = float(parameters["deployment_lightgbm_weight"])
    evidence = {
        "schema_version": 1,
        "source_type": "blend_evaluation_v1_cross_fit_threshold_v1",
        "source_run_id": evaluation_id,
        "source_manifest_sha256": str(manifest["manifest_sha256"]),
        "threshold": float(parameters["deployment_threshold"]),
        "threshold_policy_id": str(summary["threshold_policy_id"]),
        "plan_sha256": lightgbm_ref["evaluation_plan_sha256"],
        "candidate_sha256": blend_candidate_identity(
            lightgbm_ref["candidate_sha256"],
            xgboost_ref["candidate_sha256"],
            deployment_lightgbm_weight=weight,
        ),
    }
    evidence_path = package_root / "threshold_evidence.yaml"
    _write_yaml(evidence_path, evidence)
    evidence_raw = evidence_path.read_bytes()
    evidence_reference = {
        "schema_version": 1,
        "path": evidence_path.relative_to(project_root).as_posix(),
        "size_bytes": len(evidence_raw),
        "sha256": hashlib.sha256(evidence_raw).hexdigest(),
        "source_type": evidence["source_type"],
        "source_run_id": evidence["source_run_id"],
        "source_manifest_sha256": evidence["source_manifest_sha256"],
        "threshold": evidence["threshold"],
        "threshold_policy_id": evidence["threshold_policy_id"],
        "plan_sha256": evidence["plan_sha256"],
        "candidate_sha256": evidence["candidate_sha256"],
    }
    _write_yaml(package_root / "threshold_evidence_reference.yaml", evidence_reference)

    xgboost_weight = 1.0 - weight
    deployment_draft = {
        "schema_version": 1,
        "deployment_id": f"{resolved['experiment']['id']}_deployment",
        "dataset_version": "REPLACE_WITH_APPROVED_DATASET_VERSION",
        "pipeline_id": "REPLACE_WITH_APPROVED_PIPELINE_ID",
        "components": [
            {
                "component_id": "manual_lightgbm_te_v1_compat",
                "approval_artifact_path": (
                    "configs/deployment_v1/approvals/REPLACE_lightgbm_approval.yaml"
                ),
                "adapter_id": "manual_lightgbm_te_v1_compat",
                "fixed_parameters": {},
                "bag_seeds": [0],
                "component_weight": weight,
            },
            {
                "component_id": "xgboost_numeric_v1",
                "approval_artifact_path": (
                    "configs/deployment_v1/approvals/REPLACE_xgboost_approval.yaml"
                ),
                "adapter_id": "xgboost_numeric_v1",
                "fixed_parameters": {},
                "bag_seeds": [0],
                "component_weight": xgboost_weight,
            },
        ],
        "blend": {
            "method": "fixed_weighted_mean",
            "weight_sum_tolerance": 1.0e-12,
        },
        "threshold": {
            "value": float(parameters["deployment_threshold"]),
            "evidence": evidence_reference,
            "comparison": "greater_than_or_equal",
        },
        "bagging": {
            "method": "full_data_seed_bagging",
            "aggregation": "arithmetic_mean",
            "model_persistence": False,
        },
        "test_data": {
            "path": "data/competition/REPLACE_test.parquet",
            "sha256": "0" * 64,
            "expected_rows": 1,
            "ordered_schema_sha256": "0" * 64,
        },
        "sample_submission": {
            "path": "data/competition/REPLACE_sample.csv",
            "sha256": "0" * 64,
            "expected_rows": 1,
            "id_column": "index",
            "target_column": "y",
        },
        "output": {
            "root": "artifacts/deployments",
            "submission_filename": "submission.csv",
        },
        "runtime": {
            "tracking_enabled": False,
            "network_enabled": False,
        },
    }
    _write_yaml(package_root / "deployment_config.draft.yaml", deployment_draft)
    notes = {
        "schema_version": 1,
        "status": "draft_only",
        "requires_manual_candidate_approvals": True,
        "automatic_submission": False,
        "deployment_lightgbm_weight": weight,
        "deployment_threshold": float(parameters["deployment_threshold"]),
        "lightgbm_run": lightgbm_ref,
        "xgboost_run": xgboost_ref,
        "submission_variants": [asdict(spec) for spec in VARIANT_SPECS],
        "instructions": [
            "Author approved candidate approval YAMLs for both components.",
            "Point both approvals at the shared blend threshold evidence.",
            "Fill competition test/sample hashes and row counts.",
            "Validate and dry-run Final Deployment v1 before any competition run.",
            "After one successful deployment, generate submission variants without retraining.",
        ],
    }
    _write_json(package_root / "prepare_deployment_notes.json", notes)
    return {
        "evaluation_root": root.relative_to(project_root).as_posix(),
        "deployment_package_dir": package_root.relative_to(project_root).as_posix(),
        "threshold_evidence_path": evidence_path.relative_to(project_root).as_posix(),
        "deployment_lightgbm_weight": weight,
        "deployment_threshold": float(parameters["deployment_threshold"]),
    }


def generate_submission_variants(
    deployment_dir: Path,
    *,
    project_root: Path,
    output_dir: Path | None = None,
) -> Path:
    deployment_root = resolve_repository_path(
        deployment_dir,
        project_root=project_root,
        role="completed_deployment",
        field_path="paths.deployment_dir",
        must_exist=True,
        require_directory=True,
    )
    if not (deployment_root / "_SUCCESS").is_file():
        raise BlendEvaluationError(
            "Deployment must be completed before variant generation.",
            reason_code="DEPLOYMENT_INCOMPLETE",
            field_path="paths.deployment_dir",
        )
    component_frame = pd.read_parquet(
        deployment_root / "component_probabilities.parquet"
    )
    prediction_summary = _read_json(deployment_root / "prediction_summary.json")
    resolved = _read_yaml(deployment_root / "resolved_deployment_config.yaml")
    sample_config = resolved["sample_submission"]
    sample = pd.read_csv(project_root / sample_config["path"])
    lightgbm_id, xgboost_id = _component_ids(resolved["components"])
    base_weight = float(
        next(
            item["component_weight"]
            for item in resolved["components"]
            if item["component_id"] == lightgbm_id
        )
    )
    threshold = float(prediction_summary["threshold"])
    if output_dir is None:
        variants_root = (
            project_root / "artifacts" / "blend_submissions" / resolved["deployment_id"]
        )
    else:
        variants_root = resolve_repository_path(
            output_dir,
            project_root=project_root,
            role="blend_submission_variants",
            field_path="paths.output_dir",
            must_exist=False,
            require_directory=True,
        )
    if variants_root.exists():
        raise BlendEvaluationArtifactError(
            f"Variant output directory already exists: {variants_root}."
        )
    variants_root.mkdir(parents=True, exist_ok=False)

    lightgbm_probability = component_frame[lightgbm_id].to_numpy(dtype=float)
    xgboost_probability = component_frame[xgboost_id].to_numpy(dtype=float)
    records: list[dict[str, Any]] = []
    for spec in VARIANT_SPECS:
        weight = clip_weight(base_weight + float(spec.lightgbm_weight_delta))
        blend = apply_fixed_blend_probabilities(
            lightgbm_probability,
            xgboost_probability,
            lightgbm_weight=weight,
        )
        labels = classify_fixed(blend, threshold)
        submission = sample[[sample_config["id_column"]]].copy()
        submission[sample_config["target_column"]] = labels.astype(np.int8)
        filename = f"{spec.id}.csv"
        path = variants_root / filename
        submission.to_csv(path, index=False)
        raw = path.read_bytes()
        records.append(
            {
                "id": spec.id,
                "filename": filename,
                "lightgbm_weight": weight,
                "lightgbm_weight_delta": float(spec.lightgbm_weight_delta),
                "threshold": threshold,
                "leaderboard_probe": bool(spec.leaderboard_probe),
                "positive_count": int(labels.sum()),
                "positive_rate": float(labels.mean()),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "retrained": False,
            }
        )
    manifest = {
        "schema_version": 1,
        "deployment_id": resolved["deployment_id"],
        "source_deployment": deployment_root.relative_to(project_root).as_posix(),
        "source_component_probabilities": "component_probabilities.parquet",
        "variants": records,
        "retrained": False,
    }
    (variants_root / "variants_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return variants_root


def _component_ids(components: list[Mapping[str, Any]]) -> tuple[str, str]:
    lightgbm_ids = [
        str(item["component_id"])
        for item in components
        if item["adapter_id"] == "manual_lightgbm_te_v1_compat"
    ]
    xgboost_ids = [
        str(item["component_id"])
        for item in components
        if item["adapter_id"] == "xgboost_numeric_v1"
    ]
    if len(lightgbm_ids) != 1 or len(xgboost_ids) != 1:
        raise BlendEvaluationError(
            "Variant generation requires exactly one LightGBM and one XGBoost component.",
            reason_code="BLEND_COMPONENT_SHAPE_INVALID",
            field_path="deployment.components",
        )
    return lightgbm_ids[0], xgboost_ids[0]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise BlendEvaluationArtifactError(f"JSON root must be a mapping: {path}")
    return dict(payload)


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise BlendEvaluationArtifactError(f"YAML root must be a mapping: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
