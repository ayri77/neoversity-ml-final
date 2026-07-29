from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.churn_ml.blend_evaluation_v1 import (
    BLEND_EVALUATION_SCHEMA_VERSION,
    BlendEvaluationError,
    BlendEvaluationResult,
    blend_candidate_identity,
)
from src.churn_ml.blend_evaluation_v1_config import BlendEvaluationConfig
from src.churn_ml.paired_comparison_paths import (
    PairedPathSafetyError,
    assert_pairwise_disjoint_paths,
    resolve_repository_path,
)
from src.churn_ml.research_data import canonical_sha256


SAFE_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
BLEND_SOURCE_PATHS = (
    "scripts/run_blend_evaluation_v1.py",
    "src/churn_ml/blend_evaluation_v1.py",
    "src/churn_ml/blend_evaluation_v1_artifacts.py",
    "src/churn_ml/blend_evaluation_v1_cli.py",
    "src/churn_ml/blend_evaluation_v1_config.py",
    "src/churn_ml/blend_evaluation_v1_deployment.py",
)
PRIMARY_ARTIFACTS = {
    "resolved_config.yaml",
    "compatibility_report.json",
    "lightgbm_run_reference.json",
    "xgboost_run_reference.json",
    "fold_selections.csv",
    "held_out_predictions.parquet",
    "fold_metrics.csv",
    "repeat_metrics.csv",
    "aggregate_summary.json",
    "deployment_parameters.json",
    "sensitivity_table.csv",
}
ARTIFACT_DIRECTORIES = {"provenance"}
PROVENANCE_ARTIFACT = "provenance/source_provenance.json"


class BlendEvaluationArtifactError(RuntimeError):
    """Raised when blend evaluation artifacts cannot satisfy their lifecycle."""


def create_blend_evaluation_artifacts(
    *,
    config: BlendEvaluationConfig,
    result: BlendEvaluationResult,
    evaluation_id: str | None = None,
) -> Path:
    slug = evaluation_id or config.evaluation_id
    if not SAFE_SLUG.match(slug):
        raise BlendEvaluationArtifactError(
            "evaluation_id must match ^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$."
        )
    output_root = config.artifact_root
    evaluation_root = (output_root / slug).resolve()
    if evaluation_root.exists():
        raise BlendEvaluationArtifactError(
            f"Blend evaluation directory already exists: {slug}."
        )
    try:
        assert_pairwise_disjoint_paths(
            {
                "lightgbm_run": (
                    config.project_root
                    / result.lightgbm_reference.repository_relative_path
                ).resolve(),
                "xgboost_run": (
                    config.project_root
                    / result.xgboost_reference.repository_relative_path
                ).resolve(),
                "output_root": output_root.resolve(),
            }
        )
    except PairedPathSafetyError as error:
        raise BlendEvaluationArtifactError(str(error)) from error

    evaluation_root.mkdir(parents=True, exist_ok=False)
    (evaluation_root / "provenance").mkdir()

    try:
        _write_yaml(evaluation_root / "resolved_config.yaml", config.resolved_payload())
        _write_json(
            evaluation_root / "compatibility_report.json",
            result.compatibility.to_dict(),
        )
        _write_json(
            evaluation_root / "lightgbm_run_reference.json",
            result.lightgbm_reference.to_dict(),
        )
        _write_json(
            evaluation_root / "xgboost_run_reference.json",
            result.xgboost_reference.to_dict(),
        )
        result.fold_selections.to_csv(
            evaluation_root / "fold_selections.csv", index=False
        )
        result.held_out_predictions.to_parquet(
            evaluation_root / "held_out_predictions.parquet", index=False
        )
        result.fold_metrics.to_csv(evaluation_root / "fold_metrics.csv", index=False)
        result.repeat_metrics.to_csv(
            evaluation_root / "repeat_metrics.csv", index=False
        )
        _write_json(evaluation_root / "aggregate_summary.json", result.summary)
        _write_json(
            evaluation_root / "deployment_parameters.json",
            result.deployment_parameters,
        )
        result.sensitivity_table.to_csv(
            evaluation_root / "sensitivity_table.csv", index=False
        )
        _write_json(
            evaluation_root / PROVENANCE_ARTIFACT,
            blend_source_provenance(config.project_root),
        )
        inventory = build_inventory(evaluation_root)
        _write_json(evaluation_root / "artifact_inventory.json", inventory)
        manifest = build_manifest(evaluation_root)
        _write_json(evaluation_root / "manifest.json", manifest)
        (evaluation_root / "_SUCCESS").write_text(
            json.dumps(
                {
                    "schema_version": BLEND_EVALUATION_SCHEMA_VERSION,
                    "status": "completed",
                    "evaluation_id": slug,
                    "finished_at_utc": _utc_now(),
                    "manifest_sha256": manifest["manifest_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception as error:
        _write_json(
            evaluation_root / "_FAILED",
            {
                "schema_version": BLEND_EVALUATION_SCHEMA_VERSION,
                "status": "failed",
                "error_type": type(error).__name__,
                "message": str(error),
                "failed_at_utc": _utc_now(),
            },
        )
        raise
    return evaluation_root


def build_threshold_evidence_draft(
    *,
    config: BlendEvaluationConfig,
    result: BlendEvaluationResult,
    evaluation_id: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    del config
    weight = float(result.deployment_parameters["deployment_lightgbm_weight"])
    threshold = float(result.deployment_parameters["deployment_threshold"])
    candidate = blend_candidate_identity(
        result.lightgbm_reference.candidate_sha256,
        result.xgboost_reference.candidate_sha256,
        deployment_lightgbm_weight=weight,
    )
    return {
        "schema_version": 1,
        "source_type": "blend_evaluation_v1_cross_fit_threshold_v1",
        "source_run_id": evaluation_id,
        "source_manifest_sha256": source_manifest_sha256,
        "threshold": threshold,
        "threshold_policy_id": str(result.summary["threshold_policy_id"]),
        "plan_sha256": result.lightgbm_reference.evaluation_plan_sha256,
        "candidate_sha256": candidate,
    }


def blend_source_provenance(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    records: list[dict[str, Any]] = []
    for relative in BLEND_SOURCE_PATHS:
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise BlendEvaluationArtifactError(
                f"Blend source is missing or outside the repository: {relative}."
            )
        raw = path.read_bytes()
        records.append(
            {
                "path": relative,
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return {
        "schema_version": BLEND_EVALUATION_SCHEMA_VERSION,
        "records": records,
        "records_sha256": canonical_sha256(records),
    }


def build_inventory(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {
            "artifact_inventory.json",
            "manifest.json",
            "_SUCCESS",
            "_FAILED",
        }:
            continue
        raw = path.read_bytes()
        files.append(
            {
                "path": relative,
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return {
        "schema_version": BLEND_EVALUATION_SCHEMA_VERSION,
        "files": files,
        "inventory_sha256": canonical_sha256(files),
    }


def build_manifest(root: Path) -> dict[str, Any]:
    inventory = build_inventory(root)
    payload = {
        "schema_version": BLEND_EVALUATION_SCHEMA_VERSION,
        "inventory_sha256": inventory["inventory_sha256"],
        "files": inventory["files"],
    }
    return {
        **payload,
        "manifest_sha256": canonical_sha256(payload),
    }


def load_completed_blend_evaluation(
    evaluation_dir: Path,
    *,
    project_root: Path,
) -> Path:
    try:
        root = resolve_repository_path(
            evaluation_dir,
            project_root=project_root,
            role="blend_evaluation",
            field_path="paths.evaluation_dir",
            must_exist=True,
            require_directory=True,
        )
    except PairedPathSafetyError as error:
        raise BlendEvaluationError(
            error.detail,
            reason_code=error.reason_code,
            field_path=error.field_path,
        ) from error
    if not (root / "_SUCCESS").is_file():
        raise BlendEvaluationError(
            "Blend evaluation is not marked successful.",
            reason_code="BLEND_EVALUATION_INCOMPLETE",
            field_path="paths.evaluation_dir",
        )
    if (root / "_FAILED").exists():
        raise BlendEvaluationError(
            "Blend evaluation contains _FAILED.",
            reason_code="BLEND_EVALUATION_FAILED",
            field_path="paths.evaluation_dir",
        )
    return root


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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise BlendEvaluationArtifactError(f"JSON root must be a mapping: {path}")
    return dict(payload)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
