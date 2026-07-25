from __future__ import annotations

import hashlib
import os
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from src.churn_ml.final_config import FinalConfig
from src.churn_ml.final_data import FinalData
from src.churn_ml.final_promotion import OperationalThreshold, PromotionEvidence
from src.churn_ml.research_protocol import canonical_sha256
from src.churn_ml.research_provenance import build_source_provenance
from src.churn_ml.run_artifacts import collect_git_state


FINAL_SOURCE_PATHS = (
    "configs/experiments/manual_lightgbmprep_r31.yaml",
    "configs/final/manual_lightgbmprep_r31_submission_v1.yaml",
    "configs/research/manual_lightgbmprep_r31_nested_rs5x5.yaml",
    "configs/research/plans/telecom_v3_nested_rs5x5_v1.yaml",
    "docs/reproducibility/baseline_manifest.yaml",
    "scripts/run_final_submission.py",
    "src/churn_ml/config.py",
    "src/churn_ml/final_artifact_validation.py",
    "src/churn_ml/final_artifacts.py",
    "src/churn_ml/final_cli.py",
    "src/churn_ml/final_config.py",
    "src/churn_ml/final_data.py",
    "src/churn_ml/final_manual_lightgbm.py",
    "src/churn_ml/final_promotion.py",
    "src/churn_ml/final_provenance.py",
    "src/churn_ml/manual_lightgbm.py",
    "src/churn_ml/research_artifact_validation.py",
    "src/churn_ml/research_config.py",
    "src/churn_ml/research_data.py",
    "src/churn_ml/research_protocol.py",
    "src/churn_ml/research_provenance.py",
    "src/churn_ml/research_runtime_provenance.py",
    "src/churn_ml/run_artifacts.py",
    "src/churn_ml/target_encoding.py",
)
REQUIRED_DISTRIBUTIONS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit_learn": "scikit-learn",
    "lightgbm": "lightgbm",
    "pyarrow": "pyarrow",
    "pyyaml": "PyYAML",
    "joblib": "joblib",
}


class FinalProvenanceError(RuntimeError):
    """Raised when final executable provenance cannot be proven."""


def collect_final_environment() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for name, distribution in REQUIRED_DISTRIBUTIONS.items():
        try:
            value = version(distribution)
        except PackageNotFoundError as error:
            raise FinalProvenanceError(
                f"Required distribution is unavailable: {distribution}"
            ) from error
        if not value:
            raise FinalProvenanceError(f"Empty version for {distribution}.")
        result[name] = value
    return result


def build_final_source_provenance(
    project_root: Path,
) -> tuple[dict[str, Any], str]:
    return build_source_provenance(project_root, FINAL_SOURCE_PATHS)


def build_final_loaded_module_provenance(
    project_root: Path,
    source_manifest: Mapping[str, Any],
    *,
    entrypoint_module_name: str = "__main__",
    loaded_modules: Mapping[str, ModuleType] | None = None,
) -> tuple[dict[str, Any], str]:
    root = project_root.resolve()
    modules = sys.modules if loaded_modules is None else loaded_modules
    records: list[dict[str, Any]] = []
    observations: list[dict[str, str]] = []
    for source in source_manifest.get("files", []):
        relative = source["path"]
        if relative == "scripts/run_final_submission.py":
            module_name: str | None = entrypoint_module_name
        elif relative.startswith("src/") and relative.endswith(".py"):
            module_name = relative[:-3].replace("/", ".")
        else:
            module_name = None
        if module_name is None:
            continue
        module = modules.get(module_name)
        if module is None:
            raise FinalProvenanceError(
                f"Expected final implementation module is not loaded: {module_name}"
            )
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise FinalProvenanceError(f"Loaded module has no file: {module_name}")
        resolved = Path(module_file).resolve()
        try:
            actual_relative = resolved.relative_to(root).as_posix()
        except ValueError as error:
            raise FinalProvenanceError(
                f"Loaded final module is outside repository: {module_name}"
            ) from error
        if actual_relative != relative:
            raise FinalProvenanceError(
                f"Loaded final module path differs: {module_name} -> {actual_relative}"
            )
        actual_hash = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if actual_hash != source["sha256"]:
            raise FinalProvenanceError(
                f"Loaded final module bytes differ: {module_name}"
            )
        records.append(
            {
                "module_name": module_name,
                "expected_repository_relative_path": relative,
                "actual_repository_relative_path": actual_relative,
                "expected_source_sha256": source["sha256"],
                "loaded_source_sha256": actual_hash,
                "match": True,
            }
        )
        observations.append({"module_name": module_name, "module_file": module_file})
    canonical = {
        "schema_version": 1,
        "hashing_method": "sha256_loaded_module_raw_source_bytes_v1",
        "modules": sorted(records, key=lambda item: item["module_name"]),
    }
    return {
        "canonical": canonical,
        "observations": observations,
    }, canonical_sha256(canonical)


def build_invocation(
    *,
    process_started_at_utc: datetime,
    entry_point: str = "scripts/run_final_submission.py",
) -> dict[str, Any]:
    if (
        process_started_at_utc.tzinfo is None
        or process_started_at_utc.utcoffset()
        != timezone.utc.utcoffset(process_started_at_utc)
    ):
        raise FinalProvenanceError("Process start must be timezone-aware UTC.")
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "sys_argv": list(sys.argv),
        "entry_point": entry_point,
        "current_working_directory": str(Path.cwd().resolve()),
        "process_id": os.getpid(),
        "process_started_at_utc": process_started_at_utc.isoformat(),
    }


def build_promotion_identity(
    config: FinalConfig,
    promotion: PromotionEvidence,
    threshold: OperationalThreshold,
) -> tuple[dict[str, Any], str]:
    approved = config.payload["promotion"]
    research = approved["research"]
    canonical = {
        "schema_version": 1,
        "promotion_id": approved["id"],
        "approval": approved["approval"],
        "approved_research": {
            "run_path": research["run_path"],
            "run_id": research["run_id"],
            "artifact_manifest_sha256": promotion.research_manifest_hash,
            "plan_id": research["plan_id"],
            "plan_sha256": research["plan_sha256"],
            "candidate_id": research["candidate_id"],
            "candidate_sha256": research["candidate_sha256"],
            "candidate_source_sha256": research["candidate_source_sha256"],
            "run_implementation_sha256": research["run_implementation_sha256"],
            "loaded_modules_sha256": research["loaded_modules_sha256"],
            "research_metrics": research["approved_metrics"],
        },
        "threshold_strategy": {
            **config.payload["threshold"],
            "averaged_oos_sha256": threshold.averaged_oos_sha256,
            "selected_threshold": threshold.selected_threshold,
            "diagnostic_balanced_accuracy": threshold.diagnostic_balanced_accuracy,
        },
        "final_fit_policy": config.payload["final_fit"],
        "submission_policy": {
            "sample_submission_sha256": config.payload["inputs"]["sample_submission"][
                "sha256"
            ],
            "columns": ["index", "y"],
            "row_count": 2500,
            "only_target_replaced": True,
            "no_overwrite": True,
            "network_calls": "prohibited",
        },
    }
    return canonical, canonical_sha256(canonical)


def build_final_model_identity(
    config: FinalConfig,
    promotion_hash: str,
    data: FinalData,
    source_hash: str,
    loaded_hash: str,
    environment: Mapping[str, str],
    candidate_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    canonical = {
        "schema_version": 1,
        "promotion_sha256": promotion_hash,
        "data_fingerprints": data.fingerprints,
        "feature_schema": data.feature_schema.to_dict(),
        "submission_schema": data.submission_schema,
        "encoder_contract": candidate_contract["target_encoder"],
        "model_contract": candidate_contract["lightgbm"],
        "final_fit_policy": config.payload["final_fit"],
        "executable_provenance": {
            "source_manifest_sha256": source_hash,
            "loaded_modules_sha256": loaded_hash,
        },
        "environment": dict(sorted(environment.items())),
    }
    return canonical, canonical_sha256(canonical)


def collect_validated_git_state(
    project_root: Path,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    state = collect_git_state(project_root)
    if set(state) != {"branch", "commit", "dirty", "status_porcelain"}:
        raise FinalProvenanceError("Complete Git state is unavailable.")
    implementation_paths = {item["path"] for item in source_manifest["files"]}
    state["untracked_implementation_files"] = sorted(
        line[3:].replace("\\", "/")
        for line in state["status_porcelain"]
        if line.startswith("?? ")
        and line[3:].replace("\\", "/") in implementation_paths
    )
    return state
