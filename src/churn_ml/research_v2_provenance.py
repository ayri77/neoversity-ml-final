from __future__ import annotations

import hashlib
import importlib.metadata
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.churn_ml.research_data import canonical_sha256


SOURCE_PATHS = (
    "scripts/run_research_v2.py",
    "src/churn_ml/config.py",
    "src/churn_ml/experiment_v2.py",
    "src/churn_ml/experiment_v2_adapter.py",
    "src/churn_ml/experiment_v2_catboost_adapter.py",
    "src/churn_ml/experiment_v2_contract.py",
    "src/churn_ml/experiment_v2_model_registry.py",
    "src/churn_ml/experiment_v2_numeric_adapter.py",
    "src/churn_ml/experiment_v2_pipeline.py",
    "src/churn_ml/experiment_v2_schema.py",
    "src/churn_ml/experiment_v2_xgboost_adapter.py",
    "src/churn_ml/research_v2_artifact_validation.py",
    "src/churn_ml/research_v2_artifacts.py",
    "src/churn_ml/research_v2_cli.py",
    "src/churn_ml/research_v2_config.py",
    "src/churn_ml/research_v2_data.py",
    "src/churn_ml/research_v2_identity.py",
    "src/churn_ml/research_v2_provenance.py",
    "src/churn_ml/research_evaluation.py",
    "src/churn_ml/research_manual_lightgbm.py",
    "src/churn_ml/research_protocol.py",
    "src/churn_ml/target_encoding.py",
)


def file_identity(
    root: Path, relative_paths: Iterable[str]
) -> tuple[dict[str, Any], str]:
    records = []
    for relative in sorted(set(relative_paths)):
        path = (root / relative).resolve()
        if root.resolve() not in path.parents:
            raise RuntimeError(f"Source path escapes repository: {relative}.")
        records.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    canonical = {
        "schema_version": 2,
        "hashing_method": "sha256_file_bytes_sorted_repository_relative_paths",
        "files": records,
    }
    return canonical, canonical_sha256(canonical)


def loaded_module_identity(root: Path) -> tuple[dict[str, Any], str]:
    records: list[dict[str, str]] = []
    for name, module in sorted(sys.modules.items()):
        if not name.startswith("src.churn_ml"):
            continue
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        path = Path(module_file).resolve()
        if root.resolve() not in path.parents or path.suffix != ".py":
            continue
        relative = path.relative_to(root.resolve()).as_posix()
        records.append(
            {
                "module": name,
                "path": relative,
                "loaded_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    canonical = {
        "schema_version": 2,
        "fresh_process_required": True,
        "modules": records,
    }
    return canonical, canonical_sha256(canonical)


def environment_identity(adapter_id: str | None = None) -> dict[str, str]:
    result = {"python": sys.version.split()[0]}
    for distribution, label in (
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "scikit_learn"),
        ("lightgbm", "lightgbm"),
        ("pyarrow", "pyarrow"),
    ):
        result[label] = importlib.metadata.version(distribution)
    adapter_distributions = {
        "xgboost_numeric_v1": ("xgboost", "xgboost"),
        "catboost_numeric_v1": ("catboost", "catboost"),
    }
    adapter_distribution = (
        None if adapter_id is None else adapter_distributions.get(adapter_id)
    )
    if adapter_distribution is not None:
        distribution, label = adapter_distribution
        result[label] = importlib.metadata.version(distribution)
    return result


def runtime_identity(
    *,
    process_started_at_utc: datetime,
    entry_point: str,
) -> dict[str, Any]:
    return {
        "process_started_at_utc": process_started_at_utc.astimezone(
            timezone.utc
        ).isoformat(),
        "preflight_at_utc": datetime.now(timezone.utc).isoformat(),
        "entry_point": entry_point,
        "fresh_process_required": True,
        "network_access": "disabled_during_execution",
        "competition_assets_accessed": False,
    }
