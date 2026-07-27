from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.churn_ml.experiment_v2 import get_candidate_adapter
from src.churn_ml.optuna_search_config import OptunaSearchConfig
from src.churn_ml.optuna_search_space import build_resolved_adapter_contract
from src.churn_ml.research_v2_config import load_research_v2_config


class OptunaSearchExportError(RuntimeError):
    """Raised when a fixed production candidate cannot be exported safely."""


def build_best_candidate_config(
    config: OptunaSearchConfig,
    *,
    best_trial_number: int,
    resolved_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a fixed Experiment Core v2 config without search distributions."""
    payload = deepcopy(config.base_config.payload)
    payload["experiment"]["id"] = (
        f"{config.adapter_id}_optuna_{config.search_identity_sha256[:12]}"
        f"_t{best_trial_number}"
    )
    contract = build_resolved_adapter_contract(
        payload["candidate_adapter"]["contract"],
        adapter_id=config.adapter_id,
        tuned_parameters=resolved_parameters,
    )
    get_candidate_adapter(config.adapter_id).validate_contract(contract)
    payload["candidate_adapter"]["contract"] = contract
    payload["search_provenance"] = {
        "schema_version": 1,
        "search_id": config.search_id,
        "study_name": config.payload["study_name"],
        "best_trial_number": best_trial_number,
        "search_identity_sha256": config.search_identity_sha256,
        "search_space_id": config.search_space.search_space_id,
        "search_space_sha256": config.search_space.sha256,
        "evidence_scope": "tuning_only_not_unbiased_final_evidence",
    }
    return payload


def export_best_candidate(
    search_dir: Path,
    output: Path,
    *,
    project_root: Path,
) -> Path:
    from src.churn_ml.optuna_search_artifacts import load_optuna_search_result

    result = load_optuna_search_result(search_dir, project_root=project_root)
    target = output if output.is_absolute() else project_root / output
    target = target.resolve()
    root = project_root.resolve()
    if target == root or root not in target.parents:
        raise OptunaSearchExportError("Export output must be repository-contained.")
    if target.exists():
        raise OptunaSearchExportError("Export output already exists.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                result.best_candidate_config,
                file,
                sort_keys=False,
                allow_unicode=True,
            )
        load_research_v2_config(temporary, project_root=root)
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target
