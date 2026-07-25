from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.churn_ml.research_config import (
    ResearchConfigurationError,
    load_research_config,
)
from src.churn_ml.research_protocol import build_candidate_contract_identity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_CONFIG = PROJECT_ROOT / "configs/research/manual_lightgbmprep_r31_nested_rs5x5.yaml"
PLAN_CONFIG = PROJECT_ROOT / "configs/research/plans/telecom_v3_nested_rs5x5_v1.yaml"


def test_research_config_resolves_only_frozen_candidate_sections() -> None:
    config = load_research_config(RUN_CONFIG, project_root=PROJECT_ROOT)

    assert config.plan_id == "telecom_v3_nested_rs5x5_v1"
    assert config.candidate_id == "manual_lightgbmprep_r31_v3"
    assert set(config.candidate_contract) == {
        "candidate_id",
        "implementation",
        "source",
        "features",
        "target_encoder",
        "lightgbm",
    }
    assert "parity" not in config.candidate_contract
    assert "thresholds" not in config.candidate_contract
    assert config.candidate_contract["lightgbm"]["parameters"]["n_estimators"] == 376


@pytest.mark.parametrize(
    ("kind", "section", "key"),
    [
        ("run", "candidate", "unexpected"),
        ("run", "artifacts", "submission_path"),
        ("plan", "threshold_policy", "unexpected"),
        ("plan", "outer_evaluation", "random_state"),
    ],
)
def test_unknown_nested_keys_fail(
    tmp_path: Path,
    kind: str,
    section: str,
    key: str,
) -> None:
    run_payload = yaml.safe_load(RUN_CONFIG.read_text(encoding="utf-8"))
    plan_payload = yaml.safe_load(PLAN_CONFIG.read_text(encoding="utf-8"))
    run_payload["evaluation_plan_path"] = str(tmp_path / "plan.yaml")
    if kind == "run":
        run_payload[section][key] = "forbidden"
    else:
        plan_payload[section][key] = "forbidden"
    (tmp_path / "plan.yaml").write_text(
        yaml.safe_dump(plan_payload, sort_keys=False),
        encoding="utf-8",
    )
    run_path = tmp_path / "run.yaml"
    run_path.write_text(
        yaml.safe_dump(run_payload, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ResearchConfigurationError, match=key):
        load_research_config(run_path, project_root=PROJECT_ROOT)


def test_missing_plan_key_fails(tmp_path: Path) -> None:
    run_payload = yaml.safe_load(RUN_CONFIG.read_text(encoding="utf-8"))
    plan_payload = yaml.safe_load(PLAN_CONFIG.read_text(encoding="utf-8"))
    del plan_payload["threshold_selection"]["random_state"]
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        yaml.safe_dump(plan_payload, sort_keys=False),
        encoding="utf-8",
    )
    run_payload["evaluation_plan_path"] = str(plan_path)
    run_path = tmp_path / "run.yaml"
    run_path.write_text(
        yaml.safe_dump(run_payload, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ResearchConfigurationError, match="random_state"):
        load_research_config(run_path, project_root=PROJECT_ROOT)


def test_candidate_contract_hash_is_stable_and_sensitive() -> None:
    config = load_research_config(RUN_CONFIG, project_root=PROJECT_ROOT)
    identity_kwargs = {
        "feature_schema": {"model_feature_names": ["feature"]},
        "source_provenance": {
            "schema_version": 1,
            "hashing_method": "sha256_raw_working_tree_bytes_v1",
            "files": [{"path": "candidate.py", "sha256": "a" * 64}],
        },
        "runtime_dependencies": {"lightgbm": "4.7.0"},
    }
    canonical, digest = build_candidate_contract_identity(
        config.candidate_contract,
        **identity_kwargs,
    )
    canonical_again, digest_again = build_candidate_contract_identity(
        deepcopy(config.candidate_contract),
        **deepcopy(identity_kwargs),
    )
    changed = deepcopy(config.candidate_contract)
    changed["lightgbm"]["parameters"]["num_leaves"] = 17
    _, changed_digest = build_candidate_contract_identity(changed, **identity_kwargs)

    assert canonical == canonical_again
    assert digest == digest_again
    assert digest != changed_digest
