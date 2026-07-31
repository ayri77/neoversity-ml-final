"""Unit tests for dataset-driven Experiment Core materialization."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.churn_ml.control_panel.dataset_experiment_materializer import (
    REGISTERED_PREPARED_PASSTHROUGH_CONTRACT,
    REGISTERED_PREPARED_PASSTHROUGH_V1,
    DatasetExperimentMaterializerError,
    discover_registry_summaries,
    list_base_templates,
    prepare_dataset_driven_experiment,
    selection_fingerprint,
)
from src.churn_ml.dataset_registry import discover_registered_datasets
from src.churn_ml.research_v2_config import load_research_v2_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
LIGHTGBM_SMOKE = "configs/research_v2/manual_lightgbm_te_v1_compat_smoke.yaml"
EDITABLE_ROOT = "artifacts/ui_configs_materializer_tests"


def _has_v7() -> bool:
    return (
        PROCESSED_ROOT / "v7_compact_zero_indicators" / "dataset_manifest.json"
    ).is_file()


def _clean_editable_root() -> Path:
    root = PROJECT_ROOT / EDITABLE_ROOT
    if root.exists():
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        if root.exists():
            root.rmdir()
    root.mkdir(parents=True, exist_ok=True)
    return root


pytestmark = pytest.mark.skipif(
    not PROCESSED_ROOT.is_dir(),
    reason="local data/processed Registry is required",
)


def test_discover_registry_summaries_dynamic_sorted() -> None:
    summaries = discover_registry_summaries(PROJECT_ROOT)
    ids = [item.dataset_id for item in summaries]
    assert ids == sorted(ids)
    assert ids == [
        item.dataset_id for item in discover_registered_datasets(PROCESSED_ROOT)
    ]
    assert "v1_basic_clean" not in ids
    assert all(item.schema_hash for item in summaries)


def test_discover_registry_missing_root(tmp_path: Path) -> None:
    with pytest.raises(DatasetExperimentMaterializerError, match="does not exist"):
        discover_registry_summaries(tmp_path, processed_root_relative="data/processed")


def test_discover_registry_empty(tmp_path: Path) -> None:
    (tmp_path / "data" / "processed").mkdir(parents=True)
    with pytest.raises(
        DatasetExperimentMaterializerError, match="no strictly validated"
    ):
        discover_registry_summaries(tmp_path)


def test_list_base_templates_excludes_comparison_policy() -> None:
    templates = list_base_templates(PROJECT_ROOT)
    assert templates
    assert all("comparison_policy" not in item.relative_path for item in templates)
    assert any(item.mode == "Smoke" for item in templates)
    assert any(item.model_family == "LightGBM" for item in templates)


@pytest.mark.skipif(not _has_v7(), reason="v7 package not present")
def test_prepare_v7_lightgbm_smoke_materializes_passthrough() -> None:
    _clean_editable_root()
    dataset_id = "v7_compact_zero_indicators"
    bundle = prepare_dataset_driven_experiment(
        PROJECT_ROOT,
        editable_root=EDITABLE_ROOT,
        dataset_id=dataset_id,
        base_config_relative=LIGHTGBM_SMOKE,
        validate=True,
    )
    assert bundle.dataset_id == dataset_id
    assert bundle.config_relative.startswith(f"{EDITABLE_ROOT}/")
    assert bundle.plan_relative.startswith(f"{EDITABLE_ROOT}/plans/")
    assert Path(bundle.plan_relative).name not in {
        path.name for path in (PROJECT_ROOT / EDITABLE_ROOT).glob("*.yaml")
    }

    config = load_research_v2_config(
        PROJECT_ROOT / bundle.config_relative, project_root=PROJECT_ROOT
    )
    assert config.dataset_version == dataset_id
    assert config.plan_payload["dataset"]["version"] == dataset_id
    assert config.pipeline_id == REGISTERED_PREPARED_PASSTHROUGH_V1
    assert config.pipeline_contract == REGISTERED_PREPARED_PASSTHROUGH_CONTRACT
    assert config.adapter_id == "manual_lightgbm_te_v1_compat"

    base = yaml.safe_load((PROJECT_ROOT / LIGHTGBM_SMOKE).read_text(encoding="utf-8"))
    assert config.adapter_contract == base["candidate_adapter"]["contract"]
    base_plan = yaml.safe_load(
        (PROJECT_ROOT / base["evaluation_plan_path"]).read_text(encoding="utf-8")
    )
    for key in (
        "outer_evaluation",
        "threshold_selection",
        "threshold_policy",
        "metrics",
        "aggregation",
    ):
        assert config.plan_payload[key] == base_plan[key]
    assert config.plan_payload["dataset"]["expected_source_features"] == 234
    assert config.plan_payload["dataset"]["expected_rows"] == 10000
    assert len(config.plan_payload["dataset"]["ordered_source_schema_sha256"]) == 64


@pytest.mark.skipif(not _has_v7(), reason="v7 package not present")
def test_prepare_is_idempotent_for_identical_content() -> None:
    _clean_editable_root()
    first = prepare_dataset_driven_experiment(
        PROJECT_ROOT,
        editable_root=EDITABLE_ROOT,
        dataset_id="v7_compact_zero_indicators",
        base_config_relative=LIGHTGBM_SMOKE,
    )
    second = prepare_dataset_driven_experiment(
        PROJECT_ROOT,
        editable_root=EDITABLE_ROOT,
        dataset_id="v7_compact_zero_indicators",
        base_config_relative=LIGHTGBM_SMOKE,
    )
    assert first.config_relative == second.config_relative
    assert first.plan_relative == second.plan_relative
    assert second.reused is True


@pytest.mark.skipif(not _has_v7(), reason="v7 package not present")
def test_prepare_rejects_content_conflict() -> None:
    _clean_editable_root()
    bundle = prepare_dataset_driven_experiment(
        PROJECT_ROOT,
        editable_root=EDITABLE_ROOT,
        dataset_id="v7_compact_zero_indicators",
        base_config_relative=LIGHTGBM_SMOKE,
    )
    target = PROJECT_ROOT / bundle.config_relative
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    payload["artifacts"] = {"root": "artifacts/research_v2_conflict_probe"}
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(DatasetExperimentMaterializerError, match="different content"):
        prepare_dataset_driven_experiment(
            PROJECT_ROOT,
            editable_root=EDITABLE_ROOT,
            dataset_id="v7_compact_zero_indicators",
            base_config_relative=LIGHTGBM_SMOKE,
            validate=False,
        )


def test_selection_fingerprint_changes_with_dataset_or_mode() -> None:
    first = selection_fingerprint(
        dataset_id="v7_compact_zero_indicators",
        base_config_relative=LIGHTGBM_SMOKE,
        model_family="LightGBM",
        mode="Smoke",
    )
    second = selection_fingerprint(
        dataset_id="v0_raw_minimal",
        base_config_relative=LIGHTGBM_SMOKE,
        model_family="LightGBM",
        mode="Smoke",
    )
    third = selection_fingerprint(
        dataset_id="v7_compact_zero_indicators",
        base_config_relative=LIGHTGBM_SMOKE,
        model_family="LightGBM",
        mode="Development",
    )
    assert first != second
    assert first != third


def test_path_traversal_base_config_rejected() -> None:
    with pytest.raises(DatasetExperimentMaterializerError, match="unsafe"):
        prepare_dataset_driven_experiment(
            PROJECT_ROOT,
            editable_root=EDITABLE_ROOT,
            dataset_id="v7_compact_zero_indicators",
            base_config_relative="../secrets/config.yaml",
            validate=False,
        )


def test_unsafe_editable_relative_rejected() -> None:
    from src.churn_ml.control_panel.config_editor import (
        ConfigEditError,
        publish_editable_text_file,
    )

    with pytest.raises(ConfigEditError, match="safe"):
        publish_editable_text_file(
            PROJECT_ROOT,
            editable_root=EDITABLE_ROOT,
            relative_path="../escape.yaml",
            text="schema_version: 1\n",
        )


def test_control_panel_package_import_still_avoids_experiment_v2() -> None:
    code = (
        "import sys; import src.churn_ml.control_panel; "
        "forbidden=('research_evaluation','experiment_v2','deployment_v1','metrics','optuna'); "
        "assert not any(any(x in name for x in forbidden) for name in sys.modules)"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        check=True,
        timeout=20,
    )
