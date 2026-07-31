"""Tests for Dataset Campaign / Matrix Runner v1."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.churn_ml.dataset_campaign.artifacts import (
    load_frozen_manifest,
    load_status,
    write_frozen_manifest,
)
from src.churn_ml.dataset_campaign.constants import UNBIASED_SCREENING_DATASET_IDS
from src.churn_ml.dataset_campaign.errors import (
    CampaignConfigurationError,
    CampaignManifestError,
    CampaignValidationError,
)
from src.churn_ml.dataset_campaign.execute import inspect_campaign, run_campaign, validate_only
from src.churn_ml.dataset_campaign.matrix import _load_model_snapshot, expand_matrix
from src.churn_ml.dataset_campaign.oof import validate_outer_validation_oof
from src.churn_ml.dataset_campaign.resolve import resolve_campaign
from src.churn_ml.dataset_campaign.schema import CampaignModelEntry, parse_campaign_spec
from src.churn_ml.dataset_registry import discover_registered_datasets
from src.churn_ml.research_data import canonical_sha256
from tests.dataset_campaign_support import (
    FakeCellRunner,
    campaign_payload,
    create_campaign_project,
)


def test_strict_config_unknown_key_rejection(tmp_path: Path) -> None:
    create_campaign_project(tmp_path)
    payload = campaign_payload()
    payload["unexpected"] = True
    with pytest.raises(CampaignConfigurationError, match="unknown"):
        parse_campaign_spec(payload, project_root=tmp_path)


def test_duplicate_dataset_rejection(tmp_path: Path) -> None:
    create_campaign_project(tmp_path)
    payload = campaign_payload(
        dataset_ids=["v0_raw_minimal", "v0_raw_minimal"],
        families=["LightGBM"],
    )
    with pytest.raises(CampaignConfigurationError, match="Duplicate dataset"):
        parse_campaign_spec(payload, project_root=tmp_path)


def test_registry_discovery(tmp_path: Path) -> None:
    create_campaign_project(tmp_path)
    summaries = discover_registered_datasets(tmp_path / "data" / "processed")
    ids = [item.dataset_id for item in summaries]
    assert ids == sorted(ids)
    for dataset_id in UNBIASED_SCREENING_DATASET_IDS:
        assert dataset_id in ids
    assert "v3_targeted_missingness" in ids


def test_deterministic_7x3_expansion(tmp_path: Path) -> None:
    create_campaign_project(tmp_path)
    payload = campaign_payload(preset="unbiased_screening_v1")
    spec = parse_campaign_spec(payload, project_root=tmp_path)
    cells = expand_matrix(spec)
    assert len(cells) == 21
    assert [cell.dataset_id for cell in cells[:7]] == list(
        UNBIASED_SCREENING_DATASET_IDS
    )
    assert cells[0].model_family == "LightGBM"
    assert cells[7].model_family == "XGBoost"
    assert cells[14].model_family == "CatBoost"
    assert [cell.execution_order for cell in cells] == list(range(21))


def test_v3_excluded_from_unbiased(tmp_path: Path) -> None:
    create_campaign_project(tmp_path)
    payload = campaign_payload(
        dataset_ids=["v0_raw_minimal", "v3_targeted_missingness"],
        families=["LightGBM"],
    )
    with pytest.raises(CampaignConfigurationError, match="exploratory"):
        parse_campaign_spec(payload, project_root=tmp_path)


def test_exploratory_v3_campaign_allowed(tmp_path: Path) -> None:
    create_campaign_project(tmp_path)
    payload = campaign_payload(
        campaign_id="exploratory_v3",
        campaign_type="exploratory",
        classification="exploratory",
        dataset_ids=["v3_targeted_missingness"],
        families=["LightGBM"],
    )
    spec = parse_campaign_spec(payload, project_root=tmp_path)
    assert spec.dataset_ids == ("v3_targeted_missingness",)
    assert expand_matrix(spec)[0].dataset_id == "v3_targeted_missingness"


def test_invariant_conflicting_protocols(tmp_path: Path) -> None:
    create_campaign_project(tmp_path)
    other_plan = {
        "schema_version": 1,
        "plan": {"id": "other"},
        "dataset": {
            "processed_dir": "data/processed",
            "version": "v0_raw_minimal",
            "files": {
                "train_features": {"name": "X_train.parquet", "sha256": "0" * 64},
                "target": {"name": "y_train.parquet", "sha256": "0" * 64},
                "metadata": {"name": "metadata.json", "sha256": "0" * 64},
            },
            "expected_rows": 6,
            "expected_source_features": 3,
            "ordered_source_schema_sha256": "0" * 64,
            "ordered_dtype_schema_sha256": "0" * 64,
            "target": {
                "name": "y",
                "dtype": "int64",
                "negative_label": 0,
                "positive_label": 1,
                "expected_negative_rows": 3,
                "expected_positive_rows": 3,
                "values_sha256": "0" * 64,
            },
        },
        "outer_evaluation": {
            "splitter": "stratified_kfold",
            "n_splits": 5,
            "shuffle": True,
            "repeat_seeds": [0, 1],
        },
        "threshold_selection": {
            "splitter": "stratified_kfold",
            "n_splits": 2,
            "shuffle": True,
            "random_state": 1,
        },
        "threshold_policy": {
            "id": "grid_balanced_accuracy_v1",
            "metric": "balanced_accuracy",
            "minimum": 0.01,
            "maximum": 0.99,
            "step": 0.001,
            "comparison": "greater_than_or_equal",
            "maximizer_absolute_tolerance": 1.0e-12,
            "tie_break": "median_maximizer_lower_on_even",
            "constant_probability_fallback": 0.5,
        },
        "metrics": {
            "primary": "balanced_accuracy",
            "secondary": ["sensitivity"],
            "diagnostic": ["tn"],
        },
        "aggregation": {
            "repeat_method": "pooled_predictions",
            "aggregate_unit": "repeat",
            "standard_deviation": "sample",
            "fold_standard_deviation": "descriptive_only",
            "confidence_interval": "none",
        },
    }
    other_rel = "configs/research_v2/plans/other_plan.yaml"
    (tmp_path / other_rel).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / other_rel).write_text(yaml.safe_dump(other_plan), encoding="utf-8")
    xgb_path = tmp_path / "configs/research_v2/xgboost_numeric_v1_smoke.yaml"
    xgb = yaml.safe_load(xgb_path.read_text(encoding="utf-8"))
    xgb["evaluation_plan_path"] = other_rel
    xgb_path.write_text(yaml.safe_dump(xgb), encoding="utf-8")
    payload = campaign_payload(families=["LightGBM", "XGBoost"])
    payload["evaluation_plan"] = {"path": None}
    spec = parse_campaign_spec(payload, project_root=tmp_path)
    with pytest.raises(CampaignValidationError, match="protocol"):
        resolve_campaign(
            spec,
            project_root=tmp_path,
            materialize=False,
            validate_prepared=False,
        )


def test_manifest_hash_immutability(tmp_path: Path) -> None:
    create_campaign_project(tmp_path)
    payload = campaign_payload(preset="unbiased_screening_v1")
    spec = parse_campaign_spec(payload, project_root=tmp_path)
    campaign_dir = tmp_path / "artifacts" / "dataset_campaigns" / spec.campaign_id
    resolved = resolve_campaign(
        spec,
        project_root=tmp_path,
        campaign_dir=campaign_dir,
        materialize=True,
        validate_prepared=False,
    )
    assert len(resolved.manifest["cells"]) == 21
    write_frozen_manifest(campaign_dir, resolved.manifest)
    loaded = load_frozen_manifest(campaign_dir)
    assert loaded["manifest_hash"] == resolved.manifest_hash
    body = {key: value for key, value in loaded.items() if key != "manifest_hash"}
    assert canonical_sha256(body) == loaded["manifest_hash"]
    mutated = dict(loaded)
    mutated["campaign_name"] = "mutated"
    with pytest.raises(CampaignManifestError, match="does not match canonical"):
        write_frozen_manifest(campaign_dir, mutated)


def test_validate_only_allocates_no_experiment_runs(tmp_path: Path) -> None:
    create_campaign_project(tmp_path)
    payload = campaign_payload(preset="unbiased_screening_v1")
    spec = parse_campaign_spec(payload, project_root=tmp_path)
    runner = FakeCellRunner()
    before = {
        path
        for path in (tmp_path / "artifacts" / "research_v2").rglob("*")
        if path.is_file()
    }
    resolved = validate_only(
        spec,
        project_root=tmp_path,
        freeze=False,
        runner=runner,
        validate_prepared=False,
    )
    after = {
        path
        for path in (tmp_path / "artifacts" / "research_v2").rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not any(
        (tmp_path / "artifacts" / "research_v2").rglob("_SUCCESS")
    )
    assert not (tmp_path / "artifacts" / "dataset_campaigns" / spec.campaign_id / "campaign_manifest.json").exists()
    assert resolved.manifest["cell_count"] == 21
    assert len(runner.validated) == 21
    assert runner.runs == []


def test_validate_freeze_then_execute_resume_retry(tmp_path: Path) -> None:
    create_campaign_project(tmp_path)
    payload = campaign_payload(
        dataset_ids=["v0_raw_minimal", "v7_compact_zero_indicators"],
        families=["LightGBM", "XGBoost"],
    )
    spec = parse_campaign_spec(payload, project_root=tmp_path)
    runner = FakeCellRunner()
    validate_only(
        spec,
        project_root=tmp_path,
        freeze=True,
        runner=runner,
        validate_prepared=False,
    )
    campaign_dir = tmp_path / "artifacts" / "dataset_campaigns" / spec.campaign_id
    manifest = load_frozen_manifest(campaign_dir)
    assert manifest["cell_count"] == 4

    # Fail the second cell on first pass.
    second_cell = sorted(manifest["cells"], key=lambda item: item["execution_order"])[1]
    runner.fail_prefixes = {second_cell["cell_id"]}
    result = run_campaign(
        project_root=tmp_path,
        campaign_dir=campaign_dir,
        runner=runner,
        resume=False,
    )
    assert result["status"]["state"] == "partial"
    status = load_status(campaign_dir)
    assert status["cells"][second_cell["cell_id"]]["state"] == "failed"
    assert len(status["cells"][second_cell["cell_id"]]["attempts"]) == 1
    succeeded_before = sum(
        1 for cell in status["cells"].values() if cell["state"] == "succeeded"
    )
    assert succeeded_before == 3

    # Resume: skip succeeded cells, retry failed with a new attempt.
    runner.fail_prefixes = set()
    runs_before = len(runner.runs)
    result2 = run_campaign(
        project_root=tmp_path,
        campaign_dir=campaign_dir,
        runner=runner,
        resume=True,
    )
    assert result2["status"]["state"] == "completed"
    status2 = load_status(campaign_dir)
    assert status2["cells"][second_cell["cell_id"]]["state"] == "succeeded"
    assert len(status2["cells"][second_cell["cell_id"]]["attempts"]) == 2
    # Only the failed cell should have been re-executed.
    assert len(runner.runs) == runs_before + 1

    summary = result2["summary"]
    assert summary["counts"]["succeeded"] == 4
    row = summary["rows"][0]
    assert row["balanced_accuracy"] == 0.75
    assert row["sensitivity"] == 0.7
    assert row["specificity"] == 0.8
    assert row["roc_auc"] == 0.82
    assert row["average_precision"] == 0.6
    assert row["brier_score"] == 0.15
    assert row["threshold_summary"]["median"] == 0.2
    assert row["oof_relative"]
    validate_outer_validation_oof(tmp_path / row["oof_relative"])


def test_no_overwrite_same_run_id(tmp_path: Path) -> None:
    create_campaign_project(tmp_path)
    payload = campaign_payload(
        dataset_ids=["v0_raw_minimal"],
        families=["LightGBM"],
    )
    spec = parse_campaign_spec(payload, project_root=tmp_path)
    runner = FakeCellRunner()
    validate_only(
        spec,
        project_root=tmp_path,
        freeze=True,
        runner=runner,
        validate_prepared=False,
    )
    campaign_dir = tmp_path / "artifacts" / "dataset_campaigns" / spec.campaign_id
    run_campaign(
        project_root=tmp_path,
        campaign_dir=campaign_dir,
        runner=runner,
        resume=False,
    )
    # Force a second attempt with the same run_id by calling runner directly.
    cell = load_frozen_manifest(campaign_dir)["cells"][0]
    first_run_id = f"{cell['cell_id']}_attempt_01"
    result = runner.run(
        Path(cell["prepared_config_relative"]),
        project_root=tmp_path,
        run_id=first_run_id,
    )
    assert result.success is False
    assert "overwrite" in (result.error or "").lower()


def test_manifest_drift_rejection(tmp_path: Path) -> None:
    create_campaign_project(tmp_path)
    payload = campaign_payload(
        dataset_ids=["v0_raw_minimal"],
        families=["LightGBM"],
    )
    spec = parse_campaign_spec(payload, project_root=tmp_path)
    runner = FakeCellRunner()
    validate_only(
        spec,
        project_root=tmp_path,
        freeze=True,
        runner=runner,
        validate_prepared=False,
    )
    campaign_dir = tmp_path / "artifacts" / "dataset_campaigns" / spec.campaign_id
    status = load_status(campaign_dir)
    status["manifest_hash"] = "0" * 64
    (campaign_dir / "campaign_status.json").write_text(
        json.dumps(status, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(CampaignManifestError, match="drift"):
        run_campaign(
            project_root=tmp_path,
            campaign_dir=campaign_dir,
            runner=runner,
            resume=True,
        )


def test_oof_schema_validation(tmp_path: Path) -> None:
    path = tmp_path / "outer_validation.parquet"
    pd.DataFrame(
        {
            "repeat": [1],
            "repeat_seed": [0],
            "outer_fold": [1],
            "row_position": [0],
            "target": [1],
            "probability": [0.4],
        }
    ).to_parquet(path, index=False)
    info = validate_outer_validation_oof(path)
    assert info["row_count"] == 1
    assert info["repeat_min"] == 1
    assert info["outer_fold_min"] == 1

    bad = tmp_path / "bad.parquet"
    pd.DataFrame({"repeat": [1], "probability": [0.2]}).to_parquet(bad, index=False)
    with pytest.raises(Exception, match="missing required columns"):
        validate_outer_validation_oof(bad)


def test_oof_rejects_zero_based_repeat_and_outer_fold(tmp_path: Path) -> None:
    """Research v2 OOF uses 1-based repeat/outer_fold; Stage D must not accept 0."""
    zero_repeat = tmp_path / "zero_repeat.parquet"
    pd.DataFrame(
        {
            "repeat": [0],
            "repeat_seed": [0],
            "outer_fold": [1],
            "row_position": [0],
            "target": [1],
            "probability": [0.4],
        }
    ).to_parquet(zero_repeat, index=False)
    with pytest.raises(Exception, match="1-based"):
        validate_outer_validation_oof(zero_repeat)

    zero_fold = tmp_path / "zero_fold.parquet"
    pd.DataFrame(
        {
            "repeat": [1],
            "repeat_seed": [0],
            "outer_fold": [0],
            "row_position": [0],
            "target": [1],
            "probability": [0.4],
        }
    ).to_parquet(zero_fold, index=False)
    with pytest.raises(Exception, match="1-based"):
        validate_outer_validation_oof(zero_fold)


def test_symlink_campaign_dir_rejected(tmp_path: Path) -> None:
    create_campaign_project(tmp_path)
    payload = campaign_payload(
        dataset_ids=["v0_raw_minimal"],
        families=["LightGBM"],
    )
    spec = parse_campaign_spec(payload, project_root=tmp_path)
    runner = FakeCellRunner()
    validate_only(
        spec,
        project_root=tmp_path,
        freeze=True,
        runner=runner,
        validate_prepared=False,
    )
    real = tmp_path / "artifacts" / "dataset_campaigns" / spec.campaign_id
    link = tmp_path / "campaign_link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable on this platform/permissions")
    with pytest.raises(CampaignManifestError):
        inspect_campaign(link)


def test_mlflow_failure_preserves_training_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_campaign_project(tmp_path)
    payload = campaign_payload(
        dataset_ids=["v0_raw_minimal"],
        families=["LightGBM"],
    )
    payload["execution"]["index_mlflow"] = True
    mlflow_rel = "configs/mlflow/local.yaml"
    (tmp_path / mlflow_rel).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / mlflow_rel).write_text(
        "tracking_uri: sqlite:///mlflow.db\n", encoding="utf-8"
    )
    spec = parse_campaign_spec(payload, project_root=tmp_path)
    runner = FakeCellRunner()

    def boom(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        raise RuntimeError("mlflow unavailable")

    monkeypatch.setattr(
        "src.churn_ml.dataset_campaign.execute.maybe_index_run_directory",
        boom,
    )
    validate_only(
        spec,
        project_root=tmp_path,
        freeze=True,
        runner=runner,
        validate_prepared=False,
    )
    campaign_dir = tmp_path / "artifacts" / "dataset_campaigns" / spec.campaign_id
    result = run_campaign(
        project_root=tmp_path,
        campaign_dir=campaign_dir,
        runner=runner,
        resume=False,
    )
    assert result["status"]["state"] == "completed"
    cell_id = next(iter(result["status"]["cells"]))
    attempt = result["status"]["cells"][cell_id]["attempts"][-1]
    assert attempt["error"] is None
    assert str(attempt["mlflow_index_status"]).startswith("failed")


def test_cli_project_root_is_repository_root_not_src() -> None:
    """Regression: parents[2] resolved to repo/src; parents[3] is the repo root."""
    from src.churn_ml.dataset_campaign import cli as campaign_cli

    cli_file = Path(campaign_cli.__file__).resolve()
    assert cli_file.name == "cli.py"
    parents2 = cli_file.parents[2]
    parents3 = cli_file.parents[3]
    assert parents2.name == "src"
    assert campaign_cli.PROJECT_ROOT == parents3
    assert campaign_cli.PROJECT_ROOT != parents2
    # Relative to the package layout, not a machine-specific absolute path string.
    assert (campaign_cli.PROJECT_ROOT / "src" / "churn_ml" / "dataset_campaign").is_dir()
    assert (campaign_cli.PROJECT_ROOT / "scripts" / "run_dataset_campaign.py").is_file()


def test_cli_resolves_repository_relative_campaign_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_campaign_project(tmp_path)
    payload = campaign_payload(
        campaign_id="cli_relative_config_v1",
        dataset_ids=["v0_raw_minimal"],
        families=["LightGBM"],
    )
    config_rel = Path("configs/dataset_campaign/cli_relative_config_v1.yaml")
    config_path = tmp_path / config_rel
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    from src.churn_ml.dataset_campaign import cli as campaign_cli
    from src.churn_ml.dataset_campaign.errors import EXIT_SUCCESS

    # Former parents[2] made project_root == <repo>/src, so repository-relative
    # model/plan/dataset paths under configs/ and data/ could not resolve.
    wrong_root = Path(campaign_cli.__file__).resolve().parents[2]
    assert wrong_root.name == "src"
    with pytest.raises(CampaignConfigurationError, match="does not exist"):
        parse_campaign_spec(payload, project_root=wrong_root)

    spec = parse_campaign_spec(payload, project_root=tmp_path)
    assert spec.campaign_id == "cli_relative_config_v1"
    assert (tmp_path / spec.models[0].config_path).is_file()

    # Exercise the CLI entrypoint with the corrected repository root. Stub
    # validate_only so this stays a path-resolution regression (no real fit).
    def fake_validate_only(loaded_spec, *, project_root, freeze=False):  # type: ignore[no-untyped-def]
        assert project_root.resolve() == tmp_path.resolve()
        assert (project_root / loaded_spec.models[0].config_path).is_file()
        assert freeze is False

        class _Resolved:
            manifest = {
                "campaign_id": loaded_spec.campaign_id,
                "classification": loaded_spec.classification,
                "campaign_type": loaded_spec.campaign_type,
                "cell_count": 1,
                "cells": [
                    {
                        "execution_order": 0,
                        "cell_id": "cell_stub",
                        "dataset_id": "v0_raw_minimal",
                        "model_family": "LightGBM",
                        "adapter_id": "manual_lightgbm_te_v1_compat",
                    }
                ],
                "manifest_hash": "a" * 64,
            }
            manifest_hash = "a" * 64

        return _Resolved()

    monkeypatch.setattr(campaign_cli, "validate_only", fake_validate_only)
    code = campaign_cli.execute(
        campaign_cli.parse_args(["validate", "--config", config_rel.as_posix()]),
        project_root=tmp_path,
    )
    assert code == EXIT_SUCCESS


def test_mlflow_index_passes_repository_root_and_persists_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_campaign_project(tmp_path)
    payload = campaign_payload(
        dataset_ids=["v0_raw_minimal"],
        families=["LightGBM"],
    )
    payload["execution"]["index_mlflow"] = True
    mlflow_rel = "configs/mlflow/local.yaml"
    (tmp_path / mlflow_rel).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / mlflow_rel).write_text(
        "tracking_uri: sqlite:///mlflow.db\n", encoding="utf-8"
    )
    load_calls: list[dict[str, Path]] = []
    sync_calls: list[Path] = []

    class _Summary:
        has_failures = False

    def fake_load(path: Path, *, repository_root: Path):  # type: ignore[no-untyped-def]
        load_calls.append({"path": Path(path), "repository_root": Path(repository_root)})
        return object()

    def fake_sync(config, **kwargs):  # type: ignore[no-untyped-def]
        del config
        sync_calls.append(Path(kwargs["run_dir"]))
        return _Summary()

    monkeypatch.setattr(
        "src.churn_ml.mlflow_config.load_mlflow_config",
        fake_load,
    )
    monkeypatch.setattr(
        "src.churn_ml.mlflow_sources.default_source_registry",
        lambda: object(),
    )
    monkeypatch.setattr(
        "src.churn_ml.mlflow_sync.sync_sources",
        fake_sync,
    )

    spec = parse_campaign_spec(payload, project_root=tmp_path)
    runner = FakeCellRunner()
    validate_only(
        spec,
        project_root=tmp_path,
        freeze=True,
        runner=runner,
        validate_prepared=False,
    )
    campaign_dir = tmp_path / "artifacts" / "dataset_campaigns" / spec.campaign_id
    first_manifest = (campaign_dir / "campaign_manifest.json").read_text(encoding="utf-8")
    result = run_campaign(
        project_root=tmp_path,
        campaign_dir=campaign_dir,
        runner=runner,
        resume=False,
    )
    assert result["status"]["state"] == "completed"
    cell_id = next(iter(result["status"]["cells"]))
    attempt = result["status"]["cells"][cell_id]["attempts"][-1]
    assert attempt["mlflow_index_status"] == "succeeded"
    assert attempt["error"] is None
    assert len(load_calls) == 1
    assert load_calls[0]["repository_root"].resolve() == tmp_path.resolve()
    assert load_calls[0]["path"].resolve() == (tmp_path / mlflow_rel).resolve()
    assert len(sync_calls) == 1
    # Frozen historical identity must not be rewritten by indexing.
    assert (campaign_dir / "campaign_manifest.json").read_text(encoding="utf-8") == (
        first_manifest
    )


def test_mlflow_index_idempotent_no_duplicate_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.churn_ml.dataset_campaign.runner import maybe_index_run_directory

    run_dir = tmp_path / "artifacts" / "research_v2" / "synthetic_run"
    run_dir.mkdir(parents=True)
    (tmp_path / "configs" / "mlflow").mkdir(parents=True)
    (tmp_path / "configs" / "mlflow" / "local.yaml").write_text(
        "tracking_uri: sqlite:///mlflow.db\n", encoding="utf-8"
    )
    allocated: list[str] = []

    class _Summary:
        has_failures = False

    def fake_load(path: Path, *, repository_root: Path):  # type: ignore[no-untyped-def]
        assert repository_root.resolve() == tmp_path.resolve()
        assert path.resolve() == (tmp_path / "configs/mlflow/local.yaml").resolve()
        return object()

    def fake_sync(config, **kwargs):  # type: ignore[no-untyped-def]
        del config
        key = str(Path(kwargs["run_dir"]).resolve())
        if key not in allocated:
            allocated.append(key)
        return _Summary()

    monkeypatch.setattr(
        "src.churn_ml.mlflow_config.load_mlflow_config",
        fake_load,
    )
    monkeypatch.setattr(
        "src.churn_ml.mlflow_sources.default_source_registry",
        lambda: object(),
    )
    monkeypatch.setattr(
        "src.churn_ml.mlflow_sync.sync_sources",
        fake_sync,
    )

    first = maybe_index_run_directory(
        run_directory=run_dir,
        project_root=tmp_path,
        mlflow_config_path="configs/mlflow/local.yaml",
    )
    second = maybe_index_run_directory(
        run_directory=run_dir,
        project_root=tmp_path,
        mlflow_config_path="configs/mlflow/local.yaml",
    )
    assert first == "succeeded"
    assert second == "succeeded"
    assert allocated == [str(run_dir.resolve())]


def test_family_config_mismatch(tmp_path: Path) -> None:
    create_campaign_project(tmp_path)
    with pytest.raises(CampaignConfigurationError, match="mismatch"):
        _load_model_snapshot(
            CampaignModelEntry(
                family="LightGBM",
                config_path="configs/research_v2/xgboost_numeric_v1_smoke.yaml",
            ),
            project_root=tmp_path,
        )
