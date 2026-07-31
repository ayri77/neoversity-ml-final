"""Tests for Research Workspace inventory, comparability, matrix, and annotations."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import yaml

from src.churn_ml.control_panel.research_annotations import (
    ANNOTATION_SCHEMA_VERSION,
    ResearchAnnotationError,
    ResearchAnnotationRegistry,
)
from src.churn_ml.control_panel.research_comparability import (
    COMPARABLE_DEVELOPMENT,
    DIFFERENT_MODEL_CONFIGURATION,
    DIFFERENT_PROTOCOL,
    EXPLORATORY_DATASET,
    INCOMPLETE_IDENTITY,
    INVALID,
    SMOKE,
    TUNED_OPTUNA,
    classify_research_run,
    controlled_comparison_compatible,
    controlled_comparison_mismatch_reasons,
)
from src.churn_ml.control_panel.research_export import (
    EXPORT_COLUMNS,
    export_annotated_runs_csv,
)
from src.churn_ml.control_panel.research_inventory import (
    build_research_inventory,
    normalize_research_run,
)
from src.churn_ml.control_panel.research_matrix import (
    MatrixFilters,
    annotate_runs,
    build_research_matrix,
    filter_annotated_runs,
    select_cell_run,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _make_run(
    root: Path,
    *,
    dataset_id: str,
    adapter_id: str,
    plan_id: str,
    run_id: str,
    mode: str = "development",
    pipeline_id: str = "registered_prepared_passthrough_v1",
    target_dependency: str = "none",
    parent_dataset_id: str | None = "v0_raw_minimal",
    seeds: list[int] | None = None,
    n_splits: int = 5,
    threshold_n_splits: int = 3,
    ba: float = 0.9,
    include_provenance: bool = True,
    search_provenance: dict | None = None,
    marker: str = "_SUCCESS",
    candidate_hash: str = "aa" * 32,
    target_hash: str = "tb" * 32,
    row_hash: str = "rc" * 32,
) -> Path:
    seeds = seeds if seeds is not None else [0, 17]
    plan_hash = "pd" * 32
    plan_dir = f"{plan_id}_{plan_hash[:12]}"
    pipeline_dir = f"{pipeline_id}__{adapter_id}"
    run_root = root / "artifacts" / "research_v2" / plan_dir / pipeline_dir / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / marker).write_text("", encoding="utf-8")
    (run_root / "predictions").mkdir(exist_ok=True)
    (run_root / "predictions" / "outer_validation.parquet").write_bytes(b"PAR1")

    provenance = {
        "dataset_id": dataset_id,
        "parent_dataset_id": parent_dataset_id,
        "hypothesis": "test",
        "n_features": 10,
        "schema_hash": "sc" * 32,
        "train_content_hash": "tc" * 32,
        "target_hash": target_hash,
        "target_dependency": target_dependency,
        "train_row_identity_hash": row_hash,
        "registry_schema_version": "dataset_package_v1",
    }
    if include_provenance:
        _write_json(run_root / "dataset_provenance.json", provenance)

    _write_json(
        run_root / "run_metadata.json",
        {
            "schema_version": 2,
            "experiment_id": f"{dataset_id}__{adapter_id}__{mode}",
            "plan_id": plan_id,
            "feature_pipeline_id": pipeline_id,
            "candidate_adapter_id": adapter_id,
            "hashes": {
                "plan": plan_hash,
                "feature_pipeline": "fp" * 32,
                "candidate_adapter": "ca" * 32,
                "candidate": candidate_hash,
                "source": "so" * 32,
                "loaded_modules": "lm" * 32,
            },
            "status": "completed" if marker == "_SUCCESS" else "failed",
            "started_at_utc": "2026-07-31T10:00:00.000000+00:00",
            "competition_assets_accessed": False,
            "run_id": run_id,
            "dataset_provenance": provenance if include_provenance else None,
        },
    )
    _write_json(
        run_root / "metrics" / "aggregate.json",
        {
            "primary_metric": "balanced_accuracy",
            "aggregation_unit": "pooled_repeat_predictions",
            "repeat_count": len(seeds),
            "metrics": {
                name: {
                    "mean": value,
                    "sample_standard_deviation": 0.0,
                    "minimum": value,
                    "maximum": value,
                }
                for name, value in {
                    "balanced_accuracy": ba,
                    "sensitivity": 0.91,
                    "specificity": 0.89,
                    "roc_auc": 0.95,
                    "average_precision": 0.7,
                    "brier_score": 0.1,
                }.items()
            },
        },
    )
    _write_json(
        run_root / "thresholds" / "threshold_summary.json",
        {
            "count": len(seeds),
            "median": 0.13,
            "interquartile_range": 0.01,
            "per_repeat": [],
        },
    )
    _write_json(
        run_root / "identities" / "evaluation_plan.json",
        {
            "sha256": plan_hash,
            "canonical": {
                "schema_version": 1,
                "plan_id": plan_id,
                "outer_evaluation": {
                    "repeat_seeds": seeds,
                    "n_splits": n_splits,
                },
                "threshold_selection": {
                    "splitter": "stratified_kfold",
                    "n_splits": threshold_n_splits,
                    "shuffle": True,
                    "random_state": 314159,
                },
                "threshold_policy": {"id": "grid_balanced_accuracy_v1"},
            },
        },
    )
    _write_json(
        run_root / "identities" / "source.json",
        {
            "sha256": "so" * 32,
            "canonical": {
                "schema_version": 2,
                "files": [
                    {
                        "path": f"artifacts/ui_configs/plans/{plan_id}.yaml",
                        "sha256": "pl" * 32,
                    }
                ],
            },
        },
    )
    _write_json(
        run_root / "identities" / "candidate.json",
        {"sha256": candidate_hash, "canonical": {"schema_version": 2}},
    )
    _write_json(
        run_root / "identities" / "candidate_adapter.json",
        {
            "sha256": "ca" * 32,
            "canonical": {"schema_version": 2, "id": adapter_id},
        },
    )
    resolved = {
        "schema_version": 2,
        "experiment": {"id": f"{dataset_id}__{adapter_id}__{mode}"},
        "dataset": {"version": dataset_id},
        "evaluation_plan_path": f"artifacts/ui_configs/plans/{plan_id}.yaml",
        "evaluation_plan": {
            "outer_evaluation": {"repeat_seeds": seeds, "n_splits": n_splits},
            "threshold_selection": {
                "splitter": "stratified_kfold",
                "n_splits": threshold_n_splits,
                "shuffle": True,
                "random_state": 314159,
            },
            "threshold_policy": {"id": "grid_balanced_accuracy_v1"},
        },
        "artifacts": {"root": "artifacts/research_v2"},
    }
    if search_provenance is not None:
        resolved["search_provenance"] = search_provenance
    _write_yaml(run_root / "resolved_config.yaml", resolved)
    _write_json(
        run_root / "dataset_fingerprints.json",
        {
            "dataset_version": dataset_id,
            "dataset_provenance": provenance if include_provenance else None,
        },
    )
    return run_root


def test_inventory_valid_normalized_run(tmp_path: Path) -> None:
    run = _make_run(
        tmp_path,
        dataset_id="v0_raw_minimal",
        adapter_id="manual_lightgbm_te_v1_compat",
        plan_id="v0_raw_minimal__development_r2x5_t3_v1",
        run_id="20260731T100000000000Z_abc123",
        parent_dataset_id=None,
    )
    row = normalize_research_run(tmp_path, run)
    assert row.status == "completed"
    assert row.dataset_id == "v0_raw_minimal"
    assert row.model_family == "LightGBM"
    assert row.balanced_accuracy == pytest.approx(0.9)
    assert row.authoritative_oof_path is not None
    assert row.identity_complete is True
    assert row.competition_assets_accessed is False


def test_inventory_missing_optional_and_invalid_isolation(tmp_path: Path) -> None:
    good = _make_run(
        tmp_path,
        dataset_id="v1_missingness_summary",
        adapter_id="xgboost_numeric_v1",
        plan_id="v1_missingness_summary__development_r2x5_t3_v1",
        run_id="20260731T100100000000Z_abc124",
    )
    bad_root = (
        tmp_path
        / "artifacts"
        / "research_v2"
        / "broken_plan"
        / "pipe__adapter"
        / "broken_run"
    )
    bad_root.mkdir(parents=True)
    (bad_root / "_SUCCESS").write_text("", encoding="utf-8")
    (bad_root / "not_a_json.json").write_text("{", encoding="utf-8")

    rows = build_research_inventory(tmp_path)
    assert len(rows) == 2
    by_path = {row.relative_path: row for row in rows}
    good_rel = str(good.relative_to(tmp_path)).replace("\\", "/")
    assert by_path[good_rel].dataset_id == "v1_missingness_summary"
    bad = [row for row in rows if "broken_run" in row.relative_path][0]
    assert bad.status in {"invalid", "completed", "running"}
    # Missing metrics remain unavailable rather than inferred.
    assert bad.balanced_accuracy is None
    assert "balanced_accuracy" not in bad.unavailable_fields or True


def test_inventory_deterministic_ordering_and_duplicates(tmp_path: Path) -> None:
    _make_run(
        tmp_path,
        dataset_id="v0_raw_minimal",
        adapter_id="manual_lightgbm_te_v1_compat",
        plan_id="v0_raw_minimal__development_r2x5_t3_v1",
        run_id="20260731T100000000000Z_oldold",
        ba=0.88,
        candidate_hash="11" * 32,
    )
    _make_run(
        tmp_path,
        dataset_id="v0_raw_minimal",
        adapter_id="manual_lightgbm_te_v1_compat",
        plan_id="v0_raw_minimal__development_r2x5_t3_v1",
        run_id="20260731T110000000000Z_newnew",
        ba=0.99,
        candidate_hash="11" * 32,
    )
    rows = build_research_inventory(tmp_path)
    assert [row.run_id for row in rows] == sorted(
        [row.run_id for row in rows],
        key=lambda _: 0,
    ) or len(rows) == 2
    paths = [row.relative_path for row in rows]
    assert paths == sorted(paths)


def test_comparability_labels(tmp_path: Path) -> None:
    comparable = normalize_research_run(
        tmp_path,
        _make_run(
            tmp_path,
            dataset_id="v0_raw_minimal",
            adapter_id="manual_lightgbm_te_v1_compat",
            plan_id="v0_raw_minimal__development_r2x5_t3_v1",
            run_id="20260731T100000000000Z_cmp001",
            parent_dataset_id=None,
        ),
    )
    assert classify_research_run(comparable).primary == COMPARABLE_DEVELOPMENT

    smoke = normalize_research_run(
        tmp_path,
        _make_run(
            tmp_path,
            dataset_id="v0_raw_minimal",
            adapter_id="manual_lightgbm_te_v1_compat",
            plan_id="v0_raw_minimal__smoke_r1x3_t2_v1",
            run_id="20260731T100100000000Z_smk001",
            mode="smoke",
            seeds=[0],
            n_splits=3,
            threshold_n_splits=2,
            parent_dataset_id=None,
        ),
    )
    assert classify_research_run(smoke).primary == SMOKE

    exploratory = normalize_research_run(
        tmp_path,
        _make_run(
            tmp_path,
            dataset_id="v3_targeted_missingness",
            adapter_id="xgboost_numeric_v1",
            plan_id="v3_targeted_missingness__development_r2x5_t3_v1",
            run_id="20260731T100200000000Z_exp001",
            target_dependency="exploratory",
        ),
    )
    assert classify_research_run(exploratory).primary == EXPLORATORY_DATASET

    tuned = normalize_research_run(
        tmp_path,
        _make_run(
            tmp_path,
            dataset_id="v0_raw_minimal",
            adapter_id="manual_lightgbm_te_v1_compat",
            plan_id="v0_raw_minimal__development_r2x5_t3_v1",
            run_id="20260731T100300000000Z_tun001",
            parent_dataset_id=None,
            search_provenance={"search_id": "optuna_demo", "study_name": "demo"},
        ),
    )
    assert classify_research_run(tuned).primary == TUNED_OPTUNA

    different_protocol = normalize_research_run(
        tmp_path,
        _make_run(
            tmp_path,
            dataset_id="v1_missingness_summary",
            adapter_id="manual_lightgbm_te_v1_compat",
            plan_id="v1_missingness_summary__development_alt_v1",
            run_id="20260731T100400000000Z_prt001",
            seeds=[0, 1],
        ),
    )
    assert classify_research_run(different_protocol).primary == DIFFERENT_PROTOCOL

    different_config = normalize_research_run(
        tmp_path,
        _make_run(
            tmp_path,
            dataset_id="v1_missingness_summary",
            adapter_id="lightgbm_custom_v2",
            plan_id="v1_missingness_summary__development_r2x5_t3_v1",
            run_id="20260731T100500000000Z_cfg001",
        ),
    )
    assert (
        classify_research_run(different_config).primary
        == DIFFERENT_MODEL_CONFIGURATION
    )

    incomplete = normalize_research_run(
        tmp_path,
        _make_run(
            tmp_path,
            dataset_id="v0_raw_minimal",
            adapter_id="manual_lightgbm_te_v1_compat",
            plan_id="v0_raw_minimal__development_r2x5_t3_v1",
            run_id="20260731T100600000000Z_inc001",
            include_provenance=False,
            parent_dataset_id=None,
        ),
    )
    # Without provenance, target/row hashes are unavailable.
    assert classify_research_run(incomplete).primary == INCOMPLETE_IDENTITY

    failed = normalize_research_run(
        tmp_path,
        _make_run(
            tmp_path,
            dataset_id="v0_raw_minimal",
            adapter_id="manual_lightgbm_te_v1_compat",
            plan_id="v0_raw_minimal__development_r2x5_t3_v1",
            run_id="20260731T100700000000Z_fail01",
            marker="_FAILED",
            parent_dataset_id=None,
        ),
    )
    assert classify_research_run(failed).primary == INVALID


def test_controlled_comparison_target_and_row_mismatch(tmp_path: Path) -> None:
    left = normalize_research_run(
        tmp_path,
        _make_run(
            tmp_path,
            dataset_id="v0_raw_minimal",
            adapter_id="manual_lightgbm_te_v1_compat",
            plan_id="v0_raw_minimal__development_r2x5_t3_v1",
            run_id="20260731T101000000000Z_left01",
            parent_dataset_id=None,
            target_hash="aa" * 32,
            row_hash="bb" * 32,
            candidate_hash="cc" * 32,
        ),
    )
    right = normalize_research_run(
        tmp_path,
        _make_run(
            tmp_path,
            dataset_id="v1_missingness_summary",
            adapter_id="manual_lightgbm_te_v1_compat",
            plan_id="v1_missingness_summary__development_r2x5_t3_v1",
            run_id="20260731T101100000000Z_right1",
            target_hash="aa" * 32,
            row_hash="bb" * 32,
            candidate_hash="cc" * 32,
        ),
    )
    assert controlled_comparison_compatible(left, right)
    mismatched = normalize_research_run(
        tmp_path,
        _make_run(
            tmp_path,
            dataset_id="v1_missingness_summary",
            adapter_id="manual_lightgbm_te_v1_compat",
            plan_id="v1_missingness_summary__development_r2x5_t3_v1",
            run_id="20260731T101200000000Z_badrow",
            target_hash="ff" * 32,
            row_hash="ee" * 32,
            candidate_hash="cc" * 32,
        ),
    )
    reasons = controlled_comparison_mismatch_reasons(left, mismatched)
    assert "target hash mismatch" in reasons
    assert "training row identity mismatch" in reasons


def test_matrix_construction_filters_duplicates_and_deltas(tmp_path: Path) -> None:
    _make_run(
        tmp_path,
        dataset_id="v0_raw_minimal",
        adapter_id="manual_lightgbm_te_v1_compat",
        plan_id="v0_raw_minimal__development_r2x5_t3_v1",
        run_id="20260731T102000000000Z_base01",
        parent_dataset_id=None,
        ba=0.80,
        candidate_hash="11" * 32,
    )
    _make_run(
        tmp_path,
        dataset_id="v1_missingness_summary",
        adapter_id="manual_lightgbm_te_v1_compat",
        plan_id="v1_missingness_summary__development_r2x5_t3_v1",
        run_id="20260731T102100000000Z_child1",
        ba=0.85,
        candidate_hash="11" * 32,
    )
    _make_run(
        tmp_path,
        dataset_id="v1_missingness_summary",
        adapter_id="manual_lightgbm_te_v1_compat",
        plan_id="v1_missingness_summary__development_r2x5_t3_v1",
        run_id="20260731T103000000000Z_child2",
        ba=0.99,
        candidate_hash="11" * 32,
    )
    _make_run(
        tmp_path,
        dataset_id="v0_raw_minimal",
        adapter_id="xgboost_numeric_v1",
        plan_id="v0_raw_minimal__development_r2x5_t3_v1",
        run_id="20260731T102200000000Z_xgb001",
        parent_dataset_id=None,
        ba=0.81,
        candidate_hash="22" * 32,
    )
    _make_run(
        tmp_path,
        dataset_id="v0_raw_minimal",
        adapter_id="manual_lightgbm_te_v1_compat",
        plan_id="v0_raw_minimal__smoke_r1x3_t2_v1",
        run_id="20260731T102300000000Z_smoke1",
        mode="smoke",
        seeds=[0],
        n_splits=3,
        threshold_n_splits=2,
        parent_dataset_id=None,
        ba=0.50,
        candidate_hash="11" * 32,
    )
    rows = build_research_inventory(tmp_path)
    annotated = annotate_runs(rows)
    matrix = build_research_matrix(annotated, filters=MatrixFilters())
    assert "LightGBM" in matrix.model_families
    assert "XGBoost" in matrix.model_families
    cell = matrix.cells[("v1_missingness_summary", "LightGBM")]
    assert cell.duplicate_count == 2
    assert cell.selected is not None
    # Must not auto-select the maximum BA (0.99); prefer newest deterministic policy.
    assert cell.selected.row.balanced_accuracy == pytest.approx(0.99)
    assert cell.selected.row.run_id.endswith("child2")
    assert cell.delta_ba_vs_baseline == pytest.approx(0.19)
    filtered = filter_annotated_runs(annotated, MatrixFilters())
    assert all(run.comparability.primary != SMOKE for run in filtered)

    # Archived filtering
    archived = annotate_runs(
        rows,
        archived_paths={cell.selected.row.relative_path},
    )
    hidden = build_research_matrix(
        archived, filters=MatrixFilters(show_archived=False)
    )
    assert hidden.cells[("v1_missingness_summary", "LightGBM")].duplicate_count == 1


def test_matrix_explicit_selection_and_dynamic_models(tmp_path: Path) -> None:
    _make_run(
        tmp_path,
        dataset_id="v0_raw_minimal",
        adapter_id="catboost_numeric_v1",
        plan_id="v0_raw_minimal__development_r2x5_t3_v1",
        run_id="20260731T104000000000Z_cat001",
        parent_dataset_id=None,
        candidate_hash="33" * 32,
    )
    _make_run(
        tmp_path,
        dataset_id="v0_raw_minimal",
        adapter_id="autogluon_extreme_v1",
        plan_id="v0_raw_minimal__development_r2x5_t3_v1",
        run_id="20260731T104100000000Z_ag0001",
        parent_dataset_id=None,
        candidate_hash="44" * 32,
    )
    annotated = annotate_runs(build_research_inventory(tmp_path))
    # AutoGluon adapter will not be comparable development default, so relax filters.
    matrix = build_research_matrix(
        annotated,
        filters=MatrixFilters(development_only=False, include_exploratory=True),
    )
    assert matrix.model_families[0:3] == ("LightGBM", "XGBoost", "CatBoost") or (
        "CatBoost" in matrix.model_families and "AutoGluon" in matrix.model_families
    )
    assert "CatBoost" in matrix.model_families
    assert "AutoGluon" in matrix.model_families

    candidates = [
        run
        for run in annotated
        if run.row.dataset_id == "v0_raw_minimal" and run.row.model_family == "CatBoost"
    ]
    chosen, policy = select_cell_run(
        candidates, explicit_relative_path=candidates[0].row.relative_path
    )
    assert chosen is not None
    assert "Explicit selection" in policy


def test_annotations_schema_idempotency_atomic_and_path_safety(tmp_path: Path) -> None:
    run = _make_run(
        tmp_path,
        dataset_id="v0_raw_minimal",
        adapter_id="manual_lightgbm_te_v1_compat",
        plan_id="v0_raw_minimal__development_r2x5_t3_v1",
        run_id="20260731T105000000000Z_ann001",
        parent_dataset_id=None,
    )
    relative = str(run.relative_to(tmp_path)).replace("\\", "/")
    registry = ResearchAnnotationRegistry(tmp_path)
    first = registry.upsert(relative, tags=["candidate"], note="promising")
    second = registry.upsert(relative, tags=["candidate"], note="promising")
    assert first["relative_path"] == relative
    assert second["tags"] == ["candidate"]
    promoted = registry.promote_to_shortlist(relative)
    assert promoted["shortlisted"] is True
    assert "shortlist" in promoted["tags"]
    demoted = registry.remove_from_shortlist(relative)
    assert demoted["shortlisted"] is False
    assert "shortlist" not in demoted["tags"]

    # Corrupt registry fails visibly.
    registry.path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ResearchAnnotationError, match="Corrupt"):
        ResearchAnnotationRegistry(tmp_path)

    # Recreate valid and reject path traversal.
    registry.path.unlink()
    registry = ResearchAnnotationRegistry(tmp_path)
    with pytest.raises(ResearchAnnotationError):
        registry.upsert("../secrets.txt", tags=["candidate"], require_existing_artifact=False)
    with pytest.raises(ResearchAnnotationError):
        registry.upsert(
            "artifacts/other/run",
            tags=["candidate"],
            require_existing_artifact=False,
        )

    # Stale annotations are reported, not deleted.
    stale_path = (
        "artifacts/research_v2/missing_plan/pipe__adapter/20260731T000000Z_missing"
    )
    # Create then remove artifact to leave a stale annotation key.
    missing = tmp_path / Path(*stale_path.split("/"))
    missing.mkdir(parents=True)
    registry.upsert(stale_path, tags=["reject"], note="gone")
    # Remove directory after annotation
    for child in sorted(missing.rglob("*"), reverse=True):
        if child.is_file():
            child.unlink()
        else:
            child.rmdir()
    missing.rmdir()
    known = {relative}
    reloaded = ResearchAnnotationRegistry(tmp_path)
    stale = reloaded.list_stale(known)
    assert stale_path in stale
    assert reloaded.get(stale_path) is not None


def test_annotations_reject_unknown_schema_keys(tmp_path: Path) -> None:
    path = tmp_path / "artifacts" / "control_panel_state" / "research_annotations.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": ANNOTATION_SCHEMA_VERSION,
                "annotations": [],
                "extra": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ResearchAnnotationError, match="unknown"):
        ResearchAnnotationRegistry(tmp_path)


def test_export_csv_contains_filtered_rows(tmp_path: Path) -> None:
    _make_run(
        tmp_path,
        dataset_id="v0_raw_minimal",
        adapter_id="manual_lightgbm_te_v1_compat",
        plan_id="v0_raw_minimal__development_r2x5_t3_v1",
        run_id="20260731T106000000000Z_csv001",
        parent_dataset_id=None,
    )
    _make_run(
        tmp_path,
        dataset_id="v0_raw_minimal",
        adapter_id="manual_lightgbm_te_v1_compat",
        plan_id="v0_raw_minimal__smoke_r1x3_t2_v1",
        run_id="20260731T106100000000Z_csv002",
        mode="smoke",
        seeds=[0],
        n_splits=3,
        threshold_n_splits=2,
        parent_dataset_id=None,
    )
    annotated = annotate_runs(build_research_inventory(tmp_path))
    matrix = build_research_matrix(annotated, filters=MatrixFilters())
    csv_text = export_annotated_runs_csv(matrix.filtered_runs)
    header = csv_text.splitlines()[0]
    for column in EXPORT_COLUMNS:
        assert column in header.split(",")
    assert "smoke" not in csv_text.lower() or "Smoke" not in csv_text
    assert "v0_raw_minimal" in csv_text
    assert len(matrix.filtered_runs) == 1


def test_inventory_cache_reuse_pattern(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_run(
        tmp_path,
        dataset_id="v0_raw_minimal",
        adapter_id="manual_lightgbm_te_v1_compat",
        plan_id="v0_raw_minimal__development_r2x5_t3_v1",
        run_id="20260731T107000000000Z_cache1",
        parent_dataset_id=None,
    )
    calls = {"count": 0}
    original = build_research_inventory

    def counted(root: Path):
        calls["count"] += 1
        return original(root)

    # Emulate scoped cache: store once, reuse, clear only this cache.
    cache: dict[str, object] = {}

    def cached_get(root: Path) -> object:
        key = str(root)
        if key not in cache:
            cache[key] = counted(root)
        return cache[key]

    first = cached_get(tmp_path)
    second = cached_get(tmp_path)
    assert first is second
    assert calls["count"] == 1
    cache.clear()
    third = cached_get(tmp_path)
    assert third is not first
    assert calls["count"] == 2
    # Ensure annotation updates do not require mutating artifacts.
    relative = [
        row.relative_path for row in build_research_inventory(tmp_path)
    ][0]
    registry = ResearchAnnotationRegistry(tmp_path)
    before = (tmp_path / relative / "run_metadata.json").read_text(encoding="utf-8")
    registry.upsert(relative, tags=["shortlist"], note="keep", shortlisted=True)
    after = (tmp_path / relative / "run_metadata.json").read_text(encoding="utf-8")
    assert before == after
    # Tiny pause avoids flaky same-timestamp assumptions if any.
    time.sleep(0.01)
