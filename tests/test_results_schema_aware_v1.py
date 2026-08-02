"""Focused tests for schema-aware Results entity views and read-only MLflow status."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from streamlit.testing.v1 import AppTest

import apps.experiment_control_panel as control_panel_app
from src.churn_ml.control_panel.artifacts import discover_artifacts
from src.churn_ml.control_panel.registry import load_registry
from src.churn_ml.control_panel.results_adapters import (
    ENTITY_VIEW_READERS,
    adapt_artifact,
)
from src.churn_ml.control_panel.results_entities import ResultEntityKind
from src.churn_ml.control_panel.results_fields import (
    FieldStatus,
    available,
    invalid,
    missing_unexpectedly,
    not_applicable,
    render_field,
    unavailable_in_schema,
)
from src.churn_ml.control_panel.results_mlflow import (
    MLflowResultStatus,
    build_mlflow_projection,
    mlflow_projection_fingerprint,
    resolve_model_run_mlflow_status,
)
from src.churn_ml.control_panel.results_tables import table_rows_for_view


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_field_status_zero_and_false_remain_available() -> None:
    assert available(0).status is FieldStatus.AVAILABLE
    assert available(False).status is FieldStatus.AVAILABLE
    assert render_field(available(0)) == "0"
    assert render_field(available(False)) == "false"
    assert render_field(not_applicable("x")) == "N/A"
    assert render_field(unavailable_in_schema("x")) == "Not recorded"
    assert render_field(missing_unexpectedly("gone")) == "Missing (gone)"
    assert render_field(invalid("bad", value=1)).startswith("Invalid")


def test_available_value_not_blank() -> None:
    assert render_field(available("")) == ""
    assert render_field(available(None)) == "None"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _research_fixture(root: Path, *, run_id: str = "20260731T102957829301Z_83e78e3f") -> Path:
    artifact = (
        root
        / "artifacts"
        / "research_v2"
        / "v0_raw_minimal__development_r2x5_t3_v1_abc"
        / "pipeline__manual_lightgbm_te_v1_compat"
        / run_id
    )
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "_SUCCESS").write_text("", encoding="utf-8")
    _write_json(
        artifact / "run_metadata.json",
        {
            "schema_version": 2,
            "run_id": run_id,
            "candidate_adapter_id": "manual_lightgbm_te_v1_compat",
            "feature_pipeline_id": "registered_prepared_passthrough_v1",
            "experiment_id": "lightgbm_development",
            "started_at_utc": "2026-07-31T10:29:57.829301Z",
            "status": "completed",
        },
    )
    _write_json(
        artifact / "metrics" / "aggregate.json",
        {
            "primary_metric": "balanced_accuracy",
            "repeat_count": 2,
            "metrics": {
                "balanced_accuracy": {
                    "mean": 0.9,
                    "sample_standard_deviation": 0.01,
                    "minimum": 0.89,
                    "maximum": 0.91,
                },
                "sensitivity": {"mean": 0.8},
                "specificity": {"mean": 0.85},
                "roc_auc": {"mean": 0.95},
                "average_precision": {"mean": 0.7},
                "brier_score": {"mean": 0.05},
            },
        },
    )
    _write_json(artifact / "thresholds" / "threshold_summary.json", {"median": 0.12})
    _write_json(
        artifact / "dataset_provenance.json",
        {
            "dataset_id": "v0_raw_minimal",
            "parent_dataset_id": None,
            "n_features": 10,
            "target_dependency": "none",
        },
    )
    return artifact


def _autogluon_fixture(
    root: Path,
    *,
    run_id: str = "ag-v3-demo",
    failed: bool = False,
) -> Path:
    artifact = root / "artifacts" / "autogluon_runs" / run_id
    artifact.mkdir(parents=True, exist_ok=True)
    marker = "_FAILED" if failed else "_SUCCESS"
    (artifact / marker).write_text("", encoding="utf-8")
    _write_json(
        artifact / "run_metadata.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "dataset_version": "v3_targeted_missingness",
            "profile_id": "tabm_only_gpu_v1",
            "requested_seed": 42,
            "duration_seconds": 12.5,
            "started_at_utc": "2026-07-30T00:00:00Z",
        },
    )
    _write_json(
        artifact / "execution_status.json",
        {
            "run_id": run_id,
            "status": "failed" if failed else "completed",
            "failure_reason": "boom" if failed else None,
        },
    )
    if not failed:
        _write_json(
            artifact / "inspection" / "summary.json",
            {
                "best_model": "WeightedEnsemble_L2",
                "decision_threshold": 0.14,
                "predictor_info": {
                    "eval_metric": "balanced_accuracy",
                    "best_model_score_val": 0.76,
                    "num_bag_folds": 8,
                },
                "leaderboard": [
                    {
                        "model": "WeightedEnsemble_L2",
                        "score_val": 0.76,
                        "eval_metric": "balanced_accuracy",
                    }
                ],
            },
        )
    return artifact


def _copy_project_readers(tmp_path: Path) -> None:
    src = PROJECT_ROOT / "configs" / "ui"
    dest = tmp_path / "configs" / "ui"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("ui_readers.yaml", "ui_commands.yaml", "ui_settings.json"):
        (dest / name).write_bytes((src / name).read_bytes())
    # Registry validates approved public CLIs exist under scripts/.
    commands = yaml.safe_load((dest / "ui_commands.yaml").read_text(encoding="utf-8"))
    scripts = tmp_path / "scripts"
    scripts.mkdir(exist_ok=True)
    for public_cli in commands.get("approved_public_clis", []):
        stub = tmp_path / public_cli
        stub.parent.mkdir(parents=True, exist_ok=True)
        if not stub.is_file():
            stub.write_text("# test stub\n", encoding="utf-8")


def test_research_adapter_maps_completed_run(tmp_path: Path) -> None:
    _copy_project_readers(tmp_path)
    artifact_dir = _research_fixture(tmp_path)
    loaded = load_registry(tmp_path)
    record = discover_artifacts(tmp_path, loaded.readers["research_v2"])[0]
    entity = adapt_artifact(record, repository_root=tmp_path)
    assert entity is not None
    assert entity.kind is ResultEntityKind.MODEL_RUN
    assert entity.fields["balanced_accuracy"].value == 0.9
    assert entity.dataset_id.value == "v0_raw_minimal"
    assert entity.fields["parent_dataset_id"].status is FieldStatus.NOT_APPLICABLE
    assert artifact_dir.name in entity.entity_id


def test_research_missing_required_metric(tmp_path: Path) -> None:
    _copy_project_readers(tmp_path)
    artifact_dir = _research_fixture(tmp_path)
    (artifact_dir / "metrics" / "aggregate.json").write_text("{}", encoding="utf-8")
    loaded = load_registry(tmp_path)
    record = discover_artifacts(tmp_path, loaded.readers["research_v2"])[0]
    entity = adapt_artifact(record, repository_root=tmp_path)
    assert entity is not None
    assert entity.fields["balanced_accuracy"].status is FieldStatus.MISSING_UNEXPECTEDLY


def test_autogluon_adapter_completed_and_failed(tmp_path: Path) -> None:
    _copy_project_readers(tmp_path)
    _autogluon_fixture(tmp_path, run_id="ag-ok")
    _autogluon_fixture(tmp_path, run_id="ag-bad", failed=True)
    loaded = load_registry(tmp_path)
    assert "autogluon_v1" in loaded.readers
    records = {
        item.root.name: item
        for item in discover_artifacts(tmp_path, loaded.readers["autogluon_v1"])
    }
    ok = adapt_artifact(records["ag-ok"], repository_root=tmp_path)
    bad = adapt_artifact(records["ag-bad"], repository_root=tmp_path)
    assert ok is not None and bad is not None
    assert ok.state.value == "completed"
    assert bad.state.value == "failed"
    assert ok.fields["best_model"].value == "WeightedEnsemble_L2"
    assert ok.fields["primary_metric_value"].value == 0.76
    assert ok.fields["sensitivity"].status is FieldStatus.NOT_APPLICABLE
    assert bad.fields["best_model"].status is FieldStatus.UNAVAILABLE_IN_SCHEMA


def test_comparison_never_labelled_as_blend(tmp_path: Path) -> None:
    _copy_project_readers(tmp_path)
    root = tmp_path / "artifacts" / "research_v2_comparisons" / "v0_lgbm_vs_xgb"
    root.mkdir(parents=True)
    (root / "_SUCCESS").write_text("", encoding="utf-8")
    _write_json(
        root / "aggregate_summary.json",
        {
            "schema_version": 1,
            "metrics": {
                "balanced_accuracy": {
                    "candidate_minus_baseline": {"mean": 0.01}
                }
            },
        },
    )
    _write_json(root / "decision_report.json", {"schema_version": 1, "status": "improved"})
    loaded = load_registry(tmp_path)
    record = discover_artifacts(tmp_path, loaded.readers["paired_comparison"])[0]
    entity = adapt_artifact(record, repository_root=tmp_path)
    assert entity is not None
    assert entity.kind is ResultEntityKind.COMPARISON
    assert entity.fields["model_family"].status is FieldStatus.NOT_APPLICABLE
    row = table_rows_for_view("Comparisons", [entity])[0]
    assert "Blend" not in str(row["Comparison"])
    assert row["Delta"] == "0.01"


def test_candidate_blend_submission_adapters(tmp_path: Path) -> None:
    _copy_project_readers(tmp_path)
    cand = tmp_path / "artifacts" / "prediction_candidates" / "pc1_demo"
    cand.mkdir(parents=True)
    (cand / "_SUCCESS").write_text("", encoding="utf-8")
    _write_json(
        cand / "candidate_manifest.json",
        {
            "schema_version": "prediction_candidate_v1",
            "candidate_id": "pc1_demo",
            "dataset_id": "v5_joint_missingness_pattern",
            "source_kind": "research_v2",
            "source_model_name": "lightgbm",
            "source_metric_name": "mean_repeat_balanced_accuracy",
            "source_metric_value": 0.8982,
            "exploratory": False,
            "oof_prediction_reference": {"row_count": 10000},
            "test_row_count": 2500,
            "created_at_utc": "2026-08-02T00:00:00Z",
        },
    )
    blend = tmp_path / "artifacts" / "prediction_blends" / "pb1_demo"
    blend.mkdir(parents=True)
    (blend / "_SUCCESS").write_text("", encoding="utf-8")
    _write_json(
        blend / "blend_manifest.json",
        {
            "schema_version": "prediction_blend_v1",
            "blend_id": "pb1_demo",
            "candidate_ids": ["pc1_a", "pc1_b"],
            "final_deployment_weights": {"pc1_a": 0.8, "pc1_b": 0.2, "pc1_c": 0.0},
            "final_deployment_threshold": 0.17,
            "exploratory": False,
            "created_at_utc": "2026-08-02T00:00:00Z",
            "identity_reference_dataset_id": "v5_joint_missingness_pattern",
            "settings": {
                "optimizer_backend": "native",
                "strategy": "optimized",
                "folds": 5,
                "repeats": 2,
                "seed": 42,
            },
            "honest_meta_cv_metrics": {
                "mean_repeat_balanced_accuracy": 0.8982157,
                "std_repeat_balanced_accuracy": 0.0003,
                "min_repeat_balanced_accuracy": 0.898,
                "max_repeat_balanced_accuracy": 0.8985,
            },
            "submission_readiness": {"state": "ready"},
            "canonical_candidate_id": "pc1_demo",
        },
    )
    sub = tmp_path / "artifacts" / "candidate_submissions" / "sub_demo"
    sub.mkdir(parents=True)
    (sub / "_SUCCESS").write_text("", encoding="utf-8")
    _write_json(
        sub / "submission_manifest.json",
        {
            "schema_version": "candidate_submission_v1",
            "submission_id": "sub_demo",
            "candidate_id": "pc1_demo",
            "blend_id": "pb1_demo",
            "row_count": 2500,
            "threshold": 0.17,
            "network_access": False,
            "kaggle_upload": False,
            "created_at_utc": "2026-08-02T00:00:00Z",
        },
    )
    _write_json(sub / "source_metadata.json", {"schema_version": 1, "exploratory": False})

    loaded = load_registry(tmp_path)
    cand_e = adapt_artifact(
        discover_artifacts(tmp_path, loaded.readers["prediction_candidate_v1"])[0],
        repository_root=tmp_path,
    )
    blend_e = adapt_artifact(
        discover_artifacts(tmp_path, loaded.readers["prediction_blend_v1"])[0],
        repository_root=tmp_path,
    )
    sub_e = adapt_artifact(
        discover_artifacts(tmp_path, loaded.readers["candidate_submission_v1"])[0],
        repository_root=tmp_path,
    )
    assert cand_e is not None and blend_e is not None and sub_e is not None
    assert cand_e.dataset_id.value == "v5_joint_missingness_pattern"
    assert cand_e.fields["source_metric_value"].value == 0.8982
    assert cand_e.fields["sensitivity"].status is FieldStatus.NOT_APPLICABLE
    assert blend_e.fields["honest_mean_ba"].value == pytest.approx(0.8982157)
    assert blend_e.fields["threshold"].value == 0.17
    assert blend_e.fields["model_family"].status is FieldStatus.NOT_APPLICABLE
    assert len(blend_e.fields["nonzero_weights"].value) == 2
    assert sub_e.fields["row_count"].value == 2500
    assert sub_e.fields["network_access"].value is False
    assert sub_e.fields["kaggle_upload"].value is False
    assert sub_e.fields["kaggle_public_score"].status is FieldStatus.UNAVAILABLE_IN_SCHEMA
    assert render_field(sub_e.fields["network_access"]) == "false"


def _make_mlflow_db(path: Path, *, source_path: str, source_type: str = "research_v2") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE experiments (
              experiment_id INTEGER PRIMARY KEY,
              name TEXT,
              artifact_location TEXT,
              lifecycle_stage TEXT
            );
            CREATE TABLE runs (
              run_uuid TEXT PRIMARY KEY,
              experiment_id INTEGER,
              status TEXT,
              lifecycle_stage TEXT,
              name TEXT
            );
            CREATE TABLE tags (
              key TEXT,
              value TEXT,
              run_uuid TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO experiments VALUES (3, 'neoversity-churn-research-v2', '', 'active')"
        )
        conn.execute(
            "INSERT INTO runs VALUES ('run-1', 3, 'FINISHED', 'active', 'demo')"
        )
        tags = {
            "mlflow_index.source_type": source_type,
            "mlflow_index.source_relative_path": source_path,
            "mlflow_index.source_identity": "abc",
            "mlflow_index.source_run_id": "run",
            "mlflow_index.source_key": "key",
        }
        for key, value in tags.items():
            conn.execute(
                "INSERT INTO tags VALUES (?, ?, 'run-1')",
                (key, value),
            )
        conn.commit()
    finally:
        conn.close()


def test_mlflow_readonly_resolver(tmp_path: Path) -> None:
    source_path = "artifacts/research_v2/plan/pipe/run"
    (tmp_path / source_path).mkdir(parents=True)
    _make_mlflow_db(tmp_path / "artifacts" / "mlflow" / "mlflow.db", source_path=source_path)
    jobs = tmp_path / "artifacts" / "ui_jobs" / "job-failed"
    jobs.mkdir(parents=True)
    _write_json(
        jobs / "mlflow_index.json",
        {
            "schema_version": 1,
            "attempted": True,
            "status": "failed",
            "source_type": "research_v2",
            "artifact_path": source_path,
            "message": "old failure",
        },
    )
    # orphan
    conn = sqlite3.connect(tmp_path / "artifacts" / "mlflow" / "mlflow.db")
    conn.execute(
        "INSERT INTO runs VALUES ('run-orphan', 3, 'FINISHED', 'active', 'orphan')"
    )
    conn.execute(
        "INSERT INTO tags VALUES ('mlflow_index.source_type', 'research_v2', 'run-orphan')"
    )
    conn.execute(
        "INSERT INTO tags VALUES (?, ?, 'run-orphan')",
        (
            "mlflow_index.source_relative_path",
            "artifacts/research_v2/missing/path",
        ),
    )
    conn.commit()
    conn.close()

    projection = build_mlflow_projection(tmp_path)
    indexed = resolve_model_run_mlflow_status(
        source_type="research_v2",
        source_path=source_path,
        projection=projection,
    )
    assert indexed.status is MLflowResultStatus.INDEXED_WITH_STALE_FAILURE_RECORD
    missing = resolve_model_run_mlflow_status(
        source_type="research_v2",
        source_path="artifacts/research_v2/other",
        projection=projection,
    )
    assert missing.status is MLflowResultStatus.NOT_INDEXED
    na = resolve_model_run_mlflow_status(
        source_type="prediction_candidate",
        source_path="artifacts/prediction_candidates/x",
        projection=projection,
    )
    assert na.status is MLflowResultStatus.NOT_APPLICABLE
    assert len(projection.orphans) == 1
    assert projection.diagnostics.stale_failure_records == 1

    # missing store
    empty = build_mlflow_projection(tmp_path / "empty")
    assert empty.store_available is False
    unavailable = resolve_model_run_mlflow_status(
        source_type="research_v2",
        source_path=source_path,
        projection=empty,
    )
    assert unavailable.status is MLflowResultStatus.STORE_UNAVAILABLE

    # fingerprint changes with db mtime/size
    fp1 = mlflow_projection_fingerprint(tmp_path)
    db = tmp_path / "artifacts" / "mlflow" / "mlflow.db"
    db.write_bytes(db.read_bytes() + b"\0")
    fp2 = mlflow_projection_fingerprint(tmp_path)
    assert fp1 != fp2


def test_results_navigation_lazy_entity_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import apps.experiment_control_panel as panel

    calls: list[str] = []

    class _Session(dict):
        def pop(self, key, default=None):
            return dict.pop(self, key, default)

    with (
        patch.object(panel, "st") as fake_st,
        patch.object(panel, "ArchiveRegistry"),
        patch.object(
            panel,
            "_results_model_runs_tab",
            side_effect=lambda *a, **k: calls.append("Model runs"),
        ),
        patch.object(
            panel,
            "_results_entity_tab",
            side_effect=lambda *a, **k: calls.append(k.get("view_name") or a[1]),
        ),
        patch.object(
            panel,
            "_results_inspect_tab",
            side_effect=lambda *a, **k: calls.append("Inspect artifact"),
        ),
        patch.object(
            panel,
            "_results_research_workspace_tab",
            side_effect=lambda *a, **k: calls.append("Research Workspace"),
        ),
    ):
        fake_st.session_state = _Session({"results-active-view": "Blends"})
        fake_st.checkbox.return_value = False
        fake_st.segmented_control.return_value = "Blends"
        panel.results_page()
        options = fake_st.segmented_control.call_args.kwargs.get("options") or list(
            fake_st.segmented_control.call_args.args[1]
        )
    assert options == [
        "Model runs",
        "Comparisons",
        "Prediction candidates",
        "Blends",
        "Submissions",
        "Inspect artifact",
        "Research Workspace",
    ]
    assert calls == ["Blends"]


def test_entity_view_readers_are_scoped() -> None:
    assert ENTITY_VIEW_READERS["Model runs"] == ("research_v2", "autogluon_v1")
    assert "prediction_candidate_v1" not in ENTITY_VIEW_READERS["Model runs"]
    assert "research_v2" not in ENTITY_VIEW_READERS["Prediction candidates"]


def test_apptest_model_runs_and_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_project_readers(tmp_path)
    _research_fixture(tmp_path)
    _autogluon_fixture(tmp_path)
    cand = tmp_path / "artifacts" / "prediction_candidates" / "pc1_demo"
    cand.mkdir(parents=True)
    (cand / "_SUCCESS").write_text("", encoding="utf-8")
    _write_json(
        cand / "candidate_manifest.json",
        {
            "schema_version": "prediction_candidate_v1",
            "candidate_id": "pc1_demo",
            "dataset_id": "v5_joint_missingness_pattern",
            "source_kind": "research_v2",
            "source_model_name": "lightgbm",
            "source_metric_name": "mean_repeat_balanced_accuracy",
            "source_metric_value": 0.8982,
            "exploratory": False,
            "oof_prediction_reference": {"row_count": 10000},
            "test_row_count": 2500,
            "created_at_utc": "2026-08-02T00:00:00Z",
        },
    )
    monkeypatch.setattr(control_panel_app, "REPOSITORY_ROOT", tmp_path)
    original_load = control_panel_app.load_registry

    def load_tmp(root: Path):
        del root
        return original_load(tmp_path)

    monkeypatch.setattr(control_panel_app, "load_registry", load_tmp)
    control_panel_app.registry.clear()

    def _page() -> None:
        import apps.experiment_control_panel as panel

        panel.results_page()

    at = AppTest.from_function(_page, default_timeout=30).run()
    assert not at.exception
    assert any(title.value == "Results" for title in at.title)
    # Default Model runs
    assert any("shown" in str(c.value) and "archived" in str(c.value) for c in at.caption)
    assert len(at.dataframe) >= 1
    # Switch to candidates
    at.segmented_control[0].set_value("Prediction candidates").run()
    df = at.dataframe[0].value
    assert "Candidate" in list(df.columns)
    assert "Metric value" in list(df.columns)
    assert str(df.iloc[0]["Dataset"]) == "v5_joint_missingness_pattern"
    assert "Not available" not in str(df.iloc[0]["Metric value"])
