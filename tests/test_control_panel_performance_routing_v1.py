"""Routing/cache regressions for Control Panel performance work."""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def panel():
    return importlib.import_module("apps.experiment_control_panel")


def test_canonical_submission_skips_research_inventory(panel) -> None:
    with (
        patch.object(panel, "load_cached_candidate_inventory", return_value=[]) as inv,
        patch.object(panel, "_cached_research_inventory_rows") as research,
        patch.object(panel, "discover_registry_summaries") as registry,
        patch.object(panel, "st") as fake_st,
    ):
        fake_st.session_state = {}
        fake_st.button.return_value = False
        panel._canonical_candidate_submission_controls(
            MagicMock(), action_id="generate", handoff=None
        )
        inv.assert_called_once()
        research.assert_not_called()
        registry.assert_not_called()


def test_canonical_checkbox_uses_cached_eligibility_not_strict(panel) -> None:
    row = {
        "candidate_id": "pc1_demo",
        "label": "Demo",
        "dataset_label": "v0",
        "dataset_id": "v0_raw_minimal",
        "source_kind_label": "AutoGluon",
        "exploratory": False,
        "manifest_sha256": "abc",
    }
    eligibility = {
        "candidate_id": "pc1_demo",
        "state": "eligible_to_attempt",
        "eligible_to_attempt": True,
        "source_kind": "autogluon_standalone_v1",
        "exploratory": False,
        "threshold": 0.17,
        "test_row_count": 2500,
        "dataset_id": "v0_raw_minimal",
        "blend_id": None,
        "parent_candidate_ids": [],
        "blockers": [],
        "warnings": [],
        "strict_validation_required": True,
        "note": "Metadata checks passed.",
    }
    with (
        patch.object(panel, "load_cached_candidate_inventory", return_value=[row]),
        patch.object(
            panel,
            "_cached_selected_submission_eligibility",
            return_value=eligibility,
        ) as cached,
        patch.object(
            panel,
            "eligibility_cache_fingerprint",
            return_value="fp",
        ),
        patch(
            "src.churn_ml.prediction_candidates.submission_v1.evaluate_submission_readiness",
            side_effect=AssertionError("strict readiness must not run on UI reruns"),
        ),
        patch.object(panel, "st") as fake_st,
    ):
        fake_st.session_state = {
            "canonical-submission-candidate": "pc1_demo",
            "_consumed_launch_nonces": set(),
        }
        fake_st.button.return_value = False
        fake_st.selectbox.return_value = "pc1_demo"
        fake_st.checkbox.return_value = True
        fake_st.text_input.return_value = "sub_demo"
        panel._canonical_candidate_submission_controls(
            MagicMock(), action_id="generate", handoff=None
        )
        assert cached.call_count == 1


def test_jobs_list_uses_refresh_false(panel) -> None:
    manager = MagicMock()
    manager.list_jobs.return_value = []
    loaded = MagicMock()
    loaded.settings.jobs_root = "artifacts/ui_jobs"
    loaded.settings.working_directory = "."
    loaded.commands = {}
    with (
        patch.object(panel, "registry", return_value=loaded),
        patch.object(panel, "job_manager", return_value=manager),
        patch.object(panel, "ArchiveRegistry") as archive_cls,
        patch.object(panel, "st") as fake_st,
    ):
        archive_cls.return_value.archived_ui_job_ids.return_value = set()
        fake_st.session_state = {}
        fake_st.checkbox.return_value = False
        fake_st.button.return_value = False
        panel.jobs_page()
    manager.list_jobs.assert_called_with(refresh=False)


def test_results_only_selected_view_runs(panel) -> None:
    loaded = MagicMock()
    loaded.settings.mlflow_url = None
    loaded.readers = {"r1": MagicMock(id="r1", title="R1")}
    calls: list[str] = []
    with (
        patch.object(panel, "registry", return_value=loaded),
        patch.object(panel, "ArchiveRegistry"),
        patch.object(panel, "st") as fake_st,
        patch.object(
            panel,
            "_results_model_runs_tab",
            side_effect=lambda *a, **k: calls.append("model_runs"),
        ),
        patch.object(
            panel,
            "_results_entity_tab",
            side_effect=lambda *a, **k: calls.append("entity"),
        ),
        patch.object(
            panel,
            "_results_inspect_tab",
            side_effect=lambda *a, **k: calls.append("inspect"),
        ),
        patch.object(
            panel,
            "_results_research_workspace_tab",
            side_effect=lambda *a, **k: calls.append("research"),
        ),
    ):
        fake_st.session_state = {"results-active-view": "Inspect artifact"}
        fake_st.checkbox.return_value = False
        fake_st.segmented_control.return_value = "Inspect artifact"
        panel.results_page()
    assert calls == ["inspect"]


def test_dashboard_jobs_list_is_lightweight(panel) -> None:
    loaded = MagicMock()
    loaded.settings.mlflow_url = None
    loaded.commands = {}
    loaded.readers = {}
    manager = MagicMock()
    manager.list_jobs.return_value = []
    metric = MagicMock()
    with (
        patch.object(panel, "registry", return_value=loaded),
        patch.object(panel, "job_manager", return_value=manager),
        patch.object(panel, "_all_artifacts", return_value=[]),
        patch.object(panel, "st") as fake_st,
    ):
        fake_st.session_state = {}
        fake_st.columns.return_value = [metric, metric, metric, metric]
        panel.dashboard_page()
    manager.list_jobs.assert_called_with(refresh=False)


def test_configuration_page_exposes_perf_toggle(panel) -> None:
    loaded = panel.registry()
    with patch.object(panel, "st") as fake_st:
        fake_st.session_state = {}
        fake_st.button.return_value = False
        fake_st.checkbox.return_value = False
        panel.configuration_page()
    fake_st.checkbox.assert_any_call(
        "Enable performance diagnostics (session)",
        value=False,
        key="config-perf-enabled",
        help=(
            "Records stage timings in memory for this Streamlit process. "
            "Also enabled by CHURN_ML_CONTROL_PANEL_PERF=1. Does not write "
            "profiling files under artifacts/."
        ),
    )
    assert loaded is not None


def test_native_candidate_and_submission_remain_discoverable() -> None:
    from src.churn_ml.control_panel.blend_workspace import discover_candidate_rows
    from src.churn_ml.control_panel.blend_workspace import discover_materialized_blends

    candidates = discover_candidate_rows(PROJECT_ROOT, include_readiness=True)
    assert any(row["candidate_id"] == "pc1_c54c97e9c32f33a2" for row in candidates)
    blends = discover_materialized_blends(PROJECT_ROOT, include_loaded=False)
    assert any(item.get("blend_id") == "pb1_d047ebe82811ef1a" for item in blends)
    submission = (
        PROJECT_ROOT
        / "artifacts"
        / "candidate_submissions"
        / "sub_Native-blend-3-models-v5-v6_c54c97e9"
    )
    assert (submission / "_SUCCESS").is_file()
    assert (submission / "submission.csv").is_file()


def test_saved_blend_search_recovery_still_importable() -> None:
    from src.churn_ml.control_panel.saved_blend_search import (
        discover_saved_blend_searches,
        recover_search_for_request,
    )

    assert callable(discover_saved_blend_searches)
    assert callable(recover_search_for_request)


def test_candidate_preparation_recovery_still_importable() -> None:
    from src.churn_ml.control_panel.candidate_preparation import (
        discover_managed_autogluon_runs,
        restore_selection_from_jobs,
    )

    assert callable(discover_managed_autogluon_runs)
    assert callable(restore_selection_from_jobs)
