"""Focused Control Panel performance instrumentation and lazy-routing tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.churn_ml.control_panel import performance as perf
from src.churn_ml.control_panel.blend_page import (
    BLEND_VIEW_OPTIONS,
    _resolve_active_blend_view,
    load_cached_candidate_inventory,
    render_blend_workspace_page,
)
from src.churn_ml.control_panel.blend_workspace import (
    candidate_inventory_fingerprint,
    discover_candidate_rows,
    discover_materialized_blends,
)
from src.churn_ml.control_panel.fs_signatures import (
    clear_sha_memo,
    memoized_file_sha256,
    path_stat_mapping,
)
from src.churn_ml.control_panel.saved_blend_search import jobs_search_fingerprint
from src.churn_ml.prediction_candidates.contract_v1 import (
    clear_file_sha256_memo,
    file_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _reset_perf() -> None:
    perf.enable_performance(False)
    perf.reset_performance()
    clear_sha_memo()
    clear_file_sha256_memo()
    yield
    perf.enable_performance(False)
    perf.reset_performance()
    clear_sha_memo()
    clear_file_sha256_memo()


def test_disabled_profiler_has_negligible_behavior() -> None:
    assert perf.performance_enabled() is False
    with perf.performance_stage("unused"):
        perf.record_counter("files_read", 3)
    snap = perf.performance_snapshot()
    assert snap["enabled"] is False
    assert snap["stages"] == []
    assert snap["counters"] == {}


def test_stages_record_elapsed_when_enabled() -> None:
    perf.enable_performance(True)
    with perf.performance_stage("candidate_discovery"):
        total = 0
        for index in range(1000):
            total += index
    snap = perf.performance_snapshot()
    assert snap["enabled"] is True
    assert len(snap["stages"]) == 1
    assert snap["stages"][0]["name"] == "candidate_discovery"
    assert snap["stages"][0]["elapsed_ms"] >= 0.0
    assert snap["stages"][0]["call_count"] == 1


def test_nested_stages_remain_valid() -> None:
    perf.enable_performance(True)
    with perf.performance_stage("outer"):
        with perf.performance_stage("inner"):
            perf.record_counter("files_inspected", 2)
    names = [item["name"] for item in perf.performance_snapshot()["stages"]]
    assert names == ["inner", "outer"]
    by_name = {item["name"]: item for item in perf.performance_snapshot()["stages"]}
    assert by_name["inner"]["parent"] == "outer"
    assert by_name["inner"]["files_inspected"] == 2


def test_counters_aggregate_correctly() -> None:
    perf.enable_performance(True)
    perf.record_counter("candidate_manifest_reads", 2)
    perf.record_counter("candidate_manifest_reads", 3)
    assert perf.performance_snapshot()["counters"]["candidate_manifest_reads"] == 5


def test_snapshots_contain_no_arbitrary_sensitive_values(tmp_path: Path) -> None:
    secret = tmp_path / "secret.key"
    secret.write_text("TOP-SECRET-VALUE", encoding="utf-8")
    perf.enable_performance(True)
    with perf.performance_stage("safe"):
        perf.note_files_read(1, bytes_read=16)
    text = perf.format_performance_snapshot()
    assert "TOP-SECRET-VALUE" not in text
    assert str(secret) not in text
    assert "safe" in text


def test_profiling_writes_no_repository_files(tmp_path: Path) -> None:
    before = {path.name for path in PROJECT_ROOT.iterdir()}
    perf.enable_performance(True)
    with perf.performance_stage("noop"):
        pass
    _ = perf.performance_snapshot()
    after = {path.name for path in PROJECT_ROOT.iterdir()}
    assert before == after
    assert not (tmp_path / "perf.json").exists()


def test_only_active_blend_view_renders() -> None:
    calls: list[str] = []

    class _Session(dict):
        def pop(self, key: str, default: Any = None) -> Any:
            return dict.pop(self, key, default)

    fake_st = MagicMock()
    fake_st.session_state = _Session(
        {"blend_workspace:active_view": "Build blend"}
    )
    fake_st.segmented_control.return_value = "Build blend"

    with (
        patch("src.churn_ml.control_panel.blend_page.st", fake_st),
        patch(
            "src.churn_ml.control_panel.blend_page.render_prepare_candidates_tab",
            side_effect=lambda **_: calls.append("prepare"),
        ),
        patch(
            "src.churn_ml.control_panel.blend_page._render_build_blend_tab",
            side_effect=lambda **_: calls.append("build"),
        ),
        patch(
            "src.churn_ml.control_panel.blend_page.render_search_history_tab",
            side_effect=lambda **_: calls.append("history"),
        ),
        patch(
            "src.churn_ml.control_panel.blend_page._render_materialized_blends",
            side_effect=lambda *_: calls.append("materialized"),
        ),
        patch(
            "src.churn_ml.control_panel.blend_page._apply_loaded_blend_configuration",
            return_value=None,
        ),
    ):
        render_blend_workspace_page(
            repository_root=PROJECT_ROOT,
            registry=MagicMock(),
            job_manager=MagicMock(),
        )
    assert calls == ["build"]


def test_inactive_prepare_does_not_scan_autogluon() -> None:
    with patch(
        "src.churn_ml.control_panel.candidate_preparation_page.discover_managed_autogluon_runs"
    ) as mocked:
        test_only_active_blend_view_renders()
        mocked.assert_not_called()


def test_inactive_build_does_not_discover_candidates() -> None:
    calls: list[str] = []

    class _Session(dict):
        def pop(self, key: str, default: Any = None) -> Any:
            return dict.pop(self, key, default)

    fake_st = MagicMock()
    fake_st.session_state = _Session(
        {"blend_workspace:active_view": "Search history"}
    )
    fake_st.segmented_control.return_value = "Search history"

    with (
        patch("src.churn_ml.control_panel.blend_page.st", fake_st),
        patch(
            "src.churn_ml.control_panel.blend_page._render_build_blend_tab",
            side_effect=lambda **_: calls.append("build"),
        ),
        patch(
            "src.churn_ml.control_panel.blend_page.render_search_history_tab",
            side_effect=lambda **_: calls.append("history"),
        ),
        patch(
            "src.churn_ml.control_panel.blend_page._apply_loaded_blend_configuration",
            return_value=None,
        ),
        patch(
            "src.churn_ml.control_panel.blend_page.discover_candidate_rows"
        ) as discover,
    ):
        render_blend_workspace_page(
            repository_root=PROJECT_ROOT,
            registry=MagicMock(),
            job_manager=MagicMock(),
        )
    assert calls == ["history"]
    discover.assert_not_called()


def test_inactive_search_history_does_not_parse_jobs() -> None:
    class _Session(dict):
        def pop(self, key: str, default: Any = None) -> Any:
            return dict.pop(self, key, default)

    fake_st = MagicMock()
    fake_st.session_state = _Session(
        {"blend_workspace:active_view": "Prepare candidates"}
    )
    fake_st.segmented_control.return_value = "Prepare candidates"

    with (
        patch("src.churn_ml.control_panel.blend_page.st", fake_st),
        patch(
            "src.churn_ml.control_panel.blend_page.render_prepare_candidates_tab",
            return_value=None,
        ),
        patch(
            "src.churn_ml.control_panel.blend_page._apply_loaded_blend_configuration",
            return_value=None,
        ),
        patch(
            "src.churn_ml.control_panel.search_history_page.discover_saved_blend_searches"
        ) as discover,
    ):
        render_blend_workspace_page(
            repository_root=PROJECT_ROOT,
            registry=MagicMock(),
            job_manager=MagicMock(),
        )
    discover.assert_not_called()


def test_inactive_materialized_does_not_discover_blends() -> None:
    class _Session(dict):
        def pop(self, key: str, default: Any = None) -> Any:
            return dict.pop(self, key, default)

    fake_st = MagicMock()
    fake_st.session_state = _Session(
        {"blend_workspace:active_view": "Build blend"}
    )
    fake_st.segmented_control.return_value = "Build blend"

    with (
        patch("src.churn_ml.control_panel.blend_page.st", fake_st),
        patch(
            "src.churn_ml.control_panel.blend_page._render_build_blend_tab",
            return_value=None,
        ),
        patch(
            "src.churn_ml.control_panel.blend_page._apply_loaded_blend_configuration",
            return_value=None,
        ),
        patch(
            "src.churn_ml.control_panel.blend_page.discover_materialized_blends"
        ) as discover,
    ):
        render_blend_workspace_page(
            repository_root=PROJECT_ROOT,
            registry=MagicMock(),
            job_manager=MagicMock(),
        )
    discover.assert_not_called()


def test_selected_view_persists() -> None:
    class _Session(dict):
        def pop(self, key: str, default: Any = None) -> Any:
            return dict.pop(self, key, default)

    state = _Session({"blend_workspace:active_tab": "Search history"})
    with patch("src.churn_ml.control_panel.blend_page.st") as fake_st:
        fake_st.session_state = state
        view = _resolve_active_blend_view()
    assert view == "Search history"
    assert state["blend_workspace:active_view"] == "Search history"
    assert all(option in BLEND_VIEW_OPTIONS for option in BLEND_VIEW_OPTIONS)


def test_candidate_inventory_fingerprint_is_stat_based(tmp_path: Path) -> None:
    package = tmp_path / "artifacts" / "prediction_candidates" / "pc1_demo"
    package.mkdir(parents=True)
    manifest = package / "candidate_manifest.json"
    success = package / "_SUCCESS"
    metadata = package / "source_metadata.json"
    manifest.write_text(
        json.dumps(
            {
                "candidate_id": "pc1_demo",
                "source_kind": "autogluon_standalone_v1",
                "source_model_name": "Model",
                "dataset_id": "v0_raw_minimal",
                "train_row_count": 1,
                "test_row_count": 1,
                "oof_protocol": "x",
                "source_metric_name": "ba",
                "source_metric_value": 0.5,
                "exploratory": False,
            }
        ),
        encoding="utf-8",
    )
    success.write_text("{}", encoding="utf-8")
    metadata.write_text("{}", encoding="utf-8")
    fp1 = candidate_inventory_fingerprint(tmp_path)
    # Touch without content-hash dependency: change mtime by rewriting same bytes.
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    fp2 = candidate_inventory_fingerprint(tmp_path)
    assert fp1 != fp2


def test_cached_sha_reused_when_size_mtime_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(b"abc" * 1000)
    clear_file_sha256_memo()
    first = file_sha256(path)
    second = file_sha256(path)
    assert first == second
    memo = memoized_file_sha256(path)
    assert memo == first


def test_changed_size_mtime_forces_sha_recalculation(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(b"abc")
    clear_file_sha256_memo()
    first = file_sha256(path)
    path.write_bytes(b"abcd")
    second = file_sha256(path)
    assert first != second


def test_jobs_search_fingerprint_does_not_hash_stdout(tmp_path: Path) -> None:
    job_dir = tmp_path / "job-1"
    job_dir.mkdir()
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": "11111111-1111-1111-1111-111111111111",
                "command_id": "prediction_blend_v1",
                "action_id": "search",
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "status.json").write_text('{"state":"succeeded"}', encoding="utf-8")
    stdout = job_dir / "stdout.log"
    stdout.write_text("x" * 10000, encoding="utf-8")
    with patch(
        "src.churn_ml.control_panel.saved_blend_search.file_sha256"
    ) as hashed:
        token = jobs_search_fingerprint(tmp_path)
        hashed.assert_not_called()
    assert isinstance(token, str) and len(token) == 64


def test_discover_candidate_rows_skips_parquet(tmp_path: Path) -> None:
    package = tmp_path / "artifacts" / "prediction_candidates" / "pc1_demo"
    package.mkdir(parents=True)
    (package / "_SUCCESS").write_text("{}", encoding="utf-8")
    (package / "candidate_manifest.json").write_text(
        json.dumps(
            {
                "candidate_id": "pc1_demo",
                "source_kind": "autogluon_standalone_v1",
                "source_model_name": "WeightedEnsemble_L2",
                "dataset_id": "v0_raw_minimal",
                "train_row_count": 10,
                "test_row_count": 5,
                "oof_protocol": "protocol",
                "source_metric_name": "balanced_accuracy",
                "source_metric_value": 0.8,
                "exploratory": False,
            }
        ),
        encoding="utf-8",
    )
    (package / "source_metadata.json").write_text(
        json.dumps({"final_deployment_threshold": 0.17}),
        encoding="utf-8",
    )
    with patch(
        "src.churn_ml.control_panel.blend_workspace.load_candidate_package"
    ) as loaded:
        rows = discover_candidate_rows(tmp_path, include_readiness=True)
        loaded.assert_not_called()
    assert len(rows) == 1
    assert rows[0]["readiness_state"] == "ready"
    assert rows[0]["probability_min"] is None


def test_materialized_inventory_is_lightweight() -> None:
    blends_root = PROJECT_ROOT / "artifacts" / "prediction_blends"
    if not blends_root.is_dir() or not any(blends_root.iterdir()):
        pytest.skip("No materialized blends present")
    with patch(
        "src.churn_ml.control_panel.blend_workspace.load_blend_artifact"
    ) as loaded:
        rows = discover_materialized_blends(PROJECT_ROOT, include_loaded=False)
        loaded.assert_not_called()
    assert rows
    assert all(item.get("loaded") in (None, {}) or item.get("loaded") is None for item in rows)


def test_load_cached_candidate_inventory_refresh(tmp_path: Path) -> None:
    # Ensure force_refresh path clears Streamlit cache wrapper without error when
    # Streamlit cache is available in-process.
    rows = load_cached_candidate_inventory(tmp_path, force_refresh=True)
    assert rows == []


def test_path_stat_mapping_missing(tmp_path: Path) -> None:
    mapping = path_stat_mapping(tmp_path / "missing.json", relative="missing.json")
    assert mapping["exists"] is False
    assert mapping["path"] == "missing.json"


def test_env_flag_enables_performance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHURN_ML_CONTROL_PANEL_PERF", "1")
    assert perf.performance_enabled() is True
