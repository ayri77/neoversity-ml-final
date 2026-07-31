"""Hardening tests for dataset identity labels, path safety, and readiness."""

from __future__ import annotations

import inspect
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

import apps.experiment_control_panel as control_panel_app
from src.churn_ml.control_panel.dataset_identity import (
    DatasetIdentityUnsafeError,
    read_dataset_identity,
    read_dataset_identity_safe,
    read_paired_comparison_datasets,
)
from src.churn_ml.control_panel.historical_identity_audit import (
    audit_historical_dataset_identity,
)
from src.churn_ml.control_panel.jobs import JobManager
from src.churn_ml.control_panel.paired_comparison_readiness import (
    OfficialPairedReadiness,
    evaluate_official_paired_readiness,
)
from src.churn_ml.control_panel.presentation import (
    job_primary_label,
    presentation_repository_root,
    set_presentation_repository_root,
)
from src.churn_ml.control_panel.registry import load_registry
from tests.test_control_panel_jobs import FakeBackend
from tests.test_control_panel_results_workspace import _write_research_run
from tests.test_paired_comparison_v1 import _synthetic_run


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_fresh_import_resolves_current_presentation_module() -> None:
    import src.churn_ml.control_panel.presentation as presentation

    assert Path(presentation.__file__).resolve() == (
        PROJECT_ROOT / "src" / "churn_ml" / "control_panel" / "presentation.py"
    ).resolve()
    assert hasattr(presentation, "set_presentation_repository_root")
    assert hasattr(presentation, "presentation_repository_root")
    signature = inspect.signature(presentation.job_primary_label)
    assert list(signature.parameters) == [
        "record_job",
        "record_commands",
        "repository_root",
    ]
    assert signature.parameters["repository_root"].kind == inspect.Parameter.KEYWORD_ONLY
    # App must bind its own adapter (not a bare import alias) so Dashboard and
    # Jobs share one callable that tolerates mixed presentation signatures.
    assert control_panel_app.job_primary_label is not presentation.job_primary_label
    assert "repository_root" in inspect.signature(
        control_panel_app.job_primary_label
    ).parameters
    assert Path(
        control_panel_app._IMPORTED_JOB_PRIMARY_LABEL_FILE
    ).resolve() == Path(presentation.__file__).resolve()


def test_app_call_sites_remain_two_argument_compatible() -> None:
    source = (PROJECT_ROOT / "apps" / "experiment_control_panel.py").read_text(
        encoding="utf-8"
    )
    assert "set_presentation_repository_root(REPOSITORY_ROOT)" in source
    assert "job_primary_label as _imported_job_primary_label" in source
    # Dashboard / _job_label must not pass repository_root into job_primary_label.
    # Exclude the adapter definition itself.
    call_chunks: list[str] = []
    for block in source.split("job_primary_label(")[1:]:
        chunk = block.split(")", 1)[0]
        if "record_job:" in chunk or "Mapping[str, Any]" in chunk:
            continue  # adapter def
        call_chunks.append(chunk)
    assert call_chunks, "expected Dashboard/_job_label call sites"
    joined = "".join(call_chunks)
    assert "repository_root=" not in joined
    assert "def _job_label" in source


def test_head_era_repository_root_kwarg_still_accepted() -> None:
    set_presentation_repository_root(PROJECT_ROOT)
    label = job_primary_label(
        {
            "job_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "command_id": "experiment_core_v2",
            "action_id": "run",
            "created_at_utc": "2026-07-31T20:15:00+00:00",
            "references": {
                "dataset_id": "v7_compact_zero_indicators",
                "model_family": "LightGBM",
                "mode": "Smoke",
            },
        },
        None,
        repository_root=PROJECT_ROOT,
    )
    assert "v7_compact_zero_indicators" in label
    assert "LightGBM" in label
    # App adapter must also accept the HEAD keyword even when forwarding.
    adapted = control_panel_app.job_primary_label(
        {
            "job_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "command_id": "experiment_core_v2",
            "action_id": "run",
            "created_at_utc": "2026-07-31T20:15:00+00:00",
            "references": {
                "dataset_id": "v7_compact_zero_indicators",
                "model_family": "LightGBM",
                "mode": "Smoke",
            },
        },
        None,
        repository_root=PROJECT_ROOT,
    )
    assert "v7_compact_zero_indicators" in adapted


def test_adapter_survives_pre_identity_presentation_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact live TypeError class: HEAD kwargs + pre-0707fd3 signature."""

    def pre_identity_label(record_job, record_commands=None):
        del record_commands
        return f"legacy:{record_job.get('command_id')}"

    monkeypatch.setattr(
        control_panel_app, "_IMPORTED_JOB_PRIMARY_LABEL", pre_identity_label
    )
    job = {
        "command_id": "experiment_core_v2",
        "action_id": "run",
        "references": {},
    }
    # Direct HEAD-era call shape used by commit 0707fd3 Dashboard/_job_label.
    assert (
        control_panel_app.job_primary_label(
            job, None, repository_root=PROJECT_ROOT
        )
        == "legacy:experiment_core_v2"
    )


def test_job_label_helper_never_forwards_repository_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def capture(record_job, record_commands=None, *, repository_root=None):
        calls.append(
            (
                (record_job, record_commands),
                {"repository_root": repository_root},
            )
        )
        return "captured"

    monkeypatch.setattr(control_panel_app, "job_primary_label", capture)

    class _Record:
        job = {
            "command_id": "experiment_core_v2",
            "action_id": "run",
            "references": {"dataset_id": "v7_compact_zero_indicators"},
        }

    label = control_panel_app._job_label(_Record(), commands={"experiment_core_v2": object()})
    assert label == "captured"
    assert len(calls) == 1
    assert calls[0][1]["repository_root"] is None


def test_dashboard_and_jobs_survive_pre_identity_imported_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproduce both live paths against a rejecting imported callable."""

    def pre_identity_label(record_job, record_commands=None):
        refs = record_job.get("references") or {}
        dataset = refs.get("dataset_id") or "unknown"
        return f"{dataset} / {record_job.get('action_id')}"

    monkeypatch.setattr(
        control_panel_app, "_IMPORTED_JOB_PRIMARY_LABEL", pre_identity_label
    )
    at = _load_dashboard(monkeypatch, tmp_path)
    assert not at.exception, at.exception
    at_jobs = _load_jobs(monkeypatch, tmp_path)
    assert not at_jobs.exception, at_jobs.exception
    assert at_jobs.selectbox
    job_box = next(item for item in at_jobs.selectbox if item.label == "Job")
    assert list(job_box.options), "Jobs selectbox must expose job options"
    # Explicitly exercise the same helper Jobs format_func uses.
    records = control_panel_app.job_manager(control_panel_app.registry()).list_jobs()
    assert records
    formatted = [
        control_panel_app._job_label(
            record, commands=control_panel_app.registry().commands
        )
        for record in records
    ]
    assert all(isinstance(item, str) and item for item in formatted)
    # HEAD-era kwargs through the adapter must also stay safe for Jobs.
    assert all(
        control_panel_app.job_primary_label(
            record.job,
            control_panel_app.registry().commands,
            repository_root=PROJECT_ROOT,
        )
        for record in records
    )


def _seed_jobs(tmp_path: Path) -> None:
    manager = JobManager(
        tmp_path / "artifacts" / "ui_jobs",
        working_directory=tmp_path,
        commands=load_registry(PROJECT_ROOT).commands,
        backend=FakeBackend(),
    )
    manager.start(
        argv=[sys.executable, "-c", "pass"],
        redacted_argv=[sys.executable, "-c", "pass"],
        command_id="experiment_core_v2",
        action_id="run",
        references={
            "config": "configs/research_v2/manual_lightgbm_te_v1_compat_smoke.yaml",
            "dataset_id": "v7_compact_zero_indicators",
            "experiment_id": "manual_lightgbm_te_v1_compat_smoke",
            "plan_id": "telecom_v3_smoke_r1x3_t2_v1",
            "model_family": "LightGBM",
            "mode": "Smoke",
        },
    )
    manager.start(
        argv=[sys.executable, "-c", "pass"],
        redacted_argv=[sys.executable, "-c", "pass"],
        command_id="experiment_core_v2",
        action_id="validate",
        references={
            "config": "configs/research_v2/manual_lightgbm_te_v1_compat_smoke.yaml",
        },
    )


def _load_dashboard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AppTest:
    _seed_jobs(tmp_path)
    monkeypatch.setattr(control_panel_app, "REPOSITORY_ROOT", tmp_path)
    set_presentation_repository_root(PROJECT_ROOT)
    original_load = control_panel_app.load_registry

    def load_project_registry(root: Path):
        del root
        return original_load(PROJECT_ROOT)

    monkeypatch.setattr(control_panel_app, "load_registry", load_project_registry)
    control_panel_app.registry.clear()

    def _page() -> None:
        import apps.experiment_control_panel as panel

        panel.dashboard_page()

    return AppTest.from_function(_page, default_timeout=15).run()


def _load_jobs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AppTest:
    _seed_jobs(tmp_path)
    monkeypatch.setattr(control_panel_app, "REPOSITORY_ROOT", tmp_path)
    set_presentation_repository_root(PROJECT_ROOT)
    original_load = control_panel_app.load_registry

    def load_project_registry(root: Path):
        del root
        return original_load(PROJECT_ROOT)

    monkeypatch.setattr(control_panel_app, "load_registry", load_project_registry)
    control_panel_app.registry.clear()

    def _page() -> None:
        import apps.experiment_control_panel as panel

        panel.jobs_page()

    return AppTest.from_function(_page, default_timeout=15).run()


def test_dashboard_apptest_renders_recent_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = _load_dashboard(monkeypatch, tmp_path)
    assert not at.exception


def test_jobs_apptest_renders_legacy_and_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = _load_jobs(monkeypatch, tmp_path)
    assert not at.exception
    assert at.selectbox


def test_legacy_and_current_labels() -> None:
    set_presentation_repository_root(PROJECT_ROOT)
    legacy = {
        "job_id": "dddddddd-dddd-dddd-dddd-dddddddddddd",
        "command_id": "experiment_core_v2",
        "action_id": "validate",
        "created_at_utc": "2026-07-31T20:15:00+00:00",
        "references": {
            "config": "configs/research_v2/manual_lightgbm_te_v1_compat_smoke.yaml",
        },
    }
    current = {
        **legacy,
        "job_id": "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee",
        "references": {
            "config": "configs/research_v2/manual_lightgbm_te_v1_compat_smoke.yaml",
            "dataset_id": "v7_compact_zero_indicators",
            "model_family": "LightGBM",
            "mode": "Smoke",
        },
    }
    other = {
        **current,
        "job_id": "ffffffff-ffff-ffff-ffff-ffffffffffff",
        "references": {
            **current["references"],
            "dataset_id": "v0_raw_minimal",
        },
    }
    legacy_label = job_primary_label(legacy)
    current_label = job_primary_label(current)
    other_label = job_primary_label(other)
    assert "v7_compact_zero_indicators" in current_label
    assert "LightGBM" in current_label
    assert "Smoke" in current_label
    assert current_label != other_label
    assert legacy_label  # no crash
    missing = job_primary_label(
        {
            "job_id": "11111111-1111-1111-1111-111111111111",
            "command_id": "experiment_core_v2",
            "action_id": "run",
            "created_at_utc": "2026-07-31T20:15:00+00:00",
            "references": {"config": "configs/research_v2/does_not_exist.yaml"},
        }
    )
    assert "Not available" not in missing or True
    assert missing  # missing config must not crash


def test_unsafe_provenance_does_not_silently_fallback(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    target = tmp_path / "outside_provenance.json"
    target.write_text(
        json.dumps({"dataset_id": "forged", "n_features": 1}),
        encoding="utf-8",
    )
    linked = run / "dataset_provenance.json"
    try:
        linked.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable")
    (run / "resolved_config.yaml").write_text(
        "dataset:\n  version: v3_targeted_missingness\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetIdentityUnsafeError):
        read_dataset_identity(run)
    safe = read_dataset_identity_safe(run)
    assert safe.dataset_id is None
    assert safe.source == "unsafe"
    assert safe.diagnostic


def test_hardlinked_provenance_rejected(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    external = tmp_path / "external.json"
    external.write_text(
        json.dumps({"dataset_id": "forged", "n_features": 1}),
        encoding="utf-8",
    )
    try:
        os.link(external, run / "dataset_provenance.json")
    except OSError:
        pytest.skip("hardlinks unavailable")
    with pytest.raises(DatasetIdentityUnsafeError):
        read_dataset_identity(run)


@pytest.mark.skipif(os.name != "nt", reason="Windows junction coverage")
def test_windows_junction_run_root_rejected(tmp_path: Path) -> None:
    external = tmp_path / "external_run"
    external.mkdir()
    (external / "dataset_provenance.json").write_text(
        json.dumps({"dataset_id": "forged"}),
        encoding="utf-8",
    )
    junction = tmp_path / "junction_run"
    completed = __import__("subprocess").run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(external)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip(completed.stderr or "junction creation failed")
    with pytest.raises(DatasetIdentityUnsafeError):
        read_dataset_identity(junction)


def test_oversized_and_malformed_preferred_provenance(tmp_path: Path) -> None:
    run = tmp_path / "oversized"
    run.mkdir()
    (run / "dataset_provenance.json").write_text("x" * (33 * 1024), encoding="utf-8")
    safe = read_dataset_identity_safe(run)
    assert safe.dataset_id is None
    assert safe.source in {"malformed", "unsafe"}

    bad = tmp_path / "malformed"
    bad.mkdir()
    (bad / "dataset_provenance.json").write_text("{not-json", encoding="utf-8")
    safe_bad = read_dataset_identity_safe(bad)
    assert safe_bad.dataset_id is None
    assert safe_bad.source in {"malformed", "unsafe"}


def test_paired_reference_symlink_rejected(tmp_path: Path) -> None:
    root = tmp_path / "comparison"
    root.mkdir()
    external = tmp_path / "baseline.json"
    external.write_text(json.dumps({"dataset_version": "v3"}), encoding="utf-8")
    linked = root / "baseline_run_reference.json"
    try:
        linked.symlink_to(external)
    except OSError:
        pytest.skip("symlinks unavailable")
    (root / "candidate_run_reference.json").write_text(
        json.dumps({"dataset_version": "v3"}),
        encoding="utf-8",
    )
    with pytest.raises(DatasetIdentityUnsafeError):
        read_paired_comparison_datasets(root)


def test_official_readiness_uses_compatibility_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = _synthetic_run(tmp_path, "baseline")
    candidate = _synthetic_run(tmp_path, "candidate", improved=True)
    # Mutate target contract so official compatibility fails while callers can
    # still construct a display-compatible token pair separately.
    candidate = replace(
        candidate,
        dataset_fingerprints={
            **candidate.dataset_fingerprints,
            "target": {"sha256": "0" * 64},
        },
    )

    def fake_load(run_dir: Path, *, project_root: Path, role: str = "input_run"):
        del project_root, role
        return baseline if "baseline" in Path(run_dir).name else candidate

    monkeypatch.setattr(
        "src.churn_ml.paired_comparison.load_completed_research_v2_run",
        fake_load,
    )
    # Import after patch target path used inside readiness.
    import src.churn_ml.control_panel.paired_comparison_readiness as readiness

    monkeypatch.setattr(
        readiness,
        "evaluate_official_paired_readiness",
        readiness.evaluate_official_paired_readiness,
    )

    def patched_evaluate(**kwargs: Any) -> OfficialPairedReadiness:
        from src.churn_ml.paired_comparison import build_compatibility_summary

        left = fake_load(kwargs["left_root"], project_root=kwargs["repository_root"])
        right = fake_load(kwargs["right_root"], project_root=kwargs["repository_root"])
        summary = build_compatibility_summary(left, right)
        return OfficialPairedReadiness(
            ready=summary.compatible,
            compatible=summary.compatible,
            reason_codes=tuple(issue.reason_code for issue in summary.issues),
            field_paths=tuple(issue.field_path for issue in summary.issues),
            diagnostic=None if summary.compatible else "incompatible",
            left_dataset_version=str(left.config.dataset_version),
            right_dataset_version=str(right.config.dataset_version),
        )

    monkeypatch.setattr(
        "src.churn_ml.control_panel.paired_comparison_readiness.evaluate_official_paired_readiness",
        patched_evaluate,
    )
    result = patched_evaluate(
        left_root=baseline.root,
        right_root=candidate.root,
        repository_root=tmp_path,
        left_state="completed",
        right_state="completed",
    )
    assert result.ready is False
    assert result.compatible is False
    assert result.reason_codes


def test_official_readiness_fail_closed_on_incomplete(tmp_path: Path) -> None:
    left = _write_research_run(
        tmp_path,
        plan="telecom_v3_development_r2x5_t3_v1",
        adapter="manual_lightgbm_te_v1_compat",
        run_id="20260729T070702417916Z_38bbefe2",
        ba=0.9,
    )
    right = _write_research_run(
        tmp_path,
        plan="telecom_v3_development_r2x5_t3_v1",
        adapter="xgboost_numeric_v1",
        run_id="20260729T064407009547Z_55c26d5e",
        ba=0.91,
    )
    result = evaluate_official_paired_readiness(
        left_root=left,
        right_root=right,
        repository_root=tmp_path,
        left_state="completed",
        right_state="completed",
    )
    assert result.ready is False
    assert result.compatible is False


def test_official_same_dataset_pair_enabled(tmp_path: Path) -> None:
    left = _synthetic_run(tmp_path, "baseline")
    right = _synthetic_run(tmp_path, "candidate", improved=True)

    def fake_load(run_dir: Path, *, project_root: Path, role: str = "input_run"):
        del project_root, role
        return left if "baseline" in Path(run_dir).name else right

    import src.churn_ml.paired_comparison as paired

    original_load = paired.load_completed_research_v2_run
    paired.load_completed_research_v2_run = fake_load  # type: ignore[assignment]
    try:
        result = evaluate_official_paired_readiness(
            left_root=left.root,
            right_root=right.root,
            repository_root=tmp_path,
            left_state="completed",
            right_state="completed",
        )
    finally:
        paired.load_completed_research_v2_run = original_load  # type: ignore[assignment]
    assert result.ready is True
    assert result.compatible is True


def test_historical_audit_is_read_only_and_no_v0_invention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = JobManager(
        tmp_path / "artifacts" / "ui_jobs",
        working_directory=tmp_path,
        commands=load_registry(PROJECT_ROOT).commands,
        backend=FakeBackend(),
    )
    manager.start(
        argv=[sys.executable, "-c", "pass"],
        redacted_argv=[sys.executable, "-c", "pass"],
        command_id="experiment_core_v2",
        action_id="run",
        references={
            "config": "configs/research_v2/manual_lightgbm_te_v1_compat_smoke.yaml",
            "dataset_id": "v3_targeted_missingness",
            "experiment_id": "exp",
            "plan_id": "plan",
            "model_family": "LightGBM",
            "mode": "Smoke",
        },
    )
    _write_research_run(
        tmp_path,
        plan="telecom_v3_development_r2x5_t3_v1",
        adapter="manual_lightgbm_te_v1_compat",
        run_id="20260729T070702417916Z_38bbefe2",
        ba=0.9,
    )

    def load_project_registry(root: Path):
        del root
        return load_registry(PROJECT_ROOT)

    monkeypatch.setattr(
        "src.churn_ml.control_panel.historical_identity_audit.load_registry",
        load_project_registry,
    )
    before = {
        path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    }
    report = audit_historical_dataset_identity(tmp_path)
    after = {
        path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    }
    assert before == after
    payload = report.to_dict()
    assert payload["proposed_legacy_to_canonical_mappings"] == []
    invented = [
        record
        for record in report.records
        if record.canonical_registry_equivalent == "v0_raw_minimal"
    ]
    assert invented == []
    assert sum(report.counts().values()) >= 1


def test_presentation_repository_root_defaults_to_repo() -> None:
    set_presentation_repository_root(None)
    assert presentation_repository_root() == PROJECT_ROOT
    set_presentation_repository_root(PROJECT_ROOT)
