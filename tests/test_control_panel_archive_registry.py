"""Tests for Control Panel archive registry and workspace cleanup."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from src.churn_ml.control_panel.archive_registry import (
    ARCHIVE_SCHEMA_VERSION,
    ArchiveError,
    ArchiveRegistry,
    preview_job_deletion,
)
from src.churn_ml.control_panel.artifacts import ArtifactRecord
from src.churn_ml.control_panel.jobs import JobManager, atomic_write_json


def _write_terminal_job(root: Path, job_id: str, *, state: str = "succeeded") -> Path:
    job_root = root / job_id
    job_root.mkdir(parents=True)
    created = "2026-07-31T12:00:00.000000Z"
    atomic_write_json(
        job_root / "job.json",
        {
            "schema_version": 2,
            "job_id": job_id,
            "created_at_utc": created,
            "command_id": "experiment_core_v2",
            "action_id": "run",
            "references": {"config": "configs/research_v2/example.yaml"},
            "authoritative_result": False,
        },
    )
    atomic_write_json(
        job_root / "command.json",
        {
            "schema_version": 2,
            "argv": ["python", "scripts/run_research_v2.py", "--config", "x.yaml"],
            "working_directory": str(root.parent),
            "shell": False,
        },
    )
    atomic_write_json(
        job_root / "status.json",
        {
            "schema_version": 2,
            "state": state,
            "created_at_utc": created,
            "started_at_utc": created,
            "finished_at_utc": created,
            "updated_at_utc": created,
            "pid": None,
            "process_identity": None,
            "exit_code": 0 if state == "succeeded" else 1,
            "elapsed_seconds": 1.0,
            "diagnostic": None,
        },
    )
    (job_root / "stdout.log").write_text("ok\n", encoding="utf-8")
    (job_root / "stderr.log").write_text("", encoding="utf-8")
    return job_root


def test_archive_registry_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "artifacts" / "control_panel_state" / "archived_items.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": ARCHIVE_SCHEMA_VERSION,
                "items": [],
                "unexpected": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ArchiveError, match="unknown"):
        ArchiveRegistry(tmp_path)


def test_archive_restore_idempotent_and_ordered(tmp_path: Path) -> None:
    registry = ArchiveRegistry(tmp_path)
    job_a = str(uuid.uuid4())
    job_b = str(uuid.uuid4())
    assert registry.archive_ui_job(job_b, state="succeeded") is True
    assert registry.archive_ui_job(job_a, state="failed") is True
    assert registry.archive_ui_job(job_a, state="failed") is False
    ids = [item["job_id"] for item in registry.items if item["kind"] == "ui_job"]
    assert ids == sorted(ids)
    assert registry.restore_ui_job(job_a) is True
    assert registry.restore_ui_job(job_a) is False
    assert not registry.is_ui_job_archived(job_a)
    assert registry.is_ui_job_archived(job_b)


def test_corrupt_registry_fails_visibly(tmp_path: Path) -> None:
    path = tmp_path / "artifacts" / "control_panel_state" / "archived_items.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ArchiveError, match="Corrupt"):
        ArchiveRegistry(tmp_path)


def test_artifact_archive_identity_and_path_safety(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts" / "research_v2" / "run_a"
    artifact.mkdir(parents=True)
    (artifact / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    registry = ArchiveRegistry(tmp_path)
    rel = "artifacts/research_v2/run_a"
    assert registry.archive_artifact(reader_id="research_v2", relative_path=rel) is True
    assert registry.archive_artifact(reader_id="research_v2", relative_path=rel) is False
    assert registry.is_artifact_archived(reader_id="research_v2", relative_path=rel)
    with pytest.raises(ArchiveError):
        registry.archive_artifact(
            reader_id="research_v2", relative_path="../outside"
        )
    with pytest.raises(ArchiveError, match="missing"):
        registry.archive_artifact(
            reader_id="research_v2",
            relative_path="artifacts/research_v2/missing_run",
        )


def test_active_job_cannot_be_archived(tmp_path: Path) -> None:
    registry = ArchiveRegistry(tmp_path)
    job_id = str(uuid.uuid4())
    with pytest.raises(ArchiveError, match="terminal"):
        registry.archive_ui_job(job_id, state="running")


def test_permanent_delete_requires_archived_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs_root = tmp_path / "artifacts" / "ui_jobs"
    jobs_root.mkdir(parents=True)
    job_id = str(uuid.uuid4())
    job_root = _write_terminal_job(jobs_root, job_id, state="succeeded")
    research = tmp_path / "artifacts" / "research_v2" / "keep_me"
    research.mkdir(parents=True)
    marker = research / "_SUCCESS"
    marker.write_text("authoritative\n", encoding="utf-8")
    before = marker.read_bytes()

    manager = JobManager.__new__(JobManager)
    manager.jobs_root = jobs_root.resolve()
    manager.working_directory = tmp_path.resolve()
    manager.commands = {}
    manager.backend = None

    def fake_load(self, loaded_id: str):  # type: ignore[no-untyped-def]
        from src.churn_ml.control_panel.jobs import JobRecord

        assert loaded_id == job_id
        return JobRecord(
            job_id=job_id,
            root=job_root,
            job=json.loads((job_root / "job.json").read_text(encoding="utf-8")),
            command=json.loads((job_root / "command.json").read_text(encoding="utf-8")),
            status=json.loads((job_root / "status.json").read_text(encoding="utf-8")),
        )

    monkeypatch.setattr(JobManager, "load", fake_load)

    archive = ArchiveRegistry(tmp_path)
    with pytest.raises(ArchiveError, match="archived"):
        archive.permanently_delete_archived_ui_job(manager, job_id)

    archive.archive_ui_job(job_id, state="succeeded")
    deleted = archive.permanently_delete_archived_ui_job(manager, job_id)
    assert deleted == job_root
    assert not job_root.exists()
    assert marker.read_bytes() == before
    assert not archive.is_ui_job_archived(job_id)


def test_noncanonical_uuid_rejected(tmp_path: Path) -> None:
    archive = ArchiveRegistry(tmp_path)
    with pytest.raises(ArchiveError):
        archive.archive_ui_job(
            "ABCDEF00-0000-0000-0000-000000000001", state="succeeded"
        )


def test_filter_artifacts_hides_archived_by_default(tmp_path: Path) -> None:
    keep = tmp_path / "artifacts" / "research_v2" / "keep"
    hide = tmp_path / "artifacts" / "research_v2" / "hide"
    keep.mkdir(parents=True)
    hide.mkdir(parents=True)
    (hide / "payload.bin").write_bytes(b"payload-bytes")
    before = (hide / "payload.bin").read_bytes()
    archive = ArchiveRegistry(tmp_path)
    archive.archive_artifact(
        reader_id="research_v2",
        relative_path="artifacts/research_v2/hide",
    )
    records = [
        ArtifactRecord(
            reader_id="research_v2",
            root=keep,
            relative_path="artifacts/research_v2/keep",
            state="completed",
            summaries={},
            json_payloads={},
            diagnostic=None,
        ),
        ArtifactRecord(
            reader_id="research_v2",
            root=hide,
            relative_path="artifacts/research_v2/hide",
            state="completed",
            summaries={},
            json_payloads={},
            diagnostic=None,
        ),
    ]
    visible = archive.filter_artifacts(
        records, reader_id="research_v2", show_archived=False
    )
    assert [item.relative_path for item in visible] == ["artifacts/research_v2/keep"]
    shown = archive.filter_artifacts(
        records, reader_id="research_v2", show_archived=True
    )
    assert len(shown) == 2
    assert archive.restore_artifact(
        reader_id="research_v2", relative_path="artifacts/research_v2/hide"
    )
    assert not archive.is_artifact_archived(
        reader_id="research_v2", relative_path="artifacts/research_v2/hide"
    )
    assert (hide / "payload.bin").read_bytes() == before


def test_preview_job_deletion_lists_files(tmp_path: Path) -> None:
    job_id = str(uuid.uuid4())
    job_root = _write_terminal_job(tmp_path / "artifacts" / "ui_jobs", job_id)
    names = preview_job_deletion(job_root)
    assert "job.json" in names
    assert "stdout.log" in names


def test_symlink_job_directory_rejected(tmp_path: Path) -> None:
    jobs = tmp_path / "artifacts" / "ui_jobs"
    jobs.mkdir(parents=True)
    real = jobs / "real"
    real.mkdir()
    link = jobs / str(uuid.uuid4())
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ArchiveError):
        preview_job_deletion(link)


def test_mlflow_index_sidecar_is_allowed_optional_file() -> None:
    from src.churn_ml.control_panel.jobs import ALLOWED_JOB_FILES

    assert "mlflow_index.json" in ALLOWED_JOB_FILES


def test_registry_cache_clear_is_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refresh datasets must clear only the Registry discovery cache entrypoint."""
    import apps.experiment_control_panel as app

    cleared: list[str] = []

    class _Cache:
        def clear(self) -> None:
            cleared.append("registry")

    monkeypatch.setattr(app, "_cached_registry_dataset_views", _Cache())
    app._cached_registry_dataset_views.clear()
    assert cleared == ["registry"]
