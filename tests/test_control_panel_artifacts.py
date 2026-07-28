from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path

from src.churn_ml.control_panel.artifacts import (
    comparison_rows,
    csv_preview,
    discover_artifacts,
    extract_dot_path,
    text_tail,
)
from src.churn_ml.control_panel.schemas import ReaderSpec


def _reader(root_name: str) -> ReaderSpec:
    return ReaderSpec.from_dict(
        {
            "id": "test",
            "title": "Test reader",
            "artifact_roots": [root_name],
            "discovery_glob": "*",
            "success_markers": ["_SUCCESS"],
            "failure_markers": ["_FAILED"],
            "summary_files": [
                {
                    "path": "summary.json",
                    "fields": {
                        "Score": "metrics.score",
                        "First": "items.0.value",
                    },
                }
            ],
            "csv_previews": ["preview.csv"],
            "log_files": ["run.log"],
            "compare_fields": ["Score", "First"],
        },
        "reader",
    )


def test_artifact_discovery_markers_and_summary(tmp_path: Path) -> None:
    root_name = "artifacts"
    root = tmp_path / root_name
    completed = root / "completed"
    completed.mkdir(parents=True)
    (completed / "_SUCCESS").write_text("", encoding="utf-8")
    (completed / "summary.json").write_text(
        json.dumps({"metrics": {"score": 0.8}, "items": [{"value": 2}]}),
        encoding="utf-8",
    )
    failed = root / "failed"
    failed.mkdir()
    (failed / "_FAILED").write_text("", encoding="utf-8")
    running = root / "running"
    running.mkdir()

    records = discover_artifacts(tmp_path, _reader(root_name))
    by_name = {record.root.name: record for record in records}
    assert by_name["completed"].state == "completed"
    assert by_name["failed"].state == "failed"
    assert by_name["running"].state == "running"
    assert by_name["completed"].summaries == {"Score": 0.8, "First": 2}


def test_dot_path_csv_tail_and_comparison_limits(tmp_path: Path) -> None:
    assert extract_dot_path({"a": [{"b": 3}]}, "a.0.b") == 3
    assert extract_dot_path({"a": 1}, "a.missing") is None
    csv_path = tmp_path / "preview.csv"
    csv_path.write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9\n", encoding="utf-8")
    preview = csv_preview(csv_path, row_limit=2, column_limit=2)
    assert preview.shape == (2, 2)
    log_path = tmp_path / "run.log"
    log_path.write_text("one\ntwo\nthree\n", encoding="utf-8")
    assert text_tail(log_path, lines=2) == "two\nthree"

    reader = _reader("artifacts")
    root = tmp_path / "artifacts"
    for name, score in (("left", 0.7), ("right", 0.8)):
        artifact = root / name
        artifact.mkdir(parents=True)
        (artifact / "_SUCCESS").touch()
        (artifact / "summary.json").write_text(
            json.dumps({"metrics": {"score": score}, "items": [{"value": 1}]}),
            encoding="utf-8",
        )
    records = {item.root.name: item for item in discover_artifacts(tmp_path, reader)}
    rows = comparison_rows(records["left"], records["right"], reader.compare_fields)
    assert rows[0] == {"field": "Score", "left": 0.7, "right": 0.8}


def test_conflicting_terminal_markers_are_explicitly_invalid(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    both = root / "both"
    both.mkdir(parents=True)
    (both / "_SUCCESS").touch()
    (both / "_FAILED").touch()
    record = discover_artifacts(tmp_path, _reader("artifacts"))[0]
    assert record.state == "invalid"
    assert record.diagnostic == "Conflicting success and failure markers are present."
    assert record.summaries == {}


def test_hardlinked_external_marker_is_invalid(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts" / "linked-marker"
    artifact.mkdir(parents=True)
    external = tmp_path / "external-marker"
    external.write_text("done", encoding="utf-8")
    os.link(external, artifact / "_SUCCESS")
    record = discover_artifacts(tmp_path, _reader("artifacts"))[0]
    assert record.state == "invalid"
    assert record.diagnostic == "A configured marker path is unsafe."


def test_linked_marker_ancestor_is_invalid(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts" / "linked-ancestor"
    artifact.mkdir(parents=True)
    external = tmp_path / "external-directory"
    external.mkdir()
    (external / "_SUCCESS").touch()
    linked = artifact / "nested"
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(linked), str(external)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode != 0:
            import pytest

            pytest.skip("Junction creation is unavailable")
    else:
        linked.symlink_to(external, target_is_directory=True)
    reader = ReaderSpec.from_dict(
        {
            "id": "test",
            "title": "Test reader",
            "artifact_roots": ["artifacts"],
            "discovery_glob": "*",
            "success_markers": ["nested/_SUCCESS"],
            "failure_markers": ["_FAILED"],
            "summary_files": [],
            "csv_previews": [],
            "log_files": [],
            "compare_fields": [],
        },
        "reader",
    )
    record = discover_artifacts(tmp_path, reader)[0]
    assert record.state == "invalid"
    assert record.diagnostic == "A configured marker path is unsafe."


def test_external_artifact_directory_link_is_not_discovered(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    external = tmp_path / "external-artifact"
    external.mkdir()
    (external / "_SUCCESS").touch()
    linked = root / str(uuid.uuid4())
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(linked), str(external)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode != 0:
            import pytest

            pytest.skip("Junction creation is unavailable")
    else:
        linked.symlink_to(external, target_is_directory=True)
    assert discover_artifacts(tmp_path, _reader("artifacts")) == []
