from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tests.research_test_support import build_persisted_test_run


@pytest.mark.parametrize(
    "corruption",
    [
        "aggregate_metric",
        "selected_threshold",
        "prediction_label",
        "assignment_key",
        "progress_artifact",
        "canonical_hash",
        "candidate_hash",
        "run_hash",
        "loaded_hash",
        "missing_column",
        "duplicate_fold_metric",
        "incomplete_artifact_set",
    ],
)
def test_completion_rejects_persisted_corruption(
    tmp_path: Path,
    corruption: str,
) -> None:
    store, result, metadata = build_persisted_test_run(tmp_path)
    root = store.root

    if corruption == "aggregate_metric":
        path = root / "metrics" / "aggregate.json"
        payload = _read_json(path)
        payload["metrics"]["balanced_accuracy"]["mean"] += 0.01
        _write_json(path, payload)
    elif corruption == "selected_threshold":
        path = root / "thresholds" / "selected_thresholds.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "selected_threshold"] += 0.1
        frame.to_csv(path, index=False)
    elif corruption == "prediction_label":
        path = root / "predictions" / "outer_validation.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "prediction"] = 1 - frame.loc[0, "prediction"]
        frame.to_parquet(path, index=False)
    elif corruption == "assignment_key":
        path = root / "splits" / "outer_assignments.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "row_position"] = frame.loc[1, "row_position"]
        frame.to_parquet(path, index=False)
    elif corruption == "progress_artifact":
        path = next((root / "fold_progress").rglob("outer_validation.parquet"))
        frame = pd.read_parquet(path)
        frame.loc[0, "probability"] += 0.01
        frame.to_parquet(path, index=False)
    elif corruption == "canonical_hash":
        path = root / "evaluation_plan.json"
        payload = _read_json(path)
        payload["sha256"] = "0" * 64
        _write_json(path, payload)
    elif corruption in {"candidate_hash", "run_hash", "loaded_hash"}:
        filenames = {
            "candidate_hash": "candidate_contract.json",
            "run_hash": "run_implementation.json",
            "loaded_hash": "loaded_modules.json",
        }
        path = root / filenames[corruption]
        payload = _read_json(path)
        payload["sha256"] = "0" * 64
        _write_json(path, payload)
    elif corruption == "missing_column":
        path = root / "predictions" / "outer_validation.parquet"
        frame = pd.read_parquet(path).drop(columns=["probability"])
        frame.to_parquet(path, index=False)
    elif corruption == "duplicate_fold_metric":
        path = root / "metrics" / "outer_folds.csv"
        frame = pd.read_csv(path)
        pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(
            path,
            index=False,
        )
    elif corruption == "incomplete_artifact_set":
        (root / "thresholds" / "threshold_summary.json").unlink()
    else:
        raise AssertionError(f"Unhandled corruption case: {corruption}")

    with pytest.raises(RuntimeError):
        store.complete(metadata, result)

    assert not (root / "_SUCCESS").exists()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "corruption",
    [
        "metadata_extra_key",
        "metadata_run_id",
        "metadata_status",
        "metadata_loaded_hash",
        "metadata_non_utc",
        "metadata_record_count",
        "metadata_duration",
        "status_fold_count",
        "status_failure",
        "status_metadata_disagreement",
    ],
)
def test_completion_rejects_terminal_metadata_corruption_after_atomic_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    store, result, metadata = build_persisted_test_run(tmp_path)
    original_write = store._write_json

    def corrupting_write(path: Path, payload: dict) -> None:
        persisted = dict(payload)
        if path.name == "run_metadata.json" and payload.get("status") == "completed":
            if corruption == "metadata_extra_key":
                persisted["unexpected"] = True
            elif corruption == "metadata_run_id":
                persisted["run_id"] = "wrong-run"
            elif corruption == "metadata_status":
                persisted["status"] = "failed"
            elif corruption == "metadata_loaded_hash":
                persisted["loaded_module_sha256"] = "0" * 64
            elif corruption == "metadata_non_utc":
                persisted["finished_at_utc"] = "2026-01-01T01:00:00+01:00"
            elif corruption == "metadata_record_count":
                persisted["record_counts"] = dict(persisted["record_counts"])
                persisted["record_counts"]["outer_prediction_count"] -= 1
            elif corruption == "metadata_duration":
                persisted["duration_seconds"] = -1.0
        if (
            path.name == "execution_status.json"
            and payload.get("status") == "completed"
        ):
            if corruption == "status_fold_count":
                persisted["completed_outer_folds"] -= 1
            elif corruption == "status_failure":
                persisted["failure"] = {"type": "Injected", "message": "bad"}
            elif corruption == "status_metadata_disagreement":
                persisted["status"] = "failed"
        original_write(path, persisted)

    monkeypatch.setattr(store, "_write_json", corrupting_write)
    with pytest.raises(RuntimeError):
        store.complete(metadata, result)

    assert not (store.root / "_SUCCESS").exists()
    assert not (store.root / "artifact_manifest.json").exists()
