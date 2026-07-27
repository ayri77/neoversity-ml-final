from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path

from src.churn_ml.autogluon_artifacts import build_inventory, write_json, write_yaml
from src.churn_ml.autogluon_config import config_identity_sha256
from src.churn_ml.autogluon_inspection import inspect_run, predictor_report
from tests.test_autogluon_config import valid_payload


class FakeLeaderboard:
    def to_dict(self, *, orient: str) -> list[dict[str, object]]:
        assert orient == "records"
        return [{"model": "WeightedEnsemble_L2", "score_val": 0.8}]


class FakePredictor:
    model_best = "WeightedEnsemble_L2"
    decision_threshold = 0.117

    def __init__(self, info: dict[str, object] | None = None) -> None:
        self._info = info or {
            "model_info": {
                "WeightedEnsemble_L2": {
                    "model_weights": {"ModelA": 1.0},
                }
            }
        }

    def model_names(self) -> list[str]:
        return ["ModelA", "WeightedEnsemble_L2"]

    def info(self) -> dict[str, object]:
        return self._info

    def leaderboard(self, *, silent: bool) -> FakeLeaderboard:
        assert silent
        return FakeLeaderboard()


def make_valid_completed_run(run_dir: Path) -> None:
    run_dir.mkdir()
    config = valid_payload()
    config_hash = config_identity_sha256(config)
    profile_identity = {"profile_id": config["profile_id"], "seed": config["seed"]}
    profile_hash = hashlib.sha256(
        json.dumps(profile_identity, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    run_id = "valid-fake-run"
    child_pid = 1234
    write_yaml(run_dir / "resolved_config.yaml", config)
    write_json(
        run_dir / "run_metadata.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "config_identity_sha256": config_hash,
            "profile_id": config["profile_id"],
            "profile_sha256": profile_hash,
            "profile": profile_identity,
            "dataset_version": config["dataset"]["version"],
            "requested_seed": config["seed"],
            "child_pid": child_pid,
        },
    )
    write_json(run_dir / "environment.json", {"schema_version": 1})
    write_json(
        run_dir / "execution_status.json",
        {
            "status": "completed",
            "run_id": run_id,
            "config_identity_sha256": config_hash,
            "profile_sha256": profile_hash,
            "child_pid": child_pid,
        },
    )
    write_json(
        run_dir / "dataset_manifest.json",
        {"dataset_version": config["dataset"]["version"]},
    )
    write_json(
        run_dir / "profile_resolution.json",
        {
            "profile_id": config["profile_id"],
            "profile_sha256": profile_hash,
            "profile_identity": profile_identity,
            "dataset_version": config["dataset"]["version"],
            "requested_seed": config["seed"],
            "resolved_families": ["REALTABPFN-V2"],
            "top_level_num_gpus_passed_to_fit": False,
        },
    )
    write_json(
        run_dir / "worker_result.json",
        {
            "schema_version": 1,
            "status": "completed",
            "worker_pid": child_pid,
            "started_at_utc": "2026-07-27T10:00:00Z",
            "completed_at_utc": "2026-07-27T10:00:02Z",
            "duration_seconds": 2.0,
            "profile_id": config["profile_id"],
            "profile_sha256": profile_hash,
            "dataset_version": config["dataset"]["version"],
            "predictor_relative_path": "predictor",
            "resolved_families": ["REALTABPFN-V2"],
            "model_names": ["ModelA", "WeightedEnsemble_L2"],
            "best_model": "WeightedEnsemble_L2",
            "decision_threshold": 0.117,
            "autogluon_version": "1.5.0",
            "python_version": "3.12.12",
            "requested_seed": config["seed"],
            "effective_seed": None,
        },
    )
    inspection = run_dir / "inspection"
    inspection.mkdir()
    (inspection / "leaderboard.csv").write_text(
        "model,score_val\nWeightedEnsemble_L2,0.8\n", encoding="utf-8"
    )
    write_json(
        inspection / "summary.json",
        {
            "models": ["ModelA", "WeightedEnsemble_L2"],
            "best_model": "WeightedEnsemble_L2",
            "decision_threshold": 0.117,
            "requested_seed": 42,
            "effective_seed": None,
            "effective_seeds_observed": [],
            "effective_seed_status": "unavailable",
        },
    )
    predictor = run_dir / "predictor"
    predictor.mkdir()
    for name in ("predictor.pkl", "learner.pkl", "version.txt"):
        (predictor / name).write_text("fake", encoding="utf-8")
    logs = run_dir / "logs"
    logs.mkdir()
    for name in ("supervisor.log", "worker.stdout.log", "worker.stderr.log"):
        (logs / name).write_text("", encoding="utf-8")
    write_json(run_dir / "artifact_inventory.json", build_inventory(run_dir))
    write_json(
        run_dir / "_SUCCESS",
        {"status": "completed", "run_id": run_id},
    )


def test_inspect_valid_completed_fake_run_loads_by_default(tmp_path: Path) -> None:
    run_dir = tmp_path / "complete"
    make_valid_completed_run(run_dir)
    report = inspect_run(
        run_dir,
        predictor_loader=lambda _path: FakePredictor(),
    )
    assert report["classification"] == "complete"
    assert report["reason_codes"] == []
    assert report["worker_completion_valid"] is True
    assert report["predictor_loading_succeeded"] is True
    assert report["predictor"]["best_model"] == "WeightedEnsemble_L2"
    assert report["predictor"]["ensemble_weights"] == {"ModelA": 1.0}


def test_inspection_is_read_only_and_idempotent(tmp_path: Path) -> None:
    run_dir = tmp_path / "complete"
    make_valid_completed_run(run_dir)
    before = {
        path.relative_to(run_dir).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    first = inspect_run(run_dir, attempt_load=False)
    second = inspect_run(run_dir, attempt_load=False)
    after = {
        path.relative_to(run_dir).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    assert first == second
    assert before == after


def test_success_with_missing_predictor_is_corrupt_and_not_loaded(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "missing-predictor"
    make_valid_completed_run(run_dir)
    (run_dir / "predictor" / "learner.pkl").unlink()
    report = inspect_run(run_dir, predictor_loader=lambda _path: FakePredictor())
    assert report["classification"] == "corrupt"
    assert "success_predictor_structure_incomplete" in report["reason_codes"]
    assert report["predictor_loading_attempted"] is False


def test_success_with_missing_worker_result_is_corrupt(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing-result"
    make_valid_completed_run(run_dir)
    (run_dir / "worker_result.json").unlink()
    report = inspect_run(run_dir)
    assert report["classification"] == "corrupt"
    assert "worker_result_missing" in report["reason_codes"]
    assert report["predictor_loading_attempted"] is False


def test_success_with_malformed_worker_result_is_corrupt(tmp_path: Path) -> None:
    run_dir = tmp_path / "malformed-result"
    make_valid_completed_run(run_dir)
    (run_dir / "worker_result.json").write_text('{"status":', encoding="utf-8")
    report = inspect_run(run_dir)
    assert report["classification"] == "corrupt"
    assert "worker_result_invalid_json" in report["reason_codes"]


def test_success_with_failed_execution_status_is_corrupt(tmp_path: Path) -> None:
    run_dir = tmp_path / "failed-status"
    make_valid_completed_run(run_dir)
    status = json.loads((run_dir / "execution_status.json").read_text(encoding="utf-8"))
    status["status"] = "failed"
    write_json(run_dir / "execution_status.json", status)
    report = inspect_run(run_dir)
    assert report["classification"] == "corrupt"
    assert "execution_status_terminal_mismatch" in report["reason_codes"]


def test_both_terminal_markers_are_corrupt(tmp_path: Path) -> None:
    run_dir = tmp_path / "both-markers"
    make_valid_completed_run(run_dir)
    write_json(run_dir / "_FAILED", {"status": "failed", "run_id": "valid-fake-run"})
    report = inspect_run(run_dir)
    assert report["classification"] == "corrupt"
    assert "terminal_markers_conflict" in report["reason_codes"]


def test_no_terminal_marker_is_not_complete(tmp_path: Path) -> None:
    run_dir = tmp_path / "no-marker"
    make_valid_completed_run(run_dir)
    (run_dir / "_SUCCESS").unlink()
    report = inspect_run(run_dir)
    assert report["classification"] in {"incomplete", "corrupt"}
    assert report["classification"] != "complete"
    assert "terminal_marker_missing" in report["reason_codes"]
    assert report["predictor_loading_attempted"] is False


def test_profile_hash_mismatch_is_corrupt(tmp_path: Path) -> None:
    run_dir = tmp_path / "profile-hash-mismatch"
    make_valid_completed_run(run_dir)
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    metadata["profile"]["seed"] = 99
    write_json(run_dir / "run_metadata.json", metadata)
    report = inspect_run(run_dir)
    assert report["classification"] == "corrupt"
    assert "run_metadata_profile_hash_mismatch" in report["reason_codes"]


def test_inventory_mismatch_is_corrupt(tmp_path: Path) -> None:
    run_dir = tmp_path / "inventory-mismatch"
    make_valid_completed_run(run_dir)
    (run_dir / "unexpected.txt").write_text("after inventory", encoding="utf-8")
    report = inspect_run(run_dir)
    assert report["classification"] == "corrupt"
    assert "artifact_inventory_mismatch" in report["reason_codes"]


def test_incomplete_run_requires_explicit_attempt_load(tmp_path: Path) -> None:
    run_dir = tmp_path / "partial"
    run_dir.mkdir()
    model = run_dir / "predictor" / "models" / "CompletedFamily"
    model.mkdir(parents=True)
    write_json(run_dir / "_FAILED", {"status": "failed", "run_id": "partial"})
    called: list[Path] = []

    def fail_load(path: Path) -> object:
        called.append(path)
        raise RuntimeError("partial predictor is not loadable")

    default_report = inspect_run(run_dir, predictor_loader=fail_load)
    attempted_report = inspect_run(
        run_dir, attempt_load=True, predictor_loader=fail_load
    )
    assert default_report["classification"] != "complete"
    assert default_report["predictor_loading_attempted"] is False
    assert attempted_report["predictor_loading_attempted"] is True
    assert attempted_report["predictor_loading_succeeded"] is False
    assert "partial predictor is not loadable" in attempted_report["load_error"]
    assert len(called) == 1


def test_predictor_info_paths_are_confined_to_nonportable_section() -> None:
    fake_user_path = r"C:\\Users\\review-user\\runs\\predictor"
    report = predictor_report(
        FakePredictor(
            {
                "path": fake_user_path,
                "nested": {
                    "artifact": fake_user_path + r"\\models\\model.pkl",
                    "model_random_seed": 42,
                },
            }
        ),
        requested_seed=42,
    )
    portable = copy.deepcopy(report)
    nonportable = portable.pop("local_operational_nonportable")
    portable_text = json.dumps(portable)
    assert "review-user" not in portable_text
    assert fake_user_path not in portable_text
    assert "review-user" in json.dumps(nonportable)
    assert report["effective_seed"] == 42
    assert report["effective_seed_status"] == "matched"


def test_effective_seed_mismatch_is_reported() -> None:
    report = predictor_report(
        FakePredictor({"model_info": {"model_random_seed": 43}}),
        requested_seed=42,
    )
    assert report["effective_seed"] == 43
    assert report["effective_seed_status"] == "mismatch"


def test_importing_public_runner_does_not_import_autogluon() -> None:
    for name in tuple(sys.modules):
        if name == "autogluon" or name.startswith("autogluon."):
            del sys.modules[name]
    module = importlib.import_module("src.churn_ml.autogluon_cli")
    importlib.reload(module)
    assert not any(
        name == "autogluon" or name.startswith("autogluon.") for name in sys.modules
    )
