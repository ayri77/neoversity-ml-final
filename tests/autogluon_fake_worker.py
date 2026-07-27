"""Cheap subprocess used by standalone AutoGluon supervisor tests."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path


RESULT_MODES = (
    "success",
    "missing-result",
    "invalid-json",
    "minimal",
    "missing-field",
    "unknown-field",
    "wrong-type",
    "wrong-pid",
    "wrong-profile",
    "wrong-profile-hash",
    "wrong-dataset",
    "wrong-path",
    "inconsistent-time",
    "mismatch-inspection",
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())


def wait_for_metadata(run_dir: Path) -> dict[str, object]:
    deadline = time.monotonic() + 5
    metadata_path = run_dir / "run_metadata.json"
    while time.monotonic() < deadline:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            time.sleep(0.01)
            continue
        if metadata.get("child_pid") == os.getpid():
            return metadata
        time.sleep(0.01)
    raise RuntimeError("supervisor did not publish child metadata")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=RESULT_MODES + ("python-failure", "native-like", "partial"),
        required=True,
    )
    parser.add_argument("--exit-code", type=int, default=23)
    args = parser.parse_args()
    run_dir: Path = args.run_dir
    print("AUTOGLUON_STAGE:fake_worker_started", flush=True)

    if args.mode == "python-failure":
        raise RuntimeError("intentional fake worker exception")
    if args.mode == "native-like":
        if os.name == "nt" and args.exit_code > 0x7FFFFFFF:
            ctypes.windll.kernel32.ExitProcess(args.exit_code)
        os._exit(args.exit_code)
    if args.mode == "partial":
        partial = run_dir / "predictor" / "models" / "CompletedFamily" / "model.pkl"
        partial.parent.mkdir(parents=True)
        partial.write_text("partial model", encoding="utf-8")
        print("AUTOGLUON_STAGE:partial_model_written", flush=True)
        print("simulated native failure", file=sys.stderr, flush=True)
        os._exit(args.exit_code)

    metadata = wait_for_metadata(run_dir)
    start = datetime.now(UTC)
    for relative in (
        "predictor/predictor.pkl",
        "predictor/learner.pkl",
        "predictor/version.txt",
    ):
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake", encoding="utf-8")
    (run_dir / "inspection").mkdir()
    (run_dir / "inspection" / "leaderboard.csv").write_text(
        "model,score_val\nFakeModel,0.8\n",
        encoding="utf-8",
    )
    inspection = {
        "models": ["FakeModel"],
        "best_model": "FakeModel",
        "decision_threshold": 0.25,
        "requested_seed": metadata["requested_seed"],
        "effective_seed_scope": "configured_core_base_models_only",
        "configured_families": ["REALTABPFN-V2"],
        "effective_seed": metadata["requested_seed"],
        "effective_seeds_observed": [metadata["requested_seed"]],
        "effective_seed_status": "verified",
        "base_seed_metadata": [
            {
                "model_name": "FakeModel",
                "model_types": ["RealTabPFNv2Model"],
                "model_random_seeds": [metadata["requested_seed"]],
            }
        ],
        "auxiliary_effective_seeds_observed": [0],
        "auxiliary_seed_metadata": [
            {
                "model_name": "WeightedEnsemble_L2",
                "model_types": ["WeightedEnsembleModel"],
                "model_random_seeds": [0],
            }
        ],
        "unclassified_seed_metadata": [],
    }
    if args.mode == "mismatch-inspection":
        inspection["best_model"] = "DifferentModel"
    write_json(run_dir / "inspection" / "summary.json", inspection)
    write_json(
        run_dir / "dataset_manifest.json",
        {
            "dataset_version": metadata["dataset_version"],
            "ordered_features": ["feature_a"],
            "row_count": 4,
        },
    )
    write_json(
        run_dir / "profile_resolution.json",
        {
            "schema_version": 1,
            "profile_id": metadata["profile_id"],
            "profile_sha256": metadata["profile_sha256"],
            "profile_identity": metadata["profile"],
            "dataset_version": metadata["dataset_version"],
            "requested_seed": metadata["requested_seed"],
            "resolved_families": ["REALTABPFN-V2"],
            "family_resources": {"REALTABPFN-V2": {"num_gpus": [1]}},
            "resolved_hyperparameters": {"REALTABPFN-V2": [{}]},
            "top_level_num_gpus_passed_to_fit": False,
        },
    )
    completed = datetime.now(UTC)
    result = {
        "schema_version": 1,
        "status": "completed",
        "worker_pid": os.getpid(),
        "started_at_utc": start.isoformat().replace("+00:00", "Z"),
        "completed_at_utc": completed.isoformat().replace("+00:00", "Z"),
        "duration_seconds": float((completed - start).total_seconds()),
        "profile_id": metadata["profile_id"],
        "profile_sha256": metadata["profile_sha256"],
        "dataset_version": metadata["dataset_version"],
        "predictor_relative_path": "predictor",
        "resolved_families": ["REALTABPFN-V2"],
        "model_names": ["FakeModel"],
        "best_model": "FakeModel",
        "decision_threshold": 0.25,
        "autogluon_version": "1.5.0",
        "python_version": platform.python_version(),
        "requested_seed": metadata["requested_seed"],
        "effective_seed": metadata["requested_seed"],
    }
    if args.mode == "missing-result":
        pass
    elif args.mode == "invalid-json":
        (run_dir / "worker_result.json").write_text('{"status":', encoding="utf-8")
    elif args.mode == "minimal":
        write_json(run_dir / "worker_result.json", {"status": "completed"})
    else:
        if args.mode == "missing-field":
            result.pop("python_version")
        elif args.mode == "unknown-field":
            result["unexpected"] = True
        elif args.mode == "wrong-type":
            result["worker_pid"] = True
        elif args.mode == "wrong-pid":
            result["worker_pid"] = os.getpid() + 1
        elif args.mode == "wrong-profile":
            result["profile_id"] = "wrong_profile"
        elif args.mode == "wrong-profile-hash":
            result["profile_sha256"] = "0" * 64
        elif args.mode == "wrong-dataset":
            result["dataset_version"] = "wrong_dataset"
        elif args.mode == "wrong-path":
            result["predictor_relative_path"] = "other"
        elif args.mode == "inconsistent-time":
            result["completed_at_utc"] = (
                (start - timedelta(seconds=10)).isoformat().replace("+00:00", "Z")
            )
        write_json(run_dir / "worker_result.json", result)
    print("AUTOGLUON_STAGE:worker_result_written", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
