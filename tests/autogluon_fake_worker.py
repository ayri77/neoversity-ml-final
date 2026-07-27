"""Cheap subprocess used by standalone AutoGluon supervisor tests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("success", "python-failure", "native-like", "partial"),
        required=True,
    )
    parser.add_argument("--exit-code", type=int, default=23)
    args = parser.parse_args()
    run_dir: Path = args.run_dir
    print("AUTOGLUON_STAGE:fake_worker_started", flush=True)

    if args.mode == "python-failure":
        raise RuntimeError("intentional fake worker exception")
    if args.mode == "native-like":
        os._exit(args.exit_code)
    if args.mode == "partial":
        partial = run_dir / "predictor" / "models" / "CompletedFamily" / "model.pkl"
        partial.parent.mkdir(parents=True)
        partial.write_text("partial model", encoding="utf-8")
        print("AUTOGLUON_STAGE:partial_model_written", flush=True)
        print("simulated native failure", file=sys.stderr, flush=True)
        os._exit(args.exit_code)

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
    write_json(
        run_dir / "inspection" / "summary.json",
        {"best_model": "FakeModel", "decision_threshold": 0.25},
    )
    write_json(
        run_dir / "dataset_manifest.json",
        {"ordered_features": ["feature_a"], "row_count": 4},
    )
    write_json(
        run_dir / "profile_resolution.json",
        {"profile_id": "fake", "resolved_hyperparameters": {"FAKE": [{}]}},
    )
    write_json(run_dir / "worker_result.json", {"status": "completed"})
    print("AUTOGLUON_STAGE:worker_result_written", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
