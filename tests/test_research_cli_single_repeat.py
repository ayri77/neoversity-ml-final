from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.churn_ml import research_cli as cli
from src.churn_ml.research_data import load_research_training_data
from src.churn_ml.research_protocol import build_evaluation_assignments
from tests.research_test_support import build_persisted_test_run


def test_one_repeat_cli_completes_with_undefined_sd_as_na(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    store, result, _ = build_persisted_test_run(tmp_path)
    data = load_research_training_data(store.config)
    assignments = build_evaluation_assignments(data.y, store.config.plan_payload)
    plan = _read_json(store.root / "evaluation_plan.json")
    candidate = _read_json(store.root / "candidate_contract.json")
    implementation = _read_json(store.root / "run_implementation.json")
    loaded = _read_json(store.root / "loaded_modules.json")
    prepared = cli.ResearchPreflight(
        config=store.config,
        data=data,
        assignments=assignments,
        plan_identity=plan["canonical"],
        plan_hash=plan["sha256"],
        candidate_identity=candidate["canonical"],
        candidate_hash=candidate["sha256"],
        candidate_source_manifest_hash=store.candidate_source_manifest_hash,
        run_implementation_identity=implementation["canonical"],
        run_implementation_hash=implementation["sha256"],
        loaded_module_identity={
            "canonical": loaded["canonical"],
            "observations": loaded["observations"],
        },
        loaded_module_hash=loaded["sha256"],
        environment=_read_json(store.root / "run_metadata.json")["environment"],
    )
    monkeypatch.setattr(cli, "preflight", lambda config_path, **kwargs: prepared)
    monkeypatch.setattr(
        cli,
        "ResearchArtifactStore",
        lambda *args, **kwargs: store,
    )
    monkeypatch.setattr(
        cli,
        "run_research_evaluation",
        lambda *args, **kwargs: result,
    )

    return_code = cli.execute(
        argparse.Namespace(
            config=tmp_path / "ignored.yaml",
            validate_only=False,
            run_id="unit-run",
        )
    )

    output = capsys.readouterr()
    assert return_code == 0, output.err
    assert "sample_sd=N/A" in output.out
    assert "Status: completed" in output.out
    assert (store.root / "_SUCCESS").is_file()
    assert not (store.root / "_FAILED").exists()
    success = _read_json(store.root / "_SUCCESS")
    manifest = _read_json(store.root / "artifact_manifest.json")
    assert success["artifact_manifest_sha256"] == manifest["manifest_sha256"]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
