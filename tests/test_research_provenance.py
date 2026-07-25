from __future__ import annotations

from pathlib import Path

from src.churn_ml import research_provenance
from src.churn_ml.research_protocol import build_candidate_contract_identity
from src.churn_ml.research_provenance import (
    build_source_provenance,
    run_implementation_provenance,
)


def test_candidate_hash_changes_with_prediction_source_bytes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "candidate.py"
    source.write_bytes(b"prediction = 0.2\n")
    first_provenance, first_source_hash = build_source_provenance(
        tmp_path,
        ["candidate.py"],
    )
    _, first_hash = _candidate_identity(first_provenance, first_source_hash)

    source.write_bytes(b"prediction = 0.8\n")
    second_provenance, second_source_hash = build_source_provenance(
        tmp_path,
        ["candidate.py"],
    )
    _, second_hash = _candidate_identity(second_provenance, second_source_hash)

    assert first_hash != second_hash


def test_run_hash_changes_with_evaluation_source_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        research_provenance,
        "RUN_IMPLEMENTATION_SOURCE_PATHS",
        ("runner.py",),
    )
    runner = tmp_path / "runner.py"
    config = tmp_path / "plan.yaml"
    runner.write_bytes(b"aggregation = 'pooled'\n")
    config.write_bytes(b"folds: 5\n")
    _, first_hash = run_implementation_provenance(
        tmp_path,
        config_paths={"evaluation_plan": config},
    )

    runner.write_bytes(b"aggregation = 'fold_mean'\n")
    _, second_hash = run_implementation_provenance(
        tmp_path,
        config_paths={"evaluation_plan": config},
    )

    assert first_hash != second_hash


def test_source_hash_order_location_and_unrelated_docs_are_stable(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    for root in (first_root, second_root):
        root.mkdir()
        (root / "a.py").write_bytes(b"a = 1\n")
        (root / "b.py").write_bytes(b"b = 2\n")

    first_manifest, first_hash = build_source_provenance(
        first_root,
        ["b.py", "a.py"],
    )
    second_manifest, second_hash = build_source_provenance(
        second_root,
        ["a.py", "b.py"],
    )
    (first_root / "README.md").write_bytes(b"unrelated documentation\n")
    _, after_docs_hash = build_source_provenance(
        first_root,
        ["a.py", "b.py"],
    )

    assert first_manifest == second_manifest
    assert first_hash == second_hash == after_docs_hash
    assert [item["path"] for item in first_manifest["files"]] == ["a.py", "b.py"]
    assert str(first_root) not in str(first_manifest)
    assert str(second_root) not in str(second_manifest)


def _candidate_identity(
    source_provenance: dict,
    source_hash: str,
) -> tuple[dict, str]:
    return build_candidate_contract_identity(
        {
            "candidate_id": "unit-candidate",
            "implementation": "unit",
            "features": {"model": ["feature"]},
            "target_encoder": {"alpha": 10},
            "lightgbm": {"parameters": {"n_estimators": 1}},
        },
        feature_schema={"model_feature_names": ["feature"]},
        source_provenance={
            **source_provenance,
            "manifest_sha256": source_hash,
        },
        runtime_dependencies={"lightgbm": "unit"},
    )
