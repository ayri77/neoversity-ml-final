from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.churn_ml.paired_comparison import build_comparison_result
from src.churn_ml.paired_comparison_artifacts import (
    PairedComparisonArtifactError,
    create_comparison_artifacts,
    validate_comparison_artifacts,
)
from tests.test_paired_comparison_v1 import (
    POLICY,
    _patch_artifact_validation,
    _synthetic_run,
)


def test_manifest_tampering_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _synthetic_run(tmp_path, "baseline")
    candidate = _synthetic_run(tmp_path, "candidate", improved=True)
    _patch_artifact_validation(monkeypatch, baseline, candidate)
    result = build_comparison_result(baseline, candidate, POLICY)
    root = create_comparison_artifacts(
        result=result,
        policy=POLICY,
        project_root=tmp_path,
        output_root=Path("comparisons"),
        comparison_id="tamper-check",
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PairedComparisonArtifactError, match="manifest"):
        validate_comparison_artifacts(
            root,
            project_root=tmp_path,
            require_success=True,
            verify_manifest=True,
        )
