"""Focused cheap submission eligibility and UI-rerun safety tests."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.churn_ml.control_panel.submission_eligibility import (
    BLOCKED_STATE,
    ELIGIBLE_STATE,
    eligibility_cache_fingerprint,
    eligibility_label,
    project_candidate_submission_eligibility,
)
from src.churn_ml.prediction_candidates.contract_v1 import (
    MANIFEST_FILENAME,
    SOURCE_METADATA_FILENAME,
    SUCCESS_FILENAME,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_candidate(
    root: Path,
    candidate_id: str,
    *,
    source_kind: str = "autogluon_standalone_v1",
    threshold: float | None = 0.17,
    exploratory: bool = False,
    blend_id: str | None = None,
    parent_ids: list[str] | None = None,
    parent_hashes: list[str] | None = None,
    positive_class_label: int = 1,
    probability_semantics: str = "P(y=1)",
    test_row_count: int = 2500,
    write_success: bool = True,
    write_metadata: bool = True,
    malformed_manifest: bool = False,
    dataset_id: str = "v0_raw_minimal",
) -> Path:
    package = root / "artifacts" / "prediction_candidates" / candidate_id
    package.mkdir(parents=True, exist_ok=True)
    if malformed_manifest:
        (package / MANIFEST_FILENAME).write_text("{not-json", encoding="utf-8")
    else:
        (package / MANIFEST_FILENAME).write_text(
            json.dumps(
                {
                    "candidate_id": candidate_id,
                    "source_kind": source_kind,
                    "source_model_name": "Model",
                    "dataset_id": dataset_id,
                    "train_row_count": 100,
                    "test_row_count": test_row_count,
                    "oof_protocol": "protocol",
                    "source_metric_name": "balanced_accuracy",
                    "source_metric_value": 0.9,
                    "exploratory": exploratory,
                    "positive_class_label": positive_class_label,
                    "probability_semantics": probability_semantics,
                }
            ),
            encoding="utf-8",
        )
    if write_metadata:
        metadata: dict = {}
        if threshold is not None:
            metadata["final_deployment_threshold"] = threshold
        if blend_id is not None:
            metadata["blend_id"] = blend_id
        if parent_ids is not None:
            metadata["parent_candidate_ids"] = parent_ids
        if parent_hashes is not None:
            metadata["parent_manifest_hashes"] = parent_hashes
        (package / SOURCE_METADATA_FILENAME).write_text(
            json.dumps(metadata), encoding="utf-8"
        )
    if write_success:
        (package / SUCCESS_FILENAME).write_text("{}", encoding="utf-8")
    return package


def _write_competition_identity(root: Path) -> None:
    sample = root / "data" / "raw" / "final_proj_sample_submission.csv"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text("id,churn\n1,0\n", encoding="utf-8")
    identity = root / "data" / "competition" / "test_row_identity_v1.json"
    identity.parent.mkdir(parents=True, exist_ok=True)
    identity.write_text("{}", encoding="utf-8")


def _write_blend_artifact(root: Path, blend_id: str) -> None:
    blend_dir = root / "artifacts" / "prediction_blends" / blend_id
    blend_dir.mkdir(parents=True, exist_ok=True)
    (blend_dir / "blend_manifest.json").write_text(
        json.dumps({"blend_id": blend_id}), encoding="utf-8"
    )
    (blend_dir / SUCCESS_FILENAME).write_text("{}", encoding="utf-8")


def test_valid_atomic_candidate_is_eligible(tmp_path: Path) -> None:
    _write_competition_identity(tmp_path)
    _write_candidate(tmp_path, "pc1_atomic")
    result = project_candidate_submission_eligibility(
        "pc1_atomic", repository_root=tmp_path
    )
    assert result.state == ELIGIBLE_STATE
    assert result.eligible_to_attempt is True
    assert result.threshold == 0.17
    assert result.strict_validation_required is True


def test_valid_blend_candidate_is_eligible(tmp_path: Path) -> None:
    _write_competition_identity(tmp_path)
    parents = ["pc1_p1", "pc1_p2"]
    for parent in parents:
        _write_candidate(tmp_path, parent)
    _write_blend_artifact(tmp_path, "pb1_demo")
    _write_candidate(
        tmp_path,
        "pc1_blend",
        source_kind="canonical_probability_blend_v1",
        blend_id="pb1_demo",
        parent_ids=parents,
        parent_hashes=["a" * 64, "b" * 64],
    )
    result = project_candidate_submission_eligibility(
        "pc1_blend", repository_root=tmp_path
    )
    assert result.state == ELIGIBLE_STATE
    assert result.blend_id == "pb1_demo"
    assert result.parent_candidate_ids == tuple(parents)


def test_missing_and_malformed_metadata_block(tmp_path: Path) -> None:
    _write_competition_identity(tmp_path)
    _write_candidate(tmp_path, "pc1_missing_manifest", write_success=True)
    (tmp_path / "artifacts/prediction_candidates/pc1_missing_manifest" / MANIFEST_FILENAME).unlink()
    missing_manifest = project_candidate_submission_eligibility(
        "pc1_missing_manifest", repository_root=tmp_path
    )
    assert missing_manifest.state == BLOCKED_STATE
    assert any(item.code == "manifest_missing" for item in missing_manifest.blockers)

    _write_candidate(tmp_path, "pc1_bad", malformed_manifest=True)
    bad = project_candidate_submission_eligibility("pc1_bad", repository_root=tmp_path)
    assert any(item.code == "manifest_invalid" for item in bad.blockers)

    _write_candidate(tmp_path, "pc1_nometa", write_metadata=False)
    nometa = project_candidate_submission_eligibility(
        "pc1_nometa", repository_root=tmp_path
    )
    assert any(item.code == "source_metadata_missing" for item in nometa.blockers)

    _write_candidate(tmp_path, "pc1_nosuccess", write_success=False)
    nosuccess = project_candidate_submission_eligibility(
        "pc1_nosuccess", repository_root=tmp_path
    )
    assert any(item.code == "success_missing" for item in nosuccess.blockers)


def test_threshold_and_exploratory_and_ids(tmp_path: Path) -> None:
    _write_competition_identity(tmp_path)
    _write_candidate(tmp_path, "pc1_nothresh", threshold=None)
    no_thresh = project_candidate_submission_eligibility(
        "pc1_nothresh", repository_root=tmp_path
    )
    assert any(item.code == "final_threshold_missing" for item in no_thresh.blockers)

    _write_candidate(tmp_path, "pc1_badthresh", threshold=1.5)
    bad_thresh = project_candidate_submission_eligibility(
        "pc1_badthresh", repository_root=tmp_path
    )
    assert any(item.code == "threshold_invalid" for item in bad_thresh.blockers)

    _write_candidate(tmp_path, "pc1_expl", exploratory=True)
    expl = project_candidate_submission_eligibility(
        "pc1_expl", repository_root=tmp_path
    )
    assert expl.eligible_to_attempt is True
    assert any(item.code == "exploratory_candidate" for item in expl.warnings)

    package = _write_candidate(tmp_path, "pc1_mismatch")
    manifest = json.loads((package / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    manifest["candidate_id"] = "pc1_other"
    (package / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    mismatch = project_candidate_submission_eligibility(
        "pc1_mismatch", repository_root=tmp_path
    )
    assert any(item.code == "candidate_id_mismatch" for item in mismatch.blockers)


def test_blend_reference_blockers(tmp_path: Path) -> None:
    _write_competition_identity(tmp_path)
    _write_candidate(
        tmp_path,
        "pc1_noblend",
        source_kind="canonical_probability_blend_v1",
        blend_id=None,
        parent_ids=["pc1_p1"],
        parent_hashes=["a" * 64],
    )
    no_blend = project_candidate_submission_eligibility(
        "pc1_noblend", repository_root=tmp_path
    )
    assert any(item.code == "blend_id_missing" for item in no_blend.blockers)

    _write_candidate(
        tmp_path,
        "pc1_noparent",
        source_kind="canonical_probability_blend_v1",
        blend_id="pb1_x",
        parent_ids=[],
        parent_hashes=[],
    )
    _write_blend_artifact(tmp_path, "pb1_x")
    no_parent = project_candidate_submission_eligibility(
        "pc1_noparent", repository_root=tmp_path
    )
    assert any(item.code == "parent_refs_missing" for item in no_parent.blockers)

    _write_candidate(
        tmp_path,
        "pc1_missing_parent",
        source_kind="canonical_probability_blend_v1",
        blend_id="pb1_y",
        parent_ids=["pc1_absent"],
        parent_hashes=["a" * 64],
    )
    _write_blend_artifact(tmp_path, "pb1_y")
    missing_parent = project_candidate_submission_eligibility(
        "pc1_missing_parent", repository_root=tmp_path
    )
    assert any(
        item.code == "parent_manifest_missing" for item in missing_parent.blockers
    )

    _write_candidate(tmp_path, "pc1_parent_ok")
    _write_candidate(
        tmp_path,
        "pc1_missing_blend_art",
        source_kind="canonical_probability_blend_v1",
        blend_id="pb1_missing",
        parent_ids=["pc1_parent_ok"],
        parent_hashes=["a" * 64],
    )
    missing_blend = project_candidate_submission_eligibility(
        "pc1_missing_blend_art", repository_root=tmp_path
    )
    assert any(
        item.code == "blend_artifact_missing" for item in missing_blend.blockers
    )


def test_projection_avoids_strict_and_payload_work(tmp_path: Path) -> None:
    _write_competition_identity(tmp_path)
    _write_candidate(tmp_path, "pc1_safe")
    with (
        patch(
            "src.churn_ml.prediction_candidates.submission_v1.evaluate_submission_readiness"
        ) as strict,
        patch(
            "src.churn_ml.prediction_candidates.contract_v1.validate_candidate_package"
        ) as validate,
        patch(
            "src.churn_ml.blending.artifact_v1.load_blend_artifact"
        ) as load_blend,
        patch(
            "src.churn_ml.prediction_candidates.contract_v1.file_sha256"
        ) as hashed,
        patch("pandas.read_parquet") as parquet,
    ):
        result = project_candidate_submission_eligibility(
            "pc1_safe", repository_root=tmp_path
        )
    assert result.eligible_to_attempt is True
    strict.assert_not_called()
    validate.assert_not_called()
    load_blend.assert_not_called()
    hashed.assert_not_called()
    parquet.assert_not_called()


def test_fingerprint_changes_with_manifest_and_parent_stats(tmp_path: Path) -> None:
    _write_competition_identity(tmp_path)
    _write_candidate(tmp_path, "pc1_p1")
    _write_blend_artifact(tmp_path, "pb1_z")
    _write_candidate(
        tmp_path,
        "pc1_blend",
        source_kind="canonical_probability_blend_v1",
        blend_id="pb1_z",
        parent_ids=["pc1_p1"],
        parent_hashes=["a" * 64],
    )
    fp1 = eligibility_cache_fingerprint("pc1_blend", repository_root=tmp_path)
    manifest = (
        tmp_path
        / "artifacts"
        / "prediction_candidates"
        / "pc1_blend"
        / MANIFEST_FILENAME
    )
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    fp2 = eligibility_cache_fingerprint("pc1_blend", repository_root=tmp_path)
    assert fp1 != fp2
    parent_manifest = (
        tmp_path / "artifacts" / "prediction_candidates" / "pc1_p1" / MANIFEST_FILENAME
    )
    parent_manifest.write_text(
        parent_manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    fp3 = eligibility_cache_fingerprint("pc1_blend", repository_root=tmp_path)
    assert fp2 != fp3


def test_real_native_blend_candidate_is_eligible() -> None:
    candidate_id = "pc1_c54c97e9c32f33a2"
    package = PROJECT_ROOT / "artifacts" / "prediction_candidates" / candidate_id
    if not package.is_dir():
        pytest.skip("Native blend candidate not present")
    result = project_candidate_submission_eligibility(
        candidate_id, repository_root=PROJECT_ROOT
    )
    assert result.eligible_to_attempt is True
    assert result.threshold == 0.17
    assert result.test_row_count == 2500
    assert result.blend_id == "pb1_d047ebe82811ef1a"
    assert eligibility_label(result.state) == "Eligible to attempt generation"


def test_ui_reruns_never_call_strict_readiness() -> None:
    panel = importlib.import_module("apps.experiment_control_panel")
    row = {
        "candidate_id": "pc1_c54c97e9c32f33a2",
        "label": "Native blend",
        "dataset_label": "v5 + v6",
        "dataset_id": "v5_joint_missingness_pattern",
        "source_kind_label": "Canonical blend",
        "exploratory": False,
    }
    eligibility = project_candidate_submission_eligibility(
        "pc1_c54c97e9c32f33a2", repository_root=PROJECT_ROOT
    ).to_mapping()
    with (
        patch.object(panel, "load_cached_candidate_inventory", return_value=[row]),
        patch.object(
            panel,
            "_cached_selected_submission_eligibility",
            return_value=eligibility,
        ),
        patch.object(panel, "eligibility_cache_fingerprint", return_value="fp"),
        patch(
            "src.churn_ml.prediction_candidates.submission_v1.evaluate_submission_readiness",
            side_effect=AssertionError("strict readiness forbidden on UI rerun"),
        ) as strict,
        patch.object(panel, "st") as fake_st,
    ):
        fake_st.session_state = {
            "canonical-submission-candidate": "pc1_c54c97e9c32f33a2",
            "_consumed_launch_nonces": set(),
        }
        fake_st.button.return_value = False
        fake_st.selectbox.return_value = "pc1_c54c97e9c32f33a2"
        fake_st.checkbox.return_value = False
        fake_st.text_input.return_value = "sub_demo"
        panel._canonical_candidate_submission_controls(
            MagicMock(), action_id="generate", handoff=None
        )
        fake_st.checkbox.return_value = True
        fake_st.text_input.return_value = "sub_demo_2"
        panel._canonical_candidate_submission_controls(
            MagicMock(), action_id="generate", handoff=None
        )
    strict.assert_not_called()


def test_generate_launch_does_not_call_strict_synchronously() -> None:
    panel = importlib.import_module("apps.experiment_control_panel")
    row = {
        "candidate_id": "pc1_c54c97e9c32f33a2",
        "label": "Native blend",
        "dataset_label": "v5 + v6",
        "dataset_id": "v5_joint_missingness_pattern",
        "source_kind_label": "Canonical blend",
        "exploratory": False,
    }
    eligibility = project_candidate_submission_eligibility(
        "pc1_c54c97e9c32f33a2", repository_root=PROJECT_ROOT
    ).to_mapping()
    loaded = MagicMock()
    manager = MagicMock()
    manager.start.return_value = MagicMock(job_id="job-1")
    built = MagicMock()
    rendered = MagicMock()
    authorized = MagicMock(argv=("python", "-m", "x"), redacted_argv=("python", "-m", "x"))
    def _button(*args, **kwargs):
        del args
        # Order: Refresh candidates, Validate readiness (Job), Generate submission.
        key = str(kwargs.get("key") or "")
        return key == "canonical-generate-submission"

    with (
        patch.object(panel, "load_cached_candidate_inventory", return_value=[row]),
        patch.object(
            panel,
            "_cached_selected_submission_eligibility",
            return_value=eligibility,
        ),
        patch.object(panel, "eligibility_cache_fingerprint", return_value="fp"),
        patch.object(panel, "build_command", return_value=built),
        patch.object(panel, "rendered_launch", return_value=rendered),
        patch.object(panel, "authorize_launch", return_value=authorized),
        patch.object(panel, "job_manager", return_value=manager),
        patch(
            "src.churn_ml.prediction_candidates.submission_v1.evaluate_submission_readiness",
            side_effect=AssertionError("no sync strict readiness"),
        ),
        patch.object(panel, "st") as fake_st,
    ):
        fake_st.session_state = {
            "canonical-submission-candidate": "pc1_c54c97e9c32f33a2",
            "_consumed_launch_nonces": set(),
        }
        fake_st.button.side_effect = _button
        fake_st.selectbox.return_value = "pc1_c54c97e9c32f33a2"
        fake_st.checkbox.return_value = True
        fake_st.text_input.return_value = "sub_Native_ok"
        panel._canonical_candidate_submission_controls(
            loaded, action_id="generate", handoff=None
        )
    manager.start.assert_called_once()
    assert manager.start.call_args.kwargs["command_id"] == "candidate_submission_v1"
    assert manager.start.call_args.kwargs["action_id"] == "generate"


def test_backend_still_calls_strict_readiness(tmp_path: Path) -> None:
    from src.churn_ml.prediction_candidates import submission_v1 as mod

    with patch.object(
        mod,
        "_evaluate_submission_readiness_with_package",
        return_value=(
            {
                "ready": False,
                "blockers": [{"code": "x", "message": "blocked"}],
            },
            None,
        ),
    ) as strict:
        with pytest.raises(mod.CandidateSubmissionError):
            mod.generate_candidate_submission(
                "pc1_x",
                submission_id="sub_x",
                repository_root=tmp_path,
            )
    strict.assert_called_once()


def test_existing_submission_remains_discoverable() -> None:
    submission = (
        PROJECT_ROOT
        / "artifacts"
        / "candidate_submissions"
        / "sub_Native-blend-3-models-v5-v6_c54c97e9"
    )
    assert (submission / "_SUCCESS").is_file()
    assert (submission / "submission.csv").is_file()


def test_refresh_clears_eligibility_cache() -> None:
    panel = importlib.import_module("apps.experiment_control_panel")
    with (
        patch.object(panel, "load_cached_candidate_inventory", return_value=[]) as inv,
        patch.object(panel, "_cached_selected_submission_eligibility") as cached,
        patch.object(panel, "st") as fake_st,
    ):
        fake_st.session_state = {}
        fake_st.button.side_effect = lambda *a, **k: k.get("key") == (
            "canonical-refresh-candidates"
        )
        panel._canonical_candidate_submission_controls(
            MagicMock(), action_id="generate", handoff=None
        )
    inv.assert_any_call(panel.REPOSITORY_ROOT, force_refresh=True)
    cached.clear.assert_called_once()


def test_failed_strict_validation_writes_no_final_artifact(tmp_path: Path) -> None:
    from src.churn_ml.prediction_candidates import submission_v1 as mod

    out_root = tmp_path / "artifacts" / "candidate_submissions"
    with patch.object(
        mod,
        "_evaluate_submission_readiness_with_package",
        return_value=(
            {
                "ready": False,
                "blockers": [{"code": "payload_corrupt", "message": "bad payload"}],
            },
            None,
        ),
    ):
        with pytest.raises(mod.CandidateSubmissionError):
            mod.generate_candidate_submission(
                "pc1_corrupt",
                submission_id="sub_corrupt_probe",
                repository_root=tmp_path,
            )
    assert not (out_root / "sub_corrupt_probe").exists()


def test_profiler_current_vs_cumulative_and_no_files(tmp_path: Path) -> None:
    from src.churn_ml.control_panel import performance as perf

    perf.enable_performance(True)
    perf.reset_performance()
    perf.begin_render()
    with perf.performance_stage("alpha"):
        perf.record_counter("files_read", 1)
    perf.begin_render()
    with perf.performance_stage("beta"):
        perf.record_counter("files_read", 4)
    text = perf.format_performance_snapshot()
    assert "Current render" in text
    assert "Session cumulative" in text
    snap = perf.performance_snapshot()
    assert snap["current"]["counters"]["files_read"] == 4
    assert snap["cumulative"]["counters"]["files_read"] == 5
    assert snap["cumulative"]["renders"] == 2
    assert not list(tmp_path.rglob("*perf*"))
    perf.enable_performance(False)
    perf.reset_performance()
