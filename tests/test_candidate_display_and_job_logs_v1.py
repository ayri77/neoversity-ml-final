"""Focused tests for candidate display labels and Job log/summary helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.churn_ml.control_panel.candidate_display import (
    CandidateDisplay,
    candidate_display_map,
    compact_model_name,
    dataset_display_label,
    disambiguate_compact_labels,
    experiment_models_label,
    load_candidate_metadata,
    project_candidate_quality_rows,
    project_heatmap_labels,
    project_optuna_study_rows,
    project_pairwise_rows,
    project_weight_rows,
    replace_candidate_ids_with_labels,
    resolve_candidate_display,
)
from src.churn_ml.control_panel.job_logs import (
    INLINE_RENDER_MAX_BYTES,
    JobLogError,
    format_log_window_status,
    inspect_job_log,
    job_log_download_bytes,
    job_log_fingerprint,
    read_job_log_full,
    read_job_log_head,
    read_job_log_tail,
    resolve_job_log_path,
)
from src.churn_ml.control_panel.job_summary import (
    build_job_summary,
    generic_job_summary,
    lightweight_job_list_label,
    resolve_job_references,
)


REPO = Path(__file__).resolve().parents[1]


def test_atomic_autogluon_full_and_compact_labels() -> None:
    display = resolve_candidate_display(
        {
            "candidate_id": "pc1_34ffeec20b01058f",
            "source_model_name": "LightGBMPrep_r31_BAG_L1",
            "source_kind": "autogluon_standalone_v1",
            "dataset_id": "v6_compact_missingness_indicators",
        }
    )
    assert display.primary_label == (
        "LightGBMPrep_r31_BAG_L1 · AutoGluon · v6 compact missingness"
    )
    assert display.compact_label == "LightGBMPrep_r31 · v6"
    assert compact_model_name("LightGBMPrep_r31_BAG_L1") == "LightGBMPrep_r31"
    assert display.candidate_id == "pc1_34ffeec20b01058f"


def test_dataset_labels_v5_and_v6() -> None:
    assert dataset_display_label("v5_joint_missingness_pattern") == "v5 joint missingness"
    assert (
        dataset_display_label("v6_compact_missingness_indicators")
        == "v6 compact missingness"
    )


def test_blend_candidate_label_and_parent_resolution() -> None:
    metadata = {
        "pc1_parent_a": {
            "candidate_id": "pc1_parent_a",
            "source_model_name": "LightGBMPrep_r31_BAG_L1",
            "source_kind": "autogluon_standalone_v1",
            "dataset_id": "v6_compact_missingness_indicators",
        },
        "pc1_parent_b": {
            "candidate_id": "pc1_parent_b",
            "source_model_name": "WeightedEnsemble_L2",
            "source_kind": "autogluon_standalone_v1",
            "dataset_id": "v5_joint_missingness_pattern",
        },
        "pc1_blend": {
            "candidate_id": "pc1_blend",
            "source_model_name": "blend_optimized_native",
            "source_kind": "canonical_probability_blend_v1",
            "dataset_id": "v6_compact_missingness_indicators",
            "parent_candidate_ids": ["pc1_parent_a", "pc1_parent_b"],
            "final_deployment_weights": {
                "pc1_parent_a": 0.88,
                "pc1_parent_b": 0.12,
            },
            "strategy": "optimized",
            "optimizer_backend": "native",
        },
    }
    display = resolve_candidate_display(
        "pc1_blend", metadata_by_id=metadata
    )
    assert display.primary_label.startswith("Native blend · 2 models")
    assert "v5" in display.primary_label and "v6" in display.primary_label
    assert "LGB" in display.compact_label or "LightGBMPrep" in display.compact_label
    assert "88%" in display.compact_label


def test_missing_metadata_fallback_and_no_mutation() -> None:
    payload = {"candidate_id": "pc1_deadbeefdeadbeef"}
    before = json.dumps(payload, sort_keys=True)
    display = resolve_candidate_display(payload)
    assert display.primary_label == "Candidate pc1_deadbeefdeadbeef"
    assert display.missing_reason
    assert json.dumps(payload, sort_keys=True) == before


def test_duplicate_compact_labels_disambiguated() -> None:
    displays = [
        CandidateDisplay(
            candidate_id="pc1_aaaaaaaaaaaaaaaa",
            primary_label="A",
            compact_label="LightGBMPrep_r31 · v6",
            model_name="LightGBMPrep_r31_BAG_L1",
            source_label="AutoGluon",
            dataset_label="v6 compact missingness",
            dataset_id="v6_compact_missingness_indicators",
            source_kind="autogluon_standalone_v1",
            exploratory=False,
        ),
        CandidateDisplay(
            candidate_id="pc1_bbbbbbbbbbbbbbbb",
            primary_label="B",
            compact_label="LightGBMPrep_r31 · v6",
            model_name="LightGBMPrep_r31_BAG_L1",
            source_label="AutoGluon",
            dataset_label="v6 compact missingness",
            dataset_id="v6_compact_missingness_indicators",
            source_kind="autogluon_standalone_v1",
            exploratory=False,
        ),
    ]
    unique = disambiguate_compact_labels(displays)
    assert unique["pc1_aaaaaaaaaaaaaaaa"] != unique["pc1_bbbbbbbbbbbbbbbb"]
    assert "[pc1_aaaa" in unique["pc1_aaaaaaaaaaaaaaaa"]


def test_quality_pairwise_heatmap_and_weight_projections() -> None:
    displays = candidate_display_map(
        ["pc1_a", "pc1_b"],
        metadata_by_id={
            "pc1_a": {
                "candidate_id": "pc1_a",
                "source_model_name": "LightGBMPrep_r31_BAG_L1",
                "source_kind": "autogluon_standalone_v1",
                "dataset_id": "v6_compact_missingness_indicators",
            },
            "pc1_b": {
                "candidate_id": "pc1_b",
                "source_model_name": "WeightedEnsemble_L2",
                "source_kind": "autogluon_standalone_v1",
                "dataset_id": "v5_joint_missingness_pattern",
            },
        },
    )
    quality = project_candidate_quality_rows(
        [
            {
                "candidate_id": "pc1_a",
                "threshold_0_5": {"balanced_accuracy": 0.8},
                "descriptive_oof_optimal_threshold": 0.2,
                "descriptive_oof_optimal_metrics": {
                    "balanced_accuracy": 0.9,
                    "sensitivity": 0.91,
                    "specificity": 0.89,
                },
            }
        ],
        displays,
    )
    assert list(quality[0])[0] == "Model"
    assert "LightGBMPrep_r31_BAG_L1" in quality[0]["Model"]
    assert quality[0]["Candidate ID"] == "pc1_a"

    pairwise = project_pairwise_rows(
        [
            {
                "candidate_a": "pc1_a",
                "candidate_b": "pc1_b",
                "pearson": 0.5,
                "spearman": 0.4,
                "mean_abs_diff": 0.01,
                "rmse": 0.02,
                "prediction_disagreement_rate": 0.03,
                "joint_error_rate": 0.04,
                "error_overlap": 0.05,
                "positive_class_disagreement": 0.06,
                "negative_class_disagreement": 0.07,
            }
        ],
        displays,
    )
    assert "Model A" in pairwise[0]
    assert pairwise[0]["Candidate A ID"] == "pc1_a"
    assert pairwise[0]["Candidate B ID"] == "pc1_b"

    axes = project_heatmap_labels(["pc1_a", "pc1_b"], displays)
    assert axes[0] != axes[1]
    assert "pc1_" not in axes[0] or "·" in axes[0]

    weights = {"pc1_a": 0.88, "pc1_b": 0.12}
    rows = project_weight_rows(weights, displays)
    assert rows[0]["Weight"] == 0.88
    assert "LightGBMPrep" in rows[0]["Model"]
    assert rows[0]["Candidate ID"] == "pc1_a"
    replaced = replace_candidate_ids_with_labels(weights, displays)
    assert "pc1_a" not in replaced

    optuna = project_optuna_study_rows(
        [{"repeat": 0, "fold": 1, "best_value": 0.9, "threshold": 0.17, "weights": [0.8, 0.2]}],
        displays,
        ["pc1_a", "pc1_b"],
    )
    assert optuna[0]["Repeat"] == 0
    assert any("LightGBMPrep" in key for key in optuna[0])


def test_experiment_models_label_uses_central_forms() -> None:
    displays = resolve_candidate_display(
        {
            "candidate_id": "pc1_x",
            "source_model_name": "LightGBMPrep_r31_BAG_L1",
            "source_kind": "autogluon_standalone_v1",
            "dataset_id": "v6_compact_missingness_indicators",
        }
    )
    we = resolve_candidate_display(
        {
            "candidate_id": "pc1_y",
            "source_model_name": "WeightedEnsemble_L2",
            "source_kind": "autogluon_standalone_v1",
            "dataset_id": "v5_joint_missingness_pattern",
        }
    )
    label = experiment_models_label([displays, we], optimizer_backend="native")
    assert label.startswith("Native · ")
    assert "LightGBMPrep_r31" in label
    assert "WeightedEnsemble v5" in label


def test_real_candidate_manifest_not_mutated() -> None:
    candidate_id = "pc1_34ffeec20b01058f"
    package = REPO / "artifacts" / "prediction_candidates" / candidate_id
    if not (package / "candidate_manifest.json").is_file():
        pytest.skip("local candidate artifact unavailable")
    before = (package / "candidate_manifest.json").read_bytes()
    display = resolve_candidate_display(candidate_id, repository_root=REPO)
    after = (package / "candidate_manifest.json").read_bytes()
    assert before == after
    assert "LightGBMPrep_r31" in display.primary_label
    meta = load_candidate_metadata(candidate_id, repository_root=REPO)
    assert meta["candidate_id"] == candidate_id


def test_job_log_empty_and_newline_variants(tmp_path: Path) -> None:
    empty = tmp_path / "stdout.log"
    empty.write_bytes(b"")
    info = inspect_job_log(empty)
    assert info.exists
    assert info.line_count == 0
    assert read_job_log_full(empty) == ""

    one = tmp_path / "one.log"
    one.write_bytes(b"hello")
    assert inspect_job_log(one).line_count == 1
    assert read_job_log_full(one) == "hello"

    one_nl = tmp_path / "one_nl.log"
    one_nl.write_bytes(b"hello\n")
    assert inspect_job_log(one_nl).line_count == 1

    crlf = tmp_path / "crlf.log"
    crlf.write_bytes(b"a\r\nb\r\nc")
    assert inspect_job_log(crlf).line_count == 3
    assert read_job_log_head(crlf, 2) == "a\nb"
    assert read_job_log_tail(crlf, 2) == "b\nc"


def test_job_log_invalid_utf8_display_and_exact_download(tmp_path: Path) -> None:
    path = tmp_path / "stdout.log"
    raw = b"ok\n\xff\xfe bad\nend"
    path.write_bytes(raw)
    text = read_job_log_full(path)
    assert "bad" in text
    assert "\ufffd" in text or "bad" in text
    assert job_log_download_bytes(path) == raw


def test_job_log_missing_path_escape_and_growing(tmp_path: Path) -> None:
    job_root = tmp_path / "job"
    job_root.mkdir()
    missing = inspect_job_log(job_root / "stdout.log")
    assert missing.exists is False
    assert read_job_log_tail(job_root / "stdout.log", 10) == ""

    path = resolve_job_log_path(job_root, "stdout.log")
    path.write_text("line1\nline2\n", encoding="utf-8")
    first = inspect_job_log(path)
    path.write_text("line1\nline2\nline3\n", encoding="utf-8")
    second = inspect_job_log(path)
    assert job_log_fingerprint(first) != job_log_fingerprint(second)

    with pytest.raises(JobLogError):
        resolve_job_log_path(job_root, "../stdout.log")


def test_log_window_status_and_inline_threshold() -> None:
    assert "last 300 of 1,487" in format_log_window_status(
        mode="last",
        requested_lines=300,
        total_lines=1487,
        rendered_lines=300,
    )
    assert "complete log" in format_log_window_status(
        mode="full",
        requested_lines=10,
        total_lines=10,
        rendered_lines=10,
    )
    assert INLINE_RENDER_MAX_BYTES == 2 * 1024 * 1024


def test_native_and_optuna_job_summaries_from_proof_artifacts() -> None:
    native_id = "2d058521-1c7a-4f27-bfff-f40461ea22e4"
    optuna_id = "19cd9ce6-fda7-40c4-95ec-39b48f2d63ff"
    native_root = REPO / "artifacts" / "ui_jobs" / native_id
    optuna_root = REPO / "artifacts" / "ui_jobs" / optuna_id
    if not (native_root / "stdout.log").is_file():
        pytest.skip("local job artifacts unavailable")

    native_job = json.loads((native_root / "job.json").read_text(encoding="utf-8"))
    native_status = json.loads((native_root / "status.json").read_text(encoding="utf-8"))
    native = build_job_summary(
        job=native_job,
        status=native_status,
        job_root=native_root,
        repository_root=REPO,
    )
    assert native.available
    row_map = dict(native.rows)
    assert abs(float(row_map["Honest mean BA"]) - 0.8982157138796905) < 1e-12
    assert row_map["Artifacts written"] == "No"
    assert native.tables
    weight_models = " ".join(str(row["Model"]) for row in native.tables[0][1])
    assert "LightGBMPrep" in weight_models
    assert lightweight_job_list_label(native_job) == "Blend search · Native · 3 models"

    optuna_job = json.loads((optuna_root / "job.json").read_text(encoding="utf-8"))
    optuna_status = json.loads((optuna_root / "status.json").read_text(encoding="utf-8"))
    # Tail-only would miss metrics; summary must use complete stdout.
    optuna = build_job_summary(
        job=optuna_job,
        status=optuna_status,
        job_root=optuna_root,
        repository_root=REPO,
    )
    assert optuna.available
    optuna_rows = dict(optuna.rows)
    assert abs(float(optuna_rows["Honest mean BA"]) - 0.8961171369461904) < 1e-12
    assert "Descriptive" not in optuna_rows
    assert any("Honest" in note or "Descriptive" in note for note in optuna.notes)
    info = inspect_job_log(optuna_root / "stdout.log")
    assert info.line_count > 300
    assert len(job_log_download_bytes(optuna_root / "stdout.log")) == info.size_bytes


def test_validation_summary_and_generic_unknown() -> None:
    validate_id = "e055e027-7e6b-484c-93cf-a22cee89b5f8"
    root = REPO / "artifacts" / "ui_jobs" / validate_id
    if not (root / "stdout.log").is_file():
        pytest.skip("local validation job unavailable")
    job = json.loads((root / "job.json").read_text(encoding="utf-8"))
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    summary = build_job_summary(
        job=job,
        status=status,
        job_root=root,
        repository_root=REPO,
    )
    assert summary.available
    rows = dict(summary.rows)
    assert rows["Models validated"] == "5 of 5"
    assert rows["Artifacts written"] == "No"
    refs = resolve_job_references(job, repository_root=REPO)
    assert "RealTabPFN-v2_r11_BAG_L1" in refs["selected_models"]

    unknown = generic_job_summary(
        {"command_id": "other_cmd", "action_id": "run", "created_at_utc": "t"},
        {"state": "succeeded", "exit_code": 0, "elapsed_seconds": 1},
    )
    assert dict(unknown.rows)["Command"] == "other_cmd"


def test_partial_and_malformed_stdout_safe(tmp_path: Path) -> None:
    job_root = tmp_path / "job"
    job_root.mkdir()
    (job_root / "stdout.log").write_text('{"ok": true, "command":', encoding="utf-8")
    summary = build_job_summary(
        job={
            "command_id": "prediction_blend_v1",
            "action_id": "search",
            "references": {"request_id": "br1_x"},
        },
        status={"state": "running"},
        job_root=job_root,
        repository_root=REPO,
    )
    assert summary.pending or not summary.available

    (job_root / "stdout.log").write_text("not-json-at-all {{{", encoding="utf-8")
    bad = build_job_summary(
        job={
            "command_id": "prediction_blend_v1",
            "action_id": "search",
            "references": {"request_id": "br1_x"},
        },
        status={"state": "succeeded"},
        job_root=job_root,
        repository_root=REPO,
    )
    assert bad.available is False
    assert bad.title == "Structured summary unavailable"


def test_jobs_list_label_and_references_resolution() -> None:
    label = lightweight_job_list_label(
        {
            "command_id": "prediction_blend_v1",
            "action_id": "search",
            "references": {
                "optimizer": "native",
                "candidate_ids": "a,b,c",
            },
        }
    )
    assert label == "Blend search · Native · 3 models"
    refs = resolve_job_references(
        {
            "references": {
                "candidate_ids": "pc1_34ffeec20b01058f,pc1_8da68e3d2071fb71",
                "request_id": "br1_452a4681d17e939f",
            }
        },
        repository_root=REPO,
    )
    assert refs["request_id"] == "br1_452a4681d17e939f"
    if (REPO / "artifacts/prediction_candidates/pc1_34ffeec20b01058f/candidate_manifest.json").is_file():
        assert any("LightGBMPrep" in item for item in refs["candidates"])
