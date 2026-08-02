"""Focused recovery and independent-candidate UX tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.churn_ml.control_panel.candidate_preparation import (
    INDEPENDENT_CANDIDATE_COPY,
    durable_preparation_context_from_session,
    ensemble_component_warning,
    independent_candidate_summary,
    list_preparation_jobs,
    package_status_label,
    persist_preparation_context,
    preparation_button_label,
    preview_independent_candidate_rows,
    preview_request_for_selection,
    recover_validation_for_request,
    resolve_request_for_selection,
    restore_selection_from_jobs,
    verify_validation_result,
)
from src.churn_ml.control_panel.selection_state import LOGICAL_SELECTION_KEY
from src.churn_ml.prediction_candidates import cli as candidate_cli
from src.churn_ml.autogluon_artifacts import build_inventory, write_json
from src.churn_ml.prediction_candidates.preparation_request_v1 import (
    PreparationRequestError,
    load_preparation_request,
    materialize_preparation_request,
    preview_preparation_request,
    verify_request_run_identity,
)
from tests.dataset_registry_support import (
    create_synthetic_legacy_tree,
    make_v0_frames,
    register_package,
    write_package_files,
)
from tests.test_autogluon_candidate_preparation_ui_v1 import (
    DATASET_ID,
    PROJECT_ROOT,
    _complete_run,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    processed = root / "data" / "processed"
    create_synthetic_legacy_tree(processed)
    register_package(
        processed,
        "v0_raw_minimal",
        parent_dataset_id=None,
        hypothesis="synthetic v0",
    )
    x_train, y_train, x_test = make_v0_frames()
    write_package_files(
        processed / DATASET_ID,
        dataset_id=DATASET_ID,
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        metadata={"source": "v0_raw_minimal", "description": "synthetic v3"},
    )
    register_package(
        processed,
        DATASET_ID,
        parent_dataset_id="v0_raw_minimal",
        hypothesis="synthetic v3 for AG fixture",
    )
    return root


def test_preview_does_not_write_and_resolve_existing(repo: Path) -> None:
    run_dir = _complete_run(repo)
    run_path = run_dir.relative_to(repo).as_posix().replace("\\", "/")
    models = ["RealTabPFN-v2_r11", "NeuralNetTorch_r37"]
    request_root = repo / "artifacts" / "ui_candidate_preparation_requests"
    assert not request_root.exists() or not any(request_root.glob("apr1_*.json"))
    preview = preview_preparation_request(
        repository_root=repo,
        run_path=run_path,
        selected_models=models,
    )
    assert preview.request_id.startswith("apr1_")
    assert not (repo / preview.relative_path).exists()
    assert resolve_request_for_selection(
        repository_root=repo,
        run_path=run_path,
        selected_models=models,
    ) is None

    materialized = materialize_preparation_request(
        repository_root=repo,
        run_path=run_path,
        selected_models=models,
    )
    assert materialized.request_id == preview.request_id
    resolved = resolve_request_for_selection(
        repository_root=repo,
        run_path=run_path,
        selected_models=models,
    )
    assert resolved is not None
    assert resolved.request_id == preview.request_id

    # Changed selection does not resolve the old request.
    assert (
        resolve_request_for_selection(
            repository_root=repo,
            run_path=run_path,
            selected_models=["RealTabPFN-v2_r11", "LightGBMPrep_r31"],
        )
        is None
    )
    # Order change follows identity semantics (different request ID).
    reordered = preview_preparation_request(
        repository_root=repo,
        run_path=run_path,
        selected_models=["NeuralNetTorch_r37", "RealTabPFN-v2_r11"],
    )
    assert reordered.request_id != preview.request_id


def test_stale_and_cross_run_request_rejected(repo: Path) -> None:
    run_a = _complete_run(repo, name="run-a")
    run_b = _complete_run(repo, name="run-b")
    path_a = run_a.relative_to(repo).as_posix().replace("\\", "/")
    path_b = run_b.relative_to(repo).as_posix().replace("\\", "/")
    models = ["RealTabPFN-v2_r11"]
    request = materialize_preparation_request(
        repository_root=repo,
        run_path=path_a,
        selected_models=models,
    )
    metadata = json.loads((run_a / "run_metadata.json").read_text(encoding="utf-8"))
    metadata["note"] = "tampered-after-request"
    write_json(run_a / "run_metadata.json", metadata)
    write_json(run_a / "artifact_inventory.json", build_inventory(run_a))
    # Current selection identity changes with the run, so the old request is not recovered.
    assert (
        resolve_request_for_selection(
            repository_root=repo,
            run_path=path_a,
            selected_models=models,
        )
        is None
    )
    loaded = load_preparation_request(request.request_id, repository_root=repo)
    with pytest.raises(PreparationRequestError, match="identity|stale"):
        verify_request_run_identity(loaded, repository_root=repo)
    # Request from another run must not resolve for path_b selection.
    assert (
        resolve_request_for_selection(
            repository_root=repo,
            run_path=path_b,
            selected_models=models,
        )
        is None
    )


def test_durable_selection_and_job_restore(repo: Path, tmp_path: Path) -> None:
    run_dir = _complete_run(repo)
    run_path = run_dir.relative_to(repo).as_posix().replace("\\", "/")
    models = ["RealTabPFN-v2_r11", "NeuralNetTorch_r37", "LightGBMPrep_r31"]
    request = materialize_preparation_request(
        repository_root=repo,
        run_path=run_path,
        selected_models=models,
    )
    session: dict = {}
    persist_preparation_context(
        session,
        run_path=run_path,
        selected_models=models,
        request_id=request.request_id,
    )
    assert LOGICAL_SELECTION_KEY in session
    durable = durable_preparation_context_from_session(session)
    assert durable is not None
    assert durable["selected_models"] == models
    assert durable["run_path"] == run_path

    jobs_root = tmp_path / "jobs"
    job_id = "job-validate-1"
    job_dir = jobs_root / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "job_id": job_id,
                "command_id": "autogluon_candidate_preparation_v1",
                "action_id": "validate",
                "created_at_utc": "2026-08-02T08:00:00Z",
                "references": {
                    "request_id": request.request_id,
                    "run_path": run_path,
                    "selected_models": ",".join(models),
                    "source_type": "managed_autogluon",
                    "operation": "validate",
                },
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "status.json").write_text(
        json.dumps({"schema_version": 2, "state": "succeeded"}),
        encoding="utf-8",
    )
    restored = restore_selection_from_jobs(
        jobs_root=jobs_root,
        repository_root=repo,
        available_run_paths=[run_path],
        preferred_run_path=run_path,
    )
    assert restored is not None
    assert restored["selected_models"] == models
    assert restored["request_id"] == request.request_id

    # Unrelated job cannot restore.
    unrelated = jobs_root / "other"
    unrelated.mkdir()
    (unrelated / "job.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "job_id": "other",
                "command_id": "prediction_blend_v1",
                "action_id": "search",
                "created_at_utc": "2026-08-02T09:00:00Z",
                "references": {
                    "request_id": "x",
                    "run_path": run_path,
                    "source_type": "other",
                },
            }
        ),
        encoding="utf-8",
    )
    assert list_preparation_jobs(jobs_root, run_path=run_path)
    # Stale request identity blocks restore.
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    metadata["note"] = "tampered-after-request"
    write_json(run_dir / "run_metadata.json", metadata)
    write_json(run_dir / "artifact_inventory.json", build_inventory(run_dir))
    assert (
        restore_selection_from_jobs(
            jobs_root=jobs_root,
            repository_root=repo,
            available_run_paths=[run_path],
            preferred_run_path=run_path,
        )
        is None
    )


def test_validation_recovery_without_request_obj(repo: Path, tmp_path: Path) -> None:
    run_dir = _complete_run(repo)
    run_path = run_dir.relative_to(repo).as_posix().replace("\\", "/")
    models = ["RealTabPFN-v2_r11", "NeuralNetTorch_r37"]
    request = materialize_preparation_request(
        repository_root=repo,
        run_path=run_path,
        selected_models=models,
    )
    jobs_root = tmp_path / "jobs"
    job_id = "e055-like"
    job_dir = jobs_root / job_id
    job_dir.mkdir(parents=True)
    payload = {
        "ok": True,
        "command": "validate",
        "artifacts_written": False,
        "all_models_valid": True,
        "request_id": request.request_id,
        "run_path": run_path,
        "selected_models": models,
        "results": [
            {
                "ok": True,
                "model_name": models[0],
                "train_row_count": 6,
                "test_row_count": 3,
                "positive_class_label": 1,
                "candidate_id": "pc1_a",
            },
            {
                "ok": True,
                "model_name": models[1],
                "train_row_count": 6,
                "test_row_count": 3,
                "positive_class_label": 1,
                "candidate_id": "pc1_b",
            },
        ],
    }
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "job_id": job_id,
                "command_id": "autogluon_candidate_preparation_v1",
                "action_id": "validate",
                "created_at_utc": "2026-08-02T08:00:00Z",
                "references": {
                    "request_id": request.request_id,
                    "run_path": run_path,
                    "selected_models": ",".join(models),
                    "source_type": "managed_autogluon",
                    "operation": "validate",
                },
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "status.json").write_text(
        json.dumps({"schema_version": 2, "state": "succeeded"}),
        encoding="utf-8",
    )
    (job_dir / "stdout.log").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    resolved = resolve_request_for_selection(
        repository_root=repo,
        run_path=run_path,
        selected_models=models,
    )
    assert resolved is not None
    recovered = recover_validation_for_request(jobs_root=jobs_root, request=resolved)
    assert recovered is not None
    assert recovered["recovery_state"] == "succeeded"
    assert recovered["payload"]["request_id"] == request.request_id

    with pytest.raises(Exception):
        verify_validation_result(
            {**payload, "request_id": "wrong"},
            request_id=request.request_id,
            run_path=run_path,
            selected_models=models,
        )
    with pytest.raises(Exception):
        verify_validation_result(
            {**payload, "selected_models": [models[0]]},
            request_id=request.request_id,
            run_path=run_path,
            selected_models=models,
        )
    with pytest.raises(Exception):
        verify_validation_result(
            {**payload, "artifacts_written": True},
            request_id=request.request_id,
            run_path=run_path,
            selected_models=models,
        )
    with pytest.raises(Exception):
        verify_validation_result(
            {**payload, "results": [payload["results"][0]]},
            request_id=request.request_id,
            run_path=run_path,
            selected_models=models,
        )

    # Failed job does not unlock preparation.
    (job_dir / "status.json").write_text(
        json.dumps({"schema_version": 2, "state": "failed"}),
        encoding="utf-8",
    )
    failed = recover_validation_for_request(jobs_root=jobs_root, request=resolved)
    assert failed is not None
    assert failed["payload"] is None
    assert failed["recovery_state"] == "failed"


def test_all_models_valid_cli_helper() -> None:
    assert candidate_cli._all_models_valid(
        [{"ok": True}, {"ok": True}, {"ok": True}, {"ok": True}, {"ok": True}]
    )
    assert not candidate_cli._all_models_valid([{"ok": True}, {"ok": False}])
    assert not candidate_cli._all_models_valid([])
    assert not candidate_cli._all_models_valid([{"ok": True}, "bad"])
    assert not candidate_cli._all_models_valid([{"model_name": "x"}])


def test_independent_candidate_ux_helpers(repo: Path) -> None:
    run_dir = _complete_run(repo)
    run_path = run_dir.relative_to(repo).as_posix().replace("\\", "/")
    models = [
        "RealTabPFN-v2_r11",
        "NeuralNetTorch_r37",
        "LightGBMPrep_r31",
        "LightGBMPrep_r41",
        "WeightedEnsemble_L2",
    ]
    preview = preview_request_for_selection(
        repository_root=repo,
        run_path=run_path,
        selected_models=models,
    )
    rows = preview_independent_candidate_rows(preview)
    assert len(rows) == 5
    ids = [row["Future candidate ID"] for row in rows]
    assert len(set(ids)) == 5
    assert "separate canonical candidate" in INDEPENDENT_CANDIDATE_COPY
    assert independent_candidate_summary(5).startswith("5 selected models")
    assert preparation_button_label(1) == "Prepare 1 independent candidate"
    assert preparation_button_label(5) == "Prepare 5 independent candidates"
    warning = ensemble_component_warning(models)
    assert warning is not None
    assert "Preparation still creates independent candidates" in warning
    assert package_status_label("already_prepared") == "Prepared"
    assert package_status_label("ready_to_validate") == "Not prepared"
    assert package_status_label("invalid_existing_candidate") == "Invalid"


def test_real_request_and_job_recoverable() -> None:
    request_id = "apr1_0492a26999afd138"
    job_id = "e055e027-7e6b-484c-93cf-a22cee89b5f8"
    run_path = "artifacts/autogluon_runs/ag-v6-focused-hybrid-s42-20260801-183821"
    models = [
        "RealTabPFN-v2_r11_BAG_L1",
        "NeuralNetTorch_r37_BAG_L1",
        "LightGBMPrep_r31_BAG_L1",
        "LightGBMPrep_r41_BAG_L1",
        "WeightedEnsemble_L2",
    ]
    request_path = (
        PROJECT_ROOT / "artifacts/ui_candidate_preparation_requests" / f"{request_id}.json"
    )
    job_dir = PROJECT_ROOT / "artifacts/ui_jobs" / job_id
    if not request_path.is_file() or not job_dir.is_dir():
        pytest.skip("Real preparation request/job artifacts not present")

    preview = preview_preparation_request(
        repository_root=PROJECT_ROOT,
        run_path=run_path,
        selected_models=models,
    )
    assert preview.request_id == request_id
    resolved = resolve_request_for_selection(
        repository_root=PROJECT_ROOT,
        run_path=run_path,
        selected_models=models,
    )
    assert resolved is not None
    assert resolved.request_id == request_id
    recovered = recover_validation_for_request(
        jobs_root=PROJECT_ROOT / "artifacts/ui_jobs",
        request=resolved,
    )
    assert recovered is not None
    assert recovered["job_id"] == job_id
    assert recovered["recovery_state"] == "succeeded"
    assert recovered["payload"] is not None
    assert recovered["payload"]["all_models_valid"] is True
    assert len(recovered["payload"]["results"]) == 5


def test_app_tabs_still_declared() -> None:
    blend_page = (
        PROJECT_ROOT / "src/churn_ml/control_panel/blend_page.py"
    ).read_text(encoding="utf-8")
    prepare = (
        PROJECT_ROOT / "src/churn_ml/control_panel/candidate_preparation_page.py"
    ).read_text(encoding="utf-8")
    assert "Prepare candidates" in blend_page
    assert "Build blend" in blend_page
    assert "Materialized blends" in blend_page
    assert "st.text_input(\"Run" not in prepare
    assert "Open Jobs" in prepare
    assert "INDEPENDENT_CANDIDATE_COPY" in prepare
    assert "separate canonical candidate" in INDEPENDENT_CANDIDATE_COPY
