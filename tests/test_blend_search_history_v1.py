"""Focused tests for blend search history, resume, and historical materialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.churn_ml.blending.artifact_v1 import search_blend
from src.churn_ml.blending.compatibility_v1 import load_compatible_candidates
from src.churn_ml.blending.request_v1 import materialize_blend_ui_request
from src.churn_ml.control_panel.blend_workspace import (
    authorized_blend_argv_values,
    expected_blend_id,
)
from src.churn_ml.control_panel.command_builder import build_command
from src.churn_ml.control_panel.launch import (
    LaunchAuthorizationError,
    authorize_launch,
    rendered_launch,
)
from src.churn_ml.control_panel.registry import load_registry
from src.churn_ml.control_panel.saved_blend_search import (
    compare_materialized_to_saved,
    configuration_from_saved_search,
    consume_loaded_blend_configuration_pending,
    discover_saved_blend_searches,
    filter_materialize_jobs_for_saved_search,
    historical_materialize_launch_plan,
    load_persisted_blend_configuration,
    persist_loaded_blend_configuration,
    recover_search_for_request,
    resolve_displayed_search_job_id,
)
from src.churn_ml.prediction_candidates.contract_v1 import (
    MANIFEST_FILENAME,
    SCHEMA_VERSION,
    create_candidate_package,
)
from tests.dataset_registry_support import (
    create_synthetic_legacy_tree,
    register_package,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "run_prediction_blend.py").write_text(
        "# stub\n", encoding="utf-8"
    )
    return root


def _make_candidate(
    repo: Path,
    *,
    name: str,
    oof_probs: np.ndarray,
    test_probs: np.ndarray,
) -> str:
    dataset_id = "v0_raw_minimal"
    y = pd.read_parquet(
        repo / "data" / "processed" / dataset_id / "y_train.parquet"
    ).iloc[:, 0]
    oof = pd.DataFrame(
        {
            "row_position": np.arange(len(y), dtype=np.int64),
            "target": y.astype("int64").to_numpy(),
            "probability_positive": oof_probs.astype(np.float64),
        }
    )
    test = pd.DataFrame(
        {
            "row_position": np.arange(len(test_probs), dtype=np.int64),
            "probability_positive": test_probs.astype(np.float64),
        }
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "source_kind": "autogluon_standalone_v1",
        "dataset_id": dataset_id,
        "source_run_path": f"artifacts/fake/{name}",
        "source_config_sha256": (name.encode().hex() + "0" * 64)[:64],
        "source_model_name": name,
        "oof_protocol": "unit_test_protocol",
        "positive_class_label": 1,
        "probability_semantics": "P(y=1)",
    }
    package = create_candidate_package(
        repository_root=repo,
        identity=identity,
        oof=oof,
        test=test,
        manifest_fields={
            "dataset_id": dataset_id,
            "exploratory": False,
            "source_run_path": identity["source_run_path"],
            "source_config_path": f"artifacts/fake/{name}/config.yaml",
            "source_config_sha256": identity["source_config_sha256"],
            "source_model_name": name,
            "source_model_type": "UnitModel",
            "source_predictor_path": "unavailable",
            "source_autogluon_version": "unavailable",
            "source_metric_name": "balanced_accuracy",
            "source_metric_value": 0.7,
            "source_leaderboard_metadata": {"model": name},
            "provenance": {"adapter": "unit_test"},
        },
        source_metadata={"schema_version": 1, "name": name},
        candidates_root_relative="artifacts/prediction_candidates",
    )
    return package.candidate_id


def _three_candidates(repo: Path) -> tuple[str, str, str]:
    y = pd.read_parquet(
        repo / "data" / "processed" / "v0_raw_minimal" / "y_train.parquet"
    ).iloc[:, 0]
    n = len(y)
    rng = np.random.default_rng(1)
    base = np.clip(0.2 + 0.6 * y.to_numpy() + rng.normal(0, 0.05, size=n), 0.05, 0.95)
    alt = np.clip(0.8 - 0.5 * y.to_numpy() + rng.normal(0, 0.05, size=n), 0.05, 0.95)
    third = np.clip(0.5 + 0.2 * y.to_numpy() + rng.normal(0, 0.05, size=n), 0.05, 0.95)
    a = _make_candidate(
        repo,
        name="WeightedEnsemble_L2",
        oof_probs=base,
        test_probs=np.linspace(0.2, 0.8, 3),
    )
    b = _make_candidate(
        repo,
        name="NeuralNetTorch_r37_BAG_L1",
        oof_probs=alt,
        test_probs=np.linspace(0.8, 0.2, 3),
    )
    c = _make_candidate(
        repo,
        name="LightGBMPrep_r31_BAG_L1",
        oof_probs=third,
        test_probs=np.linspace(0.3, 0.7, 3),
    )
    return a, b, c


def _write_job(
    jobs_root: Path,
    *,
    job_id: str,
    request_id: str,
    request_path: str,
    state: str = "succeeded",
    created_at_utc: str = "2026-08-01T10:00:00Z",
    stdout: str | None = None,
    action_id: str = "search",
    references: dict[str, Any] | None = None,
) -> Path:
    job_dir = jobs_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    refs = {
        "request_id": request_id,
        "request_path": request_path,
        **(references or {}),
    }
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "job_id": job_id,
                "command_id": "prediction_blend_v1",
                "action_id": action_id,
                "created_at_utc": created_at_utc,
                "references": refs,
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "status.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": state,
                "exit_code": 0 if state == "succeeded" else 1,
                "started_at_utc": created_at_utc,
                "finished_at_utc": created_at_utc,
            }
        ),
        encoding="utf-8",
    )
    if stdout is not None:
        (job_dir / "stdout.log").write_text(stdout, encoding="utf-8")
    return job_dir


def _search_stdout(repo: Path, request: Any) -> str:
    pool = load_compatible_candidates(
        list(request.candidate_ids), repository_root=repo
    )
    payload = search_blend(pool, request.blend_settings())
    payload["command"] = "search"
    payload["request_id"] = request.request_id
    return json.dumps(payload)


def _native_request(repo: Path, *, seed: int = 7) -> Any:
    a, b, c = _three_candidates(repo)
    return materialize_blend_ui_request(
        repository_root=repo,
        candidate_ids=[a, b, c],
        method="optimized_native",
        folds=2,
        repeats=1,
        seed=seed,
        max_active_models=3,
        dirichlet_draws=8,
        pairwise_grid_step=0.25,
    )


def _prepare_native_search(repo: Path) -> tuple[Any, Path]:
    request = _native_request(repo)
    jobs_root = repo / "artifacts" / "ui_jobs"
    _write_job(
        jobs_root,
        job_id="job_native_ok",
        request_id=request.request_id,
        request_path=request.relative_path,
        stdout=_search_stdout(repo, request),
        created_at_utc="2026-08-01T12:00:00Z",
    )
    return request, jobs_root


def test_discover_valid_succeeded_search_without_session(repo: Path) -> None:
    request, jobs_root = _prepare_native_search(repo)
    rows = discover_saved_blend_searches(repository_root=repo, jobs_root=jobs_root)
    assert len(rows) == 1
    saved = rows[0]
    assert saved.reusable is True
    assert saved.request_id == request.request_id
    assert saved.max_active_models == 3
    assert saved.optimizer_backend == "native"
    assert saved.honest_mean_balanced_accuracy is not None
    joined = " ".join(saved.candidate_labels)
    assert "LightGBMPrep_r31" in joined or "LGB" in joined
    assert all(not label.startswith("pc1_") for label in saved.candidate_labels)
    assert saved.search_result is not None
    assert "honest_meta_cv_metrics" in saved.search_result
    assert saved.reusable is True
    plan = historical_materialize_launch_plan(saved)
    assert plan["request_path"] == request.relative_path
    assert plan["requires_confirmation"] is True


def test_missing_and_malformed_request_blocked(repo: Path) -> None:
    jobs_root = repo / "artifacts" / "ui_jobs"
    _write_job(
        jobs_root,
        job_id="job_missing_request",
        request_id="br1_missing_request",
        request_path="artifacts/ui_blend_requests/br1_missing_request.json",
        stdout='{"ok": true, "command": "search", "blend_id": "pb1_x"}',
    )
    malformed = repo / "artifacts" / "ui_blend_requests" / "br1_malformed.json"
    malformed.parent.mkdir(parents=True, exist_ok=True)
    malformed.write_text("{not-json", encoding="utf-8")
    _write_job(
        jobs_root,
        job_id="job_malformed_request",
        request_id="br1_malformed",
        request_path="artifacts/ui_blend_requests/br1_malformed.json",
        stdout='{"ok": true, "command": "search", "blend_id": "pb1_x"}',
    )
    rows = {
        item.request_id: item
        for item in discover_saved_blend_searches(
            repository_root=repo, jobs_root=jobs_root
        )
    }
    assert rows["br1_missing_request"].reusable is False
    assert "request_missing" in rows["br1_missing_request"].blocker_codes
    assert rows["br1_malformed"].reusable is False
    assert any(
        code in rows["br1_malformed"].blocker_codes
        for code in ("request_invalid", "request_missing")
    )


def test_request_id_mismatch_blocked(repo: Path) -> None:
    request = _native_request(repo, seed=11)
    jobs_root = repo / "artifacts" / "ui_jobs"
    _write_job(
        jobs_root,
        job_id="job_wrong_ref_id",
        request_id="br1_wrong_id",
        request_path=request.relative_path,
        stdout=_search_stdout(repo, request),
    )
    rows = discover_saved_blend_searches(repository_root=repo, jobs_root=jobs_root)
    # Projection rewrites to the file request_id but keeps the mismatch blocker.
    saved = next(item for item in rows if item.request_id == request.request_id)
    assert saved.reusable is False
    assert "request_id_mismatch" in saved.blocker_codes


def test_stdout_missing_malformed_and_blend_mismatch(repo: Path) -> None:
    request = _native_request(repo, seed=12)
    jobs_root = repo / "artifacts" / "ui_jobs"
    _write_job(
        jobs_root,
        job_id="job_no_stdout",
        request_id=request.request_id,
        request_path=request.relative_path,
        stdout=None,
    )
    rows = discover_saved_blend_searches(repository_root=repo, jobs_root=jobs_root)
    assert rows[0].reusable is False
    assert "stdout_missing" in rows[0].blocker_codes

    request2 = _native_request(repo, seed=13)
    _write_job(
        jobs_root,
        job_id="job_bad_stdout",
        request_id=request2.request_id,
        request_path=request2.relative_path,
        stdout="not json",
    )
    rows2 = {
        item.request_id: item
        for item in discover_saved_blend_searches(
            repository_root=repo, jobs_root=jobs_root
        )
    }
    assert rows2[request2.request_id].reusable is False
    assert "stdout_malformed" in rows2[request2.request_id].blocker_codes

    request3 = _native_request(repo, seed=14)
    bad = json.loads(_search_stdout(repo, request3))
    bad["blend_id"] = "pb1_wrong_blend"
    _write_job(
        jobs_root,
        job_id="job_wrong_blend",
        request_id=request3.request_id,
        request_path=request3.relative_path,
        stdout=json.dumps(bad),
    )
    rows3 = {
        item.request_id: item
        for item in discover_saved_blend_searches(
            repository_root=repo, jobs_root=jobs_root
        )
    }
    assert rows3[request3.request_id].reusable is False
    assert "blend_id_mismatch" in rows3[request3.request_id].blocker_codes


def test_failed_stopped_orphaned_non_reusable(repo: Path) -> None:
    request = _native_request(repo, seed=15)
    jobs_root = repo / "artifacts" / "ui_jobs"
    for index, state in enumerate(("failed", "stopped", "orphaned")):
        _write_job(
            jobs_root,
            job_id=f"job_{state}",
            request_id=request.request_id,
            request_path=request.relative_path,
            state=state,
            created_at_utc=f"2026-08-01T0{index}:00:00Z",
            stdout=None,
        )
    rows = discover_saved_blend_searches(repository_root=repo, jobs_root=jobs_root)
    assert len(rows) == 1
    assert rows[0].reusable is False
    assert any(code.startswith("job_") for code in rows[0].blocker_codes)


def test_missing_candidate_and_changed_manifest(repo: Path) -> None:
    request = _native_request(repo, seed=16)
    jobs_root = repo / "artifacts" / "ui_jobs"
    stdout = _search_stdout(repo, request)
    _write_job(
        jobs_root,
        job_id="job_before_delete",
        request_id=request.request_id,
        request_path=request.relative_path,
        stdout=stdout,
    )
    missing_id = request.candidate_ids[0]
    candidate_dir = repo / "artifacts" / "prediction_candidates" / missing_id
    for path in sorted(candidate_dir.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    if candidate_dir.exists():
        candidate_dir.rmdir()
    rows = discover_saved_blend_searches(repository_root=repo, jobs_root=jobs_root)
    saved = next(item for item in rows if item.request_id == request.request_id)
    assert saved.reusable is False
    assert "candidate_missing" in saved.blocker_codes

    request2 = _native_request(repo, seed=17)
    jobs_root2 = repo / "artifacts" / "ui_jobs"
    _write_job(
        jobs_root2,
        job_id="job_manifest_drift",
        request_id=request2.request_id,
        request_path=request2.relative_path,
        stdout=_search_stdout(repo, request2),
        created_at_utc="2026-08-01T15:00:00Z",
    )
    manifest_path = (
        repo
        / "artifacts"
        / "prediction_candidates"
        / request2.candidate_ids[0]
        / MANIFEST_FILENAME
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_metric_value"] = 0.123456
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rows2 = {
        item.request_id: item
        for item in discover_saved_blend_searches(
            repository_root=repo, jobs_root=jobs_root2
        )
    }
    drifted = rows2[request2.request_id]
    assert drifted.reusable is False
    assert (
        "blend_id_mismatch" in drifted.blocker_codes
        or "candidate_manifest_changed" in drifted.blocker_codes
    )


def test_dedupe_prefers_latest_valid_succeeded(repo: Path) -> None:
    a, b, c = _three_candidates(repo)
    request = materialize_blend_ui_request(
        repository_root=repo,
        candidate_ids=[a, b, c],
        method="optimized_optuna",
        folds=2,
        repeats=1,
        seed=3,
        max_active_models=3,
        optuna_trials=5,
        optuna_seed=3,
    )
    jobs_root = repo / "artifacts" / "ui_jobs"
    stdout = _search_stdout(repo, request)
    _write_job(
        jobs_root,
        job_id="job_old_ok",
        request_id=request.request_id,
        request_path=request.relative_path,
        stdout=stdout,
        created_at_utc="2026-08-01T09:00:00Z",
    )
    _write_job(
        jobs_root,
        job_id="job_mid_fail",
        request_id=request.request_id,
        request_path=request.relative_path,
        state="failed",
        created_at_utc="2026-08-01T10:00:00Z",
        stdout=None,
    )
    _write_job(
        jobs_root,
        job_id="job_new_ok",
        request_id=request.request_id,
        request_path=request.relative_path,
        stdout=stdout,
        created_at_utc="2026-08-01T11:00:00Z",
    )
    other = materialize_blend_ui_request(
        repository_root=repo,
        candidate_ids=[c, b, a],
        method="optimized_optuna",
        folds=2,
        repeats=1,
        seed=3,
        max_active_models=3,
        optuna_trials=5,
        optuna_seed=3,
    )
    _write_job(
        jobs_root,
        job_id="job_other_request",
        request_id=other.request_id,
        request_path=other.relative_path,
        stdout=_search_stdout(repo, other),
        created_at_utc="2026-08-01T12:00:00Z",
    )
    rows = discover_saved_blend_searches(repository_root=repo, jobs_root=jobs_root)
    by_request = {item.request_id: item for item in rows}
    assert set(by_request) == {request.request_id, other.request_id}
    primary = by_request[request.request_id]
    assert primary.reusable is True
    assert primary.job_id == "job_new_ok"
    assert primary.attempt_count == 3
    assert "job_old_ok" in primary.attempt_job_ids
    assert "job_mid_fail" in primary.attempt_job_ids


def test_configuration_load_restores_max_active_and_identity(repo: Path) -> None:
    request, jobs_root = _prepare_native_search(repo)
    saved = discover_saved_blend_searches(
        repository_root=repo, jobs_root=jobs_root
    )[0]
    config = configuration_from_saved_search(saved)
    assert config["candidate_ids"] == list(request.candidate_ids)
    assert config["max_active_models"] == 3
    assert config["max_active_models"] != 0
    assert config["dirichlet_draws"] == 8
    assert config["pairwise_grid_step"] == 0.25
    session: dict[str, Any] = {}
    persist_loaded_blend_configuration(session, config, pending_apply=True)
    loaded = load_persisted_blend_configuration(session)
    assert loaded is not None
    assert loaded["max_active_models"] == 3
    assert consume_loaded_blend_configuration_pending(session) is True
    assert consume_loaded_blend_configuration_pending(session) is False
    # Simulate Build blend widget seeding from the durable loaded config.
    assert int(loaded["max_active_models"]) == 3
    assert int(loaded["folds"]) == 2
    assert int(loaded["repeats"]) == 1
    assert int(loaded["dirichlet_draws"]) == 8
    assert float(loaded["pairwise_grid_step"]) == 0.25

    optuna = materialize_blend_ui_request(
        repository_root=repo,
        candidate_ids=list(request.candidate_ids),
        method="optimized_optuna",
        folds=2,
        repeats=1,
        seed=21,
        max_active_models=3,
        optuna_trials=50,
        optuna_seed=21,
    )
    _write_job(
        jobs_root,
        job_id="job_optuna_cfg",
        request_id=optuna.request_id,
        request_path=optuna.relative_path,
        stdout=_search_stdout(repo, optuna),
    )
    optuna_saved = next(
        item
        for item in discover_saved_blend_searches(
            repository_root=repo, jobs_root=jobs_root
        )
        if item.request_id == optuna.request_id
    )
    optuna_config = configuration_from_saved_search(optuna_saved)
    assert optuna_config["optuna_trials"] == 50
    assert optuna_config["optuna_seed"] == 21

    edited = materialize_blend_ui_request(
        repository_root=repo,
        candidate_ids=list(config["candidate_ids"]),
        method=str(config["method"]),
        folds=int(config["folds"]),
        repeats=int(config["repeats"]),
        seed=int(config["seed"]) + 1,
        max_active_models=config["max_active_models"],
        dirichlet_draws=int(config["dirichlet_draws"]),
        pairwise_grid_step=float(config["pairwise_grid_step"]),
    )
    assert edited.request_id != request.request_id
    original = (repo / request.relative_path).read_text(encoding="utf-8")
    assert request.request_id in original


def test_historical_materialize_plan_and_authorize(repo: Path) -> None:
    request, jobs_root = _prepare_native_search(repo)
    saved = discover_saved_blend_searches(
        repository_root=repo, jobs_root=jobs_root
    )[0]
    plan = historical_materialize_launch_plan(saved)
    assert plan["request_path"] == request.relative_path
    assert plan["requires_confirmation"] is True
    assert plan["references"]["source_search_job_id"] == saved.job_id
    assert plan["references"]["request_id"] == request.request_id

    registry = load_registry(PROJECT_ROOT)
    values = authorized_blend_argv_values(str(plan["request_path"]))
    built = build_command(
        registry.commands,
        str(plan["command_id"]),
        str(plan["action_id"]),
        values,
        repository_root=repo,
    )
    assert built.argv[-2:] == ("--request", request.relative_path)
    consumed: set[str] = set()
    rendered = rendered_launch(built, None, consumed_nonces=consumed)
    with pytest.raises(LaunchAuthorizationError):
        authorize_launch(
            registry.commands,
            command_id=str(plan["command_id"]),
            action_id=str(plan["action_id"]),
            values=values,
            repository_root=repo,
            confirmed=False,
            high_risk_acknowledged=True,
            rendered=rendered,
            consumed_nonces=consumed,
        )
    authorized = authorize_launch(
        registry.commands,
        command_id=str(plan["command_id"]),
        action_id=str(plan["action_id"]),
        values=values,
        repository_root=repo,
        confirmed=True,
        high_risk_acknowledged=True,
        rendered=rendered,
        consumed_nonces=consumed,
    )
    assert request.relative_path in authorized.argv


def test_materialize_filter_and_mismatch_blocks_handoff(repo: Path) -> None:
    request, jobs_root = _prepare_native_search(repo)
    saved = discover_saved_blend_searches(
        repository_root=repo, jobs_root=jobs_root
    )[0]
    related = [
        {
            "job_id": "mat_ok",
            "action_id": "materialize",
            "references": {
                "source_search_job_id": saved.job_id,
                "request_id": request.request_id,
                "expected_blend_id": saved.expected_blend_id,
            },
        },
        {
            "job_id": "mat_unrelated",
            "action_id": "materialize",
            "references": {
                "source_search_job_id": "job_other",
                "request_id": "br1_other_request",
                "expected_blend_id": "pb1_other",
            },
        },
        {
            "job_id": "mat_wrong_source",
            "action_id": "materialize",
            "references": {
                "source_search_job_id": "job_different_search",
                "request_id": request.request_id,
                "expected_blend_id": "pb1_different",
            },
        },
    ]
    filtered = filter_materialize_jobs_for_saved_search(related, saved)
    assert [item["job_id"] for item in filtered] == ["mat_ok"]

    mismatches = compare_materialized_to_saved(
        saved=saved,
        materialized_payload={
            "request_id": request.request_id,
            "blend_id": "pb1_different",
            "candidate_ids": list(saved.candidate_ids),
            "optimizer_backend": saved.optimizer_backend,
            "honest_meta_cv_metrics": {"mean_repeat_balanced_accuracy": 0.1},
            "final_deployment_threshold": 0.99,
            "final_deployment_weights": {key: 0.0 for key in saved.final_weights},
        },
        loaded_artifact={
            "blend_id": "pb1_different",
            "candidate_ids": list(saved.candidate_ids),
            "optimizer_backend": saved.optimizer_backend,
            "honest_meta_cv_metrics": {"mean_repeat_balanced_accuracy": 0.1},
            "final_deployment_threshold": 0.99,
            "final_deployment_weights": {key: 0.0 for key in saved.final_weights},
        },
    )
    assert "blend_id" in mismatches
    assert "honest_mean_balanced_accuracy" in mismatches
    del jobs_root


def test_stale_job_and_request_recovery(repo: Path) -> None:
    request, jobs_root = _prepare_native_search(repo)
    other = materialize_blend_ui_request(
        repository_root=repo,
        candidate_ids=list(request.candidate_ids),
        method="optimized_native",
        folds=2,
        repeats=1,
        seed=8,
        max_active_models=None,
        dirichlet_draws=8,
        pairwise_grid_step=0.25,
    )
    assert other.request_id != request.request_id
    assert (
        resolve_displayed_search_job_id(
            session_job_id="job_native_ok",
            session_request_id=request.request_id,
            current_request_id=other.request_id,
            request_search_job_ids=[],
        )
        is None
    )
    assert (
        resolve_displayed_search_job_id(
            session_job_id="job_native_ok",
            session_request_id=request.request_id,
            current_request_id=request.request_id,
            request_search_job_ids=["job_native_ok"],
        )
        == "job_native_ok"
    )
    recovered = recover_search_for_request(
        jobs_root=jobs_root,
        request=request,
        expected_blend_id_value=expected_blend_id(request, repo),
    )
    assert recovered is not None
    assert recovered["recovery_state"] == "succeeded"
    assert recovered["payload"] is not None
    assert (
        recover_search_for_request(
            jobs_root=jobs_root,
            request=other,
            expected_blend_id_value=expected_blend_id(other, repo),
        )
        is None
    )


def test_page_contracts_include_search_history_tab() -> None:
    blend_page = (
        PROJECT_ROOT / "src/churn_ml/control_panel/blend_page.py"
    ).read_text(encoding="utf-8")
    history_page = (
        PROJECT_ROOT / "src/churn_ml/control_panel/search_history_page.py"
    ).read_text(encoding="utf-8")
    assert "Prepare candidates" in blend_page
    assert "Build blend" in blend_page
    assert "Search history" in blend_page
    assert "Materialized blends" in blend_page
    assert "Open Search history" in blend_page
    assert "Materialize this saved search" in history_page
    assert "Load configuration in Build blend" in history_page
    assert "deterministically re-evaluates" in history_page


def test_real_artifact_proof_optional() -> None:
    """Read-only discovery against the real workspace when artifacts exist."""
    jobs_root = PROJECT_ROOT / "artifacts" / "ui_jobs"
    if not jobs_root.is_dir():
        pytest.skip("real ui_jobs absent")
    rows = discover_saved_blend_searches(
        repository_root=PROJECT_ROOT, jobs_root=jobs_root
    )
    by_request = {item.request_id: item for item in rows}
    native = by_request.get("br1_452a4681d17e939f")
    optuna = by_request.get("br1_e3fccec058b97919")
    if native is None or optuna is None:
        pytest.skip("expected proof searches not present")
    assert native.reusable is True
    assert optuna.reusable is True
    assert native.optimizer_backend == "native"
    assert optuna.optimizer_backend == "optuna"
    assert native.max_active_models == 3
    assert optuna.max_active_models == 3
    assert native.expected_blend_id == "pb1_d047ebe82811ef1a"
    assert optuna.expected_blend_id == "pb1_8a09dbec68cc01ea"
    assert abs(float(native.honest_mean_balanced_accuracy) - 0.8982157139) < 1e-9
    assert abs(float(optuna.honest_mean_balanced_accuracy) - 0.8961171369) < 1e-9
    assert float(native.honest_mean_balanced_accuracy) > float(
        optuna.honest_mean_balanced_accuracy
    )
    assert native.final_threshold == 0.17
    draft = by_request.get("br1_ea0706c93e6dee86")
    if draft is not None:
        assert draft.reusable is False
    assert all(not label.startswith("pc1_") for label in native.candidate_labels)
