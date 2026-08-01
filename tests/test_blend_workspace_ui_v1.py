"""Focused Blend Workspace UI, request-contract, and submission-handoff tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.churn_ml.blending.artifact_v1 import materialize_blend, search_blend
from src.churn_ml.blending.cli import build_parser, main as blend_cli_main
from src.churn_ml.blending.compatibility_v1 import load_compatible_candidates
from src.churn_ml.blending.request_v1 import (
    BlendUIRequestError,
    build_request_payload,
    load_blend_ui_request,
    materialize_blend_ui_request,
    method_to_strategy_optimizer,
    settings_from_request_payload,
    strategy_optimizer_to_method,
    validate_candidate_ids,
)
from src.churn_ml.control_panel.blend_workspace import (
    CANONICAL_SUBMISSION_HANDOFF_KEY,
    BlendWorkspaceError,
    apply_canonical_submission_handoff,
    authorized_blend_argv_values,
    authorized_submission_argv_values,
    candidate_display_label,
    candidate_inventory_fingerprint,
    discover_candidate_rows,
    discover_materialized_blends,
    filter_candidate_rows,
    list_jobs_for_request,
    load_validated_submission_csv_bytes,
    optimizer_label,
    parse_job_stdout_json,
    readiness_label,
    resolve_canonical_submission_handoff,
    run_compatibility,
    run_diversity,
    selection_fingerprint,
    source_kind_label,
    suggest_submission_id,
    validate_selection_count,
    verify_search_result,
)
from src.churn_ml.control_panel.command_builder import build_command
from src.churn_ml.control_panel.launch import authorize_launch, rendered_launch
from src.churn_ml.control_panel.registry import load_registry
from src.churn_ml.control_panel.workflow_navigation import (
    ADVANCED_ONLY_COMMAND_IDS,
    is_legacy_command,
    visible_command_ids,
    workflow_command_ids,
    workflow_label,
)
from src.churn_ml.prediction_candidates.contract_v1 import (
    CandidateConflictError,
    MANIFEST_FILENAME,
    SCHEMA_VERSION,
    create_candidate_package,
    file_sha256,
)
from src.churn_ml.prediction_candidates.submission_v1 import (
    evaluate_submission_readiness,
    generate_candidate_submission,
)
from src.churn_ml.competition_assets_v1 import (
    DEFAULT_ID_COLUMN,
    DEFAULT_TARGET_COLUMN,
    SAMPLE_SUBMISSION_PATH,
    ensure_competition_row_identity,
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
    return root


def _seed_competition(repo: Path, *, n_test: int = 3) -> None:
    sample = pd.DataFrame(
        {
            DEFAULT_ID_COLUMN: list(range(n_test)),
            DEFAULT_TARGET_COLUMN: [0] * n_test,
        }
    )
    sample_path = repo / Path(*Path(SAMPLE_SUBMISSION_PATH).parts)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(sample_path, index=False, lineterminator="\n")
    test = pd.DataFrame(
        {
            "feature_one": [float(i) for i in range(n_test)],
            "feature_two": [float(i * 2) for i in range(n_test)],
        }
    )
    test_path = repo / "data" / "raw" / "final_proj_test.csv"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test.to_csv(test_path, index=False, lineterminator="\n")
    resolution = ensure_competition_row_identity(
        repo, dataset_version="v0_raw_minimal"
    )
    assert resolution.ready is True


def _make_candidate(
    repo: Path,
    *,
    name: str,
    oof_probs: np.ndarray,
    test_probs: np.ndarray,
    exploratory: bool = False,
    final_threshold: float | None = None,
    source_kind: str = "autogluon_standalone_v1",
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
        "source_kind": source_kind,
        "dataset_id": dataset_id,
        "source_run_path": f"artifacts/fake/{name}",
        "source_config_sha256": (name.encode().hex() + "0" * 64)[:64],
        "source_model_name": name,
        "oof_protocol": "unit_test_protocol",
        "positive_class_label": 1,
        "probability_semantics": "P(y=1)",
    }
    source_metadata: dict[str, Any] = {"schema_version": 1, "name": name}
    if final_threshold is not None:
        source_metadata["final_deployment_threshold"] = float(final_threshold)
    package = create_candidate_package(
        repository_root=repo,
        identity=identity,
        oof=oof,
        test=test,
        manifest_fields={
            "dataset_id": dataset_id,
            "exploratory": exploratory,
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
        source_metadata=source_metadata,
        candidates_root_relative="artifacts/prediction_candidates",
    )
    return package.candidate_id


def _three_candidates(repo: Path) -> tuple[str, str, str]:
    y = pd.read_parquet(
        repo / "data" / "processed" / "v0_raw_minimal" / "y_train.parquet"
    ).iloc[:, 0]
    n = len(y)
    n_test = 3
    rng = np.random.default_rng(0)
    base = np.clip(0.2 + 0.6 * y.to_numpy() + rng.normal(0, 0.05, size=n), 0.05, 0.95)
    alt = np.clip(0.8 - 0.5 * y.to_numpy() + rng.normal(0, 0.05, size=n), 0.05, 0.95)
    third = np.clip(0.5 + 0.2 * y.to_numpy() + rng.normal(0, 0.05, size=n), 0.05, 0.95)
    a = _make_candidate(
        repo,
        name="WeightedEnsemble_L2",
        oof_probs=base,
        test_probs=np.linspace(0.2, 0.8, n_test),
        final_threshold=0.45,
    )
    b = _make_candidate(
        repo,
        name="LightGBM_BAG_L1",
        oof_probs=alt,
        test_probs=np.linspace(0.8, 0.2, n_test),
        final_threshold=0.5,
        exploratory=True,
    )
    c = _make_candidate(
        repo,
        name="XGBoost_BAG_L1",
        oof_probs=third,
        test_probs=np.linspace(0.3, 0.7, n_test),
        final_threshold=0.55,
        source_kind="unit_test_source_v1",
    )
    return a, b, c


def test_navigation_and_workflow_contracts() -> None:
    loaded = load_registry(PROJECT_ROOT)
    assert "Blend" in {
        "Dashboard",
        "Run",
        "Blend",
        "Jobs",
        "Results",
        "Configuration",
    }
    assert workflow_command_ids() == (
        "experiment_core_v2",
        "paired_comparison",
        "optuna_search_v1",
        "final_deployment_v1",
    )
    assert "blend_evaluation_v1" not in workflow_command_ids()
    assert is_legacy_command("blend_evaluation_v1")
    standard = visible_command_ids(list(loaded.commands))
    assert "blend_evaluation_v1" not in standard
    assert "prediction_blend_v1" not in standard
    assert "candidate_submission_v1" not in standard
    assert "prediction_blend_v1" in ADVANCED_ONLY_COMMAND_IDS
    assert "candidate_submission_v1" in ADVANCED_ONLY_COMMAND_IDS
    assert workflow_label("final_deployment_v1", fallback="x") == (
        "📤 Generate submission"
    )
    assert "prediction_blend_v1" in loaded.commands
    assert "candidate_submission_v1" in loaded.commands
    assert "prediction_candidate_v1" in loaded.readers
    assert "prediction_blend_v1" in loaded.readers
    assert "candidate_submission_v1" in loaded.readers


def test_request_identity_order_reuse_and_conflict(repo: Path) -> None:
    a, b, c = _three_candidates(repo)
    first = materialize_blend_ui_request(
        repository_root=repo,
        candidate_ids=[a, b],
        method="optimized_native",
        folds=3,
        repeats=1,
        seed=7,
    )
    reused = materialize_blend_ui_request(
        repository_root=repo,
        candidate_ids=[a, b],
        method="optimized_native",
        folds=3,
        repeats=1,
        seed=7,
    )
    assert reused.request_id == first.request_id
    assert reused.absolute_path == first.absolute_path

    reordered = build_request_payload(
        candidate_ids=[b, a],
        method="optimized_native",
        folds=3,
        repeats=1,
        seed=7,
    )
    assert reordered["request_id"] != first.request_id

    changed = build_request_payload(
        candidate_ids=[a, b],
        method="optimized_optuna",
        folds=3,
        repeats=1,
        seed=7,
        optuna_trials=25,
    )
    assert changed["request_id"] != first.request_id

    conflict_path = first.absolute_path
    payload = json.loads(conflict_path.read_text(encoding="utf-8"))
    payload["seed"] = 999
    conflict_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(CandidateConflictError):
        materialize_blend_ui_request(
            repository_root=repo,
            candidate_ids=[a, b],
            method="optimized_native",
            folds=3,
            repeats=1,
            seed=7,
        )

    with pytest.raises(BlendUIRequestError, match="traversal|Invalid"):
        validate_candidate_ids(["../escape", a])
    with pytest.raises(BlendUIRequestError):
        validate_candidate_ids([a])
    with pytest.raises(BlendUIRequestError):
        validate_candidate_ids([a, b, c] + [a] * 8)

    del c


def test_request_cli_equivalence_and_conflict(repo: Path) -> None:
    a, b, _ = _three_candidates(repo)
    request = materialize_blend_ui_request(
        repository_root=repo,
        candidate_ids=[a, b],
        method="equal",
        folds=3,
        repeats=1,
        seed=11,
    )
    loaded = load_blend_ui_request(request.relative_path, repository_root=repo)
    assert loaded.candidate_ids == [a, b]
    assert loaded.blend_settings().strategy == "equal"

    parser = build_parser()
    args_request = parser.parse_args(
        ["search", "--request", request.relative_path]
    )
    args_direct = parser.parse_args(
        [
            "search",
            "--candidate",
            a,
            "--candidate",
            b,
            "--strategy",
            "equal",
            "--folds",
            "3",
            "--repeats",
            "1",
            "--seed",
            "11",
        ]
    )
    from src.churn_ml.blending import cli as blend_cli

    cand_r, settings_r, *_rest_r = blend_cli._resolve_search_inputs(args_request, repo)
    cand_d, settings_d, *_rest_d = blend_cli._resolve_search_inputs(args_direct, repo)
    assert cand_r == cand_d == [a, b]
    assert settings_r.normalized() == settings_d.normalized()

    with pytest.raises(BlendUIRequestError, match="mixed"):
        blend_cli._resolve_search_inputs(
            parser.parse_args(
                [
                    "search",
                    "--request",
                    request.relative_path,
                    "--candidate",
                    a,
                ]
            ),
            repo,
        )

    code = blend_cli_main(
        ["search", "--request", request.relative_path],
        repository_root=repo,
    )
    assert code == 0


def test_method_mapping_and_manual_weight_validation() -> None:
    assert method_to_strategy_optimizer("equal") == ("equal", "native")
    assert method_to_strategy_optimizer("manual") == ("manual", "native")
    assert method_to_strategy_optimizer("optimized_native") == ("optimized", "native")
    assert method_to_strategy_optimizer("optimized_optuna") == ("optimized", "optuna")
    assert strategy_optimizer_to_method("optimized", "optuna") == "optimized_optuna"
    with pytest.raises(BlendUIRequestError):
        method_to_strategy_optimizer("invalid")
    with pytest.raises(BlendUIRequestError):
        strategy_optimizer_to_method("optimized", "bogus")


def test_candidate_inventory_labels_filters_and_bounds(repo: Path) -> None:
    a, b, c = _three_candidates(repo)
    rows = discover_candidate_rows(repo, include_readiness=True)
    assert len(rows) == 3
    labels = {row["candidate_id"]: row["label"] for row in rows}
    assert "WeightedEnsemble_L2" in labels[a]
    assert "AutoGluon" in labels[a]
    assert source_kind_label("autogluon_standalone_v1") == "AutoGluon"
    assert source_kind_label("canonical_probability_blend_v1") == "Canonical blend"
    assert optimizer_label("optimized", "optuna") == "Optimized — Optuna"
    exploratory = [row for row in rows if row["candidate_id"] == b][0]
    assert exploratory["exploratory"] is True
    filtered = filter_candidate_rows(rows, exploratory=True)
    assert [row["candidate_id"] for row in filtered] == [b]
    validate_selection_count([a, b])
    with pytest.raises(BlendUIRequestError):
        validate_selection_count([a])
    with pytest.raises(BlendUIRequestError):
        validate_selection_count([a, b, c] * 4)
    empty_repo = repo / "empty"
    empty_repo.mkdir()
    assert discover_candidate_rows(empty_repo) == []


def test_readiness_cache_fingerprint_and_invalid_candidate(repo: Path) -> None:
    a, b, _ = _three_candidates(repo)
    fp1 = candidate_inventory_fingerprint(repo)
    rows = discover_candidate_rows(repo, include_readiness=True)
    blocked = next(row for row in rows if row["candidate_id"] == a)
    assert blocked["readiness_state"] in {"ready", "blocked"}
    # Tamper the authenticated manifest so inventory fingerprint changes and load fails.
    manifest_path = repo / "artifacts" / "prediction_candidates" / b / MANIFEST_FILENAME
    manifest_path.write_text("{not-valid-json", encoding="utf-8")
    fp2 = candidate_inventory_fingerprint(repo)
    assert fp1 != fp2
    rows2 = discover_candidate_rows(repo, include_readiness=True)
    statuses = {row["candidate_id"]: row["package_status"] for row in rows2}
    assert statuses[b] == "invalid"


def test_compatibility_diversity_and_state_invalidation(repo: Path) -> None:
    a, b, c = _three_candidates(repo)
    ok = run_compatibility([a, b], repo)
    assert ok["ok"] is True
    assert ok["summary"]["status"] == "compatible"
    diversity = run_diversity([a, b], repo)
    assert diversity["ok"] is True
    assert diversity["candidate_metrics"]
    assert diversity["pairwise_analysis"]
    fp_ab = selection_fingerprint([a, b])
    fp_ac = selection_fingerprint([a, c])
    assert fp_ab != fp_ac


def test_search_authorize_recover_and_materialize_argv(repo: Path) -> None:
    a, b, _ = _three_candidates(repo)
    request = materialize_blend_ui_request(
        repository_root=repo,
        candidate_ids=[a, b],
        method="optimized_native",
        folds=2,
        repeats=1,
        seed=3,
        dirichlet_draws=8,
        pairwise_grid_step=0.25,
    )
    optuna_request = materialize_blend_ui_request(
        repository_root=repo,
        candidate_ids=[a, b],
        method="optimized_optuna",
        folds=2,
        repeats=1,
        seed=3,
        optuna_trials=5,
        optuna_seed=3,
    )
    assert request.request_id != optuna_request.request_id
    assert request.payload["optimizer_backend"] == "native"
    assert optuna_request.payload["optimizer_backend"] == "optuna"

    loaded = load_registry(PROJECT_ROOT)
    values = authorized_blend_argv_values(request.relative_path)
    # Path resolution needs the request under repository_root; scripts path is argv only.
    (repo / "scripts").mkdir(exist_ok=True)
    (repo / "scripts" / "run_prediction_blend.py").write_text("# stub\n", encoding="utf-8")
    built = build_command(
        loaded.commands,
        "prediction_blend_v1",
        "search",
        values,
        repository_root=repo,
    )
    assert built.argv[-2:] == ("--request", request.relative_path)
    assert all(isinstance(token, str) for token in built.argv)
    consumed: set[str] = set()
    rendered = rendered_launch(built, None, consumed_nonces=consumed)
    authorized = authorize_launch(
        loaded.commands,
        command_id="prediction_blend_v1",
        action_id="search",
        values=values,
        repository_root=repo,
        confirmed=True,
        high_risk_acknowledged=True,
        rendered=rendered,
        consumed_nonces=consumed,
    )
    assert authorized.argv == built.argv

    materialize_built = build_command(
        loaded.commands,
        "prediction_blend_v1",
        "materialize",
        values,
        repository_root=repo,
    )
    assert materialize_built.argv[-2:] == ("--request", request.relative_path)

    pool = load_compatible_candidates([a, b], repository_root=repo)
    settings = request.blend_settings()
    search_payload = search_blend(pool, settings)
    search_payload["command"] = "search"
    search_payload["request_id"] = request.request_id
    verified = verify_search_result(
        search_payload,
        request_id=request.request_id,
        expected_blend_id=search_payload["blend_id"],
    )
    assert "honest_meta_cv_metrics" in verified
    assert verified["request_id"] == request.request_id

    with pytest.raises(BlendWorkspaceError):
        verify_search_result(
            {"ok": True, "blend_id": "x", "request_id": "br1_other"},
            request_id=request.request_id,
        )
    with pytest.raises(BlendWorkspaceError):
        parse_job_stdout_json("not json at all")
    parsed = parse_job_stdout_json(
        'noise\n{"ok": true, "command": "search", "blend_id": "pb1_x", '
        f'"request_id": "{request.request_id}"}}\n'
    )
    assert parsed["blend_id"] == "pb1_x"

    jobs_root = repo / "artifacts" / "ui_jobs"
    job_dir = jobs_root / "job_search_1"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "job_id": "job_search_1",
                "command_id": "prediction_blend_v1",
                "action_id": "search",
                "created_at_utc": "2026-08-01T00:00:00Z",
                "references": {
                    "request_id": request.request_id,
                    "candidate_ids": f"{a},{b}",
                },
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "status.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": "succeeded",
                "exit_code": 0,
                "started_at_utc": "2026-08-01T00:00:01Z",
                "finished_at_utc": "2026-08-01T00:00:02Z",
            }
        ),
        encoding="utf-8",
    )
    matches = list_jobs_for_request(jobs_root, request.request_id)
    assert matches[0]["job_id"] == "job_search_1"
    assert matches[0]["state"] == "succeeded"


def test_materialized_blend_handoff_and_submission(repo: Path) -> None:
    _seed_competition(repo)
    a, b, _ = _three_candidates(repo)
    request = materialize_blend_ui_request(
        repository_root=repo,
        candidate_ids=[a, b],
        method="equal",
        folds=2,
        repeats=1,
        seed=5,
    )
    pool = load_compatible_candidates([a, b], repository_root=repo)
    result = materialize_blend(
        pool, request.blend_settings(), blend_root_relative="artifacts/prediction_blends"
    )
    blends = discover_materialized_blends(repo)
    assert any(item["blend_id"] == result.blend_id and item["status"] == "completed" for item in blends)
    # Tampered blend appears blocked.
    bad_dir = repo / "artifacts" / "prediction_blends" / "pb1_tampered"
    bad_dir.mkdir(parents=True)
    (bad_dir / "_SUCCESS").write_text("{}", encoding="utf-8")
    blends2 = discover_materialized_blends(repo)
    assert any(item["blend_id"] == "pb1_tampered" and item["status"] == "blocked" for item in blends2)

    candidate_id = result.candidate_package.candidate_id
    manifest_path = (
        repo / "artifacts" / "prediction_candidates" / candidate_id / MANIFEST_FILENAME
    )
    session: dict[str, Any] = {}
    apply_canonical_submission_handoff(
        session,
        candidate_id=candidate_id,
        manifest_sha256=file_sha256(manifest_path),
        blend_id=result.blend_id,
    )
    handoff = resolve_canonical_submission_handoff(session, repo)
    assert handoff is not None
    assert handoff["candidate_id"] == candidate_id

    # Stale manifest hash invalidates handoff without mutating the package.
    session[CANONICAL_SUBMISSION_HANDOFF_KEY]["manifest_sha256"] = "0" * 64
    assert resolve_canonical_submission_handoff(session, repo) is None

    apply_canonical_submission_handoff(
        session,
        candidate_id=candidate_id,
        manifest_sha256=file_sha256(manifest_path),
        blend_id=result.blend_id,
    )
    readiness = evaluate_submission_readiness(candidate_id, repository_root=repo)
    assert readiness["ready"] is True
    assert readiness_label("ready") == "Ready"

    submission_id = suggest_submission_id(
        candidate_display_label(
            {
                "source_model_name": "Optimized Equal blend",
                "source_kind": "canonical_probability_blend_v1",
                "dataset_id": "v0_raw_minimal",
            }
        ),
        candidate_id=candidate_id,
    )
    generate_candidate_submission(
        candidate_id,
        repository_root=repo,
        submission_id=submission_id,
    )
    csv_bytes = load_validated_submission_csv_bytes(repo, submission_id)
    assert DEFAULT_ID_COLUMN.encode() in csv_bytes

    loaded = load_registry(PROJECT_ROOT)
    (repo / "scripts").mkdir(exist_ok=True)
    (repo / "scripts" / "generate_candidate_submission.py").write_text(
        "# stub\n", encoding="utf-8"
    )
    validate_values = authorized_submission_argv_values(candidate_id)
    generate_values = authorized_submission_argv_values(
        candidate_id, submission_id=submission_id
    )
    built_validate = build_command(
        loaded.commands,
        "candidate_submission_v1",
        "validate",
        validate_values,
        repository_root=repo,
    )
    assert "--candidate-id" in built_validate.argv
    assert "generate" not in built_validate.argv
    built_generate = build_command(
        loaded.commands,
        "candidate_submission_v1",
        "generate",
        generate_values,
        repository_root=repo,
    )
    assert built_generate.argv.count("--submission-id") == 1
    generate_action = loaded.commands["candidate_submission_v1"].actions["generate"]
    assert generate_action.competition_test is True
    assert generate_action.confirmation == "acknowledge"

    y_len = len(
        pd.read_parquet(repo / "data" / "processed" / "v0_raw_minimal" / "y_train.parquet")
    )
    no_thresh = _make_candidate(
        repo,
        name="no_thresh",
        oof_probs=np.linspace(0.1, 0.9, y_len),
        test_probs=np.linspace(0.2, 0.8, 3),
        final_threshold=None,
    )
    blocked = evaluate_submission_readiness(no_thresh, repository_root=repo)
    assert blocked["ready"] is False
    assert any(
        item["code"] == "final_threshold_missing" for item in blocked["blockers"]
    )


def test_app_navigation_includes_blend_page() -> None:
    import apps.experiment_control_panel as panel

    source = Path(panel.__file__).read_text(encoding="utf-8")
    assert 'st.Page(blend_page, title="Blend"' in source
    assert "render_blend_workspace_page" in source
    assert "Submission source" in source
    assert "Canonical prediction candidate" in source
    assert "Completed Research v2 run / deployment package" in source


def test_settings_from_request_payload_roundtrip(repo: Path) -> None:
    a, b, _ = _three_candidates(repo)
    payload = build_request_payload(
        candidate_ids=[a, b],
        method="manual",
        folds=4,
        repeats=2,
        seed=9,
        manual_weights={a: 0.4, b: 0.6},
    )
    settings = settings_from_request_payload(payload)
    assert settings.strategy == "manual"
    assert abs(sum(float(item.split("=")[1]) for item in settings.manual_weights) - 1.0) < 1e-9


def test_real_root_one_candidate_blocker() -> None:
    rows = discover_candidate_rows(PROJECT_ROOT, include_readiness=True)
    if not rows:
        pytest.skip("No real candidates present")
    if len(rows) != 1:
        pytest.skip("Real root does not have exactly one candidate")
    row = rows[0]
    assert "AutoGluon" in str(row["label"]) or row["source_kind_label"]
    assert row["readiness_state"] == "blocked"
    assert any(
        blocker.get("code") == "final_threshold_missing"
        for blocker in row.get("readiness_blockers") or []
    )
    with pytest.raises(BlendUIRequestError):
        validate_selection_count([row["candidate_id"]])
