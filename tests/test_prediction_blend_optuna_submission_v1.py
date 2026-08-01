"""Focused tests for Optuna blending, honest meta-CV, and candidate submissions."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.churn_ml.blending.artifact_v1 import (
    build_blend_id,
    load_blend_artifact,
    materialize_blend,
)
from src.churn_ml.blending.cli import main as blend_main
from src.churn_ml.blending.compatibility_v1 import load_compatible_candidates
from src.churn_ml.blending.evaluation_v1 import BlendSettings, run_blend_evaluation
from src.churn_ml.blending.optimization_v1 import (
    BlendOptimizationError,
    enforce_max_active,
    softmax_from_latents,
    validate_max_active,
)
from src.churn_ml.blending.optuna_v1 import OptunaStudyRecord, derive_study_seed
from src.churn_ml.competition_assets_v1 import (
    DEFAULT_ID_COLUMN,
    DEFAULT_TARGET_COLUMN,
    SAMPLE_SUBMISSION_PATH,
    ensure_competition_row_identity,
)
from src.churn_ml.prediction_candidates.contract_v1 import (
    SCHEMA_VERSION,
    create_candidate_package,
    file_sha256,
    load_candidate_package,
)
from src.churn_ml.prediction_candidates.submission_cli import main as submission_main
from src.churn_ml.prediction_candidates.submission_v1 import (
    evaluate_submission_readiness,
    generate_candidate_submission,
)
from tests.dataset_registry_support import (
    create_synthetic_legacy_tree,
    register_package,
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
    register_package(
        processed,
        "v1_missingness_summary",
        parent_dataset_id="v0_raw_minimal",
        hypothesis="synthetic v1",
        summary_features=("missing_count_total", "missing_rate_total"),
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
    source_kind: str = "unit_test_source_v1",
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
    source_metadata: dict = {"schema_version": 1, "name": name}
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


def _two_candidates(
    repo: Path, *, exploratory: bool = False
) -> tuple[str, str]:
    y = pd.read_parquet(
        repo / "data" / "processed" / "v0_raw_minimal" / "y_train.parquet"
    ).iloc[:, 0]
    n = len(y)
    rng = np.random.default_rng(0)
    base = np.clip(0.2 + 0.6 * y.to_numpy() + rng.normal(0, 0.05, size=n), 0.05, 0.95)
    alt = np.clip(0.8 - 0.5 * y.to_numpy() + rng.normal(0, 0.05, size=n), 0.05, 0.95)
    a = _make_candidate(
        repo,
        name="model_a",
        oof_probs=base,
        test_probs=np.linspace(0.2, 0.8, 3),
        exploratory=exploratory,
    )
    b = _make_candidate(
        repo,
        name="model_b",
        oof_probs=alt,
        test_probs=np.linspace(0.8, 0.2, 3),
        source_kind="unit_test_source_v2",
        exploratory=exploratory,
    )
    return a, b


def _stub_study_runner_factory(captured: list):
    def runner(**kwargs):
        y_true = np.asarray(kwargs["y_true"])
        captured.append(
            {
                "y_true": y_true.copy(),
                "probability_matrix": np.asarray(kwargs["probability_matrix"]).copy(),
                "study_role": kwargs["study_role"],
                "repeat": kwargs["repeat"],
                "fold": kwargs["fold"],
                "resolved_seed": kwargs["resolved_seed"],
            }
        )
        n = int(kwargs["probability_matrix"].shape[1])
        weights = np.full(n, 1.0 / n, dtype=np.float64)
        return OptunaStudyRecord(
            study_role=str(kwargs["study_role"]),
            repeat=kwargs["repeat"],
            fold=kwargs["fold"],
            resolved_seed=int(kwargs["resolved_seed"]),
            optuna_version="stub",
            sampler_type="TPESampler",
            sampler_settings={"seed": int(kwargs["resolved_seed"])},
            requested_trials=int(kwargs["n_trials"]),
            completed_trials=int(kwargs["n_trials"]),
            timeout_seconds=kwargs["timeout_seconds"],
            best_trial_number=0,
            best_objective=0.75,
            best_latents=[0.0] * n,
            resolved_weights=weights.tolist(),
            resolved_threshold=0.5,
            trial_state_counts={"COMPLETE": int(kwargs["n_trials"])},
            trials=(
                {
                    "study_role": kwargs["study_role"],
                    "repeat": kwargs["repeat"],
                    "fold": kwargs["fold"],
                    "trial_number": 0,
                    "state": "COMPLETE",
                    "objective": 0.75,
                    "latents": [0.0] * n,
                    "weights": weights.tolist(),
                    "threshold": 0.5,
                    "duration_seconds": 0.0,
                },
            ),
            status="completed",
        )

    return runner


def test_lazy_optuna_import_for_non_optuna_paths(repo: Path) -> None:
    a, b = _two_candidates(repo)
    pool = load_compatible_candidates([a, b], repository_root=repo)
    before = "optuna" in sys.modules
    run_blend_evaluation(
        pool, BlendSettings(strategy="equal", folds=2, repeats=1, seed=1)
    )
    run_blend_evaluation(
        pool,
        BlendSettings(
            strategy="optimized",
            folds=2,
            repeats=1,
            seed=1,
            optimizer_backend="native",
            dirichlet_draws=4,
        ),
    )
    if not before:
        assert "optuna" not in sys.modules


def test_softmax_and_max_active_constraints() -> None:
    weights = softmax_from_latents(np.array([1.0, 2.0, 3.0]))
    assert (weights >= 0).all()
    assert weights.sum() == pytest.approx(1.0)
    limited = enforce_max_active(np.array([0.4, 0.4, 0.2]), 1)
    assert int(np.sum(limited > 1e-8)) == 1
    # Tie: prefer canonical lower index.
    tied = enforce_max_active(np.array([0.5, 0.5]), 1)
    assert tied.tolist() == pytest.approx([1.0, 0.0])
    with pytest.raises(BlendOptimizationError, match="exceeds"):
        validate_max_active(5, n_candidates=2)
    with pytest.raises(BlendOptimizationError, match=">= 1"):
        validate_max_active(0, n_candidates=2)


def test_optimizer_backend_and_budget_change_blend_id(repo: Path) -> None:
    a, b = _two_candidates(repo)
    pool = load_compatible_candidates([a, b], repository_root=repo)
    native = BlendSettings(
        strategy="optimized",
        folds=2,
        repeats=1,
        seed=7,
        optimizer_backend="native",
        dirichlet_draws=4,
    )
    optuna = BlendSettings(
        strategy="optimized",
        folds=2,
        repeats=1,
        seed=7,
        optimizer_backend="optuna",
        optuna_trials=5,
        optuna_seed=7,
    )
    optuna_more = BlendSettings(
        strategy="optimized",
        folds=2,
        repeats=1,
        seed=7,
        optimizer_backend="optuna",
        optuna_trials=6,
        optuna_seed=7,
    )
    optuna_seed = BlendSettings(
        strategy="optimized",
        folds=2,
        repeats=1,
        seed=7,
        optimizer_backend="optuna",
        optuna_trials=5,
        optuna_seed=8,
    )
    assert build_blend_id(pool, native) != build_blend_id(pool, optuna)
    assert build_blend_id(pool, optuna) != build_blend_id(pool, optuna_more)
    assert build_blend_id(pool, optuna) != build_blend_id(pool, optuna_seed)


def test_optuna_flags_rejected_for_misuse(repo: Path) -> None:
    with pytest.raises(BlendOptimizationError, match="Optuna-specific|optimizer"):
        BlendSettings(
            strategy="equal",
            optimizer_backend="optuna",
            optuna_trials=10,
        ).normalized()
    with pytest.raises(BlendOptimizationError, match="Optuna-specific"):
        BlendSettings(
            strategy="optimized",
            optimizer_backend="native",
            optuna_options_provided=True,
            optuna_trials=10,
        ).normalized()
    with pytest.raises(BlendOptimizationError, match="Native search-budget"):
        BlendSettings(
            strategy="optimized",
            optimizer_backend="optuna",
            optuna_trials=10,
            native_search_options_provided=True,
        ).normalized()
    with pytest.raises(BlendOptimizationError, match="positive"):
        BlendSettings(
            strategy="optimized",
            optimizer_backend="optuna",
            optuna_trials=0,
        ).normalized()
    with pytest.raises(BlendOptimizationError, match="timeout"):
        BlendSettings(
            strategy="optimized",
            optimizer_backend="optuna",
            optuna_trials=5,
            optuna_timeout_seconds=-1.0,
        ).normalized()


def test_optuna_meta_val_labels_never_reach_objective(repo: Path) -> None:
    a, b = _two_candidates(repo)
    pool = load_compatible_candidates([a, b], repository_root=repo)
    captured: list = []
    settings = BlendSettings(
        strategy="optimized",
        folds=2,
        repeats=1,
        seed=3,
        optimizer_backend="optuna",
        optuna_trials=3,
        optuna_seed=3,
    )
    result = run_blend_evaluation(
        pool,
        settings,
        optuna_study_runner=_stub_study_runner_factory(captured),
    )
    assert len(captured) == 3  # 2 folds + final
    roles = {item["study_role"] for item in captured}
    assert roles == {"meta_fold", "final_deployment"}
    fold_calls = [item for item in captured if item["study_role"] == "meta_fold"]
    for call in fold_calls:
        repeat = call["repeat"]
        fold = call["fold"]
        group = result.assignments[
            (result.assignments["repeat"] == repeat)
            & (result.assignments["fold"] == fold)
        ]
        train_rows = set(
            group.loc[group["role"] == "meta_train", "row_position"].tolist()
        )
        val_rows = set(
            group.loc[group["role"] == "meta_validation", "row_position"].tolist()
        )
        assert train_rows.isdisjoint(val_rows)
        assert set(np.arange(len(call["y_true"]))) == set(range(len(train_rows)))
        # The y_true passed is the train slice values, not full vector.
        assert len(call["y_true"]) == len(train_rows)
        assert len(call["y_true"]) != len(pool.target) or len(train_rows) == len(
            pool.target
        )
    seeds = {
        (item["study_role"], item["repeat"], item["fold"], item["resolved_seed"])
        for item in captured
    }
    assert len(seeds) == len(captured)


def test_optuna_deterministic_and_history(repo: Path) -> None:
    a, b = _two_candidates(repo)
    pool = load_compatible_candidates([a, b], repository_root=repo)
    settings = BlendSettings(
        strategy="optimized",
        folds=2,
        repeats=1,
        seed=11,
        optimizer_backend="optuna",
        optuna_trials=8,
        optuna_seed=11,
        max_active_models=2,
    )
    first = run_blend_evaluation(pool, settings)
    second = run_blend_evaluation(pool, settings)
    assert first.fixed_or_final_weights.tolist() == pytest.approx(
        second.fixed_or_final_weights.tolist()
    )
    assert first.final_threshold == pytest.approx(second.final_threshold)
    assert first.optuna_trial_history is not None
    assert len(first.optuna_trial_history) >= 8
    assert any(
        item["study_role"] == "final_deployment"
        for item in first.optuna_study_summaries
    )
    limited = BlendSettings(
        strategy="optimized",
        folds=2,
        repeats=1,
        seed=11,
        optimizer_backend="optuna",
        optuna_trials=6,
        optuna_seed=11,
        max_active_models=1,
    )
    limited_result = run_blend_evaluation(pool, limited)
    assert int(np.sum(limited_result.fixed_or_final_weights > 1e-8)) == 1


def test_incomplete_optuna_study_cannot_succeed(repo: Path) -> None:
    a, b = _two_candidates(repo)
    pool = load_compatible_candidates([a, b], repository_root=repo)

    def failed_runner(**kwargs):
        return OptunaStudyRecord(
            study_role=str(kwargs["study_role"]),
            repeat=kwargs["repeat"],
            fold=kwargs["fold"],
            resolved_seed=int(kwargs["resolved_seed"]),
            optuna_version="stub",
            sampler_type="TPESampler",
            sampler_settings={},
            requested_trials=int(kwargs["n_trials"]),
            completed_trials=0,
            timeout_seconds=None,
            best_trial_number=-1,
            best_objective=float("nan"),
            best_latents=[],
            resolved_weights=[],
            resolved_threshold=float("nan"),
            trial_state_counts={"FAIL": 1},
            trials=(),
            status="failed",
        )

    with pytest.raises(BlendOptimizationError, match="incomplete"):
        run_blend_evaluation(
            pool,
            BlendSettings(
                strategy="optimized",
                folds=2,
                repeats=1,
                seed=1,
                optimizer_backend="optuna",
                optuna_trials=3,
            ),
            optuna_study_runner=failed_runner,
        )


def test_honest_meta_cv_primary_and_descriptive_separation(repo: Path) -> None:
    a, b = _two_candidates(repo)
    pool = load_compatible_candidates([a, b], repository_root=repo)
    settings = BlendSettings(strategy="equal", folds=2, repeats=2, seed=5)
    result = run_blend_evaluation(pool, settings)
    assert len(result.cross_fitted_oof) == len(pool.target)
    assert not np.isnan(result.cross_fitted_oof).any()
    for repeat in (1, 2):
        subset = result.held_out_decisions[
            result.held_out_decisions["repeat"] == repeat
        ]
        assert len(subset) == len(pool.target)
        assert set(subset["row_position"].tolist()) == set(range(len(pool.target)))
        assert set(subset["prediction"].unique()).issubset({0, 1})
    mean_ba = float(result.repeat_metrics["balanced_accuracy"].mean())
    assert result.honest_meta_cv_metrics["mean_repeat_balanced_accuracy"] == pytest.approx(
        mean_ba
    )
    assert (
        result.honest_meta_cv_metrics["primary_score"]
        == "mean_repeat_balanced_accuracy"
    )
    descriptive = result.cross_fitted_probability_descriptive_metrics
    assert "descriptive" in descriptive["label"]
    assert "not the primary" in descriptive["note"].lower() or "Descriptive" in descriptive["note"]
    assert "descriptive" in result.full_oof_descriptive_metrics["label"].lower() or (
        "full_OOF_descriptive" in result.full_oof_descriptive_metrics["label"]
    )


def test_materialize_integrity_optuna_and_strict_load(repo: Path) -> None:
    _seed_competition(repo)
    a, b = _two_candidates(repo)
    pool = load_compatible_candidates([a, b], repository_root=repo)
    settings = BlendSettings(
        strategy="optimized",
        folds=2,
        repeats=1,
        seed=9,
        optimizer_backend="optuna",
        optuna_trials=5,
        optuna_seed=9,
    )
    first = materialize_blend(
        pool, settings, blend_root_relative="artifacts/prediction_blends"
    )
    loaded = load_blend_artifact(
        first.blend_id,
        repository_root=repo,
        blend_root_relative="artifacts/prediction_blends",
    )
    assert loaded["ok"] is True
    manifest = loaded["manifest"]
    assert "manifest_sha256" not in manifest
    success = json.loads((first.blend_dir / "_SUCCESS").read_text(encoding="utf-8"))
    assert success["manifest_sha256"] == file_sha256(
        first.blend_dir / "blend_manifest.json"
    )
    assert (first.blend_dir / "optuna_trials.parquet").is_file()
    assert (first.blend_dir / "held_out_decisions.parquet").is_file()
    package = load_candidate_package(
        first.candidate_package.package_dir, repository_root=repo
    )
    assert package.manifest["source_metric_value"] == pytest.approx(
        first.evaluation.honest_meta_cv_metrics["mean_repeat_balanced_accuracy"]
    )
    assert package.source_metadata["identity_reference_dataset_id"] == "v0_raw_minimal"
    assert package.source_metadata["final_deployment_threshold"] == pytest.approx(
        first.evaluation.final_threshold
    )

    second = materialize_blend(
        pool, settings, blend_root_relative="artifacts/prediction_blends"
    )
    assert second.blend_id == first.blend_id

    # Tamper evaluation -> strict load fails.
    evaluation_path = first.blend_dir / "evaluation.json"
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    payload["assignment_hash"] = "0" * 64
    evaluation_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(Exception, match="SHA-256|hash"):
        load_blend_artifact(
            first.blend_id,
            repository_root=repo,
            blend_root_relative="artifacts/prediction_blends",
        )


def test_tamper_manifest_weights_parent_candidate_rejected(repo: Path) -> None:
    _seed_competition(repo)
    a, b = _two_candidates(repo)
    pool = load_compatible_candidates([a, b], repository_root=repo)
    settings = BlendSettings(strategy="equal", folds=2, repeats=1, seed=2)
    result = materialize_blend(
        pool, settings, blend_root_relative="artifacts/prediction_blends"
    )
    # Tamper manifest without updating _SUCCESS.
    manifest_path = result.blend_dir / "blend_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["exploratory"] = not manifest["exploratory"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(Exception, match="SHA-256|hash"):
        load_blend_artifact(
            result.blend_id,
            repository_root=repo,
            blend_root_relative="artifacts/prediction_blends",
        )

    settings2 = BlendSettings(strategy="equal", folds=2, repeats=1, seed=4)
    pool2 = load_compatible_candidates([a, b], repository_root=repo)
    result2 = materialize_blend(
        pool2, settings2, blend_root_relative="artifacts/prediction_blends"
    )
    weights_path = result2.blend_dir / "final_weights.json"
    weights = json.loads(weights_path.read_text(encoding="utf-8"))
    weights["threshold"] = 0.123456
    weights_path.write_text(
        json.dumps(weights, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(Exception, match="SHA-256|hash"):
        load_blend_artifact(
            result2.blend_id,
            repository_root=repo,
            blend_root_relative="artifacts/prediction_blends",
        )


def test_submission_readiness_and_generation(repo: Path) -> None:
    _seed_competition(repo)
    a, b = _two_candidates(repo, exploratory=True)
    pool = load_compatible_candidates([a, b], repository_root=repo)
    settings = BlendSettings(
        strategy="optimized",
        folds=2,
        repeats=1,
        seed=12,
        optimizer_backend="native",
        dirichlet_draws=8,
    )
    blend = materialize_blend(
        pool, settings, blend_root_relative="artifacts/prediction_blends"
    )
    readiness = evaluate_submission_readiness(
        blend.candidate_package.candidate_id,
        repository_root=repo,
    )
    assert readiness["ready"] is True
    assert any(item["code"] == "exploratory_candidate" for item in readiness["warnings"])

    # Standalone without threshold blocked.
    alone = _make_candidate(
        repo,
        name="no_threshold",
        oof_probs=pool.oof_matrix[:, 0],
        test_probs=np.linspace(0.1, 0.9, 3),
        source_kind="unit_test_source_v3",
    )
    alone_ready = evaluate_submission_readiness(alone, repository_root=repo)
    assert alone_ready["ready"] is False
    assert any(
        item["code"] == "final_threshold_missing" for item in alone_ready["blockers"]
    )

    # Invalid threshold blocked.
    bad = _make_candidate(
        repo,
        name="bad_threshold",
        oof_probs=pool.oof_matrix[:, 0],
        test_probs=np.linspace(0.1, 0.9, 3),
        final_threshold=1.5,
        source_kind="unit_test_source_v4",
    )
    bad_ready = evaluate_submission_readiness(bad, repository_root=repo)
    assert bad_ready["ready"] is False
    assert any(item["code"] == "threshold_invalid" for item in bad_ready["blockers"])

    # Validate CLI is read-only.
    before = {
        path: path.read_bytes()
        for path in (repo / "artifacts" / "prediction_candidates").rglob("*")
        if path.is_file()
    }
    assert (
        submission_main(
            ["validate", "--candidate-id", blend.candidate_package.candidate_id],
            repository_root=repo,
        )
        == 0
    )
    after = {
        path: path.read_bytes()
        for path in (repo / "artifacts" / "prediction_candidates").rglob("*")
        if path.is_file()
    }
    assert before == after

    assert (
        submission_main(
            [
                "generate",
                "--candidate-id",
                blend.candidate_package.candidate_id,
                "--submission-id",
                "sub_proof_001",
            ],
            repository_root=repo,
        )
        == 0
    )
    submission_dir = repo / "artifacts" / "candidate_submissions" / "sub_proof_001"
    assert (submission_dir / "_SUCCESS").is_file()
    frame = pd.read_csv(submission_dir / "submission.csv")
    assert len(frame) == 3
    assert frame[DEFAULT_ID_COLUMN].tolist() == [0, 1, 2]
    threshold = float(blend.evaluation.final_threshold)
    expected = (
        blend.evaluation.test_probabilities >= threshold
    ).astype(np.int8)
    assert frame[DEFAULT_TARGET_COLUMN].tolist() == expected.tolist()
    source_meta = json.loads(
        (submission_dir / "source_metadata.json").read_text(encoding="utf-8")
    )
    assert source_meta["source_type"] == "canonical_prediction_candidate"
    assert source_meta["candidate_id"] == blend.candidate_package.candidate_id
    assert source_meta["exploratory"] is True
    assert source_meta["kaggle_upload"] is False
    assert source_meta["network_access"] is False

    # No-overwrite / idempotent same content.
    again = generate_candidate_submission(
        blend.candidate_package.candidate_id,
        submission_id="sub_proof_001",
        repository_root=repo,
    )
    assert again["idempotent"] is True

    # Conflicting overwrite blocked.
    other_settings = BlendSettings(
        strategy="equal", folds=2, repeats=1, seed=99
    )
    other = materialize_blend(
        pool, other_settings, blend_root_relative="artifacts/prediction_blends"
    )
    with pytest.raises(Exception):
        generate_candidate_submission(
            other.candidate_package.candidate_id,
            submission_id="sub_proof_001",
            repository_root=repo,
        )


def test_cli_optuna_and_native_search(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a, b = _two_candidates(repo)
    assert (
        blend_main(
            [
                "search",
                "--candidate",
                a,
                "--candidate",
                b,
                "--strategy",
                "optimized",
                "--optimizer",
                "native",
                "--folds",
                "2",
                "--repeats",
                "1",
                "--seed",
                "1",
                "--dirichlet-draws",
                "4",
            ],
            repository_root=repo,
        )
        == 0
    )
    native_payload = json.loads(capsys.readouterr().out)
    assert native_payload["optimizer_backend"] == "native"
    assert "honest_meta_cv_metrics" in native_payload
    assert (
        blend_main(
            [
                "search",
                "--candidate",
                a,
                "--candidate",
                b,
                "--strategy",
                "optimized",
                "--optimizer",
                "optuna",
                "--folds",
                "2",
                "--repeats",
                "1",
                "--seed",
                "1",
                "--optuna-trials",
                "5",
                "--optuna-seed",
                "1",
            ],
            repository_root=repo,
        )
        == 0
    )
    optuna_payload = json.loads(capsys.readouterr().out)
    assert optuna_payload["optimizer_backend"] == "optuna"
    assert optuna_payload["blend_id"] != native_payload["blend_id"]
    assert (
        blend_main(
            [
                "search",
                "--candidate",
                a,
                "--candidate",
                b,
                "--strategy",
                "equal",
                "--optimizer",
                "optuna",
            ],
            repository_root=repo,
        )
        == 2
    )


def test_derive_study_seed_distinct() -> None:
    seeds = {
        derive_study_seed(
            global_seed=42,
            optuna_seed=7,
            study_role=role,
            repeat=repeat,
            fold=fold,
        )
        for role, repeat, fold in (
            ("meta_fold", 1, 1),
            ("meta_fold", 1, 2),
            ("meta_fold", 2, 1),
            ("final_deployment", None, None),
        )
    }
    assert len(seeds) == 4


def test_optuna_module_import_does_not_import_optuna() -> None:
    if "optuna" in sys.modules:
        pytest.skip("optuna already imported in this process")
    importlib.import_module("src.churn_ml.blending.optuna_v1")
    assert "optuna" not in sys.modules
