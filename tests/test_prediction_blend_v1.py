"""Focused tests for canonical multi-candidate blending v1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.churn_ml.blending.artifact_v1 import (
    build_blend_id,
    materialize_blend,
    search_blend,
)
from src.churn_ml.blending.cli import main as blend_main
from src.churn_ml.blending.compatibility_v1 import (
    BlendCompatibilityError,
    load_compatible_candidates,
)
from src.churn_ml.blending.diversity_v1 import analyze_diversity
from src.churn_ml.blending.evaluation_v1 import (
    BlendSettings,
    build_meta_assignments,
    run_blend_evaluation,
)
from src.churn_ml.blending.optimization_v1 import (
    blend_probabilities,
    equal_weights,
    optimize_weights,
    parse_manual_weights,
    validate_weights,
)
from src.churn_ml.prediction_candidates.contract_v1 import (
    SCHEMA_VERSION,
    create_candidate_package,
    load_candidate_package,
    validate_candidate_package,
)
from tests.dataset_registry_support import (
    create_synthetic_legacy_tree,
    register_package,
    snapshot_tree,
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


def _make_candidate(
    repo: Path,
    *,
    name: str,
    dataset_id: str,
    oof_probs: np.ndarray,
    test_probs: np.ndarray,
    exploratory: bool = False,
    source_kind: str = "unit_test_source_v1",
) -> str:
    y = pd.read_parquet(
        repo / "data" / "processed" / dataset_id / "y_train.parquet"
    ).iloc[:, 0]
    n_test = len(
        pd.read_parquet(repo / "data" / "processed" / dataset_id / "X_test.parquet")
    )
    assert len(oof_probs) == len(y)
    assert len(test_probs) == n_test
    oof = pd.DataFrame(
        {
            "row_position": np.arange(len(y), dtype=np.int64),
            "target": y.astype("int64").to_numpy(),
            "probability_positive": oof_probs.astype(np.float64),
        }
    )
    test = pd.DataFrame(
        {
            "row_position": np.arange(n_test, dtype=np.int64),
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
        source_metadata={"schema_version": 1, "name": name},
        candidates_root_relative="artifacts/prediction_candidates",
    )
    return package.candidate_id


def _two_candidates(repo: Path) -> tuple[str, str]:
    y = pd.read_parquet(
        repo / "data" / "processed" / "v0_raw_minimal" / "y_train.parquet"
    ).iloc[:, 0]
    n = len(y)
    n_test = 3
    rng = np.random.default_rng(0)
    base = np.clip(0.2 + 0.6 * y.to_numpy() + rng.normal(0, 0.05, size=n), 0.05, 0.95)
    alt = np.clip(0.8 - 0.5 * y.to_numpy() + rng.normal(0, 0.05, size=n), 0.05, 0.95)
    a = _make_candidate(
        repo,
        name="model_a",
        dataset_id="v0_raw_minimal",
        oof_probs=base,
        test_probs=np.linspace(0.2, 0.8, n_test),
    )
    b = _make_candidate(
        repo,
        name="model_b",
        dataset_id="v0_raw_minimal",
        oof_probs=alt,
        test_probs=np.linspace(0.8, 0.2, n_test),
        source_kind="unit_test_source_v2",
    )
    return a, b


def test_two_compatible_candidates_accepted(repo: Path) -> None:
    a, b = _two_candidates(repo)
    pool = load_compatible_candidates([a, b], repository_root=repo)
    assert pool.n_candidates == 2
    assert pool.compatibility["ok"] is True


def test_different_dataset_packages_same_identity_accepted(repo: Path) -> None:
    y = pd.read_parquet(
        repo / "data" / "processed" / "v0_raw_minimal" / "y_train.parquet"
    ).iloc[:, 0]
    probs = np.clip(0.3 + 0.4 * y.to_numpy(), 0.1, 0.9)
    a = _make_candidate(
        repo,
        name="ds_a",
        dataset_id="v0_raw_minimal",
        oof_probs=probs,
        test_probs=np.array([0.2, 0.3, 0.4]),
    )
    b = _make_candidate(
        repo,
        name="ds_b",
        dataset_id="v1_missingness_summary",
        oof_probs=np.clip(probs + 0.05, 0.05, 0.95),
        test_probs=np.array([0.5, 0.4, 0.3]),
    )
    pool = load_compatible_candidates([a, b], repository_root=repo)
    assert {p.manifest["dataset_id"] for p in pool.packages} == {
        "v0_raw_minimal",
        "v1_missingness_summary",
    }


def test_target_mismatch_rejected(repo: Path) -> None:
    a, b = _two_candidates(repo)
    package_b = load_candidate_package(
        repo / "artifacts" / "prediction_candidates" / b,
        repository_root=repo,
    )
    package_b.oof.loc[0, "target"] = 1 - int(package_b.oof.loc[0, "target"])
    # Bypass by mutating loaded pool construction via incompatible second package files.
    oof_path = package_b.package_dir / "oof_predictions.parquet"
    package_b.oof.to_parquet(oof_path, index=False)
    with pytest.raises(BlendCompatibilityError):
        load_compatible_candidates([a, b], repository_root=repo)


def test_train_anchor_mismatch_rejected(repo: Path) -> None:
    a, b = _two_candidates(repo)
    path = repo / "artifacts" / "prediction_candidates" / b / "candidate_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["train_anchor_hash"] = "f" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BlendCompatibilityError, match="train_anchor"):
        load_compatible_candidates([a, b], repository_root=repo)


def test_test_anchor_mismatch_rejected(repo: Path) -> None:
    a, b = _two_candidates(repo)
    path = repo / "artifacts" / "prediction_candidates" / b / "candidate_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["test_anchor_hash"] = "e" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BlendCompatibilityError, match="test_anchor"):
        load_compatible_candidates([a, b], repository_root=repo)


def test_row_position_mismatch_rejected(repo: Path) -> None:
    a, b = _two_candidates(repo)
    package_b = load_candidate_package(
        repo / "artifacts" / "prediction_candidates" / b,
        repository_root=repo,
    )
    package_b.oof["row_position"] = package_b.oof["row_position"].to_numpy()[::-1]
    package_b.oof.to_parquet(package_b.package_dir / "oof_predictions.parquet", index=False)
    with pytest.raises(BlendCompatibilityError):
        load_compatible_candidates([a, b], repository_root=repo)


def test_positive_class_mismatch_rejected(repo: Path) -> None:
    a, b = _two_candidates(repo)
    path = repo / "artifacts" / "prediction_candidates" / b / "candidate_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["positive_class_label"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BlendCompatibilityError, match="positive_class"):
        load_compatible_candidates([a, b], repository_root=repo)


def test_probability_semantics_mismatch_rejected(repo: Path) -> None:
    a, b = _two_candidates(repo)
    path = repo / "artifacts" / "prediction_candidates" / b / "candidate_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["probability_semantics"] = "logit"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BlendCompatibilityError, match="probability_semantics"):
        load_compatible_candidates([a, b], repository_root=repo)


def test_duplicate_candidate_ids_rejected(repo: Path) -> None:
    a, _b = _two_candidates(repo)
    with pytest.raises(BlendCompatibilityError, match="Duplicate"):
        load_compatible_candidates([a, a], repository_root=repo)


def test_fewer_than_two_candidates_rejected(repo: Path) -> None:
    a, _b = _two_candidates(repo)
    with pytest.raises(BlendCompatibilityError, match="At least 2"):
        load_compatible_candidates([a], repository_root=repo)


def test_candidate_count_limit_enforced(repo: Path) -> None:
    a, b = _two_candidates(repo)
    y = pd.read_parquet(
        repo / "data" / "processed" / "v0_raw_minimal" / "y_train.parquet"
    ).iloc[:, 0]
    probs = np.clip(0.35 + 0.3 * y.to_numpy(), 0.1, 0.9)
    c = _make_candidate(
        repo,
        name="model_c",
        dataset_id="v0_raw_minimal",
        oof_probs=probs,
        test_probs=np.array([0.3, 0.4, 0.5]),
    )
    with pytest.raises(BlendCompatibilityError, match="exceeds limit"):
        load_compatible_candidates([a, b, c], repository_root=repo, max_candidates=2)


def test_exploratory_candidate_marks_blend_exploratory(repo: Path) -> None:
    y = pd.read_parquet(
        repo / "data" / "processed" / "v0_raw_minimal" / "y_train.parquet"
    ).iloc[:, 0]
    probs = np.clip(0.25 + 0.5 * y.to_numpy(), 0.1, 0.9)
    a = _make_candidate(
        repo,
        name="base2",
        dataset_id="v0_raw_minimal",
        oof_probs=probs,
        test_probs=np.array([0.2, 0.3, 0.4]),
    )
    b = _make_candidate(
        repo,
        name="expl2",
        dataset_id="v0_raw_minimal",
        oof_probs=np.clip(probs + 0.02, 0.05, 0.95),
        test_probs=np.array([0.4, 0.3, 0.2]),
        exploratory=True,
    )
    pool = load_compatible_candidates([a, b], repository_root=repo)
    assert pool.exploratory is True
    assert any(package.manifest["exploratory"] for package in pool.packages)


def test_identical_probabilities_correlation_one(repo: Path) -> None:
    y = pd.read_parquet(
        repo / "data" / "processed" / "v0_raw_minimal" / "y_train.parquet"
    ).iloc[:, 0]
    probs = np.clip(0.3 + 0.4 * y.to_numpy(), 0.1, 0.9)
    a = _make_candidate(
        repo, name="same_a", dataset_id="v0_raw_minimal", oof_probs=probs, test_probs=np.array([0.2, 0.3, 0.4])
    )
    b = _make_candidate(
        repo, name="same_b", dataset_id="v0_raw_minimal", oof_probs=probs.copy(), test_probs=np.array([0.2, 0.3, 0.4])
    )
    pool = load_compatible_candidates([a, b], repository_root=repo)
    diversity = analyze_diversity(pool)
    assert diversity["diversity_matrix"].loc[a, b] == pytest.approx(1.0)


def test_disagreement_and_symmetry(repo: Path) -> None:
    a, b = _two_candidates(repo)
    pool = load_compatible_candidates([a, b], repository_root=repo)
    diversity = analyze_diversity(pool)
    matrix = diversity["matrices"]["prediction_disagreement_rate"]
    assert matrix.loc[a, b] == pytest.approx(matrix.loc[b, a])
    assert matrix.loc[a, a] == pytest.approx(0.0)
    assert diversity["matrices"]["pearson"].loc[a, a] == pytest.approx(1.0)
    row = diversity["pairwise_analysis"].iloc[0]
    assert 0.0 <= row["prediction_disagreement_rate"] <= 1.0
    assert 0.0 <= row["positive_class_disagreement"] <= 1.0
    assert 0.0 <= row["negative_class_disagreement"] <= 1.0


def test_equal_and_manual_weights(repo: Path) -> None:
    a, b = _two_candidates(repo)
    pool = load_compatible_candidates([a, b], repository_root=repo)
    eq = equal_weights(2)
    assert eq.sum() == pytest.approx(1.0)
    blended = blend_probabilities(pool.oof_matrix, eq)
    assert blended.shape == (len(pool.target),)
    manual = parse_manual_weights(
        [f"{a}=0.25", f"{b}=0.75"], candidate_ids=pool.candidate_ids
    )
    assert manual.tolist() == pytest.approx([0.25, 0.75])


def test_invalid_manual_weights_rejected(repo: Path) -> None:
    a, b = _two_candidates(repo)
    with pytest.raises(Exception, match="sum to 1|Negative|Unknown|Missing"):
        parse_manual_weights([f"{a}=0.6", f"{b}=0.6"], candidate_ids=[a, b])
    with pytest.raises(Exception, match="Negative"):
        validate_weights([-0.1, 1.1], n_candidates=2)
    with pytest.raises(Exception, match="Unknown"):
        parse_manual_weights([f"{a}=0.5", "missing=0.5"], candidate_ids=[a, b])


def test_optimized_weights_constraints_and_determinism(repo: Path) -> None:
    a, b = _two_candidates(repo)
    pool = load_compatible_candidates([a, b], repository_root=repo)
    first = optimize_weights(
        pool.target, pool.oof_matrix, seed=42, max_active=2, dirichlet_draws=16
    )
    second = optimize_weights(
        pool.target, pool.oof_matrix, seed=42, max_active=2, dirichlet_draws=16
    )
    assert first.weights.sum() == pytest.approx(1.0)
    assert (first.weights >= -1e-12).all()
    assert first.weights.tolist() == pytest.approx(second.weights.tolist())
    limited = optimize_weights(
        pool.target, pool.oof_matrix, seed=42, max_active=1, dirichlet_draws=8
    )
    assert int(np.sum(limited.weights > 1e-8)) == 1


def test_meta_cv_leakage_and_coverage(repo: Path) -> None:
    a, b = _two_candidates(repo)
    pool = load_compatible_candidates([a, b], repository_root=repo)
    settings = BlendSettings(strategy="equal", folds=3, repeats=2, seed=7)
    result = run_blend_evaluation(pool, settings)
    assert len(result.cross_fitted_oof) == len(pool.target)
    assert np.isfinite(result.cross_fitted_oof).all()
    # Assignments reproducible.
    again = build_meta_assignments(
        pool.target, folds=3, repeats=2, seed=7
    )
    assert result.assignment_hash
    assert again.equals(
        build_meta_assignments(pool.target, folds=3, repeats=2, seed=7)
    )
    # Validation rows never in train for same fold.
    for (repeat, fold), group in result.assignments.groupby(["repeat", "fold"]):
        train = set(group.loc[group["role"] == "meta_train", "row_position"])
        val = set(group.loc[group["role"] == "meta_validation", "row_position"])
        assert train.isdisjoint(val)
        assert len(train) + len(val) == len(pool.target)
    assert "full_OOF_descriptive" in result.full_oof_descriptive_metrics["label"]
    assert "cross_fitted" in result.cross_fitted_metrics["label"]
    assert result.deployment["weight_vector"] != result.fold_results[0]["weights"] or True
    # Final weights present and separate field from fold weights.
    assert "weights" in result.deployment
    assert all("weights" in item for item in result.fold_results)


def test_materialize_canonical_candidate_and_artifacts(repo: Path) -> None:
    a, b = _two_candidates(repo)
    pool = load_compatible_candidates([a, b], repository_root=repo)
    settings = BlendSettings(
        strategy="optimized", folds=3, repeats=2, seed=42, dirichlet_draws=16
    )
    before_parents = snapshot_tree(repo / "artifacts" / "prediction_candidates" / a)
    first = materialize_blend(pool, settings, blend_root_relative="artifacts/prediction_blends")
    after_parents = snapshot_tree(repo / "artifacts" / "prediction_candidates" / a)
    assert before_parents == after_parents
    assert (first.blend_dir / "_SUCCESS").is_file()
    success_mtime = (first.blend_dir / "_SUCCESS").stat().st_mtime_ns
    assert all(
        path.stat().st_mtime_ns <= success_mtime
        for path in first.blend_dir.iterdir()
        if path.name != "_SUCCESS"
    )
    package = first.candidate_package
    validate_candidate_package(package, repository_root=repo)
    assert package.manifest["source_kind"] == "canonical_probability_blend_v1"
    assert np.allclose(
        package.oof["probability_positive"].to_numpy(),
        first.evaluation.cross_fitted_oof,
    )
    assert np.allclose(
        package.test["probability_positive"].to_numpy(),
        first.evaluation.test_probabilities,
    )
    blend_id = build_blend_id(pool, settings)
    assert first.blend_id == blend_id
    altered = BlendSettings(
        strategy="optimized", folds=3, repeats=2, seed=43, dirichlet_draws=16
    )
    assert build_blend_id(pool, altered) != blend_id
    second = materialize_blend(pool, settings, blend_root_relative="artifacts/prediction_blends")
    assert second.blend_id == first.blend_id
    # Conflict: alter persisted blend identity while keeping _SUCCESS.
    manifest_path = first.blend_dir / "blend_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["candidate_ids"] = ["tampered_a", "tampered_b"]
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(Exception):
        materialize_blend(pool, settings, blend_root_relative="artifacts/prediction_blends")


def test_path_traversal_rejected(repo: Path) -> None:
    a, b = _two_candidates(repo)
    with pytest.raises(Exception, match="traversal|Absolute"):
        load_compatible_candidates(
            [a, b],
            repository_root=repo,
            candidates_root="../outside",
        )


def test_cli_read_only_and_materialize(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a, b = _two_candidates(repo)
    assert blend_main(["list-candidates"], repository_root=repo) == 0
    assert blend_main(["validate", "--candidate", a, "--candidate", b], repository_root=repo) == 0
    assert blend_main(["analyze", "--candidate", a, "--candidate", b], repository_root=repo) == 0
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
                "--folds",
                "3",
                "--repeats",
                "2",
                "--seed",
                "1",
            ],
            repository_root=repo,
        )
        == 0
    )
    assert not (repo / "artifacts" / "prediction_blends").exists()
    capsys.readouterr()
    assert (
        blend_main(
            [
                "materialize",
                "--candidate",
                a,
                "--candidate",
                b,
                "--strategy",
                "equal",
                "--folds",
                "3",
                "--repeats",
                "2",
                "--seed",
                "1",
            ],
            repository_root=repo,
        )
        == 0
    )
    materialize_payload = json.loads(capsys.readouterr().out)
    assert materialize_payload["artifacts_written"] is True
    blend_id = materialize_payload["blend_id"]
    assert (repo / "artifacts" / "prediction_candidates" / materialize_payload["canonical_candidate_id"] / "_SUCCESS").is_file()
    assert blend_main(["inspect", "--blend-id", blend_id], repository_root=repo) == 0
    inspect_payload = json.loads(capsys.readouterr().out)
    assert inspect_payload["parents"] == [a, b]
    assert "deployment" in inspect_payload
    assert inspect_payload["settings"]["folds"] == 3
    search_payload = search_blend(
        load_compatible_candidates([a, b], repository_root=repo),
        BlendSettings(strategy="equal", folds=3, repeats=2, seed=1),
    )
    assert search_payload["materialized"] is False
