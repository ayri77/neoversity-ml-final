"""Focused tests for Blend Campaign Runner v1."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.churn_ml.blend_campaign.artifact_v1 import (
    ENVIRONMENT,
    FAILURES,
    FROZEN_CONFIG,
    PLAN,
    RANKING,
    RESULTS_CSV,
    RESULTS_JSON,
    SELECTED,
    STATUS,
    SUBMISSIONS,
    SUCCESS,
    BlendCampaignArtifactError,
    campaign_dir,
)
from src.churn_ml.blend_campaign.config_v1 import (
    BlendCampaignConfigurationError,
    parse_campaign_config,
)
from src.churn_ml.blend_campaign.runner_v1 import (
    BlendCampaignRunner,
    BlendCampaignRunnerError,
    rank_results,
)
from src.churn_ml.blending.artifact_v1 import build_blend_id
from src.churn_ml.prediction_candidates.contract_v1 import (
    MANIFEST_FILENAME,
    file_sha256,
)
from tests.dataset_registry_support import (
    create_synthetic_legacy_tree,
    register_package,
)
from tests.test_prediction_blend_v1 import _two_candidates


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


def campaign_payload(
    candidate_a: str,
    candidate_b: str,
    *,
    campaign_id: str = "test_campaign",
    experiment_ids: tuple[str, ...] = ("blend_one",),
    top_k: int = 0,
    submission_enabled: bool = False,
) -> dict:
    experiment = {
        "experiment_id": "blend_one",
        "candidates": ["A", "B"],
        "strategy": "equal",
        "optimizer": "native",
        "folds": 2,
        "repeats": 1,
        "seed": 7,
        "max_active_models": 2,
        "manual_weights": {},
        "native_search": {"pairwise_grid_step": 0.1, "dirichlet_draws": 0},
        "optuna": {"trials": 5, "timeout_seconds": None, "seed": 7},
        "threshold_policy": {
            "minimum": 0.05,
            "maximum": 0.30,
            "step": 0.001,
            "maximizer_absolute_tolerance": 1.0e-12,
            "constant_probability_fallback": 0.5,
        },
        "exploratory": False,
    }
    experiments = []
    for experiment_id in experiment_ids:
        row = deepcopy(experiment)
        row["experiment_id"] = experiment_id
        experiments.append(row)
    return {
        "schema_version": "blend_campaign_v1",
        "campaign_id": campaign_id,
        "candidate_root": "artifacts/prediction_candidates",
        "blend_root": "artifacts/prediction_blends",
        "submission_root": "artifacts/candidate_submissions",
        "candidates": {"A": candidate_a, "B": candidate_b},
        "experiments": experiments,
        "materialize_policy": {"top_k": top_k, "experiment_ids": []},
        "submission_policy": {
            "enabled": submission_enabled,
            "experiment_ids": ["blend_one"] if submission_enabled else [],
            "id_prefix": "test_campaign",
        },
    }


def test_config_validation_and_duplicate_experiment_rejection(repo: Path) -> None:
    a, b = _two_candidates(repo)
    payload = campaign_payload(a, b)
    spec = parse_campaign_config(payload)
    assert spec.experiments[0].threshold_policy["step"] == 0.001

    unknown = deepcopy(payload)
    unknown["unexpected"] = True
    with pytest.raises(BlendCampaignConfigurationError, match="unknown"):
        parse_campaign_config(unknown)

    duplicate_id = campaign_payload(a, b, experiment_ids=("same", "same"))
    with pytest.raises(
        BlendCampaignConfigurationError, match="Duplicate experiment_id"
    ):
        parse_campaign_config(duplicate_id)

    duplicate_identity = campaign_payload(a, b, experiment_ids=("one", "two"))
    with pytest.raises(BlendCampaignRunnerError, match="Duplicate experiment identity"):
        BlendCampaignRunner(repo).validate(parse_campaign_config(duplicate_identity))


def test_alias_resolution_hash_freezing_and_deterministic_plan(repo: Path) -> None:
    a, b = _two_candidates(repo)
    spec = parse_campaign_config(campaign_payload(a, b))
    runner = BlendCampaignRunner(repo)
    first = runner.validate(spec)
    second = BlendCampaignRunner(repo).validate(spec)
    assert first == second
    assert first["frozen_aliases"]["A"]["candidate_id"] == a
    assert first["frozen_aliases"]["A"]["manifest_sha256"] == file_sha256(
        repo / "artifacts" / "prediction_candidates" / a / MANIFEST_FILENAME
    )
    assert first["experiments"][0]["ordered_candidate_ids"] == [a, b]


def test_exploratory_and_threshold_policy_propagation(repo: Path) -> None:
    a, b = _two_candidates(repo)
    payload = campaign_payload(a, b)
    payload["experiments"][0]["exploratory"] = True
    with pytest.raises(BlendCampaignRunnerError, match="exactly match"):
        BlendCampaignRunner(repo).validate(parse_campaign_config(payload))

    plan = BlendCampaignRunner(repo).validate(
        parse_campaign_config(campaign_payload(a, b))
    )
    assert plan["experiments"][0]["exploratory"] is False
    assert plan["experiments"][0]["threshold_policy"]["step"] == 0.001


def test_ranking_and_top_k_selection_order() -> None:
    base = {
        "status": "succeeded",
        "honest_mean_balanced_accuracy": 0.80,
        "honest_balanced_accuracy_std": 0.02,
        "min_repeat_balanced_accuracy": 0.78,
        "active_model_count": 2,
        "exploratory": False,
        "expected_blend_id": "pb1_x",
    }
    rows = [
        {**base, "experiment_id": "b"},
        {**base, "experiment_id": "a", "active_model_count": 1},
        {**base, "experiment_id": "c", "honest_mean_balanced_accuracy": 0.81},
        {**base, "experiment_id": "failed", "status": "failed"},
    ]
    ranking = rank_results(rows)
    assert [row["experiment_id"] for row in ranking] == ["c", "a", "b"]
    assert ranking[0]["rank"] == 1


def test_failed_experiment_continues_and_resume_recovers_cached_result(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b = _two_candidates(repo)
    payload = campaign_payload(a, b, experiment_ids=("first", "second"))
    # Avoid duplicate identity by changing the seed of the second experiment.
    payload["experiments"][1]["seed"] = 8
    spec = parse_campaign_config(payload)
    calls: list[int] = []

    def fake_search(pool, settings):
        calls.append(settings.seed)
        if settings.seed == 7:
            raise RuntimeError("synthetic failure")
        return _fake_search_payload(pool, settings)

    monkeypatch.setattr(
        "src.churn_ml.blend_campaign.runner_v1.search_blend", fake_search
    )
    runner = BlendCampaignRunner(repo)
    result = runner.run(spec)
    states = result["status"]["experiments"]
    assert states["first"]["state"] == "failed"
    assert states["second"]["state"] == "succeeded"
    assert result["status"]["state"] == "completed_with_failures"

    calls.clear()
    resumed = runner.resume(campaign_dir(repo, spec.campaign_id))
    assert resumed["status"]["experiments"]["second"]["state"] == "succeeded"
    assert calls == [7]  # completed second search was skipped with exact hashes.


def test_plan_no_overwrite_and_candidate_hash_drift_fail_closed(repo: Path) -> None:
    a, b = _two_candidates(repo)
    payload = campaign_payload(a, b)
    runner = BlendCampaignRunner(repo)
    runner.plan(parse_campaign_config(payload))

    changed = deepcopy(payload)
    changed["experiments"][0]["seed"] = 99
    with pytest.raises(BlendCampaignArtifactError, match="Refusing to overwrite"):
        runner.plan(parse_campaign_config(changed))

    manifest = repo / "artifacts" / "prediction_candidates" / a / MANIFEST_FILENAME
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["created_at_utc"] = "tampered"
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(BlendCampaignRunnerError, match="changed"):
        runner.resume(campaign_dir(repo, "test_campaign"))


def test_materialize_top_submission_handoff_no_network_and_success_last(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b = _two_candidates(repo)
    spec = parse_campaign_config(
        campaign_payload(a, b, top_k=1, submission_enabled=True)
    )
    handoffs: list[dict] = []

    def fake_submission(candidate_id: str, **kwargs):
        handoffs.append({"candidate_id": candidate_id, **kwargs})
        return {
            "ok": True,
            "submission_id": kwargs["submission_id"],
            "candidate_id": candidate_id,
            "network_access": False,
            "kaggle_upload": False,
        }

    monkeypatch.setattr(
        "src.churn_ml.blend_campaign.runner_v1.generate_candidate_submission",
        fake_submission,
    )
    result = BlendCampaignRunner(repo).run(spec)
    directory = campaign_dir(repo, spec.campaign_id)
    selected = result["selected"]["materializations"]
    assert selected[0]["status"] == "succeeded"
    assert handoffs[0]["candidate_id"] == selected[0]["canonical_candidate_id"]
    assert handoffs[0]["candidates_root_relative"] == spec.candidate_root
    assert result["submissions"]["network_access"] is False
    assert result["submissions"]["kaggle_upload"] is False

    required = [
        FROZEN_CONFIG,
        PLAN,
        STATUS,
        RESULTS_CSV,
        RESULTS_JSON,
        FAILURES,
        RANKING,
        SELECTED,
        SUBMISSIONS,
        ENVIRONMENT,
    ]
    success_mtime = (directory / SUCCESS).stat().st_mtime_ns
    assert all((directory / name).is_file() for name in required)
    assert all(
        success_mtime >= (directory / name).stat().st_mtime_ns for name in required
    )


def _fake_search_payload(pool, settings) -> dict:
    blend_id = build_blend_id(pool, settings)
    weights = {
        candidate_id: 1.0 / len(pool.candidate_ids)
        for candidate_id in pool.candidate_ids
    }
    return {
        "blend_id": blend_id,
        "exploratory": pool.exploratory,
        "final_deployment_weights": weights,
        "final_deployment_threshold": 0.2,
        "honest_meta_cv_metrics": {
            "mean_repeat_balanced_accuracy": 0.75,
            "std_repeat_balanced_accuracy": 0.01,
            "min_repeat_balanced_accuracy": 0.74,
            "max_repeat_balanced_accuracy": 0.76,
            "pooled_repeated_held_out_confusion_metrics": {
                "sensitivity": 0.7,
                "specificity": 0.8,
            },
        },
        "full_oof_descriptive_metrics": {"balanced_accuracy": 0.77},
    }
