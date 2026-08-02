"""Resumable orchestration over the canonical probability-blending backend."""

from __future__ import annotations

import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from src.churn_ml.blend_campaign.artifact_v1 import (
    ENVIRONMENT,
    FAILURES,
    FROZEN_CONFIG,
    PLAN,
    SELECTED,
    STATUS,
    SUBMISSIONS,
    BlendCampaignArtifactError,
    atomic_write_json,
    campaign_dir,
    environment_payload,
    load_json,
    remove_success_marker,
    utc_now,
    write_immutable_json,
    write_immutable_yaml,
    write_ranking,
    write_result_views,
    write_success_last,
)
from src.churn_ml.blend_campaign.config_v1 import (
    BlendCampaignConfig,
    parse_campaign_config,
)
from src.churn_ml.blending.artifact_v1 import (
    build_blend_id,
    materialize_blend,
    search_blend,
)
from src.churn_ml.blending.compatibility_v1 import (
    CompatibleCandidateSet,
    load_compatible_candidates,
)
from src.churn_ml.prediction_candidates.contract_v1 import (
    MANIFEST_FILENAME,
    file_sha256,
)
from src.churn_ml.prediction_candidates.submission_v1 import (
    generate_candidate_submission,
)
from src.churn_ml.research_data import canonical_sha256


class BlendCampaignRunnerError(RuntimeError):
    """Raised when campaign state cannot be safely planned or resumed."""


class BlendCampaignRunner:
    """One-process runner with candidate-set/package compatibility caching."""

    def __init__(self, repository_root: Path) -> None:
        self.repository_root = repository_root.resolve()
        self._pool_cache: dict[tuple[str, tuple[str, ...]], CompatibleCandidateSet] = {}

    def validate(self, config: BlendCampaignConfig) -> dict[str, Any]:
        return self._build_plan(config)

    def plan(self, config: BlendCampaignConfig) -> dict[str, Any]:
        plan = self._build_plan(config)
        directory = campaign_dir(self.repository_root, config.campaign_id)
        directory.mkdir(parents=True, exist_ok=True)
        write_immutable_yaml(directory / FROZEN_CONFIG, config.payload)
        write_immutable_json(directory / PLAN, plan)
        if not (directory / STATUS).exists():
            atomic_write_json(directory / STATUS, _initial_status(plan))
            write_result_views(directory, [])
            atomic_write_json(
                directory / FAILURES, {"schema_version": 1, "failures": []}
            )
            write_ranking(directory, [])
            atomic_write_json(
                directory / SELECTED,
                {"schema_version": 1, "selected": [], "materializations": []},
            )
            atomic_write_json(
                directory / SUBMISSIONS,
                {
                    "schema_version": 1,
                    "generated": [],
                    "network_access": False,
                    "kaggle_upload": False,
                },
            )
            write_immutable_json(directory / ENVIRONMENT, environment_payload())
        else:
            self._assert_artifacts_match(config, plan, directory)
        return {"campaign_dir": directory, "plan": plan}

    def run(self, config: BlendCampaignConfig) -> dict[str, Any]:
        planned = self.plan(config)
        directory = planned["campaign_dir"]
        status = load_json(directory / STATUS)
        if status.get("state") not in {"planned", "interrupted"}:
            raise BlendCampaignRunnerError(
                f"Campaign state is {status.get('state')!r}; use resume for an existing campaign."
            )
        return self._execute(config, planned["plan"], directory, retry_failed=False)

    def resume(self, directory: Path) -> dict[str, Any]:
        directory = directory.resolve()
        config, plan = self._load_frozen(directory)
        self._assert_artifacts_match(config, plan, directory)
        return self._execute(config, plan, directory, retry_failed=True)

    def inspect(self, directory: Path) -> dict[str, Any]:
        directory = directory.resolve()
        _config, plan = self._load_frozen(directory)
        return {
            "campaign_dir": str(directory),
            "plan": plan,
            "status": load_json(directory / STATUS),
            "results": load_json(directory / "experiment_results.json"),
            "failures": load_json(directory / FAILURES),
            "selected": load_json(directory / SELECTED),
            "submissions": load_json(directory / SUBMISSIONS),
        }

    def materialize_top(
        self, directory: Path, *, top_k: int | None = None
    ) -> dict[str, Any]:
        directory = directory.resolve()
        config, plan = self._load_frozen(directory)
        self._assert_artifacts_match(config, plan, directory)
        status = load_json(directory / STATUS)
        if not _all_searches_terminal(status):
            raise BlendCampaignRunnerError(
                "All planned searches must finish before materialization."
            )
        remove_success_marker(directory)
        results = _result_rows(status, plan)
        ranking = rank_results(results)
        payload, failures = self._materialize_selected(
            config, plan, directory, ranking, top_k=top_k
        )
        _merge_failures(directory, failures)
        atomic_write_json(directory / SELECTED, payload)
        write_success_last(directory, _success_payload(config, plan, status))
        return payload

    def generate_submissions(self, directory: Path) -> dict[str, Any]:
        directory = directory.resolve()
        config, plan = self._load_frozen(directory)
        self._assert_artifacts_match(config, plan, directory)
        if not config.submission_enabled:
            raise BlendCampaignRunnerError(
                "submission_policy.enabled is false in frozen config."
            )
        remove_success_marker(directory)
        selected = load_json(directory / SELECTED)
        payload, failures = self._generate_submissions(config, directory, selected)
        _merge_failures(directory, failures)
        atomic_write_json(directory / SUBMISSIONS, payload)
        status = load_json(directory / STATUS)
        write_success_last(directory, _success_payload(config, plan, status))
        return payload

    def _execute(
        self,
        config: BlendCampaignConfig,
        plan: dict[str, Any],
        directory: Path,
        *,
        retry_failed: bool,
    ) -> dict[str, Any]:
        remove_success_marker(directory)
        status = load_json(directory / STATUS)
        status["state"] = "running"
        status["updated_at_utc"] = utc_now()
        atomic_write_json(directory / STATUS, status)
        experiments_by_id = {item.experiment_id: item for item in config.experiments}

        for planned in plan["experiments"]:
            experiment_id = planned["experiment_id"]
            cell = status["experiments"][experiment_id]
            if cell["state"] == "succeeded":
                _emit_progress(
                    "search_skipped",
                    experiment_id=experiment_id,
                    reason="completed_identity_match",
                )
                continue
            if cell["state"] == "failed" and not retry_failed:
                continue
            cached_result = self._load_cached_search_result(directory, planned)
            if cached_result is not None:
                cell.update(
                    {
                        "state": "succeeded",
                        "result": cached_result,
                        "failure_reason": None,
                    }
                )
                atomic_write_json(directory / STATUS, status)
                _emit_progress("search_recovered", experiment_id=experiment_id)
                continue

            cell["state"] = "running"
            cell["attempts"] = int(cell.get("attempts", 0)) + 1
            cell["failure_reason"] = None
            atomic_write_json(directory / STATUS, status)
            _emit_progress("search_started", experiment_id=experiment_id)
            started = time.perf_counter()
            try:
                pool = self._pool(
                    planned["ordered_candidate_ids"], config.candidate_root
                )
                experiment = experiments_by_id[experiment_id]
                payload = search_blend(pool, experiment.settings(pool.candidate_ids))
                runtime = time.perf_counter() - started
                result = _normalize_result(planned, payload, pool, runtime)
                search_path = directory / "search_results" / f"{experiment_id}.json"
                result["source_result_path"] = _relative(
                    search_path, self.repository_root
                )
                envelope = {
                    "schema_version": 1,
                    "experiment_identity_hash": planned["experiment_identity_hash"],
                    "candidate_manifest_hashes": planned["candidate_manifest_hashes"],
                    "result": result,
                }
                write_immutable_json(search_path, envelope)
                cell.update({"state": "succeeded", "result": result})
                _emit_progress(
                    "search_succeeded",
                    experiment_id=experiment_id,
                    expected_blend_id=planned["expected_blend_id"],
                    honest_mean_balanced_accuracy=result[
                        "honest_mean_balanced_accuracy"
                    ],
                )
            except Exception as error:  # noqa: BLE001 - failure isolation is a campaign requirement
                runtime = time.perf_counter() - started
                reason = f"{type(error).__name__}: {error}"
                cell.update(
                    {
                        "state": "failed",
                        "failure_reason": reason,
                        "result": _failed_result(planned, reason, runtime),
                    }
                )
                _emit_progress(
                    "search_failed", experiment_id=experiment_id, reason=reason
                )
            status["updated_at_utc"] = utc_now()
            atomic_write_json(directory / STATUS, status)

        results = _result_rows(status, plan)
        ranking = rank_results(results)
        failures = _search_failures(status)
        write_result_views(directory, results)
        write_ranking(directory, ranking)
        atomic_write_json(
            directory / FAILURES, {"schema_version": 1, "failures": failures}
        )

        selected, materialization_failures = self._materialize_selected(
            config, plan, directory, ranking, top_k=None
        )
        failures.extend(materialization_failures)
        atomic_write_json(directory / SELECTED, selected)

        submissions = {
            "schema_version": 1,
            "generated": [],
            "network_access": False,
            "kaggle_upload": False,
        }
        if config.submission_enabled:
            submissions, submission_failures = self._generate_submissions(
                config, directory, selected
            )
            failures.extend(submission_failures)
        atomic_write_json(directory / SUBMISSIONS, submissions)
        atomic_write_json(
            directory / FAILURES, {"schema_version": 1, "failures": failures}
        )

        status["state"] = "completed_with_failures" if failures else "completed"
        status["updated_at_utc"] = utc_now()
        atomic_write_json(directory / STATUS, status)
        write_success_last(directory, _success_payload(config, plan, status))
        _emit_progress(
            "campaign_completed", campaign_id=config.campaign_id, state=status["state"]
        )
        return self.inspect(directory)

    def _build_plan(self, config: BlendCampaignConfig) -> dict[str, Any]:
        frozen_aliases: dict[str, dict[str, str]] = {}
        for alias, candidate_id in config.aliases.items():
            package_dir = self.repository_root / config.candidate_root / candidate_id
            manifest_path = package_dir / MANIFEST_FILENAME
            if not manifest_path.is_file():
                raise BlendCampaignRunnerError(
                    f"Candidate alias {alias!r} resolves to missing exact candidate ID {candidate_id!r}."
                )
            frozen_aliases[alias] = {
                "candidate_id": candidate_id,
                "manifest_sha256": file_sha256(manifest_path),
            }

        planned_experiments: list[dict[str, Any]] = []
        seen_identities: set[str] = set()
        for order, experiment in enumerate(config.experiments, start=1):
            candidate_ids = tuple(config.aliases[alias] for alias in experiment.aliases)
            pool = self._pool(candidate_ids, config.candidate_root)
            if experiment.exploratory != pool.exploratory:
                raise BlendCampaignRunnerError(
                    f"{experiment.experiment_id}: exploratory={experiment.exploratory} does not "
                    f"exactly match canonical candidate propagation ({pool.exploratory})."
                )
            hashes = [
                file_sha256(package.package_dir / MANIFEST_FILENAME)
                for package in pool.packages
            ]
            settings = experiment.settings(pool.candidate_ids)
            settings_payload = settings.identity_payload(
                parent_candidate_ids=pool.candidate_ids,
                parent_manifest_hashes=hashes,
            )
            identity = canonical_sha256(
                {
                    "ordered_candidate_ids": list(pool.candidate_ids),
                    "candidate_manifest_hashes": hashes,
                    "settings": settings_payload,
                    "exploratory": experiment.exploratory,
                }
            )
            if identity in seen_identities:
                raise BlendCampaignRunnerError(
                    f"Duplicate experiment identity rejected at {experiment.experiment_id}."
                )
            seen_identities.add(identity)
            planned_experiments.append(
                {
                    "execution_order": order,
                    "experiment_id": experiment.experiment_id,
                    "aliases": list(experiment.aliases),
                    "ordered_candidate_ids": list(pool.candidate_ids),
                    "candidate_manifest_hashes": hashes,
                    "strategy": settings.strategy,
                    "optimizer": settings.optimizer_backend,
                    "folds": settings.folds,
                    "repeats": settings.repeats,
                    "seed": settings.seed,
                    "max_active_models": settings.max_active_models,
                    "threshold_policy": dict(settings.threshold_policy or {}),
                    "settings": settings_payload,
                    "exploratory": experiment.exploratory,
                    "expected_blend_id": build_blend_id(pool, settings),
                    "experiment_identity_hash": identity,
                }
            )
        body = {
            "schema_version": "blend_campaign_plan_v1",
            "campaign_id": config.campaign_id,
            "config_hash": config.config_hash,
            "candidate_root": config.candidate_root,
            "blend_root": config.blend_root,
            "submission_root": config.submission_root,
            "frozen_aliases": frozen_aliases,
            "experiments": planned_experiments,
            "materialize_policy": {
                "top_k": config.materialize_top_k,
                "experiment_ids": list(config.materialize_experiment_ids),
            },
            "submission_policy": {
                "enabled": config.submission_enabled,
                "experiment_ids": list(config.submission_experiment_ids),
                "id_prefix": config.submission_id_prefix,
            },
        }
        return {**body, "plan_hash": canonical_sha256(body)}

    def _pool(
        self, candidate_ids: tuple[str, ...] | list[str], candidate_root: str
    ) -> CompatibleCandidateSet:
        key = (candidate_root, tuple(candidate_ids))
        if key not in self._pool_cache:
            self._pool_cache[key] = load_compatible_candidates(
                key[1],
                repository_root=self.repository_root,
                candidates_root=candidate_root,
            )
        return self._pool_cache[key]

    def _load_frozen(
        self, directory: Path
    ) -> tuple[BlendCampaignConfig, dict[str, Any]]:
        config_path = directory / FROZEN_CONFIG
        if not config_path.is_file():
            raise BlendCampaignArtifactError(f"Frozen config missing: {config_path}")
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = parse_campaign_config(payload, source_path=config_path)
        return config, load_json(directory / PLAN)

    def _assert_artifacts_match(
        self, config: BlendCampaignConfig, plan: Mapping[str, Any], directory: Path
    ) -> None:
        if directory != campaign_dir(self.repository_root, config.campaign_id):
            raise BlendCampaignRunnerError(
                "Campaign directory does not match frozen campaign_id."
            )
        body = {
            key: deepcopy(value) for key, value in plan.items() if key != "plan_hash"
        }
        if canonical_sha256(body) != plan.get("plan_hash"):
            raise BlendCampaignRunnerError("Frozen campaign plan hash is invalid.")
        fresh = self._build_plan(config)
        if fresh["plan_hash"] != plan.get("plan_hash"):
            raise BlendCampaignRunnerError(
                "Candidate identities or manifests changed; campaign resume failed closed."
            )
        status = load_json(directory / STATUS)
        if status.get("plan_hash") != plan.get("plan_hash"):
            raise BlendCampaignRunnerError(
                "Campaign status does not match frozen plan identity."
            )

    def _load_cached_search_result(
        self, directory: Path, planned: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        path = directory / "search_results" / f"{planned['experiment_id']}.json"
        if not path.is_file():
            return None
        payload = load_json(path)
        if (
            payload.get("experiment_identity_hash")
            != planned["experiment_identity_hash"]
            or payload.get("candidate_manifest_hashes")
            != planned["candidate_manifest_hashes"]
        ):
            raise BlendCampaignRunnerError(
                f"Cached search result identity drift: {planned['experiment_id']}"
            )
        result = payload.get("result")
        if (
            not isinstance(result, dict)
            or result.get("expected_blend_id") != planned["expected_blend_id"]
        ):
            raise BlendCampaignRunnerError(
                f"Cached search result is invalid: {planned['experiment_id']}"
            )
        return result

    def _materialize_selected(
        self,
        config: BlendCampaignConfig,
        plan: Mapping[str, Any],
        directory: Path,
        ranking: list[dict[str, Any]],
        *,
        top_k: int | None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        limit = config.materialize_top_k if top_k is None else int(top_k)
        if limit < 0:
            raise BlendCampaignRunnerError("top_k must be >= 0.")
        ranked_ids = [row["experiment_id"] for row in ranking[:limit]]
        selected_ids = list(
            dict.fromkeys([*config.materialize_experiment_ids, *ranked_ids])
        )
        plan_by_id = {row["experiment_id"]: row for row in plan["experiments"]}
        experiments_by_id = {row.experiment_id: row for row in config.experiments}
        existing_payload = load_json(directory / SELECTED)
        existing = {
            row["experiment_id"]: row
            for row in existing_payload.get("materializations", [])
            if isinstance(row, dict) and row.get("status") == "succeeded"
        }
        records: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for experiment_id in selected_ids:
            if experiment_id not in {row["experiment_id"] for row in ranking}:
                records.append(
                    {"experiment_id": experiment_id, "status": "not_search_successful"}
                )
                continue
            if experiment_id in existing:
                records.append(existing[experiment_id])
                continue
            planned = plan_by_id[experiment_id]
            try:
                pool = self._pool(
                    planned["ordered_candidate_ids"], config.candidate_root
                )
                settings = experiments_by_id[experiment_id].settings(pool.candidate_ids)
                materialized = materialize_blend(
                    pool, settings, blend_root_relative=config.blend_root
                )
                if materialized.blend_id != planned["expected_blend_id"]:
                    raise BlendCampaignRunnerError(
                        "Materialized blend ID differs from frozen plan."
                    )
                record = {
                    "experiment_id": experiment_id,
                    "status": "succeeded",
                    "blend_id": materialized.blend_id,
                    "blend_path": _relative(
                        materialized.blend_dir, self.repository_root
                    ),
                    "canonical_candidate_id": materialized.candidate_package.candidate_id,
                    "canonical_candidate_path": _relative(
                        materialized.candidate_package.package_dir, self.repository_root
                    ),
                    "exploratory": bool(
                        materialized.candidate_package.manifest.get("exploratory")
                    ),
                }
                records.append(record)
                _emit_progress("materialization_succeeded", experiment_id=experiment_id)
            except Exception as error:  # noqa: BLE001
                reason = f"{type(error).__name__}: {error}"
                records.append(
                    {
                        "experiment_id": experiment_id,
                        "status": "failed",
                        "failure_reason": reason,
                    }
                )
                failures.append(
                    {
                        "phase": "materialization",
                        "experiment_id": experiment_id,
                        "reason": reason,
                    }
                )
                _emit_progress(
                    "materialization_failed", experiment_id=experiment_id, reason=reason
                )
        return (
            {
                "schema_version": 1,
                "selection_policy": {
                    "top_k": limit,
                    "explicit_experiment_ids": list(config.materialize_experiment_ids),
                },
                "selected": selected_ids,
                "materializations": records,
            },
            failures,
        )

    def _generate_submissions(
        self,
        config: BlendCampaignConfig,
        directory: Path,
        selected: Mapping[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        materialized = {
            row["experiment_id"]: row
            for row in selected.get("materializations", [])
            if isinstance(row, Mapping) and row.get("status") == "succeeded"
        }
        requested = list(config.submission_experiment_ids) or list(materialized)
        generated: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for experiment_id in requested:
            row = materialized.get(experiment_id)
            if row is None:
                reason = (
                    "Experiment has no successfully materialized canonical candidate."
                )
                generated.append(
                    {
                        "experiment_id": experiment_id,
                        "status": "failed",
                        "failure_reason": reason,
                    }
                )
                failures.append(
                    {
                        "phase": "submission",
                        "experiment_id": experiment_id,
                        "reason": reason,
                    }
                )
                continue
            submission_id = f"{config.submission_id_prefix}_{experiment_id}"
            try:
                result = generate_candidate_submission(
                    str(row["canonical_candidate_id"]),
                    submission_id=submission_id,
                    repository_root=self.repository_root,
                    candidates_root_relative=config.candidate_root,
                    submission_root_relative=config.submission_root,
                    blend_root_relative=config.blend_root,
                )
                generated.append(
                    {"experiment_id": experiment_id, "status": "succeeded", **result}
                )
                _emit_progress(
                    "submission_succeeded",
                    experiment_id=experiment_id,
                    submission_id=submission_id,
                )
            except Exception as error:  # noqa: BLE001
                reason = f"{type(error).__name__}: {error}"
                generated.append(
                    {
                        "experiment_id": experiment_id,
                        "status": "failed",
                        "failure_reason": reason,
                    }
                )
                failures.append(
                    {
                        "phase": "submission",
                        "experiment_id": experiment_id,
                        "reason": reason,
                    }
                )
                _emit_progress(
                    "submission_failed", experiment_id=experiment_id, reason=reason
                )
        return (
            {
                "schema_version": 1,
                "generated": generated,
                "network_access": False,
                "kaggle_upload": False,
            },
            failures,
        )


def rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successful = [row for row in results if row.get("status") == "succeeded"]
    successful.sort(
        key=lambda row: (
            -float(row["honest_mean_balanced_accuracy"]),
            float(row["honest_balanced_accuracy_std"]),
            -float(row["min_repeat_balanced_accuracy"]),
            int(row["active_model_count"]),
            str(row["experiment_id"]),
        )
    )
    return [
        {
            "rank": rank,
            "experiment_id": row["experiment_id"],
            "honest_mean_balanced_accuracy": row["honest_mean_balanced_accuracy"],
            "honest_balanced_accuracy_std": row["honest_balanced_accuracy_std"],
            "min_repeat_balanced_accuracy": row["min_repeat_balanced_accuracy"],
            "active_model_count": row["active_model_count"],
            "exploratory": row["exploratory"],
            "expected_blend_id": row["expected_blend_id"],
        }
        for rank, row in enumerate(successful, start=1)
    ]


def _initial_status(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": plan["campaign_id"],
        "plan_hash": plan["plan_hash"],
        "state": "planned",
        "updated_at_utc": utc_now(),
        "experiments": {
            row["experiment_id"]: {
                "experiment_id": row["experiment_id"],
                "experiment_identity_hash": row["experiment_identity_hash"],
                "candidate_manifest_hashes": row["candidate_manifest_hashes"],
                "state": "pending",
                "attempts": 0,
                "result": None,
                "failure_reason": None,
            }
            for row in plan["experiments"]
        },
    }


def _normalize_result(
    planned: Mapping[str, Any],
    payload: Mapping[str, Any],
    pool: CompatibleCandidateSet,
    runtime: float,
) -> dict[str, Any]:
    if payload.get("blend_id") != planned["expected_blend_id"]:
        raise BlendCampaignRunnerError(
            "Search returned a blend ID different from the frozen plan."
        )
    honest = payload["honest_meta_cv_metrics"]
    pooled = honest["pooled_repeated_held_out_confusion_metrics"]
    full = payload["full_oof_descriptive_metrics"]
    weights = payload["final_deployment_weights"]
    threshold = float(payload["final_deployment_threshold"])
    weight_vector = np.array([float(weights[value]) for value in pool.candidate_ids])
    test_probs = pool.test_matrix @ weight_vector
    return {
        "experiment_id": planned["experiment_id"],
        "ordered_candidate_ids": list(planned["ordered_candidate_ids"]),
        "aliases": list(planned["aliases"]),
        "candidate_manifest_hashes": list(planned["candidate_manifest_hashes"]),
        "strategy": planned["strategy"],
        "optimizer": planned["optimizer"],
        "folds": planned["folds"],
        "repeats": planned["repeats"],
        "seed": planned["seed"],
        "max_active_models": planned["max_active_models"],
        "threshold_policy": dict(planned["threshold_policy"]),
        "expected_blend_id": planned["expected_blend_id"],
        "status": "succeeded",
        "runtime_seconds": float(runtime),
        "honest_mean_balanced_accuracy": float(honest["mean_repeat_balanced_accuracy"]),
        "honest_balanced_accuracy_std": float(honest["std_repeat_balanced_accuracy"]),
        "min_repeat_balanced_accuracy": float(honest["min_repeat_balanced_accuracy"]),
        "max_repeat_balanced_accuracy": float(honest["max_repeat_balanced_accuracy"]),
        "pooled_sensitivity": float(pooled["sensitivity"]),
        "pooled_specificity": float(pooled["specificity"]),
        "descriptive_full_oof_balanced_accuracy": float(full["balanced_accuracy"]),
        "final_weights": dict(weights),
        "final_threshold": threshold,
        "predicted_positive_count": int(np.sum(test_probs >= threshold)),
        "active_model_count": int(np.sum(weight_vector > 1.0e-8)),
        "exploratory": bool(payload["exploratory"]),
        "failure_reason": None,
        "source_result_path": None,
    }


def _failed_result(
    planned: Mapping[str, Any], reason: str, runtime: float
) -> dict[str, Any]:
    row = {
        "experiment_id": planned["experiment_id"],
        "ordered_candidate_ids": list(planned["ordered_candidate_ids"]),
        "aliases": list(planned["aliases"]),
        "candidate_manifest_hashes": list(planned["candidate_manifest_hashes"]),
        "strategy": planned["strategy"],
        "optimizer": planned["optimizer"],
        "folds": planned["folds"],
        "repeats": planned["repeats"],
        "seed": planned["seed"],
        "max_active_models": planned["max_active_models"],
        "threshold_policy": dict(planned["threshold_policy"]),
        "expected_blend_id": planned["expected_blend_id"],
        "status": "failed",
        "runtime_seconds": float(runtime),
        "exploratory": bool(planned["exploratory"]),
        "failure_reason": reason,
        "source_result_path": None,
    }
    for key in (
        "honest_mean_balanced_accuracy",
        "honest_balanced_accuracy_std",
        "min_repeat_balanced_accuracy",
        "max_repeat_balanced_accuracy",
        "pooled_sensitivity",
        "pooled_specificity",
        "descriptive_full_oof_balanced_accuracy",
        "final_weights",
        "final_threshold",
        "predicted_positive_count",
        "active_model_count",
    ):
        row[key] = None
    return row


def _result_rows(
    status: Mapping[str, Any], plan: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for planned in plan["experiments"]:
        result = status["experiments"][planned["experiment_id"]].get("result")
        if isinstance(result, dict):
            rows.append(result)
    return rows


def _search_failures(status: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "phase": "search",
            "experiment_id": experiment_id,
            "reason": cell.get("failure_reason"),
            "attempts": cell.get("attempts", 0),
        }
        for experiment_id, cell in status["experiments"].items()
        if cell.get("state") == "failed"
    ]


def _all_searches_terminal(status: Mapping[str, Any]) -> bool:
    return all(
        cell.get("state") in {"succeeded", "failed"}
        for cell in status["experiments"].values()
    )


def _success_payload(
    config: BlendCampaignConfig, plan: Mapping[str, Any], status: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": config.campaign_id,
        "plan_hash": plan["plan_hash"],
        "status": status["state"],
        "completed_at_utc": utc_now(),
        "network_access": False,
        "kaggle_upload": False,
    }


def _merge_failures(directory: Path, new_failures: list[dict[str, Any]]) -> None:
    payload = load_json(directory / FAILURES)
    existing = list(payload.get("failures") or [])
    keys = {
        (row.get("phase"), row.get("experiment_id"), row.get("reason"))
        for row in existing
    }
    for row in new_failures:
        key = (row.get("phase"), row.get("experiment_id"), row.get("reason"))
        if key not in keys:
            existing.append(row)
            keys.add(key)
    atomic_write_json(directory / FAILURES, {"schema_version": 1, "failures": existing})


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _emit_progress(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, sort_keys=True), flush=True)
    experiment = payload.get("experiment_id")
    suffix = f" [{experiment}]" if experiment else ""
    print(f"STATUS: {event}{suffix}", file=sys.stderr, flush=True)


__all__ = ["BlendCampaignRunner", "BlendCampaignRunnerError", "rank_results"]
