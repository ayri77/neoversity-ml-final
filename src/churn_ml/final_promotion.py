from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.churn_ml.final_config import FinalConfig
from src.churn_ml.research_artifact_validation import validate_artifact_manifest
from src.churn_ml.research_config import ResearchConfig, load_research_config
from src.churn_ml.research_data import ResearchTrainingData, load_research_training_data
from src.churn_ml.research_protocol import (
    build_candidate_contract_identity,
    build_evaluation_assignments,
    build_evaluation_plan_identity,
    canonical_sha256,
    select_balanced_accuracy_threshold,
)
from src.churn_ml.research_provenance import (
    candidate_source_provenance,
    run_implementation_provenance,
)
from src.churn_ml.research_runtime_provenance import (
    collect_research_environment_versions,
)


class PromotionVerificationError(RuntimeError):
    """Raised when the explicitly approved research evidence is not promotable."""


@dataclass(frozen=True)
class PromotionEvidence:
    research_config: ResearchConfig
    training_data: ResearchTrainingData
    run_root: Path
    research_manifest_hash: str
    plan_identity: dict[str, Any]
    candidate_identity: dict[str, Any]
    research_run_identity: dict[str, Any]
    loaded_modules: dict[str, Any]
    research_metadata: dict[str, Any]
    research_status: dict[str, Any]
    research_metrics: dict[str, Any]


@dataclass(frozen=True)
class OperationalThreshold:
    averaged_oos: pd.DataFrame
    threshold_curve: pd.DataFrame
    selected_threshold: float
    diagnostic_balanced_accuracy: float
    averaged_oos_sha256: str
    selection_record: dict[str, Any]


def verify_approved_research_run(config: FinalConfig) -> PromotionEvidence:
    approved = config.payload["promotion"]["research"]
    root = config.research_run
    if root.name != approved["run_id"] or not root.is_dir():
        raise PromotionVerificationError("Approved research run path or ID is invalid.")
    if not (root / "_SUCCESS").is_file():
        raise PromotionVerificationError("Approved research run lacks _SUCCESS.")
    if (root / "_FAILED").exists():
        raise PromotionVerificationError("Approved research run contains _FAILED.")

    manifest = _read_json(root / "artifact_manifest.json")
    validate_artifact_manifest(root, manifest)
    if manifest.get("manifest_sha256") != approved["artifact_manifest_sha256"]:
        raise PromotionVerificationError("Research artifact manifest hash differs.")
    success = _read_json(root / "_SUCCESS")
    if success != {
        "schema_version": 1,
        "artifact_manifest_sha256": approved["artifact_manifest_sha256"],
    }:
        raise PromotionVerificationError(
            "Research _SUCCESS does not link the manifest."
        )

    metadata = _read_json(root / "run_metadata.json")
    status = _read_json(root / "execution_status.json")
    _validate_terminal_metadata(metadata, status, approved)

    research_config = load_research_config(
        config.research_config,
        project_root=config.project_root,
    )
    if (
        research_config.plan_id != approved["plan_id"]
        or research_config.candidate_id != approved["candidate_id"]
    ):
        raise PromotionVerificationError("Tracked research config identity differs.")
    training = load_research_training_data(research_config)
    assignments = build_evaluation_assignments(
        training.y,
        research_config.plan_payload,
    )
    plan_identity, plan_hash = build_evaluation_plan_identity(
        research_config.plan_payload,
        training.fingerprints,
        assignments,
    )
    if plan_hash != approved["plan_sha256"]:
        raise PromotionVerificationError("Current source does not reproduce plan hash.")
    _validate_identity(root / "evaluation_plan.json", plan_identity, plan_hash)

    environment = collect_research_environment_versions()
    candidate_sources, candidate_source_hash = candidate_source_provenance(
        config.project_root
    )
    if candidate_source_hash != approved["candidate_source_sha256"]:
        raise PromotionVerificationError(
            "Current source does not reproduce candidate-source hash."
        )
    candidate_identity, candidate_hash = build_candidate_contract_identity(
        research_config.candidate_contract,
        feature_schema=training.feature_schema.to_dict(),
        source_provenance={
            **candidate_sources,
            "manifest_sha256": candidate_source_hash,
        },
        runtime_dependencies={
            name: environment[name]
            for name in (
                "numpy",
                "pandas",
                "scikit_learn",
                "lightgbm",
                "pyarrow",
            )
        },
    )
    if candidate_hash != approved["candidate_sha256"]:
        raise PromotionVerificationError(
            "Current source/runtime does not reproduce candidate hash."
        )
    _validate_identity(
        root / "candidate_contract.json", candidate_identity, candidate_hash
    )

    source = research_config.candidate_contract["source"]
    run_identity, run_hash = run_implementation_provenance(
        config.project_root,
        config_paths={
            "research_run": research_config.source_path,
            "evaluation_plan": research_config.plan_path,
            "historical_candidate": Path(str(source["historical_config_path"])),
            "baseline_manifest": Path(str(source["manifest_path"])),
        },
    )
    if run_hash != approved["run_implementation_sha256"]:
        raise PromotionVerificationError(
            "Current committed source does not reproduce research implementation."
        )
    _validate_identity(root / "run_implementation.json", run_identity, run_hash)

    loaded = _read_json(root / "loaded_modules.json")
    _validate_stored_loaded_modules(
        loaded,
        approved["loaded_modules_sha256"],
        config.project_root,
    )

    outer = pd.read_parquet(root / "predictions" / "outer_validation.parquet")
    selected = pd.read_csv(root / "thresholds" / "selected_thresholds.csv")
    repeats = pd.read_csv(root / "metrics" / "repeats.csv")
    aggregate = _read_json(root / "metrics" / "aggregate.json")
    _validate_complete_outputs(outer, selected, repeats, aggregate, approved)

    return PromotionEvidence(
        research_config=research_config,
        training_data=training,
        run_root=root,
        research_manifest_hash=str(manifest["manifest_sha256"]),
        plan_identity=_read_json(root / "evaluation_plan.json"),
        candidate_identity=_read_json(root / "candidate_contract.json"),
        research_run_identity=_read_json(root / "run_implementation.json"),
        loaded_modules=loaded,
        research_metadata=metadata,
        research_status=status,
        research_metrics=aggregate,
    )


def derive_operational_threshold(
    config: FinalConfig,
    promotion: PromotionEvidence,
) -> OperationalThreshold:
    source = pd.read_parquet(
        promotion.run_root / "predictions" / "outer_validation.parquet"
    )
    required = {
        "repeat",
        "row_position",
        "target",
        "probability",
    }
    if not required.issubset(source.columns):
        raise PromotionVerificationError("Outer predictions lack threshold inputs.")
    source = source.sort_values(["row_position", "repeat"], ignore_index=True)
    if source.duplicated(["repeat", "row_position"]).any():
        raise PromotionVerificationError("Outer predictions contain duplicate rows.")
    probabilities = source["probability"].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise PromotionVerificationError("Outer probabilities are invalid.")
    if set(source["target"].unique().tolist()) != {0, 1}:
        raise PromotionVerificationError("Threshold target must contain both classes.")

    expected_repeats = int(config.payload["threshold"]["expected_repeats_per_row"])
    records: list[dict[str, Any]] = []
    for row_position, group in source.groupby("row_position", sort=True):
        if len(group) != expected_repeats:
            raise PromotionVerificationError(
                f"Row {row_position} does not have {expected_repeats} predictions."
            )
        if group["repeat"].tolist() != list(range(1, expected_repeats + 1)):
            raise PromotionVerificationError(
                f"Row {row_position} has incomplete repeat identities."
            )
        targets = group["target"].to_numpy()
        if np.unique(targets).size != 1:
            raise PromotionVerificationError(
                f"Row {row_position} has inconsistent targets."
            )
        values = group["probability"].to_numpy(dtype=np.float64)
        records.append(
            {
                "row_position": int(row_position),
                "target": int(targets[0]),
                "repeat_count": expected_repeats,
                "probability": float(np.mean(values, dtype=np.float64)),
            }
        )
    averaged = pd.DataFrame.from_records(records)
    expected_rows = int(
        promotion.research_config.plan_payload["dataset"]["expected_rows"]
    )
    if len(averaged) != expected_rows or averaged["row_position"].tolist() != list(
        range(expected_rows)
    ):
        raise PromotionVerificationError("Averaged OOS rows are missing or reordered.")

    policy = {
        name: config.payload["threshold"][name]
        for name in (
            "metric",
            "minimum",
            "maximum",
            "step",
            "comparison",
            "maximizer_absolute_tolerance",
            "tie_break",
            "constant_probability_fallback",
        )
    }
    policy["id"] = "grid_balanced_accuracy_v1"
    result = select_balanced_accuracy_threshold(
        averaged["target"].to_numpy(),
        averaged["probability"].to_numpy(dtype=np.float64),
        policy,
    )
    expected_threshold = float(
        config.payload["threshold"]["expected_selected_threshold"]
    )
    expected_score = float(
        config.payload["threshold"]["expected_diagnostic_balanced_accuracy"]
    )
    if result.threshold != expected_threshold or not math.isclose(
        result.balanced_accuracy,
        expected_score,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise PromotionVerificationError(
            "Derived operational threshold or diagnostic differs from approval."
        )
    digest = _averaged_oos_sha256(averaged)
    record = {
        "strategy": config.payload["threshold"]["strategy"],
        "diagnostic_label": config.payload["threshold"]["diagnostic_label"],
        "selected_threshold": result.threshold,
        "diagnostic_balanced_accuracy": result.balanced_accuracy,
        "comparison": "greater_than_or_equal",
        "input_row_count": len(averaged),
        "probabilities_per_row": expected_repeats,
        "averaged_oos_sha256": digest,
        "status": result.status,
        "degenerate": result.degenerate,
        "prohibited_claims": [
            "unbiased_estimate",
            "test_accuracy",
            "expected_kaggle_score",
            "replacement_for_research_metrics",
        ],
    }
    return OperationalThreshold(
        averaged_oos=averaged,
        threshold_curve=result.scores,
        selected_threshold=result.threshold,
        diagnostic_balanced_accuracy=result.balanced_accuracy,
        averaged_oos_sha256=digest,
        selection_record=record,
    )


def _validate_terminal_metadata(
    metadata: Mapping[str, Any],
    status: Mapping[str, Any],
    approved: Mapping[str, Any],
) -> None:
    required_metadata = {
        "run_id",
        "status",
        "plan_id",
        "candidate_id",
        "plan_sha256",
        "candidate_sha256",
        "candidate_source_manifest_sha256",
        "run_implementation_sha256",
        "loaded_module_sha256",
        "competition_assets_accessed",
        "started_at_utc",
        "finished_at_utc",
        "record_counts",
    }
    if not required_metadata.issubset(metadata):
        raise PromotionVerificationError("Research metadata is incomplete.")
    expected_values = {
        "run_id": approved["run_id"],
        "status": "completed",
        "plan_id": approved["plan_id"],
        "candidate_id": approved["candidate_id"],
        "plan_sha256": approved["plan_sha256"],
        "candidate_sha256": approved["candidate_sha256"],
        "candidate_source_manifest_sha256": approved["candidate_source_sha256"],
        "run_implementation_sha256": approved["run_implementation_sha256"],
        "loaded_module_sha256": approved["loaded_modules_sha256"],
        "competition_assets_accessed": False,
    }
    for name, expected in expected_values.items():
        if metadata.get(name) != expected:
            raise PromotionVerificationError(f"Research metadata {name} differs.")
    for name in ("started_at_utc", "finished_at_utc"):
        _parse_utc(metadata[name], f"research metadata {name}")
    if _parse_utc(metadata["finished_at_utc"], "finished") < _parse_utc(
        metadata["started_at_utc"], "started"
    ):
        raise PromotionVerificationError("Research timestamps are reversed.")

    expected_status = {
        "run_id": approved["run_id"],
        "status": "completed",
        "completed_outer_folds": 25,
        "expected_outer_folds": 25,
        "failure": None,
    }
    for name, expected in expected_status.items():
        if status.get(name) != expected:
            raise PromotionVerificationError(f"Research status {name} differs.")
    for name in ("started_at_utc", "finished_at_utc"):
        _parse_utc(status.get(name), f"research status {name}")

    expected_counts = {
        "repeat_count": 5,
        "outer_fold_count": 25,
        "outer_prediction_count": 50000,
        "selected_threshold_count": 25,
    }
    counts = metadata["record_counts"]
    for name, expected in expected_counts.items():
        if counts.get(name) != expected:
            raise PromotionVerificationError(f"Research record count {name} differs.")


def _validate_complete_outputs(
    outer: pd.DataFrame,
    selected: pd.DataFrame,
    repeats: pd.DataFrame,
    aggregate: Mapping[str, Any],
    approved: Mapping[str, Any],
) -> None:
    if len(outer) != 50000 or outer.duplicated(["repeat", "row_position"]).any():
        raise PromotionVerificationError("Research outer predictions are incomplete.")
    if sorted(outer["repeat"].unique().tolist()) != [1, 2, 3, 4, 5]:
        raise PromotionVerificationError("Research repeat identities are incomplete.")
    for repeat, group in outer.groupby("repeat"):
        if (
            len(group) != 10000
            or group["row_position"].nunique() != 10000
            or sorted(group["row_position"].tolist()) != list(range(10000))
        ):
            raise PromotionVerificationError(
                f"Research repeat {repeat} lacks complete rows."
            )
    fold_pairs = outer[["repeat", "outer_fold"]].drop_duplicates()
    if len(fold_pairs) != 25:
        raise PromotionVerificationError("Research does not contain 25 outer folds.")
    if (
        len(selected) != 25
        or selected.duplicated(["repeat", "outer_fold"]).any()
        or len(repeats) != 5
    ):
        raise PromotionVerificationError("Research threshold/repeat outputs differ.")
    metric = aggregate.get("metrics", {}).get("balanced_accuracy", {})
    expected = approved["approved_metrics"]
    if (
        aggregate.get("repeat_count") != 5
        or metric.get("mean") != expected["balanced_accuracy_mean"]
        or metric.get("sample_standard_deviation")
        != expected["balanced_accuracy_sample_sd"]
    ):
        raise PromotionVerificationError("Approved research metric summary differs.")


def _validate_identity(path: Path, canonical: dict[str, Any], digest: str) -> None:
    payload = _read_json(path)
    if payload != {"sha256": digest, "canonical": canonical}:
        raise PromotionVerificationError(f"Research identity differs: {path.name}")


def _validate_stored_loaded_modules(
    payload: Mapping[str, Any],
    expected_hash: str,
    project_root: Path,
) -> None:
    if set(payload) != {"sha256", "canonical", "observations"}:
        raise PromotionVerificationError("Loaded-module artifact keys differ.")
    canonical = payload["canonical"]
    if (
        payload["sha256"] != expected_hash
        or canonical_sha256(canonical) != expected_hash
    ):
        raise PromotionVerificationError("Loaded-module hash differs.")
    modules = canonical.get("modules")
    if not isinstance(modules, list) or not modules:
        raise PromotionVerificationError("Loaded-module evidence is empty.")
    root = project_root.resolve()
    for record in modules:
        if (
            not isinstance(record, Mapping)
            or record.get("match") is not True
            or record.get("expected_repository_relative_path")
            != record.get("actual_repository_relative_path")
            or record.get("loaded_source_sha256")
            != record.get("expected_source_sha256")
        ):
            raise PromotionVerificationError("Loaded-module record is invalid.")
        relative = record["actual_repository_relative_path"]
        if (
            not isinstance(relative, str)
            or "\\" in relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise PromotionVerificationError("Loaded-module path is not portable.")
        source = (root / Path(*relative.split("/"))).resolve()
        if root not in source.parents or not source.is_file():
            raise PromotionVerificationError("Loaded-module source is unavailable.")
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != record["loaded_source_sha256"]:
            raise PromotionVerificationError(
                f"Current loaded-module source differs: {relative}"
            )


def _averaged_oos_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(b"averaged_repeated_outer_oos_v1")
    for name, dtype in (
        ("row_position", "<i8"),
        ("target", "<i8"),
        ("repeat_count", "<i8"),
        ("probability", "<f8"),
    ):
        digest.update(name.encode("utf-8"))
        digest.update(frame[name].to_numpy(dtype=dtype, copy=True).tobytes(order="C"))
    return digest.hexdigest()


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise PromotionVerificationError(f"{label} is missing UTC.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise PromotionVerificationError(f"{label} is invalid.") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise PromotionVerificationError(f"{label} is not timezone-aware UTC.")
    return parsed


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PromotionVerificationError(
            f"Cannot read JSON artifact: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise PromotionVerificationError(f"JSON artifact is not an object: {path}")
    return payload
