from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

import yaml


SAFE_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")

ROOT_KEYS = {
    "schema_version",
    "promotion",
    "threshold",
    "final_fit",
    "inputs",
    "artifacts",
    "verification",
}
PROMOTION_KEYS = {"id", "approval", "research"}
APPROVAL_KEYS = {"mode", "approved_at_utc", "metric_policy"}
RESEARCH_KEYS = {
    "run_path",
    "run_id",
    "config_path",
    "artifact_manifest_sha256",
    "plan_id",
    "plan_sha256",
    "candidate_id",
    "candidate_sha256",
    "candidate_source_sha256",
    "run_implementation_sha256",
    "loaded_modules_sha256",
    "approved_metrics",
}
METRIC_KEYS = {
    "label",
    "balanced_accuracy_mean",
    "balanced_accuracy_sample_sd",
}
THRESHOLD_KEYS = {
    "strategy",
    "expected_repeats_per_row",
    "metric",
    "minimum",
    "maximum",
    "step",
    "comparison",
    "maximizer_absolute_tolerance",
    "tie_break",
    "constant_probability_fallback",
    "expected_selected_threshold",
    "expected_diagnostic_balanced_accuracy",
    "diagnostic_label",
}
FINAL_FIT_KEYS = {
    "policy",
    "implementation",
    "model_count",
    "boosting_rounds",
    "parameter_tuning",
    "test_probability_aggregation",
}
INPUT_KEYS = {
    "processed_dir",
    "dataset_version",
    "train_features",
    "target",
    "test_features",
    "metadata",
    "sample_submission",
}
FILE_KEYS = {"name", "sha256"}
SAMPLE_KEYS = {"path", "sha256", "columns", "target_column", "expected_rows"}
ARTIFACT_KEYS = {
    "root",
    "submission_filename",
    "persist_encoder",
    "persist_model",
    "persist_probabilities",
    "overwrite",
}
VERIFICATION_KEYS = {
    "reload_encoder",
    "reload_model",
    "encoded_test_comparison",
    "probability_comparison",
    "prediction_comparison",
}


class FinalConfigurationError(ValueError):
    """Raised when final-promotion configuration is not exact and safe."""


@dataclass(frozen=True)
class FinalConfig:
    payload: dict[str, Any]
    source_path: Path
    project_root: Path

    @property
    def promotion_id(self) -> str:
        return str(self.payload["promotion"]["id"])

    @property
    def research_run(self) -> Path:
        return resolve_portable_path(
            self.payload["promotion"]["research"]["run_path"],
            self.project_root,
            "promotion.research.run_path",
        )

    @property
    def research_config(self) -> Path:
        return resolve_portable_path(
            self.payload["promotion"]["research"]["config_path"],
            self.project_root,
            "promotion.research.config_path",
        )

    @property
    def dataset_dir(self) -> Path:
        processed = resolve_portable_path(
            self.payload["inputs"]["processed_dir"],
            self.project_root,
            "inputs.processed_dir",
        )
        return contained_child(
            processed,
            str(self.payload["inputs"]["dataset_version"]),
            "inputs.dataset_version",
        )

    @property
    def sample_submission(self) -> Path:
        return resolve_portable_path(
            self.payload["inputs"]["sample_submission"]["path"],
            self.project_root,
            "inputs.sample_submission.path",
        )

    @property
    def artifact_root(self) -> Path:
        return resolve_portable_path(
            self.payload["artifacts"]["root"],
            self.project_root,
            "artifacts.root",
        )

    def resolved_payload(self) -> dict[str, Any]:
        result = deepcopy(self.payload)
        result["promotion"]["research"]["run_path"] = str(self.research_run)
        result["promotion"]["research"]["config_path"] = str(self.research_config)
        result["inputs"]["processed_dir"] = str(self.dataset_dir.parent)
        result["inputs"]["sample_submission"]["path"] = str(self.sample_submission)
        result["artifacts"]["root"] = str(self.artifact_root)
        return result


def load_final_config(path: Path, *, project_root: Path) -> FinalConfig:
    root = project_root.resolve()
    source = path.resolve()
    if not source.is_file():
        raise FinalConfigurationError(f"Final config does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    root_mapping = _mapping(payload, "root")
    _keys(root_mapping, ROOT_KEYS, "root")
    if _integer(root_mapping["schema_version"], "schema_version") != 1:
        raise FinalConfigurationError("schema_version must be 1.")

    promotion = _section(root_mapping, "promotion")
    _keys(promotion, PROMOTION_KEYS, "promotion")
    _slug(promotion["id"], "promotion.id")
    approval = _section(promotion, "approval")
    _keys(approval, APPROVAL_KEYS, "promotion.approval")
    if approval["mode"] != "manual":
        raise FinalConfigurationError("approval.mode must be manual.")
    if approval["metric_policy"] != "immutable_unbiased_research_evidence":
        raise FinalConfigurationError("approval.metric_policy is invalid.")
    _utc(approval["approved_at_utc"], "promotion.approval.approved_at_utc")

    research = _section(promotion, "research")
    _keys(research, RESEARCH_KEYS, "promotion.research")
    for name in ("run_id", "plan_id", "candidate_id"):
        _slug(research[name], f"promotion.research.{name}")
    for name in (
        "artifact_manifest_sha256",
        "plan_sha256",
        "candidate_sha256",
        "candidate_source_sha256",
        "run_implementation_sha256",
        "loaded_modules_sha256",
    ):
        _sha(research[name], f"promotion.research.{name}")
    resolve_portable_path(research["run_path"], root, "promotion.research.run_path")
    resolve_portable_path(
        research["config_path"], root, "promotion.research.config_path"
    )
    if PurePosixPath(str(research["run_path"])).name != research["run_id"]:
        raise FinalConfigurationError("Research run path basename differs from run_id.")
    approved = _section(research, "approved_metrics")
    _keys(approved, METRIC_KEYS, "promotion.research.approved_metrics")
    if approved["label"] != "unbiased_nested_research_estimate":
        raise FinalConfigurationError("Approved research metric label is invalid.")
    _finite(approved["balanced_accuracy_mean"], "approved mean")
    _finite_nonnegative(approved["balanced_accuracy_sample_sd"], "approved SD")

    threshold = _section(root_mapping, "threshold")
    _keys(threshold, THRESHOLD_KEYS, "threshold")
    expected_threshold = {
        "strategy": "averaged_repeated_outer_oos_grid_ba_v1",
        "expected_repeats_per_row": 5,
        "metric": "balanced_accuracy",
        "comparison": "greater_than_or_equal",
        "tie_break": "median_maximizer_lower_on_even",
        "diagnostic_label": "operational_threshold_selection_diagnostic",
    }
    for name, expected in expected_threshold.items():
        if threshold[name] != expected:
            raise FinalConfigurationError(f"threshold.{name} must be {expected!r}.")
    for name in (
        "minimum",
        "maximum",
        "step",
        "maximizer_absolute_tolerance",
        "constant_probability_fallback",
        "expected_selected_threshold",
        "expected_diagnostic_balanced_accuracy",
    ):
        _finite(threshold[name], f"threshold.{name}")
    if (
        float(threshold["minimum"]) != 0.01
        or float(threshold["maximum"]) != 0.99
        or float(threshold["step"]) != 0.001
        or float(threshold["constant_probability_fallback"]) != 0.5
        or float(threshold["maximizer_absolute_tolerance"]) != 1e-12
    ):
        raise FinalConfigurationError("Threshold numerical policy differs.")

    final_fit = _section(root_mapping, "final_fit")
    _keys(final_fit, FINAL_FIT_KEYS, "final_fit")
    expected_fit = {
        "policy": "single_full_data_model_oof_target_encoding_v1",
        "implementation": "manual_lightgbm_r31",
        "model_count": 1,
        "boosting_rounds": 376,
        "parameter_tuning": False,
        "test_probability_aggregation": "none",
    }
    if dict(final_fit) != expected_fit:
        raise FinalConfigurationError("Final-fit policy differs from the approved one.")

    inputs = _section(root_mapping, "inputs")
    _keys(inputs, INPUT_KEYS, "inputs")
    _portable(inputs["processed_dir"], "inputs.processed_dir")
    _slug(inputs["dataset_version"], "inputs.dataset_version")
    for role in ("train_features", "target", "test_features", "metadata"):
        item = _section(inputs, role)
        _keys(item, FILE_KEYS, f"inputs.{role}")
        _basename(item["name"], f"inputs.{role}.name")
        _sha(item["sha256"], f"inputs.{role}.sha256")
    sample = _section(inputs, "sample_submission")
    _keys(sample, SAMPLE_KEYS, "inputs.sample_submission")
    resolve_portable_path(sample["path"], root, "inputs.sample_submission.path")
    _sha(sample["sha256"], "inputs.sample_submission.sha256")
    if sample["columns"] != ["index", "y"] or sample["target_column"] != "y":
        raise FinalConfigurationError(
            "Sample submission schema must be ['index', 'y']."
        )
    if _integer(sample["expected_rows"], "sample expected_rows") != 2500:
        raise FinalConfigurationError("Sample submission must contain 2500 rows.")

    artifacts = _section(root_mapping, "artifacts")
    _keys(artifacts, ARTIFACT_KEYS, "artifacts")
    resolve_portable_path(artifacts["root"], root, "artifacts.root")
    _basename(artifacts["submission_filename"], "artifacts.submission_filename")
    for name in ("persist_encoder", "persist_model", "persist_probabilities"):
        if _boolean(artifacts[name], f"artifacts.{name}") is not True:
            raise FinalConfigurationError(f"artifacts.{name} must be true.")
    if _boolean(artifacts["overwrite"], "artifacts.overwrite") is not False:
        raise FinalConfigurationError("Artifact overwrite must be false.")

    verification = _section(root_mapping, "verification")
    _keys(verification, VERIFICATION_KEYS, "verification")
    if verification != {
        "reload_encoder": True,
        "reload_model": True,
        "encoded_test_comparison": "exact",
        "probability_comparison": "exact",
        "prediction_comparison": "exact",
    }:
        raise FinalConfigurationError("Reload verification must require exact parity.")

    return FinalConfig(deepcopy(dict(root_mapping)), source, root)


def resolve_portable_path(value: Any, root: Path, label: str) -> Path:
    portable = _portable(value, label)
    resolved = (root / Path(*PurePosixPath(portable).parts)).resolve()
    if resolved != root and root not in resolved.parents:
        raise FinalConfigurationError(f"{label} escapes the repository.")
    return resolved


def contained_child(parent: Path, name: str, label: str) -> Path:
    child = (parent.resolve() / name).resolve()
    if parent.resolve() not in child.parents:
        raise FinalConfigurationError(f"{label} escapes its parent.")
    return child


def _portable(value: Any, label: str) -> str:
    text = _string(value, label)
    if "\\" in text:
        raise FinalConfigurationError(f"{label} must use portable '/' separators.")
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if posix.is_absolute() or windows.is_absolute() or ".." in posix.parts:
        raise FinalConfigurationError(f"{label} must be a contained relative path.")
    if not posix.parts or "." in posix.parts:
        raise FinalConfigurationError(f"{label} is not a normalized portable path.")
    return text


def _basename(value: Any, label: str) -> str:
    text = _string(value, label)
    if PurePosixPath(text).name != text or PureWindowsPath(text).name != text:
        raise FinalConfigurationError(f"{label} must be a plain basename.")
    if text in {".", ".."}:
        raise FinalConfigurationError(f"{label} is unsafe.")
    return text


def _slug(value: Any, label: str) -> str:
    text = _string(value, label)
    if SAFE_SLUG.fullmatch(text) is None:
        raise FinalConfigurationError(f"{label} is not a safe slug.")
    return text


def _sha(value: Any, label: str) -> str:
    text = _string(value, label)
    if SHA256.fullmatch(text) is None:
        raise FinalConfigurationError(f"{label} must be a lowercase SHA-256.")
    return text


def _utc(value: Any, label: str) -> datetime:
    text = _string(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise FinalConfigurationError(f"{label} is not valid ISO-8601.") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise FinalConfigurationError(f"{label} must be timezone-aware UTC.")
    return parsed


def _keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise FinalConfigurationError(
            f"{label} keys differ: expected={sorted(expected)}, actual={sorted(value)}"
        )


def _section(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return _mapping(value.get(name), name)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalConfigurationError(f"{label} must be a mapping.")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise FinalConfigurationError(f"{label} must be a non-empty string.")
    return value


def _integer(value: Any, label: str) -> int:
    if type(value) is not int:
        raise FinalConfigurationError(f"{label} must be an integer.")
    return value


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise FinalConfigurationError(f"{label} must be boolean.")
    return value


def _finite(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise FinalConfigurationError(f"{label} must be finite.")
    return float(value)


def _finite_nonnegative(value: Any, label: str) -> float:
    number = _finite(value, label)
    if number < 0:
        raise FinalConfigurationError(f"{label} must be non-negative.")
    return number
