from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.churn_ml.blend_evaluation_v1 import (
    BLEND_EVALUATION_SCHEMA_VERSION,
    BlendEvaluationError,
    LIGHTGBM_ADAPTER,
    XGBOOST_ADAPTER,
    build_weight_grid,
)
from src.churn_ml.paired_comparison_paths import (
    PairedPathSafetyError,
    resolve_repository_path,
)


REQUIRED_ROOT_KEYS = {
    "schema_version",
    "experiment",
    "components",
    "blend",
    "deployment_parameters",
    "artifacts",
}


@dataclass(frozen=True)
class BlendEvaluationConfig:
    path: Path
    project_root: Path
    payload: dict[str, Any]
    experiment_id: str
    lightgbm_run_dir: Path
    xgboost_run_dir: Path
    lightgbm_adapter_id: str
    xgboost_adapter_id: str
    weight_minimum: float
    weight_maximum: float
    weight_step: float
    maximizer_absolute_tolerance: float
    prefer_closest_to: float
    forbid_pooled_oof_fallback: bool
    sensitivity_weight_deltas: tuple[float, ...]
    artifact_root: Path
    evaluation_id: str

    def weight_grid(self) -> Any:
        return build_weight_grid(
            minimum=self.weight_minimum,
            maximum=self.weight_maximum,
            step=self.weight_step,
        )

    def resolved_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.payload["schema_version"],
            "experiment": dict(self.payload["experiment"]),
            "components": {
                "lightgbm": {
                    "role": "lightgbm",
                    "adapter_id": self.lightgbm_adapter_id,
                    "run_dir": self.lightgbm_run_dir.relative_to(
                        self.project_root
                    ).as_posix(),
                },
                "xgboost": {
                    "role": "xgboost",
                    "adapter_id": self.xgboost_adapter_id,
                    "run_dir": self.xgboost_run_dir.relative_to(
                        self.project_root
                    ).as_posix(),
                },
            },
            "blend": dict(self.payload["blend"]),
            "deployment_parameters": dict(self.payload["deployment_parameters"]),
            "artifacts": {
                "root": self.artifact_root.relative_to(self.project_root).as_posix(),
                "evaluation_id": self.evaluation_id,
            },
        }


def load_blend_evaluation_config(
    path: Path,
    *,
    project_root: Path,
) -> BlendEvaluationConfig:
    try:
        config_path = resolve_repository_path(
            path,
            project_root=project_root,
            role="blend_evaluation_config",
            field_path="paths.config",
            must_exist=True,
            require_directory=False,
        )
    except PairedPathSafetyError as error:
        raise BlendEvaluationError(
            error.detail,
            reason_code=error.reason_code,
            field_path=error.field_path,
        ) from error
    payload = _read_yaml(config_path)
    if set(payload) != REQUIRED_ROOT_KEYS:
        raise BlendEvaluationError(
            "Blend evaluation config root keys differ.",
            field_path="config",
        )
    if payload["schema_version"] != BLEND_EVALUATION_SCHEMA_VERSION:
        raise BlendEvaluationError(
            "schema_version must be 1.",
            field_path="schema_version",
        )
    experiment = _mapping(payload["experiment"], "experiment")
    if set(experiment) != {"id", "name"}:
        raise BlendEvaluationError(
            "experiment keys must be exactly {id, name}.",
            field_path="experiment",
        )
    experiment_id = _non_empty_string(experiment["id"], "experiment.id")
    _non_empty_string(experiment["name"], "experiment.name")

    components = _mapping(payload["components"], "components")
    if set(components) != {"lightgbm", "xgboost"}:
        raise BlendEvaluationError(
            "components must contain exactly lightgbm and xgboost.",
            field_path="components",
        )
    lightgbm = _component(components["lightgbm"], "lightgbm", LIGHTGBM_ADAPTER)
    xgboost = _component(components["xgboost"], "xgboost", XGBOOST_ADAPTER)
    lightgbm_run_dir = _resolve_run_dir(
        lightgbm["run_dir"],
        project_root=project_root,
        role="lightgbm_run",
    )
    xgboost_run_dir = _resolve_run_dir(
        xgboost["run_dir"],
        project_root=project_root,
        role="xgboost_run",
    )

    blend = _mapping(payload["blend"], "blend")
    if set(blend) != {
        "method",
        "lightgbm_weight_grid",
        "weight_selection",
        "cross_fitting",
    }:
        raise BlendEvaluationError("blend keys differ.", field_path="blend")
    if blend["method"] != "linear_probability_mean":
        raise BlendEvaluationError(
            "blend.method must be linear_probability_mean.",
            field_path="blend.method",
        )
    grid = _mapping(blend["lightgbm_weight_grid"], "blend.lightgbm_weight_grid")
    if set(grid) != {"minimum", "maximum", "step"}:
        raise BlendEvaluationError(
            "lightgbm_weight_grid keys differ.",
            field_path="blend.lightgbm_weight_grid",
        )
    weight_minimum = _finite_float(
        grid["minimum"], "blend.lightgbm_weight_grid.minimum"
    )
    weight_maximum = _finite_float(
        grid["maximum"], "blend.lightgbm_weight_grid.maximum"
    )
    weight_step = _finite_float(grid["step"], "blend.lightgbm_weight_grid.step")
    build_weight_grid(
        minimum=weight_minimum,
        maximum=weight_maximum,
        step=weight_step,
    )

    selection = _mapping(blend["weight_selection"], "blend.weight_selection")
    if set(selection) != {
        "metric",
        "maximizer_absolute_tolerance",
        "prefer_closest_to",
        "secondary_tie_break",
    }:
        raise BlendEvaluationError(
            "weight_selection keys differ.",
            field_path="blend.weight_selection",
        )
    if selection["metric"] != "balanced_accuracy":
        raise BlendEvaluationError(
            "weight_selection.metric must be balanced_accuracy.",
            field_path="blend.weight_selection.metric",
        )
    if selection["secondary_tie_break"] != "lower_lightgbm_weight":
        raise BlendEvaluationError(
            "secondary_tie_break must be lower_lightgbm_weight.",
            field_path="blend.weight_selection.secondary_tie_break",
        )
    maximizer_absolute_tolerance = _finite_float(
        selection["maximizer_absolute_tolerance"],
        "blend.weight_selection.maximizer_absolute_tolerance",
    )
    prefer_closest_to = _finite_float(
        selection["prefer_closest_to"],
        "blend.weight_selection.prefer_closest_to",
    )

    cross_fitting = _mapping(blend["cross_fitting"], "blend.cross_fitting")
    if set(cross_fitting) != {"protocol", "forbid_pooled_oof_fallback"}:
        raise BlendEvaluationError(
            "cross_fitting keys differ.",
            field_path="blend.cross_fitting",
        )
    if cross_fitting["protocol"] != "leave_one_outer_fold_out_within_repeat":
        raise BlendEvaluationError(
            "Unsupported cross-fitting protocol.",
            field_path="blend.cross_fitting.protocol",
        )
    if cross_fitting["forbid_pooled_oof_fallback"] is not True:
        raise BlendEvaluationError(
            "forbid_pooled_oof_fallback must be true.",
            reason_code="OPTIMISTIC_FALLBACK_FORBIDDEN",
            field_path="blend.cross_fitting.forbid_pooled_oof_fallback",
        )

    deployment = _mapping(payload["deployment_parameters"], "deployment_parameters")
    if set(deployment) != {
        "weight_aggregation",
        "threshold_aggregation",
        "sensitivity_weight_deltas",
    }:
        raise BlendEvaluationError(
            "deployment_parameters keys differ.",
            field_path="deployment_parameters",
        )
    if deployment["weight_aggregation"] != "median":
        raise BlendEvaluationError(
            "weight_aggregation must be median.",
            field_path="deployment_parameters.weight_aggregation",
        )
    if deployment["threshold_aggregation"] != "median":
        raise BlendEvaluationError(
            "threshold_aggregation must be median.",
            field_path="deployment_parameters.threshold_aggregation",
        )
    deltas_raw = deployment["sensitivity_weight_deltas"]
    if not isinstance(deltas_raw, list) or not deltas_raw:
        raise BlendEvaluationError(
            "sensitivity_weight_deltas must be a non-empty list.",
            field_path="deployment_parameters.sensitivity_weight_deltas",
        )
    deltas = tuple(
        _finite_float(value, "deployment_parameters.sensitivity_weight_deltas")
        for value in deltas_raw
    )
    if any(delta <= 0.0 for delta in deltas):
        raise BlendEvaluationError(
            "sensitivity_weight_deltas must be positive.",
            field_path="deployment_parameters.sensitivity_weight_deltas",
        )

    artifacts = _mapping(payload["artifacts"], "artifacts")
    if set(artifacts) != {"root", "evaluation_id"}:
        raise BlendEvaluationError("artifacts keys differ.", field_path="artifacts")
    evaluation_id = _non_empty_string(
        artifacts["evaluation_id"], "artifacts.evaluation_id"
    )
    try:
        artifact_root = resolve_repository_path(
            Path(str(artifacts["root"])),
            project_root=project_root,
            role="blend_evaluation_output",
            field_path="artifacts.root",
            must_exist=False,
            require_directory=True,
        )
    except PairedPathSafetyError as error:
        raise BlendEvaluationError(
            error.detail,
            reason_code=error.reason_code,
            field_path=error.field_path,
        ) from error
    expected_root = (project_root / "artifacts" / "blend_evaluations").resolve()
    if artifact_root != expected_root:
        raise BlendEvaluationError(
            "artifacts.root must be exactly artifacts/blend_evaluations.",
            field_path="artifacts.root",
        )

    return BlendEvaluationConfig(
        path=config_path,
        project_root=project_root.resolve(),
        payload=payload,
        experiment_id=experiment_id,
        lightgbm_run_dir=lightgbm_run_dir,
        xgboost_run_dir=xgboost_run_dir,
        lightgbm_adapter_id=str(lightgbm["adapter_id"]),
        xgboost_adapter_id=str(xgboost["adapter_id"]),
        weight_minimum=weight_minimum,
        weight_maximum=weight_maximum,
        weight_step=weight_step,
        maximizer_absolute_tolerance=maximizer_absolute_tolerance,
        prefer_closest_to=prefer_closest_to,
        forbid_pooled_oof_fallback=True,
        sensitivity_weight_deltas=deltas,
        artifact_root=artifact_root,
        evaluation_id=evaluation_id,
    )


def _component(
    value: Any,
    role: str,
    expected_adapter: str,
) -> dict[str, Any]:
    component = _mapping(value, f"components.{role}")
    if set(component) != {"role", "adapter_id", "run_dir"}:
        raise BlendEvaluationError(
            f"components.{role} keys differ.",
            field_path=f"components.{role}",
        )
    if component["role"] != role:
        raise BlendEvaluationError(
            f"components.{role}.role must equal {role!r}.",
            field_path=f"components.{role}.role",
        )
    adapter_id = _non_empty_string(
        component["adapter_id"], f"components.{role}.adapter_id"
    )
    if adapter_id != expected_adapter:
        raise BlendEvaluationError(
            f"components.{role}.adapter_id must be {expected_adapter}.",
            reason_code="ADAPTER_ID_MISMATCH",
            field_path=f"components.{role}.adapter_id",
        )
    _non_empty_string(component["run_dir"], f"components.{role}.run_dir")
    return component


def _resolve_run_dir(value: Any, *, project_root: Path, role: str) -> Path:
    try:
        return resolve_repository_path(
            Path(str(value)),
            project_root=project_root,
            role=role,
            field_path=f"components.{role}.run_dir",
            must_exist=True,
            require_directory=True,
        )
    except PairedPathSafetyError as error:
        raise BlendEvaluationError(
            error.detail,
            reason_code=error.reason_code,
            field_path=error.field_path,
        ) from error


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BlendEvaluationError(f"{label} must be a mapping.", field_path=label)
    return dict(value)


def _non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BlendEvaluationError(
            f"{label} must be a non-empty string.",
            field_path=label,
        )
    return value


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BlendEvaluationError(
            f"{label} must be a finite number.",
            field_path=label,
        )
    result = float(value)
    if not math.isfinite(result):
        raise BlendEvaluationError(
            f"{label} must be a finite number.",
            field_path=label,
        )
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise BlendEvaluationError(
            "YAML root must be a mapping.",
            field_path=str(path),
        )
    return dict(payload)
