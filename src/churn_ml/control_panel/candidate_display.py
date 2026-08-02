"""Central human-readable labels for prediction candidates.

Pure presentation helpers. Candidate artifacts and backend contracts are never
mutated. Technical IDs remain available for provenance and selection identity.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.churn_ml.prediction_candidates.contract_v1 import (
    CANDIDATE_ROOT_RELATIVE,
    MANIFEST_FILENAME,
    SOURCE_METADATA_FILENAME,
    SUCCESS_FILENAME,
)


BLEND_SOURCE_KIND = "canonical_probability_blend_v1"
AUTOGLUON_SOURCE_KIND = "autogluon_standalone_v1"

_SOURCE_KIND_LABELS = {
    AUTOGLUON_SOURCE_KIND: "AutoGluon",
    BLEND_SOURCE_KIND: "Canonical blend",
}

_BAG_SUFFIX = re.compile(r"_BAG_L\d+$")
_SHORT_MODEL = re.compile(
    r"^(?P<family>LightGBMPrep|NeuralNetTorch|RealTabPFN-v2|RealTabPFN|"
    r"WeightedEnsemble|ExtraTreesGini|ExtraTrees)"
    r"(?:_(?P<tag>r?\d+|L\d+))?",
    re.IGNORECASE,
)
_WEIGHT_ALIASES = {
    "LightGBMPrep": "LGB",
    "NeuralNetTorch": "NNT",
    "RealTabPFN-v2": "RealTabPFN",
    "RealTabPFN": "RealTabPFN",
    "WeightedEnsemble": "WE",
    "ExtraTreesGini": "ExtraTrees",
    "ExtraTrees": "ExtraTrees",
}


@dataclass(frozen=True)
class CandidateDisplay:
    candidate_id: str
    primary_label: str
    compact_label: str
    model_name: str
    source_label: str
    dataset_label: str
    dataset_id: str
    source_kind: str
    exploratory: bool
    missing_reason: str | None = None


def source_kind_display_label(source_kind: str) -> str:
    kind = str(source_kind or "").strip()
    if not kind:
        return "Unknown source"
    return _SOURCE_KIND_LABELS.get(kind, kind.replace("_", " ").title())


def dataset_display_label(dataset_id: str | None) -> str:
    raw = str(dataset_id or "").strip()
    if not raw:
        return "unknown dataset"
    known = {
        "v5_joint_missingness_pattern": "v5 joint missingness",
        "v6_compact_missingness_indicators": "v6 compact missingness",
    }
    if raw in known:
        return known[raw]
    if raw.startswith("v") and "_" in raw:
        version, rest = raw.split("_", 1)
        words = rest.replace("_", " ").strip()
        # Drop trailing generic suffixes for compact readability.
        for suffix in (" pattern", " indicators"):
            if words.endswith(suffix):
                words = words[: -len(suffix)].strip()
        return f"{version} {words}".strip()
    return raw.replace("_", " ")


def compact_model_name(model_name: str) -> str:
    text = str(model_name or "").strip()
    if not text:
        return "Unknown model"
    return _BAG_SUFFIX.sub("", text)


def ultra_short_model_label(model_name: str, *, dataset_id: str | None = None) -> str:
    """History-style abbreviations such as ``LGB r31`` or ``v5 WE``."""
    text = str(model_name or "model")
    match = _SHORT_MODEL.match(text)
    if not match:
        short = compact_model_name(text)
    else:
        family = match.group("family")
        tag = match.group("tag")
        short = _WEIGHT_ALIASES.get(family, family)
        if tag:
            short = f"{short} {tag}"
    if dataset_id and str(dataset_id).startswith("v") and "WE" in short:
        return f"{str(dataset_id).split('_', 1)[0]} {short}"
    return short


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _metadata_from_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    source_metadata = payload.get("source_metadata")
    if not isinstance(source_metadata, Mapping):
        source_metadata = {}
    settings = payload.get("settings")
    if not isinstance(settings, Mapping):
        settings = {}
    candidate_id = _safe_str(
        payload.get("candidate_id") or payload.get("canonical_candidate_id")
    )
    parent_ids = (
        payload.get("parent_candidate_ids")
        or source_metadata.get("parent_candidate_ids")
        or payload.get("candidate_ids")
        or []
    )
    if not isinstance(parent_ids, Sequence) or isinstance(parent_ids, (str, bytes)):
        parent_ids = []
    deployment = payload.get("deployment")
    deployment_weights = (
        deployment.get("weights") if isinstance(deployment, Mapping) else None
    )
    weights = (
        payload.get("final_deployment_weights")
        or source_metadata.get("final_deployment_weights")
        or deployment_weights
        or source_metadata.get("final_weights")
        or {}
    )
    if not isinstance(weights, Mapping):
        weights = {}
    strategy = _safe_str(
        settings.get("strategy")
        or payload.get("strategy")
        or source_metadata.get("strategy")
    )
    optimizer = _safe_str(
        settings.get("optimizer_backend")
        or payload.get("optimizer_backend")
        or source_metadata.get("optimizer_backend")
    )
    return {
        "candidate_id": candidate_id,
        "source_model_name": _safe_str(
            payload.get("source_model_name") or payload.get("model")
        ),
        "source_kind": _safe_str(payload.get("source_kind")),
        "dataset_id": _safe_str(payload.get("dataset_id")),
        "exploratory": bool(payload.get("exploratory", False)),
        "parent_candidate_ids": [str(item) for item in parent_ids],
        "final_deployment_weights": {
            str(key): value for key, value in weights.items()
        },
        "strategy": strategy,
        "optimizer_backend": optimizer,
        "blend_id": _safe_str(
            payload.get("blend_id") or source_metadata.get("blend_id")
        ),
    }


def load_candidate_metadata(
    candidate_id: str,
    *,
    repository_root: Path | str,
    candidates_root_relative: str | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    relative = (candidates_root_relative or CANDIDATE_ROOT_RELATIVE).replace("\\", "/")
    package_dir = root / relative / candidate_id
    manifest_path = package_dir / MANIFEST_FILENAME
    metadata_path = package_dir / SOURCE_METADATA_FILENAME
    if not manifest_path.is_file():
        return {
            "candidate_id": candidate_id,
            "source_model_name": "",
            "source_kind": "",
            "dataset_id": "",
            "exploratory": False,
            "parent_candidate_ids": [],
            "final_deployment_weights": {},
            "strategy": "",
            "optimizer_backend": "",
            "blend_id": "",
            "missing_reason": f"Missing {MANIFEST_FILENAME}",
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return {
            "candidate_id": candidate_id,
            "source_model_name": "",
            "source_kind": "",
            "dataset_id": "",
            "exploratory": False,
            "parent_candidate_ids": [],
            "final_deployment_weights": {},
            "strategy": "",
            "optimizer_backend": "",
            "blend_id": "",
            "missing_reason": f"Unreadable manifest: {error}",
        }
    source_metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                source_metadata = loaded
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            source_metadata = {}
    merged = dict(manifest)
    merged["source_metadata"] = source_metadata
    merged["candidate_id"] = candidate_id
    meta = _metadata_from_mapping(merged)
    meta["source_metadata"] = source_metadata
    meta["parent_dataset_id"] = manifest.get("parent_dataset_id")
    meta["target_dependency"] = manifest.get("target_dependency")
    for key in ("final_deployment_threshold", "final_threshold"):
        if key in source_metadata:
            meta[key] = source_metadata.get(key)
    if not (package_dir / SUCCESS_FILENAME).is_file():
        meta["missing_reason"] = f"Missing {SUCCESS_FILENAME}"
    return meta


def _blend_optimizer_label(strategy: str, optimizer: str, model_name: str) -> str:
    if strategy == "optimized" and optimizer == "native":
        return "Native blend"
    if strategy == "optimized" and optimizer == "optuna":
        return "Optuna blend"
    if strategy == "equal":
        return "Equal-weight blend"
    if strategy == "manual":
        return "Manual blend"
    if model_name.startswith("blend_"):
        parts = model_name.split("_")
        if len(parts) >= 3:
            return _blend_optimizer_label(parts[1], parts[2], "")
    if optimizer:
        return f"{optimizer.title()} blend"
    return "Blend"


def _parent_dataset_versions(parent_displays: Sequence[CandidateDisplay]) -> str:
    versions: list[str] = []
    seen: set[str] = set()
    for item in parent_displays:
        dataset_id = item.dataset_id
        version = dataset_id.split("_", 1)[0] if dataset_id.startswith("v") else ""
        if version and version not in seen:
            seen.add(version)
            versions.append(version)
    return " + ".join(versions)


def _weighted_parent_detail(
    parent_displays: Sequence[CandidateDisplay],
    weights: Mapping[str, Any],
) -> str:
    ranked: list[tuple[float, str]] = []
    for display in parent_displays:
        raw = weights.get(display.candidate_id)
        try:
            weight = float(raw)
        except (TypeError, ValueError):
            continue
        short = ultra_short_model_label(
            display.model_name, dataset_id=display.dataset_id or None
        )
        percent = int(round(weight * 100))
        ranked.append((weight, f"{short} {percent}%"))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return " + ".join(label for _, label in ranked)


def resolve_candidate_display(
    candidate_id: str | Mapping[str, Any],
    *,
    repository_root: Path | str | None = None,
    candidates_root_relative: str | None = None,
    metadata_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    _stack: frozenset[str] | None = None,
) -> CandidateDisplay:
    """Resolve a readable display model for one candidate.

    Accepts either a candidate ID or a metadata/summary mapping. Parent blend
    labels are resolved recursively through ``metadata_by_id`` or disk lookup.
    """
    stack = _stack or frozenset()
    if isinstance(candidate_id, Mapping):
        meta = _metadata_from_mapping(candidate_id)
        cid = meta["candidate_id"] or "unknown"
        missing_reason = _safe_str(candidate_id.get("missing_reason")) or None
    else:
        cid = str(candidate_id)
        if metadata_by_id and cid in metadata_by_id:
            meta = _metadata_from_mapping(metadata_by_id[cid])
            missing_reason = _safe_str(metadata_by_id[cid].get("missing_reason")) or None
        elif repository_root is not None:
            loaded = load_candidate_metadata(
                cid,
                repository_root=repository_root,
                candidates_root_relative=candidates_root_relative,
            )
            meta = loaded
            missing_reason = _safe_str(loaded.get("missing_reason")) or None
        else:
            meta = {
                "candidate_id": cid,
                "source_model_name": "",
                "source_kind": "",
                "dataset_id": "",
                "exploratory": False,
                "parent_candidate_ids": [],
                "final_deployment_weights": {},
                "strategy": "",
                "optimizer_backend": "",
                "blend_id": "",
            }
            missing_reason = "Candidate metadata unavailable"

    model_name = meta["source_model_name"]
    source_kind = meta["source_kind"]
    dataset_id = meta["dataset_id"]
    source_label = source_kind_display_label(source_kind)
    dataset_label = dataset_display_label(dataset_id)
    exploratory = bool(meta.get("exploratory"))

    if not model_name and not source_kind and not dataset_id:
        fallback = f"Candidate {cid}" if cid else "Unknown candidate"
        return CandidateDisplay(
            candidate_id=cid,
            primary_label=fallback,
            compact_label=fallback,
            model_name="",
            source_label=source_label,
            dataset_label=dataset_label,
            dataset_id=dataset_id,
            source_kind=source_kind,
            exploratory=exploratory,
            missing_reason=missing_reason or "Incomplete candidate metadata",
        )

    is_blend = source_kind == BLEND_SOURCE_KIND or model_name.startswith("blend_")
    if is_blend:
        parent_ids = list(meta.get("parent_candidate_ids") or [])
        parent_displays: list[CandidateDisplay] = []
        if cid in stack:
            parent_displays = []
        else:
            nested = stack | {cid}
            for parent_id in parent_ids:
                parent_displays.append(
                    resolve_candidate_display(
                        parent_id,
                        repository_root=repository_root,
                        candidates_root_relative=candidates_root_relative,
                        metadata_by_id=metadata_by_id,
                        _stack=nested,
                    )
                )
        blend_name = _blend_optimizer_label(
            str(meta.get("strategy") or ""),
            str(meta.get("optimizer_backend") or ""),
            model_name,
        )
        versions = _parent_dataset_versions(parent_displays)
        count = len(parent_ids) or len(parent_displays)
        primary = f"{blend_name} · {count} models"
        if versions:
            primary = f"{primary} · {versions}"
        detail = _weighted_parent_detail(
            parent_displays, meta.get("final_deployment_weights") or {}
        )
        if detail:
            # Keep primary blend identity short; detail is available via compact
            # when weights exist and as a richer secondary form.
            compact = f"{blend_name} · {detail}"
        else:
            parent_compacts = [
                item.compact_label for item in parent_displays if item.compact_label
            ]
            compact = (
                f"{blend_name} · {' + '.join(parent_compacts)}"
                if parent_compacts
                else primary
            )
        return CandidateDisplay(
            candidate_id=cid,
            primary_label=primary,
            compact_label=compact,
            model_name=model_name or blend_name,
            source_label=source_label,
            dataset_label=dataset_label,
            dataset_id=dataset_id,
            source_kind=source_kind,
            exploratory=exploratory,
            missing_reason=missing_reason,
        )

    compact_model = compact_model_name(model_name or "Unknown model")
    version = dataset_id.split("_", 1)[0] if dataset_id.startswith("v") else ""
    primary = f"{model_name or compact_model} · {source_label} · {dataset_label}"
    compact = f"{compact_model} · {version}" if version else compact_model
    return CandidateDisplay(
        candidate_id=cid,
        primary_label=primary,
        compact_label=compact,
        model_name=model_name or compact_model,
        source_label=source_label,
        dataset_label=dataset_label,
        dataset_id=dataset_id,
        source_kind=source_kind,
        exploratory=exploratory,
        missing_reason=missing_reason,
    )


def resolve_candidate_displays(
    candidate_ids: Sequence[str],
    *,
    repository_root: Path | str | None = None,
    candidates_root_relative: str | None = None,
    metadata_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[CandidateDisplay]:
    return [
        resolve_candidate_display(
            candidate_id,
            repository_root=repository_root,
            candidates_root_relative=candidates_root_relative,
            metadata_by_id=metadata_by_id,
        )
        for candidate_id in candidate_ids
    ]


def candidate_display_map(
    candidate_ids: Sequence[str] | Iterable[str],
    *,
    repository_root: Path | str | None = None,
    candidates_root_relative: str | None = None,
    metadata_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, CandidateDisplay]:
    return {
        item.candidate_id: item
        for item in resolve_candidate_displays(
            list(candidate_ids),
            repository_root=repository_root,
            candidates_root_relative=candidates_root_relative,
            metadata_by_id=metadata_by_id,
        )
    }


def disambiguate_compact_labels(
    displays: Sequence[CandidateDisplay],
) -> dict[str, str]:
    """Return candidate_id → unique compact axis label."""
    counts: dict[str, int] = {}
    for item in displays:
        counts[item.compact_label] = counts.get(item.compact_label, 0) + 1
    result: dict[str, str] = {}
    for item in displays:
        label = item.compact_label
        if counts.get(label, 0) > 1:
            suffix = item.candidate_id
            if suffix.startswith("pc1_") and len(suffix) >= 8:
                suffix = suffix[:8]
            label = f"{label} [{suffix}]"
        result[item.candidate_id] = label
    return result


def replace_candidate_ids_with_labels(
    mapping: Mapping[str, Any],
    displays: Mapping[str, CandidateDisplay],
    *,
    style: str = "primary",
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in mapping.items():
        display = displays.get(str(key))
        if display is None:
            label = str(key)
        elif style == "compact":
            label = display.compact_label
        else:
            label = display.primary_label
        result[label] = value
    return result


def project_weight_rows(
    weights: Mapping[str, Any] | None,
    displays: Mapping[str, CandidateDisplay],
    *,
    style: str = "primary",
) -> list[dict[str, Any]]:
    if not isinstance(weights, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for candidate_id, weight in weights.items():
        display = displays.get(str(candidate_id))
        if display is None:
            model = str(candidate_id)
        elif style == "compact":
            model = display.compact_label
        else:
            model = display.primary_label
        try:
            numeric = float(weight)
        except (TypeError, ValueError):
            numeric = weight
        rows.append(
            {
                "Model": model,
                "Weight": numeric,
                "Candidate ID": str(candidate_id),
            }
        )
    rows.sort(
        key=lambda row: (
            -(float(row["Weight"]) if isinstance(row["Weight"], (int, float)) else 0.0),
            str(row["Model"]),
        )
    )
    return rows


def project_candidate_quality_rows(
    quality_rows: Sequence[Mapping[str, Any]],
    displays: Mapping[str, CandidateDisplay],
) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for row in quality_rows:
        candidate_id = str(row.get("candidate_id") or "")
        display = displays.get(candidate_id)
        threshold_05 = row.get("threshold_0_5") or {}
        optimal = row.get("descriptive_oof_optimal_metrics") or {}
        if not isinstance(threshold_05, Mapping):
            threshold_05 = {}
        if not isinstance(optimal, Mapping):
            optimal = {}
        projected.append(
            {
                "Model": display.primary_label if display else candidate_id,
                "Dataset": display.dataset_label if display else "",
                "Threshold 0.5 BA": threshold_05.get("balanced_accuracy"),
                "Descriptive optimal threshold": row.get(
                    "descriptive_oof_optimal_threshold"
                ),
                "Descriptive optimal BA": optimal.get("balanced_accuracy"),
                "Sensitivity": optimal.get("sensitivity"),
                "Specificity": optimal.get("specificity"),
                "Candidate ID": candidate_id,
            }
        )
    return projected


def project_pairwise_rows(
    pairwise_rows: Sequence[Mapping[str, Any]],
    displays: Mapping[str, CandidateDisplay],
    *,
    unique_labels: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    labels = unique_labels or {
        candidate_id: display.compact_label
        for candidate_id, display in displays.items()
    }
    projected: list[dict[str, Any]] = []
    for row in pairwise_rows:
        a = str(row.get("candidate_a") or "")
        b = str(row.get("candidate_b") or "")
        projected.append(
            {
                "Model A": labels.get(a) or (displays[a].primary_label if a in displays else a),
                "Model B": labels.get(b) or (displays[b].primary_label if b in displays else b),
                "Pearson": row.get("pearson", row.get("pearson_correlation")),
                "Spearman": row.get("spearman"),
                "Mean absolute difference": row.get("mean_abs_diff"),
                "RMSE": row.get("rmse"),
                "Prediction disagreement": row.get("prediction_disagreement_rate"),
                "Joint error rate": row.get("joint_error_rate"),
                "Error overlap": row.get("error_overlap"),
                "Positive-class disagreement": row.get("positive_class_disagreement"),
                "Negative-class disagreement": row.get("negative_class_disagreement"),
                "Candidate A ID": a,
                "Candidate B ID": b,
            }
        )
    return projected


def project_heatmap_labels(
    candidate_ids: Sequence[str],
    displays: Mapping[str, CandidateDisplay],
) -> list[str]:
    ordered = [
        displays[candidate_id]
        if candidate_id in displays
        else CandidateDisplay(
            candidate_id=candidate_id,
            primary_label=candidate_id,
            compact_label=candidate_id,
            model_name="",
            source_label="",
            dataset_label="",
            dataset_id="",
            source_kind="",
            exploratory=False,
            missing_reason="Missing display",
        )
        for candidate_id in candidate_ids
    ]
    unique = disambiguate_compact_labels(ordered)
    return [unique[candidate_id] for candidate_id in candidate_ids]


def project_optuna_study_rows(
    summaries: Sequence[Mapping[str, Any]],
    displays: Mapping[str, CandidateDisplay],
    ordered_candidate_ids: Sequence[str],
) -> list[dict[str, Any]]:
    columns = []
    for candidate_id in ordered_candidate_ids:
        display = displays.get(str(candidate_id))
        columns.append(
            (
                str(candidate_id),
                display.compact_label if display else str(candidate_id),
            )
        )
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        row: dict[str, Any] = {
            "Repeat": summary.get("repeat"),
            "Fold": summary.get("fold"),
            "Best objective": summary.get("best_value")
            or summary.get("best_objective"),
            "Threshold": summary.get("threshold")
            or summary.get("best_threshold"),
            "Best trial": summary.get("best_trial")
            or summary.get("n_trials"),
        }
        weights = (
            summary.get("resolved_weights")
            or summary.get("best_weights")
            or summary.get("weights")
            or {}
        )
        if isinstance(weights, Mapping):
            for candidate_id, label in columns:
                if candidate_id in weights:
                    row[label] = weights[candidate_id]
        elif isinstance(weights, Sequence) and not isinstance(weights, (str, bytes)):
            for index, (candidate_id, label) in enumerate(columns):
                if index < len(weights):
                    row[label] = weights[index]
        rows.append(row)
    return rows


def experiment_models_label(
    displays: Sequence[CandidateDisplay],
    *,
    optimizer_backend: str,
) -> str:
    optimizer = str(optimizer_backend or "unknown")
    if optimizer == "native":
        head = "Native"
    elif optimizer == "optuna":
        head = "Optuna"
    else:
        head = optimizer.title()
    parts: list[str] = []
    for item in displays:
        if item.source_kind == BLEND_SOURCE_KIND:
            parts.append(item.compact_label)
            continue
        model = item.model_name or item.compact_label
        if "WeightedEnsemble" in model:
            version = (
                item.dataset_id.split("_", 1)[0]
                if item.dataset_id.startswith("v")
                else ""
            )
            parts.append(
                f"WeightedEnsemble {version}".strip()
                if version
                else compact_model_name(model)
            )
        else:
            parts.append(compact_model_name(model) if model else item.compact_label)
    models = " + ".join(parts) if parts else "no models"
    return f"{head} · {models}"


__all__ = [
    "AUTOGLUON_SOURCE_KIND",
    "BLEND_SOURCE_KIND",
    "CandidateDisplay",
    "candidate_display_map",
    "compact_model_name",
    "dataset_display_label",
    "disambiguate_compact_labels",
    "experiment_models_label",
    "load_candidate_metadata",
    "project_candidate_quality_rows",
    "project_heatmap_labels",
    "project_optuna_study_rows",
    "project_pairwise_rows",
    "project_weight_rows",
    "replace_candidate_ids_with_labels",
    "resolve_candidate_display",
    "resolve_candidate_displays",
    "source_kind_display_label",
    "ultra_short_model_label",
]
