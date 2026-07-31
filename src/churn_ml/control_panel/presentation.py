from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


_MAX_CONFIG_BYTES = 32 * 1024
_RUN_ID_RE = re.compile(
    r"^(?P<stamp>\d{8}T\d{6}(?:\d{0,6})?Z)_(?P<hash>[0-9a-fA-F]{6,})$"
)
_NA = "Not available"


def normalize_model_family(raw: str) -> str:
    lower = raw.lower()
    if "lightgbm" in lower and "xgboost" in lower:
        return "LightGBM + XGBoost Blend"
    if lower.startswith("xgboost"):
        return "XGBoost"
    if lower.startswith("catboost"):
        return "CatBoost"
    if lower.startswith("manual_lightgbm") or lower.startswith("lightgbm"):
        return "LightGBM"
    if lower.startswith("autogluon"):
        return "AutoGluon"
    if "blend" in lower:
        return "LightGBM + XGBoost Blend"
    if raw == raw.lower():
        return raw.title()
    return raw


def normalize_mode(raw: str) -> str:
    lower = raw.lower()
    if lower == "smoke":
        return "Smoke"
    if lower == "development":
        return "Development"
    if lower == "deployment":
        return "Deployment"
    return raw.title()


def normalize_source_kind(path_str: str, content: dict | None = None) -> str:
    del content
    if path_str.startswith("artifacts/optuna_exports"):
        return "Optuna export"
    if path_str.startswith("artifacts/optuna_searches"):
        return "Optuna study"
    if path_str.startswith("artifacts/ui_configs"):
        return "UI config copy"
    if path_str.startswith("artifacts/research_v2_comparisons"):
        return "Paired comparison"
    if path_str.startswith("artifacts/blend_evaluations"):
        return "Blend evaluation"
    if path_str.startswith("artifacts/blend_deployment_packages"):
        return "Blend deployment package"
    if path_str.startswith("artifacts/research_v2"):
        return "Experiment Core run"
    if path_str.startswith("artifacts/research/"):
        return "Research v1 run"
    if path_str.startswith("artifacts/deployments"):
        return "Deployment artifact"
    if path_str.startswith("artifacts/deployment_fixtures"):
        return "Deployment fixture"
    if path_str.startswith("configs/"):
        return "Canonical config"
    return "Other"


def config_badge(source_kind: str) -> str:
    mapping = {
        "Optuna export": "EXPORTED",
        "Optuna study": "STUDY",
        "UI config copy": "COPY",
        "Canonical config": "CANONICAL",
        "Experiment Core run": "RUN",
        "Research v1 run": "RUN",
        "Paired comparison": "COMPARE",
        "Blend evaluation": "BLEND",
        "Blend deployment package": "BLEND",
        "Deployment artifact": "DEPLOY",
        "Deployment fixture": "FIXTURE",
    }
    return mapping.get(source_kind, "OTHER")


def mode_badge(mode: str) -> str:
    mapping = {
        "Smoke": "SMOKE",
        "Development": "DEVELOPMENT",
        "Deployment": "DEPLOYMENT",
    }
    return mapping.get(mode, mode.upper())


def _read_file_bounded(path: Path) -> dict[str, Any] | None:
    try:
        if not path.is_file():
            return None
        size = path.stat().st_size
        if size > _MAX_CONFIG_BYTES:
            return None
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            result = json.loads(text)
            return result if isinstance(result, dict) else None
        result = yaml.safe_load(text)
        if isinstance(result, dict):
            return result
        return None
    except Exception:
        return None


def _read_config_bounded(path: Path) -> dict[str, Any] | None:
    try:
        if path.is_dir():
            for candidate in (
                "resolved_config.yaml",
                "run_metadata.json",
                "study_summary.json",
                "best_trial.json",
                "metrics/aggregate.json",
                "decision_report.json",
                "prediction_summary.json",
                "aggregate_summary.json",
            ):
                nested = path / candidate
                loaded = _read_file_bounded(nested)
                if loaded is not None:
                    return loaded
            return None
        return _read_file_bounded(path)
    except Exception:
        return None


def parse_run_id_timestamp(run_id: str) -> str | None:
    """Convert a research run_id stamp into an ISO-8601 UTC timestamp string."""
    match = _RUN_ID_RE.match(run_id.strip())
    if match is None:
        return None
    stamp = match.group("stamp")
    # YYYYMMDDTHHMMSS[ffffff]Z
    date_part = stamp[:8]
    time_part = stamp[9:]
    if time_part.endswith("Z"):
        time_part = time_part[:-1]
    if len(time_part) < 6:
        return None
    hour = time_part[0:2]
    minute = time_part[2:4]
    second = time_part[4:6]
    fraction = time_part[6:]
    iso = f"{date_part[0:4]}-{date_part[4:6]}-{date_part[6:8]}T{hour}:{minute}:{second}"
    if fraction:
        iso = f"{iso}.{fraction}"
    return f"{iso}+00:00"


def is_raw_run_id(value: str) -> bool:
    return _RUN_ID_RE.match(value.strip()) is not None


def humanize_experiment_token(raw: str, model_family: str = "") -> str:
    """Turn adapter/experiment tokens into a short human label."""
    text = raw.strip()
    if not text:
        return ""
    if is_raw_run_id(text):
        return ""
    lower = text.lower()
    for prefix in (
        "manual_lightgbm_",
        "lightgbm_",
        "xgboost_",
        "catboost_",
        "autogluon_",
        "manual_",
    ):
        if lower.startswith(prefix):
            text = text[len(prefix) :]
            lower = text.lower()
            break
    for suffix in ("_smoke", "_development", "_deployment"):
        if lower.endswith(suffix):
            text = text[: -len(suffix)]
            lower = text.lower()
            break
    replacements = {
        "te": "TE",
        "compat": "compatibility",
        "v1": "v1",
        "v2": "v2",
        "v3": "v3",
        "numeric": "numeric",
        "pipeline": "pipeline",
    }
    words: list[str] = []
    for token in text.replace("__", "_").split("_"):
        if not token:
            continue
        mapped = replacements.get(token.lower())
        if mapped is not None:
            words.append(mapped)
        else:
            words.append(token)
    label = " ".join(words).strip()
    if not label:
        return model_family or raw
    if model_family and not label.lower().startswith(model_family.lower()):
        return f"{model_family} {label}".strip()
    return label


def format_primary_metric(value: Any, *, prefix: str = "BA") -> str | None:
    try:
        if value is None or value == "":
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return f"{prefix} {number:.6f}"


def _merge_metadata(result: dict[str, str], content: dict[str, Any]) -> None:
    experiment = content.get("experiment", {})
    if isinstance(experiment, dict):
        exp_id = experiment.get("id", "")
        if exp_id and "experiment_name" not in result:
            result["experiment_name"] = str(exp_id)
        model = experiment.get("model", {})
        if isinstance(model, dict):
            adapter = model.get("adapter", "")
            if adapter:
                result["adapter_id"] = str(adapter)
                result.setdefault("model_family", normalize_model_family(str(adapter)))
        cv = experiment.get("cv", {})
        if isinstance(cv, dict):
            repeats = cv.get("repeats")
            folds = cv.get("folds")
            if repeats is not None:
                result["repeats"] = str(repeats)
            if folds is not None:
                result["folds"] = str(folds)

    dataset = content.get("dataset", {})
    if isinstance(dataset, dict):
        version = dataset.get("version")
        if version:
            result.setdefault("dataset_id", str(version))
            result.setdefault("dataset_version", str(version))

    if "dataset_id" in content and content.get("dataset_id"):
        result.setdefault("dataset_id", str(content["dataset_id"]))
        result.setdefault("dataset_version", str(content["dataset_id"]))
    if "parent_dataset_id" in content and content.get("parent_dataset_id") is not None:
        result.setdefault("parent_dataset_id", str(content["parent_dataset_id"]))
    if "target_dependency" in content and content.get("target_dependency"):
        result.setdefault("target_dependency", str(content["target_dependency"]))
    if "n_features" in content and content.get("n_features") is not None:
        result.setdefault("n_features", str(content["n_features"]))
    if "schema_hash" in content and content.get("schema_hash"):
        result.setdefault("schema_hash", str(content["schema_hash"]))
    if "train_content_hash" in content and content.get("train_content_hash"):
        result.setdefault("train_content_hash", str(content["train_content_hash"]))
    if "target_hash" in content and content.get("target_hash"):
        result.setdefault("target_hash", str(content["target_hash"]))
    if "train_row_identity_hash" in content and content.get("train_row_identity_hash"):
        result.setdefault(
            "train_row_identity_hash", str(content["train_row_identity_hash"])
        )

    provenance = content.get("dataset_provenance")
    if isinstance(provenance, dict):
        if provenance.get("dataset_id"):
            result.setdefault("dataset_id", str(provenance["dataset_id"]))
            result.setdefault("dataset_version", str(provenance["dataset_id"]))
        if provenance.get("parent_dataset_id") is not None:
            result.setdefault(
                "parent_dataset_id", str(provenance.get("parent_dataset_id"))
            )
        if provenance.get("target_dependency"):
            result.setdefault(
                "target_dependency", str(provenance["target_dependency"])
            )
        if provenance.get("n_features") is not None:
            result.setdefault("n_features", str(provenance["n_features"]))

    plan = content.get("plan", {})
    if isinstance(plan, dict):
        plan_id = plan.get("id", "")
        if plan_id:
            result["plan_id"] = str(plan_id)

    optuna = content.get("optuna", {})
    if isinstance(optuna, dict):
        n_trials = optuna.get("n_trials")
        if n_trials is not None:
            result["n_trials"] = str(n_trials)

    if "best_trial_number" in content:
        result["best_trial"] = str(content["best_trial_number"])
    if "best_objective" in content:
        result["best_objective"] = str(content["best_objective"])
    if "status" in content and "status" not in result:
        result["status"] = str(content["status"])

    if "study_name" in content and "experiment_name" not in result:
        result["experiment_name"] = str(content["study_name"])
    if "search_id" in content and "experiment_name" not in result:
        result["experiment_name"] = str(content["search_id"])

    if "experiment_id" in content and "experiment_name" not in result:
        result["experiment_name"] = str(content["experiment_id"])
    if "plan_id" in content and "plan_id" not in result:
        result["plan_id"] = str(content["plan_id"])
    if "candidate_adapter_id" in content:
        adapter = str(content["candidate_adapter_id"])
        result["adapter_id"] = adapter
        result.setdefault("model_family", normalize_model_family(adapter))
    if "started_at_utc" in content and "created_at_utc" not in result:
        result["created_at_utc"] = str(content["started_at_utc"])
    if "finished_at_utc" in content and "created_at_utc" not in result:
        result["created_at_utc"] = str(content["finished_at_utc"])

    metrics = content.get("metrics")
    if isinstance(metrics, dict):
        ba = metrics.get("balanced_accuracy")
        if isinstance(ba, dict) and ba.get("mean") is not None:
            result["balanced_accuracy"] = str(ba["mean"])
        elif content.get("primary_metric") == "balanced_accuracy" and isinstance(
            metrics.get("balanced_accuracy"), (int, float)
        ):
            result["balanced_accuracy"] = str(metrics["balanced_accuracy"])


def parse_config_metadata(path_str: str, repo_root: Path) -> dict[str, str]:
    try:
        full_path = repo_root / path_str
        content = _read_config_bounded(full_path)
        source_kind = normalize_source_kind(path_str, content)
        result: dict[str, str] = {"source_kind": source_kind}

        stem = Path(path_str).stem
        basename = Path(path_str).name
        parts = [p for p in Path(path_str).parts if p not in (".",)]
        path_tokens: list[str] = []
        for part in parts:
            path_tokens.append(part)
            path_tokens.extend(part.split("_"))
            path_tokens.extend(part.split("__"))
        path_tokens.extend(stem.split("_"))

        mode_candidates = ["smoke", "development", "deployment"]
        for part in reversed(path_tokens):
            if part.lower() in mode_candidates:
                result["mode"] = normalize_mode(part)
                break

        run_stamp = parse_run_id_timestamp(basename)
        if run_stamp is not None:
            result["created_at_utc"] = run_stamp
            result["run_id"] = basename

        if full_path.is_dir():
            for relative in (
                "dataset_provenance.json",
                "run_metadata.json",
                "resolved_config.yaml",
                "metrics/aggregate.json",
                "study_summary.json",
                "best_trial.json",
            ):
                nested = _read_file_bounded(full_path / relative)
                if nested is not None:
                    _merge_metadata(result, nested)
            # Parent pipeline__adapter directory often encodes model family.
            if len(parts) >= 2:
                parent = parts[-2]
                if "__" in parent:
                    adapter_token = parent.split("__", 1)[1]
                    result.setdefault(
                        "model_family", normalize_model_family(adapter_token)
                    )
                    result.setdefault("adapter_id", adapter_token)
                    if "experiment_name" not in result:
                        result["experiment_name"] = adapter_token

        if content is not None:
            _merge_metadata(result, content)

        if "model_family" not in result:
            for token in path_tokens:
                family = normalize_model_family(token)
                if family in {
                    "LightGBM",
                    "XGBoost",
                    "CatBoost",
                    "AutoGluon",
                    "LightGBM + XGBoost Blend",
                }:
                    result["model_family"] = family
                    break
            if "model_family" not in result:
                family = normalize_model_family(stem)
                if family not in (stem, stem.title()):
                    result["model_family"] = family
                else:
                    for prefix in (
                        "xgboost",
                        "catboost",
                        "lightgbm",
                        "manual_lightgbm",
                        "autogluon",
                    ):
                        if stem.lower().startswith(prefix):
                            result["model_family"] = normalize_model_family(stem)
                            break

        return result
    except Exception:
        return {}


def _stem_description(stem: str) -> str:
    known_prefixes = (
        "manual_lightgbm",
        "lightgbm",
        "xgboost",
        "catboost",
        "autogluon",
    )
    known_suffixes = ("smoke", "development", "deployment")
    lower = stem.lower()
    stripped = lower
    for prefix in known_prefixes:
        if stripped.startswith(prefix + "_"):
            stripped = stripped[len(prefix) + 1 :]
            break
        if stripped == prefix:
            stripped = ""
            break
    for suffix in known_suffixes:
        if stripped.endswith("_" + suffix):
            stripped = stripped[: -(len(suffix) + 1)]
            break
        if stripped == suffix:
            stripped = ""
            break
    stripped = stripped.strip("_")
    if not stripped:
        return ""
    words = stripped.split("_")
    friendly = " ".join(words)
    return friendly


def readable_config_label(path_str: str, repo_root: Path) -> str:
    try:
        meta = parse_config_metadata(path_str, repo_root)
        stem = Path(path_str).stem

        model_family = meta.get("model_family") or normalize_model_family(stem)
        mode = meta.get("mode", "")
        source_kind = meta.get("source_kind") or normalize_source_kind(path_str)
        badge = config_badge(source_kind)

        desc = _stem_description(stem)

        parts = [model_family]
        if mode:
            parts.append(normalize_mode(mode))
        if desc:
            parts.append(desc)

        label = " — ".join(parts)
        return f"{label} [{badge}]"
    except Exception:
        return Path(path_str).name


def format_date(utc_iso: str | None) -> str:
    if utc_iso is None:
        return "—"
    try:
        cleaned = re.sub(r"Z$", "+00:00", utc_iso)
        dt = datetime.fromisoformat(cleaned).astimezone(timezone.utc)
        day = str(dt.day)
        month = dt.strftime("%b")
        year = str(dt.year)
        return f"{day} {month} {year}"
    except Exception:
        return "Invalid timestamp"


def format_time(utc_iso: str | None) -> str:
    if utc_iso is None:
        return "—"
    try:
        cleaned = re.sub(r"Z$", "+00:00", utc_iso)
        dt = datetime.fromisoformat(cleaned).astimezone(timezone.utc)
        return dt.strftime("%H:%M:%S")
    except Exception:
        return ""


def format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "—"
    value = max(0, int(seconds))
    if value < 60:
        return f"{value}s"
    if value < 3600:
        minutes = value // 60
        secs = value % 60
        return f"{minutes}m {secs:02d}s"
    hours = value // 3600
    remainder = value % 3600
    minutes = remainder // 60
    secs = remainder % 60
    return f"{hours}h {minutes}m {secs:02d}s"


def job_primary_label(
    record_job: Mapping[str, Any],
    record_commands: dict | None = None,
    *,
    repository_root: Path | None = None,
) -> str:
    try:
        command_id = record_job.get("command_id", "")
        action_id = record_job.get("action_id", "")
        references = record_job.get("references", {})
        if not isinstance(references, dict):
            references = {}
        config_path = references.get("config", "")
        repo = repository_root if repository_root is not None else Path(".")

        parts: list[str] = []
        dataset_id = references.get("dataset_id")
        model_family = references.get("model_family")
        mode = references.get("mode")
        if not dataset_id or not model_family or not mode:
            meta: dict[str, str] = {}
            if config_path:
                meta = parse_config_metadata(str(config_path), repo)
            dataset_id = dataset_id or meta.get("dataset_id") or meta.get(
                "dataset_version"
            )
            model_family = model_family or meta.get("model_family")
            mode = mode or meta.get("mode")
        if dataset_id:
            parts.append(str(dataset_id))
        if model_family:
            parts.append(str(model_family))
        if mode:
            parts.append(str(mode))

        operation = ""
        if record_commands and command_id in record_commands:
            command = record_commands[command_id]
            operation = getattr(command, "title", "") or str(command_id)
            actions = getattr(command, "actions", {})
            if action_id in actions:
                action_title = getattr(actions[action_id], "title", "")
                if action_title:
                    operation = f"{operation} {action_title}".strip()
        if not operation:
            cmd_part = command_id.replace("_", " ").title() if command_id else ""
            act_part = action_id.replace("_", " ").title() if action_id else ""
            operation = " ".join(p for p in (cmd_part, act_part) if p)

        if operation:
            parts.append(operation)

        created = record_job.get("created_at_utc")
        if isinstance(created, str) and created:
            date_label = format_date(created)
            time_label = format_time(created)
            if date_label and date_label not in {"—", "Invalid timestamp"}:
                parts.append(date_label)
            if time_label:
                parts.append(time_label)

        job_id = record_job.get("job_id")
        if isinstance(job_id, str) and len(job_id) >= 8:
            parts.append(job_id[:8])

        if parts:
            return " · ".join(parts)
        return f"{command_id} / {action_id}" if command_id else "Unknown job"
    except Exception:
        command_id = record_job.get("command_id", "")
        action_id = record_job.get("action_id", "")
        return f"{command_id} / {action_id}" if command_id else "Unknown job"


def build_pre_run_summary(
    command_id: str,
    action_id: str,
    command_title: str,
    action_title: str,
    values: dict,
    repo_root: Path,
) -> dict[str, str]:
    summary: dict[str, str] = {}

    summary["Operation"] = command_title
    summary["Action"] = action_title

    config_path = values.get("config", "")
    if config_path:
        meta = parse_config_metadata(str(config_path), repo_root)
        model_family = meta.get("model_family", "")
        mode = meta.get("mode", "")
        source_kind = meta.get("source_kind", "")
        badge = config_badge(source_kind) if source_kind else ""

        if model_family:
            summary["Model"] = model_family
        if mode:
            summary["Mode"] = mode
        if source_kind:
            summary["Source"] = source_kind

        config_base = Path(str(config_path)).name
        if badge:
            summary["Config"] = f"{config_base} [{badge}]"
        else:
            summary["Config"] = config_base

        repeats = meta.get("repeats", "")
        folds = meta.get("folds", "")
        if repeats and folds:
            summary["Plan"] = f"{repeats} repeats × {folds} folds"

        n_trials = meta.get("n_trials", "")
        if n_trials:
            summary["Trials"] = n_trials

    return summary


# ---------------------------------------------------------------------------
# Cascade config selector helpers
# ---------------------------------------------------------------------------

_SOURCE_LABEL_MAP: dict[str, str] = {
    "Canonical config": "Canonical config",
    "Optuna export": "Optuna export",
    "Optuna study": "Optuna study",
    "UI config copy": "UI copy",
    "Experiment Core run": "Experiment Core run",
    "Research v1 run": "Research v1 run",
    "Paired comparison": "Paired comparison",
    "Blend evaluation": "Blend evaluation",
    "Blend deployment package": "Blend deployment package",
    "Deployment artifact": "Deployment artifact",
    "Deployment fixture": "Deployment fixture",
    "Other": "Other approved source",
}

_MODEL_LABEL_MAP: dict[str, str] = {
    "LightGBM": "LightGBM",
    "XGBoost": "XGBoost",
    "CatBoost": "CatBoost",
    "AutoGluon": "AutoGluon",
    "LightGBM + XGBoost Blend": "LightGBM + XGBoost Blend",
}

_MODE_LABEL_MAP: dict[str, str] = {
    "Smoke": "Smoke",
    "Development": "Development",
    "Deployment": "Deployment",
}


@dataclass
class ConfigCascadeOption:
    """One config file with its extracted metadata for cascade filtering."""

    path: str
    source_kind: str
    model_family: str
    mode: str
    display_label: str
    basename: str


def _experiment_core_label(meta: dict[str, str], basename: str) -> str:
    model_family = meta.get("model_family", "")
    experiment_token = (
        meta.get("experiment_name")
        or meta.get("adapter_id")
        or _stem_description(Path(basename).stem)
        or ""
    )
    experiment = humanize_experiment_token(str(experiment_token), model_family)
    if not experiment or is_raw_run_id(experiment):
        experiment = humanize_experiment_token(
            meta.get("adapter_id", ""), model_family
        ) or (model_family or "Experiment")

    created = meta.get("created_at_utc")
    date_label = format_date(created) if created else ""
    time_label = format_time(created) if created else ""
    # Prefer HH:MM for compact selector labels when seconds are available.
    if time_label and len(time_label) >= 5:
        time_label = time_label[:5]

    metric = format_primary_metric(meta.get("balanced_accuracy"))
    if metric is None:
        metric = format_primary_metric(meta.get("best_objective"), prefix="obj")

    dataset = meta.get("dataset_id") or meta.get("dataset_version") or ""
    parts = [part for part in (dataset, experiment) if part]
    if not parts:
        parts = [experiment or basename]
    if date_label and date_label not in {"—", "Invalid timestamp"}:
        parts.append(date_label)
    if time_label:
        parts.append(time_label)
    if metric:
        parts.append(metric)
    return " · ".join(parts)


def build_cascade_options(
    paths: list[str], repo_root: Path
) -> list[ConfigCascadeOption]:
    """Parse metadata for each path and return structured cascade options."""
    options: list[ConfigCascadeOption] = []
    seen_labels: dict[str, int] = {}
    for path in paths:
        meta = parse_config_metadata(path, repo_root)
        source_kind = meta.get("source_kind", "Other")
        model_family = meta.get("model_family", "")
        mode = meta.get("mode", "")
        basename = Path(path).name

        if source_kind in {"Optuna export", "Optuna study"}:
            trial = meta.get("best_trial", "")
            obj = meta.get("best_objective", "")
            name = meta.get("experiment_name", "")
            created = meta.get("created_at_utc")
            date_label = format_date(created) if created else ""
            time_label = format_time(created) if created else ""
            if time_label and len(time_label) >= 5:
                time_label = time_label[:5]
            metric = format_primary_metric(obj, prefix="BA")
            if trial and obj not in ("", "None", "none"):
                label_core = f"Optuna trial {trial}"
            elif name:
                label_core = str(name)
            elif trial:
                label_core = f"Optuna trial {trial}"
            else:
                label_core = basename
            extras = [
                part
                for part in (date_label, time_label, metric)
                if part and part not in {"—", "Invalid timestamp"}
            ]
            if extras:
                label_core = " · ".join([label_core, *extras])
        elif source_kind in {"Experiment Core run", "Research v1 run"} or is_raw_run_id(
            basename
        ):
            label_core = _experiment_core_label(meta, basename)
        else:
            stem = Path(path).stem
            desc = _stem_description(stem)
            if desc and not is_raw_run_id(desc.replace(" ", "_")):
                label_core = desc
            else:
                label_core = meta.get("experiment_name") or basename
                if is_raw_run_id(str(label_core)):
                    label_core = _experiment_core_label(meta, basename)

        if label_core in seen_labels:
            seen_labels[label_core] += 1
            label_core = f"{label_core} ({seen_labels[label_core]})"
        else:
            seen_labels[label_core] = 1

        options.append(
            ConfigCascadeOption(
                path=path,
                source_kind=source_kind,
                model_family=model_family,
                mode=mode,
                display_label=label_core,
                basename=basename,
            )
        )
    return options


def cascade_available_sources(options: list[ConfigCascadeOption]) -> list[str]:
    seen: list[str] = []
    for opt in options:
        if opt.source_kind not in seen:
            seen.append(opt.source_kind)
    return seen


def cascade_available_models(
    options: list[ConfigCascadeOption], source: str | None = None
) -> list[str]:
    seen: list[str] = []
    for opt in options:
        if source is not None and opt.source_kind != source:
            continue
        if opt.model_family and opt.model_family not in seen:
            seen.append(opt.model_family)
    return seen


def cascade_available_modes(
    options: list[ConfigCascadeOption],
    source: str | None = None,
    model: str | None = None,
) -> list[str]:
    seen: list[str] = []
    for opt in options:
        if source is not None and opt.source_kind != source:
            continue
        if model is not None and opt.model_family != model:
            continue
        if opt.mode and opt.mode not in seen:
            seen.append(opt.mode)
    return seen


def cascade_filter_configs(
    options: list[ConfigCascadeOption],
    source: str | None = None,
    model: str | None = None,
    mode: str | None = None,
) -> list[ConfigCascadeOption]:
    return [
        opt
        for opt in options
        if (source is None or opt.source_kind == source)
        and (model is None or not model or opt.model_family == model)
        and (mode is None or not mode or opt.mode == mode)
    ]


def readable_path_label(path_str: str, repo_root: Path) -> str:
    """Human-readable primary label for any config or artifact path."""
    opts = build_cascade_options([path_str], repo_root)
    if not opts:
        return Path(path_str).name
    opt = opts[0]
    parts = []
    if opt.model_family:
        parts.append(opt.model_family)
    if opt.mode:
        parts.append(opt.mode)
    # Avoid duplicating model/mode already present in the experiment label.
    display = opt.display_label
    if opt.model_family and display.startswith(f"{opt.model_family} "):
        display = display
    parts.append(display)
    label = " · ".join(parts)
    if is_raw_run_id(Path(path_str).name) and Path(path_str).name in label:
        # Never present timestamp hashes as the primary readable label.
        meta = parse_config_metadata(path_str, repo_root)
        rebuilt = _experiment_core_label(meta, Path(path_str).name)
        parts = []
        if opt.model_family:
            parts.append(opt.model_family)
        if opt.mode:
            parts.append(opt.mode)
        parts.append(rebuilt)
        label = " · ".join(parts)
    return label


def experiment_selector_label(path_str: str, repo_root: Path) -> str:
    """Final-cascade experiment label (date/time/metric; no raw run id)."""
    opts = build_cascade_options([path_str], repo_root)
    if not opts:
        return Path(path_str).name
    return opts[0].display_label


def enum_human_label(value: str) -> str:
    mapping = {
        "research_v2": "Experiment Core v2",
        "autogluon": "AutoGluon",
        "all": "All sources",
    }
    return mapping.get(value, value.replace("_", " ").title())


def source_human_label(source_kind: str) -> str:
    return _SOURCE_LABEL_MAP.get(source_kind, source_kind)


def model_human_label(model_family: str) -> str:
    return _MODEL_LABEL_MAP.get(model_family, model_family)


def mode_human_label(mode: str) -> str:
    return _MODE_LABEL_MAP.get(mode, mode)
