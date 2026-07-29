from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


_MAX_CONFIG_BYTES = 32 * 1024


def normalize_model_family(raw: str) -> str:
    lower = raw.lower()
    if lower.startswith("xgboost"):
        return "XGBoost"
    if lower.startswith("catboost"):
        return "CatBoost"
    if lower.startswith("manual_lightgbm") or lower.startswith("lightgbm"):
        return "LightGBM"
    if lower.startswith("autogluon"):
        return "AutoGluon"
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


def _read_config_bounded(path: Path) -> dict[str, Any] | None:
    try:
        if path.is_dir():
            for candidate in (
                "study_summary.json",
                "best_trial.json",
                "metrics/aggregate.json",
                "decision_report.json",
                "prediction_summary.json",
                "aggregate_summary.json",
            ):
                nested = path / candidate
                loaded = _read_config_bounded(nested)
                if loaded is not None:
                    return loaded
            return None
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


def parse_config_metadata(path_str: str, repo_root: Path) -> dict[str, str]:
    try:
        full_path = repo_root / path_str
        content = _read_config_bounded(full_path)
        source_kind = normalize_source_kind(path_str, content)
        result: dict[str, str] = {"source_kind": source_kind}

        stem = Path(path_str).stem
        parts = [p for p in Path(path_str).parts if p not in (".",)]
        path_tokens = list(parts) + stem.split("_")

        mode_candidates = ["smoke", "development", "deployment"]
        for part in reversed(path_tokens):
            if part.lower() in mode_candidates:
                result["mode"] = normalize_mode(part)
                break

        if content is not None:
            experiment = content.get("experiment", {})
            if isinstance(experiment, dict):
                exp_id = experiment.get("id", "")
                if exp_id:
                    result["experiment_name"] = str(exp_id)
                model = experiment.get("model", {})
                if isinstance(model, dict):
                    adapter = model.get("adapter", "")
                    if adapter:
                        result["adapter_id"] = str(adapter)
                        result["model_family"] = normalize_model_family(str(adapter))
                cv = experiment.get("cv", {})
                if isinstance(cv, dict):
                    repeats = cv.get("repeats")
                    folds = cv.get("folds")
                    if repeats is not None:
                        result["repeats"] = str(repeats)
                    if folds is not None:
                        result["folds"] = str(folds)

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
            if "status" in content:
                result["status"] = str(content["status"])

            # Optuna study_summary / best_trial payloads under artifact dirs
            if "study_name" in content and "best_trial" not in result:
                result["experiment_name"] = str(content["study_name"])
            if "search_id" in content and "experiment_name" not in result:
                result["experiment_name"] = str(content["search_id"])

        if "model_family" not in result:
            for token in path_tokens:
                family = normalize_model_family(token)
                if family in {"LightGBM", "XGBoost", "CatBoost", "AutoGluon"}:
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


def job_primary_label(record_job: dict, record_commands: dict | None = None) -> str:
    try:
        command_id = record_job.get("command_id", "")
        action_id = record_job.get("action_id", "")
        references = record_job.get("references", {})
        config_path = (
            references.get("config", "") if isinstance(references, dict) else ""
        )

        parts = []
        if config_path:
            meta = parse_config_metadata(str(config_path), Path("."))
            model_family = meta.get("model_family", "")
            mode = meta.get("mode", "")
            if model_family:
                parts.append(model_family)
            if mode:
                parts.append(mode)

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
    "Deployment artifact": "Deployment artifact",
    "Deployment fixture": "Deployment fixture",
    "Other": "Other approved source",
}

_MODEL_LABEL_MAP: dict[str, str] = {
    "LightGBM": "LightGBM",
    "XGBoost": "XGBoost",
    "CatBoost": "CatBoost",
    "AutoGluon": "AutoGluon",
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
            if trial and obj not in ("", "None", "none"):
                try:
                    obj_f = float(obj)
                    label_core = f"trial {trial} · objective {obj_f:.6f}"
                except ValueError:
                    label_core = f"trial {trial} · {obj}"
            elif name:
                label_core = str(name)
            elif trial:
                label_core = f"trial {trial}"
            else:
                label_core = basename
        else:
            stem = Path(path).stem
            desc = _stem_description(stem)
            label_core = desc if desc else (meta.get("experiment_name") or basename)

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
    parts.append(opt.display_label)
    return " · ".join(parts)


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
