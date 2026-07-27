"""Read-only inspection and worker-side inspection export for AutoGluon runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.churn_ml.autogluon_artifacts import discover_model_directories, write_json


def inspect_run(
    run_dir: Path,
    *,
    attempt_load: bool | None = None,
    predictor_loader: Any | None = None,
) -> dict[str, Any]:
    """Inspect status first and load only when explicitly requested or safely complete."""
    run_dir = run_dir.resolve(strict=True)
    success = (run_dir / "_SUCCESS").is_file()
    failed = (run_dir / "_FAILED").is_file()
    status = _read_json(run_dir / "execution_status.json")
    predictor_dir = run_dir / "predictor"
    structurally_complete = all(
        (predictor_dir / name).is_file()
        for name in ("predictor.pkl", "learner.pkl", "version.txt")
    )
    should_load = (
        success and not failed and structurally_complete
        if attempt_load is None
        else attempt_load
    )
    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "classification": (
            "complete"
            if success and not failed
            else "failed_or_partial"
            if failed
            else "incomplete"
        ),
        "success_marker": success,
        "failed_marker": failed,
        "execution_status": status,
        "predictor_structure_complete": structurally_complete,
        "discovered_model_directories": discover_model_directories(predictor_dir),
        "predictor_loading_attempted": should_load,
        "predictor_loading_succeeded": False,
        "load_error": None,
    }
    if not should_load:
        return report

    try:
        loader = predictor_loader or _load_predictor
        predictor = loader(predictor_dir)
        report["predictor"] = predictor_report(predictor)
        report["predictor_loading_succeeded"] = True
    except Exception as error:  # inspection must report corrupt/partial predictors
        report["load_error"] = f"{type(error).__name__}: {error}"
    return report


def export_worker_inspection(run_dir: Path, predictor: Any) -> dict[str, Any]:
    """Export inspection artifacts after a completed fit, before worker completion."""
    inspection_dir = run_dir / "inspection"
    inspection_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = predictor.leaderboard(silent=True)
    leaderboard.to_csv(inspection_dir / "leaderboard.csv", index=False)
    summary = predictor_report(predictor)
    write_json(inspection_dir / "summary.json", summary)
    return summary


def predictor_report(predictor: Any) -> dict[str, Any]:
    """Gather metadata through public predictor APIs with best-effort weights."""
    models = list(predictor.model_names())
    best_model = getattr(predictor, "model_best", None)
    leaderboard_error: str | None = None
    try:
        leaderboard = predictor.leaderboard(silent=True).to_dict(orient="records")
    except Exception as error:
        leaderboard = []
        leaderboard_error = f"{type(error).__name__}: {error}"
    info_error: str | None = None
    try:
        info = predictor.info()
    except Exception as error:
        info = {}
        info_error = f"{type(error).__name__}: {error}"
    weights = _find_ensemble_weights(info, best_model)
    return {
        "decision_threshold": _json_safe(
            getattr(predictor, "decision_threshold", None)
        ),
        "models": models,
        "best_model": best_model,
        "leaderboard": _json_safe(leaderboard),
        "leaderboard_error": leaderboard_error,
        "ensemble_weights": weights,
        "ensemble_weights_source": (
            "best-effort traversal of public TabularPredictor.info()"
            if weights
            else None
        ),
        "predictor_info": _json_safe(info),
        "predictor_info_error": info_error,
    }


def _load_predictor(predictor_dir: Path) -> Any:
    from autogluon.tabular import TabularPredictor  # type: ignore[import-not-found]

    return TabularPredictor.load(str(predictor_dir))


def _find_ensemble_weights(
    info: Any, best_model: str | None
) -> dict[str, float] | None:
    if not isinstance(info, dict):
        return None
    model_info = info.get("model_info")
    root: Any = model_info.get(best_model, {}) if isinstance(model_info, dict) else info
    stack = [root]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"model_weights", "weights"} and isinstance(child, dict):
                    numeric = {
                        str(name): float(weight)
                        for name, weight in child.items()
                        if type(weight) in {int, float}
                    }
                    if numeric:
                        return numeric
                stack.append(child)
        elif isinstance(value, list):
            stack.extend(value)
    return None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(child) for child in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)
