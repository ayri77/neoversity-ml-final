from __future__ import annotations

from pathlib import Path

from src.churn_ml.control_panel.presentation import (
    build_pre_run_summary,
    config_badge,
    format_date,
    format_duration,
    format_time,
    job_primary_label,
    mode_badge,
    normalize_model_family,
    normalize_mode,
    normalize_source_kind,
    readable_config_label,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_normalize_model_family_xgboost() -> None:
    assert normalize_model_family("xgboost_dev") == "XGBoost"
    assert normalize_model_family("xgboost") == "XGBoost"


def test_normalize_model_family_lightgbm_manual() -> None:
    assert normalize_model_family("manual_lightgbm_te_v1") == "LightGBM"
    assert normalize_model_family("lightgbm_development") == "LightGBM"


def test_normalize_model_family_catboost() -> None:
    assert normalize_model_family("catboost_development") == "CatBoost"


def test_normalize_model_family_unknown() -> None:
    result = normalize_model_family("some_model")
    assert isinstance(result, str)
    assert len(result) > 0


def test_normalize_mode_smoke() -> None:
    assert normalize_mode("smoke") == "Smoke"
    assert normalize_mode("SMOKE") == "Smoke"


def test_normalize_mode_development() -> None:
    assert normalize_mode("development") == "Development"


def test_format_date_valid() -> None:
    result = format_date("2026-07-29T04:44:07Z")
    assert "29" in result
    assert "Jul" in result
    assert "2026" in result


def test_format_date_none() -> None:
    assert format_date(None) == "—"


def test_format_date_invalid() -> None:
    result = format_date("not-a-date")
    assert result == "Invalid timestamp"


def test_format_time_valid() -> None:
    result = format_time("2026-07-29T04:44:07Z")
    assert result == "04:44:07"


def test_format_duration_none() -> None:
    assert format_duration(None) == "—"


def test_format_duration_seconds() -> None:
    assert format_duration(45) == "45s"
    assert format_duration(0) == "0s"


def test_format_duration_minutes() -> None:
    assert format_duration(305) == "5m 05s"
    assert format_duration(60) == "1m 00s"


def test_format_duration_hours() -> None:
    result = format_duration(3600 + 16 * 60 + 52)
    assert result == "1h 16m 52s"


def test_normalize_source_kind_canonical() -> None:
    assert (
        normalize_source_kind("configs/optuna/catboost_development.yaml")
        == "Canonical config"
    )


def test_normalize_source_kind_optuna_export() -> None:
    assert (
        normalize_source_kind("artifacts/optuna_exports/xgboost_best.yaml")
        == "Optuna export"
    )


def test_normalize_source_kind_ui_copy() -> None:
    assert (
        normalize_source_kind("artifacts/ui_configs/my_copy.yaml") == "UI config copy"
    )


def test_readable_config_label_lightgbm_development() -> None:
    label = readable_config_label(
        "configs/research_v2/manual_lightgbm_te_v1_compat_development.yaml",
        PROJECT_ROOT,
    )
    assert "LightGBM" in label
    assert "Development" in label
    assert "CANONICAL" in label


def test_readable_config_label_fallback() -> None:
    label = readable_config_label("nonexistent/path/some_config.yaml", PROJECT_ROOT)
    assert isinstance(label, str)
    assert len(label) > 0


def test_config_badge_exported() -> None:
    assert config_badge("Optuna export") == "EXPORTED"
    assert config_badge("Canonical config") == "CANONICAL"
    assert config_badge("UI config copy") == "COPY"


def test_mode_badge_smoke() -> None:
    assert mode_badge("Smoke") == "SMOKE"


def test_mode_badge_development() -> None:
    assert mode_badge("Development") == "DEVELOPMENT"


def test_build_pre_run_summary_keys() -> None:
    summary = build_pre_run_summary(
        "experiment_core_v2",
        "run",
        "🧪 Train",
        "Run",
        {},
        PROJECT_ROOT,
    )
    assert "Operation" in summary
    assert "Action" in summary
    assert summary["Operation"] == "🧪 Train"
    assert summary["Action"] == "Run"


def test_normalize_source_kind_generated_deployment_draft() -> None:
    assert (
        normalize_source_kind(
            "artifacts/deployment_drafts/run-abc/deployment_config.yaml"
        )
        == "Generated deployment draft"
    )


def test_build_pre_run_summary_generated_deployment_draft() -> None:
    summary = build_pre_run_summary(
        "final_deployment_v1",
        "validate",
        "📤 Generate submission",
        "Validate",
        {
            "config": (
                "artifacts/deployment_drafts/run-abc/deployment_config.yaml"
            )
        },
        PROJECT_ROOT,
    )
    assert summary["Operation"] == "📤 Generate submission"
    assert summary["Source"] == "Generated deployment draft"
    assert summary["Config type"] == "Deployment"
    assert "[DRAFT]" in summary["Config"]


def test_job_primary_label_basic() -> None:
    job = {"command_id": "experiment_core_v2", "action_id": "run", "references": {}}
    label = job_primary_label(job)
    assert isinstance(label, str)
    assert len(label) > 0


def test_job_primary_label_fallback() -> None:
    job = {"command_id": "", "action_id": ""}
    label = job_primary_label(job)
    assert isinstance(label, str)
