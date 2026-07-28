from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.churn_ml.control_panel.command_builder import (
    CommandBuildError,
    build_command,
)
from src.churn_ml.control_panel.process import build_child_environment
from src.churn_ml.control_panel.registry import load_registry
from src.churn_ml.control_panel.schemas import SchemaError, parse_commands


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_optuna_validate_builds_without_authority_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CHURN_ML_OPTUNA_LIFECYCLE_AUTHORITY_KEY_FILE", raising=False)
    loaded = load_registry(PROJECT_ROOT)
    built = build_command(
        loaded.commands,
        "optuna_search_v1",
        "validate",
        {"config": "configs/optuna/xgboost_numeric_v1_smoke.yaml"},
        repository_root=PROJECT_ROOT,
        python_executable="python-safe",
    )
    assert built.argv == (
        "python-safe",
        "-u",
        "scripts/run_optuna_search.py",
        "validate",
        "--config",
        "configs/optuna/xgboost_numeric_v1_smoke.yaml",
    )
    environment, sensitive = build_child_environment(
        loaded.commands["optuna_search_v1"].environment,
        parent={"PATH": "safe-path"},
    )
    assert "CHURN_ML_OPTUNA_LIFECYCLE_AUTHORITY_KEY_FILE" not in environment
    assert sensitive == ()


def test_optuna_runtime_environment_passes_sensitive_key_without_display(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    key_path = tmp_path / "outside-authority.key"
    key_path.write_bytes(os.urandom(32))
    monkeypatch.setenv("CHURN_ML_OPTUNA_LIFECYCLE_AUTHORITY_KEY_FILE", str(key_path))
    loaded = load_registry(PROJECT_ROOT)
    environment, sensitive = build_child_environment(
        loaded.commands["optuna_search_v1"].environment
    )
    assert environment["CHURN_ML_OPTUNA_LIFECYCLE_AUTHORITY_KEY_FILE"] == str(key_path)
    assert sensitive == (str(key_path),)

    built = build_command(
        loaded.commands,
        "optuna_search_v1",
        "run",
        {"config": "configs/optuna/catboost_numeric_v1_smoke.yaml"},
        repository_root=PROJECT_ROOT,
        python_executable="python-safe",
    )
    assert "CHURN_ML_OPTUNA_LIFECYCLE_AUTHORITY_KEY_FILE" not in " ".join(built.argv)
    assert str(key_path) not in " ".join(built.redacted_argv)


def test_optuna_authority_init_redacts_external_output(tmp_path: Path) -> None:
    outside = tmp_path / "secrets" / "optuna-lifecycle-authority.key"
    loaded = load_registry(PROJECT_ROOT)
    built = build_command(
        loaded.commands,
        "optuna_search_v1",
        "authority_init",
        {"output": str(outside)},
        repository_root=PROJECT_ROOT,
        python_executable="python-safe",
    )
    assert built.argv == (
        "python-safe",
        "-u",
        "scripts/run_optuna_search.py",
        "authority-init",
        "--output",
        str(outside.resolve(strict=False)),
    )
    assert built.redacted_argv[-1] == "<redacted>"
    assert built.references["output"] == "<redacted>"


def test_optuna_authority_init_rejects_repository_relative_output() -> None:
    loaded = load_registry(PROJECT_ROOT)
    with pytest.raises(CommandBuildError, match="absolute"):
        build_command(
            loaded.commands,
            "optuna_search_v1",
            "authority_init",
            {"output": "artifacts/ui_configs/authority.key"},
            repository_root=PROJECT_ROOT,
        )


def test_optuna_authority_init_rejects_in_repository_absolute_output() -> None:
    loaded = load_registry(PROJECT_ROOT)
    inside = PROJECT_ROOT / "artifacts" / "ui_configs" / "authority.key"
    with pytest.raises(CommandBuildError, match="outside the repository"):
        build_command(
            loaded.commands,
            "optuna_search_v1",
            "authority_init",
            {"output": str(inside)},
            repository_root=PROJECT_ROOT,
        )


def test_optuna_export_and_inspect_actions_build() -> None:
    loaded = load_registry(PROJECT_ROOT)
    search_dir = PROJECT_ROOT / "artifacts" / "optuna_searches" / "example_search"
    search_dir.mkdir(parents=True, exist_ok=True)
    try:
        inspect_built = build_command(
            loaded.commands,
            "optuna_search_v1",
            "inspect",
            {"search_dir": "artifacts/optuna_searches/example_search"},
            repository_root=PROJECT_ROOT,
            python_executable="python-safe",
        )
        assert inspect_built.argv[3:6] == (
            "inspect",
            "--search-dir",
            "artifacts/optuna_searches/example_search",
        )
        export_built = build_command(
            loaded.commands,
            "optuna_search_v1",
            "export_best",
            {
                "search_dir": "artifacts/optuna_searches/example_search",
                "output": "artifacts/ui_configs/optuna_best_candidate.yaml",
            },
            repository_root=PROJECT_ROOT,
            python_executable="python-safe",
        )
        assert export_built.argv[3:] == (
            "export-best",
            "--search-dir",
            "artifacts/optuna_searches/example_search",
            "--output",
            "artifacts/ui_configs/optuna_best_candidate.yaml",
        )
    finally:
        if search_dir.exists():
            search_dir.rmdir()
        parent = search_dir.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()


def test_optuna_external_absolute_schema_rejects_roots() -> None:
    payload = yaml.safe_load(
        (PROJECT_ROOT / "configs/ui/ui_commands.yaml").read_text(encoding="utf-8")
    )
    optuna = next(
        item for item in payload["commands"] if item["id"] == "optuna_search_v1"
    )
    action = next(item for item in optuna["actions"] if item["id"] == "authority_init")
    action["placeholders"]["output"]["roots"] = ["artifacts/ui_configs"]
    with pytest.raises(SchemaError, match="roots must be empty"):
        parse_commands(payload)


def test_optuna_public_cli_has_no_candidate_validate_only() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_optuna_search.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0
    assert "{authority-init,validate,run,inspect,export-best}" in completed.stdout
    invalid = subprocess.run(
        [sys.executable, "scripts/run_optuna_search.py", "candidate", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert invalid.returncode != 0
    assert "invalid choice: 'candidate'" in invalid.stdout + invalid.stderr


def test_optuna_registry_defaults_to_validate_action() -> None:
    loaded = load_registry(PROJECT_ROOT)
    action_ids = list(loaded.commands["optuna_search_v1"].actions)
    assert action_ids[0] == "validate"
    assert "authority_init" in action_ids
