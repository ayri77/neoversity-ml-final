from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import apps.experiment_control_panel as control_panel_app
from streamlit.testing.v1 import AppTest

from src.churn_ml.control_panel.artifacts import artifact_selector_options
from src.churn_ml.control_panel.command_builder import (
    CommandBuildError,
    build_command,
)
from src.churn_ml.control_panel.placeholder_suggestions import (
    SuggestionError,
    render_suggested_value_template,
    sanitize_path_basename,
)
from src.churn_ml.control_panel.process import build_child_environment
from src.churn_ml.control_panel.registry import load_registry
from src.churn_ml.control_panel.schemas import ActionSpec, SchemaError, parse_commands
from src.churn_ml.optuna_search_export import (
    OptunaSearchExportError,
    export_best_candidate,
)
from tests.test_control_panel_security import StartSpy, _run_page, _select


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_optuna_artifact(
    root: Path,
    name: str,
    *,
    state: str,
    best_objective: float | None = None,
) -> Path:
    artifact = root / name
    artifact.mkdir(parents=True, exist_ok=True)
    if state == "completed":
        (artifact / "_SUCCESS").write_text("", encoding="utf-8")
    elif state == "failed":
        (artifact / "_FAILED").write_text("", encoding="utf-8")
    (artifact / "study_summary.json").write_text(
        json.dumps(
            {
                "study_name": name,
                "search_id": name,
                "best_trial_number": 1,
                "best_objective": best_objective,
                "requested_trials": 2,
                "actual_trials": 2 if state == "completed" else 0,
            }
        ),
        encoding="utf-8",
    )
    if best_objective is not None:
        (artifact / "best_trial.json").write_text(
            json.dumps({"objective": best_objective, "trial_number": 1}),
            encoding="utf-8",
        )
    return artifact


def test_windows_home_variables_copied_when_present() -> None:
    parent = {
        "PATH": "safe-path",
        "USERPROFILE": r"C:\Users\example",
        "HOMEDRIVE": "C:",
        "HOMEPATH": r"\Users\example",
        "OPENAI_API_KEY": "ambient-secret",
    }
    environment, sensitive = build_child_environment({}, parent=parent)
    assert environment["USERPROFILE"] == r"C:\Users\example"
    assert environment["HOMEDRIVE"] == "C:"
    assert environment["HOMEPATH"] == r"\Users\example"
    assert "OPENAI_API_KEY" not in environment
    assert sensitive == ()


def test_absent_windows_home_variables_remain_absent() -> None:
    environment, sensitive = build_child_environment({}, parent={"PATH": "safe-path"})
    assert "USERPROFILE" not in environment
    assert "HOMEDRIVE" not in environment
    assert "HOMEPATH" not in environment
    assert sensitive == ()


def test_child_can_resolve_path_home_with_windows_home_vars(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    parent = {
        "PATH": "safe-path",
        "HOME": str(home),
        "USERPROFILE": str(home),
        "HOMEDRIVE": "C:",
        "HOMEPATH": r"\Users\example",
        "SYSTEMROOT": r"C:\Windows",
    }
    environment, _ = build_child_environment({}, parent=parent)
    completed = subprocess.run(
        [sys.executable, "-c", "from pathlib import Path; print(Path.home())"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )
    assert completed.returncode == 0
    resolved = Path(completed.stdout.strip()).resolve()
    assert resolved == home.resolve()


def test_apptest_config_options_follow_command_and_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "mlflow_local_index")
    config = next(item for item in at.selectbox if item.label == "Config")
    assert "configs/mlflow/local.yaml" in list(config.options)
    assert config.value == "configs/mlflow/local.yaml"

    at = _select(at, "Operation", "optuna_search_v1")
    config = next(item for item in at.selectbox if item.label == "Config")
    assert "configs/mlflow/local.yaml" not in list(config.options)
    assert all(
        str(option).startswith("configs/optuna/")
        or str(option).startswith("artifacts/ui_configs/")
        for option in config.options
    )
    assert config.value in list(config.options)

    at = _select(at, "Action", "run")
    config = next(item for item in at.selectbox if item.label == "Config")
    assert config.value in list(config.options)


def test_stale_config_session_value_is_discarded() -> None:
    options = control_panel_app._config_options(
        load_registry(PROJECT_ROOT),
        ("configs/optuna/*.yaml",),
    )
    assert options
    stale = "configs/mlflow/local.yaml"
    assert stale not in options
    selected = stale if stale in options else options[0]
    assert selected == options[0]


def test_artifact_selector_lists_only_completed_optuna(tmp_path: Path) -> None:
    searches = tmp_path / "artifacts" / "optuna_searches"
    _write_optuna_artifact(
        searches, "completed_search", state="completed", best_objective=0.91
    )
    _write_optuna_artifact(searches, "failed_search", state="failed")
    _write_optuna_artifact(searches, "running_search", state="running")
    loaded = load_registry(PROJECT_ROOT)
    options = artifact_selector_options(
        tmp_path,
        loaded.readers["optuna_search_v1"],
        roots=("artifacts/optuna_searches",),
        statuses=("completed",),
    )
    assert [path for path, _label in options] == [
        "artifacts/optuna_searches/completed_search"
    ]
    label = options[0][1]
    assert "completed_search" in label
    assert "best=0.91" in label
    assert "completed" in label


def test_inspect_and_export_artifact_paths_build_exact_argv() -> None:
    loaded = load_registry(PROJECT_ROOT)
    search_dir = PROJECT_ROOT / "artifacts" / "optuna_searches" / "ux_selector_probe"
    search_dir.mkdir(parents=True, exist_ok=True)
    export_root = PROJECT_ROOT / "artifacts" / "optuna_exports"
    export_root.mkdir(parents=True, exist_ok=True)
    try:
        inspect_built = build_command(
            loaded.commands,
            "optuna_search_v1",
            "inspect",
            {"search_dir": "artifacts/optuna_searches/ux_selector_probe"},
            repository_root=PROJECT_ROOT,
            python_executable="python-safe",
        )
        assert inspect_built.argv[-1] == "artifacts/optuna_searches/ux_selector_probe"
        export_built = build_command(
            loaded.commands,
            "optuna_search_v1",
            "export_best",
            {
                "search_dir": "artifacts/optuna_searches/ux_selector_probe",
                "output": "artifacts/optuna_exports/ux_selector_probe_best.yaml",
            },
            repository_root=PROJECT_ROOT,
            python_executable="python-safe",
        )
        assert export_built.argv[-1] == (
            "artifacts/optuna_exports/ux_selector_probe_best.yaml"
        )
        with pytest.raises(CommandBuildError):
            build_command(
                loaded.commands,
                "optuna_search_v1",
                "inspect",
                {"search_dir": "artifacts/optuna_searches/../ui_configs"},
                repository_root=PROJECT_ROOT,
            )
        with pytest.raises(CommandBuildError):
            build_command(
                loaded.commands,
                "optuna_search_v1",
                "inspect",
                {"search_dir": "artifacts/optuna_searches/missing_manual"},
                repository_root=PROJECT_ROOT,
            )
    finally:
        if search_dir.exists():
            search_dir.rmdir()
        for path in (export_root, search_dir.parent):
            if path.exists() and not any(path.iterdir()):
                path.rmdir()


def test_suggested_export_path_updates_and_rejects_unsafe() -> None:
    assert (
        render_suggested_value_template(
            "artifacts/optuna_exports/{search_dir.basename}_best.yaml",
            {"search_dir": "artifacts/optuna_searches/study_a"},
        )
        == "artifacts/optuna_exports/study_a_best.yaml"
    )
    assert (
        render_suggested_value_template(
            "artifacts/optuna_exports/{search_dir.basename}_best.yaml",
            {"search_dir": "artifacts/optuna_searches/study_b"},
        )
        == "artifacts/optuna_exports/study_b_best.yaml"
    )
    assert sanitize_path_basename("artifacts/optuna_searches/../escape") == "escape"
    with pytest.raises(SuggestionError):
        sanitize_path_basename("..")
    with pytest.raises(SuggestionError):
        sanitize_path_basename("")
    payload = yaml.safe_load(
        (PROJECT_ROOT / "configs/ui/ui_commands.yaml").read_text(encoding="utf-8")
    )
    optuna = next(
        item for item in payload["commands"] if item["id"] == "optuna_search_v1"
    )
    action = next(item for item in optuna["actions"] if item["id"] == "export_best")
    action["placeholders"]["output"]["suggested_value_template"] = (
        "artifacts/optuna_exports/{search_dir.basename}/../evil.yaml"
    )
    with pytest.raises(SchemaError):
        parse_commands(payload)


def test_export_best_allows_optuna_exports_and_refuses_existing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded = load_registry(PROJECT_ROOT)
    output_spec = (
        loaded.commands["optuna_search_v1"]
        .actions["export_best"]
        .placeholders["output"]
    )
    assert "artifacts/optuna_exports" in output_spec.roots
    assert output_spec.suggested_value_template == (
        "artifacts/optuna_exports/{search_dir.basename}_best.yaml"
    )
    existing = tmp_path / "already.yaml"
    existing.write_text("name: existing\n", encoding="utf-8")

    class _FakeResult:
        best_candidate_config = {"experiment": {"id": "x"}}

    monkeypatch.setattr(
        "src.churn_ml.optuna_search_artifacts.load_optuna_search_result",
        lambda *_args, **_kwargs: _FakeResult(),
    )
    with pytest.raises(OptunaSearchExportError, match="already exists"):
        export_best_candidate(tmp_path / "search", existing, project_root=tmp_path)


def test_results_action_rows_include_export_best() -> None:
    loaded = load_registry(PROJECT_ROOT)
    command = loaded.commands["optuna_search_v1"]
    rows = [
        {
            "action": action.title,
            "enabled": action.enabled,
            "where": "Run page",
            "additional_input_required": (
                control_panel_app._action_requires_additional_input(action)
            ),
        }
        for action in command.actions.values()
        if action.enabled
    ]
    titles = {row["action"] for row in rows}
    assert "Inspect" in titles
    assert "Export best candidate" in titles
    assert "New study" in titles
    export_row = next(row for row in rows if row["action"] == "Export best candidate")
    assert export_row["additional_input_required"] is True
    inspect_row = next(row for row in rows if row["action"] == "Inspect")
    assert inspect_row["additional_input_required"] is False


def _apptest_results_page() -> None:
    import apps.experiment_control_panel as panel

    panel.results_page()


def test_apptest_side_by_side_defaults_to_distinct_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    searches = tmp_path / "artifacts" / "optuna_searches"
    _write_optuna_artifact(searches, "alpha_search", state="completed")
    _write_optuna_artifact(searches, "beta_search", state="completed")
    monkeypatch.setattr(control_panel_app, "REPOSITORY_ROOT", tmp_path)
    original_load = control_panel_app.load_registry

    def load_project_registry(root: Path):
        del root
        return original_load(PROJECT_ROOT)

    monkeypatch.setattr(control_panel_app, "load_registry", load_project_registry)
    control_panel_app.registry.clear()
    at = AppTest.from_function(_apptest_results_page, default_timeout=10).run()
    assert not at.exception
    reader = next(item for item in at.selectbox if item.label == "Artifact type")
    reader.select("optuna_search_v1")
    at = at.run()
    assert not at.exception
    left = next(item for item in at.selectbox if item.label == "Left")
    right = next(item for item in at.selectbox if item.label == "Right")
    assert left.value != right.value
    assert left.value in left.options
    assert right.value in right.options


def test_action_requires_additional_input_helper() -> None:
    inspect = ActionSpec.from_dict(
        {
            "id": "inspect",
            "title": "Inspect",
            "description": "d",
            "argv": ["inspect", "{search_dir}"],
            "placeholders": {
                "search_dir": {
                    "type": "path",
                    "role": "input",
                    "required": True,
                    "roots": ["artifacts/optuna_searches"],
                }
            },
            "confirmation": "none",
            "enabled": True,
            "competition_test": False,
            "success_markers": [],
            "failure_markers": [],
        },
        "action",
    )
    assert control_panel_app._action_requires_additional_input(inspect) is False
