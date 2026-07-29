from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import psutil

import src.churn_ml.control_panel.process as process_module
from streamlit.testing.v1 import AppTest

import apps.experiment_control_panel as control_panel_app
from src.churn_ml.control_panel.command_builder import build_command
from src.churn_ml.control_panel.jobs import JobManager
from src.churn_ml.control_panel.launch import (
    LaunchAuthorizationError,
    RenderedLaunch,
    authorize_launch,
    rendered_launch,
)
from src.churn_ml.control_panel.process import (
    ProcessIdentityError,
    ProcessIdentity,
    argv_fingerprint,
    verify_process_identity,
    build_child_environment,
)
from src.churn_ml.control_panel.registry import load_registry
from src.churn_ml.control_panel.schemas import EnvironmentVariableSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StartSpy:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def start(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return type("Record", (), {"job_id": "spy-job"})()


def _select(at: AppTest, label: str, value: str) -> AppTest:
    element = next(item for item in at.selectbox if item.label == label)
    element.select(value)
    return at.run()


def _check(at: AppTest, prefix: str) -> AppTest:
    element = next(item for item in at.checkbox if item.label.startswith(prefix))
    element.check()
    return at.run()


def _click(at: AppTest, label: str) -> AppTest:
    element = next(item for item in at.button if item.label == label)
    element.click()
    return at.run()


def _apptest_run_page() -> None:
    # Inline import is required: AppTest.from_function re-executes only this
    # function's source in an isolated module, so module-level names are absent.
    import apps.experiment_control_panel as panel

    panel.run_page()


def _run_page(monkeypatch: pytest.MonkeyPatch, spy: StartSpy) -> AppTest:
    monkeypatch.setattr(control_panel_app, "job_manager", lambda loaded: spy)
    return AppTest.from_function(_apptest_run_page, default_timeout=10).run()


def test_apptest_disabled_action_and_crafted_session_never_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "final_deployment_v1")
    at = _select(at, "Action", "run")
    at = _check(at, "I confirm")
    at = _check(at, "I explicitly acknowledge")
    at.session_state["enabled"] = True
    _click(at, "Start background job")
    assert spy.calls == []


def test_apptest_missing_normal_confirmation_never_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Action", "run")
    confirm = next(item for item in at.checkbox if item.label.startswith("I confirm"))
    assert confirm.value is False
    _click(at, "Start background job")
    assert spy.calls == []


def test_apptest_missing_high_risk_acknowledgement_never_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load_registry(PROJECT_ROOT)
    deployment = loaded.commands["final_deployment_v1"]
    enabled_run = replace(deployment.actions["run"], enabled=True)
    enabled_deployment = replace(
        deployment, actions={**deployment.actions, "run": enabled_run}
    )
    loaded = replace(
        loaded,
        commands={**loaded.commands, "final_deployment_v1": enabled_deployment},
    )
    spy = StartSpy()
    monkeypatch.setattr(control_panel_app, "registry", lambda: loaded)
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "final_deployment_v1")
    at = _select(at, "Action", "run")
    at = _check(at, "I confirm")
    _click(at, "Start background job")
    assert spy.calls == []


def test_apptest_failed_build_and_stale_action_never_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = StartSpy()
    # Empty repository root so paired-comparison inputs cannot be auto-selected.
    monkeypatch.setattr(control_panel_app, "REPOSITORY_ROOT", tmp_path)
    original_load = control_panel_app.load_registry

    def load_project_registry(root: Path):
        del root
        return original_load(PROJECT_ROOT)

    monkeypatch.setattr(control_panel_app, "load_registry", load_project_registry)
    control_panel_app.registry.clear()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "paired_comparison")
    _click(at, "Start background job")
    assert spy.calls == []

    monkeypatch.setattr(
        control_panel_app,
        "rendered_launch",
        lambda built, previous, *, consumed_nonces=None: RenderedLaunch(
            "stale-command",
            built.action_id,
            "0" * 64,
            "00000000-0000-0000-0000-000000000001",
        ),
    )
    at = _run_page(monkeypatch, spy)
    _click(at, "Start background job")
    assert spy.calls == []


def test_apptest_legitimate_launch_once_and_replay_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Action", "run")
    at = _check(at, "I confirm")
    at = _click(at, "Start background job")
    assert len(spy.calls) == 1
    consumed = at.session_state["_consumed_launch_nonces"]
    assert isinstance(consumed, set)
    assert consumed
    forged_nonce = next(iter(consumed))

    def forge_consumed_token(
        built: Any,
        previous: RenderedLaunch | None,
        *,
        consumed_nonces: Any = None,
    ) -> RenderedLaunch:
        del previous, consumed_nonces
        return RenderedLaunch(
            built.command_id,
            built.action_id,
            argv_fingerprint(built.argv),
            forged_nonce,
        )

    monkeypatch.setattr(control_panel_app, "rendered_launch", forge_consumed_token)
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Action", "run")
    at = _check(at, "I confirm")
    # Restore the consumed-nonce set after the fresh AppTest session.
    at.session_state["_consumed_launch_nonces"] = set(consumed)
    _click(at, "Start background job")
    assert len(spy.calls) == 1


def test_apptest_optuna_defaults_to_validate_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = StartSpy()
    at = _run_page(monkeypatch, spy)
    at = _select(at, "Operation", "optuna_search_v1")
    action = next(item for item in at.selectbox if item.label == "Action")
    assert action.value == "validate"


def test_authorization_revalidates_current_paths_and_consumes_nonce() -> None:
    loaded = load_registry(PROJECT_ROOT)
    values = {"config": "configs/research_v2/catboost_numeric_v1_smoke.yaml"}
    built = build_command(
        loaded.commands,
        "experiment_core_v2",
        "run",
        values,
        repository_root=PROJECT_ROOT,
    )
    rendered = rendered_launch(built, None)
    consumed: set[str] = set()
    authorized = authorize_launch(
        loaded.commands,
        command_id="experiment_core_v2",
        action_id="run",
        values=values,
        repository_root=PROJECT_ROOT,
        confirmed=True,
        high_risk_acknowledged=True,
        rendered=rendered,
        consumed_nonces=consumed,
    )
    assert authorized == built
    with pytest.raises(LaunchAuthorizationError, match="already consumed"):
        authorize_launch(
            loaded.commands,
            command_id="experiment_core_v2",
            action_id="run",
            values=values,
            repository_root=PROJECT_ROOT,
            confirmed=True,
            high_risk_acknowledged=True,
            rendered=rendered,
            consumed_nonces=consumed,
        )


def test_explicit_environment_allowlist_excludes_ambient_credentials() -> None:
    declared = {
        "CONTROL_PANEL_REQUIRED": EnvironmentVariableSpec(
            name="CONTROL_PANEL_REQUIRED", required=True, sensitive=False
        ),
        "CONTROL_PANEL_AUTHORITY_KEY": EnvironmentVariableSpec(
            name="CONTROL_PANEL_AUTHORITY_KEY", required=True, sensitive=True
        ),
    }
    parent = {
        "PATH": "safe-path",
        "CONTROL_PANEL_REQUIRED": "present",
        "CONTROL_PANEL_AUTHORITY_KEY": "authority-canary",
        "OPENAI_API_KEY": "ambient-canary",
        "GITHUB_TOKEN": "ambient-github-canary",
    }
    environment, sensitive = build_child_environment(declared, parent=parent)
    assert environment["CONTROL_PANEL_REQUIRED"] == "present"
    assert environment["CONTROL_PANEL_AUTHORITY_KEY"] == "authority-canary"
    assert "OPENAI_API_KEY" not in environment
    assert "GITHUB_TOKEN" not in environment
    assert sensitive == ("authority-canary",)
    with pytest.raises(ProcessIdentityError, match="absent"):
        build_child_environment(declared, parent={"PATH": "safe-path"})


def test_child_receives_declared_environment_only_and_sensitive_value_never_persists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    required = "CONTROL_PANEL_REQUIRED"
    authority = "CONTROL_PANEL_AUTHORITY_KEY"
    authority_secret = "AUTHORITY_CANARY_1e7c"
    ambient_secret = "AMBIENT_OPENAI_CANARY_93bc"
    monkeypatch.setenv(required, "present")
    monkeypatch.setenv(authority, authority_secret)
    monkeypatch.setenv("OPENAI_API_KEY", ambient_secret)
    loaded = load_registry(PROJECT_ROOT)
    command = loaded.commands["experiment_core_v2"]
    command = replace(
        command,
        environment={
            required: EnvironmentVariableSpec(required, True, False),
            authority: EnvironmentVariableSpec(authority, True, True),
        },
    )
    commands = {**loaded.commands, command.id: command}
    manager = JobManager(
        tmp_path / "jobs",
        working_directory=tmp_path,
        commands=commands,
    )
    code = (
        "import os; print(os.environ.get('CONTROL_PANEL_REQUIRED')); "
        "print(os.environ.get('CONTROL_PANEL_AUTHORITY_KEY')); "
        "print(os.environ.get('OPENAI_API_KEY', '<missing>'))"
    )
    record = manager.start(
        argv=[sys.executable, "-c", code],
        redacted_argv=[sys.executable, "-c", "<environment probe>"],
        command_id=command.id,
        action_id="validate",
        references={},
    )
    deadline = time.monotonic() + 10
    while (
        manager.refresh(record.job_id).status["state"] == "running"
        and time.monotonic() < deadline
    ):
        time.sleep(0.02)
    time.sleep(0.1)
    stdout = (record.root / "stdout.log").read_text(encoding="utf-8")
    assert "present" in stdout
    assert "<redacted>" in stdout
    assert "<missing>" in stdout
    for path in record.root.iterdir():
        text = path.read_text(encoding="utf-8", errors="replace")
        assert authority_secret not in text
        assert ambient_secret not in text


class _InspectableProcess:
    def __init__(self, *, denied: bool = False) -> None:
        self.denied = denied

    def status(self) -> str:
        return psutil.STATUS_RUNNING

    def create_time(self) -> float:
        if self.denied:
            raise psutil.AccessDenied(pid=41001)
        return 1_700_000_000.25

    def exe(self) -> str:
        return sys.executable

    def cmdline(self) -> list[str]:
        return [sys.executable, "-c", "harmless"]


def _identity() -> ProcessIdentity:
    argv = [sys.executable, "-c", "harmless"]
    return ProcessIdentity(
        pid=41001,
        creation_time_utc=process_module._canonical_creation_time(1_700_000_000.25),
        executable_path=process_module.canonical_executable(sys.executable),
        argv_sha256=argv_fingerprint(argv),
        process_group=None,
        session_id=None,
    )


def test_process_identity_verifier_matches_all_persisted_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        process_module.psutil, "Process", lambda pid: _InspectableProcess()
    )
    assert verify_process_identity(_identity()).state == "matching"


def test_reused_pid_with_different_creation_time_is_mismatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        process_module.psutil, "Process", lambda pid: _InspectableProcess()
    )
    identity = replace(_identity(), creation_time_utc="2026-01-01T00:00:00.000000Z")
    result = verify_process_identity(identity)
    assert result.state == "mismatched"
    assert "creation time" in str(result.diagnostic)


def test_matching_pid_with_different_executable_is_mismatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        process_module.psutil, "Process", lambda pid: _InspectableProcess()
    )
    identity = replace(_identity(), executable_path=str(PROJECT_ROOT / "different.exe"))
    result = verify_process_identity(identity)
    assert result.state == "mismatched"
    assert "executable" in str(result.diagnostic)


def test_process_access_denial_is_unverifiable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        process_module.psutil,
        "Process",
        lambda pid: _InspectableProcess(denied=True),
    )
    result = verify_process_identity(_identity())
    assert result.state == "unverifiable"
    assert "denied" in str(result.diagnostic)
