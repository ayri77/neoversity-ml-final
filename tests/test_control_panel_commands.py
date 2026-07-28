from __future__ import annotations

import subprocess
from dataclasses import replace
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import src.churn_ml.control_panel.process as process_module
from src.churn_ml.control_panel.command_builder import (
    CommandBuildError,
    build_command,
)
from src.churn_ml.control_panel.process import (
    LocalProcessBackend,
    ProcessIdentity,
    build_child_environment,
)
from src.churn_ml.control_panel.registry import load_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_allowlist_and_placeholder_substitution() -> None:
    loaded = load_registry(PROJECT_ROOT)
    built = build_command(
        loaded.commands,
        "experiment_core_v2",
        "validate",
        {"config": "configs/research_v2/catboost_numeric_v1_smoke.yaml"},
        repository_root=PROJECT_ROOT,
        python_executable="python-safe",
    )
    assert built.argv == (
        "python-safe",
        "-u",
        "scripts/run_research_v2.py",
        "--config",
        "configs/research_v2/catboost_numeric_v1_smoke.yaml",
        "--validate-only",
    )


def test_unknown_commands_actions_and_missing_placeholders_are_rejected() -> None:
    loaded = load_registry(PROJECT_ROOT)
    with pytest.raises(CommandBuildError, match="Unknown command"):
        build_command(
            loaded.commands,
            "arbitrary",
            "run",
            {},
            repository_root=PROJECT_ROOT,
        )
    with pytest.raises(CommandBuildError, match="Unknown action"):
        build_command(
            loaded.commands,
            "experiment_core_v2",
            "arbitrary",
            {},
            repository_root=PROJECT_ROOT,
        )
    with pytest.raises(CommandBuildError, match="Missing required"):
        build_command(
            loaded.commands,
            "experiment_core_v2",
            "run",
            {},
            repository_root=PROJECT_ROOT,
        )


def test_disabled_deployment_action_cannot_be_built() -> None:
    loaded = load_registry(PROJECT_ROOT)
    with pytest.raises(CommandBuildError, match="disabled"):
        build_command(
            loaded.commands,
            "final_deployment_v1",
            "run",
            {"config": "artifacts/ui_configs/deployment.yaml"},
            repository_root=PROJECT_ROOT,
        )


def test_injection_text_remains_one_inert_argv_value(tmp_path: Path) -> None:
    backend = LocalProcessBackend()
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    injected = "value; Write-Output INJECTION"
    environment, _ = build_child_environment({})
    spawned = backend.spawn(
        [
            sys.executable,
            "-c",
            "import sys; print(repr(sys.argv[1]))",
            injected,
        ],
        cwd=tmp_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        environment=environment,
        redactions=(),
    )
    process = backend._processes[spawned.pid]
    assert process.wait(timeout=10) == 0
    for thread in backend._log_threads:
        thread.join(timeout=5)
    assert stdout_path.read_text(encoding="utf-8").strip() == repr(injected)
    assert not (tmp_path / "INJECTION").exists()


def test_process_backend_hard_codes_shell_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class DummyPopen:
        pid = 123
        stdout = object()
        stderr = object()

        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(subprocess, "Popen", DummyPopen)
    backend = LocalProcessBackend()
    identity = ProcessIdentity(
        pid=123,
        creation_time_utc="2026-01-01T00:00:00.000000Z",
        executable_path=str(Path(sys.executable).resolve()),
        argv_sha256="0" * 64,
        process_group=None,
        session_id=None,
    )
    monkeypatch.setattr(process_module, "_capture_identity", lambda *args: identity)
    monkeypatch.setattr(backend, "_start_log_drain", lambda *args: None)
    backend.spawn(
        ["safe"],
        cwd=tmp_path,
        stdout_path=tmp_path / "out",
        stderr_path=tmp_path / "err",
        environment={},
        redactions=(),
    )
    assert captured["shell"] is False


@pytest.mark.parametrize("unsafe", ["../secret.yaml", "C:/secret.yaml", "/secret.yaml"])
def test_traversal_and_absolute_paths_are_rejected(unsafe: str) -> None:
    loaded = load_registry(PROJECT_ROOT)
    with pytest.raises(CommandBuildError):
        build_command(
            loaded.commands,
            "experiment_core_v2",
            "validate",
            {"config": unsafe},
            repository_root=PROJECT_ROOT,
        )


def test_config_must_match_declared_glob_not_only_root() -> None:
    loaded = load_registry(PROJECT_ROOT)
    with pytest.raises(CommandBuildError, match="allowed globs"):
        build_command(
            loaded.commands,
            "experiment_core_v2",
            "validate",
            {"config": "configs/research_v2/plans/telecom_v3_smoke_r1x3_t2_v1.yaml"},
            repository_root=PROJECT_ROOT,
        )


def test_symlink_or_reparse_config_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = load_registry(PROJECT_ROOT)
    config_dir = tmp_path / "configs/research_v2"
    config_dir.mkdir(parents=True)
    target = config_dir / "target.yaml"
    target.write_text("schema_version: 1\n", encoding="utf-8")
    link = config_dir / "link.yaml"
    try:
        link.symlink_to(target)
    except OSError:
        link.write_text("schema_version: 1\n", encoding="utf-8")
        original_lstat = Path.lstat

        def fake_lstat(path: Path) -> Any:
            result = original_lstat(path)
            if path == link:
                return SimpleNamespace(st_mode=result.st_mode, st_file_attributes=0x400)
            return result

        monkeypatch.setattr(Path, "lstat", fake_lstat)
    with pytest.raises(CommandBuildError, match="Symlink|reparse"):
        build_command(
            loaded.commands,
            "experiment_core_v2",
            "validate",
            {"config": "configs/research_v2/link.yaml"},
            repository_root=tmp_path,
        )


def test_input_output_overlap_is_rejected(tmp_path: Path) -> None:
    loaded = load_registry(PROJECT_ROOT)
    command = loaded.commands["paired_comparison"]
    action = command.actions["run"]
    output = replace(
        action.placeholders["output_root"],
        roots=("artifacts/research_v2",),
    )
    action = replace(
        action,
        placeholders={**action.placeholders, "output_root": output},
    )
    command = replace(command, actions={"run": action})
    for name in ("baseline", "candidate"):
        (tmp_path / "artifacts/research_v2" / name).mkdir(parents=True)
    with pytest.raises(CommandBuildError, match="overlap"):
        build_command(
            {"paired_comparison": command},
            "paired_comparison",
            "run",
            {
                "baseline_run_dir": "artifacts/research_v2/baseline",
                "candidate_run_dir": "artifacts/research_v2/candidate",
                "comparison_id": "test-comparison",
                "output_root": "artifacts/research_v2",
            },
            repository_root=tmp_path,
        )
