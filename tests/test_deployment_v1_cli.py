from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from src.churn_ml.deployment_v1_cli import execute


def test_validate_cli_emits_machine_json_without_execution(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    validated = SimpleNamespace(
        config=SimpleNamespace(deployment_id="synthetic"),
        identity_sha256="a" * 64,
        approvals=(object(),),
    )
    monkeypatch.setattr(
        "src.churn_ml.deployment_v1_cli._resolve_project_path",
        lambda path: path.absolute(),
    )
    monkeypatch.setattr(
        "src.churn_ml.deployment_v1_cli.validate_only",
        lambda *args, **kwargs: validated,
    )
    monkeypatch.setattr(
        "src.churn_ml.deployment_v1_cli.execute_deployment",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("validate must not execute")
        ),
    )
    args = argparse.Namespace(command="validate", config=tmp_path / "config.yaml")
    assert execute(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["competition_test_accessed"] is False
    assert payload["output_allocated"] is False


def test_run_cli_requires_explicit_competition_confirmation(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        "src.churn_ml.deployment_v1_cli._resolve_project_path",
        lambda path: path.absolute(),
    )
    monkeypatch.setattr(
        "src.churn_ml.deployment_v1_cli.validate_only",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    args = argparse.Namespace(
        command="run",
        config=tmp_path / "config.yaml",
        allow_competition_test=False,
    )
    assert execute(args) == 3
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"]["code"] == "competition_test_confirmation_required"


def test_dry_run_cli_uses_only_explicit_fixture_and_output(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    validated = SimpleNamespace()
    monkeypatch.setattr(
        "src.churn_ml.deployment_v1_cli._resolve_project_path",
        lambda path: path.absolute(),
    )
    monkeypatch.setattr(
        "src.churn_ml.deployment_v1_cli.validate_only",
        lambda *args, **kwargs: validated,
    )
    monkeypatch.setattr(
        "src.churn_ml.deployment_v1_cli._resolve_path",
        lambda path, **kwargs: path.absolute(),
    )
    observed = {}

    def fake_execute(prepared, **kwargs):
        observed["prepared"] = prepared
        observed.update(kwargs)
        return SimpleNamespace(
            root=tmp_path / "output",
            deployment_identity_sha256="b" * 64,
            positive_count=1,
            positive_rate=0.5,
        )

    monkeypatch.setattr(
        "src.churn_ml.deployment_v1_cli.execute_deployment",
        fake_execute,
    )
    args = argparse.Namespace(
        command="dry-run",
        config=tmp_path / "config.yaml",
        fixture_dir=tmp_path / "fixture",
        output_dir=tmp_path / "output",
    )
    assert execute(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert observed["mode"] == "dry-run"
    assert observed["fixture_dir"] == (tmp_path / "fixture").resolve()
    assert observed["output_dir"] == (tmp_path / "output").resolve()
