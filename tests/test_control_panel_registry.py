from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.churn_ml.control_panel.registry import load_registry
from src.churn_ml.control_panel.schemas import SchemaError, UISettings, parse_commands
from src.churn_ml.control_panel.schemas import parse_readers


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_repository_registry_is_strict_and_cross_referenced() -> None:
    loaded = load_registry(PROJECT_ROOT)
    assert loaded.settings.schema_version == 1
    assert set(loaded.commands) == {
        "experiment_core_v2",
        "research_v1",
        "paired_comparison",
        "mlflow_local_index",
        "final_deployment_v1",
    }
    assert set(loaded.readers) == {
        "research_v2",
        "research_v1",
        "paired_comparison",
        "deployment_v1",
    }
    assert loaded.commands["final_deployment_v1"].actions["run"].enabled is False
    assert (
        loaded.commands["final_deployment_v1"].actions["run"].competition_test is True
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema_version", True),
        ("poll_interval_seconds", "3"),
        ("allow_process_stop", 1),
        ("jobs_root", "../jobs"),
    ],
)
def test_settings_reject_wrong_primitive_types_and_paths(
    key: str, value: object
) -> None:
    payload = json.loads(
        (PROJECT_ROOT / "configs/ui/ui_settings.json").read_text(encoding="utf-8")
    )
    payload[key] = value
    with pytest.raises(SchemaError):
        UISettings.from_dict(payload)


def test_settings_reject_unknown_keys() -> None:
    payload = json.loads(
        (PROJECT_ROOT / "configs/ui/ui_settings.json").read_text(encoding="utf-8")
    )
    payload["unexpected"] = True
    with pytest.raises(SchemaError, match="unknown keys"):
        UISettings.from_dict(payload)


def test_command_registry_rejects_unknown_keys_and_partial_placeholders() -> None:
    payload = yaml.safe_load(
        (PROJECT_ROOT / "configs/ui/ui_commands.yaml").read_text(encoding="utf-8")
    )
    payload["commands"][0]["unexpected"] = "value"
    with pytest.raises(SchemaError, match="unknown keys"):
        parse_commands(payload)

    payload = yaml.safe_load(
        (PROJECT_ROOT / "configs/ui/ui_commands.yaml").read_text(encoding="utf-8")
    )
    payload["commands"][0]["actions"][0]["argv"] = ["--config={config}"]
    with pytest.raises(SchemaError, match="complete argv token"):
        parse_commands(payload)


@pytest.mark.parametrize(
    "prefix",
    [
        ["$PYTHON", "-c", "print('unsafe')"],
        ["$PYTHON", "-m", "module"],
        ["$PYTHON", "-"],
        ["$PYTHON", "outside.py"],
        ["$PYTHON", "scripts/../outside.py"],
        ["$PYTHON", "C:/outside.py"],
        ["$PYTHON", "--isolated", "scripts/run_research_v2.py"],
    ],
)
def test_command_prefix_contract_rejects_arbitrary_python(prefix: list[str]) -> None:
    payload = yaml.safe_load(
        (PROJECT_ROOT / "configs/ui/ui_commands.yaml").read_text(encoding="utf-8")
    )
    payload["commands"][0]["argv_prefix"] = prefix
    with pytest.raises(SchemaError):
        parse_commands(payload)


def test_public_cli_entrypoints_have_real_help() -> None:
    loaded = load_registry(PROJECT_ROOT)
    for script in sorted({command.public_cli for command in loaded.commands.values()}):
        completed = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout.lower()


@pytest.mark.parametrize(
    "marker",
    [
        "",
        ".",
        "..",
        "../_SUCCESS",
        "/tmp/_SUCCESS",
        "C:/_SUCCESS",
        "a//b",
        "a\\b",
        "bad\x00marker",
    ],
)
def test_reader_markers_must_be_normalized_safe_relative_paths(marker: str) -> None:
    payload = yaml.safe_load(
        (PROJECT_ROOT / "configs/ui/ui_readers.yaml").read_text(encoding="utf-8")
    )
    payload["readers"][0]["success_markers"] = [marker]
    with pytest.raises(SchemaError):
        parse_readers(payload)


@pytest.mark.parametrize(
    "dot_path",
    [
        "metrics..score",
        ".metrics",
        "metrics.",
        "metrics.*.score",
        "metrics[0].score",
        "metrics.__class__()",
        "items.01.value",
    ],
)
def test_reader_dot_path_grammar_rejects_malformed_paths(dot_path: str) -> None:
    payload = yaml.safe_load(
        (PROJECT_ROOT / "configs/ui/ui_readers.yaml").read_text(encoding="utf-8")
    )
    payload["readers"][0]["summary_files"][0]["fields"]["test"] = dot_path
    with pytest.raises(SchemaError, match="dot path|segments"):
        parse_readers(payload)


def test_reader_dot_path_grammar_accepts_mapping_and_list_segments() -> None:
    payload = yaml.safe_load(
        (PROJECT_ROOT / "configs/ui/ui_readers.yaml").read_text(encoding="utf-8")
    )
    payload["readers"][0]["summary_files"][0]["fields"]["test"] = "items.0.score"
    parsed = parse_readers(payload)
    assert parsed["research_v2"].summary_files[0].fields["test"] == "items.0.score"


def test_reader_registry_rejects_unknown_keys_and_wrong_types() -> None:
    path = PROJECT_ROOT / "configs/ui/ui_readers.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["readers"][0]["unexpected"] = True
    with pytest.raises(SchemaError, match="unknown keys"):
        parse_readers(payload)

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["readers"][0]["artifact_roots"] = "artifacts/research_v2"
    with pytest.raises(SchemaError, match="list of strings"):
        parse_readers(payload)
