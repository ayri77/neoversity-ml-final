from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/final/manual_lightgbmprep_r31_submission_v1.yaml"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/final_submissions"


def _run(*arguments: str, cwd: Path = PROJECT_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/run_final_submission.py",
            "--config",
            CONFIG,
            *arguments,
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_directories() -> set[Path]:
    if not ARTIFACT_ROOT.exists():
        return set()
    return {
        path
        for path in ARTIFACT_ROOT.rglob("*")
        if path.is_dir() and (path / "execution_status.json").exists()
    }


def test_validate_only_allocates_no_run_and_proves_loaded_modules() -> None:
    before = _run_directories()

    result = _run("--validate-only")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "threshold=0.107" in result.stdout
    assert "No artifact run was allocated." in result.stdout
    assert _run_directories() == before


def test_final_cli_requires_repository_root(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/run_final_submission.py"),
            "--config",
            str(PROJECT_ROOT / CONFIG),
            "--validate-only",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "repository root" in result.stderr


def test_final_source_contains_no_network_or_kaggle_client() -> None:
    forbidden = (
        "import requests",
        "from requests",
        "import httpx",
        "from httpx",
        "import kaggle",
        "from kaggle",
        "urllib.request",
        "socket.create_connection",
    )
    sources = [
        PROJECT_ROOT / "scripts/run_final_submission.py",
        *sorted((PROJECT_ROOT / "src/churn_ml").glob("final_*.py")),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in sources)

    for token in forbidden:
        assert token not in combined
