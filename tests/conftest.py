from __future__ import annotations

import os
import secrets
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.churn_ml.optuna_search_authority import AUTHORITY_KEY_FILE_ENV

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAFE_TMP = _REPO_ROOT / ".tmp" / "control-panel-tests"
_SAFE_TMP.mkdir(parents=True, exist_ok=True)

# Prefer a repository-local temp root when the platform default temp ACLs
# interfere with Streamlit AppTest and subprocess job probes.
os.environ.setdefault("TMP", str(_SAFE_TMP))
os.environ.setdefault("TEMP", str(_SAFE_TMP))
os.environ.setdefault("TMPDIR", str(_SAFE_TMP))


@pytest.fixture(scope="session", autouse=True)
def external_optuna_lifecycle_authority_key(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """Give Optuna tests an out-of-repository binary lifecycle authority key."""
    key_root = tmp_path_factory.mktemp("external_optuna_lifecycle_authority")
    key_path = key_root / "authority.key"
    key_path.write_bytes(secrets.token_bytes(32))
    previous = os.environ.get(AUTHORITY_KEY_FILE_ENV)
    os.environ[AUTHORITY_KEY_FILE_ENV] = str(key_path)
    try:
        yield
    finally:
        key_path.unlink(missing_ok=True)
        if previous is None:
            os.environ.pop(AUTHORITY_KEY_FILE_ENV, None)
        else:
            os.environ[AUTHORITY_KEY_FILE_ENV] = previous
