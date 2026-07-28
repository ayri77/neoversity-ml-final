from __future__ import annotations

import os
import secrets
from collections.abc import Iterator

import pytest

from src.churn_ml.optuna_search_authority import AUTHORITY_KEY_FILE_ENV


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
