from __future__ import annotations

import concurrent.futures
import json
import os
import stat
import threading
import uuid
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from src.churn_ml import optuna_search_authority as authority_mod
from src.churn_ml.optuna_search_authority import (
    AuthorityKeyInitError,
    initialize_lifecycle_authority_key,
)
from src.churn_ml.optuna_search_cli import (
    EXIT_STUDY,
    execute,
    parse_args,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATH_CANARY = f"CANARY_PATH_{uuid.uuid4().hex}"
KEY_CANARY = f"CANARY_KEY_{uuid.uuid4().hex}"


def _assert_no_canaries(*parts: str, path_canary: str, key_canary: str) -> None:
    blob = "\n".join(parts)
    assert path_canary not in blob
    assert key_canary not in blob
    assert PATH_CANARY not in blob
    assert KEY_CANARY not in blob


def _cli_authority_init(
    output: Path, *, create_parent: bool = False
) -> tuple[int, str, str]:
    argv = ["authority-init", "--output", str(output)]
    if create_parent:
        argv.append("--create-parent")
    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        status = execute(parse_args(argv))
    return status, stdout.getvalue(), stderr.getvalue()


def test_authority_init_success_omits_path_and_key_material(tmp_path: Path) -> None:
    target = tmp_path / f"{PATH_CANARY}.key"
    status, out, err = _cli_authority_init(target)
    assert status == 0
    payload = json.loads(out)
    assert payload["status"] == "authority_initialized"
    assert payload["key_bytes"] == 32
    assert "path" not in payload
    assert str(target) not in out
    assert PATH_CANARY not in out
    assert PATH_CANARY not in err
    assert len(target.read_bytes()) == 32


@pytest.mark.parametrize(
    "case",
    (
        "relative",
        "repo_contained",
        "artifact_contained",
        "study_root_contained",
        "missing_parent",
        "destination_directory",
        "already_exists",
    ),
)
def test_authority_init_sanitized_error_contract_canaries(
    case: str,
    tmp_path: Path,
) -> None:
    path_canary = f"{PATH_CANARY}_{case}"
    key_canary = f"{KEY_CANARY}_{case}"
    if case == "relative":
        output = Path(f"{path_canary}.key")
    elif case == "repo_contained":
        output = PROJECT_ROOT / "tmp_authority_canary" / f"{path_canary}.key"
    elif case == "artifact_contained":
        output = PROJECT_ROOT / "artifacts" / "optuna" / f"{path_canary}.key"
    elif case == "study_root_contained":
        output = (
            PROJECT_ROOT / "artifacts" / "optuna" / "studies" / f"{path_canary}.key"
        )
    elif case == "missing_parent":
        output = tmp_path / path_canary / "missing" / f"{path_canary}.key"
    elif case == "destination_directory":
        output = tmp_path / path_canary
        output.mkdir()
    else:
        output = tmp_path / f"{path_canary}.key"
        output.write_bytes(key_canary.encode("utf-8") + b"\x00" * 16)

    with pytest.raises(AuthorityKeyInitError) as caught:
        initialize_lifecycle_authority_key(
            output,
            project_root=PROJECT_ROOT,
            create_parent=False,
            forbidden_roots=(
                PROJECT_ROOT / "artifacts",
                PROJECT_ROOT / "artifacts" / "optuna",
            ),
        )
    error = caught.value
    _assert_no_canaries(
        str(error),
        repr(error),
        error.message,
        error.code,
        path_canary=path_canary,
        key_canary=key_canary,
    )
    assert error.code.startswith("AUTHORITY_KEY_INIT_")

    status, out, err = _cli_authority_init(output)
    assert status == EXIT_STUDY
    payload = json.loads(err)
    assert payload["error_kind"] == "authority_key_init_error"
    assert payload["error_code"].startswith("AUTHORITY_KEY_INIT_")
    _assert_no_canaries(
        out,
        err,
        json.dumps(payload),
        path_canary=path_canary,
        key_canary=key_canary,
    )
    assert str(output) not in out
    assert str(output) not in err


def test_authority_init_parent_create_failure_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path_canary = f"{PATH_CANARY}_parent_fail"
    target = tmp_path / path_canary / "authority.key"

    def boom(self: Path) -> None:
        raise OSError(13, f"permission denied: {path_canary}")

    monkeypatch.setattr(Path, "mkdir", boom)
    with pytest.raises(AuthorityKeyInitError) as caught:
        initialize_lifecycle_authority_key(
            target,
            project_root=PROJECT_ROOT,
            create_parent=True,
        )
    assert caught.value.code == "AUTHORITY_KEY_INIT_PARENT_CREATE_FAILED"
    _assert_no_canaries(
        str(caught.value),
        repr(caught.value),
        path_canary=path_canary,
        key_canary=KEY_CANARY,
    )


def test_authority_init_write_flush_verify_cleanup_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path_canary = f"{PATH_CANARY}_io"
    target = tmp_path / f"{path_canary}.key"

    real_open = os.open
    real_write = os.write
    real_fsync = os.fsync

    def fail_write(fd: int, data: bytes) -> int:
        raise OSError(28, f"write failed for {path_canary}")

    monkeypatch.setattr(os, "write", fail_write)
    with pytest.raises(AuthorityKeyInitError) as caught:
        initialize_lifecycle_authority_key(
            target,
            project_root=PROJECT_ROOT,
            create_parent=False,
        )
    assert caught.value.code == "AUTHORITY_KEY_INIT_WRITE_FAILED"
    assert not target.exists()
    assert list(tmp_path.glob(".churn-ml-authority-*.tmp")) == []
    _assert_no_canaries(
        str(caught.value), path_canary=path_canary, key_canary=KEY_CANARY
    )
    monkeypatch.setattr(os, "write", real_write)

    def fail_fsync(fd: int) -> None:
        raise OSError(5, f"fsync failed for {path_canary}")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    with pytest.raises(AuthorityKeyInitError) as caught:
        initialize_lifecycle_authority_key(
            target,
            project_root=PROJECT_ROOT,
            create_parent=False,
        )
    assert caught.value.code == "AUTHORITY_KEY_INIT_FLUSH_FAILED"
    assert list(tmp_path.glob(".churn-ml-authority-*.tmp")) == []
    monkeypatch.setattr(os, "fsync", real_fsync)

    def fail_verify(*args: Any, **kwargs: Any) -> Any:
        raise OSError(13, f"verify blocked {path_canary}")

    calls = {"n": 0}
    original_link = getattr(os, "link", None)

    def counting_open(
        path: str | bytes | os.PathLike[str], flags: int, *rest: Any
    ) -> int:
        path_text = str(os.fspath(path))
        fd = real_open(path, flags, *rest) if rest else real_open(path, flags)
        if path_text.endswith(".tmp") or Path(path_text) == target:
            calls["n"] += 1
        return fd

    # Force verification failure after successful publication by patching loader.
    def boom_load(*args: Any, **kwargs: Any) -> Any:
        raise OSError(1, f"cannot load {path_canary}")

    monkeypatch.setattr(
        authority_mod,
        "load_lifecycle_authority_key_from_path",
        boom_load,
    )
    with pytest.raises(AuthorityKeyInitError) as caught:
        initialize_lifecycle_authority_key(
            target,
            project_root=PROJECT_ROOT,
            create_parent=False,
        )
    assert caught.value.code == "AUTHORITY_KEY_INIT_VERIFY_FAILED"
    _assert_no_canaries(
        str(caught.value), path_canary=path_canary, key_canary=KEY_CANARY
    )
    del original_link, counting_open, fail_verify


def test_authority_init_rejects_reparse_or_link_destination(
    tmp_path: Path,
) -> None:
    path_canary = f"{PATH_CANARY}_link"
    real_target = tmp_path / "real.key"
    real_target.write_bytes(os.urandom(32))
    linked = tmp_path / f"{path_canary}.key"
    try:
        linked.symlink_to(real_target)
    except OSError:
        pytest.skip("Local Windows policy does not allow test symlink creation.")
    with pytest.raises(AuthorityKeyInitError) as caught:
        initialize_lifecycle_authority_key(
            linked,
            project_root=PROJECT_ROOT,
            create_parent=False,
        )
    assert caught.value.code in {
        "AUTHORITY_KEY_INIT_INVALID_DESTINATION",
        "AUTHORITY_KEY_INIT_ALREADY_EXISTS",
    }
    _assert_no_canaries(
        str(caught.value), path_canary=path_canary, key_canary=KEY_CANARY
    )


def test_authority_init_concurrent_creators_one_winner(tmp_path: Path) -> None:
    target = tmp_path / "shared.key"
    barrier = threading.Barrier(2)
    results: list[tuple[str, Any]] = []

    def worker() -> None:
        barrier.wait(timeout=10)
        try:
            fingerprint = initialize_lifecycle_authority_key(
                target,
                project_root=PROJECT_ROOT,
                create_parent=False,
            )
            results.append(("ok", fingerprint))
        except AuthorityKeyInitError as error:
            results.append(("err", error.code))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker), pool.submit(worker)]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    assert len(results) == 2
    oks = [item for item in results if item[0] == "ok"]
    errs = [item for item in results if item[0] == "err"]
    assert len(oks) == 1
    assert len(errs) == 1
    assert errs[0][1] in {
        "AUTHORITY_KEY_INIT_ALREADY_EXISTS",
        "AUTHORITY_KEY_INIT_CREATE_FAILED",
    }
    assert target.is_file()
    assert len(target.read_bytes()) == 32
    assert getattr(target.stat(), "st_nlink", 1) == 1
    assert list(tmp_path.glob(".churn-ml-authority-*.tmp")) == []


def test_authority_init_does_not_delete_foreign_file_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path_canary = f"{PATH_CANARY}_race"
    target = tmp_path / f"{path_canary}.key"
    foreign = b"FOREIGN_KEY_MATERIAL_SHOULD_SURVIVE_32b"

    real_link = os.link

    def publish_then_fail(src: str, dst: str) -> None:
        real_link(src, dst)
        # Replace published file with a foreign creator's content identity race:
        # simulate another process replacing after our publish by writing foreign bytes
        # through a new inode when possible; at minimum ensure cleanup does not delete
        # an independently created path when verification fails after identity change.
        raise OSError(1, f"post-publish fault {path_canary}")

    if not hasattr(os, "link"):
        pytest.skip("os.link is required for this race probe")

    # Force failure during verification after our temp exists but before/without
    # deleting a pre-created foreign destination that appears between checks.
    target.write_bytes(foreign)
    with pytest.raises(AuthorityKeyInitError) as caught:
        initialize_lifecycle_authority_key(
            target,
            project_root=PROJECT_ROOT,
            create_parent=False,
        )
    assert caught.value.code == "AUTHORITY_KEY_INIT_ALREADY_EXISTS"
    assert target.read_bytes() == foreign
    _assert_no_canaries(
        str(caught.value), path_canary=path_canary, key_canary=KEY_CANARY
    )

    target.unlink()
    created_temps: list[Path] = []
    real_open = os.open

    def tracking_open(
        path: str | bytes | os.PathLike[str], flags: int, *rest: Any
    ) -> int:
        path_obj = Path(str(os.fspath(path)))
        fd = real_open(path, flags, *rest) if rest else real_open(path, flags)
        if (
            path_obj.name.startswith(".churn-ml-authority-")
            and path_obj.suffix == ".tmp"
        ):
            created_temps.append(path_obj)
            # Inject foreign destination before publish.
            if not target.exists():
                target.write_bytes(foreign)
        return fd

    monkeypatch.setattr(os, "open", tracking_open)
    with pytest.raises(AuthorityKeyInitError):
        initialize_lifecycle_authority_key(
            target,
            project_root=PROJECT_ROOT,
            create_parent=False,
        )
    assert target.read_bytes() == foreign
    assert all(not path.exists() for path in created_temps)
    del publish_then_fail


def test_authority_init_cleanup_only_own_temp_on_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "authority.key"
    foreign_temp = tmp_path / f".churn-ml-authority-{uuid.uuid4().hex}.tmp"
    foreign_temp.write_bytes(b"x" * 32)

    def fail_write(fd: int, data: bytes) -> int:
        raise OSError(5, "disk fault")

    monkeypatch.setattr(os, "write", fail_write)
    with pytest.raises(AuthorityKeyInitError) as caught:
        initialize_lifecycle_authority_key(
            target,
            project_root=PROJECT_ROOT,
            create_parent=False,
        )
    assert caught.value.code == "AUTHORITY_KEY_INIT_WRITE_FAILED"
    assert foreign_temp.is_file()
    assert foreign_temp.read_bytes() == b"x" * 32
    assert (
        list(
            path
            for path in tmp_path.glob(".churn-ml-authority-*.tmp")
            if path != foreign_temp
        )
        == []
    )


def test_authority_init_final_key_permissions_and_link_count(tmp_path: Path) -> None:
    target = tmp_path / "authority.key"
    fingerprint = initialize_lifecycle_authority_key(
        target,
        project_root=PROJECT_ROOT,
        create_parent=False,
    )
    assert len(fingerprint) == 64
    metadata = target.lstat()
    assert stat.S_ISREG(metadata.st_mode)
    assert not stat.S_ISLNK(metadata.st_mode)
    assert metadata.st_nlink == 1
    assert len(target.read_bytes()) == 32
    if os.name != "nt":
        assert metadata.st_mode & 0o777 == 0o600
