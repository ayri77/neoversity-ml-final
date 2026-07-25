from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType

import pytest

from src.churn_ml import research_runtime_provenance
from src.churn_ml.research_runtime_provenance import (
    ResearchRuntimeProvenanceError,
    build_invocation_provenance,
    build_loaded_module_provenance,
    collect_research_environment_versions,
)


def _module(name: str, path: Path) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = str(path)
    return module


def _manifest(path: str, raw: bytes) -> dict:
    return {"files": [{"path": path, "sha256": hashlib.sha256(raw).hexdigest()}]}


def test_loaded_module_manifest_binds_name_path_and_raw_bytes(tmp_path: Path) -> None:
    relative = "src/unit/example.py"
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    raw = b"value = 1\r\n"
    source.write_bytes(raw)
    identity, digest = build_loaded_module_provenance(
        tmp_path,
        (_manifest(relative, raw),),
        entrypoint_module_name=None,
        loaded_modules={"src.unit.example": _module("src.unit.example", source)},
    )

    record = identity["canonical"]["modules"][0]
    assert record["module_name"] == "src.unit.example"
    assert record["loaded_source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert record["match"] is True
    repeated_identity, repeated_digest = build_loaded_module_provenance(
        tmp_path,
        (_manifest(relative, raw),),
        entrypoint_module_name=None,
        loaded_modules={"src.unit.example": _module("src.unit.example", source)},
    )
    assert repeated_identity == identity
    assert repeated_digest == digest
    assert len(digest) == 64


@pytest.mark.parametrize("failure", ["missing", "outside", "path", "bytes", "alias"])
def test_loaded_module_manifest_rejects_unverified_runtime(
    tmp_path: Path,
    failure: str,
) -> None:
    relative = "src/unit/example.py"
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    raw = b"value = 1\n"
    source.write_bytes(raw)
    modules: dict[str, ModuleType] = {
        "src.unit.example": _module("src.unit.example", source)
    }
    if failure == "missing":
        modules = {}
    elif failure == "outside":
        outside = tmp_path.parent / "outside.py"
        outside.write_bytes(raw)
        modules["src.unit.example"] = _module("src.unit.example", outside)
    elif failure == "path":
        other = tmp_path / "src/unit/other.py"
        other.write_bytes(raw)
        modules["src.unit.example"] = _module("src.unit.example", other)
    elif failure == "bytes":
        source.write_bytes(b"value = 2\n")
    elif failure == "alias":
        modules["unexpected.alias"] = _module("unexpected.alias", source)

    with pytest.raises(ResearchRuntimeProvenanceError):
        build_loaded_module_provenance(
            tmp_path,
            (_manifest(relative, raw),),
            entrypoint_module_name=None,
            loaded_modules=modules,
        )


def test_environment_capture_requires_pyarrow_and_pyyaml() -> None:
    versions = collect_research_environment_versions()
    assert versions["pyarrow"]
    assert versions["pyyaml"]


def test_environment_capture_fails_when_pyarrow_version_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = research_runtime_provenance.version

    def missing(distribution: str) -> str:
        if distribution == "pyarrow":
            raise research_runtime_provenance.PackageNotFoundError(distribution)
        return original(distribution)

    monkeypatch.setattr(research_runtime_provenance, "version", missing)
    with pytest.raises(ResearchRuntimeProvenanceError, match="pyarrow"):
        collect_research_environment_versions()


def test_invocation_requires_utc_process_start() -> None:
    non_utc = datetime.now(timezone(timedelta(hours=1)))
    with pytest.raises(ResearchRuntimeProvenanceError, match="UTC"):
        build_invocation_provenance(
            entry_point="unit",
            process_started_at_utc=non_utc,
        )
