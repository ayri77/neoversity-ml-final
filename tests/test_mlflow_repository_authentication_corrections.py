from __future__ import annotations

import hashlib
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from src.churn_ml.mlflow_sources import (
    ResearchV2SourceAdapter,
    SourceValidationError,
    default_source_registry,
)
from src.churn_ml.mlflow_sync import sync_sources
from src.churn_ml.research_v2_provenance import SOURCE_PATHS, file_identity
from tests.test_mlflow_final_review_corrections import (
    PROJECT_ROOT,
    _config,
    _resolved_config,
    _write_early_failure,
    _write_identity,
)


CONFIG_RELATIVE = "configs/research_v2/manual_lightgbm_te_v1_compat_smoke.yaml"


def _read_resolved(run: Path) -> dict[str, Any]:
    return yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8"))


def _write_resolved(run: Path, resolved: dict[str, Any]) -> None:
    (run / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )


def _plan_run(tmp_path: Path) -> tuple[Any, Path, dict[str, Any]]:
    config = _config(tmp_path)
    resolved = _resolved_config()
    run = config.paths.research_v2_root / "plan" / "candidate" / "run-1"
    _write_early_failure(run, resolved_config=resolved)
    return config, run, resolved


def _mutate_persisted_plan(resolved: dict[str, Any], corruption: str) -> None:
    plan = resolved["evaluation_plan"]
    if corruption == "dataset_version":
        resolved["dataset"]["version"] = "forged_dataset"
        plan["dataset"]["version"] = "forged_dataset"
    elif corruption == "plan_id":
        plan["plan"]["id"] = "forged_plan"
    elif corruption == "outer_folds":
        plan["outer_evaluation"]["n_splits"] = 4
    elif corruption == "repeat_count":
        plan["outer_evaluation"]["repeat_seeds"] = [0, 1]
    elif corruption == "seeds":
        plan["outer_evaluation"]["repeat_seeds"] = [42]
    elif corruption == "threshold_folds":
        plan["threshold_selection"]["n_splits"] = 3
    elif corruption == "threshold_grid":
        plan["threshold_policy"]["step"] = 0.002
    elif corruption == "threshold_policy":
        plan["threshold_policy"]["tie_break"] = "highest_threshold"
    else:
        raise AssertionError(corruption)


def test_matching_persisted_plan_authenticates_against_repository(
    tmp_path: Path,
) -> None:
    config, run, resolved = _plan_run(tmp_path)

    record = ResearchV2SourceAdapter().prepare(run, config)

    assert record.params["dataset_version"] == resolved["dataset"]["version"]
    assert "resolved_config.yaml" in record.artifact_relative_paths


@pytest.mark.parametrize(
    "corruption",
    [
        "dataset_version",
        "plan_id",
        "outer_folds",
        "repeat_count",
        "seeds",
        "threshold_folds",
        "threshold_grid",
        "threshold_policy",
    ],
)
def test_persisted_plan_behavior_mutations_are_rejected(
    tmp_path: Path,
    corruption: str,
) -> None:
    config, run, _ = _plan_run(tmp_path)
    resolved = _read_resolved(run)
    _mutate_persisted_plan(resolved, corruption)
    _write_resolved(run, resolved)

    with pytest.raises(SourceValidationError):
        ResearchV2SourceAdapter().prepare(run, config)


def test_reference_to_different_valid_repository_plan_is_rejected(
    tmp_path: Path,
) -> None:
    config, run, _ = _plan_run(tmp_path)
    resolved = _read_resolved(run)
    different_relative = (
        "configs/research_v2/plans/telecom_v3_development_r2x5_t3_v1.yaml"
    )
    different_source = PROJECT_ROOT / different_relative
    different_target = tmp_path / different_relative
    different_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(different_source, different_target)
    resolved["evaluation_plan_path"] = different_relative
    _write_resolved(run, resolved)

    with pytest.raises(SourceValidationError):
        ResearchV2SourceAdapter().prepare(run, config)


def test_missing_referenced_repository_plan_is_rejected(tmp_path: Path) -> None:
    config, run, resolved = _plan_run(tmp_path)
    (tmp_path / resolved["evaluation_plan_path"]).unlink()

    with pytest.raises(SourceValidationError, match="missing"):
        ResearchV2SourceAdapter().prepare(run, config)


@pytest.mark.parametrize(
    "bad_path",
    [
        "../plan.yaml",
        "/tmp/plan.yaml",
        "C:/plan.yaml",
        r"\\server\share\plan.yaml",
        r"\rooted\plan.yaml",
        r"configs\research_v2\plans\plan.yaml",
    ],
)
def test_referenced_plan_path_escape_forms_are_rejected(
    tmp_path: Path,
    bad_path: Any,
) -> None:
    config, run, _ = _plan_run(tmp_path)
    resolved = _read_resolved(run)
    resolved["evaluation_plan_path"] = bad_path
    _write_resolved(run, resolved)

    with pytest.raises(SourceValidationError):
        ResearchV2SourceAdapter().prepare(run, config)


def test_forged_plan_stops_before_mapping_copy_and_mlflow_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, run, _ = _plan_run(tmp_path)
    resolved = _read_resolved(run)
    _mutate_persisted_plan(resolved, "threshold_grid")
    _write_resolved(run, resolved)
    calls = {"copy": False, "mapping": False}

    def copied(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        calls["copy"] = True
        return ()

    def mapped(*args: Any, **kwargs: Any) -> Any:
        calls["mapping"] = True
        raise AssertionError("mapping must not run")

    monkeypatch.setattr("src.churn_ml.mlflow_sources.select_indexed_artifacts", copied)
    monkeypatch.setattr("src.churn_ml.mlflow_sources.build_research_mapping", mapped)

    summary = sync_sources(
        config,
        registry=default_source_registry(),
        source_types=("research_v2",),
    )

    assert summary.items[0].outcome == "rejected"
    assert calls == {"copy": False, "mapping": False}
    assert config.paths.backend_store.exists() is False
    assert config.paths.artifact_root.exists() is False


def _copy_repository_identity_inputs(root: Path, resolved: dict[str, Any]) -> None:
    paths = {*SOURCE_PATHS, CONFIG_RELATIVE, resolved["evaluation_plan_path"]}
    for relative in paths:
        source = PROJECT_ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)


def _authenticated_source_run(
    tmp_path: Path,
) -> tuple[Any, Path, dict[str, Any], dict[str, Any], str]:
    config, run, resolved = _plan_run(tmp_path)
    _copy_repository_identity_inputs(tmp_path, resolved)
    relative_paths = (*SOURCE_PATHS, CONFIG_RELATIVE, resolved["evaluation_plan_path"])
    canonical, digest = file_identity(tmp_path, relative_paths)
    _write_identity(run, "source", canonical)
    return config, run, resolved, canonical, digest


def _rewrite_source_identity(
    run: Path,
    canonical: dict[str, Any],
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    changed = deepcopy(canonical)
    mutation(changed)
    _write_identity(run, "source", changed)


def test_valid_source_identity_recomputes_repository_bytes_and_is_copyable(
    tmp_path: Path,
) -> None:
    config, run, _, _, digest = _authenticated_source_run(tmp_path)

    record = ResearchV2SourceAdapter().prepare(run, config)

    assert record.params["source_provenance_sha256"] == digest
    assert "identities/source.json" in record.artifact_relative_paths
    summary = sync_sources(
        config,
        registry=default_source_registry(),
        source_types=("research_v2",),
    )
    assert summary.items[0].outcome == "created"


@pytest.mark.parametrize(
    "method",
    ["bogus", "SHA256_FILE_BYTES_SORTED_REPOSITORY_RELATIVE_PATHS", "", 7],
)
def test_source_identity_requires_exact_hashing_method(
    tmp_path: Path,
    method: Any,
) -> None:
    config, run, _, canonical, _ = _authenticated_source_run(tmp_path)
    _rewrite_source_identity(
        run,
        canonical,
        lambda value: value.update({"hashing_method": method}),
    )

    with pytest.raises(SourceValidationError, match="hashing method"):
        ResearchV2SourceAdapter().prepare(run, config)


def test_source_identity_rejects_uppercase_digest(tmp_path: Path) -> None:
    config, run, _, canonical, _ = _authenticated_source_run(tmp_path)
    _rewrite_source_identity(
        run,
        canonical,
        lambda value: value["files"][0].update(
            {"sha256": value["files"][0]["sha256"].upper()}
        ),
    )

    with pytest.raises(SourceValidationError):
        ResearchV2SourceAdapter().prepare(run, config)


def test_source_identity_rejects_correct_looking_wrong_digest(tmp_path: Path) -> None:
    config, run, _, canonical, _ = _authenticated_source_run(tmp_path)
    _rewrite_source_identity(
        run,
        canonical,
        lambda value: value["files"][0].update({"sha256": "0" * 64}),
    )

    with pytest.raises(SourceValidationError, match="byte hash"):
        ResearchV2SourceAdapter().prepare(run, config)


def test_source_identity_rejects_digest_from_different_file(tmp_path: Path) -> None:
    config, run, _, canonical, _ = _authenticated_source_run(tmp_path)

    def mutate(value: dict[str, Any]) -> None:
        value["files"][0]["sha256"] = value["files"][1]["sha256"]

    _rewrite_source_identity(run, canonical, mutate)

    with pytest.raises(SourceValidationError, match="byte hash"):
        ResearchV2SourceAdapter().prepare(run, config)


def test_source_identity_rejects_file_modified_after_identity(tmp_path: Path) -> None:
    config, run, _, canonical, _ = _authenticated_source_run(tmp_path)
    path = tmp_path / canonical["files"][0]["path"]
    path.write_bytes(path.read_bytes() + b"\n# modified after identity\n")

    with pytest.raises(SourceValidationError, match="byte hash"):
        ResearchV2SourceAdapter().prepare(run, config)


def test_source_identity_rejects_recorded_size_field(tmp_path: Path) -> None:
    config, run, _, canonical, _ = _authenticated_source_run(tmp_path)
    _rewrite_source_identity(
        run,
        canonical,
        lambda value: value["files"][0].update({"size_bytes": -1}),
    )

    with pytest.raises(SourceValidationError, match="keys differ"):
        ResearchV2SourceAdapter().prepare(run, config)


def test_source_identity_rejects_missing_referenced_file(tmp_path: Path) -> None:
    config, run, _, canonical, _ = _authenticated_source_run(tmp_path)
    (tmp_path / canonical["files"][0]["path"]).unlink()

    with pytest.raises(SourceValidationError, match="missing"):
        ResearchV2SourceAdapter().prepare(run, config)


def test_source_identity_rejects_extra_file_entry(tmp_path: Path) -> None:
    config, run, _, canonical, _ = _authenticated_source_run(tmp_path)
    extra = tmp_path / "src/churn_ml/extra_auth_source.py"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"extra")

    def mutate(value: dict[str, Any]) -> None:
        value["files"].append(
            {
                "path": "src/churn_ml/extra_auth_source.py",
                "sha256": hashlib.sha256(b"extra").hexdigest(),
            }
        )
        value["files"] = sorted(value["files"], key=lambda item: item["path"])

    _rewrite_source_identity(run, canonical, mutate)

    with pytest.raises(SourceValidationError, match="file set"):
        ResearchV2SourceAdapter().prepare(run, config)


def test_source_identity_rejects_duplicate_file_entry(tmp_path: Path) -> None:
    config, run, _, canonical, _ = _authenticated_source_run(tmp_path)
    _rewrite_source_identity(
        run,
        canonical,
        lambda value: value["files"].append(deepcopy(value["files"][0])),
    )

    with pytest.raises(SourceValidationError, match="duplicate"):
        ResearchV2SourceAdapter().prepare(run, config)


@pytest.mark.parametrize(
    "bad_path",
    [
        "../source.py",
        "/tmp/source.py",
        "C:/source.py",
        r"\\server\share\source.py",
        r"src\churn_ml\source.py",
        7,
    ],
)
def test_source_identity_rejects_nonrepository_paths(
    tmp_path: Path,
    bad_path: Any,
) -> None:
    config, run, _, canonical, _ = _authenticated_source_run(tmp_path)
    _rewrite_source_identity(
        run,
        canonical,
        lambda value: value["files"][0].update({"path": bad_path}),
    )

    with pytest.raises(SourceValidationError):
        ResearchV2SourceAdapter().prepare(run, config)


def test_source_identity_rejects_symlink_reference_where_supported(
    tmp_path: Path,
) -> None:
    config, run, _, canonical, _ = _authenticated_source_run(tmp_path)
    path = tmp_path / canonical["files"][0]["path"]
    target = tmp_path / canonical["files"][1]["path"]
    path.unlink()
    try:
        path.symlink_to(target)
    except OSError as error:
        pytest.skip(f"Symlink creation unavailable: {error}")

    with pytest.raises(SourceValidationError, match="reparse"):
        ResearchV2SourceAdapter().prepare(run, config)


def test_authenticated_source_dry_run_allocates_no_mlflow_storage(
    tmp_path: Path,
) -> None:
    config, _, _, _, _ = _authenticated_source_run(tmp_path)

    summary = sync_sources(
        config,
        registry=default_source_registry(),
        source_types=("research_v2",),
        dry_run=True,
    )

    assert summary.items[0].outcome == "validated"
    assert config.paths.backend_store.exists() is False
    assert config.paths.artifact_root.exists() is False


def test_forged_source_identity_stops_before_copy_mapping_and_mlflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, run, _, canonical, _ = _authenticated_source_run(tmp_path)
    _rewrite_source_identity(
        run,
        canonical,
        lambda value: value["files"][0].update({"sha256": "0" * 64}),
    )
    calls = {"copy": False, "mapping": False}

    def copied(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        calls["copy"] = True
        return ()

    def mapped(*args: Any, **kwargs: Any) -> Any:
        calls["mapping"] = True
        raise AssertionError("mapping must not run")

    monkeypatch.setattr("src.churn_ml.mlflow_sources.select_indexed_artifacts", copied)
    monkeypatch.setattr("src.churn_ml.mlflow_sources.build_research_mapping", mapped)

    summary = sync_sources(
        config,
        registry=default_source_registry(),
        source_types=("research_v2",),
    )

    assert summary.items[0].outcome == "rejected"
    assert calls == {"copy": False, "mapping": False}
    assert config.paths.backend_store.exists() is False
    assert config.paths.artifact_root.exists() is False
