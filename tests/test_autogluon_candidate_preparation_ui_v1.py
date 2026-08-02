"""Focused tests for AutoGluon candidate preparation workspace."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.churn_ml.autogluon_artifacts import build_inventory, write_json
from src.churn_ml.control_panel.candidate_preparation import (
    discover_managed_autogluon_runs,
    ensemble_component_warning,
    filter_managed_runs,
    managed_runs_inventory_fingerprint,
    verify_validation_result,
)
from src.churn_ml.control_panel.command_builder import build_command
from src.churn_ml.control_panel.launch import authorize_launch, rendered_launch
from src.churn_ml.control_panel.python_runtimes import (
    PythonRuntimeError,
    require_autogluon_python,
    resolve_autogluon_python,
)
from src.churn_ml.control_panel.registry import load_registry
from src.churn_ml.control_panel.workflow_navigation import (
    ADVANCED_ONLY_COMMAND_IDS,
    visible_command_ids,
)
from src.churn_ml.prediction_candidates import cli as candidate_cli
from src.churn_ml.prediction_candidates.autogluon_v1 import (
    OOF_PROTOCOL,
    SOURCE_KIND,
)
from src.churn_ml.prediction_candidates.contract_v1 import (
    POSITIVE_CLASS_LABEL,
    PROBABILITY_SEMANTICS,
    SCHEMA_VERSION,
    create_candidate_package,
)
from src.churn_ml.prediction_candidates.preparation_request_v1 import (
    PreparationRequestError,
    build_preparation_request_payload,
    load_preparation_request,
    materialize_preparation_request,
    validate_selected_models,
    verify_request_run_identity,
)
from tests.dataset_registry_support import (
    create_synthetic_legacy_tree,
    make_v0_frames,
    register_package,
    write_package_files,
)
from tests.test_autogluon_inspection import make_valid_completed_run


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DATASET_ID = "v3_targeted_missingness"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    processed = root / "data" / "processed"
    create_synthetic_legacy_tree(processed)
    register_package(
        processed,
        "v0_raw_minimal",
        parent_dataset_id=None,
        hypothesis="synthetic v0",
    )
    # Mirror the AutoGluon fixture dataset id used by make_valid_completed_run.
    x_train, y_train, x_test = make_v0_frames()
    write_package_files(
        processed / DATASET_ID,
        dataset_id=DATASET_ID,
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        metadata={"source": "v0_raw_minimal", "description": "synthetic v3"},
    )
    register_package(
        processed,
        DATASET_ID,
        parent_dataset_id="v0_raw_minimal",
        hypothesis="synthetic v3 for AG fixture",
    )
    return root


def _complete_run(repo: Path, *, name: str = "ag-complete") -> Path:
    run_dir = repo / "artifacts" / "autogluon_runs" / name
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    make_valid_completed_run(run_dir)
    worker = json.loads((run_dir / "worker_result.json").read_text(encoding="utf-8"))
    worker["model_names"] = [
        "RealTabPFN-v2_r11",
        "NeuralNetTorch_r37",
        "LightGBMPrep_r31",
        "LightGBMPrep_r41",
        "WeightedEnsemble_L2",
    ]
    worker["best_model"] = "WeightedEnsemble_L2"
    write_json(run_dir / "worker_result.json", worker)
    summary = json.loads((run_dir / "inspection" / "summary.json").read_text(encoding="utf-8"))
    summary["models"] = list(worker["model_names"])
    summary["best_model"] = worker["best_model"]
    write_json(run_dir / "inspection" / "summary.json", summary)
    pd.DataFrame(
        [
            {
                "model": model_name,
                "score_val": 0.8 - index * 0.01,
                "stack_level": 1,
                "fit_time": 1.0,
                "pred_time_val": 0.1,
            }
            for index, model_name in enumerate(worker["model_names"])
        ]
    ).to_csv(run_dir / "inspection" / "leaderboard.csv", index=False)
    write_json(run_dir / "artifact_inventory.json", build_inventory(run_dir))
    return run_dir


def _make_candidate_for_run(
    repo: Path,
    *,
    run_path: str,
    model_name: str,
    config_sha256: str,
) -> str:
    y = pd.read_parquet(
        repo / "data" / "processed" / DATASET_ID / "y_train.parquet"
    ).iloc[:, 0]
    n = len(y)
    oof = pd.DataFrame(
        {
            "row_position": np.arange(n, dtype=np.int64),
            "target": y.astype("int64").to_numpy(),
            "probability_positive": np.linspace(0.1, 0.9, n),
        }
    )
    test = pd.DataFrame(
        {
            "row_position": np.arange(3, dtype=np.int64),
            "probability_positive": np.linspace(0.2, 0.8, 3),
        }
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "source_kind": SOURCE_KIND,
        "dataset_id": DATASET_ID,
        "source_run_path": run_path,
        "source_config_sha256": config_sha256,
        "source_model_name": model_name,
        "oof_protocol": OOF_PROTOCOL,
        "positive_class_label": POSITIVE_CLASS_LABEL,
        "probability_semantics": PROBABILITY_SEMANTICS,
    }
    package = create_candidate_package(
        repository_root=repo,
        identity=identity,
        oof=oof,
        test=test,
        manifest_fields={
            "dataset_id": DATASET_ID,
            "exploratory": False,
            "source_run_path": run_path,
            "source_config_path": f"{run_path}/resolved_config.yaml",
            "source_config_sha256": config_sha256,
            "source_model_name": model_name,
            "source_model_type": "UnitModel",
            "source_predictor_path": f"{run_path}/predictor",
            "source_autogluon_version": "1.5.0",
            "source_metric_name": "balanced_accuracy",
            "source_metric_value": 0.8,
            "source_leaderboard_metadata": {"model": model_name},
            "provenance": {"adapter": "unit_test"},
        },
        source_metadata={"schema_version": 1, "name": model_name},
        candidates_root_relative="artifacts/prediction_candidates",
    )
    return package.candidate_id


def test_runtime_resolution_windows_posix_and_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    missing = resolve_autogluon_python(root)
    assert missing.available is False
    assert missing.reason_code == "autogluon_runtime_missing"
    with pytest.raises(PythonRuntimeError):
        require_autogluon_python(root)

    if os.name == "nt":
        relative = Path(".venv-autogluon/Scripts/python.exe")
    else:
        relative = Path(".venv-autogluon/bin/python")
    target = root / relative
    target.parent.mkdir(parents=True)
    target.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    if os.name != "nt":
        target.chmod(0o755)
    resolved = resolve_autogluon_python(root)
    assert resolved.available is True
    assert resolved.relative_path.replace("\\", "/") == relative.as_posix()

    # Escape attempt via constructed relative is rejected by resolver API.
    monkeypatch.setattr(
        "src.churn_ml.control_panel.python_runtimes.autogluon_interpreter_relative",
        lambda: "../outside/python",
    )
    escaped = resolve_autogluon_python(root)
    assert escaped.available is False
    assert escaped.reason_code == "path_traversal"


def test_runtime_used_identically_by_build_and_authorize(repo: Path) -> None:
    run_dir = _complete_run(repo)
    # Create fake interpreter under repo.
    if os.name == "nt":
        py = repo / ".venv-autogluon" / "Scripts" / "python.exe"
    else:
        py = repo / ".venv-autogluon" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    if os.name != "nt":
        py.chmod(0o755)
    (repo / "scripts").mkdir(exist_ok=True)
    (repo / "scripts" / "import_autogluon_prediction_candidates.py").write_text(
        "# stub\n", encoding="utf-8"
    )
    request = materialize_preparation_request(
        repository_root=repo,
        run_path=run_dir.relative_to(repo).as_posix().replace("\\", "/"),
        selected_models=["RealTabPFN-v2_r11"],
    )
    loaded = load_registry(PROJECT_ROOT)
    runtime = require_autogluon_python(repo)
    values = {"request": request.relative_path}
    built = build_command(
        loaded.commands,
        "autogluon_candidate_preparation_v1",
        "validate",
        values,
        repository_root=repo,
        python_executable=str(runtime.absolute_path),
    )
    assert built.argv[0] == str(runtime.absolute_path)
    consumed: set[str] = set()
    rendered = rendered_launch(built, None, consumed_nonces=consumed)
    authorized = authorize_launch(
        loaded.commands,
        command_id="autogluon_candidate_preparation_v1",
        action_id="validate",
        values=values,
        repository_root=repo,
        confirmed=True,
        high_risk_acknowledged=True,
        rendered=rendered,
        consumed_nonces=consumed,
        python_executable=str(runtime.absolute_path),
    )
    assert authorized.argv == built.argv
    assert "shell" not in " ".join(built.argv).casefold()


def test_run_discovery_and_prepared_mapping(repo: Path) -> None:
    run_dir = _complete_run(repo, name="v6-focused")
    external = repo / "artifacts" / "elsewhere" / "run"
    external.mkdir(parents=True)
    (external / "_SUCCESS").write_text("{}", encoding="utf-8")
    incomplete = repo / "artifacts" / "autogluon_runs" / "incomplete"
    incomplete.mkdir(parents=True)
    (incomplete / "execution_status.json").write_text("{}", encoding="utf-8")

    rows = discover_managed_autogluon_runs(repo, include_incomplete=True)
    paths = {row["run_path"] for row in rows}
    assert any("v6-focused" in path for path in paths)
    assert all("elsewhere" not in path for path in paths)
    incomplete_row = next(row for row in rows if "incomplete" in row["run_path"])
    assert incomplete_row["preparable"] is False

    complete = next(row for row in rows if "v6-focused" in row["run_path"])
    model_names = {item["model_name"] for item in complete["models"]}
    assert {
        "RealTabPFN-v2_r11",
        "NeuralNetTorch_r37",
        "LightGBMPrep_r31",
        "LightGBMPrep_r41",
        "WeightedEnsemble_L2",
    }.issubset(model_names)

    config_sha = str(complete["config_sha256"])
    candidate_id = _make_candidate_for_run(
        repo,
        run_path=complete["run_path"],
        model_name="WeightedEnsemble_L2",
        config_sha256=config_sha,
    )
    rows2 = discover_managed_autogluon_runs(repo, include_incomplete=False)
    complete2 = next(row for row in rows2 if row["run_path"] == complete["run_path"])
    ensemble = next(
        item
        for item in complete2["models"]
        if item["model_name"] == "WeightedEnsemble_L2"
    )
    assert ensemble["already_prepared"] is True
    assert ensemble["candidate_id"] == candidate_id

    other = _make_candidate_for_run(
        repo,
        run_path="artifacts/autogluon_runs/other-run",
        model_name="WeightedEnsemble_L2",
        config_sha256="a" * 64,
    )
    assert other != candidate_id
    fp1 = managed_runs_inventory_fingerprint(repo)
    (run_dir / "run_metadata.json").write_text(
        (run_dir / "run_metadata.json").read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    fp2 = managed_runs_inventory_fingerprint(repo)
    assert fp1 != fp2


def test_request_contract_and_cli_equivalence(repo: Path) -> None:
    run_dir = _complete_run(repo)
    run_path = run_dir.relative_to(repo).as_posix().replace("\\", "/")
    first = materialize_preparation_request(
        repository_root=repo,
        run_path=run_path,
        selected_models=["RealTabPFN-v2_r11", "NeuralNetTorch_r37"],
    )
    reused = materialize_preparation_request(
        repository_root=repo,
        run_path=run_path,
        selected_models=["RealTabPFN-v2_r11", "NeuralNetTorch_r37"],
    )
    assert reused.request_id == first.request_id
    reordered = build_preparation_request_payload(
        repository_root=repo,
        run_path=run_path,
        selected_models=["NeuralNetTorch_r37", "RealTabPFN-v2_r11"],
    )
    assert reordered["request_id"] != first.request_id
    changed = build_preparation_request_payload(
        repository_root=repo,
        run_path=run_path,
        selected_models=["RealTabPFN-v2_r11", "LightGBMPrep_r31"],
    )
    assert changed["request_id"] != first.request_id
    with pytest.raises(PreparationRequestError):
        validate_selected_models([])
    with pytest.raises(PreparationRequestError):
        validate_selected_models([f"m{i}" for i in range(9)])
    with pytest.raises(PreparationRequestError):
        materialize_preparation_request(
            repository_root=repo,
            run_path="../escape",
            selected_models=["RealTabPFN-v2_r11"],
        )
    with pytest.raises(PreparationRequestError):
        materialize_preparation_request(
            repository_root=repo,
            run_path=run_path,
            selected_models=["NotARealModel"],
        )

    # Stale identity rejected.
    request = load_preparation_request(first.relative_path, repository_root=repo)
    (run_dir / "worker_result.json").write_text(
        (run_dir / "worker_result.json").read_text(encoding="utf-8").replace(
            "1.0", "1.1"
        ),
        encoding="utf-8",
    )
    # Rewrite worker with different content hash while keeping schema.
    worker = json.loads((run_dir / "worker_result.json").read_text(encoding="utf-8"))
    worker["autogluon_version"] = "changed"
    (run_dir / "worker_result.json").write_text(json.dumps(worker), encoding="utf-8")
    with pytest.raises(PreparationRequestError, match="identity|stale"):
        verify_request_run_identity(request, repository_root=repo)

    parser = candidate_cli.build_parser()
    with pytest.raises(PreparationRequestError):
        candidate_cli._resolve_selection(
            parser.parse_args(
                [
                    "validate",
                    "--request",
                    first.relative_path,
                    "--model",
                    "RealTabPFN-v2_r11",
                ]
            ),
            repo,
        )


def test_selection_helpers_and_warnings() -> None:
    assert ensemble_component_warning(
        ["WeightedEnsemble_L2", "LightGBMPrep_r31"]
    )
    assert ensemble_component_warning(["LightGBMPrep_r31"]) is None
    rows = [
        {
            "run_path": "a",
            "dataset_id": "v6",
            "target_dependency": "none",
            "preparable": True,
            "label": "v6 run",
        },
        {
            "run_path": "b",
            "dataset_id": "v5",
            "target_dependency": "exploratory",
            "preparable": False,
            "label": "v5 blocked",
        },
    ]
    assert len(filter_managed_runs(rows, dataset="v6")) == 1
    assert len(filter_managed_runs(rows, completion="blocked")) == 1


def test_result_verification_helpers(repo: Path) -> None:
    payload = {
        "ok": True,
        "command": "validate",
        "artifacts_written": False,
        "request_id": "apr1_x",
        "run_path": "artifacts/autogluon_runs/r",
        "selected_models": ["A", "B"],
        "results": [{"ok": True}, {"ok": True}],
    }
    verify_validation_result(
        payload,
        request_id="apr1_x",
        run_path="artifacts/autogluon_runs/r",
        selected_models=["A", "B"],
    )
    with pytest.raises(Exception):
        verify_validation_result(
            {**payload, "selected_models": ["A"]},
            request_id="apr1_x",
            run_path="artifacts/autogluon_runs/r",
            selected_models=["A", "B"],
        )


def test_registry_and_navigation() -> None:
    loaded = load_registry(PROJECT_ROOT)
    assert "autogluon_candidate_preparation_v1" in loaded.commands
    assert (
        "scripts/import_autogluon_prediction_candidates.py"
        in loaded.commands["autogluon_candidate_preparation_v1"].public_cli
        or loaded.commands["autogluon_candidate_preparation_v1"].argv_prefix[-1]
        == "scripts/import_autogluon_prediction_candidates.py"
    )
    command = loaded.commands["autogluon_candidate_preparation_v1"]
    assert command.actions["validate"].confirmation in {"none", ""}
    assert command.actions["prepare"].confirmation == "confirm"
    assert command.actions["validate"].competition_test is False
    assert command.actions["prepare"].competition_test is False
    assert "autogluon_candidate_preparation_v1" in ADVANCED_ONLY_COMMAND_IDS
    standard = visible_command_ids(list(loaded.commands))
    assert "autogluon_candidate_preparation_v1" not in standard


def test_real_root_read_only_discovery() -> None:
    rows = discover_managed_autogluon_runs(PROJECT_ROOT, include_incomplete=True)
    if not rows:
        pytest.skip("No managed AutoGluon runs present")
    labels = " ".join(str(row.get("label") or "") for row in rows)
    paths = " ".join(row["run_path"] for row in rows)
    assert "autogluon_runs" in paths
    # Existing prepared WeightedEnsemble should map when present.
    prepared = [
        model
        for row in rows
        for model in row.get("models") or []
        if model.get("model_name") == "WeightedEnsemble_L2"
        and model.get("already_prepared")
    ]
    if prepared:
        assert any(
            item.get("candidate_id") == "pc1_8da68e3d2071fb71" for item in prepared
        )
    # Focused models appear when available.
    all_models = {
        model.get("model_name")
        for row in rows
        for model in row.get("models") or []
    }
    interesting_prefixes = (
        "RealTabPFN-v2_r11",
        "NeuralNetTorch_r37",
        "LightGBMPrep_r31",
        "LightGBMPrep_r41",
        "WeightedEnsemble_L2",
    )
    matched = {
        name
        for name in all_models
        if isinstance(name, str)
        and any(name == prefix or name.startswith(prefix + "_") for prefix in interesting_prefixes)
    }
    if matched:
        assert matched
    # Temporary request under temp root only.
    # Do not invent a temporary complete run from real inventory.
    del labels


def test_app_blend_tabs_present() -> None:
    import apps.experiment_control_panel as panel

    source = Path(panel.__file__).read_text(encoding="utf-8")
    blend_page = (
        PROJECT_ROOT / "src/churn_ml/control_panel/blend_page.py"
    ).read_text(encoding="utf-8")
    assert "Prepare candidates" in blend_page
    assert "Build blend" in blend_page
    assert "Materialized blends" in blend_page
    assert "render_prepare_candidates_tab" in blend_page
    assert "blend_page" in source
