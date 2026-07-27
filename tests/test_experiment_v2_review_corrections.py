from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from src.churn_ml import research_v2_cli
from src.churn_ml.experiment_v2 import (
    get_candidate_adapter,
    get_feature_pipeline,
)
from src.churn_ml.research_evaluation import run_research_evaluation
from src.churn_ml.research_protocol import (
    ThresholdSelectionResult,
    build_evaluation_assignments,
)
from src.churn_ml.research_v2_artifact_validation import (
    ResearchV2SemanticValidationError,
    validate_portable_payload_paths,
    validate_research_v2_run,
)
from src.churn_ml.research_v2_artifacts import (
    ResearchV2ArtifactError,
    ResearchV2ArtifactStore,
)
from src.churn_ml.research_v2_config import (
    ResearchV2ConfigurationError,
    load_research_v2_config,
)
from src.churn_ml.research_v2_data import (
    ResearchV2DataError,
    load_research_v2_training_data,
)
from src.churn_ml.research_v2_identity import (
    ADAPTER_IMPLEMENTATION_SOURCES,
    PIPELINE_IMPLEMENTATION_SOURCES,
    build_component_identities,
)
from tests.research_v2_test_support import (
    PROJECT_ROOT,
    SMOKE_CONFIG,
    build_persisted_v2_test_run,
    build_v2_test_preflight,
)


TRAIN_PATH = PROJECT_ROOT / "data/processed/v3_targeted_missingness/X_train.parquet"


@pytest.mark.parametrize(
    "component,path,value",
    [
        ("pipeline", "source_feature_count", "217"),
        ("pipeline", "source_feature_count", False),
        ("pipeline", "expected_model_feature_count", 213.0),
        ("adapter", "target_encoder.inner_splits", 5.0),
        ("adapter", "target_encoder.shuffle", 1),
        ("adapter", "target_encoder.random_state", False),
        ("adapter", "target_encoder.alpha", 10),
        ("adapter", "target_encoder.keep_original_categorical_features", 0),
        ("adapter", "lightgbm.parameters.n_estimators", 376.0),
        ("adapter", "lightgbm.parameters.extra_trees", 0),
        ("adapter", "lightgbm.parameters.random_state", False),
        ("adapter", "lightgbm.parameters.n_jobs", -1.0),
    ],
)
def test_contracts_reject_equality_compatible_wrong_types(
    component: str,
    path: str,
    value: Any,
) -> None:
    payload = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
    section = (
        payload["feature_pipeline"]["contract"]
        if component == "pipeline"
        else payload["candidate_adapter"]["contract"]
    )
    target = section
    parts = path.split(".")
    for name in parts[:-1]:
        target = target[name]
    target[parts[-1]] = value
    validator = (
        get_feature_pipeline(payload["feature_pipeline"]["id"])
        if component == "pipeline"
        else get_candidate_adapter(payload["candidate_adapter"]["id"])
    )
    with pytest.raises(ValueError, match=re.escape(path)):
        validator.validate_contract(section)


def test_component_identities_are_independent_complete_and_portable() -> None:
    paths = set(PIPELINE_IMPLEMENTATION_SOURCES) | set(ADAPTER_IMPLEMENTATION_SOURCES)
    sources = {path: f"{index:064x}" for index, path in enumerate(sorted(paths), 1)}

    def build_identities(
        source_records: dict[str, str],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        return build_component_identities(
            pipeline_inputs={"id": "pipeline", "contract": {"version": 1}},
            adapter_inputs={"id": "adapter", "contract": {"version": 1}},
            resolved_feature_schema={"ordered": ["a", "b"]},
            runtime_dependencies={"python": "3.12"},
            dataset_version="unit",
            source_records=source_records,
        )

    identities, hashes = build_identities(sources)
    adapter_changed = dict(sources)
    adapter_changed["src/churn_ml/experiment_v2_adapter.py"] = "a" * 64
    _, adapter_hashes = build_identities(adapter_changed)
    assert adapter_hashes["feature_pipeline"] == hashes["feature_pipeline"]
    assert adapter_hashes["candidate_adapter"] != hashes["candidate_adapter"]
    assert adapter_hashes["candidate"] != hashes["candidate"]

    pipeline_changed = dict(sources)
    pipeline_changed["src/churn_ml/experiment_v2_pipeline.py"] = "b" * 64
    _, pipeline_hashes = build_identities(pipeline_changed)
    assert pipeline_hashes["feature_pipeline"] != hashes["feature_pipeline"]
    assert pipeline_hashes["candidate_adapter"] == hashes["candidate_adapter"]
    assert pipeline_hashes["candidate"] != hashes["candidate"]

    encoder_changed = dict(sources)
    encoder_changed["src/churn_ml/target_encoding.py"] = "c" * 64
    _, encoder_hashes = build_identities(encoder_changed)
    assert encoder_hashes["feature_pipeline"] == hashes["feature_pipeline"]
    assert encoder_hashes["candidate_adapter"] != hashes["candidate_adapter"]
    assert encoder_hashes["candidate"] != hashes["candidate"]
    validate_portable_payload_paths(
        identities,
        label="identities",
        project_root=PROJECT_ROOT,
    )
    assert not any(
        Path(record["path"]).is_absolute()
        for identity in identities.values()
        for record in identity["canonical"]
        .get("implementation_sources", {})
        .get("files", [])
    )


@pytest.mark.parametrize(
    "nonportable_path",
    [
        r"C:\absolute\path",
        "C:/absolute/path",
        r"C:relative",
        "C:",
        r"\\server\share",
        r"\rooted",
        "/absolute/path",
    ],
)
def test_portable_payload_validator_rejects_all_host_absolute_path_forms(
    nonportable_path: str,
) -> None:
    payload = {"canonical": {"source": {"path": nonportable_path}}}
    with pytest.raises(
        ResearchV2SemanticValidationError,
        match=r"identities\.candidate\.canonical\.source\.path",
    ):
        validate_portable_payload_paths(
            payload["canonical"],
            label="identities.candidate.canonical",
            project_root=PROJECT_ROOT,
        )


def test_portable_payload_validator_allows_relative_paths_and_punctuation() -> None:
    validate_portable_payload_paths(
        {
            "source": {"path": "src/churn_ml/experiment_v2.py"},
            "semantics": "binary_positive_class_label_1",
            "punctuation": "metric: balanced_accuracy (v2)!",
        },
        label="identities.candidate.canonical",
        project_root=PROJECT_ROOT,
    )


def test_dataset_processed_root_traversal_is_mandatorily_rejected() -> None:
    config = load_research_v2_config(SMOKE_CONFIG, project_root=PROJECT_ROOT)
    plan = deepcopy(config.plan_payload)
    plan["dataset"]["processed_dir"] = "../escape"
    invalid = type(config)(
        payload=config.payload,
        plan_payload=plan,
        source_path=config.source_path,
        plan_path=config.plan_path,
        project_root=config.project_root,
    )
    with pytest.raises(ResearchV2ConfigurationError, match="repository-relative"):
        load_research_v2_training_data(invalid)


def test_dataset_version_symlink_escape_is_rejected_where_supported(
    tmp_path: Path,
) -> None:
    config = load_research_v2_config(SMOKE_CONFIG, project_root=PROJECT_ROOT)
    link_name = f"v2_symlink_escape_{tmp_path.name}"
    link = PROJECT_ROOT / "data" / "processed" / link_name
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        try:
            os.symlink(outside, link, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            pytest.skip(f"Directory symlink creation is unavailable: {error}")
        payload = deepcopy(config.payload)
        payload["dataset"]["version"] = link_name
        plan = deepcopy(config.plan_payload)
        plan["dataset"]["version"] = link_name
        escaped = type(config)(
            payload=payload,
            plan_payload=plan,
            source_path=config.source_path,
            plan_path=config.plan_path,
            project_root=config.project_root,
        )
        with pytest.raises(ResearchV2DataError, match="version directory escapes"):
            load_research_v2_training_data(escaped)
    finally:
        if link.exists() or link.is_symlink():
            link.unlink()


def test_production_evaluation_classifies_probability_equal_to_threshold_positive() -> (
    None
):
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], name="y", dtype="int64")
    X = pd.DataFrame({"row": np.arange(len(y), dtype=float)})
    plan = {
        "outer_evaluation": {
            "n_splits": 2,
            "shuffle": True,
            "repeat_seeds": [0],
        },
        "threshold_selection": {
            "n_splits": 2,
            "shuffle": True,
            "random_state": 17,
        },
        "threshold_policy": {
            "minimum": 0.1,
            "maximum": 0.9,
            "step": 0.1,
            "maximizer_absolute_tolerance": 1e-12,
            "constant_probability_fallback": 0.5,
        },
    }
    assignments = build_evaluation_assignments(y, plan)

    def fit_predict(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_prediction: pd.DataFrame,
        contract: dict[str, Any],
        *,
        model_training_positions: np.ndarray,
        prediction_positions: np.ndarray,
        audit_callback: Any,
    ) -> np.ndarray:
        del X_train, y_train, X_prediction, contract, model_training_positions
        del audit_callback
        return np.where(
            prediction_positions == 0, 0.5, 0.2 + 0.6 * (prediction_positions % 2)
        )

    def fixed_selector(
        targets: pd.Series | np.ndarray,
        probabilities: np.ndarray,
        policy: dict[str, Any],
    ) -> ThresholdSelectionResult:
        del targets, probabilities, policy
        return ThresholdSelectionResult(
            threshold=0.5,
            balanced_accuracy=0.5,
            degenerate=False,
            status="selected",
            scores=pd.DataFrame({"threshold": [0.5], "balanced_accuracy": [0.5]}),
        )

    result = run_research_evaluation(
        X,
        y,
        assignments,
        plan,
        {},
        fit_predict=fit_predict,
        threshold_selector=fixed_selector,
    )
    equality = result.outer_validation.loc[
        result.outer_validation["probability"] == 0.5
    ]
    assert len(equality) == 1
    assert equality["prediction"].tolist() == [1]


@pytest.mark.skipif(not TRAIN_PATH.exists(), reason="Local v3 data unavailable.")
@pytest.mark.parametrize(
    "corruption",
    [
        "wrong_fold_key",
        "missing_row_coverage",
        "duplicate_row_coverage",
        "mismatched_status",
        "altered_threshold",
        "altered_metric",
        "manifest_hash",
        "forbidden_artifact",
    ],
)
def test_independent_semantic_validator_rejects_corruption(
    tmp_path: Path,
    corruption: str,
) -> None:
    store, result, metadata, prepared = build_persisted_v2_test_run(tmp_path)
    root = store.root
    try:
        store.complete(metadata, result)
        if corruption == "wrong_fold_key":
            path = root / "thresholds/selected_thresholds.csv"
            frame = pd.read_csv(path)
            frame.loc[0, "outer_fold"] = 99
            frame.to_csv(path, index=False)
        elif corruption in {"missing_row_coverage", "duplicate_row_coverage"}:
            path = root / "predictions/outer_validation.parquet"
            frame = pd.read_parquet(path)
            if corruption == "missing_row_coverage":
                frame = frame.iloc[:-1].copy()
            else:
                frame.loc[0, "row_position"] = frame.loc[1, "row_position"]
            frame.to_parquet(path, index=False)
        elif corruption == "mismatched_status":
            path = root / "execution_status.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["status"] = "failed"
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        elif corruption == "altered_threshold":
            path = root / "thresholds/selected_thresholds.csv"
            frame = pd.read_csv(path)
            frame.loc[0, "selected_threshold"] += 0.001
            frame.to_csv(path, index=False)
        elif corruption == "altered_metric":
            path = root / "metrics/outer_folds.csv"
            frame = pd.read_csv(path)
            frame.loc[0, "balanced_accuracy"] += 0.01
            frame.to_csv(path, index=False)
        elif corruption == "manifest_hash":
            path = root / "artifact_manifest.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["manifest_sha256"] = "0" * 64
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        elif corruption == "forbidden_artifact":
            (root / "submission.csv").write_text("y\n1\n", encoding="utf-8")
        else:
            raise AssertionError(corruption)
        with pytest.raises(ResearchV2SemanticValidationError):
            validate_research_v2_run(
                root,
                prepared.config,
                expected_hashes=prepared.hashes,
                require_success=True,
                verify_manifest=True,
            )
    finally:
        shutil.rmtree(prepared.config.artifact_root, ignore_errors=True)


@pytest.mark.skipif(not TRAIN_PATH.exists(), reason="Local v3 data unavailable.")
def test_success_is_newest_file_and_independently_validated(tmp_path: Path) -> None:
    store, result, metadata, prepared = build_persisted_v2_test_run(tmp_path)
    try:
        store.complete(metadata, result)
        validate_research_v2_run(
            store.root,
            prepared.config,
            expected_hashes=prepared.hashes,
            require_success=True,
            verify_manifest=True,
        )
        success_mtime = (store.root / "_SUCCESS").stat().st_mtime_ns
        ordinary_mtimes = [
            path.stat().st_mtime_ns
            for path in store.root.rglob("*")
            if path.is_file() and path.name != "_SUCCESS"
        ]
        assert success_mtime >= max(ordinary_mtimes)
        assert not (store.root / "_FAILED").exists()
    finally:
        shutil.rmtree(prepared.config.artifact_root, ignore_errors=True)


@pytest.mark.skipif(not TRAIN_PATH.exists(), reason="Local v3 data unavailable.")
def test_last_pre_success_failure_creates_failed_and_command_reports_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = build_v2_test_preflight(tmp_path)
    monkeypatch.setattr(research_v2_cli, "preflight", lambda path: prepared)

    def fail_last(
        self: ResearchV2ArtifactStore,
        output: Any,
    ) -> None:
        del self, output
        raise RuntimeError("last pre-success operation failed")

    monkeypatch.setattr(
        ResearchV2ArtifactStore,
        "_run_pre_success_output",
        fail_last,
    )
    try:
        code = research_v2_cli.execute(
            argparse.Namespace(
                config=SMOKE_CONFIG,
                validate_only=False,
                run_id="pre-success-failure",
            )
        )
        captured = capsys.readouterr()
        failed_markers = list(prepared.config.artifact_root.rglob("_FAILED"))
        assert code == 1
        assert "Status: failed" in captured.err
        assert len(failed_markers) == 1
        root = failed_markers[0].parent
        assert not (root / "_SUCCESS").exists()
    finally:
        shutil.rmtree(prepared.config.artifact_root, ignore_errors=True)


def test_explicit_run_id_safe_slug_boundaries(tmp_path: Path) -> None:
    config = load_research_v2_config(SMOKE_CONFIG, project_root=PROJECT_ROOT)
    payload = deepcopy(config.payload)
    artifact_root = PROJECT_ROOT / "artifacts/research_v2_tests" / tmp_path.name
    payload["artifacts"]["root"] = artifact_root.relative_to(PROJECT_ROOT).as_posix()
    test_config = type(config)(
        payload=payload,
        plan_payload=config.plan_payload,
        source_path=config.source_path,
        plan_path=config.plan_path,
        project_root=config.project_root,
    )
    valid = "a" * 128
    try:
        store = ResearchV2ArtifactStore(
            test_config,
            plan_hash="a" * 64,
            candidate_hash="b" * 64,
            run_id=valid,
        )
        assert store.run_id == valid
        for invalid in ("a" * 129, "unsafe.value", "-leading"):
            with pytest.raises(ResearchV2ArtifactError, match="safe slug"):
                ResearchV2ArtifactStore(
                    test_config,
                    plan_hash="a" * 64,
                    candidate_hash="b" * 64,
                    run_id=invalid,
                )
    finally:
        shutil.rmtree(artifact_root, ignore_errors=True)
