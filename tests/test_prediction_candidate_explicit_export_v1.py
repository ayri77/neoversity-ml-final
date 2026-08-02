"""Focused tests for source-neutral explicit prediction export import."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.churn_ml.prediction_candidates import cli as candidate_cli
from src.churn_ml.prediction_candidates.contract_v1 import (
    SUCCESS_FILENAME,
    CandidateValidationError,
    PredictionCandidateError,
)
from src.churn_ml.prediction_candidates.explicit_export_v1 import (
    ExplicitExportRequest,
    import_explicit_export_candidate,
    prepare_explicit_export,
    validate_explicit_export,
)
from tests.dataset_registry_support import (
    create_synthetic_legacy_tree,
    register_package,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    processed = create_synthetic_legacy_tree(root / "data" / "processed")
    register_package(
        processed,
        "v0_raw_minimal",
        parent_dataset_id=None,
        hypothesis="synthetic v0",
    )
    register_package(
        processed,
        "v1_missingness_summary",
        parent_dataset_id="v0_raw_minimal",
        hypothesis="synthetic exploratory v1",
        target_dependency="exploratory",
        summary_features=("missing_count_total", "missing_rate_total"),
    )
    (root / "artifacts" / "legacy" / "run" / "exports").mkdir(
        parents=True, exist_ok=True
    )
    _write_exports(root)
    return root


def _write_exports(
    repo: Path,
    *,
    oof: pd.DataFrame | None = None,
    test: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    export_dir = repo / "artifacts" / "legacy" / "run" / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    y = pd.read_parquet(
        repo / "data" / "processed" / "v0_raw_minimal" / "y_train.parquet"
    ).iloc[:, 0]
    oof_frame = oof
    if oof_frame is None:
        oof_frame = pd.DataFrame(
            {
                "row_index": np.arange(len(y), dtype=np.int64),
                "target": y.to_numpy(dtype=np.int64),
                "probability": np.linspace(0.1, 0.9, len(y), dtype=np.float64),
            }
        )
    test_frame = test
    if test_frame is None:
        test_frame = pd.DataFrame(
            {
                "row_index": np.arange(3, dtype=np.int64),
                "probability": np.array([0.05, 0.4, 0.8], dtype=np.float64),
            }
        )
    oof_path = export_dir / "oof.parquet"
    test_path = export_dir / "test.parquet"
    oof_frame.to_parquet(oof_path, index=False)
    test_frame.to_parquet(test_path, index=False)
    return oof_path, test_path


def _request(**overrides) -> ExplicitExportRequest:
    request = ExplicitExportRequest(
        oof_path=Path("artifacts/legacy/run/exports/oof.parquet"),
        test_path=Path("artifacts/legacy/run/exports/test.parquet"),
        dataset_id="v0_raw_minimal",
        source_model_name="LegacyModel",
        source_kind="legacy_explicit_export_v1",
        source_run_path="artifacts/legacy/run",
        threshold=0.25,
        exploratory=False,
        validation_balanced_accuracy=0.8,
        historical_kaggle_public_score=0.7,
        source_model_type="LegacyModelType",
    )
    return replace(request, **overrides)


def test_valid_import_schema_normalization_and_provenance(repo: Path) -> None:
    package = import_explicit_export_candidate(_request(), repository_root=repo)
    assert list(package.oof.columns) == [
        "row_position",
        "target",
        "probability_positive",
    ]
    assert list(package.test.columns) == ["row_position", "probability_positive"]
    assert package.manifest["source_kind"] == "legacy_explicit_export_v1"
    assert package.source_metadata["predictor_loaded"] is False
    assert package.source_metadata["predictions_generated"] is False
    assert package.source_metadata["external_evidence"]["role"].startswith(
        "external_evidence_only"
    )
    assert package.source_metadata["oof_export"]["detected_schema"]["columns"] == [
        "row_index",
        "target",
        "probability",
    ]


def test_canonical_schema_and_optional_oof_target_are_normalized(repo: Path) -> None:
    y = pd.read_parquet(
        repo / "data" / "processed" / "v0_raw_minimal" / "y_train.parquet"
    ).iloc[:, 0]
    _write_exports(
        repo,
        oof=pd.DataFrame(
            {
                "row_position": np.arange(len(y), dtype=np.int64),
                "probability_positive": np.linspace(0.1, 0.9, len(y)),
            }
        ),
        test=pd.DataFrame(
            {
                "index": np.arange(3, dtype=np.int64),
                "proba": [0.1, 0.2, 0.3],
            }
        ),
    )
    prepared = prepare_explicit_export(_request(), repository_root=repo)
    assert prepared.oof["target"].tolist() == y.tolist()
    assert prepared.test["probability_positive"].tolist() == [0.1, 0.2, 0.3]


@pytest.mark.parametrize(
    ("mutation", "reason_code"),
    [
        (lambda frame: frame.iloc[:-1].copy(), "explicit_missing_rows"),
        (
            lambda frame: frame.assign(row_index=[0, 1, 2, 3, 4, 4]),
            "explicit_duplicate_rows",
        ),
        (
            lambda frame: frame.assign(probability=[0.1, 0.2, np.nan, 0.4, 0.5, 0.6]),
            "probability_nonfinite",
        ),
        (
            lambda frame: frame.assign(probability=[0.1, 0.2, 1.1, 0.4, 0.5, 0.6]),
            "probability_out_of_range",
        ),
        (
            lambda frame: frame.iloc[[1, 0, 2, 3, 4, 5]].reset_index(drop=True),
            "explicit_row_order_mismatch",
        ),
    ],
)
def test_invalid_rows_and_probabilities_rejected(
    repo: Path, mutation, reason_code: str
) -> None:
    path = repo / "artifacts" / "legacy" / "run" / "exports" / "oof.parquet"
    frame = mutation(pd.read_parquet(path))
    frame.to_parquet(path, index=False)
    with pytest.raises(CandidateValidationError) as captured:
        prepare_explicit_export(_request(), repository_root=repo)
    assert captured.value.reason_code == reason_code


def test_equal_counts_without_row_identity_evidence_rejected(repo: Path) -> None:
    path = repo / "artifacts" / "legacy" / "run" / "exports" / "test.parquet"
    frame = pd.read_parquet(path).drop(columns="row_index")
    frame.to_parquet(path, index=False)
    with pytest.raises(CandidateValidationError) as captured:
        prepare_explicit_export(_request(), repository_root=repo)
    assert captured.value.reason_code == "explicit_identity_evidence_missing"


def test_wrong_dataset_target_identity_rejected(repo: Path) -> None:
    path = repo / "artifacts" / "legacy" / "run" / "exports" / "oof.parquet"
    frame = pd.read_parquet(path)
    frame["target"] = 1 - frame["target"]
    frame.to_parquet(path, index=False)
    with pytest.raises(CandidateValidationError) as captured:
        prepare_explicit_export(_request(), repository_root=repo)
    assert captured.value.reason_code == "target_mismatch"


def test_absolute_paths_and_missing_provenance_rejected(repo: Path) -> None:
    absolute = (
        repo / "artifacts" / "legacy" / "run" / "exports" / "oof.parquet"
    ).resolve()
    with pytest.raises(PredictionCandidateError) as captured:
        prepare_explicit_export(_request(oof_path=absolute), repository_root=repo)
    assert captured.value.reason_code == "path_traversal"

    with pytest.raises(PredictionCandidateError) as captured2:
        prepare_explicit_export(_request(source_model_name=""), repository_root=repo)
    assert captured2.value.reason_code == "explicit_provenance_missing"


def test_exploratory_declaration_propagates_exactly(repo: Path) -> None:
    with pytest.raises(CandidateValidationError) as captured:
        prepare_explicit_export(
            _request(dataset_id="v1_missingness_summary", exploratory=False),
            repository_root=repo,
        )
    assert captured.value.reason_code == "exploratory_declaration_mismatch"

    package = import_explicit_export_candidate(
        _request(dataset_id="v1_missingness_summary", exploratory=True),
        repository_root=repo,
    )
    assert package.manifest["exploratory"] is True
    assert package.manifest["target_dependency"] == "exploratory"


def test_immutable_idempotence_and_success_written_last(repo: Path) -> None:
    first = import_explicit_export_candidate(_request(), repository_root=repo)
    success = first.package_dir / SUCCESS_FILENAME
    first_success_mtime = success.stat().st_mtime_ns
    second = import_explicit_export_candidate(_request(), repository_root=repo)
    assert second.candidate_id == first.candidate_id
    assert success.stat().st_mtime_ns == first_success_mtime
    success_mtime = success.stat().st_mtime_ns
    assert all(
        path.name == SUCCESS_FILENAME or path.stat().st_mtime_ns <= success_mtime
        for path in first.package_dir.iterdir()
    )


def test_cli_import_explicit_never_loads_predictor_or_autogluon(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        candidate_cli,
        "load_predictor",
        lambda *_args, **_kwargs: pytest.fail("predictor loading is prohibited"),
    )
    before = {
        name
        for name in sys.modules
        if name == "autogluon" or name.startswith("autogluon.")
    }
    code = candidate_cli.main(
        [
            "import-explicit",
            "--oof",
            "artifacts/legacy/run/exports/oof.parquet",
            "--test",
            "artifacts/legacy/run/exports/test.parquet",
            "--dataset-id",
            "v0_raw_minimal",
            "--source-model-name",
            "LegacyModel",
            "--source-kind",
            "legacy_explicit_export_v1",
            "--source-run-path",
            "artifacts/legacy/run",
            "--threshold",
            "0.25",
            "--non-exploratory",
        ],
        repository_root=repo,
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidate_id"].startswith("pc1_")
    assert payload["predictor_loaded"] is False
    after = {
        name
        for name in sys.modules
        if name == "autogluon" or name.startswith("autogluon.")
    }
    assert after == before


def test_historical_lightgbmprep_export_when_available() -> None:
    root = Path(__file__).resolve().parents[1]
    run = Path("artifacts/autogluon/autogluon_v3_extreme_8h_20260726")
    oof = run / "exports/lightgbmprep_r31/oof_predictions.parquet"
    test = run / "exports/lightgbmprep_r31/test_predictions.parquet"
    if not (root / oof).is_file() or not (root / test).is_file():
        pytest.skip(
            "Historical LightGBMPrep explicit exports are not available locally."
        )
    result = validate_explicit_export(
        ExplicitExportRequest(
            oof_path=oof,
            test_path=test,
            dataset_id="v3_targeted_missingness",
            source_model_name="LightGBMPrep_r31_BAG_L1",
            source_kind="autogluon_legacy_explicit_export_v1",
            source_run_path=run.as_posix(),
            threshold=0.117,
            exploratory=True,
            validation_balanced_accuracy=0.895158,
            historical_kaggle_public_score=0.9112,
        ),
        repository_root=root,
    )
    assert result["oof_export"]["row_count"] == 10_000
    assert result["test_export"]["row_count"] == 2_500
    assert result["predicted_positive_count"] == 541
    historical = root / "submissions/autogluon_v3_extreme_threshold_0117.csv"
    if historical.is_file():
        probabilities = pd.read_parquet(root / test)["probability"].to_numpy()
        labels = (probabilities >= 0.117).astype(np.int8)
        saved = pd.read_csv(historical)
        assert list(saved.columns) == ["index", "y"]
        assert np.array_equal(saved["y"].to_numpy(dtype=np.int8), labels)
