from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from src.churn_ml.config import ManualExperimentConfig
from src.churn_ml.manual_lightgbm import (
    CrossValidationResult,
    EvaluationResult,
    FeatureSchema,
    FoldResult,
    ParityReport,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def collect_environment_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "scikit_learn": _package_version("scikit-learn"),
        "lightgbm": _package_version("lightgbm"),
    }


def validate_environment_contract(
    actual: dict[str, str],
    expected: dict[str, Any],
) -> list[str]:
    return [
        f"{name}: expected {expected[name]}, actual {actual.get(name)}"
        for name in expected
        if str(expected[name]) != actual.get(name)
    ]


def collect_git_state(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    try:
        status = run("status", "--porcelain")
        return {
            "branch": run("branch", "--show-current"),
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(status),
            "status_porcelain": status.splitlines(),
        }
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return {
            "branch": None,
            "commit": None,
            "dirty": None,
            "error": str(error),
        }


def fingerprint_file(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "sha256": digest.hexdigest(),
    }


class RunFailurePersistenceError(RuntimeError):
    """Raised when one or more failed-run records cannot be persisted."""

    def __init__(self, errors: list[BaseException]) -> None:
        self.errors = errors
        details = "; ".join(
            f"{type(error).__name__}: {error}" for error in errors
        )
        super().__init__(f"Failed-run persistence errors: {details}")


class RunFailureFinalizationError(RuntimeError):
    """Preserve both the primary run error and failure-finalization error."""

    def __init__(
        self,
        primary_error: BaseException,
        finalization_error: BaseException,
    ) -> None:
        self.primary_error = primary_error
        self.finalization_error = finalization_error
        super().__init__(
            "Run failed and failure finalization also failed: "
            f"primary={type(primary_error).__name__}: {primary_error}; "
            "finalization="
            f"{type(finalization_error).__name__}: {finalization_error}"
        )


class RunArtifactStore:
    """Own the non-overwriting lifecycle of one manual experiment run."""

    def __init__(
        self,
        config: ManualExperimentConfig,
        *,
        train_index: pd.Index,
        test_index: pd.Index,
        run_id: str | None = None,
    ) -> None:
        self.config = config
        self.started_at = utc_now()
        self.train_index = train_index.to_numpy()
        self.test_index = test_index.to_numpy()
        self.fold_records: list[dict[str, Any]] = []
        self.run_id = run_id or self._generate_run_id()
        if not self.run_id or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in self.run_id
        ):
            raise ValueError(
                "run_id may contain only letters, numbers, hyphens, and underscores."
            )

        self.root = config.artifact_root / self.run_id
        initial_metadata: dict[str, Any] = {
            "experiment_id": config.experiment_id,
            "run_id": self.run_id,
            "status": "running",
            "started_at_utc": self.started_at.isoformat(),
        }
        self.root.mkdir(parents=True, exist_ok=False)
        try:
            (self.root / "models").mkdir()
            (self.root / "threshold_diagnostics").mkdir()
            (self.root / "submission").mkdir()
            self._write_status("running", completed_folds=0)
            self._write_yaml(
                self.root / "resolved_config.yaml",
                config.resolved_payload(),
            )
        except BaseException as error:
            self.fail_preserving(error, initial_metadata)
            raise

    def save_initial_contract(
        self,
        metadata: dict[str, Any],
        schema: FeatureSchema,
        fold_assignments: pd.DataFrame,
    ) -> None:
        self._write_json(self.root / "run_metadata.json", metadata)
        self._write_json(self.root / "feature_schema.json", schema.to_dict())
        fold_assignments.to_parquet(
            self.root / "fold_assignments.parquet",
            index=False,
        )

    def save_fold(self, fold: FoldResult) -> None:
        fold_dir = self.root / "models" / f"fold_{fold.fold:02d}"
        fold_dir.mkdir(exist_ok=False)
        artifacts = self.config.payload["artifacts"]
        if artifacts["save_fold_models"]:
            joblib.dump(fold.model, fold_dir / "model.joblib")
        if artifacts["save_encoder_states"]:
            joblib.dump(fold.encoder, fold_dir / "encoder.joblib")

        validation = pd.DataFrame(
            {
                "row_position": fold.validation_positions,
                "row_index": self.train_index[fold.validation_positions],
                "target": fold.validation_targets,
                "probability": fold.validation_probabilities,
            }
        )
        test = pd.DataFrame(
            {
                "row_position": np.arange(len(self.test_index), dtype=np.int64),
                "row_index": self.test_index,
                "probability": fold.test_probabilities,
            }
        )
        validation.to_parquet(
            fold_dir / "validation_predictions.parquet",
            index=False,
        )
        test.to_parquet(fold_dir / "test_predictions.parquet", index=False)
        fold_metadata = {
            "fold": fold.fold,
            "train_rows": len(fold.train_positions),
            "validation_rows": len(fold.validation_positions),
            "balanced_accuracy": fold.balanced_accuracy,
            "duration_seconds": fold.duration_seconds,
            "model_parameters": self.config.model_parameters,
            "encoder": {
                "n_splits": fold.encoder.n_splits,
                "alpha": fold.encoder.alpha,
                "random_state": fold.encoder.random_state,
                "categorical_features": fold.encoder.categorical_columns_,
                "transformed_features": (
                    fold.encoder.passthrough_columns_
                    + [
                        f"{column}__te"
                        for column in fold.encoder.categorical_columns_
                    ]
                ),
            },
        }
        self._write_json(fold_dir / "fold_metadata.json", fold_metadata)

        self.fold_records.append(
            {
                "fold": fold.fold,
                "train_rows": len(fold.train_positions),
                "validation_rows": len(fold.validation_positions),
                "decision_threshold": self.config.payload["thresholds"]["submission"],
                "balanced_accuracy": fold.balanced_accuracy,
                "predicted_positive_count": int(
                    (
                        fold.validation_probabilities
                        >= self.config.payload["thresholds"]["submission"]
                    ).sum()
                ),
                "duration_seconds": fold.duration_seconds,
            }
        )
        pd.DataFrame(self.fold_records).to_csv(
            self.root / "fold_metrics.csv",
            index=False,
        )
        self._write_status(
            "running",
            completed_folds=len(self.fold_records),
        )

    def save_final_outputs(
        self,
        result: CrossValidationResult,
        evaluation: EvaluationResult,
        submission: pd.DataFrame,
        parity_report: ParityReport | None,
    ) -> None:
        result.fold_metrics.to_csv(self.root / "fold_metrics.csv", index=False)
        result.oof_predictions.to_parquet(
            self.root / "oof_predictions.parquet",
            index=False,
        )
        result.test_predictions.to_parquet(
            self.root / "test_predictions.parquet",
            index=False,
        )
        self._write_json(self.root / "metrics.json", evaluation.metrics)
        evaluation.global_threshold_curve.to_csv(
            self.root / "threshold_diagnostics" / "global_threshold_curve.csv",
            index=False,
        )
        evaluation.legacy_cross_fitted_fold_metrics.to_csv(
            self.root
            / "threshold_diagnostics"
            / "legacy_cross_fitted_folds.csv",
            index=False,
        )
        evaluation.legacy_cross_fitted_predictions.to_parquet(
            self.root
            / "threshold_diagnostics"
            / "legacy_cross_fitted_predictions.parquet",
            index=False,
        )
        submission.to_csv(
            self.root
            / "submission"
            / "manual_lightgbmprep_r31_threshold_0117.csv",
            index=False,
        )
        if parity_report is not None:
            self._write_json(
                self.root / "parity_report.json",
                parity_report.to_dict(),
            )

    def complete(self, metadata: dict[str, Any]) -> None:
        self._validate_completion_preconditions()
        finished_at = utc_now()
        metadata["finished_at_utc"] = finished_at.isoformat()
        metadata["duration_seconds"] = (
            finished_at - self.started_at
        ).total_seconds()
        metadata["status"] = "completed"
        metadata["artifact_inventory"] = sorted(
            {*self.inventory(), "_SUCCESS"}
        )
        self._write_json(self.root / "run_metadata.json", metadata)
        self._write_status(
            "completed",
            completed_folds=len(self.fold_records),
            finished_at=finished_at,
        )
        (self.root / "_SUCCESS").touch(exist_ok=False)

    def fail_preserving(
        self,
        error: BaseException,
        metadata: dict[str, Any],
    ) -> None:
        """Persist failure or raise an error retaining both failure causes."""
        try:
            self.fail(error, metadata)
        except BaseException as finalization_error:
            raise RunFailureFinalizationError(
                error,
                finalization_error,
            ) from error

    def fail(self, error: BaseException, metadata: dict[str, Any]) -> None:
        if (self.root / "_SUCCESS").exists():
            raise RuntimeError("A completed run cannot be finalized as failed.")

        finished_at = utc_now()
        reason = {
            "type": type(error).__name__,
            "message": str(error),
        }
        metadata["finished_at_utc"] = finished_at.isoformat()
        metadata["duration_seconds"] = (
            finished_at - self.started_at
        ).total_seconds()
        metadata["status"] = "failed"
        metadata["failure"] = reason
        metadata["artifact_inventory"] = sorted(
            {*self.inventory(), "_FAILED"}
        )

        persistence_errors: list[BaseException] = []
        try:
            self._write_json(self.root / "run_metadata.json", metadata)
        except BaseException as persistence_error:
            persistence_errors.append(persistence_error)
        try:
            self._write_status(
                "failed",
                completed_folds=len(self.fold_records),
                finished_at=finished_at,
                failure=reason,
            )
        except BaseException as persistence_error:
            persistence_errors.append(persistence_error)
        try:
            (self.root / "_FAILED").touch(exist_ok=True)
        except BaseException as persistence_error:
            persistence_errors.append(persistence_error)
        if persistence_errors:
            raise RunFailurePersistenceError(persistence_errors)

    def _validate_completion_preconditions(self) -> None:
        errors: list[str] = []
        expected_folds = int(self.config.payload["outer_cv"]["n_splits"])
        expected_fold_numbers = set(range(1, expected_folds + 1))
        recorded_fold_numbers = {
            int(record["fold"]) for record in self.fold_records
        }
        if len(self.fold_records) != expected_folds:
            errors.append(
                f"expected {expected_folds} fold records, got {len(self.fold_records)}"
            )
        if recorded_fold_numbers != expected_fold_numbers:
            errors.append(
                "recorded fold numbers differ from the complete expected set"
            )

        required_files = (
            "resolved_config.yaml",
            "run_metadata.json",
            "execution_status.json",
            "feature_schema.json",
            "fold_assignments.parquet",
            "fold_metrics.csv",
            "oof_predictions.parquet",
            "test_predictions.parquet",
            "metrics.json",
            "threshold_diagnostics/global_threshold_curve.csv",
            "threshold_diagnostics/legacy_cross_fitted_folds.csv",
            "threshold_diagnostics/legacy_cross_fitted_predictions.parquet",
            "submission/manual_lightgbmprep_r31_threshold_0117.csv",
        )
        missing_files = [
            relative_path
            for relative_path in required_files
            if not (self.root / relative_path).is_file()
        ]
        if missing_files:
            errors.append(f"missing required final artifacts: {missing_files}")

        expected_fold_directories = {
            f"fold_{fold:02d}" for fold in expected_fold_numbers
        }
        models_root = self.root / "models"
        actual_fold_directories = (
            {
                path.name
                for path in models_root.iterdir()
                if path.is_dir()
            }
            if models_root.is_dir()
            else set()
        )
        if actual_fold_directories != expected_fold_directories:
            errors.append(
                "fold artifact directories differ from the expected eight-fold set"
            )
        fold_files = (
            "model.joblib",
            "encoder.joblib",
            "fold_metadata.json",
            "validation_predictions.parquet",
            "test_predictions.parquet",
        )
        for fold_directory in sorted(expected_fold_directories):
            for filename in fold_files:
                path = models_root / fold_directory / filename
                if not path.is_file():
                    errors.append(
                        f"missing fold artifact: models/{fold_directory}/{filename}"
                    )

        if (self.root / "_FAILED").exists():
            errors.append("_FAILED already exists")
        if (self.root / "_SUCCESS").exists():
            errors.append("_SUCCESS already exists")

        if self.config.payload["parity"]["required"]:
            parity_path = self.root / "parity_report.json"
            if not parity_path.is_file():
                errors.append("required parity_report.json is missing")
            else:
                try:
                    with parity_path.open("r", encoding="utf-8") as file:
                        parity_report = json.load(file)
                    primary_gates = parity_report.get("primary_gates")
                    if (
                        parity_report.get("enabled") is not True
                        or parity_report.get("required") is not True
                        or parity_report.get("passed") is not True
                        or not isinstance(primary_gates, dict)
                        or not primary_gates
                        or not all(
                            isinstance(gate, dict) and gate.get("passed") is True
                            for gate in primary_gates.values()
                        )
                    ):
                        errors.append("required parity report is absent or failed")
                except (OSError, ValueError, TypeError) as error:
                    errors.append(f"required parity report is unreadable: {error}")

        if errors:
            raise RuntimeError(
                "Completion preconditions failed:\n- " + "\n- ".join(errors)
            )
    def inventory(self) -> list[str]:
        return sorted(
            str(path.relative_to(self.root)).replace("\\", "/")
            for path in self.root.rglob("*")
            if path.is_file()
        )

    def _generate_run_id(self) -> str:
        config_json = json.dumps(
            self.config.resolved_payload(),
            sort_keys=True,
            separators=(",", ":"),
        )
        config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:8]
        timestamp = self.started_at.strftime("%Y%m%dT%H%M%S%fZ")
        return f"{timestamp}_{config_hash}"

    def _write_status(
        self,
        status: str,
        *,
        completed_folds: int,
        finished_at: datetime | None = None,
        failure: dict[str, str] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "status": status,
            "started_at_utc": self.started_at.isoformat(),
            "finished_at_utc": (
                finished_at.isoformat() if finished_at is not None else None
            ),
            "completed_folds": completed_folds,
            "expected_folds": self.config.payload["outer_cv"]["n_splits"],
            "failure": failure,
        }
        self._write_json(self.root / "execution_status.json", payload)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(
                payload,
                file,
                indent=2,
                ensure_ascii=False,
                default=_json_default,
            )
            file.write("\n")
        temporary.replace(path)

    @staticmethod
    def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=True)
        temporary.replace(path)


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}.")
