from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path
    models: Path
    reports: Path
    metadata: Path
    config: Path
    environment: Path
    metrics: Path
    fold_metrics: Path
    oof_predictions: Path
    test_predictions: Path
    feature_importance: Path
    submission: Path
    index: Path


class Experiment:
    INDEX_COLUMNS = [
        "run_id",
        "name",
        "model_type",
        "created_at",
        "status",
        "cv_score",
        "cv_std",
        "oof_score",
        "threshold",
        "public_score",
        "duration_seconds",
        "seed",
        "n_folds",
        "git_commit",
        "git_dirty",
        "notes",
        "artifact_path",
    ]

    TRACKED_PACKAGES = [
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "catboost",
        "lightgbm",
        "xgboost",
        "optuna",
        "shap",
        "mlflow",
    ]

    def __init__(
        self,
        name: str,
        model_type: str,
        *,
        description: str = "",
        notes: str = "",
        seed: int | None = None,
        n_folds: int | None = None,
        artifacts_dir: str | Path = "artifacts/experiments",
        train_path: str | Path | None = None,
        test_path: str | Path | None = None,
    ) -> None:
        self.name = name
        self.model_type = model_type
        self.description = description
        self.notes = notes
        self.seed = seed
        self.n_folds = n_folds

        self.created_at = datetime.now()
        self.started_at = datetime.now()
        self.finished_at: datetime | None = None
        self.status = "running"

        artifacts_root = Path(artifacts_dir)
        artifacts_root.mkdir(parents=True, exist_ok=True)

        timestamp = self.created_at.strftime("%Y%m%d_%H%M%S")
        safe_name = self._make_safe_name(name)
        self.run_id = f"{timestamp}_{safe_name}"

        run_root = artifacts_root / self.run_id
        run_root.mkdir(parents=True, exist_ok=False)

        models_dir = run_root / "models"
        models_dir.mkdir()

        reports_dir = run_root / "reports"
        reports_dir.mkdir()

        self.paths = ExperimentPaths(
            root=run_root,
            models=models_dir,
            reports=reports_dir,
            metadata=run_root / "metadata.json",
            config=run_root / "config.yaml",
            environment=run_root / "environment.json",
            metrics=run_root / "metrics.json",
            fold_metrics=run_root / "fold_metrics.csv",
            oof_predictions=run_root / "oof_predictions.csv",
            test_predictions=run_root / "test_predictions.csv",
            feature_importance=run_root / "feature_importance.csv",
            submission=run_root / "submission.csv",
            index=artifacts_root / "experiment_index.csv",
        )

        self.git_info = self._get_git_info()
        self.data_fingerprints = {
            "train": self._get_file_info(train_path),
            "test": self._get_file_info(test_path),
        }

        self._metrics: dict[str, Any] = {}

        self.save_environment()
        self.save_metadata()
        self._upsert_index()

    def save_config(self, config: dict[str, Any]) -> None:
        with self.paths.config.open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                config,
                file,
                sort_keys=False,
                allow_unicode=True,
            )

    def save_metadata(self) -> None:
        metadata = {
            "run_id": self.run_id,
            "name": self.name,
            "model_type": self.model_type,
            "description": self.description,
            "notes": self.notes,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat(),
            "finished_at": (
                self.finished_at.isoformat() if self.finished_at is not None else None
            ),
            "duration_seconds": self.duration_seconds,
            "seed": self.seed,
            "n_folds": self.n_folds,
            "git": self.git_info,
            "data": self.data_fingerprints,
            "paths": {key: str(value) for key, value in asdict(self.paths).items()},
        }

        self._write_json(self.paths.metadata, metadata)

    def save_environment(self) -> None:
        environment = {
            "python": {
                "version": sys.version,
                "executable": sys.executable,
                "implementation": platform.python_implementation(),
            },
            "system": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "packages": {
                package: self._get_package_version(package)
                for package in self.TRACKED_PACKAGES
            },
        }

        self._write_json(self.paths.environment, environment)

    def save_metrics(self, metrics: dict[str, Any]) -> None:
        self._metrics.update(metrics)
        self._write_json(self.paths.metrics, self._metrics)
        self._upsert_index()

    def save_fold_metrics(self, metrics: pd.DataFrame) -> None:
        metrics.to_csv(self.paths.fold_metrics, index=False)

    def save_model(
        self,
        model: Any,
        *,
        filename: str = "final_model.joblib",
    ) -> Path:
        model_path = self.paths.models / filename
        joblib.dump(model, model_path)
        return model_path

    def save_oof_predictions(self, predictions: pd.DataFrame) -> None:
        predictions.to_csv(self.paths.oof_predictions, index=False)

    def save_test_predictions(self, predictions: pd.DataFrame) -> None:
        predictions.to_csv(self.paths.test_predictions, index=False)

    def save_submission(self, submission: pd.DataFrame) -> None:
        submission.to_csv(self.paths.submission, index=False)

    def save_feature_importance(
        self,
        feature_importance: pd.DataFrame,
    ) -> None:
        feature_importance.to_csv(
            self.paths.feature_importance,
            index=False,
        )

    def save_dataframe(
        self,
        dataframe: pd.DataFrame,
        filename: str,
        *,
        index: bool = False,
    ) -> Path:
        path = self.paths.reports / filename

        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")

        dataframe.to_csv(path, index=index)
        return path

    def save_json(
        self,
        data: dict[str, Any],
        filename: str,
    ) -> Path:
        path = self.paths.reports / filename

        if path.suffix.lower() != ".json":
            path = path.with_suffix(".json")

        self._write_json(path, data)
        return path

    def complete(
        self,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if metrics:
            self._metrics.update(metrics)
            self._write_json(self.paths.metrics, self._metrics)

        self.status = "completed"
        self.finished_at = datetime.now()

        self.save_metadata()
        self._upsert_index()

    def fail(self, error: Exception | str) -> None:
        self.status = "failed"
        self.finished_at = datetime.now()

        self._metrics["error"] = str(error)
        self._write_json(self.paths.metrics, self._metrics)

        self.save_metadata()
        self._upsert_index()

    @property
    def duration_seconds(self) -> float | None:
        if self.finished_at is None:
            return None

        return round(
            (self.finished_at - self.started_at).total_seconds(),
            3,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "model_type": self.model_type,
            "status": self.status,
            "artifact_path": str(self.paths.root),
        }

    def _upsert_index(self) -> None:
        row = self._build_index_row()
        rows: list[dict[str, Any]] = []

        if self.paths.index.exists():
            with self.paths.index.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as file:
                reader = csv.DictReader(file)
                rows = [
                    existing_row
                    for existing_row in reader
                    if existing_row.get("run_id") != self.run_id
                ]

        rows.append(row)

        with self.paths.index.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.INDEX_COLUMNS,
            )
            writer.writeheader()
            writer.writerows(rows)

    def _build_index_row(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "model_type": self.model_type,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "cv_score": self._metrics.get("cv_score", ""),
            "cv_std": self._metrics.get("cv_std", ""),
            "oof_score": self._metrics.get("oof_score", ""),
            "threshold": self._metrics.get("threshold", ""),
            "public_score": self._metrics.get("public_score", ""),
            "duration_seconds": self.duration_seconds or "",
            "seed": self.seed if self.seed is not None else "",
            "n_folds": (self.n_folds if self.n_folds is not None else ""),
            "git_commit": self.git_info.get("commit", ""),
            "git_dirty": self.git_info.get("dirty", ""),
            "notes": self.notes,
            "artifact_path": str(self.paths.root),
        }

    @staticmethod
    def _make_safe_name(value: str) -> str:
        normalized = value.strip().lower()
        safe = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in normalized
        )
        return safe.strip("_") or "experiment"

    @staticmethod
    def _get_package_version(package: str) -> str | None:
        try:
            return version(package)
        except PackageNotFoundError:
            return None

    @staticmethod
    def _get_git_info() -> dict[str, Any]:
        def run_git(*args: str) -> str:
            result = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()

        try:
            status = run_git("status", "--porcelain")

            return {
                "branch": run_git(
                    "rev-parse",
                    "--abbrev-ref",
                    "HEAD",
                ),
                "commit": run_git("rev-parse", "HEAD"),
                "dirty": bool(status),
            }
        except (subprocess.CalledProcessError, FileNotFoundError):
            return {
                "branch": None,
                "commit": None,
                "dirty": None,
            }

    @classmethod
    def _get_file_info(
        cls,
        path: str | Path | None,
    ) -> dict[str, Any] | None:
        if path is None:
            return None

        file_path = Path(path)

        if not file_path.exists():
            return {
                "path": str(file_path),
                "exists": False,
            }

        return {
            "path": str(file_path),
            "exists": True,
            "size_bytes": file_path.stat().st_size,
            "sha256": cls._calculate_sha256(file_path),
        }

    @staticmethod
    def _calculate_sha256(
        path: Path,
        chunk_size: int = 1024 * 1024,
    ) -> str:
        digest = hashlib.sha256()

        with path.open("rb") as file:
            while chunk := file.read(chunk_size):
                digest.update(chunk)

        return digest.hexdigest()

    @classmethod
    def _write_json(
        cls,
        path: Path,
        data: dict[str, Any],
    ) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
                default=cls._json_serializer,
            )

    @staticmethod
    def _json_serializer(value: Any) -> Any:
        if hasattr(value, "item"):
            return value.item()

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, datetime):
            return value.isoformat()

        raise TypeError(
            f"Object of type {type(value).__name__} is not JSON serializable"
        )
