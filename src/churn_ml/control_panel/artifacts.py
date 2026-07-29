from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import pandas as pd

from src.churn_ml.control_panel.command_builder import (
    CommandBuildError,
    resolve_safe_path,
)
from src.churn_ml.control_panel.path_safety import (
    PathSafetyError,
    path_exists_nonfollowing,
    require_regular_file,
    require_safe_directory,
    require_safe_existing_ancestors,
)
from src.churn_ml.control_panel.presentation import (
    format_date,
    format_time,
    humanize_experiment_token,
    parse_config_metadata,
)
from src.churn_ml.control_panel.schemas import ReaderSpec


class ArtifactReadError(ValueError):
    """Raised when a configured artifact cannot be read safely."""


@dataclass(frozen=True)
class ArtifactRecord:
    reader_id: str
    root: Path
    relative_path: str
    state: str
    summaries: Mapping[str, Any]
    json_payloads: Mapping[str, Any]
    diagnostic: str | None


def configured_artifact_file(
    repository_root: Path,
    artifact: ArtifactRecord,
    relative_path: str,
) -> Path:
    try:
        _, path = resolve_safe_path(
            repository_root,
            f"{artifact.relative_path}/{relative_path}",
            allowed_roots=(artifact.relative_path,),
            must_exist=True,
        )
    except CommandBuildError as error:
        raise ArtifactReadError(str(error)) from error
    if not path.is_file():
        raise ArtifactReadError(f"Configured artifact file is not regular: {path}.")
    try:
        require_regular_file(path, reject_hardlinks=True)
    except PathSafetyError as error:
        raise ArtifactReadError(str(error)) from error
    return path


def safe_json_load(path: Path, *, max_bytes: int = 2_000_000) -> Any:
    _require_regular_file(path)
    if path.stat().st_size > max_bytes:
        raise ArtifactReadError(f"JSON file exceeds {max_bytes} bytes: {path}.")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactReadError(f"Could not read JSON file {path}: {error}") from error


def extract_dot_path(payload: Any, path: str) -> Any:
    current = payload
    for part in path.split("."):
        if not part:
            raise ArtifactReadError("Dot paths must not contain empty segments.")
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return None
            current = current[index]
        else:
            return None
    return current


def csv_preview(
    path: Path,
    *,
    row_limit: int = 100,
    column_limit: int = 40,
    max_bytes: int = 5_000_000,
) -> pd.DataFrame:
    if row_limit < 1 or column_limit < 1:
        raise ArtifactReadError("CSV preview limits must be positive.")
    _require_regular_file(path)
    if path.stat().st_size > max_bytes:
        raise ArtifactReadError(f"CSV file exceeds {max_bytes} bytes: {path}.")
    try:
        frame = pd.read_csv(path, nrows=row_limit)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
        raise ArtifactReadError(
            f"Could not preview CSV file {path}: {error}"
        ) from error
    return frame.iloc[:, :column_limit]


def text_tail(
    path: Path,
    *,
    lines: int,
    max_bytes: int = 2_000_000,
) -> str:
    if lines < 1:
        raise ArtifactReadError("Tail line count must be positive.")
    _require_regular_file(path)
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(max(0, size - max_bytes))
        content = handle.read(max_bytes)
    return "\n".join(content.decode("utf-8", errors="replace").splitlines()[-lines:])


def discover_artifacts(
    repository_root: Path,
    reader: ReaderSpec,
) -> list[ArtifactRecord]:
    root = repository_root.resolve(strict=True)
    results: list[ArtifactRecord] = []
    seen: set[tuple[str, str]] = set()
    for configured_root in reader.artifact_roots:
        try:
            _, artifact_root = resolve_safe_path(
                root,
                configured_root,
                allowed_roots=(configured_root,),
                must_exist=True,
            )
        except CommandBuildError:
            continue
        for candidate in artifact_root.glob(reader.discovery_glob):
            try:
                require_safe_directory(candidate)
                canonical = candidate.resolve(strict=True)
                canonical.relative_to(artifact_root)
                record = read_artifact(root, reader, canonical)
            except (OSError, ValueError, ArtifactReadError, PathSafetyError):
                continue
            identity = (record.reader_id, record.relative_path)
            if identity in seen:
                continue
            seen.add(identity)
            results.append(record)
    return sorted(results, key=lambda item: item.relative_path, reverse=True)


def read_artifact(
    repository_root: Path,
    reader: ReaderSpec,
    artifact_root: Path,
) -> ArtifactRecord:
    root = repository_root.resolve(strict=True)
    canonical = artifact_root.resolve(strict=True)
    require_safe_directory(canonical)
    allowed_roots = tuple(reader.artifact_roots)
    try:
        relative, validated = resolve_safe_path(
            root,
            canonical.relative_to(root),
            allowed_roots=allowed_roots,
            must_exist=True,
        )
    except (CommandBuildError, ValueError) as error:
        raise ArtifactReadError(str(error)) from error
    state, diagnostic = _marker_state(validated, reader)
    if state == "invalid":
        return ArtifactRecord(reader.id, validated, relative, state, {}, {}, diagnostic)
    summaries: dict[str, Any] = {}
    payloads: dict[str, Any] = {}
    for summary in reader.summary_files:
        try:
            _, path = resolve_safe_path(
                root,
                f"{relative}/{summary.path}",
                allowed_roots=(relative,),
                must_exist=True,
            )
        except CommandBuildError:
            continue
        if not path.is_file():
            continue
        payload = safe_json_load(path)
        payloads[summary.path] = payload
        for label, dot_path in summary.fields.items():
            summaries[label] = extract_dot_path(payload, dot_path)
    return ArtifactRecord(
        reader_id=reader.id,
        root=validated,
        relative_path=relative,
        state=state,
        summaries=summaries,
        json_payloads=payloads,
        diagnostic=diagnostic,
    )


def comparison_rows(
    left: ArtifactRecord,
    right: ArtifactRecord,
    compare_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in compare_fields:
        left_value = left.summaries.get(field)
        right_value = right.summaries.get(field)
        rows.append(
            {
                "Metric": field,
                "Left": left_value,
                "Right": right_value,
                "Delta (Right - Left)": _numeric_delta(left_value, right_value),
                # Backward-compatible aliases used by older tests/callers.
                "field": field,
                "left": left_value,
                "right": right_value,
                "delta": _numeric_delta(left_value, right_value),
            }
        )
    return rows


def _numeric_delta(left: Any, right: Any) -> float | None:
    try:
        if left is None or right is None:
            return None
        return float(right) - float(left)
    except (TypeError, ValueError):
        return None


def artifact_selector_options(
    repository_root: Path,
    reader: ReaderSpec,
    *,
    roots: tuple[str, ...],
    statuses: tuple[str, ...],
) -> list[tuple[str, str]]:
    """Return unique (relative_path, label) pairs for declarative path selectors."""
    allowed_states = set(statuses)
    options: list[tuple[str, str]] = []
    seen: set[str] = set()
    for artifact in discover_artifacts(repository_root, reader):
        if allowed_states and artifact.state not in allowed_states:
            continue
        if roots and not _relative_within_roots(artifact.relative_path, roots):
            continue
        if artifact.relative_path in seen:
            continue
        seen.add(artifact.relative_path)
        options.append((artifact.relative_path, artifact_option_label(artifact)))
    return options


def artifact_option_label(artifact: ArtifactRecord) -> str:
    name = (
        artifact.summaries.get("Study name")
        or artifact.summaries.get("Search ID")
        or Path(artifact.relative_path).name
    )
    parts = [str(name)]
    objective = artifact.summaries.get("Best objective")
    if objective is None:
        objective = artifact.summaries.get("Best trial objective")
    if objective is not None:
        parts.append(f"best={objective}")
    parts.append(artifact.state)
    return " · ".join(parts)


def _summary_or_na(summaries: Mapping[str, Any], key: str) -> Any:
    value = summaries.get(key)
    return value if value is not None else "Not available"


def build_experiment_table_rows(
    artifacts: list[ArtifactRecord],
    *,
    repo_root: Path,
) -> list[dict[str, Any]]:
    """Build human-readable experiment catalog rows (newest first)."""
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        try:
            meta = parse_config_metadata(artifact.relative_path, repo_root)
            created = meta.get("created_at_utc")
            experiment = humanize_experiment_token(
                meta.get("experiment_name") or meta.get("adapter_id") or "",
                meta.get("model_family", ""),
            )
            if not experiment:
                experiment = Path(artifact.relative_path).name
            model = meta.get("model_family") or "Not available"
            mode = meta.get("mode") or "Not available"
            created_date = format_date(created) if created else "Not available"
            created_time = format_time(created) if created else "Not available"
            rows.append(
                {
                    "Model": model,
                    "Mode": mode,
                    "Experiment": experiment,
                    "Created date": created_date,
                    "Created time": created_time,
                    "Balanced Accuracy": _summary_or_na(
                        artifact.summaries, "Balanced Accuracy mean"
                    ),
                    "Sensitivity": _summary_or_na(artifact.summaries, "Sensitivity"),
                    "Specificity": _summary_or_na(artifact.summaries, "Specificity"),
                    "ROC AUC": _summary_or_na(artifact.summaries, "ROC AUC"),
                    "Average Precision": _summary_or_na(
                        artifact.summaries, "Average Precision"
                    ),
                    "Brier score": _summary_or_na(artifact.summaries, "Brier score"),
                    "Threshold median": _summary_or_na(
                        artifact.summaries, "Threshold median"
                    ),
                    "Status": artifact.state,
                    "Artifact path": artifact.relative_path,
                    "_created_sort": created or "",
                    # Unique per artifact so Plotly never sums duplicate labels.
                    "_chart_key": artifact.relative_path,
                    "_chart_label": _chart_tick_label(
                        model=model,
                        experiment=experiment,
                        created_date=created_date,
                        created_time=created_time,
                    ),
                    "_ba": artifact.summaries.get("Balanced Accuracy mean"),
                    "_sensitivity": artifact.summaries.get("Sensitivity"),
                    "_specificity": artifact.summaries.get("Specificity"),
                    "_roc_auc": artifact.summaries.get("ROC AUC"),
                    "_average_precision": artifact.summaries.get("Average Precision"),
                }
            )
        except Exception:
            fallback_name = Path(artifact.relative_path).name
            rows.append(
                {
                    "Model": "Not available",
                    "Mode": "Not available",
                    "Experiment": fallback_name,
                    "Created date": "Not available",
                    "Created time": "Not available",
                    "Balanced Accuracy": "Not available",
                    "Sensitivity": "Not available",
                    "Specificity": "Not available",
                    "ROC AUC": "Not available",
                    "Average Precision": "Not available",
                    "Brier score": "Not available",
                    "Threshold median": "Not available",
                    "Status": artifact.state,
                    "Artifact path": artifact.relative_path,
                    "_created_sort": "",
                    "_chart_key": artifact.relative_path,
                    "_chart_label": fallback_name,
                    "_ba": None,
                    "_sensitivity": None,
                    "_specificity": None,
                    "_roc_auc": None,
                    "_average_precision": None,
                }
            )
    rows.sort(key=lambda item: str(item.get("_created_sort") or ""), reverse=True)
    return rows


def _chart_tick_label(
    *,
    model: str,
    experiment: str,
    created_date: str,
    created_time: str,
) -> str:
    """Readable bar tick/hover label: model, experiment, date, and time."""
    parts = [part for part in (model, experiment) if part and part != "Not available"]
    if created_date and created_date != "Not available":
        parts.append(created_date)
    if created_time and created_time != "Not available":
        parts.append(created_time)
    return " · ".join(parts) if parts else experiment


def display_compatibility_summary(
    left: ArtifactRecord,
    right: ArtifactRecord,
) -> dict[str, Any]:
    """UI-local compatibility display for research artifacts.

    This does not replace the official Paired Comparison gate; it only surfaces
    fingerprint equality checks already present on disk.
    """
    left_plan = _identity_hash(left.root, "identities/evaluation_plan.json")
    right_plan = _identity_hash(right.root, "identities/evaluation_plan.json")
    left_dataset = _dataset_fingerprint_token(left.root)
    right_dataset = _dataset_fingerprint_token(right.root)
    left_folds = _fold_assignment_token(left.root)
    right_folds = _fold_assignment_token(right.root)

    same_plan = (
        left_plan is not None and right_plan is not None and left_plan == right_plan
    )
    same_dataset = (
        left_dataset is not None
        and right_dataset is not None
        and left_dataset == right_dataset
    )
    same_folds = (
        left_folds is not None and right_folds is not None and left_folds == right_folds
    )
    compatible = bool(same_plan and same_dataset and same_folds)
    return {
        "same_evaluation_plan": same_plan,
        "same_dataset_fingerprint": same_dataset,
        "same_fold_assignments": same_folds,
        "compatible": compatible,
        "status": "compatible" if compatible else "incompatible",
    }


COMPARISON_ID_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def default_comparison_id(
    left: ArtifactRecord,
    right: ArtifactRecord,
    *,
    repo_root: Path,
) -> str:
    left_meta = parse_config_metadata(left.relative_path, repo_root)
    right_meta = parse_config_metadata(right.relative_path, repo_root)
    left_model = _slug(left_meta.get("model_family") or "left")
    right_model = _slug(right_meta.get("model_family") or "right")
    mode = _slug(left_meta.get("mode") or right_meta.get("mode") or "development")
    plan = left_meta.get("plan_id") or right_meta.get("plan_id") or "compare"
    plan_slug = _slug(plan)
    # Prefer trailing plan tokens such as r2x5-t3-v1 when present.
    tokens = plan_slug.split("-")
    short_plan = plan_slug
    for index, token in enumerate(tokens):
        if re.fullmatch(r"r\d+x\d+", token):
            short_plan = "-".join(tokens[index:])
            break
    return sanitize_comparison_id(f"{left_model}-vs-{right_model}-{mode}-{short_plan}")


def sanitize_comparison_id(value: str) -> str:
    """Normalize a comparison ID to the existing paired-comparison safe-slug rules."""
    text = _slug(value)
    if COMPARISON_ID_SAFE.fullmatch(text) is None:
        text = re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-_") or "compare"
        if text[0] in "-_":
            text = f"c{text}"
        text = text[:128]
    if COMPARISON_ID_SAFE.fullmatch(text) is None:
        return "compare"
    return text


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower()).strip("-")
    return text or "item"


def _identity_hash(root: Path, relative: str) -> str | None:
    path = root.joinpath(*relative.split("/"))
    if not path.is_file():
        return None
    try:
        payload = safe_json_load(path)
    except ArtifactReadError:
        return None
    if isinstance(payload, dict):
        sha = payload.get("sha256")
        if isinstance(sha, str) and sha:
            return sha
        hashes = payload.get("hashes")
        if isinstance(hashes, dict):
            plan = hashes.get("plan")
            if isinstance(plan, str) and plan:
                return plan
    return _file_sha256(path)


def _dataset_fingerprint_token(root: Path) -> str | None:
    path = root / "dataset_fingerprints.json"
    if not path.is_file():
        # Fall back to evaluation-plan embedded dataset identity.
        return _identity_hash(root, "identities/evaluation_plan.json")
    try:
        payload = safe_json_load(path)
    except ArtifactReadError:
        return None
    if not isinstance(payload, dict):
        return None
    version = payload.get("dataset_version")
    files = payload.get("files")
    train_sha = None
    if isinstance(files, dict):
        train = files.get("train_features")
        if isinstance(train, dict):
            train_sha = train.get("sha256")
    row = payload.get("row_position_identity")
    row_sha = row.get("sha256") if isinstance(row, dict) else None
    parts = [str(version or ""), str(train_sha or ""), str(row_sha or "")]
    token = "|".join(parts)
    return token if any(parts) else _file_sha256(path)


def _fold_assignment_token(root: Path) -> str | None:
    splits = root / "splits"
    names = (
        "outer_assignments.parquet",
        "threshold_selection_assignments.parquet",
    )
    digests: list[str] = []
    for name in names:
        path = splits / name
        if not path.is_file():
            return None
        digest = _file_sha256(path)
        if digest is None:
            return None
        digests.append(digest)
    return "|".join(digests)


def _file_sha256(path: Path) -> str | None:
    try:
        require_regular_file(path, reject_hardlinks=True)
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except (OSError, PathSafetyError):
        return None


def _relative_within_roots(relative_path: str, roots: tuple[str, ...]) -> bool:
    path = PurePosixPath(relative_path)
    for root in roots:
        try:
            path.relative_to(PurePosixPath(root))
            return True
        except ValueError:
            continue
    return False


def _marker_state(root: Path, reader: ReaderSpec) -> tuple[str, str | None]:
    try:
        success = any(
            _safe_marker_exists(root, marker) for marker in reader.success_markers
        )
        failure = any(
            _safe_marker_exists(root, marker) for marker in reader.failure_markers
        )
    except (PathSafetyError, OSError):
        return "invalid", "A configured marker path is unsafe."
    if success and failure:
        return "invalid", "Conflicting success and failure markers are present."
    if success:
        return "completed", None
    if failure:
        return "failed", None
    return "running", None


def _safe_marker_exists(root: Path, relative: str) -> bool:
    path = root.joinpath(*relative.split("/"))
    require_safe_existing_ancestors(path)
    if not path_exists_nonfollowing(path):
        return False
    require_regular_file(path, reject_hardlinks=True)
    return True


def _require_regular_file(path: Path) -> None:
    try:
        require_regular_file(path, reject_hardlinks=True)
    except PathSafetyError as error:
        raise ArtifactReadError(str(error)) from error
