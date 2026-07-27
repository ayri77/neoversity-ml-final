from __future__ import annotations

import hashlib
import socket
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

import numpy as np
import pandas as pd

from src.churn_ml.deployment_v1_auth import (
    SyntheticFixture,
    load_synthetic_fixture,
)
from src.churn_ml.deployment_v1_paths import (
    prewalk_regular_tree,
    validate_new_path,
    validate_regular_file,
)
from src.churn_ml.deployment_v1_artifacts import (
    DeploymentArtifactError,
    DeploymentArtifactStore,
    build_submission,
    environment_record,
    source_provenance,
)
from src.churn_ml.deployment_v1_contracts import (
    ValidatedDeployment,
    validate_deployment,
)
from src.churn_ml.deployment_v1_features import (
    EncodingResult,
    build_full_data_encoding,
    load_deployment_data,
)
from src.churn_ml.deployment_v1_models import (
    EstimatorFactory,
    build_fixed_blend,
    classify_fixed,
    fit_component_bags,
    probability_sha256,
)
from src.churn_ml.research_data import canonical_sha256


@dataclass(frozen=True)
class DeploymentExecution:
    root: Path
    deployment_identity_sha256: str
    positive_count: int
    positive_rate: float


def execute_deployment(
    validated: ValidatedDeployment,
    *,
    mode: str,
    output_dir: Path | None = None,
    fixture_dir: Path | None = None,
    estimator_factory: EstimatorFactory | None = None,
) -> DeploymentExecution:
    if mode not in {"dry-run", "run"}:
        raise ValueError("Deployment execution mode must be dry-run or run.")
    fixture: SyntheticFixture | None = None
    if mode == "dry-run":
        if fixture_dir is None or output_dir is None:
            raise ValueError("dry-run requires fixture_dir and output_dir.")
        root = output_dir.absolute()
        fixture = _assert_dry_run_paths(validated, fixture_dir, root)
    else:
        if fixture_dir is not None or output_dir is not None:
            raise ValueError("run cannot accept dry-run path overrides.")
        root = (
            validated.config.project_root
            / Path(validated.config.payload["output"]["root"])
            / validated.config.deployment_id
        ).absolute()
    _assert_inputs_outputs_disjoint(
        validated, root, fixture.root if fixture is not None else None
    )
    before = _input_authentication(validated, fixture)
    store: DeploymentArtifactStore | None = None
    started = perf_counter()
    started_at = datetime.now(timezone.utc)
    try:
        store = DeploymentArtifactStore(root)
        # Test/sample loading remains behind complete approval validation.
        data = load_deployment_data(validated, fixture=fixture)
        encodings: dict[str, EncodingResult] = {}
        encoding_identities: dict[str, Any] = {}
        common_assignments: pd.DataFrame | None = None
        predictions = []
        bag_records: list[dict[str, Any]] = []
        for component, approval in zip(
            validated.config.payload["components"],
            validated.approvals,
            strict=True,
        ):
            contract = approval.research_run.config.adapter_contract
            encoder_contract = (
                contract["target_encoder"]
                if approval.adapter_id == "manual_lightgbm_te_v1_compat"
                else contract["numeric_features"]
            )
            encoding = build_full_data_encoding(
                data.X_train,
                data.y_train,
                data.X_test,
                encoder_contract,
            )
            encodings[approval.component_id] = encoding
            encoding_identities[approval.component_id] = encoding.identity
            if common_assignments is None:
                common_assignments = encoding.assignments
            else:
                try:
                    pd.testing.assert_frame_equal(
                        common_assignments,
                        encoding.assignments,
                        check_exact=True,
                    )
                except AssertionError as error:
                    raise ValueError(
                        "Component OOF encoding assignments differ."
                    ) from error
            result = fit_component_bags(
                component,
                approval,
                encoding,
                data.y_train,
                row_keys=data.test_row_keys,
                estimator_factory=estimator_factory,
            )
            predictions.append(result)
            bag_records.extend(result.bag_records)
        assert common_assignments is not None

        weights = {
            str(item["component_id"]): float(item["component_weight"])
            for item in validated.config.payload["components"]
        }
        blend = build_fixed_blend(predictions, weights)
        threshold = float(validated.config.payload["threshold"]["value"])
        labels = classify_fixed(blend, threshold)
        sample_config = validated.config.payload["sample_submission"]
        submission = build_submission(
            data.sample_submission,
            labels,
            id_column=str(sample_config["id_column"]),
            target_column=str(sample_config["target_column"]),
        )
        store.write_yaml(
            "resolved_deployment_config.yaml",
            validated.config.resolved_payload(),
        )
        store.write_json(
            "deployment_identity.json",
            {
                "sha256": validated.identity_sha256,
                "canonical": validated.identity,
            },
        )
        for approval in validated.approvals:
            store.write_yaml(
                f"approval_snapshot/{approval.component_id}.yaml",
                approval.payload,
            )
        store.write_json("dataset_identity.json", data.dataset_identity)
        store.write_json("fixture_identity.json", data.fixture_identity)
        store.write_json("train_schema.json", data.train_schema)
        store.write_json("test_schema.json", data.test_schema)
        pipeline_identity = validated.approvals[0].research_run.evaluation_plan_identity
        store.write_json(
            "feature_identity.json",
            {
                "schema_version": 1,
                "pipeline_id": validated.config.payload["pipeline_id"],
                "pipeline_sha256": validated.approvals[0].payload["research_run"][
                    "pipeline_sha256"
                ],
                "transformed_columns": data.X_train.columns.tolist(),
                "transformed_columns_sha256": canonical_sha256(
                    data.X_train.columns.tolist()
                ),
                "evaluation_plan_sha256": pipeline_identity["sha256"],
            },
        )
        store.write_json(
            "encoding_identity.json",
            {"schema_version": 1, "components": encoding_identities},
        )
        store.write_parquet("encoding_assignments.parquet", common_assignments)
        store.write_json(
            "component_summary.json",
            {
                "schema_version": 1,
                "components": [item.summary for item in predictions],
            },
        )
        store.write_csv("bag_summary.csv", pd.DataFrame.from_records(bag_records))
        component_frame = pd.DataFrame(
            {
                "row_position": np.arange(len(blend), dtype=np.int64),
                "row_id": list(data.test_row_keys),
            }
        )
        for item in predictions:
            component_frame[item.component_id] = item.probabilities
        store.write_parquet("component_probabilities.parquet", component_frame)
        bag_frame = pd.DataFrame(
            {
                "row_position": np.arange(len(blend), dtype=np.int64),
                "row_id": list(data.test_row_keys),
            }
        )
        for item in predictions:
            for bag in item.bags:
                bag_frame[bag.column] = bag.probabilities
        store.write_parquet("bag_probabilities.parquet", bag_frame)
        blend_frame = pd.DataFrame(
            {
                "row_position": np.arange(len(blend), dtype=np.int64),
                "row_id": list(data.test_row_keys),
                "probability": blend,
                "prediction": labels,
            }
        )
        store.write_parquet("blend_probabilities.parquet", blend_frame)
        store.write_csv("submission.csv", submission)
        submission_raw = (root / "submission.csv").read_bytes()
        prediction_summary = {
            "schema_version": 1,
            "row_count": len(blend),
            "blend_probability_sha256": probability_sha256(blend),
            "threshold": threshold,
            "comparison": "greater_than_or_equal",
            "positive_count": int(labels.sum()),
            "positive_rate": float(labels.mean()),
            "submission_sha256": hashlib.sha256(submission_raw).hexdigest(),
        }
        store.write_json("prediction_summary.json", prediction_summary)
        finished_at = datetime.now(timezone.utc)
        store.write_json(
            "runtime.json",
            {
                "schema_version": 1,
                "status": "completed",
                "mode": mode,
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": finished_at.isoformat(),
                "duration_seconds": perf_counter() - started,
                "competition_test_access_authorized": mode == "run",
                "network_access": False,
                "tracking_enabled": False,
            },
        )
        store.write_json("environment.json", environment_record())
        store.write_json(
            "source_provenance.json",
            source_provenance(validated.config.project_root),
        )
        if _input_authentication(validated, fixture) != before:
            raise DeploymentArtifactError(
                "Approval or research inputs changed during deployment."
            )
        store.complete(validated, data=data)
        return DeploymentExecution(
            root=root,
            deployment_identity_sha256=validated.identity_sha256,
            positive_count=int(labels.sum()),
            positive_rate=float(labels.mean()),
        )
    except BaseException as error:
        if store is not None and not (store.root / "_SUCCESS").exists():
            store.fail(error)
        raise


def validate_only(config_path: Path, *, project_root: Path) -> ValidatedDeployment:
    """Validate approvals/research only; never allocate or read P4 test paths."""
    return validate_deployment(config_path, project_root=project_root)


@contextmanager
def network_disabled() -> Iterator[None]:
    original_socket = socket.socket
    original_connection = socket.create_connection

    def deny(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("Network access is prohibited during P4 deployment.")

    socket.socket = deny  # type: ignore[misc,assignment]
    socket.create_connection = deny  # type: ignore[misc,assignment]
    try:
        yield
    finally:
        socket.socket = original_socket  # type: ignore[misc,assignment]
        socket.create_connection = original_connection


def _assert_dry_run_paths(
    validated: ValidatedDeployment,
    fixture_dir: Path,
    output_dir: Path,
) -> SyntheticFixture:
    forbidden_hashes = {
        str(validated.config.payload["test_data"]["sha256"]),
        str(validated.config.payload["sample_submission"]["sha256"]),
    }
    fixture = load_synthetic_fixture(
        fixture_dir,
        project_root=validated.config.project_root,
        forbidden_hashes=forbidden_hashes,
    )
    configured = {
        (
            validated.config.project_root
            / validated.config.payload["test_data"]["path"]
        ).resolve(strict=False),
        (
            validated.config.project_root
            / validated.config.payload["sample_submission"]["path"]
        ).resolve(strict=False),
    }
    if any(path == fixture.root or fixture.root in path.parents for path in configured):
        raise ValueError("Dry-run fixtures overlap configured competition paths.")
    if output_dir.exists():
        raise FileExistsError("Dry-run output directory already exists.")
    return fixture


def _assert_inputs_outputs_disjoint(
    validated: ValidatedDeployment,
    output: Path,
    fixture_dir: Path | None,
) -> None:
    output = validate_new_path(output)
    inputs = [
        validated.config.source_path,
        *[item.source_path for item in validated.approvals],
        *[item.research_run.root for item in validated.approvals],
        (
            validated.config.project_root
            / validated.config.payload["test_data"]["path"]
        ).resolve(),
        (
            validated.config.project_root
            / validated.config.payload["sample_submission"]["path"]
        ).resolve(),
    ]
    if fixture_dir is not None:
        inputs.append(fixture_dir.resolve())
    for item in inputs:
        resolved = item.resolve()
        if (
            output == resolved
            or output in resolved.parents
            or resolved in output.parents
        ):
            raise ValueError("Deployment output overlaps an immutable input.")
    if output.exists():
        raise FileExistsError("Deployment output already exists.")


def _input_authentication(
    validated: ValidatedDeployment,
    fixture: SyntheticFixture | None,
) -> dict[str, str]:
    records: dict[str, str] = {
        validated.config.source_path.relative_to(
            validated.config.project_root
        ).as_posix(): _tree_sha256(validated.config.source_path)
    }
    for approval in validated.approvals:
        records[
            approval.source_path.relative_to(validated.config.project_root).as_posix()
        ] = _tree_sha256(approval.source_path)
        records[
            approval.research_run.root.relative_to(
                validated.config.project_root
            ).as_posix()
        ] = _tree_sha256(approval.research_run.root)
    if fixture is not None:
        records[fixture.root.relative_to(validated.config.project_root).as_posix()] = (
            _tree_sha256(fixture.root)
        )
    return records


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        safe = validate_regular_file(path, reject_hardlinks=True)
        digest.update(safe.read_bytes())
        return digest.hexdigest()
    tree = prewalk_regular_tree(path, reject_hardlinks=True)
    for relative in sorted(tree.files, key=lambda item: item.as_posix()):
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update((tree.root / relative).read_bytes())
    return digest.hexdigest()
