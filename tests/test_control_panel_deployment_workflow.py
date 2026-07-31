"""Focused tests for the canonical run to deployment workflow.

These tests build synthetic Research v2 run directories and Dataset Package
manifests on disk. No model is fitted, no competition asset is read, and no
Streamlit session is started.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.churn_ml.control_panel.config_schema_guard import (
    DEPLOYMENT_KIND,
    EXPERIMENT_CORE_KIND,
    EXPERIMENT_CORE_MESSAGE,
    classify_config_document,
    guard_config_for_command,
)
from src.churn_ml.control_panel.deployment_candidates import (
    SUBMISSION_ACTION_ID,
    SUBMISSION_COMMAND_ID,
    SUBMISSION_HANDOFF_KEY,
    UNRESOLVED_SAMPLE_SUBMISSION_PATH,
    apply_submission_handoff,
    candidate_durable_key,
    evaluate_deployment_readiness,
    list_deployment_candidates,
)
from src.churn_ml.control_panel.config_editor import ConfigEditError, read_config
from src.churn_ml.control_panel.deployment_draft_builder import (
    APPROVAL_NAME,
    DEFAULT_EXCEPTION_REASON,
    DEFAULT_INTENDED_ROLE,
    DEPLOYMENT_CONFIG_NAME,
    DEPLOYMENT_DRAFT_ROOT,
    DeploymentDraftConflictError,
    DeploymentDraftError,
    prepare_deployment_draft,
)
from src.churn_ml.control_panel.presentation import (
    build_pre_run_summary,
    config_preview_allowed_roots,
    is_read_only_config_path,
    normalize_source_kind,
)
from src.churn_ml.control_panel.registry import load_registry
from src.churn_ml.control_panel.research_inventory import build_research_inventory
from src.churn_ml.control_panel.selection_state import (
    get_durable_value,
    set_durable_value,
    ui_durable_key,
)
from src.churn_ml.control_panel.workflow_navigation import (
    LEGACY_BADGE,
    RESEARCH_V2_RUN_CONTRACT,
    TRAIN_CONFIG_CONTRACT,
    contract_declaration,
    is_legacy_command,
    visible_command_ids,
    workflow_command_ids,
    workflow_label,
)
from src.churn_ml.deployment_v1_contracts import (
    CONFIG_KEYS,
    DEPLOYMENT_SCHEMA_VERSION,
    load_deployment_config,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ID = "manual_lightgbm_te_v1_compat"
PIPELINE_ID = "registered_prepared_passthrough_v1"
DATASET_ID = "v0_raw_minimal"
PLAN_ID = "v0_raw_minimal__development_r2x5_t3_v1"
RUN_ID = "20260731T100000000000Z_abc123"
APPROVER = "focused-test-operator"
APPROVED_AT = "2026-07-31T12:00:00+00:00"

_PLAN_HASH = "1d" * 32
_PIPELINE_HASH = "2f" * 32
_ADAPTER_HASH = "ca" * 32
_CANDIDATE_HASH = "aa" * 32
_MANIFEST_HASH = "bd" * 32
_SOURCE_HASH = "50" * 32
_SCHEMA_HASH = "5c" * 32


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _make_dataset_package(
    root: Path, *, dataset_id: str = DATASET_ID, target_dependency: str = "none"
) -> None:
    _write_json(
        root / "data" / "processed" / dataset_id / "dataset_manifest.json",
        {
            "schema_version": "dataset_package_v1",
            "dataset_id": dataset_id,
            "target_dependency": target_dependency,
            "schema_hash": _SCHEMA_HASH,
            "train_row_count": 4000,
            "test_row_count": 1000,
            "files": {"X_test": "X_test.parquet"},
            "content_hashes": {"X_test": "e7" * 32},
            "features": [
                {"name": "feature_one", "dtype": "float64"},
                {"name": "feature_two", "dtype": "float64"},
            ],
        },
    )


def _make_run(
    root: Path,
    *,
    run_id: str = RUN_ID,
    dataset_id: str = DATASET_ID,
    adapter_id: str = ADAPTER_ID,
    plan_id: str = PLAN_ID,
    mode: str = "development",
    marker: str = "_SUCCESS",
    threshold_median: float | None = 0.13,
    include_manifest: bool = True,
    include_adapter_contract: bool = True,
    include_identities: bool = True,
) -> str:
    """Create one synthetic Research v2 run and return its relative path."""
    plan_dir = f"{plan_id}_{_PLAN_HASH[:12]}"
    run_root = (
        root
        / "artifacts"
        / "research_v2"
        / plan_dir
        / f"{PIPELINE_ID}__{adapter_id}"
        / run_id
    )
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / marker).write_text("", encoding="utf-8")
    (run_root / "predictions").mkdir(exist_ok=True)
    (run_root / "predictions" / "outer_validation.parquet").write_bytes(b"PAR1")

    provenance = {
        "dataset_id": dataset_id,
        "parent_dataset_id": None,
        "hypothesis": "focused test",
        "n_features": 2,
        "schema_hash": _SCHEMA_HASH,
        "train_content_hash": "7c" * 32,
        "target_hash": "7b" * 32,
        "target_dependency": "none",
        "train_row_identity_hash": "bc" * 32,
        "registry_schema_version": "dataset_package_v1",
    }
    _write_json(run_root / "dataset_provenance.json", provenance)
    _write_json(
        run_root / "run_metadata.json",
        {
            "schema_version": 2,
            "experiment_id": f"{dataset_id}__{adapter_id}__{mode}",
            "plan_id": plan_id,
            "feature_pipeline_id": PIPELINE_ID,
            "candidate_adapter_id": adapter_id,
            "hashes": {
                "plan": _PLAN_HASH,
                "feature_pipeline": _PIPELINE_HASH,
                "candidate_adapter": _ADAPTER_HASH,
                "candidate": _CANDIDATE_HASH,
                "source": _SOURCE_HASH,
                "loaded_modules": "1b" * 32,
            },
            "status": "completed" if marker == "_SUCCESS" else "failed",
            "started_at_utc": "2026-07-31T10:00:00.000000+00:00",
            "competition_assets_accessed": False,
            "run_id": run_id,
            "dataset_provenance": provenance,
        },
    )
    if include_manifest:
        _write_json(
            run_root / "artifact_manifest.json",
            {"schema_version": 1, "manifest_sha256": _MANIFEST_HASH, "files": []},
        )
    _write_json(
        run_root / "metrics" / "aggregate.json",
        {
            "primary_metric": "balanced_accuracy",
            "aggregation_unit": "pooled_repeat_predictions",
            "repeat_count": 2,
            "metrics": {
                "balanced_accuracy": {
                    "mean": 0.9,
                    "sample_standard_deviation": 0.0,
                    "minimum": 0.9,
                    "maximum": 0.9,
                }
            },
        },
    )
    threshold_summary: dict[str, object] = {
        "count": 2,
        "interquartile_range": 0.01,
        "per_repeat": [],
    }
    if threshold_median is not None:
        threshold_summary["median"] = threshold_median
    _write_json(run_root / "thresholds" / "threshold_summary.json", threshold_summary)
    if include_identities:
        _write_json(
            run_root / "identities" / "evaluation_plan.json",
            {
                "sha256": _PLAN_HASH,
                "canonical": {"schema_version": 1, "plan_id": plan_id},
            },
        )
        _write_json(
            run_root / "identities" / "feature_pipeline.json",
            {
                "sha256": _PIPELINE_HASH,
                "canonical": {"schema_version": 2, "id": PIPELINE_ID},
            },
        )
        _write_json(
            run_root / "identities" / "candidate_adapter.json",
            {
                "sha256": _ADAPTER_HASH,
                "canonical": {"schema_version": 2, "id": adapter_id},
            },
        )
        _write_json(
            run_root / "identities" / "candidate.json",
            {"sha256": _CANDIDATE_HASH, "canonical": {"schema_version": 2}},
        )
        _write_json(
            run_root / "identities" / "source.json",
            {"sha256": _SOURCE_HASH, "canonical": {"schema_version": 2, "files": []}},
        )
    adapter_section: dict[str, object] = {"id": adapter_id}
    if include_adapter_contract:
        adapter_section["contract"] = {
            "lightgbm": {
                "parameters": {
                    "n_estimators": 400,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                }
            }
        }
    _write_yaml(
        run_root / "resolved_config.yaml",
        {
            "schema_version": 2,
            "experiment": {"id": f"{dataset_id}__{adapter_id}__{mode}"},
            "dataset": {"version": dataset_id},
            "feature_pipeline": {"id": PIPELINE_ID},
            "candidate_adapter": adapter_section,
            "evaluation_plan_path": f"artifacts/ui_configs/plans/{plan_id}.yaml",
            "evaluation_plan": {
                "plan": {"id": plan_id},
                "outer_evaluation": {"repeat_seeds": [0, 17], "n_splits": 5},
                "threshold_policy": {"id": "grid_balanced_accuracy_v1"},
            },
            "artifacts": {"root": "artifacts/research_v2"},
        },
    )
    _write_json(
        run_root / "dataset_fingerprints.json",
        {"dataset_version": dataset_id, "dataset_provenance": provenance},
    )
    return str(run_root.relative_to(root)).replace("\\", "/")


@pytest.fixture()
def canonical_run(tmp_path: Path) -> str:
    _make_dataset_package(tmp_path)
    return _make_run(tmp_path)


def test_completed_canonical_run_passes_deployment_readiness(
    tmp_path: Path, canonical_run: str
) -> None:
    readiness = evaluate_deployment_readiness(tmp_path, canonical_run)
    assert readiness.blocking_reasons == ()
    assert readiness.supported is True
    assert readiness.competition_test_assets_accessed is False
    facts = readiness.facts
    assert facts.run_id == RUN_ID
    assert facts.candidate_id == f"{RUN_ID}-{_MANIFEST_HASH[:12]}"
    assert facts.dataset_version == DATASET_ID
    assert facts.adapter_id == ADAPTER_ID
    assert facts.threshold_value == pytest.approx(0.13)
    assert facts.test_data_path == f"data/processed/{DATASET_ID}/X_test.parquet"
    assert facts.test_expected_rows == 1000
    summary = readiness.summary()
    assert summary["Readiness"] == "supported"
    assert summary["Competition-test assets accessed"] == "false"
    assert summary["Deployment schema to produce"].startswith("deployment_v1")


@pytest.mark.parametrize(
    ("kwargs", "expected_reason_fragment"),
    [
        ({"marker": "_FAILED"}, "Run status is failed"),
        ({"include_manifest": False}, "artifact manifest hash is unavailable"),
        ({"include_identities": False}, "Identity artifact identities/"),
        ({"threshold_median": None}, "Selected threshold evidence"),
        ({"include_adapter_contract": False}, "fixed model parameters are unavailable"),
        ({"adapter_id": "unsupported_adapter_v9"}, "not supported by the deployment"),
    ],
)
def test_incomplete_or_invalid_run_is_blocked(
    tmp_path: Path, kwargs: dict[str, object], expected_reason_fragment: str
) -> None:
    _make_dataset_package(tmp_path)
    run = _make_run(tmp_path, **kwargs)  # type: ignore[arg-type]
    readiness = evaluate_deployment_readiness(tmp_path, run)
    assert readiness.supported is False
    assert any(
        expected_reason_fragment in reason for reason in readiness.blocking_reasons
    ), readiness.blocking_reasons
    with pytest.raises(DeploymentDraftError):
        prepare_deployment_draft(
            tmp_path, run, approver=APPROVER, approved_at_utc=APPROVED_AT
        )


def test_smoke_and_archived_runs_are_blocked(tmp_path: Path) -> None:
    _make_dataset_package(tmp_path)
    smoke = _make_run(
        tmp_path,
        run_id="20260731T100500000000Z_smoke1",
        plan_id="v0_raw_minimal__smoke_r1x2_t2_v1",
        mode="smoke",
    )
    smoke_readiness = evaluate_deployment_readiness(tmp_path, smoke)
    assert smoke_readiness.supported is False
    assert "Smoke runs are not deployment candidates." in (
        smoke_readiness.blocking_reasons
    )

    good = _make_run(tmp_path)
    archived = evaluate_deployment_readiness(
        tmp_path, good, archived_paths=frozenset({good})
    )
    assert archived.supported is False
    assert "Run is archived in the Control Panel workspace." in (
        archived.blocking_reasons
    )


def test_candidate_selector_filters_and_labels_runs(tmp_path: Path) -> None:
    _make_dataset_package(tmp_path)
    _make_dataset_package(
        tmp_path, dataset_id="v9_exploratory", target_dependency="exploratory"
    )
    standard = _make_run(tmp_path)
    _make_run(
        tmp_path,
        run_id="20260731T100600000000Z_smoke2",
        plan_id="v0_raw_minimal__smoke_r1x2_t2_v1",
        mode="smoke",
    )
    failed = _make_run(
        tmp_path, run_id="20260731T100700000000Z_failed", marker="_FAILED"
    )
    rows = build_research_inventory(tmp_path)

    candidates = list_deployment_candidates(rows)
    paths = [item.relative_path for item in candidates]
    assert standard in paths
    assert failed not in paths
    assert not any("smoke" in path for path in paths)

    selected = next(item for item in candidates if item.relative_path == standard)
    for fragment in (DATASET_ID, "LightGBM", "Development", "BA 0.9", "run "):
        assert fragment in selected.label
    assert selected.exploratory is False

    archived_only = list_deployment_candidates(
        rows, archived_paths=frozenset({standard})
    )
    assert standard not in [item.relative_path for item in archived_only]


def test_builder_produces_the_actual_deployment_schema(
    tmp_path: Path, canonical_run: str
) -> None:
    draft = prepare_deployment_draft(
        tmp_path, canonical_run, approver=APPROVER, approved_at_utc=APPROVED_AT
    )
    config = draft.payload.deployment_config
    assert set(config) == CONFIG_KEYS
    assert config["schema_version"] == DEPLOYMENT_SCHEMA_VERSION
    assert config["dataset_version"] == DATASET_ID
    assert config["pipeline_id"] == PIPELINE_ID
    assert config["threshold"]["value"] == pytest.approx(0.13)
    assert config["threshold"]["comparison"] == "greater_than_or_equal"
    assert config["components"][0]["adapter_id"] == ADAPTER_ID
    assert config["components"][0]["fixed_parameters"]["n_estimators"] == 400
    assert config["test_data"]["expected_rows"] == 1000
    assert config["output"]["root"] == "artifacts/deployments"
    assert config["runtime"] == {"tracking_enabled": False, "network_enabled": False}

    provenance = draft.payload.provenance
    assert provenance["source"]["run_path"] == canonical_run
    assert provenance["source"]["run_id"] == RUN_ID
    assert provenance["source"]["run_manifest_sha256"] == _MANIFEST_HASH
    assert provenance["dataset"]["dataset_id"] == DATASET_ID
    assert provenance["model"]["adapter_id"] == ADAPTER_ID
    assert provenance["protocol"]["plan_sha256"] == _PLAN_HASH
    assert provenance["safety"]["competition_test_assets_accessed"] is False
    assert provenance["safety"]["sample_submission_identity_resolved"] is False
    assert provenance["safety"]["real_competition_run_ready"] is False
    assert "submission_row_identity" not in config
    approval = draft.payload.candidate_approval
    assert approval["research_run"]["path"] == canonical_run
    assert approval["manual_approval"]["approver"] == APPROVER
    assert approval["paired_comparison"]["exception"]["granted"] is True


def test_generated_draft_is_accepted_by_the_production_config_loader(
    tmp_path: Path, canonical_run: str
) -> None:
    draft = prepare_deployment_draft(
        tmp_path, canonical_run, approver=APPROVER, approved_at_utc=APPROVED_AT
    )
    loaded = load_deployment_config(
        Path(draft.deployment_config_path), project_root=tmp_path
    )
    assert loaded.payload["deployment_id"] == draft.candidate_id
    assert loaded.payload["dataset_version"] == DATASET_ID
    assert (
        loaded.payload["sample_submission"]["path"] == UNRESOLVED_SAMPLE_SUBMISSION_PATH
    )
    on_disk = yaml.safe_load(
        (tmp_path / draft.deployment_config_path).read_text(encoding="utf-8")
    )
    assert classify_config_document(on_disk) == DEPLOYMENT_KIND


def test_generated_draft_uses_the_dedicated_deployment_root(
    tmp_path: Path, canonical_run: str
) -> None:
    draft = prepare_deployment_draft(
        tmp_path, canonical_run, approver=APPROVER, approved_at_utc=APPROVED_AT
    )
    assert draft.draft_dir == f"{DEPLOYMENT_DRAFT_ROOT}/{draft.candidate_id}"
    for relative in (
        draft.deployment_config_path,
        draft.threshold_evidence_path,
        draft.candidate_approval_path,
        draft.provenance_path,
    ):
        assert relative.startswith(f"{DEPLOYMENT_DRAFT_ROOT}/")
        assert "ui_configs" not in relative
        assert not Path(relative).is_absolute()
        assert (tmp_path / relative).is_file()
    assert not (tmp_path / "artifacts" / "ui_configs").exists()


def test_regeneration_is_idempotent_and_conflicts_are_reported(
    tmp_path: Path, canonical_run: str
) -> None:
    first = prepare_deployment_draft(
        tmp_path, canonical_run, approver=APPROVER, approved_at_utc=APPROVED_AT
    )
    assert first.reused is False
    config_path = tmp_path / first.deployment_config_path
    original_config = config_path.read_bytes()
    original_approval = (tmp_path / first.candidate_approval_path).read_bytes()

    second = prepare_deployment_draft(
        tmp_path, canonical_run, approver=APPROVER, approved_at_utc=APPROVED_AT
    )
    assert second.reused is True
    assert config_path.read_bytes() == original_config

    # Same immutable approval identity, different audit metadata: the recorded
    # historical approval evidence is preserved rather than rewritten.
    third = prepare_deployment_draft(
        tmp_path,
        canonical_run,
        approver="someone-else",
        approved_at_utc="2026-07-31T18:00:00+00:00",
    )
    assert third.reused is True
    assert (tmp_path / first.candidate_approval_path).read_bytes() == original_approval

    # A draft with different deployment content must never be replaced silently.
    config_path.write_bytes(original_config + b"# manually edited\n")
    with pytest.raises(DeploymentDraftConflictError):
        prepare_deployment_draft(
            tmp_path, canonical_run, approver=APPROVER, approved_at_utc=APPROVED_AT
        )
    assert config_path.read_bytes() == original_config + b"# manually edited\n"


def test_research_v2_config_is_rejected_by_the_deployment_guard(
    tmp_path: Path, canonical_run: str
) -> None:
    research_config = "artifacts/ui_configs/experiment_core_v2/candidate.yaml"
    _write_yaml(
        tmp_path / research_config,
        {
            "schema_version": 2,
            "experiment": {"id": "v0_raw_minimal__lightgbm__development"},
            "dataset": {"version": DATASET_ID},
            "feature_pipeline": {"id": PIPELINE_ID},
            "candidate_adapter": {"id": ADAPTER_ID},
            "evaluation_plan_path": "configs/experiment_v2/plan.yaml",
        },
    )
    guard = guard_config_for_command(tmp_path, "final_deployment_v1", research_config)
    assert guard.ok is False
    assert guard.kind == EXPERIMENT_CORE_KIND
    assert guard.expected_kind == DEPLOYMENT_KIND
    assert guard.message == EXPERIMENT_CORE_MESSAGE

    draft = prepare_deployment_draft(
        tmp_path, canonical_run, approver=APPROVER, approved_at_utc=APPROVED_AT
    )
    accepted = guard_config_for_command(
        tmp_path, "final_deployment_v1", draft.deployment_config_path
    )
    assert accepted.ok is True
    assert accepted.kind == DEPLOYMENT_KIND


def test_deployment_config_is_rejected_by_the_experiment_core_guard(
    tmp_path: Path, canonical_run: str
) -> None:
    draft = prepare_deployment_draft(
        tmp_path, canonical_run, approver=APPROVER, approved_at_utc=APPROVED_AT
    )
    guard = guard_config_for_command(
        tmp_path, "experiment_core_v2", draft.deployment_config_path
    )
    assert guard.ok is False
    assert guard.kind == DEPLOYMENT_KIND
    assert guard.expected_kind == EXPERIMENT_CORE_KIND
    assert "Final Deployment configuration" in str(guard.message)


def test_results_prefill_transfers_run_identity_not_source_yaml(
    canonical_run: str,
) -> None:
    session: dict[str, object] = {}
    apply_submission_handoff(session, canonical_run)

    assert session[SUBMISSION_HANDOFF_KEY] == canonical_run
    assert get_durable_value(session, candidate_durable_key()) == canonical_run
    assert session["run-command"] == SUBMISSION_COMMAND_ID
    assert session[f"run-action-{SUBMISSION_COMMAND_ID}"] == SUBMISSION_ACTION_ID
    prefill = session["run_prefill"]
    assert prefill == {
        "command_id": SUBMISSION_COMMAND_ID,
        "action_id": SUBMISSION_ACTION_ID,
        "values": {},
    }
    serialized = json.dumps(session, default=str)
    assert "resolved_config.yaml" not in serialized
    assert "ui_configs" not in serialized

    with pytest.raises(ValueError):
        apply_submission_handoff({}, "C:/absolute/run")


def test_workflow_order_labels_and_legacy_visibility() -> None:
    registered = list(load_registry(REPOSITORY_ROOT).commands)

    assert workflow_command_ids() == (
        "experiment_core_v2",
        "paired_comparison",
        "optuna_search_v1",
        "blend_evaluation_v1",
        "final_deployment_v1",
    )
    assert all(command_id in registered for command_id in workflow_command_ids())

    standard = visible_command_ids(registered)
    workflow = list(workflow_command_ids())
    assert standard[: len(workflow)] == workflow
    assert "research_v1" not in standard
    assert [workflow_label(item, fallback=item) for item in workflow] == [
        "🧪 Train",
        "🔎 Compare",
        "🎛️ Tune",
        "🧬 Blend",
        "📤 Generate submission",
    ]
    for command_id in workflow:
        label = workflow_label(command_id, fallback=command_id)
        for token in ("v1", "v2", "Experiment Core", "Final Deployment", "Research"):
            assert token not in label

    advanced = visible_command_ids(registered, include_legacy=True)
    assert advanced[: len(standard)] == standard
    assert "research_v1" in advanced
    assert "dataset_comparison_v1" in advanced
    assert sorted(advanced) == sorted(registered)
    assert is_legacy_command("research_v1") is True
    assert is_legacy_command("experiment_core_v2") is False
    assert LEGACY_BADGE in workflow_label("research_v1", fallback="research_v1")

    assert contract_declaration("experiment_core_v2") == {
        "input": (TRAIN_CONFIG_CONTRACT,),
        "intermediate": (),
        "output": (RESEARCH_V2_RUN_CONTRACT,),
    }
    submission = contract_declaration("final_deployment_v1")
    assert RESEARCH_V2_RUN_CONTRACT in submission["input"]
    assert "deployment_draft_v1" in submission["intermediate"]
    assert submission["output"] == ("deployment_v1_submission_artifact",)


def test_competition_test_safety_and_typed_deployment_inputs_unchanged() -> None:
    deployment = load_registry(REPOSITORY_ROOT).commands["final_deployment_v1"]
    actions = deployment.actions

    real_run = actions["run"]
    assert real_run.enabled is True
    assert real_run.title == "Generate submission"
    assert real_run.competition_test is True
    assert real_run.confirmation == "acknowledge"
    for action_id in ("validate", "dry_run", "inspect"):
        assert actions[action_id].enabled is True
        assert actions[action_id].competition_test is False

    # The generic mixed-schema UI config root can no longer reach deployment.
    assert not any(
        "ui_configs" in glob for glob in deployment.allowed_config_globs
    ), deployment.allowed_config_globs
    assert not any("ui_configs" in root for root in deployment.allowed_input_roots)
    assert f"{DEPLOYMENT_DRAFT_ROOT}/*/{DEPLOYMENT_CONFIG_NAME}" in (
        deployment.allowed_config_globs
    )
    assert DEPLOYMENT_DRAFT_ROOT in deployment.allowed_input_roots
    assert deployment.allowed_output_roots == ("artifacts/deployments",)
    for action_id in ("validate", "dry_run", "run"):
        roots = actions[action_id].placeholders["config"].roots
        assert DEPLOYMENT_DRAFT_ROOT in roots
        assert not any("ui_configs" in root for root in roots)


def test_draft_artifacts_contain_only_deployment_schema_content(
    tmp_path: Path, canonical_run: str
) -> None:
    draft = prepare_deployment_draft(
        tmp_path, canonical_run, approver=APPROVER, approved_at_utc=APPROVED_AT
    )
    config = yaml.safe_load(
        (tmp_path / draft.deployment_config_path).read_text(encoding="utf-8")
    )
    assert set(config) == CONFIG_KEYS
    assert "evaluation_plan" not in config
    assert "candidate_adapter" not in config

    approval = yaml.safe_load(
        (tmp_path / draft.candidate_approval_path).read_text(encoding="utf-8")
    )
    assert approval["component_id"] == ADAPTER_ID
    assert Path(draft.candidate_approval_path).name == APPROVAL_NAME
    assert config["components"][0]["approval_artifact_path"] == (
        draft.candidate_approval_path
    )


def test_generated_draft_preview_uses_operation_specific_roots(
    tmp_path: Path, canonical_run: str
) -> None:
    draft = prepare_deployment_draft(
        tmp_path, canonical_run, approver=APPROVER, approved_at_utc=APPROVED_AT
    )
    registry = load_registry(REPOSITORY_ROOT)
    deployment = registry.commands["final_deployment_v1"]
    train = registry.commands["experiment_core_v2"]
    deploy_roots = config_preview_allowed_roots(
        allowed_input_roots=deployment.allowed_input_roots,
        config_placeholder_roots=deployment.actions["validate"].placeholders[
            "config"
        ].roots,
        allowed_config_globs=deployment.allowed_config_globs,
    )
    train_roots = config_preview_allowed_roots(
        allowed_input_roots=train.allowed_input_roots,
        config_placeholder_roots=train.actions["validate"].placeholders["config"].roots,
        allowed_config_globs=train.allowed_config_globs,
    )

    assert DEPLOYMENT_DRAFT_ROOT in deploy_roots
    assert "artifacts/ui_configs" not in deploy_roots
    assert "artifacts/ui_configs" in train_roots
    assert DEPLOYMENT_DRAFT_ROOT not in train_roots

    text, canonical = read_config(
        tmp_path, draft.deployment_config_path, allowed_roots=deploy_roots
    )
    assert "deployment_id" in text
    assert canonical.name == DEPLOYMENT_CONFIG_NAME
    assert is_read_only_config_path(draft.deployment_config_path) is True
    assert (
        normalize_source_kind(draft.deployment_config_path)
        == "Generated deployment draft"
    )

    with pytest.raises(ConfigEditError, match="outside declared roots"):
        read_config(
            tmp_path, draft.deployment_config_path, allowed_roots=train_roots
        )


def test_user_facing_summary_for_generated_draft(
    tmp_path: Path, canonical_run: str
) -> None:
    draft = prepare_deployment_draft(
        tmp_path, canonical_run, approver=APPROVER, approved_at_utc=APPROVED_AT
    )
    operation = workflow_label("final_deployment_v1", fallback="final_deployment_v1")
    summary = build_pre_run_summary(
        "final_deployment_v1",
        "validate",
        operation,
        "Validate",
        {"config": draft.deployment_config_path},
        tmp_path,
    )
    assert operation == "📤 Generate submission"
    assert summary["Operation"] == "📤 Generate submission"
    assert summary["Source"] == "Generated deployment draft"
    assert summary["Config type"] == "Deployment"
    assert "[OTHER]" not in summary["Config"]
    assert "[DRAFT]" in summary["Config"]


def test_approver_durable_state_key_and_advanced_defaults() -> None:
    session: dict[str, object] = {}
    durable = ui_durable_key("run", "deployment_approver")
    set_durable_value(session, durable, "local-operator")
    assert get_durable_value(session, durable) == "local-operator"
    # Advanced approval fields keep their contract defaults when not customized.
    assert DEFAULT_INTENDED_ROLE
    assert DEFAULT_EXCEPTION_REASON


def test_submission_blocked_message_is_context_specific() -> None:
    # The Generate submission page must not reuse the dataset-driven Train message.
    import apps.experiment_control_panel as panel

    source = Path(panel.__file__).read_text(encoding="utf-8")
    assert "Prepare a deployment draft before running validation." in source
    assert "Validate and test" in source
    assert "Approved by" in source
    assert "Advanced approval details" in source
    assert "Enter the approver name to prepare the draft." in source
    # Dataset-driven Train message remains only for that entry mode.
    assert (
        "Prepare a valid dataset-driven configuration before Validate/Run."
        in source
    )
    assert 'command_id == "final_deployment_v1"' in source
    assert "use_dataset_driven" in source
