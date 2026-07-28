from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

import optuna
import pytest
import yaml

from src.churn_ml import optuna_search_authority as authority_mod
from src.churn_ml.optuna_search_artifacts import load_optuna_search_result
from src.churn_ml.optuna_search_authority import (
    AUTHORITY_KEY_FILE_ENV,
    EPOCHS_ATTR,
    LEGACY_FINAL_LEDGER_ATTR,
    LifecycleAuthorityRecorder,
    OptunaLifecycleAuthorityError,
    initialize_lifecycle_authority_key,
    load_lifecycle_authority_key,
)
from src.churn_ml.optuna_search_config import load_optuna_search_config
from src.churn_ml.optuna_search_lifecycle import (
    portable_dataset_identity,
    run_optuna_study,
)
from src.churn_ml.optuna_search_objective import build_search_assignments
from src.churn_ml.research_v2_data import load_research_v2_training_data
from tests.optuna_auth_support import (
    DeterministicAdapter,
    ensure_train_only_files,
    shrink_space,
    write_search_plan,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _prepare_extension_workspace(tmp_token: str | None = None) -> dict[str, Any]:
    ensure_train_only_files()
    token = tmp_token or uuid.uuid4().hex
    operational = PROJECT_ROOT / "artifacts" / "optuna" / "epoch_tests" / token
    reports = PROJECT_ROOT / "artifacts" / "optuna_searches" / "epoch_tests" / token
    if operational.exists():
        shutil.rmtree(operational)
    if reports.exists():
        shutil.rmtree(reports)
    operational.mkdir(parents=True)
    reports.mkdir(parents=True)
    space = yaml.safe_load(
        (
            PROJECT_ROOT / "configs/optuna/search_spaces/xgboost_numeric_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    assert isinstance(space, dict)
    shrink_space(space)
    space_path = operational / "space.yaml"
    _write_yaml(space_path, space)
    first_plan = write_search_plan(
        space_path=space_path,
        storage_path=operational / "study.db",
        reports=reports,
        study_name=f"epoch_{token}",
        n_trials=1,
    )
    second_plan = deepcopy(first_plan)
    second_plan["n_trials"] = 2
    first_path = operational / "first.yaml"
    second_path = operational / "second.yaml"
    _write_yaml(first_path, first_plan)
    _write_yaml(second_path, second_plan)
    return {
        "token": token,
        "operational": operational,
        "reports": reports,
        "first_path": first_path,
        "second_path": second_path,
        "storage": operational / "study.db",
    }


def _run_plan(config_path: Path) -> Any:
    config = load_optuna_search_config(config_path, project_root=PROJECT_ROOT)
    data = load_research_v2_training_data(config.base_config)
    return run_optuna_study(
        config,
        X=data.X,
        y=data.y,
        dataset_identity=portable_dataset_identity(
            data.fingerprints,
            project_root=PROJECT_ROOT,
        ),
        assignments=build_search_assignments(
            data.y,
            repeats=1,
            folds=3,
            assignment_seed=23,
        ),
        adapter=DeterministicAdapter(fail_first=False),
    )


def _load_epochs(storage: Path, study_name: str) -> list[dict[str, Any]]:
    with sqlite3.connect(storage) as connection:
        study_id = connection.execute(
            "SELECT study_id FROM studies WHERE study_name = ?",
            (study_name,),
        ).fetchone()[0]
        raw = connection.execute(
            "SELECT value_json FROM study_user_attributes "
            "WHERE study_id = ? AND key = ?",
            (study_id, EPOCHS_ATTR),
        ).fetchone()
    assert raw is not None
    epochs = json.loads(raw[0])
    assert isinstance(epochs, list)
    return epochs


def test_historical_epoch0_report_validates_after_epoch1(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    key_path = tmp_path / "authority.key"

    initialize_lifecycle_authority_key(
        key_path,
        project_root=PROJECT_ROOT,
        create_parent=False,
    )
    monkeypatch.setenv(AUTHORITY_KEY_FILE_ENV, str(key_path))
    workspace = _prepare_extension_workspace()
    try:
        first = _run_plan(workspace["first_path"])
        epoch0_report = first.search_dir
        first_ledger = json.loads(
            (epoch0_report / "lifecycle_authority_ledger.json").read_text(
                encoding="utf-8"
            )
        )
        assert first_ledger["epoch_number"] == 0
        assert "epoch_ledger" in first_ledger
        assert "final_ledger" not in first_ledger

        second = _run_plan(workspace["second_path"])
        epoch1_report = second.search_dir
        second_ledger = json.loads(
            (epoch1_report / "lifecycle_authority_ledger.json").read_text(
                encoding="utf-8"
            )
        )
        assert second_ledger["epoch_number"] == 1

        loaded0 = load_optuna_search_result(epoch0_report, project_root=PROJECT_ROOT)
        loaded1 = load_optuna_search_result(epoch1_report, project_root=PROJECT_ROOT)
        assert loaded0.study_summary["requested_trials"] == 1
        assert loaded1.study_summary["requested_trials"] == 2

        epochs = _load_epochs(
            workspace["storage"],
            study_name=json.loads(
                (epoch0_report / "study_summary.json").read_text(encoding="utf-8")
            )["study_name"],
        )
        assert len(epochs) == 2
        assert "signature_sha256" in epochs[0]
        assert "signature_sha256" in epochs[1]
        assert epochs[0] == first_ledger["epoch_ledger"]
        assert epochs[1] == second_ledger["epoch_ledger"]

        # Mutating epoch 0 SQLite authority rejects epoch 0 report.
        with sqlite3.connect(workspace["storage"]) as connection:
            study_id = connection.execute(
                "SELECT study_id FROM studies WHERE study_name = ?",
                (
                    json.loads(
                        (epoch0_report / "study_summary.json").read_text(
                            encoding="utf-8"
                        )
                    )["study_name"],
                ),
            ).fetchone()[0]
            mutated = deepcopy(epochs)
            mutated[0]["payload"]["configured_trial_target"] = 99
            connection.execute(
                "UPDATE study_user_attributes SET value_json = ? "
                "WHERE study_id = ? AND key = ?",
                (json.dumps(mutated), study_id, EPOCHS_ATTR),
            )
            connection.commit()
        with pytest.raises(Exception):
            load_optuna_search_result(epoch0_report, project_root=PROJECT_ROOT)

        # Restore epoch 0, mutate epoch 1 only: epoch 0 report still validates.
        with sqlite3.connect(workspace["storage"]) as connection:
            restored = deepcopy(epochs)
            restored[1] = deepcopy(epochs[0])
            connection.execute(
                "UPDATE study_user_attributes SET value_json = ? "
                "WHERE study_id = ? AND key = ?",
                (json.dumps(restored), study_id, EPOCHS_ATTR),
            )
            connection.commit()
        # Epoch history integrity should fail closed on duplicate/reorder.
        with pytest.raises(Exception):
            load_optuna_search_result(epoch1_report, project_root=PROJECT_ROOT)

        with sqlite3.connect(workspace["storage"]) as connection:
            connection.execute(
                "UPDATE study_user_attributes SET value_json = ? "
                "WHERE study_id = ? AND key = ?",
                (json.dumps(epochs), study_id, EPOCHS_ATTR),
            )
            connection.commit()
        load_optuna_search_result(epoch0_report, project_root=PROJECT_ROOT)

        # Attacker cannot move old report to a newer epoch.
        moved = json.loads(
            (epoch0_report / "lifecycle_authority_ledger.json").read_text(
                encoding="utf-8"
            )
        )
        moved["epoch_number"] = 1
        (epoch0_report / "lifecycle_authority_ledger.json").write_text(
            json.dumps(moved, sort_keys=True),
            encoding="utf-8",
        )
        with pytest.raises(Exception):
            load_optuna_search_result(epoch0_report, project_root=PROJECT_ROOT)
    finally:
        shutil.rmtree(workspace["operational"], ignore_errors=True)
        shutil.rmtree(workspace["reports"], ignore_errors=True)


class ExtensionInterrupt(BaseException):
    """Process-kill probe that must escape Optuna's Exception catch."""


@pytest.mark.parametrize(
    "interrupt_point",
    (
        "after_epoch_open",
        "after_allocation",
        "after_running",
        "after_terminal",
        "after_study_completed",
        "after_report_before_epoch_ledger",
        "after_epoch_ledger_before_success",
    ),
)
def test_target_extension_interruption_matrix(
    interrupt_point: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    key_path = tmp_path / "authority.key"
    initialize_lifecycle_authority_key(
        key_path,
        project_root=PROJECT_ROOT,
        create_parent=False,
    )
    monkeypatch.setenv(AUTHORITY_KEY_FILE_ENV, str(key_path))
    workspace = _prepare_extension_workspace()
    try:
        first = _run_plan(workspace["first_path"])
        assert first.study_summary["requested_trials"] == 1

        original_append = authority_mod.LifecycleAuthorityRecorder.append_event
        original_finalize = authority_mod.LifecycleAuthorityRecorder.finalize
        original_extend = (
            authority_mod.LifecycleAuthorityRecorder.extend_configured_trial_target
        )
        state = {"seen_extension": False}

        def extend_proxy(self: Any, new_target: int) -> None:
            original_extend(self, new_target)
            state["seen_extension"] = True
            if interrupt_point == "after_epoch_open":
                raise ExtensionInterrupt("interrupt after epoch open")

        def append_proxy(self: Any, *args: Any, **kwargs: Any) -> Any:
            event = original_append(self, *args, **kwargs)
            event_type = kwargs.get("event_type")
            if event_type == "trial_allocated" and state["seen_extension"]:
                if interrupt_point == "after_allocation":
                    raise ExtensionInterrupt("interrupt after allocation")
            if event_type == "trial_started" and state["seen_extension"]:
                if interrupt_point == "after_running":
                    raise ExtensionInterrupt("interrupt after running")
            if (
                event_type
                in {
                    "trial_completed",
                    "trial_execution_failed",
                }
                and state["seen_extension"]
            ):
                if interrupt_point == "after_terminal":
                    raise ExtensionInterrupt("interrupt after terminal")
            if event_type == "study_completed" and state["seen_extension"]:
                if interrupt_point == "after_study_completed":
                    raise ExtensionInterrupt("interrupt after study_completed")
            return event

        def finalize_proxy(self: Any, *args: Any, **kwargs: Any) -> Any:
            if interrupt_point == "after_report_before_epoch_ledger":
                raise ExtensionInterrupt("interrupt before epoch ledger")
            result = original_finalize(self, *args, **kwargs)
            if interrupt_point == "after_epoch_ledger_before_success":
                raise ExtensionInterrupt("interrupt before success")
            return result

        monkeypatch.setattr(
            authority_mod.LifecycleAuthorityRecorder,
            "extend_configured_trial_target",
            extend_proxy,
        )
        monkeypatch.setattr(
            authority_mod.LifecycleAuthorityRecorder,
            "append_event",
            append_proxy,
        )
        monkeypatch.setattr(
            authority_mod.LifecycleAuthorityRecorder,
            "finalize",
            finalize_proxy,
        )

        with pytest.raises(ExtensionInterrupt):
            _run_plan(workspace["second_path"])

        monkeypatch.setattr(
            authority_mod.LifecycleAuthorityRecorder,
            "extend_configured_trial_target",
            original_extend,
        )
        monkeypatch.setattr(
            authority_mod.LifecycleAuthorityRecorder,
            "append_event",
            original_append,
        )
        monkeypatch.setattr(
            authority_mod.LifecycleAuthorityRecorder,
            "finalize",
            original_finalize,
        )

        resumed = _run_plan(workspace["second_path"])
        assert resumed.study_summary["requested_trials"] == 2
        loaded = load_optuna_search_result(
            resumed.search_dir,
            project_root=PROJECT_ROOT,
        )
        assert loaded.study_summary["actual_trials"] == 2
        load_optuna_search_result(first.search_dir, project_root=PROJECT_ROOT)
    finally:
        shutil.rmtree(workspace["operational"], ignore_errors=True)
        shutil.rmtree(workspace["reports"], ignore_errors=True)


def test_singleton_final_ledger_schema_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    key_path = tmp_path / "authority.key"
    initialize_lifecycle_authority_key(
        key_path,
        project_root=PROJECT_ROOT,
        create_parent=False,
    )
    monkeypatch.setenv(AUTHORITY_KEY_FILE_ENV, str(key_path))
    workspace = _prepare_extension_workspace()
    try:
        first = _run_plan(workspace["first_path"])
        study_name = json.loads(
            (first.search_dir / "study_summary.json").read_text(encoding="utf-8")
        )["study_name"]
        with sqlite3.connect(workspace["storage"]) as connection:
            study_id = connection.execute(
                "SELECT study_id FROM studies WHERE study_name = ?",
                (study_name,),
            ).fetchone()[0]
            connection.execute(
                "DELETE FROM study_user_attributes WHERE study_id = ? AND key = ?",
                (study_id, EPOCHS_ATTR),
            )
            connection.execute(
                "INSERT INTO study_user_attributes(study_id, key, value_json) "
                "VALUES (?, ?, ?)",
                (
                    study_id,
                    LEGACY_FINAL_LEDGER_ATTR,
                    json.dumps({"authority_schema_version": 1, "final_ledger": {}}),
                ),
            )
            connection.commit()
        key = load_lifecycle_authority_key(project_root=PROJECT_ROOT)

        study = optuna.load_study(
            study_name=study_name,
            storage=f"sqlite:///{workspace['storage'].as_posix()}",
        )
        with pytest.raises(
            OptunaLifecycleAuthorityError, match="singleton|unsupported"
        ):
            LifecycleAuthorityRecorder.initialize_or_load(
                study=study,
                key=key,
                base_search_identity_sha256=study.user_attrs[
                    "lifecycle_authority_initial_statement"
                ]["payload"]["base_search_identity_sha256"],
                dataset_identity_sha256=study.user_attrs[
                    "lifecycle_authority_initial_statement"
                ]["payload"]["dataset_identity_sha256"],
                assignment_identity_sha256=study.user_attrs[
                    "lifecycle_authority_initial_statement"
                ]["payload"]["assignment_identity_sha256"],
                source_closure_identity_sha256=study.user_attrs[
                    "lifecycle_authority_initial_statement"
                ]["payload"]["source_closure_identity_sha256"],
                runtime_identity_sha256=study.user_attrs[
                    "lifecycle_authority_initial_statement"
                ]["payload"]["runtime_identity_sha256"],
                sampler_pruner_identity_sha256=study.user_attrs[
                    "lifecycle_authority_initial_statement"
                ]["payload"]["sampler_pruner_identity_sha256"],
                configured_trial_count=2,
                allow_initialize=False,
            )
    finally:
        shutil.rmtree(workspace["operational"], ignore_errors=True)
        shutil.rmtree(workspace["reports"], ignore_errors=True)
