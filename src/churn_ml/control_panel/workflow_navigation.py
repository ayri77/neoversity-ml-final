"""User-facing Control Panel workflow navigation and artifact-contract mapping.

The Control Panel exposes a short ordered workflow instead of technical
operation identities. Internal command IDs, action IDs, schema versions, and the
allowlisted CLI backend are unchanged; only presentation and the declarative
input/output contract mapping live here.

Each workflow step declares which artifact/config contract it accepts and which
contract it produces. That declaration is the minimum practical metadata needed
to prevent a config of one schema from being offered to an incompatible
operation.
"""

from __future__ import annotations

from dataclasses import dataclass


TRAIN_CONFIG_CONTRACT = "experiment_core_v2_config_v1"
RESEARCH_V2_RUN_CONTRACT = "research_v2_completed_run_v1"
RESEARCH_V1_CONFIG_CONTRACT = "research_v1_config_v1"
OPTUNA_CONFIG_CONTRACT = "optuna_search_config_v1"
OPTUNA_SEARCH_CONTRACT = "optuna_search_report_v1"
BLEND_CONFIG_CONTRACT = "blend_evaluation_config_v1"
BLEND_EVALUATION_CONTRACT = "blend_evaluation_v1_completed_evaluation"
BLEND_DEPLOYMENT_PACKAGE_CONTRACT = "blend_evaluation_v1_deployment_package"
PREDICTION_CANDIDATE_CONTRACT = "prediction_candidate_v1"
PREDICTION_BLEND_CONTRACT = "prediction_blend_v1"
CANDIDATE_SUBMISSION_CONTRACT = "candidate_submission_v1"
BLEND_UI_REQUEST_CONTRACT = "blend_ui_request_v1"
DEPLOYMENT_DRAFT_CONTRACT = "deployment_draft_v1"
DEPLOYMENT_CONFIG_CONTRACT = "deployment_v1_config"
DEPLOYMENT_ARTIFACT_CONTRACT = "deployment_v1_submission_artifact"
PAIRED_COMPARISON_CONTRACT = "paired_comparison_v1_artifact"
DATASET_COMPARISON_CONTRACT = "dataset_comparison_v1_artifact"
MLFLOW_INDEX_CONTRACT = "mlflow_local_index_mirror"

LEGACY_BADGE = "Legacy"


@dataclass(frozen=True)
class WorkflowStep:
    """One ordered user-facing workflow step backed by one registry command."""

    key: str
    label: str
    command_id: str
    description: str
    input_contracts: tuple[str, ...]
    output_contracts: tuple[str, ...]
    intermediate_contracts: tuple[str, ...] = ()

    @property
    def is_legacy(self) -> bool:
        return False


@dataclass(frozen=True)
class AdvancedOperation:
    """A registry command that is not one of the ordered workflow steps."""

    command_id: str
    label: str
    description: str
    badge: str | None
    input_contracts: tuple[str, ...]
    output_contracts: tuple[str, ...]

    @property
    def is_legacy(self) -> bool:
        return self.badge == LEGACY_BADGE


WORKFLOW_STEPS: tuple[WorkflowStep, ...] = (
    WorkflowStep(
        key="train",
        label="🧪 Train",
        command_id="experiment_core_v2",
        description=(
            "Train a candidate on a registered Dataset Package and record one "
            "complete, reproducible run with metrics, thresholds, and provenance."
        ),
        input_contracts=(TRAIN_CONFIG_CONTRACT,),
        output_contracts=(RESEARCH_V2_RUN_CONTRACT,),
    ),
    WorkflowStep(
        key="compare",
        label="🔎 Compare",
        command_id="paired_comparison",
        description=(
            "Compare completed runs across three scopes: same dataset with "
            "different candidates (official paired inference), same model across "
            "different datasets, or descriptive Research Workspace deltas."
        ),
        input_contracts=(RESEARCH_V2_RUN_CONTRACT,),
        output_contracts=(PAIRED_COMPARISON_CONTRACT,),
    ),
    WorkflowStep(
        key="tune",
        label="🎛️ Tune",
        command_id="optuna_search_v1",
        description=(
            "Search model parameters on training data only and export the best "
            "candidate as a new training configuration."
        ),
        input_contracts=(OPTUNA_CONFIG_CONTRACT,),
        output_contracts=(OPTUNA_SEARCH_CONTRACT, TRAIN_CONFIG_CONTRACT),
    ),
    WorkflowStep(
        key="generate_submission",
        label="📤 Generate submission",
        command_id="final_deployment_v1",
        description=(
            "Select a completed Research v2 run or a canonical prediction "
            "candidate, validate readiness, and generate a local Kaggle "
            "submission CSV without upload."
        ),
        input_contracts=(
            RESEARCH_V2_RUN_CONTRACT,
            BLEND_DEPLOYMENT_PACKAGE_CONTRACT,
            PREDICTION_CANDIDATE_CONTRACT,
            PREDICTION_BLEND_CONTRACT,
        ),
        intermediate_contracts=(
            DEPLOYMENT_DRAFT_CONTRACT,
            DEPLOYMENT_CONFIG_CONTRACT,
            CANDIDATE_SUBMISSION_CONTRACT,
        ),
        output_contracts=(DEPLOYMENT_ARTIFACT_CONTRACT, CANDIDATE_SUBMISSION_CONTRACT),
    ),
)

ADVANCED_OPERATIONS: tuple[AdvancedOperation, ...] = (
    AdvancedOperation(
        command_id="blend_evaluation_v1",
        label="Fixed Blend Evaluation v1",
        description=(
            "Superseded fixed LightGBM + XGBoost blend evaluation. Prefer the "
            "Blend Workspace for new multi-candidate blends. Kept only so "
            "historical blend evaluations remain reproducible."
        ),
        badge=LEGACY_BADGE,
        input_contracts=(BLEND_CONFIG_CONTRACT, RESEARCH_V2_RUN_CONTRACT),
        output_contracts=(BLEND_EVALUATION_CONTRACT, BLEND_DEPLOYMENT_PACKAGE_CONTRACT),
    ),
    AdvancedOperation(
        command_id="prediction_blend_v1",
        label="Prediction Blend v1",
        description=(
            "Supporting CLI for the Blend Workspace. Prefer the Blend page; "
            "this registry entry exists for authorized background jobs."
        ),
        badge=None,
        input_contracts=(PREDICTION_CANDIDATE_CONTRACT, BLEND_UI_REQUEST_CONTRACT),
        output_contracts=(PREDICTION_BLEND_CONTRACT, PREDICTION_CANDIDATE_CONTRACT),
    ),
    AdvancedOperation(
        command_id="candidate_submission_v1",
        label="Candidate Submission v1",
        description=(
            "Supporting CLI for canonical prediction-candidate submissions. "
            "Prefer Generate submission with the Canonical prediction candidate "
            "source."
        ),
        badge=None,
        input_contracts=(PREDICTION_CANDIDATE_CONTRACT,),
        output_contracts=(CANDIDATE_SUBMISSION_CONTRACT,),
    ),
    AdvancedOperation(
        command_id="dataset_comparison_v1",
        label="Dataset Comparison v1",
        description=(
            "Supporting cross-dataset comparison CLI. Prefer the Compare workflow "
            "scope selector; this registry entry exists for advanced launches."
        ),
        badge=None,
        input_contracts=(RESEARCH_V2_RUN_CONTRACT,),
        output_contracts=(DATASET_COMPARISON_CONTRACT,),
    ),
    AdvancedOperation(
        command_id="research_v1",
        label="Research evaluation v1",
        description=(
            "Superseded training backend kept only so historical runs stay "
            "reproducible. Do not use it for new training."
        ),
        badge=LEGACY_BADGE,
        input_contracts=(RESEARCH_V1_CONFIG_CONTRACT,),
        output_contracts=("research_v1_completed_run",),
    ),
    AdvancedOperation(
        command_id="mlflow_local_index",
        label="MLflow local index",
        description=(
            "Optional read-only metadata mirror. Filesystem artifacts remain "
            "authoritative."
        ),
        badge=None,
        input_contracts=("mlflow_local_index_config_v1",),
        output_contracts=(MLFLOW_INDEX_CONTRACT,),
    ),
)

_STEPS_BY_COMMAND = {step.command_id: step for step in WORKFLOW_STEPS}
_ADVANCED_BY_COMMAND = {item.command_id: item for item in ADVANCED_OPERATIONS}
ADVANCED_ONLY_COMMAND_IDS: frozenset[str] = frozenset(
    {
        "dataset_comparison_v1",
        "prediction_blend_v1",
        "candidate_submission_v1",
    }
)


def workflow_command_ids() -> tuple[str, ...]:
    """Deterministic ordered command IDs of the normal workflow."""
    return tuple(step.command_id for step in WORKFLOW_STEPS)


def advanced_command_ids() -> tuple[str, ...]:
    """Command IDs that are reachable only through the Advanced section."""
    return tuple(item.command_id for item in ADVANCED_OPERATIONS)


def legacy_command_ids() -> frozenset[str]:
    """Command IDs that must be labelled Legacy and hidden by default."""
    return frozenset(
        item.command_id for item in ADVANCED_OPERATIONS if item.is_legacy
    )


def is_legacy_command(command_id: str) -> bool:
    return command_id in legacy_command_ids()


def visible_command_ids(
    available: list[str] | tuple[str, ...],
    *,
    include_legacy: bool = False,
) -> list[str]:
    """Ordered command IDs to offer, restricted to registered commands.

    The workflow steps always come first in the declared order. Supporting
    operations follow so the registry stays the single source of truth. Legacy
    operations are appended only when the caller explicitly asks for them.
    Advanced-only supporting commands stay hidden from the Operation selector.
    """
    known = list(available)
    legacy = legacy_command_ids()
    advanced_only = ADVANCED_ONLY_COMMAND_IDS
    ordered = [item for item in workflow_command_ids() if item in known]
    tail = [item for item in advanced_command_ids() if item in known]
    tail.extend(item for item in known if item not in ordered and item not in tail)
    ordered.extend(
        item
        for item in tail
        if item not in ordered and item not in legacy and item not in advanced_only
    )
    if include_legacy:
        # Legacy and advanced-only supporting commands appear only when the
        # operator explicitly enables Advanced / Legacy visibility.
        ordered.extend(item for item in tail if item not in ordered)
    return ordered


def workflow_label(command_id: str, *, fallback: str) -> str:
    """User-facing label for a command, without technical version tokens."""
    step = _STEPS_BY_COMMAND.get(command_id)
    if step is not None:
        return step.label
    advanced = _ADVANCED_BY_COMMAND.get(command_id)
    if advanced is not None:
        if advanced.badge:
            return f"{advanced.label} ({advanced.badge})"
        return advanced.label
    return fallback


def workflow_description(command_id: str, *, fallback: str) -> str:
    """Short plain-language description shown right after selection."""
    step = _STEPS_BY_COMMAND.get(command_id)
    if step is not None:
        return step.description
    advanced = _ADVANCED_BY_COMMAND.get(command_id)
    if advanced is not None:
        return advanced.description
    return fallback


def contract_declaration(command_id: str) -> dict[str, tuple[str, ...]]:
    """Accepted input, intermediate, and produced output contracts."""
    step = _STEPS_BY_COMMAND.get(command_id)
    if step is not None:
        return {
            "input": step.input_contracts,
            "intermediate": step.intermediate_contracts,
            "output": step.output_contracts,
        }
    advanced = _ADVANCED_BY_COMMAND.get(command_id)
    if advanced is not None:
        return {
            "input": advanced.input_contracts,
            "intermediate": (),
            "output": advanced.output_contracts,
        }
    return {"input": (), "intermediate": (), "output": ()}


__all__ = [
    "ADVANCED_OPERATIONS",
    "ADVANCED_ONLY_COMMAND_IDS",
    "AdvancedOperation",
    "BLEND_DEPLOYMENT_PACKAGE_CONTRACT",
    "BLEND_UI_REQUEST_CONTRACT",
    "CANDIDATE_SUBMISSION_CONTRACT",
    "DATASET_COMPARISON_CONTRACT",
    "DEPLOYMENT_ARTIFACT_CONTRACT",
    "DEPLOYMENT_CONFIG_CONTRACT",
    "DEPLOYMENT_DRAFT_CONTRACT",
    "LEGACY_BADGE",
    "PAIRED_COMPARISON_CONTRACT",
    "PREDICTION_BLEND_CONTRACT",
    "PREDICTION_CANDIDATE_CONTRACT",
    "RESEARCH_V2_RUN_CONTRACT",
    "TRAIN_CONFIG_CONTRACT",
    "WORKFLOW_STEPS",
    "WorkflowStep",
    "advanced_command_ids",
    "contract_declaration",
    "is_legacy_command",
    "legacy_command_ids",
    "visible_command_ids",
    "workflow_command_ids",
    "workflow_description",
    "workflow_label",
]
