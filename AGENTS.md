# Repository Instructions

## Project behavior

- This is an existing working ML competition project. Prefer incremental, additive changes over large restructures.
- Preserve existing behavior until parity tests exist.
- Canonical notebooks are `notebooks/01_*.ipynb` through `notebooks/08_*.ipynb`. Do not modify them unless explicitly requested.
- Experimental notebooks belong under sandbox directories.
- Do not silently change dataset definitions, feature names or order, CV splits, seeds, thresholds, metric definitions, prediction averaging, or submission behavior.
- Clearly distinguish legacy evaluation protocols from new evaluation protocols.
- Kaggle Public Score is an external benchmark, not the primary model-selection criterion.

## Environments

- Use `.venv` for the main project.
- Use `.venv-autogluon` only for AutoGluon tasks.
- Do not install or update dependencies unless explicitly requested.
- Prefer `uv` commands for the main environment.
- Do not merge AutoGluon dependencies into the main environment.

## Data and artifacts

- Raw data, processed datasets, models, MLflow runs, local artifacts, and submissions are generated or local assets. Do not commit them unless explicitly requested.
- Never overwrite historical experiment artifacts by default.
- Preserve train/test row order and submission index alignment.
- Every new experiment must record the dataset version, ordered feature schema, CV protocol, seeds, threshold methodology, model parameters, environment, OOF predictions, test predictions, and metrics.

## Editing and validation

- Do not perform Git write operations unless explicitly requested.
- Run relevant tests and lightweight validation before reporting completion.
- Report changed files, commands run, validation results, and unresolved risks.
- Code, comments, configuration names, and documentation must be written in English.

## Git branches and worktrees

- Default to the currently checked-out task branch. Do not create a new branch or worktree unless the user explicitly requests it or approves a clearly explained need for isolation.
- Use at most one feature branch per logical task. Revisions, review fixes, agent handoffs, and integration work must remain on that branch.
- Do not create successive `codex/*`, `cursor/*`, `*-corrections`, `*-final`, or `integrate-*` branches for the same task.
- Before creating a branch or worktree, state its base branch, purpose, integration target, and cleanup plan.
- A branch-related task is not complete until the required changes are integrated or otherwise preserved, relevant validation passes, and no required unique commits or file changes remain.
- Authorization to create a temporary branch or worktree includes authorization to remove that same temporary branch or worktree after verified integration, unless the user asks to preserve it. This does not apply to pre-existing or long-lived branches.
- After verified integration, remove the temporary worktree, local branch, and pushed remote branch, then run `git fetch --prune`.
- Never force-delete an unmerged branch based only on its name, age, or the existence of a newer branch. First prove ancestry or audit its patch and file contents.
- Before reporting completion of branch-related work, verify `git branch --all`, `git worktree list`, and `git status --short --branch`.
- If cleanup cannot be completed safely, report every remaining branch or worktree and ask the user how to proceed. Never leave temporary Git state behind silently.

## Current project status and handoff

- Treat `docs/current-project-status.md` as the canonical cross-session and cross-dialog operational handoff.
- Read it before continuing a multi-stage task, then verify the active branch, HEAD, and Git status because volatile state may have changed.
- Update it whenever a material milestone, dataset or evaluation contract, roadmap order, active branch, known risk, or next action changes.
- Keep it aligned with implemented code, validated artifacts, and test results. Do not record planned work as completed.
- Do not update it for trivial refactoring or temporary investigation notes.
- This document supplements rather than replaces immutable dataset manifests, frozen experiment manifests, specialist documentation, artifacts, and Git history.
- Before closing a material project stage, verify that its last-updated date, current checkpoint, completed work, remaining roadmap, and immediate next action are accurate.