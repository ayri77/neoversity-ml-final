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
