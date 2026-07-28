# Experiment Control Panel — user guide

Practical operator guide for the local Streamlit control panel. The panel builds
allowlisted argv and starts public CLIs; filesystem experiment artifacts remain
authoritative.

For the declarative contract and safety model, see
[Experiment Control Panel](experiment-control-panel.md).

## Required setup

From the repository root (main project environment, not `.venv-autogluon`):

```powershell
uv sync --extra ui
```

Initialize an Optuna lifecycle authority key **outside** the repository:

```powershell
.\.venv\Scripts\python.exe -u scripts\run_optuna_search.py authority-init `
  --output $env:USERPROFILE\churn-ml-secrets\optuna-lifecycle-authority.key `
  --create-parent
```

Set the PowerShell environment variable for the session:

```powershell
$env:CHURN_ML_OPTUNA_LIFECYCLE_AUTHORITY_KEY_FILE = "$env:USERPROFILE\churn-ml-secrets\optuna-lifecycle-authority.key"
```

Launch the panel:

```powershell
uv run --extra ui streamlit run apps/experiment_control_panel.py
```

## Pages

| Page | Purpose |
| --- | --- |
| **Dashboard** | Job counts, recent jobs/artifacts, command groups, MLflow link |
| **Run** | Choose operation/action, fill placeholders, review argv, start a job |
| **Jobs** | Refresh status, inspect redacted argv and log tails, stop when enabled |
| **Results** | Discover reader artifacts, summaries, display-only side-by-side, action list |
| **Configuration** | Registry validation summary and reload |

## Optuna actions

| Action | Notes |
| --- | --- |
| **Validate** | Config-only; no authority key required |
| **New study** | Starts a new train-only study; requires authority key |
| **Resume unfinished study** | Same public `run` CLI for an unfinished study only |
| **Inspect** | Select a completed search artifact (advanced manual path optional) |
| **Export best** | Select a completed search; output defaults under `artifacts/optuna_exports` |
| **Authority init** | Creates an external key; path is never displayed or persisted |

### Standard Optuna workflow

1. **Validate** an Optuna config on Run.
2. **New study** → confirm → start background job.
3. Monitor on **Jobs**.
4. Open **Results** (Optuna Search v1 reader) when complete.
5. **Inspect** the completed search from Run (artifact dropdown).
6. **Export best** → accept or edit the suggested
   `artifacts/optuna_exports/<search-basename>_best.yaml`.
7. Validate the exported candidate with Experiment Core (`--validate-only`).
8. Use controlled paired comparison / deployment validate only when that is the
   intended next gate.

### Operational limitation

Never increase `n_trials` on a finalized study. Create a new study, config, and
output/search ID instead. Resume is only for unfinished studies.

## MLflow workflow

On Run → **MLflow local index**:

1. **Validate** `configs/mlflow/local.yaml`.
2. **Validate sources** (dry-run) for `research_v2`, `autogluon`, or `all`.
3. **Synchronize** after confirmation to mirror approved filesystem metadata.

Launch or open the local MLflow UI separately (the panel does not start the server).
Use the Dashboard **Open MLflow** link or the documented `mlflow ui` command in
[MLflow Local Index](mlflow-local-index.md).

Rejected source artifacts mean the sync skipped or failed that source (schema,
markers, size, or policy). Fix or exclude the source; do not force-import rejected
runs. Re-sync is idempotent for already-indexed approved artifacts.

## Artifact locations

| Path | Role |
| --- | --- |
| `artifacts/optuna_searches` | Optuna search reports |
| `artifacts/optuna_exports` | Exported best-candidate YAML |
| `artifacts/ui_jobs` | UI operational job records only |
| `artifacts/ui_configs` | Editable config copies |
| `artifacts/mlflow` | Local MLflow index store |
