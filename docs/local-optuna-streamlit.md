# Local Optuna + Streamlit control panel

Concise first-launch guide for this repository on Windows PowerShell. Commands were
verified against the public CLIs in this branch.

For the full control-panel operator guide, see
[Experiment Control Panel user guide](experiment-control-panel-user-guide.md).

## 1. Install the UI extra

From the repository root, using the main project environment (not `.venv-autogluon`):

```powershell
uv sync --extra ui
```

## 2. Initialize an external Optuna authority key

Create the key outside the repository. Prefer a directory that already exists; use
`--create-parent` only when you intentionally want the CLI to create the parent.

```powershell
.\.venv\Scripts\python.exe -u scripts\run_optuna_search.py authority-init `
  --output $env:USERPROFILE\churn-ml-secrets\optuna-lifecycle-authority.key `
  --create-parent
```

The command prints a path-independent JSON success payload with a key fingerprint.
It never prints the key bytes or the destination path.

## 3. Set the authority environment variable for this session

```powershell
$env:CHURN_ML_OPTUNA_LIFECYCLE_AUTHORITY_KEY_FILE = "$env:USERPROFILE\churn-ml-secrets\optuna-lifecycle-authority.key"
```

Do not commit this path, paste the key contents into chat, or store the value in
repository files. The Streamlit control panel redacts the environment value and never
persists it.

## 4. Launch Streamlit

```powershell
uv run --extra ui streamlit run apps/experiment_control_panel.py
```

## 5. Validate an Optuna config

In the UI: **Run** → Optuna Search v1 → **Validate** → choose
`configs/optuna/xgboost_numeric_v1_smoke.yaml` (or the CatBoost smoke config).

Equivalent CLI (does not require the authority key):

```powershell
.\.venv\Scripts\python.exe -u scripts\run_optuna_search.py validate `
  --config configs\optuna\xgboost_numeric_v1_smoke.yaml
```

## 6. Start a tiny/smoke study

In the UI: **Run** → Optuna Search v1 → **New study** → same smoke config → confirm →
start the background job.

Equivalent CLI (requires the authority environment variable from step 3):

```powershell
.\.venv\Scripts\python.exe -u scripts\run_optuna_search.py run `
  --config configs\optuna\xgboost_numeric_v1_smoke.yaml
```

## 7. Monitor Jobs

Open **Jobs**, refresh, and inspect PID, exit code, redacted argv, and log tails.
Stop only after explicit confirmation when stopping is enabled.

## 8. Inspect Results

Open **Results**, select the Optuna Search v1 reader, and open a completed artifact
under `artifacts/optuna_searches/<search-id>/`.

Equivalent CLI:

```powershell
.\.venv\Scripts\python.exe -u scripts\run_optuna_search.py inspect `
  --search-dir artifacts\optuna_searches\<search-id>
```

## 9. Export the best candidate

In the UI: **Run** → Optuna Search v1 → **Export best candidate** → choose a
completed search from the artifact dropdown (or Advanced manual path) and accept or
edit the suggested output under `artifacts/optuna_exports`.

Equivalent CLI:

```powershell
.\.venv\Scripts\python.exe -u scripts\run_optuna_search.py export-best `
  --search-dir artifacts\optuna_searches\<search-id> `
  --output artifacts\optuna_exports\<search-id>_best.yaml
```

Validate the exported Experiment Core v2 candidate without fitting:

```powershell
.\.venv\Scripts\python.exe -u scripts\run_research_v2.py `
  --config artifacts\optuna_exports\<search-id>_best.yaml `
  --validate-only
```

## 10. Validate Deployment

In the UI: **Run** → Final Deployment v1 → **Validate**. Real competition-test
deployment remains disabled by default.

## Operational limitation

Never increase `n_trials` on a finalized study from this control panel. The Optuna
UI registry exposes resume only for unfinished studies. Additional trials require a
new study, a new config, and a new output/search ID.
