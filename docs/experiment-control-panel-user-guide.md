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

## The standard workflow

Run offers one ordered workflow instead of technical operation names:

```text
🧪 Train → 🔎 Compare → 🎛️ Tune → 🧬 Blend → 📤 Generate submission
```

| Step | What it does | Selects | Produces |
| --- | --- | --- | --- |
| **🧪 Train** | Trains one candidate on a registered Dataset Package | Training configuration | Completed canonical run |
| **🔎 Compare** | Compares two completed runs on identical folds and seeds | Completed runs | Paired-comparison artifact |
| **🎛️ Tune** | Searches parameters on training data only | Search configuration | Search report, exported training config |
| **🧬 Blend** | Combines completed runs with leakage-safe cross-fitting | Completed runs | Blend evaluation, deployment package |
| **📤 Generate submission** | Turns one completed run into a validated deployment draft | Completed run | Deployment draft, then deployment artifact |

Notes:

- **Train** is the standard training backend. **Research evaluation v1** is
  legacy: it stays hidden behind **Advanced / Legacy operations** on Run, is
  labelled `Legacy`, and must not be used for new training.
- Downstream steps select completed runs and artifacts, not raw YAML paths.
- Technical operation IDs, action IDs, and schema versions are shown only inside
  the **Technical details** expander.
- Dataset Comparison and canonical handoffs for Tune and Blend remain subsequent
  work; those steps still start from a configuration.

## Pages

| Page | Purpose |
| --- | --- |
| **Dashboard** | Job counts, recent jobs/artifacts, command groups, MLflow link |
| **Run** | Choose a workflow step and action, fill placeholders, review argv, start a job |
| **Jobs** | Refresh status, inspect redacted argv and log tails, stop when enabled, archive terminal jobs, permanently delete archived UI job metadata/logs |
| **Results** | Discover reader artifacts, summaries, display-only side-by-side, archive/hide results without deleting files |
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
| `artifacts/ui_configs` | Editable config copies and dataset-driven prepared configs (training schemas only) |
| `artifacts/ui_configs/plans` | Dataset-driven prepared evaluation plans (not config selectors) |
| `artifacts/deployment_drafts` | Generated deployment drafts (deployment schema only) |
| `artifacts/deployment_fixtures` | Approved synthetic dry-run fixtures |
| `artifacts/deployments` | Deployment outputs |
| `artifacts/mlflow` | Local MLflow index store |

## Understanding the control panel UI

### Operations and Actions

**Operation** is the workflow step (Train, Compare, Tune, Blend, Generate
submission) plus any supporting operation such as MLflow local index.
**Action** is the specific task within that step (e.g. Run, Validate, New study).
These are separate because they have different safety levels, configs, and CLIs.
The technical command and action IDs behind the selected step are listed under
**Technical details**, together with the artifact contract the step accepts and
the contract it produces.

### Source types

| Source | Meaning |
| --- | --- |
| Canonical config | Version-controlled config in `configs/` |
| Optuna export | Best-trial YAML exported from a completed search |
| UI config copy | Editable copy saved under `artifacts/ui_configs/` |

### Model and Mode

**Model** is inferred from the config filename (e.g. `manual_lightgbm_te_v1_compat` → LightGBM).
**Mode** is inferred similarly (e.g. `_development` → Development, `_smoke` → Smoke).

### Smoke vs Development

| Mode | Purpose |
| --- | --- |
| SMOKE | Quick technical check; small subset; not for scoring |
| DEVELOPMENT | Full evaluation protocol; use for model selection |

### Verifying the pre-run summary

Before clicking "Start background job", expand **Pre-run summary** and verify:

- Model matches your intention (e.g. LightGBM, not XGBoost).
- Mode is correct (Development for real evaluations, Smoke for checks).
- Source shows "Canonical config" unless you intentionally selected an Optuna export.
- Config basename matches the file you intended.

The exact argv below the summary is always authoritative.

### Example: dataset-driven Registry experiment

1. Operation → **Experiment Core v2**
2. Entry mode → **Dataset-driven experiment**
3. Select a registered Dataset Package (exploratory packages show a warning)
4. Choose Model, Evaluation mode, and base configuration template
5. Click **Prepare run configuration** (explicit; not automatic on rerun)
6. Review Pre-run summary: dataset, parent, target dependency, pipeline,
   prepared config/plan, and exact argv
7. Action → **Validate**, then **Run** when ready

Generated files stay under `artifacts/ui_configs/` (plans under
`artifacts/ui_configs/plans/`). They are local artifacts and are not committed.

### Example: running manual_lightgbm_te_v1_compat_development.yaml

1. Operation → **Experiment Core v2**
2. Entry mode → **Existing config**
3. Action → **Run**
4. Config → select **LightGBM — Development — manual TE compatibility [CANONICAL]**
5. Expand Pre-run summary and confirm: Model=LightGBM, Mode=Development, Source=Canonical config
6. Check the confirmation box
7. Click **Start background job**

### Example: Generate submission from a completed run

Generate submission starts from a **completed run**, never from a deployment YAML
path. The deployment configuration is generated for you.

1. Operation → **📤 Generate submission**
2. Action → **Validate**
3. **1. Select a completed run** — pick the exact run. Labels show Dataset ID,
   model family, config identity, mode, Balanced Accuracy, and short run
   identity. Nothing is auto-selected by best metric; when several runs share a
   dataset and model family the panel says so and keeps your choice.
   Exploratory Dataset Packages are hidden until you enable them and always
   carry a warning.
4. **2. Deployment readiness** — a deterministic report of run identity, dataset
   identity, model/adapter identity, resolved plan identity, selected threshold
   evidence, the competition-test access flag (`false`), and the deployment
   schema that will be produced. When something authoritative is missing the
   step is **blocked** and every blocking reason is listed. Nothing is guessed.
5. **3. Competition submission readiness** — authenticates the local sample
   submission and the separate test-row identity artifact. When ready, the panel
   shows `Competition submission readiness: ready`; otherwise it lists exact
   blocking reasons. Submission IDs are never model features.
6. **4. Prepare deployment draft** — enter **Approved by** (remembered locally as
   an audit record), optionally open **Advanced approval details**, then click
   **Prepare deployment draft**. Generated drafts are **read-only** preview
   artifacts under `artifacts/deployment_drafts/<candidate-id>/`. Resolved drafts
   use a revision suffix derived from the authenticated competition asset
   fingerprint so an older unresolved draft is never overwritten. Regenerating
   identical content is idempotent; differing content conflicts instead of
   replacing, and a recorded approval is never rewritten.
7. **5. Validate and test** — the generated draft is used as the `config`
   argument. Start the job and confirm `"ok": true` on Jobs.
8. **Synthetic dry run** — select action **Synthetic dry run** and provide an
   approved fixture directory under `artifacts/deployment_fixtures` plus a new
   output directory under `artifacts/deployments`.
9. **Generate submission** — available only when competition readiness is ready
   and a resolved draft carries authenticated sample-submission plus row-identity
   references. Requires explicit competition-test acknowledgement. Output stays
   local under `artifacts/deployments/<deployment_id>` with no-overwrite; network
   upload to Kaggle remains disabled.

You can also start from **Results → Research Workspace**: open a run and click
**Prepare for submission**. That transfers only the run identity to Generate
submission (never the Research v2 configuration) and shows the same readiness and
builder behavior.

### Dataset-aware jobs and results

- Job labels include Dataset ID so identical model/mode launches on different
  packages stay distinguishable.
- On the selected Job page, Dataset ID, experiment ID, plan ID, and config appear
  outside the technical JSON expander.
- Results → Experiments lists Dataset, Parent dataset, Target dependency, and
  Features and supports a Dataset filter.
- Results → Compare shows left/right Dataset IDs. Cross-dataset rows are
  descriptive only. Official Paired Comparison preparation uses the
  authoritative compatibility contract (not the lightweight display tokens)
  and stays disabled until that contract passes. After changing Control Panel
  presentation modules, fully restart Streamlit.

### Research Workspace

Open **Results → Research Workspace** to review Research v2 screening runs.

- Inventory lists every discoverable Research v2 run with Dataset, model,
  protocol, metrics, and identity fields. Missing fields show as unavailable.
- Comparability badges explain whether a run is comparable development,
  smoke, exploratory (`v3`), tuned/Optuna, legacy, incomplete, or invalid.
  These badges are descriptive and do **not** replace official Paired
  Comparison readiness.
- The matrix is Dataset Package × model family. Default filters hide smoke,
  failed/invalid, archived, exploratory `v3`, and legacy runs when a
  comparable development run already exists for that cell.
- When a cell has multiple eligible runs, the UI shows the duplicate count and
  lets you choose the exact run. It does not auto-crown the highest BA.
- Baseline deltas vs `v0_raw_minimal` (or another selected package) are
  descriptive aggregate deltas only — not Stage E paired inference.
- Tags, notes, and shortlist are stored under
  `artifacts/control_panel_state/research_annotations.json` and never rewrite
  Research v2 artifacts.
- **Export visible inventory CSV** downloads the filtered rows.
- **Refresh research inventory** clears only the Research Workspace discovery
  cache.

## Workspace cleanup (archive / job delete)

Stage F maintenance capability. This is separate from Dataset Campaign execution,
Stage E cross-dataset comparison, and campaign-results UI.

### Archive

- **Archive** hides an item from normal Control Panel lists and charts.
- It does **not** move, rename, edit, or delete authoritative experiment files.
- Supported for terminal UI jobs and configured Results artifacts.
- Toggle **Show archived** to reveal archived items; **Restore** returns them to
  the default lists.
- Overlay state is stored under
  `artifacts/control_panel_state/archived_items.json` (ignored with `artifacts/`).

### Permanent UI job deletion

- Only **archived terminal** UI jobs can be permanently deleted.
- Deletion removes only `artifacts/ui_jobs/<canonical-job-uuid>/` (metadata + logs,
  including an optional local `mlflow_index.json` sidecar).
- Research v2 / comparison / blend / deployment artifacts and MLflow runs/receipts
  are **not** deleted.
- Requires checkbox confirmation plus typing the exact job ID, with a file preview.
- Authoritative result directories cannot be physically deleted in this v1.

### Dataset Registry cache

- Dataset-driven Run discovery is cached (`st.cache_data`) keyed by repository root,
  processed-data root, and discovery contract version.
- **Refresh datasets** clears only that cache and rescans; it does not launch or
  validate an experiment.
- Jobs **Refresh job status** no longer calls global `st.cache_data.clear()`, so it
  does not wipe the Registry cache.