# MLflow local index

## Authority and scope

The filesystem artifact contracts remain authoritative. MLflow is an optional local
metadata index and UI over validated terminal runs. It does not participate in model
training, experiment identity, semantic validation, artifact finalization, threshold
selection, prediction generation, or submission behavior.

The index supports:

- Experiment Core v2 research runs under `artifacts/research_v2`;
- standalone AutoGluon runs under `artifacts/autogluon_runs`.

Paired-comparison artifacts are not implemented. A typed `SourceAdapter` protocol and
`SourceAdapterRegistry` are the extension point for a future comparison source.

## Local configuration and storage

The strict schema-version-1 configuration is `configs/mlflow/local.yaml`. It resolves
these ignored repository-local locations:

- SQLite metadata: `artifacts/mlflow/mlflow.db`;
- copied MLflow artifacts: `artifacts/mlflow/mlartifacts`;
- sync receipts: `artifacts/mlflow/sync_receipts`.

Config validation uses exact recursive keys and exact types, including rejecting a
boolean where an integer is expected. Paths must be repository-relative and are
rejected for POSIX roots, Windows roots, drive-qualified and drive-relative forms, UNC
forms, backslashes, traversal, and symlink or junction escapes. The SQLite database and
MLflow artifact root must stay under `artifacts/`.

MLflow 3.14.0 is already declared and locked in the main project environment. Imports
are lazy: configuration validation, dry-run, and unrelated project imports do not
import MLflow.

## Commands

Run from the repository root with the main `.venv`:

```powershell
.venv\Scripts\python.exe -u scripts\sync_mlflow.py validate `
  --config configs\mlflow\local.yaml

.venv\Scripts\python.exe -u scripts\sync_mlflow.py sync `
  --config configs\mlflow\local.yaml `
  --source-type research_v2 `
  --dry-run

.venv\Scripts\python.exe -u scripts\sync_mlflow.py sync `
  --config configs\mlflow\local.yaml `
  --source-type autogluon

.venv\Scripts\python.exe -u scripts\sync_mlflow.py sync `
  --config configs\mlflow\local.yaml `
  --source-type all
```

Use `--run-dir <path>` with one explicit source type to select one run. Use
`--fail-fast` to stop after the first rejected or errored source. Every command prints
JSON. Batch sync otherwise continues across independent invalid runs and reports exact
per-item outcomes and aggregate counts.

`validate` and `sync --dry-run` do not create the SQLite database, artifact root,
experiments, receipts, or MLflow runs. Dry-run validates and maps each discovered
source but intentionally does not query a possibly absent MLflow database.

Launch the local UI from the repository root:

```powershell
.venv\Scripts\python.exe -m mlflow ui `
  --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db `
  --host 127.0.0.1 `
  --port 5000
```

Then open `http://127.0.0.1:5000`. The two configured experiment names are
`neoversity-churn-research-v2` and `neoversity-churn-autogluon`.

## Source validation and status

Research-v2 completed runs must have one `_SUCCESS` marker. Before mapping, sync calls
the production `validate_research_v2_run()` semantic validator with success and
manifest verification enabled. This recomputes the evaluation semantics and verifies
the exact ordinary inventory, identities, metrics, thresholds, predictions, progress
artifacts, artifact manifest, and success marker. A corrupt completed source is
rejected and is never represented as successful.

Research-v2 failed runs must have one `_FAILED` marker and exact version-2 terminal
status structure. The available failed metadata must agree on run ID and failure
state. Failed runs are indexed for diagnosis but are not described as valid completed
evaluations.

AutoGluon completed runs use the standalone runner's read-only classification with
`attempt_load=False`. A run must classify as `complete`, including validated
completion, status, profile/config identities, predictor structure, inventory, and
terminal marker. Sync never loads `TabularPredictor` or model binaries.

AutoGluon failed runs require a structurally valid failed status, metadata agreement,
and matching `_FAILED` marker. Failure codes, child exit code, last observed stage,
duration, profile, and requested seed remain indexable even when completion-specific
artifacts are absent.

The explicit mapping is:

- valid completed source: MLflow `FINISHED`;
- structurally valid terminal failure: MLflow `FAILED`;
- incomplete or running source: skipped with a reason;
- corrupt source: rejected, never indexed as successful.

Source artifacts are read-only. Receipts are written only to the independent local
MLflow area.

## Identity, idempotence, and mutation policy

Each MLflow run receives immutable `mlflow_index.*` tags for sync schema, source type,
source run ID, source relative path, source key, and source identity. The source key is
a deterministic hash of source type and source run ID. Portable identity contains no
username or absolute path. A separately labelled
`mlflow_index.local_source_path_nonportable` tag records the operational local path.

Research completion identity is the verified source manifest hash. AutoGluon completion
identity covers config identity, profile identity, and validated artifact inventory.
Failed-run identity hashes the available terminal status and stable source identities.

Sync searches by deterministic source key through `MlflowClient`:

- the same source identity is unchanged and creates no duplicate run;
- an interrupted partial sync is resumable through `mlflow_index.sync_complete`;
- a changed identity for the same source run ID is reported as
  `source_mutation_detected`;
- an existing immutable parameter with a different value is reported as
  `immutable_parameter_conflict`;
- multiple pre-existing index rows for one source key are rejected.

The persisted mutation policy is deterministic rejection. Sync never overwrites
immutable parameters and never silently versions a mutated source. Failed-run re-sync
uses the same policy and is idempotent.

## Metadata mapping

Research-v2 parameters include source run ID, schema versions, plan ID/hash, dataset
version and identity, pipeline ID/hash, adapter ID/hash, complete candidate hash,
source and loaded-module provenance hashes, repeat/fold counts, threshold policy, and
source identity. Metrics include:

- Balanced Accuracy, sensitivity, specificity, ROC AUC, and average precision
  (`higher_is_better`);
- Brier score (`lower_is_better`);
- repeat-level means and sample standard deviations when defined;
- threshold median, minimum, maximum, range, and population standard deviation;
- evaluation duration.

AutoGluon parameters include run/config/profile identities, dataset version, requested
seed, AutoGluon version, included/excluded/resolved families, CPU/GPU and time budgets,
status information, duration, child and Windows exit codes, failure codes, last stage,
predictor classification, and—for valid completed runs only—best model, decision
threshold, and model count.

MLflow holds searchable scalar metadata, not evaluation authority. Kaggle scores,
submissions, competition test assets, and model-selection decisions are not part of
this sync.

## Artifact copy policy

Copying is optional and bounded by `sync.log_small_artifacts` and
`sync.max_artifact_size_bytes`. Only explicit human-readable metadata allowlists are
eligible, including resolved config, aggregate or inspection summary, status and run
metadata, manifest or inventory, worker completion, compact leaderboard CSV, and
terminal markers.

The sync never copies:

- model binaries or AutoGluon predictor directories;
- OOF or test probability tables;
- raw or processed datasets;
- competition test or sample-submission assets;
- submissions;
- logs;
- unlisted files, even when small.

Allowlisted candidates are still rejected if they are symlinks, non-regular files, or
escape the source run. Files above the size cap are omitted.
