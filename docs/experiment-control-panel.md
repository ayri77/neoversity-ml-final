# Experiment Control Panel v1

## Purpose and boundary

The Experiment Control Panel is a small local Streamlit application for operating the
project's approved public CLIs. It is a visual control surface, not an ML runtime. It
does not import fitting code, Optuna internals, Experiment Core internals, deployment
internals, or metric and threshold implementations.

The application can load declarative registries, build allowlisted argv arrays, start
public CLI commands with `subprocess` and `shell=False`, persist operational job
metadata and logs, and render configured result files. Files under `artifacts/ui_jobs`
are operational records only. The existing filesystem experiment artifacts remain
authoritative, and the optional MLflow index remains a local searchable mirror.

For day-to-day operator steps (setup, Optuna and MLflow workflows, artifact locations),
see the [Experiment Control Panel user guide](experiment-control-panel-user-guide.md).

## Installation and launch

From the repository root:

```powershell
uv run --extra ui streamlit run apps/experiment_control_panel.py
```

Streamlit and `psutil` are isolated in the `ui` optional dependency extra. The
application uses the main project environment, not `.venv-autogluon`.

## Pages

- **Dashboard** shows job-state counts, recent jobs, recent configured artifacts,
  command groups, and the configured MLflow link.
- **Run** selects an allowlisted operation and action, reads an allowed YAML or JSON
  configuration, validates syntax, displays exact redacted argv, and starts one
  background job after the required confirmation.
- **Jobs** refreshes persisted job state, shows PID, timestamps, elapsed time, exit
  code, exact redacted argv, and bounded stdout/stderr tails. Experiment Core jobs
  persist Dataset ID / experiment / plan / model / mode in `job.json` references and
  surface them in human-readable labels. A running process can be stopped only after
  explicit confirmation when stopping is enabled.
- **Results** discovers artifacts only through reader definitions, renders configured
  summary fields and JSON trees, previews bounded CSV data, tails configured logs, and
  shows a display-only side-by-side field table with explicit left/right Dataset IDs.
  Official comparison remains the Paired Comparison CLI action.
- **Configuration** reports strict registry validation, source paths, and loaded
  settings/command/reader summaries. The three UI registry files are read-only in v1.

The UI uses manual refresh. It does not use aggressive automatic reruns and never
retries a command automatically.

## Experiment Core v2 entry modes

On **Run → Experiment Core v2**, choose one entry mode:

- **Dataset-driven experiment** (recommended for registered Dataset Packages)
  discovers packages dynamically through
  `discover_registered_datasets(data/processed)`, shows lineage and
  `target_dependency`, warns on exploratory packages, and materializes a local
  Research v2 config/plan pair under `artifacts/ui_configs` after an explicit
  **Prepare run configuration** action.
- **Existing config** keeps the historical Source → Model → Mode → Config
  cascade for canonical Research v2 YAML, UI copies, and Optuna exports.

Dataset-driven preparation always forces:

```yaml
feature_pipeline:
  id: registered_prepared_passthrough_v1
  contract:
    mode: registry_prepared_passthrough_v1
    drop: []
    keep_all_features: true
```

Candidate adapter contracts and evaluation-protocol sections are copied from the
selected base template without change. Generated plans live under
`artifacts/ui_configs/plans/` so they do not appear in config selectors. Canonical
configs and plans remain read-only; publication is create-if-absent, with safe
identical-content reuse and no overwrite on content conflicts. Changing dataset,
model, mode, or template invalidates any previously prepared selection.

Validate/Run then use the prepared config path in the exact allowlisted argv.

## Declarative contracts

All UI integration lives under `configs/ui/`.

### `ui_settings.json`

Schema version 1 has an exact key set:

- `working_directory` must remain a safe repository-relative path (`.` by default);
- `jobs_root` and `editable_config_root` are safe repository-relative paths;
- polling and log-tail values are bounded integers;
- process stopping and copy editing are exact booleans;
- `mlflow_url` is an HTTP(S) URL.

Unknown keys, wrong primitive types, unsupported versions, absolute paths, and
traversal are rejected with readable errors.

### `ui_commands.yaml`

Schema version 1 defines command groups and actions. A group declares a stable ID,
title, description, category, fixed Python argv prefix, allowed configuration globs,
input/output roots, optional result reader, optional URL, and an explicit child
environment allowlist. Each action declares:

- a stable ID and description;
- a fixed argv token list;
- typed whole-token placeholders (`path`, `string`, `integer`, or `enum`);
- placeholder roles and path roots;
- confirmation and competition-test requirements;
- enabled/disabled state;
- stdout success/failure markers.

#### Approved executable-prefix contract

Every executable prefix begins with the registry sentinel `$PYTHON`, which resolves
only to the running UI interpreter. The only accepted shapes are:

- `["$PYTHON", "scripts/<public-cli>.py"]`
- `["$PYTHON", "-u", "scripts/<public-cli>.py"]`

Rejected at registry load: `$PYTHON -c`, `$PYTHON -m`, stdin (`-`), arbitrary
interpreter flags, external or absolute scripts, traversal, linked scripts, and any
path that is not an exact approved `scripts/*.py` public CLI. Widgets never accept
executable or arbitrary command text. A new supported CLI action can therefore be
added without Python changes by adding a strict action entry and its typed
placeholders.

The initial registry is derived from actual public `--help` output:

- Experiment Core v2: validate and run;
- legacy train-only research: validate and run;
- Paired Comparison v1: validate and run;
- MLflow local index: validate, dry-run source validation, and sync;
- Final Deployment v1: validate, synthetic dry-run, inspect, and a real run that is
  disabled by default;
- Optuna Search v1: validate (default), authority-init, new study, resume unfinished
  study, inspect, and export-best.

There is no public Optuna `candidate --validate-only` subcommand, no MLflow start
command, no Paired Comparison inspect/export command, and no standalone result
export command outside Optuna `export-best`. The UI does not guess missing
interfaces. MLflow is opened through its configured URL.

Optuna lifecycle runtime actions pass through the declared environment variable
`CHURN_ML_OPTUNA_LIFECYCLE_AUTHORITY_KEY_FILE` when present. The variable is
sensitive and never displayed or persisted. Structural config validation does not
require it, matching the public CLI. Run, resume, inspect, and export-best fail
clearly in the child process when the key is absent. The registry does not expose
an action that raises `n_trials` on a finalized study; additional trials require a
new study, config, and output ID.

Path placeholders may set `external_absolute: true` only for Optuna
`authority-init` output. That value must be an absolute filesystem path outside
the repository; when marked sensitive, the path is redacted from argv displays and
job metadata.

### `ui_readers.yaml`

Schema version 1 declares artifact roots, discovery globs, terminal markers, summary
JSON files, labeled dot-path fields, bounded CSV preview files, log files, and
side-by-side compare fields. Python contains no model-specific result parser.

Marker paths must be safe POSIX-style relative paths at registry load. Traversal,
absolute, drive-qualified, UNC-like, empty, `.`, and `..` segments are rejected.
Marker reads stay inside the artifact directory and reject symlinks, junctions,
reparse ancestors, hard-linked markers, and external targets.

Dot paths traverse mapping keys and canonical numeric list indexes only. Malformed
reader fields such as `metrics..score`, leading/trailing dots, wildcards, or
zero-padded indexes are rejected at registry load. Dot paths do not execute
expressions.

If both success and failure markers exist for an artifact, the reader returns an
explicit `invalid` state with a diagnostic. It does not report success.

## Fail-closed launch authorization

Disabled Streamlit widgets are not a security boundary. Immediately before
`JobManager.start()`, the Run page rechecks:

- action `enabled`;
- current command build success against the live selection;
- required confirmation;
- high-risk / competition-test acknowledgement when declared;
- a one-time launch token bound to the current command, action, and argv fingerprint.

A consumed token cannot be replayed. Streamlit AppTest coverage includes forged
session state and replayed tokens. Changing the live selection mints a new token.

## Configuration copy editing

Canonical configurations are displayed read-only. The Advanced editor validates a
primitive-only YAML or JSON mapping and writes only a new, safely named file beneath
`artifacts/ui_configs`. Publication uses create-if-absent semantics: an existing
target is never overwritten, exactly one concurrent writer succeeds, and temporary
siblings are cleaned up on failure. Registry and settings files cannot be edited in
the application.

## Jobs and logs

Each job receives a UUID directory:

```text
artifacts/ui_jobs/<job-id>/
├── job.json
├── command.json
├── status.json
├── terminal.json   # optional durable exit record from the job wrapper
├── stdout.log
└── stderr.log
```

Job directories are loaded non-followingly. `jobs_root`, UUID directory names,
ancestors, metadata files, and logs are validated; symlink/junction/reparse paths
and external targets are rejected. Persisted schemas are strict, and the directory
name must equal `job_id`.

JSON writes use a temporary sibling, flush, filesystem sync, and atomic replace.
stdout/stderr are append-only. Persisted states are `created`, `running`, `succeeded`,
`failed`, `stop_requested`, `stopped`, and `orphaned`.

Every UI-launched command runs through a small wrapper process. The wrapper is the
tracked process; it executes the allowlisted target with `shell=False`, forwards
stdout/stderr into the job logs, and atomically writes `terminal.json` with
`terminal_status` (`succeeded` or `failed`), the exit code, and UTC timestamps before
exiting. The terminal record never stores raw argv, environment values, or secrets.

### Process identity and orphan behavior

While a job is live, status persists PID, creation time, executable identity, argv
fingerprint, and process-group/session identity where the platform provides them.
Refresh and stop verify that identity before acting. A process is never signalled
based only on PID existence, which protects against PID reuse.

Refresh prefers a valid `terminal.json` over the in-memory process handle, so a short
completed job still becomes `succeeded` or `failed` after a Streamlit rerun. Mismatched
or unverifiable live jobs become `orphaned` only when there is no trustworthy terminal
record and no verifiable live process. After an application restart, a still-matching
process remains `running`. There is no database, recursive deletion, artifact cleanup,
or automatic retry.

### Minimal child environment

Child processes do not inherit the complete parent environment. The backend builds a
documented base allowlist (`PATH`, locale, temp, `HOME`, and Windows home/system roots
`USERPROFILE`, `HOMEDRIVE`, `HOMEPATH`, `SYSTEMROOT`, `WINDIR`, and `COMSPEC` when
present) plus only variable names declared on the command registry entry. Required
declared variables must be present; ambient credentials that are not declared are
excluded. Sensitive values may exist only transiently in the immediate execution
argv or child environment and are redacted from references, job JSON, command JSON,
status, logs, diagnostics, and UI output.

## Safety model

- Commands are allowlisted registry IDs and actions.
- Execution always uses an argv list and `subprocess.Popen(..., shell=False)`.
- Placeholder substitution occupies complete argv tokens, so shell metacharacters are
  inert values.
- Absolute, drive-qualified, traversal, symlink, junction, reparse, and
  repository-escape paths are rejected.
- Paths must remain within placeholder roots, and input/output paths cannot overlap or
  contain one another.
- Exact redacted argv is displayed and persisted. Sensitive placeholder values, if
  configured, are not stored.
- Run, comparison, synchronization, dry-run, export-like, and stop operations require
  confirmation as declared.
- Any competition-test/deployment action requires an additional acknowledgement.
- The real deployment action is disabled by default even though its exact public argv
  is recorded in the registry.
- Launch authorization is fail-closed and tokenized; widget disabled state is not
  trusted.

## Dataset identity visibility

Dataset identity is a first-class presentation dimension end-to-end:

```text
Prepared config
  → persisted UI job identity (job.json references)
  → dataset-aware Jobs / Dashboard labels
  → dataset-aware Results / Compare
  → validated MLflow searchable metadata
```

### Job labels and metadata

New Experiment Core Validate/Run jobs persist `dataset_id`, `experiment_id`,
`plan_id`, `model_family`, `mode`, and `config` in the existing `job.json`
`references` map (schema version 2). Values are derived only from the already
authorized repository-contained config and never influence executable argv.
Dashboard and Jobs labels include Dataset ID, model, mode, operation, timestamp,
and a short job id fragment. Old jobs without these keys continue to load.

### Results workspace

The Experiments table adds Dataset, Parent dataset, Target dependency, and
Features; Dataset filter and text search include Dataset ID; chart labels and
hover data include Dataset ID. Inspect shows a compact provenance summary
(hashes may sit in an expander) and warns when `target_dependency` is
`exploratory`. Selectors expose Dataset ID in cascade labels.

Historical Experiment Core identity is read from run-local artifacts
(`dataset_provenance.json` preferred, cross-checked against
`resolved_config.yaml`, `run_metadata.json`, and `dataset_fingerprints.json`).
The live `data/processed` Registry is never consulted to label historical runs.
Conflicts surface as an explicit diagnostic rather than a silent merge.

### Descriptive compare vs official Paired Comparison

The Results comparison table may show metrics from two different datasets, but
it always displays left/right Dataset IDs and the fingerprint match flag. When
datasets differ, the UI labels the view as descriptive only and keeps
**Prepare Paired Comparison action** disabled. Official Paired Comparison still
requires completed runs plus the existing authoritative compatibility gate
(CLI remains the final authority). Saved comparison Inspect views show baseline
and candidate dataset versions from their reference files without changing the
paired-comparison artifact schema.

## MLflow integration

The Dashboard opens `mlflow_url`. The registry can validate and synchronize the
existing optional local metadata index through `scripts/sync_mlflow.py`. The UI does
not start an MLflow server because the approved CLI has no start subcommand. Use the
documented launch command in [MLflow Local Index](mlflow-local-index.md) when needed.
Filesystem artifacts remain authoritative; MLflow remains an optional searchable
mirror.

## Windows behavior and limitations

Windows jobs start with `CREATE_NEW_PROCESS_GROUP`; POSIX jobs start in a new session.
Stopping targets the process group. Windows first sends `CTRL_BREAK_EVENT` to a
process started by the current application and uses a minimal-environment `taskkill`
argv fallback for a persisted live PID after restart, only after identity
verification. POSIX sends `SIGTERM` to the process group. Stopping is best-effort and
never recursively deletes either job or experiment artifacts.

Windows-specific limits:

- Process group and session identity fields are unavailable and stored as null;
  verification relies on PID, creation time, executable path, and argv fingerprint.
- Junction and reparse-point rejection depends on filesystem attribute inspection.
- Hard-link publication for config copies uses an exclusive create-then-replace claim
  on Windows; POSIX uses create-if-absent hard links.
- Some restricted environments may deny junction creation in tests; those cases are
  skipped rather than weakening path safety.

## Limitations

This MVP is local and single-user. It has no authentication, server API, database,
cloud deployment, WebSockets, custom components, automatic retry, authoritative
comparison calculation, artifact mutation, or canonical-config editing. Artifact
rendering is intentionally bounded. Reader dot paths that are absent render as empty
values rather than becoming project-specific Python logic.
