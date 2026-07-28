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

## Installation and launch

From the repository root:

```powershell
uv run --extra ui streamlit run apps/experiment_control_panel.py
```

Streamlit is isolated in the `ui` optional dependency extra. The application uses the
main project environment, not `.venv-autogluon`.

## Pages

- **Dashboard** shows job-state counts, recent jobs, recent configured artifacts,
  command groups, and the configured MLflow link.
- **Run** selects an allowlisted operation and action, reads an allowed YAML or JSON
  configuration, validates syntax, displays exact redacted argv, and starts one
  background job after the required confirmation.
- **Jobs** refreshes persisted job state, shows PID, timestamps, elapsed time, exit
  code, exact redacted argv, and bounded stdout/stderr tails. A running process can be
  stopped only after explicit confirmation when stopping is enabled.
- **Results** discovers artifacts only through reader definitions, renders configured
  summary fields and JSON trees, previews bounded CSV data, tails configured logs, and
  shows a display-only side-by-side field table. Official comparison remains the
  Paired Comparison CLI action.
- **Configuration** reports strict registry validation, source paths, and loaded
  settings/command/reader summaries. The three UI registry files are read-only in v1.

The UI uses manual refresh. It does not use aggressive automatic reruns and never
retries a command automatically.

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
input/output roots, optional result reader, and optional URL. Each action declares:

- a stable ID and description;
- a fixed argv token list;
- typed whole-token placeholders (`path`, `string`, `integer`, or `enum`);
- placeholder roles and path roots;
- confirmation and competition-test requirements;
- enabled/disabled state;
- stdout success/failure markers.

Every executable prefix begins with the registry sentinel `$PYTHON`, which resolves
only to the running UI interpreter. Widgets never accept executable or arbitrary
command text. A new supported CLI action can therefore be added without Python changes
by adding a strict action entry and its typed placeholders.

The initial registry is derived from actual public `--help` output at the approved
base:

- Experiment Core v2: validate and run;
- legacy train-only research: validate and run;
- Paired Comparison v1: validate and run;
- MLflow local index: validate, dry-run source validation, and sync;
- Final Deployment v1: validate, synthetic dry-run, inspect, and a real run that is
  disabled by default.

This base has no public Optuna lifecycle CLI, MLflow start command, Paired Comparison
inspect/export command, or standalone result export command. The UI does not guess
those interfaces. MLflow is opened through its configured URL.

### `ui_readers.yaml`

Schema version 1 declares artifact roots, discovery globs, terminal markers, summary
JSON files, labeled dot-path fields, bounded CSV preview files, log files, and
side-by-side compare fields. Python contains no model-specific result parser.

Dot paths traverse mapping keys and numeric list indexes only. They do not execute
expressions.

## Configuration copy editing

Canonical configurations are displayed read-only. The Advanced editor validates a
primitive-only YAML or JSON mapping and writes only a new, safely named file beneath
`artifacts/ui_configs`. Existing copies are not overwritten. Registry and settings
files cannot be edited in the application.

## Jobs and logs

Each job receives a UUID directory:

```text
artifacts/ui_jobs/<job-id>/
├── job.json
├── command.json
├── status.json
├── stdout.log
└── stderr.log
```

JSON writes use a temporary sibling, flush, filesystem sync, and atomic replace.
stdout/stderr are append-only. Persisted states are `created`, `running`, `succeeded`,
`failed`, `stop_requested`, `stopped`, and `orphaned`. After an application restart, a
still-live PID remains running; a missing process with no recoverable exit code becomes
orphaned. There is no database, recursive deletion, artifact cleanup, or automatic
retry.

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
- Environment variables and secrets are neither collected nor persisted.

## MLflow integration

The Dashboard opens `mlflow_url`. The registry can validate and synchronize the
existing optional local metadata index through `scripts/sync_mlflow.py`. The UI does
not start an MLflow server because the approved CLI has no start subcommand. Use the
documented launch command in [MLflow Local Index](mlflow-local-index.md) when needed.

## Windows behavior

Windows jobs start with `CREATE_NEW_PROCESS_GROUP`; POSIX jobs start in a new session.
Stopping targets the process group. Windows first sends `CTRL_BREAK_EVENT` to a process
started by the current application and uses the platform `taskkill` argv fallback for
a persisted live PID after restart. POSIX sends `SIGTERM` to the process group.
Stopping is best-effort and never recursively deletes either job or experiment
artifacts.

## Limitations

This MVP is local and single-user. It has no authentication, server API, database,
cloud deployment, WebSockets, custom components, automatic retry, authoritative
comparison calculation, artifact mutation, or canonical-config editing. Artifact
rendering is intentionally bounded. Reader dot paths that are absent render as empty
values rather than becoming project-specific Python logic.
