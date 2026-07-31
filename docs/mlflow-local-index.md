# MLflow local index

## Authority and scope

The filesystem artifact contracts remain authoritative. MLflow is an optional,
repository-local searchable metadata mirror over validated terminal runs. It does not
participate in training, experiment identity generation, artifact finalization,
threshold selection, prediction generation, model selection, or submission behavior.
Corrupt sources are rejected; indexing them as `FAILED` is not a substitute for a valid
failed-run lifecycle.

The index supports Experiment Core v2 research runs under
`artifacts/research_v2` and standalone AutoGluon runs under
`artifacts/autogluon_runs`. Paired comparisons are not implemented. The typed
`SourceAdapter` protocol and registry remain the extension point for a future source.
The sync never loads an AutoGluon predictor, model binary, dataset, competition test
asset, sample submission, Kaggle asset, or submission.

## Local configuration and storage

The strict config-schema-version-1 file is `configs/mlflow/local.yaml`. It resolves
three ignored, repository-local locations:

- SQLite metadata: `artifacts/mlflow/mlflow.db`;
- copied MLflow artifacts: `artifacts/mlflow/mlartifacts`;
- deterministic receipts: `artifacts/mlflow/sync_receipts`.

Validation uses exact recursive keys and exact primitive types. Paths reject POSIX and
Windows roots, drive-relative paths, UNC paths, backslashes, traversal, and resolved
symlink/junction escapes. The database, artifact root, and receipt root must remain
inside `artifacts/` and must be pairwise non-overlapping in both directions. Equality,
any location containing another, and resolved containment through a reparse path are
rejected; sibling locations are allowed.

MLflow is imported lazily. Config validation, dry-run, and unrelated imports do not
allocate the SQLite database, artifact root, experiments, runs, or receipts.

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

Use `--run-dir <path>` with one explicit source type. Use `--fail-fast` to stop
following the first rejected, recoverable, or errored item. Batch mode otherwise
continues across independent sources and returns exact per-item outcomes and counts.
A `recoverable` result identifies a partial MLflow write that retains the deterministic
run for a safe retry; it is an error exit, not success or `unchanged`.

Launch the local UI with:

```powershell
.venv\Scripts\python.exe -m mlflow ui `
  --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db `
  --host 127.0.0.1 `
  --port 5000
```

The configured experiments are `neoversity-churn-research-v2` and
`neoversity-churn-autogluon`.

## Exact source validation

Discovery first confines every run to its configured source root and rejects symlinks,
junctions, reparse points, and path escapes. A terminal status without its required
marker is corrupt rather than merely running. A source with no terminal status is
skipped.

Completed research runs are accepted only after the production
`validate_research_v2_run()` validator recomputes and verifies the exact identities,
ordinary inventory, assignments, predictions, metrics, thresholds, progress,
manifest, and `_SUCCESS` contract.

Failed research runs use a separate exact validator. It requires `_FAILED`, rejects
`_SUCCESS` and a stale success manifest, parses ordered UTC timestamps, validates
exact status and full-or-native-early metadata schemas, bounded fold progress,
nonempty typed failure details, plan/pipeline/adapter/run agreement, hash shapes, and
resolved-plan fold counts. An early failure is not accepted until every authoritative
artifact that is present has been classified. A persisted resolved config must satisfy
the strict Experiment Core v2 key, primitive-type, registered pipeline/adapter,
persistence, tracking, and repository-contained-path contracts. Its referenced
evaluation-plan file must be a regular non-reparse repository file, must pass the
production strict plan loader, and must be recursively identical by both primitive type
and value to the persisted plan. This authenticates all behavior-critical plan fields,
including dataset, folds, seeds, threshold policy/grid, metrics, aggregation, and label
semantics.

Present identity, environment, runtime, and Git artifacts must satisfy their exact
schemas, lifecycle timestamps, and every identity field available from status,
metadata, and config. The completed-run Experiment Core v2 source identity remains the
legacy schema-version-2 payload with exact `path` and `sha256` fields; MLflow does not
mutate that portable identity or its pinned hashes.

An early failure that persists the optional `identities/source.json` artifact must also
persist `identities/mlflow_source_authentication.json`. Its MLflow-owned schema-version-4
canonical manifest has exact top-level keys `schema_version`, `hashing_method`, and
`files`. Every file entry has exactly:

- `path`: a canonical repository-relative POSIX path;
- `size_bytes`: the exact nonnegative integer byte length (booleans, strings, and
  negative values are invalid);
- `sha256`: the exact lowercase SHA-256 of those same bytes.

The validator checks containment, traversal and reparse escapes, and regular-file type,
then reads current repository bytes and recomputes both exact size and digest. Both must
match. It independently rebuilds the complete sorted schema-v4 manifest and rejects
missing, extra, duplicate, or reordered entries and unknown recursive fields. The
schema-v4 path/digest projection must also exactly match the legacy source identity.
A missing or incorrect size makes the optional artifact corrupt before mapping,
artifact selection or copying, source-key/receipt construction, or MLflow client/run
allocation.

Schema v4 is an intentional migration boundary for optional early-failure repository
authentication: an earlier early-failure source identity without the companion
size-bearing manifest is no longer indexable and must be regenerated. Native early
failures with no optional source identity remain valid. Completed Experiment Core v2
runs continue through their existing production validator and are not invalidated by
this MLflow-only early-failure schema. The referenced repository config is loaded
through the production v2 loader and compared with the persisted resolved config.
Unsupported inventory/manifest-like or other partial artifacts make an early source
corrupt. Reparse paths and forbidden competition, submission, model, pickle, or
AutoGluon content are rejected.

Only the validated result returned by that lifecycle check can supply mapping fields,
immutable source identity, or additions to the failed-run artifact-copy allowlist. A
malformed or contradictory optional artifact is rejected before an MLflow client or
run is created, and cannot be copied or influence mapped parameters.

Completed AutoGluon runs use the standalone read-only inspection classification with
`attempt_load=False` and must classify exactly as `complete`. The sync never calls a
predictor loader.

Failed AutoGluon runs use an exact MLflow-owned validator derived from the standalone
runner contract; reason-code substring matching is not used. It requires exact status,
metadata, and `_FAILED` marker keys; absence of `_SUCCESS`; run/config/profile/dataset/
seed agreement; the production profile hash; exact resolved config and resource
agreement; signed and unsigned exit-code consistency; PID fields; parsed ordered UTC
timestamps; finite nonnegative duration; nonempty unique failure codes and joined
reason; last stage; predictor-loading flags; exact missing-artifact state; exact
worker-completion and log-pump fields; local operational metadata shape; discovered
model-directory diagnostics; and an exact inventory whose regular-file sizes and
SHA-256 values match the current tree. Symlink, junction, reparse, stale-success,
unknown inventory, and inconsistent early-failure states are rejected. Valid failures
before predictor creation and native process crashes remain indexable as `FAILED`.

The terminal mapping is therefore:

- validated completed source → MLflow `FINISHED`;
- exact, internally consistent terminal failure → MLflow `FAILED`;
- running/incomplete source → skipped;
- corrupt terminal source → rejected and not committed.

## Scoped identity and mutation policy

Sync schema version 3 derives the portable source key from:

- sync/source-key schema version;
- source type;
- the complete canonical source-run path relative to `repository_root`;
- leaf run ID.

The adapters derive that path from the resolved run directory and the resolved
repository root, not merely from the configured research or AutoGluon source root.
For example, keys retain `artifacts/research_v2/plan/candidate/run-id` or
`artifacts/autogluon_runs/run-id`. Backslashes normalize to `/`; absolute, rooted,
drive, empty, dot, traversal, and repository-escape paths are rejected. Path case is
preserved. Thus identical internal paths under two differently configured source roots
have different keys, while repeated sync of the same full repository-relative path
has the same key. Absolute local paths are excluded from the portable key payload.
Research and AutoGluon use this same scoping rule.

The separately computed immutable source identity covers the verified terminal source
(manifest/inventory or validated failed terminal identity). It is deliberately checked
separately from the lookup key: a new immutable identity at the same complete path is
`source_mutation_detected`, not a second row.
`mlflow_index.local_source_path_nonportable` is explicitly operational and is not part
of portable identity.

Schema-version-1 rows used an unscoped key, and schema-version-2 rows omitted the
repository-relative source-root prefix. Neither old key is automatically adopted by
schema version 3. Because this feature has not been merged, local development should
reindex into a new/empty SQLite database; old rows may be retained only as local
historical state. Version-3 sync creates and queries only schema-version-3 keys.

## Two-phase synchronization and recovery

For every validated source, production `_sync_one()` performs this order:

1. find or create the deterministic MLflow run by scoped source key;
2. reject duplicate rows, source mutation, immutable conflicts, or unremovable extra
   state;
3. keep `mlflow_index.sync_complete` absent or `false`;
4. reconcile immutable params, expected tags, and metrics;
5. revalidate and log each allowlisted artifact;
6. verify exact params, expected tags, metric keys/values, and copied artifact
   paths/sizes/SHA-256 values;
7. set MLflow terminal status to `FINISHED` or `FAILED`;
8. read back and verify that exact terminal status;
9. write `mlflow_index.sync_complete=true` as the final MLflow mutation;
10. perform a read-only verification, then atomically write the receipt.

No exception path forces a partial run to `FAILED`. A write failure leaves the commit
marker false and returns `recoverable` with the existing MLflow run ID. Retry searches
the same source key, reconciles missing or repairable state, corrects a wrong terminal
status, and completes the same run without duplication. If state cannot be reconciled
(for example conflicting immutable params or extra undeletable artifacts), it is
rejected explicitly. `unchanged` is returned only after exact identity, immutable
params/tags, metrics, copied artifacts, terminal status, and `sync_complete=true` are
verified.

MLflow does not provide one transaction spanning all run APIs. The false/true commit
marker and deterministic reconciliation are the local crash-recovery protocol.

## Deterministic receipts

A receipt is written only after the final commit marker and read-only verification
succeed. Its stable JSON body contains sync schema version, source type, scoped source
key, source run ID, canonical relative path, immutable source identity, MLflow run ID,
and final status. It contains no current timestamp, UUID, username, or absolute path.

The final filename is the SHA-256 of those canonical receipt bytes under
`sync_receipts/<source-type>/`. The write uses a temporary sibling and atomic replace.
An identical unchanged sync therefore produces the same path and exact bytes, not a
second audit record. Different stable identity/run/status content produces a different
receipt identity. Rejected, skipped, errored, dry-run, and recoverable items do not
receive receipts.

## Metadata mapping

Research parameters include source/run identity, schema and plan identity, dataset,
pipeline, adapter, candidate/source/module hashes, repeat/fold counts, and threshold
policy. Completed metrics include the defined evaluation metrics, directions, repeat
summaries, thresholds, and duration. Failed mappings expose only validated failure
diagnostics and fields from optional artifacts that passed the failed-run validator;
raw optional files never provide fallback mapping values.

Completed research `resolved_config.yaml` payloads may include exactly one optional
top-level key beyond the Experiment Core v2 base set: `search_provenance`. That block
is validated with the production research_v2 helper (exact keys, schema version 1,
safe slugs, SHA-256 digests, and
`evidence_scope=tuning_only_not_unbiased_final_evidence`). Unknown extra keys and
malformed provenance remain rejected. Failed-run resolved-config validation uses the
same optional-key rule. Full `validate_research_v2_run()` semantic validation is
unchanged for completed sources.

### Deterministic MLflow run names

Sync always sets a stable UI name and reconciles it in place via `mlflow.runName`:

- AutoGluon: `{source_run_id}`
- research_v2: `{dataset_version}__{adapter_id}__{source_run_id}` when
  `dataset_version` is available; otherwise `{adapter_id}__{source_run_id}`

Lookup identity remains `mlflow_index.source_key`. Renaming or backfilling an
existing indexed source updates the same MLflow run in place and never allocates
a second row for the same key. After create or rename reconciliation, a following
sync is `unchanged`. Never overwrite a recorded `dataset_version` with an
assumed canonical ID. Dry-run sync reports create/resume/unchanged/rejected
without allocating storage.

### Research dataset provenance parameters and tags

Filesystem artifacts remain authoritative. Completed, validated Registry-backed
research runs may additionally map consistent provenance into searchable MLflow
parameters (omit unavailable optional values):

- retained: `dataset_version`, `dataset_identity_sha256`
- added when validated: `dataset_parent_id`, `dataset_feature_count`,
  `dataset_target_dependency`, `dataset_schema_sha256`,
  `dataset_train_content_sha256`, `dataset_target_sha256`,
  `dataset_train_row_identity_sha256`, `dataset_registry_schema_version`

Searchable tags when provenance validates:

- `dataset.id`
- `dataset.target_dependency`

`dataset_provenance.json` is included in the bounded completed-research metadata
artifact allowlist and still subject to path, size, content, and post-copy
verification. Failed-run mapping must not trust arbitrary optional provenance.
Contradictory completed provenance is omitted rather than silently merged.
AutoGluon mapping is unchanged.
### AutoGluon metrics and tags

AutoGluon parameters include config/profile/dataset/seed/resource identities and
validated status diagnostics. `predictor_classification` is a tag, not a parameter.
Model count, best model, decision threshold, and worker-completion-derived predictor
metadata are populated only when terminal status is completed, read-only classification
is `complete`, and completion validation succeeded. A failed run cannot expose stale
`worker_result.json` completion fields. A valid duration of `0.0` is preserved.

For completed AutoGluon runs, native quality comes from the already validated
`inspection/summary.json`, with `inspection/leaderboard.csv` used only to
cross-check or fall back for the exported best model. Logged values when present and
finite:

- metrics: `score_val`, `best_model_fit_time_seconds`,
  `best_model_pred_time_val_seconds`, and `duration_seconds`;
- parameter: `eval_metric`;
- tags: `best_model` and, when the metric is a known AutoGluon direction,
  `metric_direction` (`higher_is_better` or `lower_is_better`).

Only the best-model row is mapped; the full leaderboard stays an optional metadata
artifact for detailed review. Failed or incomplete AutoGluon runs keep
status/duration handling and do not receive quality metrics they did not produce.

### In-place backfill

Repeated sync locates the existing MLflow run by `mlflow_index.source_key`, adds only
missing params/metrics/tags (including `mlflow.runName`), and refuses undeletable
unexpected metrics or conflicting immutable state. No duplicate rows are created.

## Strict artifact-copy policy

Copying is optional and bounded by `sync.log_small_artifacts` and
`sync.max_artifact_size_bytes`. Only explicit metadata allowlists are eligible. Each
candidate is checked both during source preparation and immediately before logging for:

- canonical contained relative path;
- regular-file type;
- no symlink, junction, or reparse point;
- configured size cap and stable pre/post-read size;
- strict UTF-8 and no NUL bytes;
- strict JSON mapping, safe YAML mapping, parseable CSV with a nonempty unique header
  and consistent row widths, or the expected JSON/empty-failure terminal marker format;
- unchanged selected size and SHA-256 immediately before copy.

Unsupported extensions/content combinations are rejected even when the filename is on
an allowlist. The post-copy verification compares path, size, and SHA-256. The sync
never copies predictors/models, predictions, datasets, logs, competition assets,
submissions, or unlisted files. Source artifacts are never modified; only independent
MLflow storage and deterministic receipts are written.