# Optuna Search v1

## Scope and statistical boundary

Optuna Search v1 is a train-only tuning stage for
`xgboost_numeric_v1` and `catboost_numeric_v1`. A search score is not an
unbiased final research result and is not a deployment decision.

The required sequence is:

```text
Optuna search plan
  -> exported fixed candidate config
  -> untouched Experiment Core v2 selection plan
  -> untouched confirmation plan
  -> Paired Comparison
  -> manual approval
```

The search loader uses the existing Experiment Core v2 train-only dataset
contract. It reads only the configured training features, training target, and
dataset metadata. It does not read competition `X_test`, sample-submission,
Kaggle-history, P2/P3, final-fit, deployment, or submission assets. It never
creates a submission and does not invoke selection, confirmation, Paired
Comparison, or MLflow.

## Cross-fitted threshold objective

Each trial uses deterministic repeated stratified folds from the dedicated
search plan. For each repeat and fold, the selected production adapter:

1. fits only the fold's training rows;
2. receives no `eval_set`;
3. uses no early stopping;
4. predicts class-1 probabilities only for the fold's validation rows.

The complete repeat forms one aligned OOF probability vector. To score a held-out
fold, its threshold is selected from the OOF targets and probabilities belonging
only to the other folds in that repeat. The held-out fold is explicitly absent
from threshold-selection membership. The production
`grid_balanced_accuracy_v1` policy, grid, tolerance, fallback, and tie break are
used unchanged. Labels use the exact `probability >= threshold` convention.

The trial objective is the arithmetic mean of fold Balanced Accuracy values,
first summarized per repeat and then across repeats. Because every repeat has the
same fold count, this is also the deterministic mean across all repeat/fold
scores. Fold assignments, threshold-source membership, selected thresholds,
fold metrics, repeat metrics, OOF predictions, aggregate objective, and
prediction-key coverage are persisted.

## Search-plan and search-space identities

Search plans under `configs/optuna/` have an exact schema and contain:

- dataset, pipeline, adapter, and fixed Experiment Core v2 candidate references;
- repeated-fold assignment settings;
- the production threshold policy and grid;
- fixed Balanced Accuracy/maximize semantics;
- bounded trial and timeout controls;
- a seeded `stateless_random_v1` sampler and `nop` pruner;
- a deterministic study name and repository-contained SQLite/report paths.

The sampler derives every parameter draw from the sampler seed, study name,
trial number, and parameter name. It has no mutable RNG position, so an
interrupted/resumed study produces the same draw for the same trial number.
This is the documented deterministic alternative to TPE for v1.

The study identity excludes operational storage paths, timeout, and the requested
trial target. This permits an explicit `n_trials` increase to resume the same
study. The completed-search identity includes `n_trials` and timeout, so the
larger completed report receives a distinct immutable search ID.

Adapter-specific spaces under `configs/optuna/search_spaces/` allow only the
approved tunable parameters. They reject unknown keys, numeric strings,
bool-as-int, non-finite values, invalid ranges/distributions, adapter mismatch,
and searches over fixed fields. XGBoost `reg_alpha` has an explicit zero mass
plus a positive log distribution.

The base production adapter remains authoritative for all fixed values:

- CPU device/task type and one thread;
- objective/loss and evaluation metric;
- XGBoost `hist`, `gbtree`, and portable `missing: IEEE_NaN`;
- CatBoost `allow_writing_files: false`, silent output, `nan_mode: Min`,
  `Bayesian` bootstrap, and `SymmetricTree` growth;
- fixed model seed policy;
- native-NaN numeric feature handling.

GPU, evaluation-set, early-stopping, filesystem-writing, and unsupported model
parameters cannot enter a v1 search space.

### Exact XGBoost search space

The following machine-readable table is normative and must match
`configs/optuna/search_spaces/xgboost_numeric_v1.yaml` exactly.

<!-- OPTUNA_SEARCH_SPACE:xgboost_numeric_v1 -->
```yaml
n_estimators: {distribution: int, low: 100, high: 600, step: 50, log: false}
learning_rate: {distribution: float, low: 0.01, high: 0.2, log: true}
max_depth: {distribution: int, low: 3, high: 10, step: 1, log: false}
min_child_weight: {distribution: float, low: 0.1, high: 20.0, log: true}
subsample: {distribution: float, low: 0.6, high: 1.0, log: false}
colsample_bytree: {distribution: float, low: 0.6, high: 1.0, log: false}
gamma: {distribution: float, low: 0.0, high: 5.0, log: false}
reg_alpha:
  distribution: zero_or_log_float
  zero_probability: 0.2
  low_positive: 1.0e-8
  high: 10.0
reg_lambda: {distribution: float, low: 1.0e-4, high: 100.0, log: true}
max_bin: {distribution: int, low: 128, high: 512, step: 64, log: false}
```
<!-- /OPTUNA_SEARCH_SPACE:xgboost_numeric_v1 -->

Integer fields use inclusive stepped ranges. Float fields marked `log: true`
use log-uniform sampling; the other float fields are linear. `reg_alpha` first
draws an explicit zero with probability `0.2`; its nonzero branch is log-uniform
on `[1e-8, 10.0]`.

### Exact CatBoost search space

The following machine-readable table is normative and must match
`configs/optuna/search_spaces/catboost_numeric_v1.yaml` exactly.

<!-- OPTUNA_SEARCH_SPACE:catboost_numeric_v1 -->
```yaml
iterations: {distribution: int, low: 100, high: 600, step: 50, log: false}
learning_rate: {distribution: float, low: 0.01, high: 0.2, log: true}
depth: {distribution: int, low: 4, high: 10, step: 1, log: false}
l2_leaf_reg: {distribution: float, low: 0.1, high: 50.0, log: true}
random_strength: {distribution: float, low: 0.0, high: 5.0, log: false}
bagging_temperature: {distribution: float, low: 0.0, high: 5.0, log: false}
border_count: {distribution: int, low: 64, high: 256, step: 32, log: false}
```
<!-- /OPTUNA_SEARCH_SPACE:catboost_numeric_v1 -->

Integer fields use inclusive stepped ranges. `learning_rate` and `l2_leaf_reg`
are log-uniform. `random_strength` and `bagging_temperature` are linear and may
be exactly zero.

### Fixed adapter-controlled fields

These fields are deliberately outside the search-space YAML. They preserve the
approved production adapters' probability, reproducibility, resource, and
side-effect contracts.

| Adapter | Fixed fields | Reason |
|---|---|---|
| XGBoost | `objective=binary:logistic`, `booster=gbtree`, `missing=IEEE_NaN`, `tree_method=hist`, `device=cpu`, `random_state=42`, `n_jobs=1`, `eval_metric=logloss`, `verbosity=0` | Preserve binary class-1 probabilities, native missing values, deterministic CPU execution, one thread, and silent fitting without an evaluation set. |
| CatBoost | `random_seed=42`, `thread_count=1`, `bootstrap_type=Bayesian`, `grow_policy=SymmetricTree`, `loss_function=Logloss`, `eval_metric=Logloss`, `nan_mode=Min`, `task_type=CPU`, `allow_writing_files=false`, `verbose=false` | Preserve deterministic CPU execution, one thread, native missing handling, the approved tree/bootstrap policy, and a no-filesystem-write fit. |
| Both | numeric fold-local target-encoding contract; positive-class label `1`; no `eval_set`; no early stopping | These are Experiment Core v2 leakage and probability contracts, not model-selection dimensions. |

## Resume authentication identity

Before any trial is allocated, v1 builds a versioned resume-authentication
identity. Its source closure contains every `optuna_search_*.py` module, the CLI
entry point, Experiment Core v2 config/resolved-config/schema/data modules,
numeric adapters and registry, threshold/metrics code, assignment logic, and
artifact/export code. Every entry uses a canonical repository-relative POSIX
path plus the size and SHA-256 of the same bytes. Symlinks, junctions, reparse
points, missing files, and non-regular files are rejected.

The runtime component records exact versions for Python, Optuna, NumPy, pandas,
scikit-learn, PyArrow, PyYAML, and the selected adapter package (XGBoost or
CatBoost). The complete identity is included in the deterministic study
identity, persisted in study attributes and the immutable report, and recomputed
before every resume. Missing legacy identity or any source/runtime mismatch is
rejected before another trial can be allocated.

## External lifecycle authority

Every new study is externally authenticated from creation, including studies in
which every trial succeeds. Runtime lifecycle operations use exactly one key
provider: `CHURN_ML_OPTUNA_LIFECYCLE_AUTHORITY_KEY_FILE`. The environment value
must be an absolute, traversal-free path to an exact regular binary file outside
the repository, artifact/report roots, and study storage roots. Symlinks,
junctions, reparse points, directories, multiply linked files, and files shorter
than 32 bytes are rejected. Bytes are read without text normalization. The key
and its path are never written or printed; reports and SQLite retain only the
authority schema version, signed payload identities, HMAC-SHA256 signatures, and
the stable key fingerprint
`SHA256("churn_ml.optuna.lifecycle.key-identifier.v1\0" || key_bytes)`.
Configuration and CLI arguments cannot select alternate report-controlled keys.

`authority-init --output <external-path>` creates 32 random bytes with exclusive
temporary creation, restrictive permissions where supported, and create-if-absent
publication (hard-link when available). It refuses overwrite, never deletes a
file it did not create, and creates a missing parent only with `--create-parent`.
Failures use stable path-independent codes such as
`AUTHORITY_KEY_INIT_INVALID_DESTINATION` and `AUTHORITY_KEY_INIT_ALREADY_EXISTS`.
JSON success and failure output is path-independent and contains no secret
material.

The signed creation statement binds authority schema 1, key fingerprint, a new
immutable study UUID, the base study/search identity, the authority-bound study
identity, dataset and assignment identities, source-closure and runtime
identities, sampler/pruner identity, creation-time configured trial count, and a
canonical microsecond UTC creation timestamp. The authority-bound identity is a
domain-separated SHA-256 over the base identity, authority schema, and key
fingerprint; unrelated Experiment Core v2 identities are unchanged.

Lifecycle event schema 1 is canonical JSON (sorted keys and fixed separators)
and contains exact primitive fields:
`authority_schema_version`, `event_schema_version`, `event_type`, `study_uuid`,
`base_search_identity_sha256`, `event_sequence`,
`previous_event_signature_sha256`, `trial_number`, `from_state`, `to_state`,
`failure_stage`, `failure_reason_code`, `exception_type`,
`failure_message_sha256`, `started_at_utc`, `event_at_utc`,
`interrupted_recovery`, target-extension bindings
(`configured_trial_target_before` / `after`, `previous_epoch_number`,
`opened_epoch_number`), and independently recomputed
`event_payload_identity_sha256`. HMAC uses the explicit
`churn_ml.optuna.lifecycle.event.v1` domain. Events cover study creation, trial
allocation/start/completion/execution failure, interrupted RUNNING/ALLOCATED
recovery, study completion, report finalization, and authenticated
`study_target_extended` epoch openings. Sequences are exact and monotonic;
each link is SHA-256 of the preceding HMAC signature. Legal transitions are
reconstructed, and interruption recovery requires an earlier signed RUNNING or
ALLOCATED state.

Finalization epochs are immutable authenticated ledgers. Epoch 0 opens at study
creation. Closing an epoch signs an epoch ledger over the exact event prefix,
trial universes, report identities, and completion time. Finalized studies may
later raise `n_trials` only by appending a signed `study_target_extended` event
and opening epoch N+1 chained to the previous epoch ledger signature. Prior
completed filesystem reports remain valid against their exact epoch even when
SQLite later contains additional epochs or a signed open suffix. At most one
open epoch may exist, and only as the latest entry, during an interrupted
extension. The singleton `lifecycle_authority_final_ledger` schema fails closed
with no silent migration. Unsigned legacy reports/studies still fail closed.

Terminal signing is deliberately non-circular:

1. write report payloads and pre-terminal signed events;
2. write inventory and manifest over those payloads;
3. append and persist the signed `report_finalized` event;
4. sign the current epoch ledger over the manifest and epoch event prefix;
5. write `_SUCCESS` last, binding that epoch ledger payload identity, HMAC, and
   epoch number;
6. persist the ordered epoch history in SQLite.

One shared validator is used before resume/recovery/allocation and for completed
loading, inspect, export, and finalization. It verifies the external key
fingerprint, every canonical payload identity/HMAC/link/transition, the report's
exact epoch ledger, exact report state/evidence, and SQLite/report equality for
that historical epoch when the database is present. The filesystem HMAC remains
independently verifiable when operational SQLite has been removed.

Threat model: this protects against coherent mutation and rehashing of report
fields, exception message/type substitution, execution-failure versus
interruption-recovery substitution, lifecycle event deletion/insertion/
reordering/replacement, removal of all failed evidence, epoch history
mutation/reordering, moving an old report onto a newer epoch, and rebuilt
inventory, manifest, identities, and `_SUCCESS`. It does not protect compromise
or replacement of the external key, compromise of the host process while the key
is loaded, or malicious code executing before signing.

## Completed-report semantic reconstruction

Completed loading does not trust a coherent inventory/manifest alone. After
non-following tree safety and byte authentication, it validates exact schemas
and independently reconstructs the authorized target row universe, repeated
stratified assignments, cross-fitted threshold membership, every completed
trial's Cartesian prediction keys, thresholds, labels, confusion counts, fold
Balanced Accuracy, repeat aggregates, final objective, lifecycle counts,
deterministic lowest-trial tie break, winning parameters, and exported candidate
linkage. `_SUCCESS` is validated last and must remain the newest file. `inspect`
and `export-best` use this complete loader and reject semantically invalid
reports even when inventory, manifest, and `_SUCCESS` were coherently rebuilt.

## Study and artifact lifecycle

Optuna `4.9.0` is already declared in `pyproject.toml` and locked in `uv.lock`.
Imports remain lazy: normal Experiment Core workflows do not import Optuna.

SQLite under `artifacts/optuna/optuna.db` is resumable operational state only.
`load_if_exists` is explicit. Existing study identity must match exactly.
Orphaned `RUNNING` trials are recovered as failed with
`INTERRUPTED_PROCESS_RECOVERY`; their trial numbers are not reused. A completed
target is not silently extended. More trials are added only when a configuration
explicitly raises `n_trials` through an authenticated finalization epoch;
lowering a previous target is refused. Prior epoch reports remain valid.

Completed reports under `artifacts/optuna_searches/<search-id>/` are the
authoritative immutable record. They contain the resolved configuration,
identities, assignments, search space, study/trial summaries, fold/repeat
metrics, OOF predictions, fixed best candidate, runtime/environment/source
provenance, recursive SHA-256 inventory, manifest, and final `_SUCCESS`. No
models, raw datasets, competition-test data, or submissions are stored.
Ordinary report files are atomically replaced, report bytes and source
provenance are checked before success, existing outputs are refused, and
`_SUCCESS` is written last. Failed attempts receive `_FAILED` separately and do
not overwrite a completed report.

Side-effect-free result loading validates the immutable tree and provides a
future boundary for optional MLflow indexing. The optimization engine does not
import or write to MLflow; the filesystem remains authoritative.

## CLI

All relative paths resolve from the repository root.

```powershell
.venv\Scripts\python.exe -u scripts\run_optuna_search.py authority-init `
  --output C:\external-secure\optuna-lifecycle-authority.key
.venv\Scripts\python.exe -u scripts\run_optuna_search.py validate `
  --config configs/optuna/xgboost_numeric_v1_smoke.yaml

.venv\Scripts\python.exe -u scripts\run_optuna_search.py run `
  --config configs/optuna/xgboost_numeric_v1_smoke.yaml

.venv\Scripts\python.exe -u scripts\run_optuna_search.py inspect `
  --search-dir artifacts/optuna_searches/<search-id>

.venv\Scripts\python.exe -u scripts\run_optuna_search.py export-best `
  --search-dir artifacts/optuna_searches/<search-id> `
  --output configs/research_v2/candidates/<candidate>.yaml
```

Every command emits a single machine-readable JSON summary. `validate` creates
nothing, `inspect` is read-only, and `export-best` refuses overwrite.

Exit codes are:

- `0`: success;
- `1`: unexpected failure;
- `2`: configuration/contract failure;
- `3`: immutable artifact failure;
- `4`: study lifecycle failure;
- `5`: export failure.

The exported YAML contains fixed primitive model parameters and strict
top-level search provenance. It contains no Optuna distributions or database
path. Its evidence scope is explicitly tuning-only, and it must pass the
production Experiment Core v2 validator. Export does not run selection,
confirmation, Paired Comparison, final fit, or submission logic.
