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

## Study and artifact lifecycle

Optuna `4.9.0` is already declared in `pyproject.toml` and locked in `uv.lock`.
Imports remain lazy: normal Experiment Core workflows do not import Optuna.

SQLite under `artifacts/optuna/optuna.db` is resumable operational state only.
`load_if_exists` is explicit. Existing study identity must match exactly.
Orphaned `RUNNING` trials are recovered as failed with
`INTERRUPTED_PROCESS_RECOVERY`; their trial numbers are not reused. A completed
target is not silently extended. More trials are added only when a configuration
explicitly raises `n_trials`; lowering a previous target is refused.

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
