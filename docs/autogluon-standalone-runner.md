# AutoGluon standalone runner

This utility runs train-only AutoGluon benchmarks outside Jupyter. It is operationally
separate from the authoritative research-v2 evaluation path. It does not establish
research validity, compare Kaggle scores, read competition test data, create
predictions, or produce submissions.

Use only the isolated `.venv-autogluon` environment from the repository root:

```powershell
.venv-autogluon\Scripts\python.exe -u scripts\run_autogluon.py validate `
  --config configs\autogluon\realtabpfn_only.yaml

.venv-autogluon\Scripts\python.exe -u scripts\run_autogluon.py train `
  --config configs\autogluon\realtabpfn_only.yaml `
  --run-id autogluon-v3-realtabpfn-20260727

.venv-autogluon\Scripts\python.exe -u scripts\run_autogluon.py inspect `
  --run-dir artifacts\autogluon_runs\autogluon-v3-realtabpfn-20260727
```

## Configuration and dataset isolation

Configuration schema version 1 is exact and rejects missing keys, unknown keys, and
wrong types. `seed` is a non-negative exact integer; booleans, floats, and strings are
rejected. The seed contributes to both configuration and profile identity.

The production data contract is fixed to one processed training pair:

- `dataset.directory` must equal exactly `data/processed/<dataset.version>`.
- `dataset.train_features_file` must equal exactly `X_train.parquet`.
- `dataset.train_target_file` must equal exactly `y_train.parquet`.
- The version is a lowercase safe slug, and the version directory and both files must
  resolve inside the repository, `data/processed`, and that exact version directory.
- Absolute, rooted, UNC, drive-qualified, drive-relative, traversal, symlink, and
  junction escapes are rejected.

The worker's production loader reads exactly those two validated paths. There are no
configurable test, raw, reference, historical-prediction, Kaggle, or submission input
paths. The only supported fit preset is `extreme_quality`.

`validate` checks this contract, profile compatibility, and resource constraints. It
allocates no run directory and does not import AutoGluon. A config-only loader mode is
used by tests and tooling when ignored training files are absent; the CLI `validate`
command requires the two training files.

## Profiles, seed, and resources

Profiles resolve the installed AutoGluon 1.5.0 registry in the worker. Before any data
load or fit, the worker durably writes `profile_resolution.json`, including the
requested seed, profile hash, resolved family/configuration list, and effective
per-family resources. Each configuration receives `model_random_seed=<seed>`,
`vary_seed_across_folds=false`, and `fold_fitting_strategy=sequential_local`.

GPU allocation is never passed as a top-level `fit()` argument for these mixed-family
profiles:

- `GBM` and `GBM_PREP` receive `ag_args_fit.num_gpus=0`.
- `CAT` receives `task_type="CPU"` and `ag_args_fit.num_gpus=0`.
- `REALTABPFN-V2` and `TABM` may receive one GPU through explicit per-configuration
  overrides when the profile GPU budget permits it.
- A family without a static resource policy is rejected before fit.

The registered profiles are:

- `realtabpfn_only_v1`: all `REALTABPFN-V2` configurations from the
  `zeroshot_2025_12_18_gpu` portfolio.
- `catboost_only_cpu_v1`: the CPU portfolio `CAT` family, forced to CPU.
- `lightgbmprep_only_cpu_v1`: the CPU portfolio `GBM_PREP` family, forced to CPU.
- `extreme_seqmem_v1`: the GPU portfolio excluding `TABDPT`, `TABICL`, and `MITRA`,
  with explicit CPU/GPU family scoping and sequential model/fold fitting. It is a
  diagnostic profile, not a recommended default.

The supervisor persists `requested_seed` and the seed-bearing profile identity before
launch, so they survive an early child crash. The requested seed is applied to every
configured core/base-family model. After fit, effective verification is scoped to
those configured families using AutoGluon 1.5.0 public `model_type` and bagged
`child_model_type` metadata. A true configured base-model mismatch fails the worker.
If supported public metadata does not expose a base-model seed, inspection records
`unavailable` and does not claim verification.

Auxiliary/meta models are reported separately. In particular, weighted ensembles may
expose their own independent internal/default `model_random_seed`; a differing
auxiliary seed does not invalidate configured base-model agreement. AutoGluon 1.5.0
normally provides model-type metadata for this classification. A documented
`WeightedEnsemble*` name fallback is used only when those public type fields are absent.

## Supervision and completion

`train` refuses an existing run directory. It launches a child using an argument list
and `shell=False`. The supervisor alone owns `_SUCCESS` and `_FAILED`; the worker never
writes terminal markers.

Worker stdout and stderr are drained by joined non-daemon threads. Read, decode, write,
tee, drain, flush, or fsync failures invalidate the run and are recorded in status and
metadata. Both worker logs and the supervisor log are flushed and `os.fsync()`-durable
before inventory and terminal-marker creation. The raw child exit code and its unsigned
32-bit Windows representation are preserved independently.

A zero exit is necessary but insufficient. `worker_result.json` is atomically written
and durably flushed by the worker, then validated by the supervisor using an exact
version-1 schema with these keys:

- `schema_version`, `status`, `worker_pid`, `started_at_utc`, `completed_at_utc`,
  `duration_seconds`;
- `profile_id`, `profile_sha256`, `dataset_version`, `predictor_relative_path`;
- `resolved_families`, `model_names`, `best_model`, `decision_threshold`;
- `autogluon_version`, `python_version`, `requested_seed`, `effective_seed`.

Missing or unknown fields, wrong exact types (including bool-as-int), non-finite values,
identity/PID/path/seed mismatches, inconsistent timing, and disagreement with exported
model names, best model, or threshold all fail completion. Success additionally
requires the predictor core files, inspection artifacts, metadata, status, and artifact
inventory. All ordinary artifacts and validation finish before the marker; `_SUCCESS`
is the final fallible filesystem operation. Any failure writes only `_FAILED`.

There is no automatic resume or overwrite. Partial artifacts are retained for diagnosis
but are not promoted to a new run and are not guaranteed to form a loadable predictor.

## Read-only inspection

`inspect` is read-only and idempotent. It classifies a run as `complete` only when:

- `_SUCCESS` exists and `_FAILED` does not;
- execution status says `completed` and agrees with run metadata;
- the exact worker completion payload validates;
- required predictor and inspection files exist;
- configuration/profile/dataset/seed identities agree;
- the artifact inventory matches the filesystem;
- the terminal marker is internally consistent.

A success marker by itself is never sufficient. Contradictions and malformed or
mismatched artifacts are `corrupt`; absent terminal state or partial failure artifacts
are `incomplete`, with explicit reason codes. Only `complete` runs load by default.
Use `--attempt-load` for an explicit best-effort attempt on an incomplete or corrupt
run, or `--no-load` for filesystem-only inspection. Failure to load a partial predictor
is expected and reported without changing the run.

Portable predictor inspection removes local absolute paths. Unavoidable local values
are isolated under `local_operational_nonportable`; the portable summary does not embed
usernames or absolute operational paths.

Exit codes are `0` for command success, `2` for invalid input/configuration, `3` for an
existing run directory, `4` for supervised worker/run failure, and `5` when an explicitly
attempted inspection load fails.
