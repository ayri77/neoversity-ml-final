# AutoGluon standalone runner

This utility runs train-only AutoGluon benchmarks outside Jupyter. It is separate from
the authoritative research-v2 evaluation path and never produces competition
predictions or submissions.

Use the isolated environment from the repository root:

```powershell
.venv-autogluon\Scripts\python.exe -u scripts\run_autogluon.py validate `
  --config configs\autogluon\realtabpfn_only.yaml

.venv-autogluon\Scripts\python.exe -u scripts\run_autogluon.py train `
  --config configs\autogluon\realtabpfn_only.yaml `
  --run-id autogluon-v3-realtabpfn-20260727

.venv-autogluon\Scripts\python.exe -u scripts\run_autogluon.py inspect `
  --run-dir artifacts\autogluon_runs\autogluon-v3-realtabpfn-20260727
```

`validate` checks the strict YAML schema, profile, train-only paths, containment, and
resource constraints without allocating a run or importing AutoGluon. `train` refuses
an existing run directory and launches a child Python process with `shell=False`.
Worker stdout and stderr are drained to durable UTF-8 logs.

The supervisor records process metadata and a raw process exit code. A successful run
must contain worker completion, core predictor files, and exported inspection files
before `_SUCCESS` is written as the final operation. Python failures and native
nonzero exits retain all partial files and end in `_FAILED`; a partial predictor is
listed as discovered but is not declared loadable.

`inspect` is read-only. By default it loads a predictor only for a structurally
complete run with `_SUCCESS`. Use `--attempt-load` to make a best-effort load attempt
for a partial run, or `--no-load` for filesystem-only inspection. Load errors are
reported without modifying the run.

## Profiles

- `realtabpfn_only_v1`: all `REALTABPFN-V2` configurations from AutoGluon 1.5.0's
  `zeroshot_2025_12_18_gpu` portfolio, with sequential folds.
- `catboost_only_cpu_v1`: the CPU portfolio `CAT` family with `task_type="CPU"` and
  zero GPUs.
- `lightgbmprep_only_cpu_v1`: the CPU portfolio `GBM_PREP` family with zero GPUs.
- `extreme_seqmem_v1`: the GPU extreme portfolio with sequential model and fold
  fitting, excluding `TABDPT`, `TABICL`, and `MITRA`. This is diagnostic, not the
  recommended default.

Exit codes are `0` for success, `2` for invalid input/configuration, `3` for an
existing run directory, `4` for worker failure, and `5` for an explicitly attempted
inspection load that failed.
