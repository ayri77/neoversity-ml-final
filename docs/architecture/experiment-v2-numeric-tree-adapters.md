# Experiment Core v2 numeric tree adapters

Experiment Core v2 registers two versioned model adapters:

- `xgboost_numeric_v1` uses `xgboost.XGBClassifier`;
- `catboost_numeric_v1` uses `catboost.CatBoostClassifier`.

Both are CPU-only first benchmarks, not tuned configurations. They use the approved
`manual_v3_pipeline_v1_compat` feature contract and its existing fold-local OOF
target-encoding behavior to create the same ordered numeric matrix. CatBoost native
categoricals are intentionally out of scope for v1 so model comparisons do not also
change feature handling.

## Native missing-value policy

The v3 numeric matrix deliberately contains IEEE NaN values. Both v1 adapters use the
exact identity-bearing contract `missing_value_policy: native_nan`: NaNs are preserved
without replacement and delegated to each estimator's native numerical missing-value
mechanism. Positive and negative infinity and nonnumeric matrix values remain invalid.
No fill, sentinel replacement, or statistical imputation is performed.

XGBoost's portable estimator contract contains the exact primitive field
`missing: IEEE_NaN`. It is retained in resolved configuration, provenance, and
identity payloads, then translated to `missing=np.nan` only when constructing
`XGBClassifier`. CatBoost receives `nan_mode="Min"` explicitly, together with
`task_type="CPU"` and `allow_writing_files=False`. Portable configuration and
identity payloads never contain raw non-finite numeric values. Future
imputation-based pipelines or adapters must use distinct versioned identities rather
than changing v1 behavior.

The adapters fit only the supplied training partition. They do not receive an
evaluation set, use early stopping, choose thresholds, create labels, fit final
models, predict competition test data, or create submissions. Their only output is
the probability for class `1`; Experiment Core v2 owns nested threshold selection
and metrics.

## Configuration and commands

Ready-to-run configurations are:

- `configs/research_v2/xgboost_numeric_v1_smoke.yaml` (1 repeat, 3 outer folds,
  2 threshold-selection folds);
- `configs/research_v2/catboost_numeric_v1_smoke.yaml` (1 x 3 x 2);
- `configs/research_v2/xgboost_numeric_v1_development.yaml` (2 x 5 x 3);
- `configs/research_v2/catboost_numeric_v1_development.yaml` (2 x 5 x 3).

Validate a configuration without allocating a run:

```powershell
uv run python scripts/run_research_v2.py --config configs/research_v2/xgboost_numeric_v1_smoke.yaml --validate-only
```

Run a smoke evaluation only when the approved train-only dataset is available:

```powershell
uv run python scripts/run_research_v2.py --config configs/research_v2/xgboost_numeric_v1_smoke.yaml
uv run python scripts/run_research_v2.py --config configs/research_v2/catboost_numeric_v1_smoke.yaml
```

Development configurations use the same command with the corresponding
`*_development.yaml` path. They are research evaluations, not deployment runs.

## Parameter overrides and identity

Each adapter contract is an exact primitive-only mapping. A future Optuna layer can
copy a configuration, replace explicit values under
`candidate_adapter.contract.<library>.parameters`, and pass the result through the
same production validator before hashing or execution. Unknown keys, numeric strings,
bool-as-int values, non-finite parameter values, unsafe ranges, GPU settings, file
writing, and missing-value policies other than `native_nan` are rejected. No Optuna
dependency or search space is embedded in the adapters.

Resolved contracts, native-missing settings, and selected-library versions are
included in portable adapter identity/provenance. Shared numeric-adapter sources
affect both new identities; XGBoost- and CatBoost-specific sources affect only their
respective identity. The legacy `manual_lightgbm_te_v1_compat` identity inputs and
source set are unchanged.

The adapters are imported lazily. If the selected package or its distribution
metadata is absent, Experiment Core v2 raises an adapter-specific diagnostic naming
the adapter, package, locked version, and the action to restore the locked project
environment. The expected v1 versions are XGBoost `3.3.0` and CatBoost `1.2.10`;
unrelated nested import failures are preserved for accurate diagnosis.
