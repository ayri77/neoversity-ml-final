# Dataset Comparison v1

Dataset Comparison v1 compares two completed Experiment Core v2 research runs
that used different Dataset Packages but share the same independent training-row
identity, target, evaluation protocol, model configuration, and OOF alignment keys.
The operation uses persisted OOF predictions only. It never fits models, mutates
historical runs or packages, or reads competition-test, submission, Kaggle,
AutoGluon, or model artifacts.

Project contract: `experiment_core_v2_dataset_comparison_v1`.

## Independent row identity

Independent train-row identity is derived read-only from each run's referenced
Dataset Package manifest plus the run's persisted row-position fingerprint:

- primary: `row_identity.train_anchor_hash` from
  `data/processed/<dataset_id>/dataset_manifest.json`;
- observation order: `dataset_fingerprints.row_position_identity` (`kind` +
  `sha256`, excluding content-bound `bound_train_row_identity_hash`);
- train row count: `dataset_fingerprints.row_count`.

Do not use `dataset_provenance.train_row_identity_hash` or
`bound_train_row_identity_hash` alone as independent identity. If the package
manifest is unavailable, compatibility fails with `ROW_IDENTITY_UNAVAILABLE`.

## Normalized identities

### Evaluation protocol hash

Includes outer/threshold protocol, assignment fingerprints, threshold grid,
metrics, aggregation, positive-class semantics, and the repeat×fold product.
Excludes dataset fingerprints, feature schema, and plan IDs that embed dataset
IDs. Full plan identity differences belong in `expected_differences`, not
blockers, when the normalized protocol hash matches.

### Model configuration hash

Includes model family, adapter and pipeline identity (excluding dataset-bound
schema hashes and feature lists), hyperparameters, random-state rules, and
probability/evaluation-boundary semantics. Excludes dataset IDs, feature names,
counts, experiment/run IDs, and timestamps.

## Compatibility

Compatible when:

- `dataset_id` differs between baseline and candidate;
- independent row identity, target identity, evaluation protocol hash, outer and
  threshold assignments, model family, and model configuration hash match;
- OOF alignment is valid.

`parent_child_relation` is recorded as `baseline_is_parent`,
`candidate_is_parent`, or `unrelated`. Unrelated pairs are not hard-blocked for
validate-only flexibility; UI filtering may require parent-child.

Exploratory comparisons are flagged when either dataset is
`v3_targeted_missingness` or has `target_dependency: exploratory`.

## Blockers

`SAME_DATASET_ID`, `ROW_IDENTITY_UNAVAILABLE`, `INDEPENDENT_ROW_IDENTITY_MISMATCH`,
`OBSERVATION_ORDER_MISMATCH`, `TARGET_IDENTITY_MISMATCH`,
`EVALUATION_PROTOCOL_MISMATCH`, `OUTER_ASSIGNMENTS_MISMATCH`,
`THRESHOLD_ASSIGNMENTS_MISMATCH`, `MODEL_FAMILY_MISMATCH`,
`MODEL_CONFIGURATION_MISMATCH`, `METRIC_CONTRACT_MISMATCH`, and OOF codes:
`OOF_DUPLICATE_KEYS`, `OOF_MISSING_KEYS`, `OOF_ROW_COVERAGE_MISMATCH`,
`OOF_TARGET_DISAGREEMENT`, `OOF_INCOMPLETE_FOLD_PRODUCT`,
`OOF_INVALID_PROBABILITIES`.

## Expected differences when compatible

`dataset_id`, `parent_dataset_id`, `schema_hash`, `feature_count`,
`feature_names`, `dtypes`, train/test content hashes, experiment/run IDs,
timestamps, full evaluation plan id/hash, and source/resolved config hashes.

## Analysis

Repeat-level deltas are primary (`candidate - baseline` for principal metrics).
Fold results are descriptive. Prediction diagnostics cover correlation, absolute
probability differences, label disagreement, and directional changes by true
class. Fit-time delta is reported only when both runs expose comparable
authoritative timing metadata; otherwise it is explicitly unavailable. No blend
diagnostic and no formal inference in v1.

## Output layout

Root: `artifacts/research_v2_dataset_comparisons/<comparison-id>/`

Persisted artifacts include `resolved_comparison_config.yaml`,
`compatibility_report.json`, `normalized_identities.json`, immutable run
references, `summary.json`, repeat/fold metrics CSVs, `oof_diagnostics.json`,
provenance, inventory/manifest, and `_SUCCESS` (or `_FAILED` on allocated
failures). Existing comparison directories are never overwritten.

## CLI

```bash
uv run python scripts/compare_research_v2_datasets.py \
  --baseline-run-dir <baseline> \
  --candidate-run-dir <candidate> \
  --comparison-id <id> \
  --output-root artifacts/research_v2_dataset_comparisons
```

`--validate-only` authenticates both runs, derives identities, evaluates
compatibility and OOF alignment, and creates no output directory.

Default comparison id:
`<model_family>__<baseline_dataset>__vs__<candidate_dataset>`.

Primary unbiased parent→child pairs and the exploratory
`v1_missingness_summary` → `v3_targeted_missingness` pair are exported as
`UNBIASED_PARENT_CHILD_PAIRS` and `EXPLORATORY_PARENT_CHILD_PAIRS`.

Planned matrix:

- 6 unbiased parent→child hypotheses × LightGBM / XGBoost / CatBoost = 18
  unbiased comparisons;
- 3 exploratory `v1` → `v3` comparisons, always labelled exploratory and
  excluded from unbiased ranking.

Validate-only readiness for all 21 pairs was confirmed against local completed
Development runs. One end-to-end proof comparison was executed using persisted
OOF only:

```text
lightgbm__v0_raw_minimal__vs__v1_missingness_summary
```

No model fitting occurred. The Dataset Campaign Runner remains paused and
unexecuted for the full screening campaign unless separately authorized.
