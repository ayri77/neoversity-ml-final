# Paired Comparison v1

Paired Comparison v1 compares two completed Experiment Core v2 research runs
without fitting a model or reading competition-test, submission, Kaggle,
AutoGluon, or model artifacts. The filesystem remains authoritative. Each input
is independently checked by the production Experiment Core v2 semantic
validator before comparison.

## Compatibility contract

The gate requires exact equality of the v2 schema and protocol versions,
evaluation-plan identity, dataset version and content fingerprints, outer and
threshold-selection assignments, target vector and row-order identity, metric
and aggregation contract, threshold policy and grid, positive-class probability
and label conventions, and the complete repeat/fold Cartesian product.

Pipeline, adapter, complete-candidate, and source identities may differ; run IDs
and timestamps may also differ. The gate never silently realigns incompatible
runs. Its JSON report gives stable reason codes and field paths for every
detected mismatch.

## Analysis

Repeat deltas are the primary comparison unit. Each repeat pools its exact outer
validation predictions and reports candidate-minus-baseline deltas for Balanced
Accuracy, sensitivity, specificity, ROC AUC, average precision, and Brier score.
A negative Brier delta favors the candidate because lower is better. Fold
results use exact repeat/fold pairs and are descriptive because folds and
repeats overlap and are not independent observations.

The aggregate report contains mean and median repeat deltas, sample standard
deviations where defined, wins/ties/losses under an explicit exact tie epsilon,
sensitivity/specificity trade-offs, and threshold stability. It performs no
formal inference and makes no scientific-significance claim.

## Decision policy

The status uses repeat-level Balanced Accuracy deltas, where
`delta = candidate - baseline`, and the exact values in
`configs/research_v2/comparison_policy_v1.yaml`. For each repeat, a win has
`delta > 1.0e-12`, a tie has `abs(delta) <= 1.0e-12`, and a loss has
`delta < -1.0e-12`. The repeat win fraction is exactly
`wins / (wins + ties + losses)`, so every classified repeat is in the
denominator.

The status boundaries are exhaustive and ordered:

- `promising` when `mean_delta >= 0.0001` and
  `wins / (wins + ties + losses) >= 0.5`;
- `mixed` when the promising rule is false and
  `mean_delta > 1.0e-12 or wins > losses`;
- `not_improved` otherwise, including an all-tie result.

The `every_repeat_improved` field is advisory only and requires every repeat
Balanced Accuracy delta to be strictly greater than `1.0e-12`. Candidate
threshold stability is reported using the sample standard deviation and the
policy limit `0.05`; threshold stability is advisory only; it does not affect the status label. Sensitivity/specificity trade-off reporting and the fixed
blend diagnostic are also advisory and do not affect the status label.

These deterministic development labels have no significance interpretation,
perform no formal inference, and are neither scientific findings nor deployment
decisions.

Prediction comparison aligns the persisted outer-validation
`repeat/repeat_seed/outer_fold/row_position` keys. It reports Pearson and
Spearman correlation, probability-distance diagnostics, label agreement,
correctness/disagreement cells, directional label changes, and disagreement by
true class. Duplicate keys or unequal coverage are rejected.

## Fixed blend diagnostic

The only ensemble diagnostic is a fixed 50/50 probability average. For each
outer repeat/fold, its threshold is selected from the aligned
threshold-selection OOF probabilities using the unchanged production policy and
grid. That fixed threshold is then applied to the averaged outer-validation
probabilities with the production `probability >= threshold` convention.
Outer-validation targets never select the threshold.

The blend is diagnostic only. It does not search weights, access competition
test data, create a submission, recommend deployment, or replace the component
runs.

## CLI

```powershell
.venv\Scripts\python.exe -u scripts\compare_research_v2.py `
  --baseline-run-dir artifacts/research_v2/<baseline-run> `
  --candidate-run-dir artifacts/research_v2/<candidate-run> `
  --comparison-id paired-example
```

Read-only validation performs both production input validations and the
compatibility gate without creating an output directory:

```powershell
.venv\Scripts\python.exe -u scripts\compare_research_v2.py `
  --baseline-run-dir artifacts/research_v2/<baseline-run> `
  --candidate-run-dir artifacts/research_v2/<candidate-run> `
  --validate-only
```

All relative CLI paths are resolved against the repository root, not the
process working directory, so the documented commands also work when invoked
from another directory. Absolute paths outside the repository are rejected.

The versioned development policy is
`configs/research_v2/comparison_policy_v1.yaml`. Before reading artifacts, each
input tree is recursively prewalked without following links. Symbolic links,
Windows junctions/reparse points, multiply linked regular files where link
counts are available, forbidden asset namespaces, and resolved descendants
outside the run or repository are rejected. Hard-link detection is limited by
the metadata exposed by the host filesystem.

Baseline, candidate, and proposed output paths must be pairwise disjoint: no
pair may be equal or have an ancestor/descendant relationship. This check runs
before allocation, including in validate-only mode. Successful comparisons are
stored under `artifacts/research_v2_comparisons/<comparison-id>/` with an exact
recursive file/directory structure. The recursive inventory and manifest use
POSIX relative paths, byte sizes, and SHA-256 hashes, including the allowed
`provenance/source_provenance.json` file. Unexpected files, empty directories,
forbidden assets, and linked descendants are rejected. `_SUCCESS` is written
last; failures after allocation receive only `_FAILED`, and failures before
allocation never mutate an input run.

The typed loaders and builders in `src/churn_ml/paired_comparison.py` are
side-effect-free foundations for a later optional MLflow mirror. Dataset
Campaign / Matrix Runner v1 (`docs/dataset-campaign-runner-v1.md`) orchestrates
Research v2 cells but does not perform cross-dataset paired inference; that
remains Stage E.
