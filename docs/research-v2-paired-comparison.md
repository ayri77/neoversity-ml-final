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

The versioned policy can label development evidence `promising`, `mixed`, or
`not_improved`. These deterministic labels expose their thresholds in the
artifact and are not scientific findings or deployment decisions.

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

The versioned development policy is
`configs/research_v2/comparison_policy_v1.yaml`. Successful comparisons are
stored under `artifacts/research_v2_comparisons/<comparison-id>/` with input
manifest hashes, source provenance, CSV/JSON reports, an inventory, a manifest,
and a final `_SUCCESS` marker. Failures never receive `_SUCCESS`.

The typed loaders and builders in `src/churn_ml/paired_comparison.py` are
side-effect-free foundations for a later optional MLflow mirror and
experiment-campaign orchestration. Neither integration is implemented here.
