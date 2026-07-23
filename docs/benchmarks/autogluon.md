# AutoGluon Benchmark

## Purpose

This benchmark evaluates AutoGluon as an independent AutoML reference point for the customer churn task. The goal is not to replace the project modeling pipeline, but to estimate how much performance can be obtained from automated model selection, bagging, preprocessing, and ensembling on the same prepared dataset.

## Experiment Setup

- AutoGluon version: `1.5.0`
- Python version: `3.12.12`
- Operating system: Windows
- CPU cores: `32`
- Available memory at startup: approximately `4.62 GB` of `31.63 GB`
- GPU requested: `1`
- GPU detected by AutoGluon: `0`
- PyTorch build: `2.9.1+cpu`
- CUDA status: unavailable
- Dataset version: `v0_raw_minimal`
- Training rows: `10,000`
- Test rows: `2,500`
- Input features: `205`
- Features used by AutoGluon after preprocessing: `202`
- Target: `target`
- Problem type: binary classification
- Evaluation metric: `balanced_accuracy`
- Preset: `best_v150`, mapped by AutoGluon to `best_quality_v150`
- Hyperparameter preset: `zeroshot_2025_12_18_cpu`
- Bagging folds: `8`
- Stack levels: `0`
- Time limit: `21,600` seconds
- Actual total runtime: `6,407.88` seconds, approximately 1 hour 47 minutes

AutoGluon automatically dropped three high-cardinality or uninformative categorical features:

- `Var214`
- `Var220`
- `Var222`

## Main Result

The best model was:

```text
WeightedEnsemble_L2
```

Validation performance before threshold calibration:

```text
Balanced Accuracy @ 0.500 = 0.8341
```

AutoGluon then calibrated the binary decision threshold:

```text
Base threshold = 0.500
Best threshold = 0.196
Balanced Accuracy @ 0.196 = 0.8732
```

This confirms that threshold selection is essential for this imbalanced classification task.

## Ensemble Composition

The final weighted ensemble combined six level-1 bagged models:

| Model | Weight |
|---|---:|
| `NeuralNetTorch_r31_BAG_L1` | 0.375 |
| `LightGBMPrep_r31_BAG_L1` | 0.188 |
| `LightGBMPrep_r19_BAG_L1` | 0.188 |
| `LightGBMPrep_r41_BAG_L1` | 0.125 |
| `NeuralNetTorch_r144_BAG_L1` | 0.062 |
| `NeuralNetTorch_r82_BAG_L1` | 0.062 |

The final ensemble did not include the AutoGluon CatBoost model.

## Best Individual Models

| Rank | Model | Validation Balanced Accuracy |
|---:|---|---:|
| 1 | `LightGBMPrep_r31_BAG_L1` | 0.8225 |
| 2 | `LightGBMPrep_r14_BAG_L1` | 0.8215 |
| 3 | `LightGBMPrep_r19_BAG_L1` | 0.8212 |
| 4 | `LightGBMPrep_r41_BAG_L1` | 0.8194 |
| 5 | `NeuralNetTorch_r31_BAG_L1` | 0.8173 |
| 6 | `CatBoost_c1_BAG_L1` | 0.8033 |

The strongest individual family was `LightGBMPrep`, which combines LightGBM with additional preprocessing such as arithmetic features, categorical interactions, and out-of-fold target encoding.

## Comparison with the Project CatBoost Baseline

Current project CatBoost results:

| Metric | Project CatBoost | AutoGluon |
|---|---:|---:|
| Balanced Accuracy @ default threshold | 0.7940 | 0.8341 |
| Optimized OOF / validation threshold | 0.174 | 0.196 |
| Optimized Balanced Accuracy | 0.8946 | 0.8732 |

The project CatBoost currently performs better after threshold optimization by approximately `0.0214` Balanced Accuracy.

This comparison is directional rather than perfectly identical because:

- the project CatBoost uses 5-fold cross-validation;
- AutoGluon uses 8-fold bagging;
- threshold calibration is performed using each framework's own validation predictions;
- the exact validation protocol and ensemble construction differ.

A Kaggle submission or a shared evaluation protocol is required for a definitive comparison.

## Training Issues and Limitations

### GPU was not used

Although `num_gpus=1` was passed to AutoGluon, the environment contained a CPU-only PyTorch build:

```text
PyTorch version: 2.9.1+cpu
CUDA is not available
Specified total num_gpus: 1, but only 0 are available
```

As a result, AutoGluon trained all models on CPU.

### FastAI models failed

The `NeuralNetFastAI` configurations failed with:

```text
AttributeError: 'list' object has no attribute 'starmap'
```

The traceback points to an incompatibility involving `fastai` and `fastcore`. These models were skipped and did not contribute to the final ensemble.

### Some LightGBM configurations failed

Several LightGBM configurations were skipped because of:

- an AutoGluon memory early-stopping callback failure:

```text
TypeError: 'NoneType' object is not iterable
```

- insufficient available RAM for some `LightGBMPrep` configurations;
- reduced fold parallelism due to memory limits.

### Memory pressure affected execution

AutoGluon frequently reduced parallelism from eight folds to one, two, or four folds. Some high-memory configurations were skipped entirely. Therefore, this benchmark does not represent the maximum possible AutoGluon result on the available hardware.

## Key Findings

1. The custom CatBoost pipeline remains the strongest current result after threshold optimization.
2. AutoGluon confirmed that a threshold near `0.17-0.20` is appropriate for the task.
3. `LightGBMPrep` is the most promising direction for the next manual model family.
4. A blend of preprocessed LightGBM models and tabular neural networks improved performance over every individual AutoGluon model.
5. The AutoGluon run was constrained by a CPU-only PyTorch build, FastAI dependency errors, and memory pressure.

## Recommended Follow-up

Before using AutoGluon for a final high-performance search:

1. Repair the isolated `.venv-autogluon` environment.
2. Install a CUDA-enabled PyTorch build compatible with the installed NVIDIA driver and AutoGluon version.
3. Verify GPU visibility with both PyTorch and AutoGluon.
4. Resolve the `fastai` / `fastcore` incompatibility.
5. Investigate the LightGBM memory callback failures.
6. Re-run a short validation experiment before launching another long benchmark.
7. Consider a later high-budget AutoGluon run only after the manual CatBoost, LightGBM, and XGBoost baselines are complete.

## Status

The benchmark is retained as an external AutoML reference. AutoGluon is not part of the primary project pipeline at this stage.