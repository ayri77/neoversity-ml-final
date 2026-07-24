# Current-State Architecture

## Status labels

This document uses the following labels:

- **Verified:** directly supported by repository source, notebook source, metadata, or saved local artifacts.
- **Inferred:** a reasonable consequence of the verified implementation, but not directly recorded as an explicit project decision.
- **Unresolved:** cannot be established confidently from the repository.

This document describes the current architecture. It does not define a final directory redesign.

## Repository map

| Path | Classification | Current responsibility |
|---|---|---|
| `src/churn_ml/` | Canonical source | Implemented data audit, feature generation, metrics, model-specific CV, visualization, and two artifact-writing abstractions. Several planned modules are empty. |
| `notebooks/` | Canonical notebooks | Numbered notebooks `01` through `08` implement the current end-to-end research and submission workflow. |
| `notebooks/sandbox/` | Experimental | Ignored AutoGluon, drift, and manual model-reproduction notebooks. |
| `sandbox/autogluon/` | Environment-specific support | Tracked AutoGluon setup instructions, direct dependency pins, and an environment diagnostic script. |
| `data/raw/` | Generated/local input | Ignored competition train, test, and sample-submission CSV files. |
| `data/interim/` | Generated/local | Ignored EDA-derived feature metadata. It currently contains two duplicate metadata directories. |
| `data/processed/` | Generated/local | Ignored versioned Parquet datasets and metadata JSON files. |
| `artifacts/experiments/` | Generated/local | Standard-model metrics, OOF/test predictions, selected models, feature importance, threshold diagnostics, and blend results. |
| `artifacts/autogluon/` | Generated/local, environment-specific | AutoGluon predictor and model directories. |
| `artifacts/reference/` | Generated/local | Exported AutoGluon LightGBMPrep OOF and test probabilities used by the manual reproduction notebook. |
| `docs/benchmarks/` | Canonical documentation | Tracked AutoGluon benchmark narrative and historical Kaggle results. |
| `submissions/` | Generated/local | Ignored competition submission CSV files. |
| `configs/` | Placeholder | Intended dataset and model YAML files; currently empty or effectively empty. |
| `scripts/` | Placeholder | Intended experiment, final-training, and submission entry points; currently empty. |
| `tests/` | Placeholder | No implemented test suite. |
| `mlruns/` | Generated/local placeholder | Empty local MLflow directory; no current MLflow logging calls. |
| `.venv/` | Environment-specific | Main project environment. |
| `.venv-autogluon/` | Environment-specific | Isolated AutoGluon environment with different dependency versions and GPU packages. |

## Current data flow

**Verified flow:**

1. `data/raw/final_proj_data.csv`, `data/raw/final_proj_test.csv`, and `data/raw/final_proj_sample_submission.csv` are loaded through `src/churn_ml/data.py::load_competition_data()`.
2. `notebooks/01_data_audit.ipynb` produces schema, constant-feature, and class-distribution reports.
3. `notebooks/02_eda.ipynb` derives missingness, type, cardinality, and feature metadata.
4. `notebooks/03_feature_engineering.ipynb` creates `v0_raw_minimal`, `v1_missingness_summary`, `v2_missingness_indicators`, and `v3_targeted_missingness`.
5. `src/churn_ml/features.py::save_dataset()` persists each version as `X_train.parquet`, `y_train.parquet`, `X_test.parquet`, and `metadata.json`.
6. Model notebooks load a selected version through `src/churn_ml/features.py::load_dataset()`.
7. Model-specific modules produce fold metrics, OOF probabilities, and averaged test probabilities.
8. Threshold logic converts probabilities into labels.
9. `notebooks/07_model_comparison.ipynb` creates aligned blend probabilities.
10. `notebooks/08_submission.ipynb` validates test order against the sample-submission index and writes a submission.
11. Historical external scores are recorded in `docs/benchmarks/kaggle_submissions.csv`.

### Current metadata path mismatch

**Verified:** `notebooks/02_eda.ipynb` currently writes feature lists to `data/interim/feature_metadata/`, while `notebooks/03_feature_engineering.ipynb` currently reads them from `data/interim/eda_metadata/`.

Both local directories were byte-identical at audit time. This is an implicit dependency on existing local state or an earlier notebook revision. It must not be changed without a parity check against all implemented dataset versions.

## Current experiment flows

### Reusable standard-model flow

**Verified:** CatBoost, LightGBM, and XGBoost share this structure:

1. Load a `PreparedDataset`.
2. Create shuffled five-fold `StratifiedKFold` with seed 42.
3. Determine categorical columns by pandas dtype.
4. Prepare training, validation, and test frames inside each fold.
5. Fit with AUC-based validation and early stopping.
6. Place validation probabilities into the original OOF positions.
7. Average test probabilities across fold models.
8. Calculate Balanced Accuracy at threshold 0.5 and probability metrics.
9. Select a global OOF threshold over a fixed grid.
10. Return metrics, predictions, and in-memory fold models.

Model-specific differences are material:

- CatBoost converts categorical values to strings, fills categorical missing values with `__MISSING__`, and currently uses GPU training.
- LightGBM derives pandas category levels from the training fold and uses the fold model's best iteration for prediction.
- XGBoost similarly derives training-fold categories and uses native categorical support.

There is no implemented full-data final-model fitting path. Standard test predictions are fold-ensemble averages.

### Notebook-only ensemble flow

**Verified:** `notebooks/07_model_comparison.ipynb` loads the v0 CatBoost, LightGBM, and XGBoost OOF/test artifacts, verifies row/fold/target alignment, evaluates six fixed probability blends, and performs a fold-wise CatBoost/XGBoost weight search. The selected submission blend is a fixed 50/50 CatBoost/XGBoost average with threshold 0.130.

The reusable blend and submission-selection logic has not yet been extracted into source modules.

### AutoGluon flow

**Verified:** AutoGluon notebooks run only in `.venv-autogluon`, load prepared Parquet datasets, fit framework-native bagged predictors, and store predictors under `artifacts/autogluon/`. The v3 extreme run exports LightGBMPrep OOF/test reference probabilities to `artifacts/reference/autogluon_lightgbmprep_r31/`.

**Verified:** the current v3 extreme notebook defines a `GPU_HYPERPARAMETERS` mapping but does not pass it to `TabularPredictor.fit()`.

**Unresolved:** the exact source configuration for every retained AutoGluon artifact is not available in the current notebook set.

### Manual LightGBMPrep reproduction flow

**Verified:** `notebooks/sandbox/experiments/02_reproduce_autogluon_lightgbm.ipynb`:

1. Loads `v3_targeted_missingness`.
2. Loads the ignored AutoGluon LightGBMPrep reference predictions.
3. Drops `Var214`, `Var220`, `Var222`, and `Var218_is_missing`.
4. Replaces categorical columns with a custom smoothed OOF target encoding.
5. Uses eight-fold outer stratified CV with seed 0.
6. Uses five-fold inner target encoding with seed 42.
7. Fits LightGBM for 376 boosting rounds with copied LightGBMPrep parameters.
8. Averages eight test probability vectors.
9. Creates the saved submission at fixed threshold 0.117.

The notebook does not persist its manual OOF probabilities, fold assignments, test probabilities, fold metrics, encoder state, or fitted models.

## Current artifact systems

### `Experiment`

`src/churn_ml/experiment.py::Experiment` creates timestamped run directories and can save:

- environment information;
- Git and raw-file fingerprints;
- metadata and YAML configuration;
- CSV metrics and predictions;
- joblib models;
- reports and submissions;
- a CSV experiment index.

Its default artifact root is relative to the current working directory. Canonical EDA notebooks therefore created runs under `notebooks/artifacts/experiments/` when launched from `notebooks/`.

### `save_experiment_result()`

`src/churn_ml/modeling.py::save_experiment_result()` uses a stable experiment ID and saves:

- `result.json`;
- `fold_metrics.csv`;
- OOF and test Parquet files;
- an optional `model.joblib`.

It does not capture the environment, Git/data fingerprints, resolved configuration, or a unified experiment index.

### AutoGluon-native storage

AutoGluon writes its own predictor, learner, model, utility, version, and environment metadata layout under `artifacts/autogluon/<run>/`. These artifacts are large, ignored, and coupled to the isolated AutoGluon environment.

### Tracked benchmark documentation

`docs/benchmarks/` contains the durable human-readable AutoGluon summary and Kaggle Public Score table. It does not contain the model or prediction artifacts required to reproduce those scores.

### MLflow status

**Verified:** MLflow is installed in the main environment, but there are no current MLflow calls and `mlruns/` is empty.

MLflow should not directly replace the existing systems before parity checks exist. The current filesystem artifacts contain behavior-specific schemas and historical local dependencies that must first be preserved. An initial MLflow integration should mirror a validated filesystem run rather than redefine it.

## Verified risks and limitations

### Verified

- All standard experiments reuse the same shuffled five-fold split with seed 42.
- Global OOF threshold optimization selects and reports the threshold on the same OOF labels, so the optimized score is optimistic.
- The existing threshold cross-fitting is legacy and not fully nested: calibration OOF predictions may come from models trained on rows in the nominal threshold-evaluation fold.
- Several notebooks contain out-of-order or missing execution counts while later cells retain outputs, demonstrating execution-state dependence.
- Notebooks 01 through 06 use `Path.cwd().parent` and therefore depend on being launched from `notebooks/`.
- Processed Parquet files omit source dataframe indexes. Row identity is reconstructed as order-based `RangeIndex`, not a persistent entity ID.
- Important processed data, experiment artifacts, sandbox notebooks, AutoGluon predictors, reference predictions, and submissions are ignored local assets.
- Manual LightGBM intermediate outputs and fitted objects are not persisted.
- The current AutoGluon requirements file is not a complete transitive lock for every extreme-preset model and downloaded checkpoint.
- There is no implemented test suite.
- Configuration and script entry-point files are placeholders.
- The repository imports `src.churn_ml` after modifying `sys.path`; the project is not installed as an importable `churn_ml` package.

### Inferred

- Repeated feature, blend, and threshold decisions on the same fixed folds can produce research-selection optimism even when individual preprocessing steps are leakage-safe.
- GPU CatBoost results may require a tolerance rather than bit-exact parity across different drivers or hardware.
- Ignored artifacts are vulnerable to local deletion unless they are backed up outside Git.

### Unresolved

- The formal rule used to select the four targeted v3 indicators is not encoded.
- The intended canonical metadata directory is not documented.
- It is not established whether future final predictions should use fold ensembles, a full-data refit, or both.
- It is not established which retained AutoGluon runs must be reconstructable as long-term baselines.
