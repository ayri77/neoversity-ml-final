# Project Context

## Project goal

This project addresses binary telecom churn classification. The competition metric is Balanced Accuracy.

The verified raw data profile is:

- 10,000 training rows;
- 2,500 test rows;
- 230 raw input features;
- 1,305 positive training targets;
- 8,695 negative training targets.

The project is an existing working competition solution. The current objective is to preserve its successful behavior while moving experiment execution and result tracking out of stateful notebooks.

## Current project phases

The project has progressed through these phases:

1. **Data audit:** `notebooks/01_data_audit.ipynb` validates the raw train/test schemas, profiles missing and constant values, and records target balance.
2. **Exploratory data analysis:** `notebooks/02_eda.ipynb` analyzes missingness, feature types, cardinality, and numerical and categorical distributions.
3. **Feature engineering:** `notebooks/03_feature_engineering.ipynb` creates the implemented versioned Parquet datasets.
4. **Standard baselines:** `notebooks/04_catboost_baseline.ipynb`, `notebooks/05_lightgbm_baseline.ipynb`, and `notebooks/06_xgboost_baseline.ipynb` run model-specific cross-validation and threshold analysis.
5. **Blending:** `notebooks/07_model_comparison.ipynb` compares aligned OOF predictions and evaluates fixed probability blends.
6. **Submission:** `notebooks/08_submission.ipynb` validates row alignment and creates the canonical CatBoost/XGBoost blend submission.
7. **AutoGluon experiments:** notebooks under `notebooks/sandbox/autogluon/` provide independent AutoML benchmarks in the isolated `.venv-autogluon` environment.
8. **Manual AutoGluon model reproduction:** `notebooks/sandbox/experiments/02_reproduce_autogluon_lightgbm.ipynb` reproduces the strongest AutoGluon LightGBMPrep model with explicit target encoding and LightGBM parameters.
9. **Experiment-platform transition:** the next phase is an additive, configuration-driven experiment runner with stronger validation and artifact contracts.

## Implemented dataset versions

### `v0_raw_minimal`

- 205 features.
- Source: raw competition data.
- Removes 25 constant features, including 18 all-missing features.
- Retains the original order and values of the remaining raw features.

### `v1_missingness_summary`

- 213 features.
- Parent: `v0_raw_minimal`.
- Adds eight row-level missingness summaries:
  - total missing count and rate;
  - numeric missing count and rate;
  - categorical missing count and rate;
  - very-high-missing-feature count and rate.

### `v2_missingness_indicators`

- 393 features.
- Parent: `v0_raw_minimal`.
- Adds four row-level missing counts and 184 binary indicators, one for every retained feature containing missing values in the training data.
- This is the broad missingness research version used to identify potentially useful targeted indicators.

### `v3_targeted_missingness`

- 217 features.
- Parent: `v1_missingness_summary`.
- Adds four targeted missing indicators:
  - `Var217_is_missing`;
  - `Var126_is_missing`;
  - `Var218_is_missing`;
  - `Var192_is_missing`.
- The stored lineage identifies `xgboost_v2_missingness_indicators_cv5` as the selection source experiment.

The following names are obsolete or unimplemented placeholders and are not active dataset versions:

- `v1_basic_clean`;
- `v2_eda_features`;
- `v3_native_categorical`.

## Important benchmark results

### CatBoost/XGBoost 50/50 blend

- Dataset: `v0_raw_minimal`.
- Legacy cross-fitted Balanced Accuracy: **0.898839**.
- Threshold: **0.130**.
- Kaggle Public Score: **0.8832**.
- Submission: `catboost_xgboost_50_50_threshold_0130.csv`.

### AutoGluon extreme

- Dataset: `v3_targeted_missingness`.
- Calibrated internal Balanced Accuracy: **0.895158**.
- Threshold: **0.117**.
- Kaggle Public Score: **0.9112**.
- Submission: `autogluon_v3_extreme_threshold_0117.csv`.

AutoGluon is retained as a strong external benchmark. It is not the primary project pipeline and remains isolated in `.venv-autogluon`.

### Manual LightGBMPrep reproduction

- Dataset: `v3_targeted_missingness`.
- Balanced Accuracy at fixed threshold 0.117: **0.900979**.
- Global optimized OOF Balanced Accuracy: **0.904078**.
- Global optimized threshold: **0.096**.
- Legacy cross-fitted threshold estimate: **0.898656**.
- Kaggle Public Score: **0.9023**.
- Test positive predictions at threshold 0.117: **569**.
- Submission: `manual_lightgbmprep_r31_threshold_0117.csv`.

The manual model is the primary first migration target because its implementation is currently notebook-only and its intermediate predictions and fitted objects are not fully persisted.

Global threshold optimization evaluates a threshold on the same OOF labels used to select it and is therefore optimistic. The existing cross-fitted threshold calculation is a historical metric and is not fully nested. Both must remain available for legacy comparison, but future validation metrics must be named separately.

Kaggle Public Leaderboard results are externally observed benchmarks. They must not be used as the sole experiment-selection mechanism.

## Current objectives

The planned direction is to:

- preserve the current datasets, baselines, predictions, and benchmark results;
- extract the manual LightGBM experiment from its sandbox notebook without changing its behavior;
- introduce validated, configuration-driven experiments;
- unify filesystem artifact persistence while retaining historical layouts;
- add MLflow as an optional tracking mirror rather than an immediate replacement;
- implement fixed, repeated, and properly nested validation protocols;
- move reusable computation out of notebooks;
- reduce canonical notebooks to human-facing analysis and reporting;
- conduct controlled feature and model ablation experiments after baseline parity is established.
