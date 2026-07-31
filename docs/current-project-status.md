# Current Project Status and Roadmap

> Operational handoff document for continuing the project across development
> sessions and assistant dialogs.

**Last updated:** 2026-07-31  
**Repository:** `ayri77/neoversity-ml-final`  
**Active task branch:** `feature/prepared-dataset-pipeline-v1`  
**Current checkpoint:** Stage C Registry ↔ Experiment Core integration and
    hardening are on the branch tip. Stage D Dataset Campaign / Matrix Runner v1
    is implemented (versioned campaign contract, validate-only, freeze, sequential
    execute/resume, CLI, and synthetic tests). The 21-run unbiased screening
    campaign has **not** been executed; do not treat Stage D screening as
    complete until a real frozen campaign run is audited. Complete the
    post-notebook Registry package audit before launching any real campaign.

## 1. Purpose and maintenance policy

This document is the canonical cross-session handoff and the current operational
roadmap. It records:

- the active checkpoint and immediate next action;
- implemented capabilities that must not be rediscovered;
- accepted dataset and evaluation contracts;
- remaining work in dependency order;
- known risks, restrictions, and deferred directions.

It is not a replacement for:

- immutable `dataset_manifest.json` files;
- frozen experiment or campaign manifests;
- saved filesystem artifacts;
- specialist architecture, registry, reproducibility, or benchmark documents;
- Git history.

Update this file when a material project fact changes, including:

- completion or failure of a roadmap stage;
- a changed dataset, manifest, evaluation, or artifact contract;
- a different active branch or next action;
- a newly accepted benchmark or decision;
- a resolved or newly discovered risk;
- a reordered or removed roadmap item.

Do not update it for trivial refactoring or temporary investigation notes.
Historical facts should remain in their authoritative manifests, artifacts, or
versioned documentation rather than being silently rewritten here.

When resuming work, verify volatile state instead of trusting an old snapshot:

```powershell
git -C $repo branch --show-current
git -C $repo rev-parse HEAD
git -C $repo status --short --branch
```

## 2. Immediate resume checkpoint

Local `data/processed` currently exposes eight Registry packages discovered by
`discover_registered_datasets` (`v0`-`v7`). The notebook working tree may still
contain user-local notebook changes; do not revert or commit them unless asked.

Immediate next actions:

1. Complete the post-notebook audit in section 9 (scan/validate Registry and
   package checks). Do not treat UI or Campaign Runner implementation as a
   substitute for that audit.
2. Keep `notebooks/03_feature_engineering.ipynb` untouched unless the user
   explicitly requests edits.
3. After the audit passes, validate-only the Stage D campaign specification
   (`docs/dataset-campaign-runner-v1.md`) before any real matrix execution.
4. Do not start the full 21-run development screening campaign until every
   package passes the audit and validate-only succeeds.

The Experiment Control Panel dataset selector is implemented and covered by
automated tests, including a validate-only smoke path for
`v7_compact_zero_indicators` + LightGBM smoke. Dataset identity is also
persisted in Experiment Core UI job references, surfaced in Results/Compare,
and mapped into searchable MLflow metadata for validated completed research
runs. That does not finish Stage A or the Dataset Campaign.

### Working-directory dependency

The notebook currently resolves the repository root as:

```python
PROJECT_ROOT = Path.cwd().parent
```

It must therefore be launched with `notebooks` as the current directory.

### Known metadata-path discrepancy

- `notebooks/02_eda.ipynb` writes metadata under
  `data/interim/feature_metadata/`.
- `notebooks/03_feature_engineering.ipynb` reads metadata under
  `data/interim/eda_metadata/`.

The two directories were previously found to be byte-identical. Do not change
the path pre-emptively. If the notebook fails, inspect both directories and
their contents before modifying code.

## 3. Project objective and verified data profile

The project addresses binary telecom churn classification. The competition
metric is **Balanced Accuracy**.

Verified raw-data profile:

| Item | Value |
|---|---:|
| Training rows | 10,000 |
| Test rows | 2,500 |
| Raw input features | 230 |
| Positive target rows (`y = 1`) | 1,305 |
| Negative target rows (`y = 0`) | 8,695 |
| Positive-class rate | 13.05% |
| Constant raw features | 25 |
| All-missing features among constants | 18 |
| Raw features containing missing values | 209 |

The current research hypothesis is that controlled analysis and representation
of missingness, zero values, and joint structural patterns can improve the
competition result more reliably than another model-only or black-box AutoML
search.

## 4. Orange Small leakage constraint

The competition rows were compared with the official Orange Small Dataset:

- all 10,000 competition training rows were found in Orange Small train;
- all 2,500 competition test rows were also found in Orange Small train;
- the competition rows do not correspond to the original Orange Small test.

The competition dataset is therefore probably a new split of Orange Small
train.

Known Orange labels must never be extracted or used for competition test rows.
Doing so would be target leakage. Orange Small may be used only as a structural
reference, not as a source of competition answers.

## 5. Repository and environment constraints

### Git

- Continue on `feature/prepared-dataset-pipeline-v1`.
- At the last verified clean checkpoint, the branch was synchronized with
  `origin`.
- Only `master` and `feature/prepared-dataset-pipeline-v1` remained locally and
  on `origin`.
- Only one worktree remained.
- Do not create another branch or worktree without explicit approval.
- Do not perform Git write operations without an explicit request.
- Do not record a commit SHA as current without verifying it in the same
  session.

### Python environments

- Use `.venv` for the main project.
- Use `.venv-autogluon` only for AutoGluon.
- Prefer `uv` for main-environment commands.
- Do not install or update dependencies without explicit approval.
- Provide commands for PowerShell.
- Write code, comments, configuration, and documentation in English.

### Data and artifacts

Do not commit raw data, processed datasets, models, MLflow runs, submissions, or
other local generated artifacts unless explicitly requested.

Do not overwrite historical experiment artifacts. Filesystem artifacts remain
authoritative; MLflow is an optional read-only tracking mirror.

## 6. Implemented project capabilities

The project already contains a substantial Experiment Platform:

- Experiment Core v2;
- nested and repeated research evaluation;
- model adapters:
  - `manual_lightgbm_te_v1_compat`;
  - `xgboost_numeric_v1`;
  - `catboost_numeric_v1`;
- paired comparison for compatible runs on one dataset;
- blend evaluation;
- final deployment;
- Optuna integration;
- standalone AutoGluon runner;
- optional MLflow mirroring;
- Streamlit Experiment Control Panel;
- saved OOF predictions, fold assignments, metrics, thresholds, identities, and
  provenance.

The first Registry-to-Experiment-Core bridge is also implemented:

- strict Dataset Package & Registry v1;
- dynamic `discover_registered_datasets()` API;
- Registry-backed loading for Research v2;
- generic `registered_prepared_passthrough_v1` pipeline;
- saved `dataset_provenance.json`;
- validation of schema, content, target, and row-identity hashes;
- support for pandas `object`, `category`, and `string` categoricals;
- categorical handling of `missingness_pattern_id`;
- fold-local target encoding;
- exclusion of competition `X_test` from research evaluation.

Implemented code is not the same as completed integration. The following still
require real-package verification or further development:

- bridge verification on generated `v0`-`v7`;
- parity between the new generic route and
  `manual_v3_pipeline_v1_compat`;
- first-class Registry documentation/catalog coverage for `v5`-`v7`;
- real execution of the Stage D 21-run screening campaign (runner implemented);
- cross-dataset paired comparison;
- campaign-results UI;
- Multi-Blend v2 for cross-model and cross-dataset blending.

The Control Panel dynamic Dataset Package selector for Experiment Core v2 is
implemented: discovery uses `discover_registered_datasets`, exploratory
packages are warned, and Validate/Run consume a prepared local config/plan.
Dataset identity is first-class in job labels, Results tables/filters/charts,
descriptive vs official Compare readiness, and validated MLflow params/tags/
run names. Filesystem artifacts remain authoritative.

## 7. Prepared dataset suite

Notebook 03 is expected to generate eight Dataset Packages.

| Dataset ID | Parent | Features | Target dependency | Purpose |
|---|---|---:|---|---|
| `v0_raw_minimal` | — | 205 | `none` | Remove 25 constant/all-missing columns while preserving all remaining values and order |
| `v1_missingness_summary` | `v0_raw_minimal` | 213 | `none` | Add eight row-level missingness summaries |
| `v2_missingness_indicators` | `v0_raw_minimal` | 393 | `none` | Add four missing counts and 184 broad missingness indicators |
| `v3_targeted_missingness` | `v1_missingness_summary` | 217 | `exploratory` | Add four target-informed missingness indicators |
| `v4_zero_value_summary` | `v0_raw_minimal` | 209 | `none` | Add four target-independent zero-value summaries |
| `v5_joint_missingness_pattern` | `v1_missingness_summary` | 214 | `none` | Add one categorical joint missingness-pattern identifier |
| `v6_compact_missingness_indicators` | `v1_missingness_summary` | 247 | `none` | Add 34 structurally unique missingness indicators |
| `v7_compact_zero_indicators` | `v4_zero_value_summary` | 234 | `none` | Add 25 structurally unique zero indicators |

### 7.1 `v0_raw_minimal`

- Removes the union of constant and all-missing raw features.
- Preserves the original order and values of the other 205 features.
- Serves as the row-identity anchor for current native packages.

### 7.2 `v1_missingness_summary`

Adds:

- `missing_count_total`;
- `missing_rate_total`;
- `missing_count_numeric`;
- `missing_rate_numeric`;
- `missing_count_categorical`;
- `missing_rate_categorical`;
- `missing_count_very_high`;
- `missing_rate_very_high`.

### 7.3 `v2_missingness_indicators`

- 184 retained features contain missing values.
- Adds one `_is_missing` indicator for each.
- Adds four missing-count summaries.
- This is a wide, target-independent research package and is intentionally
  redundant.

### 7.4 `v3_targeted_missingness`

Adds:

- `Var217_is_missing`;
- `Var126_is_missing`;
- `Var218_is_missing`;
- `Var192_is_missing`.

These features were selected using the earlier full-data experiment
`xgboost_v2_missingness_indicators_cv5`. The package is therefore exploratory.
No later nested protocol can retroactively make that feature-discovery step
unbiased.

### 7.5 `v4_zero_value_summary`

The source pool contains 171 retained numeric features.

Selection rules for supported zero features:

- `min_present = 1000`;
- `min_zero = 100`;
- `min_nonzero = 100`;
- use training features without `y`;
- deduplicate identical zero/observation masks.

Expected diagnostics:

- 33 features before deduplication;
- 25 features after deduplication;
- 8 duplicate-structure groups;
- 209 final features.

Adds:

- `zero_count_numeric`;
- `zero_rate_observed_numeric`;
- `zero_count_supported_numeric`;
- `zero_rate_observed_supported_numeric`.

### 7.6 `v5_joint_missingness_pattern`

Expected missingness structure:

- 184 features containing missing values;
- 34 unique missingness-mask representatives;
- 22 duplicate-mask groups.

Adds one pandas string/categorical feature:

```text
missingness_pattern_id
```

It contains a hex-encoded packed missingness mask.

Expected diagnostics:

- 693 training patterns;
- 372 test patterns;
- approximately 97.60% of test rows use a pattern observed in training;
- 214 final features.

Frequency encoding, rare grouping, and target encoding are not materialized in
this package.

### 7.7 `v6_compact_missingness_indicators`

- Reuses the 34 unique missingness-mask representatives from `v5`.
- Adds 34 binary `int8` features named `<feature>_is_missing`.
- Contains 247 final features.

### 7.8 `v7_compact_zero_indicators`

- Inherits the four summaries from `v4`.
- Reuses the fixed list of 25 supported zero features from `v4`.
- Adds 25 binary `int8` features named `<feature>_is_zero`.
- Treats missing values as “not zero”.
- Contains 234 final features.

## 8. Dataset Package & Registry contract

Expected package layout:

```text
data/processed/<dataset_id>/
  X_train.parquet
  y_train.parquet
  X_test.parquet
  metadata.json
  dataset_manifest.json
```

`dataset_manifest.json` is the canonical Registry contract. `manifest.json` is
not canonical.

Required manifest information includes:

- `schema_version = dataset_package_v1`;
- `dataset_id`;
- `parent_dataset_id`;
- hypothesis and transformations;
- training and test row counts;
- ordered feature names;
- dtypes and feature roles;
- `target_dependency`;
- `schema_hash`;
- file content hashes;
- target metadata and target hash;
- row identity and alignment proof.

Allowed feature roles:

- `numeric`;
- `categorical`;
- `binary_indicator`;
- `summary`.

Registered packages are immutable. Even `overwrite=True` must not replace an
already registered package.

If notebook execution raises `OverwriteRefusedError`, do not delete
`data/processed` immediately. First:

1. inspect the existing manifests;
2. scan and validate the Registry;
3. compare schemas and hashes;
4. determine whether the existing packages are already the correct result;
5. separately agree on controlled removal or a new dataset ID only if needed.

Some saved notebook outputs may be older than the current code. For example,
older output may say `row_alignment_check: passed`, while the current Registry
contract records `row_identity.alignment_status: proven`. A complete manual
execution is intended to refresh such stale output.

A `FutureWarning` involving `None` and `NaN` comparison is not by itself a
failed run when validation and package round-trip succeed. Record it as a future
compatibility risk.

## 9. Post-notebook audit

After notebook execution, inspect all eight directories and run:

```powershell
uv run python scripts/run_dataset_registry.py scan --root data/processed
uv run python scripts/run_dataset_registry.py validate --root data/processed
```

Then run the relevant tests:

```powershell
uv run pytest `
  tests/test_features.py `
  tests/test_dataset_registry.py `
  tests/test_registry_experiment_core_phase1.py
```

For every `v0`-`v7` package, verify:

- 10,000 training rows;
- 2,500 test rows;
- the expected feature count;
- identical train/test column order;
- compatible train/test dtypes;
- the same target hash across every package;
- child target hash equal to parent target hash;
- proven row alignment;
- unchanged parent projection;
- generated indicators are binary `int8`;
- `missingness_pattern_id` is pandas `string`;
- no unexpected generated missing values;
- valid manifests;
- successful package round-trip.

Do not commit the notebook merely because execution succeeded. Inspect:

```powershell
git -C $repo diff -- notebooks/03_feature_engineering.ipynb
git -C $repo status --short --branch
```

## 10. Roadmap overview

| Stage | Status | Outcome |
|---|---|---|
| A. Generate and audit `v0`-`v7` | In progress | Eight validated immutable Dataset Packages |
| B. Finish Registry/manifests/docs | Planned | `v5`-`v7` are first-class and documentation matches reality |
| C. Verify Registry ↔ Experiment Core | Partly implemented | Real-package bridge validation and v3 compatibility parity |
| D. Dataset Campaign / Matrix Runner | Implemented (not executed) | Versioned contract, validate-only, freeze, sequential execute/resume, CLI; 21-run screening not yet run |
| E. Cross-dataset paired comparison | Planned | Parent-child deltas on aligned OOF |
| F. Control Panel and Results integration | Partly implemented | Dynamic Dataset Package selector for Experiment Core; campaign-results UI still planned |
| G. Screening and decision | Planned | Evidence-based shortlist |
| H. Confirmation, blending, and Kaggle | Planned | Untouched confirmation and justified submission |
| I. AutoGluon r31 gap closure | Deferred | Controlled reproduction after dataset screening |

## 11. Stage B — finish Registry, manifests, and documentation

1. Confirm that `v5`-`v7` are first-class Registry datasets rather than only
   inline `DatasetBuildSpec` definitions in notebook 03.
2. Update when required:
   - `src/churn_ml/dataset_registry/constants.py`;
   - the native catalog and canonical IDs;
   - role assignment;
   - Registry tests.
3. Bring stale documents up to date:
   - `docs/dataset-registry-v1.md` currently describes native versions only
     through `v4`;
   - `docs/project-context.md` currently lists only `v0`-`v3`;
   - `docs/architecture/current-state.md` currently describes notebook 03 only
     through `v3`;
   - `docs/reproducibility/baseline_manifest.yaml` is a historical `v0`-`v3`
     contract.
4. Do not silently rewrite the historical baseline manifest. Add new datasets
   additively or create a separate versioned campaign manifest.
5. Ensure every selected package exposes or preserves:
   - manifest SHA;
   - schema hash;
   - training content hash;
   - target hash;
   - training row-identity hash.
6. Keep dataset discovery dynamic. Do not hardcode `v0`-`v7`.

## 12. Stage C — verify Registry-to-Experiment-Core integration

1. Confirm that `discover_registered_datasets(data/processed)` returns all eight
   valid packages.
2. Confirm that the selector/API exposes:
   - dataset ID;
   - parent ID;
   - feature count;
   - hypothesis;
   - target dependency;
   - schema hash;
   - content hash;
   - target hash;
   - row-identity hash.
3. Verify that `registered_prepared_passthrough_v1`:
   - preserves all prepared features;
   - treats `string`, `object`, and `category` as categorical;
   - does not apply target-dependent preprocessing before a fold;
   - rejects unsupported dtypes and schema collisions.
4. Exercise the real `v5` package and its `missingness_pattern_id`.
5. Run a v3 parity test between:
   - `manual_v3_pipeline_v1_compat`;
   - the Registry-backed route.

The parity test must use identical folds, seeds, and compatibility
preprocessing, and compare ordered transformed schemas, probabilities, and
metrics within a predefined tolerance.

The generic passthrough must not silently reproduce old behavior. If parity
requires dropping:

- `Var214`;
- `Var220`;
- `Var222`;
- `Var218_is_missing`;

encode those drops in an explicit versioned compatibility contract or adapter,
not as hidden behavior in the generic prepared-dataset pipeline.

## 13. Stage D — Dataset Campaign / Matrix Runner

Stop adding new prepared datasets after `v7` until the current hypotheses have
been screened.

**Implementation status:** Dataset Campaign Runner v1 is implemented. See
`docs/dataset-campaign-runner-v1.md` and
`configs/dataset_campaign/templates/`. The runner orchestrates existing Research
v2 single runs; it does not duplicate training. The 21-run screening campaign
itself is **not** complete until a real frozen campaign is executed and audited.

### 13.1 Screening matrix

The unbiased screening set contains seven target-independent packages:

- `v0_raw_minimal`;
- `v1_missingness_summary`;
- `v2_missingness_indicators`;
- `v4_zero_value_summary`;
- `v5_joint_missingness_pattern`;
- `v6_compact_missingness_indicators`;
- `v7_compact_zero_indicators`.

Run `v3_targeted_missingness` separately and label every result as exploratory.

Initial model families:

- LightGBM;
- XGBoost;
- CatBoost.

The primary matrix contains 7 datasets × 3 models = **21 comparable runs**.
Three additional v3 runs may be executed as exploratory results, but must not be
mixed into the unbiased ranking.

### 13.2 Controlled-comparison rules

Within each model family, keep constant:

- outer folds;
- repeat seeds;
- threshold protocol;
- model configuration;
- fold-local categorical preprocessing.

Do not tune each dataset independently during initial screening.

For the first controlled comparison, CatBoost should use the agreed
numeric/OOF-target-encoding path. CatBoost native categorical handling is a
separate later experiment because otherwise model pipeline and dataset change
simultaneously.

### 13.3 Runner contract

The Campaign Runner must:

1. discover datasets dynamically from the Registry;
2. freeze selected dataset IDs and hashes in a campaign manifest before
   training;
3. validate all packages and configurations before allocating run directories;
4. operate on training data only;
5. run sequentially or with safe resource limits;
6. avoid simultaneous heavy CatBoost and AutoGluon jobs under current memory
   pressure;
7. never overwrite an existing run;
8. preserve failed state and completed-run visibility;
9. aggregate:
   - dataset and parent;
   - model;
   - protocol;
   - Balanced Accuracy;
   - sensitivity;
   - specificity;
   - ROC AUC;
   - average precision;
   - Brier score;
   - threshold summary;
   - run directory;
   - hashes and provenance;
10. save aligned OOF rows with:
    - `repeat`;
    - `repeat_seed`;
    - `outer_fold`;
    - `row_position`;
    - `target`;
    - `probability`.

Execution order:

1. validate-only for the full matrix;
2. short smoke campaign;
3. frozen development campaign;
4. untouched confirmation evaluation for the selected shortlist.

Freeze smoke, development, and confirmation plans before observing their
intermediate results.

## 14. Stage E — cross-dataset paired comparison

Paired Comparison v1 requires the same dataset fingerprint and is intended for
models or pipelines on one dataset. Do not weaken that contract.

Create a separate versioned cross-dataset comparison contract. Different
feature schemas and content are allowed, but it must require:

- identical target hash;
- identical training row identity;
- identical outer assignments;
- identical threshold-selection assignments;
- identical seeds and evaluation protocol;
- identical model family/configuration for a parent-child comparison;
- exact alignment of all OOF keys.

Primary parent-child comparisons for each model:

- `v0_raw_minimal` → `v1_missingness_summary`;
- `v0_raw_minimal` → `v2_missingness_indicators`;
- `v0_raw_minimal` → `v4_zero_value_summary`;
- `v1_missingness_summary` → `v5_joint_missingness_pattern`;
- `v1_missingness_summary` → `v6_compact_missingness_indicators`;
- `v4_zero_value_summary` → `v7_compact_zero_indicators`.

The `v1_missingness_summary` → `v3_targeted_missingness` comparison remains
exploratory.

Primary effect:

```text
delta_BA = BA(child) - BA(parent)
```

Also report:

- repeat wins, ties, and losses;
- stability across repeats;
- sensitivity/specificity trade-off;
- threshold stability;
- probability correlation;
- disagreement patterns.

Do not select a dataset from a single maximum Balanced Accuracy value.

## 15. Stage F — Control Panel and Results integration

Completed for Experiment Core v2:

1. Dynamic Registry-backed dataset selector (no hardcoded dataset IDs).
2. Lineage and manifest/provenance fields in the Run page.
3. Visual warning for `target_dependency: exploratory`.
4. Explicit prepare of dataset-driven config/plan under `artifacts/ui_configs`,
   then Validate/Run through the existing allowlisted CLI path.
5. Dataset-aware job labels and persisted job references (`dataset_id`,
   experiment/plan/model/mode).
6. Results Experiments columns/filter/search/charts for Dataset identity;
   Inspect compact provenance; Compare left/right Dataset IDs with
   descriptive-only cross-dataset labeling and stricter Paired Comparison
   prepare readiness.
7. Validated MLflow research provenance params/tags and dataset-prefixed
   deterministic run names, with in-place backfill via `source_key`.

Still planned after the Campaign Runner and comparison contract exist:

1. Campaign Results matrix.
2. Parent-child deltas as a campaign view.
3. Keep filesystem artifacts authoritative and MLflow secondary (already the
   policy; campaign-scale UI still pending).

## 16. Stage G — screening and decision rules

Do not optimize only for the largest observed Balanced Accuracy.

Look for:

- gains repeated across model families;
- stable parent-child gains across repeats;
- a defensible sensitivity/specificity balance;
- threshold stability;
- evidence about the actual contributing feature family:
  - missingness summaries;
  - broad missingness indicators;
  - compact missingness indicators;
  - joint missingness pattern;
  - zero-value summaries;
  - compact zero indicators.

A feature family becomes a confirmation candidate when its gain is stable across
repeats and preferably appears in more than one model family.

A gain limited to one model or one repeat is mixed evidence and must be reported
as such.

## 17. Stage H — confirmation, blending, and Kaggle

1. Freeze the shortlist after screening.
2. Use a separate untouched nested/repeated confirmation plan.
3. Do not alter a candidate after observing interim outer-fold results.
4. Treat any changed candidate as a new experiment.
5. Keep `v3_targeted_missingness` exploratory.
6. Consider blending only from aligned OOF predictions after confirmation.
7. Implement Multi-Blend v2 for cross-dataset/cross-model blending.
8. Begin with fixed blend weights.
9. Do not search dataset, model, preprocessing, threshold, and blend weights in
   one optimization.
10. Select the threshold only after fixing the probability model and deployment
    structure.
11. Build Final Deployment and make one justified submission after the local
    decision is frozen.

Kaggle Public Score is an external benchmark only. Never:

- select features by Public Score;
- tune a threshold through repeated submissions;
- tune blend weights on the leaderboard;
- use Orange labels;
- promote a locally weak candidate solely because of Public Score.

## 18. Known benchmark results

### Historical CatBoost/XGBoost 50/50 blend on v0

- legacy cross-fitted Balanced Accuracy: `0.898839`;
- threshold: `0.130`;
- Kaggle Public Score: `0.8832`.

### AutoGluon extreme on v3

- calibrated internal Balanced Accuracy: `0.895158`;
- threshold: `0.117`;
- Kaggle Public Score: `0.9112`.

### Manual LightGBMPrep reproduction on v3

- Balanced Accuracy at `0.117`: `0.900979`;
- global optimized OOF Balanced Accuracy: `0.904078` at `0.096`, but optimistic;
- legacy cross-fitted estimate: `0.898656`;
- Kaggle Public Score: `0.9023`.

### Research v2 development

- LightGBM: Balanced Accuracy `0.897453`;
- XGBoost: Balanced Accuracy `0.897987`;
- CatBoost Optuna best objective: `0.883898`, not a primary candidate.

Threshold `0.117` is not universal.

## 19. Deferred direction — close the AutoGluon r31 gap

This is not the next task. Revisit it only after dataset screening.

Priority order:

1. arithmetic feature generation;
2. categorical interactions;
3. fold-local OOF encoding;
4. exact `Var211` handling;
5. encoder seed;
6. bag-specific rounds and repeated/eight-fold bagging;
7. threshold selection last.

Known r31 details:

- 49 eligible numeric features;
- up to 2,000 `float32` arithmetic features;
- `+`, `-`, `*`, and `/` operations;
- maximum order 3;
- correlation threshold 0.95;
- approximately 100 categorical pair interactions per bag;
- OOF encoding of approximately 130 categorical columns;
- 5 encoding folds;
- `alpha = 10`;
- `random_state = 0`;
- original categorical columns removed after encoding.

Any target-informed selection or encoding must run inside the outer-training
partition. It must not be materialized as a prepared
`target_dependency: none` dataset.

## 20. Global operating rules

- Do not silently change dataset definitions, feature order, dtypes, seeds,
  folds, thresholds, metrics, or prediction aggregation.
- Do not interpret global optimized OOF Balanced Accuracy as unbiased.
- Do not use competition test data in research evaluation.
- Do not overwrite historical artifacts.
- Do not modify canonical notebooks without an explicit request.
- Do not hardcode the discovered dataset list.
- Do not mix exploratory v3 results into the unbiased ranking.
- Do not perform Git writes without explicit authorization.
- After code changes, report:
  - changed files;
  - commands run;
  - validation results;
  - unresolved risks.
- Work step by step: provide one logical command block, explain it briefly, and
  wait for the result.

## 21. Minimal prompt for a new dialog

Use this instead of pasting the complete project history:

```text
Continue work on `neoversity-ml-final` from
`docs/current-project-status.md`.

Read that document first and treat it as the current operational handoff.
Verify the active branch and Git status before relying on volatile state.

I am manually executing `notebooks/03_feature_engineering.ipynb`.
Start by checking the final output or traceback and
`git status --short --branch`. Then continue with the v0-v7 Dataset Package
audit. Do not move to the Dataset Campaign until all eight packages are
validated.

Work step by step and give PowerShell commands.
```
