# Dataset Campaign / Matrix Runner v1

Versioned orchestration for Registry × model-family screening matrices.
Filesystem Research v2 artifacts remain authoritative. This runner does **not**
duplicate model fitting, thresholding, metrics, or MLflow indexing logic; it
calls the existing single-experiment Research v2 path for every matrix cell.

## Stage boundaries

| Stage | This document |
|---|---|
| D — Campaign / Matrix Runner | Implemented here (contract + CLI + tests) |
| E — Cross-dataset paired comparison | **Not** implemented; separate versioned contract |
| F — Campaign Results UI | **Not** implemented |
| G/H — Ranking, confirmation, blending, Kaggle | **Not** implemented |

Implementation and tests alone do **not** mean the 21-run screening campaign has
been executed. Do not mark Stage D execution complete until a real frozen
campaign has been run and audited.

## Unbiased screening matrix

Exact target-independent set:

1. `v0_raw_minimal`
2. `v1_missingness_summary`
3. `v2_missingness_indicators`
4. `v4_zero_value_summary`
5. `v5_joint_missingness_pattern`
6. `v6_compact_missingness_indicators`
7. `v7_compact_zero_indicators`

Initial model families: LightGBM, XGBoost, CatBoost.

Primary development matrix: **7 × 3 = 21** comparable runs.

`v3_targeted_missingness` is exploratory only. It must never be silently
included in, ranked with, or relabelled as part of the unbiased matrix. Use a
separate exploratory campaign (see templates).

## Controlled-comparison invariants

Within each campaign, every model family must share the same:

- outer-fold protocol;
- repeat seeds;
- threshold-selection protocol;
- evaluation-plan semantics;
- per-family model configuration (referenced, not retuned per dataset);
- fold-local categorical preprocessing contract from that configuration.

The first CatBoost screening path uses the existing numeric / fold-local OOF
target-encoding configuration (`catboost_numeric_v1`). Native CatBoost
categorical handling is a later, separate experiment.

## Contract versions

| Artifact | `schema_version` |
|---|---|
| User-facing campaign specification | `dataset_campaign_v1` |
| Frozen resolved manifest | `dataset_campaign_manifest_v1` |
| Mutable runtime status | `dataset_campaign_status_v1` |
| Descriptive summary | `dataset_campaign_summary_v1` |

Unknown keys, duplicate datasets/models, unknown Registry IDs, unsafe paths,
missing packages/configs/plans, `v3` mixed into unbiased campaigns,
family/config mismatches, and conflicting protocol identities are rejected.

### Specification shape

```yaml
schema_version: dataset_campaign_v1
campaign:
  id: screening_development_v1
  name: Unbiased development screening matrix
  type: development   # smoke | development | confirmation | exploratory
  classification: unbiased  # unbiased | exploratory
datasets:
  preset: unbiased_screening_v1   # or explicit ids: [...]
models:
  - family: LightGBM
    config_path: configs/research_v2/manual_lightgbm_te_v1_compat_development.yaml
  - family: XGBoost
    config_path: configs/research_v2/xgboost_numeric_v1_development.yaml
  - family: CatBoost
    config_path: configs/research_v2/catboost_numeric_v1_development.yaml
evaluation_plan:
  path: configs/research_v2/plans/telecom_v3_development_r2x5_t3_v1.yaml
execution:
  policy: sequential
  processed_root: data/processed
  artifacts_root: artifacts/dataset_campaigns
  index_mlflow: true
  mlflow_config_path: configs/mlflow/local.yaml
```

Templates live under `configs/dataset_campaign/templates/`. They are examples;
copy and keep explicit configuration references. Do not invent model parameters.

## Frozen identity vs mutable status

Before training, the runner resolves and freezes:

- Dataset IDs, parents, target dependencies, package hashes;
- model family, adapter id, config path/hash;
- evaluation-plan path/hash and protocol identity;
- deterministic `cell_id` values and execution order;
- campaign contract version and `manifest_hash`.

`campaign_manifest.json` is immutable. Runtime updates go only to
`campaign_status.json`. Status writes never change `manifest_hash`. Resume
rejects manifest drift.

Campaign layout:

```text
artifacts/dataset_campaigns/<campaign_id>/
  campaign_manifest.json      # frozen identity
  campaign_status.json        # mutable runtime
  campaign_summary.json       # descriptive aggregation only
  prepared/                   # Registry-backed Research v2 config/plan pairs
    <experiment_id>.yaml
    plans/<plan_id>.yaml
```

Prepared configs use `registered_prepared_passthrough_v1`. Experiment Core run
directories remain under `artifacts/research_v2/...` with existing no-overwrite
semantics.

## Lifecycle distinctions

| State | Meaning |
|---|---|
| Validate-only | Resolve/validate every cell; no Research v2 run dirs; no jobs |
| Frozen campaign | Immutable manifest persisted; execution not implied |
| Running campaign | Sequential cell execution from the frozen manifest |
| Completed screening | All cells succeeded under that frozen identity |
| Exploratory `v3` | Separate classification; never mixed into unbiased ranking |
| Stage E comparison | Future parent-child OOF alignment contract; not this runner |

## CLI

```powershell
# Validate-only (no Experiment Core run allocation)
uv run python scripts/run_dataset_campaign.py validate `
  --config configs/dataset_campaign/templates/smoke_subset_v1.example.yaml

# Validate and freeze the immutable manifest
uv run python scripts/run_dataset_campaign.py validate `
  --config configs/dataset_campaign/templates/development_screening_v1.example.yaml `
  --freeze

# Inspect frozen identity
uv run python scripts/run_dataset_campaign.py inspect `
  --campaign-dir artifacts/dataset_campaigns/screening_development_v1

# Execute (validates/freezes if needed, then runs sequentially)
uv run python scripts/run_dataset_campaign.py run `
  --config configs/dataset_campaign/templates/development_screening_v1.example.yaml

# Resume from the exact frozen manifest
uv run python scripts/run_dataset_campaign.py resume `
  --campaign-dir artifacts/dataset_campaigns/screening_development_v1

# Status / descriptive summary
uv run python scripts/run_dataset_campaign.py status `
  --campaign-dir artifacts/dataset_campaigns/screening_development_v1
```

Exit codes: `0` success, `2` config, `3` validation, `4` manifest drift,
`5` execution/partial failure.

## Execution and recovery

- Deterministic ordering: model-family order as declared, then dataset order.
- Sequential by default; no concurrent CatBoost/AutoGluon work.
- Training data only (Research v2 train-only path).
- Never overwrite an existing Research v2 run directory.
- Preserve completed and failed attempts.
- Resume skips only cells already `succeeded` under the exact frozen identity.
- Failed cells keep prior attempts; retry creates a new attempt.
- Automatic MLflow indexing failures do not rewrite training success.

## Aggregated fields

Descriptive summary only (no official paired inference, no winner selection):

- dataset / parent / target_dependency;
- model family / adapter / protocol hash;
- Balanced Accuracy, sensitivity, specificity, ROC AUC, average precision,
  Brier score;
- threshold summary;
- run directory and package hashes;
- OOF reference (`predictions/outer_validation.parquet`) with columns
  `repeat`, `repeat_seed`, `outer_fold`, `row_position`, `target`, `probability`.
  Research v2 writes `repeat` and `outer_fold` as **1-based** identifiers
  (`enumerate(..., start=1)`). `row_position` is the training-row identity used
  for alignment. Stage D must not reinterpret these as 0-based indices.

## Related documents

- `docs/current-project-status.md` — operational roadmap Stage D/E
- `docs/dataset-registry-v1.md` — Registry package contract
- `docs/research-v2-paired-comparison.md` — same-dataset paired comparison
- `docs/mlflow-local-index.md` — optional searchable mirror
- `docs/experiment-control-panel.md` — single-job UI (campaign UI is Stage F)
