# Dataset Package & Registry v1

Strict, versioned dataset-package contract and registry for prepared competition
datasets. This is an additive extension of the experiment platform. It does
**not** change Research v2 loaders, notebooks, Control Panel, MLflow, models, or
historical artifacts.

## Package layout

Each registered package is a directory under a registry `--root`:

```text
<root>/<dataset_id>/
  X_train.parquet
  y_train.parquet
  X_test.parquet
  metadata.json              # detailed feature-engineering + git provenance
  dataset_manifest.json      # sole canonical Registry contract
```

`dataset_manifest.json` is the only persisted reproducibility/lineage contract.
Generator-owned `manifest.json` is not written, not trusted, and not a Registry
contract. Author-supplied fields are represented by `DatasetBuildSpec`; all
calculated fields are produced by Registry code.

## Native generation versus legacy backfill

- **Native generation** (`features.save_dataset`): writes the four package
  artifacts, then materializes `dataset_manifest.json` through Registry
  hashing, role assignment, alignment proof, and strict validation.
- **Legacy backfill** (`dataset_registry` CLI `backfill`): adds
  `dataset_manifest.json` to existing four-file packages after proving
  lineage/alignment. Dry-run by default; never overwrites an existing
  Registry manifest; never modifies parquet/`metadata.json`.

## Manifest fields and semantics

`schema_version` is always `dataset_package_v1`. Unknown keys and missing
required keys fail validation.

| Field | Meaning |
|---|---|
| `dataset_id` | Immutable package identity; must match the directory name |
| `parent_dataset_id` | Immediate parent package id, or `null` for a root package |
| `hypothesis` | Why this package exists |
| `files` | Relative names of the four package artifacts |
| `train_row_count` / `test_row_count` | Row counts |
| `n_features` / `features` | Ordered feature names, dtypes, and roles |
| `transformations` | Declared transform records (`type` required) |
| `target_dependency` | `none` \| `exploratory` \| `fold_local` |
| `schema_hash` | Hash of ordered `{name,dtype,role}` entries |
| `content_hashes` | File SHA-256 for `X_train`, `y_train`, `X_test`, `metadata` |
| `target` | Name, dtype, class counts, and value hash |
| `row_identity` | Train/test content hashes plus alignment proof fields |

## Feature roles

Exact allowed roles:

- `numeric` — retained numeric/bool source columns after catalog checks
- `categorical` — retained object/category/string source columns
- `binary_indicator` — declared or catalog-known missingness indicator columns
- `summary` — declared missingness / zero-value summary columns

Role assignment is deterministic and never inspects `y_train` values, `X_test`
values, correlations, or cardinality. Precedence (highest first):

1. `summary` (declared name set)
2. `binary_indicator` (declared name set, or legacy v2 `_is_missing` suffix rule)
3. `categorical` (train dtype is object/category/string)
4. `numeric` (remaining supported numeric/bool train dtypes)

Unsupported dtypes fail classification. Stale roles such as `feature` or
`engineered` fail strict manifest parsing.

Roles are part of the immutable schema hash. Reordering columns, changing a
dtype, or changing a role requires a new `dataset_id`.

Native engineered roles:

- v1: eight missingness summaries → `summary`
- v2: four count summaries → `summary`; generated `_is_missing` → `binary_indicator`
- v3: eight missingness summaries → `summary`; four targeted indicators → `binary_indicator`
- v4: four zero-value aggregates → `summary` (`parent=v0_raw_minimal`, `target_dependency=none`)

## `target_dependency` meanings

- `none`: package construction does not depend on the training target
- `exploratory`: construction used a prior full-train / exploratory target-informed choice (legacy `v3_targeted_missingness`)
- `fold_local`: recognized by the Registry schema, but **not** materializable by
  `save_dataset()`. Fold-local target-dependent transforms belong inside a CV
  pipeline, not a ready prepared Parquet package.

## Immutability

Registered packages are immutable. Validation recomputes protected properties
from on-disk files and fails when any of the following diverge from the
manifest:

- file content
- schema / dtype / feature order
- row order / row counts
- target values or class counts
- metadata file content
- row-identity / alignment hashes

A modified package must receive a new `dataset_id`. `save_dataset()` refuses to
overwrite a registered package even when `overwrite=True`. Backfill never
overwrites an existing `dataset_manifest.json` and never modifies parquet or
`metadata.json`. Git provenance is stored under `metadata.json` as
`created_from_git` (commit + dirty flag), not as an unknown Registry manifest
field.

## Row-identity proof

A manually authored “aligned” flag is never trusted.

Train row identity is derived from ordered feature values, **not** from `y`
alone. For the implemented native packages
(`v0_raw_minimal`, `v1_missingness_summary`, `v2_missingness_indicators`,
`v3_targeted_missingness`, `v4_zero_value_summary`), the ordered
`v0_raw_minimal` feature projection is the alignment anchor because those
packages retain that projection with unchanged values and row order.

Legacy `metadata.json` stores lineage under nested
`source` / `base_version` (and baseline docs use `parent`). It does **not**
store an explicit row-identity key; the v0 projection is therefore the
verifiable identity assumption for safe backfill.

If alignment cannot be proven, status is `unverifiable_alignment` and
registration/backfill is refused.

## CLI

```powershell
.venv\Scripts\python.exe -u scripts\run_dataset_registry.py scan --root <processed-root>
.venv\Scripts\python.exe -u scripts\run_dataset_registry.py validate --root <processed-root>
.venv\Scripts\python.exe -u scripts\run_dataset_registry.py validate --root <processed-root> --dataset-id v0_raw_minimal
.venv\Scripts\python.exe -u scripts\run_dataset_registry.py inspect --root <processed-root> --dataset-id v1_missingness_summary
.venv\Scripts\python.exe -u scripts\run_dataset_registry.py backfill --root <processed-root>
.venv\Scripts\python.exe -u scripts\run_dataset_registry.py backfill --root <processed-root> --write
```

Every command requires explicit `--root`. Backfill is dry-run unless `--write`
is provided.

### Exit statuses

| Status | Exit code |
|---|---:|
| `valid` | 0 |
| `invalid` | 2 |
| `unregistered` | 3 |
| `unverifiable_alignment` | 4 |
| `refused_overwrite` | 5 |

## Safe legacy backfill

Canonical backfill targets:

- `v0_raw_minimal` (`target_dependency: none`)
- `v1_missingness_summary` (`target_dependency: none`)
- `v2_missingness_indicators` (`target_dependency: none`)
- `v3_targeted_missingness` (`target_dependency: exploratory`)

Intentionally **unregistered** obsolete names:

- `v1_basic_clean`
- `v2_eda_features`
- `v3_native_categorical`

Dry-run first, then write only after inspection:

```powershell
.venv\Scripts\python.exe -u scripts\run_dataset_registry.py scan --root data\processed
.venv\Scripts\python.exe -u scripts\run_dataset_registry.py backfill --root data\processed
.venv\Scripts\python.exe -u scripts\run_dataset_registry.py backfill --root data\processed --write
.venv\Scripts\python.exe -u scripts\run_dataset_registry.py validate --root data\processed --dataset-id v0_raw_minimal
.venv\Scripts\python.exe -u scripts\run_dataset_registry.py validate --root data\processed --dataset-id v1_missingness_summary
.venv\Scripts\python.exe -u scripts\run_dataset_registry.py validate --root data\processed --dataset-id v2_missingness_indicators
.venv\Scripts\python.exe -u scripts\run_dataset_registry.py validate --root data\processed --dataset-id v3_targeted_missingness
```

## Public API

```python
from src.churn_ml.dataset_registry import (
    list_dataset_packages,
    load_dataset_manifest,
    resolve_dataset_package,
    validate_dataset_package,
)
```

`resolve_dataset_package(root, dataset_id)` returns a validated package object
for later pipeline integration.

## Reserved future integration points

Out of scope for Registry v1; reserved for later packages:

- Generic Prepared-Dataset Pipeline (`prepared_dataset_v1`)
- Dataset Campaign / Matrix Runner
- Cross-Dataset Paired Comparison
- Multi-Blend v2
- MLflow / Control Panel wiring

Do not load datasets through this registry from Research v2 yet; keep existing
research dataset loading unchanged until a dedicated integration task.
