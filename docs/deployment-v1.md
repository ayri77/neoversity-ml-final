# P4 fixed deployment framework v1

## Boundary and policy

Deployment v1 is the only P4 path. P0 smoke, P1 development, P2 selection, and
P3 confirmation remain train-only. `validate` authenticates configuration,
manual approvals, completed Experiment Core v2 research runs, identities, and
manifests without reading competition test data, reading a sample submission,
allocating output, or fitting a model.

P4 never tunes parameters, searches blend weights, calibrates a threshold,
reads Optuna storage, reads Kaggle or leaderboard information, chooses a
candidate, uploads a submission, or feeds an external score back into research.
Every candidate, parameter mapping, seed list, component weight, and threshold
is a fixed input approved before execution.

Supported research adapter identities are:

- `manual_lightgbm_te_v1_compat`;
- `xgboost_numeric_v1`;
- `catboost_numeric_v1`.

Their Experiment Core v2 research identities are unchanged. Deployment uses a
separate full-data protocol and reuses their exact validated parameter
contracts and approved runtime-library versions.

## Manual candidate approval

An approval is a strict schema-v1 artifact. It authenticates a completed,
semantically valid Experiment Core v2 run by repository-relative path, run and
manifest identities, evaluation plan, dataset, pipeline, adapter, and complete
candidate hashes. It records exact resolved model parameters, threshold
evidence, approval status, approver, UTC timestamp, and intended role.

Approval is never inferred from a metric or a “best” run. `revoked` and
unapproved states are rejected. A missing Paired Comparison is accepted only
when the artifact explicitly grants an exception and records its reason. When
a comparison is referenced, its complete lifecycle and manifest are
revalidated and the approved run must be one of its authenticated inputs.
Approval snapshots and hashes are persisted; approval and research trees are
authenticated before and after execution and are never modified.

See [the approval template](templates/deployment_candidate_approval_v1.example.yaml).

## Deployment configuration

The schema-v1 config has exact recursive sections for deployment identity,
dataset and pipeline IDs, components, blend, threshold, bagging, competition
test data, sample submission, output, and runtime. Unknown keys, bool-as-int,
non-finite values, duplicate components or seeds, path traversal, optimization
fields, unsupported adapters, and parameter differences from approval are
rejected.

Weights are finite nonnegative floats and must sum to one within the fixed
absolute tolerance `1e-12`. A single component with weight `1.0` is valid. The
threshold is a finite float in `[0, 1]`, has a mandatory evidence reference,
and uses exact `probability >= threshold`. Output is fixed beneath
`artifacts/deployments/<deployment-id>/`, tracking and network access are
disabled, model persistence is false, and existing output is never replaced.

See [the config template](templates/deployment_v1.example.yaml). It contains
placeholders and is intentionally not an approved production selection.

## Full-data features and bagging

After all configuration, approval, completed-run, manifest, and identity checks
succeed, execution loads the approved full training data and the explicitly
configured test and sample paths. Train/test ordered names, dtypes, RangeIndex,
row counts, and test/sample ID order are validated before feature preparation.
The approved pipeline contract is applied independently to train and test.

Categoricals use deterministic internal stratified folds. Training features
receive OOF target encodings. The encoder then fits its retained full-training
mapping and transforms test rows; it never fits on test. Missing and unknown
categories use the approved encoder fallback. Numeric NaNs remain native for
XGBoost and CatBoost, infinities are rejected, and no imputation occurs.
Encoding assignments, mapping identities, matrix hashes, and ordered feature
identities are persisted.

Each configured bag constructs one estimator with the approved fixed
parameters. Only the adapter's seed field changes. Every estimator fits all
training rows, without an outer holdout, evaluation set, early stopping, or
callbacks. Positive-class probability is selected by estimator class label,
bag probabilities are averaged arithmetically, and no estimator binary is
written.

The fixed blend operates only on aligned persisted component probability
vectors. It is recomputed during semantic validation before success.

## Submission and lifecycle

Before success, P4 verifies exact test/sample row counts, exact ID order, exact
two-column submission schema, binary nonmissing labels, absence of a CSV index
column, deterministic positive count/rate, and an exact CSV round trip. The
runner writes but never uploads `submission.csv`; any subsequent Kaggle upload
is a separate manual action.

Ordinary files use atomic replacement. A deployment has one exact recursive
artifact set, approval snapshots, SHA-256 inventory and manifest, and one
terminal marker. Semantics are recomputed before `_SUCCESS`; the marker is
written last and must be newest. Failure after allocation writes `_FAILED`.
No store writes are allowed after success. Typed read-only completed and failed
loaders provide a future MLflow indexing boundary; filesystem artifacts remain
authoritative and current MLflow modules are unchanged.

## CLI

```powershell
.venv\Scripts\python.exe -u scripts/run_deployment_v1.py validate --config <path>
.venv\Scripts\python.exe -u scripts/run_deployment_v1.py dry-run --config <path> --fixture-dir <path> --output-dir <temporary-path>
.venv\Scripts\python.exe -u scripts/run_deployment_v1.py run --config <path> --allow-competition-test
.venv\Scripts\python.exe -u scripts/run_deployment_v1.py inspect --deployment-dir <path>
```

Relative paths resolve against the repository root. `dry-run` requires explicit
noncompetition Parquet/CSV fixtures and a caller-provided new output directory.
`run` refuses to read competition assets unless the confirmation flag is
present. Output is one JSON object; exit codes are `0` success, `2` contract or
artifact validation, `3` safety refusal, and `4` execution failure.
