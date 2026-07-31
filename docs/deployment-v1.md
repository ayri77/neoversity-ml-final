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
Approval snapshots and hashes are persisted. The deployment config, approval,
completed research tree, threshold evidence, referenced Paired Comparison tree,
synthetic fixture (or confirmed competition test/sample files), and path-chain
identities are authenticated before data/model execution, immediately before
data loading, and immediately before `_SUCCESS`. Any byte, size, recursive tree,
manifest, file identity, or path-chain change prevents success.

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
threshold is a finite float in `[0, 1]`, is byte-for-byte equal to a typed,
size- and SHA-256-authenticated threshold-evidence artifact tied to the completed
run/manifest/plan/candidate, and uses exact `probability >= threshold`. Output is fixed beneath
`artifacts/deployments/<deployment-id>/`, tracking and network access are
disabled, model persistence is false, and existing output is never replaced.

See [the config template](templates/deployment_v1.example.yaml). It contains
placeholders and is intentionally not an approved production selection.

## Generated deployment drafts

Deployment configurations are no longer written by hand. The Control Panel
**Generate submission** step selects a completed canonical Research v2 run and
generates a draft through `control_panel/deployment_draft_builder.py`:

```text
completed Research v2 run
→ deterministic readiness report
→ competition asset authentication (sample submission + row identity)
→ artifacts/deployment_drafts/<run-id>-<manifest-sha256[:12]>[-r<asset-fp8>]/
  ├── deployment_config.yaml     (this deployment_v1 schema)
  ├── threshold_evidence.yaml    (typed evidence artifact)
  ├── candidate_approval.yaml    (approval schema v1)
  └── draft_provenance.json      (run, dataset, model, protocol, threshold links)
```

The builder reads authoritative run sidecars and the immutable Dataset Package
manifest for model/feature/threshold identity. Competition submission IDs come
from a separate authenticated contract (`competition_assets_v1` /
`data/competition/test_row_identity_v1.json` and
`configs/competition/competition_assets_v1.yaml`), not from the feature matrix.
Prepared Dataset Package `X_test.parquet` remains features-only.

Optional additive config key `submission_row_identity` links the draft to that
row-identity artifact (path, SHA-256, expected rows, ID column/dtype/semantics,
ordered ID hash, row-position identity, Dataset Package `test_anchor_hash`).
Legacy synthetic fixtures without this key keep the previous ID-on-test-frame
path.

Drafts are repository-relative, written atomically, idempotent on identical
regeneration, and never silently replaced; a recorded approval is never rewritten.
Unresolved historical drafts (placeholder sample path) remain inspectable.
Resolved drafts use a deterministic `-r<asset-fingerprint[:8]>` revision so the
unresolved draft is not overwritten.

A generated draft is accepted by `load_deployment_config` and by
`run_deployment_v1.py validate`, and can be rehearsed with
`run_deployment_v1.py dry-run` against an approved synthetic fixture.

Local **Generate submission** (`run` + `--allow-competition-test`) is enabled in
the registry only with acknowledge confirmation, a resolved draft, authenticated
sample-submission identity, authenticated test-row identity, no-overwrite output
under `artifacts/deployments/<deployment_id>`, and `runtime.network_enabled: false`.
It does not upload to Kaggle.

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

Before success, P4 verifies exact test/sample row counts, exact ID order, and
independent ID validity. Integer IDs use the accepted exact integer input dtype;
string IDs must be homogeneous, nonmissing, unique, nonempty, and not whitespace
only. Values are never trimmed or normalized. The expected submission is rebuilt
only from authenticated ordered IDs and recomputed exact int8 labels. Creation
and validation share one canonical UTF-8 CSV writer (comma separator, minimal
quoting, LF terminator, no BOM, no index, exact column order). Raw bytes must
match exactly; parsed-equivalent quoting, spaces, CRLF, BOM, blank lines, float
label text, or an extra/reordered column is rejected. The runner writes but never
uploads `submission.csv`; any subsequent Kaggle upload is a separate manual
action.

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

## Review-hardening invariants

`approval_id` is the canonical SHA-256 of immutable approval fields. Component
name, approver, and approval timestamp are display/audit metadata and are the
only excluded fields; changing research, parameters, evidence, comparison,
status, role, or component identity changes the ID. Bag seeds are canonicalized
in ascending numeric order and float64 probabilities are accumulated in that
order, so seed-list permutations produce identical resolved identities and
averages.

Every test row carries its authenticated ID through each per-bag vector,
component average, fixed blend, prediction, and submission. Duplicate or null
IDs are rejected independently in test and sample inputs, and equal-length row
permutations are rejected. Completed validation rebuilds schemas, feature and
encoding identities, OOF assignments, full-data mapping semantics, per-bag
parameter/probability identities, component averages, the blend, threshold
labels, raw submission tokens, environment, provenance, runtime state, and
terminal lifecycle from authoritative inputs. Rebuilding the inventory and
manifest cannot legitimize semantically altered content.

### Non-following path-chain validation

One `validate_path_chain` primitive owns canonicalization. It first joins portable
repository-relative components syntactically, then applies `lstat`, Windows
reparse attributes, `Path.is_junction()` where available, symlink, mount-like,
kind, and policy-specific hardlink checks to every existing component from the
filesystem anchor through the containment root and requested leaf. Only that
validated object may call `resolve()` for the final canonical containment check.
A not-yet-existing output authenticates every existing parent and resolves only
the nearest already-validated parent. Recursive inspection uses non-following
`os.scandir` plus per-entry `lstat` and rejects before descending. Raw paths are
never resolved again downstream. This applies equally to config, approval,
research, comparison, threshold, fixture, test/sample, CLI, output, completed
artifact, inventory/manifest, and terminal-marker paths, including a junction
two or more ancestors above a normal leaf.

### Exact physical artifacts and terminal time

Authoritative Parquet and CSV tables have versioned physical schemas: exact
column set/order, primitive signed dtype, nonnullable status, finite/range or
enum constraints, and exact row identity. Validation occurs before any cast,
`int()`, narrowing, or NumPy dtype conversion. Row positions, folds, bag indices,
seeds, counts, and predictions cannot be floats, booleans, strings, nullable
integers, or unsigned alternatives; persisted predictions are exact int8
`{0,1}` and probabilities are nonnullable finite float64 in `[0,1]`. Threshold
labels are recomputed directly from those float64 values using exact `>=`.

Every authoritative deployment JSON object is checked recursively before
inventory/manifest authentication and before semantic equality: exact key sets,
list shape, primitive types (`type(value) is expected_type`), nullable policy,
finite floats, integer/boolean separation, and field-specific enum/range rules.
This applies to deployment, input, dataset, fixture, feature, encoding,
component, prediction, runtime, environment, provenance, inventory/manifest,
and terminal records. Resolved configuration and approval snapshots (including
threshold and paired-comparison evidence) receive the same recursive exact-type
comparison against their authenticated authorities. Python equality therefore
cannot equate `true`, `1`, and `1.0` at this boundary.

`bag_summary.csv` has a dedicated canonical writer and raw-byte reader. Its only
accepted representation is UTF-8 without BOM, LF endings, the fixed v1 header,
fixed row order, comma delimiters, no quoting, canonical nonnegative integer and
`repr(float)` duration tokens, `True`/`False` booleans, one final newline, and no
blank line. Validation preserves lexical provenance and reconstructs the exact
canonical bytes before pandas materialization, so quoted numerics, alternate
number spelling, whitespace, CRLF, BOM, quoting, and column reordering fail even
when a permissive CSV parser could infer the same values.

Runtime authority is limited to the exact schema/version, deployment ID, status,
mode, canonical UTC timestamps, wall-clock duration derived from those
timestamps, component/bag counts, model-persistence flag, and configured
competition/network/tracking isolation flags. Timestamps serialize only as
`YYYY-MM-DDTHH:MM:SS.ffffffZ`, must round-trip, and satisfy start <= finish;
`duration_seconds` must exactly equal their difference. `_SUCCESS` has an exact
schema, matches deployment ID/status/manifest and the runtime completion string,
is mutually exclusive with `_FAILED`, and is the newest file. `_FAILED` also has
an exact schema and canonical UTC timestamp.

The read-only production loader reconstructs config, approvals, Experiment Core
v2 run, threshold/comparison evidence, fixture/data, manifests, and semantics
from disk. Coherently rebuilding inventory, manifest, and `_SUCCESS` cannot make
physical-type, runtime, timestamp, terminal, canonical-CSV, identity, or model
semantic corruption valid. The adversarial suite exercises this public loader
without injecting prevalidated deployment objects or data.

All repository inputs and artifact trees are inspected without following links.
Symlinks, Windows junctions/reparse points, mount-style escapes, multiply linked
files, containment escapes, and input/output overlap are rejected before reads
or writes. Dry-run fixtures must be repository-contained, synthetic,
noncompetition trees containing exactly four files plus
`fixture_manifest.yaml`; every file has an exact role, path, byte size, and
SHA-256, and the fixture records generator identity and seed. Competition hashes
and linked or extra files are prohibited. See
[the fixture manifest template](templates/deployment_synthetic_fixture_v1.example.yaml)
and [the threshold evidence template](templates/deployment_threshold_evidence_v1.example.yaml).