# Final technical audit

**Audit date:** 2026-08-02

**Repository:** `ayri77/neoversity-ml-final`

**Finalization branch:** `feature/prepared-dataset-pipeline-v1`

## Scope and safety

The finalization was reporting, verification, documentation, and publication only.
No model was trained or refit, no AutoGluon run or blend campaign was executed, no
Optuna study was started, and nothing was uploaded to Kaggle. Historical datasets,
candidate packages, blends, submissions, and MLflow state were not moved, deleted,
relabeled, or rewritten.

A local lightweight safety tag, `safety-pre-finalization-2026-08-02`, points to
the initial audited HEAD `86f90424ab775e8e5567bbe063e6164dcd21d6dd`.

## Initial worktree audit

- Branch: `feature/prepared-dataset-pipeline-v1`
- Initial HEAD: `86f90424ab775e8e5567bbe063e6164dcd21d6dd`
- Origin: `https://github.com/ayri77/neoversity-ml-final.git`
- Actual remote default ref after fetch: `origin/master`
- Feature branch divergence from upstream: 0 ahead / 0 behind
- Feature branch relative to `origin/master`: 90 commits ahead / 0 behind
- Initial staged, unstaged, and untracked state: clean
- Ignored local artifacts were preserved.

## Compact evidence verification

```powershell
uv run ruff check scripts/export_final_project_evidence.py
uv run python scripts/export_final_project_evidence.py
```

Passed. The exporter verified 15/15 immutable identities: eight candidate
manifest SHA-256 values, two campaign plan hashes, and five submission CSV hashes.
It read metadata only and did not load predictions or models.

## Notebook and charts

```powershell
uv run python scripts/generate_final_report_assets.py
uv run jupyter nbconvert --to notebook --execute notebooks/09_final_project_report.ipynb --inplace --ExecutePreprocessor.timeout=120
```

Passed. All cells completed; no traceback or error output remains. The notebook
contains no machine-specific absolute path, trains nothing, and is about 281 KB.
Its tables and two charts are generated from tracked compact evidence.

## Control Panel screenshots

The existing Streamlit app was started headlessly on port 8501 and inspected at
1440×900. No job, training, archive/delete, or upload action was triggered.

- `docs/images/control-panel-run.png`
- `docs/images/control-panel-results.png`
- `docs/images/control-panel-compare.png`
- `docs/images/control-panel-blend.png`
- `docs/images/control-panel-submission.png`

The process was stopped and its two temporary system logs were removed.
Captures show rendered data, no traceback, and no personal absolute path.

## Static validation

```powershell
uv lock --check
uv run ruff check .
git diff --check
```

- `uv lock --check`: passed (200 packages resolved).
- Repository-wide Ruff: passed. Four existing intentional post-bootstrap imports
  in `scripts/run_experiment.py` were mechanically annotated `# noqa: E402`;
  runtime behavior is unchanged.
- `git diff --check`: passed.

The publication audit also checks required files, relative links/images, notebook
errors, machine-specific paths, final/screenshot markers, conflict markers,
forbidden artifact patterns, files over 5 MB, and common secret signatures.

## Tests

### Complete suite

```powershell
uv run pytest
```

Result after 13m42s: **1,421 passed, 19 skipped, 55 failed, 109 setup errors**
from 1,604 collected tests.

This is not reported as a passing full suite. Two representative causes were
isolated:

1. `test_paired_comparison_renders_baseline_candidate_selectors` reaches the
   Streamlit AppTest 10-second timeout while reading the unusually large local
   artifact workspace.
2. Optuna authorization fixtures fail closed because a frozen evaluation-plan
   `metadata` hash disagrees with the current local
   `v3_targeted_missingness/dataset_manifest.json` content hash.

The second condition is persisted historical state. The audit did not rewrite the
plan or Dataset Package merely to make tests pass. Failures concentrate in
Control Panel/Research/Final/MLflow/Optuna environment- and artifact-state tests.

### Broad artifact-safe core subset

```powershell
uv run pytest tests/test_features.py tests/test_dataset_registry.py tests/test_registry_experiment_core_phase1.py tests/test_blend_campaign_v1.py tests/test_prediction_candidate_v1.py tests/test_prediction_candidate_explicit_export_v1.py tests/test_results_schema_aware_v1.py
```

Result: **105 passed in 17.60s**.

## Repository and security hygiene

- Raw competition data is not included in the publication set.
- Predictors, model binaries, OOF/test arrays, submission CSVs, MLflow state,
  virtual environments, server logs, and browser profiles are not included.
- No intended publication file exceeds 5 MB; screenshots are compact PNGs.
- No common Kaggle/GitHub/AWS/private-key secret signature is present.
- No force push or history rewrite is used.

## Documentation validation

The READMEs link to each other, notebook 09, compact evidence, reproducibility,
audit, charts, and screenshots. Stale final-notebook and screenshot markers were
removed. Both distinguish internal protocols from Public Score, keep `v3`
exploratory, and make no Private Score or final-rank claim.

## Known residual risk

The complete suite is not clean under this populated local artifact state. The
state/hash and UI-timeout blockers are documented above. Resolving the hash
condition requires an explicit compatibility decision for historical frozen plans;
silently rewriting evidence is prohibited. This does not block repository-only
execution of the report or the audited five-submission evidence bundle.
