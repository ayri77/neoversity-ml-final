# Blend Campaign Runner v1

## Purpose and architecture

Blend Campaign Runner v1 is a CLI-first orchestration layer over the existing
canonical probability-blending backend. It reads only immutable
`prediction_candidate_v1` packages, calls `load_compatible_candidates` and
`search_blend`, ranks honest repeated meta-CV evidence, and calls the existing
materialization and local-submission backends only for frozen selections.

The runner does not invoke Streamlit, retrain base models, load AutoGluon
predictors, generate base-model predictions, mutate candidate packages, contact
Kaggle, or upload submissions. Candidate packages and compatibility matrices
are cached per ordered candidate set within one process. Diversity diagnostics
are produced by the canonical blend backend.

The main entry point is:

```text
scripts/run_blend_campaign.py
```

Campaign artifacts are written under:

```text
artifacts/blend_campaigns/<campaign_id>/
```

## Commands

- `validate --config`: strict read-only validation. Resolves aliases only to the
  exact candidate IDs in the YAML, validates every candidate set, freezes
  candidate manifest hashes in memory, and calculates deterministic experiment
  and plan identities. It writes nothing.
- `plan --config`: performs validation and writes the immutable frozen config
  and plan plus empty mutable campaign indexes. It does not run a search.
- `run --config`: plans a new campaign, searches every ordered experiment,
  records successes and failures, ranks successful searches, then applies the
  frozen materialization and optional submission policies.
- `resume --campaign-dir`: revalidates the frozen plan against current exact
  candidate IDs and manifest hashes, skips matching successful searches,
  recovers an immutable completed search-result handoff when present, and
  retries failed or interrupted searches.
- `inspect --campaign-dir`: reads the frozen plan, status, results, failures,
  materialization handoffs, and submission handoffs.
- `materialize-top --campaign-dir [--top-k N]`: after all searches are terminal,
  materializes the explicit frozen selections plus the requested top-K.
- `generate-submissions --campaign-dir`: when enabled in the frozen policy,
  calls the validated local submission backend for materialized canonical
  candidates. It never uploads.

Stdout is JSONL so progress can be consumed by another process. Concise status
messages go to stderr.

## Configuration

The schema version is `blend_campaign_v1`. The complete example is
`configs/blend_campaigns/exploratory_competition_v1.yaml`.

Top-level fields are strict:

- `campaign_id`;
- repository-relative candidate, blend, and submission roots;
- `candidates`, a mapping from human aliases to exact immutable candidate IDs;
- an ordered `experiments` list;
- `materialize_policy`;
- `submission_policy`.

Every experiment defines its ID, ordered aliases, strategy, optimizer, folds,
repeats, seed, active-model limit, manual weights, native search budget, Optuna
budget, threshold policy, and exploratory flag. Unknown keys and duplicate
experiment IDs are rejected. Different experiment IDs with identical candidate
hashes and settings are also rejected.

Aliases are never resolved by fuzzy model names. Validation fails if an exact ID
is absent. A campaign also fails validation when its explicit exploratory flag
does not exactly match propagation from its canonical source candidates.

The established blend default threshold grid is unchanged. Campaigns may opt
into an explicit finer grid, such as `0.05` through `0.30` in `0.001` steps.
That policy becomes part of the experiment identity and is passed unchanged to
the canonical blend backend.

## Artifact contract

Each campaign directory contains:

- `frozen_config.yaml`: immutable parsed campaign configuration;
- `campaign_plan.json`: exact alias resolution, candidate manifest hashes,
  ordered experiments, expected blend IDs, experiment identities, and plan hash;
- `campaign_status.json`: resumable per-experiment state and attempt count;
- `experiment_results.csv` and `experiment_results.json`: one record for every
  successful or failed search;
- `failures.json`: search, materialization, and submission failures;
- `ranking.csv`: successful searches in honest-evidence order;
- `selected_for_materialization.json`: selection policy and canonical blend/
  candidate handoffs;
- `generated_submissions.json`: local generation handoffs, always declaring
  `network_access: false` and `kaggle_upload: false`;
- `environment.json`: Python, platform, and relevant dependency versions;
- `search_results/<experiment_id>.json`: immutable per-search resume handoff;
- `_SUCCESS`: written only after all other campaign indexes for the operation.

The frozen config, plan, environment, and per-search handoffs use conflict-on-
change semantics. Status and aggregate indexes are atomic mutable state needed
for resume. Canonical blend, candidate, and submission backends retain their own
immutable no-overwrite contracts.

## Ranking semantics

Only successful searches are ranked. The deterministic ordering is:

1. higher mean repeat Balanced Accuracy from honest meta-CV;
2. lower sample standard deviation of repeat Balanced Accuracy;
3. higher minimum repeat Balanced Accuracy;
4. fewer non-zero final deployment weights;
5. lexicographically smaller `experiment_id`.

The full-OOF score is descriptive only. Kaggle Public Score is deliberately
excluded because it is an external benchmark, not leakage-safe local evidence,
and repeated leaderboard selection would turn the public leaderboard into a
tuning set.

## Resume and identity safety

The plan hashes exact ordered candidate IDs, their candidate-manifest hashes,
all blend settings, the threshold policy, and exploratory status. Resume first
rebuilds this plan from the frozen config. Any candidate identity or manifest
change fails closed before another search is run.

A successful experiment is skipped only when its frozen experiment identity and
candidate hashes match. Failed or interrupted experiments are retried while
other completed experiments remain untouched. Failures never stop later planned
searches. Materialization begins only after every search is terminal.

## Search, materialization, and submission generation

Search evaluates weights and thresholds and records evidence; it does not create
a blend artifact. Materialization reruns the same deterministic canonical
backend settings and creates an immutable blend artifact plus its canonical
`prediction_candidate_v1`. Submission generation accepts only that materialized
canonical candidate and produces a locally validated submission artifact. These
are separate stages so a broad search campaign does not create a large set of
deployment artifacts or submissions.

## PowerShell examples

Read-only validation:

```powershell
uv run python scripts/run_blend_campaign.py validate `
  --config configs/blend_campaigns/exploratory_competition_v1.yaml
```

Freeze a plan without searching:

```powershell
uv run python scripts/run_blend_campaign.py plan `
  --config configs/blend_campaigns/exploratory_competition_v1.yaml
```

Run and resume:

```powershell
uv run python scripts/run_blend_campaign.py run `
  --config configs/blend_campaigns/exploratory_competition_v1.yaml

uv run python scripts/run_blend_campaign.py resume `
  --campaign-dir artifacts/blend_campaigns/exploratory_competition_v1
```

Inspect, materialize, and locally generate submissions:

```powershell
uv run python scripts/run_blend_campaign.py inspect `
  --campaign-dir artifacts/blend_campaigns/exploratory_competition_v1

uv run python scripts/run_blend_campaign.py materialize-top `
  --campaign-dir artifacts/blend_campaigns/exploratory_competition_v1 `
  --top-k 3

uv run python scripts/run_blend_campaign.py generate-submissions `
  --campaign-dir artifacts/blend_campaigns/exploratory_competition_v1
```

The example contains explicit TODO candidate IDs and is intentionally not ready
to validate or execute until those placeholders are replaced from verified
tracked/local candidate inventory.
