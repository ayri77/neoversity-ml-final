# Churn ML Experiment Platform

[Українська академічна версія](README.uk.md) · [Final executed notebook](notebooks/09_final_project_report.ipynb) · [Compact evidence](reports/final/README.md) · [Reproducibility](docs/final-reproducibility.md) · [Final audit](docs/final-audit.md)

Publication-ready record of a binary telecom churn project and its experiment
platform. The metric is **Balanced Accuracy**. Train has 10,000 rows, test 2,500,
the raw schema 230 features, and the positive-class rate is 13.05%.

## Final status

> **As of 2026-08-02**

| Item | Final state |
|---|---|
| Best verified Kaggle Public Score | **0.9112**, historical AutoGluon-screened `LightGBMPrep_r31` on exploratory `v3` |
| Strongest final honest result | A+B, mean BA **0.896420**, std **0.000438**, repeated 5×5 meta-CV |
| Best new Kaggle Public result | A+D, **0.9080** |
| Final portfolio | Five frozen submissions |
| Formal report / evidence | Executed [notebook 09](notebooks/09_final_project_report.ipynb) / [identity-verified bundle](reports/final/README.md) |

No Private Score or final rank is known. Kaggle Public Score is an external
benchmark, not the project-owned model-selection metric.

## Final results

The frozen `exploratory_confirmation_v1` campaign used 5 folds × 5 repeats,
seed 42, deterministic native pair optimization, and fold-local threshold
decisions. Under that shared protocol, A+B and A+D improved on the historical
anchor. A+B was the most stable internal candidate. A+D produced the best new
Public Score, 0.9080. The historical model remained the best verified Public
result at 0.9112.

Public ordering did not exactly follow honest internal evaluation because the
Public leaderboard scores only a subset of hidden labels. It was not used to
tune features, thresholds, or blend weights.

![Final internal evidence and Public Score comparison](docs/images/final-results-internal-vs-public.png)

*Internal metrics retain protocol labels; the right panel is external Kaggle
evidence only. Red bars depend on exploratory `v3`.*

| Experiment | Honest mean BA | Std | Min repeat | Full-OOF descriptive BA | Threshold | Positives | Public |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline A | 0.892843 | 0.001462 | 0.890291 | 0.895158 | 0.117 | 541 | 0.9112 |
| A+B | **0.896420** | **0.000438** | **0.895712** | 0.900249 | 0.125 | 567 | 0.9042 |
| A+D | 0.896129 | 0.001352 | 0.894466 | 0.900558 | 0.117 | 551 | **0.9080** |
| A+H | 0.894780 | 0.001555 | 0.892477 | 0.897593 | 0.226 | 531 | 0.9039 |

All rows depend on target-informed A and remain exploratory. Baseline A's Public
score is historical external evidence, not campaign output.

### Frozen Kaggle portfolio

| # | Submission | Candidate / blend | Threshold | Positives | Public | Exploratory |
|---:|---|---|---:|---:|---:|---|
| 1 | Historical AutoGluon v3 LightGBMPrep-r31 | `pc1_d833780dcc6565cb` | 0.117 | 541 | **0.9112** | Yes |
| 2 | Final A+D | `pc1_eef84ff4999c488a` / `pb1_1c6b3a8dbd482a6a` | 0.117 | 551 | 0.9080 | Yes |
| 3 | Target-independent native probability blend | `pc1_c54c97e9c32f33a2` / `pb1_d047ebe82811ef1a` | 0.170 | 527 | 0.9048 | No |
| 4 | Final A+B | `pc1_671b589ae5111b0d` / `pb1_6ec07045204974b8` | 0.125 | 567 | 0.9042 | Yes |
| 5 | Final A+H | `pc1_9255274a6cbd3bab` / `pb1_1ca68e70755eea20` | 0.226 | 531 | 0.9039 | Yes |

Exact identities are in [selected submissions](reports/final/selected_submissions.csv)
and [the artifact manifest](reports/final/artifact_manifest.json); submission CSVs
are intentionally not tracked.

## Dataset Packages

| Dataset | Features | Parent | Hypothesis | Dependency |
|---|---:|---|---|---|
| `v0_raw_minimal` | 205 | — | Remove constant/empty columns | `none` |
| `v1_missingness_summary` | 213 | `v0` | Row-level missingness summaries | `none` |
| `v2_missingness_indicators` | 393 | `v0` | Broad missingness indicators | `none` |
| `v3_targeted_missingness` | 217 | `v1` | Four target-informed indicators | **`exploratory`** |
| `v4_zero_value_summary` | 209 | `v0` | Row-level zero summaries | `none` |
| `v5_joint_missingness_pattern` | 214 | `v1` | Joint missingness state | `none` |
| `v6_compact_missingness_indicators` | 247 | `v1` | Unique missingness indicators | `none` |
| `v7_compact_zero_indicators` | 234 | `v4` | Compact zero indicators | `none` |

Packages are immutable and record ordered schema, content/target/row hashes,
lineage, and feature roles. `v3` never enters the unbiased dataset ranking.

## Models and methodology

| Family | Role |
|---|---|
| Logistic Regression | Transparent baseline |
| CatBoost | Controlled boosted-tree comparison |
| LightGBM | Strong manual and AutoGluon candidates |
| XGBoost | Tree baseline and diversity |
| AutoGluon ensembles | Isolated bagged models, WeightedEnsemble, explicit exports |
| NeuralNetTorch | Diversity screen; rejected after no honest gain |

The 0.9112 candidate was imported from model-specific explicit AutoGluon exports:
no predictor loading, in-sample fallback, or refit. AutoGluon remains isolated in
`.venv-autogluon`.

- OOF predictions preserve held-out row identity.
- Repeated/nested validation separates evaluation from threshold selection.
- Weights are fixed before final threshold selection.
- Honest meta-CV uses held-out decisions; full-OOF deployment metrics are descriptive.
- Protocols, folds, seeds, candidates, weights, and threshold grids are frozen.

## Implemented platform

| Area | State |
|---|---|
| Dataset Registry | Immutable packages, lineage, roles, hashes, validation |
| Research v2 | Repeated evaluation, aligned OOF, threshold evidence |
| Comparisons | Strict paired gate and row-aligned cross-dataset comparison |
| Candidates | Canonical contract; managed and explicit AutoGluon import |
| Blending | Native/Optuna backends, honest meta-CV, diversity analysis |
| Campaign runner | Frozen/resumable plans, ranking, selective materialization |
| Submission | Local schema/order/positive-count/SHA validation; no upload |
| Control Panel | Allowlisted workflows, results, compare, blend, submissions |

A dedicated campaign-results matrix and some canonical Tune/Blend handoffs remain
future UI work.

## Control Panel

![Control Panel Run page](docs/images/control-panel-run.png)

*Explicit operation, action, source, model, mode, and safety summary.*

![Control Panel Results page](docs/images/control-panel-results.png)

*Schema-aware result inventory; filesystem artifacts remain authoritative.*

![Control Panel Compare view](docs/images/control-panel-compare.png)

*Recorded comparisons with normalized identity and lineage.*

![Control Panel Blend workspace](docs/images/control-panel-blend.png)

*Materialized blends, honest BA, thresholds, and parent weights.*

![Control Panel submission inventory](docs/images/control-panel-submission.png)

*Five locally generated submissions; the UI never uploads to Kaggle.*

```powershell
uv run --extra ui streamlit run apps/experiment_control_panel.py --server.headless true --server.port 8501
```

## Reproducibility

| Level | Reproduction | Requirements |
|---|---|---|
| 1 — Repository-only | Notebook 09, tables, charts | Clean clone and main `.venv` |
| 2 — Local submission | Readiness and exact regeneration | Ignored canonical artifacts |
| 3 — Full reconstruction | Datasets and historical experiments | Authorized data, two environments, substantial resources |

See [the guide](docs/final-reproducibility.md). The local workspace can exceed
22 GB and 100,000 generated files; raw/processed data, predictors, binaries,
OOF/test probabilities, campaigns, MLflow state, and submissions remain ignored.

```powershell
uv sync --dev --extra ui
uv run jupyter nbconvert --to notebook --execute notebooks/09_final_project_report.ipynb --inplace
```

## Limitations and conclusion

`v3` is target-informed exploratory work; internal protocols are not fully
interchangeable; Public ordering may differ from Private; large artifacts are not
in Git; bitwise training reproducibility across hardware/library changes is not
claimed. Honest confirmation favored A+B for stability and A+D as a close second;
A+D achieved the best new Public Score. The historical candidate still leads at
0.9112. The final portfolio, evidence, notebook, screenshots, and boundaries are
frozen for review.
