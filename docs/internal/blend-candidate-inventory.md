# Blend Candidate Inventory

## 1. Purpose and scope

This is a read-only audit of locally available prediction-producing candidates for future multi-model blending. It inventories atomic models separately from macro/ensemble candidates, records missing evidence explicitly, and distinguishes row compatibility from evaluation-protocol comparability. It does not select weights, run a blend, generate predictions, or import candidates.

The companion CSV is the row-level source of truth. It contains one record for every discovered run/model candidate with a saved OOF artifact or explicit run record. AutoGluon models in separate runs remain separate rows even when their probabilities are exact duplicates; the duplicate analysis identifies the representative that should be retained.

`standalone_balanced_accuracy` preserves the source's own recorded semantics. Research v2 values use its nested fold-local threshold protocol; legacy manual `standalone` values are at 0.5 and `calibrated` values use the recorded threshold diagnostic; AutoGluon leaderboard values are its validation score. These values are not interchangeable model-selection estimates.

## 2. Audit timestamp and Git state

| Item | Value |
|---|---|
| Audit timestamp | `2026-08-01T20:38:30Z` |
| Branch | `feature/prepared-dataset-pipeline-v1` |
| HEAD | `6be048d89afc8aaf1a6e1c95fa506b89aca74a60` |
| Start state | Clean and synchronized with `origin/feature/prepared-dataset-pipeline-v1` |
| State observed late in audit | Parallel, out-of-scope source/test changes appeared in the working tree; none were made or modified by this audit |

## 3. Sources inspected

- `artifacts/research_v2/`: 31 run records, aggregate metrics, threshold summaries, identities, resolved configs, and repeated OOF files.
- `artifacts/autogluon_runs/`: 9 completed managed runs, one incomplete run, 87 managed leaderboard candidates, worker results, inspection summaries, predictor inventories, and OOF prediction pickles.
- `artifacts/prediction_candidates/`: one immutable package, `pc1_8da68e3d2071fb71`, for the v5 AutoGluon `WeightedEnsemble_L2`.
- `artifacts/autogluon/`: six historical predictor trees, explicit exports, 72 historical model OOF artifacts, and available top-level test probabilities.
- `artifacts/autogluon_overnight/`: launcher/inspection receipts; no additional distinct predictor candidates beyond the managed run trees.
- `artifacts/experiments/`: eight classic manual atomic exports, two exact duplicate manual LightGBMPrep reproductions, and the historical fixed CatBoost/XGBoost blend.
- `artifacts/research_evaluations/` and `artifacts/final_submissions/`: three nested LightGBMPrep evaluation variants plus the durable full-data test prediction artifact.
- `artifacts/blend_evaluations/`: the cross-fitted Research v2 LightGBM/XGBoost macro evaluation.
- `artifacts/deployments/`: linked v0 and v5 Research v2 LightGBM test probabilities and explicit test-row identity evidence.
- Dataset Package manifests for v0-v7, `docs/benchmarks/kaggle_submissions.csv`, `docs/benchmarks/autogluon.md`, the baseline manifest, and relevant resolved/canonical configs.

Optuna trial caches were treated as tuning evidence rather than deployable candidates: they have no finalized candidate contract or aligned test probabilities. Some archived/test-state and Optuna-search paths were unreadable under the audit sandbox (`artifacts/control_panel_state`, `artifacts/_tmp_jobs_test/jobs`, and two `artifacts/optuna_searches/*` directories). No shortlisted prediction evidence depends on those paths. Trained model objects were not loaded; only metadata, tabular probability files, and the `_oof_pred_proba` arrays inside trusted local `oof.pkl` prediction artifacts were read.

## 4. Candidate inventory summary

| Dimension | Count |
|---|---:|
| Total candidate records | 206 |
| Research v2 | 31 |
| Legacy manual | 13 |
| AutoGluon atomic | 147 |
| AutoGluon ensemble | 13 |
| Fixed blend | 2 |
| Eligible | 12 |
| Conditionally eligible | 97 |
| Ineligible | 37 |
| Unknown | 60 |
| OOF available | 205 |
| OOF missing | 1 |
| Test probabilities available | 20 |
| Test probabilities missing | 186 |
| Exact-duplicate groups | 26 groups / 60 records |

`eligible` means complete, uniquely aligned OOF and test probabilities are already evidenced. `conditionally_eligible` means the candidate is structurally plausible but still needs a declared repeat normalization, test export, or candidate materialization. `unknown` is used mainly for unsealed historical AutoGluon internal OOF artifacts whose row identity or test ordering cannot be proved. `ineligible` covers incomplete/smoke records and non-representative exact duplicates.

All v0-v7 Dataset Packages share target hash `e583aadf...b50966`, train anchor `0e90ca93...3eb34`, and test anchor `5149a289...0491f`. Equal row counts alone were never used as compatibility evidence.

## 5. Target-independent candidate table

| Candidate | Source / model | BA evidence | OOF | Test | Eligibility | Priority |
|---|---|---:|---:|---:|---|---|
| `rv2__v5...988689f2` | Research v2 manual LightGBM / v5 | 0.899252; threshold 0.1095 | 20,000 repeated; 10,000 unique | 2,500 aligned deployment probabilities | Conditional: normalize repeats and materialize | A |
| `rv2__v6...347eb7d5` | Research v2 manual XGBoost / v6 | 0.897891; median threshold 0.0875 | 20,000 repeated; 10,000 unique | Missing | Conditional | A |
| `legacy__v0_raw_minimal__catboost__cv5` | Legacy manual CatBoost / v0 | cross-fitted 0.892612; threshold 0.174 | 10,000 aligned | 2,500 aligned | Eligible reserve | B |
| `autogluon__ag_v6...realtabpfn_v2_r11...` | focused v6 RealTabPFN-v2_r11 | leaderboard 0.879241 | 10,000 aligned | Missing | Conditional | A |
| `autogluon__ag_v6...neuralnettorch_r37...` | focused v6 NeuralNetTorch_r37 | leaderboard 0.813057 | 10,000 aligned | Missing | Conditional | A |
| `autogluon__ag_v6...neuralnettorch_r31...` | focused v6 NeuralNetTorch_r31 | leaderboard 0.824733 | 10,000 aligned | Missing | Conditional reserve | B |
| `autogluon__ag_v6...lightgbmprep_r31...` | focused v6 LightGBMPrep_r31 | leaderboard 0.821047 | 10,000 aligned | Missing | Conditional | A |
| `autogluon__ag_v6...lightgbmprep_r41...` | focused v6 LightGBMPrep_r41 | leaderboard 0.817504 | 10,000 aligned | Missing | Conditional | A |
| `autogluon__ag_v6...weightedensemble_l2` | focused v6 WeightedEnsemble_L2 macro | 0.883923; derived 0.887448 at 0.306 | 10,000 aligned | Missing | Conditional macro | A |

The best source-complete target-independent manual CatBoost is the legacy v0 export, not the Research v2 v6 CatBoost: the latter has stronger protocol metadata but no test probabilities. No target-independent atomic blend using the provisional six-candidate list is executable yet because five members lack test probabilities and the v5 Research v2 OOF still needs declared repeat normalization.

## 6. Exploratory candidate table

| Candidate | Source / model | BA / Kaggle evidence | OOF | Test | Eligibility |
|---|---|---|---:|---:|---|
| `autogluon_legacy__autogluon_v3_extreme_8h_20260726__lightgbmprep_r31_bag_l1` | historical AutoGluon LightGBMPrep_r31 | 0.895158 at 0.117; Kaggle 0.9112 | 10,000 aligned | 2,500 aligned | Eligible |
| `legacy__v3__manual_lightgbmprep_r31__20260724t211936695045z_21d37ad8` | manual LightGBMPrep_r31 reproduction | 0.900979 at 0.117; Kaggle 0.9023 | 10,000 aligned | 2,500 aligned | Eligible |
| `autogluon__ag_v3_focused_hybrid_smoke_s42_20260801_173849__realtabpfn_v2_r11_bag_l1` | focused v3 RealTabPFN-v2_r11 | leaderboard 0.879624 | 10,000 aligned | Missing | Conditional |
| `autogluon__ag_v3_focused_hybrid_smoke_s42_20260801_173849__neuralnettorch_r31_bag_l1` | focused v3 NeuralNetTorch_r31 | leaderboard 0.820852 | 10,000 aligned | Missing | Conditional |
| `autogluon__ag_v3_focused_hybrid_smoke_s42_20260801_173849__weightedensemble_l2` | focused v3 WeightedEnsemble_L2 macro | 0.881198; derived 0.885055 at 0.360 | 10,000 aligned | Missing | Conditional macro |
| `fixed__v3__research_v2_lightgbm_xgboost_crossfit` | Research v2 LGB/XGB macro | 0.899209; deployment weight 0.775, threshold 0.134 | 20,000 repeated | Missing | Conditional macro |
| v5/v6 A-priority candidates above | strongest target-independent additions | protocol-specific | aligned OOF | mixed/missing | Conditional |

The two score-linked v3 atomic exports are the only A-priority exploratory atomic candidates with both probability sides currently present. Focused v3 models should enter only after per-model test probabilities are exported through the approved adapter path.

## 7. Macro/ensemble candidates

| Macro candidate | Components / dependency | Readiness |
|---|---|---|
| v5 `WeightedEnsemble_L2` / `pc1_8da68e3d2071fb71` | 81.82% RealTabPFN-r11, 9.09% LGBPrep-r31, 4.55% each LGBPrep-r13/r41 | Eligible immutable candidate package; both tracks |
| focused v6 `WeightedEnsemble_L2` | 66.67% RealTabPFN-r11, 16.67% NNT-r37, 8.33% each LGBPrep-r31/r41 | OOF complete; test missing |
| focused v3 `WeightedEnsemble_L2` | 75% RealTabPFN-r11, 25% NNT-r31 | OOF complete; test missing; exploratory |
| historical CatBoost/XGBoost 50/50 | exact dependency on the legacy v0 CatBoost and XGBoost exports | Eligible; Kaggle 0.8832 |
| Research v2 v3 LGB/XGB cross-fitted macro | depends on run `38bbefe2` and run `55c26d5e` | repeated OOF complete; test missing |

Do not place any macro candidate and its own components in the same initial optimization. Several one-component AutoGluon weighted ensembles are exact atomic duplicates and are excluded rather than treated as independent candidates.

## 8. Compatibility groups

| Group | Records | Evidence | Permitted use |
|---|---:|---|---|
| G1 common-anchor target-independent | 70 | common target hash, train anchor, and Dataset Package test anchor; file-level row checks where predictions exist | academic target-independent track after each candidate's missing export/normalization gates pass |
| G2 common-anchor exploratory | 64 | same row/target anchors, but dataset/model path has `target_dependency: exploratory` | exploratory track only; may include G1 candidates with explicit labeling |
| G3 identity unproven | 72 | historical internal AutoGluon OOF exists, but sealed source identity or explicit test order is absent | diagnostic only; do not blend until provenance is reconstructed |

Research v2 repeated OOF has complete `(repeat, row_position)` coverage and no duplicates at that key, but it is not yet one prediction per row. The declared future normalization should be arithmetic mean across repeats and must be recorded in the candidate manifest. AutoGluon bagged OOF and classic manual OOF already provide one probability per row. Shared fold assignments are not required for row-wise blending, but their metric estimates remain protocol-specific.

## 9. Missing evidence and blockers

- All 12 focused v3 and all 12 focused v6 models have complete 10,000-row saved OOF arrays and zero saved per-model test probability files. None is currently blend-ready.
- Research v2 candidates have complete repeated OOF, but only v0 and v5 manual LightGBM have linked 2,500-row deployment probabilities. Every Research v2 candidate still needs a versioned repeat-normalization/materialization step.
- The historical 0.9112 LightGBMPrep-r31 export is the exception: its explicit OOF export has target and row indices 0-9,999, its explicit test export has row indices 0-2,499, and both align to the canonical target/sample order.
- Sixty older AutoGluon candidates are `unknown`; another 12 old records are in G3 but already excluded as duplicates. Their internal OOF arrays alone do not prove test ordering or complete provenance.
- `ag-v3-lightgbmprep-cpu-20260727` is incomplete: no worker result, predictor, leaderboard, `_SUCCESS`, OOF, or test probabilities.
- Candidate-specific thresholds are absent for AutoGluon atomic models. The ensemble threshold must not be copied to its atoms.
- No evidence indicated competition-test labels were used. Research v2 records explicitly report `competition_assets_accessed: false`; managed AutoGluon configs point only to prepared train features/targets during fit.

## 10. Duplicate/correlation findings

Probability SHA-256 was computed over normalized `float64` probability vectors. There are 26 exact OOF duplicate groups covering 60 records.

Important exact duplicates:

- The two manual LightGBMPrep reproduction runs have identical OOF and test probability hashes; retain the later parity-verified `21d37ad8` run.
- Historical v3 `WeightedEnsemble_L2` is an exact OOF duplicate of LightGBMPrep-r31 in both copied predictor trees; the ensemble contains only that atom. Retain the explicit r31 export linked to 0.9112.
- Focused v3 RealTabPFN-r11 is identical to both standalone RealTabPFN-r11 runs and their one-component ensembles. Retain the focused atomic record.
- Focused v3 LightGBMPrep-r31 is identical to the standalone LightGBMPrep-r31 run and its one-component ensemble. Retain the focused atomic record for focused-run analysis.
- The v5 smoke `WeightedEnsemble_L2` and v5 smoke LightGBMPrep-r31 are identical to the full v5 LightGBMPrep-r31 atom; they are not independent candidates.

Notable near-duplicate relationships:

| Pair | Pearson | Spearman | Classification agreement at recorded thresholds |
|---|---:|---:|---:|
| focused v3 vs v6 RealTabPFN-r11 | 0.996669 | 0.990473 | n/a |
| Research v2 v5 LightGBM vs v6 XGBoost | 0.988672 | 0.985148 | 0.9762 |
| focused v6 LGBPrep-r31 vs r41 | 0.983306 | 0.949672 | n/a |
| focused v3 vs v6 WeightedEnsemble | 0.987946 | 0.939592 | 0.9764 |
| historical vs manual v3 LightGBMPrep-r31 | 0.954545 | 0.908570 | 0.9488 |

Likely diversity sources are the neural nets: focused v6 NNT-r31 has Pearson 0.737-0.751 against several tree candidates, and NNT-r37 is roughly 0.742-0.752 against the same set. RealTabPFN versus the neural nets is approximately 0.75-0.79. These are descriptive diagnostics, not evidence that a blend will improve.

## 11. Recommended initial atomic shortlist

Target-independent activation shortlist, preserving the provisional intent:

1. Research v2 manual LightGBM / v5.
2. Research v2 manual XGBoost / v6.
3. focused v6 RealTabPFN-v2_r11.
4. focused v6 NeuralNetTorch-r37.
5. focused v6 LightGBMPrep-r31.
6. focused v6 LightGBMPrep-r41.

Reserve: focused v6 NeuralNetTorch-r31 for diversity diagnostics and legacy manual CatBoost / v0 as the best source-complete target-independent CatBoost. This list is not executable today: export the five missing test vectors and normalize/materialize the v5 Research v2 OOF first.

Exploratory activation shortlist:

1. historical v3 AutoGluon LightGBMPrep-r31 linked to Kaggle 0.9112.
2. canonical manual v3 LightGBMPrep-r31 reproduction linked to Kaggle 0.9023.
3. focused v3 RealTabPFN-v2_r11 after test export.
4. focused v3 NeuralNetTorch-r31 after test export.
5. the target-independent shortlist after its own gates pass.

For a blend that can be assembled strictly from currently complete atomic files, begin exploratory diagnostics with the historical and manual v3 r31 candidates plus selected eligible legacy manual atoms. Do not interpret this as the final model-selection set because protocols differ.

## 12. Recommended macro-candidate shortlist

- v5 AutoGluon `WeightedEnsemble_L2` (`pc1_8da68e3d2071fb71`) is the only immutable, directly consumable macro candidate and is recommended for both-track macro diagnostics.
- focused v6 `WeightedEnsemble_L2` is recommended for the target-independent macro queue after test export.
- focused v3 `WeightedEnsemble_L2` is recommended for the exploratory macro queue after test export.
- retain the historical v0 CatBoost/XGBoost 50/50 blend as a benchmark macro, not as a preferred optimizer input.
- retain the Research v2 v3 LGB/XGB macro as diagnostic-only until test probabilities exist.

## 13. Excluded candidates and reasons

- 37 records are ineligible: exact-duplicate non-representatives, Research v2 smoke records, duplicate nested-evaluation variants, and the incomplete managed AutoGluon run.
- 60 records remain unknown and diagnostic-only because historical predictor trees are unsealed and lack sufficient row/test identity evidence.
- AutoGluon one-component weighted ensembles are excluded when their probabilities exactly equal the component.
- v5 extreme-smoke candidates are excluded where the full run contains the same atomic probabilities.
- Low-performing ExtraTrees/TabM entries remain in the CSV for completeness but are C-priority diagnostics, not shortlist members.
- Optuna trial-cache predictions are excluded from candidate rows because they are tuning trials rather than finalized candidates and have no aligned test probabilities.

## 14. Recommended next actions after Multi-Blend is ready

1. Materialize the v5 Research v2 LightGBM with an explicit mean-across-repeats OOF policy and its existing deployment test probabilities.
2. Export, without retraining, per-model test probabilities for the selected focused v6 atoms and validate them against the common test anchor/order hash.
3. Export the selected focused v3 atoms and macros the same way; do not reuse ensemble thresholds for atoms.
4. Run the candidate validator, then freeze separate target-independent and exploratory candidate manifests.
5. Remove exact duplicates and keep macros disjoint from their components in each initial optimization.
6. Start with fixed weights. Only after the candidate set is frozen should any bounded OOF-only weight search occur; select the final threshold last.
7. Preserve Kaggle Public Score as external benchmark evidence only. Do not use it to tune candidate membership, weights, or thresholds.

No blending was executed as part of this audit.
