# Current project status and handoff

**Last updated:** 2026-08-02

**Checkpoint:** final academic publication and submission handoff

**Active branch at finalization start:** `feature/prepared-dataset-pipeline-v1`

## Current state

The project is finalized for human review. Dataset Packages `v0`–`v7`,
Research v2, comparison contracts, canonical prediction candidates, managed and
explicit-export AutoGluon imports, native/Optuna blending, honest repeated meta-CV,
the resumable Blend Campaign Runner, selective materialization, and local canonical
submission generation are implemented.

The formal Ukrainian report is
`notebooks/09_final_project_report.ipynb`. It executes from tracked compact
evidence in `reports/final/` and does not train or load predictors. Publication
charts and five representative Control Panel screenshots are under `docs/images/`.

## Frozen final evidence

- Historical exploratory anchor: `pc1_d833780dcc6565cb`,
  `LightGBMPrep_r31_BAG_L1`, threshold 0.117, Public Score 0.9112.
- Confirmation campaign: `exploratory_confirmation_v1`, plan hash
  `857e146a11c8591032fab8e8f3d1994e391fff2ef38e0cc4df2ade3a696dc172`.
- A+B: honest mean BA 0.896420, std 0.000438, Public Score 0.9042.
- A+D: honest mean BA 0.896129, Public Score 0.9080.
- A+H: honest mean BA 0.894780, Public Score 0.9039.
- Target-independent native blend: honest 2-repeat meta-CV BA 0.898216,
  Public Score 0.9048; its protocol is labeled separately.
- Neural screen: `exploratory_neural_screen_v1`; A+Neural did not improve A.

The frozen five-submission portfolio is documented in
`reports/final/selected_submissions.csv`. No Private Score or final rank is known.

## Methodology boundary

`v3_targeted_missingness` is target-informed and remains exploratory everywhere.
Honest 5×5 confirmation, historical explicit-OOF evidence, and native 2-repeat
meta-CV are not silently treated as identical protocols. Kaggle Public Score is
external evidence only.

## Persisted-state and artifact compatibility

No persisted schema, historical artifact, MLflow mapping, or Dataset Package was
migrated or rewritten during finalization. Compatibility therefore required no
migration. The final report adds a tracked overlay of compact summaries and
repository-relative provenance paths; ignored authoritative artifacts remain intact.

## Remaining roadmap

The academic scope is complete. Optional platform work remains outside submission:
a dedicated campaign-results matrix, canonical completed-artifact handoffs for all
Tune/Blend paths, and further deployment-result UI cleanup. No new research is
required for submission.

## Immediate next action

Verify the rendered GitHub README and notebook, confirm the same five Kaggle
selections, paste `docs/mentor-submission-message.uk.md` into the LMS, and submit.
