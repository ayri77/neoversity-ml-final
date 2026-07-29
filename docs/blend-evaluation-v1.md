# Blend Evaluation v1

Blend Evaluation v1 scores a leakage-safe linear probability blend of two
compatible Experiment Core v2 runs. It is intentionally separate from the
single-model adapter abstraction and from the fixed 50/50 paired-comparison
diagnostic.

## Statistical rule

Do not optimize blend weight or threshold on the same held-out predictions used
to report the metric. For each repeat and outer fold:

1. Treat the current outer fold as the blend evaluation fold.
2. Use outer-validation OOF probabilities from the other outer folds of the same
   repeat as blend-training data.
3. Search LightGBM weight and threshold only on that training slice.
4. Apply the selected weight and threshold to the held-out outer fold.
5. Record held-out predictions and metrics.

Pooled-OOF optimistic fallback is forbidden. Competition-test rows are never
used during blend selection.

## Blend formula

```text
p_blend = w * p_lightgbm + (1 - w) * p_xgboost
```

Weight grid: `w ∈ {0.00, 0.025, ..., 1.00}`.

Threshold policy: existing Experiment Core grid (`0.01` to `0.99`, step
`0.001`, `>=`, deterministic median-lower-on-even tie-break).

Weight tie-break:

1. maximize balanced accuracy on the blend-training slice;
2. within absolute tolerance, prefer weight closest to `0.5`;
3. then prefer the lower LightGBM weight.

## Deployment parameters

- deployment weight = median selected LightGBM weight across held-out folds
- deployment threshold = median selected threshold across held-out folds

A diagnostic sensitivity table is written at weight `±0.05` and `±0.10`,
clipped to `[0, 1]`. Overlapping-repeat confidence intervals are intentionally
omitted.

## Artifact contract sufficiency

Compatible completed Experiment Core v2 outer-validation parquet files already
contain `repeat`, `repeat_seed`, `outer_fold`, `row_position`, `target`, and
positive-class `probability`. No base-model rerun is required for blend
evaluation when those artifacts validate.

## CLI

```powershell
.venv\Scripts\python.exe -u scripts\run_blend_evaluation_v1.py validate --config configs/blend_evaluation/<config>.yaml
.venv\Scripts\python.exe -u scripts\run_blend_evaluation_v1.py run --config configs/blend_evaluation/<config>.yaml --evaluation-id lightgbm_xgboost_blend_v1
.venv\Scripts\python.exe -u scripts\run_blend_evaluation_v1.py inspect --evaluation-dir artifacts/blend_evaluations/lightgbm_xgboost_blend_v1
.venv\Scripts\python.exe -u scripts\run_blend_evaluation_v1.py prepare-deployment --evaluation-dir artifacts/blend_evaluations/lightgbm_xgboost_blend_v1
```

Successful evaluations are stored under `artifacts/blend_evaluations/<id>/`.
`prepare-deployment` writes an immutable draft package under
`artifacts/blend_deployment_packages/<id>/` including shared blend threshold
evidence for Final Deployment v1. Submission variants are generated later from a
completed deployment without retraining:

```powershell
.venv\Scripts\python.exe -u scripts\run_blend_evaluation_v1.py generate-submission-variants --deployment-dir artifacts/deployments/<deployment-id>
```

Variants:

- `blend_robust` (median weight)
- `blend_lgbm_plus` (+0.05 LightGBM weight, leaderboard probe)
- `blend_xgb_plus` (-0.05 LightGBM weight, leaderboard probe)
