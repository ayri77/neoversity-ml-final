# Final reproducibility guide

This project separates lightweight report reproduction from local submission
verification and full experimental reconstruction. The separation is deliberate:
the tracked repository stays compact, while authoritative historical artifacts are
preserved locally without being rewritten.

## Level 1 — repository-only report reproduction

This level reproduces the final notebook, tables, and charts from the tracked
compact evidence bundle. It needs neither competition data nor model artifacts.

```powershell
git clone https://github.com/ayri77/neoversity-ml-final.git
Set-Location neoversity-ml-final
uv sync --dev --extra ui
uv run python scripts/generate_final_report_assets.py
uv run jupyter nbconvert --to notebook --execute notebooks/09_final_project_report.ipynb --inplace
```

The notebook reads `reports/final/final_results.json` and
`reports/final/selected_submissions.csv`. It does not train, load AutoGluon, or
require ignored artifacts.

## Level 2 — local submission verification

This level requires the canonical ignored candidate/blend packages, competition
sample-submission identity, and test row alignment. Training is not required.
Validate the local artifacts before regenerating a submission:

```powershell
uv run python scripts/export_final_project_evidence.py
uv run python scripts/generate_candidate_submission.py --help
uv run python scripts/run_blend_campaign.py status --campaign-dir artifacts/blend_campaigns/exploratory_confirmation_v1
```

Use the documented candidate ID, threshold, and submission ID for the selected
artifact. The generator must verify 2,500 rows, columns `index` and `y`, ordered
row identity, positive count, and SHA-256. The expected identities are recorded in
`reports/final/artifact_manifest.json` and
`reports/final/selected_submissions.csv`. No Kaggle upload is part of this
workflow.

## Level 3 — full experimental reconstruction

This level requires authorized competition data, immutable Dataset Packages, the
main `.venv`, the isolated `.venv-autogluon`, versioned configs, substantial
time, and substantial disk. It is not represented as a lightweight one-command
workflow.

```powershell
uv sync --dev --extra ui
uv run python scripts/run_dataset_registry.py scan --root data/processed
uv run python scripts/run_dataset_registry.py validate --root data/processed
uv run python scripts/run_blend_campaign.py validate --config configs/blend_campaigns/exploratory_confirmation_v1.yaml
.\.venv-autogluon\Scripts\python.exe scripts\run_autogluon.py --help
```

Heavy commands are shown for orientation only. Rebuilding historical candidates
requires the exact authorized data and environment boundaries and must follow the
frozen configs. This finalization did not retrain models or rerun campaigns.

## Why the artifact tree is not in Git

The local workspace can exceed 22 GB, 104,000 files, and 16,000 directories.
Raw/processed data, predictors, model binaries, OOF/test probabilities, candidate
and blend packages, MLflow state, campaign state, and submission CSVs are generated
or local assets. Tracking them would make the academic repository impractical and
could publish competition data.

Git instead preserves source code, configs, contracts, tests, compact evidence,
hashes, report outputs, and documentation. Historical artifacts remain authoritative
and are never silently relabeled or rewritten.

## Reproducibility boundary

Level 1 is verified from a clean tracked evidence path. Level 2 is identity-checked
when local canonical artifacts exist. Level 3 is process reproducibility, not a
promise of bitwise-identical model training across hardware, operating systems, or
library changes.
