# Final project evidence bundle

This directory is the compact, tracked evidence source for the final academic
report. It deliberately contains no raw competition rows, predictions, model
binaries, MLflow state, or submission CSV files.

## Contents

- `final_results.json` — canonical structured evidence used by the notebook and charts;
- `final_results.csv` — final 5×5 confirmation campaign results;
- `selected_submissions.csv` — the frozen five-submission Kaggle portfolio;
- `artifact_manifest.json` — expected immutable identities and the status of the local verification performed during export.

The bundle was produced by `scripts/export_final_project_evidence.py`. The export
verified candidate-manifest hashes, campaign plan hashes, and final submission
hashes against the ignored local artifact tree without loading models or prediction
arrays. Repository-relative provenance paths intentionally point to local immutable
artifacts that are not committed.

To repeat the identity-checked export when those local artifacts are available:

```powershell
uv run python scripts/export_final_project_evidence.py
```

From a clean clone, the formal notebook reads the already tracked JSON and CSV
files directly; no ignored artifact is required. `--skip-local-verification` exists
only for regenerating declared summaries when the immutable local artifacts are not
present, and its status is recorded in `artifact_manifest.json`.

## Interpretation boundary

Internal Balanced Accuracy values retain their protocol labels. In particular, the
historical explicit-OOF score, the native blend's 2-repeat meta-CV score, and the
final confirmation campaign's repeated 5×5 honest mean are not silently treated as
one directly comparable ranking. Kaggle Public Score is external evidence only.
`v3_targeted_missingness` and every candidate depending on it remain explicitly
marked exploratory. No Private Score or final competition rank is known.
