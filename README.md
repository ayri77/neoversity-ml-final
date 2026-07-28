# Customer Churn Prediction

## Project Structure

## Environment

## Repository Layout

## Experiments

The local [Experiment Control Panel](docs/experiment-control-panel.md) provides a
declarative Streamlit UI for validating and launching approved public CLI commands,
monitoring operational jobs, and reading configured artifact summaries. Filesystem
experiment artifacts remain authoritative. Launch it from the repository root:

```powershell
uv run --extra ui streamlit run apps/experiment_control_panel.py
```

The optional local metadata index is documented in
[`docs/mlflow-local-index.md`](docs/mlflow-local-index.md). Filesystem artifacts remain
authoritative; MLflow is a searchable local mirror only. Sync uses a schema-v3 key
scoped by the complete repository-relative source path. Early failed research runs
validate every present authoritative optional artifact before any field is mapped or
copied. After exact source metadata and bounded human-readable artifacts are verified,
MLflow status is finalized, and
`sync_complete=true` is written last before one deterministic receipt. Corrupt terminal
runs are rejected, failed AutoGluon runs never expose stale completion-only model
fields, and no predictor or model is loaded.

## Results