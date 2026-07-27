# Experiment v2 frozen compatibility fixture

`experiment_v2_compatibility_v1.json` is a train-only frozen reference for the
`manual_lightgbm_te_v1_compat` adapter. It contains deterministic reduced input
rows and labels, the exact target-encoder and LightGBM contract, the ordered
transformed schema, an encoded-value hash, and frozen probability expectations.

The fixture was generated once on 2026-07-27 in the locked main `.venv` by:

1. constructing the 30 training rows and six prediction rows stored in the
   fixture;
2. running `AutoGluonBinaryOOFTargetEncoder.fit_transform()` with the persisted
   encoder contract;
3. hashing `{"columns": encoded.columns, "values": encoded.to_numpy().tolist()}`
   with the project canonical JSON SHA-256 helper;
4. calling the production v2 candidate adapter with training positions `0..29`
   and prediction positions `30..35`; and
5. persisting the returned probability vector and its canonical JSON SHA-256.

The regression test reads these expectations. It never regenerates or updates
them. An intentional compatibility change requires a new versioned fixture and
an explicit contract review.
