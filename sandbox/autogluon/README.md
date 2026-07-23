# AutoGluon sandbox environment (`.venv-autogluon`)

Isolated, uv-managed environment for AutoGluon experiments. Kept fully separate from the
main project `.venv` — never add AutoGluon dependencies to the root `pyproject.toml`.

- Python: 3.12.12
- AutoGluon: 1.5.0
- GPU: NVIDIA RTX 4080 Laptop (driver 610.62, supports CUDA runtime up to 13.3)

## Setup / repair

```powershell
uv pip install --python .venv-autogluon/Scripts/python.exe `
  -r sandbox/autogluon/requirements-autogluon.txt `
  --index-url https://download.pytorch.org/whl/cu128
```

`--index-url` is required so `torch`/`torchvision` resolve to the CUDA (`+cu128`) build
instead of the default CPU-only PyPI wheel. Everything else installs from PyPI normally
(`uv` falls back to it automatically for packages not present on the PyTorch index).

Always install via this requirements file (`uv pip install -r ...` or `uv pip sync ...`),
never `uv add`, so the fragile pins below can't silently drift.

Verify with:

```powershell
.venv-autogluon/Scripts/python.exe sandbox/autogluon/check_environment.py
```

## Root causes fixed

1. **GPU not detected** — the environment had shipped with the CPU-only `torch==2.9.1+cpu`
   wheel. Fixed by installing the `+cu128` build, which matches the driver's CUDA 12.8
   support and AutoGluon's declared `torch<2.10,>=2.6` requirement.

2. **`NeuralNetFastAI` crash: `AttributeError: 'list' object has no attribute 'starmap'`** —
   `fastai==2.8.7` calls `L(...).starmap(...)` (`fastai/optimizer.py`), an API that
   `fastcore` removed in its v2.0 release (replaced by the `star()` adapter). The
   environment had `fastcore==2.1.5` installed; `fastai`'s own dependency metadata only
   requires `fastcore>=1.8.0` with no upper bound, so pip/uv won't catch this on its own.
   Fixed by pinning `fastcore==1.14.5` (last pre-v2 release).

   This pin has a knock-on effect: `fastprogress>=1.1.0` unconditionally imports
   `python-fasthtml`, which itself requires `fastcore>=2.1.2` — directly incompatible
   with the fastcore pin above. Fixed by pinning `fastprogress==1.0.5`, the last release
   with no `fasthtml` dependency.

3. **LightGBM `TypeError: 'NoneType' object is not iterable` in `_mem_early_stop`** —
   `lightgbm==4.6.0` is within AutoGluon's supported range (`lightgbm<4.7,>=4.0`), so this
   is not a version mismatch. AutoGluon's custom callback
   (`autogluon/tabular/models/lgb/callbacks.py`) reads `env.evaluation_result_list`, which
   can still be `None` when the callback fires before LightGBM's internal evaluation
   callback populates it — a known, upstream-acknowledged timing sensitivity (the source
   has an explicit `FIXME` about per-model memory accounting being unreliable during
   parallel fits). It surfaces more often under memory pressure from oversubscribed Ray
   workers. Mitigate at the `fit()` call, not via a package pin:
   - Pass an explicit `num_cpus` (not `"auto"`) sized with headroom, e.g.
     `num_cpus = os.cpu_count() - 2`, to reduce concurrent Ray worker memory pressure.
   - Keep an eye on `Memory Avail` in the AutoGluon startup banner — if it's a small
     fraction of total RAM (see below), close other applications before a long run.

4. **LightGBMPrep configs skipped ("insufficient memory")** — same root cause as #3.
   On this machine, `psutil` reported only ~4-5 GB of the 34 GB total RAM available at
   the time of testing (other running applications, browser tabs, etc. were consuming
   the rest) — that is a real, current system constraint, not an AutoGluon bug. Close
   memory-heavy applications before a full run, or expect AutoGluon to skip/scale back
   memory-intensive configs accordingly.

## GPU scoping

Do not pass `num_gpus=1` at the top level of `TabularPredictor.fit()` — it applies to
every model family, including ones with no GPU support in these pip wheels
(`LightGBM`, `XGBoost` CPU builds). Scope it per model instead:

```python
hyperparameters={
    "GBM": {},                                            # CPU only, no GPU wheel here
    "NN_TORCH": {"ag_args_fit": {"num_gpus": 1}},
    "FASTAI": {"ag_args_fit": {"num_gpus": 1}},
    "CAT": {"task_type": "GPU", "ag_args_fit": {"num_gpus": 1}},
}
```

Model families that actually benefit from the RTX 4080 here: `NeuralNetTorch`,
`NeuralNetFastAI`, and `CatBoost` (via `task_type="GPU"`, verified working). LightGBM's
standard pip wheel has no OpenCL/GPU support — don't request GPU for it.

## Recommended config for a future maximum-quality run

- Keep `presets="best_v150"` (or try the newer `"extreme"` preset if TabPFN/TabICL/Mitra
  extras are installed — not covered by this repair).
- Pass `num_cpus` explicitly rather than `"auto"`.
- Scope `num_gpus` per model as shown above rather than globally.
- Free up RAM before starting (check `Memory Avail` in the startup banner via
  `check_environment.py` or the AutoGluon log itself).
- Re-run the six-hour `best_v150` benchmark only after a short (~10-30 min) supervised
  run confirms no regressions under real data volumes.

## Known limitations after this repair

- `python-fasthtml` remains installed but is now unused by `fastprogress==1.0.5`;
  harmless, left in place rather than uninstalling to minimize churn.
- The LightGBM callback timing issue (#3) is mitigated, not eliminated — it depends on
  system memory pressure and Ray scheduling, both partly outside AutoGluon's control.
- CUDA compiler toolkit installed system-wide is old (`nvcc` reports CUDA 11.8), but this
  does not matter for pip-installed CUDA wheels, which bundle their own CUDA runtime.
