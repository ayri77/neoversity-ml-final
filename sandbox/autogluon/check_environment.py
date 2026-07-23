"""Diagnostic check for the isolated .venv-autogluon environment.

Run with:
    .venv-autogluon/Scripts/python.exe sandbox/autogluon/check_environment.py
"""

import shutil
import sys


def section(title):
    print(f"\n=== {title} ===")


def main():
    ok = True

    section("Python executable")
    print(sys.executable)
    print(sys.version)
    if ".venv-autogluon" not in sys.executable:
        print("WARNING: not running inside .venv-autogluon")
        ok = False

    section("AutoGluon")
    try:
        import autogluon.core

        print("autogluon.core", autogluon.core.__version__)
        from autogluon.tabular import TabularPredictor  # noqa: F401

        print("autogluon.tabular import OK")
    except Exception as e:
        print("FAILED:", e)
        ok = False

    section("PyTorch / CUDA")
    try:
        import torch

        print("torch", torch.__version__)
        print("torch.version.cuda", torch.version.cuda)
        print("cuda available", torch.cuda.is_available())
        print("device count", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("device name", torch.cuda.get_device_name(0))
            x = torch.randn(1024, 1024, device="cuda")
            y = x @ x
            torch.cuda.synchronize()
            print("GPU matmul OK, sum =", y.sum().item())
        else:
            print("WARNING: CUDA not available")
            ok = False
    except Exception as e:
        print("FAILED:", e)
        ok = False

    section("CatBoost GPU")
    try:
        import numpy as np
        from catboost import CatBoostClassifier

        X = np.random.rand(200, 5)
        y = (np.random.rand(200) > 0.5).astype(int)
        model = CatBoostClassifier(
            iterations=10,
            task_type="GPU",
            devices="0",
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(X, y)
        print("CatBoost GPU fit OK, tree_count =", model.tree_count_)
    except Exception as e:
        print("FAILED:", e)
        ok = False

    section("FastAI")
    try:
        import fastcore

        print("fastcore", fastcore.__version__)
        from fastcore.foundation import L

        assert hasattr(L([]), "starmap"), (
            "L.starmap missing (fastcore too new for fastai)"
        )
        import fastai

        print("fastai", fastai.__version__)
        from fastai.optimizer import Adam  # noqa: F401
        from fastai.tabular.all import TabularDataLoaders  # noqa: F401

        print("fastai optimizer/tabular import OK")
    except Exception as e:
        print("FAILED:", e)
        ok = False

    section("LightGBM")
    try:
        import lightgbm

        print("lightgbm", lightgbm.__version__)
    except Exception as e:
        print("FAILED:", e)
        ok = False

    section("System resources")
    try:
        import psutil

        vm = psutil.virtual_memory()
        print(
            f"RAM total: {vm.total / 1e9:.1f} GB, available: {vm.available / 1e9:.1f} GB ({vm.percent}% used)"
        )
        total, used, free = shutil.disk_usage(".")
        print(f"Disk total: {total / 1e9:.1f} GB, free: {free / 1e9:.1f} GB")
    except Exception as e:
        print("FAILED:", e)
        ok = False

    section("Summary")
    print("ALL CHECKS PASSED" if ok else "ONE OR MORE CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
