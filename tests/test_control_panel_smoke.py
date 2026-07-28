from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_control_panel_import_does_not_import_ml_runtime_internals() -> None:
    code = (
        "import sys; import src.churn_ml.control_panel; "
        "forbidden=('research_evaluation','experiment_v2','deployment_v1','metrics','optuna'); "
        "assert not any(any(x in name for x in forbidden) for name in sys.modules)"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        check=True,
        timeout=20,
    )


def test_streamlit_app_import_smoke() -> None:
    code = (
        "import runpy; "
        "runpy.run_path('apps/experiment_control_panel.py', "
        "run_name='control_panel_import_smoke')"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        check=True,
        timeout=30,
    )
