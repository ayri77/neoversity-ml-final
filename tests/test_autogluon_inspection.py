from __future__ import annotations

import importlib
import sys
from pathlib import Path

from src.churn_ml.autogluon_inspection import inspect_run


class FakePredictor:
    model_best = "WeightedEnsemble_L2"
    decision_threshold = 0.117

    def model_names(self) -> list[str]:
        return ["ModelA", "WeightedEnsemble_L2"]

    def info(self) -> dict[str, object]:
        return {
            "model_info": {
                "WeightedEnsemble_L2": {
                    "model_weights": {"ModelA": 1.0},
                }
            }
        }


def make_predictor_structure(run_dir: Path) -> None:
    predictor = run_dir / "predictor"
    predictor.mkdir(parents=True)
    for name in ("predictor.pkl", "learner.pkl", "version.txt"):
        (predictor / name).write_text("fake", encoding="utf-8")


def test_inspect_completed_fake_run_loads_by_default(tmp_path: Path) -> None:
    run_dir = tmp_path / "complete"
    run_dir.mkdir()
    make_predictor_structure(run_dir)
    (run_dir / "_SUCCESS").write_text("{}", encoding="utf-8")
    report = inspect_run(
        run_dir,
        predictor_loader=lambda _path: FakePredictor(),
    )
    assert report["classification"] == "complete"
    assert report["predictor_loading_succeeded"] is True
    assert report["predictor"]["best_model"] == "WeightedEnsemble_L2"
    assert report["predictor"]["ensemble_weights"] == {"ModelA": 1.0}


def test_partial_run_load_failure_is_reported_without_changes(tmp_path: Path) -> None:
    run_dir = tmp_path / "partial"
    run_dir.mkdir()
    model = run_dir / "predictor" / "models" / "CompletedFamily"
    model.mkdir(parents=True)
    (run_dir / "_FAILED").write_text("{}", encoding="utf-8")
    before = sorted(path.relative_to(run_dir).as_posix() for path in run_dir.rglob("*"))

    def fail_load(_path: Path) -> object:
        raise RuntimeError("partial predictor is not loadable")

    report = inspect_run(run_dir, attempt_load=True, predictor_loader=fail_load)
    after = sorted(path.relative_to(run_dir).as_posix() for path in run_dir.rglob("*"))
    assert report["classification"] == "failed_or_partial"
    assert report["predictor_loading_attempted"] is True
    assert report["predictor_loading_succeeded"] is False
    assert "partial predictor is not loadable" in report["load_error"]
    assert before == after


def test_runner_modules_do_not_reference_competition_outputs() -> None:
    source_dir = Path(__file__).parents[1] / "src" / "churn_ml"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in source_dir.glob("autogluon_*.py")
    ).lower()
    forbidden = (
        "x_" + "test",
        "sample_" + "submission",
        "data/" + "raw",
        "kag" + "gle",
        "submissions/",
    )
    assert all(token not in source for token in forbidden)


def test_importing_public_runner_does_not_import_autogluon() -> None:
    for name in tuple(sys.modules):
        if name == "autogluon" or name.startswith("autogluon."):
            del sys.modules[name]
    module = importlib.import_module("src.churn_ml.autogluon_cli")
    importlib.reload(module)
    assert not any(
        name == "autogluon" or name.startswith("autogluon.") for name in sys.modules
    )
