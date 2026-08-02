"""Generate publication charts from the tracked final evidence bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / "reports" / "final" / "final_results.json"
IMAGE_DIR = ROOT / "docs" / "images"


def load_evidence(path: Path = EVIDENCE_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def generate_charts(evidence: dict[str, Any] | None = None) -> list[Path]:
    evidence = evidence or load_evidence()
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    submissions = evidence["selected_submissions"]

    labels = [item["label"] for item in submissions]
    internal = [item["internal_balanced_accuracy"] for item in submissions]
    public = [item["public_score"] for item in submissions]
    protocols = [item["internal_protocol"] for item in submissions]
    colors = ["#c44e52" if item["exploratory"] else "#4c72b0" for item in submissions]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    y = np.arange(len(labels))
    axes[0].barh(y, internal, color=colors)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.89, 0.901)
    axes[0].set_xlabel("Balanced Accuracy")
    axes[0].set_title("Internal evidence (protocol-specific)")
    for idx, value in enumerate(internal):
        axes[0].text(value + 0.00008, idx, f"{value:.4f}", va="center", fontsize=9)
    axes[0].text(
        0.0,
        -0.20,
        "Protocols differ: final A+B/A+D/A+H use 5×5 confirmation; native blend uses 2-repeat meta-CV; historical anchor uses legacy explicit OOF.",
        transform=axes[0].transAxes,
        fontsize=8,
        color="#444444",
        wrap=True,
    )

    axes[1].barh(y, public, color=colors)
    axes[1].set_yticks(y, [""] * len(labels))
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.90, 0.913)
    axes[1].set_xlabel("Kaggle Public Score")
    axes[1].set_title("External benchmark (not model-selection metric)")
    for idx, value in enumerate(public):
        axes[1].text(value + 0.00008, idx, f"{value:.4f}", va="center", fontsize=9)
    fig.suptitle("Final portfolio: internal evidence and Kaggle Public Score", fontsize=15, fontweight="bold")
    first = IMAGE_DIR / "final-results-internal-vs-public.png"
    fig.savefig(first, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    thresholds = [item["threshold"] for item in submissions]
    positives = [item["positive_count"] for item in submissions]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    axes[0].barh(y, thresholds, color=colors)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Probability threshold")
    axes[0].set_title("Selected thresholds")
    for idx, value in enumerate(thresholds):
        axes[0].text(value + 0.003, idx, f"{value:.3f}", va="center", fontsize=9)

    axes[1].barh(y, positives, color=colors)
    axes[1].set_yticks(y, [""] * len(labels))
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Predicted positives (of 2,500)")
    axes[1].set_title("Final operating points")
    axes[1].set_xlim(0, max(positives) * 1.15)
    for idx, value in enumerate(positives):
        axes[1].text(value + 5, idx, str(value), va="center", fontsize=9)
    fig.suptitle("Threshold and positive-count comparison", fontsize=15, fontweight="bold")
    second = IMAGE_DIR / "final-candidate-operating-points.png"
    fig.savefig(second, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if len(set(protocols)) < 2:
        raise ValueError("Protocol labels unexpectedly collapsed; refusing a misleading chart.")
    return [first, second]


if __name__ == "__main__":
    for output in generate_charts():
        print(output.relative_to(ROOT).as_posix())
