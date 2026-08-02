"""Thin repository bootstrap for Blend Campaign Runner v1."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.churn_ml.blend_campaign.cli import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main(repository_root=REPOSITORY_ROOT))
