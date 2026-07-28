from __future__ import annotations

from datetime import timedelta
from pathlib import Path


def human_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "—"
    value = max(0, int(seconds))
    return str(timedelta(seconds=value))


def safe_path_display(path: Path, repository_root: Path) -> str:
    try:
        return (
            path.resolve(strict=False)
            .relative_to(repository_root.resolve(strict=True))
            .as_posix()
        )
    except ValueError:
        return "<outside repository>"
