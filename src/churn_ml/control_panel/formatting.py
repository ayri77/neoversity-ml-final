from __future__ import annotations

from pathlib import Path


def human_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "—"
    value = max(0, int(seconds))
    if value < 60:
        return f"{value}s"
    if value < 3600:
        minutes = value // 60
        secs = value % 60
        return f"{minutes}m {secs:02d}s"
    hours = value // 3600
    remainder = value % 3600
    minutes = remainder // 60
    secs = remainder % 60
    return f"{hours}h {minutes}m {secs:02d}s"


def safe_path_display(path: Path, repository_root: Path) -> str:
    try:
        return (
            path.resolve(strict=False)
            .relative_to(repository_root.resolve(strict=True))
            .as_posix()
        )
    except ValueError:
        return "<outside repository>"
