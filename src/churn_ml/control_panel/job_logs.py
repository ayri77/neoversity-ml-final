"""Safe Job stdout/stderr inspection helpers.

Paths must remain under the selected Job directory. Display decoding uses UTF-8
with replacement; downloads return the exact stored bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class JobLogError(ValueError):
    """Raised when a Job log path is unsafe or unreadable."""

    def __init__(self, message: str, *, reason_code: str = "job_log") -> None:
        self.reason_code = reason_code
        super().__init__(message)


@dataclass(frozen=True)
class JobLogInfo:
    path: Path
    exists: bool
    size_bytes: int
    line_count: int
    encoding: str
    modified_at_ns: int


_ALLOWED_LOG_NAMES = frozenset({"stdout.log", "stderr.log"})
INLINE_RENDER_MAX_BYTES = 2 * 1024 * 1024


def resolve_job_log_path(job_root: Path | str, name: str) -> Path:
    root = Path(job_root).resolve()
    if name not in _ALLOWED_LOG_NAMES:
        raise JobLogError(
            f"Unsupported job log name: {name!r}.",
            reason_code="job_log_name_invalid",
        )
    path = (root / name).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise JobLogError(
            "Job log path escapes the selected job directory.",
            reason_code="job_log_path_escape",
        ) from error
    return path


def _count_lines(data: bytes) -> int:
    if not data:
        return 0
    # splitlines(keepends=False) semantics for both LF and CRLF, without
    # requiring a trailing newline for the final line.
    text = data.decode("utf-8", errors="replace")
    return len(text.splitlines())


def inspect_job_log(path: Path | str) -> JobLogInfo:
    target = Path(path)
    if not target.exists() or not target.is_file():
        return JobLogInfo(
            path=target,
            exists=False,
            size_bytes=0,
            line_count=0,
            encoding="utf-8",
            modified_at_ns=0,
        )
    try:
        stat = target.stat()
        data = target.read_bytes()
    except OSError:
        return JobLogInfo(
            path=target,
            exists=False,
            size_bytes=0,
            line_count=0,
            encoding="utf-8",
            modified_at_ns=0,
        )
    return JobLogInfo(
        path=target,
        exists=True,
        size_bytes=int(stat.st_size),
        line_count=_count_lines(data),
        encoding="utf-8",
        modified_at_ns=int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
    )


def _decode_lines(data: bytes) -> list[str]:
    return data.decode("utf-8", errors="replace").splitlines()


def read_job_log_head(path: Path | str, lines: int) -> str:
    if lines < 1:
        raise JobLogError("Head line count must be positive.", reason_code="lines_invalid")
    target = Path(path)
    if not target.is_file():
        return ""
    try:
        data = target.read_bytes()
    except OSError:
        return ""
    selected = _decode_lines(data)[:lines]
    return "\n".join(selected)


def read_job_log_tail(path: Path | str, lines: int, *, max_bytes: int = 8_000_000) -> str:
    if lines < 1:
        raise JobLogError("Tail line count must be positive.", reason_code="lines_invalid")
    target = Path(path)
    if not target.is_file():
        return ""
    try:
        size = target.stat().st_size
        with target.open("rb") as handle:
            handle.seek(max(0, size - max_bytes))
            data = handle.read(max_bytes)
    except OSError:
        return ""
    selected = _decode_lines(data)[-lines:]
    return "\n".join(selected)


def read_job_log_full(path: Path | str) -> str:
    target = Path(path)
    if not target.is_file():
        return ""
    try:
        data = target.read_bytes()
    except OSError:
        return ""
    return "\n".join(_decode_lines(data))


def job_log_download_bytes(path: Path | str) -> bytes:
    target = Path(path)
    if not target.is_file():
        return b""
    try:
        return target.read_bytes()
    except OSError:
        return b""


def job_log_fingerprint(info: JobLogInfo) -> tuple[int, int, int]:
    return (info.size_bytes, info.modified_at_ns, info.line_count)


def format_log_window_status(
    *,
    mode: str,
    requested_lines: int,
    total_lines: int,
    rendered_lines: int,
) -> str:
    if total_lines <= 0:
        return "Showing empty log · 0 lines"
    if mode == "full":
        return f"Showing complete log · {total_lines:,} lines"
    if mode == "first":
        shown = min(requested_lines, total_lines)
        return f"Showing first {shown:,} of {total_lines:,} lines"
    shown = min(requested_lines, total_lines)
    return f"Showing last {shown:,} of {total_lines:,} lines"


__all__ = [
    "INLINE_RENDER_MAX_BYTES",
    "JobLogError",
    "JobLogInfo",
    "format_log_window_status",
    "inspect_job_log",
    "job_log_download_bytes",
    "job_log_fingerprint",
    "read_job_log_full",
    "read_job_log_head",
    "read_job_log_tail",
    "resolve_job_log_path",
]
