"""Optional Control Panel render/performance instrumentation.

Disabled by default. Enable with environment variable
``CHURN_ML_CONTROL_PANEL_PERF=1`` or by calling ``enable_performance()``.

Captured data stays in-process (session dict / console). Nothing is written
under repository ``artifacts/``.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

try:
    import streamlit as _streamlit
except Exception:  # pragma: no cover - Streamlit optional outside UI extra
    _streamlit = None


_ENV_FLAG = "CHURN_ML_CONTROL_PANEL_PERF"
_ENABLED = False
_STAGES: list["StageRecord"] = []
_COUNTERS: dict[str, int] = {}
_STACK: list[str] = []
_RENDER_STARTED: float | None = None


@dataclass
class StageRecord:
    name: str
    elapsed_ms: float
    call_count: int = 1
    files_inspected: int = 0
    files_read: int = 0
    bytes_read: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    parent: str | None = None


@dataclass
class _ActiveStage:
    name: str
    started: float
    files_inspected: int = 0
    files_read: int = 0
    bytes_read: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    parent: str | None = None


_ACTIVE: list[_ActiveStage] = []


def performance_enabled() -> bool:
    if _ENABLED:
        return True
    value = os.environ.get(_ENV_FLAG, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def enable_performance(enabled: bool = True) -> None:
    global _ENABLED
    _ENABLED = bool(enabled)


def reset_performance() -> None:
    """Clear in-memory stage/counter state without changing the enable flag."""
    _STAGES.clear()
    _COUNTERS.clear()
    _STACK.clear()
    _ACTIVE.clear()
    global _RENDER_STARTED
    _RENDER_STARTED = None


def begin_render() -> None:
    if not performance_enabled():
        return
    global _RENDER_STARTED
    _RENDER_STARTED = time.perf_counter()


def record_counter(name: str, count: int = 1) -> None:
    if not performance_enabled():
        return
    if count == 0:
        return
    key = str(name)
    _COUNTERS[key] = int(_COUNTERS.get(key, 0)) + int(count)
    if _ACTIVE:
        current = _ACTIVE[-1]
        if key.endswith("_files_inspected") or key == "files_inspected":
            current.files_inspected += int(count)
        elif key.endswith("_files_read") or key == "files_read":
            current.files_read += int(count)
        elif key.endswith("_bytes_read") or key == "bytes_read":
            current.bytes_read += int(count)
        elif key.endswith("_cache_hit") or key == "cache_hit":
            current.cache_hits += int(count)
        elif key.endswith("_cache_miss") or key == "cache_miss":
            current.cache_misses += int(count)


def note_files_inspected(count: int = 1) -> None:
    record_counter("files_inspected", count)


def note_files_read(count: int = 1, *, bytes_read: int = 0) -> None:
    record_counter("files_read", count)
    if bytes_read:
        record_counter("bytes_read", int(bytes_read))


def note_cache_hit(count: int = 1) -> None:
    record_counter("cache_hit", count)


def note_cache_miss(count: int = 1) -> None:
    record_counter("cache_miss", count)


@contextmanager
def performance_stage(name: str) -> Iterator[None]:
    if not performance_enabled():
        yield
        return
    stage = _ActiveStage(
        name=str(name),
        started=time.perf_counter(),
        parent=_STACK[-1] if _STACK else None,
    )
    _STACK.append(stage.name)
    _ACTIVE.append(stage)
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - stage.started) * 1000.0
        _ACTIVE.pop()
        _STACK.pop()
        _STAGES.append(
            StageRecord(
                name=stage.name,
                elapsed_ms=elapsed_ms,
                files_inspected=stage.files_inspected,
                files_read=stage.files_read,
                bytes_read=stage.bytes_read,
                cache_hits=stage.cache_hits,
                cache_misses=stage.cache_misses,
                parent=stage.parent,
            )
        )


def performance_snapshot() -> dict[str, Any]:
    """Return a redacted in-memory snapshot suitable for UI/console diagnostics."""
    total_ms: float | None = None
    if _RENDER_STARTED is not None:
        total_ms = (time.perf_counter() - _RENDER_STARTED) * 1000.0
    aggregated = _aggregate_stages(_STAGES)
    return {
        "enabled": performance_enabled(),
        "total_render_ms": total_ms,
        "stages": aggregated,
        "counters": dict(sorted(_COUNTERS.items())),
    }


def format_performance_snapshot(snapshot: Mapping[str, Any] | None = None) -> str:
    data = dict(snapshot or performance_snapshot())
    lines = [
        f"enabled={data.get('enabled')}",
        f"total_render_ms={_fmt_ms(data.get('total_render_ms'))}",
        "stages:",
    ]
    for stage in data.get("stages") or []:
        lines.append(
            "  - "
            f"{stage.get('name')}: {_fmt_ms(stage.get('elapsed_ms'))} ms "
            f"(calls={stage.get('call_count', 0)}, "
            f"files_inspected={stage.get('files_inspected', 0)}, "
            f"files_read={stage.get('files_read', 0)}, "
            f"bytes_read={stage.get('bytes_read', 0)}, "
            f"cache_hits={stage.get('cache_hits', 0)}, "
            f"cache_misses={stage.get('cache_misses', 0)})"
        )
    counters = data.get("counters") or {}
    if counters:
        lines.append("counters:")
        for key, value in counters.items():
            lines.append(f"  - {key}: {value}")
    return "\n".join(lines)


def render_performance_expander(streamlit_module: Any | None = None) -> None:
    """Optional Streamlit expander. No-op when disabled or Streamlit missing."""
    if not performance_enabled():
        return
    st = streamlit_module if streamlit_module is not None else _streamlit
    if st is None:
        return
    with st.expander("Performance diagnostics", expanded=False):
        st.code(format_performance_snapshot(), language="text")


def _aggregate_stages(stages: list[StageRecord]) -> list[dict[str, Any]]:
    ordered: list[str] = []
    by_name: dict[str, dict[str, Any]] = {}
    for stage in stages:
        item = by_name.get(stage.name)
        if item is None:
            ordered.append(stage.name)
            by_name[stage.name] = {
                "name": stage.name,
                "elapsed_ms": float(stage.elapsed_ms),
                "call_count": 1,
                "files_inspected": int(stage.files_inspected),
                "files_read": int(stage.files_read),
                "bytes_read": int(stage.bytes_read),
                "cache_hits": int(stage.cache_hits),
                "cache_misses": int(stage.cache_misses),
                "parent": stage.parent,
            }
            continue
        item["elapsed_ms"] = float(item["elapsed_ms"]) + float(stage.elapsed_ms)
        item["call_count"] = int(item["call_count"]) + 1
        item["files_inspected"] = int(item["files_inspected"]) + int(
            stage.files_inspected
        )
        item["files_read"] = int(item["files_read"]) + int(stage.files_read)
        item["bytes_read"] = int(item["bytes_read"]) + int(stage.bytes_read)
        item["cache_hits"] = int(item["cache_hits"]) + int(stage.cache_hits)
        item["cache_misses"] = int(item["cache_misses"]) + int(stage.cache_misses)
    return [by_name[name] for name in ordered]


def _fmt_ms(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


__all__ = [
    "StageRecord",
    "begin_render",
    "enable_performance",
    "format_performance_snapshot",
    "note_cache_hit",
    "note_cache_miss",
    "note_files_inspected",
    "note_files_read",
    "performance_enabled",
    "performance_snapshot",
    "performance_stage",
    "record_counter",
    "render_performance_expander",
    "reset_performance",
]
