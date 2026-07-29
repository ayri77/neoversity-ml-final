"""Shared logical selection state for control-panel path placeholders.

Widget keys remain Streamlit-owned; this module stores validated values keyed by
operation + semantic placeholder role (+ name) so Validate/Run and navigation
can reuse the same selection without trusting stale or forged paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


LOGICAL_SELECTION_KEY = "_cp_logical_selection"


@dataclass(frozen=True)
class CascadeReconciliation:
    """Canonical path resolved from cascade parents in a single rerun."""

    path: str | None
    matched_paths: tuple[str, ...]
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.path is not None


def logical_selection_key(operation: str, role: str, name: str) -> str:
    """Return the durable key for an operation + semantic placeholder role."""
    return f"{operation}::{role}:{name}"


def widget_selection_key(operation: str, role: str, name: str) -> str:
    """Shared Streamlit widget key across actions of the same operation."""
    return f"value-{operation}-{role}-{name}"


def get_logical_selection(
    session_state: Any,
    operation: str,
    role: str,
    name: str,
) -> Any | None:
    store = session_state.get(LOGICAL_SELECTION_KEY)
    if not isinstance(store, dict):
        return None
    return store.get(logical_selection_key(operation, role, name))


def set_logical_selection(
    session_state: Any,
    operation: str,
    role: str,
    name: str,
    value: Any,
) -> None:
    store = session_state.setdefault(LOGICAL_SELECTION_KEY, {})
    if not isinstance(store, dict):
        store = {}
        session_state[LOGICAL_SELECTION_KEY] = store
    key = logical_selection_key(operation, role, name)
    if value in (None, ""):
        store.pop(key, None)
        return
    store[key] = value


def validate_against_allowed(
    value: Any,
    allowed: Sequence[Any] | None,
) -> Any | None:
    """Fail closed: reject values absent from the current allowed option set."""
    if value in (None, ""):
        return None
    if allowed is None:
        return value
    if value in allowed:
        return value
    return None


def resolve_logical_selection(
    session_state: Any,
    *,
    operation: str,
    role: str,
    name: str,
    allowed: Sequence[Any] | None,
    default: Any | None = None,
) -> Any | None:
    """Return a validated selection, clearing stale logical values when needed."""
    logical = get_logical_selection(session_state, operation, role, name)
    validated = validate_against_allowed(logical, allowed)
    if validated is None and logical not in (None, ""):
        set_logical_selection(session_state, operation, role, name, None)
    if validated is not None:
        return validated
    if allowed is not None and allowed:
        if default is not None and default in allowed:
            return default
        return allowed[0]
    return default


def seed_widget_from_logical(
    session_state: Any,
    *,
    operation: str,
    role: str,
    name: str,
    widget_key: str,
    allowed: Sequence[Any] | None,
    cascade_meta: Mapping[str, str] | None = None,
) -> Any | None:
    """Seed a widget from logical state without clobbering cascade parents.

    Stale or forged widget values that are not in ``allowed`` are discarded.
    Cascade parent keys (``__src`` / ``__mdl`` / ``__mode``) are only filled when
    missing so a parent change in the same rerun is not overwritten by an old
    logical path. Parent→child reconciliation happens in the cascade selector.
    """
    current = session_state.get(widget_key)
    validated_current = validate_against_allowed(current, allowed)
    if validated_current is None and current not in (None, ""):
        session_state.pop(widget_key, None)
        for suffix in ("__src", "__mdl", "__mode", "__flat", "__parent_fp"):
            session_state.pop(f"{widget_key}{suffix}", None)

    resolved = resolve_logical_selection(
        session_state,
        operation=operation,
        role=role,
        name=name,
        allowed=allowed,
        default=validated_current,
    )
    if resolved is None:
        return None

    if session_state.get(widget_key) != resolved:
        session_state[widget_key] = resolved
    if cascade_meta:
        _seed_missing_cascade_parent(
            session_state, f"{widget_key}__src", cascade_meta.get("source_kind")
        )
        _seed_missing_cascade_parent(
            session_state, f"{widget_key}__mdl", cascade_meta.get("model_family")
        )
        _seed_missing_cascade_parent(
            session_state, f"{widget_key}__mode", cascade_meta.get("mode")
        )
    return resolved


def _seed_missing_cascade_parent(
    session_state: Any, key: str, value: str | None
) -> None:
    if value in (None, ""):
        return
    if session_state.get(key) in (None, ""):
        session_state[key] = value


def reconcile_cascade_selection(
    *,
    allowed_paths: Sequence[str],
    matched_paths: Sequence[str],
    current_path: Any,
    default_index: int = 0,
) -> CascadeReconciliation:
    """Resolve one canonical path from cascade parents; fail closed on mismatch.

    The diagnostic raw selector must never override this result. Callers should
    sync widget + ``__flat`` keys from ``path`` in the same rerun.
    """
    allowed = [path for path in allowed_paths if path]
    matched = [path for path in matched_paths if path in allowed]
    if not matched:
        return CascadeReconciliation(
            path=None,
            matched_paths=(),
            error="No configuration matches the selected Source / Model / Mode.",
        )
    if current_path in matched:
        return CascadeReconciliation(
            path=str(current_path), matched_paths=tuple(matched), error=None
        )
    index = min(max(default_index, 0), len(matched) - 1)
    return CascadeReconciliation(
        path=matched[index], matched_paths=tuple(matched), error=None
    )


def apply_cascade_reconciliation(
    session_state: Any,
    *,
    widget_key: str,
    reconciliation: CascadeReconciliation,
    parent_fingerprint: tuple[str | None, ...],
) -> str | None:
    """Write canonical cascade path into widget + diagnostic keys atomically."""
    if not reconciliation.ok or reconciliation.path is None:
        session_state.pop(widget_key, None)
        session_state.pop(f"{widget_key}__flat", None)
        session_state[f"{widget_key}__parent_fp"] = parent_fingerprint
        return None
    canonical = reconciliation.path
    session_state[widget_key] = canonical
    session_state[f"{widget_key}__flat"] = canonical
    session_state[f"{widget_key}__parent_fp"] = parent_fingerprint
    return canonical


def remember_widget_selection(
    session_state: Any,
    *,
    operation: str,
    role: str,
    name: str,
    value: Any,
    allowed: Sequence[Any] | None,
) -> Any | None:
    """Persist a widget value into logical state only when currently allowed."""
    validated = validate_against_allowed(value, allowed)
    if validated is None:
        return None
    set_logical_selection(session_state, operation, role, name, validated)
    return validated
