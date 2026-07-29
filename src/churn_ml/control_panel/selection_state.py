"""Shared logical selection state for control-panel path placeholders.

Widget keys remain Streamlit-owned; this module stores validated values keyed by
operation + semantic placeholder role (+ name) so Validate/Run and navigation
can reuse the same selection without trusting stale or forged paths.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


LOGICAL_SELECTION_KEY = "_cp_logical_selection"


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
    """Seed a widget (and optional cascade parents) from logical state.

    Stale or forged widget values that are not in ``allowed`` are discarded.
    """
    current = session_state.get(widget_key)
    validated_current = validate_against_allowed(current, allowed)
    if validated_current is None and current not in (None, ""):
        session_state.pop(widget_key, None)
        for suffix in ("__src", "__mdl", "__mode", "__flat"):
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
            source = cascade_meta.get("source_kind")
            model = cascade_meta.get("model_family")
            mode = cascade_meta.get("mode")
            if source:
                session_state[f"{widget_key}__src"] = source
            if model:
                session_state[f"{widget_key}__mdl"] = model
            if mode:
                session_state[f"{widget_key}__mode"] = mode
    return resolved


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
