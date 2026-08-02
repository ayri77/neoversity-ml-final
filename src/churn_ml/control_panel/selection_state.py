"""Shared logical selection state for control-panel path placeholders.

Widget keys remain Streamlit-owned; this module stores validated values keyed by
operation + semantic placeholder role (+ name) so Validate/Run and navigation
can reuse the same selection without trusting stale or forged paths.

Streamlit removes widget values when those widgets are not rendered (for example
after leaving the Run page). Durable entries under ``LOGICAL_SELECTION_KEY`` are
non-widget session state and survive in-session navigation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


LOGICAL_SELECTION_KEY = "_cp_logical_selection"

# Non-widget keys that must survive simulated / real page navigation.
DURABLE_SESSION_KEYS = frozenset(
    {
        LOGICAL_SELECTION_KEY,
        "run_prefill",
    }
)


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


def ui_durable_key(*parts: str) -> str:
    """Return a durable key for non-placeholder UI state (Run/Results/Jobs/editor)."""
    if not parts:
        raise ValueError("ui_durable_key requires at least one part")
    return "__ui__::" + ":".join(str(part) for part in parts)


def widget_selection_key(operation: str, role: str, name: str) -> str:
    """Shared Streamlit widget key across actions of the same operation."""
    return f"value-{operation}-{role}-{name}"


def cascade_parents_durable_key(operation: str, role: str, name: str) -> str:
    return f"{logical_selection_key(operation, role, name)}::cascade_parents"


def get_durable_value(session_state: Any, key: str) -> Any | None:
    store = _session_mapping_get(session_state, LOGICAL_SELECTION_KEY)
    if not isinstance(store, dict):
        return None
    return store.get(key)


def set_durable_value(session_state: Any, key: str, value: Any) -> None:
    existing = _session_mapping_get(session_state, LOGICAL_SELECTION_KEY)
    store = dict(existing) if isinstance(existing, dict) else {}
    if value in (None, ""):
        store.pop(key, None)
    else:
        store[key] = value
    _session_mapping_set(session_state, LOGICAL_SELECTION_KEY, store)


def force_session_value(session_state: Any, key: str, value: Any) -> None:
    """Set session state even when a widget key already exists in this run.

    Prefer calling this before widgets are instantiated. When a widget key is
    already live, writes go through Streamlit's mutable inner mapping so Load /
    resume flows do not raise StreamlitAPIException.
    """
    _session_mapping_set(session_state, key, value)


def _session_mapping_get(session_state: Any, key: str, default: Any = None) -> Any:
    """Read a key from dict-like or Streamlit SafeSessionState."""
    if isinstance(session_state, dict):
        return session_state.get(key, default)
    inner = getattr(getattr(session_state, "_state", None), "_new_session_state", None)
    if isinstance(inner, dict) and key in inner:
        return inner[key]
    try:
        return session_state[key]
    except Exception:
        return default


def _session_mapping_set(session_state: Any, key: str, value: Any) -> None:
    if isinstance(session_state, dict):
        session_state[key] = value
        return
    state = getattr(session_state, "_state", None)
    inner = getattr(state, "_new_session_state", None)
    if isinstance(inner, dict):
        inner[key] = value
    try:
        session_state[key] = value
    except Exception:
        if isinstance(inner, dict):
            return
        # Last-resort write for environments where only the proxy rejects the key.
        if state is not None and hasattr(state, "_new_session_state"):
            if not isinstance(state._new_session_state, dict):
                state._new_session_state = {}
            state._new_session_state[key] = value
            return
        raise


def get_logical_selection(
    session_state: Any,
    operation: str,
    role: str,
    name: str,
) -> Any | None:
    store = _session_mapping_get(session_state, LOGICAL_SELECTION_KEY)
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
    set_durable_value(
        session_state,
        logical_selection_key(operation, role, name),
        value,
    )


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


def _session_mapping_pop(session_state: Any, key: str) -> None:
    if isinstance(session_state, dict):
        session_state.pop(key, None)
        return
    inner = getattr(getattr(session_state, "_state", None), "_new_session_state", None)
    if isinstance(inner, dict):
        inner.pop(key, None)
    try:
        del session_state[key]
    except Exception:
        return


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

    A currently valid widget value is the latest user interaction and takes
    precedence over an older logical value. Logical state seeds the widget only
    when the widget value is missing/blank (for example after page navigation)
    or invalid/forged.

    Cascade parent keys (``__src`` / ``__mdl`` / ``__mode``) are only filled when
    missing so a parent change in the same rerun is not overwritten by an old
    logical path. Parent→child reconciliation happens in the cascade selector.
    """
    current = _session_mapping_get(session_state, widget_key)
    validated_current = validate_against_allowed(current, allowed)
    if validated_current is None and current not in (None, ""):
        _session_mapping_pop(session_state, widget_key)
        for suffix in ("__src", "__mdl", "__mode", "__flat", "__parent_fp"):
            _session_mapping_pop(session_state, f"{widget_key}{suffix}")

    if validated_current is not None:
        # Latest valid widget interaction wins over older logical state.
        set_logical_selection(
            session_state, operation, role, name, validated_current
        )
        _restore_cascade_parents(
            session_state,
            operation=operation,
            role=role,
            name=name,
            widget_key=widget_key,
            cascade_meta=cascade_meta,
        )
        return validated_current

    resolved = resolve_logical_selection(
        session_state,
        operation=operation,
        role=role,
        name=name,
        allowed=allowed,
        default=None,
    )
    if resolved is None:
        return None

    if _session_mapping_get(session_state, widget_key) != resolved:
        _session_mapping_set(session_state, widget_key, resolved)
    _restore_cascade_parents(
        session_state,
        operation=operation,
        role=role,
        name=name,
        widget_key=widget_key,
        cascade_meta=cascade_meta,
    )
    return resolved


def sync_widget_with_durable(
    session_state: Any,
    *,
    widget_key: str,
    durable_key: str,
    allowed: Sequence[Any] | None,
    default: Any | None = None,
) -> Any | None:
    """Bidirectional sync for non-placeholder widgets before they render.

    Prefer a currently valid widget value; otherwise seed from durable state;
    otherwise fall back to ``default`` / first allowed option.
    """
    current = _session_mapping_get(session_state, widget_key)
    validated_current = validate_against_allowed(current, allowed)
    if validated_current is None and current not in (None, ""):
        _session_mapping_pop(session_state, widget_key)

    if validated_current is not None:
        set_durable_value(session_state, durable_key, validated_current)
        return validated_current

    durable = get_durable_value(session_state, durable_key)
    validated_durable = validate_against_allowed(durable, allowed)
    if validated_durable is None and durable not in (None, ""):
        set_durable_value(session_state, durable_key, None)

    resolved = validated_durable
    if resolved is None:
        if allowed is not None and allowed:
            if default is not None and default in allowed:
                resolved = default
            else:
                resolved = allowed[0]
        elif default not in (None, ""):
            resolved = default

    if resolved in (None, ""):
        return None
    _session_mapping_set(session_state, widget_key, resolved)
    set_durable_value(session_state, durable_key, resolved)
    return resolved


def remember_durable_value(
    session_state: Any,
    *,
    durable_key: str,
    value: Any,
    allowed: Sequence[Any] | None,
) -> Any | None:
    validated = validate_against_allowed(value, allowed)
    if validated is None:
        return None
    set_durable_value(session_state, durable_key, validated)
    return validated


def _seed_missing_cascade_parent(
    session_state: Any, key: str, value: str | None
) -> None:
    if value in (None, ""):
        return
    if _session_mapping_get(session_state, key) in (None, ""):
        _session_mapping_set(session_state, key, value)


def _restore_cascade_parents(
    session_state: Any,
    *,
    operation: str,
    role: str,
    name: str,
    widget_key: str,
    cascade_meta: Mapping[str, str] | None,
) -> None:
    stored = get_durable_value(
        session_state, cascade_parents_durable_key(operation, role, name)
    )
    parents: dict[str, str] = {}
    if isinstance(stored, Mapping):
        for key in ("source_kind", "model_family", "mode"):
            value = stored.get(key)
            if value not in (None, ""):
                parents[key] = str(value)
    if cascade_meta:
        for key in ("source_kind", "model_family", "mode"):
            value = cascade_meta.get(key)
            if key not in parents and value not in (None, ""):
                parents[key] = str(value)
    _seed_missing_cascade_parent(
        session_state, f"{widget_key}__src", parents.get("source_kind")
    )
    _seed_missing_cascade_parent(
        session_state, f"{widget_key}__mdl", parents.get("model_family")
    )
    _seed_missing_cascade_parent(
        session_state, f"{widget_key}__mode", parents.get("mode")
    )


def remember_cascade_parents(
    session_state: Any,
    *,
    operation: str,
    role: str,
    name: str,
    widget_key: str,
) -> None:
    parents = {
        "source_kind": _session_mapping_get(session_state, f"{widget_key}__src"),
        "model_family": _session_mapping_get(session_state, f"{widget_key}__mdl"),
        "mode": _session_mapping_get(session_state, f"{widget_key}__mode"),
    }
    cleaned = {
        key: str(value)
        for key, value in parents.items()
        if value not in (None, "")
    }
    set_durable_value(
        session_state,
        cascade_parents_durable_key(operation, role, name),
        cleaned or None,
    )


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
    widget_key: str | None = None,
) -> Any | None:
    """Persist a widget value into logical state only when currently allowed."""
    validated = validate_against_allowed(value, allowed)
    if validated is None:
        return None
    set_logical_selection(session_state, operation, role, name, validated)
    if widget_key is not None:
        remember_cascade_parents(
            session_state,
            operation=operation,
            role=role,
            name=name,
            widget_key=widget_key,
        )
    return validated


def _session_state_keys(session_state: Any) -> list[str]:
    """Enumerate keys for both plain dicts and Streamlit SafeSessionState."""
    if isinstance(session_state, dict):
        return [str(key) for key in session_state]
    inner = getattr(session_state, "_state", None)
    keys: set[str] = set()
    for attr in ("_new_session_state", "_old_state"):
        candidate = getattr(inner, attr, None)
        if isinstance(candidate, dict):
            keys.update(str(key) for key in candidate)
    filtered = getattr(inner, "filtered_state", None)
    if isinstance(filtered, dict):
        keys.update(str(key) for key in filtered)
    if keys:
        return sorted(keys)
    to_dict = getattr(session_state, "to_dict", None)
    if callable(to_dict):
        try:
            mapping = to_dict()
            if isinstance(mapping, dict):
                return [str(key) for key in mapping]
        except Exception:
            pass
    return []


def snapshot_durable_session(session_state: Any) -> dict[str, Any]:
    """Capture durable non-widget session entries for navigation simulations."""
    snapshot: dict[str, Any] = {}
    for key_str in _session_state_keys(session_state):
        if key_str not in DURABLE_SESSION_KEYS and not key_str.startswith("_cp_"):
            continue
        inner = getattr(getattr(session_state, "_state", None), "_new_session_state", None)
        try:
            if isinstance(inner, dict) and key_str in inner:
                snapshot[key_str] = inner[key_str]
            elif isinstance(session_state, dict):
                snapshot[key_str] = session_state[key_str]
            else:
                snapshot[key_str] = session_state[key_str]
        except Exception:
            continue
    return snapshot


def restore_durable_session(session_state: Any, snapshot: Mapping[str, Any]) -> None:
    """Replace session contents with durable snapshot only (widget keys cleared)."""
    inner = getattr(getattr(session_state, "_state", None), "_new_session_state", None)
    if isinstance(inner, dict):
        for key in list(inner.keys()):
            if str(key).startswith("$$"):
                continue
            inner.pop(key, None)
        for key, value in snapshot.items():
            inner[str(key)] = value
        return
    if isinstance(session_state, dict):
        for key in list(session_state.keys()):
            session_state.pop(key, None)
        session_state.update({str(key): value for key, value in snapshot.items()})


def drop_transient_widget_keys(session_state: Any) -> list[str]:
    """Remove Streamlit widget keys while retaining durable backing state.

    Used by tests to simulate navigation away from a page (Streamlit deletes
    values for widgets that are not rendered). Safety confirmations and other
    transient keys are intentionally dropped.
    """
    before = set(_session_state_keys(session_state))
    snapshot = snapshot_durable_session(session_state)
    restore_durable_session(session_state, snapshot)
    after = set(_session_state_keys(session_state))
    return sorted(before - after)
