from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any, Mapping

from src.churn_ml.control_panel.schemas import SAFE_BASENAME, SUGGESTED_TEMPLATE_REF


class SuggestionError(ValueError):
    """Raised when a declarative path suggestion cannot be resolved safely."""


def sanitize_path_basename(raw: str) -> str:
    text = str(raw).replace("\\", "/").strip()
    if not text or text in {".", ".."}:
        raise SuggestionError("Path basename is empty or unsafe.")
    name = PurePosixPath(text).name
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise SuggestionError("Path basename is empty or unsafe.")
    if name.startswith(".") or ".." in name:
        raise SuggestionError("Path basename must not introduce traversal.")
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    if not sanitized or SAFE_BASENAME.fullmatch(sanitized) is None:
        raise SuggestionError("Path basename could not be sanitized safely.")
    return sanitized


def render_suggested_value_template(
    template: str,
    values: Mapping[str, Any],
) -> str:
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        attribute = match.group(2)
        raw = values.get(name)
        if raw in (None, ""):
            missing.append(name)
            return "missing"
        text = str(raw)
        if attribute == "basename":
            return sanitize_path_basename(text)
        raise SuggestionError(
            "Suggested templates may only expand basename attributes."
        )

    rendered = SUGGESTED_TEMPLATE_REF.sub(replace, template)
    if missing:
        raise SuggestionError(
            f"Suggested template is missing values: {sorted(set(missing))}."
        )
    path = PurePosixPath(rendered)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or path.as_posix() != rendered
        or "\\" in rendered
    ):
        raise SuggestionError("Suggested path escaped the safe relative contract.")
    return path.as_posix()


def path_is_within_roots(relative_path: str, roots: tuple[str, ...]) -> bool:
    path = PurePosixPath(relative_path)
    for root in roots:
        root_path = PurePosixPath(root)
        try:
            path.relative_to(root_path)
            return True
        except ValueError:
            continue
    return False
