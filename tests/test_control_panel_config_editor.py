from __future__ import annotations

from pathlib import Path

import pytest

from src.churn_ml.control_panel.config_editor import (
    ConfigEditError,
    parse_config_text,
    read_config,
    save_config_copy,
)


def test_canonical_config_is_read_only_and_copy_is_new(tmp_path: Path) -> None:
    canonical = tmp_path / "configs" / "canonical.yaml"
    canonical.parent.mkdir()
    original = "schema_version: 1\nvalue: original\n"
    canonical.write_text(original, encoding="utf-8")
    text, loaded_path = read_config(
        tmp_path,
        "configs/canonical.yaml",
        allowed_roots=("configs",),
    )
    assert text == original
    assert loaded_path == canonical

    copied = save_config_copy(
        tmp_path,
        editable_root="artifacts/ui_configs",
        copy_name="experiment_copy.yaml",
        text="schema_version: 1\nvalue: changed\n",
    )
    assert copied.read_text(encoding="utf-8") == ("schema_version: 1\nvalue: changed\n")
    assert canonical.read_text(encoding="utf-8") == original
    with pytest.raises(ConfigEditError, match="already exists"):
        save_config_copy(
            tmp_path,
            editable_root="artifacts/ui_configs",
            copy_name="experiment_copy.yaml",
            text=text,
        )


@pytest.mark.parametrize(
    ("text", "suffix"),
    [
        ("[1, 2]", ".json"),
        ("- one\n- two\n", ".yaml"),
        ("key: !!python/object:unsafe {}", ".yaml"),
        ("{broken", ".json"),
    ],
)
def test_config_parser_rejects_non_mapping_or_unsafe_syntax(
    text: str, suffix: str
) -> None:
    with pytest.raises(ConfigEditError):
        parse_config_text(text, suffix)


def test_copy_name_cannot_escape_editable_root(tmp_path: Path) -> None:
    with pytest.raises(ConfigEditError):
        save_config_copy(
            tmp_path,
            editable_root="artifacts/ui_configs",
            copy_name="../escape.yaml",
            text="schema_version: 1\n",
        )
