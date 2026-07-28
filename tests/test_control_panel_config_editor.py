from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path

import src.churn_ml.control_panel.config_editor as editor_module
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


def test_atomic_publication_loses_race_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_publish = editor_module._publish_new_file
    winner = "schema_version: 1\nvalue: concurrent-winner\n"

    def publish_after_competitor(temporary: Path, target: Path) -> None:
        target.write_text(winner, encoding="utf-8")
        original_publish(temporary, target)

    monkeypatch.setattr(editor_module, "_publish_new_file", publish_after_competitor)
    with pytest.raises(ConfigEditError, match="already exists"):
        save_config_copy(
            tmp_path,
            editable_root="artifacts/ui_configs",
            copy_name="raced.yaml",
            text="schema_version: 1\nvalue: losing-writer\n",
        )
    target = tmp_path / "artifacts/ui_configs/raced.yaml"
    assert target.read_text(encoding="utf-8") == winner
    assert list(target.parent.glob(".raced.yaml.*.tmp")) == []


def test_concurrent_writers_publish_exactly_one_copy(tmp_path: Path) -> None:
    barrier = threading.Barrier(4)

    def write(index: int) -> tuple[bool, str]:
        barrier.wait(timeout=5)
        text = f"schema_version: 1\nwriter: {index}\n"
        try:
            save_config_copy(
                tmp_path,
                editable_root="artifacts/ui_configs",
                copy_name="concurrent.yaml",
                text=text,
            )
        except ConfigEditError:
            return False, text
        return True, text

    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(write, range(4)))
    winners = [text for succeeded, text in outcomes if succeeded]
    assert len(winners) == 1
    target = tmp_path / "artifacts/ui_configs/concurrent.yaml"
    assert target.read_text(encoding="utf-8") == winners[0]
    assert list(target.parent.glob(".concurrent.yaml.*.tmp")) == []
