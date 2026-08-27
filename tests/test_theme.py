"""Styling must be invisible when it cannot or should not apply.

Help output gets piped, redirected to files and read by people who have
turned colour off. Escape codes leaking into any of those is worse than
having no theme at all.
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from blobmap import theme
from blobmap.cli import build_parser

ROOT = Path(__file__).resolve().parent.parent
ESCAPE = re.compile(r"\x1b\[")

rich = pytest.importorskip("rich_argparse")


def help_text(env: dict[str, str]) -> str:
    """Render the top-level help in a subprocess with a given environment."""
    result = subprocess.run(
        [sys.executable, "-c",
         "from blobmap.cli import build_parser; build_parser().print_help()"],
        capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, "COLUMNS": "80", **env})
    assert result.returncode == 0, result.stderr
    return result.stdout


# -- when styling must not appear -----------------------------------------

def test_piped_output_has_no_escapes():
    """The common case: `blobmap --help | less`, or redirected to a file."""
    assert not ESCAPE.search(help_text({}))


def test_no_color_leaves_no_escapes_at_all():
    """Rich reads NO_COLOR itself but keeps bold, which is defensible -- bold
    is not colour -- yet someone setting it is usually capturing output to a
    file, where any escape is the thing they were avoiding. So blobmap falls
    back to plain argparse entirely."""
    assert not ESCAPE.search(help_text({"NO_COLOR": "1", "FORCE_COLOR": "1"}))


@pytest.mark.parametrize("value,expected", [
    ("1", True), ("anything", True), ("", False),
])
def test_no_color_must_be_non_empty(monkeypatch, value, expected):
    """Per no-color.org the variable counts when present and non-empty,
    regardless of its value."""
    monkeypatch.setenv("NO_COLOR", value)
    assert theme.disabled() is expected


def test_schema_output_is_never_styled():
    """It is JSON that gets piped into a file or a validator."""
    result = subprocess.run(
        [sys.executable, "-m", "blobmap.cli", "schema"],
        capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, "FORCE_COLOR": "1"})
    assert not ESCAPE.search(result.stdout)
    import json
    json.loads(result.stdout)


# -- when it should ---------------------------------------------------------

def test_the_palette_is_actually_applied():
    text = help_text({"FORCE_COLOR": "1", "COLORTERM": "truecolor",
                      "TERM": "xterm-256color"})
    red, green, blue = _rgb(theme.PALETTE["primary"])
    assert f"38;2;{red};{green};{blue}" in text, "primary colour not emitted"


def _rgb(value: str) -> tuple[int, int, int]:
    digits = value.split("#")[-1]
    return tuple(int(digits[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


@pytest.mark.parametrize("sample", [
    "s3://cordex", "50GiB", "b_tas_2", "tas/c/5000/0/0", " --dry-run",
])
def test_highlights_match_what_blobmap_talks_about(sample):
    """These patterns are the point of the theme: scanning the help for a URL
    or a size should not mean reading every word."""
    assert any(re.search(pattern, sample) for pattern in theme.HIGHLIGHTS), \
        f"nothing highlights {sample!r}"


def test_highlights_do_not_match_ordinary_prose():
    prose = "the partitioner descends into each store and cuts it"
    for pattern in theme.HIGHLIGHTS:
        assert not re.search(pattern, prose), pattern


def test_every_highlight_group_has_a_style():
    """An unstyled group silently renders as plain text."""
    for pattern in theme.HIGHLIGHTS:
        for name in re.findall(r"\(\?P<(\w+)>", pattern):
            assert name in theme.STYLES, f"{name} has no style"


# -- the fallback -----------------------------------------------------------

def test_apply_is_a_no_op_without_rich():
    """So callers need not check whether rich is installed."""
    import argparse
    plain = argparse.RawDescriptionHelpFormatter
    assert theme.apply(plain) is plain
    assert not hasattr(plain, "styles")


def test_theme_imports_without_rich():
    """blobmap.theme is stdlib only; it must not grow a rich import."""
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.modules['rich_argparse'] = None; "
         "import blobmap.theme; print(len(blobmap.theme.HIGHLIGHTS))"],
        capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(len(theme.HIGHLIGHTS))
