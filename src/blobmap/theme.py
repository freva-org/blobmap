"""Help output styling, when `rich-argparse` is installed.

Colour is the least interesting part of this. The useful part is
`HIGHLIGHTS`: patterns that pick out the things blobmap actually talks about
-- store URLs, sizes, blob identifiers, chunk keys -- so that reading
`--help` or a decision table is a matter of scanning rather than parsing.

Everything degrades. Without `rich-argparse` the formatter falls back to plain
argparse, and rich disables styling when stdout is not a terminal, so piped
output stays clean with no special handling.

`NO_COLOR` is handled here rather than left to rich. Rich reads it and drops
colour but keeps bold, which is defensible -- bold is not colour -- but
someone setting `NO_COLOR` is usually capturing output to a file, and finding
an escape sequence there is exactly what they were trying to avoid. So the
whole formatter falls back to plain argparse instead.
"""

from __future__ import annotations

import os
from typing import Any

#: Waterpark palette. This is the one thing to change to match the site.
#:
#: Any rich style string works: a named colour, a hex value like `#0b7285`, or
#: a combination such as `bold #0b7285`.
PALETTE: dict[str, str] = {
    "primary": "#1f8fa5",       # headings and group names
    "accent": "#7bc6d6",        # arguments and metavars
    "muted": "grey58",          # the program name, defaults
    "emphasis": "bold",         # literal syntax in examples
}

#: Regular expressions whose named groups are styled in help text.
#:
#: Each group name must correspond to an entry in `STYLES`. These are what
#: make the output look like it belongs to this tool rather than to argparse.
HIGHLIGHTS: tuple[str, ...] = (
    # s3://bucket, file:///path, memory://
    r"(?P<blobmap_url>\b[a-z][a-z0-9+.-]*://[^\s,)]*)",
    # 100GiB, 50G, 16 MiB
    r"(?P<blobmap_size>\b\d+(?:\.\d+)?\s?(?:[KMGTP]i?B|[KMGTP])\b)",
    # b_tas_2, b_pr_hurs_0
    r"(?P<blobmap_blob>\bb_[a-z0-9_]+\b)",
    # tas/c/5000/0/0, healpix/mean.zarr
    r"(?P<blobmap_key>\b[\w.-]+(?:/[\w.-]+){2,})",
    # --flag
    r"(?P<blobmap_flag>\s--[\w-]+)",
)

#: Style for each highlight group, plus overrides for argparse's own styles.
STYLES: dict[str, str] = {
    "argparse.args": PALETTE["accent"],
    "argparse.groups": f"bold {PALETTE['primary']}",
    "argparse.help": "default",
    "argparse.metavar": PALETTE["muted"],
    "argparse.prog": f"bold {PALETTE['primary']}",
    "argparse.syntax": PALETTE["emphasis"],
    "argparse.text": "default",
    "argparse.default": "italic " + PALETTE["muted"],
    "blobmap_url": PALETTE["accent"],
    "blobmap_size": PALETTE["primary"],
    "blobmap_blob": f"bold {PALETTE['primary']}",
    "blobmap_key": PALETTE["muted"],
    "blobmap_flag": PALETTE["accent"],
}


def disabled() -> bool:
    """Whether the user has asked for no styling.

    Returns:
        True if `NO_COLOR` is set to a non-empty value, per no-color.org.
    """
    return bool(os.environ.get("NO_COLOR", ""))


def apply(formatter: Any) -> Any:
    """Apply the theme to a `rich-argparse` formatter class.

    A no-op for the plain argparse fallback, which has no styles to set, so
    callers need not check whether rich is present.

    Args:
        formatter: The formatter class to style, modified in place.

    Returns:
        The same class, for convenience at an import site.
    """
    styles = getattr(formatter, "styles", None)
    if styles is None:
        return formatter                       # plain argparse: nothing to do

    styles.update(STYLES)
    formatter.highlights = list(
        getattr(formatter, "highlights", [])) + list(HIGHLIGHTS)
    formatter.group_name_formatter = str.lower
    return formatter
