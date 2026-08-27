"""Diagrams and the generated call graph must stay honest.

Mermaid renders on GitHub and in mkdocs, so a syntax error is invisible until
someone opens the page and sees a red box. These checks are structural rather
than a full parse, but they catch the mistakes that actually happen: an edge
to a node nobody declared, and a call graph that has drifted from the code.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = sorted((ROOT / "docs").glob("*.md"))

DECLARATION = re.compile(r"(\w+)(?:\[|\{|\(\[|\[\()")
SUBGRAPH = re.compile(r"subgraph\s+(\w+)")
EDGE = re.compile(r"(\w+)\s*-[.-]*->(?:\|[^|]*\|)?\s*(\w+)")


def blocks(path: Path) -> list[str]:
    return re.findall(r"```mermaid\n(.*?)```", path.read_text(), re.S)


ALL_BLOCKS = [(path.name, i, block)
              for path in DOCS
              for i, block in enumerate(blocks(path))]


def test_there_are_diagrams_to_check():
    assert len(ALL_BLOCKS) >= 8


@pytest.mark.parametrize("name,index,block", ALL_BLOCKS,
                         ids=[f"{n}:{i}" for n, i, _ in ALL_BLOCKS])
def test_diagram_is_well_formed(name, index, block):
    header = block.strip().splitlines()[0]
    assert header.startswith(("flowchart", "graph", "sequenceDiagram")), header

    declared = set(DECLARATION.findall(block)) | set(SUBGRAPH.findall(block))
    used: set[str] = set()
    for match in EDGE.finditer(block):
        used |= {match.group(1), match.group(2)}

    undeclared = used - declared
    assert not undeclared, f"{name} block {index} links to {undeclared}"


@pytest.mark.parametrize("name,index,block", ALL_BLOCKS,
                         ids=[f"{n}:{i}" for n, i, _ in ALL_BLOCKS])
def test_subgraphs_are_balanced(name, index, block):
    opens = len(SUBGRAPH.findall(block))
    closes = len(re.findall(r"^\s*end\s*$", block, re.M))
    assert opens == closes, f"{name} block {index}: {opens} subgraph, {closes} end"


# -- the call graph is generated, so it must match the code ----------------

def test_callgraph_tool_runs():
    result = subprocess.run(
        [sys.executable, "tools/callgraph.py", "--entry", "partition_store"],
        capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert "flowchart LR" in result.stdout


def test_callgraph_reports_a_missing_entry():
    result = subprocess.run(
        [sys.executable, "tools/callgraph.py", "--entry", "no_such_function"],
        capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 1
    assert "no function named" in result.stderr


def test_documented_callgraph_edges_still_exist():
    """The graph in internals.md is hand-laid-out for readability, so it is
    not byte-identical to the tool's output. It must still describe real
    calls: if partition_store stops calling read_arrays, this fails."""
    generated = subprocess.run(
        [sys.executable, "tools/callgraph.py", "--entry", "partition_store",
         "--depth", "4"],
        capture_output=True, text=True, cwd=ROOT).stdout

    labels = dict(re.findall(r'(n\d+)\["([^"]+)"\]', generated))
    real = {(labels[a].split(".")[-1], labels[b].split(".")[-1])
            for a, b in re.findall(r"(n\d+) --> (n\d+)", generated)}

    documented = {
        ("partition_store", "read_arrays"),
        ("partition_store", "partition"),
        ("partition_store", "diff"),
        ("read_arrays", "list_all"),
        ("write", "put_bytes"),
        ("read", "loads"),
        ("partition", "default_hot_always"),
    }
    missing = documented - real
    assert not missing, f"internals.md documents calls that no longer exist: {missing}"


# -- dark mode -------------------------------------------------------------

COLOUR_DIRECTIVE = re.compile(r"\b(fill|color|stroke)\s*:", re.I)


def _styles(block: str) -> list[str]:
    """Style and classDef declarations, one per line."""
    return [line.strip() for line in block.splitlines()
            if line.strip().startswith(("style ", "classDef ", "class "))]


@pytest.mark.parametrize("name,index,block", ALL_BLOCKS,
                         ids=[f"{n}:{i}" for n, i, _ in ALL_BLOCKS])
def test_no_hardcoded_colours(name, index, block):
    """Diagrams must inherit their colours from whatever theme is rendering
    them.

    Every renderer overrides a different part: GitHub light, GitHub dark and
    Material's palette toggle all differ, and mkdocs-material re-sets
    `.mermaid text` with higher specificity than a `classDef color:`, so a
    hardcoded light fill ends up carrying the theme's own light text. Letting
    the theme pick both is the only combination legible everywhere.

    For emphasis use a labelled subgraph or `<b>` in the label instead.
    """
    offenders = [line for line in block.splitlines()
                 if COLOUR_DIRECTIVE.search(line)]
    assert not offenders, (
        f"{name} block {index} hardcodes colours, which will be unreadable "
        f"under some theme: {offenders}")


@pytest.mark.parametrize("name,index,block", ALL_BLOCKS,
                         ids=[f"{n}:{i}" for n, i, _ in ALL_BLOCKS])
def test_no_style_or_classdef(name, index, block):
    """The only reason to reach for these is colour, so ban them outright."""
    banned = [line for line in _styles(block)
              if line.startswith(("style ", "classDef "))]
    assert not banned, f"{name} block {index}: {banned}"
