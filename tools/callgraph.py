#!/usr/bin/env python3
"""Generate a call graph of blobmap as mermaid.

Static, from the AST. Emits text rather than an image so the output is
diffable, renders on GitHub and in mkdocs without graphviz, and can be
regenerated in CI to catch drift.

Only calls *within* blobmap are drawn. Calls into obstore, the stdlib and so
on are noise for the purpose of understanding how the package hangs together.

    python tools/callgraph.py                    # whole package
    python tools/callgraph.py --entry partition_store --depth 3
    python tools/callgraph.py --module resolve --private
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "blobmap"


@dataclass
class Function:
    """One function or method found in the package."""

    name: str
    module: str
    lineno: int
    calls: set[str] = field(default_factory=set)

    @property
    def qualname(self) -> str:
        return f"{self.module}.{self.name}"

    @property
    def is_private(self) -> bool:
        return self.name.split(".")[-1].startswith("_")


class Collector(ast.NodeVisitor):
    """Walk one module, recording definitions and the calls inside them."""

    def __init__(self, module: str) -> None:
        self.module = module
        self.functions: dict[str, Function] = {}
        self._stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def _function(self, node: ast.AST, name: str) -> None:
        qual = ".".join([*self._stack, name])
        fn = Function(qual, self.module, getattr(node, "lineno", 0))
        self.functions[qual] = fn
        self._stack.append(name)
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                target = _callee(child.func)
                if target:
                    fn.calls.add(target)
        self._stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node, node.name)


def _callee(node: ast.AST) -> str | None:
    """The bare name being called, ignoring what it was called on."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def collect(package: Path) -> dict[str, Function]:
    """Every function in the package, keyed by qualified name."""
    out: dict[str, Function] = {}
    for path in sorted(package.rglob("*.py")):
        module = path.relative_to(package).with_suffix("").as_posix()
        module = module.replace("/", ".").removesuffix(".__init__")
        tree = ast.parse(path.read_text())
        collector = Collector(module or "__init__")
        collector.visit(tree)
        for fn in collector.functions.values():
            out[fn.qualname] = fn
    return out


def edges(functions: dict[str, Function], *,
          private: bool) -> set[tuple[str, str]]:
    """Resolve call names to qualified names, dropping anything external."""
    by_name: dict[str, list[str]] = defaultdict(list)
    for qual, fn in functions.items():
        by_name[fn.name.split(".")[-1]].append(qual)

    out: set[tuple[str, str]] = set()
    for qual, fn in functions.items():
        if not private and fn.is_private:
            continue
        for call in fn.calls:
            for target in by_name.get(call, []):
                if target == qual:
                    continue           # recursion adds nothing to the picture
                if not private and functions[target].is_private:
                    continue
                out.add((qual, target))
    return out


def reachable(edges_: set[tuple[str, str]], entry: str, depth: int) -> set[str]:
    """Everything within `depth` calls of `entry`."""
    frontier = {q for q in {e[0] for e in edges_} | {e[1] for e in edges_}
                if q.endswith(f".{entry}") or q == entry}
    seen = set(frontier)
    for _ in range(depth):
        nxt = {b for a, b in edges_ if a in frontier} - seen
        seen |= nxt
        frontier = nxt
    return seen


def mermaid(functions: dict[str, Function], edges_: set[tuple[str, str]],
            keep: set[str] | None) -> str:
    """Render as a mermaid flowchart, grouped by module."""
    nodes = {a for a, _ in edges_} | {b for _, b in edges_}
    if keep is not None:
        nodes &= keep
        edges_ = {(a, b) for a, b in edges_ if a in nodes and b in nodes}

    by_module: dict[str, list[str]] = defaultdict(list)
    for qual in sorted(nodes):
        by_module[functions[qual].module].append(qual)

    ids = {qual: f"n{i}" for i, qual in enumerate(sorted(nodes))}
    lines = ["```mermaid", "flowchart LR"]
    for module in sorted(by_module):
        lines.append(f'  subgraph {module}["{module}"]')
        for qual in by_module[module]:
            label = qual.split(".", 1)[1] if "." in qual else qual
            lines.append(f'    {ids[qual]}["{label}"]')
        lines.append("  end")
    for a, b in sorted(edges_):
        lines.append(f"  {ids[a]} --> {ids[b]}")
    lines.append("```")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entry", help="start from this function, following calls")
    p.add_argument("--depth", type=int, default=3,
                   help="how many call levels to follow from --entry")
    p.add_argument("--module", help="only show functions in this module")
    p.add_argument("--private", action="store_true",
                   help="include underscore-prefixed helpers")
    args = p.parse_args(argv)

    functions = collect(PACKAGE)
    graph = edges(functions, private=args.private)

    keep: set[str] | None = None
    if args.entry:
        keep = reachable(graph, args.entry, args.depth)
        if not keep:
            print(f"no function named {args.entry!r}", file=sys.stderr)
            return 1
    if args.module:
        subset = {q for q in functions if functions[q].module == args.module}
        keep = subset if keep is None else keep & subset

    print(mermaid(functions, graph, keep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
