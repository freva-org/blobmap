"""LIST-driven discovery: find zarr stores, and which of them lack a manifest.

No database. A delimited walk plus an existence check, which is what keeps
blobmap testable end to end against a memory backend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator

from ..hierarchy import detect_format
from ..manifests import ManifestStore
from ..storage import Store, list_dirs

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    """A zarr store found by a scan.

    Attributes:
        scope: The store prefix, relative to the data store root.
        fmt: `"v2"` or `"v3"`.
        has_manifest: Whether this scope has already been partitioned.
    """

    scope: str
    fmt: str
    has_manifest: bool


def scan(data: Store, root: str, manifests: ManifestStore, *,
         max_depth: int = 6) -> Iterator[Candidate]:
    """Walk a prefix and yield the zarr stores under it.

    Descent stops at a store boundary. A datatree may put an entire bucket in
    one store, so walking in to look for more would mean listing hundreds of
    thousands of chunk keys to no purpose.

    Args:
        data: A storage handle for the data.
        root: Prefix to scan under. Empty scans everything.
        manifests: Used only to find out which scopes are already known.
        max_depth: How far to descend before giving up on a branch. A store
            nested deeper than this is not found, so raise it rather than
            wonder why something is missing.

    Yields:
        One [`Candidate`][blobmap.discover.scan.Candidate] per store.

    Example:
        ```python
        for candidate in scan(data, "cordex", manifests):
            if not candidate.has_manifest:
                partition_store(data, manifests, candidate.scope)
        ```
    """
    known = {s.strip("/") for s in manifests.scopes()}
    yield from _descend(data, root.strip("/"), known, max_depth)


def _descend(data: Store, scope: str, known: set[str],
             depth: int) -> Iterator[Candidate]:
    """Recurse until a store is found or the depth budget runs out.

    Args:
        data: A storage handle.
        scope: Prefix currently being examined.
        known: Scopes that already have a manifest.
        depth: Remaining descent budget.

    Yields:
        Candidates found in this subtree.
    """
    fmt = detect_format(data, scope)
    if fmt is not None:
        yield Candidate(scope, fmt, scope in known)
        return                       # do not walk into the store
    if depth <= 0:
        log.debug("%s: max depth reached", scope)
        return
    for child in list_dirs(data, f"{scope}/" if scope else ""):
        yield from _descend(data, child.strip("/"), known, depth - 1)
