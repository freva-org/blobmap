"""Reading the format: object key to blob id.

This lives in blobmap rather than in the consuming service because it *is*
the semantics of the manifest. If a consumer reimplements the parse rules,
the format and its interpretation drift, and the failure mode is silent
misattribution of chunks to blobs.

The trie holds one node per cut, never per object. A store with 300,000
objects contributes a handful of nodes, and a bucketed array is a single node
however many blobs it spans, because the id is arithmetic rather than an
entry. A petabyte at 100 GB blobs is single digit megabytes of Python dicts.

Example:
    >>> from blobmap.model import Blob, Bucket, Manifest
    >>> manifest = Manifest("cordex/a.zarr",
    ...                     (Blob("b_tas", ("tas/c",),
    ...                           Bucket(0, 2048, "v3_slash")),),
    ...                     ("**/zarr.json",))
    >>> trie = Trie()
    >>> trie.add(manifest)
    >>> trie.lookup("cordex/a.zarr/tas/c/5000/0/0").blob_id
    'b_tas_2'
    >>> trie.lookup("cordex/a.zarr/zarr.json").kind
    'hot'
    >>> trie.lookup("somewhere/else/entirely").kind
    'unmanaged'
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .model import METADATA_BASENAMES, Blob, Bucket, Manifest

_MARK = "\0mark"  # marker key; cannot collide with a path segment


@dataclass(frozen=True)
class Resolution:
    """What a key resolves to.

    Attributes:
        kind: One of `blob`, `hot`, `pinned` or `unmanaged`. `hot` means a
            metadata object or coordinate, held back because of what it is.
            `pinned` means someone deliberately asked for it to stay on disk;
            it is reported separately so an operator can tell a structural
            decision from a human one. `unmanaged` means nothing claims this
            key, which is normal and safe rather than an error.
        blob_id: The concrete id, such as `b_tas_2`, or `None` unless `kind`
            is `blob`. This is the key against blobtier's state table.
        blob: The definition that matched, for callers that need its
            prefixes or bucket.
        bucket_index: Which bucket the key fell into, 0 for an unbucketed
            blob.

    Example:
        >>> Resolution("hot").archivable
        False
    """

    kind: str
    blob_id: str | None = None
    blob: Blob | None = None
    bucket_index: int | None = None

    @property
    def archivable(self) -> bool:
        """Whether this key may be moved to tape.

        Returns:
            True only for `kind == "blob"`. Metadata, coordinates and
            unmanaged keys all stay hot.
        """
        return self.kind == "blob"


HOT = Resolution("hot")
PINNED = Resolution("pinned")
UNMANAGED = Resolution("unmanaged")


class Trie:
    """Segment-wise longest prefix match over all known manifests.

    Deepest declaration wins, so a store that outgrows a parent scope
    manifest and gets its own simply overrides it for that subtree. There is
    no special case in the lookup and nesting needs no validation.

    Build once at startup from
    [`ManifestStore.load_all`][blobmap.manifests.ManifestStore.load_all],
    then rebind atomically on reload. In-flight lookups finish against the
    old trie and the next one sees the new, so no lock is needed.

    Example:
        >>> from blobmap.model import Blob, Manifest
        >>> trie = Trie()
        >>> trie.add_all([Manifest("a", (Blob("b_a", ("x",)),), ()),
        ...               Manifest("a/x", (Blob("b_deep", ("y",)),), ())])
        >>> trie.lookup("a/x/y/0").blob_id     # deepest declaration wins
        'b_deep_0'
    """

    def __init__(self) -> None:
        self._root: dict[str, Any] = {}
        # pins live in their own trie because they must win regardless of
        # depth. A longest-prefix match over a single trie would let a blob at
        # `tas/c` beat a pin on `tas`, which is exactly the case pinning
        # exists for.
        self._pins: dict[str, Any] = {}
        self._hot_basenames: set[str] = set(METADATA_BASENAMES)
        self._epochs: dict[str, int] = {}

    # -- build ------------------------------------------------------------

    def add(self, manifest: Manifest) -> None:
        """Insert one manifest's declarations.

        Args:
            manifest: The manifest to index. Its scope is prepended to every
                prefix, so keys are matched absolutely.
        """
        scope = manifest.scope.strip("/")
        self._epochs[scope] = manifest.epoch

        for pattern in manifest.hot_always:
            if pattern.startswith("**/"):
                # a bare basename: cheaper to check than to walk
                self._hot_basenames.add(pattern[3:])
            else:
                self._insert(_join(scope, pattern.rstrip("*/")), HOT)

        for blob in manifest.blobs:
            for prefix in blob.prefixes:
                self._insert(_join(scope, prefix), blob)

        for pin in manifest.pinned:
            self._insert(_join(scope, pin.prefix), PINNED, self._pins)

    def add_all(self, manifests: Iterable[Manifest]) -> None:
        """Insert many manifests.

        Args:
            manifests: Usually the result of `ManifestStore.load_all()`.
        """
        for m in manifests:
            self.add(m)

    def _insert(
        self, path: str, value: Any, root: dict[str, Any] | None = None
    ) -> None:
        """Place a marker at the node for `path`.

        Args:
            path: Absolute prefix, split on `/`.
            value: A `Blob` or a `Resolution` sentinel.
            root: Which trie to insert into, defaulting to the main one.
        """
        node = self._root if root is None else root
        for part in _split(path):
            node = node.setdefault(part, {})
        node[_MARK] = value

    @property
    def epochs(self) -> dict[str, int]:
        """Epoch of every indexed scope.

        Returns:
            Mapping of scope to epoch, for deciding whether a reload is
            needed without diffing the definitions.
        """
        return dict(self._epochs)

    def __len__(self) -> int:
        """Get the number of trie nodes, one per cut and never per object.

        Returns:
            Node count, useful as a sanity check that the trie has not
            accidentally grown per chunk.
        """
        return _count(self._root)

    # -- lookup -----------------------------------------------------------

    def lookup(self, key: str) -> Resolution:
        """Map an absolute object key to a blob.

        Misses are normal and safe. An unpartitioned store, foreign data or a
        brand new upload resolves to unmanaged, which means hot. No database
        call and no exception. Only registered blobs are archivable, so a
        miss is the conservative default rather than something to handle.

        Args:
            key: Full object key, including the scope prefix.

        Returns:
            A `Resolution`. Never `None`.

        Example:
            >>> from blobmap.model import Blob, Bucket, Manifest
            >>> trie = Trie()
            >>> trie.add(Manifest("s", (Blob("b_tas", ("tas/c",),
            ...                              Bucket(0, 100, "v3_slash")),), ()))
            >>> trie.lookup("s/tas/c/250/0/0").bucket_index
            2
            >>> trie.lookup("s/tas/c/notanumber/0").kind
            'unmanaged'
        """
        parts = _split(key)
        if not parts:
            return UNMANAGED
        if parts[-1] in self._hot_basenames:
            return HOT
        if self._pins and _covered(self._pins, parts):
            return PINNED

        node: dict[str, Any] | None = self._root
        best: Any = None
        best_depth = 0
        for depth, part in enumerate(parts, start=1):
            assert node is not None
            node = node.get(part)
            if node is None:
                break
            if _MARK in node:
                best, best_depth = node[_MARK], depth

        if best is None:
            return UNMANAGED
        if isinstance(best, Resolution):
            return best

        blob: Blob = best
        n = _bucket_index(blob.bucket, parts, best_depth)
        if n is None:
            return UNMANAGED
        return Resolution("blob", blob.instance(n), blob, n)


def _covered(root: dict[str, Any], parts: list[str]) -> bool:
    """Whether any prefix in `root` is an ancestor of `parts`.

    Unlike the main lookup this stops at the *first* match rather than the
    deepest, because every marker in the pin trie means the same thing.

    Args:
        root: The pin trie.
        parts: The key, already split.

    Returns:
        True if a pin covers the key.
    """
    node = root
    for part in parts:
        if _MARK in node:
            return True
        child = node.get(part)
        if child is None:
            return False
        node = child
    return _MARK in node


def _bucket_index(bucket: Bucket | None, parts: list[str], depth: int) -> int | None:
    """Parse the chunk index out of a key and divide it into a bucket.

    Args:
        bucket: The bucketing rule, or `None` for a plain prefix blob.
        parts: The key, already split on `/`.
        depth: How many segments the matched prefix consumed.

    Returns:
        The bucket number, 0 when `bucket` is `None`, or `None` when the key
        does not parse as a chunk key under this prefix. A prefix match alone
        is not enough.
    """
    if bucket is None:
        return 0
    try:
        if bucket.key_encoding == "v2_flat":
            # one segment holds the whole index: tas/137.0.0
            raw = parts[depth].split(".")[bucket.index]
        else:
            raw = parts[depth + bucket.index]
        value = int(raw)
    except (IndexError, ValueError):
        return None  # not a chunk key under this prefix after all
    if value < 0:
        return None
    return value // bucket.width


def resolve(manifest: Manifest, key: str) -> Resolution:
    """Resolve one key against one manifest.

    Builds a throwaway trie, so this is for tests and one-off inspection. A
    service should build a [`Trie`][blobmap.resolve.Trie] once and reuse it.

    Args:
        manifest: The manifest to resolve against.
        key: Key *relative to the manifest scope*, unlike
            [`Trie.lookup`][blobmap.resolve.Trie.lookup] which takes an
            absolute key.

    Returns:
        A `Resolution`.

    Example:
        >>> from blobmap.model import Blob, Manifest
        >>> resolve(Manifest("s", (Blob("b_pr", ("pr/c",)),), ()),
        ...         "pr/c/7/0/0").blob_id
        'b_pr_0'
    """
    trie = Trie()
    trie.add(manifest)
    return trie.lookup(_join(manifest.scope.strip("/"), key))


def _split(path: str) -> list[str]:
    """Split a key into non-empty segments.

    Args:
        path: A key or prefix.

    Returns:
        Segments, with empty ones dropped so leading and trailing slashes do
        not matter.
    """
    return [p for p in path.split("/") if p]


def _join(*parts: str) -> str:
    """Join path fragments, ignoring empties and stray slashes.

    Args:
        *parts: Fragments to join.

    Returns:
        A normalised path.
    """
    return "/".join(p.strip("/") for p in parts if p.strip("/"))


def _count(node: dict[str, Any]) -> int:
    """Count nodes below and including this one, excluding markers.

    Args:
        node: A trie node.

    Returns:
        Node count.
    """
    return sum(
        1 + _count(v) for k, v in node.items() if k != _MARK and isinstance(v, dict)
    )
