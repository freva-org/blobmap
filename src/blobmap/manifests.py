"""Where manifests live: a bucket you own, at a path mirroring the data.

That mirroring is what makes this work for stores you must not alter --
nothing is ever written into the source. Inline `_blob_root` attributes, where
a store owner sets them, are an *input* to the partitioner; the manifest is
always the resolved output and the single authority for lookup.

Manifests are per *scope*, not per blob, so a PB is hundreds to a few thousand
small JSON objects. `load_all` is a LIST plus parallel GETs, which is why
blobtier needs no database at startup.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterator

from .model import Manifest
from .storage import Conflict, Store, get_bytes, head, list_all, put_bytes

log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class Stored:
    """A manifest as read, with the etag needed to update it safely.

    Attributes:
        manifest: The parsed manifest.
        etag: Entity tag at the time of reading. Pass it back to
            [`write`][blobmap.manifests.ManifestStore.write] so a concurrent
            update is rejected rather than silently overwritten.
    """

    manifest: Manifest
    etag: str | None


class ManifestStore:
    """Read and write manifests in a bucket you own.

    Args:
        store: A storage handle for the *manifest* bucket, not the data.
        prefix: Optional prefix within that bucket, if manifests share it
            with something else.

    Example:
        >>> from obstore.store import MemoryStore
        >>> from blobmap.model import Blob, Manifest
        >>> manifests = ManifestStore(MemoryStore(), "blobmap")
        >>> manifests.key("cordex/a.zarr")
        'blobmap/cordex/a.zarr/manifest.json'
        >>> _ = manifests.write(Manifest("cordex/a.zarr",
        ...                              (Blob("b", ("x",)),), ()),
        ...                     expect_absent=True)
        >>> manifests.read("cordex/a.zarr").manifest.scope
        'cordex/a.zarr'
    """

    def __init__(self, store: Store, prefix: str = "") -> None:
        self.store = store
        self.prefix = prefix.strip("/")

    def key(self, scope: str) -> str:
        """Get the object key a scope's manifest lives at.

        Args:
            scope: Data prefix, such as `cordex/nukleus/eur11.zarr`.

        Returns:
            The mirrored key in the manifest bucket.
        """
        parts = [p for p in (self.prefix, scope.strip("/"), MANIFEST_NAME) if p]
        return "/".join(parts)

    # -- read -------------------------------------------------------------

    def read(self, scope: str) -> Stored | None:
        """Read one manifest.

        Args:
            scope: Data prefix whose manifest to fetch.

        Returns:
            A [`Stored`][blobmap.manifests.Stored], or `None` if this scope
            has never been partitioned.

        Raises:
            ValueError: If the object exists but is not a manifest this
                version can read.
        """
        key = self.key(scope)
        raw = get_bytes(self.store, key)
        if raw is None:
            return None
        entry = head(self.store, key)
        return Stored(Manifest.loads(raw), entry.etag if entry else None)

    def scopes(self) -> Iterator[str]:
        """Every scope that has a manifest.

        Yields:
            Scope prefixes, derived from the manifest keys. One LIST, no
            GETs, so this is cheap enough to call on every scan.
        """
        base = f"{self.prefix}/" if self.prefix else ""
        for entry in list_all(self.store, base):
            if not entry.key.endswith(MANIFEST_NAME):
                continue
            i, k = len(base), -len(MANIFEST_NAME)
            yield entry.key[i:k].strip("/")

    def load_all(self, workers: int = 16) -> list[Manifest]:
        """Load every manifest, for building a [`Trie`][blobmap.resolve.Trie].

        This is everything a resolving service needs at startup, with no
        database on the path. Manifests are per scope rather than per blob,
        so a petabyte is hundreds to a few thousand small JSON objects: one
        LIST plus parallel GETs.

        Args:
            workers: Thread pool size for the GETs.

        Returns:
            Every valid manifest. A manifest whose declared scope does not
            match its location is skipped with a warning, since it would
            otherwise claim keys it has no business claiming.
        """
        scopes = list(self.scopes())
        if not scopes:
            return []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(self.read, scopes))
        out: list[Manifest] = []
        for scope, stored in zip(scopes, results):
            if stored is None:
                continue
            if stored.manifest.scope.strip("/") != scope:
                log.warning(
                    "manifest at %s declares scope %r -- ignoring",
                    scope,
                    stored.manifest.scope,
                )
                continue
            out.append(stored.manifest)
        return out

    # -- write ------------------------------------------------------------

    def write(
        self,
        manifest: Manifest,
        *,
        etag: str | None = None,
        expect_absent: bool = False,
    ) -> str | None:
        """Validate and write a manifest.

        Validation happens before every write, because a malformed manifest
        sitting in object storage is far more expensive than a failed
        partition run.

        Args:
            manifest: The manifest to store. Its scope determines the key.
            etag: Etag from a prior read, requiring the object to be
                unchanged.
            expect_absent: Require that no manifest exists yet.

        Returns:
            The new etag, or `None` if the backend does not report one.

        Raises:
            ValueError: If the manifest fails validation. Nothing is written.
            Conflict: If a conditional write fails its precondition.
        """
        manifest.validate()
        return put_bytes(
            self.store,
            self.key(manifest.scope),
            manifest.dumps().encode(),
            etag=etag,
            expect_absent=expect_absent,
        )


__all__ = ["ManifestStore", "Stored", "Conflict", "MANIFEST_NAME"]
