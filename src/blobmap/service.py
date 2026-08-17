"""The one operation both discovery drivers call.

Ties together [`read_arrays`][blobmap.hierarchy.read_arrays],
[`partition`][blobmap.partition.partition] and
[`ManifestStore`][blobmap.manifests.ManifestStore] into something a scan or an
event handler can call with a single scope string.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .hierarchy import read_arrays
from .manifests import ManifestStore
from .model import Manifest, Policy
from .partition import Diff, diff, partition
from .storage import Conflict, Store

log = logging.getLogger(__name__)


class NotAdditive(RuntimeError):
    """A repartition would move or drop existing blob definitions.

    Unreachable while pinning is unconditional, since
    [`partition`][blobmap.partition.partition] carries previous blobs over
    verbatim. Kept as an assertion so a future change to the cut cannot
    quietly start invalidating tape addresses.
    """


@dataclass(frozen=True)
class Result:
    """Outcome of one partition run.

    Attributes:
        scope: The scope that was partitioned.
        manifest: The manifest now in effect. On a no-op this is the existing
            one, unchanged.
        diff: What changed relative to the previous manifest.
        written: Whether anything was actually stored. False for a no-op, a
            dry run, or when a concurrent writer won the race.
    """

    scope: str
    manifest: Manifest
    diff: Diff
    written: bool


def partition_store(
    data: Store,
    manifests: ManifestStore,
    scope: str,
    *,
    policy: Policy | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> Result:
    """Partition or repartition one scope.

    Repartitioning is additive *by construction*: pinned blobs are carried
    over verbatim and new cuts only fill unclaimed regions, so ids -- and the
    tape addresses blobtier holds against them -- survive. Note this means a
    policy change alone has no effect on an existing scope; cuts are frozen
    once made, which is the whole point.

    `force` drops the pinning and recomputes from scratch. That is the only
    path that can move or drop a blob, so it is the only one that can
    invalidate a tape copy, and it says so loudly.

    Args:
        data: A storage handle for the data being partitioned.
        manifests: Where manifests are read and written.
        scope: Prefix to partition, such as `cordex/nukleus/eur11.zarr`.
        policy: Thresholds. Defaults to the previous manifest's policy when
            repartitioning. Note that on the pinned path a changed policy
            affects only newly cut regions.
        force: Recompute from scratch, ignoring existing blobs. Can orphan
            tape copies, and logs what it moved.
        dry_run: Compute and return, writing nothing.

    Returns:
        A [`Result`][blobmap.service.Result].

    Raises:
        NotAZarrStore: If no zarr metadata is found under the scope.
        NotAdditive: If the cut would move existing blobs without `force`.
            Not reachable in normal operation.

    Example:
        ```python
        from obstore.store import S3Store
        from blobmap import ManifestStore, partition_store

        data = S3Store(bucket="cordex")
        manifests = ManifestStore(S3Store(bucket="waterpark-blobmap"))

        result = partition_store(data, manifests, "nukleus/eur11.zarr",
                                 dry_run=True)
        print(result.diff.describe())
        ```
    """
    stored = manifests.read(scope)
    previous = stored.manifest if stored else None

    arrays = read_arrays(data, scope)
    manifest = partition(
        scope, arrays, policy=policy, previous=None if force else previous
    )
    changes = diff(previous, manifest)

    if previous is not None and changes.is_empty:
        return Result(scope, previous, changes, written=False)

    if not changes.is_additive:
        if not force:
            # unreachable while pinning is unconditional; kept as an assertion
            # so a future change to _cut cannot quietly start moving blobs
            raise NotAdditive(
                f"{scope}: repartition would modify existing blobs, "
                f"invalidating tape addresses.\n{changes.describe()}"
            )
        log.warning(
            "%s: forced repartition moves %d and drops %d blobs; "
            "any tape copies held against those ids are now orphaned"
            "\n%s",
            scope,
            len(changes.modified),
            len(changes.removed),
            changes.describe(),
        )

    if previous is not None:
        manifest = manifest.bumped()

    if dry_run:
        return Result(scope, manifest, changes, written=False)

    try:
        manifests.write(
            manifest, etag=stored.etag if stored else None, expect_absent=stored is None
        )
    except Conflict:
        log.warning("%s: manifest changed underneath us, skipping", scope)
        return Result(scope, manifest, changes, written=False)
    return Result(scope, manifest, changes, written=True)
