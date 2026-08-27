"""The decisions: arrays in, [`Manifest`][blobmap.model.Manifest] out.

Pure functions with no I/O. Everything that actually matters lives here, and
it is testable with hand-written [`Array`][blobmap.model.Array] objects in
milliseconds.

The cut proceeds top down:

1. Coordinates and arrays under `t_hot_bytes` are pinned hot.
2. If the whole scope fits in `t_max_bytes`, it becomes one blob. This is the
   common case, and the best outcome for tape: small stores are read whole,
   so one mount beats one per variable.
3. Arrays over `t_max_bytes` are bucketed, which is the only way to cut
   *inside* an array.
4. What is left is coalesced with its neighbours until each group clears
   `t_min_bytes`, so a wide tree of small variables does not cost a tape
   mount per gigabyte.

Example:
    >>> from blobmap.model import Array, GiB, Policy
    >>> arrays = [Array("time", (400,), (400,), 8, is_coordinate=True),
    ...           Array("tas", (400, 4, 4), (10, 4, 4), 4,
    ...                 stored_bytes=400 * GiB)]
    >>> manifest = partition("cordex/a.zarr", arrays)
    >>> sorted(b.id for b in manifest.blobs)
    ['b_tas']
    >>> "time/**" in manifest.hot_always
    True
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .model import (
    Array,
    Blob,
    Bucket,
    GiB,
    Manifest,
    Pin,
    Policy,
    default_hot_always,
    now,
)

log = logging.getLogger(__name__)


def bucket_width(a: Array, policy: Policy) -> int:
    """Choose how many objects along dimension 0 go into one blob.

    Two bounds apply:

    * *ideal* hits `t_max_bytes` at the compression observed today
    * *hard* keeps the blob under `width_clamp * t_max_bytes` even if
      compression degrades to nothing, computed from uncompressed object
      size, which cannot change without rewriting the array

    The result is rounded down to a power of two so a ratio drifting from 2.0
    to 2.3 does not move the width and renumber every blob. Rounding is
    skipped below `pow2_floor`, where it would throw away too much of the
    target. That is the common case for sharded arrays, which have far fewer
    and larger objects.

    Args:
        a: The array to size. Its
            [`object_shape`][blobmap.model.Array.object_shape] is the unit,
            so a sharded array is measured in shards.
        policy: Thresholds to apply.

    Returns:
        Objects per blob, at least 1.

    Note:
        Logs a warning when a single object already exceeds `t_max_bytes`,
        since there is then nothing left to cut.

    Example:
        >>> from blobmap.model import Array, GiB, Policy
        >>> plain = Array("tas", (1_314_000, 412, 424), (128, 412, 424), 4,
        ...               stored_bytes=459 * GiB)
        >>> bucket_width(plain, Policy())
        2048

        The same bytes in a 10x sharded array is a tenth of the objects, so
        the width falls with it. Reading `chunks` instead of `shards` here
        would give both arrays the same width, and blobs 10x the target:

        >>> sharded = Array("tas", (1_314_000, 412, 424), (128, 412, 424), 4,
        ...                 shards=(1280, 412, 424), stored_bytes=459 * GiB)
        >>> bucket_width(sharded, Policy())
        128
    """
    ideal = max(1, policy.t_max_bytes // a.avg_object_bytes)
    hard = max(1, (policy.width_clamp * policy.t_max_bytes)
               // a.uncompressed_object_bytes)
    width = max(1, min(ideal, hard))
    if width >= policy.pow2_floor:
        width = 1 << (width.bit_length() - 1)
    if width == 1 and a.avg_object_bytes > policy.t_max_bytes:
        log.warning("%s: a single object is %.1f GiB, above t_max -- cannot "
                    "cut finer than one object", a.path,
                    a.avg_object_bytes / GiB)
    return width


def partition(
    scope: str,
    arrays: list[Array],
    *,
    policy: Policy | None = None,
    previous: Manifest | None = None,
) -> Manifest:
    """Compute the blob definitions for one scope.

    When `previous` is given its blobs are *pinned*: carried over byte
    identical, with new cuts only in regions no existing blob claims. That is
    what keeps blob ids, and the tape addresses blobtier holds against them,
    valid across a repartition. Growth along a bucketed dimension needs no
    repartition at all, since the id is arithmetic.

    Because pinning is unconditional, a policy change alone has no effect on
    an existing scope. Cuts are frozen once made. Pass `previous=None` to
    recompute from scratch, which is the only way to move a blob and the only
    way to orphan a tape copy.

    `hot_always` is always recomputed, since it follows from the store's
    structure: hot data is never archived, so there is no state to invalidate,
    and an array that grew past `t_hot_bytes` should stop being held back.
    `pinned` is carried over untouched, because it follows from intent rather
    than structure and can only be changed deliberately.

    Args:
        scope: Prefix these definitions apply to, relative to the data store.
        arrays: Every array under the scope, from
            [`read_arrays`][blobmap.hierarchy.read_arrays] or written by hand.
        policy: Thresholds. Defaults to `previous.policy` when repartitioning,
            else to [`Policy`][blobmap.model.Policy] defaults.
        previous: The manifest currently in effect, whose blobs are pinned.

    Returns:
        A validated `Manifest`. The epoch is carried over unchanged; callers
        that intend to write should use
        [`bumped`][blobmap.model.Manifest.bumped].

    Raises:
        ValueError: If the result fails validation, which would indicate a
            bug in the cut rather than bad input.

    Example:
        >>> from blobmap.model import Array, GiB
        >>> first = partition("s", [Array("tas", (400, 4, 4), (10, 4, 4), 4,
        ...                                stored_bytes=400 * GiB)])
        >>> grown = partition("s", [Array("tas", (800, 4, 4), (10, 4, 4), 4,
        ...                                stored_bytes=800 * GiB)],
        ...                   previous=first)
        >>> diff(first, grown).is_empty          # an append changes nothing
        True
    """
    policy = policy or (previous.policy if previous else Policy())

    # Pins are intent, so they survive a repartition. hot_always is derived
    # from structure, so it is recomputed.
    pinned: tuple[Pin, ...] = previous.pinned if previous else ()

    hot: list[str] = default_hot_always()
    payload: list[Array] = []
    for a in arrays:
        if a.is_coordinate or a.total_bytes < policy.t_hot_bytes:
            hot.append(f"{a.path}/**" if a.path else "**")
        elif any(pin.covers(a.path) for pin in pinned):
            # deliberately held hot; giving it a blob would let the tiering
            # policy archive it the moment the pin is forgotten
            hot.append(f"{a.path}/**" if a.path else "**")
        else:
            payload.append(a)

    carried: tuple[Blob, ...] = previous.blobs if previous else ()
    unclaimed = [a for a in payload if not _claimed(a, carried)]
    taken = {b.id for b in carried}
    blobs = list(carried) + _cut(unclaimed, policy, taken, fresh=not carried)

    manifest = Manifest(
        scope=scope,
        blobs=tuple(blobs),
        hot_always=tuple(hot),
        pinned=pinned,
        policy=policy,
        epoch=previous.epoch if previous else 1,
        generated_at=now(),
        provenance={
            a.path or ".": {
                "objects": a.nobjects,
                "objects_seen": a.nobjects_seen,
                "stored_bytes": a.total_bytes,
                "uncompressed_object_bytes": a.uncompressed_object_bytes,
                "sharded": a.shards is not None,
                "key_encoding": a.key_encoding,
            }
            for a in arrays
        },
    )
    manifest.validate()
    return manifest


def _claimed(a: Array, pinned: tuple[Blob, ...]) -> bool:
    """Whether an existing blob already covers this array's chunk keys.

    Args:
        a: Candidate array.
        pinned: Blobs carried over from the previous manifest.

    Returns:
        True if some pinned blob's prefix is the array's chunk prefix or an
        ancestor of it.
    """
    key = a.chunk_prefix
    for blob in pinned:
        for prefix in blob.prefixes:
            if prefix == "" or key == prefix or key.startswith(prefix + "/"):
                return True
    return False


def _cut(arrays: list[Array], policy: Policy, taken: set[str],
         fresh: bool) -> list[Blob]:
    if not arrays:
        return []

    total = sum(a.total_bytes for a in arrays)
    if fresh and total <= policy.t_max_bytes:
        # base case, and the common one: the whole scope is one blob. Best
        # outcome for tape too -- small stores are read whole, so one mount
        # beats one per variable.
        return [Blob(_uniq("b_root", taken), ("",))]

    out: list[Blob] = []
    small: list[Array] = []

    for a in arrays:
        if a.total_bytes > policy.t_max_bytes:
            # we must cut *inside* the array, so it has to be bucketed --
            # and only now does the key encoding matter
            out.append(Blob(
                id=_uniq(f"b_{_ident(a.path)}", taken),
                prefixes=(a.chunk_prefix,),
                bucket=Bucket(index=0, width=bucket_width(a, policy),
                              key_encoding=a.key_encoding),
            ))
        else:
            small.append(a)

    out.extend(_coalesce(small, policy, taken))
    return out


def _coalesce(arrays: list[Array], policy: Policy, taken: set[str]) -> list[Blob]:
    """Group adjacent small arrays so we do not spend a tape mount per GB.

    Adjacency in the listing is a crude proxy for access correlation. Real
    access logs, where available, are better evidence than path structure.

    Args:
        arrays: Arrays that individually fit inside `t_max_bytes`.
        policy: Thresholds; only `t_min_bytes` is used.
        taken: Blob ids already in use. Mutated as ids are allocated.

    Returns:
        One blob per group. A trailing group too small to stand alone is
        folded into the previous blob.
    """
    out: list[Blob] = []
    group: list[Array] = []
    size = 0

    for a in arrays:
        group.append(a)
        size += a.total_bytes
        if size >= policy.t_min_bytes:
            out.append(_group_blob(group, taken))
            group, size = [], 0

    if group:
        if out:
            # tail too small to stand alone: fold it into the last blob
            last = out[-1]
            taken.discard(last.id)
            prefixes = last.prefixes + tuple(a.chunk_prefix for a in group)
            out[-1] = Blob(_uniq(_group_id(prefixes), taken), prefixes)
        else:
            out.append(_group_blob(group, taken))
    return out


def _group_blob(group: list[Array], taken: set[str]) -> Blob:
    prefixes = tuple(a.chunk_prefix for a in group)
    return Blob(_uniq(_group_id(prefixes), taken), prefixes)


def _group_id(prefixes: tuple[str, ...]) -> str:
    """Derive an id for a blob covering one or more prefixes.

    Uses the *full* prefix, not its first segment. Truncating was safe while a
    scope was always a single store, where prefixes look like `tas/c`. For a
    deeper scope covering a datatree it collapsed every blob in the tree onto
    the same name, which `_uniq` then disambiguated by iteration order -- so a
    repartition after the listing order shifted could reassign an id to
    different data while tier state still pointed at the old meaning.

    Args:
        prefixes: The prefixes this blob claims, relative to the scope.

    Returns:
        A deterministic identifier, derived only from the prefixes themselves.

    Example:
        >>> _group_id(("pr/c", "hurs/c"))
        'b_pr_c_hurs_c'
        >>> _group_id(("multiscales/zoom_9/tas/c",))
        'b_multiscales_zoom_9_tas_c'
    """
    return "b_" + "_".join(_ident(p) for p in prefixes)


def _ident(path: str) -> str:
    """Sanitise a path into a blob identifier.

    Blob ids are the join key against blobtier's state table, so a typo is
    either permanent or costs a repartition. Keep them boring: lowercase,
    `[a-z0-9_]`, derived from the path.

    Args:
        path: Array or scope path.

    Returns:
        A safe identifier, `root` if nothing survives sanitising.
    """
    out = "".join(c if c.isalnum() else "_" for c in path.strip("/"))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_").lower() or "root"


def _uniq(ident: str, taken: set[str]) -> str:
    """Make an identifier unique by suffixing.

    Different paths can still sanitise to the same identifier -- `a-b` and
    `a_b` both become `a_b` -- and a collision would silently merge two blobs'
    tier state.

    Note the suffix depends on allocation order, so this is a last resort
    rather than a naming scheme: ids should be distinct from the prefixes
    alone. `_group_id` uses the full prefix precisely so that ordinary
    partitioning never reaches here.

    Args:
        ident: Preferred identifier.
        taken: Ids already allocated. Mutated to include the result.

    Returns:
        `ident`, or `ident_2`, `ident_3` and so on.
    """
    candidate, n = ident, 1
    while candidate in taken:
        n += 1
        candidate = f"{ident}_{n}"
    if candidate != ident:
        log.warning(
            "blob id %r collides; using %r. This suffix depends on the order "
            "arrays were listed, so it is not stable across repartitions -- "
            "check for array paths that differ only in punctuation",
            ident, candidate)
    taken.add(candidate)
    return candidate


# --------------------------------------------------------------------------
# diffing -- a repartition should be additive, and loud when it is not
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Diff:
    """What a repartition changed.

    Attributes:
        added: Blobs that did not exist before. Always safe.
        removed: Blobs that disappeared. Any tape copy held against these ids
            is now orphaned.
        modified: Pairs of (old, new) for blobs whose definition changed. The
            id still resolves but now covers different objects, so the tape
            copy no longer matches.

    Example:
        >>> from blobmap.model import Blob, Manifest
        >>> old = Manifest("s", (Blob("a", ("a",)),), ())
        >>> new = Manifest("s", (Blob("a", ("a",)), Blob("b", ("b",))), ())
        >>> diff(old, new).is_additive
        True
    """

    added: tuple[Blob, ...] = ()
    removed: tuple[Blob, ...] = ()
    modified: tuple[tuple[Blob, Blob], ...] = ()

    @property
    def is_additive(self) -> bool:
        """Whether no existing blob definition changed or disappeared.

        Anything else invalidates ids that blobtier holds tape addresses for.

        Returns:
            True when only additions were made.
        """
        return not self.removed and not self.modified

    @property
    def is_empty(self) -> bool:
        """Whether nothing changed at all.

        Returns:
            True when there is nothing to write.
        """
        return not (self.added or self.removed or self.modified)

    def describe(self) -> str:
        """Render for a log line or `--dry-run` output.

        Returns:
            One line per change, prefixed `+`, `-` or `~`, or `(no change)`.
        """
        lines = [f"+ {b.id} {list(b.prefixes)}" for b in self.added]
        lines += [f"- {b.id} {list(b.prefixes)}" for b in self.removed]
        lines += [f"~ {o.id}: {o.to_json()} -> {n.to_json()}"
                  for o, n in self.modified]
        return "\n".join(lines) or "(no change)"


def diff(old: Manifest | None, new: Manifest) -> Diff:
    """Compare two manifests by blob id.

    Args:
        old: The manifest previously in effect, or `None` for a first run.
        new: The freshly computed manifest.

    Returns:
        A `Diff`. When `old` is `None` everything counts as added.
    """
    if old is None:
        return Diff(added=new.blobs)
    a, b = old.by_id(), new.by_id()
    return Diff(
        added=tuple(b[k] for k in b if k not in a),
        removed=tuple(a[k] for k in a if k not in b),
        modified=tuple((a[k], b[k]) for k in a if k in b and a[k] != b[k]),
    )
