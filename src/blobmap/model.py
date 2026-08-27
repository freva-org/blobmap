"""Manifest format and the array metadata that feeds it.

The manifest is *pure definition*: it changes only when the set of blobs
changes. Anything measured, such as tier, last access or current size, lives
in blobtier's state table keyed by blob id.

Stdlib only. Nothing in this module knows that storage exists, which is what
makes the partitioner testable without a backend.

Attributes:
    SCHEMA_VERSION: Manifest format version. Checked strictly on read; a
        mismatch raises rather than attempting a best-effort parse, because
        silently misreading a blob definition would misroute chunks.
    VERSION: Package version, recorded in each manifest as provenance.
    GiB: Convenience constant, 1024**3.
    KEY_ENCODINGS: Supported chunk key layouts, mapped to an example key.
        See [`Array.key_encoding`][blobmap.model.Array].
    METADATA_BASENAMES: Object basenames that are never archivable. This is
        what keeps `xr.open_zarr` from triggering a tape mount.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

from ._version import __version__
from .schema import validate_document

SCHEMA_VERSION = 3
VERSION = __version__

GiB = 1024**3

# How to locate the chunk index in an object key. Per *array*, not per store:
# in v2 `dimension_separator` lives in each .zarray, and v3 arrays may opt in
# to v2-style keys, so one store can legitimately mix all three.
KEY_ENCODINGS: dict[str, str] = {
    "v3_slash": "tas/c/137/0/0",
    "v2_slash": "tas/137/0/0",
    "v2_flat": "tas/137.0.0",
}

METADATA_BASENAMES: tuple[str, ...] = (
    "zarr.json",
    ".zarray",
    ".zgroup",
    ".zattrs",
    ".zmetadata",
)


def now() -> str:
    """Get current UTC time as an ISO 8601 string, to second resolution.

    Returns:
        Timestamp such as `2026-08-03T09:12:00+00:00`.

    Example:
        >>> stamp = now()
        >>> stamp.endswith("+00:00")
        True
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# input side
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Array:
    """One zarr array: metadata for structure, LIST for size.

    This is the partitioner's input. It is deliberately a plain dataclass with
    no storage behind it, so the partitioning rules can be tested by writing
    these out by hand.

    Attributes:
        path: Array path relative to the manifest scope, such as `tas` or
            `deep/nest/tas`. Empty string for an array at the scope root.
        shape: Full array shape, from the zarr metadata.
        chunks: The *inner* chunk shape. This is not the storage unit when
            the array is sharded; see
            [`object_shape`][blobmap.model.Array.object_shape].
        itemsize: Bytes per element, derived from the dtype.
        shards: Shard shape when sharding is in use, else `None`. When set,
            this is what becomes one object.
        stored_bytes: Observed compressed total, summed from a LIST. `None`
            falls back to the uncompressed estimate, which is correct but
            pessimistic and makes the width clamp bind hard.
        nobjects_seen: Objects actually present. Differs from
            [`nobjects`][blobmap.model.Array.nobjects] for a sparsely or
            partially written array.
        is_coordinate: True for a dimension coordinate. Coordinates are
            pinned hot regardless of size, or opening the store hits tape.
        key_encoding: How to locate the chunk index in an object key. One of
            `v3_slash`, `v2_slash`, `v2_flat`. Per array rather than per
            store: in v2 `dimension_separator` lives in each `.zarray`, and a
            v3 array may opt in to v2 style keys, so one store can mix all
            three.

    Example:
        >>> tas = Array("tas", (400, 4, 4), (10, 4, 4), 4, stored_bytes=2560)
        >>> tas.nobjects
        40
        >>> tas.chunk_prefix
        'tas/c'
    """

    path: str  # relative to scope, e.g. "tas"
    shape: tuple[int, ...]
    chunks: tuple[int, ...]  # inner chunk; not the object if sharded
    itemsize: int
    shards: tuple[int, ...] | None = None
    stored_bytes: int | None = None
    nobjects_seen: int | None = None  # objects actually present
    is_coordinate: bool = False
    key_encoding: str = "v3_slash"

    @property
    def object_shape(self) -> tuple[int, ...]:
        """Shape of one stored object: the shard if sharded, else the chunk.

        zarr-python inverts the intuitive naming. `Array.chunks` is the
        *inner* chunk of a sharded array and `Array.shards` is what becomes an
        object. Using the inner chunk here understates object size by the
        shard factor, which silently produces blobs many times too large. Raw
        v3 metadata does not invert: `chunk_grid.chunk_shape` is already the
        object, and the inner shape sits in the sharding codec config.

        Returns:
            The shape of a single stored object.

        Example:
            >>> Array("tas", (400, 4), (10, 4), 4).object_shape
            (10, 4)
            >>> Array("tas", (400, 4), (10, 4), 4, shards=(50, 4)).object_shape
            (50, 4)
        """
        return self.shards or self.chunks

    @property
    def object_grid(self) -> tuple[int, ...]:
        """Number of objects along each dimension.

        Returns:
            Object counts per dimension, using
            [`object_shape`][blobmap.model.Array.object_shape] as the unit.

        Example:
            >>> Array("tas", (400, 4, 4), (10, 4, 4), 4).object_grid
            (40, 1, 1)
        """
        return tuple(math.ceil(s / c) for s, c in zip(self.shape, self.object_shape))

    @property
    def nobjects(self) -> int:
        """Objects this array would have if fully written.

        Deliberately not `nobjects_seen`: a sparsely written array should be
        partitioned for the shape it will have, not the shape it has today.

        Returns:
            Total object count across the full grid.

        Example:
            >>> Array("tas", (400, 4, 4), (10, 4, 4), 4, nobjects_seen=3).nobjects
            40
        """
        return math.prod(self.object_grid)

    @property
    def uncompressed_object_bytes(self) -> int:
        """Size of one object with no compression at all.

        Fixed for the lifetime of the array, since changing it means
        rewriting every chunk. That is what makes it usable as a hard ceiling
        in [`bucket_width`][blobmap.partition.bucket_width]: whatever the
        codec does later, an object cannot exceed this.

        Returns:
            Uncompressed bytes per stored object.

        Example:
            >>> Array("tas", (400, 4, 4), (10, 4, 4), 4).uncompressed_object_bytes
            640
        """
        return math.prod(self.object_shape) * self.itemsize

    @property
    def total_bytes(self) -> int:
        """Total size of the array, observed if known and estimated otherwise.

        Returns:
            `stored_bytes` when set, else the uncompressed estimate.

        Example:
            >>> Array("tas", (400, 4, 4), (10, 4, 4), 4).total_bytes
            25600
            >>> Array("tas", (400, 4, 4), (10, 4, 4), 4, stored_bytes=99).total_bytes
            99
        """
        if self.stored_bytes is not None:
            return self.stored_bytes
        return self.nobjects * self.uncompressed_object_bytes

    @property
    def avg_object_bytes(self) -> int:
        """Mean size of one stored object.

        Divides by the objects actually seen rather than the full grid, so a
        partially written array is not reported as compressing better than it
        does.

        Returns:
            Average bytes per object, at least 1.

        Example:
            >>> Array("tas", (400, 4, 4), (10, 4, 4), 4,
            ...       stored_bytes=2560).avg_object_bytes
            64
        """
        seen = self.nobjects_seen or self.nobjects
        return max(1, self.total_bytes // max(1, seen))

    @property
    def chunk_prefix(self) -> str:
        """Where this array's chunk keys begin, relative to the scope.

        Returns:
            The prefix a blob should claim to cover this array's chunks.

        Raises:
            ValueError: If `key_encoding` is not one of
                `KEY_ENCODINGS`.

        Example:
            >>> Array("tas", (1,), (1,), 4, key_encoding="v3_slash").chunk_prefix
            'tas/c'
            >>> Array("tas", (1,), (1,), 4, key_encoding="v2_flat").chunk_prefix
            'tas'
        """
        if self.key_encoding == "v3_slash":
            return f"{self.path}/c" if self.path else "c"
        if self.key_encoding in ("v2_slash", "v2_flat"):
            return self.path
        raise ValueError(f"unknown key_encoding {self.key_encoding!r}")


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Policy:
    """Thresholds that shape the cut.

    Defaults are a starting point, not a recommendation. The right values
    depend on tape mount and positioning times, so check them against what
    the HSM actually does before trusting them.

    Attributes:
        t_max_bytes: Target upper bound for one blob. A subtree at or under
            this becomes a single blob; above it the partitioner descends and
            eventually buckets. Too large and a restore takes hours.
        t_min_bytes: Floor below which arrays are coalesced with their
            neighbours rather than each becoming a blob. Defaults to 0, which
            disables coalescing: it groups by path adjacency, which is a guess
            about access correlation, and the cost of guessing wrong is
            restoring data nobody asked for. Aggregating small objects is the
            tape layer's job. Raise it only if the HSM cannot bundle recalls.
        t_hot_bytes: Arrays smaller than this are never archived, regardless
            of the blobs. This is a "not worth a row" threshold, not a
            "small variable" one: keeping every sub-gigabyte array hot leaves
            an unbounded amount of data permanently on disk. Coordinates are
            pinned by detection, not by size, so this can be small.
        width_clamp: Bound on the worst case. A blob may not exceed
            `width_clamp * t_max_bytes` even if compression degrades to
            nothing, computed from uncompressed object size.
        pow2_floor: Widths at or above this are rounded down to a power of
            two, so a small drift in the observed compression ratio does not
            renumber every blob. Below it, rounding would throw away too much
            of the target, which is the common case for sharded arrays.

    Example:
        >>> Policy().t_max_bytes == 100 * GiB
        True
        >>> Policy().t_min_bytes            # no floor: the tape layer bundles
        0
    """

    t_max_bytes: int = 100 * GiB
    t_min_bytes: int = 0
    t_hot_bytes: int = 16 * 1024**2
    width_clamp: int = 8
    pow2_floor: int = 64

    def to_json(self) -> dict[str, int]:
        """Serialise for embedding in a manifest.

        Returns:
            A plain dict of the five thresholds.
        """
        return {
            "t_max_bytes": self.t_max_bytes,
            "t_min_bytes": self.t_min_bytes,
            "t_hot_bytes": self.t_hot_bytes,
            "width_clamp": self.width_clamp,
            "pow2_floor": self.pow2_floor,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Policy:
        """Rebuild from a manifest, ignoring fields this version does not know.

        Args:
            d: The `policy` block of a manifest.

        Returns:
            A `Policy`, with defaults for anything absent.

        Example:
            >>> Policy.from_json({"t_max_bytes": 5, "added_in_future": 9})
            Policy(t_max_bytes=5, t_min_bytes=0, t_hot_bytes=16777216, width_clamp=8, pow2_floor=64)
        """
        fields = set(cls.__dataclass_fields__)
        return cls(**{k: int(v) for k, v in d.items() if k in fields})


@dataclass(frozen=True)
class Pin:
    """A deliberate instruction to keep a prefix hot.

    Distinct from `hot_always`, which is *derived*: metadata objects and
    coordinates are recomputed on every partition because they follow from the
    store's structure. A pin follows from someone's intent, so it is preserved
    across repartitions and can only be removed by removing it.

    That also means a pin is the one part of a manifest that cannot be
    reconstructed by re-scanning. Blob definitions can be recomputed; intent
    cannot. Back the manifest bucket up.

    Attributes:
        prefix: What to keep hot, relative to the manifest scope. Empty
            string pins the whole scope.
        reason: Why. Required, because a pin with no stated reason is
            indistinguishable from one nobody remembers setting.
        by: Who set it.
        at: ISO 8601 UTC timestamp when it was set.
        until: ISO 8601 UTC timestamp after which it should be reviewed, or
            `None` for an open-ended pin. Expiry is *reported*, not enforced:
            silently unpinning a dataset someone is working on would be a
            worse surprise than a stale pin.

    Example:
        >>> pin = Pin("multiscales/zoom_9", "active ICON analysis", "wilfred",
        ...           "2026-08-19T09:12:00+00:00", "2026-12-01T00:00:00+00:00")
        >>> pin.covers("multiscales/zoom_9/tas/c/0")
        True
        >>> pin.covers("multiscales/zoom_8/tas/c/0")
        False
    """

    prefix: str
    reason: str
    by: str = ""
    at: str = ""
    until: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Serialise for the manifest.

        Returns:
            A dict with `prefix`, `reason`, `by`, `at` and `until`.
        """
        return {
            "prefix": self.prefix,
            "reason": self.reason,
            "by": self.by,
            "at": self.at,
            "until": self.until,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Pin:
        """Rebuild from a manifest entry.

        Args:
            d: One element of the manifest's `pinned` list.

        Returns:
            A `Pin`.
        """
        return cls(
            str(d["prefix"]),
            str(d["reason"]),
            str(d.get("by", "")),
            str(d.get("at", "")),
            None if d.get("until") is None else str(d["until"]),
        )

    def covers(self, key: str) -> bool:
        """Whether this pin applies to a key, relative to the scope.

        Args:
            key: Scope-relative object key.

        Returns:
            True if the key is at or below the pinned prefix.
        """
        if not self.prefix:
            return True
        return key == self.prefix or key.startswith(self.prefix + "/")

    def expired(self, now_iso: str | None = None) -> bool:
        """Whether the review date has passed.

        Args:
            now_iso: Time to compare against, defaulting to now. ISO 8601.

        Returns:
            False for an open-ended pin, which never expires but also never
            gets reviewed -- which is its own problem, so `pin show` calls
            those out separately.

        Example:
            >>> Pin("x", "why", until="2020-01-01T00:00:00+00:00").expired()
            True
            >>> Pin("x", "why").expired()
            False
        """
        if self.until is None:
            return False
        return (now_iso or now()) > self.until


@dataclass(frozen=True)
class Bucket:
    """The arithmetic that turns a chunk index into a blob number.

    This is what makes a manifest a rule rather than a table: chunk 5,000,000
    resolves without anyone having enumerated it, so appending to a store
    needs no manifest change at all.

    Attributes:
        index: Which segment after the blob prefix holds the chunk index.
            Almost always 0, since time is conventionally the first
            dimension and is what people slice on.
        width: How many objects along that dimension go into one blob.
            Chosen by [`bucket_width`][blobmap.partition.bucket_width].
        key_encoding: How to parse the index out of the key. One of
            `v3_slash`, `v2_slash`, `v2_flat`.

    Example:
        >>> Bucket(0, 2048, "v3_slash").to_json()["width"]
        2048
    """

    index: int
    width: int
    key_encoding: str

    def to_json(self) -> dict[str, Any]:
        """Serialise for the manifest.

        Returns:
            A dict with `index`, `width` and `key_encoding`.
        """
        return {
            "index": self.index,
            "width": self.width,
            "key_encoding": self.key_encoding,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Bucket:
        """Rebuild from a manifest.

        Args:
            d: The `bucket` block of a blob entry.

        Returns:
            A `Bucket`.
        """
        return cls(int(d["index"]), int(d["width"]), str(d["key_encoding"]))


@dataclass(frozen=True)
class Blob:
    """A set of objects that move to and from tape together.

    Every blob has exactly these three fields. `bucket is None` means one
    bucket, so the resolved id is *always* `f"{id}_{n}"`. Unbucketed is the
    degenerate case of bucketed rather than a different kind of thing, which
    gives the resolver a single code path and means filling in `bucket` later
    does not change existing blob ids.

    Attributes:
        id: Stable identifier, `[a-z0-9_]`. This is the join key against
            blobtier's state table and, through it, against tape addresses.
            Changing how ids are derived is a breaking change.
        prefixes: Key prefixes this blob claims, relative to the manifest
            scope. More than one when small arrays were coalesced. A bucketed
            blob must have exactly one.
        bucket: Arithmetic for splitting inside an array, or `None` for a
            blob that is a plain prefix.

    Example:
        >>> Blob("b_pr", ("pr/c",)).instance(0)
        'b_pr_0'
        >>> Blob("b_tas", ("tas/c",), Bucket(0, 2048, "v3_slash")).instance(2)
        'b_tas_2'
    """

    id: str
    prefixes: tuple[str, ...]
    bucket: Bucket | None = None

    def to_json(self) -> dict[str, Any]:
        """Serialise for the manifest.

        Returns:
            A dict with exactly `id`, `prefixes` and `bucket`.

        Example:
            >>> sorted(Blob("b_pr", ("pr/c",)).to_json())
            ['bucket', 'id', 'prefixes']
        """
        return {
            "id": self.id,
            "prefixes": list(self.prefixes),
            "bucket": self.bucket.to_json() if self.bucket else None,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Blob:
        """Rebuild from a manifest entry.

        Args:
            d: One element of the manifest's `blobs` list.

        Returns:
            A `Blob`.
        """
        b = d.get("bucket")
        return cls(
            str(d["id"]), tuple(d["prefixes"]), Bucket.from_json(b) if b else None
        )

    def instance(self, n: int) -> str:
        """Get the concrete blob id for bucket number `n`.

        Args:
            n: Bucket number, always 0 when `bucket` is `None`.

        Returns:
            The id blobtier stores tier state against.
        """
        return f"{self.id}_{n}"


@dataclass(frozen=True)
class Manifest:
    """The blob definitions for one scope.

    Lives as a single JSON object in a bucket you own, at a path mirroring
    the data. Never inside the store, because data arrives that must not be
    altered.

    A manifest is *pure definition*. It changes only when the set of blobs
    changes: a new variable, a new cut, a forced repartition. Not on reads,
    writes, tiering or restores.

    Attributes:
        scope: Prefix these definitions apply to, relative to the data store.
            Usually one zarr store, but may be a parent prefix covering many
            small stores that should restore together.
        blobs: The blob definitions. A list of *rules*, so its length tracks
            the number of cut decisions, not the number of blobs and
            certainly not the number of objects.
        hot_always: Glob patterns that are never archivable, whatever the
            blobs say. Metadata objects and dimension coordinates. Derived
            from the store's structure and recomputed on every partition, so
            this is not where deliberate decisions belong.
        pinned: Prefixes someone has deliberately kept hot. Preserved across
            repartitions, unlike `hot_always`, and the one part of a manifest
            that cannot be reconstructed by re-scanning.
        policy: Thresholds used to produce this cut, recorded so a later run
            can tell whether they changed.
        epoch: Bumped whenever the definitions change. Lets a resolver decide
            if it needs to reload without diffing.
        generated_at: ISO 8601 UTC timestamp of the run that produced this.
        generated_by: Package and version that produced this.
        provenance: Measured numbers per array, for debugging only. These can
            go stale while the manifest is untouched, so the resolver must
            never read them.

    Example:
        >>> m = Manifest("cordex/a.zarr", (Blob("b_pr", ("pr/c",)),),
        ...              ("**/zarr.json",))
        >>> m.epoch
        1
        >>> m.bumped().epoch
        2
    """

    scope: str
    blobs: tuple[Blob, ...]
    hot_always: tuple[str, ...]
    pinned: tuple[Pin, ...] = ()
    policy: Policy = field(default_factory=Policy)
    epoch: int = 1
    generated_at: str = ""
    generated_by: str = f"blobmap {VERSION}"
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate on construction.

        There is no such thing as an invalid `Manifest` in memory, so callers
        cannot forget to check one before writing it. The cost is that
        [`validate`][blobmap.model.Manifest.validate] runs on every
        construction, which is a handful of times per partition run and never
        on a lookup path.
        """
        self.validate()

    def to_json(self) -> dict[str, Any]:
        """Serialise, including the schema version.

        Returns:
            The full manifest as a JSON-compatible dict.
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "scope": self.scope,
            "epoch": self.epoch,
            "generated_at": self.generated_at,
            "generated_by": self.generated_by,
            "policy": self.policy.to_json(),
            "hot_always": list(self.hot_always),
            "pinned": [p.to_json() for p in self.pinned],
            "blobs": [b.to_json() for b in self.blobs],
            "provenance": self.provenance,
        }

    def dumps(self, indent: int | None = 2) -> str:
        """Serialise to JSON text.

        Args:
            indent: Passed to `json.dumps`. Keep it non-`None`: manifests are
                read by humans during incidents.

        Returns:
            JSON text ready to write to object storage.
        """
        return json.dumps(self.to_json(), indent=indent)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Manifest:
        """Rebuild from a parsed manifest.

        Args:
            d: A parsed manifest object.

        Returns:
            A `Manifest`.

        Raises:
            SchemaError: If the document does not match
                `SCHEMA`, including a version this
                package does not speak. Deliberately strict: silently
                misreading a blob definition would misroute chunks to the
                wrong tape unit.

        Example:
            >>> Manifest.from_json({"schema_version": 99, "scope": "s",
            ...                     "epoch": 1, "hot_always": [], "blobs": []})
            Traceback (most recent call last):
                ...
            blobmap.schema.SchemaError: schema_version: expected 3, got 99
        """
        validate_document(d)
        return cls(
            scope=str(d["scope"]),
            blobs=tuple(Blob.from_json(b) for b in d["blobs"]),
            hot_always=tuple(d["hot_always"]),
            pinned=tuple(Pin.from_json(x) for x in d.get("pinned", [])),
            policy=Policy.from_json(d.get("policy", {})),
            epoch=int(d.get("epoch", 1)),
            generated_at=str(d.get("generated_at", "")),
            generated_by=str(d.get("generated_by", "")),
            provenance=dict(d.get("provenance", {})),
        )

    @classmethod
    def loads(cls, text: str | bytes) -> Manifest:
        """Parse from JSON text.

        Args:
            text: JSON as read from object storage.

        Returns:
            A `Manifest`.

        Raises:
            SchemaError: If the document does not match the schema.
            json.JSONDecodeError: If the text is not JSON at all.

        Example:
            >>> m = Manifest("s", (Blob("b", ("x",)),), (),
            ...              (Pin("x", "under active analysis"),),
            ...              generated_at="2026-01-01T00:00:00+00:00")
            >>> Manifest.loads(m.dumps()) == m
            True
        """
        return cls.from_json(json.loads(text))

    def by_id(self) -> dict[str, Blob]:
        """Index the blobs by id.

        Returns:
            Mapping of blob id to `Blob`, useful for diffing two manifests.
        """
        return {b.id: b for b in self.blobs}

    def bumped(self, **changes: Any) -> Manifest:
        """Copy with the epoch incremented and the timestamp refreshed.

        Args:
            **changes: Any other fields to replace.

        Returns:
            A new `Manifest`. The original is unchanged, since manifests are
            frozen.
        """
        return replace(self, epoch=self.epoch + 1, generated_at=now(), **changes)

    def pin_for(self, key: str) -> Pin | None:
        """Find the pin covering a scope-relative key, if any.

        Args:
            key: Scope-relative object key.

        Returns:
            The first matching `Pin`, or `None`.
        """
        for pin in self.pinned:
            if pin.covers(key):
                return pin
        return None

    def pinned_blob_ids(self) -> set[str]:
        """Blob ids whose definition is covered by a pin.

        Pinning a prefix that already has a blob cannot remove that blob:
        repartitioning is additive, and dropping a blob id would orphan
        whatever tape copy is held against it. So the definition survives,
        nothing resolves to it any more, and its last-read timestamp goes
        stale -- at which point an age-based tiering policy would archive
        exactly the data someone asked to keep on disk.

        A consumer must therefore exclude these before archiving. Resolution
        alone is not enough, because the policy query runs over blob state
        rather than over keys.

        Returns:
            Ids of blobs covered by at least one pin. Note these are
            definition ids, not instance ids: a bucketed blob contributes
            `b_tas`, which covers every `b_tas_n`.

        Example:
            >>> m = Manifest("s", (Blob("b_tas", ("tas/c",)),), (),
            ...              (Pin("tas", "under analysis"),))
            >>> m.pinned_blob_ids()
            {'b_tas'}
        """
        return {
            b.id
            for b in self.blobs
            for prefix in b.prefixes
            if any(pin.covers(prefix) for pin in self.pinned)
        }

    def validate(self) -> None:
        """Check the invariants that the resolver relies on.

        Called automatically on construction. These are the cross-field rules
        the JSON Schema cannot express: uniqueness of ids, and the constraint
        that a bucketed blob has exactly one prefix.

        Raises:
            ValueError: On a duplicate or malformed blob id, a blob with no
                prefixes, an unknown key encoding, a bucketed blob with more
                than one prefix, or a non-positive width.

        Example:
            >>> Manifest("s", (Blob("b", ("x",)), Blob("b", ("y",))),
            ...          ()).validate()
            Traceback (most recent call last):
                ...
            ValueError: duplicate blob id 'b'
        """
        seen: set[str] = set()
        for b in self.blobs:
            if b.id in seen:
                raise ValueError(f"duplicate blob id {b.id!r}")
            if not b.id or not all(c.isalnum() or c == "_" for c in b.id):
                raise ValueError(f"blob id {b.id!r} is not [a-z0-9_]")
            if not b.prefixes:
                raise ValueError(f"{b.id}: no prefixes")
            seen.add(b.id)
            if b.bucket is None:
                continue
            if b.bucket.key_encoding not in KEY_ENCODINGS:
                raise ValueError(
                    f"{b.id}: unknown key_encoding {b.bucket.key_encoding!r}"
                )
            if len(b.prefixes) != 1:
                raise ValueError(
                    f"{b.id}: a bucketed blob needs exactly one "
                    f"prefix, got {len(b.prefixes)}"
                )
            if b.bucket.width < 1:
                raise ValueError(f"{b.id}: width must be >= 1")
            if b.bucket.index < 0:
                raise ValueError(f"{b.id}: index must be >= 0")

        seen_prefixes: set[str] = set()
        for pin in self.pinned:
            if not pin.reason.strip():
                raise ValueError(
                    f"pin on {pin.prefix!r} has no reason; a pin "
                    f"nobody can explain is one nobody removes"
                )
            if pin.prefix in seen_prefixes:
                raise ValueError(f"duplicate pin on {pin.prefix!r}")
            seen_prefixes.add(pin.prefix)


def default_hot_always() -> list[str]:
    """Patterns for metadata objects, which are never archivable.

    This is the rule that keeps `xr.open_zarr` from triggering a tape mount,
    and incidentally why re-reading a store to partition it cannot feed its
    own event loop: everything the partitioner touches is already ineligible.

    Returns:
        Glob patterns, one per known metadata basename.

    Example:
        >>> "**/zarr.json" in default_hot_always()
        True
    """
    return [f"**/{name}" for name in METADATA_BASENAMES]
