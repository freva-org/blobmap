# Design notes

Why the pieces are shaped the way they are. Roughly in the order the
decisions were made.

## Granularity

Per-chunk tracking is impossible: one store is 10^5-10^6 objects and the
state table would be larger than the data catalogue. Per-store is useless:
a 50 TB store is one unit, so tiering it is all-or-nothing.

A **blob** is the middle: a set of objects that moves as a unit, sized so a
restore is worth a tape mount. Roughly 10 GB (below which mount plus
positioning dominates) to 100 GB (above which a restore takes too long).

## The manifest is a rule, not a table

A blob definition is a path prefix plus optional arithmetic:

    tas/c/<n>/...  ->  b_tas_{n // 2048}

This resolves chunk 5,000,000 with nobody having computed it. Appending to a
store therefore needs no manifest change, no epoch bump and no resolver
reload -- the ids extend themselves. That property is why partitioning is a
once-per-store operation rather than a recurring crawl.

Corollary: **never emit a plain prefix for a chunked array that might grow.**
A plain prefix is a size assertion that can go stale; bucketing is not.
Plain prefixes are only used where the whole subtree is under `t_max` today,
where the blast radius of being wrong is one re-archive.

## Where manifests live

In a bucket you own, at a path mirroring the data. Not inside the store,
because data lands here that we must not alter, and a "write attrs if we can,
sidecar otherwise" fallback means two code paths, two idempotency stories and
a silent third state when the write half-succeeds.

Inline `_blob_root` attributes remain meaningful as *user declaration* --
input to the partitioner for owners who want control. The manifest is always
the resolved output and the single authority for lookup.

## Hot always

Metadata objects and dimension coordinates are never archivable, regardless of
blobs. Without this, `xr.open_zarr` on a cold store triggers a tape mount and
the catalogue becomes unusable.

Two things fall out of it:

* Partitioning cannot feed its own event loop. Everything the partitioner
  reads is already ineligible, so its own writes and reads produce no
  actionable events.
* Cold data stays browsable. This is a property zarr gives us that
  netCDF/HDF5/GRIB do not, since their headers live inside the file.

## Freezing the cut

Once made, a cut does not move. Pinning is unconditional in `partition()`:
existing blobs are carried over verbatim and new cuts only fill unclaimed
regions. So a policy change has no effect on an existing scope, and
`diff().is_additive` is structurally always true on the non-forced path.

Only `--force` recomputes from scratch, and it is the only operation that can
orphan a tape copy. It logs what it moved.

## Sharding

Where sharding is in use the shard is the object, so it is the unit for
sizing, the width clamp and the smallest possible cut. zarr-python inverts the
naming -- `Array.chunks` is the *inner* chunk -- so reading the wrong field
produces blobs too large by the shard factor, silently. Raw v3 metadata does
not invert: `chunk_grid.chunk_shape` is already the object.

Never introspect a shard index: it lives inside the object, so reading it on
cold data triggers exactly the restore we are trying to avoid.

## Compression drift

Width is chosen from observed compression but clamped against *uncompressed*
object size, which cannot change without rewriting the array. A 500x
compressing array therefore does not get a width that becomes 25 TB if the
codec later changes.

Width is rounded down to a power of two above `pow2_floor`, so a ratio
drifting 2.0 -> 2.3 does not renumber every blob. Below the floor, rounding
would throw away too much of the target -- which is the common case for
sharded arrays, since they have fewer, larger objects.

## Discovery

Two drivers, one entry point.

**Scan** is LIST plus "does a manifest exist". Descent stops at a store
boundary: a datatree may put a whole bucket in one store.

**Events** poll MinIO's Postgres notification target. A broker was not
necessary: latency does not matter here (the only latency-sensitive path,
restore, is synchronous in blobtier), a cursor gives replay for free, and two
consumers can hold independent positions in the same table with no
coordination.

Two filters carry the load. Only `ObjectCreated` on a *metadata* object is
interesting, which is what stops a conversion run's 300k chunk writes from
becoming 300k queue entries. And debounce: never partition a store still
being written, or the cut lands in the wrong place and the trailing bucket is
not sealed.

`store_of()` gives the node a metadata object sits in; `find_store_root()`
climbs to the store. Both are needed -- debouncing on the node fragments the
pending set per variable and hands the partitioner a sub-array as a scope.

## Principal filtering

The tiering job reads every object in a blob to move it to tape, and the
restore path reads them coming back. Without filtering by principal, archiving
a blob marks it freshly accessed and it immediately looks hot again. Same for
verification and backup sweeps. Only user-facing principals count.

## What is deliberately not here

**Tier state.** Which blobs are cold, when they were last read, restore
status. That is blobtier's table, keyed by blob id. The manifest changes only
when the set of blobs changes; state changes constantly.

**A database.** `ManifestStore.load_all()` is a LIST plus parallel GETs over a
few thousand small JSON objects. Blobtier needs no index at startup. If a
query interface is ever wanted from outside blobtier, blobtier already parses
every manifest and can upsert definitions as derived data.

**Restore.** MinIO events are post-hoc: by the time you see the GET, it has
already failed or stalled. Restore has to live in the request path, and the
trie is called synchronously from it.

## Known gaps

* The trailing bucket of a growing array must never be archived -- appending
  read-modify-writes it, and with sharding it rewrites a whole shard. That
  rule belongs in blobtier, which knows what is currently on tape; blobmap
  does not model it.
* A write into a cold blob leaves a stale tape copy. Blocking it at the
  gateway is cleaner than repairing after the fact.
* Retirement: when a rolling window empties a bucket permanently, the
  arithmetic still generates its id. That needs an explicit state, and it is
  the one place where tiering reality feeds back into the manifest.
* Kerchunk/virtualizarr references pointing into archived targets break
  silently. There is no concept of that dependency edge here.
