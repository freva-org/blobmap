# blobmap

Decides which zarr objects move to and from tape together, and reads that
decision back.

    model       manifest types              stdlib only
    partition   arrays -> Manifest          pure, no I/O
    resolve     object key -> blob id       the semantics of the format
    storage     three ops over obstore      S3, local disk, memory
    manifests   manifest read/write         conditional, in a bucket you own
    hierarchy   zarr store -> arrays        LIST-first, never reads a chunk
    discover    scan / event drivers        what feeds the partitioner

A **blob** is a set of objects that moves as a unit. Per-chunk is too fine to
track -- 300k rows for one store -- and per-store is too coarse to tier. A
blob sits in between, sized so that a restore is worth a tape mount.

The **manifest** is a rule, not a table. `tas/c/<n>/...` -> `b_tas_{n // 2048}`
resolves chunk 5,000,000 without anyone having computed it, so appends need no
manifest change at all. It lives in a bucket you own at a path mirroring the
data, which is what makes this work for stores you must not alter.

Manifests are pure definition. Tier, last access and restore state live in
blobtier's table, keyed by blob id.

## Pins

Everything except metadata, coordinates and genuinely tiny arrays is
archivable. To hold something on disk deliberately:

    blobmap ... pin add cordex/a.zarr multiscales/zoom_9 \
        --reason "active ICON analysis" --until 2026-12-01
    blobmap ... pin show
    blobmap ... pin remove cordex/a.zarr multiscales/zoom_9

A pin is set once and persists across repartitions, rather than being a flag
passed on every run: whether a dataset stays hot should not depend on shell
history. `--reason` is required, because a pin nobody can explain is one
nobody removes, and `pin show` lists expired and open-ended pins first since
those are how a hot pool quietly fills.

Expiry is reported, never enforced. Nothing is unpinned behind your back.

## Checking what is not archivable

    blobmap ... report
    blobmap ... report --per-bucket

    scope       total   archivable        hot     pinned   unmanaged    blobs
    era5     412.0 TiB    408.1 TiB    0.3 TiB    3.6 TiB       0.0 B    4,102
    cmip6     88.0 TiB     71.2 TiB    0.1 TiB    2.5 TiB    14.2 TiB    1,880

    cmip6: 16% unmanaged. Something exists that no manifest claims -- an
    unpartitioned store, a variable added since the last run, or a layout the
    partitioner did not recognise. Run scan.

`unmanaged` is the column to watch. Hot and pinned data is held back for
reasons someone chose; unmanaged data is held back because nothing knows about
it, which is how a pool fills without anyone noticing.

## Invariants

* Every blob is `{id, prefixes, bucket}`. `bucket: null` means one bucket, so
  the resolved id is always `f"{id}_{n}"` and the resolver has one code path.
* Metadata objects and dimension coordinates are never archivable, so
  `xr.open_zarr` never triggers a tape mount -- and re-reading a store to
  partition it cannot feed its own event loop.
* Repartitioning is additive *by construction*: pinned blobs are carried over
  verbatim, new cuts only fill unclaimed regions. A policy change therefore
  has no effect on an existing scope; cuts are frozen once made. Only
  `--force` can move a blob, and it says what it orphaned.
* An unknown path resolves to unmanaged-and-hot. Misses are normal, not errors.
* `hot_always` is derived from structure and recomputed on every partition.
  `pinned` is intent, carried over untouched, and the one part of a manifest
  that cannot be reconstructed by re-scanning -- so back the manifest bucket
  up.
* There is no minimum blob size. Coalescing groups by path adjacency, which
  guesses at access correlation, and guessing wrong means restoring data
  nobody asked for. Aggregating small objects belongs to the tape layer.
* The storage unit is the shard where sharding is in use. zarr-python inverts
  the naming (`Array.chunks` is the *inner* chunk); getting this wrong makes
  blobs larger than the target by the shard factor.
* Array roots come from where the metadata objects are, never from the shape
  of chunk keys. A group named `c` is a group.

## Paths and URLs

Both `--data` and `--manifests` take either a URL or a plain path. These are
equivalent:

    --manifests /work/blobmap
    --manifests ./blobmap
    --manifests ~/blobmap
    --manifests file:///work/blobmap

A local manifest directory is created if it does not exist, since blobmap owns
it. A missing `--data` path is an error rather than being created: silently
accepting a typo there would make a scan report nothing found, which looks
identical to an empty bucket.

## Connecting to S3

`obstore` reads `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
and `AWS_REGION` from the environment. With none of them set it assumes AWS
and looks for instance credentials at `169.254.169.254`, which off EC2 hangs
for the retry budget and then fails with a Rust backtrace that never mentions
credentials.

So for anything that is not AWS, either export those variables or pass the
flags:

    blobmap --data s3://cmip6 --endpoint https://s3.example.org \
        --manifests file:///work/blobmap scan

`--anonymous` skips credentials and signing entirely, which is the quickest
way to read a public bucket and to avoid the metadata lookup. `--allow-http`
is implied when the endpoint already starts with `http://`.

## Usage

    blobmap --data s3://cordex --manifests s3://waterpark-blobmap \
        scan --partition
    blobmap --data s3://cordex --manifests s3://waterpark-blobmap \
        partition cordex/nukleus/eur11.zarr --dry-run
    blobmap --data s3://cordex --manifests s3://waterpark-blobmap \
        resolve cordex/nukleus/eur11.zarr/tas/c/5000/0/0

```python
from obstore.store import S3Store
from blobmap import ManifestStore, Trie

trie = Trie()
trie.add_all(ManifestStore(S3Store(bucket="waterpark-blobmap")).load_all())
trie.lookup(key).blob_id          # or .kind == "hot" / "unmanaged"
```

`--dry-run` prints the per-array decision table. Expect to stare at it while
tuning `t_max`.

## Testing

    pytest -q                      # memory + local backends
    docker compose up -d           # then export the vars in the compose file
    pytest -q                      # same suite, plus a real S3 backend

`MemoryStore` and `LocalStore` implement the same API as `S3Store`, so nothing
is mocked and MinIO is one extra fixture param rather than a second suite.

## Notes

`obstore` 0.11: `LocalStore` supports create-if-absent but *not*
update-if-etag, so a local repartition falls back to overwrite with a warning.
On S3 both conditions work, and two jobs partitioning the same scope produce a
conflict rather than last-writer-wins.

The event driver needs MinIO configured with `format=access` and `queue_dir`
set. Give the partitioner its own service account and list it in
`ignore_principals`, or its metadata reads will look like user traffic --
and tiering reads would mark every blob it archives as freshly accessed.

`EventPoller` needs a `root_of` callable (use `find_store_root`) to map a
metadata write to its store root. Without it, two variables in one store
debounce independently and the partitioner gets handed a sub-array as a scope.
