<div align="center">
  <img src="assets/logo-color.svg" alt="" width="88">
  <h1>blobmap</h1>
  <p><em>Decide which zarr objects move to tape together, and read that back.</em></p>
</div>

---

A zarr store is hundreds of thousands of objects with no structure S3 can see.
Tiering it needs a unit that is neither a chunk nor the whole store:

- **per chunk** is untrackable. One store is 10^5 to 10^6 objects, so the state
  table outgrows the data catalogue and a single user read fires thousands of
  updates.
- **per store** is useless. A 50 TB store is one unit, and a datatree can put
  an entire bucket in one store.

A **blob** sits in between: a set of objects that move as a unit, sized so a
restore is worth a tape mount.

blobmap decides where the cuts go, writes them down, and reads them back. It
does not move anything. Tier state, archiving and restore belong to whatever
consumes the manifests.

## The manifest is a rule, not a table

A blob definition is a prefix plus arithmetic:

```
tas/c/<n>/...   ->   b_tas_{n // 2048}
```

Chunk 5,000,000 resolves without anyone having enumerated it. So **appending
to a store needs no manifest change**, no epoch bump, and nothing running.
Lookup walks an in-memory trie holding one node per *cut*, never per object: a
store with 300,000 objects contributes a handful of nodes.

Manifests live in a bucket you own, at a path mirroring the data. Never inside
the store, because data arrives on the hub that must not be altered. The
format is published and versioned at
[waterpark.dkrz.de/blobmap](https://waterpark.dkrz.de/blobmap/).

## Install

```bash
pip install blobmap            # one runtime dependency: obstore
pip install blobmap[pretty]    # coloured help output
```

## Use

```bash
# what is here, and what has never been partitioned?
blobmap --data s3://cordex --manifests s3://waterpark-blobmap scan

# cut one store, see the decisions before writing anything
blobmap --data s3://cordex --manifests s3://waterpark-blobmap \
    partition nukleus/eur11.zarr --dry-run

# which blob does this object belong to?
blobmap --data s3://cordex --manifests s3://waterpark-blobmap \
    resolve nukleus/eur11.zarr/tas/c/5000/0/0
```

`--dry-run` prints the per-array decision table. Expect to look at it while
tuning `t_max`:

```
array          objects      stored      object  blob
tas             10,266   459.0 GiB    45.8 MiB  b_tas_0..5  (width=2048, ~91.6 GiB/blob)
hus              1,027   459.0 GiB   457.7 MiB  b_hus_0..8  (width=128,  ~57.2 GiB/blob)
pr              10,266    20.0 GiB     2.0 MiB  b_pr_0
time                 1     3.0 KiB     3.0 KiB  pinned hot
```

`tas` and `hus` hold the same bytes; `hus` is sharded 10x, so it has a tenth
of the objects and a correspondingly smaller width. The shard is the stored
object, and getting that wrong makes blobs larger than the target by the shard
factor.

### Paths and URLs

`--data` and `--manifests` take a URL or a plain path, so `/work/blobmap`,
`./blobmap`, `~/blobmap` and `file:///work/blobmap` are equivalent. A local
manifest directory is created if missing; a missing `--data` path is an error
rather than being created, since silently accepting a typo would make a scan
report nothing found, which is indistinguishable from an empty bucket.

Point `--data` at exactly one **bucket**. Scopes are bucket-relative, so a
scope string only means something paired with a bucket name, and two buckets
sharing a manifest root would collide.

### Connecting to S3

`obstore` reads `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY` and `AWS_REGION` from the environment. With none of
them set it assumes AWS and looks for instance credentials at
`169.254.169.254`, which off EC2 hangs for the retry budget and then fails
with an error that never mentions credentials. For anything that is not AWS:

```bash
blobmap --data s3://cmip6 --endpoint https://s3.example.org \
        --manifests /work/blobmap scan
```

`--anonymous` skips credentials and signing entirely, which is the quickest
way to read a public bucket.

### Scanning a gateway's backing filesystem

With versitygw's posix backend, object keys map onto paths under the bucket
directory, so blobmap can be pointed at Lustre instead of the S3 endpoint. The
relative keys are identical, so the manifests come out the same:

```bash
blobmap --data /lustre/waterpark/eerie --manifests /work/blobmap scan
```

No HTTP round trip per LIST page, and `lfs hsm_state` can tell you what is
already released. Gateway staging directories are visible this way but hidden
over S3, so `.sgwtmp`, `.versitygw` and `.snapshot` are excluded by default.
`--exclude` adds more, `--exclude ""` disables it.

## Pins

Everything except metadata, coordinates and genuinely tiny arrays is
archivable. To hold something on disk deliberately:

```bash
blobmap ... pin add cordex/a.zarr multiscales/zoom_9 \
    --reason "active ICON analysis" --until 2026-12-01
blobmap ... pin show
blobmap ... pin remove cordex/a.zarr multiscales/zoom_9
```

A pin is set once and persists across repartitions, rather than being a flag
passed on every run: whether a dataset stays hot should not depend on shell
history. `--reason` is required, because a pin nobody can explain is one
nobody removes. `pin show` lists expired and open-ended pins first, since
those are how a hot pool quietly fills.

Expiry is reported, never enforced. Nothing is unpinned behind your back.

## What is not archivable

```bash
blobmap ... report --per-bucket
```

```
scope       total   archivable        hot     pinned   unmanaged    blobs
era5    412.0 TiB    408.1 TiB    0.3 TiB    3.6 TiB       0.0 B    4,102
cmip6    88.0 TiB     71.2 TiB    0.1 TiB    2.5 TiB    14.2 TiB    1,880

cmip6: 16% unmanaged. Something exists that no manifest claims -- an
unpartitioned store, a variable added since the last run, or a layout the
partitioner did not recognise. Run scan.
```

`unmanaged` is the column to watch. Hot and pinned data is held back for
reasons someone chose; unmanaged data is held back because nothing knows about
it, which is how a pool fills without anyone noticing.

## As a library

```python
from obstore.store import S3Store
from blobmap import ManifestStore, Trie, partition_store

data = S3Store(bucket="cordex")
manifests = ManifestStore(S3Store(bucket="waterpark-blobmap"))

partition_store(data, manifests, "nukleus/eur11.zarr")

trie = Trie()
trie.add_all(manifests.load_all())
trie.lookup("nukleus/eur11.zarr/tas/c/5000/0/0").blob_id   # 'b_tas_2'
```

A consuming service builds one `Trie` per bucket at startup, which is a LIST
plus parallel GETs over a few thousand small JSON objects with no database on
the path, then calls `lookup` per event. Resolution lives here rather than in
the consumer because it *is* the semantics of the format: reimplementing the
parse rules elsewhere means they drift, and the failure mode is chunks
attributed to the wrong tape unit.

## Invariants

- Every blob is `{id, prefixes, bucket}`. `bucket: null` means one bucket, so
  the resolved id is always `{id}_{n}` and a consumer needs one code path.
- Metadata objects and dimension coordinates are never archivable, so
  `xr.open_zarr` works on a fully archived store, and re-reading a store to
  partition it cannot feed its own event loop.
- Repartitioning is additive *by construction*: existing blobs are carried
  over verbatim and new cuts only fill unclaimed regions. A policy change
  therefore has no effect on an existing scope, because cuts are frozen once
  made. Only `--force` can move a blob, and it says what it orphaned.
- Blob ids come from the full prefix and are independent of listing order,
  because an id is a join key against tier state and, through it, a tape
  address.
- An unknown path resolves to unmanaged-and-hot. Misses are normal, not
  errors: unpartitioned stores, foreign data and fresh uploads all land there.
- `hot_always` is derived from structure and recomputed every run. `pinned` is
  intent, carried across, and the one part of a manifest that cannot be
  reconstructed by re-scanning, so back the manifest bucket up.
- There is no minimum blob size. Coalescing guesses at access correlation from
  path adjacency, and guessing wrong means restoring data nobody asked for.
  Aggregating small objects belongs to the tape layer.

## Documentation

| | |
|---|---|
| [Operations](operations.md) | running it: what the parts are, what to configure, what breaks |
| [Internals](internals.md) | changing it: module layers, the two flows, a generated call graph |
| [Design](design.md) | why the pieces are shaped this way |
| [Manifest format](https://waterpark.dkrz.de/blobmap/) | the published schema, and its stability policy |

## Development

```bash
tox                 # lint, types, tests, docs
tox -e test
tox -e docs-serve
```

Tests use no mocks. `MemoryStore` and `LocalStore` are real implementations of
the same API as `S3Store`, so the same suite runs against all three and MinIO
is one extra fixture rather than a second suite:

```bash
docker compose up -d
export BLOBMAP_TEST_S3_URL=s3://blobmap-test
export BLOBMAP_TEST_S3_ENDPOINT=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_REGION=us-east-1
pytest -q
```

That run is what actually exercises conditional writes: `LocalStore` has no
update-if-etag, so the `If-Match` path is untested without a real S3.

See [Contributing](contributing.md).

## Licence

BSD-3-Clause.
