# blobmap manifest schema

A **blobmap manifest** records which objects in a zarr store move to and from
tape together. One JSON object per scope, stored alongside the data it
describes but never inside it.

This page is the permanent home of the format. The schemas below are stable
URLs: once published, a version's constraints never change.

## Versions

| version | schema | status |
|---|---|---|
| 3 | [`manifest-v3.schema.json`](manifest-v3.schema.json) | current |
| 2 | [`manifest-v2.schema.json`](manifest-v2.schema.json) | superseded |

Version 2 predates the `pinned` field and is otherwise identical. It is
published so that manifests written during development remain identifiable,
not because it is supported: no v2 manifests should exist in production, and a
consumer written today should accept version 3 only.

Version 1 was never used outside a single development branch and is not
published.

### Moving from 2 to 3

There is no migration tool, because there is nothing worth migrating. Blob
definitions are recomputed from the store, so the answer is to delete the old
manifest and partition again:

```bash
blobmap --data ... --manifests ... partition <scope>
```

The only content a repartition cannot reconstruct is pins, and version 2 had
none. Do this before anything is archived: once tape copies exist, blob
identifiers must be preserved and deleting a manifest orphans them.

## Using it

Validate a manifest against the schema in any language with a JSON Schema
implementation. Draft 2020-12.

```bash
curl -sO https://waterpark.dkrz.de/blobmap/manifest-v3.schema.json
check-jsonschema --schemafile manifest-v3.schema.json manifest.json
```

```python
import json, urllib.request, jsonschema

url = "https://waterpark.dkrz.de/tech/blobmap/manifest-v3.schema.json"
schema = json.load(urllib.request.urlopen(url))
jsonschema.validate(manifest, schema)
```

The `blobmap` package carries its own copy and validates on read, so Python
consumers need neither the download nor a JSON Schema library.

## What a manifest looks like

```json
{
  "schema_version": 3,
  "scope": "healpix/era5land/PT1H/level_9.zarr",
  "epoch": 4,
  "generated_at": "2026-08-19T09:12:00+00:00",
  "generated_by": "blobmap 0.5.0",
  "policy": {"t_max_bytes": 107374182400, "t_min_bytes": 0,
             "t_hot_bytes": 16777216, "width_clamp": 8, "pow2_floor": 64},
  "hot_always": ["**/zarr.json", "**/.zarray", "**/.zattrs",
                 "time/**", "cell/**"],
  "pinned": [],
  "blobs": [
    {"id": "b_t2m", "prefixes": ["t2m"],
     "bucket": {"index": 0, "width": 2048, "key_encoding": "v2_slash"}}
  ],
  "provenance": {}
}
```

The central idea is that a manifest is a **rule, not a table**. The entry
above resolves any chunk of `t2m` arithmetically:

```
t2m/5000/0   ->   b_t2m_2        because 5000 // 2048 == 2
```

So chunk five million resolves without anyone having enumerated it, and
appending to a store requires no change to the manifest.

Keys matching `hot_always` or covered by a `pinned` prefix are never
archivable. A key that matches nothing is unmanaged, which also means it stays
on disk: that is the safe default, not an error.

## Stability policy

**Constraints are frozen.** Once a version is published, no change may make a
previously valid manifest invalid, or a previously invalid one valid. Anything
that would is a new version at a new URL, and the old schema stays served
indefinitely.

**Prose may be corrected.** A `title` or `description` may be fixed in place.
Validators read only constraints; the prose is for people. Bumping the version
for a typo would invalidate every manifest in existence, since consumers check
`schema_version` exactly.

**Superseded versions stay online.** A manifest written years ago must remain
checkable against the schema it was written for, even when nothing is expected
to produce that version any more.

The two schemas reject each other, deliberately. A version 2 document fails
against version 3 because `schema_version` must be exactly 3, and a version 3
document fails against version 2 because `pinned` is not a known field. There
is no version in which a document is silently readable as the wrong one.

## Fields

| field | required | meaning |
|---|---|---|
| `schema_version` | yes | Format version. Reject a version you do not know. |
| `scope` | yes | Prefix these definitions cover, relative to the bucket. |
| `epoch` | yes | Bumped when definitions change, so a consumer can detect staleness without diffing. |
| `blobs` | yes | The definitions. A list of rules, so its length tracks cut decisions, not object count. |
| `hot_always` | yes | Glob patterns never archivable. Derived from the store's structure and recomputed on every partition. |
| `pinned` | no | Prefixes deliberately held on disk. Preserved across repartitions; carries who, why and an optional review date. |
| `policy` | no | Thresholds used to produce this cut. |
| `generated_at`, `generated_by` | no | Provenance of the producing run. |
| `provenance` | no | Measured sizes, for debugging. These go stale while the manifest is untouched, so a consumer must not read them. |

A blob is always `{id, prefixes, bucket}`. `bucket: null` means one bucket, so
the resolved identifier is always `{id}_{n}` and a consumer needs one code
path rather than two.

Blob identifiers are join keys against tier state and, through it, against
tape addresses. They are derived from the full prefix and are independent of
the order arrays were listed.

## Implementation

The reference implementation can be retrieved by the `schama` subcommand
which validates manifests on read and carries its own copy of the schema, 
so Python consumers need neither the download nor a JSON Schema library. 
The published files here are generated from it:

```bash
blobmap schema > manifest-v3.schema.json
```

Questions: [waterpark@support.dkrz.de](mailto:waterpark@support.dkrz.de).
