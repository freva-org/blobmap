# Internals

For anyone changing the code. Where things live, why the boundaries are where
they are, and how a request moves through.

## Module layers

Dependencies point downward only. Nothing below imports anything above it.

```mermaid
flowchart TB
    CLI["cli<br/><i>argument parsing, output</i>"]
    SVC["service<br/><i>the one operation drivers call</i>"]
    DISC["discover.scan / discover.events<br/><i>what feeds the partitioner</i>"]
    HIER["hierarchy<br/><i>zarr store to arrays</i>"]
    MAN["manifests<br/><i>read and write manifests</i>"]
    BACK["backends<br/><i>URL to store, usable errors</i>"]
    STOR["storage<br/><i>list, get, conditional put</i>"]

    subgraph purelayer["pure: no I/O, no obstore, no storage handle"]
        PART["partition<br/><i>the decisions</i>"]
        RES["resolve<br/><i>key to blob id</i>"]
        MODEL["model<br/><i>manifest types</i>"]
        SCHEMA["schema<br/><i>the contract</i>"]
    end

    OBS(["obstore &mdash; external"])

    CLI --> SVC
    CLI --> DISC
    CLI --> BACK
    DISC --> SVC
    SVC --> HIER
    SVC --> MAN
    SVC --> PART
    HIER --> STOR
    MAN --> STOR
    BACK --> STOR
    STOR --> OBS
    HIER --> MODEL
    PART --> MODEL
    RES --> MODEL
    MODEL --> SCHEMA
```

The boxed modules are **pure**: no I/O, no obstore import, no storage handle.
That is where all the logic that matters lives, and it is why the test suite
can exercise the partitioning rules with hand-written dataclasses in
milliseconds. Keep it that way.

`storage` is the only module that imports obstore. Everything else annotates a
handle as `Store`, imported from `storage`, so the dependency stays at one
edge of the package.

## The two flows

Writing definitions and reading them back are almost independent. They share
only the manifest format.

```mermaid
flowchart LR
    subgraph write["write path: runs occasionally"]
        direction TB
        W1["scan or event"] --> W2["read_arrays<br/>LIST + metadata"]
        W2 --> W3["partition<br/>pure"]
        W3 --> W4["diff vs previous"]
        W4 --> W5["conditional PUT"]
    end

    subgraph read["read path: runs constantly"]
        direction TB
        R1["load_all<br/>at startup"] --> R2["Trie.add_all"]
        R2 --> R3["lookup<br/>ten dict hits"]
    end

    W5 -.->|"manifest"| R1
```

The write path can be slow; it runs once per store. The read path is on the
event stream, so it never touches the network or a database after startup.

## Where the decisions are

`partition.py` is small and worth reading in full. The cut is four rules
applied top down:

```mermaid
flowchart TB
    START["arrays for one scope"] --> HOT{"coordinate<br/>or under t_hot?"}
    HOT -->|yes| PIN["pin hot, never archivable"]
    HOT -->|no| WHOLE{"does the whole<br/>scope fit in t_max?"}
    WHOLE -->|yes| ONE["one blob for everything<br/><b>the common case</b>"]
    WHOLE -->|no| BIG{"is this array<br/>bigger than t_max?"}
    BIG -->|yes| BUCKET["bucket it<br/><b>the only way to cut<br/>inside an array</b>"]
    BIG -->|no| SMALL["coalesce with neighbours<br/>until the group clears t_min"]
```

Two things to internalise before changing any of it:

**Pinning is unconditional.** When `previous` is passed, its blobs are carried
over verbatim and new cuts only fill unclaimed regions. So a policy change has
no effect on an existing scope. Cuts are frozen once made, because a blob id is
a join key against tape addresses.

**Bucketing is arithmetic, not enumeration.** `b_tas_{n // 2048}` resolves
chunk 5,000,000 without anyone having computed it, which is why appending to a
store needs no manifest change.

## Resolution

```mermaid
flowchart LR
    KEY["cordex/a.zarr/tas/c/5000/0/0"] --> BASE{"basename in<br/>hot_always?"}
    BASE -->|yes| HOT["hot<br/>never archive"]
    BASE -->|no| WALK["walk the trie<br/>segment by segment"]
    WALK --> MATCH{"any prefix<br/>matched?"}
    MATCH -->|no| UNM["unmanaged<br/>stays hot, not an error"]
    MATCH -->|yes| BKT{"blob has<br/>a bucket?"}
    BKT -->|no| ZERO["id_0"]
    BKT -->|yes| PARSE["parse the next segment<br/>5000 // 2048 = 2"]
    PARSE --> ID["b_tas_2"]
```

The trie holds **one node per cut, never per object**. A store with 300,000
objects contributes a handful of nodes, and a bucketed array is a single node
however many blobs it spans. `Trie.__len__` exists so a test can assert this
has not accidentally started growing per chunk.

A miss is normal and safe. Unpartitioned stores, foreign data and fresh
uploads all resolve to unmanaged, which means hot.

## Call graph

Generated from the AST, so it cannot drift:

    python tools/callgraph.py --entry partition_store --depth 3
    python tools/callgraph.py --module resolve --private
    python tools/callgraph.py                      # everything public

`partition_store`, two levels deep:

```mermaid
flowchart LR
  subgraph service["service"]
    s1["partition_store"]
  end
  subgraph hierarchy["hierarchy"]
    h1["read_arrays"]
  end
  subgraph partition["partition"]
    p1["partition"]
    p2["diff"]
    p3["Diff.describe"]
  end
  subgraph manifests["manifests"]
    m1["ManifestStore.read"]
    m2["ManifestStore.write"]
    m3["ManifestStore.key"]
  end
  subgraph storage["storage"]
    st1["list_all"]
    st2["get_bytes"]
    st3["head"]
    st4["put_bytes"]
  end
  subgraph model["model"]
    mo1["Manifest.loads"]
    mo2["Manifest.validate"]
    mo3["Manifest.bumped"]
    mo4["Manifest.dumps"]
    mo5["default_hot_always"]
  end

  s1 --> m1
  s1 --> h1
  s1 --> p1
  s1 --> p2
  s1 --> p3
  s1 --> mo3
  s1 --> m2
  h1 --> st1
  m1 --> m3
  m1 --> st2
  m1 --> st3
  m1 --> mo1
  m2 --> m3
  m2 --> mo4
  m2 --> st4
  p1 --> mo5
  p1 --> mo2
  mo1 --> mo2
```

Note `read_arrays` calls only `list_all` and `get_bytes`. It never reads a
chunk, which is what makes partitioning safe on cold data: a v3 shard index
lives *inside* the object, so introspecting one would trigger a restore.

## Testing

```mermaid
flowchart LR
    subgraph pure["no I/O, milliseconds"]
        T1["test_model"]
        T2["test_partition"]
        T3["test_resolve"]
        T4["test_schema"]
    end
    subgraph backed["parametrized over backends"]
        T5["test_storage"]
        T6["test_hierarchy"]
        T7["test_manifests"]
        T8["test_scan"]
        T9["test_service"]
    end
    subgraph fixtures["fixtures"]
        F1["MemoryStore"]
        F2["LocalStore"]
        F3["S3Store<br/><i>if BLOBMAP_TEST_S3_URL</i>"]
    end
    backed --> F1
    backed --> F2
    backed --> F3
```

Nothing is mocked. `MemoryStore` and `LocalStore` are real implementations of
the same API as `S3Store`, so MinIO is one extra fixture param rather than a
second suite.

`tests/zarrgen.py` writes synthetic stores as raw objects with no zarr-python
dependency. That is deliberate: blobmap parses this metadata itself, so the
fixtures must not be produced by the library whose output they imitate.

## Things that will bite

**Changing how a blob id is derived** is a breaking change. Ids join against
tier state and, through it, against tape addresses.

**Adding a field to the manifest** means changing the JSON Schema too. The
schema is the contract, not a derived artifact, and `tests/test_schema.py`
cross-checks the hand-written validator against the real `jsonschema` library.

**`Manifest` validates in `__post_init__`**, so an invalid one cannot exist in
memory. If a test needs a malformed document, build the dict rather than the
dataclass.

**Sharding inverts the naming.** `Array.chunks` in zarr-python is the *inner*
chunk; `Array.shards` is what becomes an object. Reading the wrong one
produces blobs too large by the shard factor, silently. Use
`Array.object_shape`.
