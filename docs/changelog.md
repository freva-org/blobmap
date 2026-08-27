# Changelog

Manifest schema versions are tracked separately from package versions and are
published at <https://waterpark.dkrz.de/blobmap/>. A consumer must reject a
schema version it does not know rather than attempt to read it.

## v0.5.0

First release.

### Manifest schema v3

* `pinned`: prefixes deliberately held on disk, carrying who, why and an
  optional review date. Preserved across repartitions, unlike `hot_always`,
  which is derived from the store's structure and recomputed each run.

Schema v2 is published as superseded. There is no migration: blob definitions
are recomputed from the store, so delete old manifests and partition again.
Do that before anything is archived, since blob identifiers must be preserved
once tape copies exist.

### Policy defaults changed

* `t_min_bytes` 10 GiB to **0**. Coalescing groups by path adjacency, which
  guesses at access correlation, and guessing wrong means restoring data
  nobody asked for. Aggregating small objects belongs to the tape layer, and
  DKRZ already has a recall proxy that bundles by tape.
* `t_hot_bytes` 1 GiB to **16 MiB**. It exists so that opening a store never
  touches tape, which needs metadata and coordinates -- coordinates are found
  by detection, not by size. Used as a general "small arrays stay hot" rule it
  left an unbounded amount of data permanently on disk: a store of twenty
  800 MB variables was entirely un-archivable.

### Commands

* `pin add|remove|show` -- hold a prefix on disk deliberately. `--reason` is
  required; `pin show` lists expired and open-ended pins first, since those
  are how a hot pool quietly fills. Expiry is reported, never enforced.
* `report` -- how much data no policy can move, split into hot, pinned and
  unmanaged. The last of those is the one that grows silently.
* `schema` -- print the JSON Schema. Needs no store arguments.
* `--version`, reporting the manifest schema version as well as the package.
* `--exclude` -- path segments to skip, defaulting to gateway staging
  directories, which are visible when scanning a backing filesystem but not
  over S3.
* S3 connection flags: `--endpoint`, `--region`, `--anonymous`,
  `--allow-http`, `--virtual-hosted`.

### Fixes made while validating against real stores

* Blob identifiers are derived from the full prefix, not its first path
  segment. On a datatree the old scheme collapsed every blob onto one name and
  disambiguated by iteration order, so a repartition could reassign an
  identifier to different data while tier state still pointed at the old
  meaning.
* `read_arrays` streams the listing in two passes instead of materialising it.
  Holding every object cost roughly 200 bytes each, so a HEALPix store at
  zoom 9 consumed gigabytes and looked like a hang. Memory is now bounded by
  the number of arrays, and progress is logged every 100k objects.
* v2 coordinates are detected. `_ARRAY_DIMENSIONS` lives in `.zattrs`, which
  was never read, so every v2 coordinate looked like a plain array and only
  the size rule kept it off tape.
* A one-dimensional array's chunk key no longer overrides the declared key
  encoding. `cell/0` is what both v2 encodings produce, so it is not evidence
  of either, and treating it as evidence produced a warning for every
  coordinate in a HEALPix store.
* Etags are round-tripped verbatim. obstore 0.11.1 returns them quoted and
  expects the same string back; normalising them broke every conditional
  write, which only shows up when two jobs race.
* Sharded arrays are sized by the shard, which is the stored object.
  zarr-python inverts the naming, and reading the inner chunk understated
  object size by the shard factor.
* A warning when more objects exist than the declared chunk grid allows,
  which skews the chosen bucket width.

### Known limitations

* Coordinate detection needs `_ARRAY_DIMENSIONS`. For stores not written by
  xarray the flag reads false and only `t_hot_bytes` keeps the coordinate on
  disk.
* Scopes are bucket-relative, so a blob identifier is only meaningful paired
  with a bucket name. A consumer needs one resolver per bucket.
* Pins cannot be reconstructed by re-scanning. Back the manifest bucket up.
