"""Reading a store: metadata objects for structure, LIST for sizes.

Array roots come from *where the metadata objects are*, never from the shape
of chunk keys. That is exact rather than heuristic: an array root is a prefix
holding a `zarr.json` with `node_type: array`, or a `.zarray`. Guessing from
key shape, by looking for a `c` segment or a numeric one, breaks on a group
legitimately named `c`, on v2 flat keys, and on datatrees.

!!! note "Nothing here reads a chunk"
    Only metadata objects are fetched, and those are in `hot_always`. So
    partitioning a store cannot trigger a tape restore, and cannot feed its
    own event loop.

Attributes:
    V3_MARKER: Basename identifying a zarr v3 node.
    V2_GROUP: Basename identifying a zarr v2 group.
    V2_ARRAY: Basename identifying a zarr v2 array.

Example:
    ```python
    from obstore.store import S3Store
    from blobmap import read_arrays

    arrays = read_arrays(S3Store(bucket="cordex"), "nukleus/eur11.zarr")
    for a in arrays:
        print(a.path, a.nobjects, a.total_bytes, a.key_encoding)
    ```
"""

from __future__ import annotations

import json
import logging
import posixpath
from dataclasses import dataclass, field
from typing import Any, Sequence

from .model import METADATA_BASENAMES, Array
from .storage import Store, get_bytes, list_all, list_names

log = logging.getLogger(__name__)

V3_MARKER = "zarr.json"
V2_GROUP = ".zgroup"
V2_ARRAY = ".zarray"
V2_ATTRS = ".zattrs"

#: Path segments to skip by default.
#:
#: These matter when scanning an S3 gateway's backing filesystem rather than
#: going through the S3 API. Versity's posix backend stages multipart uploads
#: under `.sgwtmp`, which is invisible over S3 but shows up in a directory
#: walk. Left in, staging objects inside a store prefix would be counted into
#: an array's size and skew the chosen bucket width.
DEFAULT_EXCLUDE: tuple[str, ...] = (".sgwtmp", ".versitygw", ".snapshot")

#: Log progress every this many objects.
#:
#: A store with millions of chunks takes minutes to walk. Without this the
#: command looks hung, which is indistinguishable from being hung.
PROGRESS_EVERY = 100_000

_ITEMSIZE: dict[str, int] = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "float16": 2,
    "int32": 4,
    "uint32": 4,
    "float32": 4,
    "int64": 8,
    "uint64": 8,
    "float64": 8,
    "complex64": 8,
    "complex128": 16,
}


class NotAZarrStore(ValueError):
    """No zarr metadata was found under the given prefix."""


def excluded(key: str, exclude: Sequence[str] = DEFAULT_EXCLUDE) -> bool:
    """Whether a key sits under an excluded path segment.

    Matches whole segments, so `.sgwtmp` skips `.sgwtmp/multipart/x` but a
    file merely *named* `data.sgwtmp.nc` is kept.

    Args:
        key: Object key.
        exclude: Path segments to skip.

    Returns:
        True if any segment of the key is excluded.

    Example:
        >>> excluded(".sgwtmp/multipart/staging-abc")
        True
        >>> excluded("healpix/mean.zarr/tas/0/0")
        False
        >>> excluded("healpix/data.sgwtmp.nc")
        False
    """
    return any(part in exclude for part in key.split("/"))


def detect_format(store: Store, scope: str) -> str | None:
    """Identify the zarr format at a prefix.

    Args:
        store: A storage handle.
        scope: Prefix to inspect.

    Returns:
        `"v3"`, `"v2"`, or `None` when this is not a store root. One
        delimited LIST, no recursion, so it is cheap enough to call at every
        level of a scan.
    """
    names = list_names(store, _prefix(scope))
    if V3_MARKER in names:
        return "v3"
    if names & {V2_GROUP, V2_ARRAY}:
        return "v2"
    return None


def find_store_root(store: Store, prefix: str, ceiling: str = "") -> str | None:
    """Climb to the outermost prefix that still holds zarr metadata.

    A metadata write tells you a *node* changed, not which store it belongs
    to. Debouncing on the node would fragment the pending set per variable
    and, worse, hand the partitioner a sub-array as if it were a store root.
    Pass this as `root_of` to
    [`EventPoller`][blobmap.discover.events.EventPoller].

    Args:
        store: A storage handle.
        prefix: Where to start climbing, usually a metadata object's parent.
        ceiling: Do not climb above this prefix.

    Returns:
        The store root, or `None` if nothing on the way up looks like zarr.
        Costs one delimited LIST per level.
    """
    parts = [p for p in prefix.strip("/").split("/") if p]
    floor = len([p for p in ceiling.strip("/").split("/") if p])
    found: str | None = None
    for i in range(len(parts), floor, -1):
        candidate = "/".join(parts[:i])
        if detect_format(store, candidate) is not None:
            found = candidate
    return found


@dataclass
class _Node:
    """Accumulated data objects belonging to one array.

    Attributes:
        stored_bytes: Sum of object sizes seen.
        nobjects: Count of objects seen.
        sample_keys: A few keys, kept to verify the key encoding by
            inspection rather than trusting the declared metadata.
    """

    stored_bytes: int = 0
    nobjects: int = 0
    sample_keys: list[str] = field(default_factory=list)


def read_arrays(
    store: Store, scope: str, *, exclude: Sequence[str] = DEFAULT_EXCLUDE
) -> list[Array]:
    """Describe every array under a scope, sized from a single LIST.

    Args:
        store: A storage handle for the data.
        scope: Prefix to read, usually a store root.
        exclude: Path segments to skip, defaulting to
            `DEFAULT_EXCLUDE`. Relevant
            when scanning a gateway's backing filesystem, which exposes
            staging directories the S3 API hides.

    Note:
        The listing is walked twice: once for metadata keys, once to
        accumulate sizes. That keeps memory proportional to the number of
        arrays rather than the number of objects, which matters at HEALPix
        scale where a single store holds millions of chunks. It also means the
        two passes could disagree if the store is written concurrently; an
        object appearing between them is reported as belonging to no array.
        Debouncing before partitioning is what avoids that.

    Returns:
        One [`Array`][blobmap.model.Array] per array found, sorted by path.
        Arrays whose metadata cannot be read are skipped with a warning
        rather than failing the whole run.

    Raises:
        NotAZarrStore: If no zarr metadata is found at all.

    Note:
        Logs a warning when data objects belong to no array. Those resolve as
        unmanaged and stay hot, which is safe but means storage nobody is
        tiering.
    """
    prefix = _prefix(scope)

    # Pass one keeps only metadata keys. Retaining every data entry costs
    # roughly 200 bytes each, which for a HEALPix store at zoom 9 is hundreds
    # of megabytes to gigabytes -- and looks like a hang rather than a
    # slowdown, because nothing is printed while it accumulates.
    metadata_keys: list[str] = []
    seen = 0
    for entry in list_all(store, prefix):
        seen += 1
        if seen % PROGRESS_EVERY == 0:
            log.info(
                "%s: %d objects listed, %d metadata",
                scope or ".",
                seen,
                len(metadata_keys),
            )
        if excluded(_relative(entry.key, prefix), exclude):
            continue
        if posixpath.basename(entry.key) in METADATA_BASENAMES:
            metadata_keys.append(entry.key)

    if not metadata_keys:
        raise NotAZarrStore(f"{scope}: no zarr metadata found")
    log.debug(
        "%s: %d objects, %d metadata objects", scope or ".", seen, len(metadata_keys)
    )

    # v2 keeps attributes in a separate .zattrs object, so _ARRAY_DIMENSIONS
    # is invisible unless we read it. Without this every v2 coordinate looks
    # like a plain array, and only the size rule keeps it off tape.
    attributes: dict[str, dict[str, Any]] = {}
    for key in metadata_keys:
        if posixpath.basename(key) != V2_ATTRS:
            continue
        loaded = _load(store, key)
        if loaded is not None:
            attributes[_relative(posixpath.dirname(key), prefix)] = loaded

    arrays: dict[str, dict[str, Any]] = {}
    for key in metadata_keys:
        base = posixpath.basename(key)
        if base not in (V3_MARKER, V2_ARRAY):
            continue
        meta = _load(store, key)
        if meta is None or not _is_array(meta):
            continue
        root = _relative(posixpath.dirname(key), prefix)
        if base == V2_ARRAY and root in attributes:
            # normalise onto the v3 shape so _is_coordinate has one code path
            meta = {**meta, "attributes": attributes[root]}
        arrays[root] = meta

    # Pass two streams the listing again, accumulating per array rather than
    # per object, so memory is bounded by the number of arrays. Two walks cost
    # less than holding the first one.
    nodes: dict[str, _Node] = {root: _Node() for root in arrays}
    unassigned = 0
    counted = 0
    for entry in list_all(store, prefix):
        relative = _relative(entry.key, prefix)
        if excluded(relative, exclude):
            continue
        if posixpath.basename(entry.key) in METADATA_BASENAMES:
            continue
        counted += 1
        if counted % PROGRESS_EVERY == 0:
            log.info("%s: %d objects sized", scope or ".", counted)
        owner = _owner(relative, nodes)
        if owner is None:
            unassigned += 1
            continue
        node = nodes[owner]
        node.stored_bytes += entry.size
        node.nobjects += 1
        if len(node.sample_keys) < 4:
            node.sample_keys.append(relative)
    if unassigned:
        log.warning(
            "%s: %d objects belong to no array; they will resolve as "
            "unmanaged and stay hot",
            scope,
            unassigned,
        )

    out: list[Array] = []
    for root in sorted(arrays):
        array = _to_array(root, arrays[root], nodes[root])
        if array is None:
            continue
        if array.nobjects_seen and array.nobjects_seen > array.nobjects:
            # the declared grid says fewer objects than are actually stored,
            # so something is left over from an append or a rechunk. This
            # skews avg_object_bytes and therefore the chosen bucket width.
            log.warning(
                "%s: %d objects stored but the declared grid holds %d "
                "(shape=%s, object shape=%s) -- stale chunks from an append "
                "or rechunk will skew sizing",
                array.path or ".",
                array.nobjects_seen,
                array.nobjects,
                array.shape,
                array.object_shape,
            )
        out.append(array)
    return out


def _owner(rel: str, roots: dict[str, _Node]) -> str | None:
    """Find which array a data object belongs to.

    Args:
        rel: Object key relative to the scope.
        roots: Known array roots.

    Returns:
        The deepest array root that is an ancestor of the key, or `None` if
        no array claims it.
    """
    parts = rel.split("/")
    for i in range(len(parts) - 1, -1, -1):
        candidate = "/".join(parts[:i])
        if candidate in roots:
            return candidate
    return None


def _to_array(root: str, meta: dict[str, Any], node: _Node) -> Array | None:
    """Build an [`Array`][blobmap.model.Array] from metadata plus tallies.

    Handles v2 and v3, sharded and not. Where the declared key encoding
    disagrees with the keys actually present, the keys win: foreign stores do
    not always match their own metadata, and a wrong encoding silently
    misroutes every chunk.

    Args:
        root: Array path relative to the scope.
        meta: Parsed `zarr.json` or `.zarray`.
        node: Sizes and sample keys collected from the listing.

    Returns:
        An `Array`, or `None` if the metadata is unusable.
    """
    try:
        if meta.get("zarr_format") == 3 or "chunk_grid" in meta:
            shape = tuple(int(x) for x in meta["shape"])
            objects = tuple(
                int(x) for x in meta["chunk_grid"]["configuration"]["chunk_shape"]
            )
            inner = _inner_chunk(meta)
            itemsize = _itemsize(meta["data_type"])
            encoding = _v3_encoding(meta)
            chunks = inner or objects
            shards = objects if inner and inner != objects else None
        else:
            shape = tuple(int(x) for x in meta["shape"])
            chunks = tuple(int(x) for x in meta["chunks"])
            shards = None
            itemsize = _itemsize(meta["dtype"])
            separator = str(meta.get("dimension_separator", "."))
            encoding = "v2_slash" if separator == "/" else "v2_flat"
    except (KeyError, TypeError, ValueError) as exc:
        log.warning("%s: unreadable array metadata (%s)", root or ".", exc)
        return None

    observed = _observed_encoding(root, node.sample_keys, len(shape))
    if observed and observed != encoding:
        # trust the keys: foreign stores do not always match their own
        # declared metadata, and a wrong encoding silently misroutes chunks
        log.warning(
            "%s: metadata says %s but keys look like %s; using %s",
            root or ".",
            encoding,
            observed,
            observed,
        )
        encoding = observed

    return Array(
        path=root,
        shape=shape,
        chunks=chunks,
        itemsize=itemsize,
        shards=shards,
        stored_bytes=node.stored_bytes or None,
        nobjects_seen=node.nobjects or None,
        is_coordinate=_is_coordinate(root, meta, shape),
        key_encoding=encoding,
    )


def _v3_encoding(meta: dict[str, Any]) -> str:
    """Determine the key encoding a v3 array declares.

    Args:
        meta: Parsed v3 array metadata.

    Returns:
        One of `v3_slash`, `v2_slash`, `v2_flat`. A v3 array may opt in to v2
        style keys, so this is not always `v3_slash`.
    """
    encoding = meta.get("chunk_key_encoding", {})
    name = encoding.get("name", "default")
    separator = encoding.get("configuration", {}).get(
        "separator", "/" if name == "default" else "."
    )
    if name == "v2":
        return "v2_slash" if separator == "/" else "v2_flat"
    return "v3_slash"


def _observed_encoding(root: str, samples: list[str], ndim: int) -> str | None:
    """Infer the key encoding from keys that actually exist.

    Only decisive evidence counts. For a one-dimensional array the two v2
    encodings are *indistinguishable* -- `cell/0` is what both produce -- so
    inferring from a bare digit would override correct metadata on the basis
    of nothing. That needs more than one dimension to be a real observation.

    Args:
        root: Array path relative to the scope.
        samples: A few of the array's data keys.
        ndim: Number of dimensions, used to tell an ambiguous key from an
            informative one.

    Returns:
        The inferred encoding, or `None` when the samples cannot distinguish
        one, including when the array has not been written yet.

    Example:
        >>> _observed_encoding("tas", ["tas/c/0/0"], 2)
        'v3_slash'
        >>> _observed_encoding("tas", ["tas/0.0"], 2)
        'v2_flat'
        >>> _observed_encoding("tas", ["tas/0/0"], 2)
        'v2_slash'
        >>> _observed_encoding("cell", ["cell/0"], 1) is None
        True
    """
    for key in samples:
        i = len(root)
        rest = key[i:].strip("/") if root else key
        head = rest.split("/")[0]
        if head == "c":
            return "v3_slash"
        if "." in head and head.replace(".", "").isdigit():
            return "v2_flat"
        if ndim > 1 and head.isdigit():
            # a multi-dimensional array with a bare digit segment must be
            # nesting the remaining indices
            return "v2_slash"
    return None


def _inner_chunk(meta: dict[str, Any]) -> tuple[int, ...] | None:
    """Extract the inner chunk shape from a sharding codec.

    Args:
        meta: Parsed v3 array metadata.

    Returns:
        The inner chunk shape when the array is sharded, else `None`. When
        this is set, the array's declared chunk shape is the *shard*, which
        is what becomes an object.
    """
    for codec in meta.get("codecs", []) or []:
        if isinstance(codec, dict) and codec.get("name") == "sharding_indexed":
            shape = codec.get("configuration", {}).get("chunk_shape")
            if shape:
                return tuple(int(x) for x in shape)
    return None


def _is_coordinate(root: str, meta: dict[str, Any], shape: tuple[int, ...]) -> bool:
    """Whether this array is a dimension coordinate.

    Coordinates must stay hot, or opening the store hits tape.

    In v3 the attributes are inline in `zarr.json`; in v2 they live in a
    separate `.zattrs`, which `read_arrays` folds in under the same key so
    this has one code path.

    `_ARRAY_DIMENSIONS` only exists if xarray wrote the store. For foreign
    data we fall back to the size rule, which catches these in practice since
    coordinates are almost always well under `t_hot_bytes` -- but the flag
    itself will read `False`, which matters if that threshold is ever lowered.

    Args:
        root: Array path relative to the scope.
        meta: Parsed array metadata, with v2 attributes already folded in.
        shape: The array shape.

    Returns:
        True for a one-dimensional array whose single dimension is named
        after itself.

    Example:
        >>> _is_coordinate("zoom_9/time",
        ...                {"attributes": {"_ARRAY_DIMENSIONS": ["time"]}},
        ...                (780,))
        True
        >>> _is_coordinate("zoom_9/tas",
        ...                {"attributes": {"_ARRAY_DIMENSIONS": ["time"]}},
        ...                (780, 4))
        False
    """
    attributes = meta.get("attributes") or meta.get("_ARRAY_DIMENSIONS")
    if isinstance(attributes, dict):
        dims = attributes.get("_ARRAY_DIMENSIONS")
    else:
        dims = attributes
    if not dims:
        return False
    return len(shape) == 1 and list(dims) == [posixpath.basename(root)]


def _is_array(meta: dict[str, Any]) -> bool:
    """Whether parsed metadata describes an array rather than a group.

    Args:
        meta: Parsed `zarr.json` or `.zarray`.

    Returns:
        True for an array node.
    """
    if meta.get("node_type") == "array":
        return True
    return "shape" in meta and ("chunks" in meta or "chunk_grid" in meta)


def _load(store: Store, key: str) -> dict[str, Any] | None:
    """Fetch and parse a metadata object.

    Args:
        store: A storage handle.
        key: Full object key.

    Returns:
        The parsed object, or `None` if it is missing, unparsable or not a
        JSON object. A listing that raced a delete must not crash a run.
    """
    raw = get_bytes(store, key)
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.warning("%s: not valid JSON (%s)", key, exc)
        return None
    return value if isinstance(value, dict) else None


def _itemsize(dtype: Any) -> int:
    """Bytes per element for a zarr dtype.

    Args:
        dtype: A v3 data type name such as `float32`, or a v2 numpy style
            string such as `<f4`.

    Returns:
        Element size in bytes.

    Raises:
        ValueError: For a dtype this cannot size, such as a structured or
            extension type.
    """
    if isinstance(dtype, str):
        if dtype in _ITEMSIZE:
            return _ITEMSIZE[dtype]
        # numpy-style: "<f4", ">i8", "|b1"
        tail = dtype.lstrip("<>|=")
        if len(tail) >= 2 and tail[1:].isdigit():
            return int(tail[1:])
    raise ValueError(f"cannot size dtype {dtype!r}")


def _prefix(scope: str) -> str:
    """Normalise a scope into a listing prefix.

    Args:
        scope: Scope, with or without slashes.

    Returns:
        The scope with exactly one trailing slash, or empty for the root.
    """
    scope = scope.strip("/")
    return f"{scope}/" if scope else ""


def _relative(key: str, prefix: str) -> str:
    """Strip a prefix from a key.

    Args:
        key: Full object key.
        prefix: Prefix to remove.

    Returns:
        The key relative to the prefix, unchanged if it does not match.
    """
    i = len(prefix)
    return key[i:] if prefix and key.startswith(prefix) else key
