"""The narrow storage seam, over `obstore`.

List, get and conditional put is all blobmap needs. Keeping the surface this
small means [`partition`][blobmap.partition] and [`resolve`][blobmap.resolve]
never import storage at all, and the MinIO integration tests are the same
suite with a different fixture rather than a second suite.

`obstore` gives S3, local disk and an in-memory backend through one API, with
atomic conditional writes: write-then-rename on POSIX, `If-None-Match` and
`If-Match` on S3.

!!! warning "Checked against obstore 0.11"
    `LocalStore` implements create-if-absent but *not* update-if-etag, so a
    local repartition of an existing manifest falls back to overwrite with a
    warning. The `If-Match` path is only really exercised against S3.

Attributes:
    Store: The type of a storage handle. An alias for `obstore`'s
        `ObjectStore`, so the rest of blobmap can annotate a handle without
        importing obstore. Note this is a closed union of obstore's own
        backends: a wrapper around a store, for logging or metrics, would not
        satisfy it. If that becomes wanted, switch these functions to calling
        store *methods* and make this a `Protocol`.
    MISSING: Exception types meaning "no such object". `obstore` raises its
        own `NotFoundError` on some backends and a plain `FileNotFoundError`
        on others, and neither subclasses the other.

Example:
    >>> from obstore.store import MemoryStore
    >>> store = MemoryStore()
    >>> etag = put_bytes(store, "a/b.json", b"hello", expect_absent=True)
    >>> get_bytes(store, "a/b.json")
    b'hello'
    >>> get_bytes(store, "not/here.json") is None
    True
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator, TypeAlias

import obstore as obs
from obstore.exceptions import AlreadyExistsError, NotFoundError
from obstore.store import ObjectStore

log = logging.getLogger(__name__)

Store: TypeAlias = ObjectStore

# obstore raises its own NotFoundError on some backends and a plain
# FileNotFoundError on others; neither subclasses the other.
MISSING: tuple[type[BaseException], ...] = (NotFoundError, FileNotFoundError)


class Conflict(RuntimeError):
    """Someone else wrote this key since we read it.

    Raised when a conditional write fails its precondition. This turns two
    jobs partitioning the same scope into an error you can retry rather than
    a silent last-writer-wins.
    """


@dataclass(frozen=True)
class Entry:
    """One object, as reported by a listing.

    Attributes:
        key: Full object key.
        size: Size in bytes, exact rather than sampled. This is where
            compressed sizes come from, with no need to read any content.
        etag: Entity tag, used for conditional writes. `None` when the
            backend does not report one.
    """

    key: str
    size: int
    etag: str | None = None


def list_all(store: Store, prefix: str = "") -> Iterator[Entry]:
    """Yield every object under a prefix, with its real stored size.

    This is the primary source for both structure and sizes. It reports
    compressed bytes exactly, needs no sampling, and never reads an object
    body. That last point matters: a v3 shard index lives *inside* the
    object, so introspecting it would trigger a restore on exactly the cold
    data we are trying not to touch.

    Args:
        store: A storage handle.
        prefix: Key prefix to list. Empty lists everything.

    Yields:
        One [`Entry`][blobmap.storage.Entry] per object, streamed in batches
        rather than materialised.
    """
    for batch in obs.list(store, prefix=prefix or None):
        for meta in batch:
            yield Entry(str(meta["path"]), int(meta["size"]), _etag(meta.get("e_tag")))


def list_dirs(store: Store, prefix: str = "") -> list[str]:
    """List immediate child prefixes, without recursing.

    Args:
        store: A storage handle.
        prefix: Prefix to list under. Include the trailing slash.

    Returns:
        Common prefixes one level down. One delimited LIST, so this stays
        cheap even above a store with hundreds of thousands of objects.
    """
    result = obs.list_with_delimiter(store, prefix=prefix or None)
    return [str(p) for p in result["common_prefixes"]]


def list_names(store: Store, prefix: str = "") -> set[str]:
    """Basenames of objects sitting directly under a prefix.

    Used to detect a zarr store by looking for `zarr.json` or `.zgroup`
    without listing the whole subtree.

    Args:
        store: A storage handle.
        prefix: Prefix to inspect. Include the trailing slash.

    Returns:
        Basenames, excluding anything in nested prefixes.
    """
    result = obs.list_with_delimiter(store, prefix=prefix or None)
    return {str(o["path"]).rsplit("/", 1)[-1] for o in result["objects"]}


def get_bytes(store: Store, key: str) -> bytes | None:
    """Read an object whole.

    Args:
        store: A storage handle.
        key: Full object key.

    Returns:
        The object body, or `None` if it does not exist. A missing object is
        an expected outcome here, not an error.
    """
    try:
        return bytes(obs.get(store, key).bytes())
    except MISSING:
        return None


def head(store: Store, key: str) -> Entry | None:
    """Fetch metadata for one object without reading it.

    Args:
        store: A storage handle.
        key: Full object key.

    Returns:
        An [`Entry`][blobmap.storage.Entry] with the size and etag, or `None`
        if the object does not exist.
    """
    try:
        meta = obs.head(store, key)
    except MISSING:
        return None
    return Entry(str(meta["path"]), int(meta["size"]), _etag(meta.get("e_tag")))


def put_bytes(
    store: Store,
    key: str,
    body: bytes,
    *,
    etag: str | None = None,
    expect_absent: bool = False,
) -> str | None:
    """Write an object, optionally conditionally.

    Args:
        store: A storage handle.
        key: Full object key.
        body: Bytes to write.
        etag: Require the object to still have this etag, mapping to
            `If-Match`. Ignored when `expect_absent` is set.
        expect_absent: Require the object not to exist, mapping to
            `If-None-Match: *`. Use for a first write.

    Returns:
        The new etag, or `None` if the backend does not report one.

    Raises:
        Conflict: If the precondition fails, meaning someone else wrote the
            key first.

    Note:
        Logs a warning and overwrites when the backend has no update-if-etag
        support, which is the case for `LocalStore` in obstore 0.11.

    Example:
        >>> from obstore.store import MemoryStore
        >>> store = MemoryStore()
        >>> _ = put_bytes(store, "m.json", b"1", expect_absent=True)
        >>> put_bytes(store, "m.json", b"2", expect_absent=True)
        Traceback (most recent call last):
            ...
        blobmap.storage.Conflict: m.json already exists
    """
    mode: Any = "overwrite"
    if expect_absent:
        mode = "create"
    elif etag is not None:
        mode = {"e_tag": etag}

    try:
        result = obs.put(store, key, body, mode=mode)
    except AlreadyExistsError as exc:
        raise Conflict(f"{key} already exists") from exc
    except NotImplementedError:
        # LocalStore has no update-if-etag as of obstore 0.11
        log.warning(
            "%s: no conditional update; concurrent partitioning of "
            "the same scope will last-writer-win",
            type(store).__name__,
        )
        result = obs.put(store, key, body, mode="overwrite")
    except Exception as exc:  # noqa: BLE001 - backend-specific precondition types
        if _is_precondition(exc):
            raise Conflict(f"{key} changed underneath us") from exc
        raise
    return _etag(result.get("e_tag"))


def _etag(value: Any) -> str | None:
    """Normalise an etag, stripping the quotes S3 wraps them in.

    Args:
        value: Raw etag from `obstore`, possibly `None`.

    Returns:
        The unquoted etag, or `None`.
    """
    return str(value).strip('"') if value else None


def _is_precondition(exc: BaseException) -> bool:
    """Whether an exception represents a failed write precondition.

    Backends signal this with different types and status codes, so this
    matches on the text rather than on a class.

    Args:
        exc: The exception raised by the put.

    Returns:
        True if this looks like a 409 or 412.
    """
    text = f"{type(exc).__name__} {exc}"
    return any(
        s in text for s in ("Precondition", "412", "409", "ConditionalRequestConflict")
    )
