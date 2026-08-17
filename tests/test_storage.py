"""Real backends, no mocks: MemoryStore and LocalStore implement the same API
as S3Store, so these become the MinIO tests by adding one fixture param."""

from __future__ import annotations

import pytest

import obstore as obs
from blobmap.storage import (Conflict, Entry, get_bytes, head, list_all,
                             list_dirs, list_names, put_bytes)


def test_put_and_get(store):
    etag = put_bytes(store, "a/b.json", b"hello")
    assert get_bytes(store, "a/b.json") == b"hello"
    assert head(store, "a/b.json").size == 5


def test_missing_is_none_not_an_exception(store):
    assert get_bytes(store, "nope.json") is None
    assert head(store, "nope.json") is None


def test_list_all_reports_real_sizes(store):
    obs.put(store, "d/a", b"x" * 10)
    obs.put(store, "d/sub/b", b"x" * 3)
    entries = {e.key: e.size for e in list_all(store, "d/")}
    assert entries == {"d/a": 10, "d/sub/b": 3}


def test_list_dirs_is_shallow(store):
    for key in ["d/a.zarr/zarr.json", "d/b.zarr/zarr.json", "d/deep/x/y"]:
        obs.put(store, key, b"x")
    assert {p.strip("/").split("/")[-1] for p in list_dirs(store, "d/")} \
        == {"a.zarr", "b.zarr", "deep"}


def test_list_names_gives_basenames(store):
    obs.put(store, "d/zarr.json", b"x")
    obs.put(store, "d/sub/zarr.json", b"x")
    assert list_names(store, "d/") == {"zarr.json"}


def test_create_twice_conflicts(store):
    put_bytes(store, "m.json", b"1", expect_absent=True)
    with pytest.raises(Conflict):
        put_bytes(store, "m.json", b"2", expect_absent=True)
    assert get_bytes(store, "m.json") == b"1"


def test_update_with_correct_etag_succeeds(store):
    etag = put_bytes(store, "m.json", b"1", expect_absent=True)
    put_bytes(store, "m.json", b"2", etag=etag)
    assert get_bytes(store, "m.json") == b"2"


def test_update_with_stale_etag(store, caplog):
    """S3 rejects it. LocalStore has no update-if-etag in obstore 0.11, so it
    warns and overwrites -- worth knowing rather than silently assuming
    protection."""
    put_bytes(store, "m.json", b"1", expect_absent=True)
    with caplog.at_level("WARNING"):
        try:
            put_bytes(store, "m.json", b"2", etag="definitely-stale")
        except Conflict:
            return
    assert "last-writer-win" in caplog.text
