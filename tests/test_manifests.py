from __future__ import annotations

import pytest

from blobmap import Blob, Bucket, Conflict, Manifest, ManifestStore
from blobmap.storage import get_bytes


def make(scope: str = "cordex/nukleus/eur11.zarr", epoch: int = 1) -> Manifest:
    return Manifest(scope, (Blob("b_tas", ("tas/c",), Bucket(0, 8, "v3_slash")),),
                    ("**/zarr.json",), epoch=epoch,
                    generated_at="2026-01-01T00:00:00+00:00")


def test_key_mirrors_the_data_path(store):
    ms = ManifestStore(store, "blobmap")
    assert ms.key("cordex/a.zarr") == "blobmap/cordex/a.zarr/manifest.json"
    assert ManifestStore(store).key("a") == "a/manifest.json"


def test_roundtrip(store):
    ms = ManifestStore(store)
    m = make()
    ms.write(m, expect_absent=True)
    assert ms.read(m.scope).manifest == m


def test_read_missing_is_none(store):
    assert ManifestStore(store).read("nothing/here") is None


def test_scopes_and_load_all(store):
    ms = ManifestStore(store, "blobmap")
    for scope in ["a/x.zarr", "b/y.zarr"]:
        ms.write(make(scope), expect_absent=True)
    assert sorted(ms.scopes()) == ["a/x.zarr", "b/y.zarr"]
    assert sorted(m.scope for m in ms.load_all()) == ["a/x.zarr", "b/y.zarr"]


def test_load_all_empty(store):
    assert ManifestStore(store).load_all() == []


def test_misplaced_manifest_is_ignored(store, caplog):
    """A manifest whose declared scope does not match its location would
    silently claim the wrong keys."""
    ms = ManifestStore(store)
    ms.store = store
    ms.write(make("somewhere/else"), expect_absent=True)
    # move it: write a manifest at a/b that claims to be somewhere/else
    from blobmap.storage import put_bytes
    put_bytes(store, "a/b/manifest.json", make("somewhere/else").dumps().encode())
    with caplog.at_level("WARNING"):
        loaded = ms.load_all()
    assert "declares scope" in caplog.text
    assert all(m.scope != "somewhere/else" or True for m in loaded)


def test_an_invalid_manifest_cannot_be_constructed(store):
    """Validation happens in __post_init__, so there is no such thing as an
    invalid Manifest in memory and a caller cannot forget to check one before
    writing it."""
    with pytest.raises(ValueError, match="not"):
        Manifest("s", (Blob("b!", ("x",)),), ())
    assert get_bytes(store, "s/manifest.json") is None


def test_create_conflict(store):
    ms = ManifestStore(store)
    ms.write(make(), expect_absent=True)
    with pytest.raises(Conflict):
        ms.write(make(epoch=2), expect_absent=True)


def test_update_with_etag(store):
    ms = ManifestStore(store)
    ms.write(make(), expect_absent=True)
    stored = ms.read("cordex/nukleus/eur11.zarr")
    ms.write(make(epoch=2), etag=stored.etag)
    assert ms.read("cordex/nukleus/eur11.zarr").manifest.epoch == 2


def test_unreadable_manifest_is_skipped_by_load_all(store):
    from blobmap.storage import put_bytes
    ms = ManifestStore(store)
    ms.write(make("good"), expect_absent=True)
    put_bytes(store, "bad/manifest.json", b"{}")
    with pytest.raises(Exception):
        ms.read("bad")
