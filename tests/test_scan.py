from __future__ import annotations

import obstore as obs

from blobmap import ManifestStore, scan
from tests.test_manifests import make
from tests.zarrgen import ArraySpec, write_store


def spec() -> list[ArraySpec]:
    return [ArraySpec("tas", (40, 4, 4), (10, 4, 4))]


def test_finds_stores(store, memory):
    write_store(store, "cordex/a.zarr", spec())
    write_store(store, "cordex/b.zarr", spec())
    found = list(scan(store, "", ManifestStore(memory)))
    assert {c.scope for c in found} == {"cordex/a.zarr", "cordex/b.zarr"}
    assert all(c.fmt == "v3" and not c.has_manifest for c in found)


def test_marks_known_stores(store, memory):
    write_store(store, "cordex/a.zarr", spec())
    ms = ManifestStore(memory)
    ms.write(make("cordex/a.zarr"), expect_absent=True)
    assert [c.has_manifest for c in scan(store, "", ms)] == [True]


def test_does_not_descend_into_a_store(store, memory):
    """A datatree may put a whole bucket in one store; walking in would mean
    listing every chunk key looking for more."""
    write_store(store, "top", spec())
    write_store(store, "top/nested.zarr", spec())
    assert [c.scope for c in scan(store, "", ManifestStore(memory))] == ["top"]


def test_ignores_non_zarr_prefixes(store, memory):
    obs.put(store, "junk/readme.txt", b"hi")
    write_store(store, "real.zarr", spec())
    assert [c.scope for c in scan(store, "", ManifestStore(memory))] \
        == ["real.zarr"]


def test_max_depth(store, memory):
    write_store(store, "a/b/c/d/e/f/g.zarr", spec())
    assert list(scan(store, "", ManifestStore(memory), max_depth=2)) == []
    assert len(list(scan(store, "", ManifestStore(memory), max_depth=8))) == 1


def test_scan_from_a_subprefix(store, memory):
    write_store(store, "cordex/a.zarr", spec())
    write_store(store, "era5/b.zarr", spec())
    assert [c.scope for c in scan(store, "era5", ManifestStore(memory))] \
        == ["era5/b.zarr"]
