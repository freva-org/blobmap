from __future__ import annotations

import pytest

from blobmap import (GiB, Array, Blob, Bucket, Manifest, Policy, Trie,
                     partition, resolve)
from tests.test_partition import arr, store_arrays


def test_bucketed_ids_are_arithmetic():
    m = partition("s/", store_arrays())
    width = m.by_id()["b_tas"].bucket.width
    assert resolve(m, "tas/c/0/0/0").blob_id == "b_tas_0"
    assert resolve(m, f"tas/c/{width}/0/0").blob_id == "b_tas_1"
    assert resolve(m, f"tas/c/{5 * width + 3}/0/0").blob_id == "b_tas_5"


def test_id_resolves_for_chunks_that_do_not_exist_yet():
    """The manifest is a rule, not a table: an append resolves with no edit."""
    m = partition("s/", store_arrays())
    assert resolve(m, "tas/c/99999999/0/0").blob_id is not None


def test_unbucketed_resolves_to_zero():
    m = partition("s/", store_arrays())
    assert resolve(m, "pr/c/1/0/0").blob_id.endswith("_0")


@pytest.mark.parametrize("key", [
    "zarr.json", "tas/zarr.json", "tas/.zarray", "x/y/.zattrs",
    "time/c/0", "lat/c/0",
])
def test_metadata_and_coordinates_never_archive(key):
    assert resolve(partition("s/", store_arrays()), key).kind == "hot"


def test_unknown_paths_are_unmanaged_not_an_error():
    m = partition("s/", store_arrays())
    for key in ["xyz/c/0/0/0", "", "totally/unrelated"]:
        assert resolve(m, key).kind == "unmanaged"
    assert not resolve(m, "xyz/c/0").archivable


def test_non_numeric_chunk_segment_is_unmanaged():
    """A prefix match is not enough; the key must actually parse."""
    m = partition("s/", store_arrays())
    assert resolve(m, "tas/c/notanumber/0/0").kind == "unmanaged"
    assert resolve(m, "tas/c").kind == "unmanaged"


@pytest.mark.parametrize("encoding,key,expected", [
    ("v3_slash", "tas/c/{n}/0/0", "b_tas_3"),
    ("v2_slash", "tas/{n}/0/0", "b_tas_3"),
    ("v2_flat", "tas/{n}.0.0", "b_tas_3"),
])
def test_all_key_encodings(encoding, key, expected):
    a = arr("tas", 459 * GiB, key_encoding=encoding)
    m = partition("s/", [a])
    width = m.blobs[0].bucket.width
    assert resolve(m, key.format(n=3 * width)).blob_id == expected


def test_deepest_declaration_wins():
    """A store that outgrows a parent-scope manifest and gets its own must
    override it, with no special case in the lookup."""
    parent = Manifest("cordex/", (Blob("b_parent", ("nukleus",)),), ())
    child = Manifest("cordex/nukleus/eur11.zarr/",
                     (Blob("b_child", ("tas/c",), Bucket(0, 100, "v3_slash")),),
                     ())
    trie = Trie()
    trie.add_all([parent, child])
    assert trie.lookup("cordex/nukleus/eur11.zarr/tas/c/137/0/0").blob_id \
        == "b_child_1"
    assert trie.lookup("cordex/nukleus/other.zarr/x/c/0").blob_id \
        == "b_parent_0"


def test_parent_scope_hot_rules_apply_across_stores():
    parent = Manifest("cordex/", (Blob("b", ("nukleus",)),),
                      ("**/zarr.json", "nukleus/eur11.zarr/time/**"))
    trie = Trie()
    trie.add(parent)
    assert trie.lookup("cordex/nukleus/a.zarr/zarr.json").kind == "hot"
    assert trie.lookup("cordex/nukleus/eur11.zarr/time/c/0").kind == "hot"


def test_trie_is_one_node_per_cut_not_per_object():
    """The whole reason this fits in memory."""
    m = partition("s/", store_arrays())
    trie = Trie()
    trie.add(m)
    assert len(trie) < 20        # one store, ~10^5 objects


def test_epochs_are_tracked_for_reload():
    trie = Trie()
    trie.add(Manifest("a/b/", (), (), epoch=7))
    assert trie.epochs == {"a/b": 7}


def test_resolution_carries_the_blob_and_index():
    m = partition("s/", store_arrays())
    width = m.by_id()["b_tas"].bucket.width
    res = resolve(m, f"tas/c/{2 * width}/0/0")
    assert res.blob.id == "b_tas" and res.bucket_index == 2 and res.archivable


def test_empty_key_and_no_manifests():
    assert Trie().lookup("").kind == "unmanaged"
    assert Trie().lookup("a/b/c").kind == "unmanaged"


def test_negative_chunk_index_is_unmanaged():
    m = partition("s/", store_arrays())
    assert resolve(m, "tas/c/-1/0/0").kind == "unmanaged"
