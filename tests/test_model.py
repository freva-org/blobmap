from __future__ import annotations

import math

import pytest

from blobmap import GiB, Array, Blob, Bucket, Manifest, Policy


def test_object_shape_prefers_shard():
    plain = Array("tas", (100, 4), (10, 4), 4)
    sharded = Array("tas", (100, 4), (10, 4), 4, shards=(50, 4))
    assert plain.object_shape == (10, 4)
    assert sharded.object_shape == (50, 4)
    assert sharded.nobjects == 2
    assert sharded.uncompressed_object_bytes == 50 * 4 * 4


def test_nobjects_is_the_full_grid_not_what_was_written():
    """A sparsely written array is partitioned for the shape it will have."""
    a = Array("tas", (100, 4), (10, 4), 4, nobjects_seen=3)
    assert a.nobjects == 10
    assert a.avg_object_bytes == a.total_bytes // 3


def test_total_bytes_falls_back_to_uncompressed():
    a = Array("tas", (100,), (10,), 4)
    assert a.total_bytes == 10 * 10 * 4
    assert Array("tas", (100,), (10,), 4, stored_bytes=7).total_bytes == 7


@pytest.mark.parametrize("encoding,expected", [
    ("v3_slash", "tas/c"), ("v2_slash", "tas"), ("v2_flat", "tas"),
])
def test_chunk_prefix(encoding, expected):
    assert Array("tas", (1,), (1,), 4, key_encoding=encoding).chunk_prefix \
        == expected


def test_chunk_prefix_rejects_nonsense():
    with pytest.raises(ValueError):
        Array("tas", (1,), (1,), 4, key_encoding="v9").chunk_prefix


def test_root_array_chunk_prefix():
    assert Array("", (1,), (1,), 4).chunk_prefix == "c"


def test_policy_ignores_unknown_json_fields():
    p = Policy.from_json({"t_max_bytes": 5, "future_field": 9})
    assert p.t_max_bytes == 5


def test_manifest_roundtrip():
    m = Manifest("s/", (Blob("b_x", ("x/c",), Bucket(0, 4, "v3_slash")),),
                 ("**/zarr.json",), generated_at="2026-01-01T00:00:00+00:00")
    assert Manifest.loads(m.dumps()) == m


def test_manifest_rejects_other_schema_versions():
    payload = Manifest("s/", (), ()).to_json()
    payload["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        Manifest.from_json(payload)


def test_bumped_increments_epoch():
    m = Manifest("s/", (), (), epoch=4)
    assert m.bumped().epoch == 5


@pytest.mark.parametrize("blobs,message", [
    ((Blob("b", ("a",)), Blob("b", ("c",))), "duplicate"),
    ((Blob("b!", ("a",)),), "not"),
    ((Blob("b", ()),), "no prefixes"),
    ((Blob("b", ("a", "b"), Bucket(0, 4, "v3_slash")),), "exactly one"),
    ((Blob("b", ("a",), Bucket(0, 0, "v3_slash")),), "width"),
    ((Blob("b", ("a",), Bucket(-1, 4, "v3_slash")),), "index"),
    ((Blob("b", ("a",), Bucket(0, 4, "v9")),), "key_encoding"),
])
def test_validate_rejects(blobs, message):
    with pytest.raises(ValueError, match=message):
        Manifest("s/", blobs, ()).validate()
