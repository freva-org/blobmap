from __future__ import annotations

import math

import pytest

from blobmap import (GiB, Array, Blob, Bucket, Manifest, Policy, bucket_width,
                     diff, partition, resolve)

NTIME, NLAT, NLON = 1_314_000, 412, 424
SHAPE = (NTIME, NLAT, NLON)
CHUNKS = (128, NLAT, NLON)


def arr(path: str, stored: int, **kw) -> Array:
    return Array(path, SHAPE, CHUNKS, 4, stored_bytes=stored, **kw)


def store_arrays() -> list[Array]:
    return [
        Array("time", (NTIME,), (NTIME,), 8, is_coordinate=True),
        Array("lat", (NLAT,), (NLAT,), 8, is_coordinate=True),
        arr("tas", 459 * GiB),
        arr("hus", 459 * GiB, shards=(1280, NLAT, NLON)),
        arr("pr", 20 * GiB),
        arr("hurs", 4 * GiB),
    ]


# -- width -----------------------------------------------------------------

def test_shards_change_the_width():
    """Same bytes, 10x fewer objects: the width must fall with it. Using the
    inner chunk instead would give both arrays the same width, i.e. a blob 10x
    the target."""
    policy = Policy()
    plain, sharded = arr("tas", 459 * GiB), arr("tas", 459 * GiB,
                                                shards=(1280, NLAT, NLON))
    assert sharded.nobjects == math.ceil(plain.nobjects / 10)
    assert bucket_width(sharded, policy) < bucket_width(plain, policy)


def test_width_clamp_bounds_the_worst_case():
    """A 500x-compressing array must not get a width that becomes 25 TB if the
    codec ever changes."""
    policy = Policy()
    tiny = arr("pr", 2 * GiB)
    assert bucket_width(tiny, policy) * tiny.uncompressed_object_bytes \
        <= policy.width_clamp * policy.t_max_bytes


def test_width_is_a_power_of_two_when_large():
    width = bucket_width(arr("tas", 459 * GiB), Policy())
    assert width & (width - 1) == 0


def test_small_widths_skip_pow2_rounding():
    """Below pow2_floor, rounding down would throw away half the target."""
    policy = Policy(pow2_floor=64)
    chunky = Array("tas", SHAPE, (128, NLAT, NLON), 4,
                   shards=(12800, NLAT, NLON), stored_bytes=4590 * GiB)
    width = bucket_width(chunky, policy)
    assert width < 64 and width & (width - 1) != 0 or width in (1, 2, 4, 8)


def test_oversized_single_object_warns(caplog):
    huge = Array("tas", (10, 1), (1, 1), 8, stored_bytes=2000 * GiB)
    with caplog.at_level("WARNING"):
        assert bucket_width(huge, Policy()) == 1
    assert "cannot cut finer" in caplog.text


# -- shape -----------------------------------------------------------------

def test_every_blob_has_the_same_three_keys():
    for blob in partition("s/", store_arrays()).blobs:
        assert set(blob.to_json()) == {"id", "prefixes", "bucket"}


def test_small_store_is_one_blob():
    m = partition("s/", [arr("tas", 20 * GiB), arr("pr", 20 * GiB)])
    assert len(m.blobs) == 1
    assert m.blobs[0].prefixes == ("",)
    assert m.blobs[0].bucket is None


def test_coordinates_and_small_arrays_are_pinned_hot():
    m = partition("s/", store_arrays())
    assert "time/**" in m.hot_always and "lat/**" in m.hot_always
    assert "**/zarr.json" in m.hot_always


def test_coalescing_groups_small_arrays():
    m = partition("s/", store_arrays())
    grouped = [b for b in m.blobs if len(b.prefixes) > 1]
    assert grouped and set(grouped[0].prefixes) == {"pr/c", "hurs/c"}


def test_tail_below_t_min_folds_into_previous_blob():
    arrays = [arr("a", 12 * GiB), arr("b", 1500 * GiB), arr("c", 2 * GiB)]
    m = partition("s/", arrays, policy=Policy(t_hot_bytes=1))
    unbucketed = [b for b in m.blobs if b.bucket is None]
    assert unbucketed[0].prefixes == ("a/c", "c/c")


def test_provenance_is_not_read_back():
    """Measured numbers are recorded for debugging but must not affect
    resolution."""
    m = partition("s/", store_arrays())
    stripped = Manifest.from_json({**m.to_json(), "provenance": {}})
    assert stripped.blobs == m.blobs


# -- incremental -----------------------------------------------------------

def test_append_needs_no_repartition():
    before = partition("s/", store_arrays())
    grown = [Array("tas", (NTIME * 2, NLAT, NLON), CHUNKS, 4,
                   stored_bytes=918 * GiB)] + store_arrays()[3:]
    after = partition("s/", store_arrays()[:2] + grown, previous=before)
    assert diff(before, after).is_empty


def test_new_variable_is_additive():
    before = partition("s/", store_arrays())
    after = partition("s/", store_arrays() + [arr("psl", 400 * GiB)],
                      previous=before)
    changes = diff(before, after)
    assert changes.is_additive
    assert [b.id for b in changes.added] == ["b_psl"]
    assert before.by_id().items() <= after.by_id().items()


def test_without_pinning_a_repartition_renumbers():
    """The failure pinning exists to prevent."""
    before = partition("s/", store_arrays())
    naive = partition("s/", store_arrays() + [arr("psl", 4 * GiB)])
    assert not diff(before, naive).is_additive


def test_pinned_blob_survives_even_if_policy_changes():
    before = partition("s/", store_arrays())
    after = partition("s/", store_arrays(), previous=before,
                      policy=Policy(t_max_bytes=1 * GiB))
    assert diff(before, after).is_additive


def test_previous_policy_is_inherited():
    before = partition("s/", store_arrays(), policy=Policy(t_max_bytes=7 * GiB))
    after = partition("s/", store_arrays(), previous=before)
    assert after.policy.t_max_bytes == 7 * GiB


def test_array_that_grew_past_t_hot_stops_being_pinned():
    before = partition("s/", [arr("tas", 400 * GiB), arr("pr", 100)])
    assert "pr/**" in before.hot_always
    after = partition("s/", [arr("tas", 400 * GiB), arr("pr", 400 * GiB)],
                      previous=before)
    assert "pr/**" not in after.hot_always
    assert any("pr/c" in b.prefixes for b in after.blobs)


def test_diff_describe_covers_all_three_kinds():
    old = Manifest("s/", (Blob("a", ("a",)), Blob("b", ("b",))), ())
    new = Manifest("s/", (Blob("a", ("a2",)), Blob("c", ("c",))), ())
    text = diff(old, new).describe()
    assert "+ c" in text and "- b" in text and "~ a" in text
    assert diff(None, new).added == new.blobs
    assert "no change" in diff(new, new).describe()


def test_colliding_idents_get_distinct_blob_ids():
    """Different paths can sanitise to the same identifier; blob ids are a
    join key, so a collision would silently merge two blobs' state."""
    arrays = [arr("a-b", 400 * GiB), arr("a_b", 400 * GiB)]
    m = partition("s/", arrays)
    assert sorted(b.id for b in m.blobs) == ["b_a_b", "b_a_b_2"]
    m.validate()
