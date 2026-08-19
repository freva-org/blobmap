"""Pins are the one part of a manifest that encodes intent rather than
structure, so they behave differently from everything else: they survive
repartitioning, they cannot be recomputed, and removing one is deliberate."""

from __future__ import annotations

import pytest

from blobmap import (GiB, Array, ManifestStore, NotPartitioned, Pin, Policy,
                     Trie, partition, partition_store, resolve)
from blobmap import pins
from tests.zarrgen import ArraySpec, coordinate, write_store

SHAPE, CHUNKS = (400, 4, 4), (10, 4, 4)


def specs():
    return [coordinate("time", 400),
            ArraySpec("tas", SHAPE, CHUNKS, dimensions=["time", "y", "x"]),
            ArraySpec("pr", SHAPE, CHUNKS, dimensions=["time", "y", "x"])]


@pytest.fixture
def tiny():
    return Policy(t_max_bytes=1280, t_min_bytes=0, t_hot_bytes=32,
                  pow2_floor=10**9)


@pytest.fixture
def partitioned(store, memory, tiny):
    write_store(store, "s.zarr", specs())
    ms = ManifestStore(memory)
    partition_store(store, ms, "s.zarr", policy=tiny)
    return store, ms


# -- the model ------------------------------------------------------------

def test_covers_matches_whole_segments():
    pin = Pin("zoom_9", "why")
    assert pin.covers("zoom_9") and pin.covers("zoom_9/tas/c/0")
    assert not pin.covers("zoom_90/tas")


def test_empty_prefix_pins_the_whole_scope():
    assert Pin("", "why").covers("anything/at/all")


def test_expiry_is_reported_not_enforced():
    assert Pin("x", "why", until="2020-01-01T00:00:00+00:00").expired()
    assert not Pin("x", "why", until="2999-01-01T00:00:00+00:00").expired()
    assert not Pin("x", "why").expired()      # open-ended never expires


def test_a_pin_needs_a_reason():
    """A pin nobody can explain is one nobody removes."""
    from blobmap import Blob, Manifest
    with pytest.raises(ValueError, match="no reason"):
        Manifest("s", (Blob("b", ("x",)),), (), (Pin("x", "  "),))


def test_duplicate_pins_are_rejected():
    from blobmap import Manifest
    with pytest.raises(ValueError, match="duplicate pin"):
        Manifest("s", (), (), (Pin("x", "a"), Pin("x", "b")))


# -- partitioning ---------------------------------------------------------

def test_pins_survive_repartition():
    """The property that separates a pin from hot_always: hot_always is
    recomputed from structure, a pin is carried over untouched."""
    arrays = [Array("tas", SHAPE, CHUNKS, 4, stored_bytes=400 * GiB),
              Array("pr", SHAPE, CHUNKS, 4, stored_bytes=400 * GiB)]
    first = partition("s", arrays)
    held = first.bumped(pinned=(Pin("pr", "active analysis", "wilfred"),))

    second = partition("s", arrays, previous=held)
    assert second.pinned == held.pinned


def test_pinning_an_existing_blob_reports_it_rather_than_removing_it():
    """A consumer must exclude these before archiving. Resolution alone is not
    enough: the tiering policy runs over blob state, not over keys, so it
    would see a blob whose last_read has simply gone stale."""
    arrays = [Array("tas", SHAPE, CHUNKS, 4, stored_bytes=400 * GiB),
              Array("pr", SHAPE, CHUNKS, 4, stored_bytes=400 * GiB)]
    base = partition("s", arrays)
    held = partition("s", arrays,
                     previous=base.bumped(pinned=(Pin("pr", "in use"),)))

    assert {b.id for b in held.blobs} == {"b_tas", "b_pr"}   # still additive
    assert held.pinned_blob_ids() == {"b_pr"}


def test_a_newly_seen_pinned_array_gets_no_blob():
    """Otherwise the tiering policy could archive it the moment someone
    removes the pin, with no repartition in between."""
    arrays = [Array("tas", SHAPE, CHUNKS, 4, stored_bytes=400 * GiB),
              Array("pr", SHAPE, CHUNKS, 4, stored_bytes=400 * GiB)]
    base = partition("s", arrays)
    assert {b.id for b in base.blobs} == {"b_tas", "b_pr"}

    held = base.bumped(pinned=(Pin("pr", "active analysis"),))
    after = partition("s", arrays, previous=held)
    assert "pr/**" in after.hot_always


def test_pinned_keys_resolve_as_pinned_not_blob():
    arrays = [Array("tas", SHAPE, CHUNKS, 4, stored_bytes=400 * GiB)]
    m = partition("s", arrays)
    held = partition("s", arrays,
                     previous=m.bumped(pinned=(Pin("tas", "in use"),)))
    res = resolve(held, "tas/c/0/0/0")
    assert res.kind == "pinned" and not res.archivable


def test_a_pin_wins_over_an_existing_blob():
    """Pinning something already partitioned must take effect without
    waiting for a repartition."""
    from blobmap import Blob, Manifest
    m = Manifest("s", (Blob("b_tas", ("tas/c",)),), (),
                 (Pin("tas", "in use"),))
    trie = Trie()
    trie.add(m)
    assert trie.lookup("s/tas/c/0").kind == "pinned"


# -- the operations -------------------------------------------------------

def test_add_and_show(partitioned):
    store, ms = partitioned
    pins.add(ms, "s.zarr", "tas", "active ICON analysis", by="wilfred",
             until="2999-01-01T00:00:00+00:00")
    records = pins.show(ms)
    assert len(records) == 1
    assert records[0].scope == "s.zarr"
    assert records[0].pin.reason == "active ICON analysis"
    assert records[0].pin.by == "wilfred"


def test_add_bumps_the_epoch(partitioned):
    store, ms = partitioned
    before = ms.read("s.zarr").manifest.epoch
    pins.add(ms, "s.zarr", "tas", "why")
    assert ms.read("s.zarr").manifest.epoch == before + 1


def test_add_refuses_an_unpartitioned_scope(memory):
    """Almost always a mistyped scope."""
    with pytest.raises(NotPartitioned, match="Partition it first"):
        pins.add(ManifestStore(memory), "nope.zarr", "x", "why")


def test_add_refuses_a_duplicate(partitioned):
    store, ms = partitioned
    pins.add(ms, "s.zarr", "tas", "first")
    with pytest.raises(ValueError, match="already pinned"):
        pins.add(ms, "s.zarr", "tas", "second")


def test_remove(partitioned):
    store, ms = partitioned
    pins.add(ms, "s.zarr", "tas", "why")
    pins.remove(ms, "s.zarr", "tas")
    assert pins.show(ms) == []


def test_remove_refuses_what_is_not_pinned(partitioned):
    store, ms = partitioned
    with pytest.raises(LookupError, match="not pinned"):
        pins.remove(ms, "s.zarr", "tas")


def test_show_puts_the_worst_first(partitioned):
    """Expired, then open-ended, then dated. Both of the first two are how a
    hot pool quietly fills."""
    store, ms = partitioned
    pins.add(ms, "s.zarr", "a", "dated", until="2999-01-01T00:00:00+00:00")
    pins.add(ms, "s.zarr", "b", "open ended")
    pins.add(ms, "s.zarr", "c", "stale", until="2020-01-01T00:00:00+00:00")

    order = [r.pin.prefix for r in pins.show(ms)]
    assert order == ["c", "b", "a"]


def test_show_expired_only(partitioned):
    store, ms = partitioned
    pins.add(ms, "s.zarr", "a", "fine", until="2999-01-01T00:00:00+00:00")
    pins.add(ms, "s.zarr", "b", "stale", until="2020-01-01T00:00:00+00:00")
    assert [r.pin.prefix for r in pins.show(ms, expired_only=True)] == ["b"]


def test_show_for_one_scope(partitioned, memory, store, tiny):
    _, ms = partitioned
    write_store(store, "other.zarr", specs())
    partition_store(store, ms, "other.zarr", policy=tiny)
    pins.add(ms, "s.zarr", "a", "why")
    pins.add(ms, "other.zarr", "b", "why")
    assert [r.pin.prefix for r in pins.show(ms, "s.zarr")] == ["a"]
    assert pins.show(ms, "missing.zarr") == []


def test_pin_survives_a_real_repartition(partitioned, tiny):
    """End to end: pin, add a variable, repartition, pin is still there and
    still keeps its prefix out of the blobs."""
    store, ms = partitioned
    pins.add(ms, "s.zarr", "pr", "active analysis")

    write_store(store, "s.zarr", [ArraySpec("psl", SHAPE, CHUNKS)])
    result = partition_store(store, ms, "s.zarr", policy=tiny)

    assert [p.prefix for p in result.manifest.pinned] == ["pr"]
    # the existing blob definition survives -- removing it would orphan any
    # tape copy held against that id -- but it is now reported as pinned
    assert "b_pr" in result.manifest.pinned_blob_ids()
    assert resolve(result.manifest, "pr/c/0/0/0").kind == "pinned"
