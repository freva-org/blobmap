from __future__ import annotations

import json

import obstore as obs
import pytest

from blobmap import (GiB, ManifestStore, NotAdditive, Policy, partition_store,
                     resolve)
from tests.zarrgen import ArraySpec, coordinate, write_store


def specs(**kw) -> list[ArraySpec]:
    return [coordinate("time", 400),
            ArraySpec("tas", (400, 4, 4), (10, 4, 4),
                      dimensions=["time", "y", "x"], **kw)]


@pytest.fixture
def tiny() -> Policy:
    """Thresholds low enough that a fixture-sized store still gets cut."""
    return Policy(t_max_bytes=1280, t_min_bytes=640, t_hot_bytes=32,
                  pow2_floor=1_000_000)


def test_end_to_end(store, memory, tiny):
    write_store(store, "cordex/a.zarr", specs())
    ms = ManifestStore(memory, "blobmap")
    result = partition_store(store, ms, "cordex/a.zarr", policy=tiny)
    assert result.written
    assert ms.read("cordex/a.zarr").manifest == result.manifest
    assert resolve(result.manifest, "tas/c/0/0/0").archivable
    assert resolve(result.manifest, "tas/zarr.json").kind == "hot"


def test_second_run_is_a_no_op(store, memory, tiny):
    write_store(store, "s.zarr", specs())
    ms = ManifestStore(memory)
    first = partition_store(store, ms, "s.zarr", policy=tiny)
    second = partition_store(store, ms, "s.zarr", policy=tiny)
    assert second.diff.is_empty and not second.written
    assert second.manifest.epoch == first.manifest.epoch


def test_dry_run_writes_nothing(store, memory, tiny):
    write_store(store, "s.zarr", specs())
    ms = ManifestStore(memory)
    result = partition_store(store, ms, "s.zarr", policy=tiny, dry_run=True)
    assert not result.written and ms.read("s.zarr") is None


def test_new_variable_bumps_the_epoch_additively(store, memory, tiny):
    write_store(store, "s.zarr", specs())
    ms = ManifestStore(memory)
    first = partition_store(store, ms, "s.zarr", policy=tiny)

    write_store(store, "s.zarr", [ArraySpec("psl", (40, 4, 4), (10, 4, 4))])
    second = partition_store(store, ms, "s.zarr", policy=tiny)

    assert second.diff.is_additive and second.written
    assert second.manifest.epoch == first.manifest.epoch + 1
    assert set(first.manifest.by_id()) < set(second.manifest.by_id())


def test_a_policy_change_alone_does_nothing(store, memory, tiny):
    """Cuts are frozen once made. Pinning makes this structural: a
    repartition cannot move an existing blob, whatever the new policy says."""
    write_store(store, "s.zarr", specs())
    ms = ManifestStore(memory)
    first = partition_store(store, ms, "s.zarr", policy=tiny)
    again = partition_store(store, ms, "s.zarr",
                            policy=Policy(t_max_bytes=64, t_min_bytes=64,
                                          t_hot_bytes=1))
    assert again.diff.is_empty
    assert again.manifest.by_id() == first.manifest.by_id()


def test_force_recomputes_and_says_what_it_orphaned(store, memory, tiny, caplog):
    """The only path that can invalidate a tape copy, so it must be loud."""
    write_store(store, "s.zarr", specs())
    ms = ManifestStore(memory)
    first = partition_store(store, ms, "s.zarr", policy=tiny)
    with caplog.at_level("WARNING"):
        forced = partition_store(store, ms, "s.zarr", force=True,
                                 policy=Policy(t_max_bytes=64, t_min_bytes=64,
                                               t_hot_bytes=1))
    assert forced.written
    assert not forced.diff.is_additive
    assert "orphaned" in caplog.text
    assert forced.manifest.by_id() != first.manifest.by_id()


def test_partitioning_only_reads_metadata(store, memory, tiny):
    """A chunk GET on cold data would trigger a tape restore, so the
    partitioner must never issue one."""
    write_store(store, "s.zarr", specs())
    ms = ManifestStore(memory)
    reads: list[str] = []
    real_get = obs.get

    def spy(target, key, *a, **kw):
        reads.append(key)
        return real_get(target, key, *a, **kw)

    obs.get = spy
    try:
        partition_store(store, ms, "s.zarr", policy=tiny)
    finally:
        obs.get = real_get

    chunk_reads = [k for k in reads
                   if not k.rsplit("/", 1)[-1].startswith((".z", "zarr.json"))
                   and "manifest" not in k]
    assert chunk_reads == []


def test_concurrent_write_is_skipped_not_lost(store, memory, tiny):
    write_store(store, "s.zarr", specs())
    ms = ManifestStore(memory)
    partition_store(store, ms, "s.zarr", policy=tiny)

    stale = ms.read("s.zarr")

    class StaleStore(ManifestStore):
        def read(self, scope):          # pretend we read before someone else wrote
            return stale

    ms.write(stale.manifest.bumped(), etag=stale.etag)   # the other writer
    write_store(store, "s.zarr", [ArraySpec("psl", (40, 4, 4), (10, 4, 4))])
    result = partition_store(store, StaleStore(memory), "s.zarr", policy=tiny)
    assert not result.written
