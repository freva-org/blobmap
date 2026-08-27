"""The report exists to make invisible problems visible, so the tests are
mostly about whether it actually surfaces them."""

from __future__ import annotations

import obstore as obs
import pytest

from blobmap import ManifestStore, Policy, partition_store, pins
from blobmap.report import Report, human, render, report
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
def counted(store, memory, tiny):
    write_store(store, "s.zarr", specs())
    ms = ManifestStore(memory)
    partition_store(store, ms, "s.zarr", policy=tiny)
    return store, ms


def test_everything_is_accounted_for(counted):
    """Every object lands in exactly one category, so the four sum to the
    listing. If they did not, the report would understate the problem."""
    from blobmap.storage import list_all
    store, ms = counted
    listed = list(list_all(store, ""))

    r = report(store, ms)
    assert r.objects == len(listed)
    assert r.total_bytes == sum(e.size for e in listed)


def test_metadata_and_coordinates_count_as_hot(counted):
    store, ms = counted
    r = report(store, ms)
    assert r.hot_bytes > 0
    assert r.archivable_bytes > 0


def test_an_unpartitioned_store_shows_as_unmanaged(counted, tiny):
    """The category that grows silently: someone adds a store and nothing
    ever archives it."""
    store, ms = counted
    write_store(store, "forgotten.zarr", specs())

    r = report(store, ms)
    assert r.unmanaged_bytes > 0
    assert "unmanaged" in render([r])
    assert "Run scan" in render([r])


def test_a_fully_partitioned_bucket_reports_no_warning(counted):
    store, ms = counted
    text = render([report(store, ms)])
    assert "Run scan" not in text


def test_pinned_bytes_are_separated_from_hot(counted):
    store, ms = counted
    before = report(store, ms)
    pins.add(ms, "s.zarr", "tas", "active analysis")
    after = report(store, ms)

    assert after.pinned_bytes > 0
    assert after.archivable_bytes < before.archivable_bytes
    assert after.total_bytes == before.total_bytes


def test_expired_pins_are_called_out(counted):
    store, ms = counted
    pins.add(ms, "s.zarr", "tas", "stale", until="2020-01-01T00:00:00+00:00")
    text = render([report(store, ms)])
    assert "past their review date" in text


def test_held_fraction():
    r = Report("s", archivable_bytes=750, hot_bytes=50, pinned_bytes=100,
               unmanaged_bytes=100)
    assert r.total_bytes == 1000
    assert r.held_bytes == 250
    assert r.held_fraction == 0.25


def test_render_puts_the_worst_first():
    reports = [Report("clean", archivable_bytes=1000),
               Report("leaky", archivable_bytes=1000, unmanaged_bytes=500)]
    text = render(reports)
    assert text.index("leaky") < text.index("clean")


def test_render_handles_nothing():
    assert render([]) == "nothing to report"


def test_empty_scope_does_not_divide_by_zero():
    assert Report("s").held_fraction == 0.0
    assert "s" in render([Report("s")])


@pytest.mark.parametrize("n,expected", [(1536, "1.5 KiB"), (0, "0.0 B")])
def test_human(n, expected):
    assert human(n) == expected
