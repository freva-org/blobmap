"""The poller against a real sqlite table standing in for MinIO's Postgres
notification target -- same DB-API surface, so no mock is needed."""

from __future__ import annotations

import sqlite3

import pytest

from blobmap import EventPoller, PollConfig, store_of


@pytest.fixture
def events():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE minio_events (id INTEGER PRIMARY KEY, "
                 "key TEXT, event_name TEXT, principal TEXT)")
    return conn


@pytest.fixture
def roots():
    """Stand-in for find_store_root: everything under x.zarr belongs to it."""
    def resolve(prefix: str) -> str:
        parts = prefix.split("/")
        for i, part in enumerate(parts):
            if part.endswith(".zarr"):
                return "/".join(parts[:i + 1])
        return prefix
    return resolve


def emit(conn, key, event_name="s3:ObjectCreated:Put", principal="user"):
    conn.execute("INSERT INTO minio_events(key, event_name, principal) "
                 "VALUES (?, ?, ?)", (key, event_name, principal))


def poller(conn, root_of=None, **kw):
    return EventPoller(conn.cursor(), config=PollConfig(placeholder="?", **kw),
                       root_of=root_of)


@pytest.mark.parametrize("key,expected", [
    ("a/b.zarr/tas/zarr.json", "a/b.zarr/tas"),
    ("a/b.zarr/.zgroup", "a/b.zarr"),
    ("a/b.zarr/tas/c/0/0", None),          # a chunk write is blobtier's problem
    ("a/b.zarr/tas/0.0.0", None),
    ("zarr.json", None),                   # no parent
])
def test_store_of(key, expected):
    assert store_of(key) == expected


def test_only_metadata_creates_become_pending(events):
    emit(events, "s/a.zarr/zarr.json")
    for i in range(100):
        emit(events, f"s/a.zarr/tas/c/{i}/0")
    p = poller(events)
    assert p.poll_once(now=0) == 101
    assert set(p.pending()) == {"s/a.zarr"}


def test_reads_are_ignored(events):
    emit(events, "s/a.zarr/zarr.json", event_name="s3:ObjectAccessed:Get")
    p = poller(events)
    p.poll_once(now=0)
    assert p.pending() == {}


def test_own_principal_is_filtered(events):
    """Without this, the partitioner's own metadata reads would re-enqueue the
    store it just partitioned."""
    emit(events, "s/a.zarr/zarr.json", principal="blobmap-svc")
    emit(events, "s/b.zarr/zarr.json", principal="user")
    p = poller(events, ignore_principals=("blobmap-svc",))
    p.poll_once(now=0)
    assert set(p.pending()) == {"s/b.zarr"}


def test_debounce_holds_until_quiet(events):
    emit(events, "s/a.zarr/zarr.json")
    p = poller(events, quiet_seconds=100)
    p.poll_once(now=1000)
    assert p.due(now=1050) == []
    assert p.due(now=1101) == ["s/a.zarr"]


def test_continued_writing_pushes_the_deadline_out(events, roots):
    """Never partition a store that is still being written: the cut would
    land in the wrong place and the trailing bucket would not be sealed."""
    emit(events, "s/a.zarr/zarr.json")
    p = poller(events, root_of=roots, quiet_seconds=100)
    p.poll_once(now=1000)
    emit(events, "s/a.zarr/pr/zarr.json")   # a new variable in the same store
    p.poll_once(now=1090)
    assert p.due(now=1150) == []
    assert p.due(now=1200) == ["s/a.zarr"]


def test_cursor_advances_and_survives(events):
    emit(events, "s/a.zarr/zarr.json")
    p = poller(events)
    p.poll_once(now=0)
    position = p.position
    assert position > 0
    assert p.poll_once(now=0) == 0
    assert p.position == position


def test_drain_handles_multiple_batches(events):
    for i in range(25):
        emit(events, f"s/{i}.zarr/zarr.json")
    p = poller(events, batch=10)
    assert p.drain(now=0) == 25
    assert len(p.pending()) == 25


def test_step_dispatches_and_clears(events):
    emit(events, "s/a.zarr/zarr.json")
    p = poller(events, quiet_seconds=0)
    handled = []
    assert p.step(handled.append, now=10) == ["s/a.zarr"]
    assert handled == ["s/a.zarr"]
    assert p.pending() == {}


def test_a_failing_store_is_retried_not_dropped(events, caplog):
    emit(events, "s/a.zarr/zarr.json")
    p = poller(events, quiet_seconds=0)

    def boom(scope):
        raise RuntimeError("versity is down")

    with caplog.at_level("ERROR"):
        assert p.step(boom, now=10) == []
    assert "will retry" in caplog.text
    assert set(p.pending()) == {"s/a.zarr"}
