"""Poll-driven discovery from MinIO's Postgres notification target.

Polling a table is a fine event source here. Latency does not matter: the one
thing that needed it -- restore -- is a synchronous path in blobtier, not an
event. A cursor also gives replay for free, and lets blobtier hold its own
position in the same table with no coordination between the two consumers.

MinIO must be configured with `format=access` (an append-only log, not one
upserted row per key) and with `queue_dir` set, so a database blip spools to
local disk instead of silently dropping events.

Two filters do the work:

  * only ObjectCreated on a *metadata* object is interesting here. A new
    variable, group or store always writes one. Chunk writes are blobtier's
    business, and filtering them here is what stops a conversion run's 300k
    events from becoming 300k queue entries.
  * debounce. Never partition a store that is still being written: sizes are
    half complete, the cut lands in the wrong place, and the trailing bucket
    is not sealed.
"""

from __future__ import annotations

import logging
import posixpath
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from ..model import METADATA_BASENAMES

log = logging.getLogger(__name__)


class Cursor(Protocol):
    """The DB-API subset the poller needs.

    Supplied by the caller, so blobmap carries no database driver dependency
    of its own and can be tested against sqlite.
    """

    def execute(self, sql: str, params: Sequence[Any] = ...) -> Any: ...
    def fetchall(self) -> list[tuple[Any, ...]]: ...


@dataclass(frozen=True)
class PollConfig:
    """How to read the notification table and when to act on it.

    Attributes:
        table: Table MinIO writes notifications to.
        consumer: Name this poller stores its cursor under. Give blobtier a
            different one and the two consume the same table independently.
        batch: Rows per query.
        quiet_seconds: How long a store must be silent before it is safe to
            partition. Set it above your normal gap between writes, or a
            conversion run gets partitioned halfway through.
        poll_seconds: Sleep between polls in
            [`run`][blobmap.discover.events.EventPoller.run].
        ignore_principals: Service accounts whose activity does not count.
            Without this, the partitioner's own metadata reads re-enqueue
            the store it just finished.
        placeholder: Parameter style. `%s` for psycopg, `?` for sqlite.
    """

    table: str = "minio_events"
    consumer: str = "blobmap"
    batch: int = 5000
    quiet_seconds: float = 1800
    poll_seconds: float = 30
    ignore_principals: tuple[str, ...] = ()
    placeholder: str = "%s"


class EventPoller:
    """Turns a stream of object keys into debounced store scopes.

    State is SQLite: the poll cursor and the pending set. Local, single
    writer, survives a restart. That is what SQLite is actually good at,
    unlike holding manifests, where one file would need a lock across writers
    and rewrite everything to record one store.

    Args:
        cursor: A DB-API cursor on the notification database.
        state_path: SQLite file for the cursor and pending set. The default
            keeps it in memory, which loses the position on restart.
        config: Polling and debounce settings.
        root_of: Maps the node a metadata object sits in to its store root,
            normally [`find_store_root`][blobmap.hierarchy.find_store_root]
            bound to a store. Without it, two variables in one store debounce
            independently and the partitioner is handed a sub-array as a
            scope.

    Example:
        ```python
        import psycopg
        from blobmap import EventPoller, PollConfig, find_store_root

        conn = psycopg.connect("postgresql://...")
        poller = EventPoller(
            conn.cursor(),
            state_path="/var/lib/blobmap/state.sqlite",
            config=PollConfig(ignore_principals=("blobmap-svc",)),
            root_of=lambda prefix: find_store_root(data, prefix),
        )
        poller.run(lambda scope: partition_store(data, manifests, scope))
        ```
    """

    def __init__(self, cursor: Cursor, state_path: str = ":memory:",
                 config: PollConfig | None = None,
                 root_of: Callable[[str], str | None] | None = None) -> None:
        self.cursor = cursor
        self.config = config or PollConfig()
        # maps the node a metadata object sits in to its store root. Without
        # it, two variables in one store debounce independently and the
        # partitioner is handed a sub-array as a scope.
        self.root_of = root_of or (lambda prefix: prefix)
        self.state = sqlite3.connect(state_path, isolation_level=None)
        self._migrate()

    def _migrate(self) -> None:
        """Create the state tables if they do not exist."""
        self.state.executescript("""
            CREATE TABLE IF NOT EXISTS cursor (
                consumer  TEXT PRIMARY KEY,
                position  INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS pending (
                scope     TEXT PRIMARY KEY,
                last_seen REAL NOT NULL);
        """)

    def close(self) -> None:
        """Close the SQLite state connection."""
        self.state.close()

    # -- polling ----------------------------------------------------------

    def poll_once(self, now: float | None = None) -> int:
        """Fetch one batch and mark the affected scopes pending.

        Args:
            now: Timestamp to record. Defaults to the wall clock; pass a
                value in tests.

        Returns:
            Rows read. Equal to `config.batch` when more may remain.
        """
        now = time.time() if now is None else now
        position = self.position
        ph = self.config.placeholder
        self.cursor.execute(
            f"SELECT id, key, event_name, principal FROM {self.config.table} "
            f"WHERE id > {ph} ORDER BY id LIMIT {ph}",
            (position, self.config.batch),
        )
        rows = self.cursor.fetchall()
        for row_id, key, event_name, principal in rows:
            position = max(position, int(row_id))
            if principal in self.config.ignore_principals:
                # our own partitioner, tiering, verification and backup reads.
                # without this, archiving a blob marks it freshly accessed and
                # it immediately looks hot again
                continue
            if not str(event_name).startswith("s3:ObjectCreated"):
                continue
            node = store_of(str(key))
            if node is None:
                continue
            scope = self.root_of(node)
            if scope is not None:
                self.touch(scope, now)
        self._set_position(position)
        return len(rows)

    def drain(self, now: float | None = None) -> int:
        """Poll until the table is caught up.

        Args:
            now: Timestamp to record against touched scopes.

        Returns:
            Total rows read across all batches.
        """
        total = 0
        while True:
            seen = self.poll_once(now)
            total += seen
            if seen < self.config.batch:
                return total

    def due(self, now: float | None = None) -> list[str]:
        """Scopes that have been quiet long enough to partition safely.

        Args:
            now: Reference time.

        Returns:
            Scope prefixes, sorted. A store still being written is held back:
            its sizes are half complete, so the cut would land in the wrong
            place.
        """
        now = time.time() if now is None else now
        cutoff = now - self.config.quiet_seconds
        rows = self.state.execute(
            "SELECT scope FROM pending WHERE last_seen <= ? ORDER BY scope",
            (cutoff,)).fetchall()
        return [str(r[0]) for r in rows]

    def pending(self) -> dict[str, float]:
        """Everything waiting for quiet.

        Returns:
            Mapping of scope to the time it was last seen being written.
        """
        return {str(s): float(t) for s, t in
                self.state.execute("SELECT scope, last_seen FROM pending")}

    def touch(self, scope: str, now: float | None = None) -> None:
        """Mark a scope as recently written, resetting its debounce.

        Args:
            scope: Store prefix.
            now: Time to record.
        """
        self.state.execute(
            "INSERT INTO pending(scope, last_seen) VALUES (?, ?) "
            "ON CONFLICT(scope) DO UPDATE SET last_seen = excluded.last_seen",
            (scope, time.time() if now is None else now))

    def clear(self, scope: str) -> None:
        """Drop a scope from the pending set, once handled.

        Args:
            scope: Store prefix.
        """
        self.state.execute("DELETE FROM pending WHERE scope = ?", (scope,))

    def step(self, handler: Callable[[str], Any],
             now: float | None = None) -> list[str]:
        """One poll and dispatch cycle.

        Args:
            handler: Called with each due scope, normally
                [`partition_store`][blobmap.service.partition_store] bound to
                its stores.
            now: Reference time.

        Returns:
            Scopes handled successfully. A scope whose handler raised is
            logged and left pending, so one unreachable store cannot stall
            the rest.
        """
        self.drain(now)
        done: list[str] = []
        for scope in self.due(now):
            try:
                handler(scope)
            except Exception:  # noqa: BLE001 - one bad store must not stall
                log.exception("partitioning %s failed; will retry", scope)
            else:
                self.clear(scope)
                done.append(scope)
        return done

    def run(self, handler: Callable[[str], Any]) -> None:
        """Poll and dispatch forever.

        Args:
            handler: Called with each due scope.
        """
        while True:
            self.step(handler)
            time.sleep(self.config.poll_seconds)

    # -- state ------------------------------------------------------------

    @property
    def position(self) -> int:
        """Last event id consumed.

        Returns:
            The cursor position, 0 before the first poll.
        """
        row = self.state.execute(
            "SELECT position FROM cursor WHERE consumer = ?",
            (self.config.consumer,)).fetchone()
        return int(row[0]) if row else 0

    def _set_position(self, position: int) -> None:
        """Record the cursor position.

        Args:
            position: Highest event id consumed.
        """
        self.state.execute(
            "INSERT INTO cursor(consumer, position) VALUES (?, ?) "
            "ON CONFLICT(consumer) DO UPDATE SET position = excluded.position",
            (self.config.consumer, position))


def store_of(key: str) -> str | None:
    """The prefix to re-examine after a write, or `None` to ignore it.

    Deliberately returns the metadata object's parent rather than the store
    root: resolving the root needs a LIST, which belongs in `root_of`. What
    matters here is that chunk writes produce nothing at all, which is what
    stops a conversion run's 300,000 events from becoming 300,000 queue
    entries.

    Args:
        key: Object key from a notification row.

    Returns:
        The parent prefix for a metadata write, else `None`.

    Example:
        >>> store_of("a/b.zarr/tas/zarr.json")
        'a/b.zarr/tas'
        >>> store_of("a/b.zarr/tas/c/0/0") is None
        True
    """
    if posixpath.basename(key) not in METADATA_BASENAMES:
        return None
    return posixpath.dirname(key) or None
