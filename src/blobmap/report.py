"""What is archivable, and what is not.

The number that matters when tiering is not working: how much data no policy
can ever move. Three things put data in that category, and only one of them is
visible without asking.

* **hot** -- metadata objects and dimension coordinates, held back so that
  opening a store never touches tape. Small and expected.
* **pinned** -- someone asked for it. Legitimate, but a pin nobody revisits is
  indistinguishable from a leak.
* **unmanaged** -- nothing claims it. A store that was never partitioned, a
  variable added after the last run, or a layout the partitioner did not
  recognise. This is the one that grows silently.

A bucket whose unmanaged share is climbing has something the partitioner is
not seeing, and that is worth knowing long before the pool fills.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .manifests import ManifestStore
from .resolve import Trie
from .storage import Store, list_all

log = logging.getLogger(__name__)


@dataclass
class Report:
    """Byte and object counts for one scope or bucket.

    Attributes:
        scope: What was counted.
        archivable_bytes: Data a tiering policy is free to move.
        hot_bytes: Metadata and coordinates, held back structurally.
        pinned_bytes: Held back deliberately.
        unmanaged_bytes: Claimed by nothing. The category that grows silently.
        blobs: Distinct blob instances seen.
        objects: Objects counted.
        pins: Pins in effect.
        expired_pins: Pins whose review date has passed.
    """

    scope: str
    archivable_bytes: int = 0
    hot_bytes: int = 0
    pinned_bytes: int = 0
    unmanaged_bytes: int = 0
    blobs: set[str] = field(default_factory=set)
    objects: int = 0
    pins: int = 0
    expired_pins: int = 0

    @property
    def total_bytes(self) -> int:
        """Get the total size of a blob."""
        return (
            self.archivable_bytes
            + self.hot_bytes
            + self.pinned_bytes
            + self.unmanaged_bytes
        )

    @property
    def held_bytes(self) -> int:
        """Get everything no policy can move."""
        return self.hot_bytes + self.pinned_bytes + self.unmanaged_bytes

    @property
    def held_fraction(self) -> float:
        """Get the share of the scope that tiering cannot touch, 0 to 1."""
        return self.held_bytes / self.total_bytes if self.total_bytes else 0.0

    def add(self, kind: str, size: int) -> None:
        """Count one object.

        Args:
            kind: A `Resolution.kind`.
            size: Object size in bytes.
        """
        self.objects += 1
        if kind == "blob":
            self.archivable_bytes += size
        elif kind == "hot":
            self.hot_bytes += size
        elif kind == "pinned":
            self.pinned_bytes += size
        else:
            self.unmanaged_bytes += size


def report(data: Store, manifests: ManifestStore, root: str = "") -> Report:
    """Walk a prefix and classify every object against the manifests.

    This is a full listing, so it costs what a partition run costs. It is a
    thing to run nightly or on demand, not per request.

    Args:
        data: Storage handle for the data.
        manifests: Where manifests live.
        root: Prefix to count, or empty for everything.

    Returns:
        A `Report`.
    """
    loaded = manifests.load_all()
    trie = Trie()
    trie.add_all(loaded)

    out = Report(scope=root or ".")
    for manifest in loaded:
        out.pins += len(manifest.pinned)
        out.expired_pins += sum(1 for p in manifest.pinned if p.expired())

    for entry in list_all(data, root):
        resolution = trie.lookup(entry.key)
        out.add(resolution.kind, entry.size)
        if resolution.blob_id:
            out.blobs.add(resolution.blob_id)
    return out


def human(n: float) -> str:
    """Format a byte count.

    Args:
        n: Size in bytes.

    Returns:
        A short binary-unit string.

    Example:
        >>> human(1536)
        '1.5 KiB'
    """
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if n < 1024 or unit == "PiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return ""


def render(reports: list[Report]) -> str:
    """Format reports as a table, worst first.

    Args:
        reports: One per bucket or scope.

    Returns:
        A table, followed by a warning for anything with a meaningful
        unmanaged share.
    """
    if not reports:
        return "nothing to report"

    width = max(len(r.scope) for r in reports)
    lines = [
        f"{'scope':<{width}}  {'total':>10}  {'archivable':>11}  "
        f"{'hot':>9}  {'pinned':>9}  {'unmanaged':>10}  {'blobs':>7}",
        "-" * (width + 68),
    ]

    for r in sorted(reports, key=lambda x: -x.unmanaged_bytes):
        lines.append(
            f"{r.scope:<{width}}  {human(r.total_bytes):>10}  "
            f"{human(r.archivable_bytes):>11}  {human(r.hot_bytes):>9}  "
            f"{human(r.pinned_bytes):>9}  {human(r.unmanaged_bytes):>10}  "
            f"{len(r.blobs):>7,}"
        )

    notes: list[str] = []
    for r in reports:
        if r.total_bytes and r.unmanaged_bytes / r.total_bytes > 0.01:
            share = 100 * r.unmanaged_bytes / r.total_bytes
            notes.append(
                f"{r.scope}: {share:.0f}% unmanaged. Something exists that no "
                f"manifest claims -- an unpartitioned store, a variable added "
                f"since the last run, or a layout the partitioner did not "
                f"recognise. Run scan."
            )
    expired = sum(r.expired_pins for r in reports)
    if expired:
        notes.append(
            f"{expired} pin(s) past their review date. Run: blobmap pin show --expired"
        )

    return "\n".join(lines) + ("\n\n" + "\n".join(notes) if notes else "")
