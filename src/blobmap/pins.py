"""Adding, removing and reporting pins.

A pin is a deliberate instruction to keep a prefix on disk. It lives in the
manifest, not in the store, for the same reason everything else does: data
arrives that must not be altered, and a pin is most needed exactly for
someone else's data that a colleague is actively working on.

Pins are set once and persist, rather than being a flag on every partition
run. A flag passed at partition time is one somebody forgets, and then whether
a dataset stays hot depends on shell history.
"""

from __future__ import annotations

import getpass
import logging
from dataclasses import dataclass

from .manifests import ManifestStore
from .model import Manifest, Pin, now

log = logging.getLogger(__name__)


class NotPartitioned(LookupError):
    """No manifest exists for this scope.

    Pinning something that has never been partitioned is almost always a typo
    in the scope, so this is an error rather than an invitation to create one.
    """


@dataclass(frozen=True)
class PinRecord:
    """A pin together with the scope it belongs to.

    Attributes:
        scope: The manifest scope.
        pin: The pin itself.
    """

    scope: str
    pin: Pin


def add(manifests: ManifestStore, scope: str, prefix: str, reason: str, *,
        by: str | None = None, until: str | None = None) -> Manifest:
    """Pin a prefix.

    Args:
        manifests: Where manifests live.
        scope: The manifest scope, which must already be partitioned.
        prefix: What to keep hot, relative to the scope. Empty pins the
            whole scope.
        reason: Why. Required.
        by: Who is setting it, defaulting to the current user.
        until: ISO 8601 UTC after which it should be reviewed. `None` for
            open-ended, which never expires and therefore never prompts
            anyone to look at it again.

    Returns:
        The updated manifest.

    Raises:
        NotPartitioned: If the scope has no manifest.
        ValueError: If the prefix is already pinned, or the reason is empty.
    """
    stored = manifests.read(scope)
    if stored is None:
        raise NotPartitioned(
            f"{scope} has no manifest. Partition it first; pinning an "
            f"unpartitioned scope is usually a mistyped scope.")

    prefix = prefix.strip("/")
    if any(p.prefix == prefix for p in stored.manifest.pinned):
        raise ValueError(f"{scope}: {prefix or '(whole scope)'} is already "
                         f"pinned; remove it first to change the reason")

    pin = Pin(prefix=prefix, reason=reason,
              by=by or getpass.getuser(), at=now(), until=until)
    updated = stored.manifest.bumped(
        pinned=stored.manifest.pinned + (pin,))
    manifests.write(updated, etag=stored.etag)
    log.info("pinned %s/%s", scope, prefix)
    return updated


def remove(manifests: ManifestStore, scope: str, prefix: str) -> Manifest:
    """Remove a pin.

    Note this does not archive anything. It makes the prefix eligible again;
    the tiering policy decides what happens next.

    Args:
        manifests: Where manifests live.
        scope: The manifest scope.
        prefix: The pinned prefix, relative to the scope.

    Returns:
        The updated manifest.

    Raises:
        NotPartitioned: If the scope has no manifest.
        LookupError: If the prefix is not pinned.
    """
    stored = manifests.read(scope)
    if stored is None:
        raise NotPartitioned(f"{scope} has no manifest")

    prefix = prefix.strip("/")
    kept = tuple(p for p in stored.manifest.pinned if p.prefix != prefix)
    if len(kept) == len(stored.manifest.pinned):
        raise LookupError(f"{scope}: {prefix or '(whole scope)'} is not pinned")

    updated = stored.manifest.bumped(pinned=kept)
    manifests.write(updated, etag=stored.etag)
    log.info("unpinned %s/%s -- now eligible, but nothing is archived until "
             "the tiering policy runs", scope, prefix)
    return updated


def show(manifests: ManifestStore, scope: str | None = None, *,
         expired_only: bool = False) -> list[PinRecord]:
    """Every pin, worst first.

    Ordering puts expired pins first, then open-ended ones, then the rest.
    Both of the first two categories are how a hot pool quietly fills: someone
    pins a dataset for a paper, the paper ships, nobody unpins it.

    Args:
        manifests: Where manifests live.
        scope: Restrict to one scope, or `None` for all.
        expired_only: Only pins whose review date has passed.

    Returns:
        Matching pins, ordered.
    """
    if scope is None:
        sources = manifests.load_all()
    else:
        stored = manifests.read(scope)
        sources = [stored.manifest] if stored else []

    records = [PinRecord(m.scope, p) for m in sources for p in m.pinned]
    if expired_only:
        records = [r for r in records if r.pin.expired()]

    def rank(record: PinRecord) -> tuple[int, str]:
        if record.pin.expired():
            return (0, record.pin.until or "")
        if record.pin.until is None:
            return (1, record.scope)
        return (2, record.pin.until)

    return sorted(records, key=rank)
