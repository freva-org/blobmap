"""Command line interface.

Four subcommands: `partition`, `scan`, `resolve` and `show`. Every invocation
needs both `--data` (where the zarr stores are) and `--manifests` (where the
blob definitions live), because they are deliberately different buckets.

Example:
    ```console
    $ blobmap --data s3://cordex --manifests s3://waterpark-blobmap \
        partition nukleus/eur11.zarr --dry-run
    ```
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from typing import Sequence

from .backends import S3Options, StoreUnreachable, diagnose, open_store
from .discover import scan
from .hierarchy import DEFAULT_EXCLUDE, read_arrays
from .manifests import ManifestStore
from . import pins
from .model import Array, GiB, Manifest, Policy
from .resolve import Trie
from .schema import SCHEMA
from .service import partition_store
from .storage import Store

log = logging.getLogger(__name__)


def parse_size(text: str) -> int:
    """Parse a human byte size.

    Args:
        text: A number with an optional binary unit suffix, such as `100GiB`,
            `1.5G`, `2M` or a bare byte count. Units are powers of 1024;
            `GB` and `GiB` mean the same thing here.

    Returns:
        The size in bytes.

    Raises:
        ValueError: If the text is not a number.

    Example:
        >>> parse_size("1.5G") == int(1.5 * GiB)
        True
        >>> parse_size("512")
        512
    """
    units = {"K": 1024, "M": 1024**2, "G": GiB, "T": 1024**4}
    clean = text.strip().upper().removesuffix("B").removesuffix("I")
    if clean and clean[-1] in units:
        return int(float(clean[:-1]) * units[clean[-1]])
    return int(float(clean))


def human(n: float) -> str:
    """Format a byte count for display.

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


def explain(arrays: Sequence[Array], manifest: Manifest) -> str:
    """Render the per-array decision table.

    This is what `--dry-run` prints, and what you will stare at while tuning
    `t_max`. Note it reports the decision *per array*: coalesced arrays each
    show the shared blob they were folded into.

    Args:
        arrays: The arrays that were partitioned.
        manifest: The resulting manifest.

    Returns:
        A fixed-width table, one row per array. `UNMANAGED` in the blob
        column means nothing claims that array, so it will stay hot.
    """
    by_prefix = {p: b for b in manifest.blobs for p in b.prefixes}
    lines = [
        f"{'array':14} {'objects':>9} {'stored':>11} {'object':>11}  blob",
        "-" * 80,
    ]
    for a in sorted(arrays, key=lambda x: x.path):
        blob = by_prefix.get(a.chunk_prefix) or by_prefix.get("")
        if a.is_coordinate or a.total_bytes < manifest.policy.t_hot_bytes:
            where = "pinned hot"
        elif blob is None:
            where = "UNMANAGED"
        elif blob.bucket is not None:
            width = blob.bucket.width
            n = math.ceil(a.object_grid[0] / width)
            where = (
                f"{blob.id}_0..{n - 1}  (width={width}, "
                f"~{human(width * a.avg_object_bytes)}/blob)"
            )
        else:
            where = f"{blob.id}_0"
        lines.append(
            f"{a.path or '.':14} {a.nobjects:>9} "
            f"{human(a.total_bytes):>11} "
            f"{human(a.avg_object_bytes):>11}  {where}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        The parser, exposed separately so docs and tests can introspect it.
    """
    p = argparse.ArgumentParser(
        prog="blobmap",
        description="Decide which zarr objects move to tape together.",
        epilog=(
            "examples:\n"
            "  blobmap --data s3://cordex --manifests s3://waterpark-blobmap \\\n"
            "      scan --partition --dry-run\n"
            "  blobmap --data s3://cordex --manifests s3://waterpark-blobmap \\\n"
            "      partition nukleus/eur11.zarr --t-max 50GiB --dry-run\n"
            "  blobmap --data file:///work --manifests file:///work/.blobmap \\\n"
            "      resolve eur11.zarr/tas/c/5000/0/0\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--data",
        metavar="URL",
        help="where the zarr data lives: s3://cordex, or a path such as "
        "/work/data or ./data. Read only; blobmap never writes here, "
        "which is what lets it manage data you must not alter. Must "
        "already exist. Required for every command except schema.",
    )
    p.add_argument(
        "--manifests",
        metavar="URL",
        help="where blob definitions are written: s3://waterpark-blobmap, or "
        "a path such as /work/blobmap. Keys mirror the data paths. Must "
        "be writable; a local directory is created if missing. Required "
        "for every command except schema.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="log at DEBUG, including skipped prefixes during a scan.",
    )
    p.add_argument(
        "--exclude",
        metavar="SEGMENT",
        action="append",
        default=None,
        help="path segment to skip, repeatable. Defaults to "
        + ", ".join(DEFAULT_EXCLUDE)
        + ". Relevant when --data points at "
        "an S3 gateway's backing filesystem rather than its S3 endpoint: "
        "a directory walk sees staging directories the S3 API hides, and "
        "staging objects inside a store would be counted into an array's "
        "size. Pass an empty value to exclude nothing.",
    )

    s3 = p.add_argument_group(
        "s3 options",
        "Only needed for s3:// URLs. Each falls back to the matching AWS_* "
        "environment variable. With none of them set and no credentials in "
        "the environment, obstore assumes AWS and hangs looking for instance "
        "credentials.",
    )
    s3.add_argument(
        "--endpoint",
        metavar="URL",
        help="S3 endpoint, e.g. https://s3.example.org. Required for any "
        "gateway that is not AWS. Env: AWS_ENDPOINT_URL.",
    )
    s3.add_argument(
        "--region",
        metavar="NAME",
        help="region name. Defaults to us-east-1, which most on-premise "
        "gateways accept regardless. Env: AWS_REGION.",
    )
    s3.add_argument(
        "--anonymous",
        action="store_true",
        help="do not sign requests or look for credentials. For public "
        "buckets. Env: AWS_SKIP_SIGNATURE.",
    )
    s3.add_argument(
        "--allow-http",
        action="store_true",
        help="permit an unencrypted http endpoint. Implied when --endpoint "
        "already starts with http://.",
    )
    s3.add_argument(
        "--virtual-hosted",
        action="store_true",
        help="use bucket.host addressing instead of host/bucket. Most "
        "on-premise gateways want this off, which is the default.",
    )

    sub = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    q = sub.add_parser(
        "partition",
        help="partition or repartition one store",
        description="Compute blob definitions for one scope and write the "
        "manifest. Repartitioning is additive: existing blobs "
        "are pinned, so ids and the tape addresses behind them "
        "survive.",
    )
    q.add_argument(
        "scope",
        metavar="SCOPE",
        help="prefix to partition, relative to --data, e.g. nukleus/eur11.zarr",
    )
    q.add_argument(
        "--t-max",
        type=parse_size,
        metavar="SIZE",
        help="target upper bound for one blob, e.g. 100GiB. Defaults to the "
        "previous manifest's policy, or 100GiB on a first run. Note that "
        "on an already partitioned store this only affects newly cut "
        "regions: existing cuts are frozen.",
    )
    q.add_argument(
        "--dry-run",
        action="store_true",
        help="print the decision table and the diff, write nothing.",
    )
    q.add_argument(
        "--force",
        action="store_true",
        help="recompute from scratch instead of pinning existing blobs. This "
        "is the only operation that can move or drop a blob, orphaning "
        "any tape copy held against its id. It logs what it broke.",
    )

    s = sub.add_parser(
        "scan",
        help="find zarr stores under a prefix",
        description="Walk --data and report every zarr store, marking which "
        "already have a manifest. Descent stops at a store "
        "boundary.",
    )
    s.add_argument(
        "root",
        nargs="?",
        default="",
        metavar="PREFIX",
        help="prefix to scan under, relative to --data. Default: everything.",
    )
    s.add_argument(
        "--partition",
        action="store_true",
        help="also partition every store found that has no manifest yet.",
    )
    s.add_argument(
        "--dry-run",
        action="store_true",
        help="with --partition, compute but write nothing.",
    )

    r = sub.add_parser(
        "resolve",
        help="map object keys to blob ids",
        description="Load every manifest and resolve keys against them. The "
        "fastest way to check that a store is partitioned the "
        "way you think it is.",
    )
    r.add_argument(
        "keys",
        nargs="+",
        metavar="KEY",
        help="object keys to resolve, including the scope prefix, e.g. "
        "nukleus/eur11.zarr/tas/c/5000/0/0. Prints the blob id, or "
        "'hot' for a metadata object or pinned coordinate, or "
        "'unmanaged' when nothing claims the key.",
    )

    sub.add_parser(
        "show",
        help="list known scopes and epochs",
        description="One line per manifest: epoch, blob count and scope.",
    )

    pin = sub.add_parser(
        "pin",
        help="keep a prefix on disk deliberately",
        description="A pin is set once and persists, rather than being a flag "
        "passed on every partition run. Whether a dataset stays "
        "hot should not depend on shell history.",
    )
    pin_sub = pin.add_subparsers(dest="pin_cmd", required=True, metavar="ACTION")

    pin_add = pin_sub.add_parser(
        "add",
        help="pin a prefix",
        description="Records who pinned what, why, and optionally when it "
        "should be reviewed.",
    )
    pin_add.add_argument(
        "scope",
        metavar="SCOPE",
        help="the manifest scope, which must already be partitioned",
    )
    pin_add.add_argument(
        "prefix",
        metavar="PREFIX",
        nargs="?",
        default="",
        help="what to keep hot, relative to SCOPE. Omit to pin the whole scope.",
    )
    pin_add.add_argument(
        "--reason",
        required=True,
        metavar="TEXT",
        help="why this must stay on disk. Required: a pin "
        "nobody can explain is one nobody removes.",
    )
    pin_add.add_argument(
        "--until",
        metavar="DATE",
        help="ISO date after which this should be reviewed, "
        "e.g. 2026-12-01. Strongly encouraged. Expiry "
        "is reported, never enforced, so nothing is "
        "unpinned behind your back.",
    )
    pin_add.add_argument(
        "--by", metavar="NAME", help="who is setting it. Defaults to the current user."
    )

    pin_rm = pin_sub.add_parser(
        "remove",
        help="remove a pin",
        description="Makes the prefix eligible again. Nothing is archived "
        "until the tiering policy runs.",
    )
    pin_rm.add_argument("scope", metavar="SCOPE")
    pin_rm.add_argument("prefix", metavar="PREFIX", nargs="?", default="")

    pin_show = pin_sub.add_parser(
        "show",
        help="list pins, worst first",
        description="Expired pins first, then open-ended ones, then the rest. "
        "Those first two categories are how a hot pool quietly "
        "fills.",
    )
    pin_show.add_argument(
        "scope", metavar="SCOPE", nargs="?", help="restrict to one scope. Default: all."
    )
    pin_show.add_argument(
        "--expired", action="store_true", help="only pins whose review date has passed."
    )

    sub.add_parser(
        "schema",
        help="print the manifest JSON Schema",
        description="Write the schema to stdout. This is the contract for "
        "any consumer, including ones not written in Python.",
    )
    return p


def _pin(args: argparse.Namespace, store: ManifestStore) -> int:
    """Run a `pin` subcommand.

    Args:
        args: Parsed arguments.
        store: Where manifests live.

    Returns:
        Process exit code.
    """
    if args.pin_cmd == "add":
        until = (
            f"{args.until}T00:00:00+00:00"
            if args.until and "T" not in args.until
            else args.until
        )
        pins.add(store, args.scope, args.prefix, args.reason, by=args.by, until=until)
        print(f"pinned {args.scope}/{args.prefix}".rstrip("/"))
        return 0

    if args.pin_cmd == "remove":
        pins.remove(store, args.scope, args.prefix)
        print(f"unpinned {args.scope}/{args.prefix}".rstrip("/"))
        print("  eligible again; nothing is archived until the tiering policy runs")
        return 0

    records = pins.show(store, args.scope, expired_only=args.expired)
    if not records:
        print("no pins" if not args.expired else "no expired pins")
        return 0

    width = max(len(f"{r.scope}/{r.pin.prefix}".rstrip("/")) for r in records)
    for record in records:
        where = f"{record.scope}/{record.pin.prefix}".rstrip("/")
        if record.pin.expired() and record.pin.until:
            state = f"EXPIRED {record.pin.until[:10]}"
        elif record.pin.until is None:
            state = "no expiry"
        else:
            state = f"until {record.pin.until[:10]}"
        print(f"{where:<{width}}  {record.pin.by:<10} {state:<20} {record.pin.reason}")

    stale = sum(1 for r in records if r.pin.expired() or r.pin.until is None)
    if stale:
        print(
            f"\n{stale} pin(s) expired or open-ended -- these are how a hot pool fills"
        )
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    data: Store | None = None,
    manifests: Store | None = None,
) -> int:
    """Entry point.

    Args:
        argv: Arguments to parse. Defaults to `sys.argv[1:]`.
        data: Pre-built data handle, bypassing `--data`. For tests.
        manifests: Pre-built manifest handle, bypassing `--manifests`. For
            tests.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    if args.cmd == "schema":
        print(json.dumps(SCHEMA, indent=2))
        return 0

    missing = [name for name in ("data", "manifests") if getattr(args, name) is None]
    if missing:
        names = " and ".join(f"--{name}" for name in missing)
        verb = "are" if len(missing) > 1 else "is"
        print(f"error: {names} {verb} required for {args.cmd}", file=sys.stderr)
        return 2

    options = S3Options(
        endpoint=args.endpoint,
        region=args.region,
        anonymous=args.anonymous,
        allow_http=args.allow_http or None,
        virtual_hosted=args.virtual_hosted or None,
    )
    try:
        # create=True for manifests only: blobmap owns that bucket and writes
        # to it. Creating a mistyped --data would make a scan report nothing
        # found, which looks identical to an empty bucket.
        data = data if data is not None else open_store(args.data, options)
        store = ManifestStore(
            manifests
            if manifests is not None
            else open_store(args.manifests, options, create=True)
        )
    except StoreUnreachable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        return _dispatch(args, data, store)
    except StoreUnreachable as exc:
        print(diagnose(args.data, exc), file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - backend errors are opaque
        message = diagnose(args.data, exc)
        if message.endswith(str(exc)) and not args.verbose:
            raise  # nothing useful to add, show the trace
        print(message, file=sys.stderr)
        log.debug("underlying error", exc_info=True)
        return 2


def _dispatch(args: argparse.Namespace, data: Store, store: ManifestStore) -> int:
    """Run the selected subcommand.

    Args:
        args: Parsed arguments.
        data: Handle for the zarr data.
        store: Where manifests live.

    Returns:
        Process exit code.
    """

    exclude = (
        DEFAULT_EXCLUDE if args.exclude is None else tuple(e for e in args.exclude if e)
    )

    if args.cmd == "partition":
        policy = Policy(t_max_bytes=args.t_max) if args.t_max else None
        result = partition_store(
            data,
            store,
            args.scope,
            policy=policy,
            force=args.force,
            dry_run=args.dry_run,
            exclude=exclude,
        )
        print(explain(read_arrays(data, args.scope, exclude=exclude), result.manifest))
        print()
        print(result.diff.describe())
        print(
            f"\nepoch {result.manifest.epoch}, "
            f"{'written' if result.written else 'not written'}"
        )

    elif args.cmd == "scan":
        for candidate in scan(data, args.root, store, exclude=exclude):
            mark = "ok " if candidate.has_manifest else "NEW"
            print(f"{mark} {candidate.fmt}  {candidate.scope}", flush=True)
            if args.partition and not candidate.has_manifest:
                # a store with millions of chunks takes minutes to walk, so
                # say what is being worked on before starting
                print(f"    partitioning {candidate.scope} ...", end="", flush=True)
                result = partition_store(
                    data, store, candidate.scope, dry_run=args.dry_run, exclude=exclude
                )
                print(
                    f" {len(result.manifest.blobs)} blobs"
                    f"{'' if result.written else ' (not written)'}",
                    flush=True,
                )

    elif args.cmd == "resolve":
        trie = Trie()
        trie.add_all(store.load_all())
        for key in args.keys:
            res = trie.lookup(key)
            print(f"{key}  ->  {res.blob_id or res.kind}")

    elif args.cmd == "pin":
        return _pin(args, store)

    elif args.cmd == "schema":
        print(json.dumps(SCHEMA, indent=2))

    elif args.cmd == "show":
        for manifest in sorted(store.load_all(), key=lambda m: m.scope):
            print(
                f"epoch {manifest.epoch:>4}  {len(manifest.blobs):>3} blobs  "
                f"{manifest.scope}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
