"""Turning a URL into an `obstore` store, with useful errors when it fails.

`obstore` reads most S3 settings from the environment, but with none of them
set it assumes AWS and looks for instance credentials on the EC2 metadata
address. Off EC2 that hangs for the full retry budget and then fails with a
Rust backtrace that does not mention credentials at all, so this module
translates the common failures into something actionable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from obstore.store import LocalStore, from_url

from .storage import Store

log = logging.getLogger(__name__)

#: A URL scheme, as opposed to a bare filesystem path. Matches `s3://`,
#: `file://`, `memory://` and so on.
SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")

#: Address of the EC2 instance metadata service. A failure mentioning this
#: means no credentials were found, not that the endpoint is unreachable.
IMDS_HOST = "169.254.169.254"


class StoreUnreachable(RuntimeError):
    """A store could not be opened or listed."""


def is_local(url: str | Path) -> bool:
    """Whether a URL or path refers to the local filesystem.

    Args:
        url: A URL or a bare path.

    Returns:
        True for `file://` URLs and for anything with no scheme at all.

    Example:
        >>> all(is_local(u) for u in ("/foo", "foo", "~/foo", "file:///foo"))
        True
        >>> any(is_local(u) for u in ("s3://b", "memory://"))
        False
    """
    if isinstance(url, Path) or not SCHEME.match(url):
        return True
    return url.startswith("file://")


def local_path(url: str) -> Path:
    """The absolute filesystem path a local URL or bare path refers to.

    Args:
        url: A `file://` URL, or a bare path which may be relative and may
            start with `~`.

    Returns:
        An absolute, expanded path. Not resolved through symlinks, so a
        manifest directory that is a symlink stays where the user put it.

    Example:
        >>> local_path("file:///srv/blobmap")
        PosixPath('/srv/blobmap')
        >>> local_path("/srv/blobmap")
        PosixPath('/srv/blobmap')
        >>> local_path("relative/dir").is_absolute()
        True
    """
    raw = url[len("file://") :] if url.startswith("file://") else url
    return Path(raw).expanduser().absolute()


def normalise(url: str) -> str:
    """Turn a bare path into a URL, leaving real URLs alone.

    `obstore` rejects a bare path with "relative URL without a base", which
    says nothing useful. Accepting `/foo/bar` and `foo/bar` is what people
    actually type.

    Args:
        url: A URL or a bare path.

    Returns:
        A URL `obstore` will accept.

    Example:
        >>> normalise("/srv/blobmap")
        'file:///srv/blobmap'
        >>> normalise("s3://cmip6")
        's3://cmip6'
    """
    if SCHEME.match(url):
        return url
    return local_path(url).as_uri()


@dataclass(frozen=True)
class S3Options:
    """Connection settings that override the environment.

    Every field defaults to `None`, meaning "leave it to `obstore`", which
    falls back to the usual `AWS_*` environment variables. Only set fields are
    passed through, so these compose with an existing environment rather than
    replacing it.

    Attributes:
        endpoint: S3 endpoint URL. Required for anything that is not AWS.
            Also settable as `AWS_ENDPOINT_URL`.
        region: Region name. `obstore` defaults to `us-east-1`, which most
            on-premise gateways accept. Also `AWS_REGION`.
        anonymous: Skip credentials and request signing entirely. Use for
            public buckets, and note this is also the quickest way to avoid
            the instance metadata lookup. Also `AWS_SKIP_SIGNATURE`.
        allow_http: Permit a plain `http://` endpoint. `obstore` refuses one
            by default.
        virtual_hosted: Use `bucket.host` addressing instead of `host/bucket`.
            Most on-premise gateways want this off, which is the default.
    """

    endpoint: str | None = None
    region: str | None = None
    anonymous: bool = False
    allow_http: bool | None = None
    virtual_hosted: bool | None = None

    def config(self) -> dict[str, Any]:
        """Build the keyword arguments for `obstore.store.from_url`.

        Returns:
            Only the options that were actually set, so unset ones continue
            to come from the environment.
        """
        out: dict[str, Any] = {}
        if self.endpoint:
            out["endpoint"] = self.endpoint
        if self.region:
            out["region"] = self.region
        if self.anonymous:
            out["skip_signature"] = True
        if self.virtual_hosted is not None:
            out["virtual_hosted_style_request"] = self.virtual_hosted
        allow_http = self.allow_http
        if allow_http is None and self.endpoint:
            allow_http = self.endpoint.startswith("http://")
        if allow_http:
            out["client_options"] = {"allow_http": True}
        return out


def open_store(
    url: str | Path, options: S3Options | None = None, *, create: bool = False
) -> Store:
    """Open a store from a URL or a bare filesystem path.

    Bare paths are accepted and made absolute, so `foo/bar`, `/foo/bar` and
    `file:///foo/bar` all work.

    Args:
        url: A URL such as `s3://bucket` or `memory://`, or a filesystem path
            which may be relative and may start with `~`.
        options: S3 settings that override the environment. Ignored for
            non-S3 schemes.
        create: Create the directory if it is local and missing. Pass this
            for the manifest store, which blobmap owns and writes to. Do not
            pass it for the data store: silently creating a mistyped path
            would make a scan report nothing found, which looks exactly like
            an empty bucket.

    Returns:
        A storage handle.

    Raises:
        StoreUnreachable: If the path does not exist and `create` is not set,
            if it exists but is a file, or if the URL cannot be opened.

    Example:
        >>> type(open_store("memory://")).__name__
        'MemoryStore'
    """
    options = options or S3Options()
    url = normalise(url)

    if is_local(url):
        return _open_local(url, create=create)

    config = options.config() if url.startswith("s3://") else {}
    try:
        return from_url(url, **config)
    except Exception as exc:  # noqa: BLE001 - backend-specific error types
        raise StoreUnreachable(f"cannot open {url}: {exc}") from exc


def _open_local(url: str, *, create: bool) -> LocalStore:
    """Open a local store, checking the directory before obstore does.

    obstore reports a missing directory as a canonicalisation failure, which
    does not read as "this path does not exist".

    Args:
        url: A `file://` URL.
        create: Whether to create a missing directory.

    Returns:
        A `LocalStore` rooted at the path.

    Raises:
        StoreUnreachable: If the path is missing, is a file, or cannot be
            created.
    """
    path = local_path(url)

    if path.exists() and not path.is_dir():
        raise StoreUnreachable(f"{path} is a file, not a directory")

    if not path.exists():
        if not create:
            raise StoreUnreachable(
                f"{path} does not exist. Check the path, or create it first "
                f"if this is where manifests should live."
            )
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StoreUnreachable(f"cannot create {path}: {exc}") from exc
        log.info("created %s", path)

    try:
        return LocalStore(str(path))
    except Exception as exc:  # noqa: BLE001 - backend-specific error types
        raise StoreUnreachable(f"cannot open {path}: {exc}") from exc


def diagnose(url: str, exc: BaseException) -> str:
    """Turn a backend error into something a human can act on.

    Args:
        url: The store URL that failed.
        exc: The exception raised.

    Returns:
        A message naming the likely cause and the setting that fixes it.

    Example:
        >>> "credentials" in diagnose("s3://x", RuntimeError(IMDS_HOST))
        True
    """
    text = str(exc)

    if IMDS_HOST in text or "fd00:ec2" in text:
        return (
            f"{url}: no S3 credentials found, so obstore fell back to the EC2 "
            f"instance metadata service and timed out.\n"
            f"  Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, or pass "
            f"--anonymous for a public bucket.\n"
            f"  For anything that is not AWS you also need --endpoint (or "
            f"AWS_ENDPOINT_URL)."
        )

    if (
        "URL scheme is not allowed" in text
        or "not allowed" in text.lower()
        and "http" in text.lower()
    ):
        return (
            f"{url}: refusing a plain http endpoint. Pass --allow-http if "
            f"the endpoint really is unencrypted."
        )

    if "NoSuchBucket" in text or "404" in text:
        return (
            f"{url}: bucket not found. Check the name, and check "
            f"--endpoint points at the right gateway."
        )

    if "SignatureDoesNotMatch" in text or "403" in text or "InvalidAccessKeyId" in text:
        return (
            f"{url}: credentials rejected. If the gateway uses path style "
            f"addressing, note blobmap defaults to that already; if it "
            f"needs virtual hosted style, pass --virtual-hosted."
        )

    if "dns error" in text.lower() or "tcp connect error" in text.lower():
        return (
            f"{url}: cannot reach the endpoint. Check --endpoint and "
            f"whether it is reachable from here."
        )

    return f"{url}: {exc}"
