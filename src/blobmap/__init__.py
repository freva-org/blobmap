"""blobmap: decide which objects move to tape together, and read that back.

    model       manifest types                stdlib only
    partition   arrays -> Manifest            pure, no I/O
    resolve     object key -> blob id         the semantics of the format
    storage     three ops over obstore        S3, local disk, memory
    manifests   manifest read/write           conditional, in a bucket you own
    hierarchy   zarr store -> arrays          LIST-first, never reads a chunk
    discover    scan / event drivers          what feeds the partitioner
"""

from .discover import Candidate, EventPoller, PollConfig, scan, store_of
from .hierarchy import (NotAZarrStore, detect_format, find_store_root,
                        read_arrays)
from .manifests import MANIFEST_NAME, ManifestStore, Stored
from .model import (GiB, KEY_ENCODINGS, METADATA_BASENAMES, SCHEMA_VERSION,
                    Array, Blob, Bucket, Manifest, Policy)
from .partition import Diff, bucket_width, diff, partition
from .resolve import Resolution, Trie, resolve
from .service import NotAdditive, Result, partition_store
from .storage import Conflict, Entry, Store

__version__ = "0.4.0"

__all__ = [
    "GiB", "KEY_ENCODINGS", "METADATA_BASENAMES", "SCHEMA_VERSION",
    "Array", "Blob", "Bucket", "Manifest", "Policy",
    "Diff", "bucket_width", "diff", "partition",
    "Resolution", "Trie", "resolve",
    "Conflict", "Entry", "Store",
    "MANIFEST_NAME", "ManifestStore", "Stored",
    "NotAZarrStore", "detect_format", "find_store_root", "read_arrays",
    "Candidate", "EventPoller", "PollConfig", "scan", "store_of",
    "NotAdditive", "Result", "partition_store",
]
