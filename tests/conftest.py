"""Fixtures.

The `store` fixture is parametrized over obstore backends. MemoryStore and
LocalStore are real implementations of the same API as S3Store, so there is
nothing to mock -- and adding MinIO later means adding a third param, not a
second test suite.

Set BLOBMAP_TEST_S3_URL (plus the usual AWS_* env vars, or
BLOBMAP_TEST_S3_ENDPOINT for MinIO) to include the S3 backend.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Iterator

import pytest
from obstore.store import LocalStore, MemoryStore, S3Store

S3_URL = os.environ.get("BLOBMAP_TEST_S3_URL")

BACKENDS = ["memory", "local"] + (["s3"] if S3_URL else [])


@pytest.fixture(params=BACKENDS)
def store(request: pytest.FixtureRequest, tmp_path: Any) -> Iterator[Any]:
    if request.param == "memory":
        yield MemoryStore()
    elif request.param == "local":
        root = tmp_path / "data"
        root.mkdir()
        yield LocalStore(str(root), mkdir=True)
    else:
        bucket = S3_URL.removeprefix("s3://").strip("/")
        prefix = f"test-{uuid.uuid4().hex[:8]}"
        endpoint = os.environ.get("BLOBMAP_TEST_S3_ENDPOINT")
        config: dict[str, Any] = {"bucket": bucket, "prefix": prefix}
        if endpoint:
            config |= {"endpoint": endpoint, "virtual_hosted_style_request": False,
                       "client_options": {"allow_http": True}}
        yield S3Store(**config)


@pytest.fixture
def memory() -> Any:
    """A second store, for manifests, when the test does not care about the
    backend."""
    return MemoryStore()
