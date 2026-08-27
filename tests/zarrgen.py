"""Synthetic zarr stores, written directly as objects.

No zarr-python dependency: blobmap parses this metadata itself, so the
fixtures must be independent of the library whose output they imitate. That
also lets us produce layouts zarr-python will not easily write, such as a v3
array using v2-style flat chunk keys.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import obstore as obs

DTYPE_SIZE = {"float32": 4, "float64": 8, "int32": 4}


@dataclass
class ArraySpec:
    name: str
    shape: tuple[int, ...]
    chunks: tuple[int, ...]                # the *object* shape unless sharded
    dtype: str = "float32"
    inner_chunks: tuple[int, ...] | None = None    # set to make it sharded
    dimensions: list[str] = field(default_factory=list)
    bytes_per_object: int = 64
    written: int | None = None             # write only the first N objects

    @property
    def grid(self) -> tuple[int, ...]:
        return tuple(math.ceil(s / c) for s, c in zip(self.shape, self.chunks))

    @property
    def nobjects(self) -> int:
        return math.prod(self.grid)


def coordinate(name: str, size: int, **kw: Any) -> ArraySpec:
    return ArraySpec(name, (size,), (size,), dimensions=[name], **kw)


def write_store(store: Any, scope: str, arrays: list[ArraySpec], *,
                zarr_format: int = 3, separator: str = "/",
                v2_keys: bool = False, consolidated: bool = False) -> None:
    """Materialise a store. `separator` and `v2_keys` control key encoding."""
    prefix = f"{scope.strip('/')}/" if scope.strip("/") else ""
    if zarr_format == 3:
        _put(store, f"{prefix}zarr.json",
             {"zarr_format": 3, "node_type": "group", "attributes": {}})
    else:
        _put(store, f"{prefix}.zgroup", {"zarr_format": 2})

    consolidated_meta: dict[str, Any] = {}
    for spec in arrays:
        root = f"{prefix}{spec.name}/"
        if zarr_format == 3:
            meta = _v3_metadata(spec, separator, v2_keys)
            _put(store, f"{root}zarr.json", meta)
        else:
            meta = _v2_metadata(spec, separator)
            _put(store, f"{root}.zarray", meta)
            if spec.dimensions:
                _put(store, f"{root}.zattrs",
                     {"_ARRAY_DIMENSIONS": spec.dimensions})
        consolidated_meta[f"{spec.name}/.zarray"] = meta

        for index in _indices(spec.grid, spec.written):
            key = _chunk_key(spec, index, zarr_format, separator, v2_keys)
            obs.put(store, f"{root}{key}", b"\0" * spec.bytes_per_object)

    if consolidated and zarr_format == 2:
        _put(store, f"{prefix}.zmetadata",
             {"zarr_consolidated_format": 1, "metadata": consolidated_meta})


def _v3_metadata(spec: ArraySpec, separator: str, v2_keys: bool) -> dict[str, Any]:
    codecs: list[dict[str, Any]] = [{"name": "bytes", "configuration": {}}]
    if spec.inner_chunks:
        codecs = [{"name": "sharding_indexed",
                   "configuration": {"chunk_shape": list(spec.inner_chunks)}}]
    encoding = ({"name": "v2", "configuration": {"separator": separator}}
                if v2_keys else
                {"name": "default", "configuration": {"separator": "/"}})
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": list(spec.shape),
        "data_type": spec.dtype,
        "chunk_grid": {"name": "regular",
                       "configuration": {"chunk_shape": list(spec.chunks)}},
        "chunk_key_encoding": encoding,
        "codecs": codecs,
        "fill_value": 0,
        "attributes": ({"_ARRAY_DIMENSIONS": spec.dimensions}
                       if spec.dimensions else {}),
    }


def _v2_metadata(spec: ArraySpec, separator: str) -> dict[str, Any]:
    return {
        "zarr_format": 2,
        "shape": list(spec.shape),
        "chunks": list(spec.chunks),
        "dtype": f"<f{DTYPE_SIZE[spec.dtype]}",
        "compressor": None,
        "fill_value": 0,
        "order": "C",
        "filters": None,
        "dimension_separator": separator,
    }


def _chunk_key(spec: ArraySpec, index: tuple[int, ...], zarr_format: int,
               separator: str, v2_keys: bool) -> str:
    parts = [str(i) for i in index]
    if zarr_format == 3 and not v2_keys:
        return "c/" + "/".join(parts)
    return separator.join(parts)


def _indices(grid: tuple[int, ...], limit: int | None) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = [()]
    for extent in grid:
        out = [prev + (i,) for prev in out for i in range(extent)]
    return out[:limit] if limit is not None else out


def _put(store: Any, key: str, body: dict[str, Any]) -> None:
    obs.put(store, key, json.dumps(body).encode())
