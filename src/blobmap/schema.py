"""JSON Schema for the manifest, and a small validator that walks it.

The schema is the contract with every consumer, including ones not written
yet. It is deliberately a plain dict rather than something generated from the
dataclasses: the format has to outlive this package's class layout, and a
consumer in another language needs the schema, not a Python model.

Validation walks the schema rather than reimplementing the rules, so the two
cannot drift. Only the keywords the schema actually uses are supported, which
is a fraction of the specification and about sixty lines. `tests/test_schema.py`
cross-checks the walker against the real `jsonschema` library, so if the schema
grows a keyword this does not handle, that test fails rather than the keyword
being silently ignored.

Example:
    >>> validate_document({"schema_version": 2, "scope": "s", "epoch": 1,
    ...                    "hot_always": [], "blobs": []})
    >>> validate_document({"schema_version": 2, "scope": "s", "epoch": 0,
    ...                    "hot_always": [], "blobs": []})
    Traceback (most recent call last):
        ...
    blobmap.schema.SchemaError: epoch: 0 is less than the minimum 1
"""

from __future__ import annotations

import re
from typing import Any

#: JSON Schema draft 2020-12 for a manifest document.
SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://dkrz.de/blobmap/manifest-v2.schema.json",
    "title": "blobmap manifest",
    "description": (
        "Blob definitions for one scope. Pure definition: this changes only "
        "when the set of blobs changes, never on reads, writes, tiering or "
        "restores."),
    "type": "object",
    "required": ["schema_version", "scope", "epoch", "hot_always", "blobs"],
    "additionalProperties": False,
    "properties": {
        "schema_version": {
            "description": "Format version. Consumers must reject a version "
                           "they do not know rather than guess.",
            "type": "integer",
            "const": 2,
        },
        "scope": {
            "description": "Prefix these definitions apply to, relative to "
                           "the data store.",
            "type": "string",
        },
        "epoch": {
            "description": "Bumped whenever the definitions change, so a "
                           "resolver can detect staleness without diffing.",
            "type": "integer",
            "minimum": 1,
        },
        "generated_at": {
            "description": "ISO 8601 UTC timestamp of the producing run.",
            "type": "string",
        },
        "generated_by": {
            "description": "Package and version that produced this.",
            "type": "string",
        },
        "policy": {
            "description": "Thresholds used to produce this cut, recorded so "
                           "a later run can tell whether they changed.",
            "type": "object",
            "additionalProperties": {"type": "integer"},
        },
        "hot_always": {
            "description": "Glob patterns that are never archivable, "
                           "whatever the blobs say. Metadata objects and "
                           "dimension coordinates.",
            "type": "array",
            "items": {"type": "string"},
        },
        "blobs": {
            "description": "Blob definitions. A list of rules, so its length "
                           "tracks cut decisions rather than object count.",
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "prefixes", "bucket"],
                "additionalProperties": False,
                "properties": {
                    "id": {
                        "description": "Join key against tier state, and "
                                       "through it against tape addresses.",
                        "type": "string",
                        "pattern": "^[a-z0-9_]+$",
                    },
                    "prefixes": {
                        "description": "Key prefixes this blob claims, "
                                       "relative to the scope. More than one "
                                       "when small arrays were coalesced.",
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string"},
                    },
                    "bucket": {
                        "description": "Arithmetic for cutting inside an "
                                       "array. Null means one bucket, so the "
                                       "resolved id is always id_0.",
                        "type": ["object", "null"],
                        "required": ["index", "width", "key_encoding"],
                        "additionalProperties": False,
                        "properties": {
                            "index": {
                                "description": "Which segment after the "
                                               "prefix holds the chunk index.",
                                "type": "integer",
                                "minimum": 0,
                            },
                            "width": {
                                "description": "Objects per blob along that "
                                               "dimension.",
                                "type": "integer",
                                "minimum": 1,
                            },
                            "key_encoding": {
                                "description": "How to parse the index out "
                                               "of a key.",
                                "enum": ["v3_slash", "v2_slash", "v2_flat"],
                            },
                        },
                    },
                },
            },
        },
        "provenance": {
            "description": "Measured numbers, for debugging only. These go "
                           "stale while the manifest is untouched, so a "
                           "resolver must never read them.",
            "type": "object",
        },
    },
}


class SchemaError(ValueError):
    """A document does not match the manifest schema."""


def validate_document(document: Any) -> None:
    """Check a parsed manifest against `SCHEMA`.

    Args:
        document: A parsed JSON document, before it becomes a
            [`Manifest`][blobmap.model.Manifest].

    Raises:
        SchemaError: On the first violation, naming the path to it.

    Example:
        >>> validate_document({"schema_version": 2, "scope": "s", "epoch": 1,
        ...                    "hot_always": [],
        ...                    "blobs": [{"id": "b!", "prefixes": ["x"],
        ...                               "bucket": None}]})
        Traceback (most recent call last):
            ...
        blobmap.schema.SchemaError: blobs[0].id: 'b!' does not match ^[a-z0-9_]+$
    """
    _check(document, SCHEMA, "")


_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "null": (type(None),),
}


def _check(value: Any, schema: dict[str, Any], path: str) -> None:
    """Validate one value against one subschema.

    Args:
        value: The value under inspection.
        schema: The subschema to apply.
        path: Dotted path to `value`, for error messages.

    Raises:
        SchemaError: On the first violation.
    """
    where = path or "document"

    expected = schema.get("type")
    if expected is not None:
        names = [expected] if isinstance(expected, str) else list(expected)
        allowed = tuple(t for name in names for t in _TYPES[name])
        # bool is a subclass of int, but a boolean is never an integer here
        if isinstance(value, bool) and bool not in allowed:
            raise SchemaError(f"{where}: expected {' or '.join(names)}, "
                              f"got boolean")
        if not isinstance(value, allowed):
            raise SchemaError(f"{where}: expected {' or '.join(names)}, got "
                              f"{type(value).__name__}")

    if "const" in schema and value != schema["const"]:
        raise SchemaError(f"{where}: expected {schema['const']!r}, "
                          f"got {value!r}")

    if "enum" in schema and value not in schema["enum"]:
        raise SchemaError(f"{where}: {value!r} is not one of "
                          f"{schema['enum']}")

    if "minimum" in schema and isinstance(value, int) \
            and value < schema["minimum"]:
        raise SchemaError(f"{where}: {value} is less than the minimum "
                          f"{schema['minimum']}")

    if "pattern" in schema and isinstance(value, str) \
            and not re.match(schema["pattern"], value):
        raise SchemaError(f"{where}: {value!r} does not match "
                          f"{schema['pattern']}")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise SchemaError(f"{where}: needs at least "
                              f"{schema['minItems']} item(s)")
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(value):
                _check(item, item_schema, f"{path}[{i}]")

    if isinstance(value, dict):
        for name in schema.get("required", []):
            if name not in value:
                raise SchemaError(f"{where}: missing required field {name!r}")

        properties = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for name, item in value.items():
            child = f"{path}.{name}" if path else name
            if name in properties:
                _check(item, properties[name], child)
            elif extra is False:
                raise SchemaError(f"{where}: unexpected field {name!r}")
            elif isinstance(extra, dict):
                _check(item, extra, child)
