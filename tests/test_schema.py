"""The schema is the contract, so it has to actually be enforced.

Two things are checked here. First that our walker agrees with a real
JSON Schema implementation, which is what stops the schema growing a keyword
the walker silently ignores. Second that the schema and the dataclasses agree,
so the published contract cannot drift from what the code produces.
"""

from __future__ import annotations

import json

import pytest

from blobmap import Blob, Bucket, Manifest, Policy
from blobmap.schema import SCHEMA, SchemaError, validate_document

jsonschema = pytest.importorskip("jsonschema")


def document(**overrides) -> dict:
    base = {
        "schema_version": 3,
        "scope": "cordex/a.zarr",
        "epoch": 1,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "generated_by": "blobmap 0.4.0",
        "policy": Policy().to_json(),
        "hot_always": ["**/zarr.json"],
        "blobs": [
            {"id": "b_pr", "prefixes": ["pr/c"], "bucket": None},
            {"id": "b_tas", "prefixes": ["tas/c"],
             "bucket": {"index": 0, "width": 2048,
                        "key_encoding": "v3_slash"}},
        ],
        "provenance": {},
    }
    return {**base, **overrides}


INVALID = {
    "bad version": document(schema_version=2),
    "missing scope": {k: v for k, v in document().items() if k != "scope"},
    "epoch zero": document(epoch=0),
    "scope not a string": document(scope=42),
    "unknown top level field": document(surprise=1),
    "blob id with punctuation": document(
        blobs=[{"id": "b-tas", "prefixes": ["x"], "bucket": None}]),
    "blob missing bucket key": document(
        blobs=[{"id": "b_tas", "prefixes": ["x"]}]),
    "blob with no prefixes": document(
        blobs=[{"id": "b_tas", "prefixes": [], "bucket": None}]),
    "unknown key encoding": document(
        blobs=[{"id": "b_tas", "prefixes": ["x"],
                "bucket": {"index": 0, "width": 8, "key_encoding": "v9"}}]),
    "zero width": document(
        blobs=[{"id": "b_tas", "prefixes": ["x"],
                "bucket": {"index": 0, "width": 0,
                           "key_encoding": "v3_slash"}}]),
    "negative index": document(
        blobs=[{"id": "b_tas", "prefixes": ["x"],
                "bucket": {"index": -1, "width": 8,
                           "key_encoding": "v3_slash"}}]),
    "extra field in bucket": document(
        blobs=[{"id": "b_tas", "prefixes": ["x"],
                "bucket": {"index": 0, "width": 8,
                           "key_encoding": "v3_slash", "extra": 1}}]),
    "boolean where integer expected": document(epoch=True),
    "hot_always not strings": document(hot_always=[1, 2]),
}


def test_schema_is_itself_valid():
    jsonschema.Draft202012Validator.check_schema(SCHEMA)


def test_valid_document_passes_both():
    validate_document(document())
    jsonschema.validate(document(), SCHEMA)


@pytest.mark.parametrize("case", sorted(INVALID), ids=sorted(INVALID))
def test_walker_agrees_with_jsonschema(case):
    """If these ever disagree, the walker is ignoring a keyword the schema
    relies on, and the published contract is stricter than the code."""
    bad = INVALID[case]
    with pytest.raises(SchemaError):
        validate_document(bad)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, SCHEMA)


def test_optional_fields_may_be_absent():
    minimal = {"schema_version": 3, "scope": "s", "epoch": 1,
               "hot_always": [], "blobs": []}
    validate_document(minimal)
    jsonschema.validate(minimal, SCHEMA)
    assert Manifest.from_json(minimal).scope == "s"


def test_errors_name_the_path():
    with pytest.raises(SchemaError, match=r"blobs\[1\]\.bucket\.width"):
        validate_document(document(blobs=[
            {"id": "b_a", "prefixes": ["a"], "bucket": None},
            {"id": "b_b", "prefixes": ["b"],
             "bucket": {"index": 0, "width": 0, "key_encoding": "v3_slash"}}]))


# -- schema and dataclasses must agree -------------------------------------

def test_what_the_code_writes_validates():
    manifest = Manifest(
        "cordex/a.zarr",
        (Blob("b_tas", ("tas/c",), Bucket(0, 2048, "v3_slash")),
         Blob("b_pr_hurs", ("pr/c", "hurs/c"))),
        ("**/zarr.json", "time/**"),
        provenance={"tas": {"objects": 10266}})
    jsonschema.validate(json.loads(manifest.dumps()), SCHEMA)


def test_every_key_encoding_is_in_the_schema():
    from blobmap.model import KEY_ENCODINGS
    allowed = SCHEMA["properties"]["blobs"]["items"]["properties"]["bucket"] \
        ["properties"]["key_encoding"]["enum"]
    assert set(allowed) == set(KEY_ENCODINGS)


def test_schema_version_constant_matches():
    from blobmap.model import SCHEMA_VERSION
    assert SCHEMA["properties"]["schema_version"]["const"] == SCHEMA_VERSION


def test_every_manifest_field_is_described():
    """A field nobody documented is a field a consumer has to guess at."""
    for name, spec in SCHEMA["properties"].items():
        assert spec.get("description"), f"{name} has no description"


def test_cross_field_rules_are_left_to_validate():
    """JSON Schema cannot express these, so they live in Manifest.validate
    and must still be enforced."""
    duplicate = document(blobs=[
        {"id": "b_a", "prefixes": ["a"], "bucket": None},
        {"id": "b_a", "prefixes": ["b"], "bucket": None}])
    jsonschema.validate(duplicate, SCHEMA)       # schema is happy
    with pytest.raises(ValueError, match="duplicate"):
        Manifest.from_json(duplicate)

    multi = document(blobs=[
        {"id": "b_a", "prefixes": ["a", "b"],
         "bucket": {"index": 0, "width": 8, "key_encoding": "v3_slash"}}])
    jsonschema.validate(multi, SCHEMA)
    with pytest.raises(ValueError, match="exactly one prefix"):
        Manifest.from_json(multi)
