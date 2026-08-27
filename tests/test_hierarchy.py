from __future__ import annotations

import json

import obstore as obs
import pytest

from blobmap import NotAZarrStore, detect_format, read_arrays
from tests.zarrgen import ArraySpec, coordinate, write_store


def basic(**kw) -> list[ArraySpec]:
    return [
        coordinate("time", 40),
        ArraySpec("tas", (40, 4, 4), (10, 4, 4), dimensions=["time", "y", "x"]),
    ]


def test_detects_v3(store):
    write_store(store, "s", basic(), zarr_format=3)
    assert detect_format(store, "s") == "v3"


def test_detects_v2(store):
    write_store(store, "s", basic(), zarr_format=2, separator="/")
    assert detect_format(store, "s") == "v2"


def test_detects_nothing(store):
    obs.put(store, "s/readme.txt", b"hi")
    assert detect_format(store, "s") is None


def test_read_arrays_v3(store):
    write_store(store, "s", basic(), zarr_format=3)
    arrays = {a.path: a for a in read_arrays(store, "s")}
    assert set(arrays) == {"time", "tas"}
    tas = arrays["tas"]
    assert tas.shape == (40, 4, 4) and tas.chunks == (10, 4, 4)
    assert tas.key_encoding == "v3_slash"
    assert tas.nobjects == 4 and tas.nobjects_seen == 4
    assert tas.stored_bytes == 4 * 64
    assert arrays["time"].is_coordinate


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_coordinates_are_detected_in_both_formats(store, zarr_format):
    """v2 keeps _ARRAY_DIMENSIONS in a separate .zattrs object, so it is
    invisible unless read. Missing it makes every v2 coordinate look like a
    plain array, and only the size rule keeps it off tape."""
    write_store(store, "s",
                [coordinate("time", 780),
                 ArraySpec("tas", (780, 4), (780, 4),
                           dimensions=["time", "cell"])],
                zarr_format=zarr_format, separator="/")
    arrays = {a.path: a for a in read_arrays(store, "s")}
    assert arrays["time"].is_coordinate
    assert not arrays["tas"].is_coordinate


def test_datatree_coordinates_are_detected(store):
    """The shape that surfaced this: coordinates nested several groups deep,
    as a datatree produces."""
    deep = "hist-1950/ICON/P1M/mean.zarr/multiscales/zoom_9"
    write_store(store, deep,
                [coordinate("time", 780),
                 ArraySpec("tas", (780, 12), (780, 12),
                           dimensions=["time", "cell"])],
                zarr_format=2, separator="/")
    arrays = {a.path: a for a in read_arrays(store, deep)}
    assert arrays["time"].is_coordinate
    assert not arrays["tas"].is_coordinate


def test_coordinate_without_zattrs_falls_back_to_size(store):
    """Foreign data not written by xarray has no _ARRAY_DIMENSIONS at all.
    The flag reads False, and t_hot is what keeps it hot."""
    write_store(store, "s", [ArraySpec("time", (780,), (780,))],
                zarr_format=2, separator="/")
    time = read_arrays(store, "s")[0]
    assert not time.is_coordinate
    assert time.total_bytes < 1024**3         # under t_hot, so pinned anyway


def test_zattrs_without_array_dimensions_is_harmless(store):
    write_store(store, "s", [ArraySpec("time", (780,), (780,),
                                       dimensions=[])],
                zarr_format=2, separator="/")
    assert not read_arrays(store, "s")[0].is_coordinate


@pytest.mark.parametrize("zarr_format,separator,v2_keys,expected", [
    (3, "/", False, "v3_slash"),
    (3, "/", True, "v2_slash"),
    (3, ".", True, "v2_flat"),
    (2, "/", False, "v2_slash"),
    (2, ".", False, "v2_flat"),
])
def test_key_encodings_round_trip(store, zarr_format, separator, v2_keys,
                                  expected):
    write_store(store, "s", basic(), zarr_format=zarr_format,
                separator=separator, v2_keys=v2_keys)
    arrays = {a.path: a for a in read_arrays(store, "s")}
    assert arrays["tas"].key_encoding == expected


def test_sharded_array_reports_the_shard_as_the_object(store):
    spec = ArraySpec("tas", (40, 4, 4), (20, 4, 4), inner_chunks=(10, 4, 4))
    write_store(store, "s", [spec], zarr_format=3)
    tas = read_arrays(store, "s")[0]
    assert tas.shards == (20, 4, 4) and tas.chunks == (10, 4, 4)
    assert tas.object_shape == (20, 4, 4)
    assert tas.nobjects == 2


def test_array_roots_come_from_metadata_not_key_shape(store):
    """A group literally named 'c', which the old key-shape heuristic would
    have mistaken for a v3 chunk prefix."""
    write_store(store, "s/c", basic(), zarr_format=3)
    obs.put(store, "s/zarr.json",
            json.dumps({"zarr_format": 3, "node_type": "group"}).encode())
    paths = {a.path for a in read_arrays(store, "s")}
    assert paths == {"c/time", "c/tas"}


def test_nested_groups(store):
    write_store(store, "s/deep/nest", basic(), zarr_format=3)
    obs.put(store, "s/zarr.json",
            json.dumps({"zarr_format": 3, "node_type": "group"}).encode())
    assert {a.path for a in read_arrays(store, "s")} \
        == {"deep/nest/time", "deep/nest/tas"}


def test_partially_written_array(store):
    spec = ArraySpec("tas", (40, 4, 4), (10, 4, 4), written=2)
    write_store(store, "s", [spec], zarr_format=3)
    tas = read_arrays(store, "s")[0]
    assert tas.nobjects == 4 and tas.nobjects_seen == 2


def test_stray_objects_are_reported_not_swallowed(store, caplog):
    write_store(store, "s", basic(), zarr_format=3)
    obs.put(store, "s/stray.bin", b"x" * 5)
    with caplog.at_level("WARNING"):
        read_arrays(store, "s")
    assert "belong to no array" in caplog.text


def test_declared_encoding_loses_to_observed(store, caplog):
    """Foreign stores do not always match their own declared metadata, and a
    wrong encoding silently misroutes every chunk."""
    write_store(store, "s", basic(), zarr_format=2, separator="/")
    meta = json.loads(bytes(obs.get(store, "s/tas/.zarray").bytes()))
    meta["dimension_separator"] = "."          # a lie: keys are nested
    obs.put(store, "s/tas/.zarray", json.dumps(meta).encode())
    with caplog.at_level("WARNING"):
        arrays = {a.path: a for a in read_arrays(store, "s")}
    assert arrays["tas"].key_encoding == "v2_slash"
    assert "keys look like" in caplog.text


def test_empty_prefix_raises(store):
    obs.put(store, "s/readme.txt", b"hi")
    with pytest.raises(NotAZarrStore):
        read_arrays(store, "s")


def test_unreadable_metadata_is_skipped(store, caplog):
    write_store(store, "s", basic(), zarr_format=3)
    obs.put(store, "s/broken/zarr.json", b"{not json")
    with caplog.at_level("WARNING"):
        arrays = read_arrays(store, "s")
    assert {a.path for a in arrays} == {"time", "tas"}
    assert "not valid JSON" in caplog.text


def test_unknown_dtype_is_skipped(store, caplog):
    write_store(store, "s", basic(), zarr_format=3)
    meta = json.loads(bytes(obs.get(store, "s/tas/zarr.json").bytes()))
    meta["data_type"] = {"name": "something_exotic"}
    obs.put(store, "s/tas/zarr.json", json.dumps(meta).encode())
    with caplog.at_level("WARNING"):
        arrays = read_arrays(store, "s")
    assert {a.path for a in arrays} == {"time"}


def test_numpy_style_dtypes(store):
    write_store(store, "s", basic(), zarr_format=2, separator="/")
    tas = {a.path: a for a in read_arrays(store, "s")}["tas"]
    assert tas.itemsize == 4


def test_find_store_root_climbs_to_the_outermost_store(store):
    from blobmap import find_store_root
    write_store(store, "cordex/a.zarr", basic(), zarr_format=3)
    assert find_store_root(store, "cordex/a.zarr/tas") == "cordex/a.zarr"
    assert find_store_root(store, "cordex/a.zarr") == "cordex/a.zarr"


def test_find_store_root_returns_none_outside_a_store(store):
    from blobmap import find_store_root
    obs.put(store, "junk/readme.txt", b"hi")
    assert find_store_root(store, "junk") is None


def test_find_store_root_respects_a_ceiling(store):
    from blobmap import find_store_root
    write_store(store, "a/b.zarr", basic(), zarr_format=3)
    assert find_store_root(store, "a/b.zarr/tas", ceiling="a/b.zarr") \
        == "a/b.zarr/tas"


def test_metadata_that_is_not_an_array_is_ignored(store):
    """Group markers and .zattrs describe structure, not data."""
    write_store(store, "s", basic(), zarr_format=3)
    obs.put(store, "s/agroup/zarr.json",
            json.dumps({"zarr_format": 3, "node_type": "group"}).encode())
    assert {a.path for a in read_arrays(store, "s")} == {"time", "tas"}


def test_missing_metadata_object_is_tolerated(store):
    """A LIST that raced a delete must not crash the partitioner."""
    from blobmap.hierarchy import _load
    assert _load(store, "does/not/exist.json") is None


def test_more_objects_than_the_grid_allows_warns(store, caplog):
    """A leftover chunk from an append or rechunk. It skews
    avg_object_bytes and therefore the bucket width, so say so."""
    write_store(store, "s", [ArraySpec("time", (780,), (780,))],
                zarr_format=2, separator="/")
    obs.put(store, "s/time/1", b"\0" * 64)        # stale, outside the grid
    with caplog.at_level("WARNING"):
        time = read_arrays(store, "s")[0]
    assert time.nobjects == 1 and time.nobjects_seen == 2
    assert "stale chunks" in caplog.text


# -- gateway artefacts -----------------------------------------------------

def test_gateway_staging_is_excluded(store, caplog):
    """Versity's posix backend stages multipart uploads under .sgwtmp. That is
    invisible over S3 but shows up when scanning the backing filesystem, and
    staging objects inside a store prefix would be counted into an array's
    size and skew the chosen bucket width."""
    write_store(store, "s", basic(), zarr_format=3)
    obs.put(store, "s/.sgwtmp/multipart/staging-abc", b"\0" * 10_000)
    obs.put(store, "s/tas/.sgwtmp/leftover", b"\0" * 10_000)

    with caplog.at_level("WARNING"):
        arrays = {a.path: a for a in read_arrays(store, "s")}

    assert arrays["tas"].stored_bytes == 4 * 64      # staging not counted
    assert "belong to no array" not in caplog.text


def test_exclusion_can_be_turned_off(store):
    write_store(store, "s", basic(), zarr_format=3)
    obs.put(store, "s/tas/.sgwtmp/leftover", b"\0" * 10_000)
    arrays = {a.path: a for a in read_arrays(store, "s", exclude=())}
    assert arrays["tas"].stored_bytes == 4 * 64 + 10_000


@pytest.mark.parametrize("key,expected", [
    (".sgwtmp/multipart/x", True),
    ("a/.sgwtmp/x", True),
    (".versitygw/meta", True),
    ("a/.snapshot/b", True),
    ("healpix/mean.zarr/tas/0/0", False),
    ("healpix/data.sgwtmp.nc", False),      # named like it, but not a segment
])
def test_excluded_matches_whole_segments(key, expected):
    from blobmap import excluded
    assert excluded(key) is expected


# -- scale -----------------------------------------------------------------

def many_chunks(store, n: int, scope: str = "s") -> None:
    """A store shaped like HEALPix hourly data: one chunk per timestep."""
    meta = {"zarr_format": 2, "shape": [n, 3_145_728],
            "chunks": [1, 3_145_728], "dtype": "<f4", "compressor": None,
            "fill_value": 0, "order": "C", "filters": None,
            "dimension_separator": "/"}
    obs.put(store, f"{scope}/.zgroup", b'{"zarr_format": 2}')
    obs.put(store, f"{scope}/t2m/.zarray", json.dumps(meta).encode())
    obs.put(store, f"{scope}/t2m/.zattrs",
            b'{"_ARRAY_DIMENSIONS": ["time", "cell"]}')
    for i in range(n):
        obs.put(store, f"{scope}/t2m/{i}/0", b"\0" * 8)


def test_memory_does_not_scale_with_object_count(store):
    """Holding every listed object cost ~200 bytes each, so a store with
    millions of chunks consumed gigabytes and looked like a hang. The listing
    is streamed instead, twice, so memory tracks the number of arrays."""
    import tracemalloc

    peaks = []
    for n in (2_000, 8_000):
        many_chunks(store, n, scope=f"s{n}")
        tracemalloc.start()
        arrays = read_arrays(store, f"s{n}")
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert arrays[0].nobjects_seen == n
        peaks.append(peak)

    # 4x the objects must not mean 4x the memory
    assert peaks[1] < peaks[0] * 2, f"peaks scaled: {peaks}"


def test_progress_is_logged_for_large_stores(store, caplog, monkeypatch):
    """A silent multi-minute walk is indistinguishable from a hang."""
    import blobmap.hierarchy as h
    monkeypatch.setattr(h, "PROGRESS_EVERY", 100)
    many_chunks(store, 250)
    with caplog.at_level("INFO"):
        read_arrays(store, "s")
    assert "objects listed" in caplog.text
    assert "objects sized" in caplog.text


# -- ambiguous key encodings ----------------------------------------------

def test_one_dimensional_keys_do_not_override_metadata(store, caplog):
    """`cell/0` is what both v2 encodings produce for a single-chunk 1-D
    array, so it is not evidence of either. Overriding on that basis produced
    a warning for every coordinate in a HEALPix store."""
    write_store(store, "s", [coordinate("cell", 12), coordinate("time", 40)],
                zarr_format=2, separator=".")
    with caplog.at_level("WARNING"):
        arrays = {a.path: a for a in read_arrays(store, "s")}
    assert arrays["cell"].key_encoding == "v2_flat"
    assert "keys look like" not in caplog.text


def test_multidimensional_keys_still_override_metadata(store, caplog):
    """With more than one dimension, `0/0` versus `0.0` is decisive, and a
    store that lies about its own separator would misroute every chunk."""
    write_store(store, "s", basic(), zarr_format=2, separator="/")
    meta = json.loads(bytes(obs.get(store, "s/tas/.zarray").bytes()))
    meta["dimension_separator"] = "."
    obs.put(store, "s/tas/.zarray", json.dumps(meta).encode())
    with caplog.at_level("WARNING"):
        arrays = {a.path: a for a in read_arrays(store, "s")}
    assert arrays["tas"].key_encoding == "v2_slash"
    assert "keys look like" in caplog.text
