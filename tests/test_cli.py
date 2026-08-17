from __future__ import annotations

import pytest
from obstore.store import MemoryStore

from blobmap.cli import explain, human, main, parse_size
from blobmap.model import GiB, Array, Manifest
from blobmap.partition import partition
from tests.zarrgen import ArraySpec, coordinate, write_store


@pytest.mark.parametrize("text,expected", [
    ("100", 100), ("1K", 1024), ("2M", 2 * 1024**2),
    ("100GiB", 100 * GiB), ("1.5G", int(1.5 * GiB)), ("2TB", 2 * 1024**4),
])
def test_parse_size(text, expected):
    assert parse_size(text) == expected


def test_human():
    assert human(1536) == "1.5 KiB"
    assert human(5 * GiB) == "5.0 GiB"


def test_explain_labels_every_array():
    arrays = [Array("time", (10,), (10,), 8, is_coordinate=True),
              Array("tas", (1000, 4), (10, 4), 4, stored_bytes=400 * GiB)]
    text = explain(arrays, partition("s/", arrays))
    assert "pinned hot" in text and "b_tas_0.." in text


@pytest.fixture
def populated():
    data = MemoryStore()
    write_store(data, "cordex/a.zarr",
                [coordinate("time", 400),
                 ArraySpec("tas", (400, 4, 4), (10, 4, 4))])
    return data, MemoryStore()


def args(*rest):
    return ["--data", "memory://", "--manifests", "memory://", *rest]


def test_scan_then_partition_then_resolve(populated, capsys):
    data, manifests = populated
    main(args("scan", "--partition"), data=data, manifests=manifests)
    assert "NEW v3  cordex/a.zarr" in capsys.readouterr().out

    main(args("show"), data=data, manifests=manifests)
    assert "cordex/a.zarr" in capsys.readouterr().out

    main(args("resolve", "cordex/a.zarr/tas/c/0/0/0",
              "cordex/a.zarr/zarr.json", "nowhere/at/all"),
         data=data, manifests=manifests)
    out = capsys.readouterr().out
    assert "hot" in out and "unmanaged" in out


def test_partition_dry_run(populated, capsys):
    data, manifests = populated
    main(args("partition", "cordex/a.zarr", "--t-max", "1K", "--dry-run"),
         data=data, manifests=manifests)
    out = capsys.readouterr().out
    assert "not written" in out and "tas" in out


def test_verbose_and_explicit_partition(populated, capsys):
    data, manifests = populated
    main(args("-v", "partition", "cordex/a.zarr"), data=data,
         manifests=manifests)
    assert "written" in capsys.readouterr().out


def test_imds_timeout_is_explained():
    """The failure mode when no credentials are set: obstore falls back to the
    EC2 metadata service and times out with a Rust backtrace that never
    mentions credentials."""
    from blobmap.backends import IMDS_HOST, diagnose
    message = diagnose("s3://cmip6", RuntimeError(
        f"Generic S3 error: Error performing PUT http://{IMDS_HOST}/latest/"
        f"api/token in 13.1s, after 10 retries"))
    assert "no S3 credentials found" in message
    assert "--anonymous" in message and "--endpoint" in message


@pytest.mark.parametrize("text,expected", [
    ("NoSuchBucket", "bucket not found"),
    ("SignatureDoesNotMatch", "credentials rejected"),
    ("tcp connect error", "cannot reach the endpoint"),
])
def test_common_failures_are_translated(text, expected):
    from blobmap.backends import diagnose
    assert expected in diagnose("s3://x", RuntimeError(text))


def test_unrecognised_errors_pass_through():
    from blobmap.backends import diagnose
    assert diagnose("s3://x", RuntimeError("something novel")) \
        == "s3://x: something novel"


def test_s3_options_only_sets_what_was_given():
    from blobmap.backends import S3Options
    assert S3Options().config() == {}
    assert S3Options(anonymous=True).config() == {"skip_signature": True}


def test_http_endpoint_implies_allow_http():
    """Otherwise obstore refuses the endpoint and the error says nothing about
    why."""
    from blobmap.backends import S3Options
    config = S3Options(endpoint="http://localhost:9000").config()
    assert config["client_options"] == {"allow_http": True}
    assert "client_options" not in S3Options(
        endpoint="https://s3.example.org").config()


def test_bad_url_exits_nonzero(capsys):
    from blobmap.backends import StoreUnreachable, open_store
    with pytest.raises(StoreUnreachable):
        open_store("not-a-scheme://x")


# -- local paths -----------------------------------------------------------

@pytest.mark.parametrize("given,expected", [
    ("/srv/blobmap", "file:///srv/blobmap"),
    ("file:///srv/blobmap", "file:///srv/blobmap"),
    ("s3://cmip6", "s3://cmip6"),
    ("memory://", "memory://"),
])
def test_normalise(given, expected):
    from blobmap.backends import normalise
    assert normalise(given) == expected


def test_relative_paths_become_absolute(tmp_path, monkeypatch):
    from blobmap.backends import normalise
    monkeypatch.chdir(tmp_path)
    assert normalise("data") == (tmp_path / "data").as_uri()
    assert normalise("./data") == (tmp_path / "data").as_uri()


def test_tilde_is_expanded(monkeypatch, tmp_path):
    from blobmap.backends import local_path
    monkeypatch.setenv("HOME", str(tmp_path))
    assert local_path("~/blobmap") == tmp_path / "blobmap"


def test_manifest_dir_is_created(tmp_path):
    """A local manifest store is ours to create; obstore otherwise refuses a
    missing directory with a canonicalisation error."""
    from blobmap.backends import open_store
    target = tmp_path / "deep" / "nested" / "blobmap"
    open_store(str(target), create=True)
    assert target.is_dir()


def test_data_dir_is_not_created(tmp_path):
    """Silently creating a mistyped --data would make scan report nothing
    found, which looks exactly like an empty bucket."""
    from blobmap.backends import StoreUnreachable, open_store
    missing = tmp_path / "typo"
    with pytest.raises(StoreUnreachable, match="does not exist"):
        open_store(str(missing))
    assert not missing.exists()


def test_a_file_where_a_directory_belongs(tmp_path):
    from blobmap.backends import StoreUnreachable, open_store
    target = tmp_path / "notadir"
    target.write_text("oops")
    with pytest.raises(StoreUnreachable, match="is a file"):
        open_store(str(target), create=True)


def test_uncreatable_path_is_reported(tmp_path):
    from blobmap.backends import StoreUnreachable, open_store
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    with pytest.raises(StoreUnreachable):
        open_store(str(blocker / "under" / "a" / "file"), create=True)


def test_end_to_end_with_bare_paths(tmp_path, capsys):
    """The shape people actually type."""
    from tests.zarrgen import ArraySpec, write_store
    from blobmap.backends import open_store

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_store(open_store(str(data_dir), create=True), "cordex/a.zarr",
                [ArraySpec("tas", (40, 4, 4), (10, 4, 4))])

    rc = main(["--data", str(data_dir),
               "--manifests", str(tmp_path / "manifests"),
               "scan", "--partition"])
    assert rc == 0
    assert "cordex/a.zarr" in capsys.readouterr().out
    assert (tmp_path / "manifests" / "cordex" / "a.zarr"
            / "manifest.json").is_file()


def test_missing_data_dir_exits_cleanly(tmp_path, capsys):
    rc = main(["--data", str(tmp_path / "nope"),
               "--manifests", str(tmp_path / "m"), "scan"])
    assert rc == 2
    assert "does not exist" in capsys.readouterr().err


def test_schema_command_needs_no_stores(capsys):
    """The schema is the contract for consumers who may not have credentials,
    or may not be using Python at all."""
    import json
    assert main(["schema"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["$id"].endswith("manifest-v2.schema.json")


@pytest.mark.parametrize("argv,expected", [
    (["scan"], "--data and --manifests are required"),
    (["--data", "memory://", "scan"], "--manifests is required"),
])
def test_missing_store_arguments(argv, expected, capsys):
    assert main(argv) == 2
    assert expected in capsys.readouterr().err


def test_exclude_flag_is_threaded_through(tmp_path, capsys):
    """--data pointed at a gateway's backing filesystem sees staging dirs the
    S3 API hides."""
    import obstore as obs
    from tests.zarrgen import ArraySpec, write_store
    from blobmap.backends import open_store

    data_dir = tmp_path / "bucket"
    data_dir.mkdir()
    data = open_store(str(data_dir), create=True)
    write_store(data, "real.zarr", [ArraySpec("tas", (40, 4, 4), (10, 4, 4))])
    write_store(data, ".sgwtmp/multipart/half.zarr",
                [ArraySpec("tas", (40, 4, 4), (10, 4, 4))])

    main(["--data", str(data_dir), "--manifests", str(tmp_path / "m"), "scan"])
    out = capsys.readouterr().out
    assert "real.zarr" in out and ".sgwtmp" not in out

    main(["--data", str(data_dir), "--manifests", str(tmp_path / "m2"),
          "--exclude", "", "scan"])
    assert ".sgwtmp" in capsys.readouterr().out
