from __future__ import annotations

import argparse
import subprocess
import sys

import pytest
from obstore.store import MemoryStore

from blobmap.cli import explain, human, main, parse_size
from blobmap.model import GiB, Array
from blobmap.partition import partition
from tests.zarrgen import ArraySpec, coordinate, write_store


@pytest.mark.parametrize(
    "text,expected",
    [
        ("100", 100),
        ("1K", 1024),
        ("2M", 2 * 1024**2),
        ("100GiB", 100 * GiB),
        ("1.5G", int(1.5 * GiB)),
        ("2TB", 2 * 1024**4),
    ],
)
def test_parse_size(text, expected):
    assert parse_size(text) == expected


def test_human():
    assert human(1536) == "1.5 KiB"
    assert human(5 * GiB) == "5.0 GiB"


def test_explain_labels_every_array():
    arrays = [
        Array("time", (10,), (10,), 8, is_coordinate=True),
        Array("tas", (1000, 4), (10, 4), 4, stored_bytes=400 * GiB),
    ]
    text = explain(arrays, partition("s/", arrays))
    assert "pinned hot" in text and "b_tas_0.." in text


@pytest.fixture
def populated():
    data = MemoryStore()
    write_store(
        data,
        "cordex/a.zarr",
        [coordinate("time", 400), ArraySpec("tas", (400, 4, 4), (10, 4, 4))],
    )
    return data, MemoryStore()


def args(*rest):
    return ["--data", "memory://", "--manifests", "memory://", *rest]


def test_scan_then_partition_then_resolve(populated, capsys):
    data, manifests = populated
    main(args("scan", "--partition"), data=data, manifests=manifests)
    assert "NEW v3  cordex/a.zarr" in capsys.readouterr().out

    main(args("show"), data=data, manifests=manifests)
    assert "cordex/a.zarr" in capsys.readouterr().out

    main(
        args(
            "resolve",
            "cordex/a.zarr/tas/c/0/0/0",
            "cordex/a.zarr/zarr.json",
            "nowhere/at/all",
        ),
        data=data,
        manifests=manifests,
    )
    out = capsys.readouterr().out
    assert "hot" in out and "unmanaged" in out


def test_partition_dry_run(populated, capsys):
    data, manifests = populated
    main(
        args("partition", "cordex/a.zarr", "--t-max", "1K", "--dry-run"),
        data=data,
        manifests=manifests,
    )
    out = capsys.readouterr().out
    assert "not written" in out and "tas" in out


def test_verbose_and_explicit_partition(populated, capsys):
    data, manifests = populated
    main(args("-v", "partition", "cordex/a.zarr"), data=data, manifests=manifests)
    assert "written" in capsys.readouterr().out


def test_imds_timeout_is_explained():
    """The failure mode when no credentials are set: obstore falls back to the
    EC2 metadata service and times out with a Rust backtrace that never
    mentions credentials."""
    from blobmap.backends import IMDS_HOST, diagnose

    message = diagnose(
        "s3://cmip6",
        RuntimeError(
            f"Generic S3 error: Error performing PUT http://{IMDS_HOST}/latest/"
            f"api/token in 13.1s, after 10 retries"
        ),
    )
    assert "no S3 credentials found" in message
    assert "--anonymous" in message and "--endpoint" in message


@pytest.mark.parametrize(
    "text,expected",
    [
        ("NoSuchBucket", "bucket not found"),
        ("SignatureDoesNotMatch", "credentials rejected"),
        ("tcp connect error", "cannot reach the endpoint"),
    ],
)
def test_common_failures_are_translated(text, expected):
    from blobmap.backends import diagnose

    assert expected in diagnose("s3://x", RuntimeError(text))


def test_unrecognised_errors_pass_through():
    from blobmap.backends import diagnose

    assert (
        diagnose("s3://x", RuntimeError("something novel")) == "s3://x: something novel"
    )


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
    assert "client_options" not in S3Options(endpoint="https://s3.example.org").config()


def test_bad_url_exits_nonzero(capsys):
    from blobmap.backends import StoreUnreachable, open_store

    with pytest.raises(StoreUnreachable):
        open_store("not-a-scheme://x")


# -- local paths -----------------------------------------------------------


@pytest.mark.parametrize(
    "given,expected",
    [
        ("/srv/blobmap", "file:///srv/blobmap"),
        ("file:///srv/blobmap", "file:///srv/blobmap"),
        ("s3://cmip6", "s3://cmip6"),
        ("memory://", "memory://"),
    ],
)
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
    write_store(
        open_store(str(data_dir), create=True),
        "cordex/a.zarr",
        [ArraySpec("tas", (40, 4, 4), (10, 4, 4))],
    )

    rc = main(
        [
            "--data",
            str(data_dir),
            "--manifests",
            str(tmp_path / "manifests"),
            "scan",
            "--partition",
        ]
    )
    assert rc == 0
    assert "cordex/a.zarr" in capsys.readouterr().out
    assert (tmp_path / "manifests" / "cordex" / "a.zarr" / "manifest.json").is_file()


def test_missing_data_dir_exits_cleanly(tmp_path, capsys):
    rc = main(
        ["--data", str(tmp_path / "nope"), "--manifests", str(tmp_path / "m"), "scan"]
    )
    assert rc == 2
    assert "does not exist" in capsys.readouterr().err


def test_schema_command_needs_no_stores(capsys):
    """The schema is the contract for consumers who may not have credentials,
    or may not be using Python at all."""
    import json

    assert main(["schema"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["$id"].endswith("manifest-v3.schema.json")


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["scan"], "--data and --manifests are required"),
        (["--data", "memory://", "scan"], "--manifests is required"),
    ],
)
def test_missing_store_arguments(argv, expected, capsys):
    assert main(argv) == 2
    assert expected in capsys.readouterr().err


def test_exclude_flag_is_threaded_through(tmp_path, capsys):
    """--data pointed at a gateway's backing filesystem sees staging dirs the
    S3 API hides."""
    from tests.zarrgen import ArraySpec, write_store
    from blobmap.backends import open_store

    data_dir = tmp_path / "bucket"
    data_dir.mkdir()
    data = open_store(str(data_dir), create=True)
    write_store(data, "real.zarr", [ArraySpec("tas", (40, 4, 4), (10, 4, 4))])
    write_store(
        data, ".sgwtmp/multipart/half.zarr", [ArraySpec("tas", (40, 4, 4), (10, 4, 4))]
    )

    main(["--data", str(data_dir), "--manifests", str(tmp_path / "m"), "scan"])
    out = capsys.readouterr().out
    assert "real.zarr" in out and ".sgwtmp" not in out

    main(
        [
            "--data",
            str(data_dir),
            "--manifests",
            str(tmp_path / "m2"),
            "--exclude",
            "",
            "scan",
        ]
    )
    assert ".sgwtmp" in capsys.readouterr().out


# -- pins ------------------------------------------------------------------


@pytest.fixture
def pinnable(tmp_path):
    from tests.zarrgen import ArraySpec, coordinate, write_store
    from blobmap.backends import open_store

    data_dir = tmp_path / "bucket"
    data_dir.mkdir()
    data = open_store(str(data_dir), create=True)
    write_store(
        data,
        "s.zarr",
        [coordinate("time", 400), ArraySpec("tas", (400, 4, 4), (10, 4, 4))],
    )
    manifests = tmp_path / "m"
    base = ["--data", str(data_dir), "--manifests", str(manifests)]
    main([*base, "partition", "s.zarr"])
    return base


def test_pin_add_show_remove(pinnable, capsys):
    capsys.readouterr()
    main(
        [
            *pinnable,
            "pin",
            "add",
            "s.zarr",
            "tas",
            "--reason",
            "active ICON analysis",
            "--until",
            "2999-01-01",
        ]
    )
    assert "pinned s.zarr/tas" in capsys.readouterr().out

    main([*pinnable, "pin", "show"])
    out = capsys.readouterr().out
    assert "s.zarr/tas" in out and "active ICON analysis" in out
    assert "until 2999-01-01" in out

    main([*pinnable, "pin", "remove", "s.zarr", "tas"])
    out = capsys.readouterr().out
    assert "unpinned" in out
    assert "nothing is archived until" in out

    main([*pinnable, "pin", "show"])
    assert "no pins" in capsys.readouterr().out


def test_pin_show_flags_the_dangerous_ones(pinnable, capsys):
    main([*pinnable, "pin", "add", "s.zarr", "a", "--reason", "open ended"])
    main(
        [
            *pinnable,
            "pin",
            "add",
            "s.zarr",
            "b",
            "--reason",
            "stale",
            "--until",
            "2020-01-01",
        ]
    )
    capsys.readouterr()

    main([*pinnable, "pin", "show"])
    out = capsys.readouterr().out
    assert out.index("EXPIRED") < out.index("no expiry")  # worst first
    assert "2 pin(s) expired or open-ended" in out

    main([*pinnable, "pin", "show", "--expired"])
    assert "stale" in capsys.readouterr().out


def test_pin_add_requires_a_reason(pinnable):
    with pytest.raises(SystemExit):
        main([*pinnable, "pin", "add", "s.zarr", "tas"])


def test_version_reports_the_schema_it_speaks(capsys):
    """The first question when a manifest looks odd is which version wrote
    it, and manifests record generated_by."""
    from blobmap import SCHEMA_VERSION, __version__

    with pytest.raises(SystemExit):
        main(["--version"])
    out = capsys.readouterr().out
    assert __version__ in out and f"v{SCHEMA_VERSION}" in out


def test_report_command(pinnable, capsys):
    capsys.readouterr()
    main([*pinnable, "report"])
    out = capsys.readouterr().out
    assert "archivable" in out and "unmanaged" in out


def test_report_flags_an_unpartitioned_store(pinnable, capsys):
    from tests.zarrgen import ArraySpec, write_store
    from blobmap.backends import open_store

    data_dir = pinnable[1]
    write_store(
        open_store(data_dir),
        "forgotten.zarr",
        [ArraySpec("tas", (400, 4, 4), (10, 4, 4))],
    )
    capsys.readouterr()
    main([*pinnable, "report"])
    assert "Run scan" in capsys.readouterr().out


def test_report_per_bucket(pinnable, capsys):
    capsys.readouterr()
    main([*pinnable, "report", "--per-bucket"])
    assert "scope" in capsys.readouterr().out


def test_a_closed_pipe_is_not_an_error(monkeypatch, capsys):
    """`blobmap schema | head` is normal shell use, not a crash."""
    from blobmap.cli import _run

    def boom():
        raise BrokenPipeError

    monkeypatch.setattr("blobmap.cli.main", boom)
    assert _run() == 0


def test_interrupt_exits_conventionally(monkeypatch, capsys):
    from blobmap.cli import _run

    def boom():
        raise KeyboardInterrupt

    monkeypatch.setattr("blobmap.cli.main", boom)
    assert _run() == 130
    assert "interrupted" in capsys.readouterr().err


# -- help formatting -------------------------------------------------------


def test_subcommands_inherit_the_formatter():
    """add_parser builds a fresh ArgumentParser and inherits parser_class but
    not formatter_class, so without _Parser every subcommand silently falls
    back to plain argparse formatting."""
    from blobmap.cli import _Formatter, build_parser

    parser = build_parser()
    actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    assert actions, "no subparsers found"

    checked = 0
    for action in actions:
        for name, sub in action.choices.items():
            assert sub.formatter_class is _Formatter, name
            checked += 1
            # nested subcommands too, e.g. `pin add`
            for nested in [
                a for a in sub._actions if isinstance(a, argparse._SubParsersAction)
            ]:
                for inner_name, inner in nested.choices.items():
                    assert inner.formatter_class is _Formatter, inner_name
                    checked += 1
    assert checked >= 6


def test_the_epilog_keeps_its_line_breaks(capsys):
    """The epilog is a worked example. A non-raw formatter rewraps it onto one
    line, which is how it was rendering before."""
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "examples:" in out
    assert "scan --partition --dry-run" in out
    # the continuation lines survive as separate lines
    assert out.count("blobmap --data") >= 3


def test_help_works_without_rich_argparse(monkeypatch, capsys):
    """The dependency is optional, so absence must be invisible rather than
    an ImportError at startup."""
    import builtins
    import importlib

    real_import = builtins.__import__

    def no_rich(name, *args, **kwargs):
        if name.startswith("rich_argparse"):
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_rich)
    import blobmap.cli

    reloaded = importlib.reload(blobmap.cli)
    try:
        assert reloaded._Formatter is argparse.RawDescriptionHelpFormatter
        parser = reloaded.build_parser()
        assert parser.format_help()
    finally:
        monkeypatch.undo()
        importlib.reload(blobmap.cli)


def test_module_entry_point():
    """`python -m blobmap` must work, not just the console script."""
    result = subprocess.run(
        [sys.executable, "-m", "blobmap", "--version"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "blobmap" in result.stdout
