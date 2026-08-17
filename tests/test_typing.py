"""The Store alias should actually constrain, not just decorate.

Without a check like this, `Store` could silently drift back to `Any` -- for
instance if someone re-exported it from a module that imports obstore lazily,
or annotated a new function with `Any` out of habit.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest

import blobmap
from blobmap.storage import Store

ROOT = Path(__file__).resolve().parent.parent

#: Parameters that carry a storage handle. Anything named like this must be
#: annotated `Store`, never `Any`.
HANDLE_NAMES = {"store", "data"}

FUNCTIONS = [
    blobmap.detect_format, blobmap.find_store_root, blobmap.read_arrays,
    blobmap.partition_store, blobmap.scan,
    blobmap.ManifestStore.__init__,
]


@pytest.mark.parametrize("fn", FUNCTIONS, ids=lambda f: f.__qualname__)
def test_handle_parameters_are_annotated(fn):
    hints = inspect.get_annotations(fn, eval_str=False)
    for name in HANDLE_NAMES & set(inspect.signature(fn).parameters):
        assert hints.get(name) == "Store", (
            f"{fn.__qualname__}({name}) is {hints.get(name)!r}, expected 'Store'")


def test_store_is_a_closed_union_of_obstore_backends():
    """Documented limitation: a wrapper around a store will not satisfy this.
    If that ever becomes wanted, switch storage.py to calling store *methods*
    and make Store a Protocol."""
    from obstore.store import LocalStore, MemoryStore, S3Store
    members = set(Store.__args__)
    assert {MemoryStore, LocalStore, S3Store} <= members


def test_real_backends_satisfy_the_alias():
    from obstore.store import MemoryStore
    assert isinstance(MemoryStore(), Store.__args__)


def run_mypy(probe: Path, cache: Path) -> subprocess.CompletedProcess[str]:
    """Type-check one file.

    The cache goes somewhere disposable rather than into the repository: mypy
    mmaps its cache, so a cache directory that is cleaned while it is running
    kills the process with SIGBUS and no output at all.

    Args:
        probe: File to check.
        cache: Scratch cache directory.

    Returns:
        The completed process.
    """
    return subprocess.run(
        [sys.executable, "-m", "mypy", "--cache-dir", str(cache), str(probe)],
        capture_output=True, text=True, cwd=ROOT)


def check(probe: Path, cache: Path) -> str:
    """Type-check, skipping if mypy could not run at all.

    A crash with no output means the environment, not the annotation. Failing
    the suite for that would train people to ignore these tests.
    """
    result = run_mypy(probe, cache)
    if result.returncode < 0 or (result.returncode and not result.stdout):
        pytest.skip(f"mypy could not run here: rc={result.returncode} "
                    f"{result.stderr[:200]}")
    return result.stdout


def test_mypy_rejects_a_non_store(tmp_path):
    """The check that proves the annotation has teeth."""
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from blobmap import read_arrays\n"
        "read_arrays('not a store', 'scope')\n")
    assert "arg-type" in check(probe, tmp_path / "cache")


def test_mypy_accepts_a_real_store(tmp_path):
    probe = tmp_path / "ok.py"
    probe.write_text(
        "from obstore.store import MemoryStore\n"
        "from blobmap import read_arrays\n"
        "read_arrays(MemoryStore(), 'scope')\n")
    assert "no issues" in check(probe, tmp_path / "cache")
