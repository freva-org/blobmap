"""Blobmap command line interface."""

import sys

if __name__ == "__main__":
    from blobmap.cli import _run

    sys.exit(_run())
