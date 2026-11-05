"""Command line entry point: ``python -m telugu_eval <command>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import dataio


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy code page; force UTF-8 so Telugu
    # in messages does not crash the process.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(prog="python -m telugu_eval")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="validate the dataset files")
    v.add_argument("--data", type=Path, default=dataio.DATA_DIR, help="data directory")
    v.add_argument("--strict", action="store_true", help="fail on warnings too")
    v.add_argument("-q", "--quiet", action="store_true", help="hide warnings")

    a = sub.add_parser("author", help="start the local authoring web form")
    a.add_argument("--data", type=Path, default=dataio.DATA_DIR, help="data directory")
    a.add_argument("--port", type=int, default=8765)
    a.add_argument("--no-browser", action="store_true", help="do not open a browser tab")

    args = ap.parse_args(argv)
    if args.cmd == "validate":
        from .validate import run
        return run(args.data, strict=args.strict, quiet=args.quiet)
    if args.cmd == "author":
        from .author.server import serve
        serve(args.data, port=args.port, open_browser=not args.no_browser)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
