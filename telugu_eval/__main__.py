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
            stream.reconfigure(encoding="utf-8", line_buffering=True)  # live progress output
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

    sub.add_parser("models", help="list configured models and whether their API key is set")

    r = sub.add_parser("run", help="run models over the dataset, saving raw outputs to runs/")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", action="append", help="model name from models.toml (repeatable)")
    g.add_argument("--all", action="store_true", help="all models with enabled = true")
    r.add_argument("--split", choices=("reviewed", "all", "examples"), default="reviewed",
                   help="reviewed (default, what gets scored) | all (drafts too) | examples")
    r.add_argument("--limit", type=int, help="call the model on at most N items (for testing)")
    r.add_argument("--dry-run", action="store_true", help="show what would run and the first prompt; no API calls")
    r.add_argument("--data", type=Path, default=dataio.DATA_DIR, help="data directory")
    r.add_argument("--runs-dir", type=Path, default=None, help="output directory (default: runs/)")

    split_help = "reviewed (default) | all (drafts too, marked preliminary) | examples"
    sc = sub.add_parser("score", help="score saved outputs (offline) -> results/")
    sc.add_argument("--split", choices=("reviewed", "all", "examples"), default="reviewed", help=split_help)
    sc.add_argument("--no-judge", action="store_true", help="ignore cached LLM-judge verdicts")
    sc.add_argument("--data", type=Path, default=dataio.DATA_DIR)
    sc.add_argument("--runs-dir", type=Path, default=None)
    sc.add_argument("--out", type=Path, default=None, help="results directory (default: results/)")

    b = sub.add_parser("build-site", help="score (offline) and export site/data + README results table")
    b.add_argument("--split", choices=("reviewed", "all", "examples"), default="reviewed", help=split_help)
    b.add_argument("--no-judge", action="store_true", help="ignore cached LLM-judge verdicts")
    b.add_argument("--data", type=Path, default=dataio.DATA_DIR)
    b.add_argument("--runs-dir", type=Path, default=None)

    j = sub.add_parser("judge", help="LLM-judge failed short answers (calls the judge model's API; cached)")
    j.add_argument("--judge", default=None, help="judge model name (default: [judge] model in models.toml)")
    j.add_argument("--split", choices=("reviewed", "all", "examples"), default="reviewed", help=split_help)
    j.add_argument("--data", type=Path, default=dataio.DATA_DIR)
    j.add_argument("--runs-dir", type=Path, default=None)

    args = ap.parse_args(argv)
    if args.cmd in ("score", "build-site", "judge"):
        from .config import ConfigError, default_judge, load_dotenv, load_models
        from .runner import RUNS_DIR
        load_dotenv()
        try:
            models = load_models()
        except ConfigError as e:
            print(f"models.toml: {e}")
            return 2
        by_name = {m.name: m for m in models}
        runs_dir = args.runs_dir or RUNS_DIR
        jname = getattr(args, "judge", None) or default_judge()
        if jname and jname not in by_name:
            print(f"judge model {jname!r} is not defined in models.toml")
            return 2
        judge_cfg = by_name.get(jname) if jname else None
        if args.cmd == "judge":
            from .report import judge_models
            if judge_cfg is None:
                print("no judge configured: set [judge] model in models.toml or pass --judge")
                return 2
            return judge_models(models, judge_cfg, args.split, data_dir=args.data, runs_dir=runs_dir)
        from .report import RESULTS_DIR, build_results, export_site, results_markdown, update_readme, write_results
        if args.no_judge:
            judge_cfg = None
        res, per_item = build_results(models, args.split, data_dir=args.data, runs_dir=runs_dir, judge_cfg=judge_cfg)
        if args.cmd == "score":
            out = args.out or RESULTS_DIR
            write_results(res, per_item, out)
            print(results_markdown(res))
            print(f"\nwrote {out / 'results.json'}")
            return 0
        write_results(res, per_item)
        export_site(res, args.data)
        print(results_markdown(res))
        print("\nwrote results/, site/data/" + (", README.md results block" if update_readme(res) else ""))
        return 0

    if args.cmd in ("models", "run"):
        from .config import ConfigError, load_dotenv, load_models
        load_dotenv()
        try:
            models = load_models()
        except ConfigError as e:
            print(f"models.toml: {e}")
            return 2
    if args.cmd == "models":
        print(f"{'name':26}{'provider':19}{'kind':14}{'enabled':9}key")
        for m in models:
            key = "-" if not m.api_key_env else (f"{m.api_key_env} set" if m.api_key() else f"{m.api_key_env} MISSING")
            print(f"{m.name:26}{m.provider:19}{m.kind:14}{str(m.enabled):9}{key}")
        return 0
    if args.cmd == "run":
        from .runner import RUNS_DIR, dry_run, run_model, select_items
        runs_dir = args.runs_dir or RUNS_DIR
        by_name = {m.name: m for m in models}
        if args.all:
            chosen = [m for m in models if m.enabled]
        else:
            unknown = [n for n in args.model if n not in by_name]
            if unknown:
                print(f"unknown model(s): {', '.join(unknown)}. See: python -m telugu_eval models")
                return 2
            chosen = [by_name[n] for n in args.model]
        sel = select_items(args.split, args.data)
        if not sel.items:
            print(f"no items in split '{args.split}'"
                  + (" (mark items as reviewed, or use --split all to include drafts)" if args.split == "reviewed" else ""))
            return 1
        failed = False
        for m in chosen:
            if args.dry_run:
                dry_run(m, sel, runs_dir)
                continue
            # one model hitting its quota does not stop the others (quotas are per model)
            s = run_model(m, sel, runs_dir=runs_dir, limit=args.limit)
            failed |= bool(s.stopped_early)
        return 1 if failed else 0
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
