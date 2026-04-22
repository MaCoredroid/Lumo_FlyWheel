from __future__ import annotations

import argparse
import sys

from watchlist_report.renderers.json_renderer import render_json
from watchlist_report.renderers.markdown_renderer import render_markdown
from watchlist_report.service import build_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the watchlist report.")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--include-watchlist", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    include_watchlist = args.include_watchlist if args.format == "json" else False
    report = build_report(include_watchlist=include_watchlist)
    if args.format == "json":
        sys.stdout.write(render_json(report))
    else:
        sys.stdout.write(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
