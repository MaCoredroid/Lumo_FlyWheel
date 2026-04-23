from __future__ import annotations

import argparse
import json

from release_readiness.export import export_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="release-readiness")
    parser.add_argument("input_path")
    parser.add_argument(
        "--output",
        choices=("json", "markdown"),
        default="markdown",
        help="export format; default keeps the new markdown view front-and-center",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = {"version": 1, "ready": True, "services": ["api", "worker"]}
    rendered = export_report(report, output=args.output)
    if args.output == "json":
        print(json.dumps(rendered, sort_keys=True))
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
