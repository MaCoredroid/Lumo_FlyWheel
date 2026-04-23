#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

CURRENT_FLAG = "--config"
DEPRECATED_FLAG = "--settings"
CURRENT_ENV = "RELEASE_PREVIEW_CONFIG"
DEPRECATED_ENV = "PREVIEW_SETTINGS_PATH"
DEFAULT_CONFIG = "configs/release_preview.toml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="release-preview")
    sub = parser.add_subparsers(dest="command")
    generate = sub.add_parser("generate", help="Generate the daily release preview")
    generate.add_argument(CURRENT_FLAG, dest="config")
    generate.add_argument(DEPRECATED_FLAG, dest="deprecated_settings", help=argparse.SUPPRESS)
    generate.add_argument("--dry-run", action="store_true")
    return parser


def resolve_config(args: argparse.Namespace) -> str:
    return (
        args.config
        or os.environ.get(CURRENT_ENV)
        or args.deprecated_settings
        or os.environ.get(DEPRECATED_ENV)
        or DEFAULT_CONFIG
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command != "generate":
        parser.print_help()
        return 0
    print(f"entrypoint=python src/release_preview/cli.py generate")
    print(f"config={resolve_config(args)}")
    if args.dry_run:
        print("mode=dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
