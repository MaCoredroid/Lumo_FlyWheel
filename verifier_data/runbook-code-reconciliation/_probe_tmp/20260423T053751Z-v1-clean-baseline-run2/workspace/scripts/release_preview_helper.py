#!/usr/bin/env python3
from __future__ import annotations

import argparse

CURRENT_COMMAND = "python src/release_preview/cli.py generate --config configs/release_preview.toml"
LEGACY_ALIAS = "build-preview"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="release-preview-helper")
    sub = parser.add_subparsers(dest="command")
    build = sub.add_parser("build-preview", help="Deprecated compatibility alias")
    build.add_argument("--settings", dest="settings")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command != LEGACY_ALIAS:
        parser.print_help()
        return 0
    print("deprecated_alias=true")
    print(f"compatibility_forward_to={CURRENT_COMMAND}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
