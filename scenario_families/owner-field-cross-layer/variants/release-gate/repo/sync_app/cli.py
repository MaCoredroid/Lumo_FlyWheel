from __future__ import annotations

import argparse
import json

from sync_app.service import sync_item


def main(argv: list[str] | None = None) -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--status", required=True)
    args = parser.parse_args(argv)
    payload = sync_item(args.name, args.status)
    return json.dumps(payload, sort_keys=True)


if __name__ == "__main__":
    print(main())
