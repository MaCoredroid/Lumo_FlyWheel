#!/usr/bin/env python3
from __future__ import annotations

import argparse

CURRENT_VERIFY_COMMAND = "queue-drain verify-post-drain"
RETIRED_VERIFY_COMMAND = "queue-drain audit-post-drain"


def build_verification_command(cluster: str) -> str:
    return f"{CURRENT_VERIFY_COMMAND} --cluster {cluster} --include-stuck-shards"


def emit_shard_report(cluster: str) -> str:
    return f"python3 repo/scripts/queue_drain_helper.py --emit-shard-report {cluster}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit-verification", action="store_true")
    parser.add_argument("--emit-shard-report")
    parser.add_argument("--cluster", default="atlas-a")
    args = parser.parse_args()
    if args.emit_verification:
        print(build_verification_command(args.cluster))
        return 0
    if args.emit_shard_report:
        print(emit_shard_report(args.emit_shard_report))
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
