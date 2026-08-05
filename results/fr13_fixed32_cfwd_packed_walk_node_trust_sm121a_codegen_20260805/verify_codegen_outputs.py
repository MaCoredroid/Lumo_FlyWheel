#!/usr/bin/env python3
"""Verify two fresh-cache trusted-node packed-walk codegen builds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(root: Path) -> dict[str, object]:
    return json.loads((root / "codegen_summary.json").read_text(encoding="ascii"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--rebuild", type=Path, required=True)
    args = parser.parse_args()
    primary = _read(args.primary)
    rebuild = _read(args.rebuild)
    if primary != rebuild:
        raise SystemExit("fresh-cache summaries differ")
    if (
        primary.get("schema")
        != "fr13.fixed32.cfwd_packed_walk.node_trust.sm121a.v1"
        or primary.get("status") != "pass"
    ):
        raise SystemExit("unexpected node-trust codegen schema/status")
    builds = primary["builds"]
    for batch in ("b1", "b4"):
        base = builds["base"][batch]
        candidate = builds["candidate"][batch]
        if not (
            candidate["registers"] < base["registers"]
            and candidate["ldg"] < base["ldg"]
            and candidate["static_noncontrol_sass_instructions"]
            < base["static_noncontrol_sass_instructions"]
            and candidate["stg"] == base["stg"]
            and all(
                build[name] == 0
                for build in (base, candidate)
                for name in ("stack_bytes", "local_bytes", "ldl", "stl", "calls")
            )
        ):
            raise SystemExit(f"node-trust static gate failed for {batch}")
    print(
        json.dumps(
            {
                "schema": (
                    "fr13.fixed32.cfwd_packed_walk.node_trust.verify.v1"
                ),
                "status": "PASS",
                "builds_verified": 4,
                "fresh_cache_byte_identity": True,
                "candidate_static_improves_b1_b4": True,
                "resource_clean": True,
                "gpu_execution": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
