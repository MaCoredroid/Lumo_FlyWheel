#!/usr/bin/env python3
"""Verify reproducibility and static gates for the active-depth walk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"summary must be an object: {path}")
    return value, raw


def verify(summary: dict[str, object]) -> None:
    if (
        summary.get("schema")
        != "fr13.fixed32.cfwd_packed_walk.active_depth.sm121a.v1"
        or summary.get("status") != "pass"
    ):
        raise ValueError("active-depth codegen summary status drift")
    work = summary.get("exact_work")
    conclusion = summary.get("conclusion")
    if (
        not isinstance(work, dict)
        or work.get("base_emitted_walk_bodies") != 12
        or work.get("candidate_emitted_walk_bodies") != 1
        or work.get("maximum_iterations_before") != 12
        or work.get("maximum_iterations_after") != 12
        or work.get("topology_size_controls_loop_bound") is not False
        or work.get("output_products_changed") != 0
        or not isinstance(conclusion, dict)
        or conclusion.get("candidate_static_improves_b1_b4") is not True
        or conclusion.get("gpu_execution") is not False
        or conclusion.get("runtime_speedup_claimed") is not False
    ):
        raise ValueError("active-depth exact-work contract drift")
    builds = summary.get("builds")
    if not isinstance(builds, dict) or set(builds) != {"base", "candidate"}:
        raise ValueError("active-depth build matrix drift")
    for batch in ("b1", "b4"):
        base = builds["base"][batch]
        candidate = builds["candidate"][batch]
        if (
            candidate["registers"] > base["registers"]
            or candidate["ldg"] >= base["ldg"]
            or candidate["stg"] > base["stg"]
            or candidate["static_noncontrol_sass_instructions"]
            >= base["static_noncontrol_sass_instructions"]
            or candidate["bra"] != 2
            or any(
                candidate[name] != 0
                for name in ("stack_bytes", "local_bytes", "ldl", "stl", "calls")
            )
        ):
            raise ValueError(f"active-depth static gate failed for {batch}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--rebuild", type=Path, required=True)
    args = parser.parse_args()
    primary, primary_raw = _load(args.primary / "codegen_summary.json")
    rebuild, rebuild_raw = _load(args.rebuild / "codegen_summary.json")
    verify(primary)
    verify(rebuild)
    if primary_raw != rebuild_raw:
        raise ValueError("cold-cache codegen summaries are not byte-identical")
    print("verified active-depth B1/B4 codegen and cold-cache byte identity")


if __name__ == "__main__":
    main()
