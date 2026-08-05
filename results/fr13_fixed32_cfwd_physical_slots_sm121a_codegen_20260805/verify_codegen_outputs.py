#!/usr/bin/env python3
"""Verify two independent fixed32 physical-slot SM121a codegen builds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BASE = "dfb04b3b20e246118006ab2f4cb91a4a196f2491"
CANDIDATE = "d2348ce9260292dcf6f9c687a774ed9966b92928"
BLOCK_STATS = "_fr13_cfwd_logit_block_stats_kernel"
DIRECT_DECISION = "_fr13_cfwd_logit_direct_decision_kernel"


def load(path: Path) -> dict:
    return json.loads((path / "codegen_summary.json").read_text(encoding="ascii"))


def resource_clean(build: dict) -> bool:
    return all(
        build[name] == 0
        for name in ("stack_bytes", "local_bytes", "ldl", "stl", "calls")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--rebuild", type=Path, required=True)
    args = parser.parse_args()
    primary = load(args.primary)
    rebuild = load(args.rebuild)
    if primary != rebuild:
        raise SystemExit("independent fresh-cache codegen summaries differ")
    if (
        primary["schema"]
        != "fr13.fixed32.cfwd_physical_slots.sm121a.codegen.v1"
        or primary["status"] != "pass"
        or primary["base_revision"] != BASE
        or primary["candidate_revision"] != CANDIDATE
        or primary["claim_scope"]
        != "static_sm121a_codegen_and_exact_work_only_no_runtime_speed_claim"
    ):
        raise SystemExit("codegen summary identity or claim scope drift")
    contract = primary["compile_contract"]
    if contract != {
        "batches": [1, 4],
        "commit_programs_per_request": 1,
        "fanout": 3,
        "jit_specialization": (
            "exact_signature_or_mock_tensor_shape_stride_alignment"
        ),
        "physical_drafts": 31,
        "physical_rows": 32,
        "producer_programs_per_request": {
            "block_stats": 1_830,
            "direct_decision": 30,
        },
        "self_rows": 13,
        "target": "sm_121a",
        "target_rows": 17,
        "vocab_blocks": 61,
        "vocab_size": 248_320,
        "walk_cap": 12,
    }:
        raise SystemExit("compile contract drift")
    work = primary["logical_work"]
    if work != {
        "commit_launches_per_event_after": 1,
        "commit_launches_per_event_before": 1,
        "commit_programs_per_request_after": 1,
        "commit_programs_per_request_before": 1,
        "decision_programs_per_request_after": 30,
        "decision_programs_per_request_before": 30,
        "decision_values_stored_per_request_after": 81,
        "decision_values_stored_per_request_before": 81,
        "decision_workspace_bytes_per_request_after": 1_048,
        "decision_workspace_bytes_per_request_before": 529,
        "topology_index_scalar_loads_per_request_after": 0,
        "topology_index_scalar_loads_per_request_before": 24,
    }:
        raise SystemExit("logical work ledger drift")

    producer = primary["producer"]
    for label in ("incumbent", "candidate"):
        for name in (BLOCK_STATS, DIRECT_DECISION):
            if not resource_clean(producer[label][name]):
                raise SystemExit(f"producer resource regression: {label}/{name}")
    if (
        producer["candidate"][DIRECT_DECISION]["registers"] != 80
        or producer["incumbent"][DIRECT_DECISION]["registers"] != 80
    ):
        raise SystemExit("direct-decision register count drift")
    if (
        producer["candidate"][BLOCK_STATS]["source_function_sha256"]
        != producer["incumbent"][BLOCK_STATS]["source_function_sha256"]
    ):
        raise SystemExit("unchanged block-stats source drift")

    commit = primary["commit"]
    for batch in ("b1", "b4"):
        incumbent = commit["incumbent"][batch]
        candidate = commit["candidate"][batch]
        if not resource_clean(incumbent) or not resource_clean(candidate):
            raise SystemExit(f"commit resource regression: {batch}")
        if (
            incumbent["registers"] != 66
            or candidate["registers"] != 64
            or incumbent["ldg"] != 118
            or candidate["ldg"] != 95
            or incumbent["static_noncontrol_sass_instructions"] != 747
            or candidate["static_noncontrol_sass_instructions"] != 684
            or incumbent["stg"] != candidate["stg"]
        ):
            raise SystemExit(f"commit codegen delta drift: {batch}")

    comparator = primary["diagnostic_comparator"]
    if (
        comparator["b1"]["registers"] != 35
        or comparator["b4"]["registers"] != 32
        or not all(resource_clean(comparator[batch]) for batch in ("b1", "b4"))
    ):
        raise SystemExit("diagnostic comparator resource drift")
    conclusion = primary["conclusion"]
    if conclusion != {
        "committer_static_improves": True,
        "producer_registers_preserved": True,
        "real_swe_verified_gate_required": True,
        "resource_clean": True,
        "runtime_speedup_claimed": False,
    }:
        raise SystemExit("conclusion drift")
    print("fr13 fixed32 CFWD physical-slot SM121a codegen: PASS")


if __name__ == "__main__":
    main()
