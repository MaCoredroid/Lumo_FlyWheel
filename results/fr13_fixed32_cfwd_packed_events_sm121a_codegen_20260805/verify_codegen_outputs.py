#!/usr/bin/env python3
"""Verify two independent fixed32 packed-event SM121a codegen builds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BASE = "1f7485ade5ec6bfacf51dde7afa514531effcbcd"
CANDIDATE = "103030ea88ad7da28a4bcab187a57200be70756d"
DIRECT_DECISION = "_fr13_cfwd_logit_direct_decision_kernel"
RESOURCE_FIELDS = ("stack_bytes", "local_bytes", "ldl", "stl", "calls")


def load(path: Path) -> dict:
    return json.loads((path / "codegen_summary.json").read_text(encoding="ascii"))


def resource_clean(build: dict) -> bool:
    return all(build[name] == 0 for name in RESOURCE_FIELDS)


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
        != "fr13.fixed32.cfwd_packed_events.sm121a.codegen.v1"
        or primary["status"] != "pass"
        or primary["base_revision"] != BASE
        or primary["candidate_revision"] != CANDIDATE
        or primary["claim_scope"]
        != "static_sm121a_codegen_and_exact_cpu_semantics_no_runtime_claim"
    ):
        raise SystemExit("codegen summary identity or claim scope drift")
    if primary["source_contracts"] != {
        "candidate": {
            "name": "fixed32_cfwd_logit_direct_packed_physical_slots_v3",
            "schema": "fr13.fixed32.cfwd_logit_direct_packed_physical_slots.v3",
            "sha256": (
                "5a9107306bdc37200448a6a5add2b84dfd839dc377b11009f218662c63abcc1c"
            ),
        },
        "cfwd_integration": {
            "schema": "fr13.fixed32.cfwd_logit_direct.integration_source.v2",
            "sha256": (
                "a82ce3f5e526792ca45bb444212e5440e8444778f174fd0650accc4bb5f8558c"
            ),
        },
        "taw": {
            "schema": "fr13-fixed32-taw-all-parent-v7",
            "sha256": (
                "998bc6331177469d6890f97f3e066e1d07c2ca2d8ab4bff723f32d5229fef290"
            ),
            "unchanged_by_candidate": True,
        },
    }:
        raise SystemExit("source contract drift")
    if primary["packed_event_contract"] != {
        "accepted_node_zero_row": 1,
        "accepted_row_mask": 31,
        "accepted_row_shift": 18,
        "parent_mask": 8_388_608,
        "rejection_accepted_row": 0,
        "token_mask": 262_143,
        "verifier_vocab_fits_token_bits": True,
        "verifier_vocab_size": 248_320,
    }:
        raise SystemExit("packed-event layout drift")
    if primary["exact_work"] != {
        "commit_programs_per_request_after": 1,
        "commit_programs_per_request_before": 1,
        "decision_programs_per_request_after": 30,
        "decision_programs_per_request_before": 30,
        "decision_values_stored_per_request_after": 30,
        "decision_values_stored_per_request_before": 81,
        "decision_workspace_bytes_per_request_after": 504,
        "decision_workspace_bytes_per_request_before": 1_048,
        "physical_rows": 32,
        "tree_metadata_loads_per_request_after": 0,
        "tree_metadata_loads_per_request_before": 24,
        "walk_levels": 12,
    }:
        raise SystemExit("exact work ledger drift")

    producer = primary["producer"]
    before = producer["base"][DIRECT_DECISION]
    after = producer["candidate"][DIRECT_DECISION]
    if (
        before["registers"] != 80
        or after["registers"] != 80
        or before["ldg"] != 51
        or after["ldg"] != 51
        or before["stg"] != 5
        or after["stg"] != 2
        or before["static_noncontrol_sass_instructions"] != 2_558
        or after["static_noncontrol_sass_instructions"] != 2_565
        or not resource_clean(before)
        or not resource_clean(after)
    ):
        raise SystemExit("producer codegen delta drift")

    for batch in ("b1", "b4"):
        before = primary["commit"]["base"][batch]
        after = primary["commit"]["candidate"][batch]
        if (
            before["registers"] != 64
            or after["registers"] != 46
            or before["ldg"] != 95
            or after["ldg"] != 35
            or before["stg"] != after["stg"]
            or before["stg"] != 41
            or before["static_noncontrol_sass_instructions"] != 684
            or after["static_noncontrol_sass_instructions"] != 509
            or before["encoded_sass_instructions"] != 696
            or after["encoded_sass_instructions"] != 520
            or before["cubin_bytes"] != 59_656
            or after["cubin_bytes"] != 46_176
            or not resource_clean(before)
            or not resource_clean(after)
        ):
            raise SystemExit(f"committer codegen delta drift: {batch}")

    comparator = primary["comparator"]
    if (
        comparator["b1"]["registers"] != 40
        or comparator["b4"]["registers"] != 38
        or comparator["b1"]["ldg"] != 22
        or comparator["b4"]["ldg"] != 22
        or not all(resource_clean(comparator[batch]) for batch in ("b1", "b4"))
    ):
        raise SystemExit("comparator resource drift")
    if primary["conclusion"] != {
        "committer_static_improves": True,
        "comparator_resource_clean": True,
        "producer_registers_preserved_and_stores_reduced": True,
        "real_swe_verified_gate_required": True,
        "runtime_speedup_claimed": False,
    }:
        raise SystemExit("conclusion drift")
    print("fr13 fixed32 CFWD packed-event SM121a codegen: PASS")


if __name__ == "__main__":
    main()
