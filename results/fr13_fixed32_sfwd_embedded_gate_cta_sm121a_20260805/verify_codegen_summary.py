#!/usr/bin/env python3
"""Verify the checked-in fixed32 SFWD embedded-gate codegen summary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUMMARY_BYTES = (ROOT / "codegen_summary.json").read_bytes()
SUMMARY = json.loads(SUMMARY_BYTES)
EXPECTED_REVISIONS = {
    "baseline": "4d876d6fd1a9a9bfc4ca4f90651bfc2421439e31",
    "candidate": "086da781207322601fc4876f9f6d69292a4a71a1",
    "compiler": "ee72339c39a83282bbd86298ea4796f71020d334",
}


def main() -> int:
    assert hashlib.sha256(SUMMARY_BYTES).hexdigest() == (
        "04635c84cde3d8bebdaff444530fb7614467dca0a8f77ae4ae0e3d11d65624a0"
    )
    assert SUMMARY["schema"].endswith("embedded_gate_cta.sm121a.offline_codegen.v1")
    assert SUMMARY["status"] == "PASS"
    assert SUMMARY["static_gate_pass"] is True
    assert SUMMARY["revisions"] == EXPECTED_REVISIONS
    for claim in (
        "gpu_api_used",
        "runtime_byte_correctness",
        "timing_claim",
        "performance_claim",
        "floor_acceptance_eligible",
    ):
        assert SUMMARY[claim] is False
    assert SUMMARY["offline_only"] is True

    for profile in ("b1", "b4"):
        baseline = SUMMARY["builds"]["baseline"][profile]
        candidate = SUMMARY["builds"]["candidate"][profile]
        assert (baseline["block_c"], baseline["num_warps"]) == (256, 4)
        assert (candidate["block_c"], candidate["num_warps"]) == (256, 4)
        assert baseline["registers"] == candidate["registers"] == 56
        for build in (baseline, candidate):
            assert build["stack_bytes"] == build["local_bytes"] == 0
            assert build["elf_shared_bytes"] == build["launch_shared_bytes"] == 0
            assert build["ldl"] == build["stl"] == build["calls"] == 0
            assert build["backend_producer"]["target"] == "sm_121a"
            assert build["backend_producer"]["tool_name"] == "ptxas-blackwell"
        assert (baseline["encoded_sass_instructions"], baseline["static_sass_instructions"]) == (3040, 2889)
        assert (candidate["encoded_sass_instructions"], candidate["static_sass_instructions"]) == (3024, 2875)
        assert baseline["ldg"] == candidate["ldg"] == 85
        assert baseline["stg"] == candidate["stg"] == 336
        assert SUMMARY["codegen_deltas"][profile]["encoded_sass_instructions"] == -16
        assert SUMMARY["codegen_deltas"][profile]["static_sass_instructions"] == -14

    expected_work = {
        "b1": (2112, 1920, -192, 8448, 7680, -768),
        "b4": (8448, 7680, -768, 33792, 30720, -3072),
    }
    for profile, expected in expected_work.items():
        baseline = SUMMARY["work_model"][profile]["baseline"]
        candidate = SUMMARY["work_model"][profile]["candidate"]
        delta = SUMMARY["work_deltas"][profile]
        observed = (
            baseline["ctas_whole_batch_all_48_layers"],
            candidate["ctas_whole_batch_all_48_layers"],
            delta["ctas_whole_batch_all_48_layers"],
            baseline["launched_warps_whole_batch_all_48_layers"],
            candidate["launched_warps_whole_batch_all_48_layers"],
            delta["launched_warps_whole_batch_all_48_layers"],
        )
        assert observed == expected
        assert baseline["standalone_gate_ctas_per_request_layer"] == 4
        assert candidate["standalone_gate_ctas_per_request_layer"] == 0
        assert candidate["embedded_gate_channel_ctas_per_request_layer"] == 4
        assert delta["requested_gate_bytes_whole_batch_all_48_layers"] == 0
        assert delta["kernel_launches_all_48_layers"] == 0

    assert SUMMARY["required_next_gate"].startswith("real SWE-Verified B1 and B4")
    print("PASS: fixed32 SFWD embedded-gate offline summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
