#!/usr/bin/env python3
"""Verify the checked-in fixed32 SFWD schedule-selector codegen summary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUMMARY_BYTES = (ROOT / "codegen_summary.json").read_bytes()
SUMMARY = json.loads(SUMMARY_BYTES)
EXPECTED_REVISIONS = {
    "candidate": "7e99008327eb1b0609793277a10c282c3d85b7d8",
    "compiler": "ee72339c39a83282bbd86298ea4796f71020d334",
}


def main() -> int:
    assert hashlib.sha256(SUMMARY_BYTES).hexdigest() == (
        "94c43af1a3c2c8d9035e5c6d0df5172f8078e23c79b2fec57443a2b7d759eff8"
    )
    assert SUMMARY["schema"].endswith(
        "embedded_gate_selector.sm121a.offline_codegen.v1"
    )
    assert SUMMARY["status"] == "PASS"
    assert SUMMARY["static_gate_pass"] is True
    assert SUMMARY["revisions"] == EXPECTED_REVISIONS
    assert SUMMARY["compile_contract"]["selector_default"] == 0
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
        standalone = SUMMARY["builds"]["standalone"][profile]
        embedded = SUMMARY["builds"]["embedded"][profile]
        assert (standalone["block_c"], standalone["num_warps"]) == (256, 4)
        assert (embedded["block_c"], embedded["num_warps"]) == (256, 4)
        assert standalone["registers"] == embedded["registers"] == 56
        for build in (standalone, embedded):
            assert build["stack_bytes"] == build["local_bytes"] == 0
            assert build["elf_shared_bytes"] == build["launch_shared_bytes"] == 0
            assert build["ldl"] == build["stl"] == build["calls"] == 0
            assert build["backend_producer"]["target"] == "sm_121a"
            assert build["backend_producer"]["tool_name"] == "ptxas-blackwell"
        assert (
            standalone["encoded_sass_instructions"],
            standalone["static_sass_instructions"],
        ) == (3040, 2889)
        assert (
            embedded["encoded_sass_instructions"],
            embedded["static_sass_instructions"],
        ) == (3024, 2875)
        assert standalone["ldg"] == embedded["ldg"] == 85
        assert standalone["stg"] == embedded["stg"] == 336
        assert SUMMARY["codegen_deltas"][profile][
            "encoded_sass_instructions"
        ] == -16
        assert SUMMARY["codegen_deltas"][profile][
            "static_sass_instructions"
        ] == -14

    expected_work = {
        "b1": (2112, 1920, -192, 8448, 7680, -768),
        "b4": (8448, 7680, -768, 33792, 30720, -3072),
    }
    for profile, expected in expected_work.items():
        standalone = SUMMARY["work_model"][profile]["standalone"]
        embedded = SUMMARY["work_model"][profile]["embedded"]
        delta = SUMMARY["work_deltas"][profile]
        observed = (
            standalone["ctas_whole_batch_all_48_layers"],
            embedded["ctas_whole_batch_all_48_layers"],
            delta["ctas_whole_batch_all_48_layers"],
            standalone["launched_warps_whole_batch_all_48_layers"],
            embedded["launched_warps_whole_batch_all_48_layers"],
            delta["launched_warps_whole_batch_all_48_layers"],
        )
        assert observed == expected
        assert standalone["standalone_gate_ctas_per_request_layer"] == 4
        assert embedded["standalone_gate_ctas_per_request_layer"] == 0
        assert embedded["embedded_gate_channel_ctas_per_request_layer"] == 4
        assert delta["requested_gate_bytes_whole_batch_all_48_layers"] == 0
        assert delta["kernel_launches_all_48_layers"] == 0

    assert SUMMARY["required_next_gate"].startswith("real SWE-Verified B1")
    print("PASS: fixed32 SFWD embedded-gate selector offline summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
