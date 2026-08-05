#!/usr/bin/env python3
"""Verify the checked-in fixed32 SFWD gate-pack offline summary."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUMMARY = json.loads((ROOT / "codegen_summary.json").read_text())
EXPECTED_REVISIONS = {
    "incumbent": "e4dbf0a521e4b7c21c9ea4f5be0db1839aefc1ea",
    "candidate": "0bf56d9d4d024129c2ff485c1802546dd518da30",
}
EXPECTED_RESOURCES = {
    "incumbent": {
        "b1": (56, 0, 0, 2736, 2589, 79, 330),
        "b4": (56, 0, 0, 2736, 2591, 79, 330),
    },
    "candidate": {
        "b1": (56, 0, 0, 2832, 2687, 81, 332),
        "b4": (56, 0, 0, 2840, 2692, 81, 332),
    },
}
EXPECTED_WORK = {
    "b1": (5376, 4608, 1327104, 1105920),
    "b4": (13824, 9216, 5308416, 3981312),
}


def main() -> int:
    assert SUMMARY["offline_only"] is True
    assert SUMMARY["gpu_api_used"] is False
    assert SUMMARY["timing_claim"] is False
    assert SUMMARY["runtime_correctness_claim"] is False
    assert SUMMARY["static_gate_pass"] is True
    assert SUMMARY["revisions"] == EXPECTED_REVISIONS
    assert SUMMARY["compile_contract"]["target"] == "sm_121a"
    assert SUMMARY["compile_contract"]["physical_rows_per_request"] == 32
    for label, profiles in EXPECTED_RESOURCES.items():
        for profile, expected in profiles.items():
            build = SUMMARY["builds"][label][profile]
            observed = (
                build["registers"],
                build["stack_bytes"],
                build["local_bytes"],
                build["encoded_sass_instructions"],
                build["static_sass_instructions"],
                build["ldg"],
                build["stg"],
            )
            assert observed == expected
            assert build["elf_shared_bytes"] == 0
            assert build["launch_shared_bytes"] == 0
            assert build["ldl"] == build["stl"] == build["calls"] == 0
            assert build["backend_producer"]["target"] == "sm_121a"
            assert build["backend_producer"]["tool_name"] == "ptxas-blackwell"
    for profile, expected in EXPECTED_WORK.items():
        work = SUMMARY["work_model"][profile]
        programs = work["total_programs_all_48_layers"]
        traffic = work["requested_gate_bytes_whole_batch_all_48_layers"]
        observed = (
            programs["incumbent"],
            programs["candidate"],
            traffic["incumbent"],
            traffic["candidate"],
        )
        assert observed == expected
        assert work["kernel_launches_all_48_layers"] == {
            "incumbent": 48,
            "candidate": 48,
            "delta": 0,
        }
        assert "not_measured_dram_bytes" in work["traffic_classification"]
    print("PASS: fixed32 SFWD gate-pack offline summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
