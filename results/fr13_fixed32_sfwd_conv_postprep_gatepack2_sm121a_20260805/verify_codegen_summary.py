#!/usr/bin/env python3
"""Verify the checked-in fixed32 SFWD gate-pack2 offline summary."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUMMARY = json.loads((ROOT / "codegen_summary.json").read_text())
EXPECTED_REVISIONS = {
    "fusion_baseline": "e4dbf0a521e4b7c21c9ea4f5be0db1839aefc1ea",
    "gatepack1": "0bf56d9d4d024129c2ff485c1802546dd518da30",
    "gatepack2": "1a86df82dbe6e704e472d2a770d3290917ca57e2",
}
EXPECTED_GATE_ROWS = {
    "fusion_baseline": {"b1": 1, "b4": 1},
    "gatepack1": {"b1": 2, "b4": 4},
    "gatepack2": {"b1": 4, "b4": 8},
}
EXPECTED_RESOURCES = {
    "fusion_baseline": {
        "b1": (56, 2736, 2589, 79, 330),
        "b4": (56, 2736, 2591, 79, 330),
    },
    "gatepack1": {
        "b1": (56, 2832, 2687, 81, 332),
        "b4": (56, 2840, 2692, 81, 332),
    },
    "gatepack2": {
        "b1": (56, 3032, 2881, 85, 336),
        "b4": (56, 3040, 2889, 85, 336),
    },
}
EXPECTED_WORK = {
    "b1": {
        "fusion_baseline": (32, 112, 5376, 27648, 1327104),
        "gatepack1": (16, 96, 4608, 23040, 1105920),
        "gatepack2": (8, 88, 4224, 20736, 995328),
    },
    "b4": {
        "fusion_baseline": (32, 72, 13824, 27648, 5308416),
        "gatepack1": (8, 48, 9216, 20736, 3981312),
        "gatepack2": (4, 44, 8448, 19584, 3760128),
    },
}
EXPECTED_PRIOR_DELTAS = {
    "b1": (-8, -8, -384, -2304, -110592),
    "b4": (-4, -4, -768, -1152, -221184),
}


def main() -> int:
    assert SUMMARY["schema"].endswith("gatepack2.sm121a.offline_codegen.v1")
    assert SUMMARY["offline_only"] is True
    assert SUMMARY["gpu_api_used"] is False
    assert SUMMARY["timing_claim"] is False
    assert SUMMARY["runtime_correctness_claim"] is False
    assert SUMMARY["static_gate_pass"] is True
    assert SUMMARY["revisions"] == EXPECTED_REVISIONS
    contract = SUMMARY["compile_contract"]
    assert contract["target"] == "sm_121a"
    assert contract["physical_rows_per_request"] == 32
    assert contract["gate_rows_per_program"] == EXPECTED_GATE_ROWS

    for label, profiles in EXPECTED_RESOURCES.items():
        for profile, expected in profiles.items():
            build = SUMMARY["builds"][label][profile]
            observed = (
                build["registers"],
                build["encoded_sass_instructions"],
                build["static_sass_instructions"],
                build["ldg"],
                build["stg"],
            )
            assert observed == expected
            assert build["revision"] == EXPECTED_REVISIONS[label]
            assert build["stack_bytes"] == build["local_bytes"] == 0
            assert build["elf_shared_bytes"] == build["launch_shared_bytes"] == 0
            assert build["ldl"] == build["stl"] == build["calls"] == 0
            assert build["backend_producer"]["target"] == "sm_121a"
            assert build["backend_producer"]["tool_name"] == "ptxas-blackwell"

    for profile, variants in EXPECTED_WORK.items():
        for label, expected in variants.items():
            work = SUMMARY["work_model"][profile][label]
            observed = (
                work["gate_programs_per_request"],
                work["total_programs_per_request"],
                work["total_programs_all_48_layers"],
                work["requested_gate_bytes_per_request_layer"],
                work["requested_gate_bytes_whole_batch_all_48_layers"],
            )
            assert observed == expected
            assert work["kernel_launches_all_48_layers"] == 48
            assert "not_measured_dram_bytes" in work["traffic_classification"]
        delta = SUMMARY["work_deltas"]["gatepack2_vs_gatepack1"][profile]
        observed_delta = (
            delta["gate_programs_per_request"],
            delta["total_programs_per_request"],
            delta["total_programs_all_48_layers"],
            delta["requested_gate_bytes_per_request_layer"],
            delta["requested_gate_bytes_whole_batch_all_48_layers"],
        )
        assert observed_delta == EXPECTED_PRIOR_DELTAS[profile]
        assert delta["kernel_launches_all_48_layers"] == 0

    prior = SUMMARY["codegen_deltas"]["gatepack2_vs_gatepack1"]
    assert prior["b1"]["registers"] == prior["b4"]["registers"] == 0
    assert prior["b1"]["encoded_sass_instructions"] == 200
    assert prior["b4"]["encoded_sass_instructions"] == 200
    assert prior["b1"]["static_sass_instructions"] == 194
    assert prior["b4"]["static_sass_instructions"] == 197
    assert SUMMARY["required_next_gate"].startswith("real SWE-Verified B1 and B4")
    print("PASS: fixed32 SFWD gate-pack2 offline summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
