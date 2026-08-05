from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results/fr13_fixed32_sfwd_b1_block256_sm121a_20260805"
SUMMARY = json.loads((ARTIFACT / "codegen_summary.json").read_text())


def test_b1_block256_artifact_checksums_are_exact() -> None:
    for line in (ARTIFACT / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == expected


def test_b1_block256_artifact_is_offline_and_spill_free() -> None:
    assert SUMMARY["status"] == "PASS"
    assert SUMMARY["offline_only"] is True
    assert SUMMARY["claims"] == {
        "floor_acceptance_eligible": False,
        "gpu_runtime_used": False,
        "performance_claim": False,
        "runtime_byte_correctness": False,
        "timing_claim": False,
    }
    candidate = SUMMARY["candidate"]
    assert (candidate["block_c"], candidate["num_warps"]) == (256, 4)
    assert candidate["registers"] == 56
    assert candidate["stack_bytes"] == candidate["local_bytes"] == 0
    assert candidate["launch_shared_bytes"] == 0
    assert candidate["ldl"] == candidate["stl"] == candidate["calls"] == 0


def test_b1_block256_artifact_binds_sources_and_exact_work_reduction() -> None:
    kernel = ROOT / "src/lumo_flywheel_serving/fr13_sfwd_conv_postprep_fusion_kernel.py"
    launcher = ROOT / "src/lumo_flywheel_serving/fr13_sfwd_conv_postprep_fusion.py"
    assert (
        hashlib.sha256(kernel.read_bytes()).hexdigest()
        == SUMMARY["kernel_source_sha256"]
    )
    assert (
        hashlib.sha256(launcher.read_bytes()).hexdigest()
        == SUMMARY["launcher_source_sha256"]
    )
    assert SUMMARY["new_b1_work"] == {
        "channel_programs_all_48_layers": 1920,
        "channel_programs_per_layer": 40,
        "gate_programs_all_48_layers": 192,
        "gate_programs_per_layer": 4,
        "gate_rows_per_program": 8,
        "programs_all_48_layers": 2112,
        "programs_per_layer": 44,
        "requested_gate_bytes_all_48_layers": 940032,
        "requested_gate_bytes_per_layer": 19584,
        "scheduled_warps_per_layer": 176,
    }
    delta = SUMMARY["comparison"]["candidate_minus_previous_b1"]
    assert delta["programs_all_48_layers"] == -2112
    assert delta["scheduled_warps_per_layer"] == 0


def test_b1_block256_artifact_reuses_the_audited_b4_binary_geometry() -> None:
    prior = json.loads(
        (
            ROOT
            / "results/fr13_fixed32_sfwd_conv_postprep_gatepack2_sm121a_20260805/codegen_summary.json"
        ).read_text()
    )["builds"]["gatepack2"]["b4"]
    candidate = SUMMARY["candidate"]
    assert SUMMARY["comparison"]["candidate_equals_prior_b4_cubin"] is True
    assert SUMMARY["comparison"]["candidate_equals_prior_b4_sass"] is True
    for key in (
        "cubin_sha256",
        "sass_sha256",
        "ptx_sha256",
        "registers",
        "stack_bytes",
        "local_bytes",
        "encoded_sass_instructions",
        "static_sass_instructions",
        "ldg",
        "stg",
        "ldl",
        "stl",
        "calls",
    ):
        assert candidate[key] == prior[key]
