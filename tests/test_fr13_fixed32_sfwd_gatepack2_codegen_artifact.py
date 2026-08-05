from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT / "results/fr13_fixed32_sfwd_conv_postprep_gatepack2_sm121a_20260805"
)
SUMMARY_BYTES = (ARTIFACT / "codegen_summary.json").read_bytes()
SUMMARY = json.loads(SUMMARY_BYTES)


def test_artifact_binds_exact_source_and_is_explicitly_offline() -> None:
    assert SUMMARY["revisions"] == {
        "fusion_baseline": "e4dbf0a521e4b7c21c9ea4f5be0db1839aefc1ea",
        "gatepack1": "0bf56d9d4d024129c2ff485c1802546dd518da30",
        "gatepack2": "1a86df82dbe6e704e472d2a770d3290917ca57e2",
    }
    assert SUMMARY["offline_only"] is True
    assert SUMMARY["gpu_api_used"] is False
    assert SUMMARY["timing_claim"] is False
    assert SUMMARY["runtime_correctness_claim"] is False
    assert hashlib.sha256(SUMMARY_BYTES).hexdigest() == (
        "38b65915dcad18b667b00491ed8cb045d297e42ef82a4586d067d763c16c64e7"
    )


def test_artifact_binds_wider_fixed32_geometry_and_static_gate() -> None:
    contract = SUMMARY["compile_contract"]
    assert contract["target"] == "sm_121a"
    assert contract["physical_rows_per_request"] == 32
    assert contract["gate_rows_per_program"]["gatepack2"] == {"b1": 4, "b4": 8}
    assert SUMMARY["static_gate_pass"] is True
    for profile in ("b1", "b4"):
        build = SUMMARY["builds"]["gatepack2"][profile]
        assert build["registers"] == 56
        assert build["stack_bytes"] == build["local_bytes"] == 0
        assert build["elf_shared_bytes"] == build["launch_shared_bytes"] == 0
        assert build["ldl"] == build["stl"] == build["calls"] == 0


def test_artifact_records_less_dynamic_work_than_first_gate_pack() -> None:
    expected = {
        "b1": (-384, -110592),
        "b4": (-768, -221184),
    }
    deltas = SUMMARY["work_deltas"]["gatepack2_vs_gatepack1"]
    for profile, pair in expected.items():
        assert (
            deltas[profile]["total_programs_all_48_layers"],
            deltas[profile][
                "requested_gate_bytes_whole_batch_all_48_layers"
            ],
        ) == pair
        assert deltas[profile]["kernel_launches_all_48_layers"] == 0


def test_audit_requires_empty_cuda_visibility_and_no_runtime_claim() -> None:
    audit = (ARTIFACT / "offline_codegen_audit.py").read_text()
    readme = (ARTIFACT / "README.md").read_text()
    assert 'os.environ.get("CUDA_VISIBLE_DEVICES") != ""' in audit
    assert "GATEPACK2_REVISION" in audit
    assert '"timing_claim": False' in audit
    assert "not measured DRAM or HBM traffic" in readme
    assert "does not establish device byte equality" in readme
