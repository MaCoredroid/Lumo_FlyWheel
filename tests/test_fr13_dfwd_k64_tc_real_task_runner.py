from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/fr13_run_b1_dfwd_k64_tc_real_task.sh"
LAUNCHER = REPO / "scripts/fr13_launch_forked_fa2_tree_server.sh"
MANIFEST = REPO / "scripts/fr13_runtime_manifest.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tc_real_task_is_real_b1_but_not_acceptance_evidence() -> None:
    text = RUNNER.read_text(encoding="ascii")

    assert "subset_b1_diagnostic_one.json" in text
    assert "FR13_FIXED32_B1_DIAGNOSTIC=1" in text
    assert "MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1" in text
    assert "FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536" in text
    assert "FR10_METRICS=0 ENFORCE_EAGER=0" in text
    assert "CUDAGRAPH_MODE=FULL_AND_PIECEWISE" in text
    assert (
        "classification=one_real_swe_verified_b1_hydra27_fixed32_"
        "k64_tc_diagnostic"
    ) in text
    assert "acceptance_valid=0" in text
    assert "timing_eligible=0" in text
    assert "floor_acceptance_eligible=0" in text
    assert '"acceptance_valid": False' in text
    assert '"timing_eligible": False' in text
    assert "PROBE_ONLY" not in text


def test_tc_real_task_binds_authenticated_candidate_inputs() -> None:
    text = RUNNER.read_text(encoding="ascii")
    expected = {
        "csrc/fr13_bf16_gemm_k64_tc16x256x64_s2.cu": (
            "8c55f0c1b8dc18b37b0cf6f06b5a8c608a62868cb027019b63b28126fa622095"
        ),
        (
            "results/fr13_fixed32_dfwd_k64_tc16x256x64_s2_sm121a_20260805/"
            "fr13_bf16_k64_tc16x256x64_s2.abi3.so"
        ): "c5c4cc7051003f521bb01fd8db4a340a5f9e8b4c579ee79ffb6a4ed3b43021a8",
        (
            "results/fr13_fixed32_dfwd_k64_tc16x256x64_s2_sm121a_20260805/"
            "build_attestation.json"
        ): "8a405cad4a8f9995d8e70cb6496f08e1e1e4645ed9636ff52ed18957a8adfdb8",
        (
            "results/fr13_fixed32_dfwd_k64_tc16x256x64_s2_sm121a_20260805/"
            "manifest.json"
        ): "5f825e42985987024316d1de4f774c2a5d12fc2f717d89805f735c14f2ea5607",
        "scripts/fr13_dfwd_k64_tc_selector.py": (
            "2797f716df4aa8fe763c6779cb0465e90d9b3883cedbd907255ebd0c24af57c8"
        ),
    }
    for relative, digest in expected.items():
        path = REPO / relative
        assert path.is_file() and not path.is_symlink()
        assert _sha256(path) == digest
        assert digest in text
    assert "TC_SO_BYTES=248984" in text


def test_tc_real_task_serves_only_candidate_and_requires_full_graph() -> None:
    text = RUNNER.read_text(encoding="ascii")

    assert "FR13_DRAFT_HEAD_B14_WARP4_PAIR8=0" in text
    assert "FR13_DRAFT_HEAD_K64_TC=1" in text
    assert 'FR13_DRAFT_HEAD_K64_TC_SOURCE_COMMIT="$SOURCE_COMMIT"' in text
    assert "candidate_served=1" in text
    assert "target_authority_unchanged=1" in text
    assert "FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB=0" in text
    assert "FR13_DRAFT_HEAD_M4_R64_U8_LIVE_AB=0" in text
    assert "FR13_DRAFT_HEAD_FP8=0" in text
    assert "FR13_DFWD_K64_TOP3=0" in text
    assert "FR13_FIXED32_CUTLASS_WAVE=stock" in text
    assert "FR13_FA2_QROW16_LIVE_PAGED_AB=0" in text
    assert "FR13_FA2_QROW32_LIVE_PAGED_AB=0" in text
    assert "FR13_FIXED32_COMMITTER_LAYER_BATCH=0" in text
    assert "FR13_CFWD_PACKED_WALK_ACTIVE_DEPTH_BYTE_AB=0" in text
    assert "[FR13_DRAFT_HEAD_K64_TC] engaged batch=1 candidate_served=1" in text
    assert "[FR13_DRAFT_HEAD_K64_TC] graph captured_calls=4" in text


def test_tc_real_task_emits_full_step_breakdown_and_is_closed() -> None:
    text = RUNNER.read_text(encoding="ascii")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    manifest = MANIFEST.read_text(encoding="utf-8")

    for timer in ("SFWD", "DFWD", "CFWD"):
        assert f"FR13_{timer}_GPU_TIMER=1" in text
        assert f"FR13_{timer}_GPU_TIMER_JSON=" in text
    assert "scripts/fr13_measure.py deploy-speed" in text
    for field in (
        "step_wall_ms",
        "measured_tps_fullstep_wall",
        "s_per_fwd_gpu_per_forward",
        "drafter_gpu_ms_per_step",
        "committer_gpu_ms_per_step",
        "overhead_other_ms_per_event",
        "accept_per_event",
    ):
        assert f'"{field}"' in text
    assert '-e FR13_DRAFT_HEAD_K64_TC="$FR13_DRAFT_HEAD_K64_TC"' in launcher
    assert "scripts/fr13_run_b1_dfwd_k64_tc_real_task.sh" in manifest


def test_tc_real_task_runner_parses() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
