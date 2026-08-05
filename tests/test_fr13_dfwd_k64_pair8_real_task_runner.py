from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/fr13_run_b1_dfwd_k64_b14_pair8_real_task.sh"
LAUNCHER = REPO / "scripts/fr13_launch_forked_fa2_tree_server.sh"
MANIFEST = REPO / "scripts/fr13_runtime_manifest.py"


def test_pair8_real_task_is_real_b1_but_not_acceptance_evidence() -> None:
    text = RUNNER.read_text(encoding="ascii")

    assert "subset_b1_diagnostic_one.json" in text
    assert "FR13_FIXED32_B1_DIAGNOSTIC=1" in text
    assert "MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1" in text
    assert "FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536" in text
    assert "classification=one_real_swe_verified_b1_hydra27_fixed32_k64_pair8_diagnostic" in text
    assert "acceptance_valid=0" in text
    assert "timing_eligible=0" in text
    assert "floor_acceptance_eligible=0" in text
    assert '"acceptance_valid": False' in text
    assert '"timing_eligible": False' in text


def test_pair8_real_task_serves_only_the_authenticated_candidate() -> None:
    text = RUNNER.read_text(encoding="ascii")

    assert "FR13_DRAFT_HEAD_B14_WARP4_PAIR8=1" in text
    assert 'FR13_DRAFT_HEAD_B14_WARP4_PAIR8_SOURCE_COMMIT="$SOURCE_COMMIT"' in text
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
    assert "[FR13_DRAFT_HEAD_B14_WARP4_PAIR8] engaged batch=1 candidate_served=1" in text
    assert "[FR13_DRAFT_HEAD_B14_WARP4_PAIR8] graph captured_calls=4" in text


def test_pair8_selector_reaches_the_container_and_manifest() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    manifest = MANIFEST.read_text(encoding="utf-8")

    assert '-e FR13_DRAFT_HEAD_B14_WARP4_PAIR8="$FR13_DRAFT_HEAD_B14_WARP4_PAIR8"' in launcher
    assert (
        '-e FR13_DRAFT_HEAD_B14_WARP4_PAIR8_SOURCE_COMMIT='
        '"$FR13_DRAFT_HEAD_B14_WARP4_PAIR8_SOURCE_COMMIT"'
    ) in launcher
    assert "scripts/fr13_run_b1_dfwd_k64_b14_pair8_real_task.sh" in manifest


def test_pair8_real_task_emits_full_step_breakdown() -> None:
    text = RUNNER.read_text(encoding="ascii")

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
