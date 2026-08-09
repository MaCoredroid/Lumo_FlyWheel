from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "fr13_run_b4_draft_head_m32_timing.sh"


def _text() -> str:
    return RUNNER.read_text(encoding="ascii")


def test_runner_is_valid_shell_and_default_off_paired_exact4_b4() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    text = _text()

    assert RUNNER.stat().st_mode & 0o111
    assert "config/fr13_fixed32/subset_b4_four.json" in text
    assert "MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4 AGENT_WALL_S=5400" in text
    assert "FR13_FIXED32_B1_DIAGNOSTIC=0" in text
    assert "FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536" in text
    assert "physical_rows_root_inclusive=32" in text
    assert 'run_arm "$STOCK_ARM" 0' in text
    assert 'run_arm "$CANDIDATE_ARM" 32' in text
    assert "only_arm_delta=FR13_DRAFT_HEAD_PAD_ROWS_0_to_32" in text
    assert "production_default_enabled=0" in text


def test_runner_selects_tail23_or_hydra27_without_changing_physical32() -> None:
    text = _text()

    assert "FR13_DRAFT_HEAD_B4_FIXED32_MODE:-hydra27_fixed32" in text
    assert "tail6_fixed32)" in text
    assert "LOGICAL_TOPOLOGY=Tail23" in text
    assert "ACTIVE_DRAFTS=23" in text
    assert "VALID_MASK=0x7a9ce7ff" in text
    assert "hydra27_fixed32)" in text
    assert "LOGICAL_TOPOLOGY=Hydra27" in text
    assert "ACTIVE_DRAFTS=27" in text
    assert "VALID_MASK=0x7abdffff" in text
    assert '"$arm" "$FIXED32_MODE" "$SUBSET"' in text
    assert "--expected-tok-per-draft 31 --batch-size 4" in text


def test_runner_rebuilds_real_task_provenance_and_requires_terminal_exact4() -> None:
    text = _text()

    assert "validate_fixed32_run_subset(subset_path, b1_diagnostic=False)" in text
    assert "build_fixed32_chat_traffic_audit(" in text
    assert "pinned_dataset_record_digests(str(repo))" in text
    assert "if audit != rebuilt:" in text
    assert 'terminal["eval"]["verdict"] not in {"resolved", "failed"}' in text
    assert '"all_tasks_terminal": True' in text
    assert 'record.get("task_instance_ids") != task_ids' in text
    assert 'record.get("n_tasks") != 4' in text
    assert "four nonempty per-task timing windows" in text


def test_runner_requires_b4_eager_and_graph_engagement_and_rejects_fallbacks() -> None:
    text = _text()

    assert "[FR13_DRAFT_HEAD_PAD\\] engaged candidate_rows=32" in text
    assert 'r"source_rows=(?P<source_rows>[1-4]) eager_launch=1"' in text
    assert "[FR13_DRAFT_HEAD_PAD\\] captured candidate_rows=32 source_rows=4" in text
    assert "if not marker_rows:" in text
    assert "candidate lacks eager M32 engagement" in text
    assert "if capture_markers < 1:" in text
    assert "candidate lacks source_rows=4 M32 graph capture" in text
    assert "stock arm emitted M32 engagement or graph capture" in text
    assert "FR13_DRAFT_VOCAB\\] DISABLED" in text
    assert "draft-head fallback/error markers" in text
    assert '"b4_graph_capture_observed": capture_markers > 0' in text
    assert '"fallback_or_error_markers": failures' in text


def test_summary_has_full_wall_phase_acceptance_and_quality_tradeoff() -> None:
    text = _text()

    assert (
        "from fr13_b4_timing_math import phase_breakdown, positive, promotion_verdict"
        in text
    )
    for field in (
        '"step_wall_ms"',
        '"measured_tps_fullstep_wall"',
        '"accepted_drafts_per_event"',
        '"committed_tokens_per_event"',
        '"sfwd_gpu_ms_per_step"',
        '"dfwd_gpu_ms_per_step"',
        '"cfwd_gpu_ms_per_step"',
        '"other_wall_ms_per_step"',
        '"candidate_to_stock_full_wall_tps_ratio"',
        '"candidate_minus_stock_accepted_drafts_per_event"',
        '"candidate_to_stock_acceptance_ratio"',
        '"stock_to_candidate_dfwd_gpu_speedup"',
    ):
        assert field in text
    assert '"candidate_scope": "proposal_quality"' in text
    assert '"draft_logits_may_differ": True' in text
    assert '"acceptance_may_differ": True' in text
    assert '"served_output_byte_identity_required": False' in text
    assert '"target_rejection_sampler_remains_authoritative": True' in text
    assert '"formal_floor_acceptance_eligible": False' in text


def test_runner_closes_source_manifests_and_disables_other_candidates() -> None:
    text = _text()

    assert "runtime_manifest.at_launch.json" in text
    assert "runtime_manifest.at_end.json" in text
    assert "external_manifest.at_launch.json" in text
    assert "external_manifest.at_end.json" in text
    assert "draft-head candidate source changed during timing" in text
    for disabled in (
        "FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0",
        "FR13_DRAFT_HEAD_M32_LIVE_AB=0",
        "FR13_DRAFT_HEAD_M32_PRODUCTION=0",
        "FR13_FIXED32_CUTLASS_WAVE=stock",
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0",
        "FR13_FA2_QROW16_LIVE_PAGED_AB=0",
        "FR13_DFWD_UNIFIED_BM8_LIVE_AB=0",
        "FR13_FIXED32_BATCH_GDN_BYTE_AB=0",
        "FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0",
        "FR13_FIXED32_ATTRIBUTION_ONLY=0",
    ):
        assert disabled in text
