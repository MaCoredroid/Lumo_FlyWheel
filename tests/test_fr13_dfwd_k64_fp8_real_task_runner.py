from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/fr13_run_b1_dfwd_k64_fp8_real_task.sh"
MANIFEST = ROOT / "scripts/fr13_runtime_manifest.py"


def test_runner_uses_real_b1_and_marks_scope_non_acceptance() -> None:
    text = RUNNER.read_text(encoding="ascii")
    assert "FR13_B1_DIAGNOSTIC_TASK_PROFILE=astropy12907" in text
    assert "FR13_B1_WORKLOAD_PROFILE=k64_root" in text
    assert "classification=one_real_swe_verified_b1" in text
    assert "acceptance_valid=0" in text
    assert "timing_eligible=0" in text
    assert "floor_acceptance_eligible=0" in text
    assert "PROBE_ONLY" not in text


def test_runner_selects_only_static_fp8_full_graph() -> None:
    text = RUNNER.read_text(encoding="ascii")
    assert "scripts/fr13_dfwd_k64_fp8_selector.py" in text
    assert "FR13_DRAFT_HEAD_FP8=1" in text
    assert "FR13_DRAFT_HEAD_FP8_STATIC_IO=1" in text
    assert "ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE" in text
    assert "FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536" in text
    assert "FR13_FIXED32_PHYSICAL_DRAFTS=31" in text
    assert "FR13_FIXED32_ACTIVE_NODES=27" in text
    assert "FR13_DRAFT_HEAD_K64_TC=0" in text
    assert "FR13_DRAFT_HEAD_B14_WARP4_PAIR8=0" in text
    assert "FR13_GATE_TAW_NATIVE=0" in text
    assert "FR13_GATE_GDN_BV=0" in text


def test_runner_binds_fp8_floor_and_real_evidence() -> None:
    text = RUNNER.read_text(encoding="ascii")
    assert "FR13_MANDATORY_WEIGHT_BYTES=30989326208" in text
    assert "FR13_WEIGHT_FLOOR_MS=113.514015414" in text
    assert "one_sided_u95_cap_ms=130.541117726" in text
    assert "draft_head_fp8_real_b1_gate.json" in text
    assert "draft_head_fp8_acceptance.json" in text
    assert "fr13_draft_head_fp8.engagement.json" in text
    for timer in ("FR13_SFWD_GPU_TIMER=1", "FR13_DFWD_GPU_TIMER=1", "FR13_CFWD_GPU_TIMER=1"):
        assert timer in (ROOT / "scripts/fr13_run_b1_kernel_live_gate.sh").read_text(
            encoding="ascii"
        )


def test_runner_and_selector_are_in_runtime_manifest() -> None:
    manifest = MANIFEST.read_text(encoding="utf-8")
    assert "scripts/fr13_dfwd_k64_fp8_selector.py" in manifest
    assert "scripts/fr13_run_b1_dfwd_k64_fp8_real_task.sh" in manifest


def test_runner_parses() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
