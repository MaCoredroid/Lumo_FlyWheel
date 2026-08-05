from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/fr13_run_b1_qrow16_target_sfwd_stack_timing.sh"


def _text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_runner_is_valid_shell_executable_and_default_off() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    assert RUNNER.stat().st_mode & stat.S_IXUSR
    env = os.environ.copy()
    env.pop("FR13_RUN_B1_QROW16_TARGET_SFWD_TIMING", None)
    completed = subprocess.run(
        ["bash", str(RUNNER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "timing is disabled" in completed.stderr


def test_runner_supports_exact4_and_exact16_with_full_breakdown() -> None:
    text = _text()
    assert "TASK_SET=${TASK_SET:-exact4}" in text
    assert "TASK_SET must be exactly exact4 or exact16" in text
    assert "subset_b4_four.json" in text
    assert "subset_b4_sixteen.json" in text
    for key in (
        "step_wall_ms",
        "measured_tps_fullstep_wall",
        "sfwd_gpu_ms_per_step",
        "dfwd_gpu_ms_per_step",
        "cfwd_gpu_ms_per_step",
        "gpu_component_ms_per_step",
        "other_wall_ms_per_step",
    ):
        assert key in text
    assert '"timing_eligible": True' in text
    assert '"floor_acceptance_eligible": False' in text
    assert '"performance_claim": False' in text


def test_runner_keeps_qrow16_on_and_u8_taw_cfwd_off_in_both_arms() -> None:
    text = _text()
    required = (
        "FR13_FA2_QROW16_PRODUCTION=1",
        'FORKED_FA2_SO="$QROW16_FA2_SO"',
        "FR13_B1_U8_CFWD_SFWD_STACK_TIMING=0",
        "FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION=0",
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0",
        "FR13_CFWD_LOGIT_DIRECT_PRODUCTION=0",
        "FR13_CONV_WB_BATCHED=\"$conv_wb_batched\"",
        "local sfwd_manifest_sha=\"\" sfwd_commit=\"\" conv_wb_batched=1",
        "'FR13_CONV_WB_BATCHED=1'",
    )
    for value in required:
        assert value in text
    assert 'FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION="$production"' not in text
    assert 'FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION="$production"' not in text
    assert 'FR13_CFWD_LOGIT_DIRECT_PRODUCTION="$production"' not in text
    assert text.index('run_arm "$STOCK_ARM" 0') < text.index(
        'run_arm "$CANDIDATE_ARM" 1'
    )


def test_runner_binds_current_target_sfwd_and_source_commit() -> None:
    text = _text()
    required = (
        "1649fbe9c6886147710dc9be97567bffcac36175c26742b752be9be50c2cbb86",
        "36940fd43d11399529d1bfe7e11baa9961907193267f3bb43d41057328737b77",
        "7d762dfa793671d75d1e353bd37d76fc07370cbe387ad1e315e32584d27927d4",
        "fr13_b1_composed_stack_gate.py validate-eager-credentials",
        "fr13_cutlass_streamk_pass.py validate",
        "fr13_sfwd_conv_postprep_gate.py validate-pass",
        '--source-commit "$SOURCE_COMMIT"',
        '"$(git rev-parse \'@{upstream}\')" == "$SOURCE_COMMIT"',
        "runner_sha256=%s",
        "production engaged layer=",
        '"sfwd_engaged_layers": 48',
    )
    for value in required:
        assert value in text


def test_runner_requires_absent_disabled_production_evidence() -> None:
    text = _text()
    for filename in (
        "fr13_dfwd_k64_m1_r64_u8.production_credential.json",
        "fr13_dfwd_k64_m1_r64_u8.production_engagement.json",
        "fr13_cfwd_logit_direct.production_pass.json",
        "fr13_cfwd_logit_direct.production_engagement.json",
        "fr13_fixed32_taw_native_precompute.production_pass.json",
        "fr13_fixed32_taw_native_precompute_production.arm",
    ):
        assert filename in text
    assert "U8/TAW/CFWD production unexpectedly engaged" in text

