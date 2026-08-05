from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
RUNNER = SCRIPTS / "fr13_run_b1_u8_cfwd_sfwd_stack_timing.sh"
WRAPPER = SCRIPTS / "fr13_run_b1_composed_cfwd_stack_timing.sh"
LAUNCHER = SCRIPTS / "fr13_launch_forked_fa2_tree_server.sh"
MANIFEST = SCRIPTS / "fr13_runtime_manifest.py"
COMPOSED_GATE = SCRIPTS / "fr13_b1_composed_stack_gate.py"
COMPOSED_REDUCER = SCRIPTS / "fr13_b1_composed_stack_timing.py"
LEGACY_RUNNER = SCRIPTS / "fr13_run_b1_k64_qrow32_b1_sfwd_stack_timing.sh"

CURRENT_TARGET_SHA256 = (
    "7d762dfa793671d75d1e353bd37d76fc07370cbe387ad1e315e32584d27927d4"
)
STALE_TARGET_SHA256 = (
    "85937b5c35ec87bce12e4b5d677dd67f63004f9a9d9fb6d64473a5bd3b53b2da"
)
QROW32_SHA256 = (
    "a9d8a6887b8b27b3a83af60bba7945eb66caff174ba710c2ee2aea92b8e7081a"
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_runner_is_executable_valid_shell_and_default_off() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    assert RUNNER.stat().st_mode & stat.S_IXUSR
    env = os.environ.copy()
    env.pop("FR13_RUN_B1_U8_CFWD_SFWD_TIMING", None)
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


def test_runner_binds_fresh_u8_gate_and_raw_result_without_qrow() -> None:
    text = _text(RUNNER)
    required = (
        ': "${U8_GATE_JSON:?set U8_GATE_JSON to the fresh U8 shadow gate}"',
        (
            ': "${U8_LIVE_RESULT_JSON:?set U8_LIVE_RESULT_JSON '
            'to the shared raw U8 result}"'
        ),
        '--dfwd-gate "$U8_GATE_JSON"',
        '--dfwd-live-result "$U8_LIVE_RESULT_JSON"',
        '--live-result "$U8_LIVE_RESULT_JSON"',
        "u8_pass=$U8_LIVE_RESULT_JSON",
        'FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION="$production"',
        "FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0",
        "FR13_FA2_QROW32_LIVE_PAGED_AB=0",
        "FR13_FA2_QROW32_B1_LIVE_AB_ARM= FR13_FA2_QROW32_B1_PRODUCTION_ARM=",
        "FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION=0",
        "FR13_DFWD_K64_TOP3=0",
    )
    for value in required:
        assert value in text
    assert QROW32_SHA256 not in text
    assert "FR13_FA2_QROW32_B1_PRODUCTION_ARM=nosplit" not in text


def test_runner_binds_packed_cfwd_current_target_sfwd_and_taw() -> None:
    text = _text(RUNNER)
    required = (
        "fr13_device_multidraft_cfwd_packed_v3.py",
        "fr13_cfwd_dfwd_u8_composed_gate.py",
        "fr13_b1_composed_stack_gate.py validate-eager-credentials",
        "fr13_cutlass_streamk_pass.py validate",
        "fr13_sfwd_conv_postprep_gate.py validate-pass",
        "fr13_cfwd_logit_direct_gate.py validate",
        "fr13_taw_b1_credential.py validate-production",
        'FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION="$production"',
        'FR13_CFWD_LOGIT_DIRECT_PRODUCTION="$production"',
        'FR13_FIXED32_CUTLASS_WAVE="$target_selector"',
        'FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION="$sfwd_fusion"',
        "production engaged layer=",
        "sfwd_engaged_layers\": 48",
    )
    for value in required:
        assert value in text
    assert CURRENT_TARGET_SHA256 in text
    assert STALE_TARGET_SHA256 not in text
    assert text.index('run_arm "$STOCK_ARM" 0') < text.index(
        'run_arm "$CANDIDATE_ARM" 1'
    )
    assert text.index("fr13_taw_b1_credential.py validate-production") < text.index(
        "docker ps -aq"
    )


def test_runner_supports_only_exact4_and_exact16_and_marks_no_floor_claim() -> None:
    text = _text(RUNNER)
    assert "TASK_SET=${TASK_SET:-exact4}" in text
    assert "TASK_SET must be exactly exact4 or exact16" in text
    assert "subset_b4_four.json" in text
    assert "subset_b4_sixteen.json" in text
    assert "production_default_enabled=0" in text
    assert '"production_default_enabled": False' in text
    assert '"timing_eligible": True' in text
    assert '"floor_acceptance_eligible": False' in text
    assert '"performance_claim": False' in text


def test_wrapper_routes_cfwd_timing_to_new_full_stack_runner() -> None:
    wrapper = _text(WRAPPER)
    assert "FR13_RUN_B1_COMPOSED_CFWD_TIMING" in wrapper
    assert "FR13_RUN_B1_U8_CFWD_SFWD_TIMING=1" in wrapper
    assert "fr13_run_b1_u8_cfwd_sfwd_stack_timing.sh" in wrapper
    assert "FR13_RUN_QROW32_NOSPLIT_TIMING" not in wrapper
    assert "fr13_run_b1_k64_qrow32_b1_sfwd_stack_timing.sh" not in wrapper
    assert "scripts/fr13_run_b1_u8_cfwd_sfwd_stack_timing.sh" in _text(MANIFEST)


def test_launcher_admits_only_the_explicit_exact_u8_full_stack_tuple() -> None:
    text = _text(LAUNCHER)
    required = (
        "FR13_B1_U8_CFWD_SFWD_STACK_TIMING=${FR13_B1_U8_CFWD_SFWD_STACK_TIMING:-0}",
        "_fr13_b1_u8_cfwd_sfwd_stack=0",
        '"$FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION" == "1"',
        '"$FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION" == "1"',
        '"$FR13_CFWD_LOGIT_DIRECT_PRODUCTION" == "1"',
        '"$FR13_FA2_QROW16_PRODUCTION" == "0"',
        '-z "$FR13_FA2_QROW32_B1_PRODUCTION_ARM"',
        '"${FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION:-0}" == "0"',
        '"$FR13_DFWD_K64_TOP3" == "0"',
        '"$FR13_FIXED32_CUTLASS_WAVE" == "identity_wide256_fullgrid_b1"',
        '"$FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION" == "1"',
        "/workspace/scripts/fr13_device_multidraft_cfwd_packed_v3.py",
        '|| "$_fr13_b1_u8_cfwd_sfwd_stack" == "1"',
        '-e FR13_B1_U8_CFWD_SFWD_STACK_TIMING=',
    )
    for value in required:
        assert value in text
    assert text.index("_fr13_b1_u8_cfwd_sfwd_stack=0") < text.index(
        "FR13 draft-head U8 production inherited an incompatible B1 candidate"
    )


def test_launcher_rejects_every_u8_full_stack_tuple_near_miss() -> None:
    text = _text(LAUNCHER)
    start = text.index("_fr13_b1_u8_cfwd_sfwd_stack=0")
    end = text.index(
        'if [[ "$FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB" == "1"', start
    )
    fragment = text[start:end]
    exact = {
        "FR13_B1_U8_CFWD_SFWD_STACK_TIMING": "1",
        "FR13_FIXED32_MODE": "hydra27_fixed32",
        "MAX_NUM_SEQS": "1",
        "SWE_CONCURRENCY": "1",
        "FR13_FIXED32_B1_DIAGNOSTIC": "0",
        "ENFORCE_EAGER": "0",
        "CUDAGRAPH_MODE": "FULL_AND_PIECEWISE",
        "FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB": "0",
        "FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION": "1",
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE": "0",
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION": "1",
        "FR13_CFWD_LOGIT_DIRECT_BYTE_AB": "0",
        "FR13_CFWD_LOGIT_DIRECT_PRODUCTION": "1",
        "FR13_FA2_QROW16_LIVE_PAGED_AB": "0",
        "FR13_FA2_QROW16_PRODUCTION": "0",
        "FR13_FA2_QROW32_LIVE_PAGED_AB": "0",
        "FR13_FA2_QROW32_LIVE_PAGED_AB_ARM": "",
        "FR13_FA2_QROW32_B1_PRODUCTION_ARM": "",
        "FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION": "0",
        "FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_BATCH": "",
        "FR13_DFWD_K64_TOP3": "0",
        "FR13_FIXED32_CUTLASS_WAVE": "identity_wide256_fullgrid_b1",
        "FR13_FIXED32_CUTLASS_WAVE_PRODUCTION": "1",
        "FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB": "0",
        "FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION": "0",
        "FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB": "0",
        "FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION": "1",
        "FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB": "0",
        "FR13_DEVICE_MULTIDRAFT_KERNEL": (
            "/workspace/scripts/fr13_device_multidraft_cfwd_packed_v3.py"
        ),
    }

    def run(values: dict[str, str]) -> subprocess.CompletedProcess[str]:
        env = {"PATH": os.environ["PATH"], **values}
        return subprocess.run(
            ["bash", "-c", "set -euo pipefail\n" + fragment],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

    assert run(exact).returncode == 0
    for name, value in exact.items():
        if name == "FR13_B1_U8_CFWD_SFWD_STACK_TIMING":
            continue
        near_miss = dict(exact)
        near_miss[name] = "near-miss" if value not in ("0", "1", "") else (
            "1" if value == "0" else "0" if value == "1" else "near-miss"
        )
        result = run(near_miss)
        assert result.returncode == 2, name
        assert "requires the exact default-off production tuple" in result.stderr


def test_related_target_pins_are_current_while_legacy_qrow_is_isolated() -> None:
    for path in (COMPOSED_GATE, COMPOSED_REDUCER, LEGACY_RUNNER):
        text = _text(path)
        assert CURRENT_TARGET_SHA256 in text
        assert STALE_TARGET_SHA256 not in text
    assert QROW32_SHA256 in _text(LEGACY_RUNNER)
    assert QROW32_SHA256 not in _text(RUNNER)
