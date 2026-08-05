from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/fr13_launch_forked_fa2_tree_server.sh"


def _text() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def _exact_tuple() -> dict[str, str]:
    return {
        "FR13_B1_CFWD_TARGET_SFWD_STACK_TIMING": "1",
        "FR13_B1_U8_CFWD_SFWD_STACK_TIMING": "0",
        "FR13_FIXED32_MODE": "hydra27_fixed32",
        "MAX_NUM_SEQS": "1",
        "SWE_CONCURRENCY": "1",
        "FR13_FIXED32_B1_DIAGNOSTIC": "0",
        "ENFORCE_EAGER": "0",
        "CUDAGRAPH_MODE": "FULL_AND_PIECEWISE",
        "FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB": "0",
        "FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION": "0",
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


def _fragment() -> str:
    text = _text()
    start = text.index("_fr13_b1_cfwd_target_sfwd_stack=0")
    end = text.index(
        'if [[ "$FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB" == "1"', start
    )
    return text[start:end]


def _run_tuple(values: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = {"PATH": os.environ["PATH"], **values}
    return subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + _fragment()],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_launcher_admits_exact_u8_off_cfwd_target_sfwd_tuple() -> None:
    result = _run_tuple(_exact_tuple())
    assert result.returncode == 0, result.stderr


def test_launcher_rejects_each_u8_off_tuple_near_miss() -> None:
    exact = _exact_tuple()
    for name, value in exact.items():
        if name == "FR13_B1_CFWD_TARGET_SFWD_STACK_TIMING":
            continue
        near_miss = dict(exact)
        near_miss[name] = "near-miss" if value not in ("0", "1", "") else (
            "1" if value == "0" else "0" if value == "1" else "near-miss"
        )
        result = _run_tuple(near_miss)
        assert result.returncode == 2, name
        assert "requires the exact U8-off production tuple" in result.stderr


def test_launcher_selector_is_default_off_guarded_and_exported() -> None:
    text = _text()
    required = (
        "FR13_B1_CFWD_TARGET_SFWD_STACK_TIMING=${FR13_B1_CFWD_TARGET_SFWD_STACK_TIMING:-0}",
        'case "${FR13_B1_CFWD_TARGET_SFWD_STACK_TIMING:-0}" in',
        '_FR13_CALLER_M32_GUARD[FR13_B1_CFWD_TARGET_SFWD_STACK_TIMING]',
        '-e FR13_B1_CFWD_TARGET_SFWD_STACK_TIMING=',
        "_fr13_sfwd_no_u8_cfwd_composed=$_fr13_b1_cfwd_target_sfwd_stack",
        '&& "$_fr13_sfwd_no_u8_cfwd_composed" != "1"',
    )
    for value in required:
        assert value in text
