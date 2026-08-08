from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SHARED_ENV = ROOT / "scripts" / "fr13_fixed32_sfwd_fusion_env.sh"
DIAG = ROOT / "scripts" / "fr13_run_b1_sfwd_fusion_boot_diag.sh"
TIMING = ROOT / "scripts" / "fr13_run_b1_target_sfwd_exact4_timing.sh"


@pytest.mark.parametrize("script", (SHARED_ENV, DIAG, TIMING))
def test_scripts_parse(script: Path) -> None:
    assert script.is_file()
    subprocess.run(["bash", "-n", str(script)], check=True)


def _strip_comments(source: str) -> str:
    """Drop whole-line comments so prose cannot satisfy a code assertion."""
    return "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )


def _shared_env_array(name: str) -> tuple[str, ...]:
    """Pull one bash array literal out of the shared env file."""
    source = SHARED_ENV.read_text(encoding="utf-8")
    match = re.search(
        rf"^[ \t]*{name}=\(\n(.*?)^[ \t]*\)$", source, re.MULTILINE | re.DOTALL
    )
    assert match is not None, f"{SHARED_ENV} lacks array {name}"
    entries = []
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    return tuple(entries)


def test_both_runners_share_one_candidate_env_definition() -> None:
    """The screen is only sound if it boots the arm it screens, byte for byte.

    Duplicated env blocks drift silently, so the candidate environment lives in
    exactly one file and both runners source it.
    """
    diag = DIAG.read_text(encoding="utf-8")
    timing = TIMING.read_text(encoding="utf-8")
    for source in (diag, timing):
        assert "source scripts/fr13_fixed32_sfwd_fusion_env.sh" in source
        assert source.count("fr13_fixed32_sfwd_fusion_env ") == 1
        assert 'env "${FR13_FIXED32_SFWD_FUSION_ENV[@]}"' in source

    # The env-prefix list must not be re-inlined anywhere but the shared file.
    shared = SHARED_ENV.read_text(encoding="utf-8")
    diag_code = _strip_comments(diag)
    timing_code = _strip_comments(timing)
    # Interpolated forms exist only in the env array; the timing runner's
    # container_env.txt pin loop legitimately repeats the literal values.
    for pin in (
        'FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION="$sfwd_fusion"',
        'FR13_FIXED32_CUTLASS_WAVE_PRODUCTION="$production"',
        'FR13_DEVICE_MULTIDRAFT_KERNEL="$device_kernel"',
        'FR13_FA2_QROW16_LIVE_PASS_SHA256="$QROW16_PASS_SHA256"',
        'FORKED_FA2_SO="$QROW16_FA2_SO"',
    ):
        assert pin in shared
        assert pin not in diag_code, pin
        assert pin not in timing_code, pin


def test_shared_env_pins_the_composed_candidate_shape() -> None:
    entries = _shared_env_array("FR13_FIXED32_SFWD_FUSION_ENV")
    joined = "\n".join(entries)
    for pin in (
        "ENFORCE_EAGER=0",
        "CUDAGRAPH_MODE=FULL_AND_PIECEWISE",
        "MAX_NUM_SEQS_OVR=1",
        "SWE_CONCURRENCY=1",
        "FR13_DRAFT_VOCAB_ROOT=1",
        "FR13_DRAFT_VOCAB_K=65536",
        "FR13_FA2_QROW16_PRODUCTION=1",
        "FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB=0",
        'FR13_CONV_WB_BATCHED="$conv_wb_batched"',
        "FR13_TREE_CONV_FUSED=1",
    ):
        assert pin in joined, pin
    # Every other production selector stays off so the screen isolates SFWD.
    for off in (
        "FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0",
        "FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION=0",
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0",
        "FR13_CFWD_LOGIT_DIRECT_PRODUCTION=0",
        "FR13_DFWD_UNIFIED_BM8_PRODUCTION=0",
        "FR13_FIXED32_BATCH_GDN_PRODUCTION=0",
    ):
        assert off in joined, off


def test_forbidden_strings_match_the_timing_engagement_validator() -> None:
    """One list, two consumers: the pair's validator and the boot screen."""
    shared = tuple(
        ast.literal_eval(entry)
        for entry in _shared_env_array("FR13_FIXED32_SFWD_FUSION_FORBIDDEN")
    )
    timing = TIMING.read_text(encoding="utf-8")
    match = re.search(r"for forbidden in \(\n(.*?)^\):$", timing, re.MULTILINE | re.DOTALL)
    assert match is not None, "timing runner lacks its forbidden-string tuple"
    validator = tuple(
        ast.literal_eval(line.strip().rstrip(","))
        for line in match.group(1).splitlines()
        if line.strip()
    )
    assert shared == validator
    assert len(shared) == 7
    assert "FR13 SFWD conv/post-prep capture lacks preseeded output bindings" in shared


def test_diag_pins_the_same_binaries_and_credentials_as_the_pair() -> None:
    diag = DIAG.read_text(encoding="utf-8")
    timing = TIMING.read_text(encoding="utf-8")
    for pin in (
        "QROW16_SHA256=1649fbe9c6886147710dc9be97567bffcac36175c26742b752be9be50c2cbb86",
        "QROW16_BYTES=299507792",
        "QROW16_PASS_SHA256=36940fd43d11399529d1bfe7e11baa9961907193267f3bb43d41057328737b77",
        "TARGET_SELECTOR=identity_wide256_fullgrid_b1",
        "TARGET_SHA256=d8c6502e7a166e6d2124576a9e36814401d6dbc215516adfffa7ac436f93ba0f",
        "TARGET_BYTES=119704312",
        "TARGET_PASS_SHA256=169704fac7c544600437e7785f5d810c9df8ffaf5f9ce70d96d83b21de46236d",
        "TARGET_QUALIFICATION_SOURCE_COMMIT=a8a904ed6c27a6338d43151038c155ebb76e3656",
    ):
        assert pin in diag, pin
        assert pin in timing, pin
    # The launcher hard-requires the SFWD pass + manifest for FUSION=1; the
    # screen takes those but never the standalone gate summary.
    assert "SFWD_CONV_POSTPREP_PASS:?" in diag
    assert "SFWD_CONV_POSTPREP_SOURCE_MANIFEST:?" in diag
    assert "SFWD_CONV_POSTPREP_GATE_SUMMARY" not in diag
    assert "SFWD_CONV_POSTPREP_GATE_SUMMARY" in timing


def test_diag_is_loudly_non_citable_and_runs_no_workload() -> None:
    diag = DIAG.read_text(encoding="utf-8")
    assert "DIAGNOSTIC ONLY - produces no citable evidence" in diag
    assert "classification=diagnostic_boot_only" in diag
    assert '"classification": "diagnostic_boot_only"' in diag
    assert '"citable": False' in diag
    assert '"timing_eligible": False' in diag
    # No measurement, no promotion, no second arm.
    for forbidden in (
        "fr13_measure.py",
        "deploy_speed",
        "floor_acceptance",
        "run_arm",
    ):
        assert forbidden not in diag, forbidden


def test_diag_requires_clean_docker_and_tears_down() -> None:
    diag = DIAG.read_text(encoding="utf-8")
    assert "all Docker containers must be absent before the boot screen" in diag
    assert "recover_host_memory" in diag
    assert "docker rm -f" in diag
    assert "trap teardown EXIT" in diag
    # A dev screen must not demand the commit be pushed upstream.
    assert "boot diagnostic requires a clean source tree" in diag
    assert "@{upstream}" not in diag


def test_diag_checks_capture_time_evidence() -> None:
    diag = DIAG.read_text(encoding="utf-8")
    assert "fr13_fa2_qrow16_production_capture.json" in diag
    assert '"fr13.fixed32.fa2_qrow16_production_capture.v1"' in diag
    assert '"runtime_mode": "FULL"' in diag
    assert "[FR13_SFWD_CONV_POSTPREP] production engaged layer=" in diag
    assert "-ne 48" in diag
