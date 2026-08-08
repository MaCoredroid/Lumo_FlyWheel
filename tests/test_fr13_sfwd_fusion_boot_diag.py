from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
SHARED_ENV = ROOT / "scripts" / "fr13_fixed32_sfwd_fusion_env.sh"
DIAG = ROOT / "scripts" / "fr13_run_b1_sfwd_fusion_boot_diag.sh"
TIMING = ROOT / "scripts" / "fr13_run_b1_target_sfwd_exact4_timing.sh"
# The two files the smoke phase's forbidden fragments must stay bound to: the
# runtime validators live in the patcher, the teardown flush in the driver.
PATCHER = ROOT / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
VARIANT = ROOT / "scripts" / "fr13_bigdenom_swe_serve_variant.sh"


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


def test_both_runners_share_one_route_preamble() -> None:
    """The launcher inherits fixed32 route pins from the process environment.

    The floor sequence is the only place FR13_FIXED32_TAW_WALK_CAP is
    exported, and the launcher reads it as "${NAME:-}", so a caller that skips
    the preamble hands it the empty string and dies pre-docker with the opaque
    "fixed32 integer route pin is malformed" (boot screen, 050d5ae9b).
    """
    shared = SHARED_ENV.read_text(encoding="utf-8")
    diag = DIAG.read_text(encoding="utf-8")
    timing = TIMING.read_text(encoding="utf-8")

    assert "fr13_fixed32_sfwd_fusion_route_preamble() {" in shared
    for source in (diag, timing):
        assert source.count("fr13_fixed32_sfwd_fusion_route_preamble") == 1
        # The preamble body must not be re-inlined next to the call.
        assert "source scripts/fr13_canonical_env.sh" not in source
        assert "run_variant() { :; }" not in source
    assert "source scripts/fr13_canonical_env.sh" in shared
    assert "run_variant() { :; }" in shared
    assert "FR13_FIXED32_TAW_WALK_CAP" in shared


def test_route_preamble_exports_the_pin_the_launcher_validates(
    tmp_path: Path,
) -> None:
    """Regression: run the real preamble and read the pin back."""
    script = tmp_path / "preamble.sh"
    script.write_text(
        "set -euo pipefail\n"
        f"cd {ROOT}\n"
        "TAG=pytest\n"
        "source scripts/fr13_fixed32_sfwd_fusion_env.sh\n"
        "fr13_fixed32_sfwd_fusion_route_preamble\n"
        'printf "%s\\n" "$FR13_FIXED32_TAW_WALK_CAP" '
        '"$FR13_MANDATORY_WEIGHT_BYTES" "$LUMO_SWE_AUTOCOMMIT"\n',
        encoding="utf-8",
    )
    out = subprocess.run(
        ["bash", str(script)], check=True, capture_output=True, text=True
    ).stdout.split()
    walk_cap, weight_bytes, autocommit = out
    assert walk_cap.isdigit() and int(walk_cap) > 0
    assert weight_bytes == "32666638208"
    assert autocommit == "0"


def test_route_preamble_refuses_without_a_tag(tmp_path: Path) -> None:
    script = tmp_path / "preamble_notag.sh"
    script.write_text(
        "set -uo pipefail\n"
        f"cd {ROOT}\n"
        "unset TAG\n"
        "source scripts/fr13_fixed32_sfwd_fusion_env.sh\n"
        "fr13_fixed32_sfwd_fusion_route_preamble && echo UNEXPECTED\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(script)], capture_output=True, text=True
    )
    assert "requires TAG" in result.stderr
    assert "UNEXPECTED" not in result.stdout


def test_diag_screens_the_pair_subset_not_a_diagnostic_subset() -> None:
    """A 1-task diagnostic subset would flip FR13_FIXED32_B1_DIAGNOSTIC."""
    diag = DIAG.read_text(encoding="utf-8")
    timing = TIMING.read_text(encoding="utf-8")
    assert "SUBSET=config/fr13_fixed32/subset_b4_four.json" in diag
    assert "subset_b4_four.json" in timing
    assert "FR13_FIXED32_B1_DIAGNOSTIC=0" in _shared_env_array(
        "FR13_FIXED32_SFWD_FUSION_ENV"
    )


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


def test_diag_smokes_the_engine_after_health() -> None:
    """Health proves capture; the runtime validators need a measured forward.

    Reaching /health only proves EngineCore init and the final FULL capture. The
    census/commit/replay family is reached exclusively by a real request, so the
    screen serves a few before it decides anything.
    """
    diag = DIAG.read_text(encoding="utf-8")
    assert "FR13_BOOT_DIAG_SMOKE=${FR13_BOOT_DIAG_SMOKE:-1}" in diag
    assert "FR13_BOOT_DIAG_SMOKE must be exactly 0 or 1" in diag
    code = _strip_comments(diag)
    # Real OpenAI chat traffic through the engine's own fixed32 ingress, which
    # admits nothing without a campaign, a task key and a bound wire ID.
    for pin in (
        "/fr13/fixed32/ingress/begin",
        "/fr13/fixed32/ingress/finalize",
        "/v1/chat/completions",
        "X-Fr13-Task-Key-ID",
        "X-Request-ID",
        "load_fixed32_ingress_secrets",
        "fixed32_canonical_task_set_sha256",
        "fixed32_task_key_id",
        '"fr13-chat-"',
        # Served text alone cannot prove the drafter ran.
        "vllm:spec_decode_num_drafts_total",
        "vllm:spec_decode_num_draft_tokens_total",
    ):
        assert pin in code, pin
    # One canonical task key, deliberately not all four: authenticating every
    # key publishes the SFWD real-event arm, and a screen must mint no evidence.
    assert "SMOKE_TASK_ID=astropy__astropy-12907" in code
    # Bounded on both axes so the phase cannot grow into a run.
    assert "SMOKE_REQUESTS=${FR13_BOOT_DIAG_SMOKE_REQUESTS:-6}" in code
    assert "SMOKE_MAX_TOKENS=${FR13_BOOT_DIAG_SMOKE_MAX_TOKENS:-96}" in code
    assert "FR13_BOOT_DIAG_SMOKE_REQUESTS must be 1..8" in code
    assert "FR13_BOOT_DIAG_SMOKE_MAX_TOKENS must be 1..256" in code


def test_smoke_knobs_never_reach_the_screened_container() -> None:
    """The launcher forwards every FR13_* variable it can see into the engine.

    fr13_launch_forked_fa2_tree_server.sh builds its -e list from
    ``compgen -v | grep -E '^(FR[0-9]+_|LUMO_|VLLM_)'`` against an opt-OUT skip
    list, so any exported FR13_BOOT_DIAG_SMOKE* knob would boot an engine whose
    environment differs from the timing pair's by exactly the variable this
    screen introduced -- and the screen is sound only while they match.
    """
    launcher = (ROOT / "scripts" / "fr13_launch_forked_fa2_tree_server.sh").read_text(
        encoding="utf-8"
    )
    assert "compgen -v | grep -E '^(FR[0-9]+_|LUMO_|VLLM_)'" in launcher
    assert "FR13_BOOT_DIAG_SMOKE" not in launcher

    code = _strip_comments(DIAG.read_text(encoding="utf-8"))
    # The selector is demoted (its value is still read); the rest are deleted.
    assert "export -n FR13_BOOT_DIAG_SMOKE" in code
    for knob in (
        "FR13_BOOT_DIAG_SMOKE_REQUESTS",
        "FR13_BOOT_DIAG_SMOKE_MAX_TOKENS",
        "FR13_BOOT_DIAG_SMOKE_TEARDOWN_GRACE_S",
    ):
        assert knob in code, knob
    unset_block = code[code.index("unset FR13_BOOT_DIAG_SMOKE_REQUESTS"):]
    for knob in (
        "FR13_BOOT_DIAG_SMOKE_REQUESTS",
        "FR13_BOOT_DIAG_SMOKE_MAX_TOKENS",
        "FR13_BOOT_DIAG_SMOKE_TEARDOWN_GRACE_S",
    ):
        assert knob in unset_block[: unset_block.index("\n\n")], knob
    # Nothing may re-export any of them afterwards.
    assert "export FR13_BOOT_DIAG_SMOKE" not in code


def test_smoke_knobs_are_scrubbed_from_a_real_shell(tmp_path: Path) -> None:
    """Regression: source the header and read the child environment back."""
    script = tmp_path / "scrub.sh"
    script.write_text(
        "set -euo pipefail\n"
        "FR13_BOOT_DIAG_SMOKE=${FR13_BOOT_DIAG_SMOKE:-1}\n"
        "SMOKE_REQUESTS=${FR13_BOOT_DIAG_SMOKE_REQUESTS:-6}\n"
        "SMOKE_MAX_TOKENS=${FR13_BOOT_DIAG_SMOKE_MAX_TOKENS:-96}\n"
        "SMOKE_TEARDOWN_GRACE_S=${FR13_BOOT_DIAG_SMOKE_TEARDOWN_GRACE_S:-180}\n"
        "export -n FR13_BOOT_DIAG_SMOKE\n"
        "unset FR13_BOOT_DIAG_SMOKE_REQUESTS FR13_BOOT_DIAG_SMOKE_MAX_TOKENS \\\n"
        "  FR13_BOOT_DIAG_SMOKE_TEARDOWN_GRACE_S\n"
        'printf "%s %s %s\\n" "$FR13_BOOT_DIAG_SMOKE" "$SMOKE_REQUESTS" '
        '"$SMOKE_MAX_TOKENS"\n'
        "env | grep -c '^FR13_BOOT_DIAG_SMOKE' || true\n",
        encoding="utf-8",
    )
    # Exactly the lines the diag's header runs, with every knob set the way an
    # operator would set them.
    result = subprocess.run(
        ["bash", str(script)],
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "FR13_BOOT_DIAG_SMOKE": "0",
            "FR13_BOOT_DIAG_SMOKE_REQUESTS": "4",
            "FR13_BOOT_DIAG_SMOKE_MAX_TOKENS": "64",
            "FR13_BOOT_DIAG_SMOKE_TEARDOWN_GRACE_S": "240",
        },
    )
    values, forwarded = result.stdout.split("\n")[:2]
    # The operator's values survive inside the script ...
    assert values.split() == ["0", "4", "64"]
    # ... and none of them survives into anything the script launches.
    assert forwarded == "0"


def test_diag_freezes_the_campaign_driver_for_the_smoke() -> None:
    """The engine admits exactly one ingress campaign, so only one may open it.

    Left running past health the driver opens its own; frozen, the screen owns
    the campaign. The CONT must come back before the TERM or the driver never
    runs the EXIT trap that performs the terminal flush.
    """
    code = _strip_comments(DIAG.read_text(encoding="utf-8"))
    assert "kill -STOP" in code
    assert "VARIANT_FROZEN=1" in code
    teardown = code[code.index("teardown() {"):]
    assert "kill -CONT" in teardown
    assert teardown.index("kill -CONT") < teardown.index("kill -TERM")
    # The flush needs room to finish before the screen kills what is running it.
    assert "TEARDOWN_GRACE_S" in code
    # Freezing before the driver publishes its flush prerequisites would make it
    # report a missing PID -- a flush failure that never happened. The screen
    # waits for the evidence, and reads no flush verdict without it.
    assert "fixed32_process_identity.json" in code
    assert "fr13_fixed32_flush_ack.json" in code
    assert "SMOKE_FLUSH_READY=1" in code
    assert "SMOKE_RAN == 1 && SMOKE_FLUSH_READY == 1" in code
    assert code.index("SMOKE_FLUSH_READY=1") < code.index("kill -STOP")


def test_diag_sweeps_the_validator_and_flush_lists_from_the_shared_env() -> None:
    """Three lists, one home: the screen inlines none of them."""
    code = _strip_comments(DIAG.read_text(encoding="utf-8"))
    assert "FR13_FIXED32_SFWD_FUSION_CENSUS_FORBIDDEN[@]" in code
    assert "FR13_FIXED32_SFWD_FUSION_FLUSH_FORBIDDEN[@]" in code
    # The capture-time list is swept twice: once at boot and again after the
    # smoke, because a fallback can appear on a forward that never appeared at
    # capture.
    assert code.count("FR13_FIXED32_SFWD_FUSION_FORBIDDEN[@]") == 2


def test_census_forbidden_strings_are_live_runtime_raise_sites() -> None:
    """A renamed validator must fail the suite, not void the sweep in silence."""
    fragments = tuple(
        ast.literal_eval(entry)
        for entry in _shared_env_array("FR13_FIXED32_SFWD_FUSION_CENSUS_FORBIDDEN")
    )
    assert len(fragments) == len(set(fragments))
    assert len(fragments) >= 12
    patcher = PATCHER.read_text(encoding="utf-8")
    for fragment in fragments:
        assert fragment in patcher, fragment
    # The four validator families the screen exists to sweep.
    for family in (
        "census drift",
        "counters mismatch",
        "forward work is incomplete",
        "geometry drifted",
    ):
        assert family in fragments, family


def test_flush_forbidden_strings_are_live_teardown_failure_sites() -> None:
    fragments = tuple(
        ast.literal_eval(entry)
        for entry in _shared_env_array("FR13_FIXED32_SFWD_FUSION_FLUSH_FORBIDDEN")
    )
    assert len(fragments) == len(set(fragments))
    patcher = PATCHER.read_text(encoding="utf-8")
    variant = VARIANT.read_text(encoding="utf-8")
    for fragment in fragments:
        assert fragment in patcher or fragment in variant, fragment
    # Both sides of the flush: the engine's own report and the driver's verdict.
    assert "[FR13_FIXED32_FLUSH] failed generation" in fragments
    assert "FAIL: fixed32 terminal flush" in fragments


def test_diag_documents_and_uses_the_smoke_verdicts() -> None:
    diag = DIAG.read_text(encoding="utf-8")
    assert "#   smoke_request_failed     7" in diag
    assert "#   smoke_validator_tripped  8" in diag
    code = _strip_comments(diag)
    for verdict in ("smoke_request_failed", "smoke_validator_tripped"):
        assert f"VERDICT={verdict}" in code, verdict
    assert "exit 7" in code
    assert "exit 8" in code


def test_diag_summary_carries_the_smoke_fields_and_stays_non_citable() -> None:
    diag = DIAG.read_text(encoding="utf-8")
    assert '"schema": "fr13.fixed32.sfwd_fusion_boot_diag.v3"' in diag
    for field in (
        '"smoke_enabled"',
        '"smoke_ran"',
        '"smoke_requests_sent"',
        '"smoke_responses_ok"',
        '"smoke_validator_strings_clean"',
        '"smoke_flush_strings_clean"',
        '"smoke_producer_alive_at_teardown"',
    ):
        assert field in diag, field
    # A sweep that never ran reports null, never a default-clean False.
    assert '"unchecked": None' in diag
    assert "SMOKE_VALIDATORS_CLEAN=unchecked" in diag
    assert "SMOKE_FLUSH_CLEAN=unchecked" in diag
    assert "SMOKE_PRODUCER_ALIVE=unchecked" in diag
    # Serving real traffic must not make the screen citable.
    assert '"classification": "diagnostic_boot_only"' in diag
    assert '"citable": False' in diag
    assert '"timing_eligible": False' in diag
    assert '"fr13.fixed32.sfwd_fusion_boot_diag_smoke.v1"' in diag


def test_diag_smoke_python_blocks_compile() -> None:
    """The phase is an embedded driver; bash -n cannot see inside its heredoc."""
    source = DIAG.read_text(encoding="utf-8")
    blocks = re.findall(r"<<'PY'.*?\n(.*?)\nPY\n", source, re.DOTALL)
    assert len(blocks) == 4
    for index, block in enumerate(blocks):
        lines = block.splitlines()
        # Drop the invoking command's own backslash continuations, which the
        # heredoc marker precedes on the same line.
        while lines and not lines[0].startswith("import "):
            lines.pop(0)
        assert lines, f"block {index} has no python body"
        compile("\n".join(lines) + "\n", f"<diag block {index}>", "exec")


def _smoke_sampling() -> dict[str, float | int]:
    """The sampling shape the shared env hands the smoke driver."""
    types = {
        "temperature": float,
        "top_p": float,
        "top_k": int,
        "presence_penalty": float,
        "min_p": float,
    }
    shape: dict[str, float | int] = {}
    for entry in _shared_env_array("FR13_FIXED32_SFWD_FUSION_SMOKE_SAMPLING"):
        key, _, value = ast.literal_eval(entry).partition("=")
        shape[key] = types[key](value)
    return shape


def test_smoke_sampling_matches_the_campaign_proxy_pins() -> None:
    """The screen bypasses the proxy, so it must carry what the proxy stamps.

    The pair's bodies are normalised by inference_proxy from LUMO_PROXY_FORCE_*;
    the screen freezes the driver before that proxy exists and talks to the
    engine directly. These values are those pins, read from the helper that
    launches the campaign's proxy, so a campaign that re-pins its sampling fails
    this suite instead of quietly leaving the screen testing another engine.
    """
    helper = (
        ROOT / "scripts" / "swe_x86_helpers" / "relaunch_proxy_remote.sh"
    ).read_text(encoding="utf-8")
    pins = {
        "temperature": "LUMO_PROXY_FORCE_TEMPERATURE",
        "top_p": "LUMO_PROXY_FORCE_TOP_P",
        "top_k": "LUMO_PROXY_FORCE_TOP_K",
        "presence_penalty": "LUMO_PROXY_FORCE_PRESENCE_PENALTY",
        "min_p": "LUMO_PROXY_FORCE_MIN_P",
    }
    sampling = _smoke_sampling()
    assert sorted(sampling) == sorted(pins)
    for key, env_name in pins.items():
        match = re.search(
            rf"^[ \t]*export {env_name}=\$\{{{env_name}:-([^}}]*)\}}$",
            helper,
            re.MULTILINE,
        )
        assert match is not None, f"{env_name} default not found in {helper!r}"[:120]
        assert float(match.group(1)) == float(sampling[key]), key
    # The qwen sampling profile is the default-ON path; only the lossless A/B
    # gate opts out, and the screen is not that gate.
    assert '"${LUMO_PROXY_QWEN_SAMPLING:-1}" = "1"' in helper
    # And the campaign driver's own temperature pin agrees.
    variant = VARIANT.read_text(encoding="utf-8")
    assert 'LUMO_PROXY_FORCE_TEMPERATURE="${DEPLOY_FORCE_TEMP:-0.6}"' in variant
    assert sampling["temperature"] == 0.6


def test_smoke_sampling_is_what_the_proxy_would_have_stamped() -> None:
    """Compare against the production normaliser rather than a second opinion."""
    from lumo_flywheel_serving.inference_proxy import (
        normalize_chat_completions_request_payload,
    )

    campaign_env = {
        "LUMO_PROXY_FORCE_TEMPERATURE": "0.6",
        "LUMO_PROXY_FORCE_TOP_P": "0.95",
        "LUMO_PROXY_FORCE_TOP_K": "20",
        "LUMO_PROXY_FORCE_PRESENCE_PENALTY": "1.0",
        "LUMO_PROXY_FORCE_MIN_P": "0",
        "LUMO_PROXY_MAX_OUTPUT_TOKENS": "32768",
    }
    base = {"model": "qwen3.6-27b", "messages": [], "max_tokens": 96, "stream": False}
    with mock.patch.dict(os.environ, campaign_env, clear=False):
        stamped = normalize_chat_completions_request_payload(base)
    sampling = _smoke_sampling()
    assert {key: stamped[key] for key in sampling} == sampling
    # The screen's short generations survive the campaign's output cap untouched.
    assert stamped["max_tokens"] == 96


def test_smoke_refuses_a_greedy_or_partial_sampling_shape(tmp_path: Path) -> None:
    """fixed32 refuses greedy decoding, and refusing it late costs the engine.

    temperature 0.0 makes the rejection sampler raise "FR13 fixed32 requires
    sampled temp>0" before the proposal seal, which kills EngineCore and answers
    with a bare 500 -- the 2026-08-08 failure. The driver refuses first, where
    the message survives and no engine dies to deliver it.
    """
    source = DIAG.read_text(encoding="utf-8")
    lines = re.findall(r"<<'PY'.*?\n(.*?)\nPY\n", source, re.DOTALL)[3].splitlines()
    while not lines[0].startswith("import "):
        lines.pop(0)
    driver = tmp_path / "driver.py"
    driver.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run(*sampling: str) -> subprocess.CompletedProcess:
        # Validation precedes every network call, so no engine is needed.
        return subprocess.run(
            [sys.executable, str(driver), "secret", "subset", "task", "9950",
             "model", "1", "96", "record", "evidence", *sampling],
            capture_output=True,
            text=True,
        )

    good = [f"{key}={value}" for key, value in _smoke_sampling().items()]
    greedy = run(*[e if not e.startswith("temperature=") else "temperature=0.0"
                   for e in good])
    assert greedy.returncode != 0
    assert "temperature must be > 0" in greedy.stdout + greedy.stderr

    partial = run("temperature=0.6", "top_p=0.95")
    assert partial.returncode != 0
    assert "incomplete" in partial.stdout + partial.stderr

    unknown = run(*good, "repetition_penalty=1.2")
    assert unknown.returncode != 0
    assert "not usable" in unknown.stdout + unknown.stderr

    # The full shape gets past validation and fails later, on the network.
    ok = run(*good)
    assert "sampling" not in ok.stdout + ok.stderr


def test_smoke_payload_carries_the_shape_and_no_hardcoded_greedy() -> None:
    code = _strip_comments(DIAG.read_text(encoding="utf-8"))
    assert "**sampling," in code
    assert '"temperature": 0.0' not in code
    assert "FR13_FIXED32_SFWD_FUSION_SMOKE_SAMPLING[@]" in code
    # The shape reaches the runroot, so a run can be read back for what it sent.
    assert '"sampling": sampling,' in code
    assert "sampling=sampling," in code
    # The contract's raise site is swept like the other runtime validators.
    assert "requires sampled temp>0" in SHARED_ENV.read_text(encoding="utf-8")


def _smoke_driver(tmp_path: Path) -> Path:
    """The screen's embedded smoke driver, lifted out to be run for real."""
    lines = re.findall(
        r"<<'PY'.*?\n(.*?)\nPY\n", DIAG.read_text(encoding="utf-8"), re.DOTALL
    )[3].splitlines()
    while not lines[0].startswith("import "):
        lines.pop(0)
    driver = tmp_path / "smoke_driver.py"
    driver.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return driver


# The message shape the engine actually returned at 1b2a89d6d: the qwen thinking
# profile answered into the reasoning channel and stopped on length at 96 tokens
# with content still null.
LIVE_REASONING_ONLY = {
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "refusal": None,
                "tool_calls": [],
                "reasoning": "Here's a thinking process:\n\n1. **Analyze**",
            },
            "finish_reason": "length",
        }
    ],
    "usage": {"prompt_tokens": 24, "completion_tokens": 96, "total_tokens": 120},
}


def _run_smoke_driver(tmp_path: Path, chat_body: dict, count: int = 1):
    """Drive the smoke against a stub engine that returns `chat_body`."""
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    body_text = _json.dumps(chat_body)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def _send(self, code: int, body: str, ctype: str = "application/json"):
            raw = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/metrics":
                self._send(
                    200,
                    "vllm:spec_decode_num_drafts_total 9.0\n"
                    "vllm:spec_decode_num_draft_tokens_total 31.0\n",
                    "text/plain",
                )
            else:
                self._send(200, "")

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path == "/v1/chat/completions":
                self._send(200, body_text)
            elif self.path.endswith("/begin"):
                self._send(200, "{}")
            else:
                self._send(
                    200,
                    _json.dumps(
                        {
                            "accepted_engine_requests": count,
                            "completed_engine_requests": count,
                        }
                    ),
                )

    tmp_path.mkdir(parents=True, exist_ok=True)
    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        from lumo_flywheel_serving.inference_proxy import (
            FIXED32_INGRESS_SECRETS_SCHEMA,
        )

        secret = tmp_path / "secret.json"
        secret.write_text(
            json.dumps(
                {
                    "schema": FIXED32_INGRESS_SECRETS_SCHEMA,
                    "task_hmac_key_hex": "ab" * 32,
                    "engine_bearer": "b" * 48,
                }
            ),
            encoding="utf-8",
        )
        secret.chmod(0o600)
        task_id = "astropy__astropy-12907"
        subset = tmp_path / "subset.json"
        subset.write_text(
            json.dumps({"instance_ids": [task_id, "b", "c", "d"]}), encoding="utf-8"
        )
        record = tmp_path / "smoke_summary.json"
        evidence = tmp_path / "smoke_requests.jsonl"
        proc = subprocess.run(
            [
                sys.executable, str(_smoke_driver(tmp_path)), str(secret),
                str(subset), task_id, str(port), "qwen3.6-27b", str(count), "96",
                str(record), str(evidence),
                *[f"{key}={value}" for key, value in _smoke_sampling().items()],
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        )
    finally:
        server.shutdown()
    summary = json.loads(record.read_text(encoding="utf-8")) if record.exists() else None
    steps = (
        [json.loads(line) for line in evidence.read_text(encoding="utf-8").splitlines()]
        if evidence.exists()
        else []
    )
    return proc, summary, steps


def test_smoke_counts_a_thinking_only_turn_as_served(tmp_path: Path) -> None:
    """The 1b2a89d6d trip: real output, in the reasoning channel, called empty.

    The qwen thinking profile the screen now pins emits into the reasoning
    channel first, and at max_tokens=96 the turn stops on length while still
    inside it. The campaign accounts for that as a served turn -- its own trace
    records thinking blocks beside text and tool_use -- and so must the screen,
    whose interest is the forward rather than the answer.
    """
    proc, summary, steps = _run_smoke_driver(tmp_path, LIVE_REASONING_ONLY, count=2)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert summary is not None
    assert summary["responses_ok"] == 2
    assert summary["responses_reasoning_only"] == 2
    # The bash-side count contract is grepped off this exact prefix.
    assert proc.stdout.count("smoke response ok ") == 2
    served = [step for step in steps if step["event"] == "chat_served"]
    assert len(served) == 2
    assert served[0]["finish_reason"] == "length"
    assert served[0]["completion_tokens"] == 96
    assert served[0]["channels"] == {"reasoning": len(
        LIVE_REASONING_ONLY["choices"][0]["message"]["reasoning"]
    )}


def test_smoke_still_refuses_a_turn_that_produced_nothing(tmp_path: Path) -> None:
    """Accepting the reasoning channel must not accept an empty response."""
    import copy

    blank = copy.deepcopy(LIVE_REASONING_ONLY)
    blank["choices"][0]["message"]["reasoning"] = None
    proc, summary, _ = _run_smoke_driver(tmp_path / "blank", blank)
    assert proc.returncode != 0
    assert "produced no text" in proc.stdout + proc.stderr
    assert summary is None

    for finish_reason in (None, "content_filter"):
        unfinished = copy.deepcopy(LIVE_REASONING_ONLY)
        unfinished["choices"][0]["finish_reason"] = finish_reason
        proc, summary, _ = _run_smoke_driver(
            tmp_path / f"unfinished{finish_reason}", unfinished
        )
        assert proc.returncode != 0, finish_reason
        assert "did not finish" in proc.stdout + proc.stderr
        assert summary is None


def test_smoke_does_not_miscount_a_plain_answer(tmp_path: Path) -> None:
    """A turn that reached the content channel is served and not reasoning-only."""
    import copy

    answered = copy.deepcopy(LIVE_REASONING_ONLY)
    answered["choices"][0]["message"]["content"] = "a real answer"
    answered["choices"][0]["message"]["reasoning"] = None
    answered["choices"][0]["finish_reason"] = "stop"
    proc, summary, steps = _run_smoke_driver(tmp_path, answered)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert summary["responses_ok"] == 1
    assert summary["responses_reasoning_only"] == 0
    served = [step for step in steps if step["event"] == "chat_served"][0]
    assert served["channels"] == {"content": len("a real answer")}
    assert served["finish_reason"] == "stop"


def _diag_function(name: str) -> str:
    """Lift one bash function out of the screen so a test can run it for real."""
    match = re.search(
        rf"^{name}\(\) \{{\n(.*?)^\}}$",
        DIAG.read_text(encoding="utf-8"),
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"{DIAG} lacks function {name}"
    return f"{name}() {{\n{match.group(1)}}}\n"


def test_container_log_is_captured_before_any_teardown_signal() -> None:
    """The engine's stack trace lives in a log teardown is about to destroy.

    The 2026-08-08 screen answered "HTTP 500 EngineCore encountered an issue"
    with boot_docker.log still ending at /health: the request-failure path exited
    straight to teardown, which removes the container, and the only re-read of
    the log sat on the success path below it.
    """
    code = _strip_comments(DIAG.read_text(encoding="utf-8"))
    teardown = code[code.index("teardown() {"):]
    assert "capture_container_log" in teardown
    # Before the driver is continued, TERMed, KILLed, or any container removed.
    for signal in ("kill -CONT", "kill -TERM", "kill -KILL", "docker rm -f"):
        assert teardown.index("capture_container_log") < teardown.index(signal), signal
    # And the smoke's own failure path dumps it rather than exiting blind.
    assert code.count("capture_container_log") >= 3


def test_container_log_capture_never_clobbers_a_good_capture(tmp_path: Path) -> None:
    """A reaped container must not turn the evidence into an empty file."""
    stub = tmp_path / "bin"
    stub.mkdir()
    docker = stub / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        'case "$DOCKER_STUB" in\n'
        '  content) echo "engine traceback line"; exit 0;;\n'
        '  empty) exit 0;;\n'
        '  gone) echo "Error: No such container" >&2; exit 1;;\n'
        "esac\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    log = tmp_path / "boot_docker.log"

    def run(mode: str, container: str = "cid") -> int:
        script = (
            "set -u\n"
            f'CONTAINER_ID="{container}"\n'
            f'DOCKER_LOG="{log}"\n'
            + _diag_function("capture_container_log")
            + "capture_container_log\n"
        )
        return subprocess.run(
            ["bash", "-c", script],
            env={"PATH": f"{stub}:/usr/bin:/bin", "DOCKER_STUB": mode},
        ).returncode

    assert run("content") == 0
    assert log.read_text(encoding="utf-8") == "engine traceback line\n"
    # Container reaped, and container never promoted: the good capture stands.
    assert run("gone") == 1
    assert log.read_text(encoding="utf-8") == "engine traceback line\n"
    assert run("empty") == 1
    assert log.read_text(encoding="utf-8") == "engine traceback line\n"
    assert run("content", container="") == 1
    assert log.read_text(encoding="utf-8") == "engine traceback line\n"
    # No debris left beside the evidence.
    assert not (tmp_path / "boot_docker.log.partial").exists()


def test_teardown_probes_the_producer_before_it_signals() -> None:
    """Asked after the CONT/TERM, the answer would describe teardown's own work."""
    code = _strip_comments(DIAG.read_text(encoding="utf-8"))
    teardown = code[code.index("teardown() {"):]
    assert "SMOKE_PRODUCER_ALIVE=true" in teardown
    assert "SMOKE_PRODUCER_ALIVE=false" in teardown
    assert teardown.index("SMOKE_PRODUCER_ALIVE=true") < teardown.index("kill -CONT")
    # The probe addresses the same EngineCore the terminal flush addresses.
    assert "smoke_producer_pid" in code
    assert "engine_core" in code
    assert "docker exec" in teardown


def test_request_failure_outranks_the_flush_it_caused() -> None:
    """Cause before consequence: 0 served completions is an rc=7, not an rc=8.

    An engine that dies serving the smoke leaves the terminal flush no live
    producer, so the flush fails too. Promoting that to smoke_validator_tripped
    names the symptom and buries the cause -- which is exactly the verdict the
    2026-08-08 screen published against smoke_responses_ok=0.
    """
    code = _strip_comments(DIAG.read_text(encoding="utf-8"))
    teardown = code[code.index("teardown() {"):]
    sweep = teardown[teardown.index("SMOKE_FLUSH_CLEAN=true"):]
    # The flush finding is still published...
    assert "SMOKE_FLUSH_CLEAN=false" in sweep
    # ...but it only owns the verdict when the smoke actually served something.
    assert "(( SMOKE_OK == 0 ))" in sweep
    assert sweep.index("(( SMOKE_OK == 0 ))") < sweep.index(
        "VERDICT=smoke_validator_tripped"
    )
    guarded = sweep[sweep.index("(( SMOKE_OK == 0 ))"):]
    assert "VERDICT=smoke_request_failed" in guarded
    assert guarded.index("VERDICT=smoke_request_failed") < guarded.index(
        "VERDICT=smoke_validator_tripped"
    )
    assert "rc=7" in guarded and "rc=8" in guarded


def _run_teardown(tmp_path: Path, *, smoke_ok: int, flush_failed: bool) -> dict:
    """Run the screen's real teardown() over stubs and read back its summary."""
    stub = tmp_path / "bin"
    stub.mkdir(parents=True)
    # ps -aq must come back empty or teardown reports unclean Docker state.
    (stub / "docker").write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    (stub / "docker").chmod(0o755)
    (stub / "free").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (stub / "free").chmod(0o755)
    armdir = tmp_path / "arm"
    armdir.mkdir()
    runlog = tmp_path / "arm.runlog"
    runlog.write_text(
        "FAIL: fixed32 terminal flush rc=2\n" if flush_failed else "all clean\n",
        encoding="utf-8",
    )
    summary = tmp_path / "boot_diag_summary.json"
    source = DIAG.read_text(encoding="utf-8")
    script = "\n".join(
        (
            "set -uo pipefail",
            f'RUNROOT_ABS="{tmp_path}"',
            f'ARMDIR="{armdir}"',
            f'RUNLOG="{runlog}"',
            f'SUMMARY="{summary}"',
            f'DOCKER_LOG="{tmp_path}/boot_docker.log"',
            f'PYTHON_BIN="{sys.executable}"',
            'ARM=arm; SOURCE_COMMIT=deadbeef; DIAG_SHA256=cafe; CONTAINER_ID=""',
            'VARIANT_PID=""; VARIANT_FROZEN=0; FR13_BOOT_DIAG_SMOKE=1',
            "SMOKE_RAN=1; SMOKE_FLUSH_READY=1",
            f"SMOKE_OK={smoke_ok}; SMOKE_SENT=1",
            "SMOKE_VALIDATORS_CLEAN=unchecked; SMOKE_FLUSH_CLEAN=unchecked",
            "SMOKE_PRODUCER_ALIVE=unchecked; TEARDOWN_GRACE_S=1",
            "VERDICT=smoke_request_failed",
            'DETAIL="smoke traffic failed: HTTP 500 EngineCore encountered an issue"'
            if smoke_ok == 0
            else 'VERDICT=pass; DETAIL="boot and smoke clean"',
            'FR13_FIXED32_SFWD_FUSION_FLUSH_FORBIDDEN=("FAIL: fixed32 terminal flush")',
            _diag_function("capture_container_log"),
            _diag_function("smoke_producer_pid"),
            re.search(r"^teardown\(\) \{\n.*?^\}$", source, re.MULTILINE | re.DOTALL)
            .group(0),
            f"(exit {7 if smoke_ok == 0 else 0}); teardown",
        )
    )
    subprocess.run(
        ["bash", "-c", script],
        env={"PATH": f"{stub}:/usr/bin:/bin"},
        capture_output=True,
    )
    return json.loads(summary.read_text(encoding="utf-8"))


def test_teardown_verdict_names_the_cause_not_the_consequence(tmp_path: Path) -> None:
    """The regression this exists for, run rather than pattern-matched.

    A smoke that served nothing leaves the terminal flush no live producer, so
    the flush fails too. The 2026-08-08 screen published that flush failure as
    the verdict -- rc=8 smoke_validator_tripped against smoke_responses_ok=0 --
    which names the symptom and buries the cause.
    """
    failed = _run_teardown(tmp_path / "a", smoke_ok=0, flush_failed=True)
    assert failed["verdict"] == "smoke_request_failed"
    assert failed["exit_code"] == 7
    assert failed["smoke_responses_ok"] == 0
    # The flush finding is still published, it just does not own the verdict.
    assert failed["smoke_flush_strings_clean"] is False
    assert "dead producer" in failed["detail"]
    assert "EngineCore" in failed["detail"]

    # A flush string after a smoke that DID serve is a real validator trip.
    tripped = _run_teardown(tmp_path / "b", smoke_ok=3, flush_failed=True)
    assert tripped["verdict"] == "smoke_validator_tripped"
    assert tripped["exit_code"] == 8
    assert tripped["smoke_flush_strings_clean"] is False

    # And a clean flush stays clean.
    clean = _run_teardown(tmp_path / "c", smoke_ok=3, flush_failed=False)
    assert clean["verdict"] == "pass"
    assert clean["exit_code"] == 0
    assert clean["smoke_flush_strings_clean"] is True
    for outcome in (failed, tripped, clean):
        assert outcome["schema"] == "fr13.fixed32.sfwd_fusion_boot_diag.v3"
        assert outcome["citable"] is False
        # No container was promoted, so liveness is unknowable, never a bare False.
        assert outcome["smoke_producer_alive_at_teardown"] is None


def test_smoke_records_per_step_request_evidence() -> None:
    """The runroot must name which step failed and what the engine answered."""
    code = _strip_comments(DIAG.read_text(encoding="utf-8"))
    assert "smoke_requests.jsonl" in code
    assert "SMOKE_EVIDENCE" in code
    assert '"fr13.fixed32.sfwd_fusion_boot_diag_smoke_step.v1"' in code
    for event in (
        '"ingress_begin"',
        '"chat_dispatch"',
        '"chat_response"',
        '"ingress_finalize"',
        '"spec_decode_counters"',
        '"post_smoke_health"',
    ):
        assert event in code, event
    # Status and body, written as the phase goes: a step that dies still leaves
    # its record, and 200 chars of a 500 is not a diagnosis.
    assert "status_code=response.status_code" in code
    assert "body_head=response.text[:2000]" in code
    assert "handle.flush()" in code
    assert "response.text[:200]\n" not in code
