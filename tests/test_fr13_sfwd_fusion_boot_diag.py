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
    assert '"schema": "fr13.fixed32.sfwd_fusion_boot_diag.v2"' in diag
    for field in (
        '"smoke_enabled"',
        '"smoke_ran"',
        '"smoke_requests_sent"',
        '"smoke_responses_ok"',
        '"smoke_validator_strings_clean"',
        '"smoke_flush_strings_clean"',
    ):
        assert field in diag, field
    # A sweep that never ran reports null, never a default-clean False.
    assert '"unchecked": None' in diag
    assert "SMOKE_VALIDATORS_CLEAN=unchecked" in diag
    assert "SMOKE_FLUSH_CLEAN=unchecked" in diag
    # Serving real traffic must not make the screen citable.
    assert '"classification": "diagnostic_boot_only"' in diag
    assert '"citable": False' in diag
    assert '"timing_eligible": False' in diag
    assert '"fr13.fixed32.sfwd_fusion_boot_diag_smoke.v1"' in diag


def test_diag_smoke_python_blocks_compile() -> None:
    """The phase is an embedded driver; bash -n cannot see inside its heredoc."""
    source = DIAG.read_text(encoding="utf-8")
    blocks = re.findall(r"<<'PY'.*?\n(.*?)\nPY\n", source, re.DOTALL)
    assert len(blocks) == 3
    for index, block in enumerate(blocks):
        lines = block.splitlines()
        # Drop the invoking command's own backslash continuations, which the
        # heredoc marker precedes on the same line.
        while lines and not lines[0].startswith("import "):
            lines.pop(0)
        assert lines, f"block {index} has no python body"
        compile("\n".join(lines) + "\n", f"<diag block {index}>", "exec")
