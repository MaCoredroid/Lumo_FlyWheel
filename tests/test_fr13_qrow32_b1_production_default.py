"""The promoted B1 production FA2 arm: canonical default, launcher agreement.

PROMOTED 2026-08-13 on Mark's ruling ("B1 flip Yes"). Until this promotion the
B1 GQA-pair FA2 unit was byte-gate-qualified and timing-proven but had no
production standing: ``FR13_FA2_QROW32_B1_PRODUCTION_ARM`` defaulted empty, so
a B1 serve that did not name an arm served whatever the loaded binary's own
default path was, and the only configuration that could serve the candidate was
the timing pair.

EVIDENCE THE FLIP RESTS ON
  byte gate   Tier-A sealed PASS + re-seals for the GQA-pair B1 translation
              unit; binary pinned in three agreeing sites by 3120b3765
              (.so 3560cdc0..., 299815552 B, closure 172b5e71...).
  timing      output/fr13_fa2_qrow32_gqa_pair_b1_timing_20260812T073429Z, real
              SWE-Verified exact4 traffic at B=1/concurrency 1:
              step_wall 232.360 ms (candidate) vs 236.765 ms (qrow16 incumbent)
              = -4.405 ms / -1.86%; per-request TPS 22.769 -> 23.155;
              promotion_eligible = true, "aggregate gain with no per-request
              regression".

WHAT THESE TESTS PROTECT

1. ONE SOURCE OF TRUTH. scripts/fr13_canonical_env.sh owns the shipped value.
   The launcher restates it as a fallback, and the two may never disagree --
   the same failure mode the mamba-narrowing promotion left open for two days
   (tests/test_fr13_mamba_spec_blocks_cdiv.py records it). So compare the files
   rather than restating a literal in a third place.

2. THE DEFAULT IS BATCH-1-SCOPED AND OPT-OUTABLE. A B1 selector is illegal at
   B > 1, and the registry is sourced by fr13_b4_campaign_driver.sh before
   BSIZE is read, so the registry cannot export the selector itself. The
   launcher applies it, and only in the shape where it is legal and
   unambiguous. Naming the variable -- including naming it EMPTY, which is how
   every non-B1 arm in this tree declares itself -- is always obeyed.

3. THE CREDENTIAL CHAIN IS UNCHANGED. Promotion changes which arm is selected,
   never what that arm must prove. Every downstream refusal stays reachable.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CANONICAL_ENV = REPO / "scripts" / "fr13_canonical_env.sh"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
CAMPAIGN_DRIVER = REPO / "scripts" / "fr13_b4_campaign_driver.sh"

CANONICAL_TEXT = CANONICAL_ENV.read_text(encoding="utf-8")
LAUNCHER_TEXT = LAUNCHER.read_text(encoding="utf-8")

PROMOTED_ARM = "gqa_pair"
DEFAULT_VAR = "FR13_FA2_QROW32_B1_PRODUCTION_ARM_DEFAULT"
ARM_VAR = "FR13_FA2_QROW32_B1_PRODUCTION_ARM"


def _shell_defaults(text: str, variable: str) -> set[str]:
    """Every ``${VAR:-X}`` default X that `text` gives `variable`."""
    return set(re.findall(r"\$\{" + re.escape(variable) + r":-([^}]*)\}", text))


# --------------------------------------------------------------------------
# 1. one source of truth
# --------------------------------------------------------------------------


def test_canonical_env_exports_the_promoted_b1_production_arm() -> None:
    assert (
        f'export {DEFAULT_VAR}="${{{DEFAULT_VAR}:-{PROMOTED_ARM}}}"'
        in CANONICAL_TEXT
    )


def test_canonical_env_records_the_ruling_and_the_measured_verdict() -> None:
    """A promotion that does not carry its evidence is an unexplained default."""
    entry = next(
        line for line in CANONICAL_TEXT.splitlines() if line.startswith(f"export {DEFAULT_VAR}=")
    )
    assert "PROMOTED 2026-08-13" in entry
    assert "B1 flip Yes" in entry
    # the timing verdict, by run root and by number
    assert "fr13_fa2_qrow32_gqa_pair_b1_timing_20260812T073429Z" in entry
    assert "232.360" in entry and "236.765" in entry
    assert "promotion_eligible=true" in entry
    # the binary the arm is credentialed against
    assert (
        "3560cdc0c1ebbe3d912858ea447b350edefc0d6749950d6353e5f763185da6ae"
        in entry
    )


def test_launcher_fallback_matches_the_canonical_shipped_default() -> None:
    """The launcher may never contradict the registry.

    A future re-promotion that flips the canonical value must flip the launcher
    too, or this fails.
    """
    canonical = _shell_defaults(CANONICAL_TEXT, DEFAULT_VAR)
    assert len(canonical) == 1, f"canonical env must declare one default: {canonical}"
    launcher = _shell_defaults(LAUNCHER_TEXT, DEFAULT_VAR)
    assert launcher == canonical, (
        "launcher fallback disagrees with fr13_canonical_env.sh: "
        f"launcher={sorted(launcher)} canonical={sorted(canonical)}"
    )
    assert canonical == {PROMOTED_ARM}


def test_campaign_driver_sources_the_registry_before_launching() -> None:
    """Agreement is not resolution -- the registry must actually reach a run."""
    driver = CAMPAIGN_DRIVER.read_text(encoding="utf-8")
    assert 'source "$SCRIPT_DIR/fr13_canonical_env.sh"' in driver


def test_the_registry_cannot_export_the_selector_itself() -> None:
    """Why this is a *_DEFAULT and not the selector.

    fr13_b4_campaign_driver.sh sources the registry at line 29 and reads BSIZE
    at line 36. Exporting FR13_FA2_QROW32_B1_PRODUCTION_ARM from the registry
    would hand every B=4 campaign arm a B1 selector that the launcher must then
    refuse at boot, because a B1 selector demands MAX_NUM_SEQS == 1.
    """
    driver_lines = CAMPAIGN_DRIVER.read_text(encoding="utf-8").splitlines()
    source_index = next(
        i for i, line in enumerate(driver_lines) if "fr13_canonical_env.sh" in line
    )
    bsize_index = next(
        i for i, line in enumerate(driver_lines) if line.startswith("BSIZE=")
    )
    assert source_index < bsize_index
    assert f'export {ARM_VAR}=' not in CANONICAL_TEXT


# --------------------------------------------------------------------------
# 2. the default is shape-scoped and opt-outable
# --------------------------------------------------------------------------


def _promotion_block() -> str:
    start = LAUNCHER_TEXT.index("_FR13_FA2_QROW32_B1_PRODUCTION_ARM_NAMED == 0")
    end = LAUNCHER_TEXT.index(f'case "${ARM_VAR}" in', start)
    return LAUNCHER_TEXT[start:end]


def test_named_and_unset_are_distinguished_before_normalisation() -> None:
    """``${VAR:-}`` erases the difference; the promotion depends on it.

    Every arm in this repo that means "no B1 selector" says so by setting the
    variable to the empty string on the command line. If the promotion could
    not tell that apart from "never mentioned", it would silently retarget the
    byte gates and timing pairs it is required to leave alone.
    """
    assert "_FR13_FA2_QROW32_B1_PRODUCTION_ARM_NAMED=0" in LAUNCHER_TEXT
    assert f"[[ -v {ARM_VAR} ]]" in LAUNCHER_TEXT
    named_at = LAUNCHER_TEXT.index("_FR13_FA2_QROW32_B1_PRODUCTION_ARM_NAMED=0")
    normalised_at = LAUNCHER_TEXT.index(f"{ARM_VAR}=${{{ARM_VAR}:-}}")
    assert named_at < normalised_at, (
        "the named/unset capture must precede the ${VAR:-} normalisation"
    )


def test_the_promoted_default_only_applies_in_the_b1_serving_shape() -> None:
    block = _promotion_block()
    for guard in (
        '"${FR13_FIXED32_MODE:-}" == "hydra27_fixed32"',      # fixed32 only
        '"$MAX_NUM_SEQS" == "1"',                              # B1 only
        '"${SWE_CONCURRENCY:-}" == "1"',                       # single stream
        '"${FR13_FIXED32_B1_DIAGNOSTIC:-0}" == "0"',           # not diagnostic
        '"${FR13_FA2_QROW16_LIVE_PAGED_AB:-0}" == "0"',
        '"${FR13_FA2_QROW16_PRODUCTION:-0}" == "0"',
        '"${FR13_FA2_QROW32_LIVE_PAGED_AB:-0}" == "0"',
        '-z "$FR13_FA2_QROW32_B1_LIVE_AB_ARM"',
        '-z "$FR13_FA2_QROW32_B1_TIMING_ARM"',
        '-z "$FR13_FA2_QROW32_B4_TIMING_ARM"',
        '-z "$FR13_FA2_QROW32_B4_PRODUCTION_ARM"',
    ):
        assert guard in block, f"promotion is missing its {guard!r} scope guard"


def test_the_promotion_announces_itself_on_stderr() -> None:
    """A default that changes what is SERVED must never be silent."""
    block = _promotion_block()
    assert "B1 production arm unnamed; serving the promoted default" in block
    assert ">&2" in block


def test_the_local_env_guard_arms_for_an_unnamed_b1_launch() -> None:
    """The promoted arm is credentialed without being named.

    Every other clause of the .lumo.local.env override guard fires because the
    caller NAMED a credentialed selector. An unnamed fixed32 B1 launch now
    carries one too, so the guard has to arm on the shape.
    """
    assert (
        '   || ( "${FR13_FIXED32_MODE:-}" == "hydra27_fixed32" \\\n'
        '        && "${MAX_NUM_SEQS:-4}" == "1" \\\n'
        '        && "${SWE_CONCURRENCY:-}" == "1" ) ]] \\\n'
        "  && _FR13_M32_GUARD_ACTIVE=1" in LAUNCHER_TEXT
    )


def test_b1_timing_arms_that_must_not_move_opt_out_explicitly() -> None:
    """Single-variable B1 timing pairs keep their only-arm-delta.

    These two runners serve fixed32 B1 at concurrency 1 with
    FR13_FIXED32_B1_DIAGNOSTIC=0 and no sibling FA2 selector, which is exactly
    the promoted shape. Their arm delta is the CFWD / draft-head lever, not
    FA2, so they must pin the B1 selector off by name.
    """
    for relpath in (
        "scripts/fr13_run_b1_cfwd_logit_direct_timing.sh",
        "scripts/fr13_run_b1_draft_head_m32_timing.sh",
    ):
        text = (REPO / relpath).read_text(encoding="utf-8")
        assert f"{ARM_VAR}= \\" in text, f"{relpath} does not opt out of the promotion"
        assert "FR13_FA2_QROW32_B1_LIVE_AB_ARM= \\" in text


def test_diagnostic_b1_gates_are_out_of_scope_by_construction() -> None:
    """Byte gates and live gates run FR13_FIXED32_B1_DIAGNOSTIC=1.

    They are therefore excluded by the shape guard rather than by an opt-out,
    which is the safer of the two: a new gate runner inherits the exclusion.
    """
    for relpath in (
        "scripts/fr13_run_b1_kernel_live_gate.sh",
        "scripts/fr13_run_b1_cfwd_logit_direct_live_gate.sh",
        "scripts/fr13_run_b1_k64_taw_source_v7_gate.sh",
        "scripts/fr13_run_b1_sfwd_state_fusion_timing.sh",
    ):
        text = (REPO / relpath).read_text(encoding="utf-8")
        assert "FR13_FIXED32_B1_DIAGNOSTIC=1" in text


# --------------------------------------------------------------------------
# 3. the credential chain is unchanged
# --------------------------------------------------------------------------


def test_the_promoted_arm_still_has_to_prove_everything_it_proved_before() -> None:
    """Promotion selects an arm; it does not credential one.

    Every refusal on the GQA-pair B1 production path must survive the flip, or
    the flip would have turned a gated candidate into an ungated default.
    """
    for message in (
        # binary + source closure identity for the resolved pin arm
        "FR13 qrow32 B1 GQA-pair binary/source provenance drifted",
        # runtime shape and canonical exact4 identity
        "FR13 qrow32 B1 production requires the canonical exact4 FULL-graph identity",
        # sealed byte gate + the live result that gate binds by digest
        "FR13 qrow32 B1 GQA-pair production requires its sealed byte gate and bound live result",
        # the batch-1 K64/root1 shape that admits any B1 selector at all
        "FR13 qrow32 B1 selector requires Hydra27 K64/root1 B1 and exact binary/source provenance",
        # credentials stay launcher-issued
        "FR13 qrow32 production sidecar credentials are launcher-private",
    ):
        assert message in LAUNCHER_TEXT


def test_the_pin_arm_resolves_the_gqa_pair_binary_from_the_production_arm() -> None:
    """A production launch has no live arm, so the pin must key on production.

    Without this the promoted default would be checked against the incumbent
    no-split pins and refused -- the exact bug 0689455ff / 38ad36073 fixed for
    the explicit arm, and which the promotion now depends on.
    """
    assert (
        "_FR13_FA2_QROW32_B1_PIN_ARM=$FR13_FA2_QROW32_B1_LIVE_AB_ARM"
        in LAUNCHER_TEXT
    )
    assert (
        f"_FR13_FA2_QROW32_B1_PIN_ARM=${ARM_VAR}" in LAUNCHER_TEXT
    )


def test_the_gqa_pair_credential_is_issued_from_the_sealed_gate() -> None:
    assert "fr13_qrow32_b1_pass_sidecar.py issue-gqa-pair" in LAUNCHER_TEXT
    assert '--gate "$FR13_FA2_QROW32_B1_GQA_PAIR_GATE_HOST"' in LAUNCHER_TEXT
    assert (
        '--live-result "$FR13_FA2_QROW32_B1_GQA_PAIR_LIVE_RESULT_JSON"'
        in LAUNCHER_TEXT
    )
    # git-free verification inside the image (the sidecar fix this flip needs)
    assert "_fr13_b1_verify_command=verify-gqa-pair" in LAUNCHER_TEXT
    assert (
        '--expected-patch-source-sha256 "\\$FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256"'
        in LAUNCHER_TEXT
    )
