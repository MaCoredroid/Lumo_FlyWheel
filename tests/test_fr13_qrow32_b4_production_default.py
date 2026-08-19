"""The promoted B4 production FA2 arm: canonical default, launcher agreement.

PROMOTED 2026-08-14 on Mark's B4 production default flip ruling -- the B4
analogue of the B1 flip at 99a511319. Until this promotion the padded B4
GQA-pair FA2 unit was byte-gate-qualified, timing-sealed and QC-cleared but had
no production standing: ``FR13_FA2_QROW32_B4_PRODUCTION_ARM`` defaulted empty,
and the launcher additionally demanded a matching ``gqa_pair`` TIMING arm, so
the only configuration that could serve the kernel was the timing pair itself.

EVIDENCE THE FLIP RESTS ON
  byte gate   the b34 dual raw-byte gate: 0/0 output and LSE mismatches at
              width 4 natively and at width 3 padded, plus the poisoned-shadow
              arm (real rows byte-identical under a NaN/impossible-page shadow)
              and the shadow contract (zeros in O, +INF in LSE), on both
              qualified topologies, at two successive commits
              (output/fr13_fa2_gqa_pair_b34_dual_byte_gate_20260813T173106Z and
              ...T234051Z). Binary unmoved: .so af9e9f24..., 299813360 B,
              closure 9c3f9e75....
  timing      output/fr13_b4_hydra27_sealing_campaign_20260814T011514Z, four
              paired passes of real SWE-Verified pool16 traffic at 4 slots with
              balanced SC/CS arm order: verdict SEALED_HYDRA27_GAIN, batch-
              conditioned width-4 improvement mean 27.03 ms/step with a
              one-sided 95% lower bound of 10.82 ms/step, placebo width clean
              in every pass. The padded pair
              output/fr13_b4_width4_timing_padded_20260813T201426Z put width 3
              at 25.16 ms/step against a pre-registered 25.6 +/- 3.5.
  agent QC    results/fr13_b4_exact16_qc_20260814: 9/16 resolved = the 21-arm
              historical median, band 8-11, zero giveups, no always-resolves
              regression and no never-resolves recovery.

WHAT THESE TESTS PROTECT

1. ONE SOURCE OF TRUTH. scripts/fr13_canonical_env.sh owns the shipped value.
   The launcher restates it as a fallback, and the two may never disagree --
   the same failure mode the mamba-narrowing promotion left open for two days.
   So compare the files rather than restating a literal in a third place.

2. THE DEFAULT IS SCOPED TO A CREDENTIALED B4 SERVE, AND IS OPT-OUTABLE. This
   is where B4 differs from B1 and the difference is the whole design. The B1
   promotion could key on batch shape alone because B=1/concurrency 1 is a rare
   shape; its handful of siblings were excluded by FR13_FIXED32_B1_DIAGNOSTIC=1
   or by two explicit opt-outs. B=4/concurrency 4 fixed32 is the shape of the
   ENTIRE campaign -- floor gates, GDN/CUTLASS/SFWD timing pairs, live gates,
   nsys profiles -- and none of those can serve this kernel, because none holds
   its credential. Keying on batch alone would hand every one of them a
   selector the credential chain must then refuse at boot. The sealed b34 dual
   gate is what separates a production serve from a campaign arm, so that is
   what the promotion keys on.

3. THE CREDENTIAL CHAIN IS UNCHANGED, AND NOW REACHES THE UNPAIRED SERVE.
   Promotion changes which arm is selected, never what that arm must prove. The
   flip creates one launch shape that did not exist before -- a production serve
   with NO timing arm -- and the binary/source/shape preflight, which used to
   key on the timing arm alone, must follow it there.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
CANONICAL_ENV = REPO / "scripts" / "fr13_canonical_env.sh"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
CAMPAIGN_DRIVER = REPO / "scripts" / "fr13_b4_campaign_driver.sh"
SCRIPTS = REPO / "scripts"

CANONICAL_TEXT = CANONICAL_ENV.read_text(encoding="utf-8")
LAUNCHER_TEXT = LAUNCHER.read_text(encoding="utf-8")

PROMOTED_ARM = "gqa_pair"
DEFAULT_VAR = "FR13_FA2_QROW32_B4_PRODUCTION_ARM_DEFAULT"
ARM_VAR = "FR13_FA2_QROW32_B4_PRODUCTION_ARM"
TIMING_VAR = "FR13_FA2_QROW32_B4_TIMING_ARM"
GATE_VAR = "FR13_FA2_QROW32_B4_DUAL_GATE_JSON"


def _shell_defaults(text: str, variable: str) -> set[str]:
    """Every ``${VAR:-X}`` default X that `text` gives `variable`."""
    return set(re.findall(r"\$\{" + re.escape(variable) + r":-([^}]*)\}", text))


# --------------------------------------------------------------------------
# 1. one source of truth
# --------------------------------------------------------------------------


def test_canonical_env_exports_the_promoted_b4_production_arm() -> None:
    assert (
        f'export {DEFAULT_VAR}="${{{DEFAULT_VAR}:-{PROMOTED_ARM}}}"'
        in CANONICAL_TEXT
    )


def test_canonical_env_records_the_ruling_and_the_measured_verdict() -> None:
    """A promotion that does not carry its evidence is an unexplained default."""
    entry = next(
        line
        for line in CANONICAL_TEXT.splitlines()
        if line.startswith(f"export {DEFAULT_VAR}=")
    )
    assert "PROMOTED 2026-08-14" in entry
    assert "99a511319" in entry  # the B1 flip it mirrors
    # the sealed timing verdict, by run root and by number
    assert "fr13_b4_hydra27_sealing_campaign_20260814T011514Z" in entry
    assert "SEALED_HYDRA27_GAIN" in entry
    assert "27.03" in entry and "10.82" in entry
    # the padded width-3 result against its pre-registered prediction
    assert "fr13_b4_width4_timing_padded_20260813T201426Z" in entry
    assert "25.16" in entry and "25.6" in entry
    # the byte evidence, at both widths and at two commits
    assert "fr13_fa2_gqa_pair_b34_dual_byte_gate_20260813T173106Z" in entry
    assert "234051Z" in entry
    assert "poisoned" in entry
    # the agent quality-control pass
    assert "fr13_b4_exact16_qc_20260814" in entry
    assert "9/16" in entry
    # the scope the credential actually authorises
    assert "final_fixed32_b34_full_graph_only" in entry
    assert "production_widths 3,4" in entry
    # the binary the arm is credentialed against
    assert (
        "af9e9f24335db899468032f5b5a3eba100febe294932533cb9b87163ce2b3fdb"
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

    fr13_b4_campaign_driver.sh sources the registry before it reads BSIZE, so
    the registry cannot know whether the launch about to happen is a B4 serve
    at all -- and the B4 production selector is CREDENTIALED. Exporting it here
    would hand every arm the registry reaches, including the B=1 arms the same
    driver runs, a selector the launcher must then refuse at boot for want of a
    sealed dual gate.
    """
    driver_lines = CAMPAIGN_DRIVER.read_text(encoding="utf-8").splitlines()
    source_index = next(
        i for i, line in enumerate(driver_lines) if "fr13_canonical_env.sh" in line
    )
    bsize_index = next(
        i for i, line in enumerate(driver_lines) if line.startswith("BSIZE=")
    )
    assert source_index < bsize_index
    assert f"export {ARM_VAR}=" not in CANONICAL_TEXT


# --------------------------------------------------------------------------
# 2. the default is scoped to a credentialed B4 serve, and is opt-outable
# --------------------------------------------------------------------------


def _promotion_block() -> str:
    start = LAUNCHER_TEXT.index(f"_{ARM_VAR}_NAMED == 0")
    end = LAUNCHER_TEXT.index(f'case "${ARM_VAR}" in', start)
    return LAUNCHER_TEXT[start:end]


def test_named_and_unset_are_distinguished_before_normalisation() -> None:
    """``${VAR:-}`` erases the difference; the promotion depends on it.

    The stock side of the width-4 timing pair means "load the candidate binary,
    serve the stock dispatch" by setting this variable to the empty string. If
    the promotion could not tell that apart from "never mentioned", it would
    silently retarget the reference arm of its own sealing campaign.
    """
    assert f"_{ARM_VAR}_NAMED=0" in LAUNCHER_TEXT
    assert f"[[ -v {ARM_VAR} ]]" in LAUNCHER_TEXT
    named_at = LAUNCHER_TEXT.index(f"_{ARM_VAR}_NAMED=0")
    normalised_at = LAUNCHER_TEXT.index(f"{ARM_VAR}=${{{ARM_VAR}:-}}")
    assert named_at < normalised_at, (
        "the named/unset capture must precede the ${VAR:-} normalisation"
    )


def test_the_promoted_default_only_applies_in_the_b4_serving_shape() -> None:
    block = _promotion_block()
    for guard in (
        '"${FR13_FIXED32_MODE:-}" == "tail6_fixed32"',        # fixed32 only,
        '"${FR13_FIXED32_MODE:-}" == "hydra27_fixed32"',      # either topology
        '"$MAX_NUM_SEQS" == "4"',                              # B4 only
        '"${SWE_CONCURRENCY:-}" == "4"',                       # four streams
        '"${FR13_FIXED32_B1_DIAGNOSTIC:-0}" == "0"',           # not diagnostic
        f'-n "${GATE_VAR}"',                                   # credential shown
        '-n "$FR13_FA2_QROW32_B4_DUAL_GATE_SHA256"',
        '"${FR13_FA2_QROW16_LIVE_PAGED_AB:-0}" == "0"',
        '"${FR13_FA2_QROW16_PRODUCTION:-0}" == "0"',
        '"${FR13_FA2_QROW32_LIVE_PAGED_AB:-0}" == "0"',
        '-z "$FR13_FA2_QROW32_B1_LIVE_AB_ARM"',
        '-z "$FR13_FA2_QROW32_B1_TIMING_ARM"',
        '-z "$FR13_FA2_QROW32_B1_PRODUCTION_ARM"',
        f'-z "${TIMING_VAR}"',
    ):
        assert guard in block, f"promotion is missing its {guard!r} scope guard"


def test_the_promotion_is_gated_on_the_sealed_gate_not_on_batch_alone() -> None:
    """The clause that keeps every campaign arm out, stated as an invariant.

    Any launcher edit that drops the dual-gate clause from the promotion would
    silently retarget every fixed32 B=4 arm in this tree onto a credentialed
    selector none of them can satisfy. Assert the clause is there, and that the
    promotion cannot fire on shape alone.
    """
    block = _promotion_block()
    assert f'-n "${GATE_VAR}"' in block
    # ... and the guard is a conjunction: no `||` may separate the credential
    # clause from the batch clauses, or the two become alternatives.
    conjunctive = block[block.index('"$MAX_NUM_SEQS" == "4"') :]
    conjunctive = conjunctive[: conjunctive.index(f'-n "${GATE_VAR}"')]
    assert "||" not in conjunctive


# The promotion block, evaluated. Text assertions pin that each clause is
# PRESENT; this pins what the conjunction of them DOES. `set -u` is on, exactly
# as in the launcher, so a future clause that reads a variable the launcher has
# not yet normalised fails here instead of at boot.
_HARNESS = """
set -euo pipefail
FR13_FA2_QROW32_B4_PRODUCTION_ARM_DEFAULT=gqa_pair
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
FR13_FIXED32_MODE=${FR13_FIXED32_MODE:-}
FR13_FA2_QROW32_B4_DUAL_GATE_JSON=${FR13_FA2_QROW32_B4_DUAL_GATE_JSON:-}
FR13_FA2_QROW32_B4_DUAL_GATE_SHA256=${FR13_FA2_QROW32_B4_DUAL_GATE_SHA256:-}
FR13_FA2_QROW32_B1_LIVE_AB_ARM=${FR13_FA2_QROW32_B1_LIVE_AB_ARM:-}
FR13_FA2_QROW32_B1_TIMING_ARM=${FR13_FA2_QROW32_B1_TIMING_ARM:-}
FR13_FA2_QROW32_B1_PRODUCTION_ARM=${FR13_FA2_QROW32_B1_PRODUCTION_ARM:-}
FR13_FA2_QROW32_B4_TIMING_ARM=${FR13_FA2_QROW32_B4_TIMING_ARM:-}
_FR13_FA2_QROW32_B4_PRODUCTION_ARM_NAMED=0
[[ -v FR13_FA2_QROW32_B4_PRODUCTION_ARM ]] \
  && _FR13_FA2_QROW32_B4_PRODUCTION_ARM_NAMED=1
FR13_FA2_QROW32_B4_PRODUCTION_ARM=${FR13_FA2_QROW32_B4_PRODUCTION_ARM:-}
@BLOCK@
printf 'RESOLVED_ARM=%s\\n' "$FR13_FA2_QROW32_B4_PRODUCTION_ARM"
"""


def _executable_promotion_block() -> str:
    """The promotion `if` statement alone, from `if ((` to its closing `fi`."""
    start = LAUNCHER_TEXT.index(f"if (( _{ARM_VAR}_NAMED == 0 ))")
    end = LAUNCHER_TEXT.index(f'\ncase "${ARM_VAR}" in', start)
    return LAUNCHER_TEXT[start:end]


_CREDENTIALED = {
    "FR13_FIXED32_MODE": "hydra27_fixed32",
    "MAX_NUM_SEQS": "4",
    "SWE_CONCURRENCY": "4",
    "FR13_FA2_QROW32_B4_DUAL_GATE_JSON": "/logs/dual_gate.json",
    "FR13_FA2_QROW32_B4_DUAL_GATE_SHA256": "f" * 64,
}


def _resolve(**overrides: str) -> str:
    env = dict(_CREDENTIALED)
    env.update(overrides)
    env = {k: v for k, v in env.items() if v is not None}
    proc = subprocess.run(
        ["bash", "-c", _HARNESS.replace("@BLOCK@", _executable_promotion_block())],
        cwd=REPO,
        env={"PATH": "/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    line = next(
        l for l in proc.stdout.splitlines() if l.startswith("RESOLVED_ARM=")
    )
    return line.split("=", 1)[1]


def test_a_credentialed_unnamed_b4_serve_resolves_to_the_promoted_arm() -> None:
    assert _resolve() == PROMOTED_ARM
    # both sealed topologies, since the dual gate qualified both
    assert _resolve(FR13_FIXED32_MODE="tail6_fixed32") == PROMOTED_ARM


@pytest.mark.parametrize(
    "overrides,why",
    [
        # the campaign: fixed32 B=4/conc 4, no credential, no selector
        ({"FR13_FA2_QROW32_B4_DUAL_GATE_JSON": "",
          "FR13_FA2_QROW32_B4_DUAL_GATE_SHA256": ""}, "no credential presented"),
        ({"FR13_FA2_QROW32_B4_DUAL_GATE_SHA256": ""}, "gate path without its digest"),
        # naming it, including naming it empty, is always obeyed
        ({"FR13_FA2_QROW32_B4_PRODUCTION_ARM": ""}, "named empty = opt-out"),
        # the timing pair declares its own arms
        ({"FR13_FA2_QROW32_B4_TIMING_ARM": "stock_dispatch"}, "stock timing arm"),
        ({"FR13_FA2_QROW32_B4_TIMING_ARM": "gqa_pair"}, "candidate timing arm"),
        # the byte gate runs as a qrow32 live A/B
        ({"FR13_FA2_QROW32_LIVE_PAGED_AB": "1"}, "qrow32 live A/B"),
        ({"FR13_FA2_QROW16_LIVE_PAGED_AB": "1"}, "qrow16 live A/B"),
        ({"FR13_FA2_QROW16_PRODUCTION": "1"}, "qrow16 production"),
        # B1 selectors own the call site when they are engaged
        ({"FR13_FA2_QROW32_B1_LIVE_AB_ARM": "gqa_pair"}, "B1 live A/B arm"),
        ({"FR13_FA2_QROW32_B1_TIMING_ARM": "gqa_pair"}, "B1 timing arm"),
        ({"FR13_FA2_QROW32_B1_PRODUCTION_ARM": "gqa_pair"}, "B1 production arm"),
        # wrong shape
        ({"MAX_NUM_SEQS": "1", "SWE_CONCURRENCY": "1"}, "the B1 shape"),
        ({"SWE_CONCURRENCY": "1"}, "batch 4 served at concurrency 1"),
        ({"FR13_FIXED32_MODE": ""}, "not fixed32 at all"),
        ({"FR13_FIXED32_MODE": "hydra23"}, "a non-fixed32 topology"),
        ({"FR13_FIXED32_B1_DIAGNOSTIC": "1"}, "the diagnostic shape"),
    ],
)
def test_the_promotion_does_not_fire_outside_its_scope(
    overrides: dict[str, str], why: str
) -> None:
    assert _resolve(**overrides) == "", f"promotion wrongly fired for {why}"


def test_the_promotion_announces_itself_on_stderr() -> None:
    """A default that changes what is SERVED must never be silent."""
    block = _promotion_block()
    assert "B4 production arm unnamed; serving the promoted default" in block
    assert ">&2" in block


def test_the_local_env_guard_arms_for_an_unnamed_credentialed_b4_launch() -> None:
    """The promoted arm is credentialed without being named.

    Every other clause of the .lumo.local.env override guard fires because the
    caller NAMED a credentialed selector. An unnamed fixed32 B4 launch that
    presents the sealed dual gate now carries one too, so the guard has to arm
    on that shape.
    """
    assert (
        '   || ( -n "${FR13_FIXED32_MODE:-}" \\\n'
        '        && "${MAX_NUM_SEQS:-4}" == "4" \\\n'
        '        && "${SWE_CONCURRENCY:-}" == "4" \\\n'
        f'        && -n "${{{GATE_VAR}:-}}" ) \\\n' in LAUNCHER_TEXT
    )
    # and the gate path is itself one of the guarded names, so a local env that
    # INTRODUCES it arms the guard and is then caught by the same comparison.
    guard_block = LAUNCHER_TEXT[
        LAUNCHER_TEXT.index("_FR13_M32_GUARD_NAMES=(") : LAUNCHER_TEXT.index(
            "\n)\ndeclare -A"
        )
    ]
    assert f"\n  {GATE_VAR}\n" in guard_block


def test_every_dual_gate_presenting_runner_names_its_arm() -> None:
    """Timing, sealing and QC runners are excluded BY NAME, not by accident.

    The three runners that hand the launcher a sealed b34 dual gate are the
    GQA-pair timing pair, the width-4 timing pair the sealing campaign drives,
    and the exact16 QC gate. Each of them declares the served arm explicitly --
    the stock side by naming it EMPTY -- so the promotion never retargets a run
    whose arm is the thing under test. A new runner that presents the gate
    without declaring an arm would be silently promoted; this test refuses it.
    """
    presenters = sorted(
        path
        for path in SCRIPTS.glob("*.sh")
        if path.name != LAUNCHER.name
        and f"{GATE_VAR}=" in path.read_text(encoding="utf-8")
    )
    assert presenters, "expected the timing/QC runners to present a dual gate"
    for path in presenters:
        text = path.read_text(encoding="utf-8")
        assert f"{ARM_VAR}=" in text, (
            f"{path.name} presents a sealed dual gate without naming {ARM_VAR}; "
            "it would be silently promoted onto the sealed kernel"
        )


def test_the_byte_gate_is_out_of_scope_by_construction() -> None:
    """The b34 dual gate runs as a qrow32 live A/B, a sibling private selector.

    It is therefore excluded by the scope guard rather than by an opt-out,
    which is the safer of the two: a new gate arm inherits the exclusion.
    """
    inner = (REPO / "scripts/fr13_run_b4_fa2_qrow32_live_gate.sh").read_text(
        encoding="utf-8"
    )
    assert "FR13_FA2_QROW32_LIVE_PAGED_AB=1" in inner
    assert '"${FR13_FA2_QROW32_LIVE_PAGED_AB:-0}" == "0"' in _promotion_block()


def test_campaign_arms_without_a_credential_are_left_alone() -> None:
    """The B=4 campaign is not retargeted -- the point of the gate clause.

    These runners all serve fixed32 at batch 4 / concurrency 4 with no FA2
    selector of any kind. Under a batch-only promotion each would be handed the
    credentialed arm and refused at boot. Assert they neither name the arm nor
    present a gate, so the promotion provably cannot reach them.
    """
    for relpath in (
        "scripts/fr13_b4_formal_floor_gate.sh",
        "scripts/fr13_b4_pool16_refill_gate.sh",
        "scripts/fr13_run_b4_mamba_narrow_within_run_pair.sh",
        "scripts/fr13_run_b4_cutlass_persistent_m128_timing.sh",
        "scripts/fr13_run_b4_draft_head_m32_timing.sh",
        "scripts/fr13_run_b4_gdn_bv8_timing.sh",
        "scripts/fr13_run_b4_sfwd_state_fusion_live_gate.sh",
    ):
        text = (REPO / relpath).read_text(encoding="utf-8")
        assert f"{GATE_VAR}=" not in text, f"{relpath} unexpectedly holds a credential"
        assert f"{ARM_VAR}=" not in text


# --------------------------------------------------------------------------
# 3. the credential chain is unchanged, and now reaches the unpaired serve
# --------------------------------------------------------------------------


def test_the_promoted_arm_still_has_to_prove_everything_it_proved_before() -> None:
    """Promotion selects an arm; it does not credential one.

    Every refusal on the GQA-pair B4 production path must survive the flip, or
    the flip would have turned a gated candidate into an ungated default.
    """
    for message in (
        # binary/source identity, runtime shape and the byte-pinned task set
        # "with K64/root1 identity" was dropped from this message when the
        # lever was converted to _fr13_assert_draft_vocab_profile: the
        # identity is still bound, one line above, and now admits full_vocab
        # too. The test kept asserting the pre-conversion wording and has been
        # failing ever since -- the same generalization that produced site 12.
        "FR13 qrow32 B4 GQA-pair timing or production requires a byte-pinned "
        "canonical B4 evidence set (exact4 or pool16) "
        "and pinned binary/source provenance",
        # the identity itself, where it now lives
        '"$FR13_FA2_QROW32_B4_QUALIFICATION_PROFILE" "FR13 qrow32 B4 GQA-pair"',
        # the sealed dual raw-byte gate, bound by digest
        "FR13 qrow32 B4 production requires its bound dual raw-byte gate PASS",
        # a named timing arm still cannot disagree with the served kernel
        "FR13 qrow32 B4 gqa_pair timing arm must serve the candidate",
        # credentials stay launcher-issued
        "FR13 qrow32 B4 internal attestation is launcher-private",
        "FR13 qrow32 production sidecar credentials are launcher-private",
        # one private FA2 selector per decode call site
        "FR13 qrow32 B4 and existing FA2 private selectors are mutually exclusive",
    ):
        assert message in LAUNCHER_TEXT


def test_the_binary_preflight_follows_the_arm_onto_the_unpaired_serve() -> None:
    """The flip's one genuinely new launch shape must not lose its pins.

    Before the flip the only way to reach the production arm was from inside
    the timing pair, so keying the candidate-mode preflight on the timing arm
    covered every launch that could serve the kernel. A promoted serve carries
    NO timing arm, and would otherwise have skipped the .so digest/size pins,
    the on-disk re-hash, SOURCE_COMMIT == HEAD, the patcher digest, the
    batch-4 runtime shape and the evidence-set predicate -- every one of which
    is a precondition of the credential the launcher then ISSUES over
    caller-supplied values.
    """
    assert (
        f'if [[ -n "${TIMING_VAR}" \\\n      || -n "${ARM_VAR}" ]]; then'
        in LAUNCHER_TEXT
    )
    block = LAUNCHER_TEXT[
        LAUNCHER_TEXT.index("_FR13_FA2_QROW32_B4_CANDIDATE_MODE=0") :
        LAUNCHER_TEXT.index("_FR13_FA2_QROW32_B1_CANDIDATE_MODE=0")
    ]
    assert '"$FR13_FA2_QROW32_SOURCE_COMMIT" == "$(git rev-parse HEAD)"' in block
    assert '"$(sha256sum "$FORKED_FA2_SO" | cut -d\' \' -f1)"' in block
    assert '"$MAX_NUM_SEQS" == "4"' in block
    assert '-n "$_FR13_FA2_QROW32_B4_TASK_SET"' in block


def test_one_selector_per_call_site_survives_the_relaxed_pairing() -> None:
    """The relaxation must not open a two-selector launch.

    One patched tree_attn decode call hosts exactly one private FA2 selector.
    Until the flip a B4 production arm could not exist without a gqa_pair
    TIMING arm, so keying the mutual-exclusion refusal on the timing arm
    covered it transitively. With the pairing relaxed, a lone production arm
    beside qrow16 production or a qrow32/B1 selector would otherwise race for
    the call site with first-one-wins deciding the served kernel.
    """
    assert (
        f'if [[ -n "${TIMING_VAR}" \\\n      || -n "${ARM_VAR}" ]] && {{'
        in LAUNCHER_TEXT
    )
    # the same relaxation in the in-container contract, which independently
    # re-derives the expected FA2 identity from the same env
    contract = (REPO / "scripts/fr13_fixed32_contract.py").read_text(
        encoding="utf-8"
    )
    assert (
        "if (qrow32_b4_timing or qrow32_b4_production) and (" in contract
    )
    assert "if qrow32_b4_timing or qrow32_b4_production:" in contract
    assert (
        "if qrow32_b4_timing and (\n"
        '        (qrow32_b4_timing == "gqa_pair") != '
        '(qrow32_b4_production == "gqa_pair")\n'
        "    ):" in contract
    )


def test_the_timing_arm_requirement_is_relaxed_only_for_the_empty_arm() -> None:
    """Production serving carries no timing arm; a WRONG one is still fatal.

    The clause used to read ``!= "gqa_pair"``, which made the production arm
    reachable only from inside the timing pair. It now fires for a NON-EMPTY
    disagreeing timing arm only, so a plain production serve is legal and
    ``stock_dispatch`` paired with a gqa_pair serve is still refused.
    """
    clause_at = LAUNCHER_TEXT.index(
        "FR13 qrow32 B4 production requires the gqa_pair timing arm"
    )
    clause = LAUNCHER_TEXT[clause_at - 400 : clause_at]
    assert f'"${ARM_VAR}" == "gqa_pair"' in clause
    assert f'-n "${TIMING_VAR}"' in clause
    assert f'"${TIMING_VAR}" != "gqa_pair"' in clause


def test_a_promoted_serve_without_the_sealed_gate_still_refuses() -> None:
    """The flip promotes an arm; it hands out no credential.

    The dual-gate preflight is keyed on the ARM, so it fires for the promoted
    arm exactly as it fired for the named one: a regular non-symlink file whose
    bytes match the declared digest, or the launch refuses.
    """
    block = LAUNCHER_TEXT[
        LAUNCHER_TEXT.index(f'if [[ -n "${ARM_VAR}" ]]; then\n  [[ -f "${GATE_VAR}"') :
    ]
    block = block[: block.index("exit 2")]
    assert f'-f "${GATE_VAR}"' in block
    assert f'! -L "${GATE_VAR}"' in block
    assert '"$FR13_FA2_QROW32_B4_DUAL_GATE_SHA256" =~ ^[0-9a-f]{64}$' in block
    assert f'"$(sha256sum "${GATE_VAR}" | cut -d\' \' -f1)"' in block


def test_the_credential_is_issued_and_verified_over_the_promoted_arm() -> None:
    """Issuance and in-container verification both read the resolved arm."""
    assert "fr13_qrow32_b4_pass_sidecar.py issue" in LAUNCHER_TEXT
    assert f'--arm "${ARM_VAR}"' in LAUNCHER_TEXT
    assert "fr13_qrow32_b4_pass_sidecar.py verify" in LAUNCHER_TEXT
    assert f'--arm "\\${ARM_VAR}"' in LAUNCHER_TEXT
    # the sidecar binds the gate to the plumbing commit, which is why the flip
    # commit needs its own re-gate before anything may serve.
    sidecar = (REPO / "scripts/fr13_qrow32_b4_pass_sidecar.py").read_text(
        encoding="utf-8"
    )
    assert "dual gate was not produced at the production plumbing commit" in sidecar
