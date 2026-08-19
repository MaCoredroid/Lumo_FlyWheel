"""Every place that decides "which fixed32 profile is this?" must know all of them.

The same question is asked in three kinds of place, and each has now refused a
serve on its own: the consumers (rounds <=12), the vehicle dispatch (round 13),
and the launcher's in-container preflight (round 14).

Round 14 taught the sharper half: fixing a mode TABLE is not enough. Three lines
below the table, the same block compared the dispatched tree against
`topology.FIXED32_CHOICES` unconditionally -- and hydra31's tree genuinely
differs (ancestry 90873d81 -> 5b33c46a), so teaching only the table would have
moved the refusal three lines down. A third profile-varying comparison, the walk
cap, sat nine lines further on.

`scripts/fr14_mode_table_parity.py` answers both halves from
`fr13_fixed32_topology.PROFILES`, so it fails on the NEXT profile rather than on
the next boot.
"""

from __future__ import annotations

import subprocess
import sys
import re
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import fr13_fixed32_topology as topo  # noqa: E402
import fr14_mode_table_parity as parity  # noqa: E402

LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
VEHICLE = REPO / "scripts" / "fr13_bigdenom_swe_serve_variant.sh"


def _tmp(text):
    p = Path(tempfile.mkdtemp()) / "x.sh"
    p.write_text(text)
    return p


# ---------------------------------------------------------------------------
# The inventory at HEAD.
# ---------------------------------------------------------------------------

def test_every_site_admits_every_profile():
    stale = parity.sweep()
    assert not stale, "\n".join(f"{r['kind']}: {r['detail']}" for r in stale)


def test_the_scan_covers_all_four_shell_sites():
    for rel in parity.SHELL_SITES:
        assert (REPO / rel).exists(), f"{rel} moved"
    # the preflight block is mirrored in three launchers -- all must be scanned
    mirrored = [
        rel for rel in parity.SHELL_SITES
        if "unsupported fixed32 mode" in (REPO / rel).read_text()
    ]
    assert len(mirrored) == 3, mirrored


def test_profile_varying_and_invariant_sets_do_not_overlap():
    assert not (parity.PROFILE_VARYING & parity.PROFILE_INVARIANT)
    # and the names really do differ / agree between profiles
    a = topo.profile(topo.PROFILE_HYDRA27)
    b = topo.profile(topo.PROFILE_HYDRA31)
    assert a["walk_cap"] != b["walk_cap"]
    assert tuple(a["choices"]) != tuple(b["choices"])
    assert topo.PHYSICAL_DRAFTS == 31 and topo.PHYSICAL_ROWS == 32


# ---------------------------------------------------------------------------
# Mutation coverage: the detector must fire on each real round-14-class defect.
# ---------------------------------------------------------------------------

def test_detector_catches_a_mode_table_missing_a_profile():
    broken = LAUNCHER.read_text().replace(
        '    "hydra31_fixed32": (\n'
        "        topology.HYDRA31_VALID_MASK,\n"
        "        topology.HYDRA31_ACTIVE_DRAFTS,\n"
        "    ),\n",
        "",
    )
    assert broken != LAUNCHER.read_text(), "anchor moved"
    hits = parity.scan_mode_tables(_tmp(broken))
    assert hits and "hydra31_fixed32" in hits[0]


def test_detector_catches_the_unconditional_tree_comparison():
    """THE round-14 second refusal -- found by reading, not by booting."""
    broken = LAUNCHER.read_text().replace(
        'if tree != tuple(_profile["choices"]):',
        "if tree != topology.FIXED32_CHOICES:",
    )
    assert broken != LAUNCHER.read_text(), "anchor moved"
    hits = parity.scan_unconditional_comparisons(_tmp(broken))
    assert hits and "FIXED32_CHOICES" in hits[0]


def test_detector_catches_the_unconditional_walk_cap_comparison():
    """The third profile-varying compare in the same block."""
    broken = LAUNCHER.read_text().replace(
        'or walk_cap != int(_profile["walk_cap"])',
        "or walk_cap != topology.WALK_CAP",
    )
    assert broken != LAUNCHER.read_text(), "anchor moved"
    hits = parity.scan_unconditional_comparisons(_tmp(broken))
    assert hits and "WALK_CAP" in hits[0]


@pytest.mark.parametrize(
    "path,old,new",
    [
        ("launcher", '""|tail6_fixed32|hydra27_fixed32|hydra31_fixed32)',
         '""|tail6_fixed32|hydra27_fixed32)'),
        ("vehicle", '|| "$KIND" == "hydra31_fixed32"', ""),
    ],
)
def test_detector_catches_a_whitelist_dropping_a_profile(path, old, new):
    src = (LAUNCHER if path == "launcher" else VEHICLE).read_text()
    broken = src.replace(old, new, 1)
    assert broken != src, "anchor moved"
    hits = parity.scan_shell_whitelists(_tmp(broken))
    assert hits and "hydra31_fixed32" in hits[0]


def test_detector_leaves_single_mode_lever_preconditions_alone():
    """hydra27-qualified levers legitimately refuse hydra31; not a whitelist."""
    probe = _tmp(
        'if [[ "$FIXED32_MODE" != "hydra27_fixed32" ]]; then exit 2; fi\n'
    )
    assert parity.scan_shell_whitelists(probe) == []


# ---------------------------------------------------------------------------
# Executed: the real preflight accepts every profile and refuses the traps.
# ---------------------------------------------------------------------------

def _preflight_source():
    text = LAUNCHER.read_text()
    start = text.index("import ast\nimport json\nimport os\nimport sys")
    return text[start: text.index("\nPY\n", start)]


def _env_for(mode, tree_mode=None, walk_cap=None):
    import json as _json

    prof = topo.profile(
        topo.PROFILE_HYDRA31 if mode == "hydra31_fixed32"
        else topo.PROFILE_HYDRA27
    )
    tree_prof = topo.profile(
        topo.PROFILE_HYDRA31 if (tree_mode or mode) == "hydra31_fixed32"
        else topo.PROFILE_HYDRA27
    )
    mask = {
        "tail6_fixed32": topo.TAIL6_VALID_MASK,
        "hydra27_fixed32": topo.HYDRA27_VALID_MASK,
        "hydra31_fixed32": topo.HYDRA31_VALID_MASK,
    }[mode]
    active = {
        "tail6_fixed32": topo.TAIL6_ACTIVE_DRAFTS,
        "hydra27_fixed32": topo.HYDRA27_ACTIVE_DRAFTS,
        "hydra31_fixed32": topo.HYDRA31_ACTIVE_DRAFTS,
    }[mode]
    tree = repr(list(tree_prof["choices"]))
    return {
        "FR13_FIXED32_MODE": mode,
        "FR13_FIXED32_VALID_MASK": f"{mask:#x}",
        "FR13_FIXED32_ACTIVE_NODES": str(active),
        "FR13_FIXED32_PHYSICAL_DRAFTS": str(topo.PHYSICAL_DRAFTS),
        "FR13_FIXED32_TAW_WALK_CAP": str(
            walk_cap if walk_cap is not None else prof["walk_cap"]
        ),
        "NUM_SPECULATIVE_TOKENS": str(topo.PHYSICAL_DRAFTS),
        "TREE": tree,
        "SPEC_CONFIG": _json.dumps({
            "method": "qwen3_5_mtp",
            "num_speculative_tokens": topo.PHYSICAL_DRAFTS,
            "speculative_token_tree": tree,
        }),
    }


def _run_preflight(env):
    import os

    py = REPO / ".venv" / "bin" / "python"
    if not py.exists():
        pytest.skip("preflight needs .venv/bin/python")
    src = Path(tempfile.mkdtemp()) / "preflight.py"
    src.write_text(_preflight_source())
    full = dict(os.environ)
    full.update(env)
    return subprocess.run(
        [str(py), str(src)], capture_output=True, text=True, cwd=REPO, env=full
    )


@pytest.mark.parametrize(
    "mode", ["tail6_fixed32", "hydra27_fixed32", "hydra31_fixed32"]
)
def test_preflight_accepts_every_mode(mode):
    r = _run_preflight(_env_for(mode))
    assert r.returncode == 0, r.stdout + r.stderr


def test_preflight_refuses_hydra31_carrying_hydra27s_tree():
    """The exact trap: right mask, wrong tree. It would otherwise boot."""
    r = _run_preflight(_env_for("hydra31_fixed32", tree_mode="hydra27_fixed32"))
    assert r.returncode != 0
    assert "TREE differs" in r.stdout + r.stderr


def test_preflight_refuses_hydra31_with_the_hydra27_walk_cap():
    r = _run_preflight(_env_for("hydra31_fixed32", walk_cap=12))
    assert r.returncode != 0
    assert "shape mismatch" in r.stdout + r.stderr


def test_preflight_refuses_an_unknown_mode():
    env = _env_for("hydra27_fixed32")
    env["FR13_FIXED32_MODE"] = "hydra99_fixed32"
    r = _run_preflight(env)
    assert r.returncode != 0
    assert "unsupported fixed32 mode" in r.stdout + r.stderr


def test_vehicle_supplies_the_walk_cap_the_preflight_demands():
    """The arm has to carry it: the runner's env says 12 for every arm."""
    text = VEHICLE.read_text()
    for mode in sorted(topo.PROFILES):
        start = text.index(f"\n  {mode})\n")
        block = text[start: text.index("\n    ;;", start)]
        assert "FR13_FIXED32_TAW_WALK_CAP=$" in block, (
            f"{mode} kind block does not export a derived walk cap"
        )


# ---------------------------------------------------------------------------
# LAUNCHER-FAMILY PARITY.
#
# "both launcher families" was wrong by one for six rounds. fr14_leg3 is a live
# serving path (arm B's profile-chain legs) and carried none of the FR14 work --
# including the PROMOTED fused-topk default, so it would have served the unfused
# kernel silently while the other two served the promoted one.
#
# The roster now lives in exactly one place and every detector imports it.
# ---------------------------------------------------------------------------

def test_the_roster_is_three_and_is_the_only_enumeration():
    assert len(parity.LAUNCHER_FAMILIES) == 3
    for rel in parity.LAUNCHER_FAMILIES:
        assert (REPO / rel).exists(), rel
    # nothing else may re-enumerate the families by hand
    for rel in (
        "scripts/fr14_paired_contract_sweep.py",
        "tests/test_fr14_suffix_pass_gate.py",
        "tests/test_fr14_splitk_arm_launcher_wiring.py",
    ):
        text = (REPO / rel).read_text()
        assert "LAUNCHER_FAMILIES" in text, (
            f"{rel} does not import the roster"
        )
        assert '"scripts/fr14_armb_leg3_launch_nomiddleware.sh",\n)' not in text


def test_family_parity_is_clean_at_head():
    assert parity.scan_family_parity() == []


@pytest.mark.parametrize("marker", parity.FAMILY_PARITY_MARKERS)
def test_every_parity_marker_is_present_in_every_family(marker):
    for rel in parity.LAUNCHER_FAMILIES:
        assert marker in (REPO / rel).read_text(), (
            f"{marker} missing from {Path(rel).name}"
        )


@pytest.mark.parametrize(
    "marker",
    [
        "FR14_FUSED_DRAFT_TOPK", "_fr14_gate_incompat", "FR14_GATE_SPLIT_GRAPH",
        # F1/F2 (pass 106). The fix landed in the canonical launcher one edit
        # ahead of the twins -- literally the two-of-three window this roster
        # exists to close -- so each half is proved to fire from a twin.
        "cannot mint the selector provenance",
        "_fr13_b1_commit_bound",
        "STANDS DOWN",
        # site 12 (pass 113)
        '"$FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE" "FR13 qrow32 B1 selector" || exit 2',
        '"$FR13_FA2_QROW32_B4_QUALIFICATION_PROFILE" "FR13 qrow32 B4 GQA-pair" || exit 2',
        "FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE=${FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE:-k64_root}",
    ],
)
def test_detector_fires_when_one_family_omits_a_marker(monkeypatch, marker):
    """Exactly the defect that happened: two families promoted, one not."""
    third = parity.LAUNCHER_FAMILIES[2]
    stripped = (REPO / third).read_text().replace(marker, "FR14_REMOVED")
    tmp = Path(tempfile.mkdtemp())
    (tmp / "scripts").mkdir()
    for rel in parity.LAUNCHER_FAMILIES:
        target = tmp / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            stripped if rel == third else (REPO / rel).read_text()
        )
    monkeypatch.setattr(parity, "REPO", tmp)
    bad = parity.scan_family_parity()
    assert bad and any(marker in b for b in bad), bad
    assert any("fr14_leg3" in b for b in bad), bad


def test_the_promoted_default_is_ON_in_every_family():
    """A promoted default that is ON in two families and absent in a third is
    not promoted -- it is a silent A/B."""
    for rel in parity.LAUNCHER_FAMILIES:
        text = (REPO / rel).read_text()
        assert '-e FR14_FUSED_DRAFT_TOPK="${FR14_FUSED_DRAFT_TOPK:-1}"' in text
        assert 'case "${FR14_FUSED_DRAFT_TOPK:-1}" in' in text
        assert "PROMOTED-ON but its pinned .so is missing" in text


def test_the_refused_gate_is_guarded_in_every_family():
    """The gate is refused-final; the guards must still exist so a stale env
    cannot arm it in a family that never learned about it."""
    for rel in parity.LAUNCHER_FAMILIES:
        text = (REPO / rel).read_text()
        assert '-e FR14_SUFFIX_PASS_GATE="${FR14_SUFFIX_PASS_GATE:-0}"' in text
        assert 'echo "FR14_SUFFIX_PASS_GATE must be 0 or 1" >&2; exit 2' in text
        assert (
            "FR14_SUFFIX_PASS_GATE=1 requires FR13_TAIL_MODE=1 and "
            "FR13_DRAFT_SOURCE=merged" in text
        )
        assert 'grep -q "FR14_GATE_SPLIT_GRAPH"' in text


def test_f1_f2_are_present_in_every_family():
    """The promotion's boot-survival fix, family by family.

    Counting is not enough here: the mint has to be inside the arming branch
    (a mint that runs when the default did NOT arm would forge provenance for
    a caller's selector), and the gate's commit clause has to be SCOPED, not
    deleted -- the byte-exact route still binds HEAD.
    """
    for rel in parity.LAUNCHER_FAMILIES:
        text = (REPO / rel).read_text()
        # F1: the mint, inside the branch that arms
        arm = text.index("FR13_FA2_QROW32_B1_TIER_B_ARM=$_FR13_SPLITK_DEFAULT_ARM")
        close = text.index("\n  fi\n", arm)
        branch = text[arm:close]
        assert "FR13_FA2_QROW32_B1_SOURCE_COMMIT:-$(git rev-parse HEAD" in branch, (
            f"{Path(rel).name}: the mint is outside the arming branch"
        )
        assert "FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256:-$(sha256sum" in branch
        assert "cannot mint the selector provenance" in branch
        # F1: the gate is SCOPED, not disarmed
        assert "requires a credential earned at this HEAD" in text
        assert "tier-b selector requires a well-formed source commit" in text
        assert '"$FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256" == "$(sha256sum' in text, (
            f"{Path(rel).name}: the patcher digest stopped being bound"
        )
        # F2: the arbitration, and the opt-out it must not cost
        assert "STANDS DOWN" in text
        assert 'if [[ -n "$FR13_FA2_QROW32_B1_TIER_B_ARM" ]]; then' in text
        assert '""|nosplit|gqa_pair) ;;' in text


# ===========================================================================
# SITE 12 (pass 113): the vocab-profile conversion, and the STRUCTURAL check.
#
# Production converted five levers from a hard-coded K64/root1 predicate to
# _fr13_assert_draft_vocab_profile, which admits full_vocab as well. The
# no-middleware forks took ONE of the five. The forks were current on that
# night's F1/F2 work and stale on the earlier generalization at the same
# time -- selective staleness, which an enumerated roster cannot be relied on
# to find because the roster is written after each miss.
# ===========================================================================

VOCAB_LEVERS = (
    ("FR13 qrow32 B4 GQA-pair timing or production requires",
     "FR13_FA2_QROW32_B4_QUALIFICATION_PROFILE"),
    ("FR13 qrow32 B1 selector requires Hydra27",
     "FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE"),
    ("FR13 GDN GQA-group3 production requires exact credentialed",
     "FR13_FIXED32_GDN_GQA_GROUP3_QUALIFICATION_PROFILE"),
    ("FR13 GDN single-launch production requires exact credentialed",
     "FR13_FIXED32_GDN_SINGLE_LAUNCH_QUALIFICATION_PROFILE"),
)

_INLINE_VOCAB = re.compile(
    r'^\s*&&\s+"\$\{?FR13_DRAFT_VOCAB_(?:K|ROOT|BLOCKS)[^"]*"?\}?"\s*==\s*"[^"]*"\s*\\$'
)


def _lever_shape(text, message):
    """(calls_helper, hardcoded_clause_count) for one lever's refusal region."""
    lines = text.split("\n")
    err = next(i for i, line in enumerate(lines) if message in line)
    start = next(
        i for i in range(err, err - 80, -1) if re.match(r"^\s*\[\[ ", lines[i])
    )
    above = "\n".join(lines[max(start - 4, 0):start])
    return (
        "_fr13_assert_draft_vocab_profile" in above,
        sum(1 for i in range(start, err) if _INLINE_VOCAB.match(lines[i])),
    )


@pytest.mark.parametrize("message,var", VOCAB_LEVERS)
def test_every_family_delegates_the_vocab_identity(message, var):
    """Site 12 proper: full_vocab must be REACHABLE in every family.

    A hard-coded `K == 65536` inside a lever predicate is a lever that cannot
    be armed under K0 -- and K0 full-vocab is the only shape split-K has ever
    served in, so this made the promoted default unbootable in the forks.
    """
    for rel in parity.LAUNCHER_FAMILIES:
        text = (REPO / rel).read_text()
        calls, hardcoded = _lever_shape(text, message)
        assert calls, f"{Path(rel).name}: {message!r} does not call the helper"
        assert hardcoded == 0, (
            f"{Path(rel).name}: {message!r} still hard-codes {hardcoded} "
            "draft-vocab clauses; full_vocab is impossible there"
        )
        assert f"{var}=${{{var}:-k64_root}}" in text, (
            f"{Path(rel).name}: {var} is never defaulted, so the call above "
            "would pass an empty profile and refuse everything"
        )


def test_the_vocab_profile_scan_is_clean_at_head():
    assert parity.scan_vocab_profile_parity() == []


def _reintroduce_the_hardcode(text):
    """Put fr14_leg3's B1 selector back the way site 12 found it.

    Located POSITIONALLY, by walking back from the lever's own refusal to its
    predicate -- the opener line `[[ "${FR13_FIXED32_MODE:-}" == ...` occurs
    five times in the file, so a text-unique anchor does not exist here. This
    is the same walk the detector does, which keeps the mutation honest: it
    puts the hardcode exactly where the detector looks.
    """
    call = (
        '  _fr13_assert_draft_vocab_profile \\\n'
        '    "$FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE" '
        '"FR13 qrow32 B1 selector" || exit 2\n'
    )
    assert text.count(call) == 1, "the B1 conversion moved; restage the mutation"
    lines = text.replace(call, "", 1).split("\n")
    err = next(
        i for i, line in enumerate(lines)
        if "FR13 qrow32 B1 selector requires Hydra27" in line
    )
    start = next(
        i for i in range(err, err - 80, -1) if re.match(r"^\s*\[\[ ", lines[i])
    )
    hardcode = [
        '     && "$FR13_DRAFT_VOCAB_ROOT" == "1" \\',
        '     && "${FR13_DRAFT_VOCAB_K:-65536}" == "65536" \\',
        '     && "${FR13_DRAFT_VOCAB_BLOCKS:-}" == '
        '"/workspace/scripts/fr13_dvk_subset_blocks.json" \\',
    ]
    return "\n".join(lines[:start + 1] + hardcode + lines[start + 1:])


def _scan_with_mutated_fork(monkeypatch, mutate):
    tmp = Path(tempfile.mkdtemp())
    (tmp / "scripts").mkdir()
    third = parity.LAUNCHER_FAMILIES[2]
    for rel in parity.LAUNCHER_FAMILIES:
        target = tmp / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        text = (REPO / rel).read_text()
        target.write_text(mutate(text) if rel == third else text)
    monkeypatch.setattr(parity, "REPO", tmp)


def test_reintroducing_the_hardcode_fails_the_structural_scan(monkeypatch):
    """The mutation proof site 12 asks for: put it back, the scan must fail."""
    _scan_with_mutated_fork(monkeypatch, _reintroduce_the_hardcode)
    bad = parity.scan_vocab_profile_parity()
    assert bad, "the structural scan did not notice the hardcode coming back"
    assert any("B1 selector" in b for b in bad), bad
    assert any("fr14_leg3" in b and "hardcoded=3" in b for b in bad), bad


def test_the_structural_scan_finds_what_no_marker_names(monkeypatch):
    """The point of the detector, stated as a test.

    An enumerated roster only ratchets after each miss. Here the roster is
    emptied of everything relating to the conversion -- as it genuinely was
    before pass 113 -- and the structural scan must still find the
    divergence, because it is told no lever's name.
    """
    _scan_with_mutated_fork(monkeypatch, _reintroduce_the_hardcode)
    monkeypatch.setattr(
        parity, "FAMILY_PARITY_MARKERS",
        tuple(m for m in parity.FAMILY_PARITY_MARKERS
              if "QUALIFICATION_PROFILE" not in m),
    )
    assert parity.scan_family_parity() == [], (
        "precondition: with its markers removed the roster must be blind"
    )
    assert parity.scan_vocab_profile_parity(), (
        "the structural scan is only worth having if it sees what the roster "
        "was never told about"
    )


def test_the_region_key_carries_an_ordinal(monkeypatch):
    """Keyed on (text, ordinal), never on text.

    "FR13 GDN GQA-group3 production requires " prefixes two different
    refusals twenty lines apart. The first version of this detector kept the
    LAST one per prefix and reported two divergences where there were four --
    it missed two instances of the exact defect it was built for.
    """
    regions = parity._vocab_regions(
        (REPO / parity.LAUNCHER_FAMILIES[0]).read_text()
    )
    assert all(isinstance(k, tuple) and len(k) == 2 for k in regions)
    prefixes = [k[0] for k in regions]
    collided = {p for p in prefixes if prefixes.count(p) > 1}
    assert collided, (
        "no prefix collides, so this test proves nothing -- re-check "
        "_KEY_CHARS against the launcher's refusal messages"
    )
    for prefix in collided:
        ordinals = sorted(k[1] for k in regions if k[0] == prefix)
        assert ordinals == list(range(1, len(ordinals) + 1)), prefix
