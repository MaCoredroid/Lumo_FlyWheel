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
        assert "patch_source_sha256 identity to mint from" in branch, (
            f"{Path(rel).name}: the patcher-digest mint is outside the "
            "arming branch"
        )
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


# ===========================================================================
# SITE 14 (pass 118): the literal-table projection.
#
# Site 12's projection asks a POLICY question -- does this region delegate the
# identity or hard-code it? Site 14's asks a VALUE question: a constant that
# must track a shared authority, carried as a literal in three places. Same
# machinery, different projection, which is the point: the next class gets a
# new projection, not a marker per site.
# ===========================================================================

# The pre-NVFP4-port checkpoint fr14_leg3 was still carrying, and the current
# values it now carries. Measurement 1 died on the first pair.
SITE_14_STALE = {
    "37335563648": "25430574256",
    "29848731008": "25254282384",
    "27977022848": "25210209416",
    "136.7603064029304": "93.15228665201465",
    "109.336011018": "92.506528879",
    "102.479937172": "92.345089436",
}


def test_the_literal_table_is_clean_at_head():
    assert parity.scan_literal_table_parity() == []


def test_the_literal_projection_covers_every_family_completely():
    """The projection is exact, not heuristic -- and that is measured.

    Every key it extracts exists in all three families: the forks are forks in
    their plumbing, not in their constants. That is what makes cross-family
    EQUALITY the right rule here, where the selector-gate regions could only
    support a shape comparison. If a future edit gives one family a constant
    the others do not have, this test says so before the equality rule starts
    reporting divergences that are not defects.
    """
    tables = {
        rel: parity._literal_table((REPO / rel).read_text())
        for rel in parity.LAUNCHER_FAMILIES
    }
    keys = set()
    for table in tables.values():
        keys |= set(table)
    assert len(keys) > 500, f"the projection extracted only {len(keys)} keys"
    lonely = {
        key for key in keys
        if sum(key in table for table in tables.values()) < len(tables)
    }
    # EXACT, not a floor: a new single-family literal must fail here rather
    # than be absorbed. Each entry needs a reason, and this one has a bad one.
    #
    # FR14_REQUIRE_NVFP4_LMHEAD is the fail-closed guard that makes the arm B
    # floor honest -- production and the armb twin arm it, fr14_leg3 has
    # neither it nor the loader patch it guards. That is not a fork being a
    # fork; it is the same selective staleness as site 14, and it is tracked
    # by test_the_nvfp4_lmhead_guard_is_armed_in_every_family in
    # tests/test_fr13_fixed32_floor_propagation.py, which is RED on purpose
    # until Mark rules on regenerating leg3. When it goes green, this
    # exception must be deleted, and this assertion is what makes that happen.
    # Empty since the leg3 stopgap: FR14_REQUIRE_NVFP4_LMHEAD was the last
    # entry here, and it was not a fork being a fork -- it was the same
    # selective staleness as site 14, tracked red until leg3 was aligned.
    # The exception list is EXACT so a new entry has to be justified in
    # writing rather than absorbed.
    justified = set()
    assert lonely == justified, (
        "the set of literals that exist in some families and not others "
        f"changed: {sorted(lonely ^ justified)}. Cross-family equality is the "
        "rule for every other key, so a new entry here is either a defect or "
        "a deliberate divergence that needs writing down."
    )


@pytest.mark.parametrize("stale,current", sorted(SITE_14_STALE.items()))
def test_re_staling_one_value_fires_the_scanner(monkeypatch, stale, current):
    """The mutation proof: put ONE pre-port value back, the scanner fires."""
    def mutate(text):
        assert text.count(f"={current}\n") >= 1, current
        return text.replace(f"={current}\n", f"={stale}\n", 1)

    _scan_with_mutated_fork(monkeypatch, mutate)
    bad = parity.scan_literal_table_parity()
    assert bad, f"re-staling {current} -> {stale} went unnoticed"
    assert any(stale in b for b in bad), bad
    assert any("fr14_leg3" in b for b in bad), bad


def test_the_ordinal_is_what_makes_the_table_visible(monkeypatch):
    """Keyed on (name, ORDINAL), for the reason site 12 taught.

    The floor table assigns _fixed32_expected_mandatory_weight_bytes three
    times, once per vocabulary row. Keyed on the name alone, only one row
    would ever be compared and the other two would be invisible -- which is
    exactly the failure mode of the first vocab-profile detector.
    """
    table = parity._literal_table(
        (REPO / parity.LAUNCHER_FAMILIES[0]).read_text()
    )
    ordinals = sorted(
        key[2] for key in table
        if key[1] == "_fixed32_expected_mandatory_weight_bytes"
    )
    assert ordinals == [1, 2, 3], ordinals
    values = {
        table[("assignment", "_fixed32_expected_mandatory_weight_bytes", n)]
        for n in ordinals
    }
    assert len(values) == 3, "three rows, three distinct values, three keys"


def test_the_projection_ignores_comments():
    """A superseded number named in prose is documentation, not a defect."""
    table = parity._literal_table(
        "# _fixed32_expected_weight_floor_ms=102.479937172 was arm A\n"
        "_fixed32_expected_weight_floor_ms=92.345089436\n"
    )
    assert list(table.values()) == ["92.345089436"]


# ===========================================================================
# SITE 15: import-precedes-owner.
#
# The first projection whose question is INTRA-file. Sites 12 and 14 were both
# "these three files disagree"; this one is "these two lines in ONE file
# disagree about who owns a name", and all three families would have been
# equally wrong, so no cross-family comparison could ever have seen it.
# ===========================================================================


def test_import_precedes_owner_is_clean_at_head():
    assert parity.scan_import_precedes_owner() == []


def test_the_pointer_whitelist_is_read_from_the_launcher_not_listed_here():
    """The projection covers names added to the whitelist tomorrow.

    A roster of the twelve current names would have to be edited every time
    the pointer's contract widens -- which is the ratchet this campaign keeps
    paying for. It reads the launcher's own `case` instead.
    """
    text = (REPO / parity.LAUNCHER_FAMILIES[0]).read_text()
    names, call_line = parity._pointer_imported_names(text)
    assert len(names) >= 12, names
    assert "FR13_FA2_QROW32_B1_SO_SIZE" in names
    assert call_line > 0
    # a name invented here must be picked up without touching the detector
    widened = text.replace(
        "      | FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE)",
        "      | FR13_FA2_QROW32_B1_INVENTED_TOMORROW \\\n"
        "      | FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE)",
        1,
    )
    widened_names, _ = parity._pointer_imported_names(widened)
    assert "FR13_FA2_QROW32_B1_INVENTED_TOMORROW" in widened_names


def _scan_with_mutated_production(monkeypatch, mutate):
    tmp = Path(tempfile.mkdtemp())
    (tmp / "scripts").mkdir()
    first = parity.LAUNCHER_FAMILIES[0]
    for rel in parity.LAUNCHER_FAMILIES:
        target = tmp / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        text = (REPO / rel).read_text()
        target.write_text(mutate(text) if rel == first else text)
    monkeypatch.setattr(parity, "REPO", tmp)


def test_removing_the_withdrawal_fires_the_scan(monkeypatch):
    """Recognising the withdrawal is not a rubber stamp.

    Two names are still ':-' defaulted by the owner on purpose -- a caller may
    legitimately present a credential earned at an older commit, which pass
    101's re-scope made valid. What makes that safe is the withdrawal running
    first. Delete it and the scan says so.
    """
    def mutate(text):
        call = "    _fr13_b1_withdraw_pointer_imports\n"
        assert text.count(call) == 1
        return text.replace(call, "", 1)

    _scan_with_mutated_production(monkeypatch, mutate)
    bad = parity.scan_import_precedes_owner()
    assert len(bad) == 2, bad
    assert any("SOURCE_COMMIT" in b for b in bad)
    assert any("PATCH_SOURCE_SHA256" in b for b in bad)


@pytest.mark.parametrize(
    "name",
    ["FR13_FA2_QROW32_B1_SO_SIZE", "FR13_FA2_QROW32_B1_SO_SHA256",
     "FR13_FA2_QROW32_B1_FA2_HEAD", "FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256"],
)
def test_re_defaulting_an_owned_pin_fires_the_scan(monkeypatch, name):
    """Site 15 itself, one pin at a time.

    The mutation restores the ':-' AND puts the line back above the
    withdrawal, which is exactly where site 15 had it. Both parts matter: the
    invariant is not "no ':-' anywhere" but "the importer's values are gone by
    the time the owner reads them", and ORDER is what decides that.
    """
    def mutate(text):
        owned = f"    {name}=$_FR13_SPLITK_DEFAULT_"
        idx = text.index(owned)
        eol = text.index("\n", idx)
        literal = text[idx + len(f"    {name}="):eol]
        line = text[idx:eol + 1]
        restored = f"    {name}=${{{name}:-{literal}}}\n"
        text = text.replace(line, "", 1)
        # put it back BEFORE the withdrawal, i.e. exactly where site 15 had it
        call = "    _fr13_b1_withdraw_pointer_imports\n"
        return text.replace(call, restored + call, 1)

    _scan_with_mutated_production(monkeypatch, mutate)
    bad = parity.scan_import_precedes_owner()
    assert bad and any(name in b for b in bad), bad


def test_the_projection_is_intra_file_not_cross_family():
    """Stated as a test so the distinction is not lost.

    Sites 12 and 14 were cross-family questions. Site 15 is not: all three
    families would have been equally wrong, so the vocab-profile and
    literal-table scans are structurally incapable of finding it.
    """
    assert parity.scan_vocab_profile_parity() == []
    assert parity.scan_literal_table_parity() == []
    assert parity.scan_family_parity() == []
    # ... and only one family even has the importer
    with_importer = [
        rel for rel in parity.LAUNCHER_FAMILIES
        if parity._pointer_imported_names((REPO / rel).read_text())[0]
    ]
    assert with_importer == [parity.LAUNCHER_FAMILIES[0]], with_importer


# ===========================================================================
# SITE 16: mint-by-hashing-the-artifact.
# ===========================================================================


def test_mint_hash_scan_is_clean_at_head():
    assert parity.scan_mint_hashes_its_own_gate() == []


def test_restoring_the_self_hashing_mint_fires_the_scan(monkeypatch):
    """The tautology, put back."""
    def mutate(text):
        anchor = (
            "    FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256="
            "${FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256:-"
        )
        start = text.index(anchor)
        end = text.index("')}", start) + 3
        return text[:start] + (
            anchor + "$(sha256sum scripts/fr13_patch_fa2_tree_bias.py "
            "| cut -d' ' -f1)}"
        ) + text[end:]

    _scan_with_mutated_production(monkeypatch, mutate)
    bad = parity.scan_mint_hashes_its_own_gate()
    assert bad and "PATCH_SOURCE_SHA256" in bad[0], bad
    assert "the gate cannot fail" in bad[0]


def test_the_adjudicated_exception_is_exact_and_reasoned():
    """One entry, and the reason is in the source next to it.

    An exception list that grows silently is a detector being switched off one
    line at a time, so the set is asserted exactly and the module is required
    to explain it where a reader will meet it.
    """
    assert parity._MINT_HASH_ADJUDICATED == frozenset(
        {"FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL_SHA256"}
    )
    source = (REPO / "scripts" / "fr14_mode_table_parity.py").read_text()
    reason = source[source.index("_MINT_HASH_ADJUDICATED") - 1200:
                    source.index("_MINT_HASH_ADJUDICATED")]
    assert "verify-tier-b" in reason and "recorded, not repaired" in reason


# ===========================================================================
# SITE 18: commit-binding scope.
# ===========================================================================


def test_commit_binding_scope_is_clean_at_head():
    assert parity.scan_commit_binding_scope() == []


def test_unscoping_the_head_comparison_fires_the_scan(monkeypatch):
    """The pre-pass-108 launcher, restored."""
    def mutate(text):
        anchor = "  if (( _fr13_b1_commit_bound == 1 )); then\n"
        assert text.count(anchor) == 1
        return text.replace(anchor, "  if true; then\n", 1)

    _scan_with_mutated_production(monkeypatch, mutate)
    bad = parity.scan_commit_binding_scope()
    assert bad and "without a tier scope" in bad[0], bad


def test_the_scan_sees_every_family():
    """All three carry the scoped comparison, so all three are covered."""
    for rel in parity.LAUNCHER_FAMILIES:
        text = (REPO / rel).read_text()
        assert parity._HEAD_COMPARISON.search(text), (
            f"{Path(rel).name} has no B1 commit-vs-HEAD comparison at all -- "
            "the byte-gated route must still bind its commit"
        )
        assert parity._COMMIT_SCOPE_OPENER in text


# ===========================================================================
# SITE 19: the three workload tables, tied.
#
# Sites 12, 17, 18 and 19 are one disease -- a concept stated N times and
# updated N-1 times. Site 19 is its purest form: exact16_minus_13236 landed in
# the launcher and patcher tables and was structurally inexpressible in
# fr13_floor_gate.EVIDENCE_SETS, which is keyed by task COUNT rather than by
# workload, so fifteen had nowhere to go.
# ===========================================================================


def test_the_three_workload_tables_agree():
    assert parity.scan_workload_table_agreement() == []


def test_the_evidence_set_is_derived_from_its_parent():
    """Derived, not listed -- and asserted to have stayed derived."""
    import json

    sys.path.insert(0, str(REPO / "scripts"))
    import fr13_floor_gate

    fifteen = fr13_floor_gate.EVIDENCE_SETS[15]
    sixteen = fr13_floor_gate.EVIDENCE_SETS[16]
    assert set(sixteen["task_ids"]) - set(fifteen["task_ids"]) == {
        "astropy__astropy-13236"
    }
    assert fifteen["task_ids"] == tuple(
        t for t in sixteen["task_ids"] if t != "astropy__astropy-13236"
    ), "the fifteen reordered their parent; a QC resumed out of order is a "\
       "different measurement"
    # ... and the table agrees with the file it names
    body = json.loads((REPO / fifteen["relative_path"]).read_text())
    assert tuple(body["instance_ids"]) == fifteen["task_ids"]
    import hashlib

    assert hashlib.sha256(
        (REPO / fifteen["relative_path"]).read_bytes()
    ).hexdigest() == fifteen["sha256"]


def test_removing_the_evidence_set_key_fires_the_tie(monkeypatch):
    """The mutation proof site 19 asks for.

    Drop the 15-key with the other two tables intact -- exactly the state the
    runner met -- and the tie must say which table is behind.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    import fr13_floor_gate

    reduced = {
        count: row
        for count, row in fr13_floor_gate.EVIDENCE_SETS.items()
        if count != 15
    }
    monkeypatch.setattr(fr13_floor_gate, "EVIDENCE_SETS", reduced)
    bad = parity.scan_workload_table_agreement()
    assert bad, "the tie did not notice a workload missing from EVIDENCE_SETS"
    assert any("exact16_minus_13236" in b for b in bad), bad
    assert any("EVIDENCE_SETS does not know" in b for b in bad), bad
    # every launcher family reports it, not just the first
    assert len({b.split(":")[0] for b in bad}) == len(parity.LAUNCHER_FAMILIES)


def test_a_workload_missing_from_the_patcher_fires_the_tie(monkeypatch):
    """The other direction: two tables know it, the container half does not."""
    real = parity.patcher_workload_table

    def reduced(text):
        table = real(text)
        table.pop("exact16_minus_13236")
        return table

    monkeypatch.setattr(parity, "patcher_workload_table", reduced)
    bad = parity.scan_workload_table_agreement()
    assert bad and any("but the patcher names" in b for b in bad), bad


def test_a_synthetic_workload_stays_out_of_the_evidence_sets():
    """random1024_calibration has no subset, so it must not be filed as one.

    EVIDENCE_SETS binds evidence FILES. A synthetic prompt shape that acquired
    an entry there would be a subset identity for a run with no subset -- the
    pins-as-fiction move, arriving through the third table instead of the
    first.
    """
    launcher = parity.launcher_workload_table(
        (REPO / parity.LAUNCHER_FAMILIES[0]).read_text()
    )
    assert launcher["random1024_calibration"]["relative_path"] == ""
    assert launcher["random1024_calibration"]["task_ids"] == ()
    evidence = parity.evidence_sets_table()
    assert all(entry["task_ids"] for entry in evidence.values())
    assert 0 not in {entry["task_count"] for entry in evidence.values()}


# ===========================================================================
# SITE 20: literal count disjunctions in bash guards.
#
# The three-table tie compares TABLES; site 20 was a guard literal five
# hundred lines downstream of the authority that had just validated fifteen.
# Fifth statement of the rule, invisible to a table comparison.
# ===========================================================================


def test_literal_count_guards_are_clean_at_head():
    assert parity.scan_literal_count_guards() == []


def test_the_count_guard_subject_is_the_authority_validated_one():
    """Not every integer disjunction -- only counts an authority blessed.

    Widening this scan to the launcher families reported 54 guards and none of
    them were this class: MAX_NUM_SEQS 1|4, booleans, GDN BV sizes. Those own
    their values; nothing computes them and no table can outgrow them.
    Adjudicating 54 entries would be a detector switched off one line at a
    time, so the SUBJECT is narrowed instead -- to the three spellings bash
    offers for a count of an authority-validated artifact.
    """
    match = parity._COUNTED_SUBJECT.search
    assert match("${#_fixed32_task_ids[@]}")      # site 21
    assert match("${_fixed32_subset_binding[0]}")  # site 20
    assert match("$FIXED32_TASK_COUNT")            # site 20's sibling
    for owned in ("$MAX_NUM_SEQS", "$FR13_DRAFT_HEAD_FP8", "$SWE_CONCURRENCY",
                  "$_fr13_gdn_path_bv_candidate", "${FR13_FIXED32_B1_DIAGNOSTIC:-0}"):
        assert not match(owned), owned
    # ... and a file with no authority in scope is not scanned at all
    assert parity._AUTHORITY_MODULE == "fr13_floor_gate"


@pytest.mark.parametrize(
    "rel,old,new",
    [
        (
            "scripts/fr13_bigdenom_swe_serve_variant.sh",
            '    [[ "${_fixed32_subset_binding[0]}" =~ ^[0-9]+$ \\\n'
            '       && ",${_fixed32_subset_binding[2]}," == '
            '*",${_fixed32_subset_binding[0]},"* ]] \\',
            '    [[ "${_fixed32_subset_binding[0]}" == "4" \\\n'
            '       || "${_fixed32_subset_binding[0]}" == "16" ]] \\',
        ),
        (
            "scripts/fr13_b4_campaign_driver.sh",
            '  [[ "$FIXED32_TASK_COUNT" =~ ^[0-9]+$ \\\n'
            '     && ",${FIXED32_ALLOWED_TASK_COUNTS}," == '
            '*",${FIXED32_TASK_COUNT},"* ]] \\',
            '  [[ "$FIXED32_TASK_COUNT" == "4" || '
            '"$FIXED32_TASK_COUNT" == "16" ]] \\',
        ),
    ],
    ids=["serve-variant", "b4-driver"],
)
def test_restoring_the_literal_disjunction_fires_the_scan(
    monkeypatch, rel, old, new
):
    """Both call sites, put back the way the 00:11Z fire found them."""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "scripts").mkdir()
    for site in parity.COUNT_GUARD_SITES:
        text = (REPO / site).read_text()
        if site == rel:
            assert text.count(old) == 1, f"{site}: the guard moved"
            text = text.replace(old, new, 1)
        (tmp / site).parent.mkdir(parents=True, exist_ok=True)
        (tmp / site).write_text(text)
    monkeypatch.setattr(parity, "REPO", tmp)
    bad = parity.scan_literal_count_guards(parity.COUNT_GUARD_SITES)
    assert bad and any(Path(rel).name in b for b in bad), bad
    assert any("literal disjunction" in b for b in bad)


def test_the_projection_reads_both_bash_spellings_of_a_count():
    """Site 20 was a quoted array field; site 21 was an unquoted LENGTH.

    Two drafts, two misses. The first matched count-ish NAMES and missed
    ${_fixed32_subset_binding[0]}. The second matched only QUOTED subjects and
    missed ${#_fixed32_task_ids[@]} -- the same `#` whose absence from the
    runner's census grep produced a zero-hit it was right not to believe. A
    projection for "compares a count to literals" that cannot see the
    canonical way bash spells a count was never going to find anything.
    """
    for line, subject in (
        ('[[ "${_some_array[0]}" == "4" || "${_some_array[0]}" == "16" ]]',
         "${_some_array[0]}"),
        ('[[ ${#_fixed32_task_ids[@]} == 4 || ${#_fixed32_task_ids[@]} == 16 ]]',
         "${#_fixed32_task_ids[@]}"),
    ):
        found = parity._INT_COMPARISON.findall(line)
        seen = {}
        for quoted, length, literal in found:
            seen.setdefault(quoted or length, set()).add(literal)
        assert seen.get(subject) == {"4", "16"}, (subject, seen)
    # and continuations are joined, which is how the real site is spelled
    guards = list(parity._bracket_guards(
        ["    [[ ${#_fixed32_task_ids[@]} == 4 \\",
         "       || ${#_fixed32_task_ids[@]} == 16 ]]"]
    ))
    assert guards and "16" in guards[0][1]


@pytest.mark.parametrize("rel", parity.LAUNCHER_FAMILIES, ids=lambda r: Path(r).name)
def test_restoring_the_ingress_disjunction_fires_the_scan(monkeypatch, rel):
    """Site 21, one launcher at a time, on the real line.

    The subject is ${#_fixed32_task_ids[@]} -- an array LENGTH, unquoted --
    which is what made this invisible to the previous draft and to the
    runner's census grep.
    """
    tmp = Path(tempfile.mkdtemp())
    (tmp / "scripts").mkdir()
    for site in parity.COUNT_GUARD_SITES:
        text = (REPO / site).read_text()
        if site == rel:
            start = text.index("    _fixed32_allowed_task_counts=$(python3 -c")
            tail = '"${#_fixed32_task_ids[@]}" >&2; exit 2; }\n'
            end = text.index(tail, start) + len(tail)
            text = text[:start] + (
                "    [[ ${#_fixed32_task_ids[@]} == 4 "
                "|| ${#_fixed32_task_ids[@]} == 16 ]] \\\n"
                '      || { echo "bad" >&2; exit 2; }\n'
            ) + text[end:]
        (tmp / site).parent.mkdir(parents=True, exist_ok=True)
        (tmp / site).write_text(text)
    monkeypatch.setattr(parity, "REPO", tmp)
    bad = parity.scan_literal_count_guards(parity.COUNT_GUARD_SITES)
    assert bad and any("${#_fixed32_task_ids[@]}" in b for b in bad), bad
    assert all(Path(rel).name in b for b in bad), bad


@pytest.mark.parametrize("rel", parity.LAUNCHER_FAMILIES, ids=lambda r: Path(r).name)
def test_the_ingress_guard_derives_and_the_prose_names_the_authority(rel):
    """No new literal, and no comment teaching the old key set."""
    text = (REPO / rel).read_text()
    assert "must contain exactly 4 or 16 IDs" not in text
    assert "EVIDENCE_SETS 4 and 16" not in text, (
        "a comment naming the authority's CONTENTS teaches the next reader "
        "the wrong rule the moment the authority grows a key"
    )
    assert 'from fr13_floor_gate import EVIDENCE_SETS' in text
    assert '",${_fixed32_allowed_task_counts}," == *",${#_fixed32_task_ids[@]},"*' in text


# ===========================================================================
# SITE 22: the serve execution closure, and the quantifier projection.
#
# The count rule was a REGEX REPETITION QUANTIFIER -- {3} for four ids, {15}
# for sixteen -- so the literals never appeared and every count scan was blind
# by construction; and it lived in a fourth root no sweep covered. Site 21's
# "closed by predicate" was retracted for exactly that: a predicate is closed
# only over the universe it runs on, and the universe was a hand-kept roster.
# ===========================================================================

QUANTIFIER_REGEX = (
    '        [[ "$FIXED32_TASK_IDS" =~ '
    "^[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+(,[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+){3}$ \\\\\n"
    '           || "$FIXED32_TASK_IDS" =~ '
    "^[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+(,[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+){15}$ ]] \\\\\n"
    '          || { echo "FAIL: bad"; exit 5; }\n'
)


def test_the_closure_contains_the_file_site_22_lived_in():
    """The universe must include the root that hid the defect.

    The first draft matched invocation SYNTAX and missed
    OFFLOAD_HELPER=scripts/swe_x86_helpers/offload_codex_proxy.sh -- a path
    stored in a variable and run later -- so the closure excluded the very
    file it was built to find.
    """
    closure = parity.serve_execution_closure()
    assert "scripts/swe_x86_helpers/offload_codex_proxy.sh" in closure
    assert "scripts/fr13_required_tree_flags.sh" in closure, (
        "a source-with-SCRIPT_DIR edge is not being resolved"
    )
    for root in parity.CLOSURE_ROOTS:
        assert root in closure
    assert len(closure) > 100, len(closure)


def test_the_closure_is_a_fixed_point():
    """Running it from its own output must add nothing."""
    closure = parity.serve_execution_closure()
    assert parity.serve_execution_closure(tuple(closure)) == closure


def test_both_count_scans_are_clean_across_the_whole_closure():
    assert parity.scan_literal_count_guards() == []
    assert parity.scan_regex_quantifier_counts() == []


def test_restoring_the_quantifier_fires_the_new_scan(tmp_path):
    """Mutation proof: the rule back inside the regex, where nothing sees it."""
    probe = tmp_path / "scripts"
    probe.mkdir()
    (probe / "proxy.sh").write_text(QUANTIFIER_REGEX)
    import fr14_mode_table_parity as parity_mod

    original = parity_mod.REPO
    try:
        parity_mod.REPO = tmp_path
        bad = parity_mod.scan_regex_quantifier_counts(["scripts/proxy.sh"])
    finally:
        parity_mod.REPO = original
    assert bad and all("repetition quantifier" in b for b in bad), bad
    # one finding per line: the four-id rule and the sixteen-id rule
    assert {"{3}", "{15}"} <= {q for b in bad for q in ("{3}", "{15}") if q in b}, bad


def test_the_quantifier_scan_ignores_regexes_that_are_not_counting_ids():
    """A {64} on a hex digest is not a task-count rule."""
    import fr14_mode_table_parity as parity_mod

    original = parity_mod.REPO
    try:
        parity_mod.REPO = REPO
        # a {64} on a hex digest has no group-closing paren before it and no
        # id-shaped alternation, so neither half of the projection matches
        hexish = "re.fullmatch(r'[0-9a-f]{64}', payload['task_hmac_key_hex'])"
        assert not parity_mod._QUANTIFIER_COUNT.search(hexish)
        assert not parity_mod._ID_SHAPED.search(hexish)
    finally:
        parity_mod.REPO = original


def test_the_proxy_validates_format_and_count_separately():
    """The decomposition, asserted where it lives."""
    text = (REPO / "scripts/swe_x86_helpers/offload_codex_proxy.sh").read_text()
    assert "){3}$" not in text and "){15}$" not in text
    assert "not an exact 4/16 list" not in text
    assert "_fixed32_require_canonical_task_ids" in text
    assert "from fr13_floor_gate import EVIDENCE_SETS" in text
    # format regex is now unbounded (`*`), count is a separate membership test
    assert "(,[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+)*$" in text
    assert '",$_allowed," == *",$_count,"*' in text
    # and the diagnostic mode is still expressible, by DECLARED mode
    assert '"$FIXED32_TASK_IDS" == "$FIXED32_DIAGNOSTIC_TASK_ID"' in text
