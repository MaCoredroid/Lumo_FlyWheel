"""Contract lint: the sides of every enumerated pair must agree.

Three campaign blockers in a row were one defect shape — a paired structure
updated on one side only — and each cost a boot to find. This is the unit test
that makes the next one cost a test run instead.

`scripts/fr14_paired_contract_sweep.py` enumerates the pairs. This file does two
things with it:

  1. asserts the inventory is clean at HEAD;
  2. **asserts every detector can actually fail** — each pair is mutated to its
     known stale form and the detector must catch it. A lint that cannot fail is
     worse than no lint, because it reads like coverage.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import fr14_paired_contract_sweep as sweep  # noqa: E402


# ---------------------------------------------------------------------------
# 1. The inventory at HEAD.
# ---------------------------------------------------------------------------

def test_inventory_is_clean():
    rows = sweep.sweep()
    stale = [r for r in rows if not r["ok"]]
    assert not stale, "stale sides: " + "; ".join(
        f"{r['pair']}: {r['detail']}" for r in stale
    )


def test_inventory_covers_every_pair_kind():
    """Each defect-shape family that has bitten this campaign must be covered."""
    kinds = {r["kind"] for r in sweep.sweep()}
    assert {"contract/consumer", "half/half", "emitter/validator",
            "twin/twin", "bash/python",
            # Arm S's 17th refusal: an arm-identity resolver that DEFAULTS
            # instead of refusing. Its own family, because the campaign fixed
            # this shape three times as three unrelated instances before
            # anyone named it.
            "fallback-pattern"} <= kinds


def test_inventory_size_is_recorded():
    """Adding a pair without updating the count is itself a one-sided update."""
    assert len(sweep.PAIRS) == 16, (
        "pair count changed -- update the sweep note in "
        "results/fr14_nvfp4_port_20260816/suffix_pass_gating.md 13"
    )


# ---------------------------------------------------------------------------
# 2. Mutation coverage -- every detector must be able to fail.
# ---------------------------------------------------------------------------

def _blob():
    return sweep.injected_blob()


def _topo():
    return sweep._topology()


def test_detector_catches_a_stale_step_shape_literal():
    mutated = _blob().replace(
        ") not in ((4, 1), (4, 2), (2, 1))", ") not in ((4, 1),)"
    )
    ok, detail = sweep.pair_contract_vs_blob(mutated, _topo())
    assert not ok and "!=" in detail


def test_detector_catches_a_stale_graph_captures_bound():
    mutated = _blob().replace(
        'proposal["graph_captures"]) not in (0, 1, 2)',
        'proposal["graph_captures"]) not in (0, 1)',
    )
    ok, detail = sweep.pair_contract_vs_blob_captures(mutated, _topo())
    assert not ok


@pytest.mark.parametrize(
    "needle,replacement",
    [
        ('"graph_replays": int(proposal["graph_replays"]),',
         '"graph_replays": 1,'),
        ('"mtp_forward_calls": _fr14_calls,', '"mtp_forward_calls": 4,'),
        ('"mtp_forward_rows": _fr14_calls * batch,',
         '"mtp_forward_rows": 4 * batch,'),
    ],
)
def test_detector_catches_a_relapsed_runtime_emitter(needle, replacement):
    """This is the exact 12th-site regression, one literal at a time."""
    blob = _blob()
    assert needle in blob, f"anchor moved: {needle}"
    ok, detail = sweep.pair_emitter_halves(blob.replace(needle, replacement))
    assert not ok, f"detector missed {replacement}"


def test_detector_catches_a_relapsed_drafter_evidence_check():
    """The 13th site -- found by the sweep, not by a boot."""
    blob = _blob()
    mutated = blob.replace(
        'or int(drafter_evidence.get("matching_replays", -1))\n'
        '        != int(runtime.get("graph_replays", -1))',
        'or int(drafter_evidence.get("matching_replays", -1)) != 1',
    )
    assert mutated != blob, "13th-site anchor moved"
    ok, detail = sweep.pair_emitter_halves(mutated)
    assert not ok and "drafter_evidence" in str(detail)


def test_detector_catches_an_untied_proposal_end_evidence_check():
    blob = _blob().replace(
        'int(evidence.get("matching_replays", -1))\n'
        '            != int(proposal["graph_replays"])',
        'int(evidence.get("matching_replays", -1)) != 1',
    )
    ok, detail = sweep.pair_emitter_halves(blob)
    assert not ok and "not tied" in str(detail)


def test_detector_catches_a_relapsed_observer():
    """The 11th site."""
    for needle, replacement in (
        ('or proposal.get("graph_captures")\n'
         '            != int(drafter_capture.get("segment", 0)) + 1',
         'or proposal.get("graph_captures") != 1'),
        ('or tree_calls not in tuple(\n'
         '                range(int(drafter_capture.get("passes", 4)))\n'
         '            )',
         'or tree_calls not in (0, 1, 2, 3)'),
    ):
        blob = _blob()
        mutated = blob.replace(needle, replacement)
        assert mutated != blob, f"11th-site anchor moved: {replacement}"
        ok, _ = sweep.pair_observer_vs_context(mutated)
        assert not ok, f"detector missed {replacement}"


def _mutate_file(monkeypatch, attr, needle, replacement, tmp):
    src = getattr(sweep, attr).read_text()
    assert needle in src, f"anchor moved in {attr}: {needle[:60]}"
    out = Path(tmp) / "mutated"
    out.write_text(src.replace(needle, replacement))
    monkeypatch.setattr(sweep, attr, out)


@pytest.fixture
def tmp():
    import tempfile

    return tempfile.mkdtemp()


@pytest.mark.parametrize(
    "needle,replacement,expect",
    [
        ("if graph_captures not in LEGAL_GRAPH_CAPTURES:",
         "if graph_captures not in (0, 1):",
         "graph_captures bounded at 0|1"),
        ('"main_tail_length": _fr14_main,',
         '"main_tail_length": ARCTIC_MAIN_TAIL_LENGTH,',
         "main_tail_length pinned to 6"),
        ("_expect(\n        runtime_mtp_calls,\n"
         "        GATED_MTP_FORWARD_CALLS if _fr14_gated else MTP_FORWARD_CALLS,",
         "_expect(\n        runtime_mtp_calls,\n        MTP_FORWARD_CALLS,",
         "mtp_forward_calls pinned to 4"),
    ],
)
def test_detector_catches_a_stale_census_side(monkeypatch, tmp, needle,
                                              replacement, expect):
    _mutate_file(monkeypatch, "CENSUS", needle, replacement, tmp)
    ok, detail = sweep.pair_census_halves()
    assert not ok, f"detector missed: {replacement}"
    assert any(expect in d for d in detail), f"wrong reason: {detail}"


def test_detector_catches_the_census_dropping_the_contract(monkeypatch, tmp):
    _mutate_file(
        monkeypatch, "CENSUS", "LEGAL_STEP_SHAPES", "_LOCAL_STEP_SHAPES", tmp
    )
    ok, detail = sweep.pair_contract_vs_census(_topo())
    assert not ok and "LEGAL_STEP_SHAPES" in str(detail)


def test_detector_catches_a_launcher_twin_divergence(monkeypatch, tmp):
    """One launcher updated, the other not -- the bash/bash pair."""
    src = sweep.LAUNCHERS[1].read_text()
    out = Path(tmp) / "twin.sh"
    out.write_text(src.replace("FR14_GATE_SPLIT_GRAPH", "FR14_GONE"))
    monkeypatch.setattr(sweep, "LAUNCHERS", (sweep.LAUNCHERS[0], out))
    ok, detail = sweep.pair_launcher_twins()
    assert not ok and "only one launcher" in str(detail)


def test_detector_catches_a_launcher_twin_count_skew(monkeypatch, tmp):
    """Present in both but not the same number of times is still a divergence."""
    src = sweep.LAUNCHERS[1].read_text()
    out = Path(tmp) / "twin2.sh"
    out.write_text(src + "\n# FR14_SUFFIX_PASS_GATE mentioned once more\n")
    monkeypatch.setattr(sweep, "LAUNCHERS", (sweep.LAUNCHERS[0], out))
    ok, detail = sweep.pair_launcher_twins()
    assert not ok and "counts differ" in str(detail)


def test_detector_catches_a_bash_python_field_count_drift(monkeypatch, tmp):
    src = sweep.LAUNCHERS[0].read_text()
    out = Path(tmp) / "cfg.sh"
    out.write_text(
        src.replace(
            "${FR14_SUFFIX_PASS_GATE_NGRAM:-8} "
            "${FR14_SUFFIX_PASS_GATE_MIN_AGREE:-0.75} "
            "${FR14_SUFFIX_PASS_GATE_MIN_HISTORY:-256}",
            "${FR14_SUFFIX_PASS_GATE_NGRAM:-8}",
        )
    )
    monkeypatch.setattr(sweep, "LAUNCHERS", (out, sweep.LAUNCHERS[1]))
    ok, detail = sweep.pair_bash_cfg_vs_python_parser()
    assert not ok and "3-field" in str(detail)


def test_detector_catches_the_python_parser_relaxing(monkeypatch, tmp):
    gate = REPO / "scripts" / "fr14_suffix_pass_gate.py"
    out = Path(tmp) / "gate.py"
    out.write_text(gate.read_text().replace("len(raw) != 3", "len(raw) < 1"))
    real_repo = sweep.REPO
    monkeypatch.setattr(sweep, "REPO", Path(tmp))
    (Path(tmp) / "scripts").mkdir(exist_ok=True)
    (Path(tmp) / "scripts" / "fr14_suffix_pass_gate.py").write_text(out.read_text())
    monkeypatch.setattr(sweep, "LAUNCHERS", (real_repo / "scripts" /
                                             "fr13_launch_forked_fa2_tree_server.sh",))
    ok, detail = sweep.pair_bash_cfg_vs_python_parser()
    assert not ok and "3 fields" in str(detail)


# ---------------------------------------------------------------------------
# The 14th site's class: a LONE arithmetic invariant with the ungated shape
# baked in as an addend.  Not a pair -- which is why the pair sweep could not
# see it.  Restated class: no literal may encode the ungated 5-pass shape.
# ---------------------------------------------------------------------------

def test_every_injected_blob_is_enumerated():
    """The 14th site sat in the blob nobody extracted.

    `injected_blob()` returns the ONE blob bound to a named global. The eagle
    proposer is an anonymous local, so for three boots nothing scanned it.
    """
    blobs = sweep.all_injected_blobs()
    assert len(blobs) > 10, f"only {len(blobs)} blobs found -- extractor broke"
    texts = [b[1] for b in blobs]
    assert any("_fr13_t_cols" in x for x in texts), "eagle proposer not enumerated"
    assert any("_fr13_fixed32_drafter_graph_replay" in x for x in texts), (
        "gdn runtime not enumerated"
    )


def test_pack_identity_is_evaluated_not_read():
    ok, detail = sweep.pair_pack_identity_under_both_shapes(sweep._topology())
    assert ok, detail


def test_detector_catches_the_14th_site_regression(monkeypatch, tmp):
    """Put the bare 15 back and the identity must fail under the GATED shape."""
    real = sweep.eagle_blob()
    assert "_fr14_head_cols" in real, "14th-site fix anchor moved"
    monkeypatch.setattr(
        sweep, "eagle_blob", lambda: real.replace("_fr14_head_cols", "15")
    )
    ok, detail = sweep.pair_pack_identity_under_both_shapes(sweep._topology())
    assert not ok and "derived form" in str(detail)


def test_pack_identity_arithmetic_rejects_the_ungated_addend():
    """Independently of the source form: 15 + 8 + 10 is 33, and must not be 31."""
    topo = sweep._topology()
    hd_full = topo.N_MTP_HEAD_DEPTHS
    head = hd_full * (1 + topo.BRANCHES_PER_HEAD_DEPTH)
    rescue = sum(length for _r, length in topo.PHYSICAL_BRANCH_CHAINS)
    # the shipped (broken) form, under the gated shape
    broken = head + topo.GATED_ARCTIC_MAIN_TAIL_LENGTH + rescue
    assert broken == 33, broken
    assert broken != topo.PHYSICAL_DRAFTS
    # the derived form, under both
    for hd, cols in (
        (hd_full, topo.ARCTIC_MAIN_TAIL_LENGTH),
        (topo.GATED_MTP_K, topo.GATED_ARCTIC_MAIN_TAIL_LENGTH),
    ):
        assert head + (cols - (hd_full - hd)) + rescue == topo.PHYSICAL_DRAFTS


def test_shape_literal_scan_is_clean():
    ok, detail = sweep.shape_literal_scan()
    assert ok, f"unadjudicated shape literals: {detail}"


def test_shape_literal_scan_can_actually_fail_on_a_rule(monkeypatch):
    """Drop the walk-depth rule and every 12 must resurface for adjudication."""
    trimmed = tuple(
        r for r in sweep.ADJUDICATED_RULES if 12 not in r[1]
    )
    assert len(trimmed) < len(sweep.ADJUDICATED_RULES)
    monkeypatch.setattr(sweep, "ADJUDICATED_RULES", trimmed)
    ok, detail = sweep.shape_literal_scan()
    assert not ok, "scanner is inert: removing a rule surfaced nothing"
    assert any("12 in" in d for d in detail), detail


def test_shape_literal_scan_can_actually_fail_on_a_specific_entry(monkeypatch):
    """`rows + 4` is covered ONLY by the specific list, so it discriminates."""
    trimmed = tuple(
        row for row in sweep.ADJUDICATED_SHAPE_LITERALS if row[1] != "rows + 4"
    )
    assert len(trimmed) == len(sweep.ADJUDICATED_SHAPE_LITERALS) - 1
    monkeypatch.setattr(sweep, "ADJUDICATED_SHAPE_LITERALS", trimmed)
    ok, detail = sweep.shape_literal_scan()
    assert not ok and any("rows + 4" in d for d in detail), detail


def test_relevance_is_function_scoped_not_line_scoped(monkeypatch):
    """Regression on the detector's own hole.

    Filtering relevance on line text alone dropped drafter-internal lines like
    `"rescue_carry_slots": 4 * batch`; its own mutation test found that. If the
    scan ever goes back to line-scoping, dropping the batch rule stops surfacing
    those lines and this fails.
    """
    monkeypatch.setattr(
        sweep,
        "ADJUDICATED_RULES",
        tuple(r for r in sweep.ADJUDICATED_RULES if 4 not in r[1]),
    )
    monkeypatch.setattr(
        sweep,
        "ADJUDICATED_SHAPE_LITERALS",
        tuple(
            row for row in sweep.ADJUDICATED_SHAPE_LITERALS
            if "rescue_carry_slots" not in row[1]
        ),
    )
    ok, detail = sweep.shape_literal_scan()
    assert not ok
    assert any("rescue_carry_slots" in d for d in detail), (
        "function-scoped relevance regressed to line-scoped: " + str(detail)
    )


def test_handoff_contract_detector_can_fail(monkeypatch):
    blob = sweep.injected_blob().replace(
        '!= (8 if int(proposal["mtp_forward_calls"]) == 2 else 6)',
        '!= (6 if int(proposal["mtp_forward_calls"]) == 2 else 6)',
    )
    ok, detail = sweep.pair_handoff_contract_vs_blob(blob, sweep._topology())
    assert not ok and "!=" in str(detail)


def test_unparseable_blobs_are_checked_not_skipped():
    """24 blobs do not parse; one of them could hold the next 14th site."""
    ok, detail = sweep.shape_literal_scan()
    assert ok
    assert "unparseable, textually checked" in detail
    assert sweep.ADJUDICATED_TEXTUAL, "textual adjudications must exist"


# ---------------------------------------------------------------------------
# The REPLAY dimension (15th site) and why it must be keyed on position.
# ---------------------------------------------------------------------------

def test_replay_literal_scan_is_clean():
    ok, detail = sweep.replay_literal_scan()
    assert ok, f"unadjudicated replay-position literals: {detail}"


def test_replay_scan_can_actually_fail(monkeypatch):
    monkeypatch.setattr(
        sweep, "ADJUDICATED_REPLAY_FUNCTIONS",
        {k: v for k, v in sweep.ADJUDICATED_REPLAY_FUNCTIONS.items()
         if k != "_fr13_fixed32_observed_take"},
    )
    ok, detail = sweep.replay_literal_scan()
    assert not ok and any("observed_take" in d for d in detail), detail


def test_the_flush_twins_are_character_identical_and_mean_opposite_things():
    """The sharpest statement of the class this campaign kept hitting.

    `evidence.get("matching_replays") != 1` appears at TWO positions in the
    fixed_flush blob's reconcile. One guards the TARGET forward graph, which
    really is replayed once per step -- correct, must not change. The other
    guarded the DRAFTER, where an armed ungated step replays twice: the 15th
    site. Byte-for-byte identical, opposite verdicts.

    A text-keyed allowlist marks both OK and hides the real one. The runner's own
    first draft did exactly that. So the adjudication is keyed on POSITION, and
    this test asserts the two remain distinguishable only that way.
    """
    for lineno, text, _tree in sweep.all_injected_blobs():
        if lineno != 39286:
            continue
        lines = text.split("\n")
        target = [
            i + 1 for i, l in enumerate(lines)
            if l.strip() == 'or evidence.get("matching_replays") != 1'
        ]
        drafter = [
            i + 1 for i, l in enumerate(lines)
            if l.strip() == 'or evidence.get("matching_replays")'
        ]
        assert len(target) == 1, (
            "the TARGET twin must still be there and untouched: " + str(target)
        )
        assert drafter, "the DRAFTER site must now be the derived form"
        # and they are in the same function -- function name alone cannot separate
        assert min(drafter) > max(target), (
            "the drafter site follows the target site in the same reconcile"
        )
        return
    pytest.skip("fixed_flush blob not present")


def test_text_keying_would_have_hidden_the_15th_site():
    """Demonstrate the failure mode, rather than asserting it in prose."""
    for lineno, text, _tree in sweep.all_injected_blobs():
        if lineno != 39286:
            continue
        stale_text = 'or evidence.get("matching_replays") != 1'
        # restore the pre-fix source: both twins identical again
        pre_fix = re.sub(
            r'or evidence\.get\("matching_replays"\)\s*\n\s*'
            r'!= runtime\.get\("graph_replays"\)',
            stale_text,
            text,
        )
        occurrences = [
            i + 1 for i, l in enumerate(pre_fix.split("\n"))
            if l.strip() == stale_text
        ]
        assert len(occurrences) == 2, occurrences
        # a text-keyed allowlist has ONE entry and suppresses BOTH
        assert len({stale_text}) == 1
        # a position-keyed allowlist has to name each, so one stays visible
        assert len(set(occurrences)) == 2
        return
    pytest.skip("fixed_flush blob not present")


def test_boundary_path_is_covered_in_all_three_dimensions():
    """columns / passes / replays -- the flush blob is scanned by each."""
    assert any(
        lineno == 39286 for lineno, _t, tree in sweep.all_injected_blobs()
        if tree is not None
    ), "the flush blob must be parseable and scanned"
    ok_shape, _ = sweep.shape_literal_scan()
    ok_replay, _ = sweep.replay_literal_scan()
    assert ok_shape and ok_replay
    assert any(
        "_fr13_f32_flush" in fn for fn in sweep.ADJUDICATED_REPLAY_FUNCTIONS
    ) or any(
        key[1].startswith("_fr13_f32_flush")
        for key in sweep.ADJUDICATED_REPLAY_POSITIONS
    ), "the boundary path must appear in the replay adjudications"


# ---------------------------------------------------------------------------
# 3. The 17th-site detector -- a lint that cannot fail is worse than no lint.
# ---------------------------------------------------------------------------

_IDENTITY_REFUSAL_NEEDLE = (
    "    identity = _FR13_FA2_QROW32_B1_IDENTITIES.get(resolved)\n"
    "    if identity is None:\n"
    "        raise RuntimeError("
)
_IDENTITY_FALLTHROUGH = (
    "    identity = _FR13_FA2_QROW32_B1_IDENTITIES.get(resolved)\n"
    "    if identity is None:\n"
    '        identity = _FR13_FA2_QROW32_B1_IDENTITIES["split2"]\n'
    "    if False:\n"
    "        raise RuntimeError("
)
_SPLITK_ENTRY = '    "gqa_pair_splitk": {\n        "candidate_sha256"'
_SPLITK_ENTRY_RENAMED = '    "SOMETHING_ELSE": {\n        "candidate_sha256"'


def test_arm_identity_detector_passes_on_the_real_tree():
    ok, detail = sweep.arm_identity_resolvers_refuse_unknown_arms()
    assert ok, detail
    ok, detail = sweep.launcher_pin_cases_refuse_unknown_arms()
    assert ok, detail


def test_arm_identity_detector_catches_a_restored_fall_through(monkeypatch, tmp):
    """Put the 17th site BACK and the detector must find it.

    This is the exact defect Arm S hit on its fifth boot: the identity table's
    unknown-arm refusal replaced by a silent handout of the incumbent's pins.
    The detector is behavioural, so it catches this by CALLING the resolver --
    a comment that still says "refuses" cannot save it.
    """
    src = sweep.FA2_PATCHER.read_text()
    assert _IDENTITY_REFUSAL_NEEDLE in src, "anchor moved in FA2_PATCHER"
    out = Path(tmp) / "fa2_patcher_fallthrough.py"
    out.write_text(src.replace(_IDENTITY_REFUSAL_NEEDLE, _IDENTITY_FALLTHROUGH, 1))
    monkeypatch.setattr(sweep, "FA2_PATCHER", out)
    ok, detail = sweep.arm_identity_resolvers_refuse_unknown_arms()
    assert not ok
    assert "did NOT refuse an unknown arm" in detail


def test_arm_identity_detector_catches_a_missing_arm_branch(monkeypatch, tmp):
    """Drop split-K from the identity table and the detector must find it."""
    src = sweep.FA2_PATCHER.read_text()
    assert _SPLITK_ENTRY in src, "split-K identity entry moved"
    out = Path(tmp) / "fa2_patcher_no_splitk.py"
    out.write_text(src.replace(_SPLITK_ENTRY, _SPLITK_ENTRY_RENAMED, 1))
    monkeypatch.setattr(sweep, "FA2_PATCHER", out)
    ok, detail = sweep.arm_identity_resolvers_refuse_unknown_arms()
    assert not ok
    assert "gqa_pair_splitk" in detail


@pytest.mark.parametrize(
    "needle,replacement,expect",
    [
        # a bash `*)` that asserts pins instead of refusing
        ('      echo "FR13 qrow32 B1 pin arm has no pinned identity',
         '      echo "FR13 qrow32 B1 incumbent provenance drifted',
         "does not refuse"),
        # an arm losing its branch
        ("    gqa_pair_splitk)", "    gqa_pair_splitk_disabled)",
         "no gqa_pair_splitk) branch"),
        # the incumbent arms going back to a bare default
        ('    nosplit|split2|"")', "    nosplit_only)",
         "not named explicitly"),
    ],
)
def test_launcher_pin_case_detector_can_fail(
    monkeypatch, tmp, needle, replacement, expect
):
    """Every launcher twin is mutated, one at a time, in the same way.

    Mutating only the first would have passed while the third twin drifted a
    whole arm behind -- which is precisely what happened.
    """
    real = sweep.LAUNCHERS
    for index in range(len(real)):
        src = real[index].read_text()
        assert needle in src, f"anchor moved in {real[index].name}"
        out = Path(tmp) / f"launcher_{index}_{abs(hash(needle))}.sh"
        out.write_text(src.replace(needle, replacement, 1))
        monkeypatch.setattr(
            sweep, "LAUNCHERS", real[:index] + (out,) + real[index + 1:]
        )
        ok, detail = sweep.launcher_pin_cases_refuse_unknown_arms()
        assert not ok, f"{real[index].name} mutation went undetected"
        assert expect in detail
        monkeypatch.setattr(sweep, "LAUNCHERS", real)


def test_all_three_launcher_twins_are_covered():
    """The third twin was outside this tuple, which is how it drifted."""
    assert {p.name for p in sweep.LAUNCHERS} == {
        "fr13_launch_forked_fa2_tree_server.sh",
        "fr14_armb_leg3_launch_nomiddleware.sh",
        "fr14_leg3_launch_nomiddleware.sh",
    }


def test_the_fa2_patcher_is_in_scope_at_all():
    """It was scanned by nothing in this family until the 17th site."""
    assert sweep.FA2_PATCHER.name == "fr13_patch_fa2_tree_bias.py"
    assert sweep.FA2_PATCHER.is_file()


# ---------------------------------------------------------------------------
# 4. Twin equivalence by EXECUTION (site 23).
# ---------------------------------------------------------------------------

_PY_SELECTOR_TUPLE = (
    "    _b1_pin_selectors = (\n"
    '        "FR13_FA2_QROW32_B1_LIVE_AB_ARM",\n'
    '        "FR13_FA2_QROW32_B1_TIER_B_ARM",\n'
    '        "FR13_FA2_QROW32_B1_PRODUCTION_ARM",\n'
    "    )"
)
_PY_SELECTOR_TUPLE_WITHOUT_TIER_B = (
    "    _b1_pin_selectors = (\n"
    '        "FR13_FA2_QROW32_B1_LIVE_AB_ARM",\n'
    '        "FR13_FA2_QROW32_B1_PRODUCTION_ARM",\n'
    "    )"
)


def test_pin_arm_resolver_twins_agree_on_the_real_tree():
    ok, detail = sweep.pin_arm_resolver_twins()
    assert ok, detail


def test_twin_detector_catches_site_23_exactly(monkeypatch, tmp):
    """Revert site 23 and the detector must find it.

    Site 23 was the Python resolver not knowing FR13_FA2_QROW32_B1_TIER_B_ARM
    while its bash twin did, so a tier-B boot resolved the empty string -- and
    the "" key added at site 17 to REPLACE a .get() default handed back
    split2's pins. Naming a default does not remove it; the removal has to
    happen at the resolver.
    """
    real = sweep.LAUNCHERS[0]
    src = real.read_text()
    assert _PY_SELECTOR_TUPLE in src, "the python resolver's selector tuple moved"
    out = Path(tmp) / "launcher_site23.sh"
    out.write_text(
        src.replace(_PY_SELECTOR_TUPLE, _PY_SELECTOR_TUPLE_WITHOUT_TIER_B, 1)
    )
    monkeypatch.setattr(sweep, "LAUNCHERS", (out,))
    ok, detail = sweep.pin_arm_resolver_twins()
    assert not ok
    assert "TIER_B_ARM=gqa_pair_splitk" in detail
    assert "bash='gqa_pair_splitk'" in detail
    # AND NOTE WHAT THE PYTHON SIDE NOW DOES. Site 23's actual behaviour was
    # to resolve "" and hand back split2's pins in silence. With the removal
    # moved to the RESOLVER, deleting the selector no longer produces a wrong
    # answer -- the environment sweep sees a set-but-unknown selector and
    # REFUSES. The twin detector still catches the divergence, and the thing
    # it catches is now a refusal rather than a mis-attested serve.
    assert "python='REFUSED:" in detail
    assert "does not know these selec" in detail


def test_twin_detector_catches_a_bash_side_regression(monkeypatch, tmp):
    """Divergence is caught from either side, not just the python one."""
    real = sweep.LAUNCHERS[0]
    src = real.read_text()
    needle = (
        '  if [[ -z "$_FR13_FA2_QROW32_B1_PIN_ARM" ]]; then\n'
        "    _FR13_FA2_QROW32_B1_PIN_ARM=$FR13_FA2_QROW32_B1_TIER_B_ARM\n"
        "  fi\n"
    )
    assert needle in src, "the bash tier-b fallback moved"
    out = Path(tmp) / "launcher_bash_regression.sh"
    out.write_text(src.replace(needle, "", 1))
    monkeypatch.setattr(sweep, "LAUNCHERS", (out,))
    ok, detail = sweep.pin_arm_resolver_twins()
    assert not ok
    assert "bash='nosplit'" in detail
    assert "python='gqa_pair_splitk'" in detail


def test_twin_detector_reach_is_recorded_not_assumed(monkeypatch, tmp):
    """What the detector does NOT cover, written down.

    Selector PRECEDENCE only becomes observable when two selectors are named
    at once, and both resolvers refuse that outright. So a precedence swap is
    invisible to the answer, and the detector correctly stays green. Recorded
    here so the next reader knows the detector's reach rather than assuming it
    is wider than it is -- the assumption that a check covers more than it does
    is what let sites 17 and 23 share a resolver.
    """
    real = sweep.LAUNCHERS[0]
    src = real.read_text()
    swapped = (
        "    _b1_pin_selectors = (\n"
        '        "FR13_FA2_QROW32_B1_PRODUCTION_ARM",\n'
        '        "FR13_FA2_QROW32_B1_TIER_B_ARM",\n'
        '        "FR13_FA2_QROW32_B1_LIVE_AB_ARM",\n'
        "    )"
    )
    out = Path(tmp) / "launcher_precedence.sh"
    out.write_text(src.replace(_PY_SELECTOR_TUPLE, swapped, 1))
    monkeypatch.setattr(sweep, "LAUNCHERS", (out,))
    ok, _detail = sweep.pin_arm_resolver_twins()
    assert ok, "precedence is unobservable while multi-selector boots refuse"


def test_twin_detector_covers_every_arm_the_campaign_can_name():
    named = {
        value for env in sweep.TWIN_RESOLVER_ENVS for value in env.values()
    }
    assert "gqa_pair_splitk" in named, "the tier-b arm must be exercised"
    assert {"nosplit", "gqa_pair", "split2", "visibility"} <= named
    # and the no-selector case, which is where "" used to live
    assert {} in [dict(env) for env in sweep.TWIN_RESOLVER_ENVS]
    # every launcher twin is exercised, not just the canonical one
    assert len(sweep.LAUNCHERS) == 3
