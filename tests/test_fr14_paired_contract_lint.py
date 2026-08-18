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
            "twin/twin", "bash/python"} <= kinds


def test_inventory_size_is_recorded():
    """Adding a pair without updating the count is itself a one-sided update."""
    assert len(sweep.PAIRS) == 12, (
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
