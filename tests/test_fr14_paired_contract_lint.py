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
    assert len(sweep.PAIRS) == 9, (
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
