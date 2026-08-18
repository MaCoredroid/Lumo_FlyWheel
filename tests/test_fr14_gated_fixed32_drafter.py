"""The gated fixed32 drafter shape (FR14 lever 2), exercised on CPU.

`decide_fixed32(gated=True)` is the host half of the lever: it is what runs when
the gate fires and the drafter has executed 2 post-root MTP forwards instead of
4.  These tests pin the two things that must hold for a gated step to be
well-formed at the verifier:

  * the published pack is still exactly 31 columns, so the sampler's validity
    mask, its child tables and the committer graph are all untouched;
  * Arctic is asked for 8 main-chain tokens, not 6, and the two extra ones are
    the fills for head depths 4 and 5 -- the depths whose MTP passes were skipped.

They also pin that the UNGATED path is unchanged, which is the whole safety
argument for shipping this default-OFF.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

torch = pytest.importorskip("torch")

import fr13_merged_drafter as md  # noqa: E402
from fr13_fixed32_topology import (  # noqa: E402
    ARCTIC_MAIN_TAIL_LENGTH,
    GATED_ARCTIC_MAIN_TAIL_LENGTH,
    GATED_MTP_K,
    GATED_PADDED_DRAFT_IDS,
    GATED_SUFFIX_SPINE_DRAFT_IDS,
    HYDRA27_VALID,
    PHYSICAL_DRAFTS,
    validate_gate_contract,
)


class MockCache:
    """Records every speculate() call so the test can assert the Arctic ask."""

    def __init__(self, chain):
        self.chain = list(chain)
        self.calls = []

    def speculate(self, req_id, pattern, **kw):
        self.calls.append(
            {
                "req_id": req_id,
                "pattern_len": len(pattern),
                "max_spec_tokens": kw.get("max_spec_tokens"),
            }
        )
        return list(self.chain)


def _run(gated, chain, head_depths):
    md._COMMITTED["A"] = list(range(1, 25))
    cache = MockCache(chain)
    head = [[100 + d] for d in range(head_depths)]
    tail = md.decide_fixed32(
        cache,
        ["A"],
        head,
        {1: [7001], 2: [7002]},
        torch.device("cpu"),
        pad_token=9,
        vocab_size=100000,
        gated=gated,
    )
    return cache, tail, md.get_fixed32_drafter_last_work()


def test_gate_contract_is_self_consistent():
    validate_gate_contract()


def test_ungated_shape_is_unchanged():
    cache, tail, work = _run(False, [8001, 8002, 8003, 8004, 8005, 8006], 5)
    assert len(tail) == ARCTIC_MAIN_TAIL_LENGTH == 6
    assert work["gated"] is False
    assert work["head_fill_columns"] == 0
    assert work["main_tail_columns"] == 6
    assert work["arctic_requested_tokens"] == 12
    main = [c for c in cache.calls if c["max_spec_tokens"] == 6]
    assert len(main) == 1, "the main chain is still asked for 6 tokens"
    # 15 head + 6 tail + 10 rescue
    assert 5 * 3 + len(tail) + 10 == PHYSICAL_DRAFTS


def test_gated_shape_asks_arctic_for_eight_and_still_packs_31():
    chain = [8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008]
    cache, tail, work = _run(True, chain, GATED_MTP_K)
    assert len(tail) == GATED_ARCTIC_MAIN_TAIL_LENGTH == 8
    assert work["gated"] is True
    assert work["head_fill_columns"] == 2
    assert work["main_tail_columns"] == 8
    # main 8 + rank1 4 + rank2 2
    assert work["arctic_requested_tokens"] == 14
    assert work["arctic_lookup_calls"] == 3, "still exactly three Arctic calls"
    main = [c for c in cache.calls if c["max_spec_tokens"] == 8]
    assert len(main) == 1

    # the two leading columns are consumed by head depths 4 and 5, so the tail
    # contribution to the pack is 6 -- the pack width is invariant.
    assert 5 * 3 + (len(tail) - work["head_fill_columns"]) + 10 == PHYSICAL_DRAFTS


def test_gated_main_pattern_uses_only_three_mtp_tokens():
    """The Arctic walk must be seeded from the 3 MTP tokens that actually ran."""
    _, _, _ = _run(True, list(range(8001, 8009)), GATED_MTP_K)
    cache_g, _, _ = _run(True, list(range(8001, 8009)), GATED_MTP_K)
    cache_u, _, _ = _run(False, list(range(8001, 8007)), 5)
    main_g = [c for c in cache_g.calls if c["max_spec_tokens"] == 8][0]
    main_u = [c for c in cache_u.calls if c["max_spec_tokens"] == 6][0]
    # 24 committed + head_depth MTP tokens
    assert main_g["pattern_len"] == 24 + GATED_MTP_K
    assert main_u["pattern_len"] == 24 + 5
    assert main_g["pattern_len"] < main_u["pattern_len"]


def test_gated_refuses_a_five_depth_head():
    """Head width must match the passes that ran, or the seam is a lie."""
    with pytest.raises(RuntimeError, match="requires exactly 3 MTP head depths"):
        _run(True, list(range(8001, 8009)), 5)


def test_ungated_refuses_a_three_depth_head():
    with pytest.raises(RuntimeError, match="requires exactly 5 MTP head depths"):
        _run(False, list(range(8001, 8007)), 3)


def test_cold_arctic_on_a_gated_step_pads_rather_than_shortens():
    """A cold cache must still publish 8 columns -- pad, never a short pack.

    This is the well-formedness invariant: the depth-6..11 tail hangs off the
    depth-5 spine node, so a short main chain would orphan live nodes.
    """
    cache, tail, work = _run(True, [], GATED_MTP_K)
    assert len(tail) == 8
    assert all(int(col[0]) == 9 for col in tail), "cold chain must be all pad"
    assert work["main_tail_columns"] == 8


def test_padded_nodes_are_leaves_and_spine_nodes_are_not():
    """The four columns the gate stops feeding must have no descendants."""
    from fr13_fixed32_topology import DRAFT_PARENT

    children = {}
    for node, parent in enumerate(DRAFT_PARENT):
        children.setdefault(parent, []).append(node)
    for node in GATED_PADDED_DRAFT_IDS:
        assert not children.get(node), f"padding {node} would orphan its subtree"
        assert HYDRA27_VALID[node], "the gate pads FILLED nodes, it does not mask"
    for node in GATED_SUFFIX_SPINE_DRAFT_IDS:
        assert children.get(node), f"spine node {node} must keep its subtree"


# ---------------------------------------------------------------------------
# The offline work census must accept a gated step and no third shape.
# ---------------------------------------------------------------------------

def _banked_event():
    import json

    p = (
        Path(__file__).resolve().parents[1]
        / "output/fr14_b1_stock_20260817T054447Z/tail6_fixed32_b1radix"
        / "logs/fr13_fixed32_work_census.jsonl"
    )
    if not p.exists():
        pytest.skip("banked census not present")
    with p.open() as fh:
        return json.loads(fh.readline())


def test_banked_events_still_validate_unchanged():
    import fr13_fixed32_work_census as census

    census.validate_event(_banked_event(), source="banked")


def test_a_gated_event_validates():
    import fr13_fixed32_work_census as census

    ev = _banked_event()
    b = ev["batch_size"]
    ev["drafter"]["mtp_forward_calls"] = 2
    ev["drafter"]["mtp_forward_rows"] = 2 * b
    ev["drafter"]["main_tail_length"] = 8
    ev["drafter"]["arctic_requested_tokens"] = 14 * b
    rt = ev["drafter_runtime"]
    rt["mtp_forward_calls"] = 2
    rt["mtp_forward_rows"] = 2 * b
    rt["arctic_requested_tokens"] = 14 * b
    rt["merge_fill_columns"] = 18
    rt["merge_fill_rows"] = 18 * b
    rt["arctic_ledger"] = [
        dict(row, tokens=8) if row["kind"] == "main" else row
        for row in rt["arctic_ledger"]
    ]
    census.validate_event(ev, source="gated")


@pytest.mark.parametrize(
    "calls,tail",
    [(2, 6), (4, 8), (3, 7), (2, 7), (0, 6), (5, 6)],
)
def test_no_other_drafter_shape_validates(calls, tail):
    import fr13_fixed32_work_census as census

    ev = _banked_event()
    b = ev["batch_size"]
    ev["drafter"]["mtp_forward_calls"] = calls
    ev["drafter"]["mtp_forward_rows"] = calls * b
    ev["drafter"]["main_tail_length"] = tail
    with pytest.raises(census.CensusError):
        census.validate_event(ev, source="bad")


def test_gated_arctic_token_count_is_checked_too():
    import fr13_fixed32_work_census as census

    ev = _banked_event()
    b = ev["batch_size"]
    ev["drafter"]["mtp_forward_calls"] = 2
    ev["drafter"]["mtp_forward_rows"] = 2 * b
    ev["drafter"]["main_tail_length"] = 8
    ev["drafter"]["arctic_requested_tokens"] = 12 * b  # stale ungated width
    with pytest.raises(census.CensusError):
        census.validate_event(ev, source="bad")
