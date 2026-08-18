"""The split-graph manifest state machine (FR14_GATE_SPLIT_GRAPH), driven directly.

The drafter graph machinery lives inside `_FR13_FIXED32_OBSERVED_RUNTIME_SOURCE`,
a source blob the patcher injects into `gdn_linear_attn`. That blob is real
Python, so it can be exec'd into a namespace and driven on CPU -- which is the
only executable check available for this change on a host without CUDA torch.

What these tests pin:

  * the UNGATED path is byte-identical -- one 4-pass graph, manifest schema v2,
    literals 4/4, the same sha256 the shipped serve carries;
  * a split capture produces two 2-pass graphs that coexist for one batch size;
  * an ungated split step replays both (4 forwards, 2 replays) and a gated step
    replays only `lo` (2 forwards, 1 replay);
  * a half-graph can NEVER present as the shipped 4-pass one (different schema,
    therefore different signature);
  * the three legal per-step shapes are the only ones proposal_end accepts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

torch = pytest.importorskip("torch")

import fr10_phase4_patch_vllm_tree_gdn as patcher  # noqa: E402

MODE = "hydra27_fixed32"
TREE_LAYER = "mtp.layers.0.self_attn.attn"  # the shipped value


def new_runtime():
    """Exec the injected blob into a fresh namespace."""
    ns = {
        "_FR13_FIXED32_MODE": MODE,
        "_FR13_FIXED32_PRESEED_CAP": 1,
        "_FR13_FIXED32_VALID_MASK": 0x7ABDFFFF,
        "torch": torch,
    }
    exec(patcher._FR13_FIXED32_OBSERVED_RUNTIME_SOURCE, ns)
    ns["_FR13_FIXED32_DRAFTER_TREE_LAYER"] = TREE_LAYER
    ns["_FR13_FIXED32_CENSUS_EVENTS"] = []
    ns["_FR13_FIXED32_COMPLETE_EVENTS"] = 0
    return ns


def begin_proposal(ns, batch=1):
    ns["_fr13_fixed32_drafter_proposal_begin"](MODE, tuple(
        f"r{i}" for i in range(batch)
    ), batch, batch, batch)
    return ns["_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT"]


def capture(ns, graph_id, batch, passes, segment=0):
    """Record one graph of `passes` post-root MTP forwards."""
    ns["_fr13_fixed32_drafter_graph_capture_begin"](
        graph_id, batch, passes, segment
    )
    for _ in range(passes):
        # Drive the REAL tree-attention observer, exactly as tree_attn.py does.
        # Hand-incrementing these counters is what let the Arm G defect through:
        # the harness simulated the observer instead of executing it, so the
        # observer's own per-step assumptions were never under test.
        ns["_fr13_fixed32_observed_tree_attn"](TREE_LAYER, batch, (1, 1), True)
        ns["_fr13_fixed32_drafter_mtp_forward"](batch, True)
    return ns["_fr13_fixed32_drafter_graph_capture_end"](
        graph_id, batch, passes, segment
    )


# ---------------------------------------------------------------------------
# The ungated path must not move at all.
# ---------------------------------------------------------------------------

def test_ungated_capture_is_unchanged_schema_and_literals():
    ns = new_runtime()
    begin_proposal(ns)
    sig = capture(ns, 4242, 1, 4)

    _stored_sig, canonical = ns["_FR13_FIXED32_DRAFTER_GRAPH_MANIFESTS"][4242]
    import json

    manifest = json.loads(canonical)
    assert manifest["schema"] == "fr13-fixed32-drafter-graph-manifest-v2"
    assert manifest["mtp_forward_calls"] == 4
    assert manifest["mtp_forward_rows"] == 4
    assert manifest["tree_attn_calls"] == 4
    assert "split_passes" not in manifest
    assert "split_segment" not in manifest
    # THE regression proof: this is the exact drafter graph signature the
    # shipped K0 serve carries in its census
    # (output/fr14_b1_stock_20260817T054447Z/.../fr13_fixed32_work_census.jsonl,
    # drafter_runtime.graph_signature). The credential re-issue this change was
    # sanctioned to make therefore did NOT move the ungated arm's credential.
    assert sig == (
        "d9a4ddece41d146e9949b9f8ff7c2603b8948d157b28ef69244e44469b36150c"
    )


def test_ungated_default_argument_keeps_every_existing_call_site_working():
    """Existing call sites omit `passes`; they must still mean four."""
    ns = new_runtime()
    begin_proposal(ns)
    ns["_fr13_fixed32_drafter_graph_capture_begin"](77, 1)
    ctx = ns["_FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT"]
    assert ctx["passes"] == 4
    for _ in range(4):
        ctx["tree_attn_calls"] += 1
        ctx["tree_attn_rows"] += 1
        ctx["tree_attn_layer"] = TREE_LAYER
        ctx["tree_attn_bias_shape"] = (1, 1)
        ns["_fr13_fixed32_drafter_mtp_forward"](1, True)
    sig = ns["_fr13_fixed32_drafter_graph_capture_end"](77, 1)
    assert (1, 4, 0) in ns["_FR13_FIXED32_DRAFTER_GRAPH_BY_BATCH"]
    ns["_fr13_fixed32_drafter_graph_replay"](77, sig, 1)
    proposal = ns["_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT"]
    assert proposal["mtp_forward_calls"] == 4
    assert proposal["graph_replays"] == 1


def test_ungated_capture_end_rejects_a_wrong_forward_count():
    ns = new_runtime()
    begin_proposal(ns)
    ns["_fr13_fixed32_drafter_graph_capture_begin"](9, 1, 4)
    ctx = ns["_FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT"]
    for _ in range(3):  # one short
        ctx["tree_attn_calls"] += 1
        ctx["tree_attn_rows"] += 1
        ctx["tree_attn_layer"] = TREE_LAYER
        ctx["tree_attn_bias_shape"] = (1, 1)
        ns["_fr13_fixed32_drafter_mtp_forward"](1, True)
    with pytest.raises(RuntimeError, match="capture work drift"):
        ns["_fr13_fixed32_drafter_graph_capture_end"](9, 1, 4)


# ---------------------------------------------------------------------------
# The split.
# ---------------------------------------------------------------------------

def test_two_half_graphs_coexist_for_one_batch_size():
    ns = new_runtime()
    begin_proposal(ns)
    sig_lo = capture(ns, 101, 1, 2)
    sig_hi = capture(ns, 102, 1, 2, 1)
    by_batch = ns["_FR13_FIXED32_DRAFTER_GRAPH_BY_BATCH"]
    # keyed by (batch, passes, segment): the halves are distinct artifacts
    assert by_batch[(1, 2, 0)] == 101
    assert by_batch[(1, 2, 1)] == 102
    assert sig_lo != sig_hi, "lo and hi must not share a signature"
    assert ns["_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT"]["graph_captures"] == 2


def test_half_graph_can_never_present_as_the_shipped_four_pass_graph():
    ns = new_runtime()
    begin_proposal(ns)
    capture(ns, 101, 1, 2)
    import json

    _sig, canonical = ns["_FR13_FIXED32_DRAFTER_GRAPH_MANIFESTS"][101]
    manifest = json.loads(canonical)
    assert manifest["schema"] == "fr13-fixed32-drafter-graph-manifest-v3-split"
    assert manifest["split_passes"] == 2
    assert manifest["mtp_forward_calls"] == 2


def test_ungated_split_step_replays_both_halves():
    ns = new_runtime()
    proposal = begin_proposal(ns)
    sig_lo = capture(ns, 101, 1, 2)
    sig_hi = capture(ns, 102, 1, 2, 1)
    # a real step: fresh proposal, then two replays
    ns["_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT"] = None
    proposal = begin_proposal(ns)
    ns["_fr13_fixed32_drafter_graph_replay"](101, sig_lo, 1, 2)
    assert proposal["mtp_forward_calls"] == 2
    assert proposal["graph_replays"] == 1
    ns["_fr13_fixed32_drafter_graph_replay"](102, sig_hi, 1, 2, 1)
    assert proposal["mtp_forward_calls"] == 4
    assert proposal["mtp_forward_rows"] == 4
    assert proposal["graph_replays"] == 2


def test_gated_step_replays_lo_alone():
    ns = new_runtime()
    capture_proposal = begin_proposal(ns)
    sig_lo = capture(ns, 101, 1, 2)
    capture(ns, 102, 1, 2, 1)
    assert capture_proposal is not None
    ns["_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT"] = None
    proposal = begin_proposal(ns)
    ns["_fr13_fixed32_drafter_graph_replay"](101, sig_lo, 1, 2)
    assert proposal["mtp_forward_calls"] == 2
    assert proposal["graph_replays"] == 1


def test_replaying_the_same_half_twice_is_refused():
    """Accumulation must not let `lo` stand in for `hi`."""
    ns = new_runtime()
    begin_proposal(ns)
    sig_lo = capture(ns, 101, 1, 2)
    capture(ns, 102, 1, 2, 1)
    ns["_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT"] = None
    begin_proposal(ns)
    ns["_fr13_fixed32_drafter_graph_replay"](101, sig_lo, 1, 2)
    with pytest.raises(RuntimeError, match="replay drift"):
        ns["_fr13_fixed32_drafter_graph_replay"](101, sig_lo, 1, 2)


def test_a_third_replay_is_refused():
    ns = new_runtime()
    begin_proposal(ns)
    sig_lo = capture(ns, 101, 1, 2)
    sig_hi = capture(ns, 102, 1, 2, 1)
    ns["_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT"] = None
    begin_proposal(ns)
    ns["_fr13_fixed32_drafter_graph_replay"](101, sig_lo, 1, 2)
    ns["_fr13_fixed32_drafter_graph_replay"](102, sig_hi, 1, 2, 1)
    with pytest.raises(RuntimeError, match="replay drift"):
        ns["_fr13_fixed32_drafter_graph_replay"](101, sig_lo, 1, 2)


def test_four_pass_graph_may_never_be_replayed_twice():
    ns = new_runtime()
    begin_proposal(ns)
    sig = capture(ns, 55, 1, 4)
    ns["_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT"] = None
    begin_proposal(ns)
    ns["_fr13_fixed32_drafter_graph_replay"](55, sig, 1, 4)
    with pytest.raises(RuntimeError, match="replay drift"):
        ns["_fr13_fixed32_drafter_graph_replay"](55, sig, 1, 4)


def test_pass_count_must_be_two_or_four():
    ns = new_runtime()
    begin_proposal(ns)
    for bad in (0, 1, 3, 5, 8):
        with pytest.raises(RuntimeError, match="shape must be 4x1 or 2x2"):
            ns["_fr13_fixed32_drafter_graph_capture_begin"](900 + bad, 1, bad)
    # a 4-pass graph is never a half, so it can only ever be segment 0
    with pytest.raises(RuntimeError, match="shape must be 4x1 or 2x2"):
        ns["_fr13_fixed32_drafter_graph_capture_begin"](950, 1, 4, 1)


def test_registry_reports_the_pass_count():
    ns = new_runtime()
    begin_proposal(ns)
    sig_lo = capture(ns, 101, 1, 2)
    sig_hi = capture(ns, 102, 1, 2, 1)
    ns["_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT"] = None
    begin_proposal(ns)
    ns["_fr13_fixed32_drafter_graph_replay"](101, sig_lo, 1, 2)
    ns["_fr13_fixed32_drafter_graph_replay"](102, sig_hi, 1, 2, 1)
    rows = ns["_fr13_fixed32_drafter_graph_registry"]()
    assert len(rows) == 2, "both halves are registered"
    assert [r["segment"] for r in rows] == [0, 1]
    assert all(r["passes"] == 2 and r["batch_size"] == 1 for r in rows)


# ---------------------------------------------------------------------------
# proposal_end: the three legal per-step shapes, and the handoff interlock.
# ---------------------------------------------------------------------------

def _finish(ns, calls, replays, main_tail, measured=False, sync_evidence=True):
    """Drive proposal_end.

    `measured=True` is the half that matters: it executes the CENSUS emitter AND
    the RUNTIME-EVIDENCE emitter. The harness previously only ever ran
    measured=False, which returns before both -- so the runtime half's
    "graph_replays: 1 / mtp_forward_calls: 4" literals were never executed by a
    test, and shipped stale into the round-2 boot. Same mock-to-real lesson as
    the 11th site: the half you do not execute is the half that breaks.
    """
    proposal = ns["_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT"]
    proposal["mtp_execution_basis"] = "cudagraph_replay"
    proposal["mtp_forward_calls"] = calls
    proposal["mtp_forward_rows"] = calls * 1
    proposal["graph_replays"] = replays
    proposal["graph_captures"] = 0
    proposal["graph_id"] = 1
    proposal["graph_signature"] = "s" * 64
    proposal["arctic"] = {
        "main_tail_columns": main_tail,
        "main_lookup_calls": 1,
        "main_lookup_tokens": main_tail,
        "rank1_lookup_calls": 1,
        "rank1_lookup_tokens": 4,
        "rank2_lookup_calls": 1,
        "rank2_lookup_tokens": 2,
        "arctic_lookup_calls": 3,
        "arctic_requested_tokens": main_tail + 6,
        "merge_fill_calls": 1,
        "merge_fill_columns": main_tail + 10,
        "merge_fill_rows": main_tail + 10,
        "rescue_carry_slots": 4,
    }
    proposal["publish"] = {
        "publish_shape": (1, 31),
        "physical_parent_sha256": "p" * 64,
    }
    proposal["measured"] = measured
    if measured and sync_evidence:
        ev = proposal["replay_evidence"]
        ev["matching_replays"] = replays
        ev["graph_captures"] = proposal["graph_captures"]
    try:
        return ns["_fr13_fixed32_drafter_proposal_end"](
            MODE, ("r0",), (1, 31), "torch.int64", "cuda", True
        )
    except RuntimeError as exc:
        # Sealing the event needs a TAW record -- the committer's collaborator,
        # not the drafter's, and lane 3 owns it. Both drafter emitters have
        # already written into observed_work by then, which is what these tests
        # assert. Only the TAW stage is tolerated: any DRAFTER-stage failure
        # still propagates, so this cannot hide the class of bug it exists for.
        if "TAW" not in str(exc):
            raise
        return None


def begin_measured(ns, batch=1):
    """A proposal bound to a pending measured event, so the emitters run."""
    req_ids = tuple(f"r{i}" for i in range(batch))
    observed = {
        "drafter": None,
        "request_ids": req_ids,
        "mode": MODE,
        "batch_size": batch,
        "forward_step_index": 0,
    }
    ns["_FR13_FIXED32_PENDING_EVENT"] = {
        "mode": MODE,
        "batch_size": batch,
        "request_ids": req_ids,
        "forward_step_index": 0,
        "target_kv_complete": True,
        "event_index": 0,
        "observed_work": observed,
    }
    ns["_fr13_fixed32_drafter_proposal_begin"](
        MODE, req_ids, batch, batch, batch
    )
    # the drafter's KV completes AFTER the proposal begins -- proposal_begin
    # refuses a proposal that starts with the split KV lifecycle already closed
    ns["_FR13_FIXED32_PENDING_EVENT"]["drafter_kv_complete"] = True
    ns["_FR13_FIXED32_PENDING_EVENT"]["kv_complete"] = True
    # hold the emitted dict directly: completion clears the pending event
    ns["_TEST_OBSERVED"] = observed
    return ns["_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT"]


@pytest.mark.parametrize(
    "calls,replays,main_tail",
    [(4, 1, 6), (4, 2, 6), (2, 1, 8)],
)
def test_the_three_legal_step_shapes_are_accepted(calls, replays, main_tail):
    ns = new_runtime()
    begin_proposal(ns)
    _finish(ns, calls, replays, main_tail)


@pytest.mark.parametrize(
    "calls,replays,main_tail",
    [
        (2, 1, 6),   # gated pass count with an ungated handoff -> malformed
        (4, 1, 8),   # ungated pass count with a gated handoff  -> malformed
        (4, 2, 8),
        (3, 1, 6),   # a pass count that cannot happen at all
        (2, 2, 8),
        (4, 3, 6),
        (0, 0, 6),
    ],
)
def test_every_other_shape_is_fatal(calls, replays, main_tail):
    ns = new_runtime()
    begin_proposal(ns)
    with pytest.raises(RuntimeError, match="proposal work drift"):
        _finish(ns, calls, replays, main_tail)


# ---------------------------------------------------------------------------
# The tree-attention observer -- the 11th integration site, and the one that
# refused Arm G (runroot output/fr14_promoab_Giso_20260818T074147Z).
#
# The harness reaches it now because `capture()` drives the REAL observer
# instead of hand-incrementing its counters. Simulating it is precisely why the
# defect shipped: the observer's own per-step assumptions were never under test.
# ---------------------------------------------------------------------------

def test_observer_accepts_each_segment_of_a_split_capture():
    """This is the exact sequence Arm G died on."""
    ns = new_runtime()
    begin_proposal(ns)
    capture(ns, 101, 1, 2, 0)
    capture(ns, 102, 1, 2, 1)  # <- raised "tree-attention work drift" before
    assert ns["_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT"]["graph_captures"] == 2


def test_observer_still_requires_exactly_one_capture_when_ungated():
    """segment 0 => graph_captures must be 1, byte-identical to the old literal."""
    ns = new_runtime()
    proposal = begin_proposal(ns)
    ns["_fr13_fixed32_drafter_graph_capture_begin"](55, 1, 4, 0)
    proposal["graph_captures"] = 2  # forge a second capture on a 4-pass graph
    with pytest.raises(RuntimeError, match="tree-attention work drift"):
        ns["_fr13_fixed32_observed_tree_attn"](TREE_LAYER, 1, (1, 1), True)


def test_observer_refuses_hi_without_lo():
    """A `hi` capture that is not preceded by `lo` is unrepresentable."""
    ns = new_runtime()
    begin_proposal(ns)
    # jump straight to segment 1: graph_captures becomes 1, but segment says 2
    ns["_fr13_fixed32_drafter_graph_capture_begin"](102, 1, 2, 1)
    with pytest.raises(RuntimeError, match="tree-attention work drift"):
        ns["_fr13_fixed32_observed_tree_attn"](TREE_LAYER, 1, (1, 1), True)


def test_observer_bounds_forwards_by_this_segments_pass_count():
    """A 2-pass segment may not receive a third forward."""
    ns = new_runtime()
    begin_proposal(ns)
    ns["_fr13_fixed32_drafter_graph_capture_begin"](101, 1, 2, 0)
    for _ in range(2):
        ns["_fr13_fixed32_observed_tree_attn"](TREE_LAYER, 1, (1, 1), True)
        ns["_fr13_fixed32_drafter_mtp_forward"](1, True)
    with pytest.raises(RuntimeError, match="tree-attention work drift"):
        ns["_fr13_fixed32_observed_tree_attn"](TREE_LAYER, 1, (1, 1), True)


def test_observer_still_bounds_a_four_pass_graph_at_four():
    ns = new_runtime()
    begin_proposal(ns)
    ns["_fr13_fixed32_drafter_graph_capture_begin"](55, 1, 4, 0)
    for _ in range(4):
        ns["_fr13_fixed32_observed_tree_attn"](TREE_LAYER, 1, (1, 1), True)
        ns["_fr13_fixed32_drafter_mtp_forward"](1, True)
    with pytest.raises(RuntimeError, match="tree-attention work drift"):
        ns["_fr13_fixed32_observed_tree_attn"](TREE_LAYER, 1, (1, 1), True)


def test_observer_still_refuses_a_foreign_layer_and_shape():
    """The parts of the contract the split did not touch are untouched."""
    ns = new_runtime()
    begin_proposal(ns)
    ns["_fr13_fixed32_drafter_graph_capture_begin"](101, 1, 2, 0)
    with pytest.raises(RuntimeError, match="tree-attention work drift"):
        ns["_fr13_fixed32_observed_tree_attn"]("some.other.attn", 1, (1, 1), True)
    with pytest.raises(RuntimeError, match="tree-attention work drift"):
        ns["_fr13_fixed32_observed_tree_attn"](TREE_LAYER, 1, (32, 32), True)


# ---------------------------------------------------------------------------
# The 12th site: proposal_end's RUNTIME-EVIDENCE half.
#
# The census half was made pass-aware when the split landed; this half, ~20
# lines later, still hardcoded graph_replays:1 / mtp_forward_calls:4 and demanded
# matching_replays==1. An armed UNGATED step is 4 forwards over 2 replays, and
# every early step is ungated (min_history=256), so it refused immediately.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("calls,replays,tail", [(4, 1, 6), (4, 2, 6), (2, 1, 8)])
def test_measured_emitter_reports_what_the_step_actually_did(calls, replays, tail):
    ns = new_runtime()
    begin_measured(ns)
    _finish(ns, calls, replays, tail, measured=True)
    rt = ns["_TEST_OBSERVED"]["drafter_runtime"]
    assert rt["graph_replays"] == replays
    assert rt["mtp_forward_calls"] == calls
    assert rt["mtp_forward_rows"] == calls * 1
    drafter = ns["_TEST_OBSERVED"]["drafter"]
    assert drafter["mtp_forward_calls"] == calls
    assert drafter["main_tail_length"] == tail


def test_measured_emitter_requires_evidence_to_match_the_replay_count():
    ns = new_runtime()
    proposal = begin_measured(ns)
    proposal["replay_evidence"]["matching_replays"] = 1  # stale single-graph value
    with pytest.raises(RuntimeError, match="proposal evidence drifted"):
        _finish(ns, 4, 2, 6, measured=True, sync_evidence=False)


def test_emitted_event_round_trips_through_the_census():
    """Emitter and validator are a PAIR: what one writes, the other must accept.

    This is the test shape that makes a one-sided update impossible to ship --
    it fails if either half moves without the other.
    """
    census = pytest.importorskip("fr13_fixed32_work_census")
    banked = _banked_event_or_skip()
    for calls, replays, tail in ((4, 1, 6), (4, 2, 6), (2, 1, 8)):
        ns = new_runtime()
        begin_measured(ns)
        _finish(ns, calls, replays, tail, measured=True)
        emitted = ns["_TEST_OBSERVED"]
        ev = json.loads(json.dumps(banked))
        for key in ("mtp_forward_calls", "mtp_forward_rows", "main_tail_length",
                    "arctic_requested_tokens"):
            ev["drafter"][key] = emitted["drafter"][key]
        for key in ("mtp_forward_calls", "mtp_forward_rows", "graph_replays",
                    "graph_captures", "arctic_requested_tokens",
                    "merge_fill_columns", "merge_fill_rows"):
            ev["drafter_runtime"][key] = emitted["drafter_runtime"][key]
        ev["drafter_runtime"]["arctic_ledger"] = [
            dict(row, tokens=tail) if row["kind"] == "main" else row
            for row in ev["drafter_runtime"]["arctic_ledger"]
        ]
        census.validate_event(ev, source=f"emitted-{calls}x{replays}")


def _banked_event_or_skip():
    path = (
        REPO
        / "output/fr14_b1_stock_20260817T054447Z/tail6_fixed32_b1radix"
        / "logs/fr13_fixed32_work_census.jsonl"
    )
    if not path.exists():
        pytest.skip("banked census fixture not present")
    census = pytest.importorskip("fr13_fixed32_work_census")
    with path.open() as fh:
        ev = json.loads(fh.readline())
    try:
        census.validate_event(ev, source="fixture")
    except census.CensusError as exc:
        if ".drafter" not in str(exc):
            pytest.skip(f"banked fixture stale outside this lane: {exc}")
        raise
    return ev
