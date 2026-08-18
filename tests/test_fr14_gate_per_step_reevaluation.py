"""The gate must re-evaluate PER STEP. Round 5 proved it did not.

Round 5 (output/fr14_promoab_Gp5_20260818T174541Z): 98 requests, 11 fully gated,
87 fully ungated, ZERO mixed. Reconstructed from the census independently: only
11 transitions into the gated state and 11 out, across 11 291 gated steps -- runs
of ~1026 consecutive steps. A hard latch, not a soft bias.

CAUSE (a wiring bug, not a predicate bug). The gate was fed from
`stage_fixed32_step`'s `row["delta"]`, which is EMPTY on every steady-state
decode step: staging sets `safe_end = prior` when `drafts == 31` because "the
prior exact finalization owns the committed boundary". Generated tokens reach
the drafter through `finalize_fixed32_step`, which had no gate hook. So the
gate's history was the prompt, frozen; `decide()` ran every step and re-read the
same trailing 8-gram. Per-step evaluation existed in form and was a no-op in
fact.

WHY IT MATTERED. The suffix chain is a recurrence copier whose own firing
predicate is recurrence, so copying keeps the predicate true. Per-step
re-evaluation is the brake. With the history frozen there was no brake at all,
and astropy-13236 emitted a 117 739-char block repeating one 12-gram 71 times
(ttr 0.066), ending mid-token in a truncated tool call.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from fr14_suffix_pass_gate import SuffixPassGate  # noqa: E402
import fr13_merged_drafter as md  # noqa: E402


# ---------------------------------------------------------------------------
# 1. The wiring invariant. This is the test that would have caught it in 0.5 s.
# ---------------------------------------------------------------------------

def test_the_gate_is_fed_wherever_arctic_is_fed():
    """Arctic and the gate must see the same token stream, or the gate is blind.

    `cache.add_active_response(...)` is the drafter's commit point. Every such
    call site in the fixed32 path must have a gate feed in the same function --
    the round-5 latch was exactly a commit point that fed Arctic and not the gate.
    """
    import ast

    src = (SCRIPTS / "fr13_merged_drafter.py").read_text()
    tree = ast.parse(src)
    offenders = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        body = ast.get_source_segment(src, fn) or ""
        if "add_active_response" not in body:
            continue
        if "fixed32" not in fn.name:
            continue  # the non-fixed32 path has its own hooks
        if "fr14_gate()" not in body and "_fr14.observe" not in body:
            offenders.append(fn.name)
    assert not offenders, (
        "these feed Arctic but not the gate: " + repr(offenders)
    )


def test_staging_no_longer_feeds_the_gate_a_delta():
    """It is empty in steady state and double-counts on prefill."""
    src = (SCRIPTS / "fr13_merged_drafter.py").read_text()
    stage = src[src.index("def stage_fixed32_step"):src.index("def finalize_fixed32_step")]
    assert '_fr14.observe(req_id, row["delta"])' not in stage
    assert "start_request" in stage, "the prompt is still supplied at staging"


def test_finalize_feeds_the_gate_the_committed_delta():
    src = (SCRIPTS / "fr13_merged_drafter.py").read_text()
    fin = src[src.index("def finalize_fixed32_step"):]
    fin = fin[:fin.index("\n_TAIL_WIDE_TOPK")]
    assert "_fr14.observe(req_id, extra)" in fin
    # and it sits with the other commit bookkeeping, not somewhere else
    assert fin.index("_fr14.observe(req_id, extra)") > fin.index("_COMMITTED[req_id] = (")


# ---------------------------------------------------------------------------
# 2. The behaviour: a request-scoped simulation of the serve's token flow.
# ---------------------------------------------------------------------------

def _prompt_with_a_recurring_tail(ngram=8):
    """A prompt whose LAST n-gram recurs -- so a frozen gate latches ON."""
    tail = list(range(900, 900 + ngram))
    body = [i % 400 for i in range(600)]
    return body + tail + body + tail          # tail occurs twice, ends on it


def _run(feed_generated, steps=200, seed=5):
    """Simulate one request: prompt at start, then `steps` commits."""
    import random

    rng = random.Random(seed)
    gate = SuffixPassGate(enabled=True, ngram=8, min_agree=0.75, min_history=256)
    gate.start_request("r", _prompt_with_a_recurring_tail())
    decisions = []
    for _ in range(steps):
        decisions.append(gate.decide("r").fired)
        # ~5 committed tokens per step, novel text
        extra = [rng.randrange(5000, 9000) for _ in range(5)]
        if feed_generated:
            gate.observe("r", extra)
    return decisions


def test_a_frozen_history_reproduces_the_round_5_latch():
    """This is the bug: no generated tokens reach the gate -> one decision."""
    d = _run(feed_generated=False)
    assert len(set(d)) == 1, "expected a latch"
    assert d[0] is True, "and this prompt latches it ON, as 11 requests did"


def test_feeding_the_committed_delta_restores_per_step_evaluation():
    d = _run(feed_generated=True)
    assert len(set(d)) == 2, (
        "the decision must change once the request generates novel text"
    )
    assert d[0] is True and d[-1] is False, d[:12]


def test_the_decision_tracks_the_text_not_the_request():
    """Novel text turns it off; returning to copied text turns it back on."""
    gate = SuffixPassGate(enabled=True, ngram=8, min_agree=0.75, min_history=0)
    phrase = list(range(700, 730))
    gate.start_request("r", phrase * 3)
    assert gate.decide("r").fired is True
    gate.observe("r", [91000 + i for i in range(40)])      # novel
    assert gate.decide("r").fired is False
    # back onto the phrase: repeat it so the trailing n-gram's earlier
    # occurrences agree on what follows (agreement >= 0.75)
    gate.observe("r", phrase * 3)
    assert gate.decide("r").fired is True


def test_a_copier_runaway_keeps_the_predicate_true():
    """The mechanism, stated as a test rather than as prose.

    Copying its own recent output satisfies the recurrence predicate, so the
    gate stays on. Per-step re-evaluation does not break this by itself -- which
    is why the anti-runaway brake is proposed on top of the latch fix.
    """
    gate = SuffixPassGate(enabled=True, ngram=8, min_agree=0.75, min_history=0)
    phrase = list(range(700, 712))
    gate.start_request("r", phrase * 2)
    fired = []
    for _ in range(40):
        fired.append(gate.decide("r").fired)
        gate.observe("r", phrase)          # the copier emits the same 12-gram
    assert all(fired), "copying keeps the predicate true -- the runaway"


def test_mixed_requests_are_possible_at_all():
    """Round 5's zero-mixed is only explicable by a latch; assert it is gone."""
    d = _run(feed_generated=True, steps=400)
    on = sum(d)
    assert 0 < on < len(d), f"still latched: {on}/{len(d)} gated"


# ---------------------------------------------------------------------------
# 3. The anti-runaway brake (pre-registered; see suffix_pass_gating.md §16.3).
# ---------------------------------------------------------------------------

def _copier(max_run, steps=200):
    """A request that copies the same phrase forever -- the round-5 pathology."""
    gate = SuffixPassGate(
        enabled=True, ngram=8, min_agree=0.75, min_history=0, max_run=max_run
    )
    phrase = list(range(700, 712))
    gate.start_request("r", phrase * 2)
    fired = []
    for _ in range(steps):
        d = gate.decide("r")
        gate.note_step(["r"], d.fired)
        fired.append(d.fired)
        gate.observe("r", phrase)
    return gate, fired


def test_without_the_brake_the_copier_never_stops():
    _gate, fired = _copier(max_run=10**9)
    assert all(fired), "sanity: the predicate stays true under self-copying"


def test_the_brake_forces_an_ungated_step():
    gate, fired = _copier(max_run=32)
    assert not all(fired), "the brake must interrupt the run"
    runs, cur = [], 0
    for f in fired:
        if f:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    assert max(runs) <= 32, f"run cap exceeded: {max(runs)}"
    assert gate.summary()["run_capped"] > 0


@pytest.mark.parametrize("cap", [1, 4, 32])
def test_the_cap_bounds_the_run_at_any_value(cap):
    _gate, fired = _copier(max_run=cap, steps=120)
    runs, cur = [], 0
    for f in fired:
        cur = cur + 1 if f else 0
        runs.append(cur)
    assert max(runs) <= cap


def test_the_brake_costs_at_most_one_step_in_cap_plus_one():
    """It is orthogonal to the predicate: the calibration is untouched."""
    _gate, fired = _copier(max_run=32, steps=330)
    assert sum(fired) / len(fired) >= 32 / 33 - 0.02


def test_the_cap_resets_when_the_gate_goes_cold():
    gate = SuffixPassGate(
        enabled=True, ngram=8, min_agree=0.75, min_history=0, max_run=4
    )
    phrase = list(range(700, 712))
    gate.start_request("r", phrase * 2)
    for _ in range(3):
        d = gate.decide("r")
        gate.note_step(["r"], d.fired)
        gate.observe("r", phrase)
    gate.observe("r", [90000 + i for i in range(40)])   # novel text
    d = gate.decide("r")
    gate.note_step(["r"], d.fired)
    assert d.fired is False and d.reason != "run_cap"
    assert gate._state["r"]["run"] == 0, "a cold step must reset the counter"


def test_note_step_counts_the_batch_outcome_not_the_row_vote():
    """At B>1 a row can vote to gate while the batch does not."""
    gate = SuffixPassGate(
        enabled=True, ngram=8, min_agree=0.75, min_history=0, max_run=2
    )
    phrase = list(range(700, 712))
    gate.start_request("hot", phrase * 2)
    gate.start_request("cold", list(range(20000, 20120)))   # no recurrence
    for _ in range(10):
        ds = [gate.decide(r) for r in ("hot", "cold")]
        gate.note_step(("hot", "cold"), all(d.fired for d in ds))
        gate.observe("hot", phrase)
    assert gate._state["hot"]["run"] == 0, (
        "the hot row must not accrue a run the batch never gated"
    )
