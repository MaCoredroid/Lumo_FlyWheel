"""Tests for the FR14 suffix-aware MTP pass gate (lever 2).

The load-bearing test is `test_online_matches_offline_predicate`: it proves the
gate that ships makes bit-identical decisions to the predicate that was
calibrated offline in `scripts/fr14_suffix_gate_calibration.py`.  Without that,
the measured warm rate and the measured q1_gated describe a different function
from the one in the serving path.
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from fr14_suffix_pass_gate import (  # noqa: E402
    DEFAULT_MIN_AGREE,
    DEFAULT_NGRAM,
    SuffixPassGate,
    gate_from_env,
)
from fr14_suffix_gate_calibration import SuffixIndex  # noqa: E402


def _synthetic_stream(n=6000, seed=7):
    """A stream with real recurrence: repeated phrases inside random filler."""
    rng = random.Random(seed)
    phrases = [[rng.randrange(1000) for _ in range(rng.randrange(9, 25))]
               for _ in range(24)]
    out = []
    while len(out) < n:
        if rng.random() < 0.45:
            out.extend(rng.choice(phrases))
        else:
            out.extend(rng.randrange(1000) for _ in range(rng.randrange(1, 6)))
    return out[:n]


def test_online_matches_offline_predicate():
    """The shipped gate and the calibrated offline predicate are one function."""
    tokens = _synthetic_stream()
    offline = SuffixIndex(tokens)
    gate = SuffixPassGate(
        enabled=True, ngram=DEFAULT_NGRAM, min_agree=DEFAULT_MIN_AGREE,
        min_history=0,
    )
    gate.start_request("r", [])

    compared = 0
    fired = 0
    for j in range(len(tokens)):
        if j >= DEFAULT_NGRAM:
            seen, agree = offline.stats_at(tokens[max(0, j - 32): j], j, DEFAULT_NGRAM)
            want = bool(seen and agree >= DEFAULT_MIN_AGREE)
            got = gate.decide("r")
            assert got.fired == want, (
                f"divergence at j={j}: offline={want} "
                f"(seen={seen}, agree={agree:.4f}) online={got!r}"
            )
            if seen:
                assert got.agreement == pytest.approx(agree), f"agreement at j={j}"
            compared += 1
            fired += int(want)
        gate.observe("r", [tokens[j]])

    assert compared > 5000, "test stream too short to be meaningful"
    # a stream with recurrence must exercise BOTH outcomes, or the test is vacuous
    assert 0 < fired < compared, f"degenerate predicate: fired {fired}/{compared}"


def test_disabled_gate_never_fires_and_keeps_no_state():
    gate = SuffixPassGate(enabled=False, min_history=0)
    gate.start_request("r", list(range(500)))
    gate.observe("r", list(range(500)))
    assert gate.active_requests() == set()
    for _ in range(50):
        assert gate.decide("r").fired is False
    # a disabled gate must not even count steps -- it is not in the path
    assert gate.stats["steps"] == 0


def test_fails_closed_on_unknown_request_and_short_history():
    gate = SuffixPassGate(enabled=True, min_history=256)
    assert gate.decide("never-started").reason == "unknown_request"
    gate.start_request("r", list(range(100)))
    d = gate.decide("r")
    assert d.fired is False and d.reason == "short_history"


def test_fails_closed_on_corrupt_state():
    gate = SuffixPassGate(enabled=True, min_history=0)
    gate.start_request("r", [1, 2, 3] * 40)
    gate._state["r"]["index"] = None  # corrupt it
    d = gate.decide("r")
    assert d.fired is False and d.reason.startswith("error:")
    assert gate.stats["errors"] == 1


def test_low_agreement_blocks_even_on_a_match():
    # one 8-gram with maximally split continuations -> agreement 0.5
    ctx = [9] * 8
    tokens = []
    for k in range(20):
        tokens.extend(ctx + [100 + (k % 2)])
    gate = SuffixPassGate(enabled=True, ngram=8, min_agree=0.75, min_history=0)
    gate.start_request("r", tokens + ctx)
    d = gate.decide("r")
    assert d.match is True
    assert d.fired is False and d.reason == "low_agreement"
    assert d.agreement < 0.75

    loose = SuffixPassGate(enabled=True, ngram=8, min_agree=0.4, min_history=0)
    loose.start_request("r", tokens + ctx)
    assert loose.decide("r").fired is True


def test_step_shape_is_the_topology_contract():
    # gated: MTP over head depths 0..2, Arctic main chain of 8 (positions 3..10)
    assert SuffixPassGate.step_shape(True) == (3, 8, 2)
    # ungated: today's shape -- MTP 0..4, Arctic chain of 6 (positions 5..10)
    assert SuffixPassGate.step_shape(False) == (5, 6, 4)
    # both reach the same maximum draft position
    for fired in (True, False):
        mtp_k, tail, _ = SuffixPassGate.step_shape(fired)
        assert mtp_k + tail == 11


def test_stop_request_releases_state():
    gate = SuffixPassGate(enabled=True, min_history=0)
    gate.start_request("r", list(range(300)))
    assert "r" in gate.active_requests()
    gate.stop_request("r")
    assert gate.active_requests() == set()
    assert gate.decide("r").reason == "unknown_request"


def test_env_defaults_off_and_rejects_garbage(monkeypatch):
    monkeypatch.delenv("FR14_SUFFIX_PASS_GATE", raising=False)
    assert gate_from_env().enabled is False

    monkeypatch.setenv("FR14_SUFFIX_PASS_GATE", "1")
    g = gate_from_env()
    assert g.enabled is True
    assert g.ngram == DEFAULT_NGRAM
    assert g.min_agree == DEFAULT_MIN_AGREE

    # a typo must be fatal, not a silent arm/disarm of an acceptance lever
    for bad in ("true", "TRUE", "yes", "", " ", "2", "01"):
        monkeypatch.setenv("FR14_SUFFIX_PASS_GATE", bad)
        with pytest.raises(ValueError):
            gate_from_env()


def test_env_overrides_are_validated(monkeypatch):
    monkeypatch.setenv("FR14_SUFFIX_PASS_GATE", "1")
    monkeypatch.setenv("FR14_SUFFIX_PASS_GATE_NGRAM", "12")
    monkeypatch.setenv("FR14_SUFFIX_PASS_GATE_MIN_AGREE", "0.9")
    g = gate_from_env()
    assert g.ngram == 12 and g.min_agree == 0.9

    monkeypatch.setenv("FR14_SUFFIX_PASS_GATE_NGRAM", "eight")
    with pytest.raises(ValueError):
        gate_from_env()


def test_summary_reports_warm_rate():
    tokens = _synthetic_stream(n=3000, seed=3)
    gate = SuffixPassGate(enabled=True, min_history=0)
    gate.start_request("r", [])
    for tok in tokens:
        gate.decide("r")
        gate.observe("r", [tok])
    s = gate.summary()
    assert s["steps"] == len(tokens)
    assert 0.0 < s["warm_rate"] < 1.0
    assert s["fired"] + s["no_match"] + s["low_agreement"] + s["short_history"] == s["steps"]


def test_indexing_is_incremental_and_order_independent():
    """observe() in one batch or token-by-token must build the same index."""
    tokens = _synthetic_stream(n=2000, seed=11)
    a = SuffixPassGate(enabled=True, min_history=0)
    a.start_request("r", tokens)
    b = SuffixPassGate(enabled=True, min_history=0)
    b.start_request("r", [])
    for tok in tokens:
        b.observe("r", [tok])
    da, db = a.decide("r"), b.decide("r")
    assert (da.fired, da.agreement, da.occurrences) == (
        db.fired, db.agreement, db.occurrences
    )
    assert a._state["r"]["index"] == b._state["r"]["index"]


# ---------------------------------------------------------------------------
# Launcher wiring: the sidecar is how the value reaches the worker at all.
# ---------------------------------------------------------------------------

LAUNCHERS = (
    "scripts/fr13_launch_forked_fa2_tree_server.sh",
    "scripts/fr14_armb_leg3_launch_nomiddleware.sh",
)


@pytest.fixture(params=LAUNCHERS)
def launcher(request):
    return (Path(__file__).resolve().parents[1] / request.param).read_text()


def test_launcher_validates_the_flag_strictly(launcher):
    assert 'case "${FR14_SUFFIX_PASS_GATE:-0}" in' in launcher
    assert 'echo "FR14_SUFFIX_PASS_GATE must be 0 or 1" >&2; exit 2' in launcher


def test_launcher_writes_the_value_carrying_sidecar(launcher):
    """The worker drops bare FR14_* masters, so an -e alone would never arrive."""
    assert '"$LOG_DIR/fr14_suffix_pass_gate.cfg"' in launcher
    assert (
        '${FR14_SUFFIX_PASS_GATE_NGRAM:-8} ${FR14_SUFFIX_PASS_GATE_MIN_AGREE:-0.75}'
        in launcher
    )
    # and removes it when disarmed, so a stale file cannot arm a later serve
    assert 'rm -f "$LOG_DIR/fr14_suffix_pass_gate.cfg"' in launcher


def test_launcher_refuses_to_arm_without_the_seam_it_hands_off_to(launcher):
    assert (
        "FR14_SUFFIX_PASS_GATE=1 requires FR13_TAIL_MODE=1 and "
        "FR13_DRAFT_SOURCE=merged" in launcher
    )
    assert "FR14_SUFFIX_PASS_GATE=1 is incompatible with FR13_TAIL_BRANCHES" in launcher


def test_launcher_forwards_env_for_attestation(launcher):
    assert '-e FR14_SUFFIX_PASS_GATE="${FR14_SUFFIX_PASS_GATE:-0}"' in launcher
    assert '-e FR14_SUFFIX_PASS_GATE_NGRAM="${FR14_SUFFIX_PASS_GATE_NGRAM:-8}"' in launcher


def test_sidecar_absent_means_off():
    from fr14_suffix_pass_gate import gate_from_sidecar

    tmp = Path(tempfile.mkdtemp())
    assert gate_from_sidecar(str(tmp / "nope.cfg")).enabled is False


def test_sidecar_round_trips_the_launcher_format():
    from fr14_suffix_pass_gate import gate_from_sidecar

    cfg = Path(tempfile.mkdtemp()) / "fr14_suffix_pass_gate.cfg"
    cfg.write_text("8 0.75 256\n")
    g = gate_from_sidecar(str(cfg))
    assert (g.enabled, g.ngram, g.min_agree, g.min_history) == (True, 8, 0.75, 256)


def test_malformed_sidecar_is_fatal_not_defaulted():
    from fr14_suffix_pass_gate import gate_from_sidecar

    cfg = Path(tempfile.mkdtemp()) / "fr14_suffix_pass_gate.cfg"
    for bad in ("8 0.75", "8 0.75 256 9", "", "8 2.0 256", "0 0.75 256", "65 0.5 1"):
        cfg.write_text(bad + "\n")
        with pytest.raises(ValueError):
            gate_from_sidecar(str(cfg))


def test_launcher_interlocks_against_a_half_integrated_lever(launcher):
    """Arming must be impossible until the drafter split-graph half exists.

    Without this, FR14_SUFFIX_PASS_GATE=1 would hand decide_fixed32 a 3-depth MTP
    head while the drafter still ran four post-root forwards: a malformed tree.
    The interlock greps the patcher for the sentinel the split will carry, so it
    clears itself the moment that lands rather than needing a second edit.
    """
    assert 'grep -q "FR14_GATE_SPLIT_GRAPH"' in launcher
    assert "the drafter split-graph half is NOT landed" in launcher


def test_interlock_is_currently_closed():
    """Sanity: the sentinel is genuinely absent, so the guard is live, not vacuous."""
    patcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "fr10_phase4_patch_vllm_tree_gdn.py"
    ).read_text()
    if "FR14_GATE_SPLIT_GRAPH" in patcher:
        pytest.skip("split graph has landed; the interlock is now open by design")
    assert True


# ---------------------------------------------------------------------------
# Lane 1 (fused draft top-k) launcher forwarding -- landed here because the lane
# deliberately left launcher territory to this owner. Its patcher-side guards
# read os.environ inside the proposer, and that path is proven live (a banked
# serve carries FR13_DRAFTER_GRAPH=1 in container_env.txt AND graph_replays=1 in
# the census), so -e forwarding is the correct mechanism -- no sidecar needed.
# ---------------------------------------------------------------------------


def test_fused_topk_flag_is_validated_strictly(launcher):
    assert 'case "${FR14_FUSED_DRAFT_TOPK:-0}" in' in launcher
    assert 'echo "FR14_FUSED_DRAFT_TOPK must be exactly 0 or 1" >&2; exit 2' in launcher


def test_fused_topk_defaults_off_everywhere(launcher):
    assert '-e FR14_FUSED_DRAFT_TOPK="${FR14_FUSED_DRAFT_TOPK:-0}"' in launcher


def test_fused_topk_blocks_range_is_enforced_at_launch(launcher):
    """The patcher raises on out-of-range; catch it before the serve boots."""
    assert "FR14_FUSED_DRAFT_TOPK_BLOCKS must be an integer in 1..121" in launcher
    assert (
        '-e FR14_FUSED_DRAFT_TOPK_BLOCKS="${FR14_FUSED_DRAFT_TOPK_BLOCKS:-64}"'
        in launcher
    )


def test_fused_topk_so_default_is_not_empty(launcher):
    """The set-but-empty trap: os.environ.get returns "" not the code default.

    Same defect FR13_DFWD_SPLIT_JSON was shipped against; forwarding the .so path
    with a bare ':-' would silently clobber the patcher's default.
    """
    assert '-e FR14_FUSED_DRAFT_TOPK_SO="${FR14_FUSED_DRAFT_TOPK_SO:-}"' not in launcher
    assert (
        '-e FR14_FUSED_DRAFT_TOPK_SO='
        '"${FR14_FUSED_DRAFT_TOPK_SO:-/tmp/fr14_dfwd_full_topk.abi3.so}"'
    ) in launcher
    patcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "fr10_phase4_patch_vllm_tree_gdn.py"
    ).read_text()
    assert '"/tmp/fr14_dfwd_full_topk.abi3.so"' in patcher, (
        "launcher default must track the patcher default"
    )


def test_fused_topk_armed_requires_its_credential(launcher):
    assert (
        "FR14_FUSED_DRAFT_TOPK=1 requires FR14_FUSED_DRAFT_TOPK_SHA256 "
        "(64 lowercase hex)" in launcher
    )
    assert "FR14_FUSED_DRAFT_TOPK=1 requires FR14_FUSED_DRAFT_TOPK_SO" in launcher


def test_launcher_refuses_the_draft_head_credential_levers(launcher):
    """The split re-issues the drafter graph signature those levers pin as literals.

    Documented as an incompatibility when the split landed, but not enforced --
    which is the same class of gap that produced the Arm G refusal. Enforced now.
    """
    for name in (
        "FR13_DRAFT_HEAD_M32_PRODUCTION",
        "FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION",
        "FR13_DRAFT_HEAD_M4_R64_U8_PRODUCTION",
        "FR13_DRAFT_HEAD_FP8",
        "FR13_DFWD_UNIFIED_BM8_PRODUCTION",
        "FR13_DFWD_UNIFIED_BM8_LIVE_AB",
        "FR13_DRAFT_HEAD_M32_LIVE_AB",
    ):
        assert name in launcher
    assert (
        "is incompatible with $_fr14_gate_incompat "
        "(drafter graph credential is per-capture)" in launcher
    )
