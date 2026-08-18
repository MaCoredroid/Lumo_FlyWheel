#!/usr/bin/env python3
"""FR14 lever 2 -- suffix-aware MTP pass gating.

Host-side predicate that decides, BEFORE the drafter runs, whether this decode
step may run 3 MTP passes instead of 5 and let the Arctic suffix proposer fill
draft positions 3..10 instead of 5..10.

WHY THIS OWNS ITS OWN INDEX INSTEAD OF ASKING ARCTIC
---------------------------------------------------
`arctic_inference.suffix_decoding.SuffixDecodingCache.speculate()` returns
`token_ids` / `parents` / `probs` / `score`.  It does **not** report the matched
pattern length, and in the fixed32 configuration (`use_tree_spec=False`) the
adapter `arctic_draft_to_suffix_rel` discards `probs` before the drafter ever
sees them.  The only strength signal reachable through Arctic is the returned
chain length, which is a function of Arctic's own `max_spec_factor` policy --
an internal we do not control, cannot read on this host (arctic-inference is
pip-installed inside the container at prelaunch and is not vendored), and must
not build an acceptance-affecting predicate on.

So the gate keeps a purpose-built index of exactly one quantity: has the current
L-gram been seen before in this request, and how concentrated is its
continuation distribution.  That is O(1) per committed token and O(1) per step:

  update  : one dict append per committed token (~5/step)
  decide  : one dict lookup + <= VOTE_CAP counter increments

which is microseconds against a 207.87 ms step.  It is also the *exact* quantity
`scripts/fr14_suffix_gate_calibration.py` calibrated offline, so the shipped
predicate and the measured predicate are the same function -- pinned by
`tests/test_fr14_suffix_pass_gate.py::test_online_matches_offline_predicate`.

THE PREDICATE (pre-registered; see results/fr14_nvfp4_port_20260816/suffix_pass_gating.md)

    fire  iff  the last NGRAM tokens have occurred before in this request
          and  agreement(winning continuation) >= MIN_AGREE
          and  history length >= MIN_HISTORY

Calibrated values NGRAM=8, MIN_AGREE=0.75 give, on the banked K0 trajectories:
warm-step rate 0.195, q1_gated 0.820 against the bar s3 = 0.8083.

CONSERVATIVE BIAS IS ABSOLUTE: every uncertainty resolves to "run 5 passes".
Disabled, unknown request, short history, missing index, or ANY exception all
return `fired=False`.
"""

from __future__ import annotations

import os

__all__ = [
    "SuffixPassGate",
    "GateDecision",
    "gate_from_env",
    "gate_from_sidecar",
    "SIDECAR_PATH",
    "DEFAULT_NGRAM",
    "DEFAULT_MIN_AGREE",
]

# The EngineCore worker's curated env drops bare FR13_*/FR14_* masters, which is
# the defect class this campaign has already had to find twice (fr13_dfwd_split,
# FR13_HOST_TAIL_*).  So the serving path reads the launcher-written /logs
# sidecar -- the same value-carrying pattern as fr13_tail_branches.cfg -- and the
# env vars exist only so container_env.txt attests the arm.
SIDECAR_PATH = "/logs/fr14_suffix_pass_gate.cfg"

DEFAULT_NGRAM = 8
DEFAULT_MIN_AGREE = 0.75
DEFAULT_VOTE_CAP = 64
DEFAULT_MIN_HISTORY = 256
# FR14 ANTI-RUNAWAY BRAKE (pre-registered 2026-08-18, round 5).
# The suffix chain is a recurrence copier whose own firing predicate is
# recurrence, so copying keeps the predicate true: per-step re-evaluation is a
# necessary brake but not a sufficient one -- pinned by
# tests/test_fr14_gate_per_step_reevaluation.py::
# test_a_copier_runaway_keeps_the_predicate_true.
# After MAX_RUN consecutive gated steps a request is forced ungated for one
# step, which puts MTP back in the loop and breaks the copy->recurrence->copy
# cycle. Chosen ORTHOGONAL to the predicate on purpose: it does not touch the
# calibrated 8-gram/0.75 threshold, so q1_gated and the warm-rate calibration
# stand unchanged and the brake costs at most 1/(MAX_RUN+1) of the saving.
DEFAULT_MAX_RUN = 32
DEFAULT_MAX_HISTORY = 131072

# Shape of a gated step, from the topology contract.  A gated step runs MTP over
# head depths 0..2 and hands off to Arctic at depth 3, so Arctic must supply a
# main chain of 8 (draft positions 3..10) instead of 6 (positions 5..10).
GATED_MTP_K = 3
GATED_MAIN_TAIL_LENGTH = 8
UNGATED_MTP_K = 5
UNGATED_MAIN_TAIL_LENGTH = 6


class GateDecision:
    """Immutable result of one gate evaluation."""

    __slots__ = ("fired", "reason", "match", "agreement", "occurrences")

    def __init__(self, fired, reason, match=False, agreement=0.0, occurrences=0):
        self.fired = bool(fired)
        self.reason = str(reason)
        self.match = bool(match)
        self.agreement = float(agreement)
        self.occurrences = int(occurrences)

    def as_census(self):
        return {
            "gate_fired": self.fired,
            "gate_reason": self.reason,
            "gate_match": self.match,
            "gate_agreement": round(self.agreement, 4),
            "gate_occurrences": self.occurrences,
        }

    def __repr__(self):  # pragma: no cover - debug only
        return (
            f"GateDecision(fired={self.fired}, reason={self.reason!r}, "
            f"agreement={self.agreement:.3f}, occurrences={self.occurrences})"
        )


_NOT_ENABLED = GateDecision(False, "disabled")


class SuffixPassGate:
    """Per-request n-gram recurrence index + the pass-count predicate.

    Lifecycle mirrors the Arctic cache's, so it can be driven from the same call
    sites in `fr13_merged_drafter`:

        gate.start_request(rid, prompt_token_ids)
        gate.observe(rid, newly_committed_token_ids)   # every step
        decision = gate.decide(rid)                    # before the drafter
        gate.stop_request(rid)
    """

    def __init__(
        self,
        enabled=False,
        ngram=DEFAULT_NGRAM,
        min_agree=DEFAULT_MIN_AGREE,
        vote_cap=DEFAULT_VOTE_CAP,
        min_history=DEFAULT_MIN_HISTORY,
        max_history=DEFAULT_MAX_HISTORY,
        max_run=DEFAULT_MAX_RUN,
    ):
        self.enabled = bool(enabled)
        self.ngram = int(ngram)
        self.min_agree = float(min_agree)
        self.vote_cap = int(vote_cap)
        self.min_history = int(min_history)
        self.max_history = int(max_history)
        self.max_run = int(max_run)
        if self.max_run < 1:
            raise ValueError("max_run must be >= 1")
        if self.ngram < 1:
            raise ValueError("ngram must be >= 1")
        if not 0.0 <= self.min_agree <= 1.0:
            raise ValueError("min_agree must be in [0, 1]")
        if self.vote_cap < 1:
            raise ValueError("vote_cap must be >= 1")
        # rid -> {"tokens": list[int], "index": dict[bytes, list[int]]}
        self._state = {}
        self.stats = {
            "steps": 0,
            "fired": 0,
            "no_match": 0,
            "low_agreement": 0,
            "short_history": 0,
            "run_capped": 0,
            "errors": 0,
        }

    # -- lifecycle ---------------------------------------------------------

    def start_request(self, rid, prompt_token_ids=()):
        if not self.enabled:
            return
        self._state[rid] = {"tokens": [], "index": {}, "run": 0}
        self.observe(rid, prompt_token_ids)

    def stop_request(self, rid):
        self._state.pop(rid, None)

    def reset(self):
        self._state.clear()

    def active_requests(self):
        return set(self._state)

    # -- indexing ----------------------------------------------------------

    @staticmethod
    def _key(tokens, start, stop):
        return b"".join(
            int(t & 0xFFFFFFFF).to_bytes(4, "little") for t in tokens[start:stop]
        )

    def observe(self, rid, new_token_ids):
        """Append newly committed tokens and index the n-grams they complete.

        An entry `index[key].append(p)` means: the n-gram occupying
        `tokens[p-L:p]` was followed by `tokens[p]`.  Only positions already
        committed are ever indexed, so `decide` needs no boundary filter.
        """
        if not self.enabled:
            return
        state = self._state.get(rid)
        if state is None:
            return
        tokens = state["tokens"]
        index = state["index"]
        L = self.ngram
        for tok in new_token_ids:
            tokens.append(int(tok))
            if len(tokens) > self.max_history:
                continue
            p = len(tokens) - 1
            if p >= L:
                index.setdefault(self._key(tokens, p - L, p), []).append(p)

    # -- the predicate -----------------------------------------------------

    def decide(self, rid):
        """Evaluate the gate for this request's next decode step."""
        if not self.enabled:
            return _NOT_ENABLED
        self.stats["steps"] += 1
        try:
            state = self._state.get(rid)
            if state is None:
                self.stats["short_history"] += 1
                return GateDecision(False, "unknown_request")
            if int(state.get("run", 0)) >= self.max_run:
                # forced ungated step: MTP re-enters the loop and the copier's
                # own output stops being able to justify the next gate
                self.stats["run_capped"] += 1
                return GateDecision(False, "run_cap")
            tokens = state["tokens"]
            n = len(tokens)
            if n < self.min_history or n < self.ngram or n > self.max_history:
                self.stats["short_history"] += 1
                return GateDecision(False, "short_history")
            positions = state["index"].get(self._key(tokens, n - self.ngram, n))
            if not positions:
                self.stats["no_match"] += 1
                return GateDecision(False, "no_match")
            window = positions[-self.vote_cap:]
            counts = {}
            for p in window:
                nxt = tokens[p]
                counts[nxt] = counts.get(nxt, 0) + 1
            agreement = max(counts.values()) / len(window)
            if agreement < self.min_agree:
                self.stats["low_agreement"] += 1
                return GateDecision(
                    False, "low_agreement", True, agreement, len(window)
                )
            self.stats["fired"] += 1
            return GateDecision(True, "fired", True, agreement, len(window))
        except Exception as exc:  # fail closed -- never let the gate crash a step
            self.stats["errors"] += 1
            return GateDecision(False, f"error:{type(exc).__name__}")

    def note_step(self, rids, fired):
        """Record what the step ACTUALLY did, so the cap counts real gating.

        Called once per step with the batch-wide outcome. At B>1 a row can vote
        to gate while the batch does not (the decision is unanimous-or-cold), and
        counting that row's vote would cap a run that never happened.
        """
        if not self.enabled:
            return
        for rid in rids:
            state = self._state.get(rid)
            if state is None:
                continue
            state["run"] = int(state.get("run", 0)) + 1 if fired else 0

    # -- shape of the step the decision implies ----------------------------

    @staticmethod
    def step_shape(fired):
        """(mtp_k, main_tail_length, mtp_forward_calls) implied by a decision.

        `mtp_forward_calls` counts the post-root draft-model forwards, which is
        what the fixed32 drafter graph and the work census count (4 today).
        """
        if fired:
            return (GATED_MTP_K, GATED_MAIN_TAIL_LENGTH, GATED_MTP_K - 1)
        return (UNGATED_MTP_K, UNGATED_MAIN_TAIL_LENGTH, UNGATED_MTP_K - 1)

    def summary(self):
        out = dict(self.stats)
        out["enabled"] = self.enabled
        out["ngram"] = self.ngram
        out["min_agree"] = self.min_agree
        out["max_run"] = self.max_run
        out["warm_rate"] = (
            self.stats["fired"] / self.stats["steps"] if self.stats["steps"] else 0.0
        )
        return out


def _env_int(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}")


def _env_float(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a float, got {raw!r}")


def gate_from_env(env=None):
    """Build the gate from the environment.  Default OFF, strictly validated.

    `FR14_SUFFIX_PASS_GATE` must be exactly "0" or "1" when present; anything
    else is a hard error rather than a silent OFF, so a typo cannot quietly
    disarm (or arm) an acceptance-affecting lever.
    """
    env = os.environ if env is None else env
    raw = env.get("FR14_SUFFIX_PASS_GATE", "0").strip()
    if raw not in ("0", "1"):
        raise ValueError(
            f"FR14_SUFFIX_PASS_GATE must be 0 or 1, got {raw!r}"
        )
    return SuffixPassGate(
        enabled=(raw == "1"),
        ngram=_env_int("FR14_SUFFIX_PASS_GATE_NGRAM", DEFAULT_NGRAM),
        min_agree=_env_float("FR14_SUFFIX_PASS_GATE_MIN_AGREE", DEFAULT_MIN_AGREE),
        vote_cap=_env_int("FR14_SUFFIX_PASS_GATE_VOTE_CAP", DEFAULT_VOTE_CAP),
        min_history=_env_int("FR14_SUFFIX_PASS_GATE_MIN_HISTORY", DEFAULT_MIN_HISTORY),
        max_run=_env_int("FR14_SUFFIX_PASS_GATE_MAX_RUN", DEFAULT_MAX_RUN),
    )


def gate_from_sidecar(path=SIDECAR_PATH):
    """Build the gate from the launcher-written sidecar.  Absent file => OFF.

    Format, one line: ``<ngram> <min_agree> <min_history>``.  A present but
    malformed sidecar is FATAL: an acceptance-affecting lever must never run on
    a silently defaulted threshold.
    """
    try:
        with open(path) as fh:
            raw = fh.read().split()
    except FileNotFoundError:
        return SuffixPassGate(enabled=False)
    except OSError:
        return SuffixPassGate(enabled=False)
    if len(raw) != 3:
        raise ValueError(
            f"{path} must hold '<ngram> <min_agree> <min_history>', got {raw!r}"
        )
    ngram = int(raw[0])
    min_agree = float(raw[1])
    min_history = int(raw[2])
    if not 1 <= ngram <= 64:
        raise ValueError(f"{path}: ngram out of range: {ngram}")
    if not 0.0 <= min_agree <= 1.0:
        raise ValueError(f"{path}: min_agree out of range: {min_agree}")
    if min_history < 0:
        raise ValueError(f"{path}: min_history out of range: {min_history}")
    return SuffixPassGate(
        enabled=True,
        ngram=ngram,
        min_agree=min_agree,
        min_history=min_history,
    )
