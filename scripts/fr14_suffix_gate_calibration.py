#!/usr/bin/env python3
"""Offline calibration of the FR14 suffix-aware MTP pass gate (FR14_SUFFIX_PASS_GATE).

Read-only over banked SWE-bench trajectories. No GPU, no container, no serve.

WHAT THIS MEASURES
------------------
The gate skips MTP passes 4 and 5 (draft depths 4 and 5 = 0-indexed draft
positions 3 and 4) on steps where the Arctic suffix cache has a STRONG match for
the current committed context, and lets the suffix proposer fill draft positions
3..10 instead of 5..10.

The whole accept cost of that move is ONE slot: the handoff at draft position 3,
which today is MTP-fed with measured survival s3 = 0.8083. Everything deeper is
at least as good under suffix as under MTP (banked: seam_move_economics.md §4).

So this script measures, over banked emitted-token streams:

  w(theta)        = P(gate predicate fires)                    -- the warm-step rate
  q1_gated(theta) = P(suffix proposes Sigma[j+3] correctly | predicate fired)
  r_m_gated       = the suffix chain ladder on gated steps, slots 2..8

against the pre-registered bar q1_gated >= s3_MTP = 0.8083.

THE CACHE BOUNDARY (the detail that makes this a simulation of the real thing)
-----------------------------------------------------------------------------
At a step whose first draft position is stream index j, the Arctic cache holds
Sigma[:j] -- the committed prefix -- and NOTHING more. The seam-3 fill queries a
pattern that ENDS at j+3 (it includes the three MTP-proposed tokens) but every
match it can find must live at an occurrence strictly earlier than j. Both facts
are modelled: patterns extend to j+3, occurrences are filtered to p < j.

Conditioning on reaching draft position 3 means the three MTP tokens were the
true ones, so Sigma[j:j+3] is the correct proposed prefix -- no MTP model needed.

USAGE
  python3 scripts/fr14_suffix_gate_calibration.py \
      --runroot output/fr14_b1_stock_20260817T054447Z/tail6_fixed32_b1radix \
      --out results/fr14_nvfp4_port_20260816/suffix_gate_calibration.json
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import os
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# PRE-REGISTERED CONSTANTS.  Do not tune these against an outcome.
# ---------------------------------------------------------------------------

# Match-length ladder, VOTE_CAP, MAX_CHAIN, SAMPLES_PER_TASK are taken verbatim
# from the banked seam study's params block so this reimplementation can be
# checked against its published ladder.
LADDER = (24, 16, 12, 8, 6, 4, 3, 2)
VOTE_CAP = 64
MAX_CHAIN = 14
SAMPLES_PER_TASK = 3000
SEED = 20260818

# Measured on the K0 serve (21611 steps, per-position accepted counters).
MTP_SURVIVAL = (0.9472, 0.8443, 0.7709, 0.8083, 0.8169)
OBSERVED_TAIL_LADDER = (0.8032, 0.8578, 0.8875, 0.8979, 0.9071)  # slots 2..6
OBSERVED_HANDOFF_S5 = 0.5972
MEASURED_ACCEPT_PER_STEP = 4.2774
MEASURED_STEP_MS = 207.87  # sfwd 134.55 + drafter 52.674 + committer 20.642
MTP_PASS_MS = 10.3  # nsys kernel attribution, passes 2..5

# Gate thresholds swept (predicate is match-length based; see §PREDICATE).
GATE_LENGTHS = (2, 3, 4, 6, 8, 12, 16, 24)
GATE_AGREEMENTS = (0.0, 0.5, 0.75)


# ---------------------------------------------------------------------------
# Stream reconstruction
# ---------------------------------------------------------------------------

def _tool_use_text(block: dict) -> str:
    """Serialize a tool_use block the way it is emitted on the wire."""
    name = block.get("name") or ""
    args = block.get("input")
    try:
        rendered = json.dumps(args, ensure_ascii=False)
    except Exception:
        rendered = str(args)
    return f"{name}\n{rendered}"


def _tool_result_text(block: dict) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                out.append(part.get("text") or "")
            elif isinstance(part, str):
                out.append(part)
        return "\n".join(out)
    return "" if content is None else str(content)


def build_stream(task_dir: Path, tokenizer):
    """Return (tokens, emitted_flags, stats).

    Order: initial prompt, then trace records in file order.  Assistant blocks
    (thinking / text / tool_use arguments) are EMITTED; tool results and the
    prompt are context.
    """
    tokens: list[int] = []
    emitted: list[bool] = []

    def push(text: str, is_emitted: bool):
        if not text:
            return
        ids = tokenizer.encode(text, add_special_tokens=False)
        tokens.extend(ids)
        emitted.extend([is_emitted] * len(ids))

    prompt_path = task_dir / "prompt.md"
    if prompt_path.exists():
        push(prompt_path.read_text(errors="replace"), False)

    reported_output = 0
    trace = task_dir / "qwen_trace.jsonl"
    for line in trace.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        rtype = rec.get("type")
        msg = rec.get("message") or {}
        usage = msg.get("usage") or {}
        if rtype == "assistant":
            reported_output += int(usage.get("output_tokens") or 0)
            for block in msg.get("content") or []:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "thinking":
                    push(block.get("thinking") or "", True)
                elif btype == "text":
                    push(block.get("text") or "", True)
                elif btype == "tool_use":
                    push(_tool_use_text(block), True)
        elif rtype == "user":
            for block in msg.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    push(_tool_result_text(block), False)
                elif isinstance(block, dict) and block.get("type") == "text":
                    push(block.get("text") or "", False)
        elif rtype == "result":
            u = rec.get("usage") or {}
            if u.get("output_tokens"):
                reported_output = max(reported_output, int(u["output_tokens"]))

    stats = {
        "stream_tokens": len(tokens),
        "emitted_tokens": int(sum(emitted)),
        "reported_output_tokens": reported_output,
        "reconstruction_ratio": (
            (sum(emitted) / reported_output) if reported_output else None
        ),
    }
    return tokens, emitted, stats


# ---------------------------------------------------------------------------
# Suffix proposer (stand-in for arctic_inference SuffixDecodingCache)
# ---------------------------------------------------------------------------

class SuffixIndex:
    """Exact n-gram index over one stream, keyed by raw bytes (no hash collisions).

    For each L in LADDER: key = bytes of Sigma[p-L:p], value = ascending list of
    end positions p.  The continuation proposed for a match at p is Sigma[p].
    """

    def __init__(self, tokens: list[int]):
        self.tokens = tokens
        self.blob = b"".join(int(t & 0xFFFFFFFF).to_bytes(4, "little") for t in tokens)
        self.index: dict[int, dict[bytes, list[int]]] = {}
        n = len(tokens)
        for L in LADDER:
            table: dict[bytes, list[int]] = {}
            for p in range(L, n):
                key = self.blob[4 * (p - L): 4 * p]
                slot = table.get(key)
                if slot is None:
                    table[key] = [p]
                else:
                    slot.append(p)
            self.index[L] = table

    @staticmethod
    def _key(seq) -> bytes:
        return b"".join(int(t & 0xFFFFFFFF).to_bytes(4, "little") for t in seq)

    def propose(self, pattern: list[int], boundary: int):
        """Longest-match top-1 proposal.

        pattern  : the context tokens ending at the query point (may contain
                   proposed tokens beyond the cache boundary)
        boundary : cache boundary -- only occurrences with end position p <
                   boundary are visible

        Returns (token, match_len, n_considered, votes_top) or (None, 0, 0, 0).
        """
        for L in LADDER:
            if L > len(pattern):
                continue
            table = self.index[L]
            key = self._key(pattern[-L:])
            positions = table.get(key)
            if not positions:
                continue
            hi = bisect.bisect_left(positions, boundary)
            if hi == 0:
                continue
            window = positions[max(0, hi - VOTE_CAP): hi]
            counts: dict[int, int] = {}
            for p in window:
                nxt = self.tokens[p]
                counts[nxt] = counts.get(nxt, 0) + 1
            best_n = max(counts.values())
            # ties -> most recent
            token = None
            for p in reversed(window):
                if counts[self.tokens[p]] == best_n:
                    token = self.tokens[p]
                    break
            return token, L, len(window), best_n
        return None, 0, 0, 0

    def stats_at(self, pattern: list[int], boundary: int, L: int):
        """The IMPLEMENTABLE predicate's features: exactly-L-gram seen + agreement.

        This is what `Fr14SuffixPassGate` computes online in O(1) per step from a
        single fixed-length n-gram table -- no ladder search, no suffix tree.
        """
        if L > len(pattern):
            return False, 0.0
        positions = self.index[L].get(self._key(pattern[-L:]))
        if not positions:
            return False, 0.0
        hi = bisect.bisect_left(positions, boundary)
        if hi == 0:
            return False, 0.0
        window = positions[max(0, hi - VOTE_CAP): hi]
        counts: dict[int, int] = {}
        for p in window:
            counts[self.tokens[p]] = counts.get(self.tokens[p], 0) + 1
        return True, max(counts.values()) / len(window)

    def chain(self, pattern: list[int], boundary: int, truth: list[int], max_len: int):
        """Chained top-1 proposals scored against `truth`.

        Returns (n_correct_prefix, first_match_len, first_n, first_votes).
        """
        ctx = list(pattern)
        correct = 0
        first = (0, 0, 0)
        for m in range(max_len):
            token, L, n_occ, votes = self.propose(ctx, boundary)
            if m == 0:
                first = (L, n_occ, votes)
            if token is None:
                break
            if m < len(truth) and token == truth[m]:
                correct += 1
                ctx.append(token)
            else:
                break
        return correct, first[0], first[1], first[2]


# ---------------------------------------------------------------------------
# Per-task measurement
# ---------------------------------------------------------------------------

def measure_task(tokens, emitted, rng, idx):
    n = len(tokens)

    # sampleable emitted positions: need 3 MTP tokens + up to 8 suffix slots
    horizon = 3 + 8
    candidates = [j for j in range(64, n - horizon - 1) if emitted[j]]
    if not candidates:
        return None
    sample = candidates if len(candidates) <= SAMPLES_PER_TASK else rng.sample(
        candidates, SAMPLES_PER_TASK
    )
    sample.sort()

    rows = []
    for j in sample:
        # (A) cold-start chain at j: reproduces the banked unconditional ladder
        cold_correct, cold_L, cold_n, cold_v = idx.chain(
            tokens[max(0, j - 32): j], j, tokens[j: j + MAX_CHAIN], MAX_CHAIN
        )
        # (B) gate features at j -- computed from Sigma[:j] ONLY.
        gate_token, gate_L, gate_n, gate_v = idx.propose(tokens[max(0, j - 32): j], j)
        gate_agree = (gate_v / gate_n) if gate_n else 0.0

        # (C) seam-3 outcome: suffix chain covering draft positions 3..10.
        #     pattern extends to j+3 (the three MTP tokens), cache stops at j.
        seam_correct, seam_L, seam_n, seam_v = idx.chain(
            tokens[max(0, j + 3 - 32): j + 3], j, tokens[j + 3: j + 3 + 8], 8
        )
        # (D) today's handoff, for calibration: suffix chain at draft position 5.
        hand_correct, _, _, _ = idx.chain(
            tokens[max(0, j + 5 - 32): j + 5], j, tokens[j + 5: j + 5 + 6], 6
        )
        # (E) the IMPLEMENTABLE predicate's features at fixed lengths
        fix = {}
        for L in (6, 8, 12, 16):
            seen, agree = idx.stats_at(tokens[max(0, j - 32): j], j, L)
            fix[L] = (seen, agree)
        rows.append(
            {
                "j": j,
                "fix": fix,
                "cold": cold_correct,
                "cold_L": cold_L,
                "gate_L": gate_L,
                "gate_n": gate_n,
                "gate_agree": gate_agree,
                "seam": seam_correct,
                "hand": hand_correct,
            }
        )
    return rows


def renewal_simulate(tokens, emitted, idx, gate_len, gate_agree, seed, gate_on):
    """Step-weighted simulation of the decode loop over one stream.

    Position-uniform sampling over-represents easy regions: a renewal process
    spends MORE steps in hard regions (short accepts) than in easy ones, so the
    warm-step RATE a serve would see is not the warm-POSITION rate.  This walks
    the stream the way the engine does -- advance by (accepted + 1) each step --
    and reports step-weighted statistics.

    MTP survivals at draft positions 0,1,2 are context-blind in this model and
    drawn Bernoulli from the measured ladder; the suffix part is the real
    simulated chain at the real context.  Positions 3,4 in the UNGATED arm are
    drawn from the measured MTP ladder too.
    """
    rng = random.Random(seed)
    n = len(tokens)
    j = 64
    steps = 0
    warm = 0
    accepted_total = 0
    accept_hist = collections.Counter()
    warm_accept = 0
    warm_steps = 0
    cold_accept = 0
    while j < n - 16:
        steps += 1
        fired = False
        if gate_on:
            seen, agree = idx.stats_at(tokens[max(0, j - 32): j], j, gate_len)
            fired = seen and (agree >= gate_agree)
        if fired:
            warm += 1
        # draft positions 0,1,2 -- identical work in both arms
        acc = 0
        for s in MTP_SURVIVAL[:3]:
            if rng.random() < s:
                acc += 1
            else:
                break
        if acc == 3:
            if fired:
                # suffix chain owns draft positions 3..10
                extra, _, _, _ = idx.chain(
                    tokens[max(0, j + 3 - 32): j + 3], j, tokens[j + 3: j + 11], 8
                )
                acc += extra
            else:
                # MTP owns 3,4; suffix chain owns 5..10
                for s in MTP_SURVIVAL[3:5]:
                    if rng.random() < s:
                        acc += 1
                    else:
                        break
                if acc == 5:
                    extra, _, _, _ = idx.chain(
                        tokens[max(0, j + 5 - 32): j + 5], j, tokens[j + 5: j + 11], 6
                    )
                    acc += extra
        accepted_total += acc
        accept_hist[acc] += 1
        if fired:
            warm_accept += acc
            warm_steps += 1
        else:
            cold_accept += acc
        j += acc + 1
    return {
        "steps": steps,
        "warm_steps": warm,
        "warm_rate": warm / steps if steps else 0.0,
        "accept_per_step": accepted_total / steps if steps else 0.0,
        "committed_per_step": (accepted_total / steps + 1.0) if steps else 0.0,
        "warm_accept_per_step": (warm_accept / warm_steps) if warm_steps else None,
        "cold_accept_per_step": (
            cold_accept / (steps - warm_steps) if steps - warm_steps else None
        ),
    }


def ladder_from(counts_geq):
    """counts_geq[m] = #(chain >= m); return conditional survivals r_m."""
    out = {}
    for m in range(2, len(counts_geq)):
        prev = counts_geq[m - 1]
        out[m] = (counts_geq[m] / prev) if prev else None
    return out


def expected_accept(survivals):
    total = 0.0
    running = 1.0
    for s in survivals:
        running *= s
        total += running
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runroot", required=True)
    ap.add_argument("--model", default="/models/qwen3.8-27b-nvfp4-radixark")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    per_task_dir = Path(args.runroot) / "swe_out" / "verified" / "per_task"
    task_dirs = sorted(d for d in per_task_dir.iterdir() if d.is_dir())

    rng = random.Random(SEED)
    all_rows = []
    per_task = {}
    streams = []
    for td in task_dirs:
        if not (td / "qwen_trace.jsonl").exists():
            continue
        tokens, emitted, stats = build_stream(td, tok)
        print(f"[stream] {td.name}: {stats}", flush=True)
        idx = SuffixIndex(tokens)
        rows = measure_task(tokens, emitted, rng, idx)
        if not rows:
            continue
        streams.append((td.name, tokens, emitted, idx))
        per_task[td.name] = {
            **stats,
            "sampled": len(rows),
            "q1_cold": sum(1 for r in rows if r["cold"] >= 1) / len(rows),
            "q1_seam_uncond": sum(1 for r in rows if r["seam"] >= 1) / len(rows),
        }
        print(f"[task]   {td.name}: {per_task[td.name]}", flush=True)
        all_rows.extend(rows)

    N = len(all_rows)
    out = {
        "schema": "fr14.suffix_gate_calibration.v1",
        "params": {
            "LADDER": list(LADDER),
            "VOTE_CAP": VOTE_CAP,
            "MAX_CHAIN": MAX_CHAIN,
            "SAMPLES_PER_TASK": SAMPLES_PER_TASK,
            "SEED": SEED,
        },
        "per_task": per_task,
        "pooled_n": N,
    }

    # --- Gate 1 (reimplementation check): unconditional cold-start ladder -----
    cold_geq = [sum(1 for r in all_rows if r["cold"] >= m) for m in range(0, MAX_CHAIN + 1)]
    cold_geq[0] = N
    out["cold_start"] = {
        "q1": cold_geq[1] / N,
        "chain_ladder_r_m": {str(m): v for m, v in ladder_from(cold_geq).items()},
        "cov_geq_d": {str(d): cold_geq[d] / N for d in range(1, MAX_CHAIN + 1)},
    }
    sim_r = [ladder_from(cold_geq)[m] for m in range(2, 7)]
    deltas = [abs(a - b) for a, b in zip(sim_r, OBSERVED_TAIL_LADDER)]
    out["cold_start"]["ladder_validation"] = {
        "simulated_r2_r6": sim_r,
        "observed_r2_r6": list(OBSERVED_TAIL_LADDER),
        "max_abs_delta": max(deltas),
        "slots_within_0.10": sum(1 for d in deltas if d <= 0.10),
        "PASS": sum(1 for d in deltas if d <= 0.10) >= 4,
    }

    # --- Handoff calibration: simulated suffix cold start at draft position 5 -
    hand_hit = sum(1 for r in all_rows if r["hand"] >= 1) / N
    out["handoff_calibration"] = {
        "simulated_s5_unselected": hand_hit,
        "observed_s5": OBSERVED_HANDOFF_S5,
        "selection_premium": OBSERVED_HANDOFF_S5 - hand_hit,
    }

    # --- The gate sweep -------------------------------------------------------
    sweep = []
    for Lstar in GATE_LENGTHS:
        for astar in GATE_AGREEMENTS:
            fired = [
                r for r in all_rows
                if r["gate_L"] >= Lstar and r["gate_agree"] >= astar
            ]
            if not fired:
                continue
            k = len(fired)
            geq = [sum(1 for r in fired if r["seam"] >= m) for m in range(0, 9)]
            geq[0] = k
            surv = []
            for m in range(1, 9):
                prev = geq[m - 1]
                surv.append((geq[m] / prev) if prev else 0.0)
            gated_ladder = list(MTP_SURVIVAL[:3]) + surv
            e_gated = expected_accept(gated_ladder)
            sweep.append(
                {
                    "gate_len": Lstar,
                    "gate_agree": astar,
                    "warm_rate": k / N,
                    "n_fired": k,
                    "q1_gated": surv[0],
                    "seam_ladder_s3_s10": surv,
                    "E_accept_gated_step": e_gated,
                    "E_accept_ungated_step": MEASURED_ACCEPT_PER_STEP,
                }
            )
    out["gate_sweep"] = sweep

    # --- The IMPLEMENTABLE predicate: fixed-L n-gram table, O(1) per step -----
    impl = []
    for L in (6, 8, 12, 16):
        for astar in GATE_AGREEMENTS:
            fired = [
                r for r in all_rows
                if r["fix"][L][0] and r["fix"][L][1] >= astar
            ]
            if not fired:
                continue
            k = len(fired)
            geq = [sum(1 for r in fired if r["seam"] >= m) for m in range(0, 9)]
            geq[0] = k
            surv = [
                (geq[m] / geq[m - 1]) if geq[m - 1] else 0.0 for m in range(1, 9)
            ]
            impl.append(
                {
                    "ngram_len": L,
                    "min_agree": astar,
                    "warm_rate": k / N,
                    "n_fired": k,
                    "q1_gated": surv[0],
                    "seam_ladder_s3_s10": surv,
                    "E_accept_gated_step": expected_accept(
                        list(MTP_SURVIVAL[:3]) + surv
                    ),
                }
            )
    out["implementable_sweep"] = impl

    # --- Threshold selection, by the PRE-REGISTERED rule -----------------------
    # Selection runs on the IMPLEMENTABLE sweep -- that is what ships, so that is
    # what must clear the bar.  ("longest match >= L" and "the L-gram was seen
    # before" are the same event; only the agreement term's measurement point
    # differs, and it is measured here at the shipped length.)
    BAR = MTP_SURVIVAL[3]
    cand = [
        {"gate_len": r["ngram_len"], "gate_agree": r["min_agree"], **r}
        for r in impl
    ]
    clearing = [r for r in cand if r["q1_gated"] >= BAR and r["warm_rate"] >= 0.20]
    if clearing:
        # smallest gate_len, then smallest agreement that clears
        clearing.sort(key=lambda r: (r["gate_len"], r["gate_agree"]))
        chosen = clearing[0]
        verdict = "FAVORABLE"
    else:
        soft = [
            r for r in cand
            if r["q1_gated"] >= BAR - (OBSERVED_HANDOFF_S5 - hand_hit)
            and r["warm_rate"] >= 0.20
        ]
        soft.sort(key=lambda r: (r["gate_len"], r["gate_agree"]))
        chosen = soft[0] if soft else None
        verdict = "MARGINAL" if soft else "UNFAVORABLE"
    out["selection"] = {
        "bar_q1_gated": BAR,
        "verdict": verdict,
        "chosen": chosen,
    }

    # --- The counterfactual on the SAME population ----------------------------
    # E_gated vs the unconditional 4.2774 is not a fair comparison: a gated step
    # is selected for being easy, and MTP would also have done better than its
    # unconditional survival there.  Draft positions 0,1,2 are the SAME three MTP
    # passes in both arms and cancel, so the whole comparison is what happens
    # after position 2, conditional on reaching it.
    if chosen is not None:
        gl, ga = chosen["gate_len"], chosen["gate_agree"]
        fired = [
            r for r in all_rows if r["fix"][gl][0] and r["fix"][gl][1] >= ga
        ]
        k = len(fired)
        # today's handoff-at-position-5 chain, measured on the GATED population
        hgeq = [sum(1 for r in fired if r["hand"] >= m) for m in range(0, 7)]
        hgeq[0] = k
        hand_ladder = [
            (hgeq[m] / hgeq[m - 1]) if hgeq[m - 1] else 0.0 for m in range(1, 7)
        ]
        u = chosen["seam_ladder_s3_s10"]
        a_gated = expected_accept(u)  # slots at draft positions 3..10
        sens = []
        for m34 in (MTP_SURVIVAL[3], 0.88, 0.95):
            # ungated counterfactual on the gated population:
            # MTP at positions 3,4 then the suffix chain from position 5
            a_ungated = expected_accept([m34, m34] + hand_ladder)
            sens.append(
                {
                    "assumed_mtp_survival_on_gated_steps": m34,
                    "A_gated_positions_3_to_10": a_gated,
                    "A_ungated_positions_3_to_10": a_ungated,
                    "delta_accept_given_reached_pos2": a_gated - a_ungated,
                    "delta_accept_per_gated_step": (
                        (a_gated - a_ungated)
                        * MTP_SURVIVAL[0] * MTP_SURVIVAL[1] * MTP_SURVIVAL[2]
                    ),
                }
            )
        out["counterfactual"] = {
            "gate_len": gl,
            "gate_agree": ga,
            "n_gated_positions": k,
            "handoff_ladder_pos5_on_gated_population": hand_ladder,
            "seam_ladder_pos3_on_gated_population": u,
            "reach_pos2_prob": MTP_SURVIVAL[0] * MTP_SURVIVAL[1] * MTP_SURVIVAL[2],
            "sensitivity": sens,
        }

    # --- Step-weighted renewal A/B (paired on identical streams) --------------
    if chosen is not None:
        gl, ga = chosen["gate_len"], chosen["gate_agree"]
        arms = {}
        for arm, gate_on in (("gate_off", False), ("gate_on", True)):
            agg = collections.Counter()
            per = {}
            for rep_seed in (11, 22, 33):
                for name, tk, em, ix in streams:
                    res = renewal_simulate(tk, em, ix, gl, ga, rep_seed, gate_on)
                    per.setdefault(name, []).append(res)
                    agg["steps"] += res["steps"]
                    agg["warm_steps"] += res["warm_steps"]
                    agg["accepted"] += res["accept_per_step"] * res["steps"]
            arms[arm] = {
                "steps": agg["steps"],
                "warm_rate": agg["warm_steps"] / agg["steps"],
                "accept_per_step": agg["accepted"] / agg["steps"],
            }
        w = arms["gate_on"]["warm_rate"]
        base_ms = MEASURED_STEP_MS
        gate_ms = base_ms - w * 2 * MTP_PASS_MS
        base_tps = (arms["gate_off"]["accept_per_step"] + 1.0) / (base_ms / 1000.0)
        gate_tps = (arms["gate_on"]["accept_per_step"] + 1.0) / (gate_ms / 1000.0)
        out["renewal_ab"] = {
            "gate_len": gl,
            "gate_agree": ga,
            "arms": arms,
            "calibration_check": {
                "gate_off_accept_per_step": arms["gate_off"]["accept_per_step"],
                "measured_accept_per_step": MEASURED_ACCEPT_PER_STEP,
                "abs_delta": abs(
                    arms["gate_off"]["accept_per_step"] - MEASURED_ACCEPT_PER_STEP
                ),
            },
            "step_ms_gate_off": base_ms,
            "step_ms_gate_on": gate_ms,
            "saving_ms_per_step_avg": base_ms - gate_ms,
            "saving_ms_on_gated_steps": 2 * MTP_PASS_MS,
            "tps_gate_off": base_tps,
            "tps_gate_on": gate_tps,
            "vs_today_pct": 100.0 * (gate_tps / base_tps - 1.0),
        }
    out["baseline_tps_fullstep_gpu"] = (
        (MEASURED_ACCEPT_PER_STEP + 1.0) / (MEASURED_STEP_MS / 1000.0)
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["cold_start"]["ladder_validation"], indent=1))
    print(json.dumps(out["handoff_calibration"], indent=1))
    for row in sweep:
        print(
            f"gate L>={row['gate_len']:2d} agree>={row['gate_agree']:.2f}  "
            f"warm={row['warm_rate']:.3f}  q1_gated={row['q1_gated']:.3f}  "
            f"E_gated={row['E_accept_gated_step']:.3f}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
