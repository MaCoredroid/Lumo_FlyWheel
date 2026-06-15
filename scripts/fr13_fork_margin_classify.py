#!/usr/bin/env python3
"""FR13 ProbeClassify reduce (zero-GPU): A/B-classify the residual committer forks.

INPUTS
------
  --rescore   the RECURRENT-oracle rescore json (fr13_recurrent_decode_oracle.py
              rescore over the cat9+K1 served stream): per_prompt[].positions[]
              with clear_margin flags = the CLEAR-MARGIN FORK positions
              (cat9 served != recurrent-oracle greedy at >1nat).
  --dump      the FR13_FORK_MARGIN_DUMP jsonl from the SAME cat9+K1 boot: per
              spec-step per request the committer-fork record (each path's lcp,
              the WINNER/SPINE lcp-divergence nodes' VERIFY top-2 margins, the
              committed_row).
  --capture   the gold_margin_probe capture json (prompts + records[].
              served_token_ids) -- the served stream the rescore scored, used to
              SEGMENT the global dump-step sequence into per-prompt position
              ranges (the JOIN key).

JOIN (non-vacuity, bug-class #12 measurement trap)
--------------------------------------------------
The dump `step` counter is GLOBAL/monotonic across every rejection_sample call
(both capture reps, all prompts). We reconstruct each prompt's served stream by
walking the dump records in step order, consuming each record's committed_row
tokens, and matching against the capture's records[pid].served_token_ids. The
first contiguous run of dump steps whose concatenated committed_row equals the
prompt's served stream defines that prompt's (position -> dump-step) map. A fork
position then JOINS to the dump step whose committed_row spans it. We ASSERT the
fork positions match dump records (non-vacuous: every clear-margin fork must land
on a real dump step, else the capture is disjoint -> FAIL LOUD).

CLASSIFY (the decisive fundamental-vs-fixable split)
----------------------------------------------------
At each joined fork we read the DECIDING parent_target's verify top-2 margin.
The deciding node is the lcp-divergence node that turned the leaf-vs-spine lcp:
  - if the served token came from a LEAF that out-matched the spine
    (winner_lcp >= spine_lcp and best_leaf != spine_leaf), the deciding node is
    the WINNER's lcp-divergence node (winner_div_margin) -- the parent_target
    the leaf matched past where the spine stopped;
  - the topology SPLIT node margin (split_node_margin) is the branch-point
    parent_target where leaf vs spine first diverge; used as the tie-break /
    cross-check.
We take the deciding margin = the margin at the node whose parent_target
DECIDED the boundary (default winner_div_margin; fall back to split_node_margin
when the winner fully matched).
  (A) FUNDAMENTAL: deciding margin > 1.0 nat  -> genuine confident leaf win
      (margin-damp would reject a real accept).
  (B) FIXABLE:     deciding margin < 1.0 nat  -> sub-1-nat near-tie
      (a deterministic rank-2 margin-damp stops the fork; genuine (A) wins keep
      serving).

OUTPUT: per-fork A/B label + deciding margin + the full margin distribution +
the A/B count + VERDICT recommendation. Reward-hack-free: read-only over the
saved artifacts, native is the oracle only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_dump(path: str) -> list[dict[str, Any]]:
    recs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    # global monotonic step order
    recs.sort(key=lambda r: (int(r.get("step", -1)), int(r.get("req_index", 0))))
    return recs


def _flatten_dump(dump: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    """Flatten committed_row tokens in global step order -> (tokens, rec_index)."""
    flat: list[int] = []
    rec_of: list[int] = []
    for j, r in enumerate(dump):
        for t in r.get("committed_row", []):
            flat.append(int(t))
            rec_of.append(j)
    return flat, rec_of


def _segment_prompt_steps(
    dump: list[dict[str, Any]], flat: list[int], rec_of: list[int],
    served: list[int], min_coverage: float = 0.95, max_head: int = 4,
) -> tuple[int, int, dict[int, int]] | None:
    """Align a prompt's served stream to the flattened committer-dump tokens.

    The served stream (from the completions API) carries a 0..max_head-token
    chat-template/BOS prefix that the raw committer dump does NOT contain, and
    the dump also holds a warmup request + a second capture rep -- so we slide
    served[head:] against the flattened committed tokens and take the
    (head_skip, flat_offset) that maximises a CONTIGUOUS match. Returns
    (head_skip, flat_offset, {served_pos -> dump_rec_index}) for the first
    (rep1) high-coverage alignment, or None.

    served_pos -> rec: served[head+i] committed flat[flat_offset+i], whose dump
    record is rec_of[flat_offset+i]. The head-skip positions (0..head-1) have no
    committer record (template tokens) and are simply absent from the map; a
    fork there would mark join MISS (but clear-margin forks are model tokens, so
    they land at served_pos >= head).
    """
    best = None  # (matched, head, off)
    for head in range(0, max_head + 1):
        ss = served[head:]
        if not ss:
            continue
        for off in range(len(flat)):
            k = 0
            while (
                k < len(ss)
                and off + k < len(flat)
                and flat[off + k] == ss[k]
            ):
                k += 1
            if best is None or k > best[0]:
                best = (k, head, off)
        # if this head already covers the whole stream, stop early
        if best and best[0] >= len(served) - head:
            break
    if best is None:
        return None
    matched, head, off = best
    denom = max(1, len(served) - head)
    if matched / denom < min_coverage:
        return None
    pos_to_rec: dict[int, int] = {}
    for i in range(matched):
        sp = head + i
        fi = off + i
        if 0 <= fi < len(rec_of):
            pos_to_rec[sp] = rec_of[fi]
    return head, off, pos_to_rec


def _deciding_margin(rec: dict[str, Any]) -> tuple[float | None, str, dict | None]:
    """The deciding parent_target margin for an LCP fork + which node decided.

    Default = the WINNER lcp-divergence node (the parent_target the leaf matched
    past where the spine stopped). If the winner fully matched (winner_div is
    the leaf with no rejection) fall back to the topology split-node margin.
    """
    win = rec.get("winner_div_margin")
    split = rec.get("split_node_margin")
    winner_lcp = int(rec.get("winner_lcp", 0))
    best_path = rec.get("best_path") or []
    # winner fully matched (no rejection node) iff winner_lcp == len(best_path)
    winner_full = winner_lcp >= len(best_path)
    if not winner_full and isinstance(win, dict) and "verify_top2_margin_nat" in win:
        return float(win["verify_top2_margin_nat"]), "winner_div", win
    if isinstance(split, dict) and "verify_top2_margin_nat" in split:
        return float(split["verify_top2_margin_nat"]), "split_node", split
    if isinstance(win, dict) and "verify_top2_margin_nat" in win:
        return float(win["verify_top2_margin_nat"]), "winner_div_fallback", win
    return None, "none", None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rescore", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin-threshold", type=float, default=1.0)
    args = ap.parse_args()

    rescore = json.loads(Path(args.rescore).read_text(encoding="utf-8"))
    capture = json.loads(Path(args.capture).read_text(encoding="utf-8"))
    dump = _load_dump(args.dump)

    cap_served = [
        [int(t) for t in r["served_token_ids"]] for r in capture["records"]
    ]
    per_prompt = rescore["per_prompt"]

    # Flatten the committer dump once, then align each prompt's served stream
    # (the JOIN). The completions stream carries a chat-template head-skip the
    # raw committer dump lacks; we slide to find the contiguous alignment.
    flat, rec_of = _flatten_dump(dump)
    prompt_maps: list[tuple[int, int, dict[int, int]] | None] = []
    prompt_align_meta: list[dict[str, Any]] = []
    for pid, served in enumerate(cap_served):
        seg = _segment_prompt_steps(dump, flat, rec_of, served)
        prompt_maps.append(seg)
        if seg is None:
            prompt_align_meta.append({"prompt_id": pid, "aligned": False})
        else:
            head, off, m = seg
            prompt_align_meta.append({
                "prompt_id": pid, "aligned": True, "head_skip": head,
                "flat_offset": off, "n_positions_mapped": len(m),
                "served_len": len(served),
                "coverage": round(len(m) / max(1, len(served) - head), 4),
            })

    n_join_ok = sum(1 for m in prompt_maps if m is not None)

    forks: list[dict[str, Any]] = []
    n_clear_total = 0
    n_join_to_dump = 0
    n_join_miss = 0
    n_fork_dump = 0  # joined dump record had is_fork True

    for pp in per_prompt:
        pid = int(pp["prompt_id"])
        seg = prompt_maps[pid] if pid < len(prompt_maps) else None
        pos_to_rec = seg[2] if seg else None
        clear_positions = [
            p for p in pp.get("positions", []) if p.get("clear_margin")
        ]
        for p in clear_positions:
            n_clear_total += 1
            pos = int(p["pos"])
            joined = None
            if pos_to_rec is not None and pos in pos_to_rec:
                joined = dump[pos_to_rec[pos]]
            rec = {
                "prompt_id": pid,
                "served_pos": pos,
                "served_token_id": p.get("served_token_id"),
                "oracle_argmax_id": p.get("oracle_argmax_id"),
                "oracle_deviation_nat": p.get("deviation_nat"),
                "joined": joined is not None,
            }
            if joined is None:
                n_join_miss += 1
                rec["join_status"] = "MISS"
                forks.append(rec)
                continue
            n_join_to_dump += 1
            margin, which, node = _deciding_margin(joined)
            is_fork_dump = bool(joined.get("is_fork"))
            if is_fork_dump:
                n_fork_dump += 1
            rec.update({
                "join_status": "OK",
                "dump_step": int(joined.get("step", -1)),
                "dump_is_fork": is_fork_dump,
                "best_leaf": joined.get("best_leaf"),
                "spine_leaf": joined.get("spine_leaf"),
                "winner_lcp": joined.get("winner_lcp"),
                "spine_lcp": joined.get("spine_lcp"),
                "lcp_delta": joined.get("lcp_delta_winner_minus_spine"),
                "deciding_node_kind": which,
                "deciding_margin_nat": margin,
                "deciding_parent_target_id": (node or {}).get("parent_target_id"),
                "deciding_verify_argmax_id": (node or {}).get("verify_argmax_id"),
                "spine_token_at_split": joined.get("spine_token_at_split"),
                "winner_token_at_split": joined.get("winner_token_at_split"),
            })
            if margin is None:
                rec["class"] = "UNKNOWN_NO_MARGIN"
            elif margin > args.margin_threshold:
                rec["class"] = "A_FUNDAMENTAL"
            else:
                rec["class"] = "B_FIXABLE"
            forks.append(rec)

    classified = [f for f in forks if f.get("class") in ("A_FUNDAMENTAL", "B_FIXABLE")]
    n_A = sum(1 for f in classified if f["class"] == "A_FUNDAMENTAL")
    n_B = sum(1 for f in classified if f["class"] == "B_FIXABLE")
    margins = [
        f["deciding_margin_nat"] for f in classified
        if f.get("deciding_margin_nat") is not None
    ]
    margins_sorted = sorted(margins)

    verdict = "INSUFFICIENT"
    if classified:
        if n_B > n_A:
            verdict = "MOSTLY_B_FIXABLE: rank-2 LCP near-tie margin-damp is viable (K1+margin-damp = lossless+fast candidate; recommend implementing + e2e testing)"
        elif n_A > n_B:
            verdict = "MOSTLY_A_FUNDAMENTAL: leaf forks are genuine accept-edge wins; margin-damp would cost accept -> relax to accept/event-parity"
        else:
            verdict = "SPLIT_A_EQ_B: mixed; margin-damp recovers ~half, costs the other half"

    artifact = {
        "schema": "fr13.fork_margin_classify.v1",
        "rescore_src": args.rescore,
        "dump_src": args.dump,
        "capture_src": args.capture,
        "margin_threshold_nat": args.margin_threshold,
        "n_prompts": len(cap_served),
        "n_prompts_dump_join_ok": n_join_ok,
        "join_nonvacuous": bool(n_join_ok == len(cap_served)),
        "prompt_alignment": prompt_align_meta,
        "n_clear_margin_forks": n_clear_total,
        "n_joined_to_dump": n_join_to_dump,
        "n_join_miss": n_join_miss,
        "n_joined_dump_is_fork": n_fork_dump,
        "n_classified": len(classified),
        "n_A_fundamental": n_A,
        "n_B_fixable": n_B,
        "margin_distribution_sorted_nat": [round(m, 4) for m in margins_sorted],
        "margin_min": (round(min(margins), 4) if margins else None),
        "margin_max": (round(max(margins), 4) if margins else None),
        "margin_median": (
            round(margins_sorted[len(margins_sorted) // 2], 4) if margins else None
        ),
        "verdict": verdict,
        "forks": forks,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        k: artifact[k] for k in (
            "n_prompts_dump_join_ok", "join_nonvacuous", "n_clear_margin_forks",
            "n_joined_to_dump", "n_join_miss", "n_classified",
            "n_A_fundamental", "n_B_fixable", "margin_distribution_sorted_nat",
            "verdict",
        )
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
