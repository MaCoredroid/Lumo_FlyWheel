#!/usr/bin/env python3
"""FR13 per-event SUPERSET gate (zero-GPU reducer).

Spec: FR13_PEREVENT_SUPERSET_GATE.md (ce912070). Replaces the cross-trajectory
aggregate accept/event edge (cat9 ~3.198 vs native E5 3.076, "+0.12") with a
PER-EVENT, SAME-REFERENCE superset gate that unifies SPEED (the leaf-saves) and
LOSSLESS (the flips) on the SAME leaves.

THE GATE (user, precise per-event superset)
-------------------------------------------
For the cat9 caterpillar tree vs the E5 (MTP-5) spine, scored on the cat9 served
stream teacher-forced through the deployment RECURRENT decode oracle (the greedy
reference -- NOT streamed logprobs / serial-torch / a backend name, bug #10/#11):

  * SPINE NON-REGRESSION: cat9 accepts >= the spine tokens E5 accepts at every
    event. STRUCTURAL (committer best_lcp = max-over-leaves, strict tie-break in
    fr10_phase4_patch_vllm_tree_gdn.py L6894-6925 => winner_lcp >= spine_lcp) so
    spine_regressions == 0 by construction. We MEASURE it to confirm S2/S3 did
    not break it.
  * LEAF SAVE = a depth d in [spine_lcp, winner_lcp): the E5 spine CUT at d
    (d == spine_lcp is the first miss) but cat9 saved a token at d via a winning
    leaf (winner_lcp > spine_lcp on a fork). The number of such depths per event
    == winner_lcp - spine_lcp == lcp_delta_winner_minus_spine. The aggregate edge
    cat9 - E5 accept/event == GROSS leaf-saves/event EXACTLY.
  * LOSSLESS vs LOSSY: a leaf save at served position i is LOSSLESS iff the served
    leaf token == the recurrent-oracle greedy (oracle_argmax_id[i]); else it is a
    FLIP = a LOSSY save (the same leaf-fork the flip analysis tracks). We report
    BOTH the raw-flip count (served != oracle argmax, the spec's strict
    definition) and the clear-margin subset (flip beyond threshold_nat).

  GATE METRIC: net = lossless_leaf_saves - lossy_leaf_saves - spine_regressions
  (per depth d=0..4 + total). PASS = net > 0 AND spine_regressions == 0.

JOIN (non-vacuity, bug-class #9 / #12)
--------------------------------------
We reuse the EXACT JOIN proven in fr13_fork_margin_classify.py: flatten the
committer dump's committed_row tokens in global step order; slide each prompt's
served stream (head-skip = chat-template prefix) against the flattened tokens;
the first contiguous high-coverage alignment defines (served_pos -> dump record
+ position-in-committed-row = DEPTH d). FAIL LOUD if any prompt does not align or
any leaf-save position misses a dump step (non-vacuous: every counted position
lands on a real dump event). We additionally assert the rescore oracle is the
RECURRENT decode path (RECURRENT_PATH_ENGAGED) and that n_positions matches the
rescore denominator (total_positions).

DEPTH SEMANTICS (CODE-READ, verified on the banked dump)
--------------------------------------------------------
committed_row = [accepted prefix of length winner_lcp] + [1 bonus token]
  (committed_row_len == winner_lcp + 1 on ALL 250 dump records).
position-in-committed-row pir == depth d of that served token within the event:
  d in [0, winner_lcp)  = accepted draft tokens (depths 0..winner_lcp-1);
  d == winner_lcp       = the verify-argmax BONUS token (NOT a draft accept).
On every leaf-save event split_pos == spine_lcp (verified): the spine is cut at
depth spine_lcp and the winning leaf continues, so leaf-saves occupy depths
[spine_lcp, winner_lcp) exactly. lcp_delta == winner_lcp - spine_lcp (verified).

INPUTS (banked, single boot, fork_margin set -- the one with the dump<->oracle join)
  --dump      output/fr13_fork_margin_probe/logs/fr13_fork_margin_dump.jsonl
  --rescore   output/fr13_fork_margin_probe/logs/rescore_cat9_K1_forkmargin.json
  --capture   output/fr13_fork_margin_probe/logs/probe_us_k1_forkmargin.json
              (records[].served_token_ids = the served stream the rescore scored)

Read-only over the saved artifacts; the recurrent oracle is the greedy judge.
This is a BANKED-DATA estimate (one boot, 23 clear-margin flips / ~27-35 gross
saves, bug-class #12 small-sample) -- NO bake/ship decision (user call).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MAX_DEPTH = 4  # spine [0,1,3,5,7] -> depths 0..4 (MTP-5)


def _load_dump(path: str) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    recs.sort(key=lambda r: (int(r.get("step", -1)), int(r.get("req_index", 0))))
    return recs


def _flatten_dump(
    dump: list[dict[str, Any]],
) -> tuple[list[int], list[int], list[int]]:
    """Flatten committed_row in global step order -> (tokens, rec_index, pir).

    pir = position-in-committed-row == the DEPTH d of that served token within
    its event.
    """
    flat: list[int] = []
    rec_of: list[int] = []
    pir_of: list[int] = []
    for j, r in enumerate(dump):
        for k, t in enumerate(r.get("committed_row", [])):
            flat.append(int(t))
            rec_of.append(j)
            pir_of.append(k)
    return flat, rec_of, pir_of


def _segment_prompt_steps(
    flat: list[int],
    served: list[int],
    min_coverage: float = 0.95,
    max_head: int = 4,
) -> tuple[int, int, int] | None:
    """Align a prompt's served stream to the flattened dump tokens.

    Returns (head_skip, flat_offset, matched_len) for the first high-coverage
    contiguous alignment, or None. Identical logic to
    fr13_fork_margin_classify._segment_prompt_steps (the proven JOIN).
    """
    best: tuple[int, int, int] | None = None  # (matched, head, off)
    for head in range(0, max_head + 1):
        ss = served[head:]
        if not ss:
            continue
        for off in range(len(flat)):
            k = 0
            while k < len(ss) and off + k < len(flat) and flat[off + k] == ss[k]:
                k += 1
            if best is None or k > best[0]:
                best = (k, head, off)
        if best and best[0] >= len(served) - head:
            break
    if best is None:
        return None
    matched, head, off = best
    denom = max(1, len(served) - head)
    if matched / denom < min_coverage:
        return None
    return head, off, matched


def _empty_depth_tally() -> dict[str, int]:
    return {
        "spine_regressions": 0,
        "gross_leaf_saves": 0,
        "lossless_leaf_saves": 0,
        "lossy_leaf_saves": 0,
        "lossy_leaf_saves_clear_margin": 0,
        "net": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dump",
        default="output/fr13_fork_margin_probe/logs/fr13_fork_margin_dump.jsonl",
    )
    ap.add_argument(
        "--rescore",
        default="output/fr13_fork_margin_probe/logs/rescore_cat9_K1_forkmargin.json",
    )
    ap.add_argument(
        "--capture",
        default="output/fr13_fork_margin_probe/logs/probe_us_k1_forkmargin.json",
    )
    ap.add_argument(
        "--out",
        default="output/fr13_fork_margin_probe/logs/fr13_perevent_superset_gate.json",
    )
    args = ap.parse_args()

    dump = _load_dump(args.dump)
    rescore = json.loads(Path(args.rescore).read_text(encoding="utf-8"))
    capture = json.loads(Path(args.capture).read_text(encoding="utf-8"))

    # --- ENGAGEMENT / NON-VACUITY ASSERTS (bug-class #9) ---
    # The oracle MUST be the deployment recurrent decode path, not streamed
    # logprobs / serial-torch / chunked-prefill.
    recurrent_engaged = bool(rescore.get("RECURRENT_PATH_ENGAGED"))
    if not recurrent_engaged:
        raise SystemExit(
            "FAIL-LOUD (#9): rescore RECURRENT_PATH_ENGAGED is not True -- the "
            "greedy reference is NOT the deployment recurrent decode oracle."
        )
    if rescore.get("schema") != "fr13.recurrent_decode_oracle.rescore.v1":
        raise SystemExit(
            f"FAIL-LOUD (#9): unexpected rescore schema {rescore.get('schema')!r}"
        )
    if not bool(rescore.get("within_proc_det_all_prompts", True)):
        raise SystemExit(
            "FAIL-LOUD (#8): rescore within_proc_det_all_prompts is False -- "
            "oracle rep1 != rep2 (non-deterministic reference)."
        )

    per_prompt = rescore["per_prompt"]
    cap_served = [
        [int(t) for t in r["served_token_ids"]] for r in capture["records"]
    ]
    n_prompts = len(cap_served)

    threshold_nat = float(rescore.get("threshold_nat", 1.0))

    flat, rec_of, pir_of = _flatten_dump(dump)

    # --- THE JOIN: served_pos -> (dump_rec_index, depth d) per prompt ---
    prompt_align: list[dict[str, Any]] = []
    posmaps: list[dict[int, tuple[int, int]]] = []  # pid -> {served_pos: (rec, pir)}
    n_join_ok = 0
    for pid, served in enumerate(cap_served):
        seg = _segment_prompt_steps(flat, served)
        if seg is None:
            prompt_align.append({"prompt_id": pid, "aligned": False})
            posmaps.append({})
            continue
        head, off, matched = seg
        n_join_ok += 1
        posmap: dict[int, tuple[int, int]] = {}
        for i in range(matched):
            sp = head + i
            fi = off + i
            if 0 <= fi < len(rec_of):
                posmap[sp] = (rec_of[fi], pir_of[fi])
        posmaps.append(posmap)
        prompt_align.append(
            {
                "prompt_id": pid,
                "aligned": True,
                "head_skip": head,
                "flat_offset": off,
                "n_positions_mapped": len(posmap),
                "served_len": len(served),
                "coverage": round(len(posmap) / max(1, len(served) - head), 4),
            }
        )

    join_nonvacuous = n_join_ok == n_prompts
    if not join_nonvacuous:
        raise SystemExit(
            f"FAIL-LOUD (#9/#12): JOIN aligned only {n_join_ok}/{n_prompts} "
            "prompts -- the served stream is disjoint from the committer dump."
        )

    # rescore positions keyed by (pid, served_pos)
    resc_pos: dict[tuple[int, int], dict[str, Any]] = {}
    n_resc_positions = 0
    for pp in per_prompt:
        pid = int(pp["prompt_id"])
        for p in pp.get("positions", []):
            resc_pos[(pid, int(p["pos"]))] = p
            n_resc_positions += 1

    rescore_denom = int(rescore.get("total_positions", -1))
    if n_resc_positions != rescore_denom:
        raise SystemExit(
            f"FAIL-LOUD (#9): rescore positions counted {n_resc_positions} != "
            f"declared total_positions {rescore_denom}."
        )

    # --- TALLY: per depth + total ---
    per_depth: dict[int, dict[str, int]] = {
        d: _empty_depth_tally() for d in range(MAX_DEPTH + 1)
    }
    total = _empty_depth_tally()

    # Iterate the JOINED events ONCE each (a record may span many served
    # positions; spine_regression / gross are per-EVENT, leaf-save lossless/lossy
    # is per-POSITION within the event). Walk per served position, but to count
    # an event's spine_regression / gross-by-event once, we anchor on the depth.

    # Track which (pid, dump_rec) we've already counted spine_regression for.
    counted_regr: set[tuple[int, int]] = set()
    leafsave_positions: list[dict[str, Any]] = []
    n_leafsave_positions = 0
    n_oracle_lookup_miss = 0

    for pid in range(n_prompts):
        posmap = posmaps[pid]
        for sp, (rj, pir) in posmap.items():
            r = dump[rj]
            slcp = int(r["spine_lcp"])
            wlcp = int(r["winner_lcp"])
            d = pir  # depth within the event == position-in-committed-row

            # SPINE REGRESSION (per event, structural 0): winner_lcp < spine_lcp.
            # Count once per event, attribute to the cut depth (== winner_lcp).
            if wlcp < slcp:
                key = (pid, rj)
                if key not in counted_regr:
                    counted_regr.add(key)
                    cd = min(wlcp, MAX_DEPTH)
                    per_depth[cd]["spine_regressions"] += 1
                    total["spine_regressions"] += 1

            # LEAF SAVE at depth d in [spine_lcp, winner_lcp).
            if slcp <= d < wlcp:
                n_leafsave_positions += 1
                dd = min(d, MAX_DEPTH)
                per_depth[dd]["gross_leaf_saves"] += 1
                total["gross_leaf_saves"] += 1

                p = resc_pos.get((pid, sp))
                if p is None:
                    # A leaf-save position MUST have an oracle score -- else the
                    # oracle denominator is disjoint from the served stream.
                    n_oracle_lookup_miss += 1
                    raise SystemExit(
                        f"FAIL-LOUD (#9): leaf-save served pos (pid={pid}, "
                        f"sp={sp}) has no recurrent-oracle score."
                    )
                served_tok = int(p["served_token_id"])
                oracle_tok = int(p["oracle_argmax_id"])
                # int-view equality, NEVER atol (grounding rule).
                lossless = served_tok == oracle_tok
                is_flip = bool(p.get("flip"))  # served != oracle argmax
                is_clear = bool(p.get("clear_margin"))
                # Consistency: flip flag must equal the int-view inequality.
                if is_flip != (not lossless):
                    raise SystemExit(
                        f"FAIL-LOUD (#9): rescore flip flag disagrees with "
                        f"int-view served!=oracle at (pid={pid}, sp={sp})."
                    )
                if lossless:
                    per_depth[dd]["lossless_leaf_saves"] += 1
                    total["lossless_leaf_saves"] += 1
                else:
                    per_depth[dd]["lossy_leaf_saves"] += 1
                    total["lossy_leaf_saves"] += 1
                    if is_clear:
                        per_depth[dd]["lossy_leaf_saves_clear_margin"] += 1
                        total["lossy_leaf_saves_clear_margin"] += 1
                leafsave_positions.append(
                    {
                        "prompt_id": pid,
                        "served_pos": sp,
                        "depth": d,
                        "dump_step": int(r.get("step", -1)),
                        "spine_lcp": slcp,
                        "winner_lcp": wlcp,
                        "best_leaf": r.get("best_leaf"),
                        "spine_leaf": r.get("spine_leaf"),
                        "served_token_id": served_tok,
                        "oracle_argmax_id": oracle_tok,
                        "deviation_nat": p.get("deviation_nat"),
                        "lossless": lossless,
                        "flip": is_flip,
                        "clear_margin": is_clear,
                    }
                )

    # net = lossless - lossy - spine_regressions (raw-flip lossy = the spec def).
    for d in range(MAX_DEPTH + 1):
        t = per_depth[d]
        t["net"] = (
            t["lossless_leaf_saves"]
            - t["lossy_leaf_saves"]
            - t["spine_regressions"]
        )
    total["net"] = (
        total["lossless_leaf_saves"]
        - total["lossy_leaf_saves"]
        - total["spine_regressions"]
    )

    # --- N_events in the joined window (unique dump records spanned) ---
    joined_events: set[tuple[int, int]] = set()
    for pid in range(n_prompts):
        for sp, (rj, pir) in posmaps[pid].items():
            joined_events.add((pid, rj))
    n_events = len(joined_events)

    gross_pe = total["gross_leaf_saves"] / n_events if n_events else None
    net_pe = total["net"] / n_events if n_events else None
    lossless_fraction = (
        total["lossless_leaf_saves"] / total["gross_leaf_saves"]
        if total["gross_leaf_saves"]
        else None
    )

    spine_regressions = total["spine_regressions"]
    net = total["net"]
    gate_pass = bool(net > 0 and spine_regressions == 0)

    artifact = {
        "schema": "fr13.perevent_superset_gate.v1",
        "dump_src": args.dump,
        "rescore_src": args.rescore,
        "capture_src": args.capture,
        "repo_head": "f574ec77",  # informational; output/ is gitignored
        # --- ENGAGEMENT / NON-VACUITY (bug-class #9) ---
        "engagement": {
            "oracle_recurrent_path_engaged": recurrent_engaged,
            "oracle_schema": rescore.get("schema"),
            "oracle_within_proc_det_all_prompts": bool(
                rescore.get("within_proc_det_all_prompts", True)
            ),
            "oracle_recurrent_decode_calls": rescore.get("recurrent_decode_calls"),
            "oracle_n_gdn_layers_introspected": rescore.get(
                "n_gdn_layers_introspected"
            ),
            "join_nonvacuous": join_nonvacuous,
            "n_prompts": n_prompts,
            "n_prompts_join_ok": n_join_ok,
            "n_positions_mapped": sum(len(m) for m in posmaps),
            "rescore_total_positions": rescore_denom,
            "n_rescore_positions_seen": n_resc_positions,
            "n_oracle_lookup_miss_on_leafsave": n_oracle_lookup_miss,
            "int_view_equality": "served_token_id == oracle_argmax_id (NO atol)",
        },
        "prompt_alignment": prompt_align,
        "n_events_joined": n_events,
        "n_leafsave_positions": n_leafsave_positions,
        "threshold_nat": threshold_nat,
        # --- THE GATE TALLY ---
        "total": total,
        "per_depth": {str(d): per_depth[d] for d in range(MAX_DEPTH + 1)},
        # --- RECONCILIATION (§3) ---
        "reconciliation": {
            "gross_leaf_saves_per_event": (
                round(gross_pe, 4) if gross_pe is not None else None
            ),
            "net_lossless_per_event": (
                round(net_pe, 4) if net_pe is not None else None
            ),
            "lossless_fraction_of_gross": (
                round(lossless_fraction, 4)
                if lossless_fraction is not None
                else None
            ),
            "note": (
                "gross_leaf_saves_per_event is the APPLE-TO-APPLE per-event "
                "version of the cross-trajectory aggregate cat9-E5 accept/event "
                "edge (winner_lcp = spine_lcp + leaf-saves every event, so the "
                "edge IS gross-saves/event). It is measured here over this boot's "
                "n_events_joined spec-steps; the banked '+0.12' is the same KIND "
                "of quantity over a DIFFERENT cross-trajectory window (bug-class "
                "#12) so they are the same statistic, not byte-identical. The gate "
                "net strips the lossy fraction. BANKED-DATA estimate, one boot, "
                "small-sample (bug-class #12). lossy_leaf_saves uses the spec "
                "strict definition (served leaf != oracle greedy argmax); "
                "lossy_leaf_saves_clear_margin = the >threshold_nat subset (here "
                "all lossy leaf-saves are clear-margin)."
            ),
        },
        # --- VERDICT ---
        "spine_regressions": spine_regressions,
        "gross_leaf_saves": total["gross_leaf_saves"],
        "lossless_leaf_saves": total["lossless_leaf_saves"],
        "lossy_leaf_saves": total["lossy_leaf_saves"],
        "net": net,
        "gate_pass": gate_pass,
        "leafsave_positions": leafsave_positions,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    summary = {
        "join_nonvacuous": join_nonvacuous,
        "oracle_recurrent_path_engaged": recurrent_engaged,
        "n_events_joined": n_events,
        "n_leafsave_positions": n_leafsave_positions,
        "spine_regressions": spine_regressions,
        "gross_leaf_saves": total["gross_leaf_saves"],
        "lossless_leaf_saves": total["lossless_leaf_saves"],
        "lossy_leaf_saves": total["lossy_leaf_saves"],
        "lossy_leaf_saves_clear_margin": total["lossy_leaf_saves_clear_margin"],
        "net": net,
        "gate_pass": gate_pass,
        "gross_leaf_saves_per_event": (
            round(gross_pe, 4) if gross_pe is not None else None
        ),
        "lossless_fraction_of_gross": (
            round(lossless_fraction, 4) if lossless_fraction is not None else None
        ),
        "per_depth": {
            str(d): {
                "spine_reg": per_depth[d]["spine_regressions"],
                "gross": per_depth[d]["gross_leaf_saves"],
                "lossless": per_depth[d]["lossless_leaf_saves"],
                "lossy": per_depth[d]["lossy_leaf_saves"],
                "net": per_depth[d]["net"],
            }
            for d in range(MAX_DEPTH + 1)
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
