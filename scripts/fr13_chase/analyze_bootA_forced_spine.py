#!/usr/bin/env python3
"""BOOT A decisive A/B (DIAGNOSTIC): forced-spine caterpillar vs chain boot.

Both windows are TREE boots (tree_path_lcp.jsonl committer rows). Lockstep
pairs events at identical (prompt_id, committed prefix) and compares the
NEXT-EVENT drafter spine tokens: caterpillar spine nodes [0,1,3,5,7] vs
chain nodes [0,1,2,3,4].  Token-identical at every depth on every lockstep
pair => the m1 contamination is entirely in branch-commit state advance
(the only machinery the forced-spine flag disables).

Also checks: forced_spine_commit self-description on every A row, alts still
scored (5 paths), served-stream first-fork per prompt, accept/event.
DIAGNOSTIC ONLY — never a serving/gate number (FR13_FORCE_SPINE_COMMIT=1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# DISC keeps pointing at the banked DATA artifacts (gitignored output/);
# the fr13_disc_lib import now resolves to the rescued co-located copy
# (scripts/fr13_chase/, tracked) so the reducer survives a disk loss.
DISC = Path("/home/mark/shared/lumoFlyWheel/output/fr13_s1s2s3_discriminate")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr13_disc_lib import (  # noqa: E402
    load_jsonl,
    load_probe,
    probe_records,
    tree_spine_draft,
    walk_tree_events,
)

CAT_SPINE = [0, 1, 3, 5, 7]
CHAIN_SPINE = [0, 1, 2, 3, 4]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-window", required=True, help="forced-spine caterpillar window")
    ap.add_argument("--a-probe-name", default="a_forced_spine_greedy_probe.json")
    ap.add_argument("--chain-window", default=str(DISC / "chain_greedy"))
    ap.add_argument("--chain-probe-name", default="chain_greedy_probe.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    awin, cwin = Path(args.a_window), Path(args.chain_window)
    aprobe = load_probe(awin / args.a_probe_name)
    arecords = probe_records(aprobe)
    aevents, adiag = walk_tree_events(load_jsonl(awin / "logs/tree_path_lcp.jsonl"), arecords)
    cprobe = load_probe(cwin / args.chain_probe_name)
    crecords = probe_records(cprobe)
    cevents, cdiag = walk_tree_events(load_jsonl(cwin / "logs/tree_path_lcp.jsonl"), crecords)

    # forced-spine self-description audit on EVERY A committer row
    a_rows = [ev["row"] for ev in aevents]
    forced_flag_rows = sum(1 for r in a_rows if r.get("forced_spine_commit") is True)
    winner_trace = awin / "logs/independent_winner_trace.jsonl"
    if winner_trace.exists():
        wrows = load_jsonl(winner_trace)
        policies = sorted({str(r.get("policy", "<missing>")) for r in wrows})
    else:
        policies = ["<independent_winner_trace.jsonl missing>"]
    # winner_path must always be a prefix of the spine node path [0,1,3,5,7]
    spine_path = [0, 1, 3, 5, 7]
    nonspine_winners = [
        {"prompt_id": ev["prompt_id"], "gen_pos": ev["gen_pos"], "winner_path": ev["row"]["winner_path"]}
        for ev in aevents
        if ev["row"]["winner_path"] != spine_path[: len(ev["row"]["winner_path"])]
    ]
    alts_scored = sum(1 for r in a_rows if len(r.get("path_scores", [])) == 5)

    # served-stream comparison per prompt
    streams = {}
    cindex_rec = {r["prompt_id"]: r for r in crecords}
    for rec in arecords:
        cr = cindex_rec[rec["prompt_id"]]
        assert rec["prompt_token_ids"] == cr["prompt_token_ids"], f"prompt {rec['prompt_id']} prompt_token_ids differ"
        at, ct = rec["token_ids"], cr["token_ids"]
        fork = next((i for i in range(min(len(at), len(ct))) if at[i] != ct[i]), None)
        streams[rec["prompt_id"]] = {
            "a_len": len(at), "chain_len": len(ct),
            "first_fork_pos": fork,
            "identical_stream": fork is None and len(at) == len(ct),
            "fork_tokens": None if fork is None else {"a": at[fork], "chain": ct[fork]},
        }

    # lockstep drafter comparison
    cindex = {(ev["prompt_id"], tuple(ev["prefix"])): ev for ev in cevents}
    depth_n = 5
    flips = [0] * depth_n
    compared = [0] * depth_n
    identical_events = 0
    flip_details = []
    pairs = 0
    for aev in aevents:
        cev = cindex.get((aev["prompt_id"], tuple(aev["prefix"])))
        if cev is None:
            continue
        pairs += 1
        aspine = tree_spine_draft(aev["row"], CAT_SPINE)
        cspine = tree_spine_draft(cev["row"], CHAIN_SPINE)
        row_flips = []
        for d in range(depth_n):
            compared[d] += 1
            if aspine[d] != cspine[d]:
                flips[d] += 1
                row_flips.append(d)
        if not row_flips:
            identical_events += 1
        else:
            flip_details.append(
                {
                    "prompt_id": aev["prompt_id"],
                    "gen_pos": aev["gen_pos"],
                    "event_in_prompt": aev["event_idx_in_prompt"],
                    "flip_depths": row_flips,
                    "a_spine": aspine,
                    "chain_draft": cspine,
                    "prev_a_winner": None,
                }
            )

    a_acc = [ev["row"]["accepted_len"] for ev in aevents]
    out = {
        "header": {
            "DIAGNOSTIC_ONLY": "FR13_FORCE_SPINE_COMMIT=1 boot - NEVER a serving/gate number",
            "a_window": str(awin),
            "chain_window": str(cwin),
            "a_flags": "caterpillar TREE, FR13_FORCE_SPINE_COMMIT=1, FR13_CONV_COMMITTED_PATH=0, BI=1, FR13_BI_TREE_ATTN=1, FR13_TREE_BONUS_SELF=1, GPU_UTIL=0.82",
            "chain_flags": "5-node chain TREE, BI=1, FR13_BI_TREE_ATTN=1, FR13_TREE_BONUS_SELF=1 (discriminate campaign boot1c)",
            "sampling": {"temperature": aprobe["temperature"], "top_p": aprobe["top_p"], "seed": aprobe["seed"]},
            "a_walk_diag": {"leftover_rows": adiag["leftover_rows"]},
            "chain_walk_diag": {"leftover_rows": cdiag["leftover_rows"]},
        },
        "forced_spine_audit": {
            "a_events": len(aevents),
            "rows_with_forced_spine_commit_true": forced_flag_rows,
            "policies_seen": policies,
            "nonspine_winner_paths": nonspine_winners,
            "rows_with_5_paths_scored": alts_scored,
        },
        "served_streams_vs_chain": streams,
        "lockstep": {
            "pairs": pairs,
            "a_events_total": len(aevents),
            "chain_events_total": len(cevents),
            "identical_draft_events": identical_events,
            "token_identical_all_pairs": identical_events == pairs,
            "per_depth_flips": {f"d{d}": f"{flips[d]}/{compared[d]}" for d in range(depth_n)},
            "flip_details": flip_details,
        },
        "a_accept_per_event_DIAGNOSTIC": sum(a_acc) / len(a_acc) if a_acc else None,
    }
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    brief = json.loads(json.dumps(out))
    brief["lockstep"]["flip_details"] = brief["lockstep"]["flip_details"][:10]
    print(json.dumps(brief, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
