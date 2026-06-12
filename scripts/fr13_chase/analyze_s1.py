#!/usr/bin/env python3
"""S1 RE-GATE: assert the committer bonus row fix on a live greedy window.

Checks (tree_path_lcp.jsonl + probe records):
  1. every fully-accepted path's served bonus == self_target[last accepted node];
     bonus_source == 'tree_self_target' (no 'path0_native_bonus' rows at all);
  2. [0,2]-winner (acc==2) events: served bonus == st[2], and report where
     st[2] != native_bonus_token (the events the legacy bug would have corrupted);
  3. spine_path_idx/spine_leaf fields correct for the topology;
  4. fork positions of each prompt vs the reference native greedy streams
     (prompt-3 pos14 must heal);
  5. accept/event vs reference tree (pre-fix) and native.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fr13_disc_lib import (  # noqa: E402
    CATERPILLAR_SPINE,
    load_jsonl,
    load_probe,
    probe_records,
    walk_tree_events,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--lcp", required=True)
    ap.add_argument("--native-probe", required=True)
    ap.add_argument("--ref-tree-probe")
    ap.add_argument("--spine", default="0,1,3,5,7")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spine = [int(x) for x in args.spine.split(",")]
    probe = load_probe(args.probe)
    records = probe_records(probe)
    lcp_rows = load_jsonl(args.lcp)
    events, walk_diag = walk_tree_events(lcp_rows, records)

    violations = []
    full_accepts = 0
    alt02_events = []
    bonus_sources = {}
    spine_field_bad = []
    superset_violations = 0
    for ev in events:
        row = ev["row"]
        bonus_sources[row["bonus_source"]] = bonus_sources.get(row["bonus_source"], 0) + 1
        acc = row["accepted_len"]
        wp = row["winner_path"]
        st = row["self_target_ids"]
        emitted = row["emitted_tokens"]
        if row.get("superset_violation"):
            superset_violations += 1
        # spine fields (topology-derived, constant)
        expect_spine_leaf = spine[-1]
        if row.get("spine_leaf") != expect_spine_leaf:
            spine_field_bad.append(
                {"global_event_idx": ev["global_event_idx"], "spine_leaf": row.get("spine_leaf")}
            )
        if acc == len(wp) and acc > 0:
            full_accepts += 1
            expect_bonus = st[wp[-1]]
            ok = emitted[-1] == expect_bonus and row["bonus_source"] == "tree_self_target"
            entry = {
                "global_event_idx": ev["global_event_idx"],
                "prompt_id": ev["prompt_id"],
                "gen_pos": ev["gen_pos"],
                "winner_path": wp,
                "served_bonus": emitted[-1],
                "expected_st_last": expect_bonus,
                "native_bonus_token": row["native_bonus_token"],
                "bonus_source": row["bonus_source"],
                "clipped": ev["clipped"],
                "ok": ok,
            }
            if ev["clipped"] and len(emitted) - 1 >= ev["n_emitted_served"]:
                # bonus never served (request hit max_tokens) - still assert log row
                entry["bonus_served_to_stream"] = False
            if not ok:
                violations.append(entry)
            if wp == [0, 2]:
                alt02_events.append(entry)

    nat_probe = load_probe(args.native_probe)
    nat_records = probe_records(nat_probe)
    forks = []
    for rec, nrec in zip(records, nat_records):
        assert rec["prompt_id"] == nrec["prompt_id"]
        a, b = rec["token_ids"], nrec["token_ids"]
        fork = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
        forks.append(
            {
                "prompt_id": rec["prompt_id"],
                "fork_pos": fork,
                "tree_token_at_fork": a[fork] if fork is not None else None,
                "native_token_at_fork": b[fork] if fork is not None else None,
                "match_len": fork if fork is not None else min(len(a), len(b)),
                "exact_match": fork is None and len(a) == len(b),
            }
        )

    mode = list(probe["modes"]) [0]
    summ = probe["modes"][mode]
    out = {
        "header": {
            "window_probe": args.probe,
            "temperature": probe["temperature"],
            "top_p": probe["top_p"],
            "seed": probe["seed"],
            "mode": mode,
            "spine_nodes": spine,
            "flag_state": "FR13_TREE_BONUS_SELF=1 (S1 fix ON), BATCH_INVARIANT=1, FR13_BI_TREE_ATTN=1, FR10_METRICS=1",
        },
        "events_walked": len(events),
        "committer_rows": len(lcp_rows),
        "walk_diag": walk_diag,
        "bonus_sources": bonus_sources,
        "full_accept_events": full_accepts,
        "bonus_violations": violations,
        "alt02_winner_events": alt02_events,
        "alt02_count": len(alt02_events),
        "alt02_all_serve_st2": all(e["ok"] for e in alt02_events),
        "alt02_discriminating": [
            e for e in alt02_events if e["expected_st_last"] != e["native_bonus_token"]
        ],
        "spine_field_bad": spine_field_bad,
        "superset_violations": superset_violations,
        "forks_vs_native_reference": forks,
        "accept_per_event": summ["accepted_per_draft_event"],
        "accept_per_event_reference": {
            "tree_prefix_greedy": 2.0817610062893084,
            "native_greedy": 3.154471544715447,
        },
        "spec_drafts": summ["spec_drafts"],
        "spec_accepted_tokens": summ["spec_accepted_tokens"],
    }
    if args.ref_tree_probe:
        ref = load_probe(args.ref_tree_probe)
        rmode = list(ref["modes"])[0]
        out["ref_tree_accept_per_event"] = ref["modes"][rmode]["accepted_per_draft_event"]
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in {"alt02_winner_events", "forks_vs_native_reference"}}, indent=1))
    print("forks:", json.dumps(out["forks_vs_native_reference"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
