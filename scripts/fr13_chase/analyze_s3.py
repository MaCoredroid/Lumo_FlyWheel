#!/usr/bin/env python3
"""S3 lockstep drafter comparison: tree-arm spine drafts vs native chain drafts
on identical committed prefixes.  Per-depth flip counts + first-flip top1/top2
swap classification (caterpillar alts).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fr13_disc_lib import (  # noqa: E402
    CATERPILLAR_ALT_AT_DEPTH,
    load_jsonl,
    load_probe,
    probe_records,
    tree_spine_draft,
    walk_native_events,
    walk_tree_events,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree-window", required=True)
    ap.add_argument("--tree-probe-name", required=True)
    ap.add_argument("--native-window", required=True)
    ap.add_argument("--native-probe-name", required=True)
    ap.add_argument("--spine", default="0,1,3,5,7")
    ap.add_argument("--alts", default="caterpillar", choices=["caterpillar", "none"])
    ap.add_argument("--native-num-spec", type=int, default=5)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spine = [int(x) for x in args.spine.split(",")]
    twin, nwin = Path(args.tree_window), Path(args.native_window)
    tprobe = load_probe(twin / args.tree_probe_name)
    trecords = probe_records(tprobe)
    tevents, _ = walk_tree_events(load_jsonl(twin / "logs/tree_path_lcp.jsonl"), trecords)
    nprobe = load_probe(nwin / args.native_probe_name)
    nrecords = probe_records(nprobe)
    nevents, ndiag = walk_native_events(
        load_jsonl(nwin / "logs/per_req_spec_trace.jsonl"),
        load_jsonl(nwin / "logs/fr10_mtp_draft_trace.jsonl"),
        nrecords,
        num_spec=args.native_num_spec,
    )

    nindex = {(ev["prompt_id"], tuple(ev["prefix"])): ev for ev in nevents}
    pairs = []
    for tev in tevents:
        nev = nindex.get((tev["prompt_id"], tuple(tev["prefix"])))
        if nev is not None:
            pairs.append((tev, nev))

    depth_n = len(spine)
    flips = [0] * depth_n
    compared = [0] * depth_n
    first_flips = []
    identical_events = 0
    per_pair = []
    for tev, nev in pairs:
        tspine = tree_spine_draft(tev["row"], spine)
        ndraft = nev["draft"]
        row_flips = []
        for d in range(min(depth_n, len(ndraft))):
            compared[d] += 1
            if tspine[d] != ndraft[d]:
                flips[d] += 1
                row_flips.append(d)
        if not row_flips:
            identical_events += 1
        else:
            d0 = row_flips[0]
            entry = {
                "prompt_id": tev["prompt_id"],
                "tree_event_in_prompt": tev["event_idx_in_prompt"],
                "gen_pos": tev["gen_pos"],
                "first_flip_depth": d0,
                "tree_token": tspine[d0],
                "native_token": ndraft[d0],
                "all_flip_depths": row_flips,
            }
            if args.alts == "caterpillar" and d0 in CATERPILLAR_ALT_AT_DEPTH:
                alt_tok = int(tev["row"]["draft_token_ids"][CATERPILLAR_ALT_AT_DEPTH[d0]])
                entry["alt_token"] = alt_tok
                entry["top1_top2_swap"] = alt_tok == ndraft[d0]
            first_flips.append(entry)
        per_pair.append(
            {
                "prompt_id": tev["prompt_id"],
                "gen_pos": tev["gen_pos"],
                "tree_spine": tspine,
                "native_draft": list(ndraft),
                "flip_depths": row_flips,
                "tree_acc": tev["row"]["accepted_len"],
                "native_acc": nev["acc"],
            }
        )

    n_pairs = len(pairs)
    out = {
        "header": {
            "label": args.label,
            "tree_window": str(twin),
            "native_window": str(nwin),
            "spine_nodes": spine,
            "sampling": {"temperature": tprobe["temperature"], "top_p": tprobe["top_p"], "seed": tprobe["seed"]},
            "native_walk_diag": ndiag,
        },
        "lockstep_pairs": n_pairs,
        "tree_events_total": len(tevents),
        "native_events_total": len(nevents),
        "identical_draft_events": identical_events,
        "identical_rate": identical_events / n_pairs if n_pairs else None,
        "per_depth_flips": {f"d{d}": f"{flips[d]}/{compared[d]}" for d in range(depth_n)},
        "per_depth_flip_counts": flips,
        "per_depth_compared": compared,
        "first_flips": first_flips,
        "swap_summary": {
            "first_flips_at_d_ge1": sum(1 for f in first_flips if f["first_flip_depth"] >= 1),
            "top1_top2_swaps": sum(1 for f in first_flips if f.get("top1_top2_swap")),
        },
        "mean_accept_on_pairs": {
            "tree": sum(p["tree_acc"] for p in per_pair) / n_pairs if n_pairs else None,
            "native": sum(p["native_acc"] for p in per_pair) / n_pairs if n_pairs else None,
        },
        "pairs": per_pair,
    }
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in {"pairs", "first_flips"}}, indent=1))
    print("first_flips:", json.dumps(first_flips, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
