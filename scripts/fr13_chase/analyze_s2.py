#!/usr/bin/env python3
"""S2 CAPTURE: locate episodic verify-forward corruption via lockstep fp32 logits.

Pairs tree verify events (10-row captures) with native verify events (6-row
captures) at identical committed prefixes; compares the root row (always
input-identical) and chain/spine rows while the tree spine drafts match the
native chain drafts.  Flags events whose whole-forward delta is an outlier
(event-7 class: mean|d| ~10-25x baseline) and records trigger context.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from fr13_disc_lib import (  # noqa: E402
    load_jsonl,
    load_probe,
    probe_records,
    tree_spine_draft,
    walk_native_events,
    walk_tree_events,
)


def load_capture(dirpath: Path, prefix: str, call: int) -> torch.Tensor | None:
    p = dirpath / f"{prefix}.call{call}.pt"
    if not p.exists():
        return None
    d = torch.load(p, map_location="cpu", weights_only=False)
    return d["logits"].float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree-window", required=True, help="dir with probe json + logs/")
    ap.add_argument("--tree-probe-name", default="tree_greedy_probe.json")
    ap.add_argument("--native-window", required=True)
    ap.add_argument("--native-probe-name", default="native_greedy_probe.json")
    ap.add_argument("--tree-capture-prefix", default="tree_final_logits")
    ap.add_argument("--native-capture-prefix", default="native_final_logits")
    ap.add_argument("--tree-capture-call-offset", type=int, default=0,
                    help="capture call index of the window's first verify event")
    ap.add_argument("--native-capture-call-offset", type=int, default=0)
    ap.add_argument("--native-align-map", default=None,
                    help="JSON from build_align_map.py; overrides the native offset "
                         "(handles trailing request-boundary captures)")
    ap.add_argument("--spine", default="0,1,3,5,7")
    ap.add_argument("--native-num-spec", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spine = [int(x) for x in args.spine.split(",")]
    twin = Path(args.tree_window)
    nwin = Path(args.native_window)
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

    nalign = None
    if args.native_align_map:
        nalign = {
            int(k): int(v)
            for k, v in json.loads(Path(args.native_align_map).read_text())["map"].items()
        }

    results = []
    missing_captures = 0
    for tev, nev in pairs:
        tlog = load_capture(twin / "logs", args.tree_capture_prefix,
                            tev["global_event_idx"] + args.tree_capture_call_offset)
        if nalign is not None:
            ncall = nalign.get(nev["global_event_idx"])
        else:
            ncall = nev["global_event_idx"] + args.native_capture_call_offset
        nlog = (load_capture(nwin / "logs", args.native_capture_prefix, ncall)
                if ncall is not None else None)
        if tlog is None or nlog is None:
            missing_captures += 1
            continue
        row = tev["row"]
        tspine = tree_spine_draft(row, spine)
        ndraft = nev["draft"]
        # comparable rows: root (idx 0 both) + depth d while spine[:d+1]==chain[:d+1]
        comp = [(0, 0, "root")]
        for d in range(min(len(spine), len(ndraft))):
            if tspine[: d + 1] != ndraft[: d + 1]:
                break
            comp.append((spine[d] + 1, d + 1, f"d{d}"))
        rows_out = []
        for trow, nrow, label in comp:
            if trow >= tlog.shape[0] or nrow >= nlog.shape[0]:
                continue
            tv, nv = tlog[trow], nlog[nrow]
            delta = (tv - nv).abs()
            n_top2 = torch.topk(nv, 2).values
            rows_out.append(
                {
                    "row": label,
                    "mean_abs_delta": float(delta.mean()),
                    "max_abs_delta": float(delta.max()),
                    "tree_argmax": int(tv.argmax()),
                    "native_argmax": int(nv.argmax()),
                    "argmax_match": int(tv.argmax()) == int(nv.argmax()),
                    "native_margin": float(n_top2[0] - n_top2[1]),
                }
            )
        if not rows_out:
            continue
        results.append(
            {
                "prompt_id": tev["prompt_id"],
                "tree_event": tev["global_event_idx"],
                "native_event": nev["global_event_idx"],
                "tree_event_in_prompt": tev["event_idx_in_prompt"],
                "gen_pos": tev["gen_pos"],
                "rows_compared": len(rows_out),
                "event_mean_abs_delta": statistics.fmean(r["mean_abs_delta"] for r in rows_out),
                "event_max_abs_delta": max(r["max_abs_delta"] for r in rows_out),
                "any_argmax_flip": any(not r["argmax_match"] for r in rows_out),
                "flip_rows": [r for r in rows_out if not r["argmax_match"]],
                "rows": rows_out,
            }
        )

    means = [r["event_mean_abs_delta"] for r in results]
    baseline = statistics.median(means) if means else None
    corrupt = []
    for i, r in enumerate(results):
        is_outlier = baseline is not None and baseline > 0 and r["event_mean_abs_delta"] > 8 * baseline
        gross_flip = any(fr["native_margin"] > 1.0 for fr in r["flip_rows"])
        if is_outlier or gross_flip:
            # trigger context from the tree stream
            tev = next(t for t, _ in pairs if t["global_event_idx"] == r["tree_event"])
            prev = None
            for t in reversed([t for t, _ in pairs]):
                pass
            # find previous tree event in the same prompt
            prev_rows = [
                t for t in (p[0] for p in pairs)
                if t["prompt_id"] == r["prompt_id"]
                and t["event_idx_in_prompt"] == r["tree_event_in_prompt"] - 1
            ]
            ctx = {}
            if prev_rows:
                pr = prev_rows[0]["row"]
                ctx = {
                    "prev_accepted_len": pr["accepted_len"],
                    "prev_winner_path": pr["winner_path"],
                    "prev_bonus_source": pr["bonus_source"],
                    "prev_was_02_winner": pr["winner_path"] == [0, 2],
                }
            corrupt.append({**r, "outlier_vs_baseline": is_outlier, "gross_flip": gross_flip,
                            "trigger_context": ctx})

    out = {
        "header": {
            "tree_window": str(twin),
            "native_window": str(nwin),
            "greedy": {"temperature": tprobe["temperature"], "top_p": tprobe["top_p"], "seed": tprobe["seed"]},
            "flag_state": "tree: FR13_TREE_BONUS_SELF=1 BI=1 FR13_BI_TREE_ATTN=1; native window flags per run_header.json",
            "native_walk_diag": ndiag,
        },
        "lockstep_pairs": len(pairs),
        "events_compared": len(results),
        "missing_captures": missing_captures,
        "baseline_event_mean_abs_delta_median": baseline,
        "event_mean_abs_delta_max": max(means) if means else None,
        "corrupt_events": corrupt,
        "all_events": results,
    }
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    brief = {k: v for k, v in out.items() if k != "all_events"}
    brief["corrupt_events"] = [
        {k: v for k, v in c.items() if k != "rows"} for c in corrupt
    ]
    print(json.dumps(brief, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
