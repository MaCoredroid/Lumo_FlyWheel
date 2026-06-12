#!/usr/bin/env python3
"""BOOT B (FR13_CONV_COMMITTED_PATH=1, normal commits) battery analysis.

(b) accept/event vs 1.819 (pre-fix caterpillar) / 2.277 (chain)
(c) serve-stream forks vs native_greedy BI=0 reference (per-prompt first fork)
(d) same-seed repeat determinism (token streams byte-identical?)
Flag state: caterpillar TREE, FR13_CONV_COMMITTED_PATH=1, FR13_FORCE_SPINE_COMMIT=0,
BI=1, FR13_BI_TREE_ATTN=1, FR13_TREE_BONUS_SELF=1, greedy 0.0/1.0/seed 1313.
(S2 logit check is analyze_s2.py from the discriminate campaign, run separately.)
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
    walk_tree_events,
)


def first_fork(a: list[int], b: list[int]) -> int | None:
    return next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-window", required=True)
    ap.add_argument("--b-probe-name", default="b_greedy_probe.json")
    ap.add_argument("--rep2-window", required=True)
    ap.add_argument("--rep2-probe-name", default="b_greedy_rep2_probe.json")
    ap.add_argument("--native-ref-probe",
                    default="/home/mark/shared/lumoFlyWheel/output/fr13_acceptance_ladder/native_greedy/native_greedy_probe.json")
    ap.add_argument("--prefix-tree-probe",
                    default=str(DISC / "tree_greedy/tree_greedy_probe.json"),
                    help="pre-fix caterpillar window (fork context)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bwin = Path(args.b_window)
    bprobe = load_probe(bwin / args.b_probe_name)
    brecords = probe_records(bprobe)
    bevents, bdiag = walk_tree_events(load_jsonl(bwin / "logs/tree_path_lcp.jsonl"), brecords)

    rwin = Path(args.rep2_window)
    rprobe = load_probe(rwin / args.rep2_probe_name)
    rrecords = probe_records(rprobe)

    nprobe = load_probe(args.native_ref_probe)
    nrecords = probe_records(nprobe)
    pprobe = load_probe(args.prefix_tree_probe)
    precords = probe_records(pprobe)

    # (b) accept/event
    acc = [ev["row"]["accepted_len"] for ev in bevents]
    accept_per_event = sum(acc) / len(acc) if acc else None
    # full-window from raw committer rows too (incl. trailing clipped rows)
    raw_rows = load_jsonl(bwin / "logs/tree_path_lcp.jsonl")
    raw_acc = [r["accepted_len"] for r in raw_rows]

    # forced-spine must be OFF: audit self-description
    forced_rows = sum(1 for r in raw_rows if r.get("forced_spine_commit"))
    branch_wins = sum(
        1 for ev in bevents
        if ev["row"]["accepted_len"] > 0
        and ev["row"]["winner_path"][: ev["row"]["accepted_len"]] != [0, 1, 3, 5, 7][: ev["row"]["accepted_len"]]
    )

    # (c) forks vs native BI=0 reference (+ pre-fix tree forks for context)
    nidx = {r["prompt_id"]: r for r in nrecords}
    pidx = {r["prompt_id"]: r for r in precords}
    forks = {}
    for rec in brecords:
        nr = nidx[rec["prompt_id"]]
        assert rec["prompt_token_ids"] == nr["prompt_token_ids"], f"prompt {rec['prompt_id']} tokens differ vs native ref"
        f_nat = first_fork(rec["token_ids"], nr["token_ids"])
        pr = pidx.get(rec["prompt_id"])
        f_pre = first_fork(pr["token_ids"], nr["token_ids"]) if pr else None
        forks[rec["prompt_id"]] = {
            "bootB_vs_nativeBI0_first_fork": f_nat,
            "prefix_tree_vs_nativeBI0_first_fork": f_pre,
            "bootB_match_len": f_nat if f_nat is not None else min(len(rec["token_ids"]), len(nr["token_ids"])),
            "fork_tokens": None if f_nat is None else {"tree": rec["token_ids"][f_nat], "native": nr["token_ids"][f_nat]},
        }

    # (d) determinism: rep2 streams byte-identical?
    ridx = {r["prompt_id"]: r for r in rrecords}
    det = {}
    for rec in brecords:
        rr = ridx[rec["prompt_id"]]
        f = first_fork(rec["token_ids"], rr["token_ids"])
        det[rec["prompt_id"]] = {
            "identical": f is None and len(rec["token_ids"]) == len(rr["token_ids"]),
            "first_diff": f,
        }

    out = {
        "header": {
            "b_window": str(bwin),
            "rep2_window": str(rwin),
            "flag_state": "caterpillar TREE, FR13_CONV_COMMITTED_PATH=1, FR13_FORCE_SPINE_COMMIT=0, BI=1, FR13_BI_TREE_ATTN=1, FR13_TREE_BONUS_SELF=1, FR10_METRICS=1, GPU_UTIL=0.82",
            "sampling": {"temperature": bprobe["temperature"], "top_p": bprobe["top_p"], "seed": bprobe["seed"]},
            "walk_diag": {"leftover_rows": bdiag["leftover_rows"]},
            "references": {
                "pre_fix_caterpillar_accept_per_event": 1.8186813186813187,
                "chain_accept_per_event": 2.2767295597484276,
                "native_ref": args.native_ref_probe,
            },
        },
        "forced_spine_audit": {"rows_with_forced_spine_commit_true": forced_rows, "branch_winner_events": branch_wins},
        "accept_per_event": {
            "walked_events": len(bevents),
            "value": accept_per_event,
            "raw_rows": len(raw_rows),
            "raw_value": sum(raw_acc) / len(raw_acc) if raw_acc else None,
            "vs_prefix_caterpillar_1p819": (accept_per_event - 1.8186813186813187) if accept_per_event else None,
            "vs_chain_2p277": (accept_per_event - 2.2767295597484276) if accept_per_event else None,
        },
        "forks_vs_nativeBI0": forks,
        "determinism_rep2": det,
        "determinism_all_identical": all(d["identical"] for d in det.values()),
    }
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
