#!/usr/bin/env python3
"""Diagnostic (not part of frozen q1v3): which tokens did the candidate's tree rows hold?
Natural TAW products (before forcing) report accepted rows + emitted tokens; an accepted row's emitted token
is the token that row consumed. Compare to forced row tokens and natural MTP drafts."""
import json, os, sys, glob
run = sys.argv[1]
tot = acc_rows = eq_forced = eq_nat = 0
for p in sorted(glob.glob(os.path.join(run, "CAND", "cases", "CAND__*__r0.json"))):
    d = json.load(open(p)); rec = d["records"]
    cyc = d.get("cycles") or d.get("plan", {}).get("cycles")
    nd = rec.get("natural_drafts", [])
    for t in rec.get("taw", []):
        j = t["call"]; nat = t["natural"]
        if j >= len(nd):
            continue
        drafts = nd[j]["tokens"]
        rows = nat["rows"][: nat["acc"]]
        for k, r in enumerate(rows):
            tok = nat["tokens"][k]
            acc_rows += 1
            eq_nat += int(r >= 1 and drafts[r - 1] == tok)
            ft = t.get("forced_row_tokens")
            if cyc:
                ft = cyc[j]["row_tokens"][r] if j < len(cyc) else None
                eq_forced += int(ft == tok)
        tot += 1
    if "C1-spine" in p and "cal-short" in p:
        for t in rec["taw"][:3]:
            print(os.path.basename(p), "call", t["call"], "natural acc", t["natural"]["acc"], "rows", t["natural"]["rows"][:t["natural"]["acc"]],
                  "tokens", t["natural"]["tokens"][:t["natural"]["acc"] + 1], "| nat drafts", nd[t["call"]]["tokens"][:8] if t["call"] < len(nd) else None)
print(f"taw calls {tot}; naturally accepted rows {acc_rows}; accepted token == natural MTP draft at that row: {eq_nat}; == forced row token: {eq_forced} (cycles in record: {bool(cyc)})")
print("record keys:", sorted(d.keys())[:20], sorted(rec.keys()))
