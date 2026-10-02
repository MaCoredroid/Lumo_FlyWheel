#!/usr/bin/env python3
"""Diagnostic: did the forced draft tokens reach the candidate's tree forward?"""
import json, glob, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q1v3_common as Q
run = sys.argv[1]
cs = {c["case_id"]: c for c in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.v1.json")))["cases"]}
n = collections.Counter(); shown = 0
for p in sorted(glob.glob(os.path.join(run, "CAND", "cases", "CAND__*__r0.json"))):
    d = json.load(open(p)); rec = d["records"]; cid = d["case_id"]
    if cid not in cs:
        continue
    cy = cs[cid]["cycles"]; nd = rec.get("natural_drafts", [])
    for e in rec.get("diag_tree_inputs", []):
        t = e["step"]
        if "err" in e:
            n["err"] += 1; print("ERR", cid, e); continue
        if t >= len(cy):
            continue
        want = cy[t]["row_tokens"]; got = e["ids"]; nat = nd[t]["tokens"] if t < len(nd) else None
        n["steps"] += 1
        n["root_ok"] += got[0] == want[0]
        n["drafts_all_forced"] += got[1:] == want[1:]
        n["drafts_all_natural"] += bool(nat) and got[1:] == nat
        n["rows_forced"] += sum(a == b for a, b in zip(got[1:], want[1:])); n["rows"] += len(want) - 1
        n["runner_eq_arg"] += e["runner_ids"] == got
        n["draft_t_forced"] += e.get("draft_t") == want[1:]
        if e.get("pos"):
            P0 = e["pos"][0]
            n["pos_depth_ok"] += [x - P0 for x in e["pos"]] == [Q.depth(r) for r in range(Q.N_ROWS)]
        if shown < 3:
            shown += 1
            print(cid, "step", t, "arg_none", e["arg_none"])
            print("  got   ", got[:12]); print("  forced", want[:12]); print("  natural", ([want[0]] + nat)[:12] if nat else None)
            print("  draft_t", (e.get("draft_t") or [])[:11]); print("  pos-P0", [x - e["pos"][0] for x in e["pos"]][:16] if e.get("pos") else None)
print(dict(n))
