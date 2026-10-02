#!/usr/bin/env python3
"""Diagnostic (not part of frozen q1v3): per-row cycle-0 verify KL, candidate vs A, for selected cases."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import q1v3_common as Q
import reduce as RD

run = sys.argv[1]
for case in sys.argv[2:]:
    a, _ = RD.load_obs(run, "A", f"A__{case}__r0")
    x, _ = RD.load_obs(run, "CAND", f"CAND__{case}__r0")
    la = a.logits()
    rec = x.doc["records"]
    step0 = [r for r in rec["logits"] if r.get("step") == 0][0]
    t = x._t(step0["obj"]).float()
    nd = rec.get("natural_drafts", [])
    print(f"== {case}  nodes={step0['nodes']}  i={step0['i']}")
    print("   natural drafts step0 (first 12):", nd[0]["tokens"][:12] if nd else None)
    for row, (n, i) in enumerate(zip(step0["nodes"], step0["i"])):
        if i not in la:
            print(f"   node {n:2d} i={i}: no A logits"); continue
        pa = torch.log_softmax(la[i].double(), -1); px = torch.log_softmax(t[row].double(), -1)
        kl = float((pa.exp() * (pa - px)).sum())
        print(f"   node {n:2d} depth {Q.depth(n):2d} i={i:3d}  KL={kl:8.3f}  greedyA={int(la[i].argmax())} greedyX={int(t[row].argmax())}")
