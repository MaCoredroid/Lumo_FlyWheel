#!/usr/bin/env python3
"""Build the boots JSON for the frozen distribution-v3 analysis (DIST-PROTOCOL-v3.md).
Usage: build_boots_v3.py DIST_ROOT OUT_JSON"""
import glob, json, os, sys

ARMS = {"ar-v3s21": "AR", "ar-v3s22": "AR", "ar-v3s23": "AR", "ar-v3s24": "AR",
        "tree-v3a": "TREE", "tree-v3b": "TREE", "tree-v3c": "TREE",
        "mtp5-v3s31": "MTP5", "mtp5-v3s32": "MTP5", "ar-v3neg": "NEG"}

root, out = sys.argv[1], sys.argv[2]
boots = {}
for label, arm in ARMS.items():
    hits = sorted(glob.glob(os.path.join(root, f"{label}-sampled-*", "dist.jsonl")))
    if len(hits) != 1:
        sys.exit(f"{label}: expected exactly one run, found {hits}")
    rows = [json.loads(l) for l in open(hits[0])]
    errs = sum(1 for r in rows if r.get("error"))
    if len(rows) != 800 or errs:
        sys.exit(f"{label}: {len(rows)} rows, {errs} errors (expected 800, 0)")
    boots[label] = {"arm": arm, "path": hits[0]}
    print(f"{label:12s} {arm:5s} {len(rows)} rows  {hits[0]}")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(boots, open(out, "w"), indent=1)
print("wrote", out)
