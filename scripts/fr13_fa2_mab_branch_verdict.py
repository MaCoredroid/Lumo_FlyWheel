#!/usr/bin/env python3
"""Verdict parser for the FA2 MAB reorder+branch run.

Answers the user requirement DIRECTLY: does the spine-first reorder give BOTH the
spine AND the branches a proper (M-invariant, context-preserving) fix?

  SPINE  proper-fix gate: spine_all_vs_m6_max == 0.0 on every full-attn layer
         (every spine node bit-exact to the spine-only arm => M-invariant =>
          closes the cat8-vs-cat6 accept leak).
  BRANCH proper-fix gate: branch_coresident_max — each branch's output in full
         cat8 vs in a (spine + that branch only) tree. 0 => branch invariant to
         co-resident branches (rescues don't wobble with the branch set);
         ~1 bf16 ULP (<=6.3e-2 and << gross) => within-floor own-column residual
         that does NOT touch the spine accept; gross => branches NOT properly
         fixed (need fixed-slot branch layout).
  NON-CORRUPTION: nondeep_relabel_max — reorder must not corrupt non-deep rows.
"""
import json
import sys
import collections

ULP_BF16 = 6.25e-2  # 1 bf16 ULP at O(1) magnitude

recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
if not recs:
    print("NO EVENTS — harness did not fire (NOT a verdict)")
    sys.exit(3)

layers = sorted({r["layer_name"] for r in recs})
print(f"events={len(recs)}  distinct full-attn layers={len(layers)}")

spine_worst = collections.defaultdict(float)
deep_worst = collections.defaultdict(float)
branch_worst = collections.defaultdict(float)
relabel_worst = collections.defaultdict(float)
per_branch = collections.defaultdict(lambda: collections.defaultdict(float))
errs = []

for r in recs:
    L = r["layer_name"]
    rr = r.get("reorder_a_prime", {}) or {}
    if "err" in rr:
        errs.append((L, rr["err"]))
        continue
    spine_worst[L] = max(spine_worst[L], rr.get("spine_all_vs_m6_max", 0.0))
    deep_worst[L] = max(deep_worst[L], rr.get("deep_vs_m6", 0.0))
    relabel_worst[L] = max(relabel_worst[L], rr.get("nondeep_relabel_max", 0.0))
    if "branch_coresident_max" in rr:
        branch_worst[L] = max(branch_worst[L], rr["branch_coresident_max"])
    for c in rr.get("branch_check", []):
        per_branch[c["branch"]]["max"] = max(
            per_branch[c["branch"]]["max"], c["coresident_vs_solo_max"])
        per_branch[c["branch"]]["anc"] = c.get("ancestors")

def g(d):
    return max(d.values()) if d else float("nan")

spine_g, deep_g = g(spine_worst), g(deep_worst)
branch_g, relabel_g = g(branch_worst), g(relabel_worst)

print("\n--- SPINE proper-fix (M-invariance) ---")
print(f"  spine_all_vs_m6_max  (global worst over layers) = {spine_g:.3e}"
      f"   {'PASS bit-exact' if spine_g == 0.0 else 'NONZERO'}")
print(f"  deep_vs_m6           (global worst)             = {deep_g:.3e}")

print("\n--- BRANCH proper-fix (co-residency M-invariance) ---")
print(f"  branch_coresident_max (global worst over layers)= {branch_g:.3e}")
for b in sorted(per_branch):
    v = per_branch[b]["max"]
    tag = "bit-exact" if v == 0.0 else (
        "within-floor(<=1 ULP)" if v <= ULP_BF16 else "GROSS>1ULP")
    print(f"    branch node {b} (anc={per_branch[b]['anc']}): {v:.3e}  {tag}")

print("\n--- non-corruption ---")
print(f"  nondeep_relabel_max (global worst)              = {relabel_g:.3e}"
      f"   {'clean' if relabel_g <= ULP_BF16 else 'CORRUPT>1ULP'}")

if errs:
    print(f"\n  reorder arm errors on {len(errs)} events; first: {errs[0]}")

print("\n>>> VERDICT")
spine_ok = spine_g == 0.0
branch_ok = branch_g <= ULP_BF16
if spine_ok and branch_ok:
    print("  BOTH PROPER: spine bit-exact M-invariant (closes speed leak); "
          "branches within-floor (rescues stable across branch set).")
    print("  => reorder is a proper fix for BOTH. Greenlight bit-exact delivery "
          "(in-kernel within-block canonicalization or cache slot reorder).")
elif spine_ok and not branch_ok:
    print(f"  SPINE proper, BRANCH NOT ({branch_g:.3e} > 1 ULP): branch own-column "
          "M-dependence is gross => reorder alone under-fixes branches; also need "
          "fixed-slot branch layout. Investigate per-branch table above.")
else:
    print(f"  SPINE NOT bit-exact ({spine_g:.3e}) — reorder not landing; re-check "
          "spine derivation / engagement before any branch conclusion.")
