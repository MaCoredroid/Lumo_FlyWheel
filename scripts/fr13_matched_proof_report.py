#!/usr/bin/env python3
"""Matched proof: is cat8-SPINE accept == native (superset)? do BRANCHES rescue? cat8 > native?
Per-mode cat8 spine/branch = brhist snapshot delta; native = accept_speed json. B=1 (no batching)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fr13_branch_rescue_report import flat_row_classes, TREES

def load(p):
    try: return json.load(open(p))
    except Exception: return None

def delta(a, b):
    ab=(b or {}).get("rows",{}); aa=(a or {}).get("rows",{}) if a else {}
    rows={int(k): int(ab.get(k,0))-int((aa or {}).get(k,0)) for k in set(ab)|set(aa)}
    steps=(b or {}).get("steps",0) - ((a or {}).get("steps",0) if a else 0)
    return rows, steps

def main():
    RR=sys.argv[1]; cls=flat_row_classes(TREES["cat8"])
    spine={r for r,c in cls.items() if c[0]=="spine"}
    print(f"cat8 spine rows={sorted(spine)}  branch rows={sorted(r for r,c in cls.items() if c[0]=='branch')}\n")
    print(f"{'mode':<8}{'A_native':>9}{'cat8_spine':>11}{'cat8_branch(rescue)':>20}{'cat8_total':>11}{'spine-native':>13}")
    prev=None
    for m in ["greedy","temp06","temp10"]:
        cur=load(f"{RR}/probe_cat8/brhist_{m}.json"); rows,steps=delta(prev,cur); prev=cur
        if steps<=0: print(f"  {m:<8} no cat8 hist delta (steps={steps}) — VACUOUS/engage-fail"); continue
        sp=sum(v for r,v in rows.items() if r in spine); br=sum(v for r,v in rows.items() if r in cls and r not in spine)
        As, Ab = sp/steps, br/steps
        nat=load(f"{RR}/probe_native/accept_speed_{m}.json"); An=nat.get("accept_per_forward") if nat else None
        d=(As-An) if An is not None else 0.0
        print(f"  {m:<8}{(An or 0):>9.3f}{As:>11.3f}{Ab:>20.3f}{As+Ab:>11.3f}{d:>+13.3f}")
    print("\nREAD: spine-native ~0 => cat8 SPINE == native (superset holds); branch>0 => rescue; total>native => cat8>native PROVEN.")
    print("      spine-native <0 (materially) => spine PERTURBED (real M-dep, NOT batching — B=1 here).")

if __name__=="__main__": main()
