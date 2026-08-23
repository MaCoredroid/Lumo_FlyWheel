#!/usr/bin/env python3
"""ARM 0 readout: are the two 13236 trajectories BIT-IDENTICAL?

Compares, per task, the artifacts that ARE the trajectory -- the generation trace
and the emitted patch -- by sha256. Reports per-task identical/divergent, and for a
divergent trace, the first differing record index so the divergence point is named
rather than merely asserted.
"""
import hashlib, json, sys
from pathlib import Path

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None

def main(a, b):
    ra, rb = Path(a), Path(b)
    tasks = sorted({p.name for r in (ra, rb) for p in (r/"swe_out/verified/per_task").glob("*") if p.is_dir()})
    out = {"schema": "fr14.arm0.determinism.v1", "replicate_a": a, "replicate_b": b, "tasks": {}}
    verdicts = []
    for t in tasks:
        rec = {}
        for label, root in (("trace", "qwen_trace.jsonl"), ("patch", "patch.diff")):
            pa, pb = ra/"swe_out/verified/per_task"/t/root, rb/"swe_out/verified/per_task"/t/root
            sa, sb = sha(pa), sha(pb)
            rec[label] = {"a": (sa or "ABSENT")[:16], "b": (sb or "ABSENT")[:16],
                          "identical": sa is not None and sa == sb}
        # name the divergence point when the traces differ
        pa, pb = ra/"swe_out/verified/per_task"/t/"qwen_trace.jsonl", rb/"swe_out/verified/per_task"/t/"qwen_trace.jsonl"
        if pa.is_file() and pb.is_file() and not rec["trace"]["identical"]:
            la, lb = pa.read_text(errors="replace").splitlines(), pb.read_text(errors="replace").splitlines()
            first = next((i for i in range(max(len(la), len(lb)))
                          if i >= len(la) or i >= len(lb) or la[i] != lb[i]), None)
            rec["first_differing_record"] = first
            rec["n_records"] = {"a": len(la), "b": len(lb)}
        rec["IDENTICAL"] = rec["trace"]["identical"] and rec["patch"]["identical"]
        out["tasks"][t] = rec
        verdicts.append(rec["IDENTICAL"])
    out["ALL_IDENTICAL"] = bool(verdicts) and all(verdicts)
    out["VERDICT"] = ("determinism HOLDS -- bisection converges" if out["ALL_IDENTICAL"]
                      else "DIVERGENT -- stop; seeding must be pinned before further bisection GPU")
    print(json.dumps(out, indent=1))
    return 0 if out["ALL_IDENTICAL"] else 1

if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
