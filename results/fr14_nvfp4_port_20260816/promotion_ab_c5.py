#!/usr/bin/env python3
"""c5 = seam conditional: delta accepted[pos5] / delta accepted[pos4], per task.
Healthy corridor [0.40, 0.70]. Computed from the per-task metrics_pre/post bracket."""
import re, sys, json
from pathlib import Path
PAT = re.compile(r'spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"\}\s+([0-9.]+)')
def pos(p):
    d = {}
    if p.is_file():
        for m in PAT.finditer(p.read_text(errors="replace")):
            d[int(m.group(1))] = float(m.group(2))
    return d
def main(root):
    base = Path(root)/"swe_out/verified/per_task"
    out = {}
    for t in sorted(p.name for p in base.glob("*") if p.is_dir()):
        pre, post = pos(base/t/"vllm_metrics_pre.txt"), pos(base/t/"vllm_metrics_post.txt")
        if not pre or not post:
            out[t] = {"c5": None, "note": "bracket missing"}; continue
        d4 = post.get(4,0)-pre.get(4,0); d5 = post.get(5,0)-pre.get(5,0)
        c5 = (d5/d4) if d4 else None
        rec = {"delta_pos4": d4, "delta_pos5": d5,
               "c5": round(c5,4) if c5 is not None else None}
        if c5 is not None:
            rec["in_corridor"] = 0.40 <= c5 <= 0.70
            rec["FLAG"] = None if rec["in_corridor"] else ("BELOW" if c5 < 0.40 else "ABOVE")
        out[t] = rec
    print(json.dumps(out, indent=1))
if __name__ == "__main__":
    main(sys.argv[1])
