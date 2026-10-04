#!/usr/bin/env python3
"""Per-task outcomes, agent minutes and pooled decode rate (Eq. 5, server /metrics) for the penalty-fixed
LumoTree SWE arm, beside the pre-fix arms recorded in results/swe_study_20261001.json."""
import glob, json, os, re, subprocess
WT = "/home/mark/shared/lumotree-v2exp-20260930/scripts/v2exp"
root = "/home/mark/shared/lumotree-v2exp-runs/swe/"
new = sorted(glob.glob(root + "fr14_promoab_C_v2swe2026100[34]*/hydra27*/"))[-1]

def tasks(d):
    out = {}
    for l in open(d + "swe_orchestrator.log"):
        m = re.search(r"<- astropy__astropy-(\d+) verdict=(\w+) elapsed_total=([\d.]+)s", l)
        if m:
            out[m.group(1)] = {"verdict": m.group(2), "agent_s": float(m.group(3))}
    return out

def pooled(d):
    b, a = d + "metrics_before_swe.txt", d + "metrics_after_swe.txt"
    if not (os.path.exists(b) and os.path.exists(a)):
        return None
    return subprocess.run(["python3", WT + "/swe_pooled.py", b, a], capture_output=True, text=True).stdout.strip()

t = tasks(new)
old = json.load(open(WT + "/results/swe_study_20261001.json"))
print("new arm:", new)
print(f"tasks done {len(t)}/10, resolved {sum(v['verdict'] == 'resolved' for v in t.values())}, "
      f"agent minutes {sum(v['agent_s'] for v in t.values()) / 60:.1f}")
print("pooled:", pooled(new))
ot = old["lumotree"].get("tasks", {})
print(f"{'task':>6} {'fixed':>18} {'pre-fix':>18}")
for k in sorted(set(t) | set(ot)):
    a, b = t.get(k), ot.get(k)
    fa = f"{a['verdict']:>9} {a['agent_s'] / 60:5.1f}m" if a else "-"
    fb = f"{b['verdict']:>9} {b['agent_s'] / 60:5.1f}m" if b else "-"
    print(f"{k:>6} {fa:>18} {fb:>18}")
for arm in ("lumotree", "vllm_mtp5", "sglang_eagle_s7_d8"):
    o = old[arm]
    print(arm, {k: o.get(k) for k in ("resolved", "agent_min_total", "pooled")})
json.dump({"dir": new, "tasks": t, "resolved": sum(v["verdict"] == "resolved" for v in t.values()),
           "agent_min_total": round(sum(v["agent_s"] for v in t.values()) / 60, 2), "pooled": pooled(new)},
          open(WT + "/results/sampling_20261003/swe_penfix_arm.json", "w"), indent=1)
