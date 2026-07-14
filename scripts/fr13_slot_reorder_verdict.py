#!/usr/bin/env python3
"""4-arm campaign verdict table for the FR13_SLOT_REORDER live goal gate.

Aggregates per arm: resolve/give-ups (runlog verdict blocks), accept_per_event +
deploy speed (deploy_speed_b4.json), sidecar ms/step + ms/draft, engagement audit
(docker logs are gone by now — uses the armdir's saved docker_full.log), and the
trace-scan garble signature count (heuristic; native arm = matched null).

Usage: fr13_slot_reorder_verdict.py <runroot>   (default output/fr13_slot_reorder/live_b4)
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "output/fr13_slot_reorder/live_b4")
ARMS = [
    ("cat8+fix", "slreorder_cat8_cache_b4"),
    ("native", "native_ourcache_b4"),
    ("cat6+fix", "slreorder_cat6_cache_b4"),
    ("t33333+fix", "slreorder_33333_cache_b4"),
]


def arm_row(label, arm):
    row = {"arm": label, "done": False}
    rl = ROOT / f"{arm}.runlog"
    if rl.exists():
        txt = rl.read_text(errors="ignore")
        row["done"] = f"ARM_DONE {arm}" in txt
        seen = {}
        for m in re.finditer(
            r'"instance_id":\s*"([^"]+)".*?"patch_bytes":\s*(\d+),\s*"verdict":\s*"([^"]+)"',
            txt, re.S,
        ):
            seen[m.group(1)] = (int(m.group(2)), m.group(3))
        if seen:
            row["tasks"] = len(seen)
            row["resolved"] = sum(1 for pb, v in seen.values() if v == "resolved")
            row["giveups"] = sum(1 for pb, v in seen.values() if pb == 0)
    dj = ROOT / arm / "deploy_speed_b4.json"
    if dj.exists():
        d = json.loads(dj.read_text())
        for k in ("accept_per_event", "s_per_fwd_gpu", "derived_tps_gpu", "prefill_frac"):
            if d.get(k) is not None:
                row[k] = round(d[k], 4)
    for sc in Path("output/fr13_sfwd_sidecar").glob(f"{arm}.json*"):
        try:
            s = json.loads(sc.read_text())
            row["ms_step"] = round(s["decode_forward_gpu_seconds"] / s["n_pure_decode_steps_timed"] * 1000, 1)
            row["ms_draft"] = round(s["decode_forward_gpu_seconds"] / s["n_drafts_in_timed_steps"] * 1000, 1)
        except Exception:
            pass
    dl = ROOT / arm / "docker_full.log"
    if dl.exists():
        n = 0
        eng = set()
        for line in open(dl, errors="ignore"):
            if "FR13_SLOT_REORDER ENGAGED" in line:
                m = re.search(r"ENGAGED \(([a-z_ ]+)\): tree_n=(\d+) pi=(\[[^\]]*\])", line)
                if m:
                    eng.add((m.group(1), m.group(2), m.group(3)))
            if "EngineDead" in line or "committer rows" in line:
                n += 1
        row["engagement"] = sorted(eng) if eng else "NONE"
        row["crash_lines"] = n
    return row


def main():
    rows = [arm_row(l, a) for l, a in ARMS]
    hdr = ["arm", "done", "resolved", "giveups", "accept_per_event", "derived_tps_gpu",
           "prefill_frac", "ms_step", "ms_draft", "crash_lines"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        print("| " + " | ".join(str(r.get(k, "—")) for k in hdr) + " |")
    print()
    for r in rows:
        e = r.get("engagement")
        tree = r["arm"] != "native"
        if e == "NONE" and tree and r.get("done"):
            print(f"!! {r['arm']}: NO engagement lines — VACUOUS ARM")
        elif e and e != "NONE" and not tree:
            print(f"!! native: engagement lines present — CONTAMINATED")
        elif e and e != "NONE":
            print(f"   {r['arm']}: engaged {e}")
    # superset math
    by = {r["arm"]: r for r in rows}
    c8, c6, nat = by.get("cat8+fix", {}), by.get("cat6+fix", {}), by.get("native", {})
    if all(x.get("accept_per_event") for x in (c8, c6, nat)):
        print(f"\nSUPERSET: cat8-cat6 = {c8['accept_per_event']-c6['accept_per_event']:+.3f} "
              f"(predict ~+0.17) | cat8-native = {c8['accept_per_event']-nat['accept_per_event']:+.3f}")


if __name__ == "__main__":
    main()
