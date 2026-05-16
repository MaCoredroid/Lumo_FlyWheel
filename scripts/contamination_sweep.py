#!/usr/bin/env python3
"""Per-attempt contamination detector v2.

Use cell MAX tps and cell MIN power as the "clean reference" — contamination
only pushes power UP and tps DOWN, so the cleanest attempt sets the baseline.
"""
import json
import statistics
from pathlib import Path

ROOT = Path("/sessions/charming-hopeful-johnson/mnt/Lumo_FlyWheel/output")

POINTS = {
    "D": [
        ROOT / "track_b_e2e_v4a_v2" / "round_0",
        ROOT / "track_b_e2e_v4a_v2" / "round_0_phase1_task1_2_PRESERVED",
        ROOT / "track_b_e2e_v4a_v2" / "round_0_phase2_task3_4_PRESERVED",
        ROOT / "track_b_e2e_v4a_v2" / "round_0_phase3a_PRESERVED",
    ],
    "A": [ROOT / "track_b_e2e_v4a_v2_ablation" / "round_1"],
    "B": [ROOT / "track_b_e2e_v4a_v2_ablation" / "round_2"],
    "C": [ROOT / "track_b_e2e_v4a_v2_ablation" / "round_3"],
    "OFF": [ROOT / "track_b_e2e_v4a_v2_ablation" / "round_4"],
}

# Global baseline: OFF cells show power_w ~37.5W consistently — this is "host idle"
ABS_POWER_FLOOR = 37.0   # below this is "quiet host"
ABS_POWER_SUSPICIOUS = 42.0  # above this on a single attempt is suspicious

def attempt_metrics(run_dir: Path):
    metrics_file = run_dir / "vllm_request_metrics.jsonl"
    dcgm_file = run_dir / "dcgm_samples.jsonl"
    if not metrics_file.is_file():
        return None
    tps = []
    for line in metrics_file.read_text().splitlines():
        if not line.strip(): continue
        try: r = json.loads(line)
        except Exception: continue
        ct = r.get("completion_tokens") or 0
        ds = r.get("decode_sum_s") or 0
        if ct > 0 and ds > 0:
            tps.append(ct / ds)
    if not tps: return None
    pwr = []
    if dcgm_file.is_file():
        for line in dcgm_file.read_text().splitlines():
            if not line.strip(): continue
            try: d = json.loads(line)
            except Exception: continue
            p = d.get("power_w")
            if p is not None and p > 0: pwr.append(p)
    return {
        "decode_tps_med": statistics.median(tps),
        "decode_tps_p10": statistics.quantiles(tps, n=10)[0] if len(tps) >= 10 else min(tps),
        "power_w_med": statistics.median(pwr) if pwr else None,
        "power_w_p90": statistics.quantiles(pwr, n=10)[8] if len(pwr) >= 10 else (max(pwr) if pwr else None),
        "n_calls": len(tps),
        "n_samples": len(pwr),
    }

rows = []
for point, roots in POINTS.items():
    task_to_attempts = {}
    for parent in roots:
        if not parent.is_dir(): continue
        for task_dir in parent.iterdir():
            if not task_dir.is_dir(): continue
            task_name = task_dir.name
            for run_dir in sorted(task_dir.iterdir()):
                if not run_dir.is_dir() or not run_dir.name.startswith("run_"): continue
                m = attempt_metrics(run_dir)
                if m is None: continue
                task_to_attempts.setdefault(task_name, []).append({
                    "run": run_dir.name, "parent": parent.name, "path": str(run_dir), **m,
                })
    for task, attempts in task_to_attempts.items():
        if point == "D" and "fanout" in task:
            attempts = [a for a in attempts if a["parent"] == "round_0"]
        tps_vals = [a["decode_tps_med"] for a in attempts]
        pwr_vals = [a["power_w_med"] for a in attempts if a["power_w_med"]]
        cell_tps_max = max(tps_vals) if tps_vals else None
        cell_pwr_min = min(pwr_vals) if pwr_vals else None
        cell_tps_med = statistics.median(tps_vals) if tps_vals else None
        cell_pwr_med = statistics.median(pwr_vals) if pwr_vals else None
        for a in attempts:
            # Cell-reference: compare to cleanest attempt in the cell
            a["cell_tps_max"] = cell_tps_max
            a["cell_pwr_min"] = cell_pwr_min
            a["cell_tps_med"] = cell_tps_med
            a["cell_pwr_med"] = cell_pwr_med
            a["tps_vs_max"] = (a["decode_tps_med"] / cell_tps_max) if cell_tps_max else None
            a["pwr_vs_min"] = (a["power_w_med"] / cell_pwr_min) if cell_pwr_min and a["power_w_med"] else None
            # Contamination: pwr > 1.15× cell-min AND tps < 0.55× cell-max
            #   (host-level contention: power spike + throughput crash)
            a["contaminated"] = (
                a["tps_vs_max"] is not None and a["tps_vs_max"] < 0.55 and
                a["pwr_vs_min"] is not None and a["pwr_vs_min"] > 1.15
            )
            # Power-only signal (host contention, weak throughput signal)
            a["power_anomaly"] = (
                a["power_w_med"] is not None and 
                a["power_w_med"] > ABS_POWER_SUSPICIOUS and
                a["pwr_vs_min"] is not None and a["pwr_vs_min"] > 1.10
            )
            # TPS-only signal (no power signal — could be model behavior)
            a["tps_only_outlier"] = (
                a["tps_vs_max"] is not None and a["tps_vs_max"] < 0.50 and
                not a["contaminated"]
            )
            rows.append({"point": point, "task": task, **a})

# Report
def k(r): return (r["point"], r["task"], r["parent"], r["run"])
rows.sort(key=k)

print(f"{'point':<4} {'task':<46} {'run':<6} {'tps':>6} {'pwr':>5} {'tps/max':>7} {'pwr/min':>7} {'flag':<6}")
seen = {(r["point"], r["task"]) for r in rows if r["contaminated"] or r["power_anomaly"] or r["tps_only_outlier"]}
for r in rows:
    if (r["point"], r["task"]) not in seen: continue
    flag = "CONTAM" if r["contaminated"] else ("PWR-HI" if r["power_anomaly"] else ("TPS-LO" if r["tps_only_outlier"] else ""))
    tps_r = f"{r['tps_vs_max']:.2f}" if r["tps_vs_max"] else "n/a"
    pwr_r = f"{r['pwr_vs_min']:.2f}" if r["pwr_vs_min"] else "n/a"
    pwr = f"{r['power_w_med']:.1f}" if r["power_w_med"] else "n/a"
    print(f"{r['point']:<4} {r['task'][:46]:<46} {r['run']:<6} {r['decode_tps_med']:>6.2f} {pwr:>5} {tps_r:>7} {pwr_r:>7} {flag:<6}")

print()
print("="*100)
print("REMEASURE PRIORITY 1 — CONTAMINATED (power+tps both anomalous, host contention confirmed)")
print("="*100)
contam = [r for r in rows if r["contaminated"]]
for r in contam:
    print(f"  {r['point']:<3} | {r['task']:<46} | {r['run']} | tps={r['decode_tps_med']:.2f} ({r['tps_vs_max']:.2f}× cell-max) | pwr={r['power_w_med']:.1f}W ({r['pwr_vs_min']:.2f}× cell-min)")

print()
print("="*100)
print("REMEASURE PRIORITY 2 — POWER ANOMALY (high power, tps may or may not be down)")
print("="*100)
pwr_anom = [r for r in rows if r["power_anomaly"] and not r["contaminated"]]
for r in pwr_anom:
    print(f"  {r['point']:<3} | {r['task']:<46} | {r['run']} | tps={r['decode_tps_med']:.2f} ({r['tps_vs_max']:.2f}× cell-max) | pwr={r['power_w_med']:.1f}W ({r['pwr_vs_min']:.2f}× cell-min)")

print()
print("="*100)
print("REMEASURE PRIORITY 3 — TPS-ONLY OUTLIER (could be model behavior, less certain)")
print("="*100)
tps_only = [r for r in rows if r["tps_only_outlier"]]
for r in tps_only:
    pwr = f"{r['power_w_med']:.1f}" if r["power_w_med"] else "n/a"
    print(f"  {r['point']:<3} | {r['task']:<46} | {r['run']} | tps={r['decode_tps_med']:.2f} ({r['tps_vs_max']:.2f}× cell-max) | pwr={pwr}W")

print()
print("="*100)
print(f"TOTAL: P1 contaminated={len(contam)}, P2 pwr-anomaly={len(pwr_anom)}, P3 tps-low={len(tps_only)}")
print("="*100)

# Group remeasure list by cell
print("\nREMEASURE QUEUE (group by cell, ordered by priority):")
queue = {}
for r in contam:
    queue.setdefault((r["point"], r["task"]), {"P1": [], "P2": [], "P3": []})["P1"].append(r["run"])
for r in pwr_anom:
    queue.setdefault((r["point"], r["task"]), {"P1": [], "P2": [], "P3": []})["P2"].append(r["run"])
for r in tps_only:
    queue.setdefault((r["point"], r["task"]), {"P1": [], "P2": [], "P3": []})["P3"].append(r["run"])
for (point, task), tiers in sorted(queue.items()):
    parts = []
    for tier in ("P1", "P2", "P3"):
        if tiers[tier]:
            parts.append(f"{tier}:{sorted(set(tiers[tier]))}")
    print(f"  {point:<3} | {task:<46} | {' '.join(parts)}")
