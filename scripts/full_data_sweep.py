#!/usr/bin/env python3
"""Master data sweep for the Round 4b ablation report.

Produces:
  - Per-cell (point × task × attempt) metrics: decode_tps, prefill_s, accept_rate, power_w, gpu_util, mem_util
  - Per-call slices: by turn-index bucket, prompt-length bucket, acceptance-rate bin
  - Per-cell behavior: pass/fail, M_aggregate, integrity flags, ceilings, milestones
  - Cell aggregates and per-point medians for the report
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("/sessions/charming-hopeful-johnson/mnt/Lumo_FlyWheel/output")
OUT = Path("/sessions/charming-hopeful-johnson/mnt/Lumo_FlyWheel/output/track_b_e2e_v4a_v2_report_data.json")

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

CONTAMINATED = set()  # Originally 4 D-point attempts; remeasured 2026-05-17, contaminated originals archived.


def safe_median(xs):
    return statistics.median(xs) if xs else None


def safe_quantile(xs, q):
    if not xs:
        return None
    if len(xs) < 10:
        sx = sorted(xs)
        return sx[int(q * (len(sx) - 1))]
    return statistics.quantiles(xs, n=10)[int(q * 10) - 1]


def attempt_data(run_dir: Path):
    metrics = run_dir / "vllm_request_metrics.jsonl"
    dcgm = run_dir / "dcgm_samples.jsonl"
    grader = run_dir / "grader_result.json"
    runner = run_dir / "runner_metadata.json"
    if not metrics.is_file():
        return None
    rows = []
    for line in metrics.read_text().splitlines():
        if not line.strip(): continue
        try:
            r = json.loads(line)
            rows.append(r)
        except Exception: pass
    if not rows: return None

    # Per-call metrics
    per_call = []
    for r in rows:
        ct = r.get("completion_tokens") or 0
        ds = r.get("decode_sum_s") or 0
        ps = r.get("prefill_sum_s") or 0
        pt = r.get("prompt_tokens") or 0
        accept = r.get("spec_decode_num_accepted_tokens") or 0
        draft = r.get("spec_decode_num_draft_tokens") or 0
        per_call.append({
            "turn_index": r.get("oracle_turn_index"),
            "regime": r.get("regime"),
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "decode_s": ds,
            "prefill_s": ps,
            "decode_tps": (ct / ds) if ds > 0 and ct > 0 else None,
            "accept_rate": (accept / draft) if draft > 0 else None,
            "tool_call_observed": r.get("tool_call_observed"),
            "primed_text_count": r.get("oracle_primed_text_count") or 0,
            "tool_schema_count": r.get("oracle_tool_schema_count") or 0,
        })

    # Cell metrics
    tps_all = [c["decode_tps"] for c in per_call if c["decode_tps"]]
    pref_all = [c["prefill_s"] for c in per_call if c["prefill_s"] and c["prefill_s"] > 0]
    accept_all = [c["accept_rate"] for c in per_call if c["accept_rate"] is not None]
    pt_all = [c["prompt_tokens"] for c in per_call if c["prompt_tokens"]]

    # Hardware
    pwr_vals, gpu_util_vals, mem_util_vals = [], [], []
    if dcgm.is_file():
        for line in dcgm.read_text().splitlines():
            if not line.strip(): continue
            try:
                d = json.loads(line)
                if d.get("power_w") and d["power_w"] > 0: pwr_vals.append(d["power_w"])
                if d.get("gpu_util_pct") is not None: gpu_util_vals.append(d["gpu_util_pct"])
                if d.get("mem_copy_util_pct") is not None: mem_util_vals.append(d["mem_copy_util_pct"])
            except Exception: pass

    # Grader
    grader_data = None
    if grader.is_file():
        try:
            g = json.loads(grader.read_text())
            grader_data = {
                "P_benchmark": g.get("P_benchmark"),
                "M_aggregate": g.get("M_aggregate") or g.get("milestone_vector", {}).get("M_aggregate"),
                "M_training": g.get("M_training"),
                "integrity_flag": g.get("integrity_flag"),
                "integrity_rules_fired": g.get("integrity_rules_fired", []),
                "ceilings_applied": g.get("ceilings_applied", []),
                "milestones": g.get("milestones", {}),
                "passed": (g.get("P_benchmark") or 0) >= 65,
            }
        except Exception: pass

    # Runner metadata
    runner_data = None
    if runner.is_file():
        try:
            rm = json.loads(runner.read_text())
            runner_data = {
                "elapsed_s": rm.get("elapsed_s"),
                "codex_exit_code": rm.get("codex_exit_code"),
            }
        except Exception: pass

    # Per-turn-bucket slicing (turn 0 = first turn; 1-5 early; 6-20 mid; 21+ late)
    def bucket_turn(t):
        if t is None: return "unk"
        if t == 0: return "t0_first"
        if t <= 5: return "t1_5_early"
        if t <= 20: return "t6_20_mid"
        return "t21p_late"

    def bucket_prompt(p):
        if p < 5000: return "p_short_<5k"
        if p < 15000: return "p_med_5_15k"
        if p < 30000: return "p_long_15_30k"
        return "p_xlong_>30k"

    slices = defaultdict(lambda: {"tps": [], "accept": [], "prefill_s": []})
    for c in per_call:
        bt = bucket_turn(c["turn_index"])
        bp = bucket_prompt(c["prompt_tokens"])
        if c["decode_tps"] is not None:
            slices[f"turn:{bt}"]["tps"].append(c["decode_tps"])
            slices[f"prompt:{bp}"]["tps"].append(c["decode_tps"])
            slices["overall"]["tps"].append(c["decode_tps"])
        if c["accept_rate"] is not None:
            slices[f"turn:{bt}"]["accept"].append(c["accept_rate"])
            slices[f"prompt:{bp}"]["accept"].append(c["accept_rate"])
            slices["overall"]["accept"].append(c["accept_rate"])
        if c["prefill_s"] and c["prefill_s"] > 0:
            slices[f"turn:{bt}"]["prefill_s"].append(c["prefill_s"])
            slices[f"prompt:{bp}"]["prefill_s"].append(c["prefill_s"])
            slices["overall"]["prefill_s"].append(c["prefill_s"])

    slice_medians = {}
    for k, v in slices.items():
        slice_medians[k] = {
            "n_tps": len(v["tps"]),
            "tps_med": safe_median(v["tps"]),
            "tps_p10": min(v["tps"]) if v["tps"] else None,
            "tps_p90": max(v["tps"]) if len(v["tps"]) < 10 else safe_quantile(v["tps"], 0.9),
            "accept_med": safe_median(v["accept"]),
            "prefill_s_med": safe_median(v["prefill_s"]),
        }

    return {
        "n_calls": len(per_call),
        "decode_tps_med": safe_median(tps_all),
        "decode_tps_p10": safe_quantile(tps_all, 0.1) if len(tps_all) >= 10 else (min(tps_all) if tps_all else None),
        "decode_tps_p90": safe_quantile(tps_all, 0.9) if len(tps_all) >= 10 else (max(tps_all) if tps_all else None),
        "prefill_s_med": safe_median(pref_all),
        "accept_rate_med": safe_median(accept_all),
        "accept_rate_mean": (sum(accept_all) / len(accept_all)) if accept_all else None,
        "prompt_tokens_med": safe_median(pt_all),
        "power_w_med": safe_median(pwr_vals),
        "power_w_p90": safe_quantile(pwr_vals, 0.9) if len(pwr_vals) >= 10 else (max(pwr_vals) if pwr_vals else None),
        "power_w_max": max(pwr_vals) if pwr_vals else None,
        "power_w_min": min(pwr_vals) if pwr_vals else None,
        "gpu_util_med": safe_median(gpu_util_vals),
        "mem_util_med": safe_median(mem_util_vals),
        "n_dcgm_samples": len(pwr_vals),
        "grader": grader_data,
        "runner": runner_data,
        "slices": slice_medians,
    }


cells = {}
for point, roots in POINTS.items():
    for parent in roots:
        if not parent.is_dir(): continue
        for task in parent.iterdir():
            if not task.is_dir() or "__" not in task.name: continue
            task_short = task.name.replace("__v1-clean-baseline", "")
            for run in sorted(task.iterdir()):
                if not run.is_dir() or not run.name.startswith("run_"): continue
                # For D-point fanout, use only round_0
                if point == "D" and "fanout" in task.name and parent.name != "round_0": continue
                d = attempt_data(run)
                if d is None: continue
                key = (point, task_short, run.name)
                if key in cells:
                    # Already have data for this cell — likely a duplicate from another preserved dir. Skip.
                    continue
                cells[key] = {
                    "point": point,
                    "task": task_short,
                    "attempt": run.name,
                    "parent": parent.name,
                    "path": str(run),
                    "contaminated": (point, task.name, run.name) in CONTAMINATED,
                    **d,
                }

# Aggregate per cell (point × task)
cell_agg = {}
for (point, task, attempt), data in cells.items():
    k = (point, task)
    cell_agg.setdefault(k, []).append(data)

# Aggregate per point
point_agg = {}
for (point, task), attempts in cell_agg.items():
    # Skip contaminated attempts for clean cell medians
    clean = [a for a in attempts if not a["contaminated"]]
    if not clean: clean = attempts  # fallback
    tps_med = safe_median([a["decode_tps_med"] for a in clean if a["decode_tps_med"]])
    pref_med = safe_median([a["prefill_s_med"] for a in clean if a["prefill_s_med"]])
    accept_med = safe_median([a["accept_rate_med"] for a in clean if a["accept_rate_med"] is not None])
    pwr_med = safe_median([a["power_w_med"] for a in clean if a["power_w_med"]])
    gpu_med = safe_median([a["gpu_util_med"] for a in clean if a["gpu_util_med"] is not None])
    pass_count = sum(1 for a in clean if a.get("grader", {}) and a["grader"].get("passed"))
    M_agg_med = safe_median([a["grader"]["M_aggregate"] for a in clean if a.get("grader", {}) and a["grader"].get("M_aggregate") is not None])
    point_agg.setdefault(point, []).append({
        "task": task,
        "n_attempts_clean": len(clean),
        "n_attempts_contaminated": len(attempts) - len(clean),
        "decode_tps_med": tps_med,
        "prefill_s_med": pref_med,
        "accept_rate_med": accept_med,
        "power_w_med": pwr_med,
        "gpu_util_med": gpu_med,
        "pass_count": pass_count,
        "M_aggregate_med": M_agg_med,
    })

# Per-point aggregates
point_summary = {}
for point, tasks in point_agg.items():
    tps_med = safe_median([t["decode_tps_med"] for t in tasks if t["decode_tps_med"]])
    pref_med = safe_median([t["prefill_s_med"] for t in tasks if t["prefill_s_med"]])
    accept_med = safe_median([t["accept_rate_med"] for t in tasks if t["accept_rate_med"] is not None])
    pwr_med = safe_median([t["power_w_med"] for t in tasks if t["power_w_med"]])
    M_med = safe_median([t["M_aggregate_med"] for t in tasks if t["M_aggregate_med"] is not None])
    n_pass = sum(t["pass_count"] for t in tasks)
    n_total_attempts = sum(t["n_attempts_clean"] + t["n_attempts_contaminated"] for t in tasks)
    point_summary[point] = {
        "n_tasks": len(tasks),
        "n_attempts_total": n_total_attempts,
        "decode_tps_med": tps_med,
        "prefill_s_med": pref_med,
        "accept_rate_med": accept_med,
        "power_w_med": pwr_med,
        "M_aggregate_med": M_med,
        "n_pass": n_pass,
    }

# Per-slice aggregation across ablation points
slice_agg = defaultdict(lambda: defaultdict(list))
for (point, task, attempt), data in cells.items():
    if data["contaminated"]: continue
    for slice_name, vals in data["slices"].items():
        if vals["tps_med"]: slice_agg[(point, slice_name)]["tps"].append(vals["tps_med"])
        if vals["accept_med"] is not None: slice_agg[(point, slice_name)]["accept"].append(vals["accept_med"])
        if vals["prefill_s_med"]: slice_agg[(point, slice_name)]["prefill_s"].append(vals["prefill_s_med"])

slice_summary = {}
for (point, slice_name), vals in slice_agg.items():
    slice_summary[f"{point}|{slice_name}"] = {
        "tps_med": safe_median(vals["tps"]),
        "accept_med": safe_median(vals["accept"]),
        "prefill_s_med": safe_median(vals["prefill_s"]),
        "n_attempts": len(vals["tps"]),
    }

# Dump everything
output = {
    "points": {k: v for k, v in point_summary.items()},
    "per_task_per_point": {pt: tasks for pt, tasks in point_agg.items()},
    "slices": slice_summary,
    "cells_raw": [{"point": p, "task": t, "attempt": a, **v} for (p, t, a), v in cells.items()],
}
OUT.write_text(json.dumps(output, indent=2, default=str))

# Print headline tables
print("="*100)
print("PER-POINT AGGREGATE")
print("="*100)
print(f"{'point':<5} {'n_tasks':>7} {'tps_med':>9} {'prefill_s':>10} {'accept':>8} {'pwr_W':>8} {'M_agg':>7} {'pass':>5}")
def fmt(v, fmt_str=">9.2f"):
    if v is None: return f"{'—':>{int(fmt_str.lstrip('>').split('.')[0])}}"
    return f"{v:{fmt_str}}"
for pt in ["OFF", "A", "B", "C", "D"]:
    if pt not in point_summary: continue
    s = point_summary[pt]
    print(f"{pt:<5} {s['n_tasks']:>7} {fmt(s['decode_tps_med'])} {fmt(s['prefill_s_med'],'>10.2f')} {fmt(s['accept_rate_med'],'>8.3f')} {fmt(s['power_w_med'],'>8.2f')} {fmt(s['M_aggregate_med'],'>7.3f')} {s['n_pass']:>5}")

print()
print("="*100)
print("PER-TASK × POINT DECODE_TPS (cleaned)")
print("="*100)
tasks_seen = set()
for pt in ["OFF", "A", "B", "C", "D"]:
    if pt not in point_agg: continue
    for t in point_agg[pt]:
        tasks_seen.add(t["task"])
header = f"{'task':<42}"
for pt in ["OFF", "A", "B", "C", "D"]:
    header += f" {pt:>7}"
print(header)
for task in sorted(tasks_seen):
    row = f"{task[:42]:<42}"
    for pt in ["OFF", "A", "B", "C", "D"]:
        t = next((x for x in point_agg.get(pt, []) if x["task"] == task), None)
        if t and t["decode_tps_med"]:
            row += f" {t['decode_tps_med']:>7.2f}"
        else:
            row += f" {'—':>7}"
    print(row)

print(f"\nWrote: {OUT}")
print(f"Total cells: {len(cells)}")
print(f"Tasks: {len(tasks_seen)}")
