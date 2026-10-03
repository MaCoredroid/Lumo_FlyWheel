#!/usr/bin/env python3
"""Reconstruct manuscript quantities from preserved records; no model execution."""
import hashlib
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "claude-status-20261002"


def read(p):
    return json.loads(p.read_text())


def replay(p):
    rows = [json.loads(s) for s in p.read_text().splitlines()]
    assert len(rows) == 43 and not any(r.get("error") for r in rows), p
    assert len({r["request"] for r in rows}) == 43
    assert all(r["max_tokens"] == 1024 for r in rows)
    tokens = sum(r["completion_tokens"] for r in rows)
    duration = sum(r["t_e2e_s"] - r["t_ttft_s"] for r in rows)
    accepted = events = 0
    for r in rows:
        for k, v in r.get("spec_delta", {}).items():
            if "num_accepted_tokens{" in k:
                accepted += v
            if "num_drafts{" in k:
                events += v
    out = {"requests": len(rows), "output_tokens": tokens,
           "decode_seconds": duration, "pooled_tokens_s": (tokens - len(rows)) / duration,
           "request_prompt_tokens": {r["request"]: r["prompt_tokens"] for r in rows}}
    if events:
        out.update(accepted_drafts=accepted, speculative_events=events,
                   accepted_drafts_per_event=accepted / events)
    timers = {}
    timer_totals = {}
    for f in p.parent.glob("*.json.*"):
        if ".samples." in f.name:
            continue
        d = read(f)
        if d.get("schema", "").startswith("fr13.sfwd_gpu_timer"):
            timers["target_ms"] = d["decode_forward_gpu_seconds"] / d["n_pure_decode_steps_timed"] * 1000
            timers["step_wall_ms"] = d["decode_step_wall_seconds"] / d["n_wall_steps"] * 1000
            timer_totals["target"] = [d["decode_forward_gpu_seconds"], d["n_pure_decode_steps_timed"]]
            timer_totals["wall"] = [d["decode_step_wall_seconds"], d["n_wall_steps"]]
        elif d.get("label") in ["drafter", "committer"]:
            timers[d["label"] + "_ms"] = d["gpu_seconds"] / d["n_spans"] * 1000
            timer_totals[d["label"]] = [d["gpu_seconds"], d["n_spans"]]
    out["timers"] = timers
    out["timer_totals"] = timer_totals
    return rows, out


def metrics(p):
    vals = {}
    for line in p.read_text().splitlines():
        m = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([0-9.eE+-]+)', line)
        if m:
            vals[m[1]] = vals.get(m[1], 0) + float(m[2])
    return vals


def message_hash(d):
    return hashlib.sha256(json.dumps(d["messages"], sort_keys=True).encode()).hexdigest()


def conversation_hash(d):
    first = next(m for m in d["messages"] if m["role"] == "user")
    return hashlib.sha256(json.dumps(first.get("content", ""), sort_keys=True).encode()).hexdigest()


def quantile(values, p):
    s = sorted(values)
    i = (len(s) - 1) * p
    lo = int(i)
    return s[lo] + (s[min(lo + 1, len(s) - 1)] - s[lo]) * (i - lo)


manifest = read(HERE / "SYNC-MANIFEST.json")
for f in manifest["files"]:
    assert hashlib.sha256((HERE / f["path"]).read_bytes()).hexdigest() == f["sha256"], f["path"]
confirmation = read(HERE / "raw/confirm/MANIFEST.json")
for f in confirmation["files"]:
    assert hashlib.sha256((HERE / "raw/confirm/requests" / f["name"]).read_bytes()).hexdigest() == f["sha256"]
tune = [read(p) for p in (HERE / "raw/corpus/requests").glob("*.json")]
confirm = [read(p) for p in (HERE / "raw/confirm/requests").glob("*.json")]
assert len(tune) == len(confirm) == 43
assert not ({message_hash(d) for d in tune} & {message_hash(d) for d in confirm})
assert not ({conversation_hash(d) for d in tune} & {conversation_hash(d) for d in confirm})

replays = {}
records = {}
for p in sorted((HERE / "raw/replay").glob("*/replay.jsonl")):
    if "diag" in p.parent.name:
        continue
    records[p.parent.name], replays[p.parent.name] = replay(p)

cf = {key: next(k for k in records if marker in k)
      for key, marker in [("tree", "cfT1"), ("mtp5", "cfM1"), ("sglang", "cfS1")]}
paired = {a: {r["request"]: r for r in records[k]} for a, k in cf.items()}
assert all(set(d) == set(paired["tree"]) for d in paired.values())
assert set(paired["tree"]) == {f["name"] for f in confirmation["files"]}
assert all(paired["tree"][n]["prompt_tokens"] == paired["mtp5"][n]["prompt_tokens"] for n in paired["tree"])
prompt_delta = sorted({paired["sglang"][n]["prompt_tokens"] - paired["tree"][n]["prompt_tokens"] for n in paired["tree"]})
comparison = {}
for a in ["mtp5", "sglang"]:
    ratios = []
    for n in paired["tree"]:
        t, other = paired["tree"][n], paired[a][n]
        ratios.append(((t["completion_tokens"] - 1) / t["t_decode_s"]) /
                      ((other["completion_tokens"] - 1) / other["t_decode_s"]))
    comparison[a] = {"pooled_ratio": replays[cf["tree"]]["pooled_tokens_s"] / replays[cf[a]]["pooled_tokens_s"],
                     "requests_tree_faster": sum(x > 1 for x in ratios), "median_request_ratio": statistics.median(ratios)}

baseline = [d for k, d in replays.items() if "swT1" in k or "swT2" in k]
assert len(baseline) == 2
baseline_pooled = {"pooled_tokens_s": sum(d["output_tokens"] - d["requests"] for d in baseline) / sum(d["decode_seconds"] for d in baseline),
                   "accepted_drafts_per_event": sum(d["accepted_drafts"] for d in baseline) / sum(d["speculative_events"] for d in baseline),
                   "timers": {k: sum(d["timer_totals"][k][0] for d in baseline) / sum(d["timer_totals"][k][1] for d in baseline) * 1000
                              for k in ["target", "wall", "drafter", "committer"]}}

study = read(HERE / "summaries/swe_study_20261001.json")
workload = {}
for arm in ["lumotree", "vllm_mtp5", "sglang_eagle_s7_d8"]:
    b = metrics(HERE / "raw/workload" / arm / "metrics_before_swe.txt")
    a = metrics(HERE / "raw/workload" / arm / "metrics_after_swe.txt")
    prefix = "sglang:" if arm.startswith("sglang") else "vllm:"
    names = ["generation_tokens_total", "num_requests_total" if prefix == "sglang:" else "request_success_total",
             "e2e_request_latency_seconds_sum", "time_to_first_token_seconds_sum"]
    n, r, e, f = [a[prefix + x] - b.get(prefix + x, 0) for x in names]
    reports = [read(p) for p in (PREVIOUS / "swe" / arm).glob("*/eval_report.json")]
    assert len(reports) == 10
    resolved = sum(d.get("passed", False) for d in reports)
    empty = sum(d.get("synthetic_no_patch", False) for d in reports)
    failed_tests = sum(d.get("failure_mode") == "tests_failed" for d in reports)
    assert resolved == study[arm]["resolved"]
    assert resolved + empty + failed_tests == 10
    for d in reports:
        suffix = d["instance_id"].split("-")[-1]
        assert d["verdict"] == study[arm]["tasks"][suffix]["verdict"]
    for name in ["num_requests_running", "num_requests_waiting"]:
        if prefix + name in a:
            assert a[prefix + name] == 0
    workload[arm] = {"output_tokens": n, "requests": r, "e2e_s": e, "ttft_s": f,
                     "pooled_tokens_s": (n - r) / (e - f), "raw_evaluation_reports": len(reports),
                     "resolved": resolved, "empty_patches": empty, "failed_tests": failed_tests,
                     "test_evaluator_invocations": len(reports) - empty,
                     "agent_minutes": sum(x["agent_s"] for x in study[arm]["tasks"].values()) / 60,
                     "task_summary": study[arm]["tasks"]}
    assert round(workload[arm]["pooled_tokens_s"], 2) == study[arm]["pooled"]["pooled_decode_tok_s"]

v = read(HERE / "summaries/q1v3_v2_20261002/VERDICT.json")
surfaces = ["gdn_rel", "conv_rel", "kv_new_rel", "kv_hist_rel"]
worst = {s: 0 for s in surfaces}
kl = {a: [] for a in ["candidate", "B", "V", "P"]}
flips = {a: 0 for a in kl}
cells = 0
for case, rows in v["cells_candidate"].items():
    for c in rows:
        k = c["k"]
        native = {a: v["cells_native"][a][case][k] for a in ["B", "V", "P"]}
        for s in surfaces:
            worst[s] = max(worst[s], c[s] / max(0.02, *(n[s] for n in native.values())))
        for a, x in [("candidate", c), *native.items()]:
            kl[a].append(x["next_kl"])
            flips[a] += x["greedy_x"] != x["greedy_ref"]
        cells += 1
assert cells == 147 and v["verdict"] == "INCONCLUSIVE"
faults = []
for n in v["negative_controls"]:
    base = n["nc_id"].split("__")[0] + "__C2-nc-base"
    native = [v["cells_native"][a][base][n["cycle"]] for a in ["B", "V", "P"]]
    vals = []
    for label, d in n["target"].items():
        s = {"conv": "conv_rel", "gdn": "gdn_rel", "kv": "kv_new_rel", "logits": "next_kl"}[label]
        vals.append(d["value"] / max(0.02, *(x[s] for x in native)))
    faults.append({"mutation": n["mutation"], "ratio": max(vals), "detected_frozen": n["detected"]})
state_faults = [n for n in faults if n["mutation"] != "NC_STALE"]
assert len(state_faults) == 8
audit = {"schema": "codex.paper-results-audit.v1", "source_files_sha_verified": len(manifest["files"]),
         "remote_head": manifest["remote_head"], "workload": workload, "replays": replays,
         "drafter_pass_baseline_pooled": baseline_pooled,
         "confirmation": {"request_count": 43, "message_overlap": 0, "conversation_overlap": 0,
                          "conversations": len({conversation_hash(d) for d in confirm}),
                          "task_selection": confirmation["tasks"], "sglang_prompt_token_delta": prompt_delta,
                          "comparisons": comparison, "runs": cf,
                          "scope": "one run per arm; SGLang chat API, NOT exact token-ID parity"},
         "fullmodel": {"cells": cells, "frozen_verdict": v["verdict"],
                       "faults_detected_frozen": sum(n["detected"] for n in v["negative_controls"]),
                       "faults_total": len(faults), "posthoc_state_worst_ratio": worst,
                       "posthoc_state_fault_min_ratio": min(n["ratio"] for n in state_faults),
                       "posthoc_state_fault_observations": len(state_faults),
                       "next_logit_KL": {a: {"median": statistics.median(x), "p90": quantile(x, .9),
                                             "max": max(x), "greedy_flips": flips[a]} for a, x in kl.items()},
                       "posthoc_status": "descriptive after inspecting outcomes; not independent qualification"}}
(HERE / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps({"verified_files": len(manifest["files"]), "confirmation": audit["confirmation"],
                  "fullmodel": audit["fullmodel"], "workload_rates": {a: d["pooled_tokens_s"] for a, d in workload.items()}}, indent=2))
