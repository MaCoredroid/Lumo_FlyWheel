#!/usr/bin/env python3
"""Independent reduction of the penalty-history-fix records synced from GB10 on 2026-10-03/04. No inference.
Recomputes replay pools, the captured-row history check, the Monte-Carlo sampler verdicts (with an independent
top-k boundary-tie check on every residual node), the v3 continuation verdict statistics, and the SWE arm's
outcomes, pooled decode rate, acceptance and phase timers, then writes AUDIT.json."""
import glob, hashlib, json, re, runpy, statistics
from pathlib import Path
HERE = Path(__file__).resolve().parent
V2 = HERE.parent.parent
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
out = {"schema": "claude.penfix-audit.v1"}

# ---- provenance -------------------------------------------------------------------------------------------
ident = (HERE / "code/patcher-identity.txt").read_text().split("\n")
out["patcher_sha256_worktree"], out["patcher_sha256_port"] = ident[0].split()[0], ident[1].split()[0]
assert out["patcher_sha256_worktree"] == out["patcher_sha256_port"]
out["remote_heads"] = {l.split()[0]: l.split()[1] for l in ident[2:] if l.strip()}
manifest = read(HERE / "SYNC-MANIFEST.json")
for f in manifest["files"]:
    assert sha(HERE / f["path"]) == f["sha256"], f["path"]
out["synced_files_sha_verified"] = len(manifest["files"])
diff = (HERE / "code/patcher-penfix.diff").read_text()
assert "_patch_rejection_sampler_tree_penalty_history" in diff and "apply_logits_processors_tree_self" in diff
assert diff.count("apply_logits_processors_tree_self(") >= 4

# ---- replays ----------------------------------------------------------------------------------------------
def replay(p):
    rows = [json.loads(l) for l in Path(p).read_text().splitlines()]
    assert len(rows) == 43 and not any(r.get("error") for r in rows)
    assert len({r["request"] for r in rows}) == 43 and all(r["max_tokens"] == 1024 for r in rows)
    n = sum(r["completion_tokens"] for r in rows); t = sum(r["t_e2e_s"] - r["t_ttft_s"] for r in rows)
    acc = sum(x for r in rows for k, x in r.get("spec_delta", {}).items() if "num_accepted_tokens{" in k)
    ev = sum(x for r in rows for k, x in r.get("spec_delta", {}).items() if "num_drafts{" in k)
    return {"run": Path(p).parent.name, "requests": 43, "output_tokens": n, "decode_seconds": t,
            "tokens_s": (n - 43) / t, "accepted": acc, "events": ev, "accepted_per_event": acc / ev,
            "prompt_counts": {r["request"]: r["prompt_tokens"] for r in rows}}
def pool(rr):
    return {"runs": len(rr), "requests": sum(r["requests"] for r in rr), "output_tokens": sum(r["output_tokens"] for r in rr),
            "decode_seconds": sum(r["decode_seconds"] for r in rr),
            "pooled_tokens_s": sum(r["output_tokens"] - r["requests"] for r in rr) / sum(r["decode_seconds"] for r in rr),
            "run_rates": [r["tokens_s"] for r in rr],
            "accepted_per_event": sum(r["accepted"] for r in rr) / sum(r["events"] for r in rr),
            "run_accepted_per_event": [r["accepted_per_event"] for r in rr]}
def timers(d):
    tot = {}
    for f in Path(d).glob("*.json.*"):
        if ".samples." in f.name: continue
        j = read(f)
        if j.get("schema", "").startswith("fr13.sfwd_gpu_timer"):
            tot["target"] = [j["decode_forward_gpu_seconds"], j["n_pure_decode_steps_timed"]]
            tot["wall"] = [j["decode_step_wall_seconds"], j["n_wall_steps"]]
        elif j.get("label") in ("drafter", "committer"):
            tot[j["label"]] = [j["gpu_seconds"], j["n_spans"]]
    assert set(tot) == {"target", "wall", "drafter", "committer"}, d
    return tot
def timer_means(ts):
    return {k: 1000 * sum(t[k][0] for t in ts) / sum(t[k][1] for t in ts) for k in ("target", "drafter", "committer", "wall")}
tuning = [replay(p) for p in sorted((HERE / "raw/replay").glob("tree-pfT*/replay.jsonl"))]
confirm = [replay(p) for p in sorted((HERE / "raw/replay").glob("tree-cfPF*/replay.jsonl"))]
assert len(tuning) == 3 and len(confirm) == 3
ref_tuning = read(V2 / "results/claude-replay-20261001/AUDITED-RATES.json")["arms"]
ref_confirm = read(V2 / "results/claude-closure-20261002/AUDIT.json")["confirmation"]
assert all(r["prompt_counts"] == tuning[0]["prompt_counts"] for r in tuning)
assert all(r["prompt_counts"] == confirm[0]["prompt_counts"] for r in confirm)
out["tuning_corpus"] = {"tree_fixed": pool(tuning), "tree_fixed_timer_means_ms": timer_means([timers(HERE / "raw/replay" / r["run"]) for r in tuning]),
                        "mtp5_reference_pooled_tokens_s": ref_tuning["mtp5"]["pooled_rate"],
                        "tree_prefix_reference_pooled_tokens_s": ref_tuning["lumotree"]["pooled_rate"]}
out["tuning_corpus"]["ratio_fixed_over_mtp5"] = out["tuning_corpus"]["tree_fixed"]["pooled_tokens_s"] / ref_tuning["mtp5"]["pooled_rate"]
out["confirmation_set"] = {"tree_fixed": pool(confirm), "tree_fixed_timer_means_ms": timer_means([timers(HERE / "raw/replay" / r["run"]) for r in confirm]),
                           "mtp5_reference": {k: ref_confirm["mtp5"][k] for k in ("pooled_tokens_s", "run_rates")},
                           "tree_prefix_reference": {k: ref_confirm["tree"][k] for k in ("pooled_tokens_s", "run_rates")}}
out["confirmation_set"]["ratio_fixed_over_mtp5"] = out["confirmation_set"]["tree_fixed"]["pooled_tokens_s"] / ref_confirm["mtp5"]["pooled_tokens_s"]
out["confirmation_set"]["all_fixed_tree_rates_above_all_mtp5"] = min(out["confirmation_set"]["tree_fixed"]["run_rates"]) > max(ref_confirm["mtp5"]["run_rates"])

# ---- sampling: captured rows and Monte-Carlo ----------------------------------------------------------------
T = runpy.run_path(str(HERE / "code/fr13_fixed32_topology.py")); parents = list(T["DRAFT_PARENT"]); assert len(parents) == 31
trace = HERE / "raw/replay/tree-pf0-sampled-20261003T190953Z/pen_trace.jsonl"
calls = [json.loads(l) for l in trace.read_text().splitlines()]; calls = [c for c in calls if c.get("nrows") == 31]
assert len(calls) == 80
flat = correct = total = 0; TOPK, TEMP = 20, 0.6
def path_set(u, spec):
    s = set()
    while u >= 0: s.add(spec[u]); u = parents[u]
    return s
steps = [(calls[i], calls[i + 1]) for i in range(0, 80, 2)]
for tgt, slf in steps:
    spec = tgt["spec"][0]; base = set(tgt["rows"][0]["pen_ids"])
    for kind, call in (("target", tgt), ("self", slf)):
        assert call["penalties"]["presence_penalties"] == [1.0]
        for j, row in enumerate(call["rows"]):
            node = parents[j] if kind == "target" else j
            total += 1; flat += set(row["pen_ids"]) == base | set(spec[:j]); correct += set(row["pen_ids"]) == base | path_set(node, spec)
out["sampling"] = {"captured_rows": total, "rows_matching_flattened_chain": flat, "rows_matching_per_path": correct}
assert (total, correct) == (2480, 2480)
mc = read(HERE / "summaries/sampling_20261003/pen_analysis_pf0_mc.json")
assert mc["history"] == {"rows": 2480, "flat_match": flat, "correct_match": 2480}
mcs = [s for s in mc["per_step"] if "faults" in s]; assert len(mcs) == 6
out["sampling"]["mc_steps"] = 6
out["sampling"]["walk_vs_used_rows"] = {"nodes_tested": sum(s["walk_vs_used_rows"]["nodes_tested"] for s in mcs), "nodes_rejected": sum(s["walk_vs_used_rows"]["nodes_rejected"] for s in mcs)}
out["sampling"]["walk_vs_correct_rows"] = {"nodes_tested": sum(s["walk_vs_correct_rows"]["nodes_tested"] for s in mcs), "nodes_rejected": sum(s["walk_vs_correct_rows"]["nodes_rejected"] for s in mcs),
                                           "rejected": [(s["step"], r) for s in mcs for r in s["walk_vs_correct_rows"]["rejected"]]}
out["sampling"]["expected_tv_per_step"] = [s["expected_tv_per_step"] for s in mcs]
out["sampling"]["faults_steps_detected"] = {k: sum(s["faults"][k]["detected"] for s in mcs) for k in mcs[0]["faults"]}
assert all(v >= 1 for v in out["sampling"]["faults_steps_detected"].values())
tv = [(s["step"], int(k), v) for s in mc["per_step"] for k, v in s["node_tv"].items()]
out["sampling"]["node_steps"] = len(tv); out["sampling"]["node_tv_max"] = max(v for _, _, v in tv)
hot = [(si, n, v) for si, n, v in tv if v > 0.01]
out["sampling"]["nodes_tv_above_0.01"] = len(hot)
# Independent tie check: every residual node must be an exact tie at the top-k boundary of its processed row.
ch = {}
for parent, kids in T["active_child_lists"]("hydra27_fixed32").items(): ch[int(parent)] = [int(k) for k in kids]
ties = []
for si, node, v in hot:
    tgt, slf = steps[si]; kids = ch.get(node, [])
    kind, j = ("target", kids[0]) if kids else ("self", node)
    r = (tgt if kind == "target" else slf)["rows"][j]; pen = set(r["pen_ids"]); pres = -sorted(r["pen_vals"])[0] if r["pen_vals"] else 1.0
    proc = sorted((((val - (pres if t in pen else 0.0)) / TEMP), t) for t, val in zip(r["top_ids"], r["top_vals"]))[::-1]
    k20, k21 = proc[TOPK - 1][0], proc[TOPK][0]
    ties.append({"step": si, "node": node, "tv": v, "tie_at_topk_boundary": abs(k20 - k21) < 1e-9, "n_tied": sum(1 for x, _ in proc if abs(x - k20) < 1e-9)})
out["sampling"]["residual_nodes"] = ties
assert all(t["tie_at_topk_boundary"] for t in ties)
assert all(any(t["step"] == st and t["node"] == rej[0] for t in ties) for st, rej in out["sampling"]["walk_vs_correct_rows"]["rejected"])

# ---- full-model continuation v3 --------------------------------------------------------------------------
v = read(HERE / "raw/q1v3v3/VERDICT.json")
assert v["verdict"] == "EQUIVALENT" and all(v["gates"].values()) and v["candidate"]["cells"] == 126 and not v["candidate"]["failing_cells"]
assert all(v["integrity"][a]["valid"] == v["integrity"][a]["expected"] and not v["integrity"][a]["invalid"] for a in v["integrity"])
freeze = read(HERE / "code/q1v3v3/FREEZE.json"); assert read(HERE / "raw/q1v3v3/FREEZE.snapshot.json") == freeze
for n, h in freeze["files"].items(): assert sha(HERE / "code/q1v3v3" / n) == h, n
def agree(cells):
    n = m = 0
    for lst in cells.values():
        for c in lst: n += 1; m += c["greedy_x"] == c["greedy_ref"]
    return m, n
cand = agree(v["cells_candidate"]); nat = {a: agree(v["cells_native"][a]) for a in ("B", "V", "P")}
dis = [(case, c["k"], c["margin_ref"]) for case, lst in v["cells_candidate"].items() for c in lst if c["greedy_x"] != c["greedy_ref"]]
ncs = v["negative_controls"]
nc_ratios = [(n["nc_id"], s, d["ratio"]) for n in ncs for s, d in (n.get("target") or {}).items() if isinstance(d, dict) and "ratio" in d]
cr = v["cache_reuse"]["observations"]
out["fullmodel"] = {"run": v["run"], "verdict": v["verdict"], "cells": 126, "max_ratio_by_surface": v["candidate"]["max_ratio_by_surface"],
                    "max_error_by_surface": v["candidate"]["max_by_surface"], "greedy_candidate_matches_A": cand, "greedy_controls_match_A": nat,
                    "candidate_disagreements_margin_ref": dis, "negative_controls_detected": sum(n["detected"] for n in ncs), "negative_controls": len(ncs),
                    "negative_control_min_ratio": min(nc_ratios, key=lambda t: t[2]), "cache_observations": len(cr),
                    "cache_min_fraction": min(o["cached_tokens"] / o["P"] for o in cr), "candidate_repeat_bitwise": all(r["bitwise_equal_all_cycles"] for r in v["candidate_repeat"])}
assert out["fullmodel"]["negative_controls_detected"] == 10 == len(ncs) and all(r < 2.0 for r in v["candidate"]["max_ratio_by_surface"].values())
assert all(o["hit"] and o["cached_tokens"] >= 0.9 * o["P"] for o in cr) and len(cr) == 30

# ---- SWE arm ----------------------------------------------------------------------------------------------
W = HERE / "raw/workload/lumotree_penfix"
tasks = {}
for l in (W / "swe_orchestrator.log").read_text().splitlines():
    m = re.search(r"<- astropy__astropy-(\d+) verdict=(\w+) elapsed_total=([\d.]+)s", l)
    if m: tasks[m.group(1)] = {"verdict": m.group(2), "agent_s": float(m.group(3))}
assert len(tasks) == 10
reports = {Path(p).parts[-3].split("-")[-1]: read(p) for p in glob.glob(str(W / "swe_out/verified/per_task/*/eval/eval_report.json"))}
assert len(reports) == 10 and all(reports[k]["verdict"] == tasks[k]["verdict"] for k in tasks)
for k in tasks:
    r = reports[k]
    tasks[k]["outcome"] = "R" if r["passed"] else ("E" if r.get("synthetic_no_patch") else "F")
    tasks[k]["failure_mode"] = r["failure_mode"]
def metrics(p):
    d = {}
    for l in Path(p).read_text().splitlines():
        if l.startswith("#") or not l.strip(): continue
        name, val = l.rsplit(" ", 1); d[name.split("{")[0]] = d.get(name.split("{")[0], 0.0) + float(val)
    return d
b, a = metrics(W / "metrics_before_swe.txt"), metrics(W / "metrics_after_swe.txt")
n, r_, e, f = [a["vllm:" + x] - b.get("vllm:" + x, 0) for x in ("generation_tokens_total", "request_success_total", "e2e_request_latency_seconds_sum", "time_to_first_token_seconds_sum")]
acc = a["vllm:spec_decode_num_accepted_tokens_total"] - b.get("vllm:spec_decode_num_accepted_tokens_total", 0)
ev = a["vllm:spec_decode_num_drafts_total"] - b.get("vllm:spec_decode_num_drafts_total", 0)
assert a.get("vllm:num_requests_running", 0) == 0 and a.get("vllm:num_requests_waiting", 0) == 0
out["workload"] = {"output_tokens": n, "requests": r_, "e2e_s": e, "ttft_s": f, "pooled_tokens_s": (n - r_) / (e - f),
                   "accepted_per_event": acc / ev, "resolved": sum(t["outcome"] == "R" for t in tasks.values()),
                   "failed_tests": sum(t["outcome"] == "F" for t in tasks.values()), "empty_patches": sum(t["outcome"] == "E" for t in tasks.values()),
                   "agent_minutes": sum(t["agent_s"] for t in tasks.values()) / 60, "tasks": tasks,
                   "timer_means_ms": timer_means([timers(W / "sidecars")]),
                   "split_k_engaged": "gqa_pair_splitk" in (W / "launch.log").read_text()}
assert out["workload"]["resolved"] + out["workload"]["failed_tests"] + out["workload"]["empty_patches"] == 10 and out["workload"]["split_k_engaged"]
prev = read(V2 / "results/claude-results-20261002/summaries/swe_study_20261001.json")
out["workload"]["prefix_reference"] = {a_: {k: prev[a_][k] for k in ("resolved", "agent_min_total", "pooled")} for a_ in prev if isinstance(prev[a_], dict)}


# ---- output-length comparability on identical prompts (tree fixed vs MTP-5 reference runs) ----------------
def lengths(p):
    rows = [json.loads(l) for l in Path(p).read_text().splitlines()]
    return {r["request"]: r["completion_tokens"] for r in rows}
def length_compare(tree_runs, mtp_glob_dirs):
    tl = [lengths(HERE / "raw/replay" / r["run"] / "replay.jsonl") for r in tree_runs]
    ml = [lengths(p) for p in mtp_glob_dirs]
    assert len(tl) == 3 and len(ml) == 3
    reqs = sorted(tl[0]); assert all(sorted(x) == reqs for x in tl + ml)
    mt = {q: sum(x[q] for x in tl) / 3 for q in reqs}; mm = {q: sum(x[q] for x in ml) / 3 for q in reqs}
    return {"tree_mean_total": sum(mt.values()), "mtp5_mean_total": sum(mm.values()),
            "requests_tree_longer": sum(mt[q] > mm[q] for q in reqs), "requests": len(reqs),
            "tree_run_totals": [sum(x.values()) for x in tl], "mtp5_run_totals": [sum(x.values()) for x in ml]}
mtp_tune = [p for p in sorted((V2 / "results").glob("*/raw/replay/mtp5-sampled-20261001T0[67]*/replay.jsonl"))]
mtp_tune = sorted({p.parent.name: p for p in mtp_tune}.values(), key=lambda p: p.parent.name)
mtp_conf = sorted({p.parent.name: p for p in (V2 / "results").glob("*/raw/replay/mtp5-cfM*/replay.jsonl")}.values(), key=lambda p: p.parent.name)
out["output_length_comparability"] = {"tuning_corpus": length_compare(tuning, mtp_tune), "confirmation_set": length_compare(confirm, mtp_conf),
                                      "mtp5_runs": [p.parent.name for p in mtp_tune + mtp_conf]}
(HERE / "AUDIT.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({"tuning": {k: out["tuning_corpus"]["tree_fixed"][k] for k in ("pooled_tokens_s", "run_rates", "accepted_per_event")}, "tuning_ratio": out["tuning_corpus"]["ratio_fixed_over_mtp5"],
                  "tuning_timers": out["tuning_corpus"]["tree_fixed_timer_means_ms"],
                  "confirm": {k: out["confirmation_set"]["tree_fixed"][k] for k in ("pooled_tokens_s", "run_rates", "accepted_per_event")}, "confirm_ratio": out["confirmation_set"]["ratio_fixed_over_mtp5"],
                  "confirm_timers": out["confirmation_set"]["tree_fixed_timer_means_ms"], "sampling": {k: out["sampling"][k] for k in out["sampling"] if k != "residual_nodes"},
                  "ties": out["sampling"]["residual_nodes"], "fullmodel": {k: out["fullmodel"][k] for k in out["fullmodel"] if k != "run"},
                  "workload": {k: out["workload"][k] for k in out["workload"] if k not in ("tasks", "prefix_reference")}, "output_lengths": out["output_length_comparability"], "tasks": {k: (t["outcome"], round(t["agent_s"] / 60, 1)) for k, t in sorted(tasks.items())}}, indent=1, default=str))
