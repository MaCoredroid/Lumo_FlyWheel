import json, re, glob, subprocess, ast
root = "/home/mark/shared/lumotree-v2exp-runs/swe/"
arms = {"lumotree": glob.glob(root + "fr14_promoab_C_v2swe*_*/hydra27*/")[0],
        "vllm_mtp5": sorted(glob.glob(root + "mtp5-*/"))[-1],
        "sglang_eagle_s7_d8": sorted(glob.glob(root + "sglang-s7k1d8-*/"))[-1]}
out = {}
for a, d in arms.items():
    tasks = {}
    for l in open(d + "swe_orchestrator.log"):
        m = re.search(r"<- astropy__astropy-(\d+) verdict=(\w+) elapsed_total=([\d.]+)s", l)
        if m:
            tasks[m.group(1)] = {"verdict": m.group(2), "agent_s": float(m.group(3))}
    p = subprocess.run(["python3", "/home/mark/shared/lumotree-v2exp-20260930/scripts/v2exp/swe_pooled.py",
                        d + "metrics_before_swe.txt", d + "metrics_after_swe.txt"], capture_output=True, text=True).stdout
    out[a] = {"dir": d, "resolved": sum(t["verdict"] == "resolved" for t in tasks.values()),
              "agent_min_total": round(sum(t["agent_s"] for t in tasks.values()) / 60, 1),
              "pooled": ast.literal_eval(p.strip()), "tasks": tasks}
out["notes"] = ("Same 10 Astropy tasks (subset_b4_sixteen_qc_remainder_10), same Qwen Code 0.19.4 agent/proxy "
                "(temp 0.6/top_p 0.95/top_k 20/presence 1.0, 24k output cap), 9000 s budgets, network-isolated agent, "
                "one attempt per task per arm. LumoTree via the Cqc10 vehicle on latest code (split-K); MTP-5 via "
                "serve_native.sh (same forked FA2, APC/cudagraph flags); SGLang via its chat API (prompts render ~146 "
                "tokens longer than vLLM). Pooled rate = paper Eq. 5 from server counters. Single run per arm.")
json.dump(out, open("/home/mark/shared/lumotree-v2exp-20260930/scripts/v2exp/results/swe_study_20261001.json", "w"), indent=1)
print({a: (v["resolved"], v["agent_min_total"], v["pooled"]["pooled_decode_tok_s"]) for a, v in out.items() if a != "notes"})
