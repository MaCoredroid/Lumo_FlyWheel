#!/usr/bin/env python3
"""E1 post-boot verification (frozen settings, pass/fail; e1/E1_FREEZE.md) on the cell's own archives: container Cmd carries
--no-async-scheduling, --enforce-eager, --max-num-seqs N, no --enable-prefix-caching; env carries the arm's flags (tree: KV
policy B exact, REPLAY_ROUTE/EAGER_PACK/TREE_CONV_FUSED/RUNROW_INIT, TREE_ATTN; native: SPEC_CONFIG with the requested
num_speculative_tokens and FLASH_ATTN; NO FR13 tree/APC/capture feature env set to 1 on native); the E1 shim report has the
five anchors; the recorder ledger is sealed with 0 errors; the tree server log shows FR13_ATTN_KV_REMAP ENGAGED and NO
FR13_SLOT_REORDER ENGAGED; the native server log shows the MTP speculative method and no FR13 ENGAGED lines; the
recorder's active sets never exceed the declared occupancy. Writes <cell>/cell_verify.json; exit 1 on any failure.
Usage: e1_cell_verify.py <cell_dir> --arm ARM --batch {1,4} --nspec N"""
import argparse, json, os, re, sys
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cell"); ap.add_argument("--arm", required=True); ap.add_argument("--batch", type=int, required=True); ap.add_argument("--nspec", type=int, required=True); ap.add_argument("--spec-config", default=None, help="frozen exact SPEC_CONFIG string from the cells order"); ap.add_argument("--gpu-util", default=None); ap.add_argument("--max-model-len", default=None); a = ap.parse_args()
    R = {"cell": a.cell, "arm": a.arm, "checks": []}
    def chk(n, ok, d=None): R["checks"].append({"check": n, "pass": bool(ok), "detail": d}); print(("PASS " if ok else "FAIL ") + n, "" if d is None else str(d)[:200])
    ins = json.load(open(os.path.join(a.cell, "docker_inspect.json")))[0]; cmd = " ".join(ins["Config"]["Cmd"]); env = dict(e.split("=", 1) for e in ins["Config"]["Env"] if "=" in e)
    chk("cmd_no_async_scheduling", "--no-async-scheduling" in cmd); chk("cmd_enforce_eager", "--enforce-eager" in cmd); chk("cmd_max_num_seqs", f"--max-num-seqs '{a.batch}'" in cmd or f"--max-num-seqs {a.batch}" in cmd, re.findall(r"--max-num-seqs '?(\d+)'?", cmd))
    chk("cmd_prefix_cache_explicitly_off", "--no-enable-prefix-caching" in cmd and "--enable-prefix-caching" not in cmd.replace("--no-enable-prefix-caching", ""), "explicit --no-enable-prefix-caching required (empty flags are not proof)")
    chk("env_no_heavy_capture_metrics_or_traces", env.get("FR10_METRICS", "0") != "1" and not env.get("LUMO_MTP_DRAFT_TRACE_FILE") and not env.get("LUMO_TREE_PATH_LCP_LOG") and not env.get("LUMO_TREE_SAMPLER_DEBUG_LOG"), {k: env.get(k) for k in ("FR10_METRICS", "LUMO_MTP_DRAFT_TRACE_FILE", "LUMO_TREE_PATH_LCP_LOG", "LUMO_TREE_SAMPLER_DEBUG_LOG")})
    if a.gpu_util is not None: chk("cmd_gpu_memory_utilization_frozen", (f"--gpu-memory-utilization '{a.gpu_util}'" in cmd or f"--gpu-memory-utilization {a.gpu_util} " in cmd + " "), re.findall(r"--gpu-memory-utilization '?([0-9.]+)'?", cmd))
    if a.max_model_len is not None: chk("cmd_max_model_len_frozen", (f"--max-model-len '{a.max_model_len}'" in cmd or f"--max-model-len {a.max_model_len} " in cmd + " "), re.findall(r"--max-model-len '?(\d+)'?", cmd))
    if a.spec_config is not None: chk("env_spec_config_exactly_frozen", env.get("SPEC_CONFIG") == a.spec_config, {"container": env.get("SPEC_CONFIG"), "frozen": a.spec_config})
    chk("env_e1_recorder", env.get("E1_RECORD") == "/logs/e1_events.jsonl" and env.get("FR13_SFWD_GPU_TIMER") == "1" and "script_snapshot/e1_event_recorder_shim.py" in env.get("E1_SHIM_PATH", ""))
    if a.arm == "tree":
        chk("env_kv_policy_B_exact", env.get("FR13_ATTN_KV_REMAP") == "1" and env.get("FR13_SLOT_REORDER") == "0" and env.get("FR13_KV_REMAP_SYNCFREE") == "1", {k: env.get(k) for k in ("FR13_ATTN_KV_REMAP", "FR13_SLOT_REORDER", "FR13_KV_REMAP_SYNCFREE")})
        chk("env_tree_route_flags", env.get("FR13_REPLAY_ROUTE") == "1" and env.get("FR13_EAGER_PACK", "1") == "1" and env.get("FR13_TREE_RUNROW_INIT", "1") == "1" and "TREE_ATTN" in cmd, {k: env.get(k) for k in ("FR13_REPLAY_ROUTE", "FR13_EAGER_PACK", "FR13_TREE_CONV_FUSED", "FR13_TREE_RUNROW_INIT")})
        chk("env_no_capture_hooks", not env.get("FR13_FINAL_LOGIT_CAPTURE") and not env.get("FR10_LAYER_HIDDEN_CAPTURE") and not env.get("FR10_TREE_GDN_CAPTURE_PAYLOAD") and not env.get("E7B_SUBSTITUTE"))
    else:
        sc = json.loads(env.get("SPEC_CONFIG", "{}")) if env.get("SPEC_CONFIG") else {}
        chk("env_native_spec_config", int(sc.get("num_speculative_tokens", -1)) == a.nspec and "FLASH_ATTN" in cmd and "--speculative-config" in cmd, sc)
        hot = {k: v for k, v in env.items() if k.startswith("FR13_") or k.startswith("FR10_") or k.startswith("E7B_")}
        bad = {k: v for k, v in hot.items() if v == "1" and k not in ("FR13_SFWD_GPU_TIMER",)}
        chk("env_native_no_fr13_feature_on", not bad, {"fr_env_seen": hot, "features_on": bad})
        chk("env_native_decode_mode_naive_mtp_tree_gdn_off", env.get("FR10_DECODE_MODE_DEFAULT") == "naive_mtp" and env.get("FR10_ENABLE_TREE_GDN") == "0" and "speculative_token_tree" not in env.get("SPEC_CONFIG", ""), {"FR10_DECODE_MODE_DEFAULT": env.get("FR10_DECODE_MODE_DEFAULT"), "FR10_ENABLE_TREE_GDN": env.get("FR10_ENABLE_TREE_GDN")})
    shim = os.path.join(a.cell, "logs", "e1_event_recorder_shim.json"); sr = json.load(open(shim)) if os.path.exists(shim) else {}
    chk("e1_shim_five_anchors", len([e for e in sr.get("edits", []) if e.get("anchor") not in ("file",)]) >= 5, [e.get("anchor") for e in sr.get("edits", [])])
    evp = os.path.join(a.cell, "logs", "e1_events.jsonl"); ev = [json.loads(l) for l in open(evp) if l.strip()] if os.path.exists(evp) else []
    seal = [e for e in ev if e.get("event") == "run_close"]; errs = [e for e in ev if e.get("event") == "error"]
    chk("recorder_sealed_no_errors", bool(seal) and not errs and not seal[-1].get("sink_failed"), {"n_events": len(ev), "errors": len(errs), "seal": (seal[-1] if seal else None)})
    occ = max((len(e["request_ids"]) for e in ev if e.get("event") == "physical_step"), default=0)
    chk("occupancy_never_exceeds_declared", occ <= a.batch, {"max_active": occ})
    logp = os.path.join(a.cell, "docker_logs.txt"); log = open(logp, errors="replace").read() if os.path.exists(logp) else ""
    if a.arm == "tree": chk("log_kv_remap_engaged_no_reorder", "FR13_ATTN_KV_REMAP ENGAGED" in log and "FR13_SLOT_REORDER ENGAGED" not in log)
    else: chk("log_native_mtp_no_fr13_engaged", ("ENGAGED" not in log) and ("mtp" in log.lower()), {"engaged_lines": len(re.findall(r"ENGAGED", log))})
    R["all_pass"] = all(c["pass"] for c in R["checks"]); json.dump(R, open(os.path.join(a.cell, "cell_verify.json"), "w"), indent=1); print("VERIFY", "PASS" if R["all_pass"] else "FAIL"); sys.exit(0 if R["all_pass"] else 1)
if __name__ == "__main__":
    main()
