#!/usr/bin/env python3
"""CPU controls for e1_native_preflight.py (--live and --final; --scores explicit) and e1_cell_verify.py (incl. frozen gpu util /
max model len in the Cmd) on SYNTHETIC cell artifacts (docker
inspect Cmd/Env, preflight manifest, ledger prefix / sealed ledger, cohort responses with token ids, shim report, server
log). Uses the read-only checkpoint tokenizer to encode the E7a native reference first tokens. Positive: live PASS, final PASS,
verify PASS (native and tree). Negatives: gated prefix first token wrong -> live FAIL (P4); low-margin prefix wrong -> still
PASS (reported); 'ENGAGED' in the log -> FAIL; tree_mtp decode mode -> FAIL; missing output rows for a preflight request ->
FAIL (P5 live); unsealed ledger -> final FAIL; occupancy 2 with batch 1 -> FAIL; verify: SPEC_CONFIG mismatch -> FAIL;
FR10_METRICS=1 -> FAIL; missing --no-enable-prefix-caching -> FAIL. Usage: <out_json>"""
import json, os, subprocess, sys, tempfile
here = os.path.dirname(os.path.abspath(__file__)); res = {"checks": {}}
def check(n, ok, d=None): res["checks"][n] = {"pass": bool(ok), "detail": d}; print(("PASS " if ok else "FAIL ") + n, "" if d is None else str(d)[:180])
from transformers import AutoTokenizer  # type: ignore
tk = AutoTokenizer.from_pretrained("/models/qwen3.6-27b-fp8", local_files_only=True)
SC = os.path.join(here, "..", "out-20260921T231853Z-e7a-step2-native-score", "prefix_scores.json"); FR = os.path.join(here, "..", "out-20260921T231853Z-e7a-step2-native-score", "frozen_prefixes.json")
ref = {p["id"]: p for p in json.load(open(SC))["prefixes"]}; pil = [r["id"] for r in json.load(open(FR))["pilot"]]
def first_id(pid): ids = tk.encode(ref[pid]["native_top1"], add_special_tokens=False); return ids[0]
NATIVE_SPEC = '{"method":"qwen3_5_mtp","num_speculative_tokens":5}'
def make(d, arm="native-5", batch=1, wrong_gated=False, wrong_low=False, engaged=False, mode="naive_mtp", drop_rows=False, sealed=True, occ2=False, spec=NATIVE_SPEC, metrics="0", apc_flag=True, with_main=False, gpu_util="0.6", mml="16384"):
    os.makedirs(os.path.join(d, "logs"), exist_ok=True); os.makedirs(os.path.join(d, "cohort"), exist_ok=True)
    cmd = ["vllm", "serve", "/models/x", "--max-num-seqs", str(batch), "--gpu-memory-utilization", f"'{gpu_util}'", "--max-model-len", f"'{mml}'", "--attention-backend", ("TREE_ATTN" if arm == "tree" else "FLASH_ATTN"), "--speculative-config", spec, "--enforce-eager", "--no-async-scheduling"] + (["--no-enable-prefix-caching"] if apc_flag else [])
    env = {"SPEC_CONFIG": spec, "E1_RECORD": "/logs/e1_events.jsonl", "FR13_SFWD_GPU_TIMER": "1", "E1_SHIM_PATH": "/workspace/x/script_snapshot/e1_event_recorder_shim.py", "FR10_METRICS": metrics, "FR13_ENABLE_APC": "0"}
    if arm == "tree": env.update({"FR13_ATTN_KV_REMAP": "1", "FR13_SLOT_REORDER": "0", "FR13_KV_REMAP_SYNCFREE": "1", "FR13_REPLAY_ROUTE": "1", "FR13_EAGER_PACK": "1", "FR13_TREE_RUNROW_INIT": "1"})
    else: env.update({"FR10_DECODE_MODE_DEFAULT": mode, "FR10_ENABLE_TREE_GDN": "0"})
    json.dump([{"Config": {"Cmd": cmd, "Env": [f"{k}={v}" for k, v in env.items()]}}], open(os.path.join(d, "docker_inspect.json"), "w"))
    open(os.path.join(d, "docker_logs_preflight.txt"), "w").write("INFO speculative method mtp loaded\n" + ("FR13_ATTN_KV_REMAP ENGAGED\n" if engaged else "")); open(os.path.join(d, "docker_logs.txt"), "w").write("INFO speculative method mtp\n" + ("FR13_ATTN_KV_REMAP ENGAGED\n" if (engaged or arm == "tree") else ""))
    json.dump({"edits": [{"anchor": a} for a in ("F", "B1", "B2", "W", "R", "file")]}, open(os.path.join(d, "logs", "e1_event_recorder_shim.json"), "w"))
    ev = [{"event": "recorder_probe", "n": 1}]; n = 1; seq = 0; reqs = []; slots = []
    for i, pid in enumerate(pil):
        slot = f"f{i}"; api = f"cmpl-{i:02d}abc"; rid = api + "-0-x"; fid = first_id(pid)
        if wrong_gated and pid == "p015": fid = fid + 1
        if wrong_low and pid == "p017": fid = fid + 1
        ids = [fid] + [100 + j for j in range(31)]
        os.makedirs(os.path.join(d, "cohort", f"req_{slot}_{pid}"), exist_ok=True); json.dump({"response_id": api, "finish_reason": "length", "usage": {"prompt_tokens": 256, "completion_tokens": 32}, "response_logprobs_tokens": [f"token_id:{t}" for t in ids]}, open(os.path.join(d, "cohort", f"req_{slot}_{pid}", "capture_request.json"), "w"))
        reqs.append({"phase": "preflight", "slot": slot, "prefix_id": pid, "response_id": api, "n_api_tokens": 32, "finish_reason": "length", "max_tokens": 32}); slots.append(slot)
        for k in range(2):
            seq += 1; n += 1; ev.append({"event": "forward_entry", "seq": seq, "num_reqs": 1, "num_tokens": 6, "n": n})
            rids = [rid] + (["cmpl-zz-0-x"] if (occ2 and k == 1) else [])
            n += 1; ev.append({"event": "physical_step", "seq": seq, "physical_step_id": seq, "t_start": 0.1 * seq, "num_reqs": len(rids), "request_ids": rids, "n": n})
            if not (drop_rows and i == 3): n += 1; ev.append({"event": "output_rows", "seq": seq, "kind": "pure", "physical_step_id": seq, "num_reqs": len(rids), "rows": [{"row": 0, "request_id": rid, "step_idx": k, "emitted_ids": ids[k * 16:(k + 1) * 16], "n_emitted": 16, "num_draft_tokens": 5, "discarded": False}], "n": n})
    man = {"phases": {"preflight": [r["response_id"] for r in reqs], "warmup": [], "timed": []}, "batch": batch, "submission": [{"phase": "preflight", "slots": [s], "utc": "x"} for s in slots], "requests": reqs}
    json.dump(man, open(os.path.join(d, "e1_preflight_manifest.json"), "w")); json.dump(man, open(os.path.join(d, "e1_manifest.json"), "w"))
    if sealed: n += 1; ev.append({"event": "run_close", "reason": "atexit", "n_events": n, "sink_failed": False, "n": n})
    with open(os.path.join(d, "logs", "e1_events.jsonl"), "w") as f:
        for e in ev: f.write(json.dumps(e) + "\n")
def pre(d, mode, arm="native-5", batch=1, nspec=5): r = subprocess.run([sys.executable, os.path.join(here, "e1_native_preflight.py"), d, "--arm", arm, "--batch", str(batch), "--nspec", str(nspec), mode, "--scores", SC], capture_output=True, text=True); return r.returncode, r.stdout
def ver(d, arm="native-5", batch=1, nspec=5, spec=NATIVE_SPEC): r = subprocess.run([sys.executable, os.path.join(here, "e1_cell_verify.py"), d, "--arm", arm, "--batch", str(batch), "--nspec", str(nspec), "--spec-config", spec, "--gpu-util", "0.6", "--max-model-len", "16384"], capture_output=True, text=True); return r.returncode, r.stdout
with tempfile.TemporaryDirectory() as td:
    d = os.path.join(td, "ok"); make(d); rc, o = pre(d, "--live"); check("live_pass", rc == 0, o.splitlines()[-1:]); rc, o = pre(d, "--final"); check("final_pass", rc == 0, o.splitlines()[-1:]); rc, o = ver(d); check("verify_native_pass", rc == 0, [l for l in o.splitlines() if l.startswith("FAIL")])
    d = os.path.join(td, "wg"); make(d, wrong_gated=True); rc, o = pre(d, "--live"); check("gated_prefix_wrong_first_token_fail_P4", rc == 1 and "FAIL P4" in o)
    d = os.path.join(td, "wl"); make(d, wrong_low=True); rc, o = pre(d, "--live"); j = json.load(open(os.path.join(d, "native_preflight_live.json"))); p4 = [c for c in j["checks"] if c["check"].startswith("P4")][0]["detail"]
    check("low_margin_prefix_wrong_still_pass_reported", rc == 0 and any(x["prefix"] == "p017" and x["equal"] is False for x in p4["reported_low_margin_not_gated"]) and all(x["equal"] for x in p4["gated"]), {"reported": [(x["prefix"], x["equal"]) for x in p4["reported_low_margin_not_gated"]]})
    d = os.path.join(td, "eng"); make(d, engaged=True); rc, o = pre(d, "--live"); check("engaged_line_fail_P1", rc == 1 and "FAIL P1_no_fr13_engaged" in o)
    d = os.path.join(td, "mode"); make(d, mode="tree_mtp"); rc, o = pre(d, "--live"); check("tree_mtp_decode_mode_fail_P1", rc == 1 and "FAIL P1_decode_mode" in o)
    d = os.path.join(td, "rows"); make(d, drop_rows=True); rc, o = pre(d, "--live"); check("missing_output_rows_fail_P5_live", rc == 1 and "FAIL P5_live" in o)
    d = os.path.join(td, "unsealed"); make(d, sealed=False); rc, o = pre(d, "--final"); check("unsealed_ledger_final_fail_P5", rc == 1 and "FAIL P5_final" in o); rc, o = pre(d, "--live"); check("unsealed_ledger_live_still_pass_no_seal_claimed", rc == 0)
    d = os.path.join(td, "occ"); make(d, occ2=True); rc, o = pre(d, "--live"); check("occupancy_2_with_batch_1_fail_P2", rc == 1 and "FAIL P2_occupancy" in o)
    d = os.path.join(td, "spec"); make(d); rc, o = ver(d, spec='{"method":"qwen3_5_mtp","num_speculative_tokens":11}'); check("verify_spec_config_mismatch_fail", rc == 1 and "FAIL env_spec_config_exactly_frozen" in o)
    d = os.path.join(td, "met"); make(d, metrics="1"); rc, o = ver(d); check("verify_metrics_on_fail", rc == 1 and "FAIL env_no_heavy_capture" in o)
    d = os.path.join(td, "apc"); make(d, apc_flag=False); rc, o = ver(d); check("verify_missing_explicit_no_prefix_caching_fail", rc == 1 and "FAIL cmd_prefix_cache_explicitly_off" in o)
    d = os.path.join(td, "gpu"); make(d, gpu_util="0.9"); rc, o = ver(d); check("verify_gpu_util_not_frozen_fail", rc == 1 and "FAIL cmd_gpu_memory_utilization_frozen" in o)
    d = os.path.join(td, "mml"); make(d, mml="8192"); rc, o = ver(d); check("verify_max_model_len_not_frozen_fail", rc == 1 and "FAIL cmd_max_model_len_frozen" in o)
    TREE_SPEC = '{"method":"qwen3_5_mtp","num_speculative_tokens":9,"speculative_token_tree":"[(0,), (0, 0)]"}'
    d = os.path.join(td, "tree"); make(d, arm="tree", spec=TREE_SPEC); rc, o = ver(d, arm="tree", nspec=9, spec=TREE_SPEC); check("verify_tree_pass", rc == 0, [l for l in o.splitlines() if l.startswith("FAIL")])
res["all_pass"] = all(v["pass"] for v in res["checks"].values()); json.dump(res, open(sys.argv[1], "w"), indent=2); print("ALL PASS" if res["all_pass"] else "SOME FAILED"); sys.exit(0 if res["all_pass"] else 1)
