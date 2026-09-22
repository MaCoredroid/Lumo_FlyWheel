#!/usr/bin/env python3
"""Bounded CPU check of the capture LAYER selector (review 08 preflight).
1. The patcher's hook compares the env selector with str(self.prefix) for EXACT equality (source assertion).
2. The served checkpoint's config.json layer_types gives the linear_attention layers; the module naming scheme
   language_model.model.layers.<i>.linear_attn is confirmed by prefixes PRINTED AT RUNTIME in the attempt-3 docker log.
3. Simulated selector: "62" fires on no layer; the full prefix fires on exactly one layer (62, linear_attention).
4. The v4 driver refuses a bare-number selector and accepts the full prefix (its embedded validator, run standalone).
5. NUM_TOKENS filter {10} contains the real tree step size (tree_n from the attempt-3 depth-position records).
Usage: test_capture_selector.py <patcher> <attempt3_run_dir> <out_json>"""
import json, re, subprocess, sys, hashlib
patcher, run, outp = sys.argv[1:4]
res = {"checks": {}}
def check(n, ok, d=None): res["checks"][n] = {"pass": bool(ok), "detail": d}; print(("PASS " if ok else "FAIL ") + n, "" if d is None else str(d)[:160])
src = open(patcher).read(); res["patcher_sha256"] = hashlib.sha256(src.encode()).hexdigest()
check("hook_selector_is_exact_equality_with_str_self_prefix", '_fr10_capture_prefix != str(self.prefix)' in src and 'FR10_TREE_GDN_CAPTURE_PAYLOAD_LAYER_PREFIX' in src)
cfg = json.load(open('/models/qwen3.6-27b-fp8/config.json')); tc = cfg.get("text_config") or cfg; lt = tc["layer_types"]
linear = [i for i, t in enumerate(lt) if t == "linear_attention"]
check("config_layer_62_is_linear_attention", lt[62] == "linear_attention", {"n_layers": len(lt), "n_linear": len(linear)})
log = open(f"{run}/docker_logs.txt").read()
printed = sorted(set(re.findall(r"language_model\.model\.layers\.(\d+)\.linear_attn", log)), key=int)
check("runtime_printed_naming_scheme_matches", len(printed) > 0 and all(lt[int(i)] == "linear_attention" for i in printed), {"printed_layers": printed})
prefixes = [f"language_model.model.layers.{i}.linear_attn" for i in range(len(lt)) if lt[i] == "linear_attention"]
def fires(sel): return [p for p in prefixes if not (sel and sel != p)]
check("bare_number_selector_fires_on_no_layer", fires("62") == [], fires("62"))
check("full_prefix_selector_fires_on_exactly_layer_62", fires("language_model.model.layers.62.linear_attn") == ["language_model.model.layers.62.linear_attn"])
# driver validator (extract the embedded python between PY2 markers and run it standalone)
drv = open(__file__.replace("test_capture_selector.py", "serve_drivers.v4.sh")).read()
m = re.search(r"<<'PY2'.*?\n(.*?)\nPY2\n", drv, re.S); code = m.group(1).replace("\\\\", "\\") if m else ""
def run_validator(sel):
    r = subprocess.run([sys.executable, "-c", code, sel, "/models/qwen3.6-27b-fp8"], capture_output=True, text=True); return r.returncode, r.stdout.strip()
rc_bad, o_bad = run_validator("62"); rc_ok, o_ok = run_validator("language_model.model.layers.62.linear_attn"); rc_full, o_full = run_validator("language_model.model.layers.63.linear_attn")
check("driver_v4_refuses_bare_number", rc_bad != 0, o_bad); check("driver_v4_accepts_full_prefix_linear_layer", rc_ok == 0, o_ok); check("driver_v4_refuses_full_attention_layer_63", rc_full != 0, o_full)
dp = [json.loads(l) for l in open(f"{run}/logs/fr10_tree_depth_positions.jsonl") if l.strip()]
check("num_tokens_filter_10_contains_real_tree_n", all(r.get("tree_n") == 10 and r.get("num_scheduled_tokens") == [10] for r in dp) and len(dp) > 0, {"records": len(dp)})
res["all_pass"] = all(c["pass"] for c in res["checks"].values()); json.dump(res, open(outp, "w"), indent=2)
print("ALL PASS" if res["all_pass"] else "SOME FAILED"); sys.exit(0 if res["all_pass"] else 1)
