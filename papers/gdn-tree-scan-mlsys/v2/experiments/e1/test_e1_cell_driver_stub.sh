#!/usr/bin/env bash
# E1 cell-driver preflight (NO GPU, no container): exercise the REAL snapshot/re-exec path of e1_cell_driver.v1.sh with STUB
# launchers (they dump the environment they received and exit 1, so the driver aborts before any container). A scratch
# mirror of experiments/ (e1, e7a, e2 copied verbatim; the driver copy differs only in its WT= line) is used. Cells 1
# (native-5 B1), 2 (native-5 B4), 5 (tree B1), 6 (tree B4). Asserts per cell: executed driver = snapshot copy (trace line),
# SHA256SUMS lists the launchers/shim/recorder/joiner/workload/verify/summary/preflight, the stub launcher received the
# frozen env (MAX_NUM_SEQS, VLLM_SYNC_SCHED=1, ENFORCE_EAGER=1, FR13_ENABLE_APC=0, E1_* + FR13_SFWD_GPU_TIMER=1, tree: policy B
# 1/0/1 + REPLAY_ROUTE/EAGER_PACK/TREE_CONV_FUSED/RUNROW_INIT=1, TREE_ATTN, NO capture env; native: NUM_SPECULATIVE_TOKENS,
# FLASH_ATTN, FR10_DECODE_MODE_DEFAULT=naive_mtp, FR10_ENABLE_TREE_GDN=0), and no docker_inspect.json. Usage: <out_json>
set -uo pipefail
OUT=$1; WT=/home/mark/lumo-paper-v2-20260921; EXP=$WT/papers/gdn-tree-scan-mlsys/v2/experiments
S=$(mktemp -d /tmp/claude-1000/-home-mark-lumo-paper-v2-20260921/c8955ff9-f3ae-4551-869b-1a98b38ca2c3/scratchpad/e1drv.XXXX)
mkdir -p "$S/papers/gdn-tree-scan-mlsys/v2/experiments" "$S/scripts"
for d in e1 e7a e2; do cp -r "$EXP/$d" "$S/papers/gdn-tree-scan-mlsys/v2/experiments/$d"; done
mkdir -p "$S/papers/gdn-tree-scan-mlsys/v2/experiments/out-20260921T231853Z-e7a-step2-native-score"; cp "$EXP/out-20260921T231853Z-e7a-step2-native-score/frozen_prefixes.json" "$S/papers/gdn-tree-scan-mlsys/v2/experiments/out-20260921T231853Z-e7a-step2-native-score/"
cp "$WT/scripts/gpu_oom_guard.sh" "$S/scripts/"; git -C "$WT" rev-parse HEAD >/dev/null
SE="$S/papers/gdn-tree-scan-mlsys/v2/experiments"
for L in e1/e1_native_launch.v2.sh e7a/e7a_capture_launch.v7.sh; do cat > "$SE/$L" <<'STUB'
#!/usr/bin/env bash
env | grep -E '^(MAX_NUM_SEQS|VLLM_SYNC_SCHED|ENFORCE_EAGER|FR13_ENABLE_APC|E1_SHIM_PATH|E1_RECORD|E1_RUNTIME_DIR|FR13_SFWD_GPU_TIMER|FR13_ATTN_KV_REMAP|FR13_SLOT_REORDER|FR13_KV_REMAP_SYNCFREE|FR13_REPLAY_ROUTE|FR13_EAGER_PACK|FR13_TREE_CONV_FUSED|FR13_TREE_RUNROW_INIT|ATTENTION_BACKEND|NUM_SPECULATIVE_TOKENS|FR10_DECODE_MODE_DEFAULT|FR10_ENABLE_TREE_GDN|FR13_FINAL_LOGIT_CAPTURE|FR10_LAYER_HIDDEN_CAPTURE|FR10_TREE_GDN_CAPTURE_PAYLOAD|E7B_SUBSTITUTE|E7A_CAPTURE_SHIM|CONTAINER|LOG_DIR|SEED|GPU_UTIL|MAX_MODEL_LEN|FR10_METRICS|LUMO_MTP_DRAFT_TRACE_FILE|LUMO_TREE_PATH_LCP_LOG|LUMO_TREE_SAMPLER_DEBUG_LOG)=' | sort > "${STUB_RECORD:?}"; exit 1
STUB
chmod +x "$SE/$L"; done
sed "s#^WT=/home/mark/lumo-paper-v2-20260921\$#WT=$S#" "$EXP/e1/e1_cell_driver.v1.sh" > "$SE/e1/e1_cell_driver.v1.sh"
python3 - "$OUT" "$S" "$SE" <<'PY'
import json, os, subprocess, sys
out, S, SE = sys.argv[1:4]; res = {"checks": {}, "scratch": S}
def check(n, ok, d=None): res["checks"][n] = {"pass": bool(ok), "detail": d}; print(("PASS " if ok else "FAIL ") + n, "" if d is None else str(d)[:160])
cells = f"{SE}/e1/e1_cells.v1.json"; root = f"{S}/runs"; os.makedirs(root, exist_ok=True)
for idx, arm, nseq, nspec in [(1, "native-5", 1, 5), (2, "native-5", 4, 5), (5, "tree", 1, 9), (6, "tree", 4, 9), (3, "native-11", 4, 11)]:
    rec = f"{S}/stub_{idx}.txt"; env = dict(os.environ, STUB_RECORD=rec, E1_POOL="/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/out-20260921T230815Z-e7a-step2-prefixes/prefix_pool.json", E1_FROZEN="/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/out-20260921T231853Z-e7a-step2-native-score/frozen_prefixes.json")
    r = subprocess.run(["bash", f"{SE}/e1/e1_cell_driver.v1.sh", cells, str(idx), root], env=env, capture_output=True, text=True, timeout=600)
    cell = [d for d in os.listdir(root) if d.startswith(f"cell_{idx:02d}_")]; cell = os.path.join(root, cell[0]) if cell else None; tag = f"cell{idx}_{arm}_B{nseq}"
    tr = open(os.path.join(cell, "driver_trace.txt")).read() if cell and os.path.exists(os.path.join(cell, "driver_trace.txt")) else ""
    check(f"{tag}_driver_reexec_from_snapshot", "driver_exec=e1_cell_driver.v1.sh" in tr and "from_snapshot=1" in tr and f"snapshot_dir={cell}/script_snapshot" in tr, tr.splitlines()[:2])
    sums = open(os.path.join(cell, "script_snapshot", "SHA256SUMS")).read() if cell and os.path.exists(os.path.join(cell, "script_snapshot", "SHA256SUMS")) else ""
    need = ["e1_native_launch.v2.sh", "e7a_capture_launch.v7.sh", "e1_event_recorder_shim.py", "e1_recorder.py", "e1_join.py", "e1_workload.py", "e1_cell_verify.py", "e1_cell_summary.py", "e1_native_preflight.py", "e1_api_tokens_from_capture.v2.py", "e1_cells.json", "frozen_prefixes.json", "gpu_oom_guard.sh"]
    check(f"{tag}_snapshot_complete", all(n in sums for n in need), [n for n in need if n not in sums])
    stub = open(rec).read() if os.path.exists(rec) else ""; e = dict(l.split("=", 1) for l in stub.splitlines() if "=" in l)
    common = e.get("MAX_NUM_SEQS") == str(nseq) and e.get("VLLM_SYNC_SCHED") == "1" and e.get("ENFORCE_EAGER") == "1" and e.get("FR13_ENABLE_APC") == "0" and e.get("E1_RECORD") == "/logs/e1_events.jsonl" and e.get("FR13_SFWD_GPU_TIMER") == "1" and "script_snapshot/e1_event_recorder_shim.py" in e.get("E1_SHIM_PATH", "") and e.get("SEED") == "20260921"
    check(f"{tag}_no_heavy_capture", e.get("FR10_METRICS") == "0" and not e.get("LUMO_MTP_DRAFT_TRACE_FILE") and not e.get("LUMO_TREE_PATH_LCP_LOG") and not e.get("LUMO_TREE_SAMPLER_DEBUG_LOG"), {k: e.get(k) for k in ("FR10_METRICS", "LUMO_MTP_DRAFT_TRACE_FILE")})
    check(f"{tag}_frozen_common_env", common, {k: e.get(k) for k in ("MAX_NUM_SEQS", "VLLM_SYNC_SCHED", "ENFORCE_EAGER", "FR13_ENABLE_APC", "E1_RECORD", "FR13_SFWD_GPU_TIMER", "SEED")})
    if arm == "tree":
        check(f"{tag}_policy_B_and_route_flags_no_capture", e.get("FR13_ATTN_KV_REMAP") == "1" and e.get("FR13_SLOT_REORDER") == "0" and e.get("FR13_KV_REMAP_SYNCFREE") == "1" and e.get("FR13_REPLAY_ROUTE") == "1" and e.get("FR13_EAGER_PACK") == "1" and e.get("FR13_TREE_RUNROW_INIT") == "1" and e.get("ATTENTION_BACKEND") == "TREE_ATTN" and e.get("E7A_CAPTURE_SHIM") == "0" and not e.get("FR13_FINAL_LOGIT_CAPTURE") and not e.get("FR10_LAYER_HIDDEN_CAPTURE") and not e.get("FR10_TREE_GDN_CAPTURE_PAYLOAD") and not e.get("E7B_SUBSTITUTE"), {k: e.get(k) for k in ("FR13_ATTN_KV_REMAP", "FR13_SLOT_REORDER", "FR13_KV_REMAP_SYNCFREE", "ATTENTION_BACKEND", "FR13_FINAL_LOGIT_CAPTURE")})
    else:
        check(f"{tag}_native_route_env", e.get("NUM_SPECULATIVE_TOKENS") == str(nspec) and e.get("ATTENTION_BACKEND") == "FLASH_ATTN" and e.get("FR10_DECODE_MODE_DEFAULT") == "naive_mtp" and e.get("FR10_ENABLE_TREE_GDN") == "0" and not e.get("FR13_ATTN_KV_REMAP"), {k: e.get(k) for k in ("NUM_SPECULATIVE_TOKENS", "ATTENTION_BACKEND", "FR10_DECODE_MODE_DEFAULT", "FR10_ENABLE_TREE_GDN")})
    check(f"{tag}_aborted_without_container", r.returncode != 0 and cell is not None and not os.path.exists(os.path.join(cell, "docker_inspect.json")), r.returncode)
res["all_pass"] = all(v["pass"] for v in res["checks"].values()); json.dump(res, open(out, "w"), indent=2); print("ALL PASS" if res["all_pass"] else "SOME FAILED"); sys.exit(0 if res["all_pass"] else 1)
PY
rc=$?; echo "scratch: $S"; exit $rc
