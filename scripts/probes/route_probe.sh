#!/usr/bin/env bash
# route_probe.sh — FR13 turn-1 COLD-PREFILL ROUTE-FLIP replay probe (2026-07-04).
#
# GOAL: quantify the cold-prefill route split flagged by the give-up autopsy
# (research/fr13_workflows/FR13_GIVEUP_AUTOPSY.md §1/§5). The SAME byte-identical
# turn-1 qwen-code request (astropy-13453; sha-verified identical across all arms)
# is replayed N=8 times, each as a genuine COLD prefill, against three serve-only
# boots:
#     A  cat8_cache     = cat8 TREE + EXACT_SEED cache  (the GIVE-UP boot)
#     B  cat8_nocache   = cat8 TREE + APC CONFIG_ONLY   (the RESOLVING boot)
#     C  native_exseed  = native MTP-5 (FLASH_ATTN) + EXACT_SEED cache
# Readout (produced by route_probe_reduce.py): per-boot distribution of the turn-1
# ROUTE choice (Explore-subagent `agent` tool vs `read_file` vs other/NO_TOOL) plus
# the first-divergent-token position across the 8 samples.
#
# NO SWE AGENT, NO offload proxy, NO nudge net: each arm boots the server via the
# EXISTING forked launcher (scripts/fr13_launch_forked_fa2_tree_server.sh — the same
# one scripts/fr13_bigdenom_swe_serve_variant.sh drives), health-checks it, then
# POSTs the payload directly to local vLLM /v1/chat/completions. temp 0.6, no seed
# (matches deployment). Reset the prefix cache before every sample => every draw is
# a cold turn-1 prefill (turn 1 is always cold in deployment; the autopsy's first
# divergence is at cold prefill, row0_hit=False).
#
# The env per arm is reproduced VERBATIM from fr13_bigdenom_swe_serve_variant.sh's
# KIND dispatch (cat8 -> forked+CAT8_TREE; nativemtp5_exseed -> forked + native
# override XFLAGS) + the campaign cache env + guard floor 6500, and cross-checked
# against the baseline container_env.txt of each arm (output/fr13_baseline_main/).
#
# BUILD-ONLY: do NOT launch while the GPU is busy. It refuses to boot if any docker
# container is already running (fail-loud co-residency guard). Serial: one arm at a
# time, teardown between (GPU serialized on GB10).
#
# Usage:  bash route_probe.sh            # all three arms
#         ARMS="cat8_cache" bash route_probe.sh   # a single arm
# Env:    N=8  MAX_TOKENS=1024  TEMPERATURE=0.6  RESET_BETWEEN=1
#         GPU_UTIL=0.82  GPU_GUARD_FLOOR_MIB=6500
#         RR=output/fr13_route_probe   (results root; output/ is gitignored)
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true

STAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD="$STAGE/route_probe_payload.json"
PORT=9950
MODEL=qwen3.6-27b
N="${N:-16}"
RR="${RR:-output/fr13_route_probe}"
RESET_BETWEEN="${RESET_BETWEEN:-1}"
GPU_UTIL="${GPU_UTIL:-0.82}"
GPU_GUARD_FLOOR_MIB="${GPU_GUARD_FLOOR_MIB:-6500}"
ok=0  # top-level default: epilogue references ok under set -u after per-arm subshells
ARMS="${ARMS:-cat8_cache cat8_nocache native_exseed}"
export GPU_UTIL GPU_GUARD_FLOOR_MIB

# Integrity constants (canonical-JSON sha256 of the tool schema + the system+user
# messages of the ORIGINAL turn-1 request; computed at build time). Any drift in
# the payload's tools/messages vs the original request fails loud here.
EXPECT_TOOLS_SHA="9d12fd4bf2527b969da87523581cce38744ccdd9ab45eea7c433e3e8d44b9667"
EXPECT_MSGS_SHA="698ce22713371c4d9764b570e6f5b87915653df7b1415dd9caf528b51128f5df"

CAT8_TREE="[(0,),(1,),(0,0),(0,1),(0,0,0),(0,0,1),(0,0,0,0),(0,0,0,0,0)]"

mkdir -p "$RR"
echo "=== route_probe: N=$N reset_between=$RESET_BETWEEN arms=[$ARMS] -> $RR ==="
date -u +%Y-%m-%dT%H:%M:%SZ

# ---- FAIL-LOUD 0: payload exists + tools/messages match the original request ----
[[ -f "$PAYLOAD" ]] || { echo "FAIL: payload not found: $PAYLOAD"; exit 2; }
python3 - "$PAYLOAD" "$EXPECT_TOOLS_SHA" "$EXPECT_MSGS_SHA" "$MODEL" <<'PY'
import json, sys, hashlib
payload, exp_tools, exp_msgs, model = sys.argv[1:5]
d = json.load(open(payload))
def canon(o): return json.dumps(o, sort_keys=True, separators=(',',':'), ensure_ascii=False).encode()
assert d.get("model") == model, f"payload model {d.get('model')!r} != {model!r}"
assert "tools" in d and d["tools"], "payload has no tools"
assert "messages" in d and d["messages"], "payload has no messages"
ts = hashlib.sha256(canon(d["tools"])).hexdigest()
ms = hashlib.sha256(canon(d["messages"])).hexdigest()
assert ts == exp_tools, f"TOOL SCHEMA CHANGED: {ts} != {exp_tools}"
assert ms == exp_msgs,  f"MESSAGES CHANGED: {ms} != {exp_msgs}"
assert abs(float(d.get("temperature", -1)) - 0.6) < 1e-9, f"temperature != 0.6: {d.get('temperature')}"
assert "seed" not in d, "payload must NOT carry a seed (deployment uses seed=None)"
print(f"[integrity] OK model={model} n_tools={len(d['tools'])} tools_sha={ts[:12]} msgs_sha={ms[:12]} temp=0.6 no-seed")
PY
(( $? == 0 )) || { echo "FAIL: payload integrity"; exit 2; }

# ---- FAIL-LOUD 1: co-residency guard (never boot on top of a busy GPU) ----
if [[ -n "$(docker ps -q 2>/dev/null)" ]]; then
  echo "FAIL: a docker container is already running — refusing to boot (GPU is serialized). Current:"
  docker ps
  exit 2
fi

recover_host(){ PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python - <<'PY' 2>/dev/null || true
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY
}

# apply optional MAX_TOKENS / TEMPERATURE overrides -> a per-run send payload.
# Default (no overrides) copies the canonical payload byte-for-byte.
build_send_payload(){ # $1=dest
  python3 - "$PAYLOAD" "$1" "${MAX_TOKENS:-}" "${TEMPERATURE:-}" <<'PY'
import json, sys
src, dst, mx, tp = sys.argv[1:5]
d = json.load(open(src))
if mx: d["max_tokens"] = int(mx)
if tp: d["temperature"] = float(tp)
json.dump(d, open(dst, "w"), ensure_ascii=False)
PY
}

# per-arm boot env, VERBATIM from fr13_bigdenom_swe_serve_variant.sh + campaign.
arm_env(){ # echoes KEY=VAL lines for the arm (consumed via `export`)
  case "$1" in
    cat8_cache)     # == m_tree_cache_base (GIVE-UP): cat8 TREE + EXACT_SEED cache
      cat <<E
TREE=$CAT8_TREE
FR13_ENABLE_APC=1
FR13_APC_EXACT_SEED=1
MAMBA_BLOCK_SIZE=1024
MAMBA_SSM_CACHE_DTYPE=float32
E
      ;;
    cat8_nocache)   # == m_tree_nocache (RESOLVES): cat8 TREE + APC CONFIG_ONLY (cache OFF)
      cat <<E
TREE=$CAT8_TREE
FR13_ENABLE_APC=1
FR13_APC_CONFIG_ONLY=1
MAMBA_BLOCK_SIZE=1024
MAMBA_SSM_CACHE_DTYPE=float32
E
      ;;
    native_nocache) # 2x2 CONTROL (user 2026-07-04): native MTP-5 (FLASH_ATTN) + APC CONFIG_ONLY (cache OFF)
      cat <<E
ATTENTION_BACKEND=FLASH_ATTN
SPEC_CONFIG={"method":"qwen3_5_mtp","num_speculative_tokens":5}
FR10_DECODE_MODE_DEFAULT=naive_mtp
FR13_REPLAY_ROUTE=0
FR13_FA2_TREE_BIAS=0
FR13_FA2_PREFILL_NATIVE=0
FR13_TREE_SAMPLE_ROW=0
FR13_CONV_COMMITTED_PATH=0
FR13_ENABLE_APC=1
FR13_APC_CONFIG_ONLY=1
MAMBA_BLOCK_SIZE=1024
E
      ;;
    native_exseed)  # == m_native_cache: native MTP-5 (FLASH_ATTN) + EXACT_SEED cache
      cat <<E
ATTENTION_BACKEND=FLASH_ATTN
SPEC_CONFIG={"method":"qwen3_5_mtp","num_speculative_tokens":5}
FR10_DECODE_MODE_DEFAULT=naive_mtp
FR13_REPLAY_ROUTE=0
FR13_FA2_TREE_BIAS=0
FR13_FA2_PREFILL_NATIVE=0
FR13_TREE_SAMPLE_ROW=0
FR13_CONV_COMMITTED_PATH=0
FR13_ENABLE_APC=1
FR13_APC_EXACT_SEED=1
MAMBA_BLOCK_SIZE=1024
APC_BLOCK_SIZE=1024
MAMBA_SSM_CACHE_DTYPE=float32
FR13_APC_REQUIRE_SNAP_FIX=1
E
      ;;
    *) echo "FAIL_UNKNOWN_ARM"; return 1 ;;
  esac
}

# expected cache state per arm (hard-asserted vs the boot log)
arm_cache_expect(){ case "$1" in
  cat8_cache|native_exseed) echo "True";;
  cat8_nocache)             echo "False";;
esac; }

run_arm(){
  local arm="$1"
  local cdir="$RR/$arm"; local container="fr13-routeprobe-$arm"
  mkdir -p "$cdir/logs"
  echo ""
  echo "########## ARM $arm  ($(date -u +%H:%M:%S)) ##########"

  # env is scoped to a subshell so it never leaks across arms.
  (
    set -uo pipefail
    # common forked pins (mirror the serve_variant else-branch; matched all baseline
    # container_env.txt): tree numerics-neutral pins + guard + B=1.
    export FR10_METRICS=0 BATCH_INVARIANT=0 LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16
    export CONTAINER="$container" PORT="$PORT" MAX_NUM_SEQS=1
    export GPU_UTIL GPU_GUARD_FLOOR_MIB
    export LOG_DIR="$PWD/$cdir/logs" FR13_RUN_DIR="$PWD/$cdir"
    export FR13_SERVE_LOG=1
    # per-arm env
    local kv
    while IFS= read -r kv; do
      [[ -z "$kv" ]] && continue
      [[ "$kv" == "FAIL_UNKNOWN_ARM" ]] && { echo "FAIL: unknown arm $arm"; exit 2; }
      export "$kv"
    done < <(arm_env "$arm")

    docker rm -f "$container" >/dev/null 2>&1 || true
    recover_host
    # per-arm co-residency guard: no OTHER container may be running (GB10 unified mem).
    if [[ -n "$(docker ps -q 2>/dev/null)" ]]; then
      echo "FAIL: $arm — a container is still running before boot (co-residency):"; docker ps; exit 3
    fi

    echo "[$arm] boot via fr13_launch_forked_fa2_tree_server.sh (GPU_UTIL=$GPU_UTIL floor=$GPU_GUARD_FLOOR_MIB)"
    bash scripts/fr13_launch_forked_fa2_tree_server.sh > "$cdir/launch.log" 2>&1
    local rc=$?
    if (( rc != 0 )); then echo "FAIL: launcher rc=$rc"; tail -30 "$cdir/launch.log"; exit 3; fi

    # ---- health wait (<=1200s) ----
    local t0; t0=$(date +%s); local healthy=0
    while (( $(date +%s) < t0 + 1200 )); do
      if curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then healthy=1; break; fi
      if [[ "$(docker inspect -f '{{.State.Status}}' "$container" 2>/dev/null)" != "running" ]]; then
        echo "FAIL: container died before health"; docker logs "$container" 2>&1 | tail -40; exit 3
      fi
      sleep 5
    done
    (( healthy == 1 )) || { echo "FAIL: health not up in 1200s"; docker logs "$container" 2>&1 | tail -40; exit 3; }
    echo "[$arm] healthy after $(( $(date +%s) - t0 ))s"

    docker logs "$container" > "$cdir/boot_log_snapshot.txt" 2>&1
    docker exec "$container" env 2>/dev/null | sort > "$cdir/container_env.txt" || true

    # ---- FAIL-LOUD 2: served model matches the payload/original request ----
    local models_json; models_json=$(curl -fsS -m 10 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)
    echo "$models_json" > "$cdir/models.json"
    echo "$models_json" | grep -q "\"$MODEL\"" \
      || { echo "FAIL: served model does not advertise $MODEL"; echo "$models_json" | head -c 400; exit 4; }
    echo "[$arm] served model OK ($MODEL)"

    # ---- FAIL-LOUD 3: this is the RIGHT boot config (no vacuous/wrong arm) ----
    local exp_cache; exp_cache=$(arm_cache_expect "$arm")
    grep -q "enable_prefix_caching=$exp_cache" "$cdir/boot_log_snapshot.txt" \
      || { echo "FAIL: expected enable_prefix_caching=$exp_cache not in boot log for $arm"; \
           grep -aoE "enable_prefix_caching=(True|False)" "$cdir/boot_log_snapshot.txt" | head; exit 4; }
    case "$arm" in
      cat8_cache)
        grep -q "^FR13_APC_EXACT_SEED=1$" "$cdir/container_env.txt" || { echo "FAIL: $arm EXACT_SEED!=1"; exit 4; }
        grep -q "^ATTENTION_BACKEND=TREE_ATTN$" "$cdir/container_env.txt" || { echo "FAIL: $arm backend!=TREE_ATTN"; exit 4; }
        grep -q "speculative_token_tree" "$cdir/container_env.txt" || { echo "FAIL: $arm SPEC_CONFIG has no tree"; exit 4; }
        [[ -f "$cdir/logs/fr13_apc_exact_seed_eng.log" ]] || echo "WARN: $arm no fr13_apc_exact_seed_eng.log yet (fires on first cache turn)";;
      cat8_nocache)
        grep -q "^FR13_APC_EXACT_SEED=0$" "$cdir/container_env.txt" || { echo "FAIL: $arm EXACT_SEED!=0 (should be config-only)"; exit 4; }
        grep -q "^ATTENTION_BACKEND=TREE_ATTN$" "$cdir/container_env.txt" || { echo "FAIL: $arm backend!=TREE_ATTN"; exit 4; }
        grep -q "speculative_token_tree" "$cdir/container_env.txt" || { echo "FAIL: $arm SPEC_CONFIG has no tree"; exit 4; };;
      native_exseed)
        grep -q "^FR13_APC_EXACT_SEED=1$" "$cdir/container_env.txt" || { echo "FAIL: $arm EXACT_SEED!=1"; exit 4; }
        grep -q "^ATTENTION_BACKEND=FLASH_ATTN$" "$cdir/container_env.txt" || { echo "FAIL: $arm backend!=FLASH_ATTN"; exit 4; }
        grep -q "speculative_token_tree" "$cdir/container_env.txt" && { echo "FAIL: $arm SPEC_CONFIG carries a tree (should be native ns5)"; exit 4; };;
    esac
    # APC engagement bridge (mirrors serve_variant lines 414-437) — records live worker APC env
    docker exec "$container" sh -c 'cat /logs/fr13_apc_bridge_loaded.flag 2>/dev/null' > "$cdir/apc_bridge_marker.txt" 2>/dev/null || true
    echo "[$arm] boot-config asserts OK (cache=$exp_cache)"

    # ---- sample loop: N cold-prefill draws ----
    local send="$cdir/send_payload.json"; build_send_payload "$send"
    : > "$cdir/reset_log.txt"
    local i ok=0
    for (( i=1; i<=N; i++ )); do
      if [[ "$RESET_BETWEEN" == "1" ]]; then
        local rcode
        rcode=$(curl -s -o /dev/null -m 20 -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" 2>/dev/null)
        echo "sample $i reset_prefix_cache http=$rcode" >> "$cdir/reset_log.txt"
        [[ "$rcode" == "200" ]] || echo "WARN: $arm sample $i reset_prefix_cache http=$rcode (cold-prefill NOT guaranteed)"
      fi
      # PAIRED-SEED design (user 2026-07-04): per-request seed = sample index. The same seed under
      # two configs applies IDENTICAL sampling randomness to different logits (common random numbers)
      # -> any per-seed route difference is purely the config's logit shift, not sampling variance.
      python3 - "$send" "$cdir/send_seed_$i.json" "$i" <<'PYSEED'
import json, sys
d = json.load(open(sys.argv[1])); d["seed"] = int(sys.argv[3])
json.dump(d, open(sys.argv[2], "w"))
PYSEED
      curl -sS -m 900 -H 'Content-Type: application/json' -H 'Authorization: Bearer EMPTY' \
        --data-binary @"$cdir/send_seed_$i.json" "http://127.0.0.1:$PORT/v1/chat/completions" \
        > "$cdir/sample_$i.json" 2>"$cdir/sample_$i.err"
      # non-empty assert (task requirement): choices[0] must have a tool_call OR
      # non-empty content OR non-empty reasoning_content.
      if python3 - "$cdir/sample_$i.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print("  PARSE_FAIL", str(e)[:120]); sys.exit(1)
ch = (d.get("choices") or [])
if not ch: print("  NO_CHOICES", json.dumps(d)[:200]); sys.exit(1)
m = ch[0].get("message") or {}
tc = m.get("tool_calls") or []
txt = (m.get("content") or "").strip()
rsn = (m.get("reasoning_content") or "").strip()
if not (tc or txt or rsn): print("  EMPTY_COMPLETION"); sys.exit(1)
route = (tc[0].get("function",{}).get("name") if tc else "NO_TOOL")
print(f"  sample OK route={route} finish={ch[0].get('finish_reason')} n_tc={len(tc)} rsn={len(rsn)}c content={len(txt)}c")
PY
      then ok=$((ok+1)); else echo "  WARN: $arm sample $i empty/failed"; fi
    done
    echo "[$arm] non-empty samples: $ok/$N"

    # arm meta for the reducer
    python3 - "$cdir" "$arm" "$MODEL" "$exp_cache" "$N" "$ok" <<'PY'
import json, sys, os
cdir, arm, model, exp_cache, N, ok = sys.argv[1:7]
bl = open(os.path.join(cdir,"boot_log_snapshot.txt"),errors="ignore").read() if os.path.exists(os.path.join(cdir,"boot_log_snapshot.txt")) else ""
import re
pc = re.search(r"enable_prefix_caching=(True|False)", bl)
meta = {
  "arm": arm, "model_served": model, "cache_expected": exp_cache,
  "cache_boot_log": (pc.group(1) if pc else None),
  "N": int(N), "nonempty_samples": int(ok),
  "exact_seed_eng_log": os.path.exists(os.path.join(cdir,"logs","fr13_apc_exact_seed_eng.log")),
  "apc_bridge_marker": (open(os.path.join(cdir,"apc_bridge_marker.txt"),errors="ignore").read().strip()
                        if os.path.exists(os.path.join(cdir,"apc_bridge_marker.txt")) else ""),
}
json.dump(meta, open(os.path.join(cdir,"arm_meta.json"),"w"), indent=1)
print("[meta]", json.dumps(meta)[:300])
PY

    # ---- teardown ----
    docker logs "$container" > "$cdir/docker_full.log" 2>&1 || true
    docker rm -f "$container" >/dev/null 2>&1 || true
    recover_host
    sleep 2
    (( ok > 0 )) || { echo "FAIL: $arm produced ZERO non-empty completions"; exit 5; }
    echo "[$arm] DONE ($(date -u +%H:%M:%S))"
  )
  local arc=$?
  # belt-and-suspenders: ensure this arm's container is gone even on subshell failure
  docker rm -f "$container" >/dev/null 2>&1 || true
  return $arc
}

RC=0
for arm in $ARMS; do
  run_arm "$arm" || { echo "ARM $arm FAILED (rc=$?)"; RC=1; }
done

echo ""
echo "=== route_probe boots complete (rc=$RC). Reduce with: ==="
echo "    .venv/bin/python $STAGE/route_probe_reduce.py --run-root $RR"
exit $RC
