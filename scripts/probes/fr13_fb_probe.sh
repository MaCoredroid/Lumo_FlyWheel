#!/usr/bin/env bash
# fr13_fb_probe.sh — FR13 §61 FORCED-BOUNDARY attribution probe (2026-07-05).
#
# PURPOSE (mechanism attribution, NOT a behavior gate — live SWE stays the gate class):
# force >1024-token generations on the REAL astropy turn-1 payload so decode crosses
# mamba block boundaries (decode-side snapshot writes — the cell no prior probe ever
# exercised: §55 obs snapshot_events=0), then re-hit the crossed blocks on turns 2/3.
# Readout = garble detectors per turn + snapshot/hit engagement brackets.
#
# ARMS (route_probe.sh presets):
#   cat8_cache    tree + EXACT_SEED cache (fixes ON via env)   -> predict garble if
#                 restart-fold-mismatch story holds (t2/t3 amplified)
#   cat8_nocache  tree + CONFIG_ONLY (no cache machinery)      -> clean unless long-gen
#                 itself is the trigger
#   native_exseed native MTP-5 + EXACT_SEED cache              -> tree-specificity test
#
# Per seed: reset -> t1 (cold, min_tokens forces the crossing) -> t2 resend+extend
# (hits incl. crossed blocks) -> t3 extend again.
# Env: N=6 MAX_TOKENS=1600 MIN_TOKENS=1200 RR=output/fr13_fb_probe
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true

STAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD="$STAGE/route_probe_payload.json"
PORT=9950
MODEL=qwen3.6-27b
N="${N:-6}"
RR="${RR:-output/fr13_fb_probe}"
GPU_UTIL="${GPU_UTIL:-0.82}"
GPU_GUARD_FLOOR_MIB="${GPU_GUARD_FLOOR_MIB:-6500}"
MAX_TOKENS="${MAX_TOKENS:-1600}"
MIN_TOKENS="${MIN_TOKENS:-1200}"
ARMS="${ARMS:-cat8_cache cat8_nocache native_exseed}"
export GPU_UTIL GPU_GUARD_FLOOR_MIB
CAT8_TREE="[(0,),(1,),(0,0),(0,1),(0,0,0),(0,0,1),(0,0,0,0),(0,0,0,0,0)]"

mkdir -p "$RR"
echo "=== fb_probe: N=$N max=$MAX_TOKENS min=$MIN_TOKENS arms=[$ARMS] -> $RR ==="
date -u +%Y-%m-%dT%H:%M:%SZ

[[ -f "$PAYLOAD" ]] || { echo "FAIL: payload missing"; exit 2; }
if [[ -n "$(docker ps -q 2>/dev/null)" ]]; then echo "FAIL: container running"; docker ps; exit 2; fi

recover_host(){ PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python - <<'PY' 2>/dev/null || true
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY
}

arm_env(){ case "$1" in
    cat8_cache) cat <<E
TREE=$CAT8_TREE
FR13_ENABLE_APC=1
FR13_APC_EXACT_SEED=1
MAMBA_BLOCK_SIZE=1024
MAMBA_SSM_CACHE_DTYPE=float32
FR13_APC_ZERO_MAMBA_ON_ALLOC=1
E
      ;;
    cat8_nocache) cat <<E
TREE=$CAT8_TREE
FR13_ENABLE_APC=1
FR13_APC_CONFIG_ONLY=1
MAMBA_BLOCK_SIZE=1024
MAMBA_SSM_CACHE_DTYPE=float32
FR13_APC_ZERO_MAMBA_ON_ALLOC=1
E
      ;;
    native_exseed) cat <<E
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
FR13_APC_ZERO_MAMBA_ON_ALLOC=1
E
      ;;
    *) echo "FAIL_UNKNOWN_ARM"; return 1 ;;
  esac
}
arm_cache_expect(){ case "$1" in cat8_cache|native_exseed) echo "True";; cat8_nocache) echo "False";; esac; }

turn_send(){ # $1=arm_dir $2=seed $3=turn_idx $4=messages_json_file -> writes t{n}_s{seed}.json
  local cdir=$1 seed=$2 tn=$3 msgs=$4
  python3 - "$PAYLOAD" "$msgs" "$cdir/send_t${tn}_s${seed}.json" "$seed" "$MAX_TOKENS" "$MIN_TOKENS" <<'PY'
import json, sys
base, msgs_f, dst, seed, mx, mn = sys.argv[1:7]
d = json.load(open(base))
d["messages"] = json.load(open(msgs_f))
d["seed"] = int(seed); d["max_tokens"] = int(mx); d["min_tokens"] = int(mn)
d["temperature"] = 0.6
json.dump(d, open(dst, "w"), ensure_ascii=False)
PY
  curl -sS -m 900 -H 'Content-Type: application/json' -H 'Authorization: Bearer EMPTY' \
    --data-binary @"$cdir/send_t${tn}_s${seed}.json" "http://127.0.0.1:$PORT/v1/chat/completions" \
    > "$cdir/t${tn}_s${seed}.json" 2>"$cdir/t${tn}_s${seed}.err"
}

run_arm(){
  local arm="$1"; local cdir="$RR/$arm"; local container="fr13-fb-$arm"
  mkdir -p "$cdir/logs"
  echo ""; echo "########## ARM $arm ($(date -u +%H:%M:%S)) ##########"
  (
    set -uo pipefail
    export FR10_METRICS=0 BATCH_INVARIANT=0 LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16
    export CONTAINER="$container" PORT="$PORT" MAX_NUM_SEQS=1
    export GPU_UTIL GPU_GUARD_FLOOR_MIB
    export LOG_DIR="$PWD/$cdir/logs" FR13_RUN_DIR="$PWD/$cdir" FR13_SERVE_LOG=1
    local kv
    while IFS= read -r kv; do
      [[ -z "$kv" ]] && continue
      [[ "$kv" == "FAIL_UNKNOWN_ARM" ]] && { echo "FAIL: unknown arm"; exit 2; }
      export "$kv"
    done < <(arm_env "$arm")
    docker rm -f "$container" >/dev/null 2>&1 || true
    recover_host
    bash scripts/fr13_launch_forked_fa2_tree_server.sh > "$cdir/launch.log" 2>&1 || { echo "FAIL: launcher"; tail -20 "$cdir/launch.log"; exit 3; }
    local t0=$(date +%s) healthy=0
    while (( $(date +%s) < t0 + 1200 )); do
      curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { healthy=1; break; }
      [[ "$(docker inspect -f '{{.State.Status}}' "$container" 2>/dev/null)" != "running" ]] && { echo "FAIL: died"; docker logs "$container" 2>&1|tail -30; exit 3; }
      sleep 5
    done
    (( healthy )) || { echo "FAIL: health timeout"; exit 3; }
    echo "[$arm] healthy after $(( $(date +%s)-t0 ))s"
    docker exec "$container" env 2>/dev/null | sort > "$cdir/container_env.txt" || true
    docker logs "$container" > "$cdir/boot_log.txt" 2>&1
    grep -q "enable_prefix_caching=$(arm_cache_expect "$arm")" "$cdir/boot_log.txt" || { echo "FAIL: cache state"; exit 4; }

    local i
    for (( i=1; i<=N; i++ )); do
      curl -s -o /dev/null -m 20 -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" 2>/dev/null
      # t1: original messages
      python3 -c "import json,sys; json.dump(json.load(open('$PAYLOAD'))['messages'], open('$cdir/msgs_t1_s$i.json','w'))"
      curl -s -m 20 "http://127.0.0.1:$PORT/metrics" 2>/dev/null | grep -E 'prefix_cache' > "$cdir/m_t1_s${i}_pre.txt" || true
      turn_send "$cdir" "$i" 1 "$cdir/msgs_t1_s$i.json"
      curl -s -m 20 "http://127.0.0.1:$PORT/metrics" 2>/dev/null | grep -E 'prefix_cache' > "$cdir/m_t1_s${i}_post.txt" || true
      # t2/t3: extend with prior assistant text + continue prompt
      local tn prev
      for tn in 2 3; do
        prev=$((tn-1))
        python3 - "$cdir/msgs_t${prev}_s$i.json" "$cdir/t${prev}_s${i}.json" "$cdir/msgs_t${tn}_s$i.json" <<'PY'
import json, sys
msgs = json.load(open(sys.argv[1]))
try:
    r = json.load(open(sys.argv[2]))
    m = r["choices"][0]["message"]
    txt = (m.get("content") or "") or (m.get("reasoning") or "")[:6000]
except Exception:
    txt = "(previous turn unavailable)"
msgs = msgs + [{"role":"assistant","content": txt},
               {"role":"user","content":"Continue the analysis in more depth. Walk through the relevant code paths step by step and propose the complete fix with full justification."}]
json.dump(msgs, open(sys.argv[3], "w"), ensure_ascii=False)
PY
        curl -s -m 20 "http://127.0.0.1:$PORT/metrics" 2>/dev/null | grep -E 'prefix_cache' > "$cdir/m_t${tn}_s${i}_pre.txt" || true
        turn_send "$cdir" "$i" "$tn" "$cdir/msgs_t${tn}_s$i.json"
        curl -s -m 20 "http://127.0.0.1:$PORT/metrics" 2>/dev/null | grep -E 'prefix_cache' > "$cdir/m_t${tn}_s${i}_post.txt" || true
      done
      python3 - "$cdir" "$i" <<'PY'
import json, sys, re
cdir, seed = sys.argv[1], sys.argv[2]
def brief(tn):
    try:
        d = json.load(open(f"{cdir}/t{tn}_s{seed}.json"))
        c0 = d["choices"][0]; m = c0.get("message") or {}
        txt = (m.get("content") or "") + " " + (m.get("reasoning") or "")
        ct = (d.get("usage") or {}).get("completion_tokens")
        cjk = sum(1 for ch in txt if '一' <= ch <= '鿿')
        rep = bool(re.search(r'(.{2,8})\1{6,}', txt))
        off = any(k in txt for k in ("# 1. Introduction", "apt install", "White Rose", "References ["))
        return f"t{tn}: ct={ct} fin={c0.get('finish_reason')} cjk={cjk} rep={rep} offtask={off}"
    except Exception as e:
        return f"t{tn}: ERR {str(e)[:60]}"
print(f"  seed {seed}: " + " | ".join(brief(t) for t in (1,2,3)), flush=True)
PY
    done
    # engagement: decode-side snapshot writes must have fired on the cache arm
    docker logs "$container" 2>&1 | grep -aoE "snapshot_events=[0-9]+" | tail -3 > "$cdir/snapshot_needle.txt" || true
    docker logs "$container" > "$cdir/docker_full.log" 2>&1 || true
    docker rm -f "$container" >/dev/null 2>&1 || true
    recover_host
    echo "[$arm] DONE ($(date -u +%H:%M:%S)) snapshot_needle=$(cat "$cdir/snapshot_needle.txt" 2>/dev/null | tr '\n' ' ')"
  )
  local arc=$?
  docker rm -f "$container" >/dev/null 2>&1 || true
  return $arc
}

RC=0
for arm in $ARMS; do run_arm "$arm" || { echo "ARM $arm FAILED"; RC=1; }; done
echo "DONE fb_probe rc=$RC $(date -u +%FT%TZ)"
exit $RC
