#!/usr/bin/env bash
# fr13_garble_replay.sh — FR13 §65 GARBLE ATTRIBUTION via exact-request replay.
# Replays the CAPTURED s2 garble-producing request (msgs=28; its live response was
# the 7415-tok garble) N times per arm, cold (reset before each), garble-scanned.
# 2x2: cat8_cache (fixes ON) vs cat8_nocache (CONFIG_ONLY). Attributes garble to
# the cache config vs the input/context. NOT a behavior gate (replayed request).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true
PAYLOAD="${PAYLOAD:-/tmp/claude-1000/-home-mark-shared/8245ff29-3bc7-469e-950a-7b8a1ab41d2a/scratchpad/garble_producer.json}"
PORT=9950; MODEL=qwen3.6-27b
N="${N:-8}"; MAX_TOKENS="${MAX_TOKENS:-2400}"
RR="${RR:-output/fr13_garble_replay}"
GPU_UTIL="${GPU_UTIL:-0.82}"; GPU_GUARD_FLOOR_MIB="${GPU_GUARD_FLOOR_MIB:-6500}"
ARMS="${ARMS:-cat8_cache cat8_nocache}"
export GPU_UTIL GPU_GUARD_FLOOR_MIB
CAT8_TREE="[(0,),(1,),(0,0),(0,1),(0,0,0),(0,0,1),(0,0,0,0),(0,0,0,0,0)]"
mkdir -p "$RR"
[[ -f "$PAYLOAD" ]] || { echo "FAIL: payload $PAYLOAD missing"; exit 2; }
if [[ -n "$(docker ps -q 2>/dev/null)" ]]; then echo "FAIL: container running"; exit 2; fi
recover_host(){ PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python - <<'PY' 2>/dev/null || true
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY
}
arm_env(){ case "$1" in
  cat8_cache) printf 'TREE=%s\nFR13_ENABLE_APC=1\nFR13_APC_EXACT_SEED=1\nMAMBA_BLOCK_SIZE=1024\nMAMBA_SSM_CACHE_DTYPE=float32\nFR13_APC_ZERO_MAMBA_ON_ALLOC=1\n' "$CAT8_TREE";;
  cat8_nocache) printf 'TREE=%s\nFR13_ENABLE_APC=1\nFR13_APC_CONFIG_ONLY=1\nMAMBA_BLOCK_SIZE=1024\nMAMBA_SSM_CACHE_DTYPE=float32\nFR13_APC_ZERO_MAMBA_ON_ALLOC=1\n' "$CAT8_TREE";;
esac; }
arm_cache_expect(){ case "$1" in cat8_cache) echo True;; cat8_nocache) echo False;; esac; }
run_arm(){
  local arm="$1"; local cdir="$RR/$arm"; local container="fr13-gr-$arm"
  mkdir -p "$cdir/logs"; echo ""; echo "#### ARM $arm ($(date -u +%H:%M:%S)) ####"
  ( set -uo pipefail
    export FR10_METRICS=0 BATCH_INVARIANT=0 LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16
    export CONTAINER="$container" PORT="$PORT" MAX_NUM_SEQS=1 GPU_UTIL GPU_GUARD_FLOOR_MIB
    export LOG_DIR="$PWD/$cdir/logs" FR13_RUN_DIR="$PWD/$cdir" FR13_SERVE_LOG=1
    local kv; while IFS= read -r kv; do [[ -z "$kv" ]] && continue; export "$kv"; done < <(arm_env "$arm")
    docker rm -f "$container" >/dev/null 2>&1 || true; recover_host
    bash scripts/fr13_launch_forked_fa2_tree_server.sh > "$cdir/launch.log" 2>&1 || { echo "FAIL launcher"; tail -20 "$cdir/launch.log"; exit 3; }
    local t0=$(date +%s) ok=0
    while (( $(date +%s) < t0+1200 )); do curl -fsS -m3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
      [[ "$(docker inspect -f '{{.State.Status}}' "$container" 2>/dev/null)" != running ]] && { echo FAIL_died; exit 3; }; sleep 5; done
    (( ok )) || { echo FAIL_health; exit 3; }
    docker logs "$container" > "$cdir/boot.txt" 2>&1
    grep -q "enable_prefix_caching=$(arm_cache_expect "$arm")" "$cdir/boot.txt" || { echo "FAIL cache state"; exit 4; }
    echo "[$arm] healthy $(( $(date +%s)-t0 ))s"
    local i
    for (( i=1; i<=N; i++ )); do
      curl -s -o /dev/null -m20 -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" 2>/dev/null
      python3 -c "import json,sys;d=json.load(open('$PAYLOAD'));d['max_tokens']=$MAX_TOKENS;d['temperature']=0.6;d['seed']=$i;d['stream']=False;d.pop('stream_options',None);json.dump(d,open('$cdir/send_$i.json','w'))"
      curl -sS -m900 -H 'Content-Type: application/json' -H 'Authorization: Bearer EMPTY' \
        --data-binary @"$cdir/send_$i.json" "http://127.0.0.1:$PORT/v1/chat/completions" > "$cdir/s_$i.json" 2>"$cdir/s_$i.err"
      python3 - "$cdir/s_$i.json" "$i" <<'PY'
import json,sys,re
try: d=json.load(open(sys.argv[1]))
except Exception as e: print(f"  s{sys.argv[2]}: PARSE_FAIL {e}"); sys.exit()
c0=(d.get('choices') or [{}])[0]; m=c0.get('message') or {}
txt=(m.get('content') or '')+' '+(m.get('reasoning') or '')
tc=m.get('tool_calls') or []
arg=json.dumps(tc[0].get('function',{}).get('arguments','')) if tc else ''
big=txt+' '+arg
cjk=sum(1 for ch in big if '一'<=ch<='鿿')
rep=bool(re.search(r'(.{2,8})\1{6,}',big)); off=any(k in big for k in('# 1. Introduction','apt install','White Rose','References ['))
argbloat=len(arg)>4000
g = cjk>40 or rep or off or argbloat
ct=(d.get('usage') or {}).get('completion_tokens')
print(f"  s{sys.argv[2]}: ct={ct} fin={c0.get('finish_reason')} route={tc[0]['function']['name'] if tc else 'NO_TOOL'} GARBLE={g} (cjk={cjk} rep={rep} off={off} argbloat={argbloat})")
PY
    done
    docker logs "$container" > "$cdir/docker_full.log" 2>&1 || true
    docker rm -f "$container" >/dev/null 2>&1 || true; recover_host
    echo "[$arm] DONE"
  )
  docker rm -f "$container" >/dev/null 2>&1 || true
}
for arm in $ARMS; do run_arm "$arm"; done
echo "DONE garble_replay $(date -u +%FT%TZ)"
