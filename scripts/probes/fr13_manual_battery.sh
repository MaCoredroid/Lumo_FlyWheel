#!/usr/bin/env bash
# Detached minimal rp2 battery against an ALREADY-BOOTED container (survives the
# harness background-bash kill via setsid). Measures token-1 logprob for cold (P1,
# reset) vs no-reset cache-HIT (P2). Flat -0.001 on P2 == carrier fixed.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
PORT="${PORT:-9950}"
CONTAINER="${CONTAINER:-fr13-rp2-cat8_cache_stateless}"
CDIR="${CDIR:-output/fr13_rp2_stateless/cat8_cache_stateless}"
mkdir -p "$CDIR"
LOG="$CDIR/manual_battery.log"
exec > "$LOG" 2>&1
echo "=== manual battery start $(date -u +%H:%M:%S) container=$CONTAINER ==="
# error watch during remaining boot
for i in $(seq 1 300); do
  if curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo "HEALTHY after ${i}s"; break; fi
  if docker logs "$CONTAINER" 2>&1 | grep -qiE 'Traceback|EngineCore.*Error|RuntimeError|CUDA error|assert.*Error|Fatal'; then
    echo "FAIL: error in container log during boot:"; docker logs "$CONTAINER" 2>&1 | grep -iE 'Traceback|Error|assert' | tail -12; exit 4
  fi
  sleep 1
done
curl -fsS -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "FAIL: never healthy in 300s"; exit 4; }
# confirm flags took (boot-config)
echo "--- boot-config ---"
docker logs "$CONTAINER" 2>&1 | grep -m1 'enable_prefix_caching' || echo "(no prefix_caching line)"
docker inspect "$CONTAINER" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null \
  | grep -E '^(FR13_APC_COMMIT_TO_RUNNING_ROW|FR13_TREE_RUNROW_INIT|FR13_APC_BURN_NODE_BANK|FR13_APC_EXACT_SEED)=' || true
# build send payload (logprobs on)
SEND="$CDIR/send_manual.json"
python3 - scripts/probes/route_probe_payload.json "$SEND" 448 0.6 <<'PY'
import json,sys
src,dst,mx,tp=sys.argv[1:5]
d=json.load(open(src)); d["max_tokens"]=int(mx); d["temperature"]=float(tp)
d["logprobs"]=True; d["top_logprobs"]=20
json.dump(d,open(dst,"w"),ensure_ascii=False)
PY
run_one(){
  local tag="$1" reset="$2"
  [[ "$reset" == 1 ]] && curl -s -o /dev/null -m 20 -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" >/dev/null 2>&1
  python3 - "$SEND" "$CDIR/s_$tag.json" 5 <<'PY'
import json,sys; d=json.load(open(sys.argv[1])); d["seed"]=int(sys.argv[3]); json.dump(d,open(sys.argv[2],"w"))
PY
  curl -sS -m 900 -H 'Content-Type: application/json' -H 'Authorization: Bearer EMPTY' \
    --data-binary @"$CDIR/s_$tag.json" "http://127.0.0.1:$PORT/v1/chat/completions" > "$CDIR/r_$tag.json" 2>/dev/null
  python3 - "$CDIR/r_$tag.json" "$tag" <<'PY'
import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception as e: print(f"{sys.argv[2]} PARSE_FAIL {str(e)[:80]}"); sys.exit(0)
ch=(d.get("choices") or [{}])[0]; lp=ch.get("logprobs") or {}; cont=lp.get("content") or []
tc=(ch.get("message") or {}).get("tool_calls") or []
route=tc[0].get("function",{}).get("name") if tc else "NO_TOOL"
t1=f"{cont[0].get('logprob'):.4f}" if cont else "NA"
print(f"BATTERY {sys.argv[2]} tok1={t1} route={route} finish={ch.get('finish_reason')} ct={(d.get('usage') or {}).get('completion_tokens')}")
PY
}
echo "--- P1 cold (reset) ---"
for k in 1 2 3 4 5; do run_one "P1_$k" 1; done
echo "--- P2 no-reset cache HIT ---"
for k in 1 2 3 4 5 6; do run_one "P2_$k" 0; done
echo "=== manual battery DONE $(date -u +%H:%M:%S) ==="
