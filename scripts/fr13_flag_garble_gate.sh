#!/usr/bin/env bash
# FR13 generic garble-gate: boot cat8 tree with arbitrary env flags, run the temp-0.6 identifier-bait
# gate, score undefined-name rate. Reusable so we stop rewriting per-fix drivers.
#   usage: fr13_flag_garble_gate.sh <arm_label> [KEY=V ...]
# baseline default=9.62% (same session 2026-07-11), native=0%. Fix candidate ~0% => birthplace+fix.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
ARM=${1:?arm label}; shift; FLAGS=("$@")
PORT=9955; N=${N:-24}; CONC=${CONC:-4}; C=fr13-garble-flag-$ARM
CAT8='[(0,),(1,),(0,0),(0,1),(0,0,0),(0,0,1),(0,0,0,0),(0,0,0,0,0)]'
TS=$(date -u +%Y%m%dT%H%M%SZ); RUN=output/fr13_flag_garble/${ARM}_$TS; mkdir -p "$RUN"
echo "=== FLAG garble gate [$ARM]  flags=[${FLAGS[*]}]  -> $RUN ==="
docker ps -aq --filter "name=fr13" | xargs -r docker rm -f >/dev/null 2>&1 || true
PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true

env CONTAINER=$C PORT=$PORT GPU_UTIL=0.8 MAX_NUM_SEQS=8 \
  ATTENTION_BACKEND=TREE_ATTN \
  SPEC_CONFIG='{"method":"qwen3_5_mtp","num_speculative_tokens":8,"speculative_token_tree":"'"$CAT8"'"}' \
  GPU_GUARD_FLOOR_MIB=3000 "${FLAGS[@]}" \
  bash scripts/fr13_launch_forked_fa2_tree_server.sh > "$RUN/boot.log" 2>&1 &
LPID=$!; T0=$SECONDS; OK=0
while [ $((SECONDS-T0)) -lt 720 ]; do
  curl -fsS -m5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { OK=1; break; }
  [[ -n "$(docker ps -aq -f name=$C -f status=exited)" ]] && { echo "[$ARM] exited"; break; }
  sleep 10
done
if [ "$OK" != 1 ]; then
  echo "FAIL [$ARM] not healthy — crash tail:"; docker logs --tail 60 "$C" > "$RUN/dockerlogs.log" 2>&1
  grep -iE 'error|assert|raise|Traceback|Exception|CUDA' "$RUN/dockerlogs.log" 2>/dev/null | tail -15 | sed 's/^/    /'
  docker rm -f "$C" >/dev/null 2>&1; exit 2
fi
# fail-loud: print the intended flags as seen in the worker
for kv in "${FLAGS[@]}"; do k=${kv%%=*}; echo "  [$ARM] worker $k=$(docker exec $C printenv "$k" 2>/dev/null || echo MISSING)"; done
MODEL=$(curl -fsS -m5 "http://127.0.0.1:$PORT/v1/models" | .venv/bin/python -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
.venv/bin/python scripts/fr13_garble_gate.py run --endpoint "http://127.0.0.1:$PORT/v1" \
  --model "$MODEL" --arm "$ARM" --n "$N" --concurrency "$CONC" --out "$RUN/$ARM.jsonl" > "$RUN/gate.log" 2>&1
echo "  [$ARM] SCORE: $(.venv/bin/python scripts/fr13_garble_gate.py score --samples "$RUN/$ARM.jsonl" 2>&1 | tee "$RUN/score.txt" | head -1)"
echo "  (ref: default=9.62% native=0% SCAN_ALIGN=10.27% geom=8.34%)"
docker rm -f "$C" >/dev/null 2>&1 || true; wait $LPID 2>/dev/null || true
echo "=== DONE [$ARM] $RUN @ $(date -u +%H:%M:%S)Z ==="
