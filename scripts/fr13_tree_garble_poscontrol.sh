#!/usr/bin/env bash
# FR13 POSITIVE CONTROL for the native+batch garble null:
# native+cache+realB8 gave undefined-name-rate 0.00% (== B1). To make that DECISIVE (vs
# "gate insensitive"), run the SAME gate on cat8 TREE — the arm that garbled in live SWE.
# If tree > 0% and native == 0% -> garble is the tree's intra-request batched-scan, NOT batching.
# If tree also 0% -> the synthetic gate doesn't trigger it (null inconclusive; need live-SWE).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
CONT=fr13-garble-tree-cat8; PORT=9955; N=${N:-24}
CAT8='[(0,),(1,),(0,0),(0,1),(0,0,0),(0,0,1),(0,0,0,0),(0,0,0,0,0)]'
TS=$(date -u +%Y%m%dT%H%M%SZ); RUN=output/fr13_native_batch_garble/tree_pos_$TS; mkdir -p "$RUN"
echo "=== TREE(cat8) garble POSITIVE CONTROL  -> $RUN ==="
docker ps -aq --filter "name=fr13" | xargs -r docker rm -f >/dev/null 2>&1 || true
PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true

echo "[boot] cat8 forked tree (TREE_ATTN, num_spec=8) max_num_seqs=8 ..."
CONTAINER=$CONT PORT=$PORT GPU_UTIL=0.82 MAX_NUM_SEQS=8 \
  ATTENTION_BACKEND=TREE_ATTN \
  SPEC_CONFIG='{"method":"qwen3_5_mtp","num_speculative_tokens":8,"speculative_token_tree":"'"$CAT8"'"}' \
  GPU_GUARD_FLOOR_MIB=3000 \
  bash scripts/fr13_launch_forked_fa2_tree_server.sh > "$RUN/boot.log" 2>&1 &
LPID=$!
echo "[boot] waiting /health (<=12min)..."
T0=$SECONDS; OK=0
while [ $((SECONDS-T0)) -lt 720 ]; do
  curl -fsS -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { OK=1; break; }
  [[ -n "$(docker ps -aq -f name=$CONT -f status=exited)" ]] && { echo "[boot] container exited"; break; }
  sleep 10
done
[ "$OK" = 1 ] || { echo "FAIL: tree not healthy"; tail -30 "$RUN/boot.log"; exit 2; }
MODEL=$(curl -fsS -m 5 "http://127.0.0.1:$PORT/v1/models" | .venv/bin/python -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
echo "[boot] healthy at $((SECONDS-T0))s model=$MODEL"

# spec-decode ENGAGED sanity (tree must actually be speculating, not degrade to non-spec)
sleep 3
docker logs --tail 40 "$CONT" 2>&1 | grep -oE 'speculative_token_tree|TREE_ATTN|num_speculative' | head -2 | sed 's/^/  engage: /'

# ARM tree_b8: real batch
echo "[tree_b8] gate concurrency=8 n=$N ..."
.venv/bin/python scripts/fr13_garble_gate.py run --endpoint "http://127.0.0.1:$PORT/v1" \
  --model "$MODEL" --arm tree_b8 --n "$N" --concurrency 8 --out "$RUN/tree_b8.jsonl" > "$RUN/tree_b8_run.log" 2>&1 &
GATE=$!
while kill -0 $GATE 2>/dev/null; do docker logs --tail 4 "$CONT" 2>&1 | grep -oE 'Running: [0-9]+ reqs' | tail -1; sleep 3; done > "$RUN/running_tree_b8.log" 2>&1
wait $GATE

# ARM tree_b1: serial (the effective-B~1 live-SWE regime where garble was seen)
echo "[tree_b1] gate concurrency=1 n=$N ..."
.venv/bin/python scripts/fr13_garble_gate.py run --endpoint "http://127.0.0.1:$PORT/v1" \
  --model "$MODEL" --arm tree_b1 --n "$N" --concurrency 1 --out "$RUN/tree_b1.jsonl" > "$RUN/tree_b1_run.log" 2>&1

echo ""; echo "=== TREE SCORES (undefined-name rate) ==="
for a in tree_b8 tree_b1; do echo "--- $a ---"; .venv/bin/python scripts/fr13_garble_gate.py score --samples "$RUN/$a.jsonl" 2>&1 | tee "$RUN/${a}_score.txt"; done
echo "  tree_b8 max Running = $(grep -oE 'Running: [0-9]+' "$RUN/running_tree_b8.log" 2>/dev/null|grep -oE '[0-9]+'|sort -n|tail -1)"
echo "=== teardown ==="; docker rm -f "$CONT" >/dev/null 2>&1 || true
echo "=== DONE $RUN @ $(date -u +%H:%M:%S)Z ==="
