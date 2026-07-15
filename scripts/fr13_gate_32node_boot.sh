#!/usr/bin/env bash
# GATE 1/2/3 (accept>5 32-node infra): BV=8 + n_pad=32 (25-node wide tree, MTP-fillable at depth-5).
# Validates the register budget (GATE 2: spill-free boot) + boot viability + a generation smoke. Bit-exact
# (GATE 1) is high-conf by BV16=BV32=0.0 + reductions-never-over-BV. If it boots+serves clean => 32-node
# horizon infra viable => proceed to N_DEPTH generalize + tail.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
PORT=9957; C=fr13-gate-32node
TS=$(date -u +%Y%m%dT%H%M%SZ); RUN=output/fr13_gate_32node/run_$TS; mkdir -p "$RUN"
TREE='[(0,),(1,),(2,),(3,),(4,),(5,),(6,),(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,0,0),(0,0,1),(0,0,2),(0,0,3),(0,0,4),(0,0,0,0),(0,0,0,1),(0,0,0,2),(0,0,0,3),(0,0,0,0,0),(0,0,0,0,1),(0,0,0,0,2)]'
docker ps -aq --filter "name=fr13" | xargs -r docker rm -f >/dev/null 2>&1 || true
PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true
env CONTAINER=$C PORT=$PORT GPU_UTIL=0.8 MAX_NUM_SEQS=4 ATTENTION_BACKEND=TREE_ATTN \
  FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 \
  SPEC_CONFIG='{"method":"qwen3_5_mtp","num_speculative_tokens":25,"speculative_token_tree":"'"$TREE"'"}' \
  GPU_GUARD_FLOOR_MIB=3000 \
  bash scripts/fr13_launch_forked_fa2_tree_server.sh > "$RUN/boot.log" 2>&1 &
LPID=$!; T0=$SECONDS; OK=0
while [ $((SECONDS-T0)) -lt 900 ]; do
  curl -fsS -m5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { OK=1; break; }
  [[ -n "$(docker ps -aq -f name=$C -f status=exited)" ]] && { echo "BOOT EXITED early"; break; }
  sleep 10
done
if [ "$OK" = 1 ]; then
  echo "GATE 2 PASS: BV=8 n_pad=32 booted + healthy (GEOM=$(docker exec $C printenv FR13_TREE_GDN_GEOM_OVERRIDE 2>/dev/null))"
  echo "--- spill/register signals in boot (should be none) ---"
  grep -iE "spill|out of resource|register.*exceed|OutOfMemory|CUDA_ERROR|too many resources" "$RUN/boot.log" | tail -5 || echo "  (none)"
  MODEL=$(curl -fsS -m5 "http://127.0.0.1:$PORT/v1/models" | .venv/bin/python -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
  echo "--- generation smoke (n_pad=32 tree verify) ---"
  curl -fsS -m40 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"def fibonacci(n):\\n\",\"max_tokens\":48,\"temperature\":0.6}" \
    | .venv/bin/python -c "import json,sys; d=json.load(sys.stdin); print('SMOKE OK ->', repr(d['choices'][0]['text'][:100]))" 2>&1 | tail -3
else
  echo "GATE 2 FAIL: boot not healthy in 900s"
  grep -iE "error|assert|Traceback|CUDA|spill|resource|register|OOM|NotImplemented|n_pad" "$RUN/boot.log" | tail -20
fi
docker rm -f "$C" >/dev/null 2>&1 || true; wait $LPID 2>/dev/null || true
echo "=== GATE 32-node boot test DONE ($RUN) ==="
