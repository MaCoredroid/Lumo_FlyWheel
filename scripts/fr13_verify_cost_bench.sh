#!/usr/bin/env bash
# Verify-cost isolation: boot a tree (TREE + LABEL env) with FR13_SFWD_GPU_TIMER, run a B=1 long generation
# (verify GPU time is workload-INDEPENDENT -> no SWE needed), read verify_ms/step from the SFWD sidecar.
# Isolates n_pad vs depth cost: cat33333 (d5,npad16) vs T55555 (d5,npad32) vs chain15 (d15,npad16) vs tail6.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
PORT=9959; C=fr13-vbench
LABEL="${LABEL:?LABEL}"; TREE="${TREE:?TREE}"
BV="${BV:-8}"   # BV=8 for n_pad=32; harmless at n_pad<=16
TS=$(date -u +%Y%m%dT%H%M%SZ); RUN=output/fr13_vbench/${LABEL}_$TS; mkdir -p "$RUN" output/fr13_sfwd_sidecar
NSPEC=$(.venv/bin/python -c "import ast; print(len(ast.literal_eval('$TREE')))")
echo "[vbench] LABEL=$LABEL nodes=$NSPEC BV=$BV" | tee "$RUN/bench.log"
docker ps -aq --filter "name=fr13" | xargs -r docker rm -f >/dev/null 2>&1 || true
PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true
env CONTAINER=$C PORT=$PORT GPU_UTIL=0.8 MAX_NUM_SEQS=1 ATTENTION_BACKEND=TREE_ATTN \
  FR13_TREE_GDN_GEOM_OVERRIDE="BV=$BV" \
  FR13_SFWD_GPU_TIMER=1 FR13_SFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${LABEL}.json \
  FR13_CFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${LABEL}_cfwd.json \
  SPEC_CONFIG='{"method":"qwen3_5_mtp","num_speculative_tokens":'"$NSPEC"',"speculative_token_tree":"'"$TREE"'"}' \
  GPU_GUARD_FLOOR_MIB=3000 \
  bash scripts/fr13_launch_forked_fa2_tree_server.sh > "$RUN/boot.log" 2>&1 &
LPID=$!; T0=$SECONDS; OK=0
while [ $((SECONDS-T0)) -lt 1500 ]; do
  curl -fsS -m5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { OK=1; break; }
  [[ -n "$(docker ps -aq -f name=$C -f status=exited)" ]] && { echo "BOOT EXITED" | tee -a "$RUN/bench.log"; grep -iE "n_pad must|Traceback|Error" "$RUN/boot.log" 2>/dev/null | tail -8 | tee -a "$RUN/bench.log"; break; }
  sleep 10
done
if [ "$OK" = 1 ]; then
  MODEL=$(curl -fsS -m5 "http://127.0.0.1:$PORT/v1/models" | .venv/bin/python -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
  echo "[vbench] HEALTHY -> B=1 decode (3x400 tok) to accumulate pure-decode forwards" | tee -a "$RUN/bench.log"
  for i in 1 2 3; do
    curl -fsS -m120 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL\",\"prompt\":\"Write a long Python module with many functions:\\n\",\"max_tokens\":400,\"temperature\":0.6,\"seed\":$i}" >/dev/null 2>&1
  done
  sleep 3   # let the timer dump
  echo "--- SFWD verify sidecar ---" | tee -a "$RUN/bench.log"
  SF=$(ls -t output/fr13_sfwd_sidecar/${LABEL}.json* 2>/dev/null | head -1)
  CF=$(ls -t output/fr13_sfwd_sidecar/${LABEL}_cfwd.json* 2>/dev/null | head -1)
  .venv/bin/python -c "
import json
sv=json.load(open('$SF')) if '$SF' else {}
vsec=sv.get('decode_forward_gpu_seconds'); vst=sv.get('n_pure_decode_steps_timed')
print(f'LABEL=$LABEL nodes=$NSPEC : VERIFY/step = {1000*vsec/vst:.1f} ms  (n={vst})' if vsec and vst else f'LABEL=$LABEL : verify sidecar incomplete {sv}')
try:
    cv=json.load(open('$CF')); print(f'  committer/step = {1000*cv[\"gpu_seconds\"]/cv[\"n_spans\"]:.1f} ms' if cv.get('gpu_seconds') and cv.get('n_spans') else '')
except Exception: pass
" 2>&1 | tee -a "$RUN/bench.log"
else
  echo "[vbench] FAIL: not healthy in 1500s" | tee -a "$RUN/bench.log"
fi
docker rm -f "$C" >/dev/null 2>&1 || true; wait $LPID 2>/dev/null || true
echo "=== VBENCH DONE $LABEL ($RUN) ===" | tee -a "$RUN/bench.log"
