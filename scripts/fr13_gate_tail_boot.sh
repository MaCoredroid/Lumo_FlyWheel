#!/usr/bin/env bash
# GATE 3.5/4 (accept>5 TAIL): boot the 31-node tail tree (cat33333 head + 16-node arctic chain, n_pad=32,
# BV=8) in TAIL mode + merged cache lifecycle, drive a REPETITIVE workload (arctic self-warms from the
# request's own history), and (1) VERIFY the tail ENGAGES (TAIL[fired>0 hit>0] in the drafter needle --
# assert engagement before trusting any number), (2) read per-position accept to see whether the tail
# accepts at depths 6+ (the >5 signal). Preliminary; the real GATE 4 is a live SWE-Verified run.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
PORT=9958; C=fr13-gate-tail
TS=$(date -u +%Y%m%dT%H%M%SZ); RUN=output/fr13_gate_tail/run_$TS; mkdir -p "$RUN"
# tail tree SPEC from the ONE topology source (code-consistent). FR13_TAIL_LEN sweeps the chain depth:
# depth-21 (tail_len=16) HANGS graph capture (deep GDN recurrence is O(depth), 12min profiling). Start
# SHALLOW (default 6 -> depth-11 tree, 21 nodes, still n_pad=32) -- already >5-capable (accept up to 11).
TAIL_LEN=${FR13_TAIL_LEN:-6}
TREE=$(.venv/bin/python -c "import sys; sys.path.insert(0,'scripts'); from fr13_mtp_suffix_assembly import tail_tree_order as t; o=t(tail_len=$TAIL_LEN); print('['+','.join('('+','.join(str(x) for x in p)+(',' if len(p)==1 else '')+')' for p in o)+']')")
NSPEC=$(.venv/bin/python -c "import sys; sys.path.insert(0,'scripts'); from fr13_mtp_suffix_assembly import tail_tree_order as t; print(len(t(tail_len=$TAIL_LEN)))")
WIDE_D=$((5 + TAIL_LEN))
echo "[gate-tail] tail_len=$TAIL_LEN -> tree nodes=$NSPEC, wide_D=$WIDE_D (n_pad<=32); TAIL mode + merged cache" | tee "$RUN/gate.log"

docker ps -aq --filter "name=fr13" | xargs -r docker rm -f >/dev/null 2>&1 || true
PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true
env CONTAINER=$C PORT=$PORT GPU_UTIL=0.8 MAX_NUM_SEQS=4 ATTENTION_BACKEND=TREE_ATTN \
  FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 FR13_DRAFT_SOURCE=merged FR13_TAIL_MODE=1 \
  SPEC_CONFIG='{"method":"qwen3_5_mtp","num_speculative_tokens":'"$NSPEC"',"speculative_token_tree":"'"$TREE"'"}' \
  GPU_GUARD_FLOOR_MIB=3000 \
  bash scripts/fr13_launch_forked_fa2_tree_server.sh > "$RUN/boot.log" 2>&1 &
LPID=$!; T0=$SECONDS; OK=0
( while [ $((SECONDS-T0)) -lt 1800 ]; do
    if docker ps -aq -f name=$C 2>/dev/null | grep -q .; then docker logs -f "$C" > "$RUN/engine.log" 2>&1; break; fi
    sleep 2; done ) & CAPPID=$!
while [ $((SECONDS-T0)) -lt 1800 ]; do
  curl -fsS -m5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { OK=1; break; }
  [[ -n "$(docker ps -aq -f name=$C -f status=exited)" ]] && { echo "BOOT EXITED early" | tee -a "$RUN/gate.log"; break; }
  sleep 10
done
if [ "$OK" != 1 ]; then
  echo "GATE-TAIL FAIL: boot not healthy in 1800s" | tee -a "$RUN/gate.log"
  grep -iE "n_pad must be|out of resource|register|Traceback|ValueError|RuntimeError|assert" "$RUN/engine.log" 2>/dev/null | tail -20 | tee -a "$RUN/gate.log"
  docker logs "$C" > "$RUN/engine_final.log" 2>&1 || true
  docker rm -f "$C" >/dev/null 2>&1 || true; kill $CAPPID 2>/dev/null || true; wait $LPID 2>/dev/null || true
  echo "=== GATE-TAIL DONE ($RUN) ===" | tee -a "$RUN/gate.log"; exit 0
fi
MODEL=$(curl -fsS -m5 "http://127.0.0.1:$PORT/v1/models" | .venv/bin/python -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
echo "[gate-tail] HEALTHY model=$MODEL -> driving repetitive workload (arctic self-warm)" | tee -a "$RUN/gate.log"
curl -fsS "http://127.0.0.1:$PORT/metrics" 2>/dev/null | grep -E "spec_decode_num_(accepted|draft)_tokens" > "$RUN/metrics_before.txt" 2>/dev/null || true
# REPETITIVE workload: a long generation with a recurring structure -> the per-request arctic trie
# self-warms so the deep tail chain (depths 6+) should hit + accept on the repeats. 6 rounds, temp 0.6.
PROMPT='# Build a config dict with 40 similar entries.\nconfig = {}\nconfig["item_000"] = {"id": 0, "value": 0, "enabled": True, "score": 0.0}\nconfig["item_001"] = {"id": 1, "value": 2, "enabled": True, "score": 1.0}\nconfig["item_002"] = {"id": 2, "value": 4, "enabled": True, "score": 2.0}\n'
for i in $(seq 1 6); do
  curl -fsS -m120 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"$PROMPT\",\"max_tokens\":320,\"temperature\":0.6,\"seed\":$i}" \
    > "$RUN/probe_$i.json" 2>/dev/null &
  [ $((i % 4)) -eq 0 ] && wait   # keep B<=4
done
wait
curl -fsS "http://127.0.0.1:$PORT/metrics" 2>/dev/null | grep -E "spec_decode_num_(accepted|draft)_tokens" > "$RUN/metrics_after.txt" 2>/dev/null || true
echo "--- ENGAGEMENT (drafter needle: TAIL[fired hit cold]) ---" | tee -a "$RUN/gate.log"
docker logs "$C" 2>&1 | grep -oE "TAIL\[fired=[0-9]+ hit=[0-9]+ cold=[0-9]+\]" | tail -3 | tee -a "$RUN/gate.log"
echo "--- ACCEPT per-position (positions 6-20 non-zero => TAIL accepts => >5 signal) ---" | tee -a "$RUN/gate.log"
curl -fsS "http://127.0.0.1:$PORT/metrics" 2>/dev/null | grep -E "spec_decode_num_accepted_tokens_per_pos" | tail -25 | tee -a "$RUN/gate.log"
echo "--- aggregate accept/draft (before -> after) ---" | tee -a "$RUN/gate.log"
{ echo "BEFORE:"; cat "$RUN/metrics_before.txt"; echo "AFTER:"; cat "$RUN/metrics_after.txt"; } | tee -a "$RUN/gate.log"
docker logs "$C" > "$RUN/engine_final.log" 2>&1 || true
docker rm -f "$C" >/dev/null 2>&1 || true; kill $CAPPID 2>/dev/null || true; wait $LPID 2>/dev/null || true
echo "=== GATE-TAIL DONE ($RUN) ===" | tee -a "$RUN/gate.log"
